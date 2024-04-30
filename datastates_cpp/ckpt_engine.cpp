#include <ckpt_engine.hpp>

datastates_llm_t::datastates_llm_t(size_t host_cache_size, int gpu_id, int rank): _gpu_id(gpu_id), _rank(rank) {
    try {
        DBG("DataStates initing: GPU: " << _gpu_id << ", host cache (MB): " << (host_cache_size >> 20));
        checkCuda(cudaSetDevice(_gpu_id));
        is_active = true;
        checkCuda(cudaStreamCreateWithFlags(&_cpy_stream, cudaStreamNonBlocking));
        _thread_d2h = std::thread([&] { _d2h_trf(); });
        _thread_h2f = std::thread([&] { _h2f_trf(); });
        _thread_d2h.detach();
        _thread_h2f.detach();
        mem = new host_cache_t(host_cache_size, _gpu_id, _rank);
        _pending_d2h.clear();
        _pending_h2f.clear();

        DBG("Going to start alloc_thread GPU: " << _gpu_id);
        _thread_alloc = std::thread([&] { _alloc_tensor(); });
        _thread_alloc.detach();
        DBG("Started alloc_thread GPU: " << _gpu_id);
        for(int thread_id=0; thread_id < READ_THREADS; thread_id++) {
            _thread_read[thread_id] = std::thread([&, thread_id] { _read_file(thread_id); });
            _thread_read[thread_id].detach();
            DBG("Started thread_read GPU: " << _gpu_id);
        }

    } catch(std::exception& e) {
        FATAL("Standard exception caught in datastates init: " << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in datastates init.");
    }
}

void datastates_llm_t::_d2h_trf() {
    checkCuda(cudaSetDevice(_gpu_id));
    while (is_active) {
        try {
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            while(_pending_d2h.empty() && is_active)
                _cv_d2h.wait(_lock_d2h);
            if (!is_active) {
                _pending_d2h.clear();
                _lock_d2h.unlock();
                _cv_d2h.notify_all();
                return;
            }
            auto e = _pending_d2h.front();
            _lock_d2h.unlock();
            _cv_d2h.notify_all();

            int version         = std::get<0>(e);
            uint64_t uid        = std::get<1>(e);
            torch::Tensor t     = std::get<2>(e);
            size_t size         = std::get<3>(e);
            size_t file_offset  = std::get<4>(e);
            std::string path    = std::get<5>(e);

            DBG("[D2H][" << _rank << "] transfer of tensor " << uid << " version " << version << " torch sum is " << torch::sum(t) << " size " << size);
            mem_region_t* m = mem->allocate(uid, size);
            char *host_ptr = m->ptr;
            char *src_ptr = static_cast<char *>(t.data_ptr());
           
            checkCuda(cudaMemcpyAsync(host_ptr, src_ptr, size, cudaMemcpyDeviceToHost, _cpy_stream));
            checkCuda(cudaStreamSynchronize(_cpy_stream));
            std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
            _pending_h2f.push_back(std::make_tuple(version, m->uid, host_ptr, size, file_offset, path));
            _lock_h2f.unlock();
            _cv_h2f.notify_all();           
            
            _lock_d2h.lock();
            _pending_d2h.pop_front();
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
        } catch (std::exception &e) {
            FATAL("Exception caught in d2h trf." << e.what());
        } catch (...) {
            FATAL("Unknown exception caught in d2h trf.");
        }
    }
}


void datastates_llm_t::_h2f_trf() {
    checkCuda(cudaSetDevice(_gpu_id));
    while (is_active) {
        try {
            std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
            while(_pending_h2f.empty() && is_active)
                _cv_h2f.wait(_lock_h2f);
            if (!is_active) {
                _lock_h2f.unlock();
                _cv_h2f.notify_all();
                return;
            }
            TIMER_START(h2f_time);
            auto e = _pending_h2f.front();
            _lock_h2f.unlock();
            _cv_h2f.notify_all();
            
            int version         = std::get<0>(e);
            uint64_t uid        = std::get<1>(e);
            char* ptr           = std::get<2>(e);
            size_t size         = std::get<3>(e);
            size_t file_offset  = std::get<4>(e);
            std::string path    = std::get<5>(e);

            std::ofstream f;            
            f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
            // f.open(path,  std::ofstream::out | std::ofstream::app | std::ofstream::binary);
            f.open(path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(file_offset);
            f.write(ptr, size);
            f.flush();
            f.close();
            
            mem->deallocate(uid, size);
            _lock_h2f.lock();
            _pending_h2f.pop_front();
            _lock_h2f.unlock();
            _cv_h2f.notify_all();
            TIMER_STOP(h2f_time, "[H2F][" << _rank << "] Total time in h2f to save tensor " << uid << " version " << version << " of size " << size, size);
        }  catch (std::exception &e) {
            FATAL("Exception caught in h2f trf." << e.what());
        } catch (...) {
            FATAL("Unknown exception caught in h2f trf.");
        }
    }
}

void datastates_llm_t::ckpt_tensor(int version, const torch::Tensor &t, const std::uint64_t size, const std::uint64_t file_offset, std::string path) {
    try {
        uint64_t uid = local_uid++;
        
        if (t.device().is_cuda()) {
            assert((t.device().is_cuda() && t.device().index() == _gpu_id) && "Tensor not on the same GPU as ckpt engine");
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            _pending_d2h.push_back(std::make_tuple(version, uid, t, size, file_offset, path));
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            return;
        } 
        std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        _pending_h2f.push_back(std::make_tuple(version, uid, static_cast<char *>(t.data_ptr()), size, file_offset, path));
        _lock_h2f.unlock();
        _cv_h2f.notify_all();
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in ckpt_tensor." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in ckpt_tensor." << path);
    }
}

void datastates_llm_t::_alloc_tensor() {
    while (is_active) {
        std::unique_lock<std::mutex> _lock_alloc(_mutex_alloc);
        while(_pending_alloc.empty() && is_active)
            _cv_alloc.wait(_lock_alloc);
        if (!is_active)
            return;
        
        auto e          = _pending_alloc.front();
        void *ptr       = std::get<0>(e);
        size_t total_size = std::get<1>(e);
        _pending_alloc.pop_front();
        _lock_alloc.unlock();
        _cv_alloc.notify_all();
        
        TIMER_START(alloc_time);
        for (size_t i=0; i<total_size; i+=CHUNK_SIZE) {
            size_t rem = std::min(CHUNK_SIZE, total_size-i);
            memset((char *)ptr+i, 0, rem);
            _alloc_map[ptr] += rem;
            _cv_alloc.notify_all();
        }
        TIMER_STOP(alloc_time, "Time to alloc is ", total_size)
    }
}


void datastates_llm_t::alloc_tensor_queue(const torch::Tensor &t) {
    std::unique_lock<std::mutex> _lock_alloc(_mutex_alloc);
    void* tensor_ptr = static_cast<void*>(t.data_ptr());
    size_t tensor_size = t.numel()*t.element_size();
    _pending_alloc.push_back(std::make_tuple(tensor_ptr, tensor_size));
    _alloc_map[tensor_ptr] = 0;
    _lock_alloc.unlock();
    _cv_alloc.notify_all();
}


void datastates_llm_t::_read_file(const int thread_id) {
    while (is_active) {
        std::unique_lock<std::mutex> _lock_read(_mutex_read[thread_id]);
        while(_pending_read[thread_id].empty() && is_active) 
            _cv_read[thread_id].wait(_lock_read);
        if (!is_active)
            return;
        auto e              = _pending_read[thread_id].front();
        void * tensor_ptr   = std::get<0>(e);
        size_t f_start_offset = std::get<1>(e);
        size_t f_end_offset = std::get<2>(e);
        int fd              = std::get<3>(e);
        size_t tensor_size = f_end_offset-f_start_offset;

        size_t my_start_offset = thread_id*CHUNK_SIZE;
        size_t my_end_offset = my_start_offset+CHUNK_SIZE;
        std::unique_lock<std::mutex> _lock_alloc(_mutex_alloc, std::defer_lock);

        uint64_t pread_time = 0;
        TIMER_START(file_read);
        while (my_start_offset < tensor_size) {
            my_end_offset = std::min(my_start_offset+CHUNK_SIZE, tensor_size);
            if (my_end_offset > _alloc_map[tensor_ptr]) {
                _lock_alloc.lock();
                while (my_end_offset > _alloc_map[tensor_ptr])
                    _cv_alloc.wait(_lock_alloc);
                _lock_alloc.unlock();
                _cv_alloc.notify_all();
            }
            auto start_time = std::chrono::steady_clock::now();
            size_t bytesRead = pread(fd, (char *)tensor_ptr+my_start_offset, my_end_offset-my_start_offset, f_start_offset+my_start_offset);
            auto end_time = std::chrono::steady_clock::now();
            pread_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            if (bytesRead != my_end_offset-my_start_offset) {
                std::cerr << "Error reading from file. Read " << bytesRead << " intead of " << my_end_offset-my_start_offset << std::endl;
                return;
            }
            my_start_offset += READ_THREADS*CHUNK_SIZE;
        }
        TIMER_STOP(file_read, "Time to read file pread time " << pread_time << " ns for path ", tensor_size);

        _pending_read[thread_id].pop_front();
        _lock_read.unlock();
        _cv_read[thread_id].notify_all();
    }
}

void datastates_llm_t::restore_tensor(const torch::Tensor &t, std::string path, const std::uint64_t f_start_offset, const std::uint64_t f_end_offset) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }
    // Launch the read operations
    for (int thread_id=0; thread_id<READ_THREADS; thread_id++) {
        std::unique_lock<std::mutex> _lock_read(_mutex_read[thread_id]);
        _pending_read[thread_id].push_back(std::tuple(static_cast<void*>(t.data_ptr()), f_start_offset, f_end_offset, fd));
        _lock_read.unlock();
        _cv_read[thread_id].notify_all();
    }

    // Wait for all read operations to complete
    for (int thread_id=0; thread_id<READ_THREADS; thread_id++) {
        std::unique_lock<std::mutex> _lock_read(_mutex_read[thread_id]);
        while (!_pending_read[thread_id].empty())
            _cv_read[thread_id].wait(_lock_read);
        _lock_read.unlock();
    }
}


void datastates_llm_t::wait() {
    try {
        TIMER_START(wait_timer);
        std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
        while(!(_pending_d2h.empty())) {
            _cv_d2h.wait(_lock_d2h);
        }
        _lock_d2h.unlock();
        _cv_d2h.notify_all();
        TIMER_STOP(wait_timer, "[" << _rank << "] Wait D2H complete ", 1);
    }  catch (std::exception &e) {
        FATAL("Exception caught in wait D2H." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in wait D2H.");
    }
}

void datastates_llm_t::shutdown() {
    try {
        wait();
        std::unique_lock<std::mutex> _lock_h2f(_mutex_h2f);
        // Wait for D2H transfers
        while((!_pending_h2f.empty())) {
            DBG("[" << _rank << "] Waiting in h2f for " << _pending_h2f.size());
            for(auto e: _pending_h2f) {
                DBG("[" << _rank << "]" << std::get<0>(e) << " UID " << std::get<1>(e) << " size " << std::get<4>(e));
            }
            _cv_h2f.wait(_lock_h2f);
        }
        _lock_h2f.unlock();
        _cv_h2f.notify_all();
        
        is_active = false;
        delete mem;
        _cv_h2f.notify_all();
        _cv_d2h.notify_all();
        _cv_alloc.notify_all();
        DBG("Going to join alloc for " << _gpu_id);
        // _thread_alloc.join();
        for(int thread_id=0; thread_id<READ_THREADS; thread_id++) {
            _cv_read[thread_id].notify_all();
            DBG("Going to join read thread " << thread_id << " for " << _gpu_id);
            // _thread_read[thread_id].join();
        }
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in shutdown." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in shutdown.");
    }
}