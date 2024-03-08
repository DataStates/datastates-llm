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
                DBG("---- Returning from d2h thread " << _rank);
                return;
            }
            TIMER_START(d2h_time);
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
            // TIMER_STOP(d2h_time, "[D2H][" << _rank << "] Total time for GPU to process " << m->uid << " version " << version, size);
            // DBG("[D2H][" << _rank << "] transfer of tensor " << uid  << " version " << version);
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
            TIMER_START(h2f_wait);
            while(_pending_h2f.empty() && is_active)
                _cv_h2f.wait(_lock_h2f);
            if (!is_active) {
                _lock_h2f.unlock();
                _cv_h2f.notify_all();
                DBG("---- Returning from h2f thread " << _rank);
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
            DBG("[H2F][" << _rank << "] flushed for tensor uid " << uid  << " version " << version);
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
            DBG("[" << _rank << "] Enqueuing GPU tensor " << uid << " version  " << version << " size " << size << " at file offset " << file_offset);
            std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
            _pending_d2h.push_back(std::make_tuple(version, uid, t, size, file_offset, path));
            _lock_d2h.unlock();
            _cv_d2h.notify_all();
            return;
        } 
        DBG("[" << _rank << "] Enqueuing host tensor " << uid << " version  " << version << " size " << size);
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

void datastates_llm_t::wait() {
    try {
        TIMER_START(wait_timer);
        std::unique_lock<std::mutex> _lock_d2h(_mutex_d2h);
        while(!(_pending_d2h.empty())) {
            DBG("[" << _rank << "] Waiting in d2h for " << _pending_d2h.size());
            for(auto e: _pending_d2h) {
                DBG("[" << _rank << "] D2H_WAIT " << std::get<0>(e) << " UID " << std::get<1>(e) << " size " << std::get<4>(e));
            }
            _cv_d2h.wait(_lock_d2h);
        }
        DBG("[" << _rank << "] Waiting complete in d2h now size " << _pending_d2h.size());
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
        DBG("[" << _rank << "]" << "VELOC shutdown starting");
        wait();
        DBG("[" << _rank << "]" << "VELOC shutdown-wait done");
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
        DBG("[" << _rank << "]" << "VELOC shutdown-file-write done");
        mem->shutdown();
        DBG("[" << _rank << "]" << "VELOC shutdown-mem done");
        _cv_h2f.notify_all();
        _cv_d2h.notify_all();
        // DBG("[" << _rank << "]" << "VELOC joining d2h thread");
        // _thread_d2h.join();
        // DBG("[" << _rank << "]" << "VELOC shutdown-d2h thread done");
        // _thread_h2f.join();
        DBG("[" << _rank << "]" << "VELOC shutdown-h2f thread done");
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in shutdown." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in shutdown.");
    }
}