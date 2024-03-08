#include "host_cache.hpp"

host_cache_t::host_cache_t(size_t t, int d, int r): _device_id(d), _total_memory(t), _curr_size(0), _head(0), _tail(0), _rank(r) {
    try {
        is_active = true;
        checkCuda(cudaMallocHost(&_start_ptr, _total_memory))
        max_allocated = 0;
        DBG("Returned from the memory cache function");
    } catch (std::exception &e) {
        FATAL("Exception caught in memory cache constructor." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in memory cache constructor.");
    }
}

host_cache_t::~host_cache_t() {
    try {
        is_active = false;
        _mem_cv.notify_all();
        checkCuda(cudaFreeHost(_start_ptr));
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in memory cache destructor." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in memory cache destructor.");
    }
}

void host_cache_t::shutdown() {
    try {
        std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
        is_active = false;
        _mem_lock.unlock();
        _mem_cv.notify_all();
        DBG("[" << _rank << "]" << "Memory cache shutdown starting");
        DBG("[" << _rank << "]" <<"Memory cache shutdown complete");
    } catch (std::exception &e) {
        FATAL("Exception caught in memory cache destructor." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in memory cache destructor.");
    }
}

mem_region_t* host_cache_t::_assign(const uint64_t uid, size_t h, size_t s) {
    try {
        if (h+s > _total_memory) {
            std::runtime_error("Exception in assign exceeding total memory size");
            // _print_trace();
            std::abort();
        }
        char *ptr = _start_ptr + h;
        mem_region_t *m = new mem_region_t(uid, ptr, h, h+s);
        _head = h + s;
        if (_head > _total_memory)
            _head = 0;
        _curr_size += s;
        _mem_q.push_back(m);
        // _print_trace();
        DBG("[" << _rank << "]" << "Assigned " << uid << " head " << h << " of size " << s << " curr size " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
        return m;
    } catch (std::exception &e) {
        FATAL("Exception caught in _assign." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in _assign.");
    }
}

mem_region_t* host_cache_t::allocate(const uint64_t uid, size_t s) {
    try {
        DBG("[" << _rank << "]" << "Attempting to allocate for " << uid << " of size " << s << " when current memory is " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
        if (s > _total_memory) 
            FATAL("[" << _rank << "]" <<"Cannot allocate size " << s << " larger than the pool of " << _total_memory);
        mem_region_t* ptr = nullptr;
        std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
        while((_curr_size + s > _total_memory) && is_active)
            _mem_cv.wait(_mem_lock);
        if (!is_active) {
            _mem_lock.unlock();
            _mem_cv.notify_all();
            return ptr;
        }
        if (_tail == _head)
            _tail = _head = 0;
        if (_tail <= _head) {
            if (_total_memory - _head >= s) {
                ptr = _assign(uid, _head, s);
            } else {
                _head = 0;
            }
        } 
        if (ptr == nullptr) {
            // Now the tail is greater than head
            while(((_tail > _head) && (_tail - _head < s)) && is_active) {
                DBG("[" << _rank << "]" << "Waiting in second wait for " << uid << " _tail -head < s, head " << _head << " tail: " << _tail << " s " << s);
                TIMER_START(wait_time);
                _mem_cv.wait(_mem_lock);
                TIMER_STOP(wait_time, "[" << _rank << "]" << "Waiting for more memory " << _tail - _head << " uid " << uid << " size " << s << " head " << _head << " tail " << _tail, 10);
            }
            // This may happen when deallocate resets the tail pointer to 0 when tail+dealloc_size > max_buffer_cap
            if (_tail <= _head) {
                _mem_lock.unlock();
                _mem_cv.notify_all();
                return allocate(uid, s);
            }
            if (!is_active) {
                DBG("Returning from allocate function.....");
                _mem_lock.unlock();
                _mem_cv.notify_all();
                return nullptr;
            }
            ptr = _assign(uid, _head, s);
        }
        _mem_lock.unlock();
        _mem_cv.notify_all();
        DBG("[" << _rank << "]" << "Allocated for " << uid << " of size " << s << " when current memory is " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
        return ptr;
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate function." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in allocate function.");
    }
}

void host_cache_t::deallocate(uint64_t _uid, size_t s) {
    try {
        DBG("[" << _rank << "]" << "Attempting to deallocate " << _uid << " of size " << s << " cur size " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
        if (_mem_q.empty() || _uid < 1)
            return;
        mem_region_t *m = _mem_q.front();
        if (_uid != m->uid || s != (m->end_offset-m->start_offset)) {
            std::cout << "Should deallocate the tail first. Only FIFO eviction allowed" << std::endl;
            std::cout << "Tried deleting " << _uid << " of size " << s <<  " at offset " << m->start_offset 
                << " but front element was " << (void *)m->ptr << " of size " << m->end_offset-m->start_offset << std::endl;
            _print_trace();
            return;
        }
        std::unique_lock<std::mutex> _mem_lock(_mem_mutex);
        _tail += s;
        if (_tail > _total_memory)
            _tail = 0;
        _curr_size -= s;
        if (_curr_size == 0)
            _head = _tail = 0;
        _mem_q.pop_front();
        _mem_lock.unlock();
        _mem_cv.notify_all();
        DBG("[" << _rank << "]" <<"Deallocated from host " << _uid << " of size " << s << " cur size " << _curr_size << " cur head " << _head  << " cur tail " << _tail);
    } catch (std::exception &e) {
        FATAL("Exception caught in deallocate operation ." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in deallocate operation.");
    }
}

void host_cache_t::_print_trace() {
    try {
        DBG("===================================================");
        for (size_t i = 0; i < _mem_q.size(); ++i) {
            const auto e = _mem_q[i];
            DBG(e->uid << (void*)e->ptr << " : " << e->start_offset << " - " << e->end_offset);
        }
        auto e = _mem_q.front();
        DBG("First element " << e->uid << " ptr " << (void *)e->ptr << " at start offset " << e->start_offset);
        DBG("Head " << _head << ", Tail " << _tail);
        DBG("===================================================");
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate _print_trace." << e.what());
    } catch (...) {
        FATAL("Unknown exception caught in _print_trace.");
    }
}