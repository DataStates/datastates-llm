#ifndef __DATASTATES_POOL_ALLOCATOR_HPP
#define __DATASTATES_POOL_ALLOCATOR_HPP
#include <atomic>
#include <iostream>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <cassert>
#include "common/defs.hpp"
#include "common/mem_region.hpp"
#include "common/utils.hpp"
#include <rmm/detail/aligned.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>


template <typename T>
class mem_pool_t {
    TIER_TYPES tier_type_;
    std::atomic<size_t> total_size_{0};
    std::atomic<size_t> curr_size_{0};
    std::map<uint64_t, size_t> alloc_map_;
    rmm::mr::pool_memory_resource<T>* pool_mr;
    
    std::mutex mem_mutex_;
    std::condition_variable mem_cv_;
    bool is_active = true;
    int rank_ = -1;
public:
    mem_pool_t(TIER_TYPES tier_type, size_t total_size, int rank=-1): tier_type_(tier_type), total_size_(total_size), rank_(rank) {
        try {
            T* mr_rmm = new T();
            pool_mr = new rmm::mr::pool_memory_resource<T>(mr_rmm, total_size, total_size);
            is_active = true;
            DBG("Returned from the memory pool function");
        } catch (std::exception &e) {
            FATAL("Exception caught in memory pool constructor." << e.what());
        }
    };

    ~mem_pool_t() {
        try {
            is_active = false;
            mem_cv_.notify_all();
            delete pool_mr;
            return;
        } catch (std::exception &e) {
            FATAL("Exception caught in memory pool destructor." << e.what());
        }
    };

    void allocate(mem_region_t* m) {
        try {
            if (m->size > total_size_) 
                FATAL("[" << rank_ << "]" <<"Cannot allocate size " << m->size << " larger than the pool of " << total_size_);
            m->ptr = nullptr;
            DBG("[" << rank_ << "]" << "Attempting to allocate for " << m->uid << " of size " << m->size << " when current memory is " << curr_size_);
            std::unique_lock<std::mutex> mem_lock_(mem_mutex_);
            while((curr_size_ + m->size > total_size_) && is_active)
                mem_cv_.wait(mem_lock_);
            if (!is_active) {
                mem_lock_.unlock();
                mem_cv_.notify_all();
                return;
            }
            m->ptr = (char *)pool_mr->allocate(m->size);
            alloc_map_[m->uid] = m->size;
            curr_size_ += m->size;
            mem_lock_.unlock();
            mem_cv_.notify_all();
            DBG("[" << rank_ << "]" << "Allocated for " << m->uid << " of size " << m->size << " when current memory is " << curr_size_);
            print_stat();
        } catch (std::exception &e) {
            FATAL("Exception caught in allocate function." << e.what());
        }
    };

    void print_stat() {
        size_t total_size = 0;
        DBG("=======================");
        for (auto const& [key, val] : alloc_map_) {
            DBG("ID " << key << " size " << val);
            total_size += val;
        }
        DBG("Total size " << total_size << " curr_size_ " << curr_size_);
        DBG("=======================");
    }

    size_t get_capacity() {
        return total_size_;
    };
    
    void deallocate(mem_region_t* m) {
        try {
            DBG("[" << rank_ << "]" << "Attempting to deallocate " << m->uid << " of size " << m->size << " cur size " << curr_size_);
            std::unique_lock<std::mutex> mem_lock_(mem_mutex_);
            pool_mr->deallocate(m->ptr, m->size);
            curr_size_ -= m->size;
            alloc_map_[m->uid] = 0;
            DBG("[" << rank_ << "]" << "deallocated " << m->uid << " of size " << m->size << " cur size " << curr_size_);
            mem_lock_.unlock();
            mem_cv_.notify_all();
            print_stat();
        } catch (std::exception &e) {
            FATAL("Exception caught in deallocate operation ." << e.what());
        }
    }
};


#endif //__DATASTATES_POOL_ALLOCATOR_HPP