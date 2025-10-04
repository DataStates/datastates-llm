#include "mem_pool.hpp"
using namespace datastates;

mem_pool_t::mem_pool_t(char* start_ptr, size_t total_size, int gpu_id, int rank, TIER_TYPES device_type): 
    start_ptr_(start_ptr), total_size_(total_size), gpu_id_(gpu_id), rank_(rank), device_type_(device_type) {
    try {
        if (total_size <= 0 || total_size > std::numeric_limits<size_t>::max()) {
            FATAL("Total size of memory pool " << total_size_ << " must be greater than zero. Use 1 byte if not using this pool on tier " << device_type_);
        }
        cudaPointerAttributes attributes;
        checkCuda(cudaPointerGetAttributes (&attributes, start_ptr_));
        if (device_type_ != attributes.type) {
            FATAL("The device type of the memory pool " << device_type_ << " does not match the pointer type " << attributes.type);
        }
        if (!is_aligned(reinterpret_cast<uintptr_t>(start_ptr_))) {
            FATAL("The start pointer of the memory pool " << reinterpret_cast<void*>(start_ptr_) 
                << " is not aligned to " << get_fs_block_alignment() << " bytes.");
        }
        is_active = true;
        DBG("Returned from the memory pool function on tier " << device_type_);
    } catch (std::exception &e) {
        FATAL("Exception caught in memory pool constructor." << e.what());
    }
}

mem_pool_t::~mem_pool_t() {
    try {
        is_active = false;
        mem_cv_.notify_all();
        mem_q_.clear();
        return;
    } catch (std::exception &e) {
        FATAL("Exception caught in memory pool destructor." << e.what() << " on tier " << device_type_);
    }
}

size_t mem_pool_t::get_free_size() {
    return curr_size_;
}

size_t mem_pool_t::get_capacity() {
    return total_size_;
}

void mem_pool_t::assign_(std::shared_ptr<mem_region_t> m) {
    try {
        if (head_ + m->aligned_size > total_size_)
            FATAL("Exception in assign: exceeding total memory size on tier " << device_type_);
        m->ptr = start_ptr_ + head_;
        if (!is_aligned(head_)) {
            FATAL("The head pointer " << head_ 
                << " is not aligned to " << get_fs_block_alignment() << " bytes on tier " << device_type_);
        }
        if (!is_aligned(reinterpret_cast<uintptr_t>(m->ptr))) {
            FATAL("The pointer " << reinterpret_cast<void*>(m->ptr)
                << " is not aligned to " << get_fs_block_alignment() << " bytes on tier " << device_type_);
        }
        head_ += m->aligned_size;
        curr_size_ += m->aligned_size;
        alloc_map_[m->internal_uid] = m->aligned_size;
        mem_q_.push_back(m);
        DBG("[" << rank_ << "]" << "Assigned " << m->uid << " of size " << m->size << " aligned " 
            << m->aligned_size << " curr size " << curr_size_ << " cur head " << head_  << " cur tail " << tail_ << " on tier " << device_type_);
    } catch (std::exception &e) {
        FATAL("Exception caught in assign_." << e.what());
    }
}

void mem_pool_t::allocate(std::shared_ptr<mem_region_t> m) {
    try {
        if (m->size <= 0 || m->size > total_size_ || m->size > std::numeric_limits<size_t>::max()) {
            FATAL("[" << rank_ << "]" << "Invalid memory region size " << m->size << " on device " 
                << device_type_ << " capacity " << total_size_ << " tier type " << TIER_TYPE_NAMES[device_type_] << " for uid " << m->uid);
        }

        m->aligned_size = m->size;
        if (!is_aligned(m->size)) { // FS_BLOCK_SIZE_ALIGNMENT==1 means no alignment
            m->aligned_size = get_aligned_offset(m->size);
        }
        if (m->aligned_size > total_size_) {
            FATAL("[" << rank_ << "]" << "Cannot allocate size " << m->aligned_size << " aligned size " << m->aligned_size 
                << " larger than the pool of " << total_size_ << " on tier " << device_type_);
        }
        m->ptr = nullptr;
        std::unique_lock<std::mutex> mem_lock_(mem_mutex_);
        // This loop will repeatedly try to find a suitable contiguous block.
        // It will only wait on the condition variable if either A) there isn't enough
        // total free space, or B) the free space is fragmented.
        while (true) {
            if (!is_active) 
                return;
            while (curr_size_ + m->aligned_size > total_size_) {
                if (!is_active) 
                    return;
                mem_cv_.wait_for(mem_lock_, std::chrono::microseconds(WAIT_TIMEOUT_US));
            }

            // If buffer becomes empty, reset pointers.
            if (curr_size_ == 0) {
                head_ = tail_ = 0;
            }

            // Step 2: Attempt to find a CONTIGUOUS block for the allocation.
            
            // Case A: Buffer is NOT wrapped (tail is behind or at head).
            if (tail_ <= head_) {
                // Option A.1: Try to allocate in the space at the end of the buffer.
                if (total_size_ - head_ >= m->aligned_size) {
                    assign_(m);
                    break; // SUCCESS: Allocation complete, exit the while(true) loop.
                }
                // Option A.2: Not enough space at the end. Try to wrap around to the beginning.
                // This is only possible if the space from [0, tail) is large enough.
                else if (tail_ >= m->aligned_size) {
                    // This action creates the "abandoned fragment" you correctly identified.
                    head_ = 0;
                    assign_(m);
                    break; // SUCCESS: Allocation complete, exit the while(true) loop.
                }
            }
            // Case B: Buffer IS wrapped (head is behind tail).
            else { // head_ < tail_
                // The only contiguous free space is the block between head and tail.
                if (tail_ - head_ >= m->aligned_size) {
                    assign_(m);
                    break; // SUCCESS: Allocation complete, exit the while(true) loop.
                }
            }
            
            // Step 3: If we reach this point, it means Step 1 passed (enough TOTAL space) 
            // but Step 2 failed (no CONTIGUOUS block was large enough). This is the exact
            // fragmentation case you described. We must wait for a deallocation to change
            // the buffer layout, then the while(true) loop will retry the logic.
            mem_cv_.wait_for(mem_lock_, std::chrono::microseconds(WAIT_TIMEOUT_US));
        }


        mem_lock_.unlock();
        mem_cv_.notify_all();
        DBG("[" << rank_ << "]" << "Allocated for " << m->internal_uid << " of size " 
            << m->size << " aligned size " << m->aligned_size << " when current memory is " << curr_size_ << " cur head " << head_  << " cur tail " << tail_ << " on tier " << device_type_);
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate function." << e.what());
    }
}


void mem_pool_t::deallocate(std::shared_ptr<mem_region_t> m) {
    try {
        if (get_capacity() <= 0) {
            return;
        }
        if (alloc_map_.find(m->internal_uid) == alloc_map_.end()) {
            return;
        }
        if (mem_q_.empty() || m->uid < 0 || m->internal_uid < 0) {
            return;
        }
        // New deferred deallocation logic to avoid deadlocks due to out-of-order deallocations.
        std::unique_lock<std::mutex> mem_lock_(mem_mutex_);
        deferred_deallocations_.insert(m->internal_uid);
        // 2. Process the front of the queue. This loop allows for a cascade of deallocations
        // if multiple contiguous items at the front of the queue are now ready to be freed.
        while (!mem_q_.empty() && deferred_deallocations_.count(mem_q_.front()->internal_uid)) {
            // Get the region from the front of the queue (the true tail of the buffer).
            auto region_to_dealloc = mem_q_.front();
            // Perform the actual deallocation for this region.
            tail_ += region_to_dealloc->aligned_size;
            if (tail_ >= total_size_) {
                tail_ = 0;
            }
            curr_size_ -= region_to_dealloc->aligned_size;
            // Clean up tracking structures.
            alloc_map_.erase(region_to_dealloc->internal_uid);
            deferred_deallocations_.erase(region_to_dealloc->internal_uid);
            mem_q_.pop_front();
        }
        // 3. If the buffer is now empty, reset head and tail pointers.
        if (curr_size_ == 0) {
            head_ = tail_ = 0;
        }
        // Unlock and notify any waiting allocator threads that space may be available.
        mem_lock_.unlock();
        mem_cv_.notify_all();
    } catch (std::exception &e) {
        FATAL("Exception caught in deallocate operation ." << e.what());
    }
}

void mem_pool_t::print_trace_() {
    try {
        DBG("===================================================");
        for (size_t i = 0; i < mem_q_.size(); ++i) {
            const auto e = mem_q_[i];
            DBG("UID: " << e->uid << " internal UID: " << e->internal_uid << " ptr: " 
                << (void*)e->ptr << " start: " << e->file_start_offset << " end: " << e->file_start_offset+e->aligned_size 
                << " size: " << e->size << " aligned size: " << e->aligned_size << " on tier " << TIER_TYPE_NAMES[e->curr_tier_type]);
        }
        auto e = mem_q_.front();
        DBG("First element " << e->uid << " internal UID: " << e->internal_uid << " ptr " << (void *)e->ptr << " at start offset " << e->file_start_offset);
        DBG("Head " << head_ << ", Tail " << tail_ << " On tier " << device_type_);
        DBG("===================================================");
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate print_trace_." << e.what());
    }
}