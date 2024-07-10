#include "mem_pool.hpp"

mem_pool_t::mem_pool_t(char* start_ptr, size_t total_size, int rank): start_ptr_(start_ptr), 
    total_size_(total_size), rank_(rank) {
    try {
        cudaPointerAttributes attributes;
        checkCuda(cudaPointerGetAttributes (&attributes, start_ptr_));
        device_type_ = attributes.type;
        is_active = true;
        DBG("Returned from the memory pool function");
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
        FATAL("Exception caught in memory pool destructor." << e.what());
    }
}

size_t mem_pool_t::get_free_size() {
    return curr_size_;
}

size_t mem_pool_t::get_capacity() {
    return total_size_;
}

void mem_pool_t::assign_(mem_region_t* m) {
    try {
        if (head_+m->size > total_size_) 
            FATAL("Exception in assign: exceeding total memory size");
        m->ptr = start_ptr_ + head_;
        head_ += m->size;
        if (head_ > total_size_)
            head_ = 0;
        curr_size_ += m->size;
        mem_q_.push_back(m);
        DBG("[" << rank_ << "]" << "Assigned " << m->uid << " of size " << m->size << " curr size " << curr_size_ << " cur head " << head_  << " cur tail " << tail_);
    } catch (std::exception &e) {
        FATAL("Exception caught in assign_." << e.what());
    }
}

void mem_pool_t::allocate(mem_region_t* m) {
    try {
        if (m->size > total_size_) 
            FATAL("[" << rank_ << "]" <<"Cannot allocate size " << m->size << " larger than the pool of " << total_size_);
        m->ptr = nullptr;
        DBG("[" << rank_ << "]" << "Attempting to allocate for " << m->uid << " of size " << m->size << " when current memory is " << curr_size_ << " cur head " << head_  << " cur tail " << tail_);
        std::unique_lock<std::mutex> mem_lock_(mem_mutex_);
        while((curr_size_ + m->size > total_size_) && is_active)
            mem_cv_.wait(mem_lock_);
        if (!is_active) {
            mem_lock_.unlock();
            mem_cv_.notify_all();
            return;
        }
        if (tail_ == head_)
            tail_ = head_ = 0;
        if (tail_ <= head_) {
            if (total_size_ - head_ >= m->size)
                // The gap at the end of the buffer is enough to hold the incoming data.
                assign_(m);
            else
                // Simplify the design: cannot partially write in the last parts of circular buffer, start from 0.
                head_ = 0;
        } 
        if (m->ptr == nullptr) {
            // tail_ > head_ when we start writing from head_=0.
            while(((tail_ > head_) && (tail_ - head_ < m->size)) && is_active) {
                mem_cv_.wait(mem_lock_);
            }
            // Happens when deallocate function resets the tail pointer to 0 when tail+dealloc_size > max_buffer_cap
            if (tail_ <= head_) {
                mem_lock_.unlock();
                mem_cv_.notify_all();
                return allocate(m);
            }
            if (!is_active) {
                mem_lock_.unlock();
                mem_cv_.notify_all();
                return;
            }
            assign_(m);
        }
        mem_lock_.unlock();
        mem_cv_.notify_all();
        DBG("[" << rank_ << "]" << "Allocated for " << m->uid << " of size " << m->size << " when current memory is " << curr_size_ << " cur head " << head_  << " cur tail " << tail_);
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate function." << e.what());
    }
}

void mem_pool_t::deallocate(mem_region_t* m) {
    try {
        if (get_capacity() <= 0)
            return;
        DBG("[" << rank_ << "]" << "Attempting to deallocate " << m->uid << " of size " << m->size << " cur size " << curr_size_ << " cur head " << head_  << " cur tail " << tail_);
        if (mem_q_.empty() || m->uid < 1)
            return;
        mem_region_t *top_m = mem_q_.front();
        if (m->uid != top_m->uid) {
            print_trace_();
            FATAL("Should deallocate the tail first. Only FIFO eviction. Tried deleting " << m->uid << " but front element was " << top_m->uid);            
            return;
        }
        std::unique_lock<std::mutex> mem_lock_(mem_mutex_);
        tail_ += m->size;
        if (tail_ > total_size_)
            tail_ = 0;
        curr_size_ -= m->size;
        if (curr_size_ == 0)
            head_ = tail_ = 0;
        DBG("[" << rank_ << "]" << "deallocated " << m->uid << " of size " << m->size << " cur size " << curr_size_ << " cur head " << head_  << " cur tail " << tail_);
        mem_q_.pop_front();
        mem_lock_.unlock();
        mem_cv_.notify_all();
    } catch (std::exception &e) {
        FATAL("Exception caught in deallocate operation ." << e.what());
    }
}

void mem_pool_t::print_trace_() {
    try {
        std::cout << "===================================================" << std::endl;
        for (size_t i = 0; i < mem_q_.size(); ++i) {
            const auto e = mem_q_[i];
            std::cout << "UID: " << e->uid << " ptr: " << (void*)e->ptr << " start: " << e->file_start_offset << " end: " << e->file_start_offset+e->size << std::endl;
        }
        auto e = mem_q_.front();
        std::cout <<  "First element " << e->uid << " ptr " << (void *)e->ptr << " at start offset " << e->file_start_offset << std::endl;
        std::cout << "Head " << head_ << ", Tail " << tail_ << std::endl;
        std::cout << "===================================================" << std::endl;
    } catch (std::exception &e) {
        FATAL("Exception caught in allocate print_trace_." << e.what());
    }
}