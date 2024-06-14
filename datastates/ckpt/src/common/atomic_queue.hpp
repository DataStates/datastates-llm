#ifndef __DATASTATES_ATOMIC_QUEUE_HPP
#define __DATASTATES_ATOMIC_QUEUE_HPP

#include "mem_region.hpp"
#include "defs.hpp"
#include "utils.hpp"
#include <mutex>
#include <atomic>
#include <condition_variable>

class atomic_queue_t {
    std::deque<mem_region_t*> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> is_active = true;
public:
    atomic_queue_t() {};
    ~atomic_queue_t() {};
    void push(mem_region_t* src) {
        std::unique_lock<std::mutex> lck(mtx);
        q.push_back(src);
        lck.unlock();
        cv.notify_all();
    };
    mem_region_t* get_front() {
        std::unique_lock<std::mutex> lck(mtx);
        mem_region_t* e = q.front();
        lck.unlock();
        cv.notify_all();
        return e;
    };
    void pop() {
        std::unique_lock<std::mutex> lck(mtx);
        q.pop_front();
        lck.unlock();
        cv.notify_all();
    };
    void wait_for_completion() {
        std::unique_lock<std::mutex> lck(mtx);
        while(q.size() > 0)
            cv.wait(lck);
        lck.unlock();
        cv.notify_all();
    }
    void set_inactive() {
        std::unique_lock<std::mutex> lck(mtx);
        is_active = false;
        lck.unlock();
        cv.notify_all();
    };
    bool wait_for_item() {
        std::unique_lock<std::mutex> lck(mtx);
        while(q.empty() && is_active)
            cv.wait(lck);
        lck.unlock();
        cv.notify_all();
        return is_active;
    };
};

#endif //__DATASTATES_ATOMIC_QUEUE_HPP