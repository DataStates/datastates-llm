#ifndef __DATASTATES_ATOMIC_QUEUE_HPP
#define __DATASTATES_ATOMIC_QUEUE_HPP

#include "defs.hpp"
#include "utils.hpp"
#include <mutex>
#include <atomic>
#include <deque>
#include <condition_variable>

namespace datastates {

template <typename T>
class atomic_queue_t {
    std::deque<T> q;
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<bool> is_active = true;
public:
    atomic_queue_t() {};
    ~atomic_queue_t() {};
    void push(T src) {
        std::unique_lock<std::mutex> lck(mtx);
        q.push_back(src);
        lck.unlock();
        cv.notify_all();
    };
    T get_front() {
        std::unique_lock<std::mutex> lck(mtx);
        T e = q.front();
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
        try {
            std::unique_lock<std::mutex> lck(mtx);
            while(q.size() != 0)
                cv.wait(lck);
            lck.unlock();
            cv.notify_all();
        } catch (std::exception& e) {
            FATAL("Exception caught in wait_for_completion: " << e.what());
        } catch (...) {
            FATAL("Unknown exception caught in wait_for_completion.");
        }
    }
    void set_inactive() {
        wait_for_completion();
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

}

#endif //__DATASTATES_ATOMIC_QUEUE_HPP