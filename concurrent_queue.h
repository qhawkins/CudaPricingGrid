// ConcurrentQueue.h
#ifndef CONCURRENT_QUEUE_H
#define CONCURRENT_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ConcurrentQueue {
public:
    // Push a copy of the value into the queue
    void push(const T& value){
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(value);
        cond_var_.notify_one();
    }
    
    // Push an rvalue into the queue
    void push(T&& value){
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
        cond_var_.notify_one();
    }
    
    // Try to pop a value; returns false if the queue is empty
    bool pop(T& result){
        std::unique_lock<std::mutex> lock(mutex_);
        if(queue_.empty())
            return false;
        result = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    // Wait and pop a value; blocks until a value is available
    void wait_and_pop(T& result){
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]{ return !queue_.empty(); });
        result = std::move(queue_.front());
        queue_.pop();
    }
    
    // Check if the queue is empty
    bool empty() const{
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_var_;
};

#endif // CONCURRENT_QUEUE_H
