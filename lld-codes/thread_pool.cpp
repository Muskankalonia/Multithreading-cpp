#include <bits/stdc++.h>
using namespace std;

template <typename T> 
class BlockingQueue {
    private :
        queue<T> items;
        mutex mtx;
        condition_variable not_empty;
        condition_variable not_full;
        size_t max_cap;
        bool closed = false;
    
    public : 
        BlockingQueue(size_t size) : max_cap(size) {}

        void push(T item) {

            unique_lock<mutex> lock(mtx);
            not_full.wait(lock, [this](){
                return items.size() != max_cap || closed;
            });

            if(closed) return;

            items.push(move(item));
            not_empty.notify_one();

        }

        optional<T> pop() {

            unique_lock<mutex> lock(mtx);
            not_empty.wait(lock, [this]() {
                return !items.empty() || closed;
            });

            if(items.empty()) return nullopt;

            T item = move(items.front());
            items.pop();

            not_full.notify_one();
            return item;

        }

        void shutdown() {
            {
                unique_lock<mutex> lock(mtx);
                closed = true;
            }

            not_empty.notify_all();
            not_full.notify_all();

        }
};


class ThreadPool {
    private :
        BlockingQueue<function<void()>> tasks;
        vector<thread> workers;

    public : 
        ThreadPool(size_t num_threads, size_t max_cap) : tasks(max_cap) {

            for (int i = 0; i < (int)num_threads; i++) {

                workers.emplace_back([this]() {
                    while(true) {
                        
                        auto task = tasks.pop();
                        if(!task.has_value()) break;

                        (*task)();

                    }
                });

            }

        }

        void submitTask(function<void()> task) {
            tasks.push(move(task));
        }

        ~ThreadPool() {

            tasks.shutdown();
            for(auto &i : workers) i.join();

        }

};

int main() {
    //TODO -> How do you get a return value back from a submitted task?
    auto work = [](int i) {
        string msg = "Executed task " + to_string(i) + " in thread pool\n";
        cout << msg;
    };

    ThreadPool pool(3, 10);
    for(int i = 0; i < 10; i++) {
        pool.submitTask([work, i] {   // wrap it to make it void 
            work(i);
        });
    }

}

/*
Follow ups 
**1. What if tasks throws an exception? **
    Fix: wrap in try/catch, log the error, keep the thread alive.
    try {
        (*task)();
    } catch (const exception& e) {
        // log e.what() — don't rethrow, worker must stay alive
    }

**2. "How do you size the thread pool?"**
Classic answer:
- CPU-bound tasks → `std::thread::hardware_concurrency()` threads (= number of cores)
- I/O-bound tasks → `cores * 2` or `cores * (1 + wait_time/compute_time)`
- Mixed → profile and tune

**3. "What's work-stealing?"**
Each thread has its own local deque of tasks instead of one shared queue. When a thread's local deque is empty, it steals from the back of another thread's deque. Reduces contention on a single shared lock. `std::execution` and Intel TBB use this. You don't need to implement it but you should be able to explain why it's better than a single shared queue under high load.
*/