// implementing semaphores using mutex and condition variables 

#include <bits/stdc++.h>
using namespace std;

class semaphores {
    private : 
        mutex mtx;
        condition_variable cv;
        int cnt;

    public : 
        explicit semaphores(int initial_count) : cnt(initial_count) {}

        semaphores(const semaphores&) = delete;
        semaphores& operator=(const semaphores&) = delete;

        void acquire() {

            unique_lock<mutex> lock(mtx);
            cv.wait(lock, [this]() {
                return cnt > 0;
            });

            cnt--;

        }

        void release() {
            {
                unique_lock<mutex> lock(mtx);
                cnt++;
            }
            
            cv.notify_one();
        }

};

class LockGuard {
    semaphores &sm;

    public : 
        explicit LockGuard(semaphores &sm) : sm(sm) {sm.acquire();}
        ~LockGuard() {sm.release();}

        LockGuard(const LockGuard&) = delete;
        LockGuard& operator=(const LockGuard&) = delete;
};

int main() {

    semaphores binary(1);
    
    vector<thread> writes;
    int c = 5;

    for(int i = 0; i < 3; i++) {

        writes.emplace_back([&c, &binary]() {
            
            LockGuard guard(binary);
            
            string msg = "Updated val of c " + to_string(++c) + "\n";
            cout << msg;

        });

    }

    for(auto &i : writes) i.join();
}


/*
Follow ups : 
1. "What's the difference between a semaphore and a mutex?"
MutexBinary vs semaphore :
a. Ownership : In mutex the thread that locks must unlock whereas in semaphore any thread can release
b. Signaling : In mutex No signalling — only mutual exclusion whereas in semaphores yes — one thread signals another
c. Use case : In mutex Protecting shared state whereas in semaphores Coordinating between threads like in blocking producer consumer queue 
The ownership distinction is the key one. A mutex acquired by thread A can only be released by thread A. A semaphore released by thread B can unblock thread A. This makes semaphores useful for producer-consumer signaling, not just mutual exclusion.
2. "Can count go negative? What would that mean?"
In the standard model — no, your predicate count > 0 prevents that. But there's an alternative formulation where count can go negative and the absolute value represents the number of waiters. That's how some OS kernels implement it.
*/