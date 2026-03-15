#include <bits/stdc++.h>
using namespace std;

template <typename T>
class BlockingQueue {
    private : 
        size_t max_size;
        queue<T> q;
        condition_variable not_full;
        condition_variable not_empty;
        mutex mtx;
        bool closed = false;

    public :
        BlockingQueue(size_t cap) : max_size(cap) {}

        // delete copy and assignment default constructors to prevent threading disasters 
        BlockingQueue(const BlockingQueue&) = delete;
        BlockingQueue& operator=(const BlockingQueue&) = delete;

        void push(T item) {
            
            unique_lock<mutex> lock(mtx);
            
            not_full.wait(lock, [this]() {
                return q.size() < max_size || closed;
            });
            
            if(closed) return; // for proper shutdown

            q.push(move(item));
            not_empty.notify_one();
        }

        optional<T> pop() {

            unique_lock<mutex> lock(mtx);

            not_empty.wait(lock, [this](){
                return !q.empty() || closed;
            });

            if(q.empty()) return nullopt;

            T item = move(q.front()); // for high performance use move since the item can be very large this prevents unecessary copies
            q.pop();
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

int main() {
    size_t cap = 5;
    BlockingQueue<int> q(cap);

    vector<thread> producers, consumers;

    for(int i = 0; i < 3; i++) {

        producers.emplace_back([&q](){

            for (int i = 0; i < 10; i++) {
                q.push(i);
                string msg = "Pushed " + to_string(i) + "\n";
                cout << msg;

            }

        });

    }
    
    for(int i = 0; i < 3; i++) {

        consumers.emplace_back([&q]() {

            while(true) {

                optional<int> item = q.pop();
                if(!item.has_value()) break;
                
                string msg = "Recieved " + to_string(*item) + "\n";
                cout << msg;

            }

        });
    }
    

    for (auto &i : producers) i.join();
    q.shutdown();

    for (auto &i : consumers) i.join();
    return 0;
}
