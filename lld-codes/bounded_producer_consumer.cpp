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

    public :
        BlockingQueue(size_t cap) : max_size(cap) {}

        // delete copy and assignment default constructors to prevent threading disasters 
        BlockingQueue(const BlockingQueue&) = delete;
        BlockingQueue& operator=(const BlockingQueue&) = delete;

        void push(T item) {
            
            unique_lock<mutex> lock(mtx);
            
            not_full.wait(lock, [this]() {
                return q.size() < max_size;
            });
            
            q.push(move(item));
            not_empty.notify_one();
        }

        T pop() {

            unique_lock<mutex> lock(mtx);

            not_empty.wait(lock, [this](){
                return q.size() != 0;
            });

            T item = move(q.front()); // for high performance use move since the item can be very large this prevents unecessary copies
            q.pop();
            not_full.notify_one();

            return item;
        }

};

int main() {
    size_t cap = 5;
    BlockingQueue<int> q(cap);

    thread producer([&q](){
        for (int i = 0; i < 10; i++) {
            q.push(i);
            cout << "Pushed " << i << "\n";
        }
    });

    thread consumer([&q](){
        for (int i = 0; i < 10; i++) {
            int item = q.pop();
            cout << "Recieved " << item << "\n"; // don't use endl here because it is less efficient
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    });

    producer.join(); consumer.join();
    return 0;
}
