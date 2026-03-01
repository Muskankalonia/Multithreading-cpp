#include <iostream>
#include <atomic> 
#include <optional>
#include <thread>
#include <vector>
#include <string>

using namespace std;

template <typename T> 
class SPSCRingBuffer {
    private : 
        vector<T> buffer;
        // alignas(64) prevents "False Sharing" by putting indices on different cache lines
        alignas(64) atomic<size_t> head{0};
        alignas(64) atomic<size_t> tail{0};
        size_t size;

    public : 
        SPSCRingBuffer(size_t size) : buffer(size), size(size) {}

        bool push(T item) {
            int cur_head = head.load(memory_order_relaxed);
            int next_head = (cur_head + 1) % size;

            if (next_head == tail.load(memory_order_acquire)) {
                return false; // queue is full 
            }

            buffer[cur_head] = move(item);
            head.store(next_head, memory_order_release);
            return true;
        }

        optional<T> pop() {
            int cur_tail = tail.load(memory_order_relaxed);
            
            if (cur_tail == head.load(memory_order_acquire)) {
                return nullopt; // queue is empty
            }

            T value = move(buffer[cur_tail]); // only use move when there is single consumer
            tail.store((tail + 1) % size, memory_order_release);
            return value;
        }
};

int main() {
    SPSCRingBuffer<int> buffer(10);

    thread producer([&buffer](){
        for (int i = 0; i < 100; i++) {
            while(!buffer.push(i)){} // spin lock for low latency if we know the consumer will quickly consume the message
            string msg = "Pushed " + to_string(i) + "\n"; // to reduce context switch during printing a msg, for gurantee use osyncstream instead
            cout << msg;
        }
    });

    thread consumer([&buffer](){
        for(int i = 0; i < 100; i++) {
            optional<int> val;
            do {
                val = buffer.pop();
            } while(!val.has_value());
            string msg = "Popped " + to_string(i) + "\n";
            cout << msg;
        }
    });

    producer.join(); consumer.join();

}