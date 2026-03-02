#include <thread>
#include <vector> 
#include <iostream>
#include <atomic> 
#include <optional>
#include <string>

using namespace std;

template <typename T> 
struct Node {
    T data;
    atomic<size_t> sequence;
};

template <typename T>
class MPMCRingBufferQueue {

    private : 
        alignas(64) atomic<size_t> head{0};
        alignas(64) atomic<size_t> tail{0};
        size_t mask;
        vector<Node<T>> buffer;
    
    public : 
        MPMCRingBufferQueue(size_t size) : buffer(size), mask(size - 1) {
            for (size_t i = 0; i < size; ++i) {
                buffer[i].sequence.store(i, std::memory_order_relaxed);
            }
        }

        bool push(T item) {
            Node<T> *node = nullptr;
            size_t pos = head.load(memory_order_relaxed);

            for(;;) {
                node = &buffer[pos & mask];
                size_t seq = node->sequence.load(memory_order_acquire);
                intptr_t diff = (intptr_t)seq - (intptr_t)pos;

                if (diff == 0) {
                    if(head.compare_exchange_weak(pos, pos + 1, memory_order_relaxed))
                        break;
                } else if (diff < 0) {
                    return false; // yet to be consumed
                } else {
                    pos = head.load(memory_order_relaxed);
                }
            }

            node->data = item;
            node->sequence.store(pos + 1, memory_order_release);

            return true;
        }

        optional<T> pop() {
            size_t pos = tail.load(memory_order_relaxed);
            Node<T> *node = nullptr;

            for(;;) {
                node = &buffer[pos & mask];
                size_t seq = node->sequence.load(memory_order_acquire);
                intptr_t diff = (intptr_t)seq - (intptr_t)(pos + 1);

                if (diff == 0) {
                    if(tail.compare_exchange_weak(pos, pos + 1, memory_order_relaxed))
                        break;
                } else if (diff < 0) {
                    return nullopt;
                } else {
                    pos = tail.load(memory_order_relaxed);
                }
            }

            T item = move(node->data);
            node->sequence.store(pos + mask + 1, memory_order_release);
            return item;
        }
};

atomic<int> total_items{200};

void produce(MPMCRingBufferQueue<int> &buffer, int j) {
    for (int i = 1; i <= 100; i++) {
        int x = i * j;
        while(!buffer.push(x)) {}
        string msg = "Pushed " + to_string(x) + "\n";
        cout << msg;
    }
}

void consume(MPMCRingBufferQueue<int> &buffer) {
    while(total_items.load(memory_order_relaxed) > 0) {
        optional<int> item = buffer.pop();
        if(item.has_value()) {
            string msg = "Consumed " + to_string(*item) + "\n";
            cout << msg;
            total_items.fetch_sub(1);
        } else {
            this_thread::yield();
        }
    }
}

int main() {
    MPMCRingBufferQueue<int> buffer(8);
    thread producer_1(produce, ref(buffer), 1), 
           producer_2(produce, ref(buffer), 2), 
           consumer_1(consume, ref(buffer)), 
           consumer_2(consume, ref(buffer));
           
    producer_1.join(); 
    producer_2.join(); 
    consumer_1.join(); 
    consumer_2.join();
    
    return 0;
}