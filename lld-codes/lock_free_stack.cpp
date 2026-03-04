#include <iostream>
#include <thread>
#include <atomic>
#include <optional>
#include <vector>

using namespace std;

template <typename T> 
class LockFreeStack {
    private :
        struct Node {
            T data;
            Node *next;

            Node(T data) : data(data), next(nullptr) {}
        };
        atomic<Node*> head{nullptr};

    public : 
        void push(T val) {
            Node *new_head = new Node(val);

            do {
                new_head->next = head.load(memory_order_relaxed) ;
            } while(!head.compare_exchange_weak(
                new_head->next, 
                new_head, 
                memory_order_release, 
                memory_order_relaxed));
        }

        optional<T> pop() {
            Node *old_head = head.load(memory_order_acquire);

            while(old_head && !head.compare_exchange_weak(
                old_head,
                old_head->next,
                memory_order_release,
                memory_order_relaxed
            ))

            if (!old_head) return nullopt;

            T val = old_head->data;
            delete old_head; // ABA problem 
            return val;
        }

        ~LockFreeStack() {
            Node *node = head.load();
            while(node) {
                Node *next = node->next;
                delete node;
                node = next;
            }
        }
};

int main() {
    LockFreeStack<int> ls;
    constexpr int NUM_THREAD = 4;
    constexpr int OPS_PER_THREAD = 100;

    vector<thread> producers;
    for (int i = 0; i < NUM_THREAD; i++) {
        producers.emplace_back([&ls](){
            for(int j = 0; j < OPS_PER_THREAD; j++) {
                ls.push(j);
                string msg = "Pushed " + to_string(j) + "\n";
                cout << msg;
            }
        });
    }

    vector<thread> consumers;
    for (int i = 0; i < NUM_THREAD; i++) {
        consumers.emplace_back([&ls](){
            for(int j = 0; j < OPS_PER_THREAD; j++) {
                optional<int> item = ls.pop();
                if(item.has_value()) {
                    string msg = "Consumed " + to_string(*item) + "\n";
                    cout << msg;
                }
            }
        });
    }

    for (auto &i : producers) i.join();
    for (auto &i : consumers) i.join();
}