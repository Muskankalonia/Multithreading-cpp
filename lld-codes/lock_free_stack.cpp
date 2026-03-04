#include <iostream>
#include <thread>
#include <atomic>
#include <optional>
#include <vector>
#include <memory>

using namespace std;

// FIX: Use atomic<shared_ptr<Node>> instead of atomic<Node*>
//
// ABA Problem recap:
//   Thread 1 reads head = A (A->next = B)
//   Thread 1 is preempted
//   Thread 2 pops A, pops B, pushes new node at same address as A
//   Thread 1 resumes: CAS sees head == A → succeeds, sets head = B (freed!)
//
// Why shared_ptr fixes it:
//   Every make_shared() creates a UNIQUE control block.
//   CAS on atomic<shared_ptr> compares both raw ptr + control block.
//   Even if a new node lands at the same address, different control block
//   → CAS correctly detects the change and retries.
//   Ref counting ensures no node is freed while any thread holds a reference.

template <typename T>
class LockFreeStack {
private:
    struct Node {
        T data;
        shared_ptr<Node> next;  // shared_ptr chain — safe to read without locks

        Node(T data) : data(std::move(data)), next(nullptr) {}
    };

    atomic<shared_ptr<Node>> head{nullptr};

public:
    void push(T val) {
        auto new_node = make_shared<Node>(std::move(val));
        new_node->next = head.load(memory_order_relaxed);

        // CAS: atomically swing head to new_node
        // If head changed since we read it, CAS fails and updates new_node->next
        while (!head.compare_exchange_weak(
            new_node->next,
            new_node,
            memory_order_release,
            memory_order_relaxed
        ));
    }

    optional<T> pop() {
        shared_ptr<Node> old_head = head.load(memory_order_acquire);

        // CAS: atomically swing head to old_head->next
        // shared_ptr refcount keeps old_head alive even if another thread pops it
        while (old_head && !head.compare_exchange_weak(
            old_head,
            old_head->next,       // safe: old_head is kept alive by our local shared_ptr
            memory_order_acquire,
            memory_order_relaxed
        ));

        if (!old_head) return nullopt;

        T val = std::move(old_head->data);
        // old_head goes out of scope here → refcount drops → auto freed if no other refs
        return val;
    }

    // No destructor needed — shared_ptr chain cleans itself up
};

int main() {
    LockFreeStack<int> ls;
    constexpr int NUM_THREAD = 4;
    constexpr int OPS_PER_THREAD = 100;

    vector<thread> producers;
    for (int i = 0; i < NUM_THREAD; i++) {
        producers.emplace_back([&ls]() {
            for (int j = 0; j < OPS_PER_THREAD; j++) {
                ls.push(j);
                string msg = "Pushed " + to_string(j) + "\n";
                cout << msg;
            }
        });
    }

    vector<thread> consumers;
    for (int i = 0; i < NUM_THREAD; i++) {
        consumers.emplace_back([&ls]() {
            for (int j = 0; j < OPS_PER_THREAD; j++) {
                optional<int> item = ls.pop();
                if (item.has_value()) {
                    string msg = "Consumed " + to_string(*item) + "\n";
                    cout << msg;
                }
            }
        });
    }

    for (auto& i : producers) i.join();
    for (auto& i : consumers) i.join();
}
