# Tier 1 — Must Know Cold
## C++ Multithreading Interview Deep Dive

---

# 1. std::thread — Creation, Join, Detach, RAII

## Core Concepts

A `std::thread` represents a single thread of execution. The thread starts immediately upon construction.

### Thread Creation

```cpp
#include <thread>
#include <iostream>
#include <string>

// 1. Free function
void greet(int id, const std::string& msg) {
    std::cout << "Thread " << id << ": " << msg << "\n";
}

// 2. Functor (callable object)
struct Worker {
    void operator()(int n) {
        for (int i = 0; i < n; ++i)
            std::cout << "Working: " << i << "\n";
    }
};

// 3. Member function
class MyClass {
public:
    void run(int x) { std::cout << "Member: " << x << "\n"; }
};

int main() {
    // Free function
    std::thread t1(greet, 1, "hello");

    // Lambda
    std::thread t2([](int x) {
        std::cout << "Lambda: " << x << "\n";
    }, 42);

    // Functor
    std::thread t3(Worker{}, 5);

    // Member function — must pass object (pointer or ref via std::ref)
    MyClass obj;
    std::thread t4(&MyClass::run, &obj, 10);
    // OR: std::thread t4(&MyClass::run, std::ref(obj), 10);

    t1.join(); t2.join(); t3.join(); t4.join();
}
```

### join() vs detach()

```cpp
#include <thread>
#include <chrono>

void long_task() {
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

int main() {
    std::thread t(long_task);

    // join(): blocks until thread finishes. Main thread waits.
    // MUST call join() or detach() before thread object is destroyed.
    t.join();

    std::thread t2(long_task);
    // detach(): thread runs independently (daemon thread).
    // You lose all handle to it. Dangerous if thread uses local variables.
    t2.detach();
    // t2 is now not joinable — do NOT call join() after detach()

    // Checking if joinable
    std::thread t3(long_task);
    if (t3.joinable()) {
        t3.join();
    }
}
```

### Passing Arguments: Value vs Reference

```cpp
#include <thread>
#include <iostream>

void increment(int& val) { val++; }
void print_val(int val)  { std::cout << val << "\n"; }

int main() {
    int x = 0;

    // WRONG: threads copy arguments by default
    // std::thread t(increment, x); // compiles but x is copied, not referenced

    // CORRECT: use std::ref to pass by reference
    std::thread t(increment, std::ref(x));
    t.join();
    std::cout << x << "\n"; // prints 1
}
```

### RAII Thread Wrapper

The problem: if an exception is thrown between thread creation and `join()`, the thread destructor calls `std::terminate()`.

```cpp
#include <thread>
#include <stdexcept>

// Simple RAII guard — always joins on destruction
class ThreadGuard {
    std::thread t_;
public:
    explicit ThreadGuard(std::thread t) : t_(std::move(t)) {}
    ~ThreadGuard() {
        if (t_.joinable()) t_.join();
    }
    // Non-copyable, movable
    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
    ThreadGuard(ThreadGuard&&) = default;
};

void risky_work() { throw std::runtime_error("oops"); }

int main() {
    std::thread t([]{ /* some work */ });
    ThreadGuard guard(std::move(t)); // t is now managed

    try {
        risky_work(); // throws!
    } catch (...) {
        // guard's destructor will call t.join() safely
    }
    // guard destructor joins the thread here
}
```

### Hardware Concurrency

```cpp
unsigned int n = std::thread::hardware_concurrency();
// Returns number of concurrent threads supported (logical cores).
// Returns 0 if not computable.
std::cout << "Cores: " << n << "\n";
```

---

## Interview Q&A — std::thread

**Q1: What happens if a std::thread object is destroyed without join() or detach()?**
> `std::terminate()` is called, killing the program. This is why RAII wrappers are essential.

**Q2: What is the difference between join() and detach()?**
> `join()` blocks the calling thread until the target thread finishes and cleans up resources. `detach()` releases ownership — the thread runs independently and resources are freed automatically when it finishes. After `detach()`, the thread handle is no longer valid (not joinable).

**Q3: Can you call join() twice on the same thread?**
> No. After `join()` returns, `joinable()` returns false. Calling `join()` again throws `std::system_error`. Always check `joinable()` or use RAII.

**Q4: How do you pass a reference to a thread?**
> Use `std::ref()`. By default, thread arguments are copied. `std::thread t(func, std::ref(x))` passes `x` by reference. Be careful: `x` must outlive the thread.

**Q5: What is the difference between `std::thread` and `std::jthread` (C++20)?**
> `std::jthread` automatically joins in its destructor (no `std::terminate`) and supports cooperative cancellation via `std::stop_token`. It is the preferred replacement for `std::thread` in C++20.

**Q6: Why might passing a C-string literal to a thread be dangerous?**
```cpp
// DANGEROUS
std::thread t(func, "hello"); // pointer to string literal passed
// If func signature is void func(std::string), the conversion
// from const char* to std::string happens in the NEW thread.
// If the calling context has vanished, UB can occur.

// SAFE: explicitly construct std::string
std::thread t(func, std::string("hello"));
```

**Q7: How do you get the current thread's ID?**
```cpp
std::thread::id this_id = std::this_thread::get_id();
```

---

# 2. std::mutex + std::lock_guard / scoped_lock

## Core Concepts

A `mutex` (mutual exclusion) ensures only one thread accesses a critical section at a time.

### Basic Mutex Usage

```cpp
#include <mutex>
#include <thread>
#include <iostream>

std::mutex mtx;
int shared_counter = 0;

void increment(int n) {
    for (int i = 0; i < n; ++i) {
        mtx.lock();          // acquire
        ++shared_counter;
        mtx.unlock();        // release — MUST be called, even if exception!
    }
}

// Problem: if ++shared_counter throws, unlock() is never called -> deadlock
// Solution: RAII lock wrappers
```

### std::lock_guard — Simple RAII Lock

```cpp
void safe_increment(int n) {
    for (int i = 0; i < n; ++i) {
        std::lock_guard<std::mutex> lock(mtx); // locks on construction
        ++shared_counter;
    } // unlocks on destruction, even if exception is thrown
}
```

### std::scoped_lock (C++17) — Lock Multiple Mutexes

```cpp
#include <mutex>

std::mutex mtx_a, mtx_b;
int account_a = 1000, account_b = 500;

void transfer(int amount) {
    // Locks BOTH mutexes atomically — prevents deadlock
    std::scoped_lock lock(mtx_a, mtx_b); // deduced template args (C++17)
    account_a -= amount;
    account_b += amount;
} // both unlocked here

// Pre-C++17 equivalent using std::lock + std::lock_guard adopt_lock:
void transfer_old(int amount) {
    std::lock(mtx_a, mtx_b); // deadlock-safe simultaneous lock
    std::lock_guard<std::mutex> la(mtx_a, std::adopt_lock); // take ownership
    std::lock_guard<std::mutex> lb(mtx_b, std::adopt_lock);
    account_a -= amount;
    account_b += amount;
}
```

### Mutex Types

```cpp
#include <mutex>
#include <shared_mutex>

std::mutex             m1;  // basic, non-recursive
std::recursive_mutex   m2;  // same thread can lock multiple times (must unlock same count)
std::timed_mutex       m3;  // try_lock_for(), try_lock_until()
std::shared_mutex      m4;  // C++17: read-write lock (multiple readers OR one writer)
std::recursive_timed_mutex m5; // combination

// try_lock example
if (mtx.try_lock()) {
    // got the lock
    mtx.unlock();
} else {
    // didn't get it — do something else
}

// Timed mutex
if (m3.try_lock_for(std::chrono::milliseconds(100))) {
    // got the lock within 100ms
    m3.unlock();
}
```

### Full Example: Thread-safe Counter

```cpp
#include <mutex>
#include <thread>
#include <vector>
#include <iostream>

class Counter {
    int value_ = 0;
    mutable std::mutex mtx_; // mutable: can lock in const methods
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx_);
        ++value_;
    }
    int get() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return value_;
    }
};

int main() {
    Counter c;
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i)
        threads.emplace_back([&c]{ for (int j = 0; j < 1000; ++j) c.increment(); });
    for (auto& t : threads) t.join();
    std::cout << "Final: " << c.get() << "\n"; // Always 10000
}
```

---

## Interview Q&A — Mutex & Locks

**Q1: Why is `mutable` used on a mutex in a class?**
> `std::mutex::lock()` is not const. If you have a const method (like a getter) that needs to lock for thread safety, the mutex must be `mutable` so it can be modified even in const context.

**Q2: What is the difference between lock_guard and scoped_lock?**
> `lock_guard` locks exactly one mutex. `scoped_lock` (C++17) can lock zero or more mutexes atomically in a deadlock-safe manner. Prefer `scoped_lock` in new code.

**Q3: Can you unlock a lock_guard early?**
> No. `lock_guard` locks on construction and unlocks on destruction. Use `unique_lock` if you need to manually unlock early.

**Q4: What does `std::adopt_lock` do?**
> Tells `lock_guard`/`unique_lock` to assume the mutex is already locked — it will only unlock on destruction, not lock. Used after `std::lock()` to transfer ownership to RAII.

**Q5: What is `std::defer_lock`?**
> Tells `unique_lock` NOT to lock the mutex on construction. You lock manually later. Used when you want to create the `unique_lock` first, then lock at a controlled point.

---

# 3. Deadlock — Causes, std::lock(), Lock Ordering

## Core Concepts

A deadlock occurs when two or more threads are each waiting for a resource held by the other, forming a circular dependency.

### The Four Conditions (Coffman Conditions)

All four must be present for a deadlock to occur:
1. **Mutual Exclusion** — resources cannot be shared
2. **Hold and Wait** — thread holds one resource while waiting for another
3. **No Preemption** — resources cannot be forcibly taken
4. **Circular Wait** — Thread A waits for B's resource, B waits for A's resource

### Classic Deadlock Example

```cpp
#include <mutex>
#include <thread>

std::mutex mtx1, mtx2;

void thread_a() {
    std::lock_guard<std::mutex> lock1(mtx1); // acquires mtx1
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> lock2(mtx2); // waits for mtx2 — DEADLOCK
}

void thread_b() {
    std::lock_guard<std::mutex> lock2(mtx2); // acquires mtx2
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> lock1(mtx1); // waits for mtx1 — DEADLOCK
}
// Thread A holds mtx1, wants mtx2
// Thread B holds mtx2, wants mtx1 -> circular wait
```

### Fix 1: std::lock() — Deadlock-Free Simultaneous Locking

```cpp
void thread_a_fixed() {
    std::unique_lock<std::mutex> lock1(mtx1, std::defer_lock);
    std::unique_lock<std::mutex> lock2(mtx2, std::defer_lock);
    std::lock(lock1, lock2); // atomically acquires both, no deadlock
    // do work...
}

void thread_b_fixed() {
    std::unique_lock<std::mutex> lock1(mtx1, std::defer_lock);
    std::unique_lock<std::mutex> lock2(mtx2, std::defer_lock);
    std::lock(lock1, lock2); // same call order, but std::lock handles it
}
```

### Fix 2: scoped_lock (C++17, preferred)

```cpp
void thread_a_modern() {
    std::scoped_lock lock(mtx1, mtx2); // deadlock-free by design
    // do work...
}

void thread_b_modern() {
    std::scoped_lock lock(mtx1, mtx2); // same order doesn't matter
    // do work...
}
```

### Fix 3: Lock Ordering (Hierarchy)

Establish a global ordering and always lock in the same order.

```cpp
// Convention: always lock mtx1 before mtx2
void thread_a_ordered() {
    std::lock_guard<std::mutex> lock1(mtx1);
    std::lock_guard<std::mutex> lock2(mtx2);
}

void thread_b_ordered() {
    std::lock_guard<std::mutex> lock1(mtx1); // same order as thread_a!
    std::lock_guard<std::mutex> lock2(mtx2);
}
```

### Hierarchical Mutex Implementation

```cpp
#include <mutex>
#include <stdexcept>

class HierarchicalMutex {
    std::mutex internal_mutex_;
    const unsigned long hierarchy_value_;
    unsigned long previous_hierarchy_value_ = 0;
    static thread_local unsigned long this_thread_hierarchy_value_;

    void check_for_hierarchy_violation() const {
        if (this_thread_hierarchy_value_ <= hierarchy_value_)
            throw std::logic_error("Mutex hierarchy violated");
    }

    void update_hierarchy_value() {
        previous_hierarchy_value_ = this_thread_hierarchy_value_;
        this_thread_hierarchy_value_ = hierarchy_value_;
    }

public:
    explicit HierarchicalMutex(unsigned long value) : hierarchy_value_(value) {}

    void lock() {
        check_for_hierarchy_violation();
        internal_mutex_.lock();
        update_hierarchy_value();
    }

    void unlock() {
        if (this_thread_hierarchy_value_ != hierarchy_value_)
            throw std::logic_error("Mutex hierarchy violated on unlock");
        this_thread_hierarchy_value_ = previous_hierarchy_value_;
        internal_mutex_.unlock();
    }

    bool try_lock() {
        check_for_hierarchy_violation();
        if (!internal_mutex_.try_lock()) return false;
        update_hierarchy_value();
        return true;
    }
};

thread_local unsigned long HierarchicalMutex::this_thread_hierarchy_value_(ULONG_MAX);

HierarchicalMutex high_mutex(10000);
HierarchicalMutex low_mutex(5000);

void good_function() {
    std::lock_guard<HierarchicalMutex> lk1(high_mutex); // high first
    std::lock_guard<HierarchicalMutex> lk2(low_mutex);  // low second: OK
}
```

### Livelock and Starvation

```cpp
// Livelock: threads keep retrying, never making progress
// Example: two people in a corridor both step aside in the same direction repeatedly

// Starvation: a thread never gets CPU time because higher-priority
// threads always preempt it.
```

---

## Interview Q&A — Deadlock

**Q1: What are the four necessary conditions for a deadlock?**
> Mutual exclusion, hold and wait, no preemption, circular wait. Breaking ANY one prevents deadlock.

**Q2: How does std::lock() prevent deadlock?**
> It uses a deadlock-avoidance algorithm (typically try-and-backoff or equivalent) to acquire multiple locks without the circular wait condition. It guarantees all-or-nothing acquisition.

**Q3: What is the difference between deadlock and livelock?**
> In deadlock, threads are permanently blocked waiting. In livelock, threads are actively running but keep reacting to each other and making no progress (like two people in a hallway).

**Q4: What is starvation?**
> A thread is perpetually denied access to a resource because other threads always take priority. Not a deadlock but still a correctness issue.

**Q5: Is it safe to lock a std::mutex from a signal handler?**
> No. Mutexes are not async-signal-safe. Use `std::atomic` for shared state accessed from signal handlers.

**Q6: What happens if you lock a non-recursive mutex twice from the same thread?**
> Undefined behavior. The second lock call will typically deadlock (the thread blocks waiting for itself) or throw. Use `std::recursive_mutex` if you need this pattern.

---

# 4. std::condition_variable

## Core Concepts

`std::condition_variable` allows threads to wait until a condition becomes true. It works with `std::unique_lock<std::mutex>`.

### Basic Usage

```cpp
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <iostream>

std::mutex mtx;
std::condition_variable cv;
bool data_ready = false;
int data = 0;

void producer() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        std::lock_guard<std::mutex> lock(mtx);
        data = 42;
        data_ready = true;
    }
    cv.notify_one(); // wake up one waiting thread
}

void consumer() {
    std::unique_lock<std::mutex> lock(mtx); // unique_lock required for wait()
    cv.wait(lock, []{ return data_ready; }); // predicate form — safe against spurious wakes
    std::cout << "Got: " << data << "\n";
}

int main() {
    std::thread p(producer);
    std::thread c(consumer);
    p.join(); c.join();
}
```

### Spurious Wakeups

`cv.wait()` can wake up even without `notify_one()`/`notify_all()` being called. This is allowed by the C++ standard (and happens on Linux/POSIX).

```cpp
// WRONG — no predicate, vulnerable to spurious wakeups
cv.wait(lock); // may wake up prematurely!

// CORRECT — always use a predicate
cv.wait(lock, []{ return data_ready; });
// Internally equivalent to:
// while (!data_ready) cv.wait(lock);
```

### Producer-Consumer with Bounded Queue

```cpp
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>

template<typename T>
class BoundedQueue {
    std::queue<T>           queue_;
    std::mutex              mtx_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    const size_t            max_size_;

public:
    explicit BoundedQueue(size_t max) : max_size_(max) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [this]{ return queue_.size() < max_size_; });
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [this]{ return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }
};

int main() {
    BoundedQueue<int> q(5);

    std::thread producer([&]{
        for (int i = 0; i < 10; ++i) {
            q.push(i);
            std::cout << "Produced: " << i << "\n";
        }
    });

    std::thread consumer([&]{
        for (int i = 0; i < 10; ++i) {
            int v = q.pop();
            std::cout << "Consumed: " << v << "\n";
        }
    });

    producer.join();
    consumer.join();
}
```

### wait_for / wait_until

```cpp
auto status = cv.wait_for(lock, std::chrono::seconds(1), []{ return data_ready; });
if (status) {
    // condition became true within timeout
} else {
    // timed out
}
```

### notify_one vs notify_all

```cpp
cv.notify_one();  // wake exactly one waiting thread (which one is unspecified)
cv.notify_all();  // wake all waiting threads (they will compete for the mutex)
// notify_all is safer when multiple threads might satisfy the condition
// notify_one is more efficient for single-consumer scenarios
```

---

## Interview Q&A — condition_variable

**Q1: Why does condition_variable require unique_lock instead of lock_guard?**
> `wait()` needs to atomically unlock the mutex and suspend the thread. It calls `lock.unlock()` before sleeping and `lock.lock()` upon waking. `lock_guard` doesn't expose `unlock()`, so it can't be used.

**Q2: What is a spurious wakeup and why does it happen?**
> A spurious wakeup is when `wait()` returns even though `notify_one/all` was never called. It's a property of POSIX condition variables that the C++ standard preserves. Always use the predicate form of `wait()` to handle this.

**Q3: Can you call notify_one() without holding the mutex?**
> Yes, technically. It's valid to call `notify_one()` without holding the lock. However, it's often better to notify while holding the lock to avoid a race where the consumer checks the condition, finds it false, is about to call wait, but the producer notifies before wait is entered.

**Q4: What is the lost wakeup problem?**
> If `notify_one()` is called BEFORE the consumer has started `wait()`, the notification is lost and the consumer waits forever. The predicate + shared state pattern (checking a bool) solves this: the consumer checks the state before sleeping, so if the notification already happened, it proceeds immediately.

**Q5: What is condition_variable_any vs condition_variable?**
> `condition_variable` works only with `unique_lock<mutex>`. `condition_variable_any` works with any lock type (e.g., `shared_lock`, custom locks). It's slightly slower due to extra overhead.

**Q6: What is the Mesa vs Hoare semantics difference?**
> C++ uses Mesa semantics: when notified, a thread goes to the ready queue and competes for the lock. The predicate might have become false again by the time it runs. In Hoare semantics, the notified thread would run immediately. Mesa semantics require always re-checking the predicate (hence the while loop / predicate form).

---

# 5. std::atomic + memory_order

## Core Concepts

`std::atomic<T>` provides lock-free, thread-safe operations on simple types without a mutex. The memory ordering controls how operations are visible across threads.

### Basic Atomic Operations

```cpp
#include <atomic>
#include <thread>
#include <iostream>

std::atomic<int> counter{0};

void increment(int n) {
    for (int i = 0; i < n; ++i)
        counter.fetch_add(1); // atomic increment, returns old value
        // OR: counter++;  (equivalent, seq_cst by default)
        // OR: counter += 1;
}

int main() {
    std::thread t1(increment, 1000);
    std::thread t2(increment, 1000);
    t1.join(); t2.join();
    std::cout << counter.load() << "\n"; // always 2000
}
```

### All Atomic Operations

```cpp
std::atomic<int> a{10};

// Load and store
int val = a.load(std::memory_order_seq_cst);
a.store(20, std::memory_order_seq_cst);

// Read-Modify-Write
int old = a.fetch_add(5);   // old = 10, a = 15
int old2 = a.fetch_sub(3);  // old2 = 15, a = 12
int old3 = a.fetch_and(0xF);
int old4 = a.fetch_or(0x1);
int old5 = a.fetch_xor(0x3);

// Exchange
int prev = a.exchange(100); // atomically set to 100, returns old value

// Compare-and-Swap (CAS)
int expected = 100;
bool success = a.compare_exchange_strong(expected, 200);
// If a == expected (100), sets a = 200, returns true
// If a != expected, sets expected = a (current value), returns false
// Use compare_exchange_weak in a loop (may spuriously fail, but faster on some architectures)
```

### Memory Ordering — The Six Orders

```cpp
// Most restrictive to least restrictive:

// 1. seq_cst (default) — Total global order, most intuitive, most expensive
std::atomic<int> x{0};
x.store(1, std::memory_order_seq_cst);
int v = x.load(std::memory_order_seq_cst);

// 2. acquire — For loads: no reads/writes in THIS thread can be reordered BEFORE this load
x.load(std::memory_order_acquire);

// 3. release — For stores: no reads/writes in THIS thread can be reordered AFTER this store
x.store(1, std::memory_order_release);

// 4. acq_rel — For RMW: combines acquire + release
x.fetch_add(1, std::memory_order_acq_rel);

// 5. consume — Like acquire but only for data-dependent reads (rarely used, often promoted to acquire)
x.load(std::memory_order_consume);

// 6. relaxed — No ordering guarantees, only atomicity. Fastest.
x.load(std::memory_order_relaxed);
x.store(1, std::memory_order_relaxed);
```

### Acquire-Release Pattern (Synchronization Without seq_cst)

```cpp
#include <atomic>
#include <thread>
#include <cassert>

std::atomic<bool> ready{false};
int data = 0;

void producer() {
    data = 42;                               // (1) write to data
    ready.store(true, std::memory_order_release); // (2) release store
    // Guarantees: (1) happens-before (2)
    // Any thread that acquire-loads (2) will see (1)
}

void consumer() {
    while (!ready.load(std::memory_order_acquire)); // (3) acquire load
    // Synchronizes-with (2): everything before (2) is visible here
    assert(data == 42); // always true
}
```

### Relaxed Ordering — Only Atomicity

```cpp
std::atomic<int> count{0};

// Use relaxed when you only need atomicity, not ordering
// Example: collecting statistics where exact order doesn't matter
void stats_increment() {
    count.fetch_add(1, std::memory_order_relaxed);
}
// At program end, count is accurate, but we can't reason about
// which thread incremented when relative to other operations
```

### CAS Loop — Lock-free Stack Push

```cpp
#include <atomic>
#include <memory>

template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next;
        Node(T d) : data(std::move(d)), next(nullptr) {}
    };
    std::atomic<Node*> head_{nullptr};

public:
    void push(T data) {
        Node* new_node = new Node(std::move(data));
        new_node->next = head_.load(std::memory_order_relaxed);
        // CAS loop: keep trying until we successfully swap head
        while (!head_.compare_exchange_weak(
            new_node->next, new_node,
            std::memory_order_release,
            std::memory_order_relaxed));
    }
};
```

### std::atomic_flag — The Simplest Atomic

```cpp
#include <atomic>

// Only guaranteed lock-free atomic in the standard
// Only two states: set and clear
std::atomic_flag flag = ATOMIC_FLAG_INIT; // must be initialized this way

// Spinlock implementation using atomic_flag
class Spinlock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire))
            ; // spin — consider adding pause hint for performance
    }
    void unlock() {
        flag_.clear(std::memory_order_release);
    }
};
```

---

## Interview Q&A — std::atomic & memory_order

**Q1: What is the difference between atomic and volatile?**
> `volatile` prevents compiler optimizations and ensures every read/write goes to memory, but provides NO thread-safety guarantees and NO memory ordering between threads. `std::atomic` provides both atomicity and configurable memory ordering. Never use `volatile` for thread communication.

**Q2: What is the default memory order for atomic operations?**
> `std::memory_order_seq_cst`. It provides the strongest guarantees: all sequentially consistent operations appear in a single total order. It's the safest default but the most expensive on some architectures (e.g., ARM, POWER).

**Q3: What is the acquire-release semantic and when would you use it?**
> Acquire-release creates a synchronization relationship: a release store on one thread synchronizes with an acquire load on another. Everything the releasing thread did before the store is visible to the acquiring thread after the load. Use it when you have one producer and one consumer and need to transfer a set of changes atomically but don't need a global total order. It's cheaper than seq_cst.

**Q4: What is compare_exchange_weak vs compare_exchange_strong?**
> `compare_exchange_weak` can fail spuriously (even if the value equals expected) due to hardware-level instruction limitations. It's faster in a loop because it doesn't need extra checks. `compare_exchange_strong` never fails spuriously. Use `weak` in a retry loop, `strong` when you can't afford a retry.

**Q5: Is std::atomic<T> always lock-free?**
> No. For large `T` (e.g., structs > 8 bytes on most platforms), the standard library may implement atomics using a mutex internally. Check with `std::atomic<T>::is_lock_free()` or `std::atomic<T>::is_always_lock_free` (C++17).

**Q6: What is the ABA problem?**
> In CAS-based lock-free algorithms: Thread A reads value A, gets preempted. Thread B changes value A→B→A. Thread A resumes, CAS sees A and "succeeds" — but the state has changed in between. Solutions: tagged pointers (attach a version counter), hazard pointers, epoch-based reclamation.

**Q7: What is the happens-before relation?**
> An operation X happens-before Y if: X is sequenced-before Y in the same thread, OR X synchronizes-with Y (e.g., release store to acquire load of the same atomic). If X happens-before Y, all effects of X are visible to Y.

---

# 6. Race Conditions — Definition, Detection, Fixing

## Core Concepts

A **data race** occurs when two threads access the same memory location concurrently, at least one access is a write, and there is no synchronization between them. Data races are undefined behavior in C++.

A **race condition** is a broader term: the program's correctness depends on the relative ordering of events between threads, even if there is no data race (e.g., check-then-act patterns).

### Example: Data Race

```cpp
int counter = 0; // shared, no synchronization

void bad_increment() {
    ++counter; // NOT atomic: read, add, write — 3 separate steps
    // Thread A reads 5, Thread B reads 5, both write 6 -> lost update
}
// Result: undefined behavior, incorrect counts
```

### Example: Race Condition (No Data Race, Still Wrong)

```cpp
std::map<int, std::string> cache;
std::mutex mtx;

// "Check-then-act" race condition:
std::string get_or_compute(int key) {
    mtx.lock();
    if (cache.find(key) == cache.end()) {
        mtx.unlock(); // release lock to do expensive computation
        std::string val = expensive_compute(key);
        mtx.lock();
        cache[key] = val; // RACE: another thread may have inserted key already
    }
    std::string result = cache[key];
    mtx.unlock();
    return result;
}

// Fix: keep lock held, or use try_emplace atomically
```

### Detecting Races with ThreadSanitizer

```bash
# Compile with TSan
g++ -fsanitize=thread -g -O1 myfile.cpp -o myfile
./myfile
# TSan will report: "data race on counter"
```

### Fixing Race Conditions

```cpp
// Fix 1: Use std::atomic
std::atomic<int> counter{0};
void safe_increment() { counter++; }

// Fix 2: Use mutex
std::mutex mtx;
int counter = 0;
void safe_increment() {
    std::lock_guard<std::mutex> lock(mtx);
    ++counter;
}

// Fix 3: Thread-local accumulation (no sharing during computation)
thread_local int local_counter = 0;
std::atomic<int> global_counter{0};
void thread_safe_increment(int n) {
    local_counter += n; // no sharing
    global_counter.fetch_add(local_counter, std::memory_order_relaxed);
    local_counter = 0;
}
```

### Common Patterns That Lead to Races

```cpp
// Pattern 1: Singleton double-checked locking (wrong without atomic/memory_order)
MyClass* instance = nullptr;
std::mutex mtx;
MyClass* get_instance() {
    if (!instance) { // (1) read without lock
        std::lock_guard<std::mutex> lock(mtx);
        if (!instance) // (2) double check
            instance = new MyClass(); // (3) write
    }
    return instance;
}
// Problem: (1) and (3) are unsynchronized — use std::atomic or call_once

// Pattern 2: Iterator invalidation under concurrency
std::vector<int> vec = {1, 2, 3};
// Thread A: vec.push_back(4);  -> may reallocate, invalidating all iterators
// Thread B: for (auto x : vec) // UB if vec was reallocated
```

---

## Interview Q&A — Race Conditions

**Q1: What is the difference between a data race and a race condition?**
> A data race is a specific C++ undefined behavior: two threads access the same memory, at least one writes, with no synchronization. A race condition is a logic bug where correctness depends on timing. You can have a race condition without a data race (e.g., TOCTOU with proper locks but wrong granularity), and eliminating data races doesn't always fix race conditions.

**Q2: How do you detect data races in practice?**
> Use ThreadSanitizer (`-fsanitize=thread`), Helgrind (Valgrind), or Intel Inspector. Static analysis tools like Clang's thread safety annotations can catch some at compile time.

**Q3: Can a data race occur with a single 64-bit integer on x86?**
> Yes, C++ does not guarantee atomic reads/writes for plain `int` regardless of the hardware. Even if the hardware performs it atomically, the compiler can split it into multiple operations or reorder it without `std::atomic`. Always use `std::atomic` for shared mutable variables.

**Q4: What is TOCTOU?**
> Time-of-Check to Time-of-Use. You check a condition, then act on it, but the condition may have changed between check and use. Example: checking if a file exists before opening it — another process may delete it in between.

---

# 7. std::async + std::future

## Core Concepts

`std::async` launches a task asynchronously and returns a `std::future` that holds the result. It is the high-level alternative to manual thread management.

### Basic Usage

```cpp
#include <future>
#include <iostream>
#include <cmath>

int compute(int n) {
    return n * n;
}

int main() {
    // Launch asynchronously
    std::future<int> f = std::async(std::launch::async, compute, 5);

    // Do other work here...

    int result = f.get(); // blocks until result is ready
    std::cout << "Result: " << result << "\n"; // 25
}
```

### Launch Policies

```cpp
// std::launch::async — guaranteed new thread
std::future<int> f1 = std::async(std::launch::async, compute, 5);

// std::launch::deferred — lazy: runs in calling thread when get() is called
std::future<int> f2 = std::async(std::launch::deferred, compute, 5);
int val = f2.get(); // compute runs HERE, synchronously

// Default (unspecified) — implementation chooses
std::future<int> f3 = std::async(compute, 5); // may or may not start a new thread
// Problem: if deferred is chosen, f3.get() may never run in parallel!
// Always specify launch::async if you need parallelism
```

### Exception Propagation

```cpp
#include <future>
#include <stdexcept>
#include <iostream>

int risky(int n) {
    if (n < 0) throw std::invalid_argument("negative");
    return n * 2;
}

int main() {
    std::future<int> f = std::async(std::launch::async, risky, -1);

    try {
        int val = f.get(); // re-throws exception from the async thread
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }
}
```

### Parallel Computation with Multiple Futures

```cpp
#include <future>
#include <numeric>
#include <vector>
#include <iostream>

long long sum_range(const std::vector<int>& v, size_t start, size_t end) {
    return std::accumulate(v.begin() + start, v.begin() + end, 0LL);
}

int main() {
    std::vector<int> data(1000000, 1);
    size_t mid = data.size() / 2;

    auto f1 = std::async(std::launch::async, sum_range, std::cref(data), 0, mid);
    auto f2 = std::async(std::launch::async, sum_range, std::cref(data), mid, data.size());

    long long total = f1.get() + f2.get();
    std::cout << "Sum: " << total << "\n"; // 1000000
}
```

### Fire and Forget — The Trap

```cpp
// DANGEROUS: future destructor blocks if launch::async
void fire_and_forget() {
    // This looks like fire-and-forget but it's NOT
    std::async(std::launch::async, []{ long_task(); });
    // The temporary future's destructor blocks here until long_task() finishes!
}

// True fire-and-forget requires storing the future or using a detached thread
std::future<void> running_task; // keep it alive
void true_fire_and_forget() {
    running_task = std::async(std::launch::async, []{ long_task(); });
    // Now we can proceed without blocking immediately
}
```

### Future Status

```cpp
std::future<int> f = std::async(std::launch::async, compute, 10);

// Poll without blocking
auto status = f.wait_for(std::chrono::milliseconds(0));
if (status == std::future_status::ready) {
    std::cout << f.get() << "\n";
} else if (status == std::future_status::timeout) {
    std::cout << "Not ready yet\n";
} else if (status == std::future_status::deferred) {
    std::cout << "Deferred\n";
}

// Wait with timeout
f.wait_for(std::chrono::seconds(1));
f.wait_until(std::chrono::steady_clock::now() + std::chrono::seconds(5));
```

---

## Interview Q&A — std::async & std::future

**Q1: What is the difference between std::async and std::thread?**
> `std::thread` requires manual result passing (via shared state), has no built-in exception propagation, and you must `join()` or `detach()`. `std::async` returns a `std::future` that holds the return value or exception, automatically manages the thread lifecycle (for `launch::async`), and blocks on destruction. Tasks are generally preferred over raw threads.

**Q2: What does the default launch policy do?**
> It's implementation-defined: the system may run the task in a new thread (async) or lazily in the calling thread (deferred). This means you can't assume parallelism. Always use `std::launch::async` explicitly when you need parallel execution.

**Q3: Can you call get() on a future more than once?**
> No. `get()` can only be called once. After the first call, the future is invalid (`valid()` returns false). If you need multiple threads to get the same result, use `std::shared_future`.

**Q4: What happens to a future when it goes out of scope?**
> If the future was created by `std::async` with `launch::async`, its destructor blocks until the async task completes. For other futures (from promise/packaged_task), the destructor does not block.

**Q5: How do you pass arguments to async by reference?**
> Use `std::ref()` or `std::cref()`. Arguments are copied by default. Ensure the referenced object outlives the async task.

**Q6: How do you cancel a future/task?**
> C++ has no built-in cancellation for `std::async`. Use a shared `std::atomic<bool>` flag that the task checks periodically, or use `std::stop_token` with `std::jthread` (C++20).

**Q7: What is std::shared_future and when would you use it?**
> A `std::shared_future` can be copied and multiple threads can call `get()` on it. Use it when multiple consumers need to wait for the same result. Obtain from `future::share()`.

---

## Summary Table — Tier 1 Quick Reference

| Primitive | Use When | Key Pitfall |
|-----------|----------|-------------|
| `std::thread` | Need direct thread control | Must join or detach before destruction |
| `std::mutex` + `lock_guard` | Protect shared mutable data | Over-locking kills performance |
| `std::scoped_lock` | Lock multiple mutexes | N/A — use instead of manual std::lock |
| `std::condition_variable` | Wait for a condition to be true | Always use predicate; spurious wakeups |
| `std::atomic` | Single variable shared between threads | Only for simple types; not a replacement for all synchronization |
| `std::async` + `future` | Run a task, get result back | Default policy ambiguity; get() is one-shot |
| Memory order `seq_cst` | Always correct, simple | Slowest — use acquire/release when possible |
| Memory order `acquire`/`release` | Producer-consumer handoff | Must pair correctly on the right atomic variable |
