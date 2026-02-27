# Tier 2 — Very Common
## C++ Multithreading Interview Deep Dive

---

# 1. std::unique_lock vs std::lock_guard

## Core Concepts

Both are RAII mutex wrappers, but `unique_lock` is more flexible at the cost of slight overhead.

### Feature Comparison

| Feature | lock_guard | unique_lock | scoped_lock |
|---------|-----------|-------------|-------------|
| Locks on construction | Yes (always) | Optional | Yes |
| Unlock before destruction | No | Yes | No |
| Re-lock after unlock | No | Yes | No |
| Deferred locking | No | Yes | No |
| Multiple mutexes | No | No | Yes |
| Move semantics | No | Yes | No |
| Use with condition_variable | No | Yes | No |
| Overhead | Minimal | Small extra | Minimal |

### lock_guard — Simple RAII

```cpp
#include <mutex>

std::mutex mtx;
int shared = 0;

void simple_increment() {
    std::lock_guard<std::mutex> lock(mtx);
    ++shared;
} // always unlocks here

// C++17: CTAD (class template argument deduction)
void simple_increment_v2() {
    std::lock_guard lock(mtx); // type deduced
    ++shared;
}
```

### unique_lock — Flexible RAII

```cpp
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

// Use case 1: manual unlock/relock (needed for condition_variable)
void wait_for_data() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return ready; }); // wait() unlocks and relocks internally
    // process data
}

// Use case 2: deferred locking
void deferred_example() {
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock); // don't lock yet
    // ... do some non-critical setup ...
    lock.lock(); // lock now
    // critical section
    lock.unlock(); // early unlock
    // ... non-critical work ...
} // lock is already unlocked, destructor does nothing

// Use case 3: try_lock
void try_lock_example() {
    std::unique_lock<std::mutex> lock(mtx, std::try_to_lock);
    if (lock.owns_lock()) {
        // got the lock
    } else {
        // didn't get it
    }
}

// Use case 4: adopt existing lock
void adopt_example() {
    mtx.lock(); // manually locked
    std::unique_lock<std::mutex> lock(mtx, std::adopt_lock); // take ownership
    // lock will unlock on destruction
}

// Use case 5: movable — transferring lock ownership
std::unique_lock<std::mutex> get_lock() {
    std::unique_lock<std::mutex> lock(mtx);
    // ... prepare something ...
    return lock; // moves ownership to caller
}
```

### When to Use Which

```cpp
// Use lock_guard when:
// - Simple critical section, lock held for duration of scope
// - No condition variable needed
// - Clearest intention, zero overhead
void write_data(int val) {
    std::lock_guard<std::mutex> lock(mtx);
    shared = val;
}

// Use unique_lock when:
// - Working with condition_variable
// - Need to unlock before scope ends
// - Need to try_lock or defer_lock
// - Need to transfer lock ownership
void conditional_write(int val) {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, []{ return ready; });
    shared = val;
}

// Use scoped_lock when:
// - Locking multiple mutexes simultaneously
std::mutex m1, m2;
void safe_swap(int& a, int& b) {
    std::scoped_lock lock(m1, m2);
    std::swap(a, b);
}
```

### Performance Considerations

```cpp
// unique_lock stores a bool for owns_lock() tracking -> small overhead
// In tight loops, prefer lock_guard

// Example: lock_guard is faster here
void fast_increment(int n) {
    for (int i = 0; i < n; ++i) {
        std::lock_guard<std::mutex> lock(mtx); // faster
        ++shared;
    }
}
```

---

## Interview Q&A — unique_lock vs lock_guard

**Q1: When must you use unique_lock over lock_guard?**
> When using `std::condition_variable` (requires the lock to be unlocked/relocked by `wait()`), when you need to unlock before scope ends, when you need deferred or tried locking, or when you need to transfer lock ownership (move semantics).

**Q2: Does unique_lock have any overhead compared to lock_guard?**
> Yes, `unique_lock` stores an extra `bool` (`owns_lock_`) and updates it on every lock/unlock operation. It's negligible for most uses but can matter in extremely tight inner loops.

**Q3: Can lock_guard be moved or copied?**
> Neither. It is non-copyable and non-movable. `unique_lock` is movable (but not copyable), which is why it can be returned from functions.

**Q4: What is std::adopt_lock and when do you use it?**
> `std::adopt_lock` is a tag that tells the lock wrapper the mutex is already locked — take ownership without locking again. Used when `std::lock()` or `std::scoped_lock` has already locked the mutex and you want RAII cleanup.

---

# 2. std::promise / std::packaged_task

## Core Concepts

`std::promise` and `std::packaged_task` are the lower-level building blocks for `std::future`. They give you more control over when results are set and from where.

### std::promise — Manual Future Setting

```cpp
#include <future>
#include <thread>
#include <iostream>

void compute_in_thread(std::promise<int> prom, int n) {
    try {
        if (n < 0) throw std::invalid_argument("negative input");
        prom.set_value(n * n); // fulfill the promise
    } catch (...) {
        prom.set_exception(std::current_exception()); // propagate exception
    }
}

int main() {
    std::promise<int> prom;
    std::future<int> fut = prom.get_future(); // associate future with promise

    std::thread t(compute_in_thread, std::move(prom), 5);

    try {
        int result = fut.get(); // blocks until promise is fulfilled
        std::cout << "Result: " << result << "\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "Error: " << e.what() << "\n";
    }
    t.join();
}
```

### Promise for Thread Notification (No Return Value)

```cpp
#include <future>
#include <thread>

std::promise<void> signal;

void worker(std::future<void> start_signal) {
    start_signal.get(); // wait for the signal
    std::cout << "Worker started!\n";
}

int main() {
    std::future<void> fut = signal.get_future();
    std::thread t(worker, std::move(fut));

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    signal.set_value(); // signal all waiting threads

    t.join();
}
```

### std::packaged_task — Wrapping Callable for Future

```cpp
#include <future>
#include <thread>
#include <functional>
#include <iostream>

int add(int a, int b) { return a + b; }

int main() {
    // Wrap a callable
    std::packaged_task<int(int, int)> task(add);
    std::future<int> fut = task.get_future();

    // Execute in a new thread
    std::thread t(std::move(task), 3, 4);
    t.join();

    std::cout << "Sum: " << fut.get() << "\n"; // 7
}
```

### packaged_task as a Deferred Task

```cpp
#include <future>
#include <vector>
#include <functional>

// Simple task queue using packaged_task
class TaskQueue {
    std::vector<std::packaged_task<int()>> tasks_;
    std::mutex mtx_;

public:
    std::future<int> submit(std::function<int()> fn) {
        std::packaged_task<int()> task(fn);
        std::future<int> fut = task.get_future();
        std::lock_guard<std::mutex> lock(mtx_);
        tasks_.push_back(std::move(task));
        return fut;
    }

    void run_all() {
        for (auto& t : tasks_) t(); // execute each task
    }
};

int main() {
    TaskQueue q;
    auto f1 = q.submit([]{ return 1 + 1; });
    auto f2 = q.submit([]{ return 2 * 3; });

    q.run_all();

    std::cout << f1.get() << "\n"; // 2
    std::cout << f2.get() << "\n"; // 6
}
```

### Promise vs packaged_task vs async

```cpp
// async: simplest, automatic thread management
auto fut1 = std::async(std::launch::async, add, 1, 2);

// packaged_task: wrap a callable, manually control when/where it runs
std::packaged_task<int(int,int)> task(add);
auto fut2 = task.get_future();
std::thread(std::move(task), 1, 2).detach();

// promise: most manual, set value from anywhere
std::promise<int> prom;
auto fut3 = prom.get_future();
std::thread([p = std::move(prom)]() mutable { p.set_value(3); }).detach();
```

---

## Interview Q&A — promise / packaged_task

**Q1: What is the difference between promise and packaged_task?**
> `std::promise` is a raw channel to set a value/exception manually from anywhere. `std::packaged_task` wraps a callable and sets the promise automatically when the callable returns or throws. `packaged_task` is a promise + callable bound together.

**Q2: Can you set a promise value more than once?**
> No. Calling `set_value()` or `set_exception()` more than once throws `std::future_error` with code `promise_already_satisfied`.

**Q3: What happens to the future if the promise is destroyed without being fulfilled?**
> The future becomes broken: calling `get()` throws `std::future_error` with `broken_promise`. This is why promises should always be fulfilled or have their exception set.

**Q4: When would you use packaged_task over async?**
> When you need to control exactly when and where the task runs (e.g., in a thread pool, on a specific thread, or deferred). `async` gives you less control over scheduling.

**Q5: Can a packaged_task be called multiple times?**
> No. After the task is invoked once, the associated promise is fulfilled. Calling it again throws `std::future_error`. Create a new `packaged_task` for each invocation.

---

# 3. Thread-Safe Singleton

## Core Concepts

A singleton ensures only one instance of a class exists. Making it thread-safe requires care — the naive implementation has a data race.

### Wrong: Not Thread-Safe

```cpp
class Singleton {
    static Singleton* instance_;
    Singleton() = default;
public:
    static Singleton* get_instance() {
        if (!instance_) // DATA RACE: two threads may both see nullptr
            instance_ = new Singleton();
        return instance_;
    }
};
```

### Wrong: Naive Mutex Lock (Correct But Slow)

```cpp
std::mutex mtx;
class Singleton {
    static Singleton* instance_;
    Singleton() = default;
public:
    static Singleton* get_instance() {
        std::lock_guard<std::mutex> lock(mtx); // locks EVERY call — expensive
        if (!instance_)
            instance_ = new Singleton();
        return instance_;
    }
};
```

### Wrong: Broken Double-Checked Locking (Pre-C++11)

```cpp
// THIS IS BROKEN without proper memory barriers
static Singleton* get_instance() {
    if (!instance_) {             // (1) check without lock
        std::lock_guard lock(mtx);
        if (!instance_)           // (2) check with lock
            instance_ = new Singleton(); // (3) may be reordered by CPU/compiler!
    }
    return instance_;
}
// Problem: (3) can be: allocate memory, store pointer, THEN construct object
// Another thread sees non-null pointer but uninitialized object
```

### Correct: Meyers Singleton (Magic Statics — C++11 and later)

```cpp
class Singleton {
    Singleton() = default;
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
public:
    static Singleton& get_instance() {
        static Singleton instance; // C++11 guarantees thread-safe initialization
        return instance;
        // The standard mandates: if multiple threads call this simultaneously,
        // only one initializes, others block until initialization is complete.
    }
    void do_something() { /* ... */ }
};

// Usage
Singleton::get_instance().do_something();
```

### Correct: std::call_once

```cpp
#include <mutex>
#include <memory>

class Singleton {
    static std::unique_ptr<Singleton> instance_;
    static std::once_flag flag_;
    Singleton() = default;
public:
    static Singleton& get_instance() {
        std::call_once(flag_, []{
            instance_ = std::make_unique<Singleton>();
        });
        return *instance_;
    }
};

std::unique_ptr<Singleton> Singleton::instance_;
std::once_flag Singleton::flag_;
```

### Correct: Atomic with Double-Checked Locking (Fixed)

```cpp
#include <atomic>
#include <mutex>

class Singleton {
    static std::atomic<Singleton*> instance_;
    static std::mutex mtx_;
    Singleton() = default;
public:
    static Singleton* get_instance() {
        Singleton* p = instance_.load(std::memory_order_acquire);
        if (!p) {
            std::lock_guard<std::mutex> lock(mtx_);
            p = instance_.load(std::memory_order_relaxed);
            if (!p) {
                p = new Singleton();
                instance_.store(p, std::memory_order_release);
            }
        }
        return p;
    }
};
std::atomic<Singleton*> Singleton::instance_{nullptr};
std::mutex Singleton::mtx_;
// The acquire-release pair ensures the constructed Singleton is
// fully visible to threads that observe the non-null pointer.
```

### CRTP Singleton Base (Reusable Pattern)

```cpp
template<typename T>
class SingletonBase {
protected:
    SingletonBase() = default;
public:
    static T& get_instance() {
        static T instance;
        return instance;
    }
    SingletonBase(const SingletonBase&) = delete;
    SingletonBase& operator=(const SingletonBase&) = delete;
};

class Config : public SingletonBase<Config> {
    friend class SingletonBase<Config>; // allow base to construct
    Config() { /* load config */ }
public:
    std::string get(const std::string& key) { return "value"; }
};
```

---

## Interview Q&A — Thread-Safe Singleton

**Q1: Why is the Meyers singleton thread-safe in C++11?**
> C++11 §6.7: "If control enters the declaration concurrently while the variable is being initialized, the concurrent execution shall wait for completion of the initialization." The compiler emits the necessary synchronization around the static local variable.

**Q2: Why is the naive double-checked locking wrong?**
> `new T()` may be: (1) allocate memory, (2) store pointer, (3) call constructor. Steps 2 and 3 can be reordered by the compiler or CPU. A thread doing the outer check can see a non-null pointer but access an unconstructed object.

**Q3: What is the problem with a global singleton and static initialization order?**
> The "static initialization order fiasco": initialization order of static objects across translation units is undefined. If Singleton A depends on Singleton B but B isn't initialized yet, you get UB. The Meyers singleton (function-local static) solves this: it's initialized on first use.

**Q4: How do you unit-test code that uses a singleton?**
> Use dependency injection (pass the singleton as an interface), use a reset mechanism (risky in production), or use a registry pattern that allows replacement. Singletons are generally considered an anti-pattern for testability.

---

# 4. thread_local — Per-Thread Storage

## Core Concepts

`thread_local` declares storage that is unique per-thread. Each thread has its own copy. Changes in one thread are invisible to others.

### Basic Usage

```cpp
#include <thread>
#include <iostream>

thread_local int local_value = 0; // each thread has its own copy

void worker(int id) {
    local_value = id * 10; // writes to THIS thread's copy
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "Thread " << id << ": " << local_value << "\n";
    // Will always print id*10, not contaminated by other threads
}

int main() {
    std::thread t1(worker, 1);
    std::thread t2(worker, 2);
    std::thread t3(worker, 3);
    t1.join(); t2.join(); t3.join();
    // Output (in some order):
    // Thread 1: 10
    // Thread 2: 20
    // Thread 3: 30
}
```

### Thread-Local Random Number Generator

```cpp
#include <random>
#include <thread>

// Each thread gets its own RNG — no synchronization needed
thread_local std::mt19937 rng{std::random_device{}()};

int random_int(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng); // uses thread's own rng, no mutex needed
}
```

### Thread-Local Logger Buffer

```cpp
#include <string>
#include <mutex>
#include <vector>

std::mutex log_mtx;
std::vector<std::string> global_log;

// Each thread accumulates into its own buffer, flushes periodically
thread_local std::vector<std::string> local_log_buffer;

void log(const std::string& msg) {
    local_log_buffer.push_back(msg); // fast, no lock
}

void flush_logs() {
    std::lock_guard<std::mutex> lock(log_mtx);
    for (auto& entry : local_log_buffer)
        global_log.push_back(entry);
    local_log_buffer.clear();
}
```

### Initialization

```cpp
// thread_local with complex initialization — initialized once per thread
thread_local std::string thread_name = "unnamed";

// thread_local with function-scope static behavior
void register_thread(const std::string& name) {
    thread_name = name; // sets this thread's copy
}
```

### Performance: Thread-Local vs Shared Atomic

```cpp
// Benchmark: thread_local is much faster than atomic for frequent updates
thread_local long long local_count = 0; // CPU-local, no cache coherence traffic
std::atomic<long long> global_count{0}; // invalidates all CPU caches on write

void thread_local_version(int n) {
    for (int i = 0; i < n; ++i)
        ++local_count; // ~1 cycle
}

void atomic_version(int n) {
    for (int i = 0; i < n; ++i)
        global_count.fetch_add(1, std::memory_order_relaxed); // ~10-100 cycles on contested cache line
}
```

---

## Interview Q&A — thread_local

**Q1: When is a thread_local variable initialized?**
> On first access by that thread (lazy initialization per thread). Each thread gets a fresh copy. When the thread exits, thread_local objects are destroyed in reverse order of construction.

**Q2: Can thread_local be used with class members?**
> Only as `static thread_local` members. Non-static members cannot be `thread_local` (they already exist per-instance, not per-thread).

**Q3: What is the difference between thread_local and a thread ID map?**
> `thread_local` is zero-overhead (compiler transforms it to a pointer lookup through the thread's storage block). A `std::map<std::thread::id, T>` requires a lock and heap allocation. Use `thread_local` for performance-critical per-thread data.

**Q4: What happens to thread_local objects when a thread exits?**
> Their destructors are called, in reverse order of construction, before the thread's stack is destroyed.

---

# 5. False Sharing — Cache-Line Contention

## Core Concepts

**False sharing** occurs when two threads modify independent variables that happen to reside on the same CPU cache line. The CPU must invalidate and reload the entire cache line on every write, causing massive performance degradation despite no logical data dependency.

### How Cache Lines Work

```
Cache line = 64 bytes on most modern x86/x64 CPUs

Memory layout:
[  counter_a  |  counter_b  |  padding...  ]  <- same 64-byte cache line
  Thread 1 ↑      Thread 2 ↑
  writes here     writes here

Every write by Thread 1 invalidates Thread 2's cache, and vice versa.
The hardware forces coherence even though the variables are logically independent.
```

### Demonstrating False Sharing

```cpp
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>

struct Bad {
    long long a = 0; // used by thread 1
    long long b = 0; // used by thread 2
    // a and b are likely on the same cache line!
};

struct alignas(64) Good {
    long long value = 0; // each on its own 64-byte cache line
};

Bad bad_data;
Good good_data[2]; // each element on separate cache line

void bad_thread_1() {
    for (int i = 0; i < 100000000; ++i) ++bad_data.a;
}
void bad_thread_2() {
    for (int i = 0; i < 100000000; ++i) ++bad_data.b;
}

void good_thread_1() {
    for (int i = 0; i < 100000000; ++i) ++good_data[0].value;
}
void good_thread_2() {
    for (int i = 0; i < 100000000; ++i) ++good_data[1].value;
}

// bad version: ~5-10x slower due to cache ping-pong between CPUs
```

### Fixing False Sharing: Padding

```cpp
#include <new> // for std::hardware_destructive_interference_size (C++17)

// Method 1: alignas with explicit 64-byte cache line size
struct alignas(64) PaddedCounter {
    long long value = 0;
    // Compiler pads to fill the 64-byte alignment unit
};

// Method 2: Manual padding
struct ManualPadded {
    long long value = 0;
    char padding[64 - sizeof(long long)]; // fill rest of cache line
};

// Method 3: C++17 standard constant
struct StandardPadded {
    alignas(std::hardware_destructive_interference_size) long long value = 0;
};

// Thread pool counter array — each thread has its own cache line
struct ThreadCounter {
    alignas(64) std::atomic<long long> count{0};
};
std::vector<ThreadCounter> per_thread_counts(std::thread::hardware_concurrency());

void worker(int thread_id, int iterations) {
    for (int i = 0; i < iterations; ++i)
        per_thread_counts[thread_id].count.fetch_add(1, std::memory_order_relaxed);
}

long long total() {
    long long sum = 0;
    for (auto& c : per_thread_counts) sum += c.count.load(std::memory_order_relaxed);
    return sum;
}
```

### Detecting False Sharing

```bash
# Linux perf tool — look for cache-misses / LLC-load-misses
perf stat -e cache-misses,cache-references,LLC-load-misses ./your_binary

# Intel VTune, AMD uProf, Google perf
# valgrind --tool=cachegrind ./your_binary
```

---

## Interview Q&A — False Sharing

**Q1: What is false sharing and why does it degrade performance?**
> False sharing is when two threads write to different variables that reside on the same CPU cache line. The cache coherence protocol forces the cache line to be transferred between CPUs on every write, even though the threads aren't logically sharing data. This causes "cache ping-pong" with latencies of 100-300 cycles per operation instead of 1-4.

**Q2: What is the typical cache line size on modern CPUs?**
> 64 bytes on most x86/x64 and ARM processors. Use `std::hardware_destructive_interference_size` (C++17) to get it portably at compile time.

**Q3: What is the difference between true sharing and false sharing?**
> True sharing: threads access the same logical data (e.g., a shared counter) — synchronization is needed. False sharing: threads access different logical data that happens to be on the same cache line — performance problem but not a correctness problem.

**Q4: Can false sharing cause correctness issues?**
> No, only performance issues. It's the cache coherence hardware that handles consistency. But the performance penalty can be so severe (5-50x slowdown) that it effectively breaks the benefit of parallelism.

**Q5: What is std::hardware_constructive_interference_size?**
> The minimum offset between objects you want to SHARE in the same cache line (L1 cache line size for performance). Contrast with `std::hardware_destructive_interference_size` (the size you pad to in order to AVOID sharing). Both are typically 64 bytes but are logically distinct.

---

# 6. RAII for Threads — Lifetime Management

## Core Concepts

RAII (Resource Acquisition Is Initialization) applies to threads: the thread should be owned by an object whose destructor ensures the thread is properly cleaned up.

### The Problem with Raw std::thread

```cpp
void launch_threads() {
    std::thread t1(task1);
    do_work_that_might_throw(); // if this throws...
    t1.join(); // ...this is never reached -> std::terminate()!
}
```

### ThreadGuard — Join on Destruction

```cpp
#include <thread>
#include <stdexcept>

// Strict ownership guard: joins on destruction
class ThreadGuard {
    std::thread t_;
public:
    explicit ThreadGuard(std::thread t) : t_(std::move(t)) {
        if (!t_.joinable())
            throw std::logic_error("No thread");
    }

    ~ThreadGuard() {
        if (t_.joinable())
            t_.join(); // safe even if constructor threw after this
    }

    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;
    ThreadGuard(ThreadGuard&&) = default;
    ThreadGuard& operator=(ThreadGuard&&) = default;

    std::thread& get() { return t_; }
};

void safe_launch() {
    ThreadGuard g(std::thread([]{ long_task(); }));
    do_work_that_might_throw(); // exception thrown
} // g's destructor calls t_.join() — safe!
```

### ScopedThread — Stricter Version

```cpp
// Like ThreadGuard but takes ownership and prevents detach
class ScopedThread {
    std::thread t_;
public:
    explicit ScopedThread(std::thread t) : t_(std::move(t)) {
        if (!t_.joinable())
            throw std::invalid_argument("No joinable thread");
    }
    ~ScopedThread() { t_.join(); }
    ScopedThread(ScopedThread&&) = default;
    ScopedThread& operator=(ScopedThread&&) = default;
    ScopedThread(const ScopedThread&) = delete;
    ScopedThread& operator=(const ScopedThread&) = delete;
};
```

### Managing Multiple Threads with RAII

```cpp
#include <vector>
#include <thread>

class ThreadPool {
    std::vector<std::thread> threads_;
public:
    void add(std::thread t) {
        threads_.push_back(std::move(t));
    }

    ~ThreadPool() {
        for (auto& t : threads_)
            if (t.joinable()) t.join();
    }

    ThreadPool() = default;
    ThreadPool(const ThreadPool&) = delete;
};

void parallel_work(int n_threads) {
    ThreadPool pool;
    for (int i = 0; i < n_threads; ++i)
        pool.add(std::thread([i]{ do_work(i); }));
} // pool destructor joins all threads, even if exception thrown
```

### std::jthread (C++20) — Built-in RAII

```cpp
#include <thread>

void modern_raii() {
    std::jthread t([]{ long_task(); }); // jthread = RAII thread
    do_work_that_might_throw();
} // t's destructor automatically calls join() — no explicit join needed
```

### Detach Pitfalls

```cpp
// DANGEROUS: detached thread uses stack variable that's gone
void dangerous_detach() {
    int local_var = 42;
    std::thread t([&local_var]{ // captures reference
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << local_var << "\n"; // UB: local_var is gone!
    });
    t.detach(); // t is now on its own
} // local_var destroyed here, detached thread still running!

// SAFE: use shared ownership or copy the data
void safe_version() {
    auto data = std::make_shared<int>(42);
    std::thread t([data]{ // copy of shared_ptr
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << *data << "\n"; // safe: shared ownership
    });
    t.detach();
} // data ref count drops to 1, still alive in the thread
```

### When Is Detach Acceptable?

```cpp
// 1. Daemon threads that run for the lifetime of the program
// 2. Fire-and-forget logging / analytics (if data is copied)
// 3. When you can guarantee the thread finishes before any shared data is destroyed

// Rule of thumb: prefer join. Only detach when you have strong lifetime guarantees.
```

---

## Interview Q&A — RAII & Thread Lifetime

**Q1: What happens when std::thread destructor is called on a joinable thread?**
> `std::terminate()` is called. This is intentional: the designers wanted to avoid the ambiguity of "should we join or detach?". RAII wrappers solve this by always joining.

**Q2: Why is detach dangerous?**
> A detached thread can outlive the scope where it was created. If it holds references/pointers to local variables, those become dangling. Additionally, you lose all handle to the thread — you can't wait for it to finish.

**Q3: What is the difference between a ThreadGuard and std::jthread?**
> `std::jthread` (C++20) is a standardized RAII thread that automatically joins on destruction AND supports cooperative cancellation via `std::stop_token`. `ThreadGuard` is a user-defined RAII wrapper for C++11/14/17.

**Q4: How do you safely pass data to a detached thread?**
> Use `std::shared_ptr` to share ownership, pass by value (copy), or ensure the data outlives the thread via a global/static. Never pass raw references or pointers to local variables.

**Q5: What is std::stop_token and how does it work with jthread?**
> `std::stop_token` provides a cooperative cancellation mechanism. `std::jthread` passes a `stop_token` to the thread function as the first argument. The thread periodically checks `stop_token.stop_requested()`. The owner calls `jthread::request_stop()` to signal cancellation.

---

## Summary Table — Tier 2 Quick Reference

| Topic | Key Point | Common Interview Trap |
|-------|-----------|----------------------|
| `lock_guard` | Simple RAII, minimum overhead | Can't use with condition_variable |
| `unique_lock` | Flexible, movable, needed for CV | Slight overhead, easy to misuse defer_lock |
| `scoped_lock` | Lock multiple mutexes safely | C++17 only |
| `promise` | Manual future fulfillment | Broken promise throws on future.get() |
| `packaged_task` | Callable → future | Can only be invoked once |
| Meyers Singleton | Static local = thread-safe in C++11 | Pre-C++11 not safe |
| `thread_local` | Per-thread storage, zero contention | Destroyed when thread exits |
| False sharing | Same cache line = perf problem | Not a correctness bug |
| RAII thread | ThreadGuard/jthread — always join | Never detach local variable references |
