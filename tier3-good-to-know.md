# Tier 3 — Good to Know
## C++ Multithreading Interview Deep Dive

---

# 1. Parallel STL Algorithms (C++17)

## Core Concepts

C++17 added execution policies to most standard algorithms, enabling parallel and vectorized execution without manual thread management.

### Three Execution Policies

```cpp
#include <algorithm>
#include <execution>
#include <vector>
#include <numeric>

std::vector<int> v(1000000);
std::iota(v.begin(), v.end(), 0); // fill with 0..999999

// 1. Sequential (same as no policy)
std::sort(std::execution::seq, v.begin(), v.end());

// 2. Parallel — may use multiple threads, order of element operations unspecified
std::sort(std::execution::par, v.begin(), v.end());

// 3. Parallel + Vectorized — may use SIMD within each thread
std::sort(std::execution::par_unseq, v.begin(), v.end());

// 4. Unseq only (C++20) — SIMD on a single thread
std::sort(std::execution::unseq, v.begin(), v.end());
```

### Algorithms That Accept Execution Policies

```cpp
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>

std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};
std::vector<int> result(v.size());

// sort
std::sort(std::execution::par, v.begin(), v.end());

// transform
std::transform(std::execution::par_unseq, v.begin(), v.end(),
               result.begin(), [](int x){ return x * x; });

// reduce (parallel-safe version of accumulate)
long long sum = std::reduce(std::execution::par, v.begin(), v.end(), 0LL);

// transform_reduce (map-reduce in one call)
long long sum_sq = std::transform_reduce(
    std::execution::par,
    v.begin(), v.end(),
    0LL,
    std::plus<long long>{},       // reduce operation
    [](int x){ return (long long)x * x; } // transform operation
);

// for_each
std::for_each(std::execution::par, v.begin(), v.end(),
              [](int& x){ x *= 2; });

// find, count, any_of, all_of, none_of
auto it = std::find(std::execution::par, v.begin(), v.end(), 10);
int cnt = std::count_if(std::execution::par, v.begin(), v.end(),
                        [](int x){ return x > 5; });

// exclusive_scan, inclusive_scan (parallel prefix sum)
std::vector<int> prefix(v.size());
std::exclusive_scan(std::execution::par, v.begin(), v.end(), prefix.begin(), 0);
```

### reduce vs accumulate

```cpp
// accumulate: sequential, left-fold, operation must be associative for parallelism
long long seq_sum = std::accumulate(v.begin(), v.end(), 0LL);

// reduce: can be parallel, operation MUST be associative AND commutative
// reduce does NOT guarantee evaluation order — use only for commutative ops
long long par_sum = std::reduce(std::execution::par, v.begin(), v.end(), 0LL);
// std::plus is commutative: fine
// concatenating strings: NOT commutative — wrong order in parallel!
```

### Exception Handling with Parallel Algorithms

```cpp
try {
    std::sort(std::execution::par, v.begin(), v.end());
} catch (std::exception& e) {
    // If any element operation throws, std::terminate() is called
    // (NOT exception propagation for par_unseq)
    // For std::execution::par, exceptions are caught and re-thrown as std::exception
}
// Rule: element operations (comparators, transforms) should not throw
// if using par_unseq
```

### When to Use Parallel Algorithms

```cpp
// Good candidates:
// - Large datasets (overhead of parallelism only pays off at ~10000+ elements)
// - CPU-bound operations (not I/O bound)
// - Embarrassingly parallel (no dependencies between elements)
// - Associative/commutative reduce operations

// Bad candidates:
// - Small vectors (overhead > speedup)
// - Operations with side effects or dependencies
// - Memory-bound operations (parallel may not help — memory is the bottleneck)
```

---

## Interview Q&A — Parallel STL

**Q1: What is the difference between std::reduce and std::accumulate?**
> `std::accumulate` is strictly sequential and preserves order. `std::reduce` may evaluate in any order and can be parallelized, but requires the operation to be commutative and associative (e.g., addition, multiplication — not string concatenation or subtraction).

**Q2: What happens when a parallel algorithm throws an exception?**
> For `std::execution::par`, if an element operation throws, the exception is propagated to the caller (as `std::exception` or specific type). For `std::execution::par_unseq`, any exception from element operations calls `std::terminate()`. So element operations in `par_unseq` must be noexcept.

**Q3: Is std::execution::par_unseq always faster than par?**
> No. `par_unseq` allows SIMD vectorization in addition to threading, which can be faster for simple operations. But it also has more restrictions: operations must be safe for SIMD (no mutexes, no allocation). Use profiling to decide.

**Q4: Do parallel algorithms guarantee any ordering?**
> No. The order of element operations is unspecified. The final result must be deterministic for correct programs (if the operation is commutative/associative), but individual steps may happen in any order.

---

# 2. std::shared_mutex + std::shared_lock (Read-Write Lock)

## Core Concepts

`std::shared_mutex` (C++17) allows multiple readers OR one exclusive writer — the classic reader-writer pattern.

### Reader-Writer Pattern

```cpp
#include <shared_mutex>
#include <mutex>
#include <unordered_map>
#include <string>

class ThreadSafeCache {
    mutable std::shared_mutex mtx_; // mutable for const read methods
    std::unordered_map<std::string, std::string> cache_;

public:
    // Multiple threads can read concurrently
    std::string get(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(mtx_); // shared (read) lock
        auto it = cache_.find(key);
        return (it != cache_.end()) ? it->second : "";
    }

    // Only one thread can write, and no readers while writing
    void set(const std::string& key, const std::string& value) {
        std::unique_lock<std::shared_mutex> lock(mtx_); // exclusive (write) lock
        cache_[key] = value;
    }

    bool contains(const std::string& key) const {
        std::shared_lock lock(mtx_); // C++17 CTAD
        return cache_.count(key) > 0;
    }

    void clear() {
        std::unique_lock lock(mtx_);
        cache_.clear();
    }
};
```

### Upgrading from Read Lock to Write Lock

```cpp
// C++ does NOT have a built-in upgrade operation (read → write)
// You must: release shared lock, acquire exclusive lock

class UpgradableCache {
    mutable std::shared_mutex mtx_;
    std::unordered_map<int, int> data_;
public:
    int get_or_compute(int key) {
        // First try: read lock
        {
            std::shared_lock rlock(mtx_);
            auto it = data_.find(key);
            if (it != data_.end()) return it->second;
        } // release read lock

        // Not found: acquire write lock
        std::unique_lock wlock(mtx_);
        // Re-check: another thread may have inserted while we were upgrading
        auto it = data_.find(key);
        if (it != data_.end()) return it->second;

        int val = expensive_compute(key);
        data_[key] = val;
        return val;
    }
};
```

### shared_timed_mutex

```cpp
#include <shared_mutex>
std::shared_timed_mutex stm;

// With timeout
if (stm.try_lock_shared_for(std::chrono::milliseconds(100))) {
    // got shared lock within 100ms
    stm.unlock_shared();
}
```

### When to Use shared_mutex

```cpp
// Beneficial when:
// - Reads are much more frequent than writes (e.g., config cache, read-heavy map)
// - Read operations are long enough to justify the overhead

// NOT beneficial when:
// - Writes are frequent (shared_mutex has more overhead than plain mutex)
// - Critical sections are very short (overhead of shared_mutex > contention savings)
// Rule: benchmark before replacing mutex with shared_mutex
```

---

## Interview Q&A — shared_mutex

**Q1: What is the writer starvation problem?**
> If readers are constantly acquiring the shared lock, a writer may never get the exclusive lock. Many implementations prioritize writers when they're waiting to prevent starvation, but the C++ standard doesn't mandate this.

**Q2: Can you upgrade a shared_lock to a unique_lock atomically in C++?**
> No, C++17/20 has no atomic upgrade. You must release the shared lock and then acquire the exclusive lock. Between the two, always re-check the condition (double-checked locking pattern).

**Q3: shared_mutex vs mutex — when does shared_mutex win?**
> When reads dominate (>80% reads), read operations are non-trivial (not just a single integer read), and reads don't modify the data. For simple reads, the overhead of `shared_mutex` (larger internal state, more complex locking) may outweigh the benefit.

---

# 3. std::latch / std::barrier (C++20)

## Core Concepts

`std::latch` and `std::barrier` are synchronization primitives for coordinating multiple threads. Both require C++20.

### std::latch — Single-Use Countdown

```cpp
#include <latch>
#include <thread>
#include <vector>
#include <iostream>

// latch: decremented by multiple threads, waits until count reaches zero
// Cannot be reset (single use)

void parallel_init(int n) {
    std::latch ready(n); // countdown from n

    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&ready, i]{
            // do initialization work
            std::cout << "Thread " << i << " initialized\n";
            ready.count_down(); // decrement latch
        });
    }

    ready.wait(); // blocks until latch reaches 0
    std::cout << "All threads initialized, proceeding\n";

    for (auto& t : threads) t.join();
}

// Use case 2: Start gate — all threads wait for a signal
void start_gate_example(int n) {
    std::latch start_signal(1); // one countdown needed

    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i) {
        threads.emplace_back([&start_signal, i]{
            start_signal.wait(); // all threads wait here
            std::cout << "Thread " << i << " started\n";
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    start_signal.count_down(); // release all at once

    for (auto& t : threads) t.join();
}

// arrive_and_wait: count_down + wait in one atomic call
void combined_example() {
    std::latch latch(3);
    auto worker = [&]{
        // do work
        latch.arrive_and_wait(); // signal completion AND wait for all others
        // all 3 threads reach here simultaneously
    };
    std::thread t1(worker), t2(worker), t3(worker);
    t1.join(); t2.join(); t3.join();
}
```

### std::barrier — Reusable Phase Synchronization

```cpp
#include <barrier>
#include <thread>
#include <vector>
#include <iostream>

// barrier: reusable — all threads wait, then proceed together (phases)
// Optional completion function runs once when all threads arrive

void parallel_phases(int n) {
    int phase = 0;

    // Completion function runs in one thread when all arrive at the barrier
    auto on_completion = [&]() noexcept {
        ++phase;
        std::cout << "Phase " << phase << " complete\n";
    };

    std::barrier sync(n, on_completion);

    auto worker = [&](int id) {
        for (int iter = 0; iter < 3; ++iter) {
            // Phase work
            std::cout << "Thread " << id << " working phase " << (iter+1) << "\n";

            sync.arrive_and_wait(); // wait for ALL threads to finish this phase
            // All threads simultaneously proceed to next phase
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < n; ++i)
        threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();
}

// arrive_and_drop: thread arrives but removes itself from future phases
void dynamic_participation() {
    std::barrier b(4);
    auto worker = [&](int id) {
        b.arrive_and_wait(); // phase 1: all 4 participate
        if (id == 3) {
            b.arrive_and_drop(); // thread 3 exits after phase 2
            return;
        }
        b.arrive_and_wait(); // phase 2
        b.arrive_and_wait(); // phase 3 (only 3 threads)
    };
}
```

### latch vs barrier vs condition_variable

```cpp
// std::latch:
// - Single use countdown
// - Any number of threads can count_down, any can wait
// - Counts don't have to match waiters

// std::barrier:
// - Reusable phase synchronization
// - Exact count of threads must participate each phase
// - Optional completion function

// std::condition_variable:
// - Most flexible, user-managed predicate
// - Requires mutex, predicate, and manual signal management
// - Best for irregular/conditional synchronization

// Prefer latch for: "wait until N things are done"
// Prefer barrier for: "N threads sync up at each phase"
// Prefer CV for: "wait until some condition is true"
```

---

## Interview Q&A — latch / barrier

**Q1: What is the difference between std::latch and std::barrier?**
> `std::latch` is single-use: once the count reaches zero it stays open (all future waits return immediately). `std::barrier` is reusable: it resets automatically after all expected threads arrive, supporting multiple synchronization phases.

**Q2: Can different threads count down a latch to different amounts?**
> Yes. `count_down(n)` decrements by `n` (default 1). Any combination works as long as total decrements reach the initial count. Any thread can call `wait()` regardless of whether they counted down.

**Q3: What is the completion function in std::barrier?**
> An optional callable passed to the barrier constructor. It's invoked exactly once per phase by one of the arriving threads (before any threads are released). Must be `noexcept`. Used for phase-boundary operations like swapping buffers.

**Q4: Can you use std::latch instead of std::condition_variable for a start gate?**
> Yes, and it's simpler. `std::latch latch(1); latch.wait();` in workers, `latch.count_down()` from the controller — all workers are released simultaneously.

---

# 4. Memory Model Internals

## Core Concepts

The C++ memory model defines how threads interact through shared memory. Understanding it is essential for writing correct lock-free code.

### The Happens-Before Relation

```
"X happens-before Y" means: all effects of X are visible to Y.

Built from:
1. Sequenced-before: within a single thread, X ; Y means X seq-before Y
2. Synchronizes-with: release store synchronizes-with acquire load of same atomic
3. Happens-before is the transitive closure of the above two
```

```cpp
// Example: establishing happens-before
std::atomic<int> flag{0};
int data = 0;

// Thread A:
data = 42;                          // (a) sequenced-before (b)
flag.store(1, std::memory_order_release); // (b)

// Thread B:
while (flag.load(std::memory_order_acquire) == 0); // (c) synchronizes-with (b)
assert(data == 42); // (a) happens-before (b) synchronizes-with (c)
                    // Therefore (a) happens-before consumer read of data
                    // assert never fires
```

### Sequential Consistency — Total Global Order

```cpp
// With seq_cst, all threads agree on a single total order of operations

std::atomic<bool> x{false}, y{false};
std::atomic<int> z{0};

void write_x() { x.store(true, std::memory_order_seq_cst); }
void write_y() { y.store(true, std::memory_order_seq_cst); }

void read_x_then_y() {
    while (!x.load(std::memory_order_seq_cst));
    if (y.load(std::memory_order_seq_cst)) ++z;
}

void read_y_then_x() {
    while (!y.load(std::memory_order_seq_cst));
    if (x.load(std::memory_order_seq_cst)) ++z;
}

// With seq_cst: z == 0 is impossible.
// Either x was stored before y or y before x in the total order.
// The thread reading second will see both as true.
```

### Acquire-Release — Partial Order (More Efficient)

```cpp
// Only establishes order between specific acquire-release pairs

std::atomic<int> payload{0};
std::atomic<bool> ready{false};

// Thread A (producer):
payload.store(42, std::memory_order_relaxed); // (1)
ready.store(true, std::memory_order_release); // (2) release

// Thread B (consumer):
while (!ready.load(std::memory_order_acquire)); // (3) acquire syncs with (2)
int val = payload.load(std::memory_order_relaxed); // (4)
// (1) happens-before (2) synchronizes-with (3) happens-before (4)
// So (4) sees 42. Correct!

// But there is NO total order on all operations — only a happens-before chain.
// Two independent acquire-release chains don't interact.
```

### Relaxed Ordering — Only Atomicity

```cpp
// No synchronization guarantees, just prevents data races on the atomic itself
std::atomic<int> count{0};

// Multiple threads incrementing a counter — relaxed is sufficient
// because we don't need to order other operations around this
void relaxed_count() {
    count.fetch_add(1, std::memory_order_relaxed);
}
// After all threads join: count is accurate (atomicity guaranteed)
// But no visibility guarantees for OTHER memory writes in those threads
```

### The Store Buffer Problem (Why Ordering Matters)

```cpp
// On x86 CPUs: stores can be buffered and not immediately visible to other CPUs
// On ARM/POWER: even more reordering is possible

// Without proper memory ordering:
int x = 0, y = 0; // not atomic

// Thread A:                Thread B:
x = 1;                     y = 1;
int a = y;                 int b = x;

// Can a == 0 and b == 0 simultaneously? On x86: No (TSO guarantees)
// On ARM/POWER: YES — stores can be reordered past loads
// With std::atomic and seq_cst: always at least one is 1
```

### Memory Fences

```cpp
#include <atomic>

// Standalone fence — not tied to a specific atomic
std::atomic_thread_fence(std::memory_order_release);
// Prevents: any store before the fence being reordered after it
// Also prevents: any load/store after being moved before

std::atomic_thread_fence(std::memory_order_acquire);
// Prevents: any load after being reordered before the fence

// Fence-based acquire-release (more granular than per-operation ordering):
std::atomic<bool> flag{false};
int data = 0;

// Producer:
data = 42;
std::atomic_thread_fence(std::memory_order_release);
flag.store(true, std::memory_order_relaxed);

// Consumer:
while (!flag.load(std::memory_order_relaxed));
std::atomic_thread_fence(std::memory_order_acquire);
assert(data == 42); // safe: fence establishes the synchronization
```

---

## Interview Q&A — Memory Model

**Q1: Explain the difference between seq_cst, acquire-release, and relaxed.**
> `seq_cst`: Single total order of all seq_cst operations across all threads. Most expensive on weakly-ordered CPUs (ARM, POWER). `acquire-release`: Establishes happens-before chains between specific pairs of stores and loads. More efficient — only orders what you pair up. `relaxed`: Only atomicity, zero ordering. Used for counters, flags where you don't need to synchronize other data.

**Q2: What is the "synchronizes-with" relationship?**
> A release store to an atomic variable X synchronizes-with an acquire load from the same X that observes the stored value. This creates a happens-before edge: everything before the release is visible after the acquire. This is how threads communicate data safely without mutex.

**Q3: Why is memory_order_consume rarely used?**
> `consume` is a weaker form of `acquire` that only orders operations data-dependent on the loaded value. Compilers typically promote it to `acquire` because tracking data dependencies across the compiler IR is difficult. It's effectively deprecated in practice.

**Q4: Can you have a data race on std::atomic variables?**
> No. By definition, `std::atomic` operations are atomic (no torn reads/writes) and the memory model guarantees freedom from data races on atomics themselves. However, if you use `memory_order_relaxed`, you can still have logical race conditions (wrong results) if you don't properly reason about ordering.

**Q5: What is the N4 memory model?**
> The C++11 memory model is sometimes called the DRF (Data Race Free) model: programs that are data-race-free have sequentially consistent behavior. Programs with data races have undefined behavior. The memory model provides tools (`std::atomic`) to write DRF programs with explicit control over ordering performance.

---

# 5. std::jthread (C++20) — Cooperative Cancellation

## Core Concepts

`std::jthread` is an improved `std::thread` that:
1. Automatically joins in its destructor (RAII)
2. Supports cooperative cancellation via `std::stop_token`

### Basic jthread Usage

```cpp
#include <thread>
#include <iostream>

void worker(int id) {
    std::cout << "Worker " << id << " running\n";
}

int main() {
    std::jthread t(worker, 1);
    // No need to call join()!
} // t's destructor: request_stop() then join() automatically
```

### Cooperative Cancellation with stop_token

```cpp
#include <thread>
#include <chrono>
#include <iostream>

void cancellable_worker(std::stop_token stoken, int id) {
    while (!stoken.stop_requested()) { // check cancellation flag
        std::cout << "Thread " << id << " working...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "Thread " << id << " cancelled gracefully\n";
}

int main() {
    std::jthread t(cancellable_worker, 42);
    // jthread automatically passes stop_token as first arg if function accepts it

    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    t.request_stop(); // signal cancellation
    // t.join() called automatically in destructor
}
```

### stop_token with condition_variable_any

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

std::mutex mtx;
std::condition_variable_any cv; // _any version supports stop_token
std::queue<int> work_queue;

void worker(std::stop_token stoken) {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        // Wait for work OR cancellation — no spurious wakeup issues
        cv.wait(lock, stoken, []{ return !work_queue.empty(); });

        if (stoken.stop_requested()) break; // cancelled

        int item = work_queue.front();
        work_queue.pop();
        lock.unlock();

        process(item);
    }
}

int main() {
    std::jthread t(worker);

    // Add work
    { std::lock_guard lock(mtx); work_queue.push(1); }
    cv.notify_one();

    std::this_thread::sleep_for(std::chrono::seconds(1));
    // t destructor: request_stop() -> condition notified -> thread exits
    // then join()
}
```

### stop_source and stop_callback

```cpp
#include <thread>
#include <iostream>

int main() {
    std::stop_source src; // you hold the stop_source
    std::stop_token token = src.get_token(); // share token with threads

    // stop_callback: run code when cancellation is requested
    std::stop_callback cb(token, []{
        std::cout << "Cancellation callback!\n";
    });

    std::jthread t([token]{
        while (!token.stop_requested())
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    src.request_stop(); // triggers callback and signals token
    // t's destructor will also call request_stop (already requested, no-op)
}
```

### jthread vs thread Comparison

```cpp
// std::thread:
{
    std::thread t(worker);
    // Must: t.join() OR t.detach() OR std::terminate() is called
    if (t.joinable()) t.join(); // explicit cleanup required
}

// std::jthread:
{
    std::jthread t(worker);
    // Nothing required — destructor handles everything
    // For cancellable workers, also calls request_stop() before joining
}
```

---

## Interview Q&A — std::jthread

**Q1: How does jthread pass the stop_token to the thread function?**
> If the thread function's first parameter is `std::stop_token`, jthread automatically passes its internal `stop_token`. Otherwise, jthread functions like a regular `std::thread`. You can also get the token manually via `jthread::get_stop_token()`.

**Q2: What happens to pending stop_callbacks when request_stop is called?**
> All registered `std::stop_callback` objects are called synchronously by the thread that calls `request_stop()`, before `request_stop()` returns. If a callback throws, `std::terminate()` is called.

**Q3: Is std::jthread's stopping truly cooperative?**
> Yes. The thread must actively check `stop_requested()` or use `condition_variable_any::wait` with a stop_token. The requesting thread cannot force termination — it only sets a flag. The target thread must cooperate by checking and acting on it.

**Q4: Can you use jthread without cooperative cancellation?**
> Yes. If you don't check `stop_requested()`, jthread behaves exactly like a thread with automatic join in the destructor. The stop mechanism is purely opt-in.

---

## Summary Table — Tier 3 Quick Reference

| Primitive | C++ Standard | Key Use Case | Key Limitation |
|-----------|-------------|--------------|----------------|
| `std::execution::par` | C++17 | Parallel algorithms | Requires large N for speedup |
| `std::reduce` | C++17 | Parallel sum/fold | Requires commutative + associative op |
| `std::shared_mutex` | C++17 | Read-heavy caches | Writer starvation; no upgrade |
| `std::shared_lock` | C++17 | Multiple reader access | No upgrade to exclusive |
| `std::latch` | C++20 | One-time countdown barrier | Single use only |
| `std::barrier` | C++20 | Phased thread synchronization | Exact count per phase |
| `std::jthread` | C++20 | RAII thread + cancellation | Requires C++20 |
| `std::stop_token` | C++20 | Cooperative cancellation | Thread must check it |
| `memory_order_acquire/release` | C++11 | Efficient sync without seq_cst | Must pair correctly |
| `std::atomic_thread_fence` | C++11 | Fine-grained memory barriers | Complex to reason about |
