# Tier 4 — Advanced / Specialist
## C++ Multithreading Interview Deep Dive

---

# 1. Lock-Free Data Structures Using std::atomic

## Core Concepts

Lock-free programming eliminates mutexes entirely. Instead of blocking, threads use atomic compare-and-swap (CAS) operations to detect conflicts and retry. The guarantee: at least one thread makes progress at all times (no deadlock, no livelock under most conditions).

### Lock-Free Guarantees (Hierarchy)

```
Obstruction-free: A thread makes progress if all others are suspended
Lock-free:        At least one thread makes progress at all times
Wait-free:        Every thread makes progress in bounded steps (strongest, hardest)
```

### Lock-Free Stack (Treiber Stack)

```cpp
#include <atomic>
#include <memory>
#include <optional>

template<typename T>
class LockFreeStack {
    struct Node {
        T data;
        Node* next = nullptr;
        explicit Node(T d) : data(std::move(d)) {}
    };

    std::atomic<Node*> head_{nullptr};

public:
    void push(T data) {
        Node* new_node = new Node(std::move(data));
        // CAS loop: keep retrying until we win the race
        new_node->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(
            new_node->next,          // expected (updated on failure)
            new_node,                // desired
            std::memory_order_release,   // success ordering
            std::memory_order_relaxed))  // failure ordering
            ; // retry
    }

    std::optional<T> pop() {
        Node* old_head = head_.load(std::memory_order_acquire);
        while (old_head) {
            if (head_.compare_exchange_weak(
                old_head,
                old_head->next,
                std::memory_order_acquire,
                std::memory_order_acquire)) {
                T data = std::move(old_head->data);
                // Memory reclamation problem: can't safely delete here!
                // Another thread may still be reading old_head->next
                delete old_head; // UNSAFE — see memory reclamation below
                return data;
            }
            // old_head updated to current head — retry
        }
        return std::nullopt; // empty stack
    }

    ~LockFreeStack() {
        Node* curr = head_.load();
        while (curr) {
            Node* next = curr->next;
            delete curr;
            curr = next;
        }
    }
};
```

### The ABA Problem

```cpp
// Scenario:
// Stack: A -> B -> C
// Thread 1: reads head = A, about to CAS A -> B (pop A)
// Thread 1 gets preempted
// Thread 2: pops A, pops B, pushes A back (A was reused)
// Stack is now: A -> C
// Thread 1 resumes: CAS succeeds (head is still A!)
// But now head->next is C, not B — B is lost! Corruption!

// ABA occurs when:
// 1. Read value A
// 2. A changes to B then back to A
// 3. CAS sees A and "succeeds" but state has changed

// Solution 1: Tagged pointer (version counter)
struct TaggedPointer {
    Node* ptr;
    uintptr_t tag; // increment on every CAS
};
// Use std::atomic<TaggedPointer> — requires 128-bit atomics (CMPXCHG16B on x86)

// Solution 2: Hazard pointers (see below)
// Solution 3: Epoch-based reclamation (RCU-style)
```

### Hazard Pointers — Safe Memory Reclamation

```cpp
// Concept: before accessing a node, register it as "hazardous" (in use)
// Before deleting, check if any thread has it registered as hazardous

#include <atomic>
#include <vector>
#include <algorithm>

// Simplified hazard pointer system (production uses more sophisticated impl)
constexpr int MAX_THREADS = 64;
std::atomic<void*> hazard_pointers[MAX_THREADS];

class HazardGuard {
    int slot_;
public:
    HazardGuard(int slot, void* ptr) : slot_(slot) {
        hazard_pointers[slot_].store(ptr, std::memory_order_seq_cst);
    }
    ~HazardGuard() {
        hazard_pointers[slot_].store(nullptr, std::memory_order_seq_cst);
    }
};

bool is_hazardous(void* ptr) {
    for (int i = 0; i < MAX_THREADS; ++i)
        if (hazard_pointers[i].load(std::memory_order_seq_cst) == ptr)
            return true;
    return false;
}
// Nodes are only deleted when is_hazardous returns false
```

### Lock-Free Queue (Michael-Scott Queue)

```cpp
#include <atomic>
#include <memory>
#include <optional>

template<typename T>
class LockFreeQueue {
    struct Node {
        std::optional<T> data;
        std::atomic<Node*> next{nullptr};
        Node() = default;
        explicit Node(T d) : data(std::move(d)) {}
    };

    // Dummy head node (sentinel)
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;

public:
    LockFreeQueue() {
        Node* dummy = new Node(); // empty sentinel
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
    }

    void enqueue(T data) {
        Node* new_node = new Node(std::move(data));
        while (true) {
            Node* tail = tail_.load(std::memory_order_acquire);
            Node* next = tail->next.load(std::memory_order_acquire);

            if (tail == tail_.load(std::memory_order_acquire)) {
                if (next == nullptr) {
                    // Try to link new node at the end
                    if (tail->next.compare_exchange_weak(
                        next, new_node,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                        // Try to advance tail (ok if we fail — another thread will)
                        tail_.compare_exchange_strong(
                            tail, new_node,
                            std::memory_order_release,
                            std::memory_order_relaxed);
                        return;
                    }
                } else {
                    // Tail is lagging — help advance it
                    tail_.compare_exchange_strong(
                        tail, next,
                        std::memory_order_release,
                        std::memory_order_relaxed);
                }
            }
        }
    }

    std::optional<T> dequeue() {
        while (true) {
            Node* head = head_.load(std::memory_order_acquire);
            Node* tail = tail_.load(std::memory_order_acquire);
            Node* next = head->next.load(std::memory_order_acquire);

            if (head == head_.load(std::memory_order_acquire)) {
                if (head == tail) {
                    if (!next) return std::nullopt; // empty
                    // Tail is lagging — advance it
                    tail_.compare_exchange_strong(tail, next,
                        std::memory_order_release, std::memory_order_relaxed);
                } else {
                    T data = std::move(*next->data);
                    if (head_.compare_exchange_weak(head, next,
                        std::memory_order_release, std::memory_order_relaxed)) {
                        delete head; // safe: we have exclusive ownership
                        return data;
                    }
                }
            }
        }
    }
};
```

### When to Use Lock-Free vs Mutex

```cpp
// Use lock-free when:
// - Contention is very high and lock overhead is measurable
// - You need progress guarantees (real-time, interrupt handlers)
// - The critical section is very short (CAS is cheap when uncontested)

// Use mutex when:
// - Correctness is more important than max throughput
// - Logic in critical section is complex
// - Profiling confirms mutex is the bottleneck (don't optimize prematurely!)
// - Memory reclamation is complex (lock-free doesn't simplify this)

// Reality check: std::mutex on uncontested paths is ~20-50 ns
// A CAS on a shared cache line under contention: ~100-300 ns
// Lock-free is NOT always faster under contention!
```

---

## Interview Q&A — Lock-Free Data Structures

**Q1: What does "lock-free" mean? Is it the same as "non-blocking"?**
> "Non-blocking" is an umbrella term. Lock-free specifically means: at least one thread makes progress at all times. It rules out deadlock. "Wait-free" is stronger: every thread makes progress in a bounded number of steps. Lock-free algorithms may still have individual threads that spin indefinitely (livelock-like behavior under adversarial scheduling).

**Q2: What is the ABA problem and how do you prevent it?**
> The ABA problem: a thread reads value A, another thread changes A→B→A (reusing the address), the first thread's CAS sees A and "succeeds" even though the logical state has changed. Solutions: tagged pointers (version counter stored alongside the pointer in a 128-bit atomic), hazard pointers (track which nodes are in use), or epoch-based reclamation.

**Q3: Why is memory reclamation hard in lock-free data structures?**
> When you CAS a node out of the structure, another thread might still hold a pointer to it (read it before the CAS). If you `delete` it immediately, that thread has a dangling pointer. Solutions: hazard pointers, epoch-based reclamation (RCU), defer deletion until no thread can possibly be accessing the node.

**Q4: What is compare_exchange_weak vs compare_exchange_strong in lock-free code?**
> `compare_exchange_weak` can fail spuriously (return false even when value == expected) due to LL/SC instruction pairs on ARM/POWER. Since lock-free code always uses CAS in a loop anyway, use `weak` — it's faster on ARM and equivalent on x86. Use `strong` only when you can't afford a spurious failure (non-loop contexts).

**Q5: Can std::atomic operations themselves be the bottleneck?**
> Yes. Under high contention, multiple threads repeatedly failing CAS can be worse than a mutex (which puts threads to sleep instead of spinning). Exponential backoff, thread-local accumulation, or reducing the scope of sharing are techniques to reduce this.

---

# 2. Coroutines (C++20) — Concept Level

## Core Concepts

A **coroutine** is a function that can be suspended and resumed. Unlike threads, coroutines are cooperative: they yield control explicitly. They enable asynchronous programming without callbacks or threads.

### Three Coroutine Keywords

```cpp
// co_await:  suspend until awaitable is ready
// co_yield:  suspend and produce a value (generators)
// co_return: complete the coroutine with a value

// A function containing any of these is a coroutine (not a regular function)
```

### Generator — Lazy Sequence

```cpp
// Requires a coroutine frame and promise_type — boilerplate heavy
// In practice, use std::generator (C++23) or a library

#include <coroutine>
#include <optional>

// Simplified generator implementation
template<typename T>
struct Generator {
    struct promise_type {
        std::optional<T> value_;

        Generator get_return_object() {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() { return {}; } // suspend immediately
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() { std::rethrow_exception(std::current_exception()); }

        std::suspend_always yield_value(T val) {
            value_ = std::move(val);
            return {}; // suspend after yielding
        }
    };

    std::coroutine_handle<promise_type> handle_;

    explicit Generator(std::coroutine_handle<promise_type> h) : handle_(h) {}
    ~Generator() { if (handle_) handle_.destroy(); }

    bool next() {
        handle_.resume(); // resume coroutine until next co_yield or co_return
        return !handle_.done();
    }

    T value() { return *handle_.promise().value_; }
};

Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        auto tmp = a;
        a = b;
        b += tmp;
    }
}

int main() {
    auto gen = fibonacci();
    for (int i = 0; i < 10; ++i) {
        gen.next();
        std::cout << gen.value() << " ";
    }
    // 0 1 1 2 3 5 8 13 21 34
}
```

### Async Coroutine — Awaitable

```cpp
#include <coroutine>
#include <thread>
#include <functional>

// Simplified Task type for async operations
struct Task {
    struct promise_type {
        Task get_return_object() {
            return Task{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_never initial_suspend() { return {}; } // run immediately
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() {}
        void unhandled_exception() {}
    };

    std::coroutine_handle<promise_type> handle_;
    Task(std::coroutine_handle<promise_type> h) : handle_(h) {}
};

// Awaitable that resumes coroutine on a new thread
struct ThreadAwaitable {
    bool await_ready() { return false; } // always suspend
    void await_suspend(std::coroutine_handle<> h) {
        std::thread([h]{ h.resume(); }).detach(); // resume on new thread
    }
    void await_resume() {} // no return value
};

Task async_work() {
    std::cout << "Before await (thread " << std::this_thread::get_id() << ")\n";
    co_await ThreadAwaitable{}; // suspend, resume on new thread
    std::cout << "After await (thread " << std::this_thread::get_id() << ")\n";
}
```

### Coroutine Concepts (Interview Level)

```cpp
// Key coroutine concepts:

// 1. Coroutine frame: heap allocation holding local variables and state
//    (compilers optimize this to stack allocation when possible)

// 2. promise_type: controls coroutine behavior
//    - get_return_object(): creates the handle returned to caller
//    - initial_suspend(): suspend immediately or run to first co_await?
//    - final_suspend(): what to do when coroutine finishes
//    - return_value()/return_void(): handle co_return
//    - yield_value(): handle co_yield

// 3. Awaitable: any object with await_ready, await_suspend, await_resume
//    - await_ready(): if true, don't suspend (optimization)
//    - await_suspend(handle): called when suspending, schedule resumption
//    - await_resume(): called when resuming, returns value of co_await expr

// 4. Coroutine handle: std::coroutine_handle<promise_type>
//    - resume(): continue execution until next suspension
//    - done(): is coroutine at final suspend?
//    - destroy(): free coroutine frame
//    - promise(): access the promise object
```

### Coroutines vs Threads

```cpp
// Coroutines:
// - Cooperative scheduling (must explicitly yield)
// - Single-threaded by default (unless awaitable schedules on thread pool)
// - Much cheaper than threads (coroutine frame << thread stack)
// - No synchronization needed (single-threaded execution)
// - Good for: I/O bound, many concurrent tasks (10k+ coroutines)

// Threads:
// - Preemptive scheduling (OS controls)
// - True parallelism (multiple CPUs)
// - Expensive (~1MB stack, OS scheduler overhead)
// - Needs synchronization for shared data
// - Good for: CPU-bound parallel computation

// Async frameworks (Boost.Asio, cppcoro, folly) combine both:
// coroutines for concurrency + thread pool for parallelism
```

---

## Interview Q&A — Coroutines

**Q1: What is the difference between a coroutine and a thread?**
> A thread is preemptively scheduled by the OS and can run in parallel with other threads. A coroutine cooperatively yields control — it suspends only at explicit `co_await`/`co_yield` points, and resumes are scheduled by the programmer or a runtime. Coroutines are much lighter than threads (no OS scheduling overhead, no stack allocation for each).

**Q2: What are the three coroutine keywords and what do they do?**
> `co_await`: suspends the coroutine until the awaitable signals it's ready, then resumes. `co_yield`: suspends and produces a value to the caller. `co_return`: completes the coroutine, optionally returning a value.

**Q3: What is promise_type in a coroutine?**
> It's a nested type in the return type that the compiler looks for to understand how to behave. It defines: how to create the coroutine's return object, whether to suspend immediately, how to handle `co_return`, `co_yield`, and unhandled exceptions.

**Q4: What is the cost of a coroutine suspension/resumption compared to a thread context switch?**
> A coroutine suspension/resumption is typically 10-100 nanoseconds (similar to a function call). A thread context switch costs ~1-10 microseconds. This is why coroutines can efficiently handle millions of concurrent operations where threads cannot.

---

# 3. Transactional Memory — Overview

## Core Concepts

Transactional memory (TM) allows a block of code to execute atomically, like a database transaction. If the transaction conflicts with another thread's accesses, it's rolled back and retried automatically.

### Syntax (GCC Transactional Memory Extension — Not Standard C++)

```cpp
// Note: Transactional Memory is NOT in the C++ standard (as of C++23)
// Available as a GCC extension with -fgnu-tm flag
// Intel TSX provides hardware transactional memory on some x86 CPUs

// Synchronized transaction (like mutex — slower but safe)
__transaction_atomic {
    // All reads and writes in here are atomic
    account_a -= 100;
    account_b += 100;
} // Commits atomically or aborts and retries

// Relaxed transaction (hardware TM — faster, but may abort due to capacity)
__transaction_relaxed {
    counter++;
}
```

### Software Transactional Memory Concept

```cpp
// Concept (not compilable standard C++):
//
// begin_transaction:
//   - Record speculative log of reads/writes
// on_conflict:
//   - Abort: discard speculative log, retry
// on_success:
//   - Commit: make all writes visible atomically
//
// Advantages:
//   - Composable: two TM operations can be nested safely
//   - Deadlock-free: transactions abort instead of deadlocking
//   - Optimistic: assumes no conflict, fast in low-contention cases
//
// Disadvantages:
//   - High overhead in high-contention (many aborts)
//   - Cannot perform irreversible operations (I/O) in transactions
//   - HTM (hardware TM) has capacity limits — large transactions abort
//   - Poorly supported in current C++ compilers/standards
```

### Hardware Transactional Memory (Intel TSX)

```cpp
// Intel RTM (Restricted Transactional Memory) — hardware support
#include <immintrin.h>

int status = _xbegin();
if (status == _XBEGIN_STARTED) {
    // Transaction body
    shared_counter++;
    _xend(); // commit
} else {
    // Aborted — use fallback (lock)
    std::lock_guard<std::mutex> lock(mtx);
    shared_counter++;
}
// HTM aborts on: capacity overflow, unsupported instructions,
// interrupts, other thread's conflicting write
```

### Why Transactional Memory Is "Mostly Theoretical" for Interviews

```cpp
// Current status:
// - Not in C++ standard
// - GCC TM extension is experimental, rarely used in production
// - Intel TSX was disabled on some CPUs due to security vulnerabilities (TAA)
// - Requires careful handling of irreversible operations
// - Performance benefits are workload-specific

// For interviews: know the concept, motivations, and limitations
// Practical alternative: use fine-grained locking or lock-free structures
```

---

## Interview Q&A — Transactional Memory

**Q1: What problem does transactional memory solve?**
> It provides composable atomicity without locks. With locks, composing two thread-safe operations (e.g., two separate thread-safe stacks) is not automatically thread-safe. With TM, you can combine operations into a single atomic transaction.

**Q2: What is an abort in transactional memory?**
> When a transaction detects a conflict (another thread modified a variable it read), it discards all its writes and restarts from the beginning. Like a database transaction rollback.

**Q3: What are the limitations of hardware transactional memory?**
> Capacity limits (transaction data must fit in L1/L2 cache), aborts on interrupts/system calls, cannot include I/O operations or irreversible actions, and security vulnerabilities (TAA) caused Intel to disable TSX on some CPUs.

---

# 4. CppMem — Understanding Memory Model Operations

## Core Concepts

CppMem (cppmem.cl.cam.ac.uk) is a tool for exploring the C++ memory model. You describe a small concurrent program and it shows you all possible executions and which memory orderings are consistent.

### How to Use CppMem

```
Website: http://svr-pes20-cppmem.cl.cam.ac.uk/cppmem/
          (also: http://cppmem.cl.cam.ac.uk/)

Interface:
1. Write a small C-like concurrent program using CppMem syntax
2. Choose a memory model (C++11, Linux kernel, etc.)
3. Press "Run"
4. CppMem shows: all consistent executions, which are valid/invalid under the model
```

### CppMem Syntax

```cpp
// CppMem uses a simplified syntax:

// Atomic variable declaration
atomic_int x = 0;
atomic_int y = 0;

// Thread notation
{{{
    // Thread 1
    x.store(1, seq_cst) ||
    // Thread 2 (runs concurrently)
    y.store(1, seq_cst)
}}}

// Full example: store buffering (shows that SC prevents both reads being 0)
int main() {
    atomic_int x = 0;
    atomic_int y = 0;
    {{{
        { x.store(1, seq_cst);
          r1 = y.load(seq_cst); }
        |||
        { y.store(1, seq_cst);
          r2 = x.load(seq_cst); }
    }}}
    // CppMem shows: r1==0 && r2==0 is NOT a valid execution under seq_cst
}
```

### What CppMem Reveals

```cpp
// Example 1: Relaxed — allows surprising behaviors
atomic_int x = 0;
atomic_int y = 0;
{{{
    { x.store(1, relaxed); r1 = y.load(relaxed); }
    |||
    { y.store(1, relaxed); r2 = x.load(relaxed); }
}}}
// CppMem shows: r1==0, r2==0 is VALID under relaxed ordering!
// This cannot happen with seq_cst.

// Example 2: Acquire-release — establishes happens-before
atomic_int flag = 0;
int data = 0;
{{{
    { data = 42; flag.store(1, release); }
    |||
    { r1 = flag.load(acquire); r2 = data; }
}}}
// CppMem shows: if r1==1 then r2==42 always (happens-before chain)
// if r1==0 then r2 can be 0 (flag not seen yet)
```

### Key CppMem Test Cases to Know

```cpp
// 1. Message passing (acquire-release works):
//    Producer: data=42; flag.store(1, release)
//    Consumer: while(!flag.load(acquire)); assert(data==42) — always safe

// 2. Store buffering (seq_cst prevents both reads being 0):
//    Thread A: x=1; r1=y;
//    Thread B: y=1; r2=x;
//    With seq_cst: r1==0 && r2==0 is impossible
//    With relaxed: r1==0 && r2==0 is possible

// 3. Independent reads of independent writes (IRIW — shows seq_cst strength):
//    Thread A: x.store(1, sc)
//    Thread B: y.store(1, sc)
//    Thread C: r1=x.load(sc); r2=y.load(sc)
//    Thread D: r3=y.load(sc); r4=x.load(sc)
//    With seq_cst: C and D see stores in same order
//    With release: they may see different orders

// 4. Dekker's algorithm (mutual exclusion without mutex):
//    Uses seq_cst to ensure both threads agree on who goes first
```

---

## Interview Q&A — CppMem

**Q1: What is CppMem used for?**
> CppMem is an academic tool for validating small concurrent programs against the C++ memory model. It enumerates all possible executions and checks which are consistent with the chosen ordering. It's valuable for understanding subtle memory model behaviors that are hard to reason about manually.

**Q2: Why is it important to understand CppMem even if you never use the tool?**
> It teaches you to reason precisely about memory ordering. The scenarios it demonstrates (store buffering, message passing, IRIW) are the same patterns you encounter when writing lock-free code. Understanding what orderings prevent which anomalies is essential for writing correct atomics code.

**Q3: What is the "store buffer" effect and what ordering prevents it?**
> The store buffer effect: CPU A's write to X may be buffered and not visible to CPU B before CPU B reads X. This allows both `r1==0 && r2==0` in a store-buffering test. `seq_cst` prevents this by requiring a total order on all SC operations. `acquire-release` alone does NOT prevent it.

---

## Advanced Topics: Common Interview Patterns Summary

### Lock-Free Stack Interview Pattern

```cpp
// This is commonly asked as a design question:
// "Design a lock-free stack"

// Key points to mention:
// 1. Use atomic head pointer
// 2. Push: CAS on head (new_node->next = old_head; CAS head: old_head -> new_node)
// 3. Pop: CAS on head (new_head = old_head->next; CAS head: old_head -> new_head)
// 4. ABA problem: discuss tagged pointers or hazard pointers
// 5. Memory reclamation: can't delete immediately (another thread may be reading)
// 6. Use compare_exchange_weak in loops (spurious failure OK, retry anyway)
```

### Producer-Consumer System Design

```cpp
// Common system design question — use this template:

// Option 1: mutex + condition_variable (simple, recommended)
// Option 2: lock-free queue (high-performance, complex)
// Option 3: std::async + future pipeline (task-based)

// Always discuss:
// - How many producers and consumers
// - Bounded vs unbounded queue
// - Backpressure mechanism (what if queue is full)
// - Shutdown/cancellation
// - Exception handling
```

---

## Summary Table — Tier 4 Quick Reference

| Topic | C++ Standard | Interview Importance | Key Concept |
|-------|-------------|---------------------|-------------|
| Lock-free stack | C++11 | High (design question) | CAS loop, ABA problem |
| Lock-free queue | C++11 | Medium (know MS queue) | Sentinel node, helping |
| ABA problem | N/A (concept) | High | Tagged pointer solution |
| Hazard pointers | N/A (library) | Medium | Safe memory reclamation |
| Coroutines | C++20 | Medium (conceptual) | co_await, promise_type |
| Transactional memory | Not standard | Low | Concept + limitations |
| CppMem | Tool | Low | Memory model validation |
| HTM (Intel TSX) | Platform-specific | Low | Hardware acceleration for TM |

---

## Final Interview Preparation Checklist

### Things You Should Be Able to Code from Memory (Tier 1-2)
- [ ] Create and join threads, pass arguments by value and reference
- [ ] Thread-safe counter with mutex + lock_guard
- [ ] Detect and fix a deadlock (lock ordering, scoped_lock)
- [ ] Producer-consumer with condition_variable (with predicate)
- [ ] Atomic counter with memory_order
- [ ] Acquire-release message passing pattern
- [ ] std::async + future for parallel sum
- [ ] Thread-safe Meyers singleton
- [ ] RAII thread guard

### Things You Should Understand Deeply (Tier 3)
- [ ] Explain happens-before, synchronizes-with
- [ ] Difference between seq_cst, acquire-release, relaxed
- [ ] When to use shared_mutex vs mutex
- [ ] How std::barrier enables phased parallel algorithms
- [ ] What jthread provides over thread

### Things You Should Know Conceptually (Tier 4)
- [ ] How a lock-free stack CAS loop works
- [ ] What the ABA problem is and how to fix it
- [ ] What a coroutine is and the difference from a thread
- [ ] Why transactional memory is not widely used
