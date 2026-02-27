# C++ Multithreading — Interview Prep Resources

> Based on the educative.io C++ Multithreading course syllabus.
> Primary free references: **cppreference.com** and **modernescpp.com** (Rainer Grimm's blog).

---

## Primary Reference Sites

| Resource | Use Case |
|----------|----------|
| [cppreference: Thread support](https://en.cppreference.com/w/cpp/thread) | Full index of all threading primitives |
| [modernescpp.com](https://www.modernescpp.com) | Rainer Grimm's free blog — covers nearly every topic in this course with examples |
| [C++ Core Guidelines (Concurrency section)](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-concurrency) | Best practices (CP.*) |
| [isocpp.org FAQ](https://isocpp.org/wiki/faq) | General C++ guidance |

---

## Introduction & Foundation

| Topic | Free Link |
|-------|-----------|
| C++11 overview | [cppreference: C++11](https://en.cppreference.com/w/cpp/11) |
| C++14 overview | [cppreference: C++14](https://en.cppreference.com/w/cpp/14) |
| C++17 overview | [cppreference: C++17](https://en.cppreference.com/w/cpp/17) |
| C++20 overview | [cppreference: C++20](https://en.cppreference.com/w/cpp/20) |
| Thread support library | [cppreference: Thread support](https://en.cppreference.com/w/cpp/thread) |

---

## Memory Model: Contract & Foundation

| Topic | Free Link |
|-------|-----------|
| Memory model overview | [cppreference: Memory model](https://en.cppreference.com/w/cpp/language/memory_model) |
| Data races | [cppreference: Memory model (data race)](https://en.cppreference.com/w/cpp/language/memory_model) |
| `std::memory_order` (all orderings) | [cppreference: memory_order](https://en.cppreference.com/w/cpp/atomic/memory_order) |

---

## Memory Model: Atomics

| Topic | Free Link |
|-------|-----------|
| Atomic operations library | [cppreference: Atomic](https://en.cppreference.com/w/cpp/atomic) |
| `std::atomic` | [cppreference: std::atomic](https://en.cppreference.com/w/cpp/atomic/atomic) |
| `std::atomic<bool>` | [cppreference: std::atomic\<bool\>](https://en.cppreference.com/w/cpp/atomic/atomic) |
| `std::atomic_flag` | [cppreference: atomic_flag](https://en.cppreference.com/w/cpp/atomic/atomic_flag) |
| Free atomic functions | [cppreference: atomic free functions](https://en.cppreference.com/w/cpp/atomic/atomic_load) |
| `std::atomic<std::shared_ptr>` (C++20) | [cppreference: atomic shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr/atomic2) |
| Spinlock vs Mutex (concept) | [modernescpp.com blog](https://www.modernescpp.com) |
| User-defined atomics | [cppreference: std::atomic (specializations)](https://en.cppreference.com/w/cpp/atomic/atomic) |

---

## Memory Model: Synchronization & Ordering Constraints

| Topic | Free Link |
|-------|-----------|
| All memory orders | [cppreference: memory_order](https://en.cppreference.com/w/cpp/atomic/memory_order) |
| Sequential consistency (`seq_cst`) | [cppreference: memory_order_seq_cst](https://en.cppreference.com/w/cpp/atomic/memory_order) |
| Acquire-Release semantics | [cppreference: memory_order_acquire / release](https://en.cppreference.com/w/cpp/atomic/memory_order) |
| `std::memory_order_consume` | [cppreference: memory_order_consume](https://en.cppreference.com/w/cpp/atomic/memory_order) |
| Relaxed semantic | [cppreference: memory_order_relaxed](https://en.cppreference.com/w/cpp/atomic/memory_order) |

---

## Memory Model: Fences

| Topic | Free Link |
|-------|-----------|
| `std::atomic_thread_fence` | [cppreference: atomic_thread_fence](https://en.cppreference.com/w/cpp/atomic/atomic_thread_fence) |
| `std::atomic_signal_fence` | [cppreference: atomic_signal_fence](https://en.cppreference.com/w/cpp/atomic/atomic_signal_fence) |

---

## Multithreading: Threads

| Topic | Free Link |
|-------|-----------|
| `std::thread` | [cppreference: std::thread](https://en.cppreference.com/w/cpp/thread/thread) |
| Thread creation & joining | [cppreference: thread::join](https://en.cppreference.com/w/cpp/thread/thread/join) |
| `thread::detach` | [cppreference: thread::detach](https://en.cppreference.com/w/cpp/thread/thread/detach) |
| `thread::joinable` | [cppreference: thread::joinable](https://en.cppreference.com/w/cpp/thread/thread/joinable) |
| Passing arguments | [cppreference: thread constructor](https://en.cppreference.com/w/cpp/thread/thread/thread) |
| `std::this_thread` | [cppreference: this_thread](https://en.cppreference.com/w/cpp/thread/this_thread) |
| `std::jthread` (C++20) | [cppreference: jthread](https://en.cppreference.com/w/cpp/thread/jthread) |

---

## Multithreading: Shared Data (Mutexes & Locks)

| Topic | Free Link |
|-------|-----------|
| Mutex types overview | [cppreference: Mutex](https://en.cppreference.com/w/cpp/thread#Mutual_exclusion) |
| `std::mutex` | [cppreference: std::mutex](https://en.cppreference.com/w/cpp/thread/mutex) |
| `std::recursive_mutex` | [cppreference: recursive_mutex](https://en.cppreference.com/w/cpp/thread/recursive_mutex) |
| `std::timed_mutex` | [cppreference: timed_mutex](https://en.cppreference.com/w/cpp/thread/timed_mutex) |
| `std::shared_mutex` (C++17) | [cppreference: shared_mutex](https://en.cppreference.com/w/cpp/thread/shared_mutex) |
| `std::lock_guard` | [cppreference: lock_guard](https://en.cppreference.com/w/cpp/thread/lock_guard) |
| `std::unique_lock` | [cppreference: unique_lock](https://en.cppreference.com/w/cpp/thread/unique_lock) |
| `std::shared_lock` (C++17) | [cppreference: shared_lock](https://en.cppreference.com/w/cpp/thread/shared_lock) |
| `std::scoped_lock` (C++17) | [cppreference: scoped_lock](https://en.cppreference.com/w/cpp/thread/scoped_lock) |
| Deadlocks | [cppreference: deadlock avoidance](https://en.cppreference.com/w/cpp/thread/lock) |
| `std::lock` (simultaneous locking) | [cppreference: std::lock](https://en.cppreference.com/w/cpp/thread/lock) |

---

## Thread-Safe Initialization

| Topic | Free Link |
|-------|-----------|
| `std::call_once` / `std::once_flag` | [cppreference: call_once](https://en.cppreference.com/w/cpp/thread/call_once) |
| `constexpr` for thread-safe init | [cppreference: constexpr](https://en.cppreference.com/w/cpp/language/constexpr) |
| Static local variables (C++11 magic statics) | [cppreference: storage duration](https://en.cppreference.com/w/cpp/language/storage_duration) |

---

## Multithreading: Local Data

| Topic | Free Link |
|-------|-----------|
| `thread_local` storage | [cppreference: thread_local](https://en.cppreference.com/w/cpp/language/storage_duration#Storage_duration) |

---

## Multithreading: Condition Variables

| Topic | Free Link |
|-------|-----------|
| `std::condition_variable` | [cppreference: condition_variable](https://en.cppreference.com/w/cpp/thread/condition_variable) |
| `std::condition_variable_any` | [cppreference: condition_variable_any](https://en.cppreference.com/w/cpp/thread/condition_variable_any) |
| Spurious wakeups (caveats) | [cppreference: wait with predicate](https://en.cppreference.com/w/cpp/thread/condition_variable/wait) |

---

## Multithreading: Tasks (async / futures / promises)

| Topic | Free Link |
|-------|-----------|
| Threads vs Tasks (concept) | [modernescpp.com blog](https://www.modernescpp.com) |
| `std::async` | [cppreference: std::async](https://en.cppreference.com/w/cpp/thread/async) |
| Launch policies (`std::launch`) | [cppreference: std::launch](https://en.cppreference.com/w/cpp/thread/launch) |
| `std::future` | [cppreference: std::future](https://en.cppreference.com/w/cpp/thread/future) |
| `std::shared_future` | [cppreference: shared_future](https://en.cppreference.com/w/cpp/thread/shared_future) |
| `std::promise` | [cppreference: std::promise](https://en.cppreference.com/w/cpp/thread/promise) |
| `std::packaged_task` | [cppreference: packaged_task](https://en.cppreference.com/w/cpp/thread/packaged_task) |
| `std::future_error` / exceptions | [cppreference: future_error](https://en.cppreference.com/w/cpp/thread/future_error) |

---

## Parallel Algorithms (C++17 STL)

| Topic | Free Link |
|-------|-----------|
| Execution policies overview | [cppreference: Execution policies](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t) |
| `std::execution::seq/par/par_unseq` | [cppreference: execution policy tags](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag) |
| Algorithms with execution policies | [cppreference: Algorithm library](https://en.cppreference.com/w/cpp/algorithm) |
| New C++17 algorithms (`reduce`, `transform_reduce`, etc.) | [cppreference: std::reduce](https://en.cppreference.com/w/cpp/algorithm/reduce) |

---

## C++20: The Concurrent Future

| Topic | Free Link |
|-------|-----------|
| Atomic smart pointers | [cppreference: atomic\<shared_ptr\>](https://en.cppreference.com/w/cpp/memory/shared_ptr/atomic2) |
| `std::latch` | [cppreference: latch](https://en.cppreference.com/w/cpp/thread/latch) |
| `std::barrier` | [cppreference: barrier](https://en.cppreference.com/w/cpp/thread/barrier) |
| `std::stop_token` / `std::jthread` | [cppreference: stop_token](https://en.cppreference.com/w/cpp/thread/stop_token) |
| Coroutines overview | [cppreference: Coroutines](https://en.cppreference.com/w/cpp/language/coroutines) |
| `co_await`, `co_yield`, `co_return` | [cppreference: Coroutines](https://en.cppreference.com/w/cpp/language/coroutines) |
| Semaphores (`std::counting_semaphore`) | [cppreference: counting_semaphore](https://en.cppreference.com/w/cpp/thread/counting_semaphore) |

---

## Time Library

| Topic | Free Link |
|-------|-----------|
| `std::chrono` overview | [cppreference: Date and time](https://en.cppreference.com/w/cpp/chrono) |
| `std::chrono::time_point` | [cppreference: time_point](https://en.cppreference.com/w/cpp/chrono/time_point) |
| `std::chrono::duration` | [cppreference: duration](https://en.cppreference.com/w/cpp/chrono/duration) |
| Clocks (`system_clock`, `steady_clock`) | [cppreference: system_clock](https://en.cppreference.com/w/cpp/chrono/system_clock) |
| `std::this_thread::sleep_for/until` | [cppreference: sleep_for](https://en.cppreference.com/w/cpp/thread/sleep_for) |

---

## Most Important Topics for Interviews

### Tier 1 — Must Know Cold
These appear in nearly every C++ systems/senior engineering interview.

1. **`std::thread`** — creation, join, detach, RAII wrapper patterns
   - [cppreference: std::thread](https://en.cppreference.com/w/cpp/thread/thread)
2. **`std::mutex` + `std::lock_guard` / `scoped_lock`** — basic mutual exclusion
   - [cppreference: mutex](https://en.cppreference.com/w/cpp/thread/mutex) | [lock_guard](https://en.cppreference.com/w/cpp/thread/lock_guard)
3. **Deadlock** — causes, `std::lock()`, lock ordering rules
   - [cppreference: std::lock](https://en.cppreference.com/w/cpp/thread/lock)
4. **`std::condition_variable`** — producer-consumer pattern, spurious wakeups, predicate form of `wait()`
   - [cppreference: condition_variable](https://en.cppreference.com/w/cpp/thread/condition_variable)
5. **`std::atomic` + memory_order** — `seq_cst` is default; understand `acquire/release`
   - [cppreference: atomic](https://en.cppreference.com/w/cpp/atomic/atomic) | [memory_order](https://en.cppreference.com/w/cpp/atomic/memory_order)
6. **Race conditions** — definition, detection, fixing
   - [cppreference: Memory model](https://en.cppreference.com/w/cpp/language/memory_model)
7. **`std::async` + `std::future`** — fire-and-forget, getting results/exceptions back
   - [cppreference: async](https://en.cppreference.com/w/cpp/thread/async) | [future](https://en.cppreference.com/w/cpp/thread/future)

### Tier 2 — Very Common
8. **`std::unique_lock` vs `std::lock_guard`** — know when to use each
   - [cppreference: unique_lock](https://en.cppreference.com/w/cpp/thread/unique_lock)
9. **`std::promise` / `std::packaged_task`** — passing exceptions across threads
   - [cppreference: promise](https://en.cppreference.com/w/cpp/thread/promise) | [packaged_task](https://en.cppreference.com/w/cpp/thread/packaged_task)
10. **Thread-safe singleton** — Meyers singleton (magic statics), `call_once`
    - [cppreference: call_once](https://en.cppreference.com/w/cpp/thread/call_once)
11. **`thread_local`** — per-thread storage
    - [cppreference: thread_local](https://en.cppreference.com/w/cpp/language/storage_duration)
12. **False sharing** — cache-line contention, padding fix
13. **RAII for threads** — lifetime management, avoiding detach pitfalls

### Tier 3 — Good to Know
14. **Parallel STL algorithms** (C++17) — `std::execution::par`
    - [cppreference: Execution policies](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)
15. **`std::shared_mutex` + `std::shared_lock`** — read-write locks
    - [cppreference: shared_mutex](https://en.cppreference.com/w/cpp/thread/shared_mutex)
16. **`std::latch` / `std::barrier`** (C++20)
    - [cppreference: latch](https://en.cppreference.com/w/cpp/thread/latch) | [barrier](https://en.cppreference.com/w/cpp/thread/barrier)
17. **Memory model internals** — happens-before, acquire/release, relaxed (needed for lock-free code)
    - [cppreference: memory_order](https://en.cppreference.com/w/cpp/atomic/memory_order)
18. **`std::jthread`** (C++20) — cooperative cancellation with `std::stop_token`
    - [cppreference: jthread](https://en.cppreference.com/w/cpp/thread/jthread)

### Tier 4 — Advanced / Specialist
19. **Lock-free data structures** using `std::atomic`
20. **Coroutines** (C++20) — concept level sufficient for most interviews
    - [cppreference: Coroutines](https://en.cppreference.com/w/cpp/language/coroutines)
21. **Transactional memory** — mostly theoretical for interviews
22. **CppMem tool** — useful for understanding memory model, rarely asked directly

---

## Quick Study Priority Order

```
Week 1: Tier 1 (threads, mutex, locks, condition_variable, atomic, race conditions, async/future)
Week 2: Tier 2 (unique_lock, promise/packaged_task, singleton patterns, thread_local, false sharing)
Week 3: Tier 3 (parallel STL, shared_mutex, C++20 latch/barrier, memory model deep dive)
Week 4: Tier 4 + Case studies + Practice problems
```

---

## Common Interview Patterns to Practice

- **Producer-Consumer** using `std::mutex` + `std::condition_variable`
- **Thread-safe queue** implementation
- **Thread pool** design
- **Singleton** with `std::call_once` or magic statics
- **Parallel accumulate** using `std::async` or `std::thread`
- **Read-write lock** using `std::shared_mutex`
- **Lock-free counter** using `std::atomic`
