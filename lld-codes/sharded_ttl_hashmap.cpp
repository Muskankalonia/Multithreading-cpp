#include <iostream>
#include <unordered_map>
#include <shared_mutex>
#include <optional>
#include <chrono>
#include <thread>
#include <atomic>
#include <array>
#include <vector>

using namespace std;
using namespace chrono;

// ─────────────────────────────────────────────
// Design choices:
//
// 1. LOCKED over lock-free: A lock-free hashmap requires handling
//    concurrent resize, which is extremely complex. Sharded locking
//    gives near lock-free performance and is production-standard
//    (Java ConcurrentHashMap, Go sync.Map use same idea).
//
// 2. SHARDING: Split map into N independent shards.
//    key → shard = hash(key) % N
//    N threads hitting different keys → N independent locks → zero contention
//
// 3. shared_mutex: Reads (get) take shared_lock  → multiple readers at once
//                  Writes (put/erase) take unique_lock → exclusive access
//
// 4. TTL: Each entry stores an expiry time_point.
//    Checked lazily on get(). Background thread cleans up periodically.
// ─────────────────────────────────────────────

template <typename K, typename V, size_t NUM_SHARDS = 16>
class ShardedTTLHashMap {
    using Clock = steady_clock;

    struct Entry {
        V value;
        Clock::time_point expiry;
    };

    struct Shard {
        unordered_map<K, Entry> map;
        mutable shared_mutex mtx;   // shared for reads, exclusive for writes
    };

    array<Shard, NUM_SHARDS> shards;

    atomic<bool> running{true};
    thread cleanup_thread;

    size_t shard_idx(const K& key) const {
        return hash<K>{}(key) % NUM_SHARDS;
    }

    bool expired(const Entry& e) const {
        return Clock::now() > e.expiry;
    }

    // Background thread: scans all shards and evicts expired keys
    void cleanup_loop(milliseconds interval) {
        while (running) {
            this_thread::sleep_for(interval);
            for (auto& shard : shards) {
                unique_lock lock(shard.mtx);
                for (auto it = shard.map.begin(); it != shard.map.end();) {
                    it = expired(it->second) ? shard.map.erase(it) : ++it; // need to be careful here 
                }
            }
        }
    }

public:
    explicit ShardedTTLHashMap(milliseconds cleanup_interval = milliseconds{1000})
        : cleanup_thread([this, cleanup_interval] { cleanup_loop(cleanup_interval); }) {}

    ~ShardedTTLHashMap() {
        running = false;
        cleanup_thread.join();
    }

    // Non-copyable (owns a thread)
    ShardedTTLHashMap(const ShardedTTLHashMap&) = delete;
    ShardedTTLHashMap& operator=(const ShardedTTLHashMap&) = delete;

    void put(const K& key, const V& val, milliseconds ttl) {
        auto& shard = shards[shard_idx(key)];
        unique_lock lock(shard.mtx);
        shard.map[key] = {val, Clock::now() + ttl};
    }

    optional<V> get(const K& key) {
        auto& shard = shards[shard_idx(key)];
        shared_lock lock(shard.mtx);   // ← multiple threads can read same shard simultaneously
        auto it = shard.map.find(key);
        if (it == shard.map.end() || expired(it->second)) return nullopt;
        return it->second.value;
    }

    void erase(const K& key) {
        auto& shard = shards[shard_idx(key)];
        unique_lock lock(shard.mtx);
        shard.map.erase(key);
    }
};

// ─── Demo ───────────────────────────────────────
int main() {
    // Cleanup scans every 300ms
    ShardedTTLHashMap<string, int> cache(milliseconds{300});

    // 1. Basic TTL demo
    cache.put("short", 1, milliseconds{200});   // expires in 200ms
    cache.put("long",  2, milliseconds{2000});  // expires in 2s

    cout << "[before expiry]\n";
    cout << "short = " << cache.get("short").value_or(-1) << "\n";  // 1
    cout << "long  = " << cache.get("long").value_or(-1)  << "\n";  // 2

    this_thread::sleep_for(milliseconds{400});

    cout << "\n[after 400ms]\n";
    cout << "short = " << cache.get("short").value_or(-1) << "\n";  // -1 (expired)
    cout << "long  = " << cache.get("long").value_or(-1)  << "\n";  // 2

    // 2. Concurrent access demo — readers + writers
    vector<thread> threads;

    for (int i = 0; i < 4; i++) {  // writers
        threads.emplace_back([&cache, i]() {
            cache.put("k" + to_string(i), i * 10, milliseconds{1000});
        });
    }

    for (int i = 0; i < 4; i++) {  // readers
        threads.emplace_back([&cache, i]() {
            auto v = cache.get("k" + to_string(i));
            string msg = "k" + to_string(i) + " = " + (v ? to_string(*v) : "miss") + "\n";
            cout << msg;
        });
    }

    for (auto& t : threads) t.join();
}


/*
followups :
1. How to convert this into CAS
 for individual bucket operations I could replace the linked list with a CAS-based structure using compare_exchange_weak. The insert becomes a CAS loop where I read the current head, set new_node->next to it, then CAS the head to new_node. If the CAS fails another thread raced me — I retry. The main problems this introduces are the ABA problem, which I'd solve with tagged pointers packing a version counter into the pointer, and memory reclamation — I can't delete a removed node immediately because another thread might be reading it. I'd need epoch-based reclamation or hazard pointers for that. For this use case the sharded mutex version gives me sufficient parallelism with much simpler correctness guarantees — I'd only go lock-free if profiling showed the mutex itself was the bottleneck.
*/