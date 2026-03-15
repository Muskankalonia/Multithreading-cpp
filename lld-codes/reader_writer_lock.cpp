#include <bits/stdc++.h>
using namespace std;

class ReadWriteLock {

    private : 
        mutex mtx;
        condition_variable can_read;
        condition_variable can_write;

        int waiting_writers = 0;
        int active_readers = 0;
        bool active_writer = false;

    public : 

        // Reader
        void read_lock() {

            unique_lock<mutex> lock(mtx);
            can_read.wait(lock, [this]() {
                return !active_writer && waiting_writers == 0;  // this is currently writer preference code if you want to make it reader preference then remove the waiting_reader == 0 condition
            });

            active_readers++;

        }

        void read_unlock() {
            
            unique_lock<mutex> lock(mtx);
            active_readers--;

            if(active_readers == 0 && waiting_writers > 0) {
                can_write.notify_one();
            }

        }

        // Writer 
        void write_lock() {

            unique_lock<mutex> lock(mtx);
            waiting_writers++;

            can_write.wait(lock, [this]() {
                return active_readers == 0 && !active_writer;
            });

            waiting_writers--;
            active_writer = true;
        }

        void write_unlock() {

            unique_lock<mutex> lock(mtx);
            active_writer = false;

            if(waiting_writers > 0) {
                can_write.notify_one();
            } else {
                can_read.notify_all();
            }

        }
};

class ReadGuard {
    ReadWriteLock &l;

    public : 
        explicit ReadGuard(ReadWriteLock &rwl) : l(rwl) {l.read_lock();}
        ~ReadGuard() {l.read_unlock();}

        ReadGuard(const ReadGuard&) = delete;
        ReadGuard& operator=(const ReadGuard&) = delete;
};

class WriteGuard {
    ReadWriteLock &l;

    public : 
        explicit WriteGuard(ReadWriteLock &rwl) : l(rwl) {l.write_lock();}
        ~WriteGuard() {l.write_unlock();}

        WriteGuard(const WriteGuard&) = delete;
        WriteGuard& operator=(const WriteGuard&) = delete;
};

int main() {

    int c = 5;
    ReadWriteLock rwl;

    thread read1([&c, &rwl]() {
        ReadGuard guard(rwl);
        string msg = "Read val of c " + to_string(c) + "\n";
        cout << msg;
    });

    thread write1([&c, &rwl]() {
        WriteGuard guard(rwl);
        c = 10;
        string msg = "Write val of c " + to_string(c) + "\n";
        cout << msg;
    });

    write1.join(); read1.join();

    sleep(5);
    cout << "Final c val " << c << endl;
}