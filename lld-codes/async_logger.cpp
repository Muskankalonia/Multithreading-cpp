#include <bits/stdc++.h>
using namespace std;

enum class LogLevel {
    DEBUG,
    INFO,
    ERROR
};

string lvlToString (LogLevel lvl) {
    switch (lvl) {
        case LogLevel::DEBUG : 
            return "DEBUG";
        case LogLevel::INFO : 
            return "INFO";
        case LogLevel::ERROR : 
            return "ERROR";
        
        default :
            return "UNKNOWN";
    }
}

class LogEntry {
    string msg;
    LogLevel lvl;
    chrono::system_clock::time_point timestamp;
    string thread_id;

    public : 
        LogEntry() {}
        LogEntry(string msg, LogLevel lvl) : msg(msg), lvl(lvl), timestamp(chrono::system_clock::now()) {
            ostringstream oss;
            oss << this_thread::get_id();
            thread_id = oss.str();
        }

        string formatter() {
            ostringstream oss;
            auto time_t_now = chrono::system_clock::to_time_t(timestamp);
            oss << thread_id << " | " << put_time(localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << " | " << lvlToString(lvl) << " | " << msg << "\n";
            return oss.str();
        }
};

template <typename T> 
class BoundedBlockedQueue {

    private :
        queue<T> q;
        condition_variable not_empty;
        condition_variable not_full;
        mutex mtx;
        size_t max_size;
        bool shutdown_flag;

    public : 

        explicit BoundedBlockedQueue(size_t max_size) : max_size(max_size), shutdown_flag(false) {}

        void push(T item) {

            unique_lock<mutex> lock(mtx);
            not_full.wait(lock, [this]() {
                return q.size() < max_size || shutdown_flag;
            });

            if(shutdown_flag) return;

            q.push(move(item));
            not_empty.notify_one();

        }

        bool pop(T &item) {

            unique_lock<mutex> lock(mtx);
            not_empty.wait(lock, [this]() {
                return !q.empty() || shutdown_flag;
            });

            if(q.empty() && shutdown_flag) return false;

            item = move(q.front());
            q.pop();

            not_full.notify_one();
            return true;
        }

        void shutdown() {

            lock_guard<mutex> lock(mtx);
            shutdown_flag = true;

            not_full.notify_all();
            not_empty.notify_all();

        }

};

class LogSink {
    public : 
    virtual ~LogSink() = default;
    virtual void write(string &formatted_str) = 0;
    virtual void flush() = 0;

};

class FileSink : public LogSink {

    private : 
        ofstream ofs;

    public : 
        FileSink(string file_path) : ofs(file_path, ios::app) {
            if (!ofs.is_open()) {
                throw std::runtime_error("Cannot open log file: " + file_path);
            }
        }

        void write(string &formatted_str) override {

            ofs << formatted_str;

        }

        void flush() override {

            ofs.flush();

        }

};

class AsyncLogger {

    private : 
        unique_ptr<BoundedBlockedQueue<LogEntry>> q;
        vector<shared_ptr<LogSink>> sinks;
        size_t max_size;
        atomic <bool> running{false};
        thread background_thread;
        atomic <LogLevel> min_level;

        AsyncLogger() : max_size(128), min_level(LogLevel::DEBUG) {}

        void backgroundLoop() {

            LogEntry log;

            while(q->pop(log)) {

                string formatted_str = log.formatter();

                for(auto &i : sinks) {
                    i->write(formatted_str);
                }

            }

        }

        ~AsyncLogger() {
            stop();
        }

    public : 
        
        static AsyncLogger& getInstance() {

            static AsyncLogger logger;
            return logger;

        }

        void start() {

            if(running.exchange(true)) return;

            q = make_unique<BoundedBlockedQueue<LogEntry>> (max_size);

            background_thread = thread([this]() {
                backgroundLoop();
            });

        }   

        void stop() {

            if(!running.exchange(false)) return;

            q->shutdown();

            if(background_thread.joinable()) {
                background_thread.join();
            }

            for(auto &i : sinks) {
                i->flush();
            }

        }

        void log(string msg, LogLevel lvl) {
            
            if(!running.load()) return;
            if(lvl < min_level.load(memory_order_relaxed)) return;
            
            LogEntry log_buf(msg, lvl);
            q->push(move(log_buf));

        }

        void setSize(size_t max_size) {
            max_size = max_size;
        }

        void setMinLevel(LogLevel minLvl) {
            min_level.store(minLvl, memory_order_relaxed);
        }

        void setSinks(shared_ptr<LogSink> sink) {
            sinks.push_back(move(sink));
        }

};

int main() {

    AsyncLogger& instance = AsyncLogger::getInstance();

    instance.setSinks(make_shared<FileSink>("app.log"));

    instance.start();

    auto worker = [&]() {
        for(int i = 0 ; i < 5; i++) {
            string s = "Hi " + to_string(i);
            instance.log(s, LogLevel::INFO);
        }
    };

    vector<thread> threads;

    for(int i = 0; i < 3; i++) {
        threads.emplace_back(worker);
    }

    for(auto &i : threads) {
        i.join();
    }

    instance.stop();


}