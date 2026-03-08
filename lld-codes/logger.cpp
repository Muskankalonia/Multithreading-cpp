/*
Implement log levels : Info, Error, Warn
Log framework should have searchByLevel, searchByKeyword, searchByService, searchByTimestamp
We should be able to add a log to the source
*/

#include <bits/stdc++.h>
using namespace std;

enum class level {
    Info,
    Error,
    Warn
};

string to_string(level l) {
    switch(l) {
        case level::Error: return "Error";
        case level::Info:  return "Info";
        case level::Warn:  return "Warn";
        default:           return "Unknown";
    }
}

struct logs {
    level lvl;
    string msg;
    int time_stamp;
    string service;

    logs(level lvl, string msg, int time_stamp, string service) : lvl(lvl), msg(msg), time_stamp(time_stamp), service(service) {}

    void printLog() {
        string final_log = to_string(time_stamp) + "|" + to_string(lvl) + "|" + service + "|" + msg + "\n";
        cout << final_log;
    }
};

class logger {
    shared_mutex mtx;
    vector<unique_ptr<logs>> logSrc;
    
    unordered_map<level, vector<logs*>> lvlSearch;
    unordered_map<string, vector<logs*>> svcSearch;
    unordered_map<string, vector<logs*>> keywordSearch;

    atomic<int> time_stamp{0};

    public : 
        static logger* instance() {
            static logger l;
            return &l;
        }

        void addLog(level lvl, string msg, string service) {
            {
                unique_lock lock(mtx);
                logSrc.emplace_back(make_unique<logs>(lvl, msg, time_stamp.load(), service));

                logs *l = logSrc.back().get();
                lvlSearch[lvl].push_back(l);
                svcSearch[service].push_back(l);
                
                stringstream ss(msg);
                string token = "";

                while(getline(ss, token, ' ')) {
                    if(token != "") {
                        keywordSearch[token].push_back(l);
                    }
                }
            }
            
            time_stamp++;

        }

        void printLog(vector<logs*> &logs_vec) {

            for(auto i : logs_vec) {
                i->printLog();
            }

        }

        void searchByLvl(level lvl) {
            vector<logs*> l;
            {
                shared_lock lock(mtx);
                l = lvlSearch[lvl];
            }

            printLog(l);

        }

        void searchByKeyword(string keyword) {
            vector<logs*> l;
            {
                shared_lock lock(mtx);
                l = keywordSearch[keyword];
            }

            printLog(l);

        }

        void searchByService(string service) {
            vector<logs*> l;
            {
                shared_lock lock(mtx);
                l = svcSearch[service];
            }

            printLog(l);

        }

        void searchByTimeStamp(int start, int end) {
            vector<logs*> l;

            shared_lock lock(mtx);
            auto i = lower_bound(logSrc.begin(), logSrc.end(), start, [](const unique_ptr<logs> &a, int val) {
                return a->time_stamp < val;
            });

            auto j = lower_bound(logSrc.begin(), logSrc.end(), end, [](const unique_ptr<logs> &a, int val) {
                return a->time_stamp < val;
            });

            if(i != logSrc.end() && j != logSrc.end()) {
                int index_i = i - logSrc.begin();
                int index_j = j - logSrc.begin();

                while(index_i <= index_j) {
                    l.push_back(logSrc[index_i].get());
                    index_i++;
                }

                printLog(l);
            }
        }
};

int main() {

    logger *l = logger::instance();

    l->addLog(level::Error, "This is an error message", "Auth");
    l->addLog(level::Info, "This is an info msg", "Fetch User");

    l->searchByKeyword("msg");
}

