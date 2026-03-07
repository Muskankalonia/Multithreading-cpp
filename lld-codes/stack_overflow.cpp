/*
Users can post questions, answer questions, and comment on questions and answers.
Users can vote on questions and answers.
Questions should have tags associated with them.
Users can search for questions based on keywords, tags, or user profiles.
The system should assign reputation score to users based on their activity and the quality of their contributions.
The system should handle concurrent access and ensure data consistency.

Design:
  User          - id, reputation
  Comment       - id, authorId, content
  Votable       - base: votes map, comments list, mutex
  Answer        - extends Votable
  Question      - extends Votable, has tags + answers
  StackOverflow - singleton, owns all data, per-entity mutexes for concurrency
*/  

#include <bits/stdc++.h>
using namespace std;

enum class voteType {up, down};

class user {
    public : 
        string id;
        atomic<int> reputation;

        user(string id) : id(id) {}

        void changeReputation(int delta) {
            reputation += delta;
        }
};

class comment {
    public : 
        string author_id;
        string comment;
};

class votable {
    public : 
        mutable shared_mutex mtx;
        unordered_map<string, voteType> vote;
        vector<comment> comments;

        int reqVoteDelta(string voter_id, voteType v) {
            
            int final_vote = v == voteType::up ? 1 : -1;

            unique_lock lock(mtx);
            if(vote.find(voter_id) != vote.end()) {
                final_vote += (-1 * (vote[voter_id] == voteType::up ? 1 : -1));
            }

            vote[voter_id] = v;
            return final_vote;
        }

        void addComment(string author_id, string content) {
            unique_lock lock(mtx);
            comment c;
            c.author_id = author_id;
            c.comment = content;
            comments.push_back(c);
        }
};

class answer : public votable {
    public : 
        string id;
        string content;
        string author;

        answer(string id, string content, string author) : id(id), content(content), author(author) {}
};

class question : public votable {
    public : 
        string id;
        string content;
        string author;
        vector<string> tags;
        vector<string> ans;

        question(string id, string content, string author, vector<string> tags) : id(id), content(content), author(author), tags(tags) {}

        void addAnswer(string answer_id) {
            ans.push_back(answer_id);
        }
};

class stackOverflow {

    mutable shared_mutex mtx;

    unordered_map<string, shared_ptr<user>> userMap;
    unordered_map<string, shared_ptr<answer>> answerMap;
    unordered_map<string, shared_ptr<question>> questionMap;

    unordered_map<string, vector<shared_ptr<question>>> keywordSearch;
    unordered_map<string, vector<shared_ptr<question>>> tagSearch;

    public : 
        static stackOverflow* instance() {
            static stackOverflow s;
            return &s;
        }

        void addUser(string user_id) {
            unique_lock lock(mtx);
            userMap[user_id] = make_shared<user>(user_id);
        }

        void addQuestion(string ques_id, string content, string user_id, vector<string> tags) {
            shared_ptr<question> ques = make_shared<question>(ques_id, content, user_id, tags);
            unique_lock lock(mtx);
            questionMap[ques_id] = ques;
            
            stringstream ss(content);
            string token = "";
            while(getline(ss, token, ' ')) {
                if(token != "") {
                    keywordSearch[token].emplace_back(ques);
                }
            }

            for(auto i : tags) {
                tagSearch[i].emplace_back(ques);
            }
        }

        void addAnswer(string id, string content, string author, string ques_id) {
            shared_ptr<answer> ans = make_shared<answer>(id, content, author);
            unique_lock lock(mtx);
            questionMap[ques_id]->addAnswer(id);
            answerMap[id] = ans;
        }

        void addVote(string voter_id, string answer_id, string question_id, voteType v) {
            int delta;
            string user_id;
            {
                shared_lock lock(mtx);
                delta = answer_id == "" ? questionMap[question_id]->reqVoteDelta(voter_id, v) : answerMap[answer_id]->reqVoteDelta(voter_id, v);
                user_id = answer_id == "" ? questionMap[question_id]->author : answerMap[answer_id]->author;
            }

            userMap[user_id]->changeReputation(delta);
        }

        void addComment(string content, string comment_author, string answer_id, string question_id) {
            shared_lock lock(mtx);
            answer_id == "" ? questionMap[question_id]->addComment(comment_author, content) : answerMap[answer_id]->addComment(comment_author, content);

        }

};