#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <thread>
#include <boost/lockfree/queue.hpp>
#include <mutex>
using namespace std;
namespace fs = filesystem;

// Global variables
int P, B = 10, W;
map<string, int> dict;
bool producers_done = false;
boost::lockfree::queue<char *> buffer(B); // Boost Lockfree queue
mutex dictMutex;

vector<string> readfileNames(const string &path)
{
    vector<string> X;
    ifstream inputFile(path);

    if (!inputFile.is_open())
    {
        cerr << "Failed to open the file: " << path << endl;
    }

    string line;
    while (getline(inputFile, line))
    {
        size_t endPos = line.find_last_not_of(" \t\r\n");
        if (endPos != string::npos)
        {
            line = line.substr(0, endPos + 1);
        }
        X.push_back(line);
    }
    inputFile.close();

    return X;
}

void produce(const string &path)
{
    ifstream inputFile(path);

    if (!inputFile.is_open())
    {
        cerr << "Failed to open the file: " << path << endl;
    }

    string line;
    while (getline(inputFile, line))
    {
        size_t endPos = line.find_last_not_of(" \t\r\n");
        if (endPos != string::npos)
        {
            line = line.substr(0, endPos + 1);
        }
        const char *line_cstr = line.c_str();
        char *line_copy = new char[strlen(line_cstr) + 1];
        strcpy(line_copy, line_cstr);
        buffer.push(line_copy);
    }
    inputFile.close();
}

string removePunctuations(const string &line)
{
    string res = "";
    for (auto ch : line)
    {
        if (!ispunct(ch))
            res += ch;
    }
    return res;
}

void consume()
{
    while (true)
    {
        string line;
        if (buffer.pop(line))
        {
            size_t endPos = line.find_last_not_of(" \t\r\n");
            if (endPos != string::npos)
            {
                line = line.substr(0, endPos + 1);
            }
            line = removePunctuations(line);
            istringstream iss(line);
            string word;
            while (iss >> word)
            {
                lock_guard<mutex> lock(dictMutex);
                dict[word]++;
            }
        }
        else if (producers_done)
        {
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    char *prod = argv[1], *cons = argv[2];
    P = atoi(prod), W = atoi(cons);
    string path = argv[3];

    vector<string> filePaths = readfileNames(path);

    vector<thread> producer_threads(P);
    vector<thread> consumer_threads(W);

    for (int i = 0; i < P; i++)
    {
        producer_threads[i] = thread(produce, filePaths[i]);
    }

    for (int i = 0; i < W; i++)
    {
        consumer_threads[i] = thread(consume);
    }

    for (int i = 0; i < P; ++i)
    {
        producer_threads[i].join();
    }

    producers_done = true; // Set the termination condition for consumers

    for (int i = 0; i < W; ++i)
    {
        consumer_threads[i].join();
    }

    for (const auto &it : dict)
    {
        cout << it.first << ' ' << it.second << endl;
    }

    return 0;
}
