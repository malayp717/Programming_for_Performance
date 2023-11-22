#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <map>
#include <vector>
#include <pthread.h>
using namespace std;
namespace fs = filesystem;

// Global variables
int P, B = 10, W, cnt = 0;
map<string, int> dict;
vector<string> buffer;

// Locks
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t counter_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t buffer_not_full = PTHREAD_COND_INITIALIZER;
pthread_cond_t buffer_not_empty = PTHREAD_COND_INITIALIZER;
bool producers_done = false;

vector<string> readfileNames(string path)
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

void *produce(void *arg)
{

    string path = *((string *)arg);

    ifstream inputFile(path);

    if (!inputFile.is_open())
    {
        cerr << "Failed to open the file: " << path << endl;
    }

    if (inputFile.is_open())
    {
        string line;
        while (getline(inputFile, line))
        {
            pthread_mutex_lock(&mutex);

            while (buffer.size() == B)
            {
                pthread_cond_wait(&buffer_not_full, &mutex);
            }
            buffer.push_back(line);
            pthread_cond_signal(&buffer_not_empty);
            pthread_mutex_unlock(&mutex);
            // inputStream.push_back(line);
        }
        inputFile.close();
    }
    else
        cout << "Not able to read: " << path << endl;

    pthread_mutex_lock(&counter_mutex);
    cnt++;
    if (cnt == P)
        producers_done = true; // termination condition
    pthread_mutex_unlock(&counter_mutex);
    pthread_cond_broadcast(&buffer_not_empty); // Signal all consumers

    return nullptr;
}

string removePunctuations(string line)
{
    string res = "";
    for (auto ch : line)
    {
        if (!ispunct(ch))
            res += ch;
    }
    return res;
}

void *consume(void *)
{

    while (true)
    {
        pthread_mutex_lock(&mutex);
        while (buffer.empty() && !producers_done)
            pthread_cond_wait(&buffer_not_empty, &mutex);

        if (buffer.empty() && producers_done)
        {
            pthread_mutex_unlock(&mutex);
            break;
        }

        string line = buffer.back();
        buffer.pop_back();
        pthread_cond_signal(&buffer_not_full);
        pthread_mutex_unlock(&mutex);

        line = removePunctuations(line);
        istringstream iss(line);
        string word;
        while (iss >> word)
        {
            pthread_mutex_lock(&mutex);
            dict[word]++;
            pthread_mutex_unlock(&mutex);
        }
    }

    return nullptr;
}

int main(int argc, char *argv[])
{

    char *prod = argv[1], *cons = argv[2];
    P = atoi(prod), W = atoi(cons);
    string path = argv[3];

    vector<string> filePaths = readfileNames(path);

    if (filePaths.size() != P)
    {
        cerr << "Number of files are not equal to the number of producer threads given" << endl;
        return 1;
    }

    pthread_t producer_threads[P], consumer_threads[W];

    for (int i = 0; i < P; i++)
    {
        int res = pthread_create(&producer_threads[i], nullptr, produce, &filePaths[i]);
        if (res != 0)
        {
            cerr << "Producer thread creation failed" << endl;
            return 1;
        }
    }

    for (int i = 0; i < W; i++)
    {
        int res = pthread_create(&consumer_threads[i], nullptr, consume, nullptr);
        if (res != 0)
        {
            cerr << "Consumer thread creation failed" << endl;
            return 1;
        }
    }

    for (int i = 0; i < P; ++i)
    {
        int res = pthread_join(producer_threads[i], nullptr);
        if (res != 0)
        {
            cerr << "Producer thread join failed" << endl;
            return 1;
        }
    }

    for (int i = 0; i < W; ++i)
    {
        int res = pthread_join(consumer_threads[i], nullptr);
        if (res != 0)
        {
            cerr << "Consumer thread join failed" << endl;
            return 1;
        }
    }

    for (auto it : dict)
        cout << it.first << ' ' << it.second << endl;

    return 0;
}
