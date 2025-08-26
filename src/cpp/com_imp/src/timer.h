#ifndef TIMER_H
#define TIMER_H

#if defined(WIN32)
#elif defined(__CYGWIN__) // cygwin

#include <sys/time.h>

#else //linux

#include <sys/time.h>
#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#endif //omp win32

#include <iostream>
#include <set>
#include <list>

#include <cmath>
#include <queue>

#include <string>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <climits>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <map>
#include <deque>

#include "head.h"

using namespace std;

#define MP make_pair
#define F first
#define S second

#ifndef TIMES_PER_SEC
#define TIMES_PER_SEC (2393.910e6)
#endif


#define F first
#define S second

#ifdef WIN32
#include < time.h >
#include <windows.h> //I've ommited this line.
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};



int gettimeofday(struct timeval *tv, struct timezone *tz)
{
    FILETIME ft;
    unsigned __int64 tmpres = 0;
    static int tzflag;

    if (NULL != tv)
    {
        GetSystemTimeAsFileTime(&ft);

        tmpres |= ft.dwHighDateTime;
        tmpres <<= 32;
        tmpres |= ft.dwLowDateTime;

        /*converting file time to unix epoch*/
        tmpres -= DELTA_EPOCH_IN_MICROSECS;
        tmpres /= 10;  /*convert into microseconds*/
        tv->tv_sec = (long)(tmpres / 1000000UL);
        tv->tv_usec = (long)(tmpres % 1000000UL);
    }

    if (NULL != tz)
    {
        if (!tzflag)
        {
            _tzset();
            tzflag++;
        }
        tz->tz_minuteswest = _timezone / 60;
        tz->tz_dsttime = _daylight;
    }

    return 0;
}
#endif


#ifndef WIN32
#ifdef __CYGWIN__

//CYGWIN
__inline__ uint64 rdtsc() {
    uint64 t0;
    asm volatile("rdtsc" : "=A"(t0));
    return t0;

}

#else

//LINUX
__inline__ uint64 rdtsc(void) {
    unsigned a, d;
    //asm("cpuid");
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    return (((uint64) a) | (((uint64) d) << 32));
}

#endif
#endif

string nowStr();


void timer_init(string arg = "default");

int64 timer_elapse(string arg = "default");

string currentTimestampStr();

class Timer {
public:
    static vector<int64> timeUsed;
    static vector<string> timeUsedDesc;
    static map<string, vector<int64>> timeUsedMap;
    static int timer_id;
    int id;
    uint64 startTime;
    bool showOnDestroy = false;

    Timer(int id = -1, string desc = "", bool showOnDestroy = false) {
        if (id < 0) {
            id = ++timer_id;
        } else {
            timer_id = max(timer_id + 1, id);
        }
        this->id = id;
        while ((int) timeUsed.size() <= id) {
            timeUsed.push_back(0);
            timeUsedDesc.push_back("");
        }
        if (timeUsedDesc[id] != desc) {
            timeUsedDesc[id] = desc;
            if (timeUsedMap.find(desc) == timeUsedMap.end()) {
                timeUsedMap[desc] = vector<int64>();
            }
            timeUsedMap[desc].emplace_back(id);
        }
        startTime = rdtsc();
        this->showOnDestroy = showOnDestroy;
    }

    ~Timer() {
        if (showOnDestroy) {
            cout << "time spend on " << timeUsedDesc[id] << ":" << (rdtsc() - startTime) / TIMES_PER_SEC << "s" << endl;
        }
        timeUsed[id] += (rdtsc() - startTime);
    }

    static double getTime(int id = -1) {
        if (id < 0) {
            id = timer_id;
        }

        if (timeUsed[id] > 0 && id <= timer_id) {
            return (timeUsed[id] / TIMES_PER_SEC);
        }
        return -1;
    }

    static void resetTime(int id = -1) {
        if (id < 0) {
            id = timer_id;
        }

        if (id <= timer_id) {
            timeUsed[id] = 0;
        }
    }

    static void show(bool debug = false, bool avg = false) {
        int64 total = 0;
        int64 cnt = 0;
        for (int i = 0; i < (int) timeUsed.size(); i++) {
            if (timeUsed[i] > 0) {
                char str[100];
                sprintf(str, "%.6lf", timeUsed[i] / TIMES_PER_SEC);
                string s = str;
                if (avg) {
                    cnt++;
                    total += timeUsed[i];
                }
                if ((int) s.size() < 15) s = " " + s;
                char t[100];
                memset(t, 0, sizeof t);
                sprintf(t, "%4d %s %s", i, s.c_str(), timeUsedDesc[i].c_str());
                if (debug) {
                    TRACE(t);
                } else {
                    INFO(t);
                }
            }
        }
        if (avg) {
            char str[100];
            sprintf(str, "Average: %.6lf", total / TIMES_PER_SEC / cnt);
            if (debug) {
                TRACE(str);
            } else {
                INFO(str);
            }
        }
    }

    static void showByDesc(string desc, bool avg = false) {
        if (timeUsedMap.find(desc) == timeUsedMap.end()) {
            return;
        }
        int64 total = 0;
        int64 cnt = 0;
        for (int64 id: timeUsedMap[desc]) {
            if (timeUsed[id] > 0) {
                char str[100];
                sprintf(str, "%.6lf", timeUsed[id] / TIMES_PER_SEC);
                string s = str;
                if ((int) s.size() < 15) s = " " + s;
                char t[100];
                memset(t, 0, sizeof t);
                sprintf(t, "%4lld %s %s", id, s.c_str(), timeUsedDesc[id].c_str());
                INFO(t);
                if (avg) {
                    cnt++;
                    total += timeUsed[id];
                }
            }
        }
        if (avg) {
            char str[100];
            sprintf(str, "Average: %.6lf", total / TIMES_PER_SEC / cnt);
            INFO(str);
        }
    }

    static void clearAll() {
        timer_id = -1;
        timeUsed.clear();
        timeUsedDesc.clear();
    }
};


class Counter {
public:
    static int cnt[1000];

    Counter(int id = 0) {
        cnt[id]++;
    }

    ~Counter() {
    }

    static void show() {
        for (int i = 0; i < 1000; i++)
            if (cnt[i] > 0)
                continue;
//                TRACE("Counter %d %d\n",i,cnt[i]);
    }
};


// return the output of the command by string
string exec(const char *cmd);

string getIpAddress();

class OutputInfo {
public:
    OutputInfo(int argn, char **argv) {
        cout << "Program Start at: " << currentTimestampStr() << endl;
        cout << "Arguments: ";
        for (int i = 0; i < argn; i++) {
            cout << argv[i] << " ";
        }
        cout << endl;
        cout << "ip address: " << getIpAddress();
        cout << "--------------------------------------------------------------------------------" << endl;
    }

    ~OutputInfo() {
        cout << "--------------------------------------------------------------------------------" << endl;
        cout << "Program Ends Successfully at: " << currentTimestampStr() << endl;
    }
};

#endif //TIMER_H
