#include "timer.h"


int Counter::cnt[1000] = {0};
vector<int64> Timer::timeUsed;
vector<string> Timer::timeUsedDesc;
map<string, vector<int64>> Timer::timeUsedMap;
int Timer::timer_id = -1;

map<string, timeval> __head_h_start;

string nowStr() {
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char str[100];
    sprintf(str, "%d_%d_%d_%d_%d_%d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
    return string(str);
}

void timer_init(string arg) {
    timeval ts;
    gettimeofday(&ts, NULL);
    __head_h_start[arg] = ts;
}

int64 timer_elapse(string arg) { // unit ms
    struct timeval now;
    gettimeofday(&now, NULL);
    int64 sec = now.tv_sec - __head_h_start[arg].tv_sec;
    int64 usec = now.tv_usec - __head_h_start[arg].tv_usec;
    return sec * 1000 + usec / 1000;
}

string currentTimestampStr() {
    time_t t = time(0);   // get time now
    struct tm *now = localtime(&t);
    char buf[1000];
    sprintf(buf, "%04d-%02d-%02d %02d:%02d:%02d", now->tm_year + 1900, now->tm_mon + 1, now->tm_mday, now->tm_hour,
            now->tm_min, now->tm_sec);
    return string(buf);
}

// return the output of the command by string
string exec(const char *cmd) {
    FILE *pipe = popen(cmd, "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL)
            result += buffer;
    }
    pclose(pipe);
    return result;
}

string getIpAddress() {
    string cmd = " ifconfig |grep 'inet addr' |grep -v 127.0.0.1 |awk '{print $2}' |python -c 'print raw_input().split(\":\")[-1]' ";
    return exec(cmd.c_str());
}