#include "head.h"

string __n_variable(string t, int n) {
    t = t + ',';
    int i = 0;
    if (n) for (; i < SIZE(t) && n; i++) if (t[i] == ',') n--;
    n = i;
    for (; t[i] != ','; i++);
    t = t.substr(n, i - n);
    trim(t);
    if (t[0] == '"') return "";
    return t + "=";
}


void ExitMessage(const string& msg) {
    cout << msg << endl;
    exit(1);
}
