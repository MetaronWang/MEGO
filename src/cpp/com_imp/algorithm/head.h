#ifndef HEAD_H
#define HEAD_H

#include <random>
#include <iostream>
#include <sstream>
#include <fstream>
#include <deque>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <climits>

using namespace std;

#define MAX_R INT_MAX

typedef unsigned int uint;
typedef unsigned char uint8;
typedef long long int64;
typedef unsigned long long uint64;
typedef pair<int, int> ipair;
typedef pair<double, double> dpair;
#define MP make_pair

typedef char int8;
typedef unsigned char uint8;
typedef long long int64;
typedef unsigned long long uint64;

#define SIZE(t) (int)(t.size())
#define ALL(t) (t).begin(), (t).end()
#define FOR(i, n) for(int (i)=0; (i)<((int)(n)); (i)++)
#ifdef WIN32
#define FORE(i, x) for (typeid((x).begin()) i = (x).begin(); (i) != (x).end(); (i)++)
#else
#define FORE(i, x) for (__typeof((x).begin()) i = (x).begin(); (i) != (x).end(); (i)++)
#endif

static inline string &ltrim(string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace))));
    return s;
}

static inline string &rtrim(string &s) {
    s.erase(find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(), s.end());
    return s;
}

static inline string &trim(string &s) { return ltrim(rtrim(s)); }

string __n_variable(string t, int n);

#define __expand_nv(x) __n_variable(t, x)<< t##x << " "

template<class T0>
void ___debug(string t, deque<T0> t0, ostream &os) {
    os << __n_variable(t, 0);
    FOR(i, SIZE(t0))os << t0[i] << " ";
}

template<class T0>
void ___debug(string t, set<T0> t0, ostream &os) {
    os << __n_variable(t, 0);
    FORE(i, t0)os << *i << " ";
}

template<class T0>
void ___debug(string t, vector<T0> t0, ostream &os) {
    os << __n_variable(t, 0);
    FOR(i, SIZE(t0))os << t0[i] << " ";
}

template<class T0, class T1>
void ___debug(string t, vector<pair<T0, T1> > t0, ostream &os) {
    os << __n_variable(t, 0);
    FOR(i, SIZE(t0))os << t0[i].F << "," << t0[i].S << " ";
}

template<class T0>
void ___debug(string t, T0 t0, ostream &os) { os << __expand_nv(0); }

template<class T0, class T1>
void ___debug(string t, T0 t0, T1 t1, ostream &os) { os << __expand_nv(0) << __expand_nv(1); }

template<class T0, class T1, class T2>
void ___debug(string t, T0 t0, T1 t1, T2 t2, ostream &os) { os << __expand_nv(0) << __expand_nv(1) << __expand_nv(2); }

template<class T0, class T1, class T2, class T3>
void ___debug(string t, T0 t0, T1 t1, T2 t2, T3 t3, ostream &os) {
    os << __expand_nv(0) << __expand_nv(1) << __expand_nv(2) << __expand_nv(3);
}

template<class T0, class T1, class T2, class T3, class T4>
void ___debug(string t, T0 t0, T1 t1, T2 t2, T3 t3, T4 t4, ostream &os) {
    os << __expand_nv(0) << __expand_nv(1) << __expand_nv(2) << __expand_nv(3) << __expand_nv(4);
}


//#define DO_ONCE


#define RUN_TIME(...) { int64 t=rdtsc();  __VA_ARGS__; t=rdtsc()-t; cout<<  #__VA_ARGS__ << " : " << t/TIMES_PER_SEC <<"s"<<endl;  }

#ifdef NDEBUG
#define TRACE(...) ;
#define IF_TRACE(args) ;
#define TRACE_LINE(...) ;
#define TRACE_SKIP(a, ...) ;
#define TRACE_LINE_SKIP(a, ...) ;
#define TRACE_LINE_END(...) ;
#define TRACE_LOG(...) ;
#else
#define TRACE(...) {{ ___debug(#__VA_ARGS__,  __VA_ARGS__, cerr); cerr<<endl;} }
#define IF_TRACE(args) args
#define TRACE_LINE(...) { ___debug( #__VA_ARGS__,  __VA_ARGS__,cerr); cerr<<"                    \033[100D";  }
#define TRACE_SKIP(a, ...) { static int c=-1; c++; if(c%a==0)TRACE( __VA_ARGS__); }
#define TRACE_LINE_SKIP(a, ...) { static int c=-1; c++; if(c%a==0) TRACE_LINE(__VA_ARGS__);  }
#define TRACE_LINE_END(...) {cerr<<endl; }
#define TRACE_LOG(...) { __HEAD_H__LOG.close(); ofstream cerr("log.txt", ofstream::out|ofstream::app); ___debug( #__VA_ARGS__,  __VA_ARGS__, cerr); cerr<<endl;  }
#endif


#ifdef NDEBUG
#define ASSERTT(v, ...) ;
#define ASSERT(v ) ;
#define INFO(...) ;
#else
#define ASSERT(v) {if (!(v)) {cerr<<"ASSERT FAIL @ "<<__FILE__<<":"<<__LINE__<<endl; exit(1);}}
#define INFO(...) {do { ___debug( #__VA_ARGS__,  __VA_ARGS__,cout); cout<<endl;  } while(0);}
#define ASSERTT(v, ...) {if (!(v)) {cerr<<"ASSERT FAIL @ "<<__FILE__<<":"<<__LINE__<<endl; INFO(__VA_ARGS__); exit(1);}}
#endif

void ExitMessage(const string &msg);

#endif //HEAD_H
