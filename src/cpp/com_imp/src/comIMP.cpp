#include <iostream>

#include "timer.h"
#include "strategy.h"

using namespace std;

string base_dir = "/data/IMP/";
map<string, pair<string, string>> dataset_path = {
        {"Facebook",    {base_dir + "Facebook/",    "Facebook_WC"}},
        {"Flixster",    {base_dir + "Flixster/",    "Flixster_WC"}},
        {"Wiki",        {base_dir + "Wiki/",        "Wiki_WC"}},
        {"Epinions",    {base_dir + "Epinions/",    "Epinions_WC"}},
        {"LiveJournal", {base_dir + "LiveJournal/", "LiveJournal_WC"}},
        {"Twitter",     {base_dir + "Twitter/",     "Twitter_WC"}},
};

map<string, bool> dataset_directed = {
        {"Facebook",    false},
        {"Flixster",    false},
        {"Wiki",        true},
        {"Epinions",    true},
        {"LiveJournal", true},
        {"Twitter",     true},
};

vector<unode_int> load_seedSet(const string &seedSetFile) {
    vector<unode_int> seeds;
    ifstream fin(seedSetFile.c_str());
    ASSERT(fin.is_open());
    unode_int s;
    while (fin >> s) {
        seeds.push_back(s);
    }
    return seeds;
}

void test_compatible() {
//    string dataset_dir = "/home/01-Optimization-Problem-Lib/data/IMP/";
//    string dataset_name = "testcase1";
//    bool directed = true;
    string dataset = "Wiki";
    string dataset_dir = dataset_path[dataset].first;
    string dataset_name = dataset_path[dataset].second;
    cout << dataset_dir << dataset_name << endl;
    string seeds_a_path = dataset_dir + "seed/" + dataset + "_WC_50_seed";
    string seeds_b_path = dataset_dir + "seed/" + dataset + "_WC_100_seed";
    cout << seeds_a_path << endl;
    cout << seeds_b_path << endl;
    bool directed = dataset_directed[dataset];
    unode_int sample_num = 10000;
    int thread_num = 16;
//    vector<double> GAP{0.1, 0.2, 0.3, 0.4};
//    vector<unode_int> bSeeds{112586, 321, 353, 68214, 167};
//    vector<unode_int> aSeeds{966, 12168, 1652, 5355, 7216};
    vector<double> gap{0.5, 0.7, 0.5, 0.7};
//    vector<unode_int> bSeeds{2};
//    vector<unode_int> aSeeds{1};
    vector<unode_int> seeds_b = load_seedSet(seeds_b_path);
    INFO(seeds_b);
    vector<unode_int> seeds_a = load_seedSet(seeds_a_path);
    INFO(seeds_a);
    bool is_load_graph = true;

    Strategy strategy(dataset_dir, dataset_name, directed, gap, seeds_b, sample_num, thread_num, is_load_graph);
//    strategy.serializeGraph();
//    unode_int com_n = strategy.getVerNum();
//
//    vector<int> is_seed_com(com_n, 0);
//    for (unode_int s: seeds_a) {
//        is_seed_com[s] = 1;
//    }

    auto *t = new Timer();
    double comExp = strategy.evaluate(seeds_a);
    delete t;
    Timer::show();
    cout << comExp << endl;
}

int main() {
    test_compatible();
    return 0;
}