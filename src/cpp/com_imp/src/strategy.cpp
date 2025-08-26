#include "strategy.h"

Strategy::Strategy(const string &dataset_dir, const string &dataset_name, bool directed, const vector<double> &gap,
                   const vector<unode_int> &seeds_b, unode_int sample_num, int thread_num,
                   bool is_load_graph) {

    _gap = gap;
    _seeds_b = seeds_b;
    _thread_num = thread_num;
    _sample_num = sample_num;
    _graph_ptr = make_shared<Graph>(dataset_dir, dataset_name, directed, is_load_graph);
}

double Strategy::evaluate(const vector<unode_int> &seeds_a) {

    Sampler sampler(_gap, _seeds_b, _thread_num);

    // Evaluating the expected and real spread on the seeds
    double expected = sampler.sample(_graph_ptr, seeds_a, _sample_num);
    return expected;
}