#ifndef STRATEGY_H
#define STRATEGY_H

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>


#include "head.h"
#include "graph.h"
#include "sampler.h"


class Strategy {
private:
    int _thread_num;
    shared_ptr<Graph> _graph_ptr;
    unode_int _sample_num;
    vector<unode_int> _seeds_b;
    vector<double> _gap;
public:
    Strategy(const string &dataset_dir, const string &dataset_name, bool directed, const vector<double> &gap,
             const vector<unode_int> &seeds_b, unode_int sample_num = 1000, int thread_num = 4,
             bool is_load_graph = false);

    vector<unode_int> perform(unsigned int k) {
        INFO("MonteStrategy no perform.")
        return {};
    }

    double evaluate(const vector<int> &is_seed);

    void serializeGraph() {
        if (_graph_ptr) {
            _graph_ptr->saveGraphStruct();
        }
    }

    unode_int getVerNum() const {
        return _graph_ptr->_verNum;
    }

    unode_int getEdgeNum() const {
        return _graph_ptr->_edgeNum;
    }

    unode_int getSampleNum() const {
        return _sample_num;
    }

    void setSampleNum(unode_int sample_num) {
        _sample_num = sample_num;
    }

    unode_int getThreadNum() const {
        return _thread_num;
    }

    void setThreadNum(unode_int thread_num) {
        _thread_num = thread_num;
    }
};


#endif //STRATEGY_H
