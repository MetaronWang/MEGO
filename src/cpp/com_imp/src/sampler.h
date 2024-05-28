#ifndef SAMPLE_H
#define SAMPLE_H

#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <boost/bind/bind.hpp>
#include <boost/asio/post.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/asio/thread_pool.hpp>


#include "graph.h"

class Sampler {
    enum EdgeStatus {
        INLIVE,
        LIVE,
        BLOCKED
    };

    enum NodeStatus {
        INACTIVE,
        SUSPENDED,
        ADOPTED,
        REJECTED,
    };

    int _n_threads;
    boost::mutex _mt;
    boost::dynamic_bitset<> _scheduler;
    //    bitset<THREAD> scheduler_;

    // Random number generator pool, one for each thread, otherwise it will block thread.
    vector<boost::mt19937> _gen_pool;
    uniform_real_distribution<double> _urd;
    //    std::vector<std::mt19937> randn;

    double **_alpha_pool_A{};
    double **_alpha_pool_B{};
    unode_int **_visit_nodes_pool_A{};  // hold the nodes currently informed of A
    unode_int **_visit_nodes_pool_B{};  // hold the nodes currently informed of B
    NodeStatus **_status_nodes_pool_A{};  // 0: inactive, 1: informed, 2: suspended, 3: adopted (active)
    NodeStatus **_status_nodes_pool_B{};  // 0: inactive, 1: informed, 2: suspended, 3: adopted (active)
    EdgeStatus **_status_edges_pool{};

public:
    vector<unode_int> _aSeeds; // seed set to be found (for item A)
    vector<unode_int> _bSeeds; // input: the other company's seed set
//    vector<double> mg;    // marginal gain of each seed added (in greedy sequence)
    double _qao;
    double _qab;
    double _qbo;
    double _qba;

    explicit Sampler(const vector<double> &GAP, const vector<unode_int> &bSeeds, int n_threads = 1)
            : _bSeeds(bSeeds), _urd(0.0, 1.0) {
        _qao = GAP[0];
        _qab = GAP[1];
        _qbo = GAP[2];
        _qba = GAP[3];
        _gen_pool.emplace_back(seed_ns());

        if (n_threads > 1) {
            _n_threads = n_threads;
            _scheduler.resize(n_threads, false);
            for (int i = 1; i < n_threads; ++i) {
//                urd_pool.emplace_back(gen_);
                _gen_pool.emplace_back(seed_ns());
            }
        } else
            _n_threads = 1;
    };

    /**
      Samples `n_samples` from seeds.
    */
    double
    sample(const shared_ptr<Graph> &graph_ptr, const vector<unode_int> &seeds_a, unode_int n_samples) {
//        ASSERT(is_seed.size() == graph_ptr->_verNum)
//        vector<unode_int> seeds_a;
//        for (unode_int i = 0; i < graph_ptr->_verNum; ++i) {
//            if (is_seed[i] == 1)
//                seeds_a.push_back(i);
//        }
        return perform_sample(graph_ptr, seeds_a, n_samples);
    }

    /**
      Performs the test_static diffusion from selected seeds.
      Returns the set of activated users.
    */
    vector<unode_int> perform_diffusion(const shared_ptr<Graph> &graph_ptr, const vector<unode_int> &seeds);

private:

    void setBSeeds(const vector<unode_int> &bSeeds);

    void setGAP(vector<double> GAP);

    /**
      Performs `n_samples` samples starting from `seeds`.
    */
    double perform_sample(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &seeds_a,
                          unode_int n_samples);

    void diffuse_ComIC(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &seeds_a,
                       unode_int *cov_ptr, int tid = 0);

    void diffuse_ComIC_thread(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &aSeeds,
                              unode_int *cov_ptr);

    void diffuse_ComIC_parallel(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &seeds_a,
                                unode_int head, unode_int tail, unode_int *cov_ptr);

    /* if v becomes X-adopted (X = A or X = B), we examine v's out-neighbours */
    void examine_out_neighbors(const shared_ptr<Graph> &base_graph_ptr, unode_int v, unode_int *visit_nodes,
                               unode_int *visit_tail, const NodeStatus *node_status, EdgeStatus *status_edges,
                               int tid = 0);

//    double mineSeedsMC();
//
//    double mineSeedsMC_comp(double baseSpread);
//
//    double compute_coverage_comp(vector<int> set_B, int size_B);
};

#endif //SAMPLE_H
