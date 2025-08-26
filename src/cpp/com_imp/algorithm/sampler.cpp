#include "sampler.h"


vector<unode_int> Sampler::perform_diffusion(const shared_ptr<Graph> &graph_ptr,
                                             const vector<unode_int> &seeds) {
    return {};
}

void Sampler::setBSeeds(const vector<unode_int> &bSeeds) {
    _bSeeds = bSeeds;

    //ASSERT((int)aSeeds.size() == 0);
}

void Sampler::setGAP(vector<double> GAP) {
    ASSERT((int) GAP.size() == 4)
    _qao = GAP[0];
    _qab = GAP[1];
    _qbo = GAP[2];
    _qba = GAP[3];
}

double
Sampler::perform_sample(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &seeds_a,
                        unode_int n_samples) {

    unode_int n = base_graph_ptr->_verNum;
    unode_int m = base_graph_ptr->_edgeNum;

    auto *cov_result = new unode_int[n_samples]();
    _alpha_pool_A = new double *[_n_threads];
    _alpha_pool_B = new double *[_n_threads];
    _visit_nodes_pool_A = new unode_int *[_n_threads];  // hold the nodes currently informed of A
    _visit_nodes_pool_B = new unode_int *[_n_threads];  // hold the nodes currently informed of B
    _status_nodes_pool_A = new NodeStatus *[_n_threads];  // 0: inactive, 1: informed, 2: suspended, 3: adopted (active)
    _status_nodes_pool_B = new NodeStatus *[_n_threads];  // 0: inactive, 1: informed, 2: suspended, 3: adopted (active)
    _status_edges_pool = new EdgeStatus *[_n_threads];

    for (int tid = 0; tid < _n_threads; tid++) {
        _alpha_pool_A[tid] = new double[n]();
        _alpha_pool_B[tid] = new double[n]();
        _visit_nodes_pool_A[tid] = new unode_int[n]();
        _visit_nodes_pool_B[tid] = new unode_int[n]();
        _status_nodes_pool_A[tid] = new NodeStatus[n]();
        _status_nodes_pool_B[tid] = new NodeStatus[n]();
        _status_edges_pool[tid] = new EdgeStatus[m]();
        memset(_status_nodes_pool_A[tid], NodeStatus::INACTIVE, n * sizeof(NodeStatus));
        memset(_status_nodes_pool_B[tid], NodeStatus::INACTIVE, n * sizeof(NodeStatus));
        memset(_status_edges_pool[tid], EdgeStatus::INLIVE, m * sizeof(EdgeStatus));

    }
    if (_n_threads > 1) {
        for (int tid = 0; tid < _n_threads; tid++) {
            _scheduler.set(tid, true);
        }
        // Case 1 implementation
//        boost::asio::thread_pool pool(n_threads_);
//        for (unode_int sample_id = 0; sample_id < n_samples; sample_id++) {
//            if (diffusionModel_ == DiffusionModel::LT) { // Linear threshold model, this method isn't implemented for LT
//                std::cerr << "Error: this part is only run by IC model." << std::endl;
//                exit(1);
//            } else if (diffusionModel_ == DiffusionModel::IC) { // Independent Cascade model
//                auto bind_fn = boost::bind(&Sampler::diffuse_ComIC_thread, this, ref(base_graph_ptr), aSeeds,
//                                           (cov_result + sample_id));
//                boost::asio::post(pool, bind_fn);
////                diffuse_ComIC_thread(base_graph_ptr, aSeeds, cov_result + sample_id,
////                                     _visit_nodes_pool_A, _visit_nodes_pool_B,
////                                     _status_nodes_pool_A, _status_nodes_pool_B, _status_edges_pool);
//            }
//        }
//        pool.join();
        // Case 2 implementation

        unode_int each_thread = n_samples / _n_threads, rest = n_samples - _n_threads * each_thread;
        vector<boost::thread> thread_list;
        unode_int head = 0, tail;
        for (int tid = 0; tid < _n_threads; tid++) {
            tail = head + each_thread;
            if (rest) {
                tail++;
                rest--;
            }
//            INFO(head, tail)
            if (tail > n_samples) tail = n_samples;

            boost::thread new_thread(boost::bind(&Sampler::diffuse_ComIC_parallel, this, ref(base_graph_ptr),
                                                 seeds_a, head, tail, (cov_result + head)));
            thread_list.push_back(boost::move(new_thread));

            head = tail;
        }
        for (boost::thread &t: thread_list)
            t.join();
    } else {
        for (unode_int sample_id = 0; sample_id < n_samples; sample_id++) {

            diffuse_ComIC(base_graph_ptr, seeds_a, cov_result + sample_id, 0);

        }
    }

    double total_cov = 0;
    for (unode_int sample_id = 0; sample_id < n_samples; sample_id++) {
        total_cov += (double) cov_result[sample_id];
    }

    for (int tid = 0; tid < _n_threads; tid++) {
        delete[] _alpha_pool_A[tid];
        delete[] _alpha_pool_B[tid];
        delete[] _visit_nodes_pool_A[tid];
        delete[] _visit_nodes_pool_B[tid];
        delete[] _status_nodes_pool_A[tid];
        delete[] _status_nodes_pool_B[tid];
        delete[] _status_edges_pool[tid];
    }
    delete[] _alpha_pool_A;
    delete[] _alpha_pool_B;
    delete[] _visit_nodes_pool_A;
    delete[] _visit_nodes_pool_B;
    delete[] _status_nodes_pool_A;
    delete[] _status_nodes_pool_B;
    delete[] _status_edges_pool;
    delete[] cov_result;

    return total_cov / (double) n_samples;
}

void Sampler::diffuse_ComIC_thread(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &aSeeds,
                                   unode_int *cov_ptr) {

    int tid = -1;
    _mt.lock();
    tid = (int) _scheduler.find_first();
    _scheduler.flip(tid);
    _mt.unlock();
    diffuse_ComIC(base_graph_ptr, aSeeds, cov_ptr, tid);
    _mt.lock();
    _scheduler.flip(tid);
    _mt.unlock();
}


void Sampler::diffuse_ComIC_parallel(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &seeds_a,
                                     unode_int head, unode_int tail, unode_int *cov_ptr) {

    int tid;
    _mt.lock();
    tid = (int) _scheduler.find_first();
    _scheduler.flip(tid);
    _mt.unlock();
    for (auto it = head; it < tail; it++) {
        diffuse_ComIC(base_graph_ptr, seeds_a, cov_ptr, tid);
        cov_ptr++;
    }

    _mt.lock();
    _scheduler.flip(tid);
    _mt.unlock();
}

void Sampler::diffuse_ComIC(const shared_ptr<Graph> &base_graph_ptr, const vector<unode_int> &seeds_a,
                            unode_int *cov_ptr, int tid) {

    (*cov_ptr) = 0;
    unode_int visit_head_A = 0, visit_tail_A = 0, visit_head_B = 0, visit_tail_B = 0;
    unode_int n = base_graph_ptr->_verNum;

    auto alpha_A = _alpha_pool_A[tid];
    auto alpha_B = _alpha_pool_B[tid];
    auto status_nodes_A = _status_nodes_pool_A[tid];
    auto status_nodes_B = _status_nodes_pool_B[tid];
    auto visit_nodes_A = _visit_nodes_pool_A[tid];
    auto visit_nodes_B = _visit_nodes_pool_B[tid];
    auto status_edges = _status_edges_pool[tid];
//    vector<double> alpha_A(n, 0);
//    vector<double> alpha_B(n, 0);
    for (int i = 0; i < n; i++) {
        alpha_A[i] = _urd(_gen_pool[tid]);
        alpha_B[i] = _urd(_gen_pool[tid]);
    }

//    memset(status_nodes_A, INACTIVE, sizeof(NodeStatus) * n);
//    memset(status_nodes_B, INACTIVE, sizeof(NodeStatus) * n);
    memset(status_edges, EdgeStatus::INLIVE, base_graph_ptr->_edgeNum * sizeof(EdgeStatus));

    // scan all A-seeds
    for (unode_int s: seeds_a) {
        status_nodes_A[s] = NodeStatus::ADOPTED;
        visit_nodes_A[visit_tail_A++] = s;
        (*cov_ptr)++;
        // iterate over its out-neighbors
        for (int nei_id = 0; nei_id < base_graph_ptr->getOutDeg(s); nei_id++) {
            unode_int edge_id = base_graph_ptr->getEdgeID(s, nei_id, false);
            unode_int v = base_graph_ptr->getEdgeIDNeighbor(edge_id, false);
            double coin = _urd(_gen_pool[tid]);
            double prob = base_graph_ptr->getEdgeIDProb(edge_id, false);
            if (coin <= prob) {
                status_edges[edge_id] = EdgeStatus::LIVE;
                if (status_nodes_A[v] != NodeStatus::ADOPTED)
                    visit_nodes_A[visit_tail_A++] = v;
            } else {
                status_edges[edge_id] = EdgeStatus::BLOCKED;
            }
        }
    }

    // scan all B-seeds
    for (unode_int s: _bSeeds) {
        status_nodes_B[s] = NodeStatus::ADOPTED;
        visit_nodes_B[visit_tail_B++] = s;
        for (int nei_id = 0; nei_id < base_graph_ptr->getOutDeg(s); nei_id++) { // iterate over its out-neighbors
            unode_int edge_id = base_graph_ptr->getEdgeID(s, nei_id);
            unode_int v = base_graph_ptr->getEdgeIDNeighbor(edge_id, false);
            if (status_edges[edge_id] == EdgeStatus::INLIVE) {
                double coin = _urd(_gen_pool[tid]);
                double prob = base_graph_ptr->getEdgeIDProb(edge_id, false);
                if (coin <= prob) {
                    status_edges[edge_id] = EdgeStatus::LIVE;  // edge is live
                    if (status_nodes_B[v] != NodeStatus::ADOPTED)
                        visit_nodes_B[visit_tail_B++] = v;
                } else {
                    status_edges[edge_id] = EdgeStatus::BLOCKED; // edge is blocked
                }
            } else if (status_edges[edge_id] == EdgeStatus::LIVE &&
                       status_nodes_B[v] != NodeStatus::ADOPTED) {
                visit_nodes_B[visit_tail_B++] = v;
            }
        }
    }

    unode_int curr_A = visit_tail_A;
    unode_int curr_B = visit_tail_B;

    while (curr_A > 0 || curr_B > 0) {
        // A-adoption test
        for (unode_int i = 0; i < curr_A; i++) {
            unode_int v = visit_nodes_A[visit_head_A++];
            if (status_nodes_A[v] != NodeStatus::INACTIVE)
                continue;

            if (status_nodes_B[v] != NodeStatus::ADOPTED) {
                // v is NOT B-adopted, test with q_A|0
                if (alpha_A[v] <= _qao) {
                    status_nodes_A[v] = NodeStatus::ADOPTED;  // A-adopted
                    (*cov_ptr)++;
                    if (status_nodes_B[v] == NodeStatus::SUSPENDED && alpha_B[v] <= _qba) {
                        status_nodes_B[v] = NodeStatus::ADOPTED; // reconsider to adopt B
                        examine_out_neighbors(base_graph_ptr, v, visit_nodes_B, &visit_tail_B,
                                              status_nodes_B, status_edges, tid);
                    }
                } else {
                    status_nodes_A[v] = NodeStatus::SUSPENDED; // A-suspended
                }

            } else {
                // v is already B-adopted, test with q_A|B
                if (alpha_A[v] <= _qab) {
                    status_nodes_A[v] = NodeStatus::ADOPTED;
                    (*cov_ptr)++;
                } else {
                    status_nodes_A[v] = NodeStatus::SUSPENDED;
                }
            }

            // if v adopts a product for the first time, we test its outgoing edges
            if (status_nodes_A[v] == NodeStatus::ADOPTED) {
                examine_out_neighbors(base_graph_ptr, v, visit_nodes_A, &visit_tail_A,
                                      status_nodes_A, status_edges, tid);
            } // END-IF
        } // ENDFOR

        // B adoption test
        for (int i = 0; i < curr_B; i++) {
            unode_int v = visit_nodes_B[visit_head_B++];
            if (status_nodes_B[v] != NodeStatus::INACTIVE)
                continue;

            // B adoption test for v
            if (status_nodes_A[v] != NodeStatus::ADOPTED) { // not A-adopted
                if (alpha_B[v] <= _qbo) {
                    status_nodes_B[v] = NodeStatus::ADOPTED;
                    if (status_nodes_A[v] == NodeStatus::SUSPENDED && alpha_A[v] <= _qab) {
                        status_nodes_A[v] = NodeStatus::ADOPTED; // reconsideration for A!
                        (*cov_ptr)++;
                        examine_out_neighbors(base_graph_ptr, v, visit_nodes_A, &visit_tail_A,
                                              status_nodes_A, status_edges, tid);
                    }
                } else {
                    status_nodes_B[v] = NodeStatus::SUSPENDED;
                }

            } else {
                status_nodes_B[v] = (alpha_B[v] <= _qba) ? NodeStatus::ADOPTED
                                                         : NodeStatus::SUSPENDED; // already A-adopted
            }

            if (status_nodes_B[v] == NodeStatus::ADOPTED) {
                examine_out_neighbors(base_graph_ptr, v, visit_nodes_B, &visit_tail_B,
                                      status_nodes_B, status_edges, tid);
            } // END-IF
        } // END-FOR

        curr_A = visit_tail_A - visit_head_A;
        curr_B = visit_tail_B - visit_head_B;

    } // END-WHILE

    for (int i = 0; i < visit_head_A; i++)
        status_nodes_A[visit_nodes_A[i]] = NodeStatus::INACTIVE;

    for (int i = 0; i < visit_head_B; i++)
        status_nodes_B[visit_nodes_B[i]] = NodeStatus::INACTIVE;
}


void Sampler::examine_out_neighbors(const shared_ptr<Graph> &base_graph_ptr, unode_int v,
                                    unode_int *visit_nodes, unode_int *visit_tail, const NodeStatus *node_status,
                                    EdgeStatus *status_edges, int tid) {
    for (int nei_id = 0; nei_id < base_graph_ptr->getOutDeg(v); nei_id++) {
        unode_int edge_id = base_graph_ptr->getEdgeID(v, nei_id, false);
        unode_int w = base_graph_ptr->getEdgeIDNeighbor(edge_id, false);

        if (status_edges[edge_id] == LIVE && node_status[w] == INACTIVE) {
            visit_nodes[(*visit_tail)++] = w;
        } else if (status_edges[edge_id] == INLIVE) {
            double coin = _urd(_gen_pool[tid]);
            double prob = base_graph_ptr->getEdgeIDProb(edge_id, false);
            if (coin <= prob) {
                status_edges[edge_id] = LIVE;
                if (node_status[w] == INACTIVE) {
                    visit_nodes[(*visit_tail)++] = w;
                }
            } else {
                status_edges[edge_id] = BLOCKED;
            }
        } // ENDIF
    } // ENDFOR
}



///**
// *  baseSpread: \sigma_A(S_A, \emptyset)
// */
//double Sampler::mineSeedsMC_comp(double baseSpread) {
//    double *improve = new double[graph->n];
//    int *last_update = new int[graph->n];
//    int *heap = new int[graph->n];
//    vector<int> tmp_set;
//    tmp_set.resize(k);
//
//    for (int i = 0; i < n; i++) {
//        heap[i] = i;
//        last_update[i] = -1;
//        improve[i] = (double) (n + 1);
//    }
//
//    double old = 0;
//    srand(time(NULL));
//    _bSeeds.clear();
//    mg.clear();
//
//    for (int i = 0; i < k; i++) {
//        while (last_update[heap[0]] != i) {
//            last_update[heap[0]] = i;
//            tmp_set[i] = heap[0];
//            improve[heap[0]] = compute_coverage_comp(tmp_set, i + 1) - old - baseSpread;
//
//            int x = 0;
//            while (x * 2 + 2 <= n - i) {
//                int newx = x * 2 + 1;
//                if ((newx + 1 < n - i) && (improve[heap[newx]] < improve[heap[newx + 1]]))
//                    newx++;
//                if (improve[heap[x]] < improve[heap[newx]]) {
//                    int t = heap[x];
//                    heap[x] = heap[newx];
//                    heap[newx] = t;
//                    x = newx;
//                } else {
//                    break;
//                }
//            } // end-while
//        } // end-while
//
//        _bSeeds.push_back(heap[0]);
//        tmp_set[i] = heap[0];
//        mg.push_back(improve[heap[0]]);
//        old += improve[heap[0]];
//
//        cout << "\tround " << i + 1 << ": node = " << _bSeeds[i] << ", mg = " << mg[i] << ", total = "
//             << (old + baseSpread) << endl;
//
//        heap[0] = heap[n - i - 1];
//        int x = 0;
//        while (x * 2 + 2 <= n - i) {
//            int newx = x * 2 + 1;
//            if ((newx + 1 < n - i) && (improve[heap[newx]] < improve[heap[newx + 1]]))
//                newx++;
//            if (improve[heap[x]] < improve[heap[newx]]) {
//                int t = heap[x];
//                heap[x] = heap[newx];
//                heap[newx] = t;
//                x = newx;
//            } else {
//                break;
//            }
//        } // endwhile
//
//    } // end-for
//
//    int rep = 3;
//    double final_spread = 0;
//    for (int j = 0; j < rep; j++)
//        final_spread += compute_coverage_comp(_bSeeds, k);
//
//    delete[] heap;
//    delete[] last_update;
//    delete[] improve;
//
//    return final_spread / (double) rep;
//}
//
//
//double Sampler::mineSeedsMC() {
//    double *improve = new double[graph->n];
//    int *last_update = new int[graph->n];
//    int *heap = new int[graph->n];
//    int *tmp_set = new int[k];
//    memset(tmp_set, 0, sizeof(int) * k);
//
//    for (int i = 0; i < n; i++) {
//        heap[i] = i;
//        last_update[i] = -1;
//        improve[i] = (double) (n + 1);
//    }
//
//    double old = 0;
//    srand(time(NULL));
//
//    _aSeeds.clear();
//    mg.clear();
//
//    int count_nodes = 0;
//    for (int i = 0; i < k; i++) {
//        while (last_update[heap[0]] != i) {
//            last_update[heap[0]] = i;
//            tmp_set[i] = heap[0];
//            improve[heap[0]] = compute_coverage(tmp_set, i + 1) - old;
//            if (i == 0) {
//                count_nodes++;
//                if (count_nodes % 1000 == 0)
//                    printf("1st iteration with %d nodes done...\n", count_nodes);
//            }
//
//            int x = 0;
//            while (x * 2 + 2 <= n - i) {
//                int newx = x * 2 + 1;
//                if ((newx + 1 < n - i) && (improve[heap[newx]] < improve[heap[newx + 1]]))
//                    newx++;
//                if (improve[heap[x]] < improve[heap[newx]]) {
//                    int t = heap[x];
//                    heap[x] = heap[newx];
//                    heap[newx] = t;
//                    x = newx;
//                } else {
//                    break;
//                }
//            } //endwhile
//        } //endwhile
//
//        _aSeeds.push_back(heap[0]);
//        tmp_set[i] = heap[0];
//        mg.push_back(improve[heap[0]]);
//        old += improve[heap[0]];
//
//        cout << "\tround " << i + 1 << ": node = " << _aSeeds[i] << ", mg = " << mg[i] << ", total = " << old << endl;
//
//        heap[0] = heap[n - i - 1];
//        int x = 0;
//        while (x * 2 + 2 <= n - i) {
//            int newx = x * 2 + 1;
//            if ((newx + 1 < n - i) && (improve[heap[newx]] < improve[heap[newx + 1]]))
//                newx++;
//            if (improve[heap[x]] < improve[heap[newx]]) {
//                int t = heap[x];
//                heap[x] = heap[newx];
//                heap[newx] = t;
//                x = newx;
//            } else {
//                break;
//            }
//        } // endwhile
//    } //endfor
//
//    printf("\n\n");
//
//    if (_aSeeds.size() < k)
//        cout << "[warning] less than " << k << " seeds were selected." << endl;
//    int *seed_set = new int[k];
//    for (int i = 0; i < k; i++)
//        seed_set[i] = _aSeeds.at(i);
//
//    int rep = 3;
//    double final_spread = 0;
//    for (int j = 0; j < rep; j++) {
//        final_spread += compute_coverage(seed_set, k);
//    }
//
//    delete[] heap;
//    delete[] last_update;
//    delete[] improve;
//    delete[] tmp_set;
//    delete[] seed_set;
//
//    return final_spread / (double) rep;
//}
//
//
//double Sampler::compute_coverage_comp(vector<int> bSeedSet, int size_B) {
//    ASSERT((int) bSeedSet.size() >= size_B);
//    //ASSERT((int)aSeeds.size() == 50);
//
//    double cov = 0;
//    double *alpha_A = new double[graph->n];
//    double *alpha_B = new double[graph->n];
//    int *status_A = new int[graph->n]; // 0: inactive, 1: informed, 2: suspended, 3: adopted (active)
//    int *status_B = new int[graph->n]; // 0: inactive, 1: informed, 2: suspended, 3: adopted (active)
//
//    deque<int> list_A;  // hold the nodes currently informed of A
//    deque<int> list_B;  // hold the nodes currently informed of B
//
//    for (int r = 0; r < MC_RUNS; r++) {
//        list_A.clear();
//        list_B.clear();
//        memset(status_A, 0, sizeof(int) * n);
//        memset(status_B, 0, sizeof(int) * n);
//        for (int i = 0; i < n; i++) {
//            alpha_A[i] = (double) rand() / (double) RAND_MAX;
//            alpha_B[i] = (double) rand() / (double) RAND_MAX;
//        }
//        graph->reset_out_edge_status();
//
//        // scan all A-seeds
//        for (int i = 0; i < (int) _aSeeds.size(); ++i) {
//            int u = _aSeeds.at(i);
//            status_A[u] = ADOPTED;
//            cov++;
//            // iterate over its out-neighbors
//            for (int j = 0; j < graph->outDeg[u]; j++) {
//                int v = graph->gO[u][j];
//                double coin = (double) rand() / (double) RAND_MAX;
//                if (coin <= graph->probO[u][j]) {
//                    graph->outEdgeStatus[u][j] = LIVE;
//                    if (status_A[v] != ADOPTED)
//                        list_A.push_back(v);
//                } else {
//                    graph->outEdgeStatus[u][j] = BLOCKED;
//                }
//            }
//        }
//
//        // scan all B-seeds
//        //for (auto it = bSeedSet.begin(); it != bSeedSet.end(); ++it)
//        for (int i = 0; i < size_B; i++) {
//            int u = bSeedSet.at(i);
//            status_B[u] = ADOPTED;
//
//            for (int j = 0; j < graph->outDeg[u]; j++) { // iterate over its out-neighbors
//                int v = graph->gO[u][j];
//                if (graph->outEdgeStatus[u][j] == INACTIVE) {
//                    double coin = (double) rand() / (double) RAND_MAX;
//                    if (coin <= graph->probO[u][j]) {
//                        graph->outEdgeStatus[u][j] = LIVE;  // edge is live
//                        if (status_B[v] != ADOPTED)
//                            list_B.push_back(v);
//                    } else {
//                        graph->outEdgeStatus[u][j] = BLOCKED; // edge is blocked
//                    }
//                } else if (graph->outEdgeStatus[u][j] == LIVE && status_B[v] != ADOPTED) {
//                    list_B.push_back(v);
//                }
//
//            }
//        }
//
//        int curr_A = list_A.size();
//        int curr_B = list_B.size();
//        int next_A = 0, next_B = 0;
//
//        while (curr_A > 0 || curr_B > 0) {
//            // A-adoption test
//            for (int i = 0; i < curr_A; i++) {
//                int v = list_A.front();
//                list_A.pop_front();
//                if (status_A[v] == SUSPENDED || status_A[v] == ADOPTED)
//                    continue;
//
//                if (status_B[v] != ADOPTED) {
//                    // v is NOT B-adopted, test with q_A|0
//                    if (alpha_A[v] <= _qao) {
//                        status_A[v] = ADOPTED;  // A-adopted
//                        cov++;
//                        if (status_B[v] == SUSPENDED && alpha_B[v] <= _qba) {
//                            status_B[v] = ADOPTED; // reconsider to adopt B
//                            examine_out_neighbors(v, &list_B, &next_B, status_B);
//                        }
//                    } else {
//                        status_A[v] = SUSPENDED; // A-suspended
//                    }
//
//                } else {
//                    // v is already B-adopted, test with q_A|B
//                    if (alpha_A[v] <= _qab) {
//                        status_A[v] = ADOPTED;
//                        cov++;
//                    } else {
//                        status_A[v] = SUSPENDED;
//                    }
//                }
//
//                // if v adopts a product for the first time, we test its outgoing edges
//                if (status_A[v] == ADOPTED) {
//                    examine_out_neighbors(v, &list_A, &next_A, status_A);
//                } // END-IF
//            } // ENDFOR
//
//            // B adoption test
//            for (int i = 0; i < curr_B; i++) {
//                int v = list_B.front();
//                list_B.pop_front();
//                if (status_B[v] == SUSPENDED || status_B[v] == ADOPTED)
//                    continue;
//
//                // B adoption test for v
//                if (status_A[v] != ADOPTED) { // not A-adopted
//                    if (alpha_B[v] <= _qbo) {
//                        status_B[v] = ADOPTED;
//                        if (status_A[v] == SUSPENDED && alpha_A[v] <= _qab) {
//                            status_A[v] = ADOPTED; // reconsideration for A!
//                            cov++;
//                            examine_out_neighbors(v, &list_A, &next_A, status_A);
//                        }
//                    } else {
//                        status_B[v] = SUSPENDED;
//                    }
//
//                } else {
//                    status_B[v] = (alpha_B[v] <= _qba) ? ADOPTED : SUSPENDED; // already A-adopted
//                }
//
//                if (status_B[v] == ADOPTED) {
//                    examine_out_neighbors(v, &list_B, &next_B, status_B);
//                } // END-IF
//            } // END-FOR
//
//            curr_A = next_A;
//            curr_B = next_B;
//            next_A = next_B = 0;
//
//        } // END-WHILE
//    }
//
//    delete[] status_A;
//    delete[] status_B;
//    delete[] alpha_A;
//    delete[] alpha_B;
//
//    return cov / (double) MC_RUNS;
//}
