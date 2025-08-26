#ifndef GRAPH_H
#define GRAPH_H

#include<boost/filesystem.hpp>

#include "head.h"
#include "commonFun.h"

using namespace std;
typedef uint64_t unode_int; // Type for node ids (can be changed into 32 or 64 bits)
typedef pair<unode_int, unode_int> uedge_pair;

struct Edge {
    unode_int _u, _v;
    double _w;

    Edge() : _u(-1), _v(-1), _w(-1) {};

    Edge(unode_int u, unode_int v, double w) : _u(u), _v(v), _w(w) {};

    bool operator==(const Edge &edge) const {
        return (this->_u == edge._u) && (this->_v == edge._v);
    }
};

class Graph {
protected:
    vector<unode_int> _neighbor_ptr, _rev_neighbor_ptr;     // size: _verNum
    vector<unode_int> _inDegree, _outDegree;        // size: _verNum
    vector<unode_int> _neighbor, _rev_neighbor;     // size: _edgeNum
    vector<double> _probability, _rev_probability;     // size: _edgeNum

public:
    string _folder;
    string _graph_file;
    bool _directed = true;

    unode_int _verNum = 0;
    unode_int _edgeNum = 0; // when directed is true, edgeNum_ = 2 * m_

    explicit Graph() = default;

    Graph(string folder, string graph_file, int directed, bool is_load_graph = false) :
            _folder(std::move(folder)), _graph_file(std::move(graph_file)), _directed(directed) {
        if (is_load_graph)
            loadGraphStruct();
        else
            readGraph();
    }

    Graph(const Graph &graph);

    ~Graph() = default;

    void readGraph();

    unode_int getOutDeg(unode_int u) const {
        return _outDegree[u];
    }

    unode_int getInDeg(unode_int v) const {
        return _inDegree[v];
    }

    unode_int getNeighbor(unode_int u, unode_int nei_id, bool inv = false) const;

    double getNeighborProb(unode_int u, unode_int nei_id, bool inv = false) const;

    unode_int getEdgeID(unode_int u, unode_int nei_id, bool inv = false) const;

    unode_int getEdgeIDNeighbor(unode_int edge_id, bool inv = false) const;

    double getEdgeIDProb(unode_int edge_id, bool inv = false) const;

    // Save graph structure to a file
    void saveGraphStruct();

    // Check graph structure of a file
    bool checkGraphStruct() const;

    // Load graph structure from a file
    void loadGraphStruct();

    void showData() const {};
};

#endif