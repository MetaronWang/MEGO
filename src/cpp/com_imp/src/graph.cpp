#include "graph.h"


Graph::Graph(const Graph &graph) :
        _verNum(graph._verNum), _edgeNum(graph._edgeNum), _directed(graph._directed) {
    _neighbor = graph._neighbor;
    _rev_neighbor = graph._rev_neighbor;
    _neighbor_ptr = graph._neighbor_ptr;
    _rev_neighbor_ptr = graph._rev_neighbor_ptr;
    _inDegree = graph._inDegree;
    _outDegree = graph._outDegree;
}

void Graph::readGraph() {
    string probGraphFile = _folder + _graph_file;
    INFO(probGraphFile)
    ifstream fin(probGraphFile);
    ASSERT(fin.is_open())
    unode_int n, m;
    fin >> n >> m;
    INFO(n, m)
    _verNum = n;
    _edgeNum = m;
    if (!_directed)
        _edgeNum *= 2;
    _inDegree.resize(_verNum);
    _outDegree.resize(_verNum);
    _neighbor_ptr.resize(_verNum);
    _rev_neighbor_ptr.resize(_verNum);

    Edge *edge;
    edge = new Edge[_edgeNum];
    unode_int u, v;
    double p = -1;
    unode_int cnt = 0;
    for (unode_int i = 0; i < m; i++) {
        fin >> u >> v >> p;
        edge[cnt]._u = u;
        edge[cnt]._v = v;
        edge[cnt]._w = p;
        _inDegree[v]++;
        _outDegree[u]++;
        cnt++;
        if (!_directed) {
            fin >> p;
            edge[cnt]._u = v;
            edge[cnt]._v = u;
            edge[cnt]._w = p;
            _inDegree[u]++;
            _outDegree[v]++;
            cnt++;
        }
    }
    fin.close();

    _neighbor.resize(_edgeNum);
    _rev_neighbor.resize(_edgeNum);

    _probability.resize(_edgeNum);
    _rev_probability.resize(_edgeNum);


    sort(edge, edge + _edgeNum, [](const Edge &a, const Edge &b) -> bool {
        if (a._u == b._u)
            return a._v < b._v;
        return a._u < b._u;
    });
    for (unode_int i = 0; i < _edgeNum; i++) {
        _neighbor[i] = edge[i]._v;
    }

    for (unode_int i = 0; i < _edgeNum; i++) {
        _probability[i] = edge[i]._w;
    }


    sort(edge, edge + _edgeNum, [](const Edge &a, const Edge &b) -> bool {
        if (a._v == b._v)
            return a._u < b._u;
        return a._v < b._v;
    });
    for (unode_int i = 0; i < _edgeNum; i++) {
        _rev_neighbor[i] = edge[i]._u;
    }

    for (unode_int i = 0; i < _edgeNum; i++) {
        _rev_probability[i] = edge[i]._w;
    }
    _neighbor_ptr = _outDegree;
    _rev_neighbor_ptr = _inDegree;
    for (unode_int i = 1; i < _verNum; i++) {
        _neighbor_ptr[i] += _neighbor_ptr[i - 1];
        _rev_neighbor_ptr[i] += _rev_neighbor_ptr[i - 1];
    }
    delete[] edge;
}


unode_int Graph::getNeighbor(unode_int u, unode_int nei_id, bool inv) const {
    if (!inv) {
        if (u) return _neighbor[_neighbor_ptr[u - 1] + nei_id];
        return _neighbor[nei_id];
    } else {
        if (u) return _rev_neighbor[_rev_neighbor_ptr[u - 1] + nei_id];
        return _rev_neighbor[nei_id];
    }
}

double Graph::getNeighborProb(unode_int u, unode_int nei_id, bool inv) const {
    if (!inv) {
        if (u) return _probability[_neighbor_ptr[u - 1] + nei_id];
        return _probability[nei_id];
    } else {
        if (u) return _rev_probability[_rev_neighbor_ptr[u - 1] + nei_id];
        return _rev_probability[nei_id];
    }
}

unode_int Graph::getEdgeID(unode_int u, unode_int nei_id, bool inv) const {
    if (!inv) {
        if (u) return _neighbor_ptr[u - 1] + nei_id;
        return nei_id;
    } else {
        if (u) return _rev_neighbor_ptr[u - 1] + nei_id;
        return nei_id;
    }
}

unode_int Graph::getEdgeIDNeighbor(unode_int edge_id, bool inv) const {
    if (!inv) {
        return _neighbor[edge_id];
    } else {
        return _rev_neighbor[edge_id];
    }
}


double Graph::getEdgeIDProb(unode_int edge_id, bool inv) const {
    if (!inv) {
        return _probability[edge_id];
    } else {
        return _rev_probability[edge_id];
    }
}

void Graph::saveGraphStruct() {
    if (_verNum == 0 || _edgeNum == 0) {
        cerr << "Graph is empty!" << endl;
        return;
    }
    boost::filesystem::path struct_path(_folder);
    struct_path /= "serialize";
    if (!checkIfDirectory(struct_path.string())) {
        INFO("Create serialize dir: ", struct_path.string())
        boost::filesystem::create_directories(struct_path);
    }
    struct_path /= _graph_file;
    string postfix = ".graph";

    saveSerializedFile(struct_path.string() + ".neighbor" + postfix, _neighbor);
    saveSerializedFile(struct_path.string() + ".rev_neighbor" + postfix, _rev_neighbor);
    saveSerializedFile(struct_path.string() + ".neighbor_ptr" + postfix, _neighbor_ptr);
    saveSerializedFile(struct_path.string() + ".rev_neighbor_ptr" + postfix, _rev_neighbor_ptr);
    saveSerializedFile(struct_path.string() + ".outDegree" + postfix, _outDegree);
    saveSerializedFile(struct_path.string() + ".inDegree" + postfix, _inDegree);
    saveSerializedFile(struct_path.string() + ".probability" + postfix, _probability);
    saveSerializedFile(struct_path.string() + ".rev_probability" + postfix, _rev_probability);
}

bool Graph::checkGraphStruct() const {
    boost::filesystem::path struct_path(_folder);
    struct_path /= "serialize";
    struct_path /= _graph_file;
    string postfix = ".graph";
    if (!checkIfFIle(struct_path.string() + ".neighbor" + postfix))return false;
    if (!checkIfFIle(struct_path.string() + ".rev_neighbor" + postfix))return false;
    if (!checkIfFIle(struct_path.string() + ".neighbor_ptr" + postfix))return false;
    if (!checkIfFIle(struct_path.string() + ".rev_neighbor_ptr" + postfix))return false;
    if (!checkIfFIle(struct_path.string() + ".outDegree" + postfix))return false;
    if (!checkIfFIle(struct_path.string() + ".inDegree" + postfix))return false;
    if (!checkIfFIle(struct_path.string() + ".probability" + postfix))return false;
    if (!checkIfFIle(struct_path.string() + ".rev_probability" + postfix))return false;

    return true;
}


void Graph::loadGraphStruct() {
    if (!checkGraphStruct()) {
        cerr << "Graph struct files not exist!" << endl;
        return;
    }
    boost::filesystem::path struct_path(_folder);
    struct_path /= "serialize";
    struct_path /= _graph_file;
    string postfix = ".graph";

    loadSerializedFile(struct_path.string() + ".neighbor" + postfix, _neighbor);
    loadSerializedFile(struct_path.string() + ".rev_neighbor" + postfix, _rev_neighbor);
    loadSerializedFile(struct_path.string() + ".neighbor_ptr" + postfix, _neighbor_ptr);
    loadSerializedFile(struct_path.string() + ".rev_neighbor_ptr" + postfix, _rev_neighbor_ptr);
    loadSerializedFile(struct_path.string() + ".outDegree" + postfix, _outDegree);
    loadSerializedFile(struct_path.string() + ".inDegree" + postfix, _inDegree);
    loadSerializedFile(struct_path.string() + ".probability" + postfix, _probability);
    loadSerializedFile(struct_path.string() + ".rev_probability" + postfix, _rev_probability);
    _verNum = _neighbor_ptr.size();
    _edgeNum = _neighbor.size();
}