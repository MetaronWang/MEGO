#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

#include "../src/strategy.h"

namespace py = pybind11;


void init_strategy(py::module &m) {
    py::class_<Strategy>(m, "Strategy")
            .def(py::init<const string &, const string &, bool, const vector<double> &, const vector <unode_int> &, unode_int, int, bool>(),
                 py::arg("dataset_dir"), py::arg("dataset_name"), py::arg("directed"), py::arg("gap"),
                 py::arg("seeds_b"), py::arg("sample_num") = 1000, py::arg("thread_num") = 4,
                 py::arg("is_load_graph") = false)
            .def("perform", &Strategy::perform, py::arg("k"))
            .def("evaluate", &Strategy::evaluate, py::arg("seeds_a"))
            .def("serializeGraph", &Strategy::serializeGraph)
            .def("getVerNum", &Strategy::getVerNum)
            .def("getEdgeNum", &Strategy::getEdgeNum)
            .def_property("_thread_num", &Strategy::getThreadNum, &Strategy::setThreadNum)
            .def_property("_sample_num", &Strategy::getSampleNum, &Strategy::setSampleNum);
}

