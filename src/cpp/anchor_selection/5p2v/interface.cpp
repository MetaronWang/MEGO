#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//#include <pybind11/stl_bind.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <iostream>
#include "calculate_cover.hpp"
#include "evaluate_anchor.hpp"
#include "sample_instances.hpp"

namespace py = pybind11;
//PYBIND11_MAKE_OPAQUE(std::vector<int>)
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<int>>)
//PYBIND11_MAKE_OPAQUE(std::string)
//PYBIND11_MAKE_OPAQUE(std::vector<double>)
//PYBIND11_MAKE_OPAQUE(std::vector<std::vector<double>>)

PYBIND11_MODULE(LM5P2V, m) {
    m.doc() = "A CPP interface for 5points-2views Problem";
//    py::bind_vector<std::vector<int>>(m, "VectorInt");
//    py::bind_vector<std::vector<double>>(m, "VectorDouble");
//    py::bind_vector<std::vector<std::vector<int>>>(m, "VectorInt");
//    py::bind_vector<std::vector<std::vector<double>>>(m, "VectorDouble");
    //calculate_cover.hpp
    m.def("single_cover", &single_cover);
    m.def("set_cover_serial", &set_cover_serial);
    m.def("set_cover_parallel", &set_cover_parallel);
    m.def("set_cover_list", &set_cover_list);
    m.def("load_n_instances", &load_n_instances);
    m.def("load_anchors", &load_anchors);
    //evaluate_anchor.hpp
    m.def("track_evaluate", &track_evaluate);
    m.def("track_evaluate_parallel", &track_evaluate_parallel);
    m.def("track_cartesion_evaluate_parallel", &track_cartesion_evaluate_parallel);
    m.def("load_n_instances", &load_n_instances);
    //sample_instance.hpp
    m.def("sample_instances", &sample_instances);
}
