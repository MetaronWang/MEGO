#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_strategy(py::module &);

namespace mcl {

    PYBIND11_MODULE(ComIMP, m) {
        // Optional docstring
        m.doc() = "ComIMP library";

        init_strategy(m);
    }
}
