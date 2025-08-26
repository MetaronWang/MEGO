# create python module
add_library(ComIMP
        MODULE
        ${SOURCE_FILES} ${HEADER_FILES} ${PYTHON_FILES}
)

target_link_libraries(ComIMP
        PUBLIC
        pybind11::module
        pybind11::embed
        Boost::system
        Boost::filesystem
        Boost::thread
        Boost::chrono
        Boost::log
        Boost::random
)

set_target_properties(ComIMP
        PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
)