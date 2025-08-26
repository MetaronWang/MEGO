#include "commonFun.h"

/**
  Builds a seed using nanoseconds to avoid same results.
*/
int seed_ns() {
//    struct timespec ts;
//    clock_gettime(CLOCK_MONOTONIC, &ts);
//    return (int)ts.tv_nsec;
    return (int) boost::random_device{}();
}

bool checkIfFIle(const std::string &filePath) {
    try {
        // Create a Path object from given path string
        boost::filesystem::path pathObj(filePath);
        // Check if path exists and is of a regular file
        if (boost::filesystem::exists(pathObj) && boost::filesystem::is_regular_file(pathObj))
            return true;
    }
    catch (boost::filesystem::filesystem_error &e) {
        std::cerr << e.what() << std::endl;
    }
    return false;
}

bool checkIfDirectory(const std::string &filePath) {
    try {
        // Create a Path object from given path string
        boost::filesystem::path pathObj(filePath);
        // Check if path exists and is of a directory file
        if (boost::filesystem::exists(pathObj) && boost::filesystem::is_directory(pathObj))
            return true;
    }
    catch (boost::filesystem::filesystem_error &e) {
        std::cerr << e.what() << std::endl;
    }
    return false;
}