#ifndef COMMONFUN_H
#define COMMONFUN_H

#include<boost/filesystem.hpp>
#include <boost/random/random_device.hpp>

#include "head.h"
#include "serialize.h"

/**
  Builds a seed using nanoseconds to avoid same results.
*/
int seed_ns();

/**
  Check if given string path is of a file
*/
bool checkIfFIle(const std::string &filePath);

/**
  Check if given string path is of a Directory
*/
bool checkIfDirectory(const std::string &filePath);

template<typename T>
static void freeSpace(T *ptr) {
    if (ptr != nullptr)
        delete[] ptr;
}

/// Save a serialized file
template<class T>
static void saveSerializedFile(const std::string &filename, const T &output) {
    std::ofstream outfile(filename, std::ios::binary);

    if (!outfile.eof() && !outfile.fail()) {
        StreamType res;
        serialize(output, res);
        outfile.write(reinterpret_cast<char *>(&res[0]), res.size());
        outfile.close();
        res.clear();
        std::cout << "Save file successfully: " << filename << '\n';
    } else {
        std::cout << "Save file failed: " + filename << '\n';
        exit(1);
    }
}

/// Load a serialized file
template<class T>
static void loadSerializedFile(const std::string &filename, T &input) {
    std::ifstream infile(filename, std::ios::binary);

    if (!infile.eof() && !infile.fail()) {
        infile.seekg(0, std::ios_base::end);
        const std::streampos fileSize = infile.tellg();
        infile.seekg(0, std::ios_base::beg);
        std::vector<uint8_t> res(fileSize);
        infile.read(reinterpret_cast<char *>(&res[0]), fileSize);
        infile.close();
        input.clear();
        auto it = res.cbegin();
        input = deserialize<T>(it, res.cend());
        res.clear();
    } else {
        std::cout << "Cannot open file: " + filename << '\n';
        exit(1);
    }
}

#endif //COMMONFUN_H
