// Problem-solution sampler for the 4 point problem (Sec. 3)
#include <iostream>
#include <vector>
#include <string>
#include "sample_instances.hpp"

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr
                << "Run as:\n sample_instances data_folder num_samples\nwhere: data_folder is the folder where the data is located, num_samples is the number of samples per camera pair\n";
        return 0;
    }

    std::string data_folder(argv[1]);

    int samples = std::stoi(argv[2]);
    std::cerr << samples << " samples\n";
    std::vector<std::vector<double>> instances = sample_instances(data_folder, samples);
    output_instances(instances);
    return 0;
}
