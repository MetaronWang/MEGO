#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include "calculate_cover.hpp"

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr
                << "Run as:\n calculate_dominated run_type set_file run_arg\nwhere: run_type is the operation code and 1 means single eval; 2 means serial eval set; 3 means parallel eval set; 4 means get dominate list\n";
        exit(233);
    }
    int run_type = std::stoi(argv[1]);
    std::string set_file(argv[2]);
    int run_arg = std::stoi(argv[3]);
    std::vector<std::vector<double>> instances = load_instances();
    std::vector<int> anchors;
    std::vector<int> dominate;
    std::vector<std::vector<int>> dominate_list;
    switch (run_type) {
        case 1:
            dominate = single_cover(set_file, instances, run_arg);
            output_cover(dominate);
            break;
        case 2:
            anchors = load_anchors(run_arg);
            dominate = set_cover_serial(set_file, instances, anchors);
            output_cover(dominate);
            break;
        case 3:
            anchors = load_anchors(run_arg);
            dominate = set_cover_parallel(set_file, instances, anchors);
            output_cover(dominate);
            break;
        case 4:
            anchors = load_anchors(run_arg);
            dominate_list = set_cover_list(set_file, instances, anchors);
            output_cover_list(dominate_list);
            break;
        default:
            std::cerr
                    << "run_type should be 1/2/3/4";
            exit(233);
    }
    return 0;
}

