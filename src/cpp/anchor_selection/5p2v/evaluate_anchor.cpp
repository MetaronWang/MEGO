#include "evaluate_anchor.hpp"


int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr
                << "Run as:\n calculate_dominated run_type set_file problem_num anchor_num\nwhere: run_type is the operation code and 1 means evaluate the anchor_problem pairs in serial; 2 means evaluate the anchor_problem pairs in parallel; 3 means evaluate the cartesion product of anchors and problems in parallel\n";
        exit(233);
    }

    int run_type = std::stoi(argv[1]);
    std::string set_file(argv[2]);
    int problem_num = std::stoi(argv[3]);
    int anchor_num = std::stoi(argv[4]);
    std::vector<std::vector<double>> problems = load_n_instances(problem_num);
    std::vector<std::vector<double>> anchors = load_n_instances(anchor_num);
    std::vector<int> success_result;
    std::vector<std::vector<int>> success_list;
    switch (run_type) {
        case 1:
            success_result = track_evaluate(set_file, problems, anchors);
            output_success(success_result);
            break;
        case 2:
            success_result = track_evaluate_parallel(set_file, problems, anchors);
            output_success(success_result);
            break;
        case 3:
            success_list = track_cartesion_evaluate_parallel(set_file, problems, anchors);
            output_success_list(success_list);
            break;
        default:
            std::cerr
                    << "run_type should be 1/2/3";
            exit(233);
    }
}