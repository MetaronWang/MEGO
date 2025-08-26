#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>
#include <string>
#include <omp.h>
#include <thread>
#include <numeric>
#include <set>

#ifndef _COMMON_DEF_
#define _COMMON_DEF_

#include "common_defination.hpp"

#endif

using namespace std::chrono;


std::vector<int> single_cover(std::string set_file, std::vector<std::vector<Float>> &instances, int anchor) {
    track_settings settings;
    bool succ_load = load_settings(set_file, settings);
    if (!succ_load) return std::vector<int>(0);
    std::vector<std::vector<Float>> problems(instances.size());
    std::vector<std::vector<Float>> start(instances.size());
    std::vector<std::vector<Float>> depths(instances.size());
    split_instances(instances, problems, start, depths);
    int n = problems.size();

    //initialize the statistics
    std::vector<int> success;
    int num_steps;
    Float params[48];
    Float solution[12];
    success.push_back(anchor);
    //copy the start problem
    for (int a = 0; a < 24; a++) {
        params[a] = problems[anchor][a];
    }

    Float cur_start[12];
    for (int a = 0; a < 12; ++a) cur_start[a] = start[anchor][a];


    for (int j = 0; j < n; j++) {
        if (j == anchor) continue;
        for (int a = 0; a < 24; a++) {
            params[24 + a] = problems[j][a];
        }

        //track the problem
        int status = track(settings, cur_start, params, solution, &num_steps);
        if (status == 2) {
            Float diff = 0;
            for (int a = 0; a < 11; a++) {
                Float cdiff = solution[a] - start[j][a];
                diff = diff + cdiff * cdiff;
            }
            if (diff <= settings.corr_thresh_) success.push_back(j);
        }
    }
    return success;
}

std::vector<int>
set_cover_serial(std::string set_file, std::vector<std::vector<Float>> &instances, std::vector<int> &anchors) {
    track_settings settings;
    bool succ_load = load_settings(set_file, settings);
    if (!succ_load) { return {}; }
    std::vector<std::vector<Float>> problems(instances.size());
    std::vector<std::vector<Float>> start(instances.size());
    std::vector<std::vector<Float>> depths(instances.size());
    split_instances(instances, problems, start, depths);
    int n = problems.size();
    std::set<int> non_dominated;
    for (int i = 0; i < n; i++) non_dominated.insert(i);
    std::vector<int> success;
    for (int anchor: anchors) {
        success.push_back(anchor);
        non_dominated.erase(anchor);
    }
    for (int anchor: anchors) {
        int num_steps;
        Float params[48];
        double solution[12];
        //copy the start problem
        for (int a = 0; a < 24; a++) {
            params[a] = problems[anchor][a];
        }

        Float cur_start[12];
        for (int a = 0; a < 12; ++a) cur_start[a] = start[anchor][a];

        std::vector<int> new_dominated;
        for (int j: non_dominated) {
            if (j == anchor) continue;
            for (int a = 0; a < 24; a++) {
                params[24 + a] = problems[j][a];
            }

            //track the problem
            int status = track(settings, cur_start, params, solution, &num_steps);
            if (status == 2) {
                Float diff = 0;
                for (int a = 0; a < 11; a++) {
                    Float cdiff = solution[a] - start[j][a];
                    diff += cdiff * cdiff;
                }
                if (diff <= settings.corr_thresh_) {
                    new_dominated.push_back(j);
                }
            }
        }
        for (int j: new_dominated) {
            success.push_back(j);
            non_dominated.erase(j);
        }
    }
    return success;
}

std::vector<int>
set_cover_parallel(std::string set_file, std::vector<std::vector<Float>> &instances, std::vector<int> &anchors) {
    track_settings settings;
    bool succ_load = load_settings(set_file, settings);
    if (!succ_load) { return {}; }
    std::vector<std::vector<Float>> problems(instances.size());
    std::vector<std::vector<Float>> start(instances.size());
    std::vector<std::vector<Float>> depths(instances.size());
    split_instances(instances, problems, start, depths);
    unsigned long n = problems.size();
    std::vector<std::vector<int>> success;

    for (int i; i < anchors.size(); i++) {
        success.emplace_back();
    }

    int concurrency = int(std::thread::hardware_concurrency());
    if (concurrency > anchors.size()) concurrency = int(anchors.size());
//    std::cerr << "hardware concurrency:" << concurrency << std::endl;
    omp_set_num_threads(concurrency);
#pragma omp parallel for
    for (int i = 0; i < anchors.size(); i++) {
        int anchor = anchors[i];
        int num_steps;
        success[i].push_back(anchor);
        Float params[48];
        double solution[12];
        //copy the start problem
        for (int a = 0; a < 24; a++) {
            params[a] = problems[anchor][a];
        }

        Float cur_start[12];
        for (int a = 0; a < 12; ++a) cur_start[a] = start[anchor][a];

        for (int j = 0; j < n; j++) {
            if (j == anchor) continue;
            for (int a = 0; a < 24; a++) {
                params[24 + a] = problems[j][a];
            }

            //track the problem
            int status = track(settings, cur_start, params, solution, &num_steps);
            if (status == 2) {
                Float diff = 0;
                for (int a = 0; a < 11; a++) {
                    Float cdiff = solution[a] - start[j][a];
                    diff += cdiff * cdiff;
                }
                if (diff <= settings.corr_thresh_) {
                    success[i].push_back(j);
                }
            }
        }
    }

//    for (int i = 0; i < success.size(); i++) std::cout << success[i].size() << " ";
//    std::cout << "\n";
    std::set<int> total_success;
    for (const std::vector<int> &single_success: success) {
        for (int instance: single_success) {
            total_success.insert(instance);
        }
    }
    std::vector<int> success_vector;
    success_vector.assign(total_success.begin(), total_success.end());
    return success_vector;
}

std::vector<std::vector<int>>
set_cover_list(std::string set_file, std::vector<std::vector<Float>> &instances, std::vector<int> &anchors) {
    track_settings settings;
    bool succ_load = load_settings(set_file, settings);
    if (!succ_load) { return {}; }
    std::vector<std::vector<Float>> problems(instances.size());
    std::vector<std::vector<Float>> start(instances.size());
    std::vector<std::vector<Float>> depths(instances.size());
    split_instances(instances, problems, start, depths);
    unsigned long n = problems.size();
    std::vector<std::vector<int>> success;

    for (int i; i < anchors.size(); i++) {
        success.emplace_back();
    }

    int concurrency = int(std::thread::hardware_concurrency());
    if (concurrency > anchors.size()) concurrency = int(anchors.size());
//    std::cerr << "hardware concurrency:" << concurrency << std::endl;
    omp_set_num_threads(concurrency);
#pragma omp parallel for
    for (int i = 0; i < anchors.size(); i++) {
        int anchor = anchors[i];
        int num_steps;
        success[i].push_back(anchor);
        Float params[48];
        double solution[12];
        //copy the start problem
        for (int a = 0; a < 24; a++) {
            params[a] = problems[anchor][a];
        }

        Float cur_start[12];
        for (int a = 0; a < 12; ++a) cur_start[a] = start[anchor][a];

        for (int j = 0; j < n; j++) {
            if (j == anchor) continue;
            for (int a = 0; a < 24; a++) {
                params[24 + a] = problems[j][a];
            }

            //track the problem
            int status = track(settings, cur_start, params, solution, &num_steps);
            if (status == 2) {
                Float diff = 0;
                for (int a = 0; a < 11; a++) {
                    Float cdiff = solution[a] - start[j][a];
                    diff += cdiff * cdiff;
                }
                if (diff <= settings.corr_thresh_) {
                    success[i].push_back(j);
                }
            }
        }
    }
    return success;
}

std::vector<std::vector<double>> load_instances() {
    int n;
    std::cin >> n;
    std::cerr << n << " problems\n";
    std::vector<std::vector<double>> instances(n);
    for (int i = 0; i < n; i++) {
        std::vector<double> instance(37);
        for (int j = 0; j < 37; j++) {
            double u;
            std::cin >> u;
            instance[j] = u;
        }
        instances[i] = instance;
    }
    return instances;
}

std::vector<int> load_anchors(int n) {
    std::cerr << n << " anchors\n";
    std::vector<int> anchors(n);
    for (int i = 0; i < n; i++) {
        int u;
        std::cin >> u;
        anchors[i] = u;
    }
    return anchors;
}

void output_cover(std::vector<int> &cover) {
    unsigned int n = cover.size();
    std::cout << n;
    for (int i = 0; i < n; i++) {
        std::cout << " " << cover[i];
    }
}

void output_cover_list(std::vector<std::vector<int>> &cover_list) {
    unsigned int n = cover_list.size();
    std::cout << n << std::endl;
    for (int i = 0; i < n; i++) {
        unsigned int m = cover_list[i].size();
        std::cout << m << " " << i;
        for (int j = 0; j < m; j++) {
            std::cout << " " << cover_list[i][j];
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<double>> load_instances_from_file(std::string instance_file) {
    std::ifstream f;
    f.open(instance_file);
    int n;
    f >> n;
    std::cerr << n << " problems\n";
    std::vector<std::vector<double>> instances(n);
    for (int i = 0; i < n; i++) {
        std::vector<double> instance(37);
        for (int j = 0; j < 37; j++) {
            double u;
            f >> u;
            instance[j] = u;
        }
        instances[i] = instance;
    }
    f.close();
    return instances;
}

std::vector<std::vector<int>>
get_set_cover_list_from_file(std::string set_file, std::string instance_file, std::vector<int> anchors) {
    std::vector<int> dominate;
    std::vector<std::vector<int>> dominate_list;
    std::vector<std::vector<double>> instances = load_instances_from_file(instance_file);
    dominate_list = set_cover_list(set_file, instances, anchors);
    return dominate_list;
}