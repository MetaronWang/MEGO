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



std::vector<int> track_evaluate(std::string set_file, std::vector<std::vector<Float>> &instances,
                                std::vector<std::vector<Float>> &anchors) {
    assert(instances.size() == anchors.size());
    track_settings settings;
    bool succ_load = load_settings(set_file, settings);
    if (!succ_load) return std::vector<int>(0);
    std::vector<std::vector<Float>> problems(instances.size());
    std::vector<std::vector<Float>> start(instances.size());
    std::vector<std::vector<Float>> depths(instances.size());
    std::vector<std::vector<Float>> anchor_problems(anchors.size());
    std::vector<std::vector<Float>> anchor_start(anchors.size());
    std::vector<std::vector<Float>> anchor_depths(anchors.size());
    split_instances(instances, problems, start, depths);
    split_instances(anchors, anchor_problems, anchor_start, anchor_depths);
    int n = problems.size();

    //initialize the statistics
    std::vector<int> success;
    int num_steps;
    Float params[40];
    Float solution[10];

    for (int j = 0; j < n; j++) {
        //copy the start problem
        for (int a = 0; a < 20; a++) {
            params[a] = anchor_problems[j][a];
        }

        Float cur_start[9];
        for (int a = 0; a < 9; ++a) cur_start[a] = anchor_start[j][a];
        for (int a = 0; a < 20; a++) {
            params[20 + a] = problems[j][a];
        }

        //track the problem
        int status = track(settings, cur_start, params, solution, &num_steps);
        if (status == 2) {
            Float diff = 0;
            for (int a = 0; a < 9; a++) {
                Float cdiff = solution[a] - start[j][a];
                diff = diff + cdiff * cdiff;
            }
            if (diff <= settings.corr_thresh_) success.push_back(j);
        }
    }
    return success;
}

std::vector<int> track_evaluate_parallel(std::string set_file, std::vector<std::vector<Float>> &instances,
                                         std::vector<std::vector<Float>> &anchors) {
    assert(instances.size() == anchors.size());
    track_settings settings;
    bool succ_load = load_settings(set_file, settings);
    if (!succ_load) return std::vector<int>(0);
    std::vector<std::vector<Float>> problems(instances.size());
    std::vector<std::vector<Float>> start(instances.size());
    std::vector<std::vector<Float>> depths(instances.size());
    std::vector<std::vector<Float>> anchor_problems(anchors.size());
    std::vector<std::vector<Float>> anchor_start(anchors.size());
    std::vector<std::vector<Float>> anchor_depths(anchors.size());
    split_instances(instances, problems, start, depths);
    split_instances(anchors, anchor_problems, anchor_start, anchor_depths);
    int n = problems.size();

    //initialize the statistics
    std::vector<int> success;

    int concurrency = int(std::thread::hardware_concurrency());
    if (concurrency > anchors.size() / 100) concurrency = int(anchors.size() / 100);
    if (concurrency < 1) concurrency = 1;
//    std::cerr << "hardware concurrency:" << concurrency << std::endl;
    omp_set_num_threads(concurrency);
#pragma omp parallel for
    for (int j = 0; j < n; j++) {
        int num_steps;
        Float params[40];
        Float solution[9];

        //copy the start problem
        for (int a = 0; a < 20; a++) {
            params[a] = anchor_problems[j][a];
        }

        Float cur_start[9];
        for (int a = 0; a < 9; ++a) cur_start[a] = anchor_start[j][a];
        for (int a = 0; a < 20; a++) {
            params[20 + a] = problems[j][a];
        }

        //track the problem
        int status = track(settings, cur_start, params, solution, &num_steps);
        if (status == 2) {
            Float diff = 0;
            for (int a = 0; a < 9; a++) {
                Float cdiff = solution[a] - start[j][a];
                diff = diff + cdiff * cdiff;
            }
            if (diff <= settings.corr_thresh_) success.push_back(j);
        }
    }
    return success;
}

std::vector<std::vector<int>>
track_cartesion_evaluate_parallel(std::string set_file, std::vector<std::vector<Float>> &instances,
                                  std::vector<std::vector<Float>> &anchors) {
    track_settings settings;
    bool succ_load = load_settings(set_file, settings);
    if (!succ_load) { return {}; }
    std::vector<std::vector<Float>> problems(instances.size());
    std::vector<std::vector<Float>> start(instances.size());
    std::vector<std::vector<Float>> depths(instances.size());
    std::vector<std::vector<Float>> anchor_problems(anchors.size());
    std::vector<std::vector<Float>> anchor_start(anchors.size());
    std::vector<std::vector<Float>> anchor_depths(anchors.size());
    split_instances(instances, problems, start, depths);
    split_instances(anchors, anchor_problems, anchor_start, anchor_depths);
    unsigned long n = problems.size();
    unsigned long m = anchors.size();
    std::vector<std::vector<int>> success;

    int concurrency = int(std::thread::hardware_concurrency());
    if (concurrency > anchors.size()) concurrency = int(anchors.size());
//    std::cerr << "hardware concurrency:" << concurrency << std::endl;
    omp_set_num_threads(concurrency);
#pragma omp parallel for
    for (int i = 0; i < m; i++) {
        int num_steps;
        Float params[40];
        double solution[9];
        //copy the start problem
        for (int a = 0; a < 20; a++) {
            params[a] = anchor_problems[i][a];
        }

        Float cur_start[9];
        for (int a = 0; a < 9; ++a) cur_start[a] = anchor_start[i][a];

        for (int j = 0; j < n; j++) {
            for (int a = 0; a < 20; a++) {
                params[20 + a] = problems[j][a];
            }

            //track the problem
            int status = track(settings, cur_start, params, solution, &num_steps);
            if (status == 2) {
                Float diff = 0;
                for (int a = 0; a < 9; a++) {
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

std::vector<std::vector<double>> load_n_instances(int n) {
    std::cerr << n << " problems\n";
    std::vector<std::vector<double>> instances(n);
    for (int i = 0; i < n; i++) {
        std::vector<double> instance(30);
        for (int j = 0; j < 30; j++) {
            double u;
            std::cin >> u;
            instance[j] = u;
        }
        instances[i] = instance;
    }
    return instances;
}

void output_success(std::vector<int> &success) {
    unsigned int n = success.size();
    std::cout << n;
    for (int i = 0; i < n; i++) {
        std::cout << " " << success[i];
    }
}

void output_success_list(std::vector<std::vector<int>> &success_list) {
    unsigned int n = success_list.size();
    std::cout << n << std::endl;
    for (int i = 0; i < n; i++) {
        unsigned int m = success_list[i].size();
        std::cout << m;
        for (int j = 0; j < m; j++) {
            std::cout << " " << success_list[i][j];
        }
    }
}