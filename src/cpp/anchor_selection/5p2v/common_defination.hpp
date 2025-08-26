#include <iostream>
#include <vector>
#ifndef _HOMOTOPY_DEF_
#define _HOMOTOPY_DEF_
#include "homotopy.hxx"
#endif

#define Float double
typedef unsigned char ind;
static constexpr Float tol = 1e-3;
void split_instances(std::vector<std::vector<Float>> &instances,
                     std::vector<std::vector<Float>> &problems,
                     std::vector<std::vector<Float>> &start,
                     std::vector<std::vector<Float>> &depths) {
    int n = instances.size();

    //load the problems
    for (int i = 0; i < n; i++) {
        std::vector<Float> problem(20);
        std::vector<Float> cst(9);
        std::vector<Float> depth(10);

        //load the points
        for (int j = 0; j < 20; j++) {
            problem[j] = instances[i][j];
        }
        problems[i] = problem;

        //load the depths and convert them to the solution
        depth[0] = instances[i][20];
        for (int j = 0; j < 9; j++) {
            depth[j + 1] = instances[i][j + 21];
            cst[j] = depth[j + 1] / depth[0];
        }

        start[i] = cst;
        depths[i] = depth;
    }
}
