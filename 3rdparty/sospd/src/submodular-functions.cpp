#include "litiv/3rdparty/sospd/submodular-functions.hpp"

#include <cmath>
#include <iostream>
#include <memory>

#include "litiv/3rdparty/sospd/sos-graph.hpp"

const std::vector<SoSGraph::UBParam> SoSGraph::ubParamList = 
    { UBParam{ UBfn::chen, "chen", ChenUpperBound },
      UBParam{ UBfn::cvpr14, "cvpr14", UpperBoundCVPR14 },
    };


double DiffL1(const std::vector<REAL>& e1, const std::vector<REAL>& e2) {
    int n = e1.size();
    double norm = 0;
    for (int i = 0; i < n; ++i)
        norm += std::abs(static_cast<double>(e1[i] - e2[i]));
    return norm;
}

double DiffL2(const std::vector<REAL>& e1, const std::vector<REAL>& e2) {
    int n = e1.size();
    double norm = 0;
    for (int i = 0; i < n; ++i) {
        double diff = std::abs(static_cast<double>(e1[i] - e2[i]));
        norm += diff*diff;
    }
    return norm;
}

double DiffLInfty(const std::vector<REAL>& e1, const std::vector<REAL>& e2) {
    int n = e1.size();
    double norm = 0;
    for (int i = 0; i < n; ++i)
        norm = std::max(norm, std::abs(static_cast<double>(e1[i] - e2[i])));
    return norm;
}

