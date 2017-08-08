#ifndef _PARAMETRIC_SUBMODULAR_IBFS_HPP_
#define _PARAMETRIC_SUBMODULAR_IBFS_HPP_

#include "litiv/3rdparty/sospd/energy-common.hpp"
#include <algorithm>

struct ParametricSubmodularIBFS {
    typedef int NodeId;
    struct Clique {
        int Size() const;
        const std::vector<REAL>& AlphaCi() const;
        std::vector<REAL>& EnergyTable();
        const std::vector<REAL>& EnergyTable() const;
    };
    struct CliqueVec {
        Clique& operator[](int n);
        const Clique& operator[](int n) const;
    };
    void AddNode(int n);
    void AddConstantTerm(REAL c);
    void AddUnaryTerm(NodeId i, REAL E0, REAL E1);
    void AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable, bool normalize);
    void GraphInit();
    void ClearUnaries();
    REAL GetConstantTerm();
    CliqueVec& GetCliques();
    
    void Solve();
    int GetLabel(NodeId i);
};


#endif
