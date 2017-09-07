#ifndef _SUBMODULAR_IBFS_HPP_
#define _SUBMODULAR_IBFS_HPP_

/** \file submodular-ibfs.hpp
 *
 * Sum-of-submodular flow using Iterated Breadth First Search
 */

#include "litiv/3rdparty/sospd/sos-graph.hpp"
#include "litiv/3rdparty/sospd/flow-solver.hpp"

/** Algorithm for sum-of-submodular IBFS
 */
template<typename ValueType, typename IndexType/*=int*/>
class SubmodularIBFS {
    public:
        typedef SoSGraph<ValueType,IndexType> GraphType;
        typedef FlowSolver<ValueType,IndexType> FlowSolverType;
        typedef typename GraphType::REAL REAL;
        typedef typename GraphType::NodeId NodeId;

        SubmodularIBFS(const SubmodularIBFSParams& params={});
        ~SubmodularIBFS() = default; // Needed for unique_ptr with incomplete type

        /** Add n new nodes to the base set V
         *
         * \return Index of first created node
         */
        NodeId AddNode(int n = 1);

        /** Get cut label of node
         *
         * \return 1, 0 or -1 if n is in S, not in S, or haven't computed flow
         * yet, respectively
         */
        int GetLabel(NodeId n) const;

        /** Add a constant to the energy function
         */
        void AddConstantTerm(REAL c) { m_constant_term += c; }

        /** AddUnaryTerm for node n, with cost E0 for not being in S and E1
         * for being in S
         */
        void AddUnaryTerm(NodeId n, REAL E0, REAL E1);
        void AddUnaryTerm(NodeId n, REAL coeff);
        void ClearUnaries();

        // Add Clique defined by nodes and energy table given
        void AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable);
        void AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11);

        void Solve();

        // Compute the total energy across all cliques of the current labeling
        REAL ComputeEnergy() const;
        REAL ComputeEnergy(const std::vector<int>& labels) const;

        GraphType& Graph() { return m_graph; }
        const SubmodularIBFSParams& Params() const { return m_params; }
        SubmodularIBFSParams& Params() { return m_params; }
        typename GraphType::NormStats* NormStats() { return &m_normStats; }

    protected:
        /* Graph and energy function definitions */
        SubmodularIBFSParams m_params;
        GraphType m_graph;
        REAL m_constant_term = 0;
        std::vector<int> m_labels;
        std::unique_ptr<FlowSolverType> m_flowSolver;
        typename GraphType::NormStats m_normStats;

    public:
        REAL GetConstantTerm() const { return m_constant_term; }
        std::vector<int>& GetLabels() { return m_labels; }
        const std::vector<int>& GetLabels() const { return m_labels; }
};

template<typename V, typename I>
inline SubmodularIBFS<V,I>::SubmodularIBFS(const SubmodularIBFSParams& params)
        : m_params(params),
          m_flowSolver(std::move(sospd::GetSolver<V,I>(params)))
{ }

template<typename V, typename I>
inline typename SubmodularIBFS<V,I>::NodeId SubmodularIBFS<V,I>::AddNode(int n) {
    for (int i = 0; i < n; ++i)
        m_labels.push_back(-1);
    return m_graph.AddNode(n);
}

template<typename V, typename I>
inline int SubmodularIBFS<V,I>::GetLabel(NodeId n) const {
    return m_labels[n];
}

template<typename V, typename I>
inline void SubmodularIBFS<V,I>::AddUnaryTerm(NodeId n, REAL E0, REAL E1) {
    // Reparametize so that E0, E1 >= 0
    if (E0 < 0) {
        AddConstantTerm(E0);
        E1 -= E0;
        E0 = 0;
    }
    if (E1 < 0) {
        AddConstantTerm(E1);
        E0 -= E1;
        E1 = 0;
    }
    // FIXME: Shouldn't it be the other way around (E1, E0)?
    m_graph.AddTerminalWeights(n, E0, E1);
}

template<typename V, typename I>
inline void SubmodularIBFS<V,I>::AddUnaryTerm(NodeId n, REAL coeff) {
    AddUnaryTerm(n, 0, coeff);
}

template<typename V, typename I>
inline void SubmodularIBFS<V,I>::ClearUnaries() {
    m_graph.ClearTerminals();
}

template<typename V, typename I>
inline void SubmodularIBFS<V,I>::AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable) {
    m_graph.AddClique(nodes, energyTable);
}

template<typename V, typename I>
inline void SubmodularIBFS<V,I>::AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11) {
    std::vector<NodeId> nodes{i, j};
    std::vector<REAL> energyTable{E00, E01, E10, E11};
    AddClique(nodes, energyTable);
}

template<typename V, typename I>
inline typename SubmodularIBFS<V,I>::REAL SubmodularIBFS<V,I>::ComputeEnergy() const {
    return ComputeEnergy(m_labels);
}

template<typename V, typename I>
inline typename SubmodularIBFS<V,I>::REAL SubmodularIBFS<V,I>::ComputeEnergy(const std::vector<int>& labels) const {
    // FIXME: Change to actually store the original unaries, since optimization
    // might change them.
    REAL total = m_constant_term;
    for (NodeId i = 0; i < m_graph.NumNodes(); ++i) {
        if (labels[i] == 1) total += m_graph.m_c_it[i];
        else total += m_graph.m_c_si[i];
    }
    for (const auto& c : m_graph.m_cliques) {
        total += c.ComputeEnergy(labels);
    }
    return total;
}

template<typename V, typename I>
inline void SubmodularIBFS<V,I>::Solve() {
    m_flowSolver->Solve(this);
}

#endif
