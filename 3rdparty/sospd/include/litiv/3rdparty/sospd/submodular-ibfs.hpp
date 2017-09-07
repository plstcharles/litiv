#ifndef _SUBMODULAR_IBFS_HPP_
#define _SUBMODULAR_IBFS_HPP_

/** \file submodular-ibfs.hpp
 *
 * Sum-of-submodular flow using Iterated Breadth First Search
 */

#include "litiv/3rdparty/sospd/sos-graph.hpp"

struct SubmodularIBFSParams {
    enum class FlowAlgorithm {
        bidirectional, source, parametric
    };
    static std::vector<std::pair<FlowAlgorithm, std::string>> algNames;

    SubmodularIBFSParams() { }
    SubmodularIBFSParams(FlowAlgorithm _alg)
        : alg(_alg)
    { }

    FlowAlgorithm alg = FlowAlgorithm::bidirectional;
    SoSGraph::UBfn ub = SoSGraph::UBfn::cvpr14;
    std::vector<bool> fixedVars;
};

class FlowSolver;
/** Algorithm for sum-of-submodular IBFS
 */
class SubmodularIBFS {
    public:
        typedef SoSGraph::NodeId NodeId;

        SubmodularIBFS(SubmodularIBFSParams params = {});
        ~SubmodularIBFS(); // Needed for unique_ptr with incomplete type

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

        SoSGraph& Graph() { return m_graph; }
        const SubmodularIBFSParams& Params() const { return m_params; }
        SubmodularIBFSParams& Params() { return m_params; }
        SoSGraph::NormStats* NormStats() { return &m_normStats; }

    protected:
        /* Graph and energy function definitions */
        SubmodularIBFSParams m_params;
        SoSGraph m_graph;
        REAL m_constant_term = 0;
        std::vector<int> m_labels;
        std::unique_ptr<FlowSolver> m_flowSolver;
        SoSGraph::NormStats m_normStats;

    public:
        REAL GetConstantTerm() const { return m_constant_term; }
        std::vector<int>& GetLabels() { return m_labels; }
        const std::vector<int>& GetLabels() const { return m_labels; }
};


#endif
