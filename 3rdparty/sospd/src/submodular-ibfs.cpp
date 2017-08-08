#include "litiv/3rdparty/sospd/submodular-ibfs.hpp"

#include <chrono>
#include <vector>

#include "litiv/3rdparty/sospd/flow-solver.hpp"

typedef std::chrono::system_clock::time_point TimePt;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::system_clock Clock;


inline std::unique_ptr<FlowSolver> FlowSolver::GetSolver(const SubmodularIBFSParams& params) {
    typedef SubmodularIBFSParams::FlowAlgorithm Alg;
    typedef std::unique_ptr<FlowSolver> FlowPtr;
    switch (params.alg) {
        case Alg::bidirectional:
            return FlowPtr{ new BidirectionalIBFS{} };
        case Alg::source:
            return FlowPtr{ new SourceIBFS{} };
        case Alg::parametric:
            return FlowPtr{ new ParametricIBFS{} };
		default:
			throw std::logic_error("bad solver type");
    }
}

std::vector<std::pair<SubmodularIBFSParams::FlowAlgorithm, std::string>> SubmodularIBFSParams::algNames 
    = { { SubmodularIBFSParams::FlowAlgorithm::bidirectional, "bidirectional" },
        { SubmodularIBFSParams::FlowAlgorithm::source, "source" },
        { SubmodularIBFSParams::FlowAlgorithm::parametric, "parametric" }
    };

SubmodularIBFS::SubmodularIBFS(SubmodularIBFSParams params) 
    : m_params(params),
    m_flowSolver(FlowSolver::GetSolver(params))
{ }

SubmodularIBFS::~SubmodularIBFS() { }

SubmodularIBFS::NodeId SubmodularIBFS::AddNode(int n) {
    for (int i = 0; i < n; ++i)
        m_labels.push_back(-1);
    return m_graph.AddNode(n);
}

int SubmodularIBFS::GetLabel(NodeId n) const {
    return m_labels[n];
}

void SubmodularIBFS::AddUnaryTerm(NodeId n, REAL E0, REAL E1) {
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

void SubmodularIBFS::AddUnaryTerm(NodeId n, REAL coeff) {
    AddUnaryTerm(n, 0, coeff);
}

void SubmodularIBFS::ClearUnaries() {
    m_graph.ClearTerminals();
}

void SubmodularIBFS::AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable) {
    m_graph.AddClique(nodes, energyTable);
}

void SubmodularIBFS::AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11) {
    std::vector<NodeId> nodes{i, j};
    std::vector<REAL> energyTable{E00, E01, E10, E11};
    AddClique(nodes, energyTable);
}

REAL SubmodularIBFS::ComputeEnergy() const {
    return ComputeEnergy(m_labels);
}

REAL SubmodularIBFS::ComputeEnergy(const std::vector<int>& labels) const {
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

void SubmodularIBFS::Solve() {
    m_flowSolver->Solve(this);    
}

