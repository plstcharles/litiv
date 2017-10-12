#pragma once

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/intrusive/options.hpp>
#include "litiv/3rdparty/sospd/submodular-functions.hpp"

namespace sospd {

    enum class UBfn {
        chen,
        cvpr14,
    };

    enum class FlowAlgorithm {
        bidirectional,
        source,
        parametric
    };

    struct SubmodularIBFSParams {
        SubmodularIBFSParams() = default;
        SubmodularIBFSParams(sospd::FlowAlgorithm _alg) : alg(_alg) {}
        sospd::FlowAlgorithm alg = sospd::FlowAlgorithm::bidirectional;
        sospd::UBfn ub = sospd::UBfn::cvpr14;
        std::vector<bool> fixedVars;
    };

    template<typename ValueType, typename IndexType>
    class SubmodularIBFS;

    template<typename ValueType, typename IndexType>
    class FlowSolver {
    public:
        FlowSolver() = default;
        virtual ~FlowSolver() = default;
        virtual void Solve(SubmodularIBFS<ValueType,IndexType>* energy) = 0;
        FlowSolver(const FlowSolver&) = delete;
        FlowSolver(FlowSolver&&) = delete;
        FlowSolver& operator=(const FlowSolver&) = delete;
        FlowSolver& operator=(FlowSolver&&) = delete;
    };

    template<typename ValueType, typename IndexType>
    std::unique_ptr<FlowSolver<ValueType,IndexType>> GetSolver(const SubmodularIBFSParams& params);

    /** Graph structure and algorithm for sum-of-submodular IBFS
     */
    template<typename ValueType, typename IndexType>
    class SoSGraph {
        static_assert(std::is_arithmetic<ValueType>::value,"value type must be arithmetic");
        static_assert(std::is_integral<IndexType>::value,"index type must be integral");
        public:
            typedef ValueType REAL;
            typedef IndexType NodeId;
            typedef IndexType CliqueId;
            typedef std::vector<CliqueId> NeighborList;
            enum class NodeState : char {
                S, T, S_orphan, T_orphan, N
            };
            class IBFSEnergyTableClique;
            typedef std::tuple<sospd::UBfn,std::string,sospd::UpperBoundFunction<ValueType,IndexType>> UBParam;
            static const std::vector<UBParam> ubParamList;

            SoSGraph()
                : m_num_nodes(0),
                s(NodeId(-1)),
                t(NodeId(-1)),
                m_num_cliques(0)
            { }

            /** Add n new nodes to the base set V
             *
             * \return Index of first created node
             */
            NodeId AddNode(IndexType n = 1);

            /** Add weights to s-i and i-t edges, respectively
             */
            void AddTerminalWeights(NodeId n, REAL sCap, REAL tCap);

            /** Zero the capacities on the s-i and i-t edges
             */
            void ClearTerminals();

            // Add Clique defined by nodes and energy table given
            IBFSEnergyTableClique& AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable);

            /* Clique: abstract base class for user-defined clique functions
             *
             * Clique stores the list of nodes associated with a clique.
             * Actual functionality is provided by the user writing a derived
             * class with Clique as the base, and which implements the
             * ComputeEnergy and ExchangeCapacity functions
             */
            class Clique {
                public:
                typedef std::vector<NodeId> NodeVec;
                Clique() : m_nodes(), m_alpha_Ci() { }
                Clique(const NodeVec& nodes)
                    : m_nodes(nodes),
                    m_alpha_Ci(nodes.size(), 0)
                { }
                ~Clique() = default;

                // Returns the energy of the given labeling for this clique function
                virtual REAL ComputeEnergy(const std::vector<int>& labels) const = 0;

                const NodeVec& Nodes() const { return m_nodes; }
                IndexType Size() const { return IndexType(m_nodes.size()); }
                std::vector<REAL>& AlphaCi() { return m_alpha_Ci; }
                const std::vector<REAL>& AlphaCi() const { return m_alpha_Ci; }
                IndexType GetIndex(NodeId i) const {
                    return IndexType(std::find(this->m_nodes.begin(), this->m_nodes.end(), i) - this->m_nodes.begin());
                }

                protected:
                NodeVec m_nodes; // The list of nodes in the clique
                std::vector<REAL> m_alpha_Ci; // The reparameterization variables for this clique

            };
            /*
             * IBFSEnergyTableClique: stores energy as a list of 2^k values for each subset
             */
            class IBFSEnergyTableClique : public Clique {
                public:
                    typedef uint32_t Assignment;

                    IBFSEnergyTableClique() : Clique(), m_energy(), m_alpha_energy(), m_min_tight_set() { }
                    IBFSEnergyTableClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energy)
                        : Clique(nodes),
                        m_energy(energy),
                        m_alpha_energy(energy),
                        m_min_tight_set(nodes.size(), (1u << nodes.size()) - 1)
                    {
                        ASSERT(nodes.size() <= 31);
                    }

                    virtual REAL ComputeEnergy(const std::vector<int>& labels) const;
                    REAL ComputeAlphaEnergy(const std::vector<int>& labels) const;
                    REAL ExchangeCapacity(IndexType u_idx, IndexType v_idx) const;
                    bool NonzeroCapacity(IndexType u_idx, IndexType v_idx) const;
                    void NormalizeEnergy(std::vector<REAL>& psi, REAL& constantTerm);

                    void Push(IndexType u_idx, IndexType v_idx, REAL delta);
                    void ComputeMinTightSets();
                    std::vector<REAL>& EnergyTable() { return m_energy; }
                    const std::vector<REAL>& EnergyTable() const { return m_energy; }
                    std::vector<REAL>& AlphaEnergy() { return m_alpha_energy; }
                    const std::vector<REAL>& AlphaEnergy() const { return m_alpha_energy; }

                    void ResetAlpha();

                protected:
                    std::vector<REAL> m_energy;
                    std::vector<REAL> m_alpha_energy;
                    std::vector<Assignment> m_min_tight_set;

            };
            struct ArcIterator {
                NodeId source;
                typename NeighborList::iterator cIter;
                IndexType cliqueIdx;
                IndexType cliqueSize;
                SoSGraph* graph;

                bool operator!=(const ArcIterator& a) {
                    return (cIter != a.cIter) || (cliqueIdx != a.cliqueIdx);
                }
                bool operator==(const ArcIterator& a) {
                    return !(*this != a);
                }
                bool operator<(const ArcIterator& a) {
                    ASSERT(source == a.source);
                    return (cIter == a.cIter) ? (source < a.source) : (cIter < a.cIter);
                }

                ArcIterator& operator++() {
                    //ASSERT(*cIter < static_cast<int>(graph->m_cliques.size()));
                    cliqueIdx++;
                    if (cliqueIdx == cliqueSize) {
                        cliqueIdx = IndexType(0);
                        cIter++;
                        if (cIter != graph->m_neighbors[source].end())
                            cliqueSize = graph->m_cliques[*cIter].Size();
                        else
                            cliqueSize = IndexType(0);
                    }
                    //ASSERT(cIter == graph->m_neighbors[source].end() || *cIter < static_cast<int>(graph->m_cliques.size()));
                    //ASSERT(cIter == graph->m_neighbors[source].end() || cliqueIdx < static_cast<int>(graph->m_cliques[*cIter].Nodes().size()));
                    return *this;
                }
                NodeId Source() const {
                    return source;
                }
                NodeId Target() const {
                    //ASSERT(*cIter < static_cast<int>(graph->m_cliques.size()));
                    //ASSERT(cliqueIdx < static_cast<int>(graph->m_cliques[*cIter].Nodes().size()));
                    return graph->m_cliques[*cIter].Nodes()[cliqueIdx];
                }
                IndexType SourceIdx() const { return graph->m_cliques[*cIter].GetIndex(source); }
                IndexType TargetIdx() const { return cliqueIdx; }
                CliqueId cliqueId() const { return *cIter; }
                ArcIterator Reverse() const {
                    auto newSource = Target();
                    auto newCIter = std::find(graph->m_neighbors[newSource].begin(), graph->m_neighbors[newSource].end(), *cIter);
                    auto newCliqueIdx = graph->GetCliques()[*newCIter].GetIndex(source);
                    return {newSource, newCIter, newCliqueIdx, graph->m_cliques[*newCIter].Size(), graph};
                }
            };

            typedef boost::intrusive::list_base_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>> ListHook;
            typedef boost::intrusive::slist_base_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>> OrphanListHook;
            struct Node : public ListHook, OrphanListHook {
                NodeId id;
                NodeState state;
                IndexType dis;
                ArcIterator parent_arc;
                NodeId parent;
                NeighborList cliques;
                explicit Node(NodeId _id)
                    : id(_id)
                    , state(NodeState::N)
                    , dis(std::numeric_limits<IndexType>::max())
                    , parent_arc()
                    , parent()
                    , cliques() { }
            };

            typedef boost::intrusive::list<Node> NodeQueue;
            typedef boost::intrusive::slist<Node, boost::intrusive::base_hook<OrphanListHook>, boost::intrusive::cache_last<true>> OrphanList;

            ArcIterator ArcsBegin(NodeId i) {
                auto cIter = m_neighbors[i].begin();
                if (cIter == m_neighbors[i].end())
                    return ArcsEnd(i);
                return {i, cIter, 0, m_cliques[*cIter].Size(), this};
            }
            ArcIterator ArcsEnd(NodeId i) {
                auto& neighborList = m_neighbors[i];
                return {i, neighborList.end(), 0, 0, this};
            }

            typedef std::vector<IBFSEnergyTableClique> CliqueVec;

            NodeId NumNodes() const { return m_num_nodes; }
            NodeId GetS() const { return s; }
            NodeId GetT() const { return t; }
            Node& node(NodeId i) { return m_nodes[i]; }
            const Node& node(NodeId i) const { return m_nodes[i]; }
            IBFSEnergyTableClique& clique(CliqueId c) { return m_cliques[c]; }
            const IBFSEnergyTableClique& clique(CliqueId c) const { return m_cliques[c]; }
            const std::vector<REAL>& GetC_si() const { return m_c_si; }
            const std::vector<REAL>& GetC_it() const { return m_c_it; }
            const std::vector<REAL>& GetPhi_si() const { return m_phi_si; }
            const std::vector<REAL>& GetPhi_it() const { return m_phi_it; }
            CliqueId GetNumCliques() const { return m_num_cliques; }
            const CliqueVec& GetCliques() const { return m_cliques; }
            CliqueVec& GetCliques() { return m_cliques; }
            const std::vector<NeighborList>& GetNeighbors() const { return m_neighbors; }
            std::vector<Node>& GetNodes() { return m_nodes; }
            const std::vector<Node>& GetNodes() const { return m_nodes; }

            REAL ResCap(const ArcIterator& arc, bool forwardArc);
            bool NonzeroCap(const ArcIterator& arc, bool forwardArc);
            void Push(ArcIterator& arc, bool forwardArc, REAL delta);

            void ResetFlow();
            typedef void(*BoundFn)(int, const std::vector<REAL>&, std::vector<REAL>&);
            struct NormStats {
                double L1 = 0;
                double L2 = 0;
                double LInfty = 0;
            };
            template<BoundFn fn>
            void UpperBoundCliques(const std::vector<bool>& fixedVars, NormStats* stats);
            void UpperBoundCliques(sospd::UBfn ub, NormStats* stats = 0);
            void UpperBoundCliques(sospd::UBfn ub, const std::vector<bool>& fixedVars, const std::vector<int>& labels, NormStats* stats = 0);

            NodeId m_num_nodes;
            NodeId s,t;
            std::vector<REAL> m_c_si;
            std::vector<REAL> m_c_it;
            std::vector<REAL> m_phi_si;
            std::vector<REAL> m_phi_it;

            CliqueId m_num_cliques;
            CliqueVec m_cliques;
            std::vector<NeighborList> m_neighbors;

        protected:
            std::vector<Node> m_nodes;
    };

} // namespace sospd

template<typename V, typename I>
const std::vector<typename sospd::SoSGraph<V,I>::UBParam> sospd::SoSGraph<V,I>::ubParamList = {
    UBParam{ sospd::UBfn::chen, "chen", sospd::ChenUpperBound<REAL> },
    UBParam{ sospd::UBfn::cvpr14, "cvpr14", sospd::UpperBoundCVPR14<REAL> },
};

template<typename V, typename I>
inline typename sospd::SoSGraph<V,I>::NodeId sospd::SoSGraph<V,I>::AddNode(I n) {
    ASSERT(n >= I(1));
    ASSERT(s == I(-1));
    NodeId first_node = m_num_nodes;
    for(I i = 0; i < n; ++i) {
        m_nodes.push_back(Node(m_num_nodes));
        m_c_si.push_back(0);
        m_c_it.push_back(0);
        m_phi_si.push_back(0);
        m_phi_it.push_back(0);
        m_neighbors.push_back(NeighborList());
        m_num_nodes++;
    }
    return first_node;
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::AddTerminalWeights(NodeId n, REAL sCap, REAL tCap) {
    m_c_si[n] += sCap;
    m_c_it[n] += tCap;
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::ClearTerminals() {
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        m_c_si[i] = m_c_it[i] = 0;
        m_phi_si[i] = m_phi_it[i] = 0;
    }
}

template<typename V, typename I>
inline typename sospd::SoSGraph<V,I>::IBFSEnergyTableClique& sospd::SoSGraph<V,I>::AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable) {
    ASSERT(s == I(-1));
    m_cliques.emplace_back(nodes, energyTable);
    for (NodeId i : nodes) {
        ASSERT(0 <= i && i < m_num_nodes);
        m_neighbors[i].push_back(m_num_cliques);
    }
    return m_cliques[m_num_cliques++];
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::ResetFlow() {
    // Initialize source, sink (only do once)
    if (s == I(-1)) {
        s = m_num_nodes; t = m_num_nodes + 1;
        m_nodes.push_back(Node(s));
        m_nodes.push_back(Node(t));
    }
    // reset distance, state and parent
    for (I i = 0; i < m_num_nodes + 2; ++i) {
        Node& node = m_nodes[i];
        node.dis = std::numeric_limits<I>::max();
        node.state = NodeState::N;
        node.parent = i;
        m_phi_si[i] = m_phi_it[i] = 0;
    }

    // Reset Clique parameters
    for (I cid = 0; cid < m_num_cliques; ++cid) {
        auto& c = m_cliques[cid];
        c.ResetAlpha();
        c.ComputeMinTightSets();
    }

}

template<typename V, typename I>
inline typename sospd::SoSGraph<V,I>::REAL sospd::SoSGraph<V,I>::ResCap(const ArcIterator& arc, bool forwardArc) {
    ASSERT(arc.cliqueId() >= 0 && arc.cliqueId() < I(m_cliques.size()));
    if (forwardArc)
        return m_cliques[arc.cliqueId()].ExchangeCapacity(arc.SourceIdx(), arc.TargetIdx());
    else
        return m_cliques[arc.cliqueId()].ExchangeCapacity(arc.TargetIdx(), arc.SourceIdx());
}

template<typename V, typename I>
inline bool sospd::SoSGraph<V,I>::NonzeroCap(const ArcIterator& arc, bool forwardArc) {
    if (forwardArc)
        return m_cliques[arc.cliqueId()].NonzeroCapacity(arc.SourceIdx(), arc.TargetIdx());
    else
        return m_cliques[arc.cliqueId()].NonzeroCapacity(arc.TargetIdx(), arc.SourceIdx());
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::IBFSEnergyTableClique::NormalizeEnergy(std::vector<REAL>& psi, REAL& constantTerm) {
    ASSERT(false /* Should not be calling this function*/);
    const I n = I(this->m_nodes.size());
    ASSERT(sospd::CheckSubmodular((int)n,m_energy));
    const Assignment num_assignments = 1u << n;
    REAL allOnes = m_energy[num_assignments - 1];
    constantTerm += allOnes;
    psi.resize(n);
    Assignment assgn = num_assignments - 1; // The all 1 assignment
    for (I i = 0; i < n; ++i) {
        Assignment next_assgn = assgn ^ (1u << i);
        psi[i] = (m_energy[assgn] - m_energy[next_assgn]);
        assgn = next_assgn;
    }

    for (Assignment a = 0; a < num_assignments; ++a) {
        m_energy[a] -= allOnes;
        for (I i = 0; i < n; ++i) {
            if (!(a & (1u << i))) m_energy[a] += psi[i];
        }
        ASSERT(m_energy[a] >= V(0));
        m_alpha_energy[a] = m_energy[a];
    }
    ComputeMinTightSets();
    ASSERT(sospd::CheckSubmodular((int)n,m_energy));
}

template<typename V, typename I>
inline typename sospd::SoSGraph<V,I>::REAL sospd::SoSGraph<V,I>::IBFSEnergyTableClique::ComputeEnergy(const std::vector<int>& labels) const {
    Assignment assgn = 0;
    for (I i = 0; i < I(this->m_nodes.size()); ++i) {
        NodeId n = this->m_nodes[i];
        if (labels[n] == 1) {
            assgn |= 1u << i;
        }
    }
    return m_energy[assgn];
}

template<typename V, typename I>
inline typename sospd::SoSGraph<V,I>::REAL sospd::SoSGraph<V,I>::IBFSEnergyTableClique::ComputeAlphaEnergy(const std::vector<int>& labels) const {
    Assignment assgn = 0;
    for (I i = 0; i < I(this->m_nodes.size()); ++i) {
        NodeId n = this->m_nodes[i];
        if (labels[n] == 1) {
            assgn |= 1u << i;
        }
    }
    return m_alpha_energy[assgn];
}

template<typename V, typename I>
inline typename sospd::SoSGraph<V,I>::REAL sospd::SoSGraph<V,I>::IBFSEnergyTableClique::ExchangeCapacity(I u_idx, I v_idx) const {
    const I n = I(this->m_nodes.size());
    ASSERT(u_idx < n);
    ASSERT(v_idx < n);
    REAL min_energy = std::numeric_limits<REAL>::max();
    Assignment num_assgns = 1u << n;
    const Assignment bound = num_assgns-1;
    const Assignment u_mask = 1u << u_idx;
    const Assignment v_mask = 1u << v_idx;
    const Assignment uv_mask = u_mask | v_mask;
    const Assignment subset_mask = bound & ~uv_mask;
    // Terrible bit-hacks to optimize the living hell out of this function
    // Iterate over all assignments without u_idx or v_idx set
    Assignment assgn = subset_mask;
    do {
        Assignment u_sep = assgn | u_mask;
        REAL energy = m_alpha_energy[u_sep];
        if (energy < min_energy) min_energy = energy;
        assgn = ((assgn - 1) & subset_mask);
    } while (assgn != subset_mask);

    return min_energy;
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::IBFSEnergyTableClique::Push(I u_idx, I v_idx, REAL delta) {
    ASSERT(u_idx < this->m_nodes.size());
    ASSERT(v_idx < this->m_nodes.size());
    Clique::m_alpha_Ci[u_idx] += delta;
    Clique::m_alpha_Ci[v_idx] -= delta;
    const I n = I(this->m_nodes.size());
    Assignment num_assgns = 1u << n;
    const Assignment bound = num_assgns-1;
    const Assignment u_mask = 1u << u_idx;
    const Assignment v_mask = 1u << v_idx;
    const Assignment uv_mask = u_mask | v_mask;
    const Assignment subset_mask = bound & ~uv_mask;
    // Terrible bit-hacks to optimize the living hell out of this function
    // Iterate over all assignments without u_idx or v_idx set
    Assignment assgn = subset_mask;
    do {
        Assignment u_sep = assgn | u_mask;
        Assignment v_sep = assgn | v_mask;
        m_alpha_energy[u_sep] -= delta;
        m_alpha_energy[v_sep] += delta;
        assgn = ((assgn - 1) & subset_mask);
    } while (assgn != subset_mask);

    ComputeMinTightSets();
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::IBFSEnergyTableClique::ComputeMinTightSets() {
    I n = I(this->m_nodes.size());
    Assignment num_assgns = 1u << n;
    const Assignment bound = num_assgns-1;
    for (auto& a : m_min_tight_set)
        a = bound;
    for (Assignment assgn = bound-1; assgn >= 1; --assgn) {
        if (m_alpha_energy[assgn] == 0) {
            for (I i = 0; i < n; ++i) {
                //ASSERT(m_alpha_energy[m_min_tight_set[i] & assgn] == 0);
                //ASSERT(m_alpha_energy[m_min_tight_set[i] | assgn] == 0);
                if ((assgn & (1u << i)) != 0)
                    m_min_tight_set[i] = assgn;
            }
        }
    }
}

template<typename V, typename I>
inline bool sospd::SoSGraph<V,I>::IBFSEnergyTableClique::NonzeroCapacity(I u_idx, I v_idx) const {
    Assignment min_set = m_min_tight_set[u_idx];
    return (min_set & (1u << v_idx)) != 0;
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::IBFSEnergyTableClique::ResetAlpha() {
    for (auto& a : this->m_alpha_Ci) {
        a = 0;
    }
    const I n = I(this->m_nodes.size());
    const Assignment num_assignments = 1u << n;
    for (Assignment a = 0; a < num_assignments; ++a) {
        m_alpha_energy[a] = m_energy[a];
    }
}

template<typename V, typename I>
template<typename sospd::SoSGraph<V,I>::BoundFn UB>
inline void sospd::SoSGraph<V,I>::UpperBoundCliques(const std::vector<bool>& fixedVars, NormStats* stats) {
    std::vector<REAL> psi;
    //int nCliques = m_cliques.size();
    I cliquesDone = 0;
    /*
     *std::cout << "Upper Bounding Cliques: ";
     *std::cout.flush();
     */
    for (auto& c : m_cliques) {
        /*
         *if (cliquesDone % (nCliques/10) == 0) {
         *    std::cout << ".";
         *    std::cout.flush();
         *}
         */
        cliquesDone++;
        auto& newEnergy = c.AlphaEnergy();
        I k = I(c.Size());
        psi.resize(k);
        // Compute upper bound g of clique energy
        UB(k, c.EnergyTable(), newEnergy);

        if (!fixedVars.empty()) {
            I fixedSet = 0;
            for (I i = 0; i < k; ++i)
                fixedSet |= (fixedVars[c.Nodes()[i]] << i);
            sospd::ZeroMarginalSet(k, newEnergy, fixedSet);
        }

        if (stats) {
            stats->L1 += sospd::DiffL1(c.EnergyTable(), newEnergy);
            stats->L2 += sospd::DiffL2(c.EnergyTable(), newEnergy);
            stats->LInfty += sospd::DiffLInfty(c.EnergyTable(), newEnergy);
        }
        // Modify g, find psi so that g'(S) = g(S) + psi(S) >= 0
        sospd::Normalize(k, newEnergy, psi);
        /*
         *AddLinear(k, c.EnergyTable(), psi);
         */

        auto& alpha_Ci = c.AlphaCi();
        for (I i = 0; i < k; ++i) {
            alpha_Ci[i] = -psi[i];
            m_phi_it[c.Nodes()[i]] += psi[i];
        }
        c.ComputeMinTightSets();
    }
    /*
     *std::cout << "\n";
     *std::cout << "L1: " << diffL1 << "\tL2: " << diffL2 << "\tLInfty: " << diffLInfty << "\n";
     */
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::UpperBoundCliques(sospd::UBfn ub, NormStats* stats) {
    UpperBoundCliques(ub, std::vector<bool>{}, std::vector<int>{}, stats);
}

template<typename V, typename I>
inline void sospd::SoSGraph<V,I>::UpperBoundCliques(sospd::UBfn ub, const std::vector<bool>& fixedVars, const std::vector<int>& /*labels*/, NormStats* stats) {
    switch(ub) {
        case sospd::UBfn::chen:
            UpperBoundCliques<sospd::ChenUpperBound>(fixedVars, stats);
            break;
        case sospd::UBfn::cvpr14:
            UpperBoundCliques<sospd::UpperBoundCVPR14>(fixedVars, stats);
            break;
    }
}
