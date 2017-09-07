#ifndef _SOS_GRAPH_HPP_
#define _SOS_GRAPH_HPP_

/** \file sos-graph.hpp
 * Graph class for sum-of-submodular flow algorithms
 */

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/slist.hpp>
#include <boost/intrusive/options.hpp>
#include "litiv/3rdparty/sospd/submodular-functions.hpp"

/** Graph structure and algorithm for sum-of-submodular IBFS
 */
class SoSGraph {
    public:
        typedef int NodeId;
        typedef int CliqueId;
        typedef std::vector<CliqueId> NeighborList;
        enum class NodeState : char {
            S, T, S_orphan, T_orphan, N
        };
        class IBFSEnergyTableClique;
        enum class UBfn {
            chen,
            cvpr14,
        };
        typedef std::tuple<UBfn, std::string, UpperBoundFunction> UBParam;
        static const std::vector<UBParam> ubParamList;

        SoSGraph()
            : m_num_nodes(0),
            s(-1),
            t(-1),
            m_num_cliques(0)
        { }

        /** Add n new nodes to the base set V
         *
         * \return Index of first created node
         */
        NodeId AddNode(int n = 1);

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
            size_t Size() const { return m_nodes.size(); }
            std::vector<REAL>& AlphaCi() { return m_alpha_Ci; }
            const std::vector<REAL>& AlphaCi() const { return m_alpha_Ci; }
            size_t GetIndex(NodeId i) const {
                return std::find(this->m_nodes.begin(), this->m_nodes.end(), i) - this->m_nodes.begin();
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
                IBFSEnergyTableClique(const std::vector<NodeId>& nodes,
                                  const std::vector<REAL>& energy)
                    : Clique(nodes),
                    m_energy(energy),
                    m_alpha_energy(energy),
                    m_min_tight_set(nodes.size(), (1 << nodes.size()) - 1)
                {
                    ASSERT(nodes.size() <= 31);
                }

                virtual REAL ComputeEnergy(const std::vector<int>& labels) const;
                REAL ComputeAlphaEnergy(const std::vector<int>& labels) const;
                REAL ExchangeCapacity(size_t u_idx, size_t v_idx) const;
                bool NonzeroCapacity(size_t u_idx, size_t v_idx) const;
                void NormalizeEnergy(std::vector<REAL>& psi, REAL& constantTerm);

                void Push(size_t u_idx, size_t v_idx, REAL delta);
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
            NeighborList::iterator cIter;
            int cliqueIdx;
            int cliqueSize;
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
                    cliqueIdx = 0;
                    cIter++;
                    if (cIter != graph->m_neighbors[source].end())
                        cliqueSize = graph->m_cliques[*cIter].Size();
                    else
                        cliqueSize = 0;
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
            int SourceIdx() const { return graph->m_cliques[*cIter].GetIndex(source); }
            int TargetIdx() const { return cliqueIdx; }
            CliqueId cliqueId() const { return *cIter; }
            ArcIterator Reverse() const {
                auto newSource = Target();
                auto newCIter = std::find(graph->m_neighbors[newSource].begin(), graph->m_neighbors[newSource].end(), *cIter);
                auto newCliqueIdx = graph->GetCliques()[*newCIter].GetIndex(source);
                return {newSource, newCIter, static_cast<int>(newCliqueIdx), static_cast<int>(graph->m_cliques[*newCIter].Size()), graph};
            }
        };

        typedef boost::intrusive::list_base_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>> ListHook;
        typedef boost::intrusive::slist_base_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>> OrphanListHook;
        struct Node : public ListHook, OrphanListHook {
            NodeId id;
            NodeState state;
            int dis;
            ArcIterator parent_arc;
            NodeId parent;
            NeighborList cliques;
            Node(NodeId _id)
                : id(_id)
                , state(NodeState::N)
                , dis(std::numeric_limits<int>::max())
                , parent_arc()
                , cliques() { }
        };

        typedef boost::intrusive::list<Node> NodeQueue;
        typedef boost::intrusive::slist<Node, boost::intrusive::base_hook<OrphanListHook>, boost::intrusive::cache_last<true>> OrphanList;

        ArcIterator ArcsBegin(NodeId i) {
            auto cIter = m_neighbors[i].begin();
            if (cIter == m_neighbors[i].end())
                return ArcsEnd(i);
            return {i, cIter, 0, static_cast<int>(m_cliques[*cIter].Size()), this};
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
        template <BoundFn fn>
        void UpperBoundCliques(const std::vector<bool>& fixedVars, NormStats* stats);
        void UpperBoundCliques(UBfn ub, NormStats* stats = 0);
        void UpperBoundCliques(UBfn ub, const std::vector<bool>& fixedVars, const std::vector<int>& labels, NormStats* stats = 0);

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

inline SoSGraph::NodeId SoSGraph::AddNode(int n) {
    ASSERT(n >= 1);
    ASSERT(s == -1);
    NodeId first_node = m_num_nodes;
    for (int i = 0; i < n; ++i) {
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

inline void SoSGraph::AddTerminalWeights(NodeId n, REAL sCap, REAL tCap) {
    m_c_si[n] += sCap;
    m_c_it[n] += tCap;
}

inline void SoSGraph::ClearTerminals() {
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        m_c_si[i] = m_c_it[i] = 0;
        m_phi_si[i] = m_phi_it[i] = 0;
    }
}

inline SoSGraph::IBFSEnergyTableClique& SoSGraph::AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable) {
    ASSERT(s == -1);
    m_cliques.emplace_back(nodes, energyTable);
    for (NodeId i : nodes) {
        ASSERT(0 <= i && i < m_num_nodes);
        m_neighbors[i].push_back(m_num_cliques);
    }
    return m_cliques[m_num_cliques++];
}

inline void SoSGraph::ResetFlow() {
    // Initialize source, sink (only do once)
    if (s == -1) {
        s = m_num_nodes; t = m_num_nodes + 1;
        m_nodes.push_back(Node(s));
        m_nodes.push_back(Node(t));
    }
    // reset distance, state and parent
    for (int i = 0; i < m_num_nodes + 2; ++i) {
        Node& node = m_nodes[i];
        node.dis = std::numeric_limits<int>::max();
        node.state = NodeState::N;
        node.parent = i;
        m_phi_si[i] = m_phi_it[i] = 0;
    }

    // Reset Clique parameters
    for (int cid = 0; cid < m_num_cliques; ++cid) {
        auto& c = m_cliques[cid];
        c.ResetAlpha();
        c.ComputeMinTightSets();
    }

}

inline REAL SoSGraph::ResCap(const ArcIterator& arc, bool forwardArc) {
    ASSERT(arc.cliqueId() >= 0 && arc.cliqueId() < static_cast<int>(m_cliques.size()));
    if (forwardArc)
        return m_cliques[arc.cliqueId()].ExchangeCapacity(arc.SourceIdx(), arc.TargetIdx());
    else
        return m_cliques[arc.cliqueId()].ExchangeCapacity(arc.TargetIdx(), arc.SourceIdx());
}

inline bool SoSGraph::NonzeroCap(const ArcIterator& arc, bool forwardArc) {
    if (forwardArc)
        return m_cliques[arc.cliqueId()].NonzeroCapacity(arc.SourceIdx(), arc.TargetIdx());
    else
        return m_cliques[arc.cliqueId()].NonzeroCapacity(arc.TargetIdx(), arc.SourceIdx());
}

inline void CheckSubmodular(size_t n, const std::vector<REAL>& m_energy) {
    typedef int32_t Assignment;
    Assignment max_assgn = 1 << n;
    for (Assignment s = 0; s < max_assgn; ++s) {
        for (size_t i = 0; i < n; ++i) {
            Assignment si = s | (1 << i);
            if (si != s) {
                for (size_t j = i+1; j < n; ++j) {
                    Assignment t = s | (1 << j);
                    if (t != s && j != i) {
                        Assignment ti = t | (1 << i);
                        // Decreasing marginal costs, so we require
                        // f(ti) - f(t) <= f(si) - f(s)
                        // i.e. f(si) - f(s) - f(ti) + f(t) >= 0
                        REAL violation = -m_energy[si] - m_energy[t]
                            + m_energy[s] + m_energy[ti];
                        //if (violation > 0) std::cout << violation << std::endl;
                        ASSERT(violation <= 0);
                    }
                }
            }
        }
    }
}

inline void SoSGraph::IBFSEnergyTableClique::NormalizeEnergy(std::vector<REAL>& psi, REAL& constantTerm) {
    ASSERT(false /* Should not be calling this function*/);
    const size_t n = this->m_nodes.size();
    CheckSubmodular(n, m_energy);
    const Assignment num_assignments = 1 << n;
    REAL allOnes = m_energy[num_assignments - 1];
    constantTerm += allOnes;
    psi.resize(n);
    Assignment assgn = num_assignments - 1; // The all 1 assignment
    for (size_t i = 0; i < n; ++i) {
        Assignment next_assgn = assgn ^ (1 << i);
        psi[i] = (m_energy[assgn] - m_energy[next_assgn]);
        assgn = next_assgn;
    }

    for (Assignment a = 0; a < num_assignments; ++a) {
        m_energy[a] -= allOnes;
        for (size_t i = 0; i < n; ++i) {
            if (!(a & (1 << i))) m_energy[a] += psi[i];
        }
        ASSERT(m_energy[a] >= 0);
        m_alpha_energy[a] = m_energy[a];
    }
    ComputeMinTightSets();
    CheckSubmodular(n, m_energy);
}

inline REAL SoSGraph::IBFSEnergyTableClique::ComputeEnergy(const std::vector<int>& labels) const {
    Assignment assgn = 0;
    for (size_t i = 0; i < this->m_nodes.size(); ++i) {
        NodeId n = this->m_nodes[i];
        if (labels[n] == 1) {
            assgn |= 1 << i;
        }
    }
    return m_energy[assgn];
}

inline REAL SoSGraph::IBFSEnergyTableClique::ComputeAlphaEnergy(const std::vector<int>& labels) const {
    Assignment assgn = 0;
    for (size_t i = 0; i < this->m_nodes.size(); ++i) {
        NodeId n = this->m_nodes[i];
        if (labels[n] == 1) {
            assgn |= 1 << i;
        }
    }
    return m_alpha_energy[assgn];
}

inline REAL SoSGraph::IBFSEnergyTableClique::ExchangeCapacity(size_t u_idx, size_t v_idx) const {
    const size_t n = this->m_nodes.size();
    ASSERT(u_idx < n);
    ASSERT(v_idx < n);

    REAL min_energy = std::numeric_limits<REAL>::max();
    Assignment num_assgns = 1 << n;
    const Assignment bound = num_assgns-1;
    const Assignment u_mask = 1 << u_idx;
    const Assignment v_mask = 1 << v_idx;
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

inline void SoSGraph::IBFSEnergyTableClique::Push(size_t u_idx, size_t v_idx, REAL delta) {
    ASSERT(u_idx < this->m_nodes.size());
    ASSERT(v_idx < this->m_nodes.size());
    m_alpha_Ci[u_idx] += delta;
    m_alpha_Ci[v_idx] -= delta;
    const size_t n = this->m_nodes.size();
    Assignment num_assgns = 1 << n;
    const Assignment bound = num_assgns-1;
    const Assignment u_mask = 1 << u_idx;
    const Assignment v_mask = 1 << v_idx;
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

inline void SoSGraph::IBFSEnergyTableClique::ComputeMinTightSets() {
    size_t n = this->m_nodes.size();
    Assignment num_assgns = 1 << n;
    const Assignment bound = num_assgns-1;
    for (auto& a : m_min_tight_set)
        a = bound;
    for (Assignment assgn = bound-1; assgn >= 1; --assgn) {
        if (m_alpha_energy[assgn] == 0) {
            for (size_t i = 0; i < n; ++i) {
                //ASSERT(m_alpha_energy[m_min_tight_set[i] & assgn] == 0);
                //ASSERT(m_alpha_energy[m_min_tight_set[i] | assgn] == 0);
                if ((assgn & (1 << i)) != 0)
                    m_min_tight_set[i] = assgn;
            }
        }
    }
}

inline bool SoSGraph::IBFSEnergyTableClique::NonzeroCapacity(size_t u_idx, size_t v_idx) const {
    Assignment min_set = m_min_tight_set[u_idx];
    return (min_set & (1 << v_idx)) != 0;
}

inline void SoSGraph::IBFSEnergyTableClique::ResetAlpha() {
    for (auto& a : this->m_alpha_Ci) {
        a = 0;
    }
    const size_t n = this->m_nodes.size();
    const Assignment num_assignments = 1 << n;
    for (Assignment a = 0; a < num_assignments; ++a) {
        m_alpha_energy[a] = m_energy[a];
    }
}

template <SoSGraph::BoundFn UB>
void SoSGraph::UpperBoundCliques(const std::vector<bool>& fixedVars, NormStats* stats) {
    std::vector<REAL> psi;
    //int nCliques = m_cliques.size();
    int cliquesDone = 0;
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
        int k = c.Size();
        psi.resize(k);
        // Compute upper bound g of clique energy
        UB(k, c.EnergyTable(), newEnergy);

        if (!fixedVars.empty()) {
            uint32_t fixedSet = 0;
            for (int i = 0; i < k; ++i)
                fixedSet |= (fixedVars[c.Nodes()[i]] << i);
            ZeroMarginalSet(k, newEnergy, fixedSet);
        }

        if (stats) {
            stats->L1 += DiffL1(c.EnergyTable(), newEnergy);
            stats->L2 += DiffL2(c.EnergyTable(), newEnergy);
            stats->LInfty += DiffLInfty(c.EnergyTable(), newEnergy);
        }
        // Modify g, find psi so that g'(S) = g(S) + psi(S) >= 0
        Normalize(k, newEnergy, psi);
        /*
         *AddLinear(k, c.EnergyTable(), psi);
         */

        auto& alpha_Ci = c.AlphaCi();
        for (int i = 0; i < k; ++i) {
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

inline void SoSGraph::UpperBoundCliques(UBfn ub, NormStats* stats) {
    UpperBoundCliques(ub, std::vector<bool>{}, std::vector<int>{}, stats);
}

inline void SoSGraph::UpperBoundCliques(UBfn ub, const std::vector<bool>& fixedVars, const std::vector<int>& /*labels*/, NormStats* stats) {
    switch (ub) {
        case UBfn::chen: UpperBoundCliques<ChenUpperBound>(fixedVars, stats);
                    break;
        case UBfn::cvpr14: UpperBoundCliques<UpperBoundCVPR14>(fixedVars, stats);
                    break;
    }
}

#endif
