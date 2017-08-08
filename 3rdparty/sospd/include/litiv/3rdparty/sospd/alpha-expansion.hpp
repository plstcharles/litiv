#ifndef _ALPHA_EXPANSION_HPP_
#define _ALPHA_EXPANSION_HPP_

#include "litiv/3rdparty/sospd/energy-common.hpp"
#include <vector>

class SubmodularIBFS;

class MultiLabelCRF {
    public:
        typedef int Label;
        typedef int NodeId;
        class Clique;
        typedef std::shared_ptr<Clique> CliquePtr;
        typedef std::vector<REAL> UnaryCost;

        MultiLabelCRF() = delete;
        explicit MultiLabelCRF(Label max_label);

        NodeId AddNode(int i = 1);
        int GetLabel(NodeId i) const;

        void AddConstantTerm(REAL c);
        void AddUnaryTerm(NodeId i, const std::vector<REAL>& coeffs);
        void AddClique(const CliquePtr& cp);

        void AlphaExpansion();
        void Solve();

        REAL ComputeEnergy() const;
        REAL ComputeEnergy(const std::vector<Label>& labels) const;

        class Clique {
            public:
            Clique(const std::vector<NodeId>& nodes)
                : m_nodes(nodes)
            { }
            virtual ~Clique() = default;

            // labels is a vector of length m_nodes.size() with the labels
            // of m_nodes. Returns the energy of the clique at that labeling
            virtual REAL Energy(const std::vector<Label>& labels) const = 0;

            const std::vector<NodeId>& Nodes() const { return m_nodes; }
            size_t Size() const { return m_nodes.size(); }

            protected:
            std::vector<NodeId> m_nodes;

            // Remove move and copy operators to prevent slicing of base classes
            Clique(Clique&&) = delete;
            Clique& operator=(Clique&&) = delete;
            Clique(const Clique&) = delete;
            Clique& operator=(const Clique&) = delete;
        };

    protected:
        void SetupAlphaEnergy(Label alpha, SubmodularIBFS& crf) const;
        void InitialLabeling();
        const size_t m_num_labels;
        REAL m_constant_term;
        std::vector<CliquePtr> m_cliques;
        std::vector<UnaryCost> m_unary_cost;
        std::vector<Label> m_labels;
};

class PottsClique : public MultiLabelCRF::Clique {
    public:
        typedef MultiLabelCRF::NodeId NodeId;
        typedef MultiLabelCRF::Label Label;

        PottsClique(const std::vector<NodeId>& nodes, REAL same_cost, REAL diff_cost)
            : MultiLabelCRF::Clique(nodes),
            m_same_cost(same_cost),
            m_diff_cost(diff_cost)
        { }

        virtual REAL Energy(const std::vector<Label>& labels) const override {
            const Label l = labels[0];
            for (Label l2 : labels)
                if (l2 != l)
                    return m_diff_cost;
            return m_same_cost;
        }
    private:
        REAL m_same_cost;
        REAL m_diff_cost;
};

class SeparableClique : public MultiLabelCRF::Clique {
    public:
        typedef MultiLabelCRF::NodeId NodeId;
        typedef MultiLabelCRF::Label Label;
        typedef uint32_t Assgn;
        typedef std::vector<std::vector<REAL>> EnergyTable;

        SeparableClique(const std::vector<NodeId>& nodes, const EnergyTable& energy_table)
            : MultiLabelCRF::Clique(nodes),
            m_energy_table(energy_table) { }

        virtual REAL Energy(const std::vector<Label>& labels) const override {
            ASSERT(labels.size() == this->m_nodes.size());
            const Label num_labels = m_energy_table.size();
            std::vector<Assgn> per_label(num_labels, 0);
            for (size_t i = 0; i < labels.size(); ++i)
                per_label[labels[i]] |= 1 << i;
            REAL e = 0;
            for (Label l = 0; l < num_labels; ++l)
                e += m_energy_table[l][per_label[l]];
            return e;
        }
    private:
        EnergyTable m_energy_table;
};




#endif
