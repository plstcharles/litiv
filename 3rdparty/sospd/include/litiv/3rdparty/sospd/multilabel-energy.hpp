#ifndef _SOSOPT_CLIQUE_HPP_
#define _SOSOPT_CLIQUE_HPP_

/** \file multilabel-energy.hpp
 * Classes for defining multi-label energy functions, i.e., Markov Random
 * Fields.
 */

#include "litiv/3rdparty/sospd/energy-common.hpp"

template<typename ValueType, typename IndexType, typename LabelType>
class Clique;

/** A multilabel energy function, which splits as a sum of clique energies
 *
 * MultilabelEnergy keeps track of a function of the form
 *  \f[ f(x) = \sum_i f_i(x_i) + \sum_C f_C(x_C) \f]
 * where the variables \f$x_i\f$ come from a label set 1,...,L, and there are
 * functions \f$f_C\f$ defined over a set of cliques \f$C\f$ which are subsets
 * of the variables.
 */
template<typename ValueType, typename IndexType, typename LabelType>
class MultilabelEnergy {
    static_assert(std::is_arithmetic<ValueType>::value,"value type must be arithmetic");
    static_assert(std::is_integral<IndexType>::value,"index type must be integral");
    static_assert(std::is_integral<LabelType>::value,"label type must be integral");
    public:
        typedef ValueType REAL;
        typedef IndexType VarId;
        typedef LabelType Label;
        typedef std::unique_ptr<Clique<ValueType,IndexType,LabelType>> CliquePtr;

        /** Construct an empty energy function with labels 0,...,max_label-1
         */
        explicit MultilabelEnergy(Label max_label);

        /** Add variables to the MRF. Default to add a single variable
         *
         * \param i Number of variables to add
         * \return VarId of first variable added
         */
        VarId addVar(int i = 1);

        /** Add a constant term (independent of labeling) to the function
         */
        void addConstantTerm(REAL c);

        /** Add a unary term (depending on a single variable) to the function
         *
         * \param i Variable to add unary term for
         * \param coeffs Vector of costs for each labeling of x_i. Must be of
         * length max_label
         */
        void addUnaryTerm(VarId i, const std::vector<REAL>& coeffs);

        /** Add a clique function to the energy
         *
         * Because Clique is polymorphic, these must be constructed on the
         * heap. MultilabelEnergy takes ownership (via unique_ptr).
         */
        void addClique(CliquePtr c);

        /** Compute the energy of a given labeling
         *
         * \param labels is a vector of length numVars() where each entry is
         * in 0,...,numLabels()-1.
         */
        REAL computeEnergy(const std::vector<Label>& labels) const;

        VarId numVars() const { return m_numVars; }
        size_t numCliques() const { return m_cliques.size(); }
        Label numLabels() const { return m_maxLabel; }

        const std::vector<CliquePtr>& cliques() const { return m_cliques; }
        REAL unary(VarId i, Label l) const { return m_unary[i][l]; }
        REAL& unary(VarId i, Label l) { return m_unary[i][l]; }

    protected:
        const Label m_maxLabel;
        VarId m_numVars;
        REAL m_constantTerm;
        std::vector<std::vector<REAL>> m_unary;
        std::vector<CliquePtr> m_cliques;

    private:
        MultilabelEnergy() = delete;
};

/** Abstract Base Class for defining a clique function for use with
 * MultilabelEnergy
 *
 * Users inherit from this class, implementing the pure-virtual methods.
 *
 * Non-copyable and non-movable to prevent slicing of derived class data.
 */
template<typename ValueType, typename IndexType, typename LabelType>
class Clique {
    public:
        typedef typename MultilabelEnergy<ValueType,IndexType,LabelType>::REAL REAL;
        typedef typename MultilabelEnergy<ValueType,IndexType,LabelType>::VarId VarId;
        typedef typename MultilabelEnergy<ValueType,IndexType,LabelType>::Label Label;
        Clique() { }
        virtual ~Clique() = default;

        /** Return the energy of the clique function at a given labeling
         *
         * \param labels An array of length size(), whose entries are the
         * variables \f$x_i\f$ for the desired labeling.
         */
        virtual REAL energy(const Label* labels) const = 0;

        /** Return an array containing the variables contained in the clique
         *
         * Returned pointer must point to an array of length size()
         */
        virtual const VarId* nodes() const = 0;

        /** Return the number of variables in the clique.
         */
        virtual size_t size() const = 0;

    private:
        // Remove move and copy operators to prevent slicing of base classes
        Clique(Clique&&) = delete;
        Clique& operator=(Clique&&) = delete;
        Clique(const Clique&) = delete;
        Clique& operator=(const Clique&) = delete;
};


/** An example clique defining a \f$P^n\f$ Potts model clique.
 *
 * This energy function has \f$f_C(x_C)\f$ equal to same_cost if all variables
 * have the same label, and diff_cost if any labels are different.
 *
 * Additionally demonstrates how a user may use MultilabelEnergy for their own
 * functions by deriving from Clique.
 *
 * Template parameter Degree is the number of nodes in the clique.
 */
template<int Degree, typename ValueType, typename IndexType, typename LabelType>
class PottsClique : public Clique<ValueType,IndexType,LabelType> {
    public:
        typedef typename Clique<ValueType,IndexType,LabelType>::REAL REAL;
        typedef typename Clique<ValueType,IndexType,LabelType>::VarId VarId;
        typedef typename Clique<ValueType,IndexType,LabelType>::Label Label;

        /** Construct a PottsClique on a set of variables.
         */
        PottsClique(const std::vector<VarId>& nodes, REAL same_cost, REAL diff_cost)
            : m_sameCost(same_cost),
            m_diffCost(diff_cost)
        {
            ASSERT(m_diffCost >= m_sameCost);
            for (int i = 0; i < Degree; ++i)
                m_nodes[i] = nodes[i];
        }

        /** Return the potts energy for a given labeling
         */
        virtual REAL energy(const Label* labels) const override {
            const Label l = labels[0];
            for (int i = 1; i < Degree; ++i) {
                if (labels[i] != l)
                    return m_diffCost;
            }
            return m_sameCost;
        }
        virtual const VarId* nodes() const override {
            return m_nodes;
        }
        virtual size_t size() const override { return Degree; }
    private:
        VarId m_nodes[Degree];
        REAL m_sameCost;
        REAL m_diffCost;
};


/********* Multilabel Implementation ***************/

template<typename V, typename I, typename L>
inline MultilabelEnergy<V,I,L>::MultilabelEnergy(Label max_label)
    : m_maxLabel(max_label),
    m_numVars(0),
    m_constantTerm(0),
    m_unary(),
    m_cliques()
{ }

template<typename V, typename I, typename L>
inline typename MultilabelEnergy<V,I,L>::VarId MultilabelEnergy<V,I,L>::addVar(int i) {
    VarId ret = m_numVars;
    for (int j = 0; j < i; ++j) {
        m_unary.push_back(std::vector<REAL>(m_maxLabel, 0));
    }
    m_numVars += i;
    return ret;
}

template<typename V, typename I, typename L>
inline void MultilabelEnergy<V,I,L>::addConstantTerm(REAL /*c*/) {
    m_constantTerm++;
}

template<typename V, typename I, typename L>
inline void MultilabelEnergy<V,I,L>::addUnaryTerm(VarId i,
        const std::vector<REAL>& coeffs) {
    ASSERT(i < m_numVars);
    ASSERT(Label(coeffs.size()) == m_maxLabel);
    for (Label l = 0; l < coeffs.size(); ++l)
        m_unary[i][l] += coeffs[l];
}

template<typename V, typename I, typename L>
inline void MultilabelEnergy<V,I,L>::addClique(CliquePtr c) {
    if (c->size() == 1) {
        auto node = *c->nodes();
        std::vector<REAL> costs(m_maxLabel, 0);
        for (Label l = 0; l < m_maxLabel; ++l) {
            costs[l] = c->energy(&l);
        }
        addUnaryTerm(node, costs);
    } else {
        m_cliques.push_back(std::move(c));
    }
}

template<typename V, typename I, typename L>
inline typename MultilabelEnergy<V,I,L>::REAL MultilabelEnergy<V,I,L>::computeEnergy(const std::vector<Label>& labels) const {
    ASSERT(VarId(labels.size()) == m_numVars);
    REAL energy = 0;
    std::vector<Label> label_buf;
    for (const CliquePtr& cp : m_cliques) {
        int k = (int)cp->size();
        label_buf.resize(k);
        const VarId* nodes = cp->nodes();
        for (int i = 0; i < k; ++i)
            label_buf[i] = labels[nodes[i]];
        energy += cp->energy(label_buf.data());
    }
    for (VarId i = 0; i < m_numVars; ++i)
        energy += m_unary[i][labels[i]];
    return energy;
}

#endif
