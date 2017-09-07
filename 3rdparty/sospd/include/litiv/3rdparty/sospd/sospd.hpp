#ifndef _SOSPD_HPP_
#define _SOSPD_HPP_

/** \file sospd.hpp
 * Sum-of-submodular Primal Dual algorithm for multilabel problems
 */

#include "litiv/3rdparty/sospd/multilabel-energy.hpp"
#include "litiv/3rdparty/sospd/submodular-ibfs.hpp"

/** Optimizer using Sum-of-submodular Primal Dual algorithm.
 *
 * Implements SoSPD algorithm from Fix, Wang, Zabih in CVPR 14.
 */
template <typename Flow = SubmodularIBFS>
class SoSPD {
    public:
        typedef MultilabelEnergy::VarId VarId;
        typedef MultilabelEnergy::Label Label;

        /** Proposal callbacks take as input the iteration number and current
         * labeling (as a vector of labels) and write the next proposal to the
         * final parameter.
         */
        typedef std::function<
              void(int niter,
                   const std::vector<Label>& current,
                   std::vector<Label>& proposed)
            > ProposalCallback;

        /** Set up SoSPD to optimize a particular energy function
         *
         * \param energy Energy function to optimize.
         */
        explicit SoSPD(const MultilabelEnergy* energy);
        explicit SoSPD(const MultilabelEnergy* energy, SubmodularIBFSParams& params);

        /** Run SoSPD algorithm either to completion, or for a number of steps.
         *
         * Each iteration has a single proposal (determined by
         * SetProposalCallback), and solves a corresponding Sum-of-Submodular
         * flow problem.
         *
         * Resulting labeling can be queried from GetLabel.
         *
         * \param niters Number of iterations
         */
        void Solve(int niters = std::numeric_limits<int>::max());

        /** Return label of a node i, returns -1 if Solve has not been called.*/
        int GetLabel(VarId i) const;

        /** Give hint that energy is expansion submodular. Enables optimizations
         * because we don't need to find submodular upper/lower bounds for the
         * function.
         */
        void SetExpansionSubmodular(bool b) { m_expansion_submodular = b; }

        /** Choose whether to use lower/upper bound in approximating function.
         */
        void SetLowerBound(bool b) { m_lower_bound = b; }

        /** Specify method for choosing proposals. */
        void SetProposalCallback(const ProposalCallback& pc) { m_pc = pc; }

        /** Set the proposal method to alpha-expansion
         *
         * Alpha-expansion proposals simply cycle through the labels, proposing
         * a constant labeling (i.e., all "alpha") at each iteration.
         */
        void SetAlphaExpansion() {
            m_pc = [&](int, const std::vector<Label>&, std::vector<Label>&) {
                AlphaProposal();
            };
        }

        /** Set the proposal method to best-height alpha-expansion
         *
         * Best-height alpha-expansion, instead of cycling through labels,
         * chooses the single alpha with the biggest sum of differences in
         * heights.
         */
        void SetHeightAlphaExpansion() {
            m_pc = [&](int, const std::vector<Label>&, std::vector<Label>&) {
                HeightAlphaProposal();
            };
        }

        /** Return lower bound on optimum, determined by current dual */
        double LowerBound();

        REAL dualVariable(int alpha, VarId i, Label l) const;
        Flow* GetFlow() { return &m_ibfs; }

    private:
        typedef MultilabelEnergy::CliquePtr CliquePtr;
        typedef std::vector<REAL> LambdaAlpha;
        typedef std::vector<std::pair<size_t, size_t>> NodeNeighborList;
        typedef std::vector<NodeNeighborList> NodeCliqueList;

        REAL ComputeHeight(VarId, Label);
        REAL ComputeHeightDiff(VarId i, Label l1, Label l2) const;
        void SetupGraph(Flow& crf);
        // TODO(afix): redo this
        void SetupAlphaEnergy(Flow& crf);
        void InitialLabeling();
        void InitialDual();
        void InitialNodeCliqueList();
        bool InitialFusionLabeling();
        void PreEditDual(Flow& crf);
        bool UpdatePrimalDual(Flow& crf);
        void PostEditDual();
        void DualFit();
        REAL& Height(VarId i, Label l) { return m_heights[i*m_num_labels+l]; }

        REAL& dualVariable(int alpha, VarId i, Label l);
        REAL dualVariable(const LambdaAlpha& lambdaAlpha,
                VarId i, Label l) const;
        REAL& dualVariable(LambdaAlpha& lambdaAlpha,
                VarId i, Label l);
        LambdaAlpha& lambdaAlpha(int alpha);
        const LambdaAlpha& lambdaAlpha(int alpha) const;

        // Move Proposals
        void HeightAlphaProposal();
        void AlphaProposal();

        const MultilabelEnergy* m_energy;
        // Unique ptr so we can forward declare?
        Flow m_ibfs;
        const size_t m_num_labels;
        std::vector<Label> m_labels;
        /// The proposed labeling in a given iteration
        std::vector<Label> m_fusion_labels;
        // Factor this list back into a node list?
        NodeCliqueList m_node_clique_list;
        // FIXME(afix) change way m_dual is stored. Put lambda_alpha as separate
        // REAL* for each clique, indexed by i, l.
        std::vector<LambdaAlpha> m_dual;
        std::vector<REAL> m_heights;
        bool m_expansion_submodular;
        bool m_lower_bound;
        int m_iter;
        ProposalCallback m_pc;
};

#endif
