#ifndef _FUSION_MOVE_HPP_
#define _FUSION_MOVE_HPP_

/*
 * fusion-move.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Computes a fusion move between the current and proposed image.
 *
 * A fusion move takes two images (current and proposed) and tries to perform
 * the optimal move where each pixel is allowed to either stay at its current
 * value, or switch to its label in the proposed image. This is a 
 * generalization of alpha-expansion, where in alpha-expansion each pixel is 
 * allowed to either stay the same, or change to a fixed value alpha. That is,
 * alpha expansion is a fusion move where the proposed image is just the flat
 * image with value alpha at all pixels.
 */

#include <iostream>
#include <sstream>
#include <functional>
#include <vector>
#include "litiv/3rdparty/sospd/higher-order-energy.hpp"
#include "litiv/3rdparty/HOCR/HOCR.h"
#include "litiv/3rdparty/sospd/multilabel-energy.hpp"
#include "litiv/3rdparty/qpbo.hpp"
#include "litiv/3rdparty/sospd/generic-higher-order.hpp"

template <int MaxDegree>
class FusionMove {
    public:
        enum class Method { FGBZ, HOCR, GRD, SOS_UB };
        typedef MultilabelEnergy::VarId VarId;
        typedef MultilabelEnergy::Label Label;
        typedef std::vector<Label> LabelVec;
        typedef std::function<void(int, const LabelVec&, LabelVec&)> ProposalCallback;
        FusionMove(const MultilabelEnergy* energy, const ProposalCallback& pc)
            : m_energy(energy)
            , m_pc(pc)
            , m_labels(energy->numVars(), 0)
            , m_iter(0)
            , m_method(Method::FGBZ) 
        { }

        FusionMove(const MultilabelEnergy* energy,
                   const ProposalCallback& pc,
                   const LabelVec& current)
            : m_energy(energy)
            , m_pc(pc)
            , m_labels(current)
            , m_iter(0)
            , m_method(Method::FGBZ) 
        { }

        void Solve(int niters);
        Label GetLabel(VarId i) const { return m_labels[i]; }
        void SetMethod(Method m) { m_method = m; }

    protected:
        template <typename HO>
        void SetupFusionEnergy(const LabelVec& proposed,
                HO& hoe) const;
        void GetFusedImage(const LabelVec& proposed, QPBO<REAL>& qr);
        void FusionStep();
    
        const MultilabelEnergy* m_energy;
        ProposalCallback m_pc;
        LabelVec m_labels;
        int m_iter;
        Method m_method;
};

template <int MaxDegree>
void FusionMove<MaxDegree>::Solve(int niters) {
    for (int i = 0; i < niters; ++i)
        FusionStep();
}

template <int MaxDegree>
void FusionMove<MaxDegree>::FusionStep() {
    switch (m_method) {
        case Method::FGBZ: 
        {
            HigherOrderEnergy<REAL, MaxDegree> hoe;
            QPBO<REAL> qr(m_labels.size(), 0);
            LabelVec proposed(m_labels.size());
            m_pc(m_iter, m_labels, proposed);
            SetupFusionEnergy(proposed, hoe);
            hoe.ToQuadratic(qr);
            qr.MergeParallelEdges();
            qr.Solve();
            qr.ComputeWeakPersistencies();
            GetFusedImage(proposed, qr);
            break;
        }
        case Method::HOCR:
        {
            PBF<REAL, MaxDegree> pbf;
            LabelVec proposed(m_labels.size());
            m_pc(m_iter, m_labels, proposed);
            SetupFusionEnergy(proposed, pbf);
            PBF<REAL, 2> qr;
            pbf.toQuadratic(qr);
            pbf.clear();
            int numvars = qr.maxID();
            QPBO<REAL> qpbo(numvars, numvars*4);
            convert(qpbo, qr);
            qpbo.AddNode(m_labels.size());
            qr.clear();
            qpbo.MergeParallelEdges();
            qpbo.Solve();
            qpbo.ComputeWeakPersistencies();
            GetFusedImage(proposed, qpbo);
            break;
        }
        case Method::GRD:
        {
#ifdef WITH_GRD
            Petter::PseudoBoolean<REAL> pb;
            LabelVec proposed(m_labels.size());
            m_pc(m_iter, m_labels, proposed);
            SetupFusionEnergy(proposed, pb);
            std::vector<Petter::label> x(m_labels.size());
            int labeled;
            pb.minimize(x, labeled, Petter::GRD_heur);
            for (size_t i = 0; i < m_labels.size(); ++i) {
                if (x[i] == 1) {
                    m_labels[i] = proposed[i];
                }
            }
#else
            ASSERT(false && "GRD not installed!");
#endif
            break;
        }
        case Method::SOS_UB:
        {
            SubmodularIBFS ibfs;
            LabelVec proposed(m_labels.size());
            m_pc(m_iter, m_labels, proposed);
            SetupFusionEnergy(proposed, ibfs);
            ibfs.Solve();
            for (size_t i = 0; i < m_labels.size(); ++i) {
                if (ibfs.GetLabel(i) == 1)
                    m_labels[i] = proposed[i];
            }
        }
    }
    m_iter++;
}

template <int MaxDegree>
void FusionMove<MaxDegree>::GetFusedImage(const LabelVec& proposed, QPBO<REAL>& qr) {
    for (size_t i = 0; i < m_labels.size(); ++i) {
        int label = qr.GetLabel(i);
        if (label == 1) {
            m_labels[i] = proposed[i];
        }
    }
}

template <int MaxDegree>
template <typename HO>
void FusionMove<MaxDegree>::SetupFusionEnergy(const LabelVec& proposed, HO& hoe) const {
    AddVars(hoe,m_energy->numVars());
    for (VarId i = 0; i < m_energy->numVars(); ++i) {
        AddUnaryTerm(
                hoe, 
                i, 
                m_energy->unary(i, proposed[i])
                - m_energy->unary(i, m_labels[i])
                );
        AddConstantTerm(hoe, m_energy->unary(i, m_labels[i]));
    }

    std::vector<REAL> energy_table;
    for (const auto& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        VarId size = c.size();
        ASSERT(size > 1);

        uint32_t numAssignments = 1 << size;
        energy_table.resize(numAssignments);
        
        // For each boolean assignment, get the clique energy at the 
        // corresponding labeling
        std::vector<Label> cliqueLabels(size);
        for (uint32_t assignment = 0; assignment < numAssignments; ++assignment) {
            for (VarId i = 0; i < size; ++i) {
                if (assignment & (1 << i)) { 
                    cliqueLabels[i] = proposed[c.nodes()[i]];
                } else {
                    cliqueLabels[i] = m_labels[c.nodes()[i]];
                }
            }
            energy_table[assignment] = c.energy(cliqueLabels.data());
        }
        std::vector<VarId> nodes(c.nodes(), c.nodes() + c.size());
        AddClique(hoe, int(c.size()), energy_table.data(), nodes.data());
    }
}

#endif
