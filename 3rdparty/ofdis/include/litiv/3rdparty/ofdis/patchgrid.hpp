
#pragma once

#ifndef OFDIS_INTERNAL
#error "must only include 'ofdis.hpp' header for API"
#endif //ndef(OFDIS_INTERNAL)

#include "litiv/3rdparty/ofdis/patch.hpp"
#include "litiv/3rdparty/ofdis/oflow.hpp" // For camera intrinsic and opt. parameter struct

namespace ofdis {

    // Class implements step (3.) in Algorithm 1 of the paper:
    //   I holds all patch objects on a specific scale, and
    //   organizes 1) the dense inverse search, 2) densification
    template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
    class PatGridClass {
    public:
        PatGridClass(const camparam* cpt_in,const camparam* cpo_in,const optparam* op_in);
        ~PatGridClass();
        void InitializeGrid(const float* im_ao_in,const float* im_ao_dx_in,const float* im_ao_dy_in);
        void SetTargetImage(const float* im_bo_in,const float* im_bo_dx_in,const float* im_bo_dy_in);
        void InitializeFromCoarserOF(const float* flow_prev);
        void AggregateFlowDense(float* flowout) const;
        // Optimizes grid to convergence of each patch
        void Optimize();
        void SetComplGrid(PatGridClass* cg_in);
        inline const int GetNoPatches() const {return nopatches;}
        inline const int GetNoph() const {return noph;}
        inline const int GetNopw() const {return nopw;}
        inline const Eigen::Vector2f GetRefPatchPos(int i) const {return pt_ref[i];} // Get reference  patch position
        inline const Eigen::Vector2f GetQuePatchPos(int i) const {return pat[i]->GetPointPos();} // Get target/query patch position
        inline const Eigen::Vector2f GetQuePatchDis(int i) const {return pt_ref[i]-pat[i]->GetPointPos();} // Get query patch displacement from reference patch

    private:
        const float* im_ao,* im_ao_dx,* im_ao_dy;
        const float* im_bo,* im_bo_dx,* im_bo_dy;
        Eigen::Map<const Eigen::MatrixXf>* im_ao_eg,* im_ao_dx_eg,* im_ao_dy_eg;
        Eigen::Map<const Eigen::MatrixXf>* im_bo_eg,* im_bo_dx_eg,* im_bo_dy_eg;
        const camparam* cpt;
        const camparam* cpo;
        const optparam* op;
        int steps;
        int nopw;
        int noph;
        int nopatches;
        // patch objects
        std::vector<PatClass<eInput,eOutput>*> pat;
        // midpoints for reference patches
        std::vector<Eigen::Vector2f> pt_ref;
        // starting parameters for query patches, use only 1 for depth, 2 for OF, all 4 for scene flow
        using p_init_type = std::conditional_t<(eOutput==ofdis::FlowOutput_OpticalFlow),std::vector<Eigen::Vector2f>,std::vector<Eigen::Matrix<float,1,1>>>;
        p_init_type p_init;
        const PatGridClass* cg = nullptr;
    };

} // namespace ofdis