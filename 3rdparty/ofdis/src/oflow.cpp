
#define OFDIS_INTERNAL
#include <Eigen/Core>
#include "litiv/3rdparty/ofdis/fdf/image.h"
#include <litiv/3rdparty/ofdis/utils.hpp>
#include "litiv/3rdparty/ofdis/oflow.hpp"
#include "litiv/3rdparty/ofdis/patchgrid.hpp"
#include "litiv/3rdparty/ofdis/refine_variational.hpp"
#include "litiv/utils/defines.hpp" // only used here for compiler flags

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
ofdis::OFClass<eInput,eOutput>::OFClass(
        const std::vector<const float*>& im_ao_in,
        const std::vector<const float*>& im_ao_dx_in,
        const std::vector<const float*>& im_ao_dy_in,
        const std::vector<const float*>& im_bo_in,
        const std::vector<const float*>& im_bo_dx_in,
        const std::vector<const float*>& im_bo_dy_in,
        const int imgpadding_in,
        float* outflow,
        const float* initflow,
        const int width_in,
        const int height_in,
        const int sc_f_in,
        const int sc_l_in,
        const int max_iter_in,
        const int min_iter_in,
        const float dp_thresh_in,
        const float dr_thresh_in,
        const float res_thresh_in,
        const int p_samp_s_in,
        const float patove_in,
        const bool usefbcon_in,
        const int costfct_in,
        const int noc_in,
        const int patnorm_in,
        const bool usetvref_in,
        const float tv_alpha_in,
        const float tv_gamma_in,
        const float tv_delta_in,
        const int tv_innerit_in,
        const int tv_solverit_in,
        const float tv_sor_in,
        const int verbosity_in) :
        im_ao(im_ao_in),
        im_ao_dx(im_ao_dx_in),
        im_ao_dy(im_ao_dy_in),
        im_bo(im_bo_in),
        im_bo_dx(im_bo_dx_in),
        im_bo_dy(im_bo_dy_in) {

#if USING_OPENMP
    if(verbosity_in>1)
        std::cout <<  "OPENMP is ON - used in pconst, pinit, potim";
#ifdef USE_PARALLEL_ON_FLOWAGGR
    if(verbosity_in>1)
        std::cout << ", cflow ";
#endif
    if(verbosity_in>1)
        std::cout << std::endl;
#endif //USING_OPENMP

    op.nop = (eOutput==ofdis::FlowOutput_OpticalFlow)?2:1;
    op.p_samp_s = p_samp_s_in;  // patch has even border length, center pixel is at (p_samp_s/2, p_samp_s/2) (ZERO INDEXED!)
    op.outlierthresh = (float)op.p_samp_s/2;
    op.patove = patove_in;
    op.sc_f = sc_f_in;
    op.sc_l = sc_l_in;
    op.max_iter = max_iter_in;
    op.min_iter = min_iter_in;
    op.dp_thresh = dp_thresh_in*dp_thresh_in; // saves the square to compare with squared L2-norm (saves sqrt operation)
    op.dr_thresh = dr_thresh_in;
    op.res_thresh = res_thresh_in;
    op.steps = std::max(1,(int)floor(op.p_samp_s*(1-op.patove)));
    op.novals = noc_in*(p_samp_s_in)*(p_samp_s_in);
    op.usefbcon = usefbcon_in;
    op.costfct = costfct_in;
    op.noc = noc_in;
    op.patnorm = patnorm_in;
    op.verbosity = verbosity_in;
    op.noscales = op.sc_f-op.sc_l+1;
    op.usetvref = usetvref_in;
    op.tv_alpha = tv_alpha_in;
    op.tv_gamma = tv_gamma_in;
    op.tv_delta = tv_delta_in;
    op.tv_innerit = tv_innerit_in;
    op.tv_solverit = tv_solverit_in;
    op.tv_sor = tv_sor_in;

    op.normoutlier_tmpbsq = (v4sf){op.normoutlier*op.normoutlier,op.normoutlier*op.normoutlier,op.normoutlier*op.normoutlier,op.normoutlier*op.normoutlier};
    op.normoutlier_tmp2bsq = __builtin_ia32_mulps(op.normoutlier_tmpbsq,op.twos);
    op.normoutlier_tmp4bsq = __builtin_ia32_mulps(op.normoutlier_tmpbsq,op.fours);

    // Variables for algorithm timings
    struct timeval tv_start_all,tv_end_all,tv_start_all_global,tv_end_all_global;
    if(op.verbosity>0)
      gettimeofday(&tv_start_all_global,nullptr);

    // ... per each scale
    double tt_patconstr[op.noscales],tt_patinit[op.noscales],tt_patoptim[op.noscales],tt_compflow[op.noscales],tt_tvopt[op.noscales],tt_all[op.noscales];
    for(int sl = op.sc_f; sl>=op.sc_l; --sl) {
        tt_patconstr[sl-op.sc_l] = 0;
        tt_patinit[sl-op.sc_l] = 0;
        tt_patoptim[sl-op.sc_l] = 0;
        tt_compflow[sl-op.sc_l] = 0;
        tt_tvopt[sl-op.sc_l] = 0;
        tt_all[sl-op.sc_l] = 0;
    }

    if(op.verbosity>1)
        gettimeofday(&tv_start_all,nullptr);

    // Create grids on each scale
    std::vector<ofdis::PatGridClass<eInput,eOutput>*> grid_fw(op.noscales);
    std::vector<ofdis::PatGridClass<eInput,eOutput>*> grid_bw(op.noscales); // grid for backward OF computation, only needed if 'usefbcon' is set to 1.
    std::vector<float*> flow_fw(op.noscales);
    std::vector<float*> flow_bw(op.noscales);
    cpl.resize(op.noscales);
    cpr.resize(op.noscales);
    for(int sl = op.sc_f; sl>=op.sc_l; --sl) {
        int i = sl-op.sc_l;

        float sc_fct = pow(2,-sl); // scaling factor at current scale
        cpl[i].sc_fct = sc_fct;
        cpl[i].height = height_in*sc_fct;
        cpl[i].width = width_in*sc_fct;
        cpl[i].imgpadding = imgpadding_in;
        cpl[i].tmp_lb = -(float)op.p_samp_s/2;
        cpl[i].tmp_ubw = (float)(cpl[i].width+op.p_samp_s/2-2);
        cpl[i].tmp_ubh = (float)(cpl[i].height+op.p_samp_s/2-2);
        cpl[i].tmp_w = cpl[i].width+2*imgpadding_in;
        cpl[i].tmp_h = cpl[i].height+2*imgpadding_in;
        cpl[i].curr_lv = sl;
        cpl[i].camlr = 0;

        cpr[i] = cpl[i];
        cpr[i].camlr = 1;

        flow_fw[i] = new float[op.nop*cpl[i].width*cpl[i].height];
        grid_fw[i] = new ofdis::PatGridClass<eInput,eOutput>(&(cpl[i]),&(cpr[i]),&op);

        if(op.usefbcon) { // for merging forward and backward flow
            flow_bw[i] = new float[op.nop*cpr[i].width*cpr[i].height];
            grid_bw[i] = new ofdis::PatGridClass<eInput,eOutput>(&(cpr[i]),&(cpl[i]),&op);

            // Make grids known to each other, necessary for AggregateFlowDense();
            grid_fw[i]->SetComplGrid(grid_bw[i]);
            grid_bw[i]->SetComplGrid(grid_fw[i]);
        }
    }

    // Timing, Grid memory allocation
    if(op.verbosity>1) {
        gettimeofday(&tv_end_all,nullptr);
        double tt_gridconst = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f+(tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
        printf("TIME (Grid Memo. Alloc. ) (ms): %3g\n",tt_gridconst);
    }

    // *** Main loop; Operate over scales, coarse-to-fine
    for(int sl = op.sc_f; sl>=op.sc_l; --sl) {
        int ii = sl-op.sc_l;

        if(op.verbosity>1) gettimeofday(&tv_start_all,nullptr);

        // Initialize grid (Step 1 in Algorithm 1 of paper)
        grid_fw[ii]->InitializeGrid(im_ao[sl],im_ao_dx[sl],im_ao_dy[sl]);
        grid_fw[ii]->SetTargetImage(im_bo[sl],im_bo_dx[sl],im_bo_dy[sl]);
        if(op.usefbcon) {
            grid_bw[ii]->InitializeGrid(im_bo[sl],im_bo_dx[sl],im_bo_dy[sl]);
            grid_bw[ii]->SetTargetImage(im_ao[sl],im_ao_dx[sl],im_ao_dy[sl]);
        }

        // Timing, Grid construction
        if(op.verbosity>1) {
            gettimeofday(&tv_end_all,nullptr);
            tt_patconstr[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f+(tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
            tt_all[ii] += tt_patconstr[ii];
            gettimeofday(&tv_start_all,nullptr);
        }

        // Initialization from previous scale, or to zero at first iteration. (Step 2 in Algorithm 1 of paper)
        if(sl<op.sc_f) {
            grid_fw[ii]->InitializeFromCoarserOF(flow_fw[ii+1]); // initialize from flow at previous coarser scale

            // Initialize backward flow
            if(op.usefbcon)
                grid_bw[ii]->InitializeFromCoarserOF(flow_bw[ii+1]);
        }
        else if(sl==op.sc_f && initflow!=nullptr) // initialization given input flow
            grid_fw[ii]->InitializeFromCoarserOF(initflow); // initialize from flow at coarser scale

        // Timing, Grid initialization
        if(op.verbosity>1) {
            gettimeofday(&tv_end_all,nullptr);
            tt_patinit[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f+(tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
            tt_all[ii] += tt_patinit[ii];
            gettimeofday(&tv_start_all,nullptr);
        }

        // Dense Inverse Search. (Step 3 in Algorithm 1 of paper)
        grid_fw[ii]->Optimize();
        if(op.usefbcon)
            grid_bw[ii]->Optimize();

        // Timing, DIS
        if(op.verbosity>1) {
            gettimeofday(&tv_end_all,nullptr);
            tt_patoptim[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f+(tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
            tt_all[ii] += tt_patoptim[ii];
            gettimeofday(&tv_start_all,nullptr);
        }

        // Densification. (Step 4 in Algorithm 1 of paper)
        float* tmp_ptr = flow_fw[ii];
        if(sl==op.sc_l)
            tmp_ptr = outflow;

        grid_fw[ii]->AggregateFlowDense(tmp_ptr);

        if(op.usefbcon && sl>op.sc_l)  // skip at last scale, backward flow no longer needed
            grid_bw[ii]->AggregateFlowDense(flow_bw[ii]);


        // Timing, Densification
        if(op.verbosity>1) {
            gettimeofday(&tv_end_all,nullptr);
            tt_compflow[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f+(tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
            tt_all[ii] += tt_compflow[ii];
            gettimeofday(&tv_start_all,nullptr);
        }


        // Variational refinement, (Step 5 in Algorithm 1 of paper)
        if(op.usetvref) {
            ofdis::VarRefClass<eInput,eOutput> varref_fw(im_ao[sl],im_ao_dx[sl],im_ao_dy[sl],im_bo[sl],im_bo_dx[sl],im_bo_dy[sl],&(cpl[ii]),&(cpr[ii]),&op,tmp_ptr);
            if(op.usefbcon && sl>op.sc_l)    // skip at last scale, backward flow no longer needed
                ofdis::VarRefClass<eInput,eOutput> varref_bw(im_bo[sl],im_bo_dx[sl],im_bo_dy[sl],im_ao[sl],im_ao_dx[sl],im_ao_dy[sl],&(cpr[ii]),&(cpl[ii]),&op,flow_bw[ii]);
        }

        // Timing, Variational Refinement
        if(op.verbosity>1) {
            gettimeofday(&tv_end_all,nullptr);
            tt_tvopt[ii] = (tv_end_all.tv_sec-tv_start_all.tv_sec)*1000.0f+(tv_end_all.tv_usec-tv_start_all.tv_usec)/1000.0f;
            tt_all[ii] += tt_tvopt[ii];
            printf("TIME (Sc: %i, #p:%6i, pconst, pinit, poptim, cflow, tvopt, total): %8.2f %8.2f %8.2f %8.2f %8.2f -> %8.2f ms.\n",sl,grid_fw[ii]->GetNoPatches(),tt_patconstr[ii],tt_patinit[ii],tt_patoptim[ii],tt_compflow[ii],tt_tvopt[ii],tt_all[ii]);
        }
    }

    // Clean up
    for(int sl = op.sc_f; sl>=op.sc_l; --sl) {
        delete[] flow_fw[sl-op.sc_l];
        delete grid_fw[sl-op.sc_l];
        if(op.usefbcon) {
            delete[] flow_bw[sl-op.sc_l];
            delete grid_bw[sl-op.sc_l];
        }
    }

    // Timing, total algorithm run-time
    if(op.verbosity>0) {
        gettimeofday(&tv_end_all_global,nullptr);
        double tt = (tv_end_all_global.tv_sec-tv_start_all_global.tv_sec)*1000.0f+(tv_end_all_global.tv_usec-tv_start_all_global.tv_usec)/1000.0f;
        printf("TIME (O.Flow Run-Time   ) (ms): %3g\n",tt);
    }
}