
#define OFDIS_INTERNAL
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include "litiv/3rdparty/ofdis/fdf/image.h"
#include "litiv/3rdparty/ofdis/patch.hpp"

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
ofdis::PatClass<eInput,eOutput>::PatClass(const camparam* cpt_in,
                                          const camparam* cpo_in,
                                          const optparam* op_in,
                                          const int patchid_in) :
        cpt(cpt_in),
        cpo(cpo_in),
        op(op_in),
        patchid(patchid_in) {
    pc = new patchstate<eOutput>();
    CreateStatusStruct(pc);
    tmp.resize(op->novals,1);
    dxx_tmp.resize(op->novals,1);
    dyy_tmp.resize(op->novals,1);
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::CreateStatusStruct(patchstate<eOutput>* psin) {
    // get reference / template patch
    psin->pdiff.resize(op->novals,1);
    psin->pweight.resize(op->novals,1);
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
ofdis::PatClass<eInput,eOutput>::~PatClass() {
    delete pc;
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::InitializePatch(Eigen::Map<const Eigen::MatrixXf>* im_ao_in,
                                                      Eigen::Map<const Eigen::MatrixXf>* im_ao_dx_in,
                                                      Eigen::Map<const Eigen::MatrixXf>* im_ao_dy_in,
                                                      const Eigen::Vector2f pt_ref_in) {
    im_ao = im_ao_in;
    im_ao_dx = im_ao_dx_in;
    im_ao_dy = im_ao_dy_in;
    pt_ref = pt_ref_in;
    ResetPatch();
    getPatchStaticNNGrad(im_ao->data(), im_ao_dx->data(), im_ao_dy->data(), &pt_ref, &tmp, &dxx_tmp, &dyy_tmp);
    ComputeHessian();
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::ComputeHessian() {
    if(eOutput==ofdis::FlowOutput_OpticalFlow) {
        pc->Hes(0,0) = (dxx_tmp.array()*dxx_tmp.array()).sum();
        pc->Hes(0,1) = (dxx_tmp.array()*dyy_tmp.array()).sum();
        pc->Hes(1,1) = (dyy_tmp.array()*dyy_tmp.array()).sum();
        pc->Hes(1,0) = pc->Hes(0,1);
        if(pc->Hes.determinant()==0) {
            pc->Hes(0,0) += 1e-10;
            pc->Hes(1,1) += 1e-10;
        }
    }
    else {
        pc->Hes(0,0) = (dxx_tmp.array()*dxx_tmp.array()).sum();
        if(pc->Hes.sum()==0)
            pc->Hes(0,0) += 1e-10;
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::SetTargetImage(Eigen::Map<const Eigen::MatrixXf>* im_bo_in,
                                                     Eigen::Map<const Eigen::MatrixXf>* im_bo_dx_in,
                                                     Eigen::Map<const Eigen::MatrixXf>* im_bo_dy_in) {
    im_bo = im_bo_in;
    im_bo_dx = im_bo_dx_in;
    im_bo_dy = im_bo_dy_in;
    ResetPatch();
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::ResetPatch() {
    pc->hasconverged=0;
    pc->hasoptstarted=0;
    pc->pt_st = pt_ref;
    pc->pt_iter = pt_ref;
    pc->p_in.setZero();
    pc->p_iter.setZero();
    pc->delta_p.setZero();
    pc->delta_p_sqnorm = 1e-10;
    pc->delta_p_sqnorm_init = 1e-10;
    pc->mares = 1e20;
    pc->mares_old = 1e20;
    pc->cnt=0;
    pc->invalid = false;
}

inline void paramtopt(Eigen::Vector2f& pt_iter, const Eigen::Vector2f& pt_ref, const Eigen::Vector2f& p_iter) {
    pt_iter = pt_ref + p_iter; // for optical flow the point displacement and the parameter vector are equivalent
}

inline void paramtopt(Eigen::Vector2f& pt_iter, const Eigen::Vector2f& pt_ref, const Eigen::Matrix<float,1,1>& p_iter) {
    pt_iter[0] = pt_ref[0] + p_iter[0];
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::OptimizeStart(const point_type& p_in_arg) {
    pc->p_in   = p_in_arg;
    pc->p_iter = p_in_arg;

    // convert from input parameters to 2D query location(s) for patches
    paramtopt(pc->pt_iter,pt_ref,pc->p_iter);

    // save starting location, only needed for outlier check
    pc->pt_st = pc->pt_iter;

    //Check if initial position is already invalid
    if(pc->pt_iter[0] < cpt->tmp_lb  || pc->pt_iter[1] < cpt->tmp_lb || // check if patch left valid image region
       pc->pt_iter[0] > cpt->tmp_ubw || pc->pt_iter[1] > cpt->tmp_ubh) {
        pc->hasconverged=1;
        pc->pdiff = tmp;
        pc->hasoptstarted=1;
    }
    else {
        pc->cnt=0; // reset iteration counter
        pc->delta_p_sqnorm = 1e-10;
        pc->delta_p_sqnorm_init = 1e-10;  // set to arbitrary low value, s.t. that loop condition is definitely true on first iteration
        pc->mares = 1e5;          // mean absolute residual
        pc->mares_old = 1e20; // for rate of change, keep mares from last iteration in here. Set high so that loop condition is definitely true on first iteration
        pc->hasconverged=0;

        OptimizeComputeErrImg();

        pc->hasoptstarted=1;
        pc->invalid = false;
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::OptimizeIter(const point_type& p_in_arg, const bool untilconv) {
    if(!pc->hasoptstarted) {
        ResetPatch();
        OptimizeStart(p_in_arg);
    }
    int oldcnt=pc->cnt;
    // optimize patch until convergence, or do only one iteration if DIS visualization is used
    while(!(pc->hasconverged || (untilconv == false && (pc->cnt > oldcnt)))) {
        pc->cnt++;
        // projection onto sd_images
        if(eOutput==ofdis::FlowOutput_OpticalFlow) {
            pc->delta_p[0] = (dxx_tmp.array()*pc->pdiff.array()).sum();
            pc->delta_p[1] = (dyy_tmp.array()*pc->pdiff.array()).sum();
        }
        else
            pc->delta_p[0] = (dxx_tmp.array()*pc->pdiff.array()).sum();
        pc->delta_p = pc->Hes.llt().solve(pc->delta_p); // solve linear system
        pc->p_iter -= pc->delta_p; // update flow vector
        if(eOutput==ofdis::FlowOutput_StereoDepth) {
            if(cpt->camlr==0)
                pc->p_iter[0] = std::min(pc->p_iter[0],0.0f); // disparity in t can only be negative (in right image)
            else
                pc->p_iter[0] = std::max(pc->p_iter[0],0.0f); // ... positive (in left image)
        }
        // compute patch locations based on new parameter vector
        paramtopt(pc->pt_iter,pt_ref,pc->p_iter);
        // check if patch(es) moved too far from starting location, if yes, stop iteration and reset to starting location
        if((pc->pt_st - pc->pt_iter).norm() > op->outlierthresh || // check if query patch moved more than >padval from starting location -> most likely outlier
            pc->pt_iter[0] < cpt->tmp_lb  || pc->pt_iter[1] < cpt->tmp_lb || // check patch left valid image region
            pc->pt_iter[0] > cpt->tmp_ubw || pc->pt_iter[1] > cpt->tmp_ubh) {
            pc->p_iter = pc->p_in; // reset
            paramtopt(pc->pt_iter,pt_ref,pc->p_iter);
            pc->hasconverged=1;
            pc->hasoptstarted=1;
        }
        OptimizeComputeErrImg();
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::LossComputeErrorImage(Eigen::Matrix<float,Eigen::Dynamic,1>* patdest,
                                            Eigen::Matrix<float,Eigen::Dynamic,1>* wdest,
                                            const Eigen::Matrix<float,Eigen::Dynamic,1>* patin,
                                            const Eigen::Matrix<float,Eigen::Dynamic,1>* tmpin) {
    v4sf * pd = (v4sf*) patdest->data(),
         * pa = (v4sf*) patin->data(),
         * te = (v4sf*) tmpin->data(),
         * pw = (v4sf*) wdest->data();

    if(op->costfct==0) { // L2 cost function
        for(int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw) {
          (*pd) = (*pa)-(*te);  // difference image
          (*pw) = __builtin_ia32_andnps(op->negzero,(*pd));
        }
    }
    else if(op->costfct==1) { // L1 cost function
        for(int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw) {
            (*pd) = (*pa)-(*te);   // difference image
            (*pd) = __builtin_ia32_orps( __builtin_ia32_andps(op->negzero,  (*pd) )  , __builtin_ia32_sqrtps (__builtin_ia32_andnps(op->negzero,  (*pd) )) );  // sign(pdiff) * sqrt(abs(pdiff))
            (*pw) = __builtin_ia32_andnps(op->negzero,  (*pd) );
        }
    }
    else if(op->costfct==2) { // Pseudo Huber cost function
        for (int i=op->novals/4; i--; ++pd, ++pa, ++te, ++pw) {
            (*pd) = (*pa)-(*te); // difference image
            (*pd) = __builtin_ia32_orps(
                        __builtin_ia32_andps(op->negzero,(*pd)),
                        __builtin_ia32_sqrtps(
                            // PSEUDO HUBER NORM
                            __builtin_ia32_mulps(
                                __builtin_ia32_sqrtps(
                                    op->ones+__builtin_ia32_divps(__builtin_ia32_mulps((*pd),(*pd)),
                                    op->normoutlier_tmpbsq)
                                )-op->ones,
                                op->normoutlier_tmp2bsq
                            )
                        )
                    ); // sign(pdiff) * sqrt( 2*b^2*( sqrt(1+abs(pdiff)^2/b^2)+1)  )) // <- looks like this without SSE instruction
            (*pw) = __builtin_ia32_andnps(op->negzero,(*pd));
        }
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::OptimizeComputeErrImg() {
    getPatchStaticBil(im_bo->data(), &(pc->pt_iter), &(pc->pdiff));
    // Get photometric patch error
    LossComputeErrorImage(&pc->pdiff, &pc->pweight, &pc->pdiff, &tmp);
    // Compute step norm
    pc->delta_p_sqnorm = pc->delta_p.squaredNorm();
    if(pc->cnt==1)
        pc->delta_p_sqnorm_init = pc->delta_p_sqnorm;
    // Check early termination criterions
    pc->mares_old = pc->mares;
    pc->mares = pc->pweight.template lpNorm<1>() / (op->novals);
    if( ! ((pc->cnt < op->max_iter) & (pc->mares  > op->res_thresh) &
          ((pc->cnt < op->min_iter) | (pc->delta_p_sqnorm / pc->delta_p_sqnorm_init >= op->dp_thresh)) &
          ((pc->cnt < op->min_iter) | (pc->mares / pc->mares_old <= op->dr_thresh))) )
        pc->hasconverged=1;
}

// Extract patch on integer position, and gradients, No Bilinear interpolation
template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::getPatchStaticNNGrad(const float* img,
                                           const float* img_dx,
                                           const float* img_dy,
                                           const Eigen::Vector2f* mid_in,
                                           Eigen::Matrix<float,Eigen::Dynamic,1>* tmp_in_e,
                                           Eigen::Matrix<float,Eigen::Dynamic,1>* tmp_dx_in_e,
                                           Eigen::Matrix<float,Eigen::Dynamic,1>* tmp_dy_in_e) {
    float* tmp_in = tmp_in_e->data();
    float* tmp_dx_in = tmp_dx_in_e->data();
    float* tmp_dy_in = tmp_dy_in_e->data();
    Eigen::Vector2i pos;
    Eigen::Vector2i pos_it;
    pos[0] = round((*mid_in)[0])+cpt->imgpadding;
    pos[1] = round((*mid_in)[1])+cpt->imgpadding;
    int posxx = 0;
    int lb = -op->p_samp_s/2;
    int ub = op->p_samp_s/2-1;
    for(int j = lb; j<=ub; ++j) {
        for(int i = lb; i<=ub; ++i,++posxx) {
            pos_it[0] = pos[0]+i;
            pos_it[1] = pos[1]+j;
            int idx = pos_it[0]+pos_it[1]*cpt->tmp_w;
            if(eInput==ofdis::FlowInput_RGB) {
                idx *= 3;
                tmp_in[posxx] = img[idx];
                tmp_dx_in[posxx] = img_dx[idx];
                tmp_dy_in[posxx] = img_dy[idx];
                ++posxx;
                ++idx;
                tmp_in[posxx] = img[idx];
                tmp_dx_in[posxx] = img_dx[idx];
                tmp_dy_in[posxx] = img_dy[idx];
                ++posxx;
                ++idx;
                tmp_in[posxx] = img[idx];
                tmp_dx_in[posxx] = img_dx[idx];
                tmp_dy_in[posxx] = img_dy[idx];
            }
            else {
                tmp_in[posxx] = img[idx];
                tmp_dx_in[posxx] = img_dx[idx];
                tmp_dy_in[posxx] = img_dy[idx];
            }
        }
    }
    // PATCH NORMALIZATION
    if(op->patnorm>0)
        tmp_in_e->array() -= (tmp_in_e->sum()/op->novals);
}

// Extract patch on float position with bilinear interpolation, no gradients.
template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatClass<eInput,eOutput>::getPatchStaticBil(const float* img, const Eigen::Vector2f* mid_in, Eigen::Matrix<float,Eigen::Dynamic,1>* tmp_in_e) {
    float* tmp_in = tmp_in_e->data();
    Eigen::Vector2f resid;
    Eigen::Vector4f we; // bilinear weight vector
    Eigen::Vector4i pos;
    Eigen::Vector2i pos_it;
    // Compute the bilinear weight vector, for patch without orientation/scale change -> weight vector is constant for all pixels
    pos[0] = ceil((*mid_in)[0]+.00001f); // ensure rounding up to natural numbers
    pos[1] = ceil((*mid_in)[1]+.00001f);
    pos[2] = floor((*mid_in)[0]);
    pos[3] = floor((*mid_in)[1]);
    resid[0] = (*mid_in)[0]-(float)pos[2];
    resid[1] = (*mid_in)[1]-(float)pos[3];
    we[0] = resid[0]*resid[1];
    we[1] = (1-resid[0])*resid[1];
    we[2] = resid[0]*(1-resid[1]);
    we[3] = (1-resid[0])*(1-resid[1]);
    pos[0] += cpt->imgpadding;
    pos[1] += cpt->imgpadding;
    float* tmp_it = tmp_in;
    const float* img_a,* img_b,* img_c,* img_d,* img_e;
    if(eInput==ofdis::FlowInput_RGB)
        img_e = img+(pos[0]-op->p_samp_s/2)*3;
    else
        img_e = img+pos[0]-op->p_samp_s/2;
    int lb = -op->p_samp_s/2;
    int ub = op->p_samp_s/2-1;
    for(pos_it[1] = pos[1]+lb; pos_it[1]<=pos[1]+ub; ++pos_it[1]) {
        if(eInput==ofdis::FlowInput_RGB) {
            img_a = img_e+pos_it[1]*cpt->tmp_w*3;
            img_c = img_e+(pos_it[1]-1)*cpt->tmp_w*3;
            img_b = img_a-3;
            img_d = img_c-3;
        }
        else {
            img_a = img_e+pos_it[1]*cpt->tmp_w;
            img_c = img_e+(pos_it[1]-1)*cpt->tmp_w;
            img_b = img_a-1;
            img_d = img_c-1;
        }
        for(pos_it[0] = pos[0]+lb; pos_it[0]<=pos[0]+ub; ++pos_it[0],++tmp_it,++img_a,++img_b,++img_c,++img_d) {
            if(eInput==ofdis::FlowInput_RGB) {
                (*tmp_it) = we[0]*(*img_a)+we[1]*(*img_b)+we[2]*(*img_c)+we[3]*(*img_d);
                ++tmp_it;
                ++img_a;
                ++img_b;
                ++img_c;
                ++img_d;
                (*tmp_it) = we[0]*(*img_a)+we[1]*(*img_b)+we[2]*(*img_c)+we[3]*(*img_d);
                ++tmp_it;
                ++img_a;
                ++img_b;
                ++img_c;
                ++img_d;
                (*tmp_it) = we[0]*(*img_a)+we[1]*(*img_b)+we[2]*(*img_c)+we[3]*(*img_d);
            }
            else {
                (*tmp_it) = we[0]*(*img_a)+we[1]*(*img_b)+we[2]*(*img_c)+we[3]*(*img_d);
            }
        }
    }
    // PATCH NORMALIZATION
    if(op->patnorm>0) // Subtract Mean
        tmp_in_e->array() -= (tmp_in_e->sum()/op->novals);
}

template class ofdis::PatClass<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::PatClass<ofdis::FlowInput_Gradient,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::PatClass<ofdis::FlowInput_RGB,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::PatClass<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_StereoDepth>;
template class ofdis::PatClass<ofdis::FlowInput_Gradient,ofdis::FlowOutput_StereoDepth>;
template class ofdis::PatClass<ofdis::FlowInput_RGB,ofdis::FlowOutput_StereoDepth>;