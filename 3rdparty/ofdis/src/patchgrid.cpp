
#define OFDIS_INTERNAL
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include "litiv/3rdparty/ofdis/fdf/image.h"
#include "litiv/3rdparty/ofdis/patch.hpp"
#include "litiv/3rdparty/ofdis/patchgrid.hpp"
#include "litiv/utils/defines.hpp" // only used here for compiler flags

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
ofdis::PatGridClass<eInput,eOutput>::PatGridClass(
        const camparam* cpt_in,
        const camparam* cpo_in,
        const optparam* op_in) :
        cpt(cpt_in),
        cpo(cpo_in),
        op(op_in) {
    // Generate grid on current scale
    steps = op->steps;
    nopw = ceil((float)cpt->width/(float)steps);
    noph = ceil((float)cpt->height/(float)steps);
    const int offsetw = floor((cpt->width-(nopw-1)*steps)/2);
    const int offseth = floor((cpt->height-(noph-1)*steps)/2);
    nopatches = nopw*noph;
    pt_ref.resize(nopatches);
    p_init.resize(nopatches);
    pat.reserve(nopatches);
    im_ao_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
    im_ao_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
    im_ao_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
    im_bo_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
    im_bo_dx_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
    im_bo_dy_eg = new Eigen::Map<const Eigen::MatrixXf>(nullptr,cpt->height,cpt->width);
    int patchid = 0;
    for(int x=0; x<nopw; ++x) {
        for(int y=0; y<noph; ++y) {
            int i = x*noph+y;
            pt_ref[i][0] = x*steps+offsetw;
            pt_ref[i][1] = y*steps+offseth;
            p_init[i].setZero();
            pat.push_back(new PatClass<eInput,eOutput>(cpt,cpo,op,patchid));
            patchid++;
        }
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
ofdis::PatGridClass<eInput,eOutput>::~PatGridClass() {
    delete im_ao_eg;
    delete im_ao_dx_eg;
    delete im_ao_dy_eg;
    delete im_bo_eg;
    delete im_bo_dx_eg;
    delete im_bo_dy_eg;
    for(int i=0; i< nopatches; ++i)
        delete pat[i];
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatGridClass<eInput,eOutput>::SetComplGrid(PatGridClass *cg_in) {
    cg = cg_in;
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatGridClass<eInput,eOutput>::InitializeGrid(const float* im_ao_in, const float* im_ao_dx_in, const float* im_ao_dy_in) {
    im_ao = im_ao_in;
    im_ao_dx = im_ao_dx_in;
    im_ao_dy = im_ao_dy_in;
    new(im_ao_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao,cpt->height,cpt->width); // new placement operator
    new(im_ao_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dx,cpt->height,cpt->width);
    new(im_ao_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_ao_dy,cpt->height,cpt->width);
#if USING_OPENMP
    #pragma omp parallel for schedule(static)
#endif //USING_OPENMP
    for (int i = 0; i < nopatches; ++i) {
        pat[i]->InitializePatch(im_ao_eg, im_ao_dx_eg, im_ao_dy_eg, pt_ref[i]);
        p_init[i].setZero();
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatGridClass<eInput,eOutput>::SetTargetImage(const float* im_bo_in, const float* im_bo_dx_in, const float* im_bo_dy_in) {
    im_bo = im_bo_in;
    im_bo_dx = im_bo_dx_in;
    im_bo_dy = im_bo_dy_in;
    new(im_bo_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo,cpt->height,cpt->width); // new placement operator
    new(im_bo_dx_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dx,cpt->height,cpt->width); // new placement operator
    new(im_bo_dy_eg) Eigen::Map<const Eigen::MatrixXf>(im_bo_dy,cpt->height,cpt->width); // new placement operator
#if USING_OPENMP
    #pragma omp parallel for schedule(static)
#endif //USING_OPENMP
    for(int i = 0; i < nopatches; ++i)
        pat[i]->SetTargetImage(im_bo_eg, im_bo_dx_eg, im_bo_dy_eg);
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatGridClass<eInput,eOutput>::Optimize() {
#if USING_OPENMP
    #pragma omp parallel for schedule(dynamic,10)
#endif //USING_OPENMP
    for(int i = 0; i < nopatches; ++i)
        pat[i]->OptimizeIter(p_init[i], true); // optimize until convergence
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatGridClass<eInput,eOutput>::InitializeFromCoarserOF(const float* flow_prev) {
#if USING_OPENMP
    #pragma omp parallel for schedule(dynamic,10)
#endif //USING_OPENMP
    for (int ip = 0; ip < nopatches; ++ip) {
        int x = floor(pt_ref[ip][0] / 2); // better, but slower: use bil. interpolation here
        int y = floor(pt_ref[ip][1] / 2);
        int i = y*(cpt->width/2) + x;
        if(eOutput==ofdis::FlowOutput_OpticalFlow) {
            p_init[ip](0) = flow_prev[2*i]*2;
            p_init[ip](1) = flow_prev[2*i+1]*2;
        }
        else
            p_init[ip](0) = flow_prev[i]*2;
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::PatGridClass<eInput,eOutput>::AggregateFlowDense(float *flowout) const {
    float* we = new float[cpt->width * cpt->height];
    memset(flowout, 0, sizeof(float) * (op->nop * cpt->width * cpt->height) );
    memset(we,      0, sizeof(float) * (          cpt->width * cpt->height) );
#ifdef USE_PARALLEL_ON_FLOWAGGR // Using this enables OpenMP on flow aggregation. This can lead to race conditions. Experimentally we found that the result degrades only marginally. However, for our experiments we did not enable this.
    #pragma omp parallel for schedule(static)
#endif
    for(int ip = 0; ip < nopatches; ++ip) {
        if(pat[ip]->IsValid()) {
            const typename p_init_type::value_type* fl = pat[ip]->GetParam(); // flow/horiz displacement of this patch
            typename p_init_type::value_type flnew;
            const float* pweight = pat[ip]->GetpWeightPtr(); // use image error as weight
            int lb = -op->p_samp_s/2;
            int ub = op->p_samp_s/2-1;
            for(int y = lb; y <= ub; ++y) {
                for(int x = lb; x <= ub; ++x, ++pweight) {
                    int yt = (y + pt_ref[ip][1]);
                    int xt = (x + pt_ref[ip][0]);
                    if (xt >= 0 && yt >= 0 && xt < cpt->width && yt < cpt->height) {
                        int i = yt*cpt->width + xt;
                        float absw;
                        if(eInput==ofdis::FlowInput_RGB) {
                            absw = (float)(std::max(op->minerrval,*pweight)); ++pweight;
                            absw+= (float)(std::max(op->minerrval,*pweight)); ++pweight;
                            absw+= (float)(std::max(op->minerrval,*pweight));
                            absw = 1.0f/absw;
                        }
                        else
                            absw = 1.0f/(float)(std::max(op->minerrval,*pweight));
                        flnew = (*fl) * absw;
                        we[i] += absw;
                        if(eOutput==ofdis::FlowOutput_OpticalFlow) {
                            flowout[2*i]   += flnew[0];
                            flowout[2*i+1] += flnew[1];
                        }
                        else
                            flowout[i] += flnew[0];
                    }
                }
            }
        }
    }

    // if complementary (forward-backward merging) is given, integrate negative backward flow as well
    if(cg) {
        Eigen::Vector4f wbil; // bilinear weight vector
        Eigen::Vector4i pos;
#if USING_OPENMP
    #ifdef USE_PARALLEL_ON_FLOWAGGR
        #pragma omp parallel for schedule(static)
    #endif
#endif //USING_OPENMP
        for(int ip = 0; ip < cg->nopatches; ++ip) {
            if (cg->pat[ip]->IsValid()) {
                const typename p_init_type::value_type* fl = (cg->pat[ip]->GetParam()); // flow/horiz displacement of this patch
                typename p_init_type::value_type flnew;
                const Eigen::Vector2f rppos = cg->pat[ip]->GetPointPos(); // get patch position after optimization
                const float* pweight = cg->pat[ip]->GetpWeightPtr(); // use image error as weight
                Eigen::Vector2f resid;
                // compute bilinear weight vector
                pos[0] = ceil(rppos[0] +.00001); // make sure they are rounded up to natural number
                pos[1] = ceil(rppos[1] +.00001); // make sure they are rounded up to natural number
                pos[2] = floor(rppos[0]);
                pos[3] = floor(rppos[1]);
                resid[0] = rppos[0] - pos[2];
                resid[1] = rppos[1] - pos[3];
                wbil[0] = resid[0]*resid[1];
                wbil[1] = (1-resid[0])*resid[1];
                wbil[2] = resid[0]*(1-resid[1]);
                wbil[3] = (1-resid[0])*(1-resid[1]);
                int lb = -op->p_samp_s/2;
                int ub = op->p_samp_s/2-1;
                for(int y = lb; y <= ub; ++y) {
                    for(int x = lb; x <= ub; ++x, ++pweight) {
                        int yt = y + pos[1];
                        int xt = x + pos[0];
                        if(xt >= 1 && yt >= 1 && xt < (cpt->width-1) && yt < (cpt->height-1)) {
                            float absw;
                            if(eInput==ofdis::FlowInput_RGB) {
                                absw = (float)(std::max(op->minerrval,*pweight)); ++pweight;
                                absw+= (float)(std::max(op->minerrval,*pweight)); ++pweight;
                                absw+= (float)(std::max(op->minerrval,*pweight));
                                absw = 1.0f/absw;
                            }
                            else
                                absw = 1.0f/(float)(std::max(op->minerrval,*pweight));
                            flnew = (*fl) * absw;
                            int idxcc =  xt    +  yt   *cpt->width;
                            int idxfc = (xt-1) +  yt   *cpt->width;
                            int idxcf =  xt    + (yt-1)*cpt->width;
                            int idxff = (xt-1) + (yt-1)*cpt->width;
                            we[idxcc] += wbil[0] * absw;
                            we[idxfc] += wbil[1] * absw;
                            we[idxcf] += wbil[2] * absw;
                            we[idxff] += wbil[3] * absw;
                            if(eOutput==ofdis::FlowOutput_OpticalFlow) {
                                flowout[2*idxcc  ] -= wbil[0] * flnew[0];   // use reversed flow
                                flowout[2*idxcc+1] -= wbil[0] * flnew[1];
                                flowout[2*idxfc  ] -= wbil[1] * flnew[0];
                                flowout[2*idxfc+1] -= wbil[1] * flnew[1];
                                flowout[2*idxcf  ] -= wbil[2] * flnew[0];
                                flowout[2*idxcf+1] -= wbil[2] * flnew[1];
                                flowout[2*idxff  ] -= wbil[3] * flnew[0];
                                flowout[2*idxff+1] -= wbil[3] * flnew[1];
                            }
                            else {
                                flowout[idxcc] -= wbil[0] * flnew[0]; // simple averaging of inverse horizontal displacement
                                flowout[idxfc] -= wbil[1] * flnew[0];
                                flowout[idxcf] -= wbil[2] * flnew[0];
                                flowout[idxff] -= wbil[3] * flnew[0];
                            }
                        }
                    }
                }
            }
        }
    }
#if USING_OPENMP
    #pragma omp parallel for schedule(static, 100)
#endif //USING_OPENMP
    // normalize each pixel by dividing displacement by aggregated weights from all patches
    for (int yi = 0; yi < cpt->height; ++yi) {
        for (int xi = 0; xi < cpt->width; ++xi) {
            int i = yi*cpt->width + xi;
            if (we[i]>0) {
                if(eOutput==ofdis::FlowOutput_OpticalFlow) {
                    flowout[2*i  ] /= we[i];
                    flowout[2*i+1] /= we[i];
                }
                else
                    flowout[i] /= we[i];
            }
        }
    }
    delete[] we;
}

template class ofdis::PatGridClass<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::PatGridClass<ofdis::FlowInput_Gradient,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::PatGridClass<ofdis::FlowInput_RGB,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::PatGridClass<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_StereoDepth>;
template class ofdis::PatGridClass<ofdis::FlowInput_Gradient,ofdis::FlowOutput_StereoDepth>;
template class ofdis::PatGridClass<ofdis::FlowInput_RGB,ofdis::FlowOutput_StereoDepth>;