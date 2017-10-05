
#define OFDIS_INTERNAL
#include <Eigen/Core>
#include "litiv/3rdparty/ofdis/refine_variational.hpp"

inline void local_image_delete(image_t *src) {
    image_delete(src);
}

inline void local_image_delete(color_image_t *src) {
    color_image_delete(src);
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
ofdis::VarRefClass<eInput,eOutput>::VarRefClass(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in,
                                                const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in,
                                                const camparam* cpt_in,const camparam* cpo_in,const optparam* op_in, float *flowout) :
        cpt(cpt_in), cpo(cpo_in), op(op_in) {
    // initialize parameters
    tvparams.alpha = op->tv_alpha;
    tvparams.beta = 0.0f;  // for matching term, not needed for us
    tvparams.gamma = op->tv_gamma;
    tvparams.delta = op->tv_delta;
    tvparams.n_inner_iteration = op->tv_innerit * (cpt->curr_lv+1);
    tvparams.n_solver_iteration = op->tv_solverit;//5;
    tvparams.sor_omega = op->tv_sor;
    tvparams.tmp_quarter_alpha = 0.25f*tvparams.alpha;
    tvparams.tmp_half_gamma_over3 = tvparams.gamma*0.5f/3.0f;
    tvparams.tmp_half_delta_over3 = tvparams.delta*0.5f/3.0f;
    tvparams.tmp_half_beta = tvparams.beta*0.5f;
    float deriv_filter[3] = {0.0f, -8.0f/12.0f, 1.0f/12.0f};
    deriv = convolution_new(2, deriv_filter, 0);
    float deriv_filter_flow[2] = {0.0f, -0.5f};
    deriv_flow = convolution_new(1, deriv_filter_flow, 0);
    // copy flow initialization into FV structs
    static int noparam = (eOutput==ofdis::FlowOutput_OpticalFlow)?2:1; // only horizontal displacements for stereo depth
    std::vector<image_t*> flow_sep(noparam);
    for(int i = 0; i < noparam; ++i )
        flow_sep[i] = image_new(cpt->width,cpt->height);
    for(int iy = 0; iy < cpt->height; ++iy) {
        for(int ix = 0; ix<cpt->width; ++ix) {
            int i = iy*cpt->width+ix;
            int is = iy*flow_sep[0]->stride+ix;
            for(int j = 0; j<noparam; ++j)
                flow_sep[j]->c1[is] = flowout[i*noparam+j];
        }
    }
    // copy image data into FV structs
    InputImageType* im_ao, *im_bo;
    im_ao = (InputImageType*)((eInput==ofdis::FlowInput_RGB)?(void*)color_image_new(cpt->width,cpt->height):(void*)image_new(cpt->width,cpt->height));
    im_bo = (InputImageType*)((eInput==ofdis::FlowInput_RGB)?(void*)color_image_new(cpt->width,cpt->height):(void*)image_new(cpt->width,cpt->height));
    copyimage(im_ao_in, im_ao);
    copyimage(im_bo_in, im_bo);
    // call solver
    if(eOutput==ofdis::FlowOutput_OpticalFlow)
        RefLevelOF(flow_sep[0], flow_sep[1], im_ao, im_bo);
    else
        RefLevelDE(flow_sep[0], im_ao, im_bo);
    // copy flow result back
    for(int iy = 0; iy < cpt->height; ++iy) {
        for(int ix = 0; ix<cpt->width; ++ix) {
            int i = iy*cpt->width+ix;
            int is = iy*flow_sep[0]->stride+ix;
            for(int j = 0; j<noparam; ++j)
                flowout[i*noparam+j] = flow_sep[j]->c1[is];
        }
    }
    // free FV structs
    for(int i = 0; i < noparam; ++i)
        image_delete(flow_sep[i]);
    convolution_delete(deriv);
    convolution_delete(deriv_flow);
    local_image_delete(im_ao);
    local_image_delete(im_bo);
}

inline void local_image_copy_pixel(image_t *img_t, int i, const float* img_st) {
    img_t->c1[i] = (*img_st);
}

inline void local_image_copy_pixel(color_image_t *img_t, int i, const float*& img_st) {
    img_t->c1[i] = (*img_st);
    ++img_st; img_t->c2[i] =  (*img_st);
    ++img_st; img_t->c3[i] =  (*img_st);
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::VarRefClass<eInput,eOutput>::copyimage(const float* img, InputImageType* img_t) {
    // remove image padding, start at first valid pixel
    const float* img_st = img+((eInput==ofdis::FlowInput_RGB)?3:1)*(cpt->tmp_w+1)*(cpt->imgpadding);
    for(int yi = 0; yi<cpt->height; ++yi) {
        for(int xi = 0; xi<cpt->width; ++xi,++img_st) {
            local_image_copy_pixel(img_t,yi*img_t->stride+xi,img_st);
        }
        img_st += ((eInput==ofdis::FlowInput_RGB)?3:1)*2*cpt->imgpadding;
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::VarRefClass<eInput,eOutput>::RefLevelOF(image_t *wx, image_t *wy, const InputImageType* im1, const InputImageType* im2) {
    int i_inner_iteration;
    int width  = wx->width;
    int height = wx->height;
    int stride = wx->stride;
    image_t *du = image_new(width,height), *dv = image_new(width,height), // the flow increment
            *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
            *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), // horiz: (i,j) contains the diffusivity coeff. from (i,j) to (i+1,j)
            *uu = image_new(width,height), *vv = image_new(width,height), // flow plus flow increment
            *a11 = image_new(width,height), *a12 = image_new(width,height), *a22 = image_new(width,height), // system matrix A of Ax=b for each pixel
            *b1 = image_new(width,height), *b2 = image_new(width,height); // system matrix b of Ax=b for each pixel
    const auto lImageCreator = [&](){return (InputImageType*)((eInput==ofdis::FlowInput_RGB)?(void*)color_image_new(width,height):(void*)image_new(width,height));};
    InputImageType *w_im2 = lImageCreator(), // warped second image
                   *Ix = lImageCreator(), *Iy = lImageCreator(), *Iz = lImageCreator(), // first order derivatives
                   *Ixx = lImageCreator(), *Ixy = lImageCreator(), *Iyy = lImageCreator(), *Ixz = lImageCreator(), *Iyz = lImageCreator(); // second order derivatives
    // warp second image
    fdf::image_warp<eInput>(w_im2, mask, im2, wx, wy);
    // compute derivatives
    fdf::get_derivatives<eInput>(im1, w_im2, deriv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
    // erase du and dv
    image_erase(du);
    image_erase(dv);
    // initialize uu and vv
    memcpy(uu->c1,wx->c1,wx->stride*wx->height*sizeof(float));
    memcpy(vv->c1,wy->c1,wy->stride*wy->height*sizeof(float));
    // inner fixed point iterations
    for(i_inner_iteration=0; i_inner_iteration<tvparams.n_inner_iteration; i_inner_iteration++) {
        //  compute robust function and system
        fdf::compute_smoothness(smooth_horiz, smooth_vert, uu, vv, deriv_flow, tvparams.tmp_quarter_alpha );
        //compute_data_and_match(a11, a12, a22, b1, b2, mask, wx, wy, du, dv, uu, vv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, desc_weight, desc_flow_x, desc_flow_y, tvparams.tmp_half_delta_over3, tvparams.tmp_half_beta, tvparams.tmp_half_gamma_over3);
        fdf::compute_data<eInput>(a11, a12, a22, b1, b2, mask, wx, wy, du, dv, uu, vv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, tvparams.tmp_half_delta_over3, tvparams.tmp_half_beta, tvparams.tmp_half_gamma_over3);
        fdf::sub_laplacian(b1, wx, smooth_horiz, smooth_vert);
        fdf::sub_laplacian(b2, wy, smooth_horiz, smooth_vert);
        // solve system
    #ifdef WITH_OPENMP
        sor_coupled_slow_but_readable(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, tvparams.n_solver_iteration, tvparams.sor_omega); // slower but parallelized
    #else
        sor_coupled(du, dv, a11, a12, a22, b1, b2, smooth_horiz, smooth_vert, tvparams.n_solver_iteration, tvparams.sor_omega);
    #endif
        // update flow plus flow increment
        int i;
        v4sf *uup = (v4sf*) uu->c1, *vvp = (v4sf*) vv->c1, *wxp = (v4sf*) wx->c1, *wyp = (v4sf*) wy->c1, *dup = (v4sf*) du->c1, *dvp = (v4sf*) dv->c1;
        for(i=0 ; i<height*stride/4 ; i++) {
            (*uup) = (*wxp) + (*dup);
            (*vvp) = (*wyp) + (*dvp);
            uup+=1; vvp+=1; wxp+=1; wyp+=1;dup+=1;dvp+=1;
        }
    }
    // add flow increment to current flow
    memcpy(wx->c1,uu->c1,uu->stride*uu->height*sizeof(float));
    memcpy(wy->c1,vv->c1,vv->stride*vv->height*sizeof(float));
    // free memory
    image_delete(du);
    image_delete(dv);
    image_delete(mask);
    image_delete(smooth_horiz);
    image_delete(smooth_vert);
    image_delete(uu);
    image_delete(vv);
    image_delete(a11);
    image_delete(a12);
    image_delete(a22);
    image_delete(b1);
    image_delete(b2);
    local_image_delete(w_im2);
    local_image_delete(Ix);
    local_image_delete(Iy);
    local_image_delete(Iz);
    local_image_delete(Ixx);
    local_image_delete(Ixy);
    local_image_delete(Iyy);
    local_image_delete(Ixz);
    local_image_delete(Iyz);
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::VarRefClass<eInput,eOutput>::RefLevelDE(image_t *wx, const InputImageType* im1, const InputImageType* im2) {
    int i_inner_iteration;
    int width  = wx->width;
    int height = wx->height;
    int stride = wx->stride;
    image_t *du = image_new(width,height), *wy_dummy = image_new(width,height), // the flow increment
            *mask = image_new(width,height), // mask containing 0 if a point goes outside image boundary, 1 otherwise
            *smooth_horiz = image_new(width,height), *smooth_vert = image_new(width,height), // horiz: (i,j) contains the diffusivity coeff. from (i,j) to (i+1,j)
            *uu = image_new(width,height), // flow plus flow increment
            *a11 = image_new(width,height), // system matrix A of Ax=b for each pixel
            *b1 = image_new(width,height); // system matrix b of Ax=b for each pixel
    image_erase(wy_dummy);
    const auto lImageCreator = [&](){return (InputImageType*)((eInput==ofdis::FlowInput_RGB)?(void*)color_image_new(width,height):(void*)image_new(width,height));};
    InputImageType *w_im2 = lImageCreator(), // warped second image
                   *Ix = lImageCreator(), *Iy = lImageCreator(), *Iz = lImageCreator(), // first order derivatives
                   *Ixx = lImageCreator(), *Ixy = lImageCreator(), *Iyy = lImageCreator(), *Ixz = lImageCreator(), *Iyz = lImageCreator(); // second order derivatives
    // warp second image
    fdf::image_warp<eInput>(w_im2, mask, im2, wx, wy_dummy);
    // compute derivatives
    fdf::get_derivatives<eInput>(im1, w_im2, deriv, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz);
    // erase du and dv
    image_erase(du);
    // initialize uu and vv
    memcpy(uu->c1,wx->c1,wx->stride*wx->height*sizeof(float));
    // inner fixed point iterations
    for(i_inner_iteration=0; i_inner_iteration<tvparams.n_inner_iteration; i_inner_iteration++) {
        //  compute robust function and system
        fdf::compute_smoothness(smooth_horiz, smooth_vert, uu, wy_dummy, deriv_flow, tvparams.tmp_quarter_alpha );
        fdf::compute_data_DE<eInput>(a11, b1, mask, wx, du, uu, Ix, Iy, Iz, Ixx, Ixy, Iyy, Ixz, Iyz, tvparams.tmp_half_delta_over3, tvparams.tmp_half_beta, tvparams.tmp_half_gamma_over3);
        fdf::sub_laplacian(b1, wx, smooth_horiz, smooth_vert);
        // solve system
        sor_coupled_slow_but_readable_DE(du, a11, b1, smooth_horiz, smooth_vert, tvparams.n_solver_iteration, tvparams.sor_omega);
        // update flow plus flow increment
        int i;
        v4sf *uup = (v4sf*) uu->c1, *wxp = (v4sf*) wx->c1, *dup = (v4sf*) du->c1;
        if(cpt->camlr==0) { // check if right or left camera, needed to truncate values above/below zero
            for(i=0; i<height*stride/4; i++) {
                (*uup) = __builtin_ia32_minps(   (*wxp) + (*dup)   ,  op->zero);
                uup+=1; wxp+=1; dup+=1;
            }
        }
        else {
            for(i=0 ; i<height*stride/4; i++) {
                (*uup) = __builtin_ia32_maxps(   (*wxp) + (*dup)   ,  op->zero);
                uup+=1; wxp+=1; dup+=1;
            }
        }
    }
    // add flow increment to current flow
    memcpy(wx->c1,uu->c1,uu->stride*uu->height*sizeof(float));
    // free memory
    image_delete(du);
    image_delete(wy_dummy);
    image_delete(mask);
    image_delete(smooth_horiz);
    image_delete(smooth_vert);
    image_delete(uu);
    image_delete(a11);
    image_delete(b1);
    local_image_delete(w_im2);
    local_image_delete(Ix);
    local_image_delete(Iy);
    local_image_delete(Iz);
    local_image_delete(Ixx);
    local_image_delete(Ixy);
    local_image_delete(Iyy);
    local_image_delete(Ixz);
    local_image_delete(Iyz);
}

template class ofdis::VarRefClass<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::VarRefClass<ofdis::FlowInput_Gradient,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::VarRefClass<ofdis::FlowInput_RGB,ofdis::FlowOutput_OpticalFlow>;
template class ofdis::VarRefClass<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_StereoDepth>;
template class ofdis::VarRefClass<ofdis::FlowInput_Gradient,ofdis::FlowOutput_StereoDepth>;
template class ofdis::VarRefClass<ofdis::FlowInput_RGB,ofdis::FlowOutput_StereoDepth>;