
#define OFDIS_INTERNAL
#include <cstdlib>
#include <cmath>
#include <malloc.h>
#include <cstring>
#include "litiv/3rdparty/ofdis/fdf/image.h"
#include "litiv/3rdparty/ofdis/fdf/opticalflow_aux.hpp"

using ofdis::v4sf;

#define datanorm (0.1f*0.1f)//0.01f // square of the normalization factor
#define epsilon_color (0.001f*0.001f)//0.000001f
#define epsilon_grad (0.001f*0.001f)//0.000001f
#define epsilon_desc (0.001f*0.001f)//0.000001f
#define epsilon_smooth (0.001f*0.001f)//0.000001f

inline void local_warp_pixel(color_image_t* dst, int offset, const color_image_t* src, int y1, int x1, int y2, int x2, float dy, float dx) {
    dst->c1[offset] =
            src->c1[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
            src->c1[y1*src->stride+x2]*dx*(1.0f-dy) +
            src->c1[y2*src->stride+x1]*(1.0f-dx)*dy +
            src->c1[y2*src->stride+x2]*dx*dy;
    dst->c2[offset] =
            src->c2[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
            src->c2[y1*src->stride+x2]*dx*(1.0f-dy) +
            src->c2[y2*src->stride+x1]*(1.0f-dx)*dy +
            src->c2[y2*src->stride+x2]*dx*dy;
    dst->c3[offset] =
            src->c3[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
            src->c3[y1*src->stride+x2]*dx*(1.0f-dy) +
            src->c3[y2*src->stride+x1]*(1.0f-dx)*dy +
            src->c3[y2*src->stride+x2]*dx*dy;
}

inline void local_warp_pixel(image_t* dst, int offset, const image_t* src, int y1, int x1, int y2, int x2, float dy, float dx) {
    dst->c1[offset] =
            src->c1[y1*src->stride+x1]*(1.0f-dx)*(1.0f-dy) +
            src->c1[y1*src->stride+x2]*dx*(1.0f-dy) +
            src->c1[y2*src->stride+x1]*(1.0f-dx)*dy +
            src->c1[y2*src->stride+x2]*dx*dy;
}

/* warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries */
template<ofdis::FlowInputType eInput>
void fdf::image_warp(ofdis::InputImageType<eInput>* dst, image_t *mask, const ofdis::InputImageType<eInput>* src, const image_t *wx, const image_t *wy) {
    int i, j, offset, incr_line = mask->stride-mask->width, x, y, x1, x2, y1, y2;
    float xx, yy, dx, dy;
    for(j=0,offset=0 ; j<src->height ; j++)
    {
        for(i=0 ; i<src->width ; i++,offset++)
        {
	        xx = i+wx->c1[offset];
	        yy = j+wy->c1[offset];
	        x = floor(xx);
	        y = floor(yy);
	        dx = xx-x;
	        dy = yy-y;
	        mask->c1[offset] = (xx>=0 && xx<=src->width-1 && yy>=0 && yy<=src->height-1);
	        x1 = MINMAX_TA(x,src->width);
	        x2 = MINMAX_TA(x+1,src->width);
	        y1 = MINMAX_TA(y,src->height);
	        y2 = MINMAX_TA(y+1,src->height);
            local_warp_pixel(dst,offset,src,y1,x1,y2,x2,dy,dx);
	    }
        offset += incr_line;
    }
}

template void fdf::image_warp<ofdis::FlowInput_Grayscale>(ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, image_t*, const ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, const image_t*, const image_t*);
template void fdf::image_warp<ofdis::FlowInput_Gradient>(ofdis::InputImageType<ofdis::FlowInput_Gradient>*, image_t*, const ofdis::InputImageType<ofdis::FlowInput_Gradient>*, const image_t*, const image_t*);
template void fdf::image_warp<ofdis::FlowInput_RGB>(ofdis::InputImageType<ofdis::FlowInput_RGB>*, image_t*, const ofdis::InputImageType<ofdis::FlowInput_RGB>*, const image_t*, const image_t*);

inline void local_image_convolve_hv(color_image_t *dst, const color_image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv) {
    color_image_convolve_hv(dst,src,horiz_conv,vert_conv);
}

inline void local_image_convolve_hv(image_t *dst, const image_t *src, const convolution_t *horiz_conv, const convolution_t *vert_conv) {
    image_convolve_hv(dst,src,horiz_conv,vert_conv);
}

inline void local_image_delete(image_t *src) {
    image_delete(src);
}

inline void local_image_delete(color_image_t *src) {
    color_image_delete(src);
}

/* compute image first and second order spatio-temporal derivatives of a color image */
template<ofdis::FlowInputType eInput>
void fdf::get_derivatives(const ofdis::InputImageType<eInput>* im1, const ofdis::InputImageType<eInput>* im2, const convolution_t *deriv, ofdis::InputImageType<eInput>* dx, ofdis::InputImageType<eInput>* dy, ofdis::InputImageType<eInput>* dt, ofdis::InputImageType<eInput>* dxx, ofdis::InputImageType<eInput>* dxy, ofdis::InputImageType<eInput>* dyy, ofdis::InputImageType<eInput>* dxt, ofdis::InputImageType<eInput>* dyt) {
    // derivatives are computed on the mean of the first image and the warped second image
    if(eInput==ofdis::FlowInput_RGB) {
        typename ofdis::InputImageType<eInput>* tmp_im2 = (typename ofdis::InputImageType<eInput>*)color_image_new(im2->width,im2->height);
        v4sf *tmp_im2p = (v4sf*) tmp_im2->c1, *dtp = (v4sf*) dt->c1, *im1p = (v4sf*) im1->c1, *im2p = (v4sf*) im2->c1;
        const v4sf half = {0.5f,0.5f,0.5f,0.5f};
        int i=0;
        for(i=0 ; i<3*im1->height*im1->stride/4 ; i++){
            *tmp_im2p = half * ( (*im2p) + (*im1p) );
            *dtp = (*im2p)-(*im1p);
            dtp+=1; im1p+=1; im2p+=1; tmp_im2p+=1;
        }
        // compute all other derivatives
        local_image_convolve_hv(dx, tmp_im2, deriv, NULL);
        local_image_convolve_hv(dy, tmp_im2, NULL, deriv);
        local_image_convolve_hv(dxx, dx, deriv, NULL);
        local_image_convolve_hv(dxy, dx, NULL, deriv);
        local_image_convolve_hv(dyy, dy, NULL, deriv);
        local_image_convolve_hv(dxt, dt, deriv, NULL);
        local_image_convolve_hv(dyt, dt, NULL, deriv);
        // free memory
        local_image_delete(tmp_im2);
    }
    else {
        typename ofdis::InputImageType<eInput>* tmp_im2 = (typename ofdis::InputImageType<eInput>*)image_new(im2->width,im2->height);
        v4sf *tmp_im2p = (v4sf*) tmp_im2->c1, *dtp = (v4sf*) dt->c1, *im1p = (v4sf*) im1->c1, *im2p = (v4sf*) im2->c1;
        const v4sf half = {0.5f,0.5f,0.5f,0.5f};
        int i=0;
        for(i=0 ; i<im1->height*im1->stride/4 ; i++){
            *tmp_im2p = half * ( (*im2p) + (*im1p) );
            *dtp = (*im2p)-(*im1p);
            dtp+=1; im1p+=1; im2p+=1; tmp_im2p+=1;
        }
        // compute all other derivatives
        local_image_convolve_hv(dx, tmp_im2, deriv, NULL);
        local_image_convolve_hv(dy, tmp_im2, NULL, deriv);
        local_image_convolve_hv(dxx, dx, deriv, NULL);
        local_image_convolve_hv(dxy, dx, NULL, deriv);
        local_image_convolve_hv(dyy, dy, NULL, deriv);
        local_image_convolve_hv(dxt, dt, deriv, NULL);
        local_image_convolve_hv(dyt, dt, NULL, deriv);
        // free memory
        local_image_delete(tmp_im2);
    }
}

template void fdf::get_derivatives<ofdis::FlowInput_Grayscale>(const ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, const ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, const convolution_t*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*, ofdis::InputImageType<ofdis::FlowInput_Grayscale>*);
template void fdf::get_derivatives<ofdis::FlowInput_Gradient>(const ofdis::InputImageType<ofdis::FlowInput_Gradient>*, const ofdis::InputImageType<ofdis::FlowInput_Gradient>*, const convolution_t*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*, ofdis::InputImageType<ofdis::FlowInput_Gradient>*);
template void fdf::get_derivatives<ofdis::FlowInput_RGB>(const ofdis::InputImageType<ofdis::FlowInput_RGB>*, const ofdis::InputImageType<ofdis::FlowInput_RGB>*, const convolution_t*, ofdis::InputImageType<ofdis::FlowInput_RGB>*, ofdis::InputImageType<ofdis::FlowInput_RGB>*, ofdis::InputImageType<ofdis::FlowInput_RGB>*, ofdis::InputImageType<ofdis::FlowInput_RGB>*, ofdis::InputImageType<ofdis::FlowInput_RGB>*, ofdis::InputImageType<ofdis::FlowInput_RGB>*, ofdis::InputImageType<ofdis::FlowInput_RGB>*, ofdis::InputImageType<ofdis::FlowInput_RGB>*);

/* compute the smoothness term */
/* It is represented as two images, the first one for horizontal smoothness, the second for vertical
   in dst_horiz, the pixel i,j represents the smoothness weight between pixel i,j and i,j+1
   in dst_vert, the pixel i,j represents the smoothness weight between pixel i,j and i+1,j */
void fdf::compute_smoothness(image_t *dst_horiz, image_t *dst_vert, const image_t *uu, const image_t *vv, const convolution_t *deriv_flow, const float quarter_alpha){
    const int width = uu->width, height = vv->height, stride = uu->stride;
    int j;
    image_t *ux = image_new(width,height), *vx = image_new(width,height), *uy = image_new(width,height), *vy = image_new(width,height), *smoothness = image_new(width,height);
    // compute derivatives [-0.5 0 0.5]
    convolve_horiz(ux, uu, deriv_flow);
    convolve_horiz(vx, vv, deriv_flow);
    convolve_vert(uy, uu, deriv_flow);
    convolve_vert(vy, vv, deriv_flow);
    // compute smoothness
    v4sf *uxp = (v4sf*) ux->c1, *vxp = (v4sf*) vx->c1, *uyp = (v4sf*) uy->c1, *vyp = (v4sf*) vy->c1, *sp = (v4sf*) smoothness->c1;
    const v4sf qa = {quarter_alpha,quarter_alpha,quarter_alpha,quarter_alpha};
    const v4sf epsmooth = {epsilon_smooth,epsilon_smooth,epsilon_smooth,epsilon_smooth};
    for(j=0 ; j< height*stride/4 ; j++){
        *sp = qa / __builtin_ia32_sqrtps( (*uxp)*(*uxp) + (*uyp)*(*uyp) + (*vxp)*(*vxp) + (*vyp)*(*vyp) + epsmooth );
        sp+=1;uxp+=1; uyp+=1; vxp+=1; vyp+=1;
    }
    image_delete(ux); image_delete(uy); image_delete(vx); image_delete(vy);
    // compute dst_horiz
    v4sf *dsthp = (v4sf*) dst_horiz->c1; sp = (v4sf*) smoothness->c1;
    float *sp_shift = (float*) memalign(16, stride*sizeof(float)); // aligned shifted copy of the current line
    for(j=0;j<height;j++){
        // create an aligned copy
        float *spf = (float*) sp;
        memcpy(sp_shift, spf+1, sizeof(float)*(stride-1));
        v4sf *sps = (v4sf*) sp_shift;
        int i;
        for(i=0;i<stride/4;i++){
            *dsthp = (*sp) + (*sps);
            dsthp+=1; sp+=1; sps+=1;
        }
        memset( &dst_horiz->c1[j*stride+width-1], 0, sizeof(float)*(stride-width+1));
    }
    free(sp_shift);
    // compute dst_vert
    v4sf *dstvp = (v4sf*) dst_vert->c1, *sp_bottom = (v4sf*) (smoothness->c1+stride); sp = (v4sf*) smoothness->c1;
    for(j=0 ; j<(height-1)*stride/4 ; j++){
        *dstvp = (*sp) + (*sp_bottom);
        dstvp+=1; sp+=1; sp_bottom+=1;
    }
    memset( &dst_vert->c1[(height-1)*stride], 0, sizeof(float)*stride);
    image_delete(smoothness);
}

/* sub the laplacian (smoothness term) to the right-hand term */
void fdf::sub_laplacian(image_t *dst, const image_t *src, const image_t *weight_horiz, const image_t *weight_vert){
    int j;
    const int offsetline = src->stride-src->width;
    float *src_ptr = src->c1, *dst_ptr = dst->c1, *weight_horiz_ptr = weight_horiz->c1;
    // horizontal filtering
    for(j=src->height+1;--j;){ // faster than for(j=0;j<src->height;j++)
        int i;
        for(i=src->width;--i;){
	        const float tmp = (*weight_horiz_ptr)*((*(src_ptr+1))-(*src_ptr));
	        *dst_ptr += tmp;
	        *(dst_ptr+1) -= tmp;
	        dst_ptr++;
	        src_ptr++;
	        weight_horiz_ptr++;
	    }
        dst_ptr += offsetline+1;
        src_ptr += offsetline+1;
        weight_horiz_ptr += offsetline+1;
    }

    v4sf *wvp = (v4sf*) weight_vert->c1, *srcp = (v4sf*) src->c1, *srcp_s = (v4sf*) (src->c1+src->stride), *dstp = (v4sf*) dst->c1, *dstp_s = (v4sf*) (dst->c1+src->stride);
    for(j=1+(src->height-1)*src->stride/4 ; --j ;){
        const v4sf tmp = (*wvp) * ((*srcp_s)-(*srcp));
        *dstp += tmp;
        *dstp_s -= tmp;
        wvp+=1; srcp+=1; srcp_s+=1; dstp+=1; dstp_s+=1;
    }
}

/* compute the dataterm and the matching term
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
void fdf::compute_data_and_match(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, color_image_t *Ix, color_image_t *Iy, color_image_t *Iz, color_image_t *Ixx, color_image_t *Ixy, color_image_t *Iyy, color_image_t *Ixz, color_image_t *Iyz, image_t *desc_weight, image_t *desc_flow_x, image_t *desc_flow_y, const float half_delta_over3, const float half_beta, const float half_gamma_over3){

    const v4sf dnorm = {datanorm, datanorm, datanorm, datanorm};
    const v4sf hdover3 = {half_delta_over3, half_delta_over3, half_delta_over3, half_delta_over3};
    const v4sf epscolor = {epsilon_color, epsilon_color, epsilon_color, epsilon_color};
    const v4sf hgover3 = {half_gamma_over3, half_gamma_over3, half_gamma_over3, half_gamma_over3};
    const v4sf epsgrad = {epsilon_grad, epsilon_grad, epsilon_grad, epsilon_grad};
    const v4sf hbeta = {half_beta,half_beta,half_beta,half_beta};
    const v4sf epsdesc = {epsilon_desc,epsilon_desc,epsilon_desc,epsilon_desc};

    v4sf *dup = (v4sf*) du->c1, *dvp = (v4sf*) dv->c1,
        *maskp = (v4sf*) mask->c1,
        *a11p = (v4sf*) a11->c1, *a12p = (v4sf*) a12->c1, *a22p = (v4sf*) a22->c1,
        *b1p = (v4sf*) b1->c1, *b2p = (v4sf*) b2->c1,
        *ix1p=(v4sf*)Ix->c1, *iy1p=(v4sf*)Iy->c1, *iz1p=(v4sf*)Iz->c1, *ixx1p=(v4sf*)Ixx->c1, *ixy1p=(v4sf*)Ixy->c1, *iyy1p=(v4sf*)Iyy->c1, *ixz1p=(v4sf*)Ixz->c1, *iyz1p=(v4sf*) Iyz->c1,
        *ix2p=(v4sf*)Ix->c2, *iy2p=(v4sf*)Iy->c2, *iz2p=(v4sf*)Iz->c2, *ixx2p=(v4sf*)Ixx->c2, *ixy2p=(v4sf*)Ixy->c2, *iyy2p=(v4sf*)Iyy->c2, *ixz2p=(v4sf*)Ixz->c2, *iyz2p=(v4sf*) Iyz->c2,
        *ix3p=(v4sf*)Ix->c3, *iy3p=(v4sf*)Iy->c3, *iz3p=(v4sf*)Iz->c3, *ixx3p=(v4sf*)Ixx->c3, *ixy3p=(v4sf*)Ixy->c3, *iyy3p=(v4sf*)Iyy->c3, *ixz3p=(v4sf*)Ixz->c3, *iyz3p=(v4sf*) Iyz->c3,
        *uup = (v4sf*) uu->c1, *vvp = (v4sf*)vv->c1, *wxp = (v4sf*)wx->c1, *wyp = (v4sf*)wy->c1,
        *descflowxp = (v4sf*)desc_flow_x->c1, *descflowyp = (v4sf*)desc_flow_y->c1, *descweightp = (v4sf*)desc_weight->c1;

    memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a12->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a22->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);
    memset(b2->c1 , 0, sizeof(float)*uu->height*uu->stride);

    int i;
    for(i = 0 ; i<uu->height*uu->stride/4 ; i++){
        v4sf tmp, tmp2, tmp3, tmp4, tmp5, tmp6, n1, n2, n3, n4, n5, n6;
        // dpsi color
        if(half_delta_over3){
            tmp  = *iz1p + (*ix1p)*(*dup) + (*iy1p)*(*dvp);
            n1 = (*ix1p) * (*ix1p) + (*iy1p) * (*iy1p) + dnorm;
            tmp2 = *iz2p + (*ix2p)*(*dup) + (*iy2p)*(*dvp);
            n2 = (*ix2p) * (*ix2p) + (*iy2p) * (*iy2p) + dnorm;
            tmp3 = *iz3p + (*ix3p)*(*dup) + (*iy3p)*(*dvp);
            n3 = (*ix3p) * (*ix3p) + (*iy3p) * (*iy3p) + dnorm;
            tmp = (*maskp) * hdover3 / __builtin_ia32_sqrtps(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + epscolor);
            tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
            *a11p += tmp  * (*ix1p) * (*ix1p);
            *a12p += tmp  * (*ix1p) * (*iy1p);
            *a22p += tmp  * (*iy1p) * (*iy1p);
            *b1p -=  tmp  * (*iz1p) * (*ix1p);
            *b2p -=  tmp  * (*iz1p) * (*iy1p);
            *a11p += tmp2 * (*ix2p) * (*ix2p);
            *a12p += tmp2 * (*ix2p) * (*iy2p);
            *a22p += tmp2 * (*iy2p) * (*iy2p);
            *b1p -=  tmp2 * (*iz2p) * (*ix2p);
            *b2p -=  tmp2 * (*iz2p) * (*iy2p);
            *a11p += tmp3 * (*ix3p) * (*ix3p);
            *a12p += tmp3 * (*ix3p) * (*iy3p);
            *a22p += tmp3 * (*iy3p) * (*iy3p);
            *b1p -=  tmp3 * (*iz3p) * (*ix3p);
            *b2p -=  tmp3 * (*iz3p) * (*iy3p);
        }
        // dpsi gradient
        n1 = (*ixx1p) * (*ixx1p) + (*ixy1p) * (*ixy1p) + dnorm;
        n2 = (*iyy1p) * (*iyy1p) + (*ixy1p) * (*ixy1p) + dnorm;
        tmp  = *ixz1p + (*ixx1p) * (*dup) + (*ixy1p) * (*dvp);
        tmp2 = *iyz1p + (*ixy1p) * (*dup) + (*iyy1p) * (*dvp);
        n3 = (*ixx2p) * (*ixx2p) + (*ixy2p) * (*ixy2p) + dnorm;
        n4 = (*iyy2p) * (*iyy2p) + (*ixy2p) * (*ixy2p) + dnorm;
        tmp3 = *ixz2p + (*ixx2p) * (*dup) + (*ixy2p) * (*dvp);
        tmp4 = *iyz2p + (*ixy2p) * (*dup) + (*iyy2p) * (*dvp);
        n5 = (*ixx3p) * (*ixx3p) + (*ixy3p) * (*ixy3p) + dnorm;
        n6 = (*iyy3p) * (*iyy3p) + (*ixy3p) * (*ixy3p) + dnorm;
        tmp5 = *ixz3p + (*ixx3p) * (*dup) + (*ixy3p) * (*dvp);
        tmp6 = *iyz3p + (*ixy3p) * (*dup) + (*iyy3p) * (*dvp);
        tmp = (*maskp) * hgover3 / __builtin_ia32_sqrtps(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + tmp4*tmp4/n4 + tmp5*tmp5/n5 + tmp6*tmp6/n6 + epsgrad);
        tmp6 = tmp/n6; tmp5 = tmp/n5; tmp4 = tmp/n4; tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
        *a11p += tmp *(*ixx1p)*(*ixx1p) + tmp2*(*ixy1p)*(*ixy1p);
        *a12p += tmp *(*ixx1p)*(*ixy1p) + tmp2*(*ixy1p)*(*iyy1p);
        *a22p += tmp2*(*iyy1p)*(*iyy1p) + tmp *(*ixy1p)*(*ixy1p);
        *b1p -=  tmp *(*ixx1p)*(*ixz1p) + tmp2*(*ixy1p)*(*iyz1p);
        *b2p -=  tmp2*(*iyy1p)*(*iyz1p) + tmp *(*ixy1p)*(*ixz1p);
        *a11p += tmp3*(*ixx2p)*(*ixx2p) + tmp4*(*ixy2p)*(*ixy2p);
        *a12p += tmp3*(*ixx2p)*(*ixy2p) + tmp4*(*ixy2p)*(*iyy2p);
        *a22p += tmp4*(*iyy2p)*(*iyy2p) + tmp3*(*ixy2p)*(*ixy2p);
        *b1p -=  tmp3*(*ixx2p)*(*ixz2p) + tmp4*(*ixy2p)*(*iyz2p);
        *b2p -=  tmp4*(*iyy2p)*(*iyz2p) + tmp3*(*ixy2p)*(*ixz2p);
        *a11p += tmp5*(*ixx3p)*(*ixx3p) + tmp6*(*ixy3p)*(*ixy3p);
        *a12p += tmp5*(*ixx3p)*(*ixy3p) + tmp6*(*ixy3p)*(*iyy3p);
        *a22p += tmp6*(*iyy3p)*(*iyy3p) + tmp5*(*ixy3p)*(*ixy3p);
        *b1p -=  tmp5*(*ixx3p)*(*ixz3p) + tmp6*(*ixy3p)*(*iyz3p);
        *b2p -=  tmp6*(*iyy3p)*(*iyz3p) + tmp5*(*ixy3p)*(*ixz3p);
        if(half_beta){ // dpsi_match
            tmp  = *uup - (*descflowxp);
            tmp2 = *vvp - (*descflowyp);
            tmp = hbeta*(*descweightp)/__builtin_ia32_sqrtps(tmp*tmp+tmp2*tmp2+epsdesc);
            *a11p += tmp;
            *a22p += tmp;
            *b1p -= tmp*((*wxp)-(*descflowxp));
            *b2p -= tmp*((*wyp)-(*descflowyp));
        }
        dup+=1; dvp+=1; maskp+=1; a11p+=1; a12p+=1; a22p+=1; b1p+=1; b2p+=1;
        ix1p+=1; iy1p+=1; iz1p+=1; ixx1p+=1; ixy1p+=1; iyy1p+=1; ixz1p+=1; iyz1p+=1;
        ix2p+=1; iy2p+=1; iz2p+=1; ixx2p+=1; ixy2p+=1; iyy2p+=1; ixz2p+=1; iyz2p+=1;
        ix3p+=1; iy3p+=1; iz3p+=1; ixx3p+=1; ixy3p+=1; iyy3p+=1; ixz3p+=1; iyz3p+=1;
        uup+=1;vvp+=1;wxp+=1; wyp+=1;descflowxp+=1;descflowyp+=1;descweightp+=1;
    }
}

inline float* local_get_c2(color_image_t* img) {
    return img->c2;
}

inline float* local_get_c3(color_image_t* img) {
    return img->c3;
}

inline float* local_get_c2(image_t* img) {
    return nullptr;
}

inline float* local_get_c3(image_t* img) {
    return nullptr;
}

/* compute the dataterm // REMOVED MATCHING TERM
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
template<ofdis::FlowInputType eInput>
void fdf::compute_data(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, ofdis::InputImageType<eInput> *Ix, ofdis::InputImageType<eInput> *Iy, ofdis::InputImageType<eInput> *Iz, ofdis::InputImageType<eInput> *Ixx, ofdis::InputImageType<eInput> *Ixy, ofdis::InputImageType<eInput> *Iyy, ofdis::InputImageType<eInput> *Ixz, ofdis::InputImageType<eInput> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3) {
    const v4sf dnorm = {datanorm, datanorm, datanorm, datanorm};
    const v4sf hdover3 = {half_delta_over3, half_delta_over3, half_delta_over3, half_delta_over3};
    const v4sf epscolor = {epsilon_color, epsilon_color, epsilon_color, epsilon_color};
    const v4sf hgover3 = {half_gamma_over3, half_gamma_over3, half_gamma_over3, half_gamma_over3};
    const v4sf epsgrad = {epsilon_grad, epsilon_grad, epsilon_grad, epsilon_grad};
    //const v4sf hbeta = {half_beta,half_beta,half_beta,half_beta};
    //const v4sf epsdesc = {epsilon_desc,epsilon_desc,epsilon_desc,epsilon_desc};

    v4sf *dup = (v4sf*) du->c1, *dvp = (v4sf*) dv->c1,
        *maskp = (v4sf*) mask->c1,
        *a11p = (v4sf*) a11->c1, *a12p = (v4sf*) a12->c1, *a22p = (v4sf*) a22->c1,
        *b1p = (v4sf*) b1->c1, *b2p = (v4sf*) b2->c1,
        *ix1p=(v4sf*)Ix->c1, *iy1p=(v4sf*)Iy->c1, *iz1p=(v4sf*)Iz->c1, *ixx1p=(v4sf*)Ixx->c1, *ixy1p=(v4sf*)Ixy->c1, *iyy1p=(v4sf*)Iyy->c1, *ixz1p=(v4sf*)Ixz->c1, *iyz1p=(v4sf*) Iyz->c1,
        *ix2p=(v4sf*)local_get_c2(Ix), *iy2p=(v4sf*)local_get_c2(Iy), *iz2p=(v4sf*)local_get_c2(Iz), *ixx2p=(v4sf*)local_get_c2(Ixx), *ixy2p=(v4sf*)local_get_c2(Ixy), *iyy2p=(v4sf*)local_get_c2(Iyy), *ixz2p=(v4sf*)local_get_c2(Ixz), *iyz2p=(v4sf*)local_get_c2(Iyz),
        *ix3p=(v4sf*)local_get_c3(Ix), *iy3p=(v4sf*)local_get_c3(Iy), *iz3p=(v4sf*)local_get_c3(Iz), *ixx3p=(v4sf*)local_get_c3(Ixx), *ixy3p=(v4sf*)local_get_c3(Ixy), *iyy3p=(v4sf*)local_get_c3(Iyy), *ixz3p=(v4sf*)local_get_c3(Ixz), *iyz3p=(v4sf*)local_get_c3(Iyz),
        *uup = (v4sf*) uu->c1, *vvp = (v4sf*)vv->c1, *wxp = (v4sf*)wx->c1, *wyp = (v4sf*)wy->c1;

    memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a12->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(a22->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);
    memset(b2->c1 , 0, sizeof(float)*uu->height*uu->stride);

    int i;
    for(i = 0 ; i<uu->height*uu->stride/4 ; i++){
        v4sf tmp, tmp2, tmp3, tmp4, tmp5, tmp6, n1, n2, n3, n4, n5, n6;
        // dpsi color
        if(half_delta_over3){
            tmp  = *iz1p + (*ix1p)*(*dup) + (*iy1p)*(*dvp);
            n1 = (*ix1p) * (*ix1p) + (*iy1p) * (*iy1p) + dnorm;
            if(eInput==ofdis::FlowInput_RGB) {
                tmp2 = *iz2p + (*ix2p)*(*dup) + (*iy2p)*(*dvp);
                n2 = (*ix2p) * (*ix2p) + (*iy2p) * (*iy2p) + dnorm;
                tmp3 = *iz3p + (*ix3p)*(*dup) + (*iy3p)*(*dvp);
                n3 = (*ix3p) * (*ix3p) + (*iy3p) * (*iy3p) + dnorm;
                tmp = (*maskp) * hdover3 / __builtin_ia32_sqrtps(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + epscolor);
                tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
            }
            else {
                tmp = (*maskp) * hdover3 / __builtin_ia32_sqrtps(3 * tmp*tmp/n1 + epscolor);
                tmp /= n1;
            }
            *a11p += tmp  * (*ix1p) * (*ix1p);
            *a12p += tmp  * (*ix1p) * (*iy1p);
            *a22p += tmp  * (*iy1p) * (*iy1p);
            *b1p -=  tmp  * (*iz1p) * (*ix1p);
            *b2p -=  tmp  * (*iz1p) * (*iy1p);
            if(eInput==ofdis::FlowInput_RGB) {
                *a11p += tmp2 * (*ix2p) * (*ix2p);
                *a12p += tmp2 * (*ix2p) * (*iy2p);
                *a22p += tmp2 * (*iy2p) * (*iy2p);
                *b1p -=  tmp2 * (*iz2p) * (*ix2p);
                *b2p -=  tmp2 * (*iz2p) * (*iy2p);
                *a11p += tmp3 * (*ix3p) * (*ix3p);
                *a12p += tmp3 * (*ix3p) * (*iy3p);
                *a22p += tmp3 * (*iy3p) * (*iy3p);
                *b1p -=  tmp3 * (*iz3p) * (*ix3p);
                *b2p -=  tmp3 * (*iz3p) * (*iy3p);
            }
        }

        // dpsi gradient
        n1 = (*ixx1p) * (*ixx1p) + (*ixy1p) * (*ixy1p) + dnorm;
        n2 = (*iyy1p) * (*iyy1p) + (*ixy1p) * (*ixy1p) + dnorm;
        tmp  = *ixz1p + (*ixx1p) * (*dup) + (*ixy1p) * (*dvp);
        tmp2 = *iyz1p + (*ixy1p) * (*dup) + (*iyy1p) * (*dvp);
        if(eInput==ofdis::FlowInput_RGB) {
            n3 = (*ixx2p) * (*ixx2p) + (*ixy2p) * (*ixy2p) + dnorm;
            n4 = (*iyy2p) * (*iyy2p) + (*ixy2p) * (*ixy2p) + dnorm;
            tmp3 = *ixz2p + (*ixx2p) * (*dup) + (*ixy2p) * (*dvp);
            tmp4 = *iyz2p + (*ixy2p) * (*dup) + (*iyy2p) * (*dvp);
            n5 = (*ixx3p) * (*ixx3p) + (*ixy3p) * (*ixy3p) + dnorm;
            n6 = (*iyy3p) * (*iyy3p) + (*ixy3p) * (*ixy3p) + dnorm;
            tmp5 = *ixz3p + (*ixx3p) * (*dup) + (*ixy3p) * (*dvp);
            tmp6 = *iyz3p + (*ixy3p) * (*dup) + (*iyy3p) * (*dvp);
            tmp = (*maskp) * hgover3 / __builtin_ia32_sqrtps(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + tmp4*tmp4/n4 + tmp5*tmp5/n5 + tmp6*tmp6/n6 + epsgrad);
            tmp6 = tmp/n6; tmp5 = tmp/n5; tmp4 = tmp/n4; tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
        }
        else {
            tmp = (*maskp) * hgover3 / __builtin_ia32_sqrtps(3* tmp*tmp/n1 + 3* tmp2*tmp2/n2 + epsgrad);
            tmp2 = tmp/n2; tmp /= n1;
        }
        *a11p += tmp *(*ixx1p)*(*ixx1p) + tmp2*(*ixy1p)*(*ixy1p);
        *a12p += tmp *(*ixx1p)*(*ixy1p) + tmp2*(*ixy1p)*(*iyy1p);
        *a22p += tmp2*(*iyy1p)*(*iyy1p) + tmp *(*ixy1p)*(*ixy1p);
        *b1p -=  tmp *(*ixx1p)*(*ixz1p) + tmp2*(*ixy1p)*(*iyz1p);
        *b2p -=  tmp2*(*iyy1p)*(*iyz1p) + tmp *(*ixy1p)*(*ixz1p);
        if(eInput==ofdis::FlowInput_RGB) {
            *a11p += tmp3*(*ixx2p)*(*ixx2p) + tmp4*(*ixy2p)*(*ixy2p);
            *a12p += tmp3*(*ixx2p)*(*ixy2p) + tmp4*(*ixy2p)*(*iyy2p);
            *a22p += tmp4*(*iyy2p)*(*iyy2p) + tmp3*(*ixy2p)*(*ixy2p);
            *b1p -=  tmp3*(*ixx2p)*(*ixz2p) + tmp4*(*ixy2p)*(*iyz2p);
            *b2p -=  tmp4*(*iyy2p)*(*iyz2p) + tmp3*(*ixy2p)*(*ixz2p);
            *a11p += tmp5*(*ixx3p)*(*ixx3p) + tmp6*(*ixy3p)*(*ixy3p);
            *a12p += tmp5*(*ixx3p)*(*ixy3p) + tmp6*(*ixy3p)*(*iyy3p);
            *a22p += tmp6*(*iyy3p)*(*iyy3p) + tmp5*(*ixy3p)*(*ixy3p);
            *b1p -=  tmp5*(*ixx3p)*(*ixz3p) + tmp6*(*ixy3p)*(*iyz3p);
            *b2p -=  tmp6*(*iyy3p)*(*iyz3p) + tmp5*(*ixy3p)*(*ixz3p);
        }

        if(eInput!=ofdis::FlowInput_RGB) {
            // multiply system to make smoothing parameters same for RGB and single-channel image
            *a11p *= 3;
            *a12p *= 3;
            *a22p *= 3;
            *b1p *=  3;
            *b2p *=  3;
        }

        dup+=1; dvp+=1; maskp+=1; a11p+=1; a12p+=1; a22p+=1; b1p+=1; b2p+=1;
        ix1p+=1; iy1p+=1; iz1p+=1; ixx1p+=1; ixy1p+=1; iyy1p+=1; ixz1p+=1; iyz1p+=1;
        if(eInput==ofdis::FlowInput_RGB) {
            ix2p+=1; iy2p+=1; iz2p+=1; ixx2p+=1; ixy2p+=1; iyy2p+=1; ixz2p+=1; iyz2p+=1;
            ix3p+=1; iy3p+=1; iz3p+=1; ixx3p+=1; ixy3p+=1; iyy3p+=1; ixz3p+=1; iyz3p+=1;
        }
        uup+=1;vvp+=1;wxp+=1; wyp+=1;
    }
}

template void fdf::compute_data<ofdis::FlowInput_Grayscale>(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ix, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iy, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iz, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ixx, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ixy, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iyy, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ixz, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
template void fdf::compute_data<ofdis::FlowInput_Gradient>(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ix, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iy, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iz, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ixx, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ixy, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iyy, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ixz, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
template void fdf::compute_data<ofdis::FlowInput_RGB>(image_t *a11, image_t *a12, image_t *a22, image_t *b1, image_t *b2, image_t *mask, image_t *wx, image_t *wy, image_t *du, image_t *dv, image_t *uu, image_t *vv, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ix, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iy, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iz, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ixx, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ixy, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iyy, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ixz, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);

/* compute the dataterm // REMOVED MATCHING TERM
   a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
   other (color) images are input */
template<ofdis::FlowInputType eInput>
void fdf::compute_data_DE(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, ofdis::InputImageType<eInput> *Ix, ofdis::InputImageType<eInput> *Iy, ofdis::InputImageType<eInput> *Iz, ofdis::InputImageType<eInput> *Ixx, ofdis::InputImageType<eInput> *Ixy, ofdis::InputImageType<eInput> *Iyy, ofdis::InputImageType<eInput> *Ixz, ofdis::InputImageType<eInput> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3) {
    const v4sf dnorm = {datanorm, datanorm, datanorm, datanorm};
    const v4sf hdover3 = {half_delta_over3, half_delta_over3, half_delta_over3, half_delta_over3};
    const v4sf epscolor = {epsilon_color, epsilon_color, epsilon_color, epsilon_color};
    const v4sf hgover3 = {half_gamma_over3, half_gamma_over3, half_gamma_over3, half_gamma_over3};
    const v4sf epsgrad = {epsilon_grad, epsilon_grad, epsilon_grad, epsilon_grad};
    //const v4sf hbeta = {half_beta,half_beta,half_beta,half_beta};
    //const v4sf epsdesc = {epsilon_desc,epsilon_desc,epsilon_desc,epsilon_desc};

    v4sf *dup = (v4sf*) du->c1,
        *maskp = (v4sf*) mask->c1,
        *a11p = (v4sf*) a11->c1,
        *b1p = (v4sf*) b1->c1,
        *ix1p=(v4sf*)Ix->c1, *iy1p=(v4sf*)Iy->c1, *iz1p=(v4sf*)Iz->c1, *ixx1p=(v4sf*)Ixx->c1, *ixy1p=(v4sf*)Ixy->c1, *iyy1p=(v4sf*)Iyy->c1, *ixz1p=(v4sf*)Ixz->c1, *iyz1p=(v4sf*) Iyz->c1,
        *ix2p=(v4sf*)local_get_c2(Ix), *iy2p=(v4sf*)local_get_c2(Iy), *iz2p=(v4sf*)local_get_c2(Iz), *ixx2p=(v4sf*)local_get_c2(Ixx), *ixy2p=(v4sf*)local_get_c2(Ixy), *iyy2p=(v4sf*)local_get_c2(Iyy), *ixz2p=(v4sf*)local_get_c2(Ixz), *iyz2p=(v4sf*)local_get_c2(Iyz),
        *ix3p=(v4sf*)local_get_c3(Ix), *iy3p=(v4sf*)local_get_c3(Iy), *iz3p=(v4sf*)local_get_c3(Iz), *ixx3p=(v4sf*)local_get_c3(Ixx), *ixy3p=(v4sf*)local_get_c3(Ixy), *iyy3p=(v4sf*)local_get_c3(Iyy), *ixz3p=(v4sf*)local_get_c3(Ixz), *iyz3p=(v4sf*)local_get_c3(Iyz),
        *uup = (v4sf*) uu->c1, *wxp = (v4sf*)wx->c1;

    memset(a11->c1, 0, sizeof(float)*uu->height*uu->stride);
    memset(b1->c1 , 0, sizeof(float)*uu->height*uu->stride);

    int i;
    for(i = 0 ; i<uu->height*uu->stride/4 ; i++){
        v4sf tmp, tmp2, tmp3, tmp4, tmp5, tmp6, n1, n2, n3, n4, n5, n6;
        // dpsi color
        if(half_delta_over3){
            tmp  = *iz1p + (*ix1p)*(*dup);
            n1 = (*ix1p) * (*ix1p) + (*iy1p) * (*iy1p) + dnorm;
            if(eInput==ofdis::FlowInput_RGB) {
                tmp2 = *iz2p + (*ix2p)*(*dup);
                n2 = (*ix2p) * (*ix2p) + (*iy2p) * (*iy2p) + dnorm;
                tmp3 = *iz3p + (*ix3p)*(*dup);
                n3 = (*ix3p) * (*ix3p) + (*iy3p) * (*iy3p) + dnorm;
                tmp = (*maskp) * hdover3 / __builtin_ia32_sqrtps(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + epscolor);
                tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
            }
            else {
                tmp = (*maskp) * hdover3 / __builtin_ia32_sqrtps(3 * tmp*tmp/n1 + epscolor);
                tmp /= n1;
            }
            *a11p += tmp  * (*ix1p) * (*ix1p);
            *b1p -=  tmp  * (*iz1p) * (*ix1p);
            if(eInput==ofdis::FlowInput_RGB) {
                *a11p += tmp2 * (*ix2p) * (*ix2p);
                *b1p -=  tmp2 * (*iz2p) * (*ix2p);
                *a11p += tmp3 * (*ix3p) * (*ix3p);
                *b1p -=  tmp3 * (*iz3p) * (*ix3p);
            }
        }
        // dpsi gradient
        n1 = (*ixx1p) * (*ixx1p) + (*ixy1p) * (*ixy1p) + dnorm;
        n2 = (*iyy1p) * (*iyy1p) + (*ixy1p) * (*ixy1p) + dnorm;
        tmp  = *ixz1p + (*ixx1p) * (*dup);
        tmp2 = *iyz1p + (*ixy1p) * (*dup);
        if(eInput==ofdis::FlowInput_RGB) {
            n3 = (*ixx2p) * (*ixx2p) + (*ixy2p) * (*ixy2p) + dnorm;
            n4 = (*iyy2p) * (*iyy2p) + (*ixy2p) * (*ixy2p) + dnorm;
            tmp3 = *ixz2p + (*ixx2p) * (*dup);
            tmp4 = *iyz2p + (*ixy2p) * (*dup);
            n5 = (*ixx3p) * (*ixx3p) + (*ixy3p) * (*ixy3p) + dnorm;
            n6 = (*iyy3p) * (*iyy3p) + (*ixy3p) * (*ixy3p) + dnorm;
            tmp5 = *ixz3p + (*ixx3p) * (*dup);
            tmp6 = *iyz3p + (*ixy3p) * (*dup);
            tmp = (*maskp) * hgover3 / __builtin_ia32_sqrtps(tmp*tmp/n1 + tmp2*tmp2/n2 + tmp3*tmp3/n3 + tmp4*tmp4/n4 + tmp5*tmp5/n5 + tmp6*tmp6/n6 + epsgrad);
            tmp6 = tmp/n6; tmp5 = tmp/n5; tmp4 = tmp/n4; tmp3 = tmp/n3; tmp2 = tmp/n2; tmp /= n1;
        }
        else {
            tmp = (*maskp) * hgover3 / __builtin_ia32_sqrtps(3* tmp*tmp/n1 + 3* tmp2*tmp2/n2 + epsgrad);
            tmp2 = tmp/n2; tmp /= n1;
        }
        *a11p += tmp *(*ixx1p)*(*ixx1p) + tmp2*(*ixy1p)*(*ixy1p);
        *b1p -=  tmp *(*ixx1p)*(*ixz1p) + tmp2*(*ixy1p)*(*iyz1p);
        if(eInput==ofdis::FlowInput_RGB) {
            *a11p += tmp3*(*ixx2p)*(*ixx2p) + tmp4*(*ixy2p)*(*ixy2p);
            *b1p -=  tmp3*(*ixx2p)*(*ixz2p) + tmp4*(*ixy2p)*(*iyz2p);
            *a11p += tmp5*(*ixx3p)*(*ixx3p) + tmp6*(*ixy3p)*(*ixy3p);
            *b1p -=  tmp5*(*ixx3p)*(*ixz3p) + tmp6*(*ixy3p)*(*iyz3p);
        }

        if(eInput!=ofdis::FlowInput_RGB) {
            // multiply system to make smoothing parameters same for RGB and single-channel image
            *a11p *= 3;
            *b1p *=  3;
        }

        dup+=1; maskp+=1; a11p+=1; b1p+=1;
        ix1p+=1; iy1p+=1; iz1p+=1; ixx1p+=1; ixy1p+=1; iyy1p+=1; ixz1p+=1; iyz1p+=1;
        if(eInput==ofdis::FlowInput_RGB) {
            ix2p+=1; iy2p+=1; iz2p+=1; ixx2p+=1; ixy2p+=1; iyy2p+=1; ixz2p+=1; iyz2p+=1;
            ix3p+=1; iy3p+=1; iz3p+=1; ixx3p+=1; ixy3p+=1; iyy3p+=1; ixz3p+=1; iyz3p+=1;
        }
        uup+=1; wxp+=1;

    }
}

template void fdf::compute_data_DE<ofdis::FlowInput_Grayscale>(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ix, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iy, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iz, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ixx, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ixy, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iyy, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Ixz, ofdis::InputImageType<ofdis::FlowInput_Grayscale> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
template void fdf::compute_data_DE<ofdis::FlowInput_Gradient>(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ix, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iy, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iz, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ixx, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ixy, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iyy, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Ixz, ofdis::InputImageType<ofdis::FlowInput_Gradient> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);
template void fdf::compute_data_DE<ofdis::FlowInput_RGB>(image_t *a11, image_t *b1, image_t *mask, image_t *wx, image_t *du, image_t *uu, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ix, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iy, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iz, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ixx, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ixy, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iyy, ofdis::InputImageType<ofdis::FlowInput_RGB> *Ixz, ofdis::InputImageType<ofdis::FlowInput_RGB> *Iyz, const float half_delta_over3, const float half_beta, const float half_gamma_over3);

/* resize the descriptors to the new size using a weighted mean */
void fdf::descflow_resize(image_t *dst_flow_x, image_t *dst_flow_y, image_t *dst_weight, const image_t *src_flow_x, const image_t *src_flow_y, const image_t *src_weight){
    const int src_width = src_flow_x->width, src_height = src_flow_x->height, src_stride = src_flow_x->stride,
                dst_width = dst_flow_x->width, dst_height = dst_flow_x->height, dst_stride = dst_flow_x->stride;
    const float scale_x = ((float)dst_width-1)/((float)src_width-1), scale_y = ((float)dst_height-1)/((float)src_height-1);
    image_erase(dst_flow_x); image_erase(dst_flow_y); image_erase(dst_weight);
    int j;
    for( j=0 ; j<src_height ; j++){
        const float yy = ((float)j)*scale_y;
        const float yyf = floor(yy);
        const float dy = yy-yyf;
        const int y1 = MINMAX_TA( (int) yyf   , dst_height);
        const int y2 = MINMAX_TA( (int) yyf+1 , dst_height);
        int i;
        for( i=0 ; i<src_width ; i++ ){
            const float weight = src_weight->c1[j*src_stride+i];
	        if( weight<0.0000000001f ) continue;
            const float xx = ((float)i)*scale_x;
	        const float xxf = floor(xx);
	        const float dx = xx-xxf;
	        const int x1 = MINMAX_TA( (int) xxf   , dst_width);
	        const int x2 = MINMAX_TA( (int) xxf+1 , dst_width);
	        float weightxy, newweight;
	        if( dx ){
	            if( dy ){
		            weightxy = weight*dx*dy;
		            newweight = dst_weight->c1[y2*dst_stride+x2] + weightxy;
		            dst_flow_x->c1[y2*dst_stride+x2] = (dst_flow_x->c1[y2*dst_stride+x2]*dst_weight->c1[y2*dst_stride+x2] + src_flow_x->c1[j*src_stride+i]*weightxy*scale_x)/newweight;
		            dst_flow_y->c1[y2*dst_stride+x2] = (dst_flow_y->c1[y2*dst_stride+x2]*dst_weight->c1[y2*dst_stride+x2] + src_flow_y->c1[j*src_stride+i]*weightxy*scale_y)/newweight;
		            dst_weight->c1[y2*dst_stride+x2] = newweight;
		        }
	            weightxy = weight*dx*(1.0f-dy);
	            newweight = dst_weight->c1[y1*dst_stride+x2] + weightxy;
	            dst_flow_x->c1[y1*dst_stride+x2] = (dst_flow_x->c1[y1*dst_stride+x2]*dst_weight->c1[y1*dst_stride+x2] + src_flow_x->c1[j*src_stride+i]*weightxy*scale_x)/newweight;
	            dst_flow_y->c1[y1*dst_stride+x2] = (dst_flow_y->c1[y1*dst_stride+x2]*dst_weight->c1[y1*dst_stride+x2] + src_flow_y->c1[j*src_stride+i]*weightxy*scale_y)/newweight;
	            dst_weight->c1[y1*dst_stride+x2] = newweight;
            }
	        if( dy ) {
                weightxy = weight*(1.0f-dx)*dy;
                newweight = dst_weight->c1[y2*dst_stride+x1] + weightxy;
                dst_flow_x->c1[y2*dst_stride+x1] = (dst_flow_x->c1[y2*dst_stride+x1]*dst_weight->c1[y2*dst_stride+x1] + src_flow_x->c1[j*src_stride+i]*weightxy*scale_x)/newweight;
                dst_flow_y->c1[y2*dst_stride+x1] = (dst_flow_y->c1[y2*dst_stride+x1]*dst_weight->c1[y2*dst_stride+x1] + src_flow_y->c1[j*src_stride+i]*weightxy*scale_y)/newweight;
                dst_weight->c1[y2*dst_stride+x1] = newweight;
	        }
	        weightxy = weight*(1.0f-dx)*(1.0f-dy);
	        newweight = dst_weight->c1[y1*dst_stride+x1] + weightxy;
	        dst_flow_x->c1[y1*dst_stride+x1] = (dst_flow_x->c1[y1*dst_stride+x1]*dst_weight->c1[y1*dst_stride+x1] + src_flow_x->c1[j*src_stride+i]*weightxy*scale_x)/newweight;
	        dst_flow_y->c1[y1*dst_stride+x1] = (dst_flow_y->c1[y1*dst_stride+x1]*dst_weight->c1[y1*dst_stride+x1] + src_flow_y->c1[j*src_stride+i]*weightxy*scale_y)/newweight;
	        dst_weight->c1[y1*dst_stride+x1] = newweight;
	    }
    }
}

/* resize the descriptors to the new size using a nearest neighbor method while keeping the descriptor with the higher weight at the end */
void fdf::descflow_resize_nn(image_t *dst_flow_x, image_t *dst_flow_y, image_t *dst_weight, const image_t *src_flow_x, const image_t *src_flow_y, const image_t *src_weight){
    const int src_width = src_flow_x->width, src_height = src_flow_x->height, src_stride = src_flow_x->stride,
                dst_width = dst_flow_x->width, dst_height = dst_flow_x->height, dst_stride = dst_flow_x->stride;
    const float scale_x = ((float)dst_width-1)/((float)src_width-1), scale_y = ((float)dst_height-1)/((float)src_height-1);
    image_erase(dst_flow_x); image_erase(dst_flow_y); image_erase(dst_weight);
    int j;
    for( j=0 ; j<src_height ; j++){
        const float yy = ((float)j)*scale_y;
        const int y = (int) 0.5f+yy; // equivalent to round(yy)
        int i;
        for( i=0 ; i<src_width ; i++ ){
	        const float weight = src_weight->c1[j*src_stride+i];
	        if( !weight )
	            continue;
	        const float xx = ((float)i)*scale_x;
            // @@@@@@@@@@
	        const int x = (int) 0.5f+xx; // equivalent to round(xx)
	        if( dst_weight->c1[y*dst_stride+x] < weight ){
	            dst_weight->c1[y*dst_stride+x] = weight;
	            dst_flow_x->c1[y*dst_stride+x] = src_flow_x->c1[j*src_stride+i]*scale_x;
	            dst_flow_y->c1[y*dst_stride+x] = src_flow_y->c1[j*src_stride+i]*scale_y;
	        }
	    }
    }
}