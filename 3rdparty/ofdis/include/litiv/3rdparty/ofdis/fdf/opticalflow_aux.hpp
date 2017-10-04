
#pragma once

#ifndef OFDIS_INTERNAL
#error "must only include 'ofdis.hpp' header for API"
#endif //ndef(OFDIS_INTERNAL)

#include <cstdlib>
#include "litiv/3rdparty/ofdis/utils.hpp"
#include "litiv/3rdparty/ofdis/fdf/image.h"

namespace fdf {

    // warp a color image according to a flow. src is the input image, wx and wy, the input flow. dst is the warped image and mask contains 0 or 1 if the pixels goes outside/inside image boundaries
    template<ofdis::FlowInputType eInput>
    void image_warp(ofdis::InputImageType<eInput>* dst,
                    image_t *mask,
                    const ofdis::InputImageType<eInput>* src,
                    const image_t *wx,
                    const image_t *wy);

    // compute image first and second order spatio-temporal derivatives of a color image
    template<ofdis::FlowInputType eInput>
    void get_derivatives(const ofdis::InputImageType<eInput>* im1,
                         const ofdis::InputImageType<eInput>* im2,
                         const convolution_t *deriv,
                         ofdis::InputImageType<eInput>* dx,
                         ofdis::InputImageType<eInput>* dy,
                         ofdis::InputImageType<eInput>* dt,
                         ofdis::InputImageType<eInput>* dxx,
                         ofdis::InputImageType<eInput>* dxy,
                         ofdis::InputImageType<eInput>* dyy,
                         ofdis::InputImageType<eInput>* dxt,
                         ofdis::InputImageType<eInput>* dyt);

    // compute the smoothness term
    void compute_smoothness(image_t *dst_horiz,
                            image_t *dst_vert,
                            const image_t *uu,
                            const image_t *vv,
                            const convolution_t *deriv_flow,
                            const float quarter_alpha);

    // sub the laplacian (smoothness term) to the right-hand term
    void sub_laplacian(image_t *dst,
                       const image_t *src,
                       const image_t *weight_horiz,
                       const image_t *weight_vert);

    // compute the dataterm and the matching term
    // a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
    // other (color) images are input
    void compute_data_and_match(image_t *a11,
                                image_t *a12,
                                image_t *a22,
                                image_t *b1,
                                image_t *b2,
                                image_t *mask,
                                image_t *wx,
                                image_t *wy,
                                image_t *du,
                                image_t *dv,
                                image_t *uu,
                                image_t *vv,
                                color_image_t *Ix,
                                color_image_t *Iy,
                                color_image_t *Iz,
                                color_image_t *Ixx,
                                color_image_t *Ixy,
                                color_image_t *Iyy,
                                color_image_t *Ixz,
                                color_image_t *Iyz,
                                image_t *desc_weight,
                                image_t *desc_flow_x,
                                image_t *desc_flow_y,
                                const float half_delta_over3,
                                const float half_beta,
                                const float half_gamma_over3);

    // compute the dataterm and ... REMOVED THE MATCHING TERM
    // a11 a12 a22 represents the 2x2 diagonal matrix, b1 and b2 the right hand side
    // other (color) images are input
    template<ofdis::FlowInputType eInput>
    void compute_data(image_t *a11,
                      image_t *a12,
                      image_t *a22,
                      image_t *b1,
                      image_t *b2,
                      image_t *mask,
                      image_t *wx,
                      image_t *wy,
                      image_t *du,
                      image_t *dv,
                      image_t *uu,
                      image_t *vv,
                      ofdis::InputImageType<eInput> *Ix,
                      ofdis::InputImageType<eInput> *Iy,
                      ofdis::InputImageType<eInput> *Iz,
                      ofdis::InputImageType<eInput> *Ixx,
                      ofdis::InputImageType<eInput> *Ixy,
                      ofdis::InputImageType<eInput> *Iyy,
                      ofdis::InputImageType<eInput> *Ixz,
                      ofdis::InputImageType<eInput> *Iyz,
                      const float half_delta_over3,
                      const float half_beta,
                      const float half_gamma_over3);

    template<ofdis::FlowInputType eInput>
    void compute_data_DE(image_t *a11,
                         image_t *b1,
                         image_t *mask,
                         image_t *wx,
                         image_t *du,
                         image_t *uu,
                         ofdis::InputImageType<eInput> *Ix,
                         ofdis::InputImageType<eInput> *Iy,
                         ofdis::InputImageType<eInput> *Iz,
                         ofdis::InputImageType<eInput> *Ixx,
                         ofdis::InputImageType<eInput> *Ixy,
                         ofdis::InputImageType<eInput> *Iyy,
                         ofdis::InputImageType<eInput> *Ixz,
                         ofdis::InputImageType<eInput> *Iyz,
                         const float half_delta_over3,
                         const float half_beta,
                         const float half_gamma_over3);

    // resize the descriptors to the new size using a weighted mean
    void descflow_resize(image_t *dst_flow_x,
                         image_t *dst_flow_y,
                         image_t *dst_weight,
                         const image_t *src_flow_x,
                         const image_t *src_flow_y,
                         const image_t *src_weight);

    // resize the descriptors to the new size using a nearest neighbor method while keeping the descriptor with the higher weight at the end
    void descflow_resize_nn(image_t *dst_flow_x,
                            image_t *dst_flow_y,
                            image_t *dst_weight,
                            const image_t *src_flow_x,
                            const image_t *src_flow_y,
                            const image_t *src_weight);

} // namespace fdf