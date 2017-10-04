
#pragma once

#ifndef OFDIS_INTERNAL
#error "must only include 'ofdis.hpp' header for API"
#endif //ndef(OFDIS_INTERNAL)

#include "litiv/3rdparty/ofdis/fdf/image.h"
#include "litiv/3rdparty/ofdis/fdf/opticalflow_aux.hpp"
#include "litiv/3rdparty/ofdis/fdf/solver.h"
#include "litiv/3rdparty/ofdis/oflow.hpp"

namespace ofdis {

    struct TVparams {
        float alpha;             // smoothness weight
        float beta;              // matching weight
        float gamma;             // gradient constancy assumption weight
        float delta;             // color constancy assumption weight
        int n_inner_iteration;   // number of inner fixed point iterations
        int n_solver_iteration;  // number of solver iterations
        float sor_omega;         // omega parameter of sor method
        float tmp_quarter_alpha;
        float tmp_half_gamma_over3;
        float tmp_half_delta_over3;
        float tmp_half_beta;
    };

    template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
    class VarRefClass {
    public:
        VarRefClass(const float * im_ao_in, const float * im_ao_dx_in, const float * im_ao_dy_in, // expects #sc_f_in pointers to float arrays for images and gradients.
                  const float * im_bo_in, const float * im_bo_dx_in, const float * im_bo_dy_in,
                  const camparam* cpt_in, const camparam* cpo_in,const optparam* op_in, float *flowout);
    private:
        using InputImageType = ofdis::InputImageType<eInput>;
        convolution_t *deriv, *deriv_flow;
        void copyimage(const float* img, InputImageType* img_t);
        void RefLevelOF(image_t *wx, image_t *wy, const InputImageType* im1, const InputImageType* im2);
        void RefLevelDE(image_t *wx, const InputImageType* im1, const InputImageType* im2);
        TVparams tvparams;
        const camparam* cpt;
        const camparam* cpo;
        const optparam* op;
    };

} // namespace ofdis