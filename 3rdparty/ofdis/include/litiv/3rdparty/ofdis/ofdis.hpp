
#pragma once

// note: this is the litiv API entrypoint for optical flow and stereo depth computation methods
// of Kroeger et al.; see the original code at 'https://github.com/tikroeger/OF_DIS' for more info

#define OFDIS_API
#include "litiv/3rdparty/ofdis/utils.hpp"
#include <opencv2/core/core.hpp>

namespace ofdis {

    /// parameter holder for ofdis algorithm
    struct FlowParams {
        /// default constructor; will initialize all parameters with proper default values
        FlowParams(int sel_oppoint=2/*default 'operating point'*/);
        // see 'litiv/3rdparty/ofdis/oflow.hpp' for parameter definitions
        int lv_f,lv_l,maxiter,miniter,patchsz,patnorm,costfct,tv_innerit,tv_solverit,verbosity,sel_oppoint;
        float mindprate,mindrrate,minimgerr,poverl,tv_alpha,tv_gamma,tv_delta,tv_sor;
        bool usefbcon,usetvref;
    private:
        void setOpPointParams(const cv::Size& oImageSize);
        template<FlowInputType eInput, FlowOutputType eOutput>
        friend void computeFlow(const cv::Mat&,const cv::Mat&,cv::Mat&,FlowParams=FlowParams());
    };

    /// ofdis algorithm interface, with all specialized input/output combos pre-instantiated
    template<FlowInputType eInput, FlowOutputType eOutput>
    void computeFlow(const cv::Mat& oInput1, const cv::Mat& oInput2, cv::Mat& oOutput, FlowParams oParams);

} // namespace ofdis