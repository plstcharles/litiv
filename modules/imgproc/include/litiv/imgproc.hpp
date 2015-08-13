#pragma once

#include "litiv/imgproc/EdgeDetectorCanny.hpp"
#include "litiv/imgproc/EdgeDetectorLBSP.hpp"

namespace litiv {

    enum eThinningMode {
        eThinningMode_ZhangSuen=0,
        eThinningMode_LamLeeSuen
    };

    //! 'thins' the provided image (currently only works on 1ch 8UC1 images, treated as binary)
    void thinning(const cv::Mat& oInput, cv::Mat& oOutput, eThinningMode eMode=eThinningMode_LamLeeSuen);

}; //namespace litiv

