
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "litiv/utils/CxxUtils.hpp"
#include "litiv/imgproc.hpp"

//! defines the default value for EdgeDetectorLBSP::m_nLevels
#define EDGLBSP_DEFAULT_LEVEL_COUNT (3)
//! defines the default value for EdgeDetectorLBSP::m_dHystLowThrshFactor (if needed)
#define EDGLBSP_DEFAULT_HYST_LOW_THRSH_FACT (0.80)
//! defines the default integral [0,255] LBSP detection threshold value
#define EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER (30)
//! defines the default integral [0,255] edge detection threshold value
#define EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER (20)
//! defines the default value for the threshold passed to EdgeDetectorLBSP::apply
#define EDGLBSP_DEFAULT_DET_THRESHOLD ((double)EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER/UCHAR_MAX)

#define EDGLBSP_NORMALIZE_OUTPUT 1

class EdgeDetectorLBSP : public EdgeDetector {
public:
    //! full constructor
    EdgeDetectorLBSP( size_t nLevels=EDGLBSP_DEFAULT_LEVEL_COUNT,
                      double dHystLowThrshFactor=EDGLBSP_DEFAULT_HYST_LOW_THRSH_FACT,
                      bool bNormalizeOutput=EDGLBSP_NORMALIZE_OUTPUT);
    //! returns the default edge detection threshold value used in 'apply'
    virtual double getDefaultThreshold() const {return EDGLBSP_DEFAULT_DET_THRESHOLD;}
    //! returns the default LBSP descriptor internal threshold used for comparions (also used in 'apply')
    virtual uchar getDefaultLBSPThreshold() const {return EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER;}
    //! thresholded edge detection function; the edge detection threshold should be between 0 and 1 (will use default otherwise)
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dDetThreshold=EDGLBSP_DEFAULT_DET_THRESHOLD);
    //! thresholded edge detection function; the edge detection threshold should be between 0 and 1 (will use default otherwise), and the LBSP threshold should be between 0 and 255
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dDetThreshold=EDGLBSP_DEFAULT_DET_THRESHOLD, uchar nLBSPThreshold=EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    //! edge detection function; returns a confidence edge mask (0-255) instead of a thresholded/binary edge mask
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask);

protected:

    //! number of pyramid levels to analyze
    const size_t m_nLevels;
    //! base threshold multiplier used to compute the upper hysteresis threshold
    const double m_dHystLowThrshFactor;
    //! gaussian blur kernel sigma value
    const double m_dGaussianKernelSigma;
    //! defines whether the output is normalized to the full 0-255 range or not
    const bool m_bNormalizeOutput;
    //! pre-allocated image pyramid maps for multi-scale LBSP lookup
    std::vector<std::aligned_vector<uchar,32>> m_vvuInputPyrMaps;
    //! pre-allocated image pyramid LUT maps for multi-scale LBSP computation
    std::vector<std::aligned_vector<uchar,32>> m_vvuLBSPLookupMaps;
    //! multi-level image map size lookup list
    std::vector<cv::Size> m_voMapSizeList;

    //! internal single-threshold edge det function w/ explicit def for 1/2/3/4 channel(s)
    template<size_t nChannels>
    void apply_threshold_internal(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nDetThreshold, uchar nLBSPThreshold);
};
