
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

#include "litiv/imgproc/EdgeDetectionUtils.hpp"
#include "litiv/features2d/LBSP.hpp"

/// defines the default value for EdgeDetectorLBSP::m_nLevels
#define EDGLBSP_DEFAULT_LEVEL_COUNT (3)
/// defines the default value for EdgeDetectorLBSP::m_dHystLowThrshFactor (if needed)
#define EDGLBSP_DEFAULT_HYST_LOW_THRSH_FACT (0.5)
/// defines the default integral [0,LBSP::MAX_GRAD_MAG] edge detection threshold value
#define EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER (LBSP::MAX_GRAD_MAG/2)
/// defines the default value for the threshold passed to EdgeDetectorLBSP::apply_threshold
#define EDGLBSP_DEFAULT_DET_THRESHOLD ((double)EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER/LBSP::MAX_GRAD_MAG)

class EdgeDetectorLBSP : public IEdgeDetector {
public:
    /// full constructor
    EdgeDetectorLBSP(size_t nLevels=EDGLBSP_DEFAULT_LEVEL_COUNT,
                     double dHystLowThrshFactor=EDGLBSP_DEFAULT_HYST_LOW_THRSH_FACT,
                     bool bNormalizeOutput=false);
    /// returns the default edge detection threshold value used in 'apply'
    virtual double getDefaultThreshold() const {return EDGLBSP_DEFAULT_DET_THRESHOLD;}
    /// thresholded edge detection function; the edge detection threshold should be between 0 and 1 (will use default otherwise)
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dDetThreshold=EDGLBSP_DEFAULT_DET_THRESHOLD);
    /// edge detection function; returns a confidence edge mask (0-255) instead of a thresholded/binary edge mask
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask);

protected:

    /// number of pyramid levels to analyze
    const size_t m_nLevels;
    /// base threshold multiplier used to compute the upper hysteresis threshold
    const double m_dHystLowThrshFactor;
    /// gaussian blur kernel sigma value
    const double m_dGaussianKernelSigma;
    /// defines whether the output is normalized to the full 0-255 range or not
    const bool m_bNormalizeOutput;
    /// pre-allocated image pyramid maps for multi-scale LBSP lookup
    std::vector<std::aligned_vector<uchar,32>> m_vvuInputPyrMaps;
    /// pre-allocated image pyramid LUT maps for multi-scale LBSP computation
    std::vector<std::aligned_vector<uchar,32>> m_vvuLBSPLookupMaps;
    /// pre-allocated image gradient reconstruction map
    std::aligned_vector<uchar,32> m_vuLBSPGradMapData;
    /// pre-allocated image edge reconstruction map
    std::aligned_vector<uchar,32> m_vuEdgeTempMaskData;
    /// multi-level image map size lookup list
    std::vector<cv::Size> m_voMapSizeList;
    /// hysteresis recursive search stack
    std::vector<uchar*> m_vuHystStack;

    /// internal lookup/pyramiding function w/ explicit definitions for 1 to 4 channels
    template<size_t nChannels>
    void apply_internal_lookup(const cv::Mat& oInputImg);
    void apply_internal_lookup(const cv::Mat& oInputImg, size_t nChannels);
    /// internal thresholding function w/ explicit definitions for 1 to 4 channels
    template<size_t nChannels>
    void apply_internal_threshold(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nDetThreshold);
    void apply_internal_threshold(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nDetThreshold, size_t nChannels);
};
