
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

#include "litiv/video/BackgroundSubtractionUtils.hpp"
#include "litiv/features2d/LBSP.hpp"
#include "litiv/utils/RandUtils.hpp"

/*!
    Local Binary Similarity Pattern (LBSP) algorithm interface for FG/BG video segmentation via change detection.

    For more details on the different parameters, see P.-L. St-Charles and G.-A. Bilodeau, "Improving Background
    Subtraction using Local Binary Similarity Patterns", in WACV 2014, or G.-A. Bilodeau et al, "Change Detection
    in Feature Space Using Local Binary Similarity Patterns", in CRV 2013.
 */
template<ParallelUtils::eParallelAlgoType eImpl>
class BackgroundSubtractorLBSP :
        public BackgroundSubtractor_<eImpl> {
public:

    //! default impl constructor
    template<ParallelUtils::eParallelAlgoType eImplTemp = eImpl>
    BackgroundSubtractorLBSP(float fRelLBSPThreshold, size_t nLBSPThresholdOffset, typename std::enable_if<eImplTemp==ParallelUtils::eNonParallel>::type* pUnused=0);

#if HAVE_GLSL
    //! glsl impl constructor
    template<ParallelUtils::eParallelAlgoType eImplTemp = eImpl>
    BackgroundSubtractorLBSP(float fRelLBSPThreshold, size_t nLBSPThresholdOffset, size_t nLevels, size_t nComputeStages,
                             size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures,
                             int nDebugType, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat,
                             typename std::enable_if<eImplTemp==ParallelUtils::eGLSL>::type* pUnused=0);

    //! returns the GLSL compute shader source code for LBSP lookup/description functions
    template<ParallelUtils::eParallelAlgoType eImplTemp = eImpl>
    typename std::enable_if<eImplTemp==ParallelUtils::eGLSL,std::string>::type getLBSPThresholdLUTShaderSource() const;
#endif //HAVE_GLSL
#if HAVE_CUDA
    // ... @@@ add impl later
    static_assert(eImpl!=ParallelUtils::eCUDA),"Missing constr impl");
#endif //HAVE_CUDA
#if HAVE_OPENCL
    // ... @@@ add impl later
    static_assert(eImpl!=ParallelUtils::eOpenCL),"Missing constr impl");
#endif //HAVE_OPENCL

    //! (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);

protected:
    //! LBSP internal threshold offset value, used to reduce texture noise in dark regions
    const size_t m_nLBSPThresholdOffset;
    //! LBSP relative internal threshold (kept here since we don't keep an LBSP object)
    const float m_fRelLBSPThreshold;
    //! pre-allocated internal LBSP threshold values LUT for all possible 8-bit intensities
    std::array<size_t,UCHAR_MAX+1> m_anLBSPThreshold_8bitLUT;
    //! default kernel size for median blur post-proc filtering
    const int m_nDefaultMedianBlurKernelSize;
    //! copy of latest descriptors (used when refreshing model)
    cv::Mat m_oLastDescFrame;
};
