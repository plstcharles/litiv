
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

/// defines the default value for BackgroundSubtractorLBSP::m_fRelLBSPThreshold
#define BGSLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.333f)
/// defines the default value for BackgroundSubtractorLBSP::m_nLBSPThresholdOffset
#define BGSLBSP_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD (0)
/// defines the default value for BackgroundSubtractorLBSP::m_nDefaultMedianBlurKernelSize
#define BGSLBSP_DEFAULT_MEDIAN_BLUR_KERNEL_SIZE (9)

/**
    Local Binary Similarity Pattern (LBSP) algorithm interface for FG/BG video segmentation via change detection.

    For more details on the different parameters, see P.-L. St-Charles and G.-A. Bilodeau, "Improving Background
    Subtraction using Local Binary Similarity Patterns", in WACV 2014, or G.-A. Bilodeau et al, "Change Detection
    in Feature Space Using Local Binary Similarity Patterns", in CRV 2013.
*/
template<lv::ParallelAlgoType eImpl>
struct IBackgroundSubtractorLBSP_ : public IBackgroundSubtractor_<eImpl> {

    /// returns a copy of the latest reconstructed background descriptors image
    virtual void getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const = 0;

protected:
    /// default impl constructor (defined here as MSVC is very prude with template-class-template-cstor-definitions)
    template<lv::ParallelAlgoType eImplTemp = eImpl>
    IBackgroundSubtractorLBSP_(float fRelLBSPThreshold=BGSLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
                               size_t nLBSPThresholdOffset=BGSLBSP_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
                               int nDefaultMedianBlurKernelSize=BGSLBSP_DEFAULT_MEDIAN_BLUR_KERNEL_SIZE,
                               std::enable_if_t<eImplTemp==lv::NonParallel>* /*pUnused*/=0) :
            m_nLBSPThresholdOffset(nLBSPThresholdOffset),
            m_fRelLBSPThreshold(fRelLBSPThreshold),
            m_nDefaultMedianBlurKernelSize(nDefaultMedianBlurKernelSize) {
        lvAssert_(m_fRelLBSPThreshold>=0,"relative threshold for LBSP features must be non-negative");
        IIBackgroundSubtractor::m_nROIBorderSize = LBSP::PATCH_SIZE/2;
    }
#if HAVE_GLSL
    /// glsl impl constructor (defined here as MSVC is very prude with template-class-template-cstor-definitions)
    template<lv::ParallelAlgoType eImplTemp = eImpl>
    IBackgroundSubtractorLBSP_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                               size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                               bool bUseTimers, bool bUseIntegralFormat,
                               float fRelLBSPThreshold=BGSLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
                               size_t nLBSPThresholdOffset=BGSLBSP_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
                               int nDefaultMedianBlurKernelSize=BGSLBSP_DEFAULT_MEDIAN_BLUR_KERNEL_SIZE,
                               std::enable_if_t<eImplTemp==lv::GLSL>* /*pUnused*/=0) :
            IBackgroundSubtractor_GLSL(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nDebugType,bUseDisplay,bUseTimers,bUseIntegralFormat),
            m_nLBSPThresholdOffset(nLBSPThresholdOffset),
            m_fRelLBSPThreshold(fRelLBSPThreshold),
            m_nDefaultMedianBlurKernelSize(nDefaultMedianBlurKernelSize) {
        lvAssert_(m_fRelLBSPThreshold>=0,"relative threshold for LBSP features must be non-negative");
        IIBackgroundSubtractor::m_nROIBorderSize = LBSP::PATCH_SIZE/2;
    }
    /// returns the GLSL compute shader source code for LBSP lookup/description functions
    template<lv::ParallelAlgoType eImplTemp = eImpl> // dont pass arguments here!
    std::enable_if_t<eImplTemp==lv::GLSL,std::string> getLBSPThresholdLUTShaderSource() const;
#endif //HAVE_GLSL
#if HAVE_CUDA
    // ... @@@ add impl later
    static_assert(eImpl!=lv::CUDA),"Missing impl");
#endif //HAVE_CUDA
#if HAVE_OPENCL
    // ... @@@ add impl later
    static_assert(eImpl!=lv::OpenCL),"Missing impl");
#endif //HAVE_OPENCL

    /// required for derived class destruction from this interface
    virtual ~IBackgroundSubtractorLBSP_() {}
    /// common (re)initiaization method for all impl types (should be called in impl-specific initialize func)
    virtual void initialize_common(const cv::Mat& oInitImg, const cv::Mat& oROI) override;
    /// LBSP internal threshold offset value, used to reduce texture noise in dark regions
    const size_t m_nLBSPThresholdOffset;
    /// LBSP relative internal threshold (kept here since we don't keep an LBSP object)
    const float m_fRelLBSPThreshold;
    /// pre-allocated internal LBSP threshold values LUT for all possible 8-bit intensities
    std::array<uchar,UCHAR_MAX+1> m_anLBSPThreshold_8bitLUT;
    /// default kernel size for median blur post-proc filtering
    const int m_nDefaultMedianBlurKernelSize;
    /// copy of latest descriptors (used when refreshing model)
    cv::Mat m_oLastDescFrame;
};

#if HAVE_GLSL
using IBackgroundSubtractorLBSP_GLSL = IBackgroundSubtractorLBSP_<lv::GLSL>;
#endif //HAVE_GLSL
#if HAVE_CUDA
//using IBackgroundSubtractorLBSP_CUDA = IBackgroundSubtractorLBSP_<lv::CUDA>;
#endif //HAVE_CUDA
#if HAVE_OPENCL
//using IBackgroundSubtractorLBSP_OpenCL = IBackgroundSubtractorLBSP_<lv::OpenCL>;
#endif //HAVE_OPENCL
using IBackgroundSubtractorLBSP = IBackgroundSubtractorLBSP_<lv::NonParallel>;
