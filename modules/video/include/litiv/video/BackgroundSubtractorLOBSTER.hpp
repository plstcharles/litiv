
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

#include "litiv/video/BackgroundSubtractorLBSP.hpp"

/// defines the default value for IBackgroundSubtractorLOBSTER_::m_nDescDistThreshold
#define BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD (4)
/// defines the default value for IBackgroundSubtractorLOBSTER_::m_nColorDistThreshold
#define BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD (30)
/// defines the default value for IBackgroundSubtractorLOBSTER_::m_nBGSamples
#define BGSLOBSTER_DEFAULT_NB_BG_SAMPLES (35)
/// defines the default value for IBackgroundSubtractorLOBSTER_::m_nRequiredBGSamples
#define BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
/// defines the default value for the learning rate passed to cv::BackgroundSubtractor::apply
#define BGSLOBSTER_DEFAULT_LEARNING_RATE (16)

#define BGSLOBSTER_GLSL_USE_DEBUG      0
#define BGSLOBSTER_GLSL_USE_TIMERS     0
#define BGSLOBSTER_GLSL_USE_BASIC_IMPL 0
#define BGSLOBSTER_GLSL_USE_SHAREDMEM  1
#define BGSLOBSTER_GLSL_USE_POSTPROC   1

/**
    LOcal Binary Similarity segmenTER (LOBSTER) algorithm for FG/BG video segmentation via change detection.

    Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).
    For optimal grayscale results, use CV_8UC1 frames instead of CV_8UC3.

    For more details on the different parameters or on the algorithm itself, see P.-L. St-Charles and
    G.-A. Bilodeau, "Improving Background Subtraction using Local Binary Similarity Patterns", in WACV 2014.
*/
template<lv::ParallelAlgoType eImpl>
struct IBackgroundSubtractorLOBSTER_ : public IBackgroundSubtractorLBSP_<eImpl> {
    IBackgroundSubtractorLOBSTER_(size_t nDescDistThreshold=BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
                                  size_t nColorDistThreshold=BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
                                  size_t nBGSamples=BGSLOBSTER_DEFAULT_NB_BG_SAMPLES,
                                  size_t nRequiredBGSamples=BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES,
                                  size_t nLBSPThresholdOffset=BGSLBSP_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
                                  float fRelLBSPThreshold=BGSLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD);
    /// returns the default learning rate value used in 'apply'
    virtual double getDefaultLearningRate() const override {return BGSLOBSTER_DEFAULT_LEARNING_RATE;}
protected:
    /// absolute color distance threshold
    const size_t m_nColorDistThreshold;
    /// absolute descriptor distance threshold
    const size_t m_nDescDistThreshold;
    /// number of different samples per pixel/block to be taken from input frames to build the background model
    const size_t m_nBGSamples;
    /// number of similar samples needed to consider the current pixel/block as 'background'
    const size_t m_nRequiredBGSamples;
};

using IBackgroundSubtractorLOBSTER = IBackgroundSubtractorLOBSTER_<lv::NonParallel>;
template<>
IBackgroundSubtractorLOBSTER::IBackgroundSubtractorLOBSTER_(size_t nDescDistThreshold, size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples, size_t nLBSPThresholdOffset, float fRelLBSPThreshold);
#if HAVE_GLSL
using IBackgroundSubtractorLOBSTER_GLSL = IBackgroundSubtractorLOBSTER_<lv::GLSL>;
template<>
IBackgroundSubtractorLOBSTER_GLSL::IBackgroundSubtractorLOBSTER_(size_t nDescDistThreshold, size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples, size_t nLBSPThresholdOffset, float fRelLBSPThreshold);
#endif //HAVE_GLSL

template<lv::ParallelAlgoType eImpl>
struct BackgroundSubtractorLOBSTER_;

#if HAVE_GLSL
template<>
struct BackgroundSubtractorLOBSTER_<lv::GLSL> : public IBackgroundSubtractorLOBSTER_GLSL {
    /// full constructor
    using IBackgroundSubtractorLOBSTER_GLSL::IBackgroundSubtractorLOBSTER_GLSL;
    /// refreshes all samples based on the last analyzed frame
    void refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate=false);
    /// (re)initiaization method; needs to be called before starting background subtraction
    void initialize_gl(const cv::Mat& oInitImg, const cv::Mat& oROI) override;
    /// returns the GLSL compute shader source code to run for a given algo stage
    virtual std::string getComputeShaderSource(size_t nStage) const override;
    /// returns a copy of the latest reconstructed background image
    virtual void getBackgroundImage(cv::OutputArray oBGImg) const override;
    /// returns a copy of the latest reconstructed background descriptors image
    virtual void getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const override;

protected:
    /// returns the GLSL compute shader source code to run for the main processing stage
    std::string getComputeShaderSource_LOBSTER() const;
    /// returns the GLSL compute shader source code to run the post-processing stage (median blur)
    std::string getComputeShaderSource_PostProc() const;
    /// custom dispatch call function to adjust in-stage uniforms, batch workgroup size & other parameters
    virtual void dispatch(size_t nStage, GLShader& oShader) override;

    size_t m_nTMT32ModelSize;
    size_t m_nSampleStepSize;
    size_t m_nPxModelSize;
    size_t m_nPxModelPadding;
    size_t m_nColStepSize;
    size_t m_nRowStepSize;
    size_t m_nBGModelSize;
    std::aligned_vector<uint,32> m_vnBGModelData;
    std::aligned_vector<lv::gl::TMT32GenParams,32> m_voTMT32ModelData;
    enum LOBSTERStorageBufferBindingList {
        LOBSTERStorageBuffer_BGModelBinding = GLImageProcAlgo::nStorageBufferDefaultBindingsCount,
        LOBSTERStorageBuffer_TMT32ModelBinding,
        nLOBSTERStorageBufferBindingsCount
    };
};

using BackgroundSubtractorLOBSTER_GLSL = BackgroundSubtractorLOBSTER_<lv::GLSL>;
#endif //HAVE_GLSL

#if HAVE_CUDA
// BackgroundSubtractorLOBSTER_<lv::CUDA> will not compile here, missing impl
#endif //HAVE_CUDA

#if HAVE_OPENCL
// BackgroundSubtractorLOBSTER_<lv::OpenCL> will not compile here, missing impl
#endif //HAVE_OPENCL

template<>
struct BackgroundSubtractorLOBSTER_<lv::NonParallel> : public IBackgroundSubtractorLOBSTER {
public:
    /// full constructor
    using IBackgroundSubtractorLOBSTER::IBackgroundSubtractorLOBSTER;
    /// refreshes all samples based on the last analyzed frame
    void refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate=false);
    /// (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) override;
    /// model update/segmentation function (synchronous version); the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray oImage, cv::OutputArray oFGMask, double dLearningRate=BGSLOBSTER_DEFAULT_LEARNING_RATE) override;
    /// returns a copy of the latest reconstructed background image
    virtual void getBackgroundImage(cv::OutputArray oBGImg) const override;
    /// returns a copy of the latest reconstructed background descriptors image
    virtual void getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const override;

protected:
    /// background model pixel intensity samples
    std::vector<cv::Mat> m_voBGColorSamples;
    /// background model descriptors samples
    std::vector<cv::Mat> m_voBGDescSamples;
};

using BackgroundSubtractorLOBSTER = BackgroundSubtractorLOBSTER_<lv::NonParallel>;
