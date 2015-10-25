
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

#include "litiv/utils/ParallelUtils.hpp"
#include <opencv2/video/background_segm.hpp>

//! enhanced background subtractor interface (inherits from cv::BackgroundSubtractor)
template<ParallelUtils::eParallelAlgoType eImpl, typename enable=void>
class BackgroundSubtractor_;

template<ParallelUtils::eParallelAlgoType eImpl>
class IBackgroundSubtractor :
        public ParallelUtils::ParallelAlgo_<eImpl>,
        public cv::BackgroundSubtractor {
public:
    //! default impl constructor
    template<ParallelUtils::eParallelAlgoType eImplTemp = eImpl>
    IBackgroundSubtractor(size_t nROIBorderSize, typename std::enable_if<eImplTemp==ParallelUtils::eNonParallel>::type* /*pUnused*/=0) :
        ParallelUtils::ParallelAlgo_<ParallelUtils::eNonParallel>(),
        m_nROIBorderSize(nROIBorderSize),
        m_nImgChannels(0),
        m_nImgType(0),
        m_nTotPxCount(0),
        m_nTotRelevantPxCount(0),
        m_nOrigROIPxCount(0),
        m_nFinalROIPxCount(0),
        m_nFrameIdx(SIZE_MAX),
        m_nFramesSinceLastReset(0),
        m_nModelResetCooldown(0),
        m_bInitialized(false),
        m_bModelInitialized(false),
        m_bAutoModelResetEnabled(true),
        m_bUsingMovingCamera(false),
        m_nDebugCoordX(0),
        m_nDebugCoordY(0),
        m_pDebugFS(nullptr) {}
#if HAVE_GLSL
    //! glsl impl constructor
    template<ParallelUtils::eParallelAlgoType eImplTemp = eImpl>
    IBackgroundSubtractor(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                          size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                          bool bUseTimers, bool bUseIntegralFormat, size_t nROIBorderSize=0,
                          typename std::enable_if<eImplTemp==ParallelUtils::eGLSL>::type* /*pUnused*/=0) :
        ParallelUtils::ParallelAlgo_<ParallelUtils::eGLSL>(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,CV_8UC1,nDebugType,true,bUseDisplay,bUseTimers,bUseIntegralFormat),
        m_nROIBorderSize(nROIBorderSize),
        m_nImgChannels(0),
        m_nImgType(0),
        m_nTotPxCount(0),
        m_nTotRelevantPxCount(0),
        m_nOrigROIPxCount(0),
        m_nFinalROIPxCount(0),
        m_nFrameIdx(SIZE_MAX),
        m_nFramesSinceLastReset(0),
        m_nModelResetCooldown(0),
        m_bInitialized(false),
        m_bModelInitialized(false),
        m_bAutoModelResetEnabled(true),
        m_bUsingMovingCamera(false),
        m_nDebugCoordX(0),
        m_nDebugCoordY(0),
        m_pDebugFS(nullptr) {}
#endif //HAVE_GLSL
#if HAVE_CUDA
    static_assert(eImpl!=ParallelUtils::eCUDA),"Missing constr impl");
#endif //HAVE_CUDA
#if HAVE_OPENCL
    static_assert(eImpl!=ParallelUtils::eOpenCL),"Missing constr impl");
#endif //HAVE_OPENCL

    // @@@ add refresh model as virtual pure func here?
    //! (re)initiaization method; needs to be called before starting background subtraction (assumes no specific ROI)
    virtual void initialize(const cv::Mat& oInitImg);
    //! (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
    //! returns the default learning rate value used in 'apply'
    virtual double getDefaultLearningRate() const = 0;
    //! turns automatic model reset on or off
    virtual void setAutomaticModelReset(bool);
    //! modifies the given ROI so it will not cause lookup errors near borders when used in the processing step
    virtual void validateROI(cv::Mat& oROI);
    //! sets the ROI to be used for input analysis (note: this function will reinit the model and return the validated ROI)
    virtual void setROI(cv::Mat& oROI);
    //! returns a copy of the ROI used for input analysis
    cv::Mat getROICopy() const;

protected:

    struct PxInfoBase {
        int nImgCoord_Y;
        int nImgCoord_X;
        size_t nModelIdx;
    };

    //! background model ROI used for input analysis (specific to the input image size)
    cv::Mat m_oROI;
    //! input image size
    cv::Size m_oImgSize;
    //! ROI border size to be ignored, useful for descriptor-based methods
    size_t m_nROIBorderSize;
    //! input image channel size
    size_t m_nImgChannels;
    //! input image type
    int m_nImgType;
    //! total number of pixels (depends on the input frame size) & total number of relevant pixels
    size_t m_nTotPxCount, m_nTotRelevantPxCount;
    //! total number of ROI pixels before & after border cleanup
    size_t m_nOrigROIPxCount, m_nFinalROIPxCount;
    //! current frame index, frame count since last model reset & model reset cooldown counters
    size_t m_nFrameIdx, m_nFramesSinceLastReset, m_nModelResetCooldown;
    //! internal pixel index LUT for all relevant analysis regions (based on the provided ROI)
    std::vector<size_t> m_vnPxIdxLUT;
    //! internal pixel info LUT for all possible pixel indexes
    std::vector<PxInfoBase> m_voPxInfoLUT;
    //! specifies whether the algorithm parameters are fully initialized or not (must be handled by derived class)
    bool m_bInitialized;
    //! specifies whether the model has been fully initialized or not (must be handled by derived class)
    bool m_bModelInitialized;
    //! specifies whether automatic model resets are enabled or not
    bool m_bAutoModelResetEnabled;
    //! specifies whether the camera is considered moving or not
    bool m_bUsingMovingCamera;
    //! the foreground mask generated by the method at [t-1]
    cv::Mat m_oLastFGMask;
    //! copy of latest pixel intensities (used when refreshing model)
    cv::Mat m_oLastColorFrame;

private:
    IBackgroundSubtractor& operator=(const IBackgroundSubtractor&) = delete;
    IBackgroundSubtractor(const IBackgroundSubtractor&) = delete;

public:
    // #### for debug purposes only ####
    int m_nDebugCoordX, m_nDebugCoordY;
    std::string m_sDebugName;
    cv::FileStorage* m_pDebugFS;
};

#if HAVE_GLSL
template<ParallelUtils::eParallelAlgoType eImpl>
class BackgroundSubtractor_<eImpl, typename std::enable_if<eImpl==ParallelUtils::eGLSL>::type> :
        public IBackgroundSubtractor<ParallelUtils::eGLSL> {
public:
    //! glsl impl constructor
    BackgroundSubtractor_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                          size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                          bool bUseTimers, bool bUseIntegralFormat, size_t nROIBorderSize=0);

    //! returns a copy of the latest foreground mask
    void getLatestForegroundMask(cv::OutputArray _oLastFGMask);
    //! model update/segmentation function (asynchronous version, glsl interface); the learning param is used to override the internal learning speed
    void apply_async_glimpl(cv::InputArray _oNextImage, bool bRebindAll, double dLearningRate=-1);
    //! model update/segmentation function (asynchronous version); the learning param is used to override the internal learning speed
    void apply_async(cv::InputArray oNextImage, double dLearningRate=-1);
    //! model update/segmentation function (asynchronous version); the learning param is used to override the internal learning speed
    void apply_async(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, double dLearningRate=-1);
    //! overloads 'apply' from cv::BackgroundSubtractor and redirects it to apply_async
    virtual void apply(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, double dLearningRate=-1);

protected:
    //! used to pass 'apply' learning rate parameter to overloaded dispatch call, if needed
    double m_dCurrLearningRate;
};
typedef BackgroundSubtractor_<ParallelUtils::eGLSL> BackgroundSubtractor_GLSL;
#endif //!HAVE_GLSL

#if HAVE_CUDA
template<ParallelUtils::eParallelAlgoType eImpl>
class BackgroundSubtractor_<eImpl, typename std::enable_if<eImpl==ParallelUtils::eCUDA>::type> :
        public IBackgroundSubtractor<ParallelUtils::eCUDA> {
public:
    static_assert(false,"Missing CUDA impl");
};
typedef BackgroundSubtractor_<ParallelUtils::eCUDA> BackgroundSubtractor_CUDA;
#endif //HAVE_CUDA

#if HAVE_OPENCL
template<ParallelUtils::eParallelAlgoType eImpl>
class BackgroundSubtractor_<eImpl, typename std::enable_if<eImpl==ParallelUtils::eOpenCL>::type> :
        public IBackgroundSubtractor<ParallelUtils::eOpenCL> {
public:
    static_assert(false,"Missing OpenCL impl");
};
typedef BackgroundSubtractor_<ParallelUtils::eOpenCL> BackgroundSubtractor_OpenCL;
#endif //HAVE_OPENCL

template<ParallelUtils::eParallelAlgoType eImpl>
class BackgroundSubtractor_<eImpl, typename std::enable_if<eImpl==ParallelUtils::eNonParallel>::type> :
        public IBackgroundSubtractor<ParallelUtils::eNonParallel> {
public:
    //! default impl constructor
    BackgroundSubtractor_(size_t nROIBorderSize);
};
typedef BackgroundSubtractor_<ParallelUtils::eNonParallel> BackgroundSubtractor;
