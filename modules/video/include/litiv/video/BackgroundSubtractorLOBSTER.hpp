#pragma once

#include "litiv/video/BackgroundSubtractorLBSP.hpp"

//! defines the default value for BackgroundSubtractorLBSP::m_fRelLBSPThreshold
#define BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.365f)
//! defines the default value for BackgroundSubtractorLBSP::m_nLBSPThresholdOffset
#define BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD (0)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nDescDistThreshold
#define BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD (4)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nColorDistThreshold
#define BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nBGSamples
#define BGSLOBSTER_DEFAULT_NB_BG_SAMPLES (35)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nRequiredBGSamples
#define BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorLOBSTER::apply
#define BGSLOBSTER_DEFAULT_LEARNING_RATE (16)

#define BGSLOBSTER_GLSL_USE_DEBUG      0
#define BGSLOBSTER_GLSL_USE_TIMERS     0
#define BGSLOBSTER_GLSL_USE_BASIC_IMPL 0
#define BGSLOBSTER_GLSL_USE_SHAREDMEM  1
#define BGSLOBSTER_GLSL_USE_POSTPROC   1

/*!
    LOcal Binary Similarity segmenTER (LOBSTER) algorithm for FG/BG video segmentation via change detection.

    Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).
    For optimal grayscale results, use CV_8UC1 frames instead of CV_8UC3.

    For more details on the different parameters or on the algorithm itself, see P.-L. St-Charles and
    G.-A. Bilodeau, "Improving Background Subtraction using Local Binary Similarity Patterns", in WACV 2014.
 */

class IBackgroundSubtractorLOBSTER {
public:
    //! local param constructor
    IBackgroundSubtractorLOBSTER(size_t nDescDistThreshold=BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
                                 size_t nColorDistThreshold=BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
                                 size_t nBGSamples=BGSLOBSTER_DEFAULT_NB_BG_SAMPLES,
                                 size_t nRequiredBGSamples=BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES);
protected:
    //! absolute color distance threshold
    const size_t m_nColorDistThreshold;
    //! absolute descriptor distance threshold
    const size_t m_nDescDistThreshold;
    //! number of different samples per pixel/block to be taken from input frames to build the background model
    const size_t m_nBGSamples;
    //! number of similar samples needed to consider the current pixel/block as 'background'
    const size_t m_nRequiredBGSamples;
};

template<ParallelUtils::eParallelImplType eImpl=ParallelUtils::eParallelImpl_None, typename enable=void>
class BackgroundSubtractorLOBSTER;

#if HAVE_GLSL
template<ParallelUtils::eParallelImplType eImpl>
class BackgroundSubtractorLOBSTER<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_GLSL>::type> :
        public BackgroundSubtractorLBSP<ParallelUtils::eParallelImpl_GLSL>,
        public IBackgroundSubtractorLOBSTER {
public:
    //! full constructor
    BackgroundSubtractorLOBSTER(float fRelLBSPThreshold=BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
                                size_t nLBSPThresholdOffset=BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
                                size_t nDescDistThreshold=BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
                                size_t nColorDistThreshold=BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
                                size_t nBGSamples=BGSLOBSTER_DEFAULT_NB_BG_SAMPLES,
                                size_t nRequiredBGSamples=BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    //! refreshes all samples based on the last analyzed frame
    void refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate=false);
    //! (re)initiaization method; needs to be called before starting background subtraction
    void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
    //! returns the default learning rate value used in 'apply'
    virtual double getDefaultLearningRate() const {return BGSLOBSTER_DEFAULT_LEARNING_RATE;};
    //! returns the GLSL compute shader source code to run for a given algo stage
    virtual std::string getComputeShaderSource(size_t nStage) const;
    //! returns a copy of the latest reconstructed background image
    virtual void getBackgroundImage(cv::OutputArray oBGImg) const;
    //! returns a copy of the latest reconstructed background descriptors image
    virtual void getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const;

protected:
    //! returns the GLSL compute shader source code to run for the main processing stage
    std::string getComputeShaderSource_LOBSTER() const;
    //! returns the GLSL compute shader source code to run the post-processing stage (median blur)
    std::string getComputeShaderSource_PostProc() const;
    //! custom dispatch call function to adjust in-stage uniforms, batch workgroup size & other parameters
    virtual void dispatch(size_t nStage, GLShader& oShader);

    size_t m_nTMT32ModelSize;
    size_t m_nSampleStepSize;
    size_t m_nPxModelSize;
    size_t m_nPxModelPadding;
    size_t m_nColStepSize;
    size_t m_nRowStepSize;
    size_t m_nBGModelSize;
    std::vector<uint,CxxUtils::AlignAllocator<uint,32>> m_vnBGModelData;
    std::vector<RandUtils::TMT32GenParams,CxxUtils::AlignAllocator<RandUtils::TMT32GenParams,32>> m_voTMT32ModelData;
    enum eLOBSTERStorageBufferBindingList {
        eLOBSTERStorageBuffer_BGModelBinding = GLImageProcAlgo::eStorageBufferDefaultBindingsCount,
        eLOBSTERStorageBuffer_TMT32ModelBinding,
        eLOBSTERStorageBufferBindingsCount
    };
};
typedef BackgroundSubtractorLOBSTER<ParallelUtils::eParallelImpl_GLSL> BackgroundSubtractorLOBSTER_GLSL;
#endif //HAVE_GLSL

#if HAVE_CUDA
template<ParallelUtils::eParallelImplType eImpl>
class BackgroundSubtractorLOBSTER<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_CUDA>::type> :
        public BackgroundSubtractorLBSP<ParallelUtils::eParallelImpl_CUDA>,
        public IBackgroundSubtractorLOBSTER {
        static_assert(false,"Missing CUDA impl");
};
typedef BackgroundSubtractorLOBSTER<ParallelUtils::eParallelImpl_CUDA> BackgroundSubtractorLOBSTER_CUDA;
#endif //HAVE_CUDA

#if HAVE_OPENCL
template<ParallelUtils::eParallelImplType eImpl>
class BackgroundSubtractorLOBSTER<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_OpenCL>::type> :
        public BackgroundSubtractorLBSP<ParallelUtils::eParallelImpl_OpenCL>,
        public IBackgroundSubtractorLOBSTER {
        static_assert(false,"Missing OpenCL impl");
};
typedef BackgroundSubtractorLOBSTER<ParallelUtils::eParallelImpl_OpenCL> BackgroundSubtractorLOBSTER_OpenCL;
#endif //HAVE_OPENCL

template<ParallelUtils::eParallelImplType eImpl>
class BackgroundSubtractorLOBSTER<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_None>::type> :
        public BackgroundSubtractorLBSP<ParallelUtils::eParallelImpl_None>,
        public IBackgroundSubtractorLOBSTER {
public:
    //! full constructor
    BackgroundSubtractorLOBSTER(float fRelLBSPThreshold=BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
                                size_t nLBSPThresholdOffset=BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
                                size_t nDescDistThreshold=BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
                                size_t nColorDistThreshold=BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
                                size_t nBGSamples=BGSLOBSTER_DEFAULT_NB_BG_SAMPLES,
                                size_t nRequiredBGSamples=BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    //! refreshes all samples based on the last analyzed frame
    void refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate=false);
    //! (re)initiaization method; needs to be called before starting background subtraction
    void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
    //! returns the default learning rate value used in 'apply'
    virtual double getDefaultLearningRate() const {return BGSLOBSTER_DEFAULT_LEARNING_RATE;};
    //! model update/segmentation function (synchronous version); the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray oImage, cv::OutputArray oFGMask, double dLearningRate=BGSLOBSTER_DEFAULT_LEARNING_RATE);
    //! returns a copy of the latest reconstructed background image
    virtual void getBackgroundImage(cv::OutputArray oBGImg) const;
    //! returns a copy of the latest reconstructed background descriptors image
    virtual void getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const;

protected:
    //! background model pixel intensity samples
    std::vector<cv::Mat> m_voBGColorSamples;
    //! background model descriptors samples
    std::vector<cv::Mat> m_voBGDescSamples;
};
typedef BackgroundSubtractorLOBSTER<> BackgroundSubtractorLOBSTER_base;
