#pragma once

#include "BackgroundSubtractorLBSP.h"

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

/*!
    LOcal Binary Similarity segmenTER (LOBSTER) change detection algorithm.

    Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).
    For optimal grayscale results, use CV_8UC1 frames instead of CV_8UC3.

    For more details on the different parameters or on the algorithm itself, see P.-L. St-Charles and
    G.-A. Bilodeau, "Improving Background Subtraction using Local Binary Similarity Patterns", in WACV 2014.

    This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorLOBSTER :
         public BackgroundSubtractorLBSP
#if HAVE_GLSL
        ,public GLImageProcAlgo
#endif //HAVE_GLSL
        {
public:
    //! full constructor
    BackgroundSubtractorLOBSTER(float fRelLBSPThreshold=BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
                                size_t nLBSPThresholdOffset=BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
                                size_t nDescDistThreshold=BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
                                size_t nColorDistThreshold=BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
                                size_t nBGSamples=BGSLOBSTER_DEFAULT_NB_BG_SAMPLES,
                                size_t nRequiredBGSamples=BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    //! default destructor
    virtual ~BackgroundSubtractorLOBSTER();
    //! (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI);
    //! refreshes all samples based on the last analyzed frame
    virtual void refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate=false);
#if HAVE_GPU_SUPPORT
    //! primary model update function (asynchronous version); the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray oNextImage, double dLearningRate=BGSLOBSTER_DEFAULT_LEARNING_RATE);
    //! returns a copy of the latest foreground mask
    virtual void getLatestForegroundMask(cv::OutputArray oLastFGMask);
#else //!HAVE_GPU_SUPPORT
    //! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray oImage, cv::OutputArray oFGMask, double dLearningRate=BGSLOBSTER_DEFAULT_LEARNING_RATE);
#endif //!HAVE_GPU_SUPPORT
    //! returns a copy of the latest reconstructed background image
    virtual void getBackgroundImage(cv::OutputArray oBGImg) const;
    //! returns a copy of the latest reconstructed background descriptors image
    virtual void getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const;

protected:
    //! absolute color distance threshold
    const size_t m_nColorDistThreshold;
    //! absolute descriptor distance threshold
    const size_t m_nDescDistThreshold;
    //! number of different samples per pixel/block to be taken from input frames to build the background model
    const size_t m_nBGSamples;
    //! number of similar samples needed to consider the current pixel/block as 'background'
    const size_t m_nRequiredBGSamples;
    //! background model pixel intensity samples
    std::vector<cv::Mat> m_voBGColorSamples;
    //! background model descriptors samples
    std::vector<cv::Mat> m_voBGDescSamples;
    //! indicates whether the model has been fully initialized or not
    bool m_bModelInitialized;

#if HAVE_GLSL
public:
    virtual std::string getComputeShaderSource(size_t nStage) const;
    virtual void apply_glimpl(cv::InputArray oNextImage, bool bRebindAll, double dLearningRate);
protected:
    size_t m_nCurrResamplingRate;
    size_t m_nTMT32ModelSize;
    size_t m_nSampleStepSize;
    size_t m_nPxModelSize;
    size_t m_nPxModelPadding;
    size_t m_nColStepSize;
    size_t m_nRowStepSize;
    size_t m_nBGModelSize;
    std::vector<uint> m_vnBGModelData;
    std::vector<GLSLFunctionUtils::TMT32GenParams> m_voTMT32ModelData;
    enum eLOBSTERStorageBufferBindingList {
        eLOBSTERStorageBuffer_BGModelBinding = GLImageProcAlgo::eStorageBufferDefaultBindingsCount,
        eLOBSTERStorageBuffer_TMT32ModelBinding,
        eLOBSTERStorageBufferBindingsCount
    };
    virtual void dispatch(size_t nStage, GLShader& oShader);
#endif //HAVE_GLSL
};

