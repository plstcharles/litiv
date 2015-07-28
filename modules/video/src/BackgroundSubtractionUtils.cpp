#include "litiv/video/BackgroundSubtractionUtils.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>

BackgroundSubtractorImpl::BackgroundSubtractorImpl(size_t nROIBorderSize) :
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

void BackgroundSubtractorImpl::initialize(const cv::Mat& oInitImg) {
    initialize(oInitImg,cv::Mat());
}

void BackgroundSubtractorImpl::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
    CV_Assert(oInitImg.isContinuous());
    CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC4);
    if(oInitImg.type()!=CV_8UC1) {
        std::vector<cv::Mat> voInitImgChannels;
        cv::split(oInitImg,voInitImgChannels);
        bool bAllChEqual = true;
        for(int c1=0; c1<oInitImg.channels(); ++c1) {
            for(int c2=c1+1; c2<oInitImg.channels(); ++c2) {
                if(cv::countNonZero((voInitImgChannels[c1]!=voInitImgChannels[c2]))) {
                    bAllChEqual = false;
                    break;
                }
            }
        }
        if(bAllChEqual)
            std::cout << "\n\tBackgroundSubtractorLBSP_base : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance.\n" << std::endl;
    }
    cv::Mat oNewBGROI;
    if(oROI.empty() && (m_oROI.empty() || oROI.size()!=oInitImg.size())) {
        oNewBGROI.create(oInitImg.size(),CV_8UC1);
        oNewBGROI = cv::Scalar_<uchar>(UCHAR_MAX);
    }
    else if(oROI.empty())
        oNewBGROI = m_oROI;
    else {
        CV_Assert(oROI.size()==oInitImg.size() && oROI.type()==CV_8UC1);
        CV_Assert(cv::countNonZero((oROI<UCHAR_MAX)&(oROI>0))==0);
        oNewBGROI = oROI.clone();
        cv::Mat oTempROI;
        cv::dilate(oNewBGROI,oTempROI,cv::Mat(),cv::Point(-1,-1),m_nROIBorderSize);
        cv::bitwise_or(oNewBGROI,oTempROI/2,oNewBGROI);
    }
    m_nOrigROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
    CV_Assert(m_nOrigROIPxCount>0);
    validateROI(oNewBGROI);
    m_nFinalROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
    CV_Assert(m_nFinalROIPxCount>0);
    m_bInitialized = false;
    m_bModelInitialized = false;
    m_oROI = oNewBGROI;
    m_oImgSize = oInitImg.size();
    m_nImgType = oInitImg.type();
    m_nImgChannels = oInitImg.channels();
    m_nTotPxCount = m_oImgSize.area();
    m_nTotRelevantPxCount = m_nFinalROIPxCount;
    m_nFrameIdx = 0;
    m_nFramesSinceLastReset = 0;
    m_nModelResetCooldown = 0;
    m_oLastFGMask.create(m_oImgSize,CV_8UC1);
    m_oLastFGMask = cv::Scalar_<uchar>(0);
    m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
    m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
    m_vnPxIdxLUT.resize(m_nTotRelevantPxCount);
    m_voPxInfoLUT.resize(m_nTotPxCount);
    if(m_nImgChannels==1) {
        CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width && m_oLastColorFrame.step.p[1]==1);
        for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
            if(m_oROI.data[nPxIter]) {
                m_vnPxIdxLUT[nModelIter] = nPxIter;
                m_voPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nModelIdx = nModelIter;
                m_oLastColorFrame.data[nPxIter] = oInitImg.data[nPxIter];
                ++nModelIter;
            }
        }
    }
    else { //(m_nImgChannels==3 || m_nImgChannels==4)
        CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width*m_nImgChannels && m_oLastColorFrame.step.p[1]==m_nImgChannels);
        for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
            if(m_oROI.data[nPxIter]) {
                m_vnPxIdxLUT[nModelIter] = nPxIter;
                m_voPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nModelIdx = nModelIter;
                const size_t nPxRGBIter = nPxIter*m_nImgChannels;
                if(m_nImgChannels==3)
                    for(size_t c=0; c<3; ++c)
                        m_oLastColorFrame.data[nPxRGBIter+c] = oInitImg.data[nPxRGBIter+c];
                else //m_nImgChannels==4
                    for(size_t c=0; c<4; ++c)
                        m_oLastColorFrame.data[nPxRGBIter+c] = oInitImg.data[nPxRGBIter+c];
                ++nModelIter;
            }
        }
    }
}

void BackgroundSubtractorImpl::validateROI(cv::Mat& oROI) {
    CV_Assert(!oROI.empty() && oROI.type()==CV_8UC1);
    if(m_nROIBorderSize>0) {
        cv::Mat oROI_new(oROI.size(),CV_8UC1,cv::Scalar_<uchar>(0));
        const cv::Rect nROI_inner(m_nROIBorderSize,m_nROIBorderSize,oROI.cols-m_nROIBorderSize*2,oROI.rows-m_nROIBorderSize*2);
        cv::Mat(oROI,nROI_inner).copyTo(cv::Mat(oROI_new,nROI_inner));
        oROI = oROI_new;
    }
}

void BackgroundSubtractorImpl::setROI(cv::Mat& oROI) {
    validateROI(oROI);
    CV_Assert(cv::countNonZero(oROI)>0);
    if(m_bInitialized) {
        cv::Mat oLatestBackgroundImage;
        getBackgroundImage(oLatestBackgroundImage);
        initialize(oLatestBackgroundImage,oROI);
    }
    else
        m_oROI = oROI.clone();
}

cv::Mat BackgroundSubtractorImpl::getROICopy() const {
    return m_oROI.clone();
}

void BackgroundSubtractorImpl::setAutomaticModelReset(bool bVal) {
    m_bAutoModelResetEnabled = bVal;
}

#if HAVE_GLSL

template<>
BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::BackgroundSubtractorParallelImpl( size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs,
                                                                                                       size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures,
                                                                                                       int nDebugType, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
    ParallelUtils::ParallelImpl<ParallelUtils::eParallelImpl_GLSL>(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,CV_8UC1,nDebugType,true,bUseDisplay,bUseTimers,bUseIntegralFormat),
    BackgroundSubtractorImpl(0),
    m_dCurrLearningRate(-1) {}

template<>
void BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::getLatestForegroundMask(cv::OutputArray _oLastFGMask) {
    _oLastFGMask.create(m_oImgSize,CV_8UC1);
    cv::Mat oLastFGMask = _oLastFGMask.getMat();
    if(!GLImageProcAlgo::m_bFetchingOutput)
    glAssert(GLImageProcAlgo::setOutputFetching(true))
    else if(m_nFrameIdx>0)
        GLImageProcAlgo::fetchLastOutput(oLastFGMask);
    else
        oLastFGMask = cv::Scalar_<uchar>(0);
};

template<>
void BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply_async_glimpl(cv::InputArray _oNextImage, bool bRebindAll, double dLearningRate) {
    glAssert(m_bInitialized && m_bModelInitialized);
    m_dCurrLearningRate = dLearningRate;
    cv::Mat oNextInputImg = _oNextImage.getMat();
    CV_Assert(oNextInputImg.type()==m_nImgType && oNextInputImg.size()==m_oImgSize);
    CV_Assert(oNextInputImg.isContinuous());
    ++m_nFrameIdx;
    GLImageProcAlgo::apply_async(oNextInputImg,bRebindAll);
    oNextInputImg.copyTo(m_oLastColorFrame);
};

template<>
void BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply_async(cv::InputArray oNextImage, double dLearningRate) {
    apply_async_glimpl(oNextImage,false,dLearningRate);
};

template<>
void BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply_async(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, double dLearningRate) {
    apply_async(oNextImage,dLearningRate);
    getLatestForegroundMask(oLastFGMask);
};

template<>
void BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, double dLearningRate) {
    apply_async(oNextImage,oLastFGMask,dLearningRate);
}

template class BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_GLSL>;
#endif //HAVE_GLSL

#if HAVE_CUDA
// ... @@@ add impl later
//template class BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_CUDA>;
#endif //HAVE_CUDA

#if HAVE_OPENCL
// ... @@@ add impl later
//template class BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_OpenCL>;
#endif //HAVE_OPENCL

template<>
BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_None>::BackgroundSubtractorParallelImpl(size_t nROIBorderSize) :
        BackgroundSubtractorImpl(nROIBorderSize) {}

template class BackgroundSubtractorParallelImpl<ParallelUtils::eParallelImpl_None>;
