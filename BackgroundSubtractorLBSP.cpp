#include "BackgroundSubtractorLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <exception>

// local define used to determine the default median blur kernel size
#define DEFAULT_MEDIAN_BLUR_KERNEL_SIZE (9)

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(float fRelLBSPThreshold, size_t nLBSPThresholdOffset)
    :    m_nImgChannels(0)
        ,m_nImgType(0)
        ,m_nLBSPThresholdOffset(nLBSPThresholdOffset)
        ,m_fRelLBSPThreshold(fRelLBSPThreshold)
        ,m_nTotPxCount(0)
        ,m_nTotRelevantPxCount(0)
        ,m_nOrigROIPxCount(0)
        ,m_nFinalROIPxCount(0)
        ,m_nFrameIndex(SIZE_MAX)
        ,m_nFramesSinceLastReset(0)
        ,m_nModelResetCooldown(0)
        ,m_aPxIdxLUT(nullptr)
        ,m_aPxInfoLUT(nullptr)
        ,m_nDefaultMedianBlurKernelSize(DEFAULT_MEDIAN_BLUR_KERNEL_SIZE)
        ,m_bInitialized(false)
        ,m_bAutoModelResetEnabled(true)
        ,m_bUsingMovingCamera(false)
        ,m_nDebugCoordX(0)
        ,m_nDebugCoordY(0)
        ,m_pDebugFS(nullptr) {
    CV_Assert(m_fRelLBSPThreshold>=0);
}

BackgroundSubtractorLBSP::~BackgroundSubtractorLBSP() {}

void BackgroundSubtractorLBSP::initialize(const cv::Mat& oInitImg) {
    this->initialize(oInitImg,cv::Mat());
}

void BackgroundSubtractorLBSP::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
    CV_Assert(oInitImg.isContinuous());
#if HAVE_GLSL
    CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC4);
#else //!HAVE_GLSL
    CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3);
#endif //!HAVE_GLSL
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
            std::cout << std::endl << "\tBackgroundSubtractorLOBSTER : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
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
        cv::dilate(oNewBGROI,oTempROI,cv::Mat(),cv::Point(-1,-1),LBSP::PATCH_SIZE/2);
        cv::bitwise_or(oNewBGROI,oTempROI/2,oNewBGROI);
    }
    m_nOrigROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
    CV_Assert(m_nOrigROIPxCount>0);
    LBSP::validateROI(oNewBGROI);
    m_nFinalROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
    CV_Assert(m_nFinalROIPxCount>0);
    m_bInitialized = false;
    m_oROI = oNewBGROI;
    m_oImgSize = oInitImg.size();
    m_nImgType = oInitImg.type();
    m_nImgChannels = oInitImg.channels();
    m_nTotPxCount = m_oImgSize.area();
    m_nTotRelevantPxCount = m_nFinalROIPxCount;
    m_nFrameIndex = 0;
    m_nFramesSinceLastReset = 0;
    m_nModelResetCooldown = 0;
    m_oLastFGMask.create(m_oImgSize,CV_8UC1);
    m_oLastFGMask = cv::Scalar_<uchar>(0);
    m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
    m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
    m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
    m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
    m_bInitialized = true;
}

cv::AlgorithmInfo* BackgroundSubtractorLBSP::info() const {
    return nullptr;
}

cv::Mat BackgroundSubtractorLBSP::getROICopy() const {
    return m_oROI.clone();
}

void BackgroundSubtractorLBSP::setROI(cv::Mat& oROI) {
    LBSP::validateROI(oROI);
    CV_Assert(cv::countNonZero(oROI)>0);
    if(m_bInitialized) {
        cv::Mat oLatestBackgroundImage;
        getBackgroundImage(oLatestBackgroundImage);
        initialize(oLatestBackgroundImage,oROI);
    }
    else
        m_oROI = oROI.clone();
}

void BackgroundSubtractorLBSP::setAutomaticModelReset(bool bVal) {
    m_bAutoModelResetEnabled = bVal;
}
