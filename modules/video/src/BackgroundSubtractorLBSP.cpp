#include "litiv/video/BackgroundSubtractorLBSP.hpp"
#include <opencv2/imgproc.hpp>
#include <iostream>

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
        ,m_nFrameIdx(SIZE_MAX)
        ,m_nFramesSinceLastReset(0)
        ,m_nModelResetCooldown(0)
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
            std::cout << "\n\tBackgroundSubtractorLBSP : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance.\n" << std::endl;
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
    m_nFrameIdx = 0;
    m_nFramesSinceLastReset = 0;
    m_nModelResetCooldown = 0;
    m_oLastFGMask.create(m_oImgSize,CV_8UC1);
    m_oLastFGMask = cv::Scalar_<uchar>(0);
    m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
    m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
    m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
    m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
    m_vnPxIdxLUT.resize(m_nTotRelevantPxCount);
    m_voPxInfoLUT.resize(m_nTotPxCount);
    if(m_nImgChannels==1) {
        CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width && m_oLastColorFrame.step.p[1]==1);
        CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset)/3);
        for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
            if(m_oROI.data[nPxIter]) {
                m_vnPxIdxLUT[nModelIter] = nPxIter;
                m_voPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nModelIdx = nModelIter;
                m_oLastColorFrame.data[nPxIter] = oInitImg.data[nPxIter];
                const size_t nDescIter = nPxIter*2;
                LBSP::computeDescriptor<1>(oInitImg,oInitImg.data[nPxIter],m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,0,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxIter]],*((ushort*)(m_oLastDescFrame.data+nDescIter)));
                ++nModelIter;
            }
        }
    }
    else { //(m_nImgChannels==3 || m_nImgChannels==4)
        CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width*m_nImgChannels && m_oLastColorFrame.step.p[1]==m_nImgChannels);
        CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset);
        for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
            if(m_oROI.data[nPxIter]) {
                m_vnPxIdxLUT[nModelIter] = nPxIter;
                m_voPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nModelIdx = nModelIter;
                const size_t nPxRGBIter = nPxIter*m_nImgChannels;
                const size_t nDescRGBIter = nPxRGBIter*2;
                if(m_nImgChannels==3) {
                    alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE*8>,3> aanLBSPLookupVals;
                    LBSP::computeDescriptor_lookup(oInitImg,m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,aanLBSPLookupVals);
                    for(size_t c=0; c<3; ++c) {
                        m_oLastColorFrame.data[nPxRGBIter+c] = oInitImg.data[nPxRGBIter+c];
                        LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],oInitImg.data[nPxRGBIter+c],m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
                    }
                }
                else { //m_nImgChannels==4
                    alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE*8>,4> aanLBSPLookupVals;
                    LBSP::computeDescriptor_lookup(oInitImg,m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,aanLBSPLookupVals);
                    for(size_t c=0; c<4; ++c) {
                        m_oLastColorFrame.data[nPxRGBIter+c] = oInitImg.data[nPxRGBIter+c];
                        LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],oInitImg.data[nPxRGBIter+c],m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
                    }
                }
                ++nModelIter;
            }
        }
    }
    m_bInitialized = true;
}

#if HAVE_GPU_SUPPORT

void BackgroundSubtractorLBSP::apply_async(cv::InputArray _oNextImage, cv::OutputArray _oLastFGMask, double dLearningRate) {
    this->apply_async(_oNextImage,dLearningRate);
    this->getLatestForegroundMask(_oLastFGMask);
}

#if HAVE_GLSL

std::string BackgroundSubtractorLBSP::getLBSPThresholdLUTShaderSource() const {
    glAssert(m_bInitialized);
    std::stringstream ssSrc;
    ssSrc << "const uint anLBSPThresLUT[256] = uint[256](\n    ";
    for(size_t t=0; t<=UCHAR_MAX; ++t) {
        if(t>0 && (t%((UCHAR_MAX+1)/8))==(((UCHAR_MAX+1)/8)-1) && t<UCHAR_MAX)
            ssSrc << m_anLBSPThreshold_8bitLUT[t] << ",\n    ";
        else if(t<UCHAR_MAX)
            ssSrc << m_anLBSPThreshold_8bitLUT[t] << ",";
        else
            ssSrc << m_anLBSPThreshold_8bitLUT[t] << "\n";
    }
    ssSrc << ");\n";
    return ssSrc.str();
}

#endif //HAVE_GLSL
#endif //HAVE_GPU_SUPPORT

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
