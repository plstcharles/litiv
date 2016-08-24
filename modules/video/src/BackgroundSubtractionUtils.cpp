
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

#include "litiv/video/BackgroundSubtractionUtils.hpp"

void IIBackgroundSubtractor::initialize(const cv::Mat& oInitImg) {
    initialize(oInitImg,cv::Mat());
}

void IIBackgroundSubtractor::setAutomaticModelReset(bool bVal) {
    m_bAutoModelResetEnabled = bVal;
}

void IIBackgroundSubtractor::validateROI(cv::Mat& oROI) const {
    lvAssert_(!oROI.empty() && oROI.type()==CV_8UC1,"provided ROI must be non-empty and of type 8UC1");
    if(m_nROIBorderSize>0) {
        cv::Mat oROI_new(oROI.size(),CV_8UC1,cv::Scalar_<uchar>(0));
        const cv::Rect oROI_inner((int)m_nROIBorderSize,(int)m_nROIBorderSize,oROI.cols-int(m_nROIBorderSize*2),oROI.rows-int(m_nROIBorderSize*2));
        cv::Mat(oROI,oROI_inner).copyTo(cv::Mat(oROI_new,oROI_inner));
        oROI = oROI_new;
    }
}

void IIBackgroundSubtractor::setROI(cv::Mat& oROI) {
    validateROI(oROI);
    lvAssert_(cv::countNonZero(oROI)>0,"provided ROI must have at least one valid pixel");
    if(m_bInitialized) {
        cv::Mat oLatestBackgroundImage;
        getBackgroundImage(oLatestBackgroundImage);
        initialize(oLatestBackgroundImage,oROI);
    }
    else
        m_oROI = oROI.clone();
}

cv::Mat IIBackgroundSubtractor::getROICopy() const {
    return m_oROI.clone();
}

IIBackgroundSubtractor::IIBackgroundSubtractor() :
        m_nROIBorderSize(0),
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
        m_bUsingMovingCamera(false) {}

void IIBackgroundSubtractor::initialize_common(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    lvAssert_(!oInitImg.empty() && oInitImg.isContinuous() && (oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC4),"provided image for initialization must be non-empty, continuous, and of type 8UC1/3/4");
    if(oInitImg.channels()>1) {
        std::vector<cv::Mat> voInitImgs;
        cv::split(oInitImg,voInitImgs);
        bool bFoundChDiff = false;
        for(size_t c=1; c<voInitImgs.size(); ++c)
            if((bFoundChDiff=(cv::countNonZero(voInitImgs[0]!=voInitImgs[c])!=0)))
                break;
        if(!bFoundChDiff)
            std::cerr << "\n\tIIBackgroundSubtractor : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance.\n" << std::endl;
    }
    cv::Mat oNewBGROI;
    if(oROI.empty() && m_oROI.size()!=oInitImg.size()) {
        oNewBGROI.create(oInitImg.size(),CV_8UC1);
        oNewBGROI = cv::Scalar_<uchar>(UCHAR_MAX);
    }
    else if(oROI.empty())
        oNewBGROI = m_oROI; // reuse last ROI if sizes match, and no new ROI is provided
    else {
        lvAssert_(oROI.size()==oInitImg.size() && oROI.type()==CV_8UC1,"provided ROI mat size must be equal to the init frame size, and its type must be 8UC1");
        lvAssert_(cv::countNonZero((oROI<UCHAR_MAX)&(oROI>0))==0,"provided ROI mat values must be 0 or 255 only");
        oNewBGROI = oROI.clone();
        cv::Mat oTempROI;
        cv::dilate(oNewBGROI,oTempROI,cv::Mat(),cv::Point(-1,-1),(int)m_nROIBorderSize);
        cv::bitwise_or(oNewBGROI,oTempROI/2,oNewBGROI); // sets value of pixels close to ROI borders as UCHAR_MAX/2 to help internal bounds check
    }
    m_nOrigROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
    lvAssert_(m_nOrigROIPxCount>0,"provided ROI mat contains no useful pixels");
    validateROI(oNewBGROI);
    m_nFinalROIPxCount = (size_t)cv::countNonZero(oNewBGROI);
    lvAssert_(m_nFinalROIPxCount>0,"provided ROI mat contains no useful pixels away from borders (descriptors will hit image bounds)");
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
        lvAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width && m_oLastColorFrame.step.p[1]==1);
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
        lvAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width*m_nImgChannels && m_oLastColorFrame.step.p[1]==m_nImgChannels);
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

#if HAVE_GLSL

IBackgroundSubtractor_GLSL::IBackgroundSubtractor_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                                                                     size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                                                                     bool bUseTimers, bool bUseIntegralFormat) :
        lv::IParallelAlgo_GLSL(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,CV_8UC1,nDebugType,true,bUseDisplay,bUseTimers,bUseIntegralFormat),
        m_dCurrLearningRate(-1) {}

void IBackgroundSubtractor_GLSL::getLatestForegroundMask(cv::OutputArray _oLastFGMask) {
    lvAssert_(GLImageProcAlgo::m_bFetchingOutput || GLImageProcAlgo::setOutputFetching(true),"algo not initialized with mat output support")
    _oLastFGMask.create(m_oImgSize,CV_8UC1);
    cv::Mat oLastFGMask = _oLastFGMask.getMat();
    if(GLImageProcAlgo::m_nInternalFrameIdx>0)
        GLImageProcAlgo::fetchLastOutput(oLastFGMask);
    else
        oLastFGMask = cv::Scalar_<uchar>(0);
}

void IBackgroundSubtractor_GLSL::initialize_gl(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    lvAssert_(!oInitImg.empty(),"algo requires a valid initialization image as input");
    cv::Mat oCurrROI = oROI;
    if(oCurrROI.empty())
        oCurrROI = cv::Mat(oInitImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
    lv::IParallelAlgo_GLSL::initialize_gl(oInitImg,oROI);
}

void IBackgroundSubtractor_GLSL::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    initialize_gl(oInitImg,oROI);
}

void IBackgroundSubtractor_GLSL::apply_gl(cv::InputArray _oNextImage, bool bRebindAll, double dLearningRate) {
    lvAssert_(m_bInitialized && m_bModelInitialized,"algo must be initialized first");
    m_dCurrLearningRate = dLearningRate;
    cv::Mat oNextInputImg = _oNextImage.getMat();
    lvAssert_(oNextInputImg.type()==m_nImgType && oNextInputImg.size()==m_oImgSize,"input image type/size mismatch with initialization type/size");
    lvAssert_(oNextInputImg.isContinuous(),"input image data must be continuous");
    ++m_nFrameIdx;
    GLImageProcAlgo::apply_gl(oNextInputImg,bRebindAll);
    oNextInputImg.copyTo(m_oLastColorFrame);
}

void IBackgroundSubtractor_GLSL::apply_gl(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, bool bRebindAll, double dLearningRate) {
    apply_gl(oNextImage,bRebindAll,dLearningRate);
    getLatestForegroundMask(oLastFGMask);
}

void IBackgroundSubtractor_GLSL::apply(cv::InputArray oNextImage, cv::OutputArray oLastFGMask, double dLearningRate) {
    apply_gl(oNextImage,oLastFGMask,false,dLearningRate);
}

#endif //HAVE_GLSL
