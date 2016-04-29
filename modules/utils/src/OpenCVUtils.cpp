
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

#include "litiv/utils/OpenCVUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"

cv::DisplayHelperPtr cv::DisplayHelper::create(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags) {
    struct DisplayHelperWrapper : public DisplayHelper {
        DisplayHelperWrapper(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags) :
                DisplayHelper(sDisplayName,sDebugFSDirPath,oMaxSize,nWindowFlags) {}
    };
    return std::make_shared<DisplayHelperWrapper>(sDisplayName,sDebugFSDirPath,oMaxSize,nWindowFlags);
}

cv::DisplayHelper::DisplayHelper(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags) :
        m_sDisplayName(sDisplayName),
        m_oMaxDisplaySize(oMaxSize),
        m_oDebugFS(PlatformUtils::AddDirSlashIfMissing(sDebugFSDirPath)+sDisplayName+"_debug.yml",cv::FileStorage::WRITE),
        m_oLastDisplaySize(cv::Size(0,0)),
        m_bContinuousUpdates(false),
        m_bFirstDisplay(true),
        m_oMouseEventCallback(std::bind(&DisplayHelper::onMouseEventCallback,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4)) {
    cv::namedWindow(m_sDisplayName,nWindowFlags);
    cv::setMouseCallback(m_sDisplayName,onMouseEvent,(void*)&m_oMouseEventCallback);
}

cv::DisplayHelper::~DisplayHelper() {
    cv::destroyWindow(m_sDisplayName);
}

void cv::DisplayHelper::display(const cv::Mat& oImage, size_t nIdx) {
    CV_Assert(!oImage.empty() && (oImage.type()==CV_8UC1 || oImage.type()==CV_8UC3 || oImage.type()==CV_8UC4));
    cv::Mat oImageBYTE3;
    if(oImage.channels()==1)
        cv::cvtColor(oImage,oImageBYTE3,cv::COLOR_GRAY2BGR);
    else if(oImage.channels()==4)
        cv::cvtColor(oImage,oImageBYTE3,cv::COLOR_BGRA2BGR);
    else
        oImageBYTE3 = oImage;
    cv::Size oCurrDisplaySize;
    if(m_oMaxDisplaySize.area()>0 && (oImageBYTE3.cols>m_oMaxDisplaySize.width || oImageBYTE3.rows>m_oMaxDisplaySize.height)) {
        if(oImageBYTE3.cols>m_oMaxDisplaySize.width && oImageBYTE3.cols>oImageBYTE3.rows)
            oCurrDisplaySize = cv::Size(m_oMaxDisplaySize.width,m_oMaxDisplaySize.width*(oImageBYTE3.rows/oImageBYTE3.cols));
        else
            oCurrDisplaySize = cv::Size(m_oMaxDisplaySize.height*(oImageBYTE3.cols/oImageBYTE3.rows),m_oMaxDisplaySize.height);
        cv::resize(oImageBYTE3,oImageBYTE3,oCurrDisplaySize);
    }
    else
        oCurrDisplaySize = oImageBYTE3.size();
    std::stringstream sstr;
    sstr << "Image #" << nIdx;
    putText(oImageBYTE3,sstr.str(),cv::Scalar_<uchar>(0,0,255));
    if(m_bFirstDisplay) {
        putText(oImageBYTE3,"[Press space to continue]",cv::Scalar_<uchar>(0,0,255),true,cv::Point2i(oImageBYTE3.cols/2-40,15));
        m_bFirstDisplay = false;
    }
    std::lock_guard<std::mutex> oLock(m_oEventMutex);
    const cv::Point2i& oDbgPt = m_oLatestMouseEvent.oPosition;
    const cv::Size& oLastDbgSize = m_oLatestMouseEvent.oDisplaySize;
    if(oDbgPt.x>=0 && oDbgPt.y>=0 && oDbgPt.x<oLastDbgSize.width && oDbgPt.y<oLastDbgSize.height) {
        const cv::Point2i oDbgPt_rescaled(int(oCurrDisplaySize.width*(float(oDbgPt.x)/oLastDbgSize.width)),int(oCurrDisplaySize.height*(float(oDbgPt.y)/oLastDbgSize.height)));
        cv::circle(oImageBYTE3,oDbgPt_rescaled,5,cv::Scalar(255,255,255));
    }
    cv::imshow(m_sDisplayName,oImageBYTE3);
    m_oLastDisplaySize = oCurrDisplaySize;
}
void cv::DisplayHelper::display(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oOutputImg, size_t nIdx) {
    CV_Assert(!oInputImg.empty() && (oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3 || oInputImg.type()==CV_8UC4));
    CV_Assert(!oDebugImg.empty() && (oDebugImg.type()==CV_8UC1 || oDebugImg.type()==CV_8UC3 || oDebugImg.type()==CV_8UC4) && oDebugImg.size()==oInputImg.size());
    CV_Assert(!oOutputImg.empty() && (oOutputImg.type()==CV_8UC1 || oOutputImg.type()==CV_8UC3 || oOutputImg.type()==CV_8UC4) && oOutputImg.size()==oInputImg.size());
    cv::Mat oInputImgBYTE3, oDebugImgBYTE3, oOutputImgBYTE3;
    if(oInputImg.channels()==1)
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_GRAY2BGR);
    else if(oInputImg.channels()==4)
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_BGRA2BGR);
    else
        oInputImgBYTE3 = oInputImg;
    if(oDebugImg.channels()==1)
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_GRAY2BGR);
    else if(oDebugImg.channels()==4)
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_BGRA2BGR);
    else
        oDebugImgBYTE3 = oDebugImg;
    if(oOutputImg.channels()==1)
        cv::cvtColor(oOutputImg,oOutputImgBYTE3,cv::COLOR_GRAY2BGR);
    else if(oOutputImg.channels()==4)
        cv::cvtColor(oOutputImg,oDebugImgBYTE3,cv::COLOR_BGRA2BGR);
    else
        oOutputImgBYTE3 = oOutputImg;
    cv::Size oCurrDisplaySize;
    if(m_oMaxDisplaySize.area()>0 && (oOutputImgBYTE3.cols>m_oMaxDisplaySize.width || oOutputImgBYTE3.rows>m_oMaxDisplaySize.height)) {
        if(oOutputImgBYTE3.cols>m_oMaxDisplaySize.width && oOutputImgBYTE3.cols>oOutputImgBYTE3.rows)
            oCurrDisplaySize = cv::Size(m_oMaxDisplaySize.width,m_oMaxDisplaySize.width*(oOutputImgBYTE3.rows/oOutputImgBYTE3.cols));
        else
            oCurrDisplaySize = cv::Size(m_oMaxDisplaySize.height*(oOutputImgBYTE3.cols/oOutputImgBYTE3.rows),m_oMaxDisplaySize.height);
        cv::resize(oInputImgBYTE3,oInputImgBYTE3,oCurrDisplaySize);
        cv::resize(oDebugImgBYTE3,oDebugImgBYTE3,oCurrDisplaySize);
        cv::resize(oOutputImgBYTE3,oOutputImgBYTE3,oCurrDisplaySize);
    }
    else
        oCurrDisplaySize = oOutputImgBYTE3.size();
    std::stringstream sstr;
    sstr << "Input #" << nIdx;
    putText(oInputImgBYTE3,sstr.str(),cv::Scalar_<uchar>(0,0,255));
    putText(oDebugImgBYTE3,"Debug",cv::Scalar_<uchar>(0,0,255));
    putText(oOutputImgBYTE3,"Output",cv::Scalar_<uchar>(0,0,255));
    if(m_bFirstDisplay) {
        putText(oDebugImgBYTE3,"[Press space to continue]",cv::Scalar_<uchar>(0,0,255),true,cv::Point2i(oDebugImgBYTE3.cols/2-100,15),1,1.0);
        m_bFirstDisplay = false;
    }
    std::lock_guard<std::mutex> oLock(m_oEventMutex);
    const cv::Point2i& oDbgPt = m_oLatestMouseEvent.oPosition;
    const cv::Size& oLastDbgSize = m_oLatestMouseEvent.oDisplaySize;
    if(oDbgPt.x>=0 && oDbgPt.y>=0 && oDbgPt.x<oLastDbgSize.width*3 && oDbgPt.y<oLastDbgSize.height) {
        const cv::Point2i oDbgPt_rescaled(int(oCurrDisplaySize.width*(float(oDbgPt.x%oLastDbgSize.width)/oLastDbgSize.width)),int(oCurrDisplaySize.height*(float(oDbgPt.y)/oLastDbgSize.height)));
        cv::circle(oInputImgBYTE3,oDbgPt_rescaled,5,cv::Scalar(255,255,255));
        cv::circle(oDebugImgBYTE3,oDbgPt_rescaled,5,cv::Scalar(255,255,255));
        cv::circle(oOutputImgBYTE3,oDbgPt_rescaled,5,cv::Scalar(255,255,255));
    }
    cv::Mat displayH;
    cv::hconcat(oInputImgBYTE3,oDebugImgBYTE3,displayH);
    cv::hconcat(displayH,oOutputImgBYTE3,displayH);
    cv::imshow(m_sDisplayName,displayH);
    m_oLastDisplaySize = oCurrDisplaySize;
}

int cv::DisplayHelper::waitKey(int nDefaultSleepDelay) {
    int nKeyPressed;
    if(m_bContinuousUpdates)
        nKeyPressed = cv::waitKey(nDefaultSleepDelay);
    else
        nKeyPressed = cv::waitKey(0);
    if(nKeyPressed!=-1)
        nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
    if(nKeyPressed==' ')
        m_bContinuousUpdates = !m_bContinuousUpdates;
    return nKeyPressed;
}

void cv::DisplayHelper::onMouseEventCallback(int nEvent, int x, int y, int nFlags) {
    std::lock_guard<std::mutex> oLock(m_oEventMutex);
    m_oLatestMouseEvent.oPosition = cv::Point2i(x,y);
    m_oLatestMouseEvent.oDisplaySize = m_oLastDisplaySize;
    m_oLatestMouseEvent.nEvent = nEvent;
    m_oLatestMouseEvent.nFlags = nFlags;
}

void cv::DisplayHelper::onMouseEvent(int nEvent, int x, int y, int nFlags, void* pData) {
    (*(std::function<void(int,int,int,int)>*)pData)(nEvent,x,y,nFlags);
}
