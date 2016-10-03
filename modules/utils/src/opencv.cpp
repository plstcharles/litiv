
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

#include "litiv/utils/opencv.hpp"
#include "litiv/utils/platform.hpp"

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
        m_oDebugFS(lv::AddDirSlashIfMissing(sDebugFSDirPath)+sDisplayName+"_debug.yml",cv::FileStorage::WRITE),
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
    lvAssert_(!oImage.empty() && (oImage.type()==CV_8UC1 || oImage.type()==CV_8UC3 || oImage.type()==CV_8UC4),"image to display must be non-empty, and of type 8UC1/8UC3/8UC4");
    cv::Mat oImageBYTE3;
    if(oImage.channels()==1)
        cv::cvtColor(oImage,oImageBYTE3,cv::COLOR_GRAY2BGR);
    else if(oImage.channels()==4)
        cv::cvtColor(oImage,oImageBYTE3,cv::COLOR_BGRA2BGR);
    else
        oImageBYTE3 = oImage.clone();
    cv::Size oCurrDisplaySize;
    if(m_oMaxDisplaySize.area()>0 && (oImageBYTE3.cols>m_oMaxDisplaySize.width || oImageBYTE3.rows>m_oMaxDisplaySize.height)) {
        if(oImageBYTE3.cols>m_oMaxDisplaySize.width && oImageBYTE3.cols>oImageBYTE3.rows)
            oCurrDisplaySize = cv::Size(m_oMaxDisplaySize.width,int(m_oMaxDisplaySize.width*(float(oImageBYTE3.rows)/oImageBYTE3.cols)));
        else
            oCurrDisplaySize = cv::Size(int(m_oMaxDisplaySize.height*(float(oImageBYTE3.cols)/oImageBYTE3.rows)),m_oMaxDisplaySize.height);
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
    std::mutex_lock_guard oLock(m_oEventMutex);
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
    lvAssert_(!oInputImg.empty() && (oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3 || oInputImg.type()==CV_8UC4),"input image must be 8UC1/8UC3/8UC4");
    lvAssert_(!oDebugImg.empty() && (oDebugImg.type()==CV_8UC1 || oDebugImg.type()==CV_8UC3 || oDebugImg.type()==CV_8UC4),"debug image must be 8UC1/8UC3/8UC4");
    lvAssert_(!oOutputImg.empty() && (oOutputImg.type()==CV_8UC1 || oOutputImg.type()==CV_8UC3 || oOutputImg.type()==CV_8UC4),"output image must be 8UC1/8UC3/8UC4");
    lvAssert_(oOutputImg.size()==oInputImg.size() && oDebugImg.size()==oInputImg.size(),"all provided mat sizes must match");
    cv::Mat oInputImgBYTE3, oDebugImgBYTE3, oOutputImgBYTE3;
    if(oInputImg.channels()==1)
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_GRAY2BGR);
    else if(oInputImg.channels()==4)
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_BGRA2BGR);
    else
        oInputImgBYTE3 = oInputImg.clone();
    if(oDebugImg.channels()==1)
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_GRAY2BGR);
    else if(oDebugImg.channels()==4)
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_BGRA2BGR);
    else
        oDebugImgBYTE3 = oDebugImg.clone();
    if(oOutputImg.channels()==1)
        cv::cvtColor(oOutputImg,oOutputImgBYTE3,cv::COLOR_GRAY2BGR);
    else if(oOutputImg.channels()==4)
        cv::cvtColor(oOutputImg,oDebugImgBYTE3,cv::COLOR_BGRA2BGR);
    else
        oOutputImgBYTE3 = oOutputImg.clone();
    cv::Size oCurrDisplaySize;
    if(m_oMaxDisplaySize.area()>0 && (oOutputImgBYTE3.cols*3>m_oMaxDisplaySize.width || oOutputImgBYTE3.rows>m_oMaxDisplaySize.height)) {
        if(oOutputImgBYTE3.cols*3>m_oMaxDisplaySize.width && oOutputImgBYTE3.cols>oOutputImgBYTE3.rows)
            oCurrDisplaySize = cv::Size((m_oMaxDisplaySize.width/3),int((m_oMaxDisplaySize.width/3)*(float(oOutputImgBYTE3.rows)/oOutputImgBYTE3.cols)));
        else
            oCurrDisplaySize = cv::Size(int(m_oMaxDisplaySize.height*(float(oOutputImgBYTE3.cols)/oOutputImgBYTE3.rows)),m_oMaxDisplaySize.height);
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
    std::mutex_lock_guard oLock(m_oEventMutex);
    const cv::Point2i& oDisplayPt = m_oLatestMouseEvent.oPosition;
    const cv::Size& oLastDisplaySize = m_oLatestMouseEvent.oDisplaySize;
    if(oDisplayPt.x>=0 && oDisplayPt.y>=0 && oDisplayPt.x<oLastDisplaySize.width && oDisplayPt.y<oLastDisplaySize.height) {
        const cv::Point2i oDisplayPt_rescaled(int(oCurrDisplaySize.width*(float(oDisplayPt.x%(oLastDisplaySize.width/3))/(oLastDisplaySize.width/3))),int(oCurrDisplaySize.height*(float(oDisplayPt.y)/oLastDisplaySize.height)));
        cv::circle(oInputImgBYTE3,oDisplayPt_rescaled,5,cv::Scalar(255,255,255));
        cv::circle(oDebugImgBYTE3,oDisplayPt_rescaled,5,cv::Scalar(255,255,255));
        cv::circle(oOutputImgBYTE3,oDisplayPt_rescaled,5,cv::Scalar(255,255,255));
    }
    cv::Mat displayH;
    cv::hconcat(oInputImgBYTE3,oDebugImgBYTE3,displayH);
    cv::hconcat(displayH,oOutputImgBYTE3,displayH);
    cv::imshow(m_sDisplayName,displayH);
    m_oLastDisplaySize = displayH.size();
}

void cv::DisplayHelper::display(const std::vector<std::vector<std::pair<cv::Mat,std::string>>>& vvImageNamePairs, const cv::Size& oSuggestedTileSize) {
    lvAssert_(!vvImageNamePairs.empty(),"must provide at least one row to display");
    lvAssert_(oSuggestedTileSize.area()>0,"must provide non-null tile size");
    const size_t nRowCount = vvImageNamePairs.size();
    size_t nColCount = SIZE_MAX;
    for(size_t nRowIdx=0; nRowIdx<nRowCount; ++nRowIdx) {
        lvAssert_(!vvImageNamePairs[nRowIdx].empty(),"must provide at least one column to display");
        lvAssert_(nColCount==SIZE_MAX || vvImageNamePairs[nRowIdx].size()==nColCount,"image map column count mismatch");
        nColCount = vvImageNamePairs[nRowIdx].size();
        for(size_t nColIdx=0; nColIdx<nColCount; ++nColIdx) {
            const cv::Mat& oImage = vvImageNamePairs[nRowIdx][nColIdx].first;
            lvAssert_(!oImage.empty(),"all images must be non-null");
            lvAssert_(oImage.channels()==1 || oImage.channels()==3 || oImage.channels()==4,"all images must be 1/3/4 channels");
            lvAssert_(oImage.depth()==CV_8U || oImage.depth()==CV_16U || oImage.depth()==CV_32F,"all images must be 8u/16u/32f depth");
        }
    }
    cv::Size oCurrDisplaySize(oSuggestedTileSize.width*nColCount,oSuggestedTileSize.height*nRowCount);
    if(m_oMaxDisplaySize.area()>0 && (oCurrDisplaySize.width>m_oMaxDisplaySize.width || oCurrDisplaySize.height>m_oMaxDisplaySize.height)) {
        if(oCurrDisplaySize.width>m_oMaxDisplaySize.width && oCurrDisplaySize.width>oCurrDisplaySize.height)
            oCurrDisplaySize = cv::Size(m_oMaxDisplaySize.width,int(m_oMaxDisplaySize.width*float(oCurrDisplaySize.height)/oCurrDisplaySize.width));
        else
            oCurrDisplaySize = cv::Size(int(m_oMaxDisplaySize.height*(float(oCurrDisplaySize.width)/oCurrDisplaySize.height)),m_oMaxDisplaySize.height);
    }
    const cv::Size oNewTileSize(oCurrDisplaySize.width/nColCount,oCurrDisplaySize.height/nRowCount);
    const cv::Size oFinalDisplaySize(oNewTileSize.width*nColCount,oNewTileSize.height*nRowCount);
    std::mutex_lock_guard oLock(m_oEventMutex);
    const cv::Point2i& oDisplayPt = m_oLatestMouseEvent.oPosition;
    const cv::Size& oLastDisplaySize = m_oLatestMouseEvent.oDisplaySize;
    cv::Mat oOutput;
    for(size_t nRowIdx=0; nRowIdx<nRowCount; ++nRowIdx) {
        cv::Mat oOutputRow;
        for(size_t nColIdx=0; nColIdx<nColCount; ++nColIdx) {
            const cv::Mat& oImage = vvImageNamePairs[nRowIdx][nColIdx].first;
            cv::Mat oImageBYTE3;
            if(oImage.depth()==CV_16U)
                oImage.convertTo(oImageBYTE3,CV_8U,double(UCHAR_MAX)/(USHRT_MAX));
            else if(oImage.depth()==CV_32F)
                oImage.convertTo(oImageBYTE3,CV_8U,double(UCHAR_MAX));
            else
                oImageBYTE3 = oImage.clone();
            if(oImageBYTE3.channels()==1)
                cv::cvtColor(oImageBYTE3,oImageBYTE3,cv::COLOR_GRAY2BGR);
            else if(oImageBYTE3.channels()==4)
                cv::cvtColor(oImageBYTE3,oImageBYTE3,cv::COLOR_BGRA2BGR);
            if(oImageBYTE3.size()!=oNewTileSize)
                cv::resize(oImageBYTE3,oImageBYTE3,oNewTileSize);
            if(!vvImageNamePairs[nRowIdx][nColIdx].second.empty())
                putText(oImageBYTE3,vvImageNamePairs[nRowIdx][nColIdx].second,cv::Scalar_<uchar>(0,0,255));
            if(oDisplayPt.x>=0 && oDisplayPt.y>=0 && oDisplayPt.x<oLastDisplaySize.width && oDisplayPt.y<oLastDisplaySize.height && oLastDisplaySize==oFinalDisplaySize) {
                const cv::Point2i oDisplayPt_raw(oDisplayPt.x%oNewTileSize.width,oDisplayPt.y%oNewTileSize.height);
                cv::circle(oImageBYTE3,oDisplayPt_raw,5,cv::Scalar(255,255,255));
            }
            if(oOutputRow.empty())
                oOutputRow = oImageBYTE3;
            else
                cv::hconcat(oOutputRow,oImageBYTE3,oOutputRow);
        }
        if(oOutput.empty())
            oOutput = oOutputRow;
        else
            cv::vconcat(oOutput,oOutputRow,oOutput);
    }
    if(m_bFirstDisplay) {
        putText(oOutput,"[Press space to continue]",cv::Scalar_<uchar>(0,0,255),true,cv::Point2i(oOutput.cols/2-100,15),1,1.0);
        m_bFirstDisplay = false;
    }
    lvAssert(oOutput.size()==oFinalDisplaySize);
    cv::imshow(m_sDisplayName,oOutput);
    m_oLastDisplaySize = oOutput.size();
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
    std::mutex_lock_guard oLock(m_oEventMutex);
    m_oLatestMouseEvent.oPosition = cv::Point2i(x,y);
    m_oLatestMouseEvent.oDisplaySize = m_oLastDisplaySize;
    m_oLatestMouseEvent.nEvent = nEvent;
    m_oLatestMouseEvent.nFlags = nFlags;
}

void cv::DisplayHelper::onMouseEvent(int nEvent, int x, int y, int nFlags, void* pData) {
    (*(std::function<void(int,int,int,int)>*)pData)(nEvent,x,y,nFlags);
}
