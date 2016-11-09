
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

cv::DisplayHelperPtr cv::DisplayHelper::create(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags) {
    struct DisplayHelperWrapper : public DisplayHelper {
        DisplayHelperWrapper(const std::string& _sDisplayName, const std::string& _sDebugFSDirPath, const cv::Size& _oMaxSize, int _nWindowFlags) :
                DisplayHelper(_sDisplayName,_sDebugFSDirPath,_oMaxSize,_nWindowFlags) {}
    };
    return std::make_shared<DisplayHelperWrapper>(sDisplayName,sDebugFSDirPath,oMaxSize,nWindowFlags);
}

cv::DisplayHelper::DisplayHelper(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags) :
        m_sDisplayName(sDisplayName),
        m_oMaxDisplaySize(oMaxSize),
        m_oFS(lv::AddDirSlashIfMissing(sDebugFSDirPath)+sDisplayName+".yml",cv::FileStorage::WRITE),
        m_oLastDisplaySize(cv::Size(0,0)),
        m_oLastTileSize(cv::Size(0,0)),
        m_bContinuousUpdates(false),
        m_bFirstDisplay(true),
        m_lInternalCallback(std::bind(&DisplayHelper::onMouseEventCallback,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4)) {
    cv::namedWindow(m_sDisplayName,nWindowFlags); // @@@ if it blocks, recompile opencv without Qt (bug still here as of OpenCV 3.1)
    cv::setMouseCallback(m_sDisplayName,onMouseEvent,(void*)&m_lInternalCallback);
}

cv::DisplayHelper::~DisplayHelper() {
    cv::destroyWindow(m_sDisplayName);
}

void cv::DisplayHelper::display(const cv::Mat& oImage, size_t nIdx) {
    display(std::vector<std::vector<std::pair<cv::Mat,std::string>>>{{std::make_pair(oImage,cv::format("Image #%d",(int)nIdx))}},oImage.size());
}
void cv::DisplayHelper::display(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oOutputImg, size_t nIdx) {
    display(std::vector<std::vector<std::pair<cv::Mat,std::string>>>{{
        std::make_pair(oInputImg,cv::format("Input #%d",(int)nIdx)),
        std::make_pair(oDebugImg,std::string("Debug")),
        std::make_pair(oOutputImg,std::string("Output")),
    }},oInputImg.size());
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
    cv::Size oCurrDisplaySize(int(oSuggestedTileSize.width*nColCount),int(oSuggestedTileSize.height*nRowCount));
    if(m_oMaxDisplaySize.area()>0 && (oCurrDisplaySize.width>m_oMaxDisplaySize.width || oCurrDisplaySize.height>m_oMaxDisplaySize.height)) {
        if(oCurrDisplaySize.width>m_oMaxDisplaySize.width && oCurrDisplaySize.width>oCurrDisplaySize.height)
            oCurrDisplaySize = cv::Size(m_oMaxDisplaySize.width,int(m_oMaxDisplaySize.width*float(oCurrDisplaySize.height)/oCurrDisplaySize.width));
        else
            oCurrDisplaySize = cv::Size(int(m_oMaxDisplaySize.height*(float(oCurrDisplaySize.width)/oCurrDisplaySize.height)),m_oMaxDisplaySize.height);
    }
    const cv::Size oNewTileSize(int(oCurrDisplaySize.width/nColCount),int(oCurrDisplaySize.height/nRowCount));
    const cv::Size oFinalDisplaySize(int(oNewTileSize.width*nColCount),int(oNewTileSize.height*nRowCount));
    const cv::Point2i& oDisplayPt = m_oLatestMouseEvent.oInternalPosition;
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
            if(oDisplayPt.x>=0 && oDisplayPt.y>=0 && oDisplayPt.x<oNewTileSize.width && oDisplayPt.y<oNewTileSize.height && m_oLatestMouseEvent.oTileSize==oNewTileSize)
                cv::circle(oImageBYTE3,oDisplayPt,5,cv::Scalar(255,255,255));
            if(oOutputRow.empty())
                oOutputRow = oImageBYTE3;
            else
                cv::hconcat(oOutputRow,oImageBYTE3,oOutputRow);
        }
        if(nRowIdx==0)
            m_oLastDisplay = oOutputRow;
        else
            cv::vconcat(m_oLastDisplay,oOutputRow,m_oLastDisplay);
    }
    if(m_bFirstDisplay && !m_bContinuousUpdates) {
        putText(m_oLastDisplay,"[Press space to continue]",cv::Scalar_<uchar>(0,0,255),true,cv::Point2i(m_oLastDisplay.cols/2-100,15),1,1.0);
        m_bFirstDisplay = false;
    }
    lvAssert(m_oLastDisplay.size()==oFinalDisplaySize);
    cv::imshow(m_sDisplayName,m_oLastDisplay);
    m_oLastDisplaySize = m_oLastDisplay.size();
    m_oLastTileSize = oNewTileSize;
}

void cv::DisplayHelper::setMouseCallback(std::function<void(const CallbackData&)> lCallback) {
    std::mutex_lock_guard oLock(m_oEventMutex);
    m_lExternalCallback = lCallback;
}

void cv::DisplayHelper::setContinuousUpdates(bool b) {
    m_bContinuousUpdates = b;
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
    m_oLatestMouseEvent.oPosition = m_oLatestMouseEvent.oInternalPosition = cv::Point2i(x,y);
    m_oLatestMouseEvent.oTileSize = m_oLastTileSize;
    m_oLatestMouseEvent.oDisplaySize = m_oLastDisplaySize;
    if(x>=0 && y>=0 && x<m_oLastDisplaySize.width && y<m_oLastDisplaySize.height && m_oLastTileSize.area()>0)
        m_oLatestMouseEvent.oInternalPosition = cv::Point2i(x%m_oLastTileSize.width,y%m_oLastTileSize.height);
    m_oLatestMouseEvent.nEvent = nEvent;
    m_oLatestMouseEvent.nFlags = nFlags;
    if(m_lExternalCallback)
        m_lExternalCallback(m_oLatestMouseEvent);
}

void cv::DisplayHelper::onMouseEvent(int nEvent, int x, int y, int nFlags, void* pData) {
    (*(std::function<void(int,int,int,int)>*)pData)(nEvent,x,y,nFlags);
}
