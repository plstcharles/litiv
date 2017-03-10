
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
#include <fstream>

// these are really empty shells, but we need actual allocation due to ocv's virtual interface
lv::AlignedMatAllocator<16,false> g_oMatAlloc16a = lv::AlignedMatAllocator<16,false>();
lv::AlignedMatAllocator<32,false> g_oMatAlloc32a = lv::AlignedMatAllocator<32,false>();

cv::MatAllocator* lv::getMatAllocator16a() {return (cv::MatAllocator*)&g_oMatAlloc16a;}
cv::MatAllocator* lv::getMatAllocator32a() {return (cv::MatAllocator*)&g_oMatAlloc32a;}

lv::DisplayHelperPtr lv::DisplayHelper::create(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags) {
    struct DisplayHelperWrapper : public DisplayHelper {
        DisplayHelperWrapper(const std::string& _sDisplayName, const std::string& _sDebugFSDirPath, const cv::Size& _oMaxSize, int _nWindowFlags) :
                DisplayHelper(_sDisplayName,_sDebugFSDirPath,_oMaxSize,_nWindowFlags) {}
    };
    return std::make_shared<DisplayHelperWrapper>(sDisplayName,sDebugFSDirPath,oMaxSize,nWindowFlags);
}

lv::DisplayHelper::DisplayHelper(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags) :
        m_sDisplayName(sDisplayName),
        m_oMaxDisplaySize(oMaxSize),
        m_oFS(lv::addDirSlashIfMissing(sDebugFSDirPath)+sDisplayName+".yml",cv::FileStorage::WRITE),
        m_oLastDisplaySize(cv::Size(0,0)),
        m_oLastTileSize(cv::Size(0,0)),
        m_bContinuousUpdates(false),
        m_bFirstDisplay(true),
        m_lInternalCallback(std::bind(&DisplayHelper::onMouseEventCallback,this,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,std::placeholders::_4)) {
    cv::namedWindow(m_sDisplayName,nWindowFlags); // @@@ if it blocks, recompile opencv without Qt (bug still here as of OpenCV 3.1)
    cv::setMouseCallback(m_sDisplayName,onMouseEvent,(void*)&m_lInternalCallback);
}

lv::DisplayHelper::~DisplayHelper() {
    cv::destroyWindow(m_sDisplayName);
}

void lv::DisplayHelper::display(const cv::Mat& oImage, size_t nIdx) {
    display(std::vector<std::vector<std::pair<cv::Mat,std::string>>>{{std::make_pair(oImage,cv::format("Image #%d",(int)nIdx))}},oImage.size());
}
void lv::DisplayHelper::display(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oOutputImg, size_t nIdx) {
    display(std::vector<std::vector<std::pair<cv::Mat,std::string>>>{{
        std::make_pair(oInputImg,cv::format("Input #%d",(int)nIdx)),
        std::make_pair(oDebugImg,std::string("Debug")),
        std::make_pair(oOutputImg,std::string("Output")),
    }},oInputImg.size());
}

void lv::DisplayHelper::display(const std::vector<std::vector<std::pair<cv::Mat,std::string>>>& vvImageNamePairs, const cv::Size& oSuggestedTileSize) {
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
            lvAssert_(oImage.depth()==CV_8U || oImage.depth()==CV_16U || oImage.depth()==CV_32S || oImage.depth()==CV_32F,"all images must be 8u/16u/32s/32f depth");
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
            if(oImage.depth()==CV_16U) // expected input range = [0..USHRT_MAX]
                oImage.convertTo(oImageBYTE3,CV_8U,double(UCHAR_MAX)/(USHRT_MAX));
            else if(oImage.depth()==CV_32S) // expected input range = [INT_MIN..INT_MAX]
                oImage.convertTo(oImageBYTE3,CV_8U,double(UCHAR_MAX)/(INT_MAX),double(UCHAR_MAX/2));
            else if(oImage.depth()==CV_32F) // expected input range = [0,1]
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

void lv::DisplayHelper::setMouseCallback(std::function<void(const CallbackData&)> lCallback) {
    lv::mutex_lock_guard oLock(m_oEventMutex);
    m_lExternalCallback = lCallback;
}

void lv::DisplayHelper::setContinuousUpdates(bool b) {
    m_bContinuousUpdates = b;
}

int lv::DisplayHelper::waitKey(int nDefaultSleepDelay) {
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

void lv::DisplayHelper::onMouseEventCallback(int nEvent, int x, int y, int nFlags) {
    lv::mutex_lock_guard oLock(m_oEventMutex);
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

void lv::DisplayHelper::onMouseEvent(int nEvent, int x, int y, int nFlags, void* pData) {
    (*(std::function<void(int,int,int,int)>*)pData)(nEvent,x,y,nFlags);
}

void lv::write(const std::string& sFilePath, const cv::Mat& _oData, lv::MatArchiveList eArchiveType) {
    lvAssert_(!sFilePath.empty() && !_oData.empty(),"output file path and matrix must both be non-empty");
    cv::Mat oData = _oData.isContinuous()?_oData:_oData.clone();
    if(eArchiveType==MatArchive_FILESTORAGE) {
        cv::FileStorage oArchive(sFilePath,cv::FileStorage::WRITE);
        lvAssert__(oArchive.isOpened(),"could not open archive at '%s' for writing",sFilePath.c_str());
        oArchive << "htag" << lv::getVersionStamp();
        oArchive << "date" << lv::getTimeStamp();
        oArchive << "matrix" << oData;
    }
    else if(eArchiveType==MatArchive_PLAINTEXT) {
        std::ofstream ssStr(sFilePath);
        lvAssert__(ssStr.is_open(),"could not open text file at '%s' for writing",sFilePath.c_str());
        ssStr << "htag " << lv::getVersionStamp() << std::endl;
        ssStr << "date " << lv::getTimeStamp() << std::endl;
        ssStr << "nDataType " << (int32_t)oData.type() << std::endl;
        ssStr << "nDataDepth " << (int32_t)oData.depth() << std::endl;
        ssStr << "nChannels " << (int32_t)oData.channels() << std::endl;
        ssStr << "nElemSize " << (uint64_t)oData.elemSize() << std::endl;
        ssStr << "nElemCount " << (uint64_t)oData.total() << std::endl;
        ssStr << "nDims " << (int32_t)oData.dims << std::endl;
        ssStr << "anSizes";
        for(int nDimIdx=0; nDimIdx<oData.dims; ++nDimIdx)
            ssStr << " " << (int32_t)oData.size[nDimIdx];
        ssStr << std::endl << std::endl;
        if(oData.depth()!=CV_64F)
            _oData.convertTo(oData,CV_64F);
        double* pdData = (double*)oData.data;
        for(int nElemIdx=0; nElemIdx<(int)oData.total(); ++nElemIdx) {
            ssStr << *pdData++;
            for(int nElemPackIdx=1; nElemPackIdx<oData.channels(); ++nElemPackIdx)
                ssStr << " " << *pdData++;
            if(((nElemIdx+1)%oData.size[oData.dims-1])==0)
                ssStr << std::endl;
            else
                ssStr << " ";
        }
        lvAssert_(ssStr,"plain text archive write failed");
    }
    else if(eArchiveType==MatArchive_BINARY) {
        std::ofstream ssStr(sFilePath,std::ios::binary);
        lvAssert__(ssStr.is_open(),"could not open binary file at '%s' for writing",sFilePath.c_str());
        const int32_t nDataType = (int32_t)oData.type();
        ssStr.write((const char*)&nDataType,sizeof(nDataType));
        const uint64_t nElemSize = (uint64_t)oData.elemSize();
        ssStr.write((const char*)&nElemSize,sizeof(nElemSize));
        const uint64_t nElemCount = (uint64_t)oData.total();
        ssStr.write((const char*)&nElemCount,sizeof(nElemCount));
        const int32_t nDims = (int32_t)oData.dims;
        ssStr.write((const char*)&nDims,sizeof(nDims));
        for(int32_t nDimIdx=0; nDimIdx<nDims; ++nDimIdx) {
            const int32_t nDimSize = (int32_t)oData.size[nDimIdx];
            ssStr.write((const char*)&nDimSize,sizeof(nDimSize));
        }
        ssStr.write((const char*)(oData.data),nElemSize*nElemCount);
        lvAssert_(ssStr,"binary archive write failed");
    }
    else
        lvError("unrecognized mat archive type flag");
}

void lv::read(const std::string& sFilePath, cv::Mat& oData, lv::MatArchiveList eArchiveType) {
    lvAssert_(!sFilePath.empty(),"input file path must be non-empty");
    if(eArchiveType==MatArchive_FILESTORAGE) {
        cv::FileStorage oArchive(sFilePath,cv::FileStorage::READ);
        lvAssert__(oArchive.isOpened(),"could not open archive at '%s' for reading",sFilePath.c_str());
        oArchive["matrix"] >> oData;
        lvAssert_(!oData.empty(),"could not read valid matrix from storage");
    }
    else if(eArchiveType==MatArchive_PLAINTEXT) {
        std::ifstream ssStr(sFilePath);
        lvAssert__(ssStr.is_open(),"could not open text file at '%s' for reading",sFilePath.c_str());
        std::string sFieldName,sFieldValue;
        lvAssert_((ssStr >> sFieldName) && sFieldName=="htag" && std::getline(ssStr,sFieldValue),"could not parse 'htag' field from archive");
        lvAssert_((ssStr >> sFieldName) && sFieldName=="date" && std::getline(ssStr,sFieldValue),"could not parse 'date' field from archive");
        int32_t nDataType,nDataDepth,nChannels;
        lvAssert_((ssStr >> sFieldName >> nDataType) && sFieldName=="nDataType","could not parse 'nDataType' field from archive");
        lvAssert_((ssStr >> sFieldName >> nDataDepth) && sFieldName=="nDataDepth","could not parse 'nDataDepth' field from archive");
        lvAssert_((ssStr >> sFieldName >> nChannels) && sFieldName=="nChannels","could not parse 'nChannels' field from archive");
        uint64_t nElemSize,nElemCount;
        lvAssert_((ssStr >> sFieldName >> nElemSize) && sFieldName=="nElemSize","could not parse 'nElemSize' field from archive");
        lvAssert_((ssStr >> sFieldName >> nElemCount) && sFieldName=="nElemCount","could not parse 'nElemCount' field from archive");
        int32_t nDims;
        lvAssert_((ssStr >> sFieldName >> nDims) && sFieldName=="nDims","could not parse 'nDims' field from archive");
        std::vector<int32_t> anSizes(nDims);
        lvAssert_((ssStr >> sFieldName) && sFieldName=="anSizes","could not parse 'anSizes' field from archive");
        for(int32_t nDimIdx=0; nDimIdx<nDims; ++nDimIdx)
            lvAssert_((ssStr >> anSizes[nDimIdx]),"could not parse dim size value from archive");
        cv::Mat oDataTemp(nDims,anSizes.data(),CV_64FC(nChannels));
        double* pdData = (double*)oDataTemp.data;
        for(int nElemIdx=0; nElemIdx<(int)oDataTemp.total(); ++nElemIdx) {
            ssStr >> *pdData++;
            for(int nElemPackIdx=1; nElemPackIdx<oDataTemp.channels(); ++nElemPackIdx)
                ssStr >> *pdData++;
        }
        lvAssert_(ssStr,"plain text archive read failed");
        oDataTemp.convertTo(oData,nDataDepth);
    }
    else if(eArchiveType==MatArchive_BINARY) {
        std::ifstream ssStr(sFilePath,std::ios::binary);
        lvAssert__(ssStr.is_open(),"could not open binary file at '%s' for reading",sFilePath.c_str());
        int32_t nDataType;
        ssStr.read((char*)&nDataType,sizeof(nDataType));
        uint64_t nElemSize;
        ssStr.read((char*)&nElemSize,sizeof(nElemSize));
        uint64_t nElemCount;
        ssStr.read((char*)&nElemCount,sizeof(nElemCount));
        int32_t nDims;
        ssStr.read((char*)&nDims,sizeof(nDims));
        std::vector<int32_t> anSizes(nDims);
        for(int32_t nDimIdx=0; nDimIdx<nDims; ++nDimIdx)
            ssStr.read((char*)&anSizes[nDimIdx],sizeof(anSizes[nDimIdx]));
        oData.create(nDims,anSizes.data(),nDataType);
        ssStr.read((char*)(oData.data),nElemSize*nElemCount);
        lvAssert_(ssStr,"binary archive read failed");
    }
    else
        lvError("unrecognized mat archive type flag");
}

cv::Mat lv::packData(const std::vector<cv::Mat>& vMats, std::vector<lv::MatInfo>* pvOutputPackInfo) {
    if(pvOutputPackInfo!=nullptr) {
        std::vector<lv::MatInfo>& vPackInfo = *pvOutputPackInfo;
        vPackInfo.resize(vMats.size());
        for(size_t nMatIdx=0; nMatIdx<vMats.size(); ++nMatIdx) {
            vPackInfo[nMatIdx].size = vMats[nMatIdx].size;
            vPackInfo[nMatIdx].type = vMats[nMatIdx].type();
        }
    }
    if(vMats.empty())
        return cv::Mat();
    if(vMats.size()==1)
        return vMats[0].clone();
    size_t nTotPacketSize = 0;
    size_t nFirstNonEmptyMatIdx = size_t(-1);
    bool bAllSameType = true;
    for(size_t nMatIdx=0; nMatIdx<vMats.size(); ++nMatIdx) {
        const size_t nCurrPacketSize = vMats[nMatIdx].total()*vMats[nMatIdx].elemSize();
        if(nCurrPacketSize>0) {
            if(nFirstNonEmptyMatIdx==size_t(-1))
                nFirstNonEmptyMatIdx = nMatIdx;
            nTotPacketSize += nCurrPacketSize;
            bAllSameType &= vMats[nMatIdx].type()==vMats[nFirstNonEmptyMatIdx].type();
        }
    }
    if(nTotPacketSize==0)
        return cv::Mat();
    lvDbgAssert_(nTotPacketSize<(size_t)std::numeric_limits<int>::max(),"packed mat data alloc too big");
    lvDbgAssert(nFirstNonEmptyMatIdx!=size_t(-1));
    cv::Mat oPacket;
    if(bAllSameType)
        oPacket.create(1,(int)(nTotPacketSize/vMats[nFirstNonEmptyMatIdx].elemSize()),vMats[nFirstNonEmptyMatIdx].type());
    else
        oPacket.create(1,(int)nTotPacketSize,CV_8UC1);
    size_t nCurrPacketIdxOffset = 0;
    for(size_t nMatIdx=0; nMatIdx<vMats.size(); ++nMatIdx) {
        const size_t nCurrPacketSize = vMats[nMatIdx].total()*vMats[nMatIdx].elemSize();
        if(nCurrPacketSize>0) {
            lvDbgAssert_(nCurrPacketIdxOffset+nCurrPacketSize<=nTotPacketSize,"pack out-of-bounds");
            vMats[nMatIdx].copyTo(cv::Mat(vMats[nMatIdx].dims,vMats[nMatIdx].size,vMats[nMatIdx].type(),oPacket.data+nCurrPacketIdxOffset));
            nCurrPacketIdxOffset += nCurrPacketSize;
        }
    }
    lvDbgAssert_(nCurrPacketIdxOffset==nTotPacketSize,"unpack has leftover data");
    return oPacket;
}

std::vector<cv::Mat> lv::unpackData(const cv::Mat& oPacket, const std::vector<lv::MatInfo>& vPackInfo) {
    if(oPacket.empty()) {
        bool bAllEmpty = true;
        for(size_t nPackInfoIdx=0; nPackInfoIdx<vPackInfo.size(); ++nPackInfoIdx)
            bAllEmpty &= vPackInfo[nPackInfoIdx].size.empty();
        lvAssert_(bAllEmpty,"empty matrix given with non-empty pack");
        return std::vector<cv::Mat>(vPackInfo.size());
    }
    lvAssert_(oPacket.isContinuous(),"input packed matrix must be continuous");
    lvAssert_(!vPackInfo.empty(),"did not provide any mat unpacking info");
    size_t nTotPacketSize = 0;
    for(size_t nPackInfoIdx=0; nPackInfoIdx<vPackInfo.size(); ++nPackInfoIdx)
        nTotPacketSize += vPackInfo[nPackInfoIdx].size.total()*vPackInfo[nPackInfoIdx].type.elemSize();
    lvAssert_(nTotPacketSize>0,"trying to unpack non-empty mat with empty packing info");
    lvAssert_(nTotPacketSize==oPacket.total()*oPacket.elemSize(),"input mat and packing info total byte size mismatch");
    std::vector<cv::Mat> vOutputMats(vPackInfo.size());
    size_t nCurrPacketIdxOffset = 0;
    for(size_t nPackInfoIdx=0; nPackInfoIdx<vPackInfo.size(); ++nPackInfoIdx) {
        const size_t nCurrPacketSize = vPackInfo[nPackInfoIdx].size.total()*vPackInfo[nPackInfoIdx].type.elemSize();
        if(nCurrPacketSize>0) {
            lvDbgAssert_(nCurrPacketIdxOffset+nCurrPacketSize<=nTotPacketSize,"unpack out-of-bounds");
            vOutputMats[nPackInfoIdx] = cv::Mat((int)vPackInfo[nPackInfoIdx].size.dims(),vPackInfo[nPackInfoIdx].size.sizes(),vPackInfo[nPackInfoIdx].type(),(void*)(oPacket.data+nCurrPacketIdxOffset));
            nCurrPacketIdxOffset += nCurrPacketSize;
        }
    }
    lvDbgAssert_(nCurrPacketIdxOffset==nTotPacketSize,"unpack has leftover data");
    return vOutputMats;
}

/** @brief shift the values in a matrix by an (x,y) offset
 *
 * Given a matrix and an integer (x,y) offset, the matrix will be shifted
 * such that:
 *
 *     oInput(a,b) ---> oOutput(a+y,b+x)
 *
 * In the case of a non-integer offset, (e.g. cv::Point2f(-2.1, 3.7)),
 * the shift will be calculated with subpixel precision using bilinear
 * interpolation.
 *
 * All valid OpenCV datatypes are supported. If the source datatype is
 * fixed-point, and a non-integer offset is supplied, the output will
 * be a floating point matrix to preserve subpixel accuracy.
 *
 * All border types are supported. If no border type is supplied, the
 * function defaults to BORDER_CONSTANT with 0-padding.
 *
 * The function supports in-place operation.
 *
 * Some common examples are provided following:
 * \code
 *     // read an image from file
 *     Mat mat = imread(filename);
 *     Mat oOutput;
 *
 *     // Perform Matlab-esque 'circshift' in-place
 *     shift(mat, mat, Point(5, 5), BORDER_WRAP);
 *
 *     // Perform shift with subpixel accuracy, padding the missing pixels with 1s
 *     // NOTE: if mat is of type CV_8U, then it will be converted to type CV_32F
 *     shift(mat, mat, Point2f(-13.7, 3.28), BORDER_CONSTANT, 1);
 *
 *     // Perform subpixel shift, preserving the boundary values
 *     shift(mat, oOutput, Point2f(0.093, 0.125), BORDER_REPLICATE);
 *
 *     // Perform a vanilla shift, integer offset, very fast
 *     shift(mat, oOutput, Point(2, 2));
 * \endcode
 *
 * @param oInput the source matrix
 * @param oOutput the destination matrix, can be the same as source
 * @param vDelta the amount to shift the matrix in (x,y) coordinates. Can be
 * integer of floating point precision
 * @param nFillType the method used to fill the null entries, defaults to BORDER_CONSTANT
 * @param vConstantFillValue the value of the null entries if the fill type is BORDER_CONSTANT
 */
void lv::shift(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Point2f& vDelta, int nFillType, const cv::Scalar& vConstantFillValue) {
    /*
     *  ORIGINAL SOURCE : http://code.opencv.org/issues/2299
     *
     *  Software License Agreement (BSD License)
     *
     *  Copyright (c) 2012, Willow Garage, Inc.
     *  All rights reserved.
     *
     *  Redistribution and use in source and binary forms, with or without
     *  modification, are permitted provided that the following conditions
     *  are met:
     *
     *   * Redistributions of source code must retain the above copyright
     *     notice, this list of conditions and the following disclaimer.
     *   * Redistributions in binary form must reproduce the above
     *     copyright notice, this list of conditions and the following
     *     disclaimer in the documentation and/or other materials provided
     *     with the distribution.
     *   * Neither the name of Willow Garage, Inc. nor the names of its
     *     contributors may be used to endorse or promote products derived
     *     from this software without specific prior written permission.
     *
     *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
     *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
     *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
     *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
     *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
     *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
     *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
     *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
     *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
     *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
     *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
     *  POSSIBILITY OF SUCH DAMAGE.
     *
     *  Original Author:  Hilton Bristow
     *  Created: Aug 23, 2012
     *  Modified: Jan 21, 2015
     */
    lvAssert_(std::abs(vDelta.x)<oInput.cols && std::abs(vDelta.y)<oInput.rows,"delta too big; cannot loop-shift mat data");
    const cv::Point2i vDeltaInt((int)std::ceil(vDelta.x),(int)std::ceil(vDelta.y));
    const cv::Point2f vDeltaSubPx(std::abs(vDelta.x-vDeltaInt.x),std::abs(vDelta.y-vDeltaInt.y));
    const int nTop = (vDeltaInt.y>0)?vDeltaInt.y:0;
    const int nBottom = (vDeltaInt.y<0)?-vDeltaInt.y:0;
    const int nLeft = (vDeltaInt.x>0)?vDeltaInt.x:0;
    const int nRight = (vDeltaInt.x<0)?-vDeltaInt.x:0;
    cv::Mat oPaddedInput;
    cv::copyMakeBorder(oInput,oPaddedInput,nTop,nBottom,nLeft,nRight,nFillType,vConstantFillValue);
    if(vDeltaSubPx.x>std::numeric_limits<float>::epsilon() || vDeltaSubPx.y>std::numeric_limits<float>::epsilon()) {
        switch(oInput.depth()) {
            case CV_32F: {
                cv::Matx<float,1,2> dx(1-vDeltaSubPx.x,vDeltaSubPx.x);
                cv::Matx<float,2,1> dy(1-vDeltaSubPx.y,vDeltaSubPx.y);
                cv::sepFilter2D(oPaddedInput,oPaddedInput,-1,dx,dy,cv::Point(0,0),0,cv::BORDER_CONSTANT);
                break;
            }
            case CV_64F: {
                cv::Matx<double,1,2> dx(1-vDeltaSubPx.x,vDeltaSubPx.x);
                cv::Matx<double,2,1> dy(1-vDeltaSubPx.y,vDeltaSubPx.y);
                cv::sepFilter2D(oPaddedInput,oPaddedInput,-1,dx,dy,cv::Point(0,0),0,cv::BORDER_CONSTANT);
                break;
            }
            default: {
                cv::Matx<float,1,2> dx(1-vDeltaSubPx.x,vDeltaSubPx.x);
                cv::Matx<float,2,1> dy(1-vDeltaSubPx.y,vDeltaSubPx.y);
                oPaddedInput.convertTo(oPaddedInput,CV_32F);
                cv::sepFilter2D(oPaddedInput,oPaddedInput,CV_32F,dx,dy,cv::Point(0,0),0,cv::BORDER_CONSTANT);
                break;
            }
        }
    }
    const cv::Rect oROI = cv::Rect(std::max(-vDeltaInt.x,0),std::max(-vDeltaInt.y,0),0,0)+oInput.size();
    oOutput = oPaddedInput(oROI);
}

void lv::doNotOptimize(const cv::Mat& m) {
    lv::doNotOptimize(m.data);
}