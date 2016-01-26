
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

#include "litiv/datasets/DatasetUtils.hpp"

#define HARDCODE_FRAME_INDEX               0 // for sync debug only! will corrupt data for non-image packets
#define PRECACHE_CONSOLE_DEBUG             0
#define PRECACHE_REQUEST_TIMEOUT_MS        1
#define PRECACHE_QUERY_TIMEOUT_MS          10
#define PRECACHE_MAX_CACHE_SIZE_GB         6LLU
#define PRECACHE_MAX_CACHE_SIZE            (((PRECACHE_MAX_CACHE_SIZE_GB*1024)*1024)*1024)
#if (!(defined(_M_X64) || defined(__amd64__)) && PRECACHE_MAX_CACHE_SIZE_GB>2)
#error "Cache max size exceeds system limit (x86)."
#endif //(!(defined(_M_X64) || defined(__amd64__)) && PRECACHE_MAX_CACHE_SIZE_GB>2)

void litiv::writeOnImage(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom) {
    cv::putText(oImg,sText,cv::Point(4,bBottom?(oImg.rows-15):15),cv::FONT_HERSHEY_PLAIN,1.2,vColor,2,cv::LINE_AA);
}

cv::Mat litiv::getDisplayImage(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oOutputImg, size_t nIdx, cv::Point oDbgPt, cv::Size oRefSize) {
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
    if(oDbgPt.x>=0 && oDbgPt.y>=0 && oDbgPt.x<oInputImg.cols && oDbgPt.y<oInputImg.rows) {
        cv::circle(oInputImgBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
        cv::circle(oOutputImgBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
    }
    if(oRefSize.width>0 && oRefSize.height>0) {
        cv::resize(oInputImgBYTE3,oInputImgBYTE3,oRefSize);
        cv::resize(oDebugImgBYTE3,oDebugImgBYTE3,oRefSize);
        cv::resize(oOutputImgBYTE3,oOutputImgBYTE3,oRefSize);
    }

    std::stringstream sstr;
    sstr << "Input #" << nIdx;
    writeOnImage(oInputImgBYTE3,sstr.str(),cv::Scalar_<uchar>(0,0,255));
    writeOnImage(oDebugImgBYTE3,"Debug",cv::Scalar_<uchar>(0,0,255));
    writeOnImage(oOutputImgBYTE3,"Output",cv::Scalar_<uchar>(0,0,255));

    cv::Mat displayH;
    cv::hconcat(oInputImgBYTE3,oDebugImgBYTE3,displayH);
    cv::hconcat(displayH,oOutputImgBYTE3,displayH);
    return displayH;
}

void litiv::validateKeyPoints(const cv::Mat& oROI, std::vector<cv::KeyPoint>& voKPs) {
    std::vector<cv::KeyPoint> voNewKPs;
    for(size_t k=0; k<voKPs.size(); ++k) {
        if( voKPs[k].pt.x>=0 && voKPs[k].pt.x<oROI.cols &&
            voKPs[k].pt.y>=0 && voKPs[k].pt.y<oROI.rows &&
            oROI.at<uchar>(voKPs[k].pt)>0)
            voNewKPs.push_back(voKPs[k]);
    }
    voKPs = voNewKPs;
}

litiv::DataPrecacher::DataPrecacher(std::function<const cv::Mat&(size_t)> lCallback) :
        m_lCallback(lCallback) {
    CV_Assert(lCallback);
    m_bIsPrecaching = false;
    m_nLastReqIdx = size_t(-1);
}

litiv::DataPrecacher::~DataPrecacher() {
    stopPrecaching();
}

const cv::Mat& litiv::DataPrecacher::getPacket(size_t nIdx) {
    if(!m_bIsPrecaching)
        return getPacket_internal(nIdx);
    CV_Assert(nIdx<m_nPacketCount);
    std::unique_lock<std::mutex> sync_lock(m_oSyncMutex);
    m_nReqIdx = nIdx;
    std::cv_status res;
    do {
        m_oReqCondVar.notify_one();
        res = m_oSyncCondVar.wait_for(sync_lock,std::chrono::milliseconds(PRECACHE_REQUEST_TIMEOUT_MS));
#if PRECACHE_CONSOLE_DEBUG
        if(res==std::cv_status::timeout)
            std::cout << " # retrying request..." << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
    } while(res==std::cv_status::timeout);
    return m_oReqPacket;
}

bool litiv::DataPrecacher::startPrecaching(size_t nTotPacketCount, size_t nSuggestedBufferSize) {
    static_assert(PRECACHE_REQUEST_TIMEOUT_MS>0,"Precache request timeout must be a positive value");
    static_assert(PRECACHE_QUERY_TIMEOUT_MS>0,"Precache query timeout must be a positive value");
    static_assert(PRECACHE_MAX_CACHE_SIZE>=(size_t)0,"Precache size must be a non-negative value");
    CV_Assert(nTotPacketCount);
    m_nPacketCount = nTotPacketCount;
    if(m_bIsPrecaching)
        stopPrecaching();
    if(nSuggestedBufferSize>0) {
        m_bIsPrecaching = true;
        m_nBufferSize = (nSuggestedBufferSize>PRECACHE_MAX_CACHE_SIZE)?(PRECACHE_MAX_CACHE_SIZE):nSuggestedBufferSize;
        m_qoCache.clear();
        m_vcBuffer.resize(m_nBufferSize);
        m_nNextExpectedReqIdx = 0;
        m_nNextPrecacheIdx = 0;
        m_nReqIdx = m_nLastReqIdx = size_t(-1);
        m_hPrecacher = std::thread(&DataPrecacher::precache,this);
    }
    return m_bIsPrecaching;
}

void litiv::DataPrecacher::stopPrecaching() {
    if(m_bIsPrecaching) {
        m_bIsPrecaching = false;
        m_hPrecacher.join();
    }
}

void litiv::DataPrecacher::precache() {
    std::unique_lock<std::mutex> sync_lock(m_oSyncMutex);
#if PRECACHE_CONSOLE_DEBUG
    std::cout << " @ initializing precaching with buffer size = " << (m_nBufferSize/1024)/1024 << " mb" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
    m_nFirstBufferIdx = m_nNextBufferIdx = 0;
    while(m_nNextPrecacheIdx<m_nPacketCount) {
        const cv::Mat& oNextPacket = getPacket_internal(m_nNextPrecacheIdx);
        const size_t nNextPacketSize = oNextPacket.total()*oNextPacket.elemSize();
        if(m_nNextBufferIdx+nNextPacketSize<m_nBufferSize) {
            cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),m_vcBuffer.data()+m_nNextBufferIdx);
            oNextPacket.copyTo(oNextPacket_cache);
            m_qoCache.push_back(oNextPacket_cache);
            m_nNextBufferIdx += nNextPacketSize;
            ++m_nNextPrecacheIdx;
        }
        else break;
    }
    while(m_bIsPrecaching) {
        if(m_oReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(m_nNextPrecacheIdx==m_nPacketCount?PRECACHE_QUERY_TIMEOUT_MS*32:PRECACHE_QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
            if(m_nReqIdx!=m_nNextExpectedReqIdx-1) {
                if(!m_qoCache.empty()) {
                    if(m_nReqIdx<m_nNextPrecacheIdx && m_nReqIdx>=m_nNextExpectedReqIdx) {
//#if PRECACHE_CONSOLE_DEBUG
//                        std::cout << " -- popping " << m_nReqIdx-m_nNextExpectedReqIdx+1 << " Packet(s) from cache" << std::endl;
//#endif //PRECACHE_CONSOLE_DEBUG
                        while(m_nReqIdx-m_nNextExpectedReqIdx+1>0) {
                            m_oReqPacket = m_qoCache.front();
                            m_nFirstBufferIdx = (size_t)(m_oReqPacket.data-m_vcBuffer.data());
                            m_qoCache.pop_front();
                            ++m_nNextExpectedReqIdx;
                        }
                    }
                    else {
#if PRECACHE_CONSOLE_DEBUG
                        std::cout << " -- out-of-order request, destroying cache" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
                        m_qoCache.clear();
                        m_oReqPacket = getPacket_internal(m_nReqIdx);
                        m_nFirstBufferIdx = m_nNextBufferIdx = size_t(-1);
                        m_nNextExpectedReqIdx = m_nNextPrecacheIdx = m_nReqIdx+1;
                    }
                }
                else {
#if PRECACHE_CONSOLE_DEBUG
                    std::cout << " @ answering request manually, precaching is falling behind" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
                    m_oReqPacket = getPacket_internal(m_nReqIdx);
                    m_nFirstBufferIdx = m_nNextBufferIdx = size_t(-1);
                    m_nNextExpectedReqIdx = m_nNextPrecacheIdx = m_nReqIdx+1;
                }
            }
#if PRECACHE_CONSOLE_DEBUG
            else
                std::cout << " @ answering request using last Packet" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
            m_oSyncCondVar.notify_one();
        }
        else {
            size_t nUsedBufferSize = m_nFirstBufferIdx==size_t(-1)?0:(m_nFirstBufferIdx<m_nNextBufferIdx?m_nNextBufferIdx-m_nFirstBufferIdx:m_nBufferSize-m_nFirstBufferIdx+m_nNextBufferIdx);
            if(nUsedBufferSize<m_nBufferSize/4 && m_nNextPrecacheIdx<m_nPacketCount) {
#if PRECACHE_CONSOLE_DEBUG
                std::cout << " @ filling precache buffer... (current size = " << (nUsedBufferSize/1024)/1024 << " mb, " << m_nPacketCount-m_nNextPrecacheIdx << " todo)" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
                size_t nFillCount = 0;
                while(nUsedBufferSize<m_nBufferSize && m_nNextPrecacheIdx<m_nPacketCount && nFillCount<10) {
                    const cv::Mat& oNextPacket = getPacket_internal(m_nNextPrecacheIdx);
                    const size_t nNextPacketSize = (oNextPacket.total()*oNextPacket.elemSize());
                    if(m_nFirstBufferIdx<=m_nNextBufferIdx) {
                        if(m_nNextBufferIdx==size_t(-1) || (m_nNextBufferIdx+nNextPacketSize>=m_nBufferSize)) {
                            if((m_nFirstBufferIdx!=size_t(-1) && nNextPacketSize>=m_nFirstBufferIdx) || nNextPacketSize>=m_nBufferSize)
                                break;
                            cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),m_vcBuffer.data());
                            oNextPacket.copyTo(oNextPacket_cache);
                            m_qoCache.push_back(oNextPacket_cache);
                            m_nNextBufferIdx = nNextPacketSize;
                            if(m_nFirstBufferIdx==size_t(-1))
                                m_nFirstBufferIdx = 0;
                        }
                        else { // m_nNextBufferIdx+nNextPacketSize<m_nBufferSize
                            cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),m_vcBuffer.data()+m_nNextBufferIdx);
                            oNextPacket.copyTo(oNextPacket_cache);
                            m_qoCache.push_back(oNextPacket_cache);
                            m_nNextBufferIdx += nNextPacketSize;
                        }
                    }
                    else if(m_nNextBufferIdx+nNextPacketSize<m_nFirstBufferIdx) {
                        cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),m_vcBuffer.data()+m_nNextBufferIdx);
                        oNextPacket.copyTo(oNextPacket_cache);
                        m_qoCache.push_back(oNextPacket_cache);
                        m_nNextBufferIdx += nNextPacketSize;
                    }
                    else // m_nNextBufferIdx+nNextPacketSize>=m_nFirstBufferIdx
                        break;
                    nUsedBufferSize += nNextPacketSize;
                    ++m_nNextPrecacheIdx;
                }
            }
        }
    }
}

const cv::Mat& litiv::DataPrecacher::getPacket_internal(size_t nIdx) {
    if(m_nLastReqIdx!=nIdx) {
        m_oLastReqPacket = m_lCallback(nIdx);
        m_nLastReqIdx = nIdx;
    }
    return m_oLastReqPacket;
}

litiv::IDataLoader_<TNoGroup>::IDataLoader_() :
        m_oInputPrecacher(std::bind(&IDataLoader_<TNoGroup>::_getInputPacket_redirect,this,std::placeholders::_1)),
        m_oGTPrecacher(std::bind(&IDataLoader_<TNoGroup>::_getGTPacket_redirect,this,std::placeholders::_1)) {}

void litiv::IDataLoader_<TNoGroup>::startPrecaching(bool bUsingGT, size_t nSuggestedBufferSize) {
    CV_Assert(m_oInputPrecacher.startPrecaching(getTotPackets(),nSuggestedBufferSize));
    CV_Assert(!bUsingGT || m_oGTPrecacher.startPrecaching(getTotPackets(),nSuggestedBufferSize));
}

void litiv::IDataLoader_<TNoGroup>::stopPrecaching() {
    m_oInputPrecacher.stopPrecaching();
    m_oGTPrecacher.stopPrecaching();
}

const cv::Mat& litiv::IDataLoader_<TNoGroup>::_getInputPacket_redirect(size_t nIdx) {
    CV_Assert(nIdx<getTotPackets());
    m_oLatestInputPacket = _getInputPacket_impl(nIdx);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    writeOnImage(m_oLatestInputPacket,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return m_oLatestInputPacket;
}

const cv::Mat& litiv::IDataLoader_<TNoGroup>::_getGTPacket_redirect(size_t nIdx) {
    CV_Assert(nIdx<getTotPackets());
    m_oLatestGTPacket = _getGTPacket_impl(nIdx);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    writeOnImage(m_oLatestGTPacket,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return m_oLatestGTPacket;
}

#if 0

litiv::Video::Segm::DatasetInfo::DatasetInfo() : DatasetInfoBase(), m_eDatasetID(eDataset_Custom), m_nResultIdxOffset(0) {}

litiv::Video::Segm::DatasetInfo::DatasetInfo(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                                                    const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                                                    const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                                                    bool bForce4ByteDataAlign, double dScaleFactor, eDatasetList eDatasetID, size_t nResultIdxOffset) :
        DatasetInfoBase(sDatasetName,sDatasetRootPath,sResultsRootPath,sResultNamePrefix,sResultNameSuffix,vsWorkBatchPaths,vsSkippedNameTokens,vsGrayscaleNameTokens,bForce4ByteDataAlign,dScaleFactor),
        m_eDatasetID(eDatasetID),
        m_nResultIdxOffset(nResultIdxOffset) {}

void litiv::Video::Segm::DatasetInfo::WriteEvalResults(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) const {
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014)
        CDnetEvaluator::WriteEvalResults(*this,vpGroups,true);
    else if(m_eDatasetID==eDataset_Wallflower || m_eDatasetID==eDataset_PETS2001_D3TC1)
        BinarySegmEvaluator::WriteEvalResults(*this,vpGroups,true);
    else
        throw std::logic_error(cv::format("litiv::Video::Segm::DatasetInfo::WriteEvalResults: missing dataset evaluator impl, cannot write results"));
}

std::shared_ptr<litiv::Video::Segm::DatasetInfo> litiv::Video::Segm::GetDatasetInfo(const eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign) {
    std::shared_ptr<DatasetInfo> pInfo;
    if(eDatasetID==eDataset_CDnet2012) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "CDnet 2012";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/CDNet/dataset/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/CDNet/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "bin";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_vsWorkBatchPaths           = {"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {"thermal"};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_CDnet2012;
        pInfo->m_nResultIdxOffset           = 1;
    }
    else if(eDatasetID==eDataset_CDnet2014) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "CDnet 2014";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/CDNet2014/dataset/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/CDNet2014/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "bin";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_vsWorkBatchPaths           = {"baseline_highway"};//{"shadow_cubicle"};//{"dynamicBackground_fall"},//{"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {"thermal","turbulence"};//{"baseline_highway"};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_CDnet2014;
        pInfo->m_nResultIdxOffset           = 1;
    }
    else if(eDatasetID==eDataset_Wallflower) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "Wallflower";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/Wallflower/dataset/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/Wallflower/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "bin";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_vsWorkBatchPaths           = {"global"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_Wallflower;
        pInfo->m_nResultIdxOffset           = 0;
    }
    else if(eDatasetID==eDataset_PETS2001_D3TC1) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "PETS2001 Dataset#3";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/PETS2001/DATASET3/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/PETS2001/DATASET3/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "bin";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_vsWorkBatchPaths           = {"TESTING"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_PETS2001_D3TC1;
        pInfo->m_nResultIdxOffset           = 0;
    }
    else if(eDatasetID==eDataset_Custom)
        throw std::logic_error(cv::format("litiv::Video::Segm::GetDatasetInfo: custom dataset info struct (eDataset_Custom) can only be filled manually"));
    else
        throw std::logic_error(cv::format("litiv::Video::Segm::GetDatasetInfo: unknown dataset type, cannot use predefined dataset info struct"));
    return pInfo;
}

litiv::Video::Segm::Sequence::Sequence(const std::string& sSeqName, const DatasetInfo& oDatasetInfo, const std::string& sRelativePath) :
        WorkBatch(sSeqName,oDatasetInfo,sRelativePath),
        m_eDatasetID(oDatasetInfo.m_eDatasetID),
        m_nResultIdxOffset(oDatasetInfo.m_nResultIdxOffset),
        m_dExpectedLoad(0),
        m_nTotFrameCount(0),
        m_nNextExpectedVideoReaderFrameIdx(0),
        m_dScaleFactor(oDatasetInfo.m_dScaleFactor) {
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014) {

    }
    else if(m_eDatasetID==eDataset_Wallflower) {
        std::vector<std::string> vsImgPaths;
        PlatformUtils::GetFilesFromDir(m_sDatasetPath,vsImgPaths);
        bool bFoundScript=false, bFoundGTFile=false;
        const std::string sGTFilePrefix("hand_segmented_");
        const size_t nInputFileNbDecimals = 5;
        const std::string sInputFileSuffix(".bmp");
        for(auto iter=vsImgPaths.begin(); iter!=vsImgPaths.end(); ++iter) {
            if(*iter==m_sDatasetPath+"/script.txt")
                bFoundScript = true;
            else if(iter->find(sGTFilePrefix)!=std::string::npos) {
                m_mTestGTIndexes.insert(std::pair<size_t,size_t>(atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),m_vsGTFramePaths.size()));
                m_vsGTFramePaths.push_back(*iter);
                bFoundGTFile = true;
            }
            else {
                if(iter->find(sInputFileSuffix)!=iter->size()-sInputFileSuffix.size())
                    throw std::runtime_error(cv::format("Sequence '%s' contained an unknown file ('%s')",sSeqName.c_str(),iter->c_str()));
                m_vsInputFramePaths.push_back(*iter);
            }
        }
        if(!bFoundGTFile || !bFoundScript || m_vsInputFramePaths.empty() || m_vsGTFramePaths.size()!=1)
            throw std::runtime_error(cv::format("Sequence '%s' did not possess the required groundtruth and input files",sSeqName.c_str()));
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(cv::format("Sequence '%s' did not possess a valid GT file",sSeqName.c_str()));
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
        m_oSize = oTempImg.size();
        m_nTotFrameCount = m_vsInputFramePaths.size();
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
        m_pEvaluator = std::shared_ptr<EvaluatorBase>(new BinarySegmEvaluator("WALLFLOWER_EVAL"));
    }
    else if(m_eDatasetID==eDataset_PETS2001_D3TC1) {
        std::vector<std::string> vsVideoSeqPaths;
        PlatformUtils::GetFilesFromDir(m_sDatasetPath,vsVideoSeqPaths);
        if(vsVideoSeqPaths.size()!=1)
            throw std::runtime_error(cv::format("Sequence '%s': bad subdirectory for PETS2001 parsing (should contain only one video sequence file)",sSeqName.c_str()));
        std::vector<std::string> vsGTSubdirPaths;
        PlatformUtils::GetSubDirsFromDir(m_sDatasetPath,vsGTSubdirPaths);
        if(vsGTSubdirPaths.size()!=1)
            throw std::runtime_error(cv::format("Sequence '%s': bad subdirectory for PETS2001 parsing (should contain only one GT subdir)",sSeqName.c_str()));
        m_voVideoReader.open(vsVideoSeqPaths[0]);
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(cv::format("Sequence '%s': video file could not be opened",sSeqName.c_str()));
        PlatformUtils::GetFilesFromDir(vsGTSubdirPaths[0],m_vsGTFramePaths);
        if(m_vsGTFramePaths.empty())
            throw std::runtime_error(cv::format("Sequence '%s': did not possess any valid GT frames",m_sDatasetPath.c_str()));
        const std::string sGTFilePrefix("image_");
        const size_t nInputFileNbDecimals = 4;
        for(auto iter=m_vsGTFramePaths.begin(); iter!=m_vsGTFramePaths.end(); ++iter)
            m_mTestGTIndexes.insert(std::pair<size_t,size_t>((size_t)atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),iter-m_vsGTFramePaths.begin()));
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(cv::format("Sequence '%s': did not possess valid GT file(s)",sSeqName.c_str()));
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        m_nTotFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
        m_pEvaluator = std::shared_ptr<EvaluatorBase>(new BinarySegmEvaluator("PETS2001_EVAL"));
    }
    else if(m_eDatasetID==eDataset_Custom) {
        cv::Mat oTempImg;
        if(m_nResultIdxOffset==size_t(-1)) {
            PlatformUtils::GetFilesFromDir(m_sDatasetPath,m_vsInputFramePaths);
            oTempImg = cv::imread(m_vsInputFramePaths[0]);
            std::cout << "path = " << m_vsInputFramePaths[0] << std::endl;
            m_nTotFrameCount = m_vsInputFramePaths.size();
        }
        else {
            m_voVideoReader.open(m_sDatasetPath);
            if(!m_voVideoReader.isOpened())
                throw std::runtime_error(cv::format("Sequence '%s': video could not be opened",sSeqName.c_str()));
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
            m_voVideoReader >> oTempImg;
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
            m_nTotFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        }
        if(oTempImg.empty())
            throw std::runtime_error(cv::format("Sequence '%s': video could not be read",sSeqName.c_str()));
        m_oOrigSize = oTempImg.size();
        if(m_dScaleFactor!=1.0)
            cv::resize(oTempImg,oTempImg,cv::Size(),m_dScaleFactor,m_dScaleFactor,cv::INTER_NEAREST);
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
    }
    else
        throw std::logic_error(cv::format("Sequence '%s': unknown dataset type, cannot use any known parsing strategy",sSeqName.c_str()));
}

void litiv::Video::Segm::Sequence::WriteResult(size_t nIdx, const cv::Mat& oResult) {
    if(m_oOrigSize==m_oSize)
        WorkBatch::WriteResult(nIdx+m_nResultIdxOffset,oResult);
    else {
        cv::Mat oResizedResult;
        cv::resize(oResult,oResizedResult,m_oOrigSize,0,0,cv::INTER_NEAREST);
        WorkBatch::WriteResult(nIdx+m_nResultIdxOffset,oResult);
    }
}

bool litiv::Video::Segm::Sequence::StartPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return WorkBatch::StartPrecaching(bUsingGT,m_oSize.height*m_oSize.width*(m_nTotFrameCount+1)*(m_bForcingGrayscale?1:m_bForcing4ByteDataAlign?4:3));
}

cv::Mat litiv::Video::Segm::Sequence::GetInputFromIndex_external(size_t nFrameIdx) {
    cv::Mat oFrame;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014 || m_eDatasetID==eDataset_Wallflower || (m_eDatasetID==eDataset_Custom && m_nResultIdxOffset==size_t(-1)))
        oFrame = cv::imread(m_vsInputFramePaths[nFrameIdx],m_bForcingGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    else if(m_eDatasetID==eDataset_PETS2001_D3TC1 || m_eDatasetID==eDataset_Custom) {
        if(m_nNextExpectedVideoReaderFrameIdx!=nFrameIdx) {
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,(double)nFrameIdx);
            m_nNextExpectedVideoReaderFrameIdx = nFrameIdx+1;
        }
        else
            ++m_nNextExpectedVideoReaderFrameIdx;
        m_voVideoReader >> oFrame;
        if(m_bForcingGrayscale && oFrame.channels()>1)
            cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2GRAY);
    }
    if(m_bForcing4ByteDataAlign && oFrame.channels()==3)
        cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2BGRA);
    if(m_dScaleFactor!=1)
        cv::resize(oFrame,oFrame,cv::Size(),m_dScaleFactor,m_dScaleFactor,cv::INTER_NEAREST);
    CV_Assert(oFrame.size()==m_oSize);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    WriteOnImage(oFrame,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return oFrame;
}

cv::Mat litiv::Video::Segm::Sequence::GetGTFromIndex_external(size_t nFrameIdx) {
    cv::Mat oFrame;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID == eDataset_CDnet2014)
        oFrame = cv::imread(m_vsGTFramePaths[nFrameIdx],cv::IMREAD_GRAYSCALE);
    else if(m_eDatasetID == eDataset_Wallflower || m_eDatasetID == eDataset_PETS2001_D3TC1) {
        auto res = m_mTestGTIndexes.find(nFrameIdx);
        if(res != m_mTestGTIndexes.end())
            oFrame = cv::imread(m_vsGTFramePaths[res->second],cv::IMREAD_GRAYSCALE);
    }
    if(oFrame.empty())
        oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL));
    else if(m_dScaleFactor!=1)
        cv::resize(oFrame,oFrame,cv::Size(),m_dScaleFactor,m_dScaleFactor,cv::INTER_NEAREST);
    CV_Assert(oFrame.size()==m_oSize);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    WriteOnImage(oFrame,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return oFrame;
}

litiv::Video::Registr::DatasetInfo::DatasetInfo() : DatasetInfoBase(), m_eDatasetID(eDataset_Custom), m_nResultIdxOffset(0) {}

litiv::Video::Registr::DatasetInfo::DatasetInfo(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                                                       const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                                                       const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                                                       bool bForce4ByteDataAlign, double dScaleFactor, eDatasetList eDatasetID, size_t nResultIdxOffset) :
        DatasetInfoBase(sDatasetName,sDatasetRootPath,sResultsRootPath,sResultNamePrefix,sResultNameSuffix,vsWorkBatchPaths,vsSkippedNameTokens,vsGrayscaleNameTokens,bForce4ByteDataAlign,dScaleFactor),
        m_eDatasetID(eDatasetID),
        m_nResultIdxOffset(nResultIdxOffset) {}

void litiv::Video::Registr::DatasetInfo::WriteEvalResults(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) const {
    if(m_eDatasetID==eDataset_LITIV2012b)
        HomographyEvaluator::WriteEvalResults(*this,vpGroups,true);
    else
        throw std::logic_error(cv::format("litiv::Video::Registr::DatasetInfo::WriteEvalResults: missing dataset evaluator impl, cannot write results"));
}

std::shared_ptr<litiv::Video::Registr::DatasetInfo> litiv::Video::Registr::GetDatasetInfo(const eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign) {
    std::shared_ptr<DatasetInfo> pInfo;
    if(eDatasetID==eDataset_LITIV2012b) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "LITIV 2012b (CVPRW2015 update)";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/litiv/litiv2012_dataset/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/litiv/litiv2012_dataset/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "homography";
        pInfo->m_sResultNameSuffix          = ".cvmat";
        pInfo->m_vsWorkBatchPaths           = {""};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {"THERMAL"};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_LITIV2012b;
        pInfo->m_nResultIdxOffset           = 0;
    }
    else if(eDatasetID==eDataset_Custom)
        throw std::logic_error(cv::format("litiv::Video::Registr::GetDatasetInfo: custom dataset info struct (eDataset_Custom) can only be filled manually"));
    else
        throw std::logic_error(cv::format("litiv::Video::Registr::GetDatasetInfo: unknown dataset type, cannot use predefined dataset info struct"));
    return pInfo;
}

litiv::Video::Registr::Sequence::Sequence(const std::string& sSeqName, const DatasetInfo& oDatasetInfo, const std::string& sRelativePath) :
        WorkBatch(sSeqName,oDatasetInfo,sRelativePath),
        m_eDatasetID(oDatasetInfo.m_eDatasetID),
        m_nResultIdxOffset(oDatasetInfo.m_nResultIdxOffset),
        m_dExpectedLoad(0),
        m_nTotFrameCount(0),
        m_nNextExpectedVideoReaderFrameIdx(0),
        m_dScaleFactor(oDatasetInfo.m_dScaleFactor) {
    if(m_eDatasetID==eDataset_LITIV2012b) {
        std::vector<std::string> vsSubDirs;
        PlatformUtils::GetSubDirsFromDir(m_sDatasetPath,vsSubDirs);
        auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),m_sDatasetPath+"/groundtruth");
        auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),m_sDatasetPath+"/input");
        if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
            throw std::runtime_error(cv::format("Sequence '%s' did not possess the required groundtruth and input directories",sSeqName.c_str()));
        PlatformUtils::GetFilesFromDir(*inputDir,m_vsInputFramePaths);
        PlatformUtils::GetFilesFromDir(*gtDir,m_vsGTFramePaths);
        if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
            throw std::runtime_error(cv::format("Sequence '%s' did not possess same amount of GT & input frames",sSeqName.c_str()));
        m_oROI = cv::imread(m_sDatasetPath+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
        if(m_oROI.empty())
            throw std::runtime_error(cv::format("Sequence '%s' did not possess a ROI.bmp file",sSeqName.c_str()));
        m_oROI = m_oROI>0;
        m_oSize = m_oROI.size();
        m_nTotFrameCount = m_vsInputFramePaths.size();
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)cv::countNonZero(m_oROI)*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
        m_pEvaluator = std::shared_ptr<EvaluatorBase>(new CDnetEvaluator());
        // note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
    }
    else
        throw std::logic_error(cv::format("Sequence '%s': unknown dataset type, cannot use any known parsing strategy",sSeqName.c_str()));
}

void litiv::Video::Registr::Sequence::WriteResult(size_t nIdx, const cv::Mat& oResult) {
    if(m_oOrigSize==m_oSize)
        WorkBatch::WriteResult(nIdx+m_nResultIdxOffset,oResult);
    else {
        cv::Mat oResizedResult;
        cv::resize(oResult,oResizedResult,m_oOrigSize,0,0,cv::INTER_NEAREST);
        WorkBatch::WriteResult(nIdx+m_nResultIdxOffset,oResult);
    }
}

bool litiv::Video::Registr::Sequence::StartPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return WorkBatch::StartPrecaching(bUsingGT,m_oSize.height*m_oSize.width*(m_nTotFrameCount+1)*(m_bForcingGrayscale?1:m_bForcing4ByteDataAlign?4:3));
}

cv::Mat litiv::Video::Registr::Sequence::GetInputFromIndex_external(size_t nFrameIdx) {
    cv::Mat oFrame;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014 || m_eDatasetID==eDataset_Wallflower || (m_eDatasetID==eDataset_Custom && m_nResultIdxOffset==size_t(-1)))
        oFrame = cv::imread(m_vsInputFramePaths[nFrameIdx],m_bForcingGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    else if(m_eDatasetID==eDataset_PETS2001_D3TC1 || m_eDatasetID==eDataset_Custom) {
        if(m_nNextExpectedVideoReaderFrameIdx!=nFrameIdx) {
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,(double)nFrameIdx);
            m_nNextExpectedVideoReaderFrameIdx = nFrameIdx+1;
        }
        else
            ++m_nNextExpectedVideoReaderFrameIdx;
        m_voVideoReader >> oFrame;
        if(m_bForcingGrayscale && oFrame.channels()>1)
            cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2GRAY);
    }
    if(m_bForcing4ByteDataAlign && oFrame.channels()==3)
        cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2BGRA);
    if(m_dScaleFactor!=1)
        cv::resize(oFrame,oFrame,cv::Size(),m_dScaleFactor,m_dScaleFactor,cv::INTER_NEAREST);
    CV_Assert(oFrame.size()==m_oSize);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    WriteOnImage(oFrame,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return oFrame;
}

cv::Mat litiv::Video::Registr::Sequence::GetGTFromIndex_external(size_t nFrameIdx) {
    cv::Mat oFrame;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID == eDataset_CDnet2014)
        oFrame = cv::imread(m_vsGTFramePaths[nFrameIdx],cv::IMREAD_GRAYSCALE);
    else if(m_eDatasetID == eDataset_Wallflower || m_eDatasetID == eDataset_PETS2001_D3TC1) {
        auto res = m_mTestGTIndexes.find(nFrameIdx);
        if(res != m_mTestGTIndexes.end())
            oFrame = cv::imread(m_vsGTFramePaths[res->second],cv::IMREAD_GRAYSCALE);
    }
    if(oFrame.empty())
        oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL));
    else if(m_dScaleFactor!=1)
        cv::resize(oFrame,oFrame,cv::Size(),m_dScaleFactor,m_dScaleFactor,cv::INTER_NEAREST);
    CV_Assert(oFrame.size()==m_oSize);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    WriteOnImage(oFrame,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return oFrame;
}

litiv::Image::Segm::DatasetInfo::DatasetInfo() : DatasetInfoBase(), m_eDatasetID(eDataset_Custom) {}

litiv::Image::Segm::DatasetInfo::DatasetInfo(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                                                    const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                                                    const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                                                    bool bForce4ByteDataAlign, double dScaleFactor, eDatasetList eDatasetID) :
        DatasetInfoBase(sDatasetName,sDatasetRootPath,sResultsRootPath,sResultNamePrefix,sResultNameSuffix,vsWorkBatchPaths,vsSkippedNameTokens,vsGrayscaleNameTokens,bForce4ByteDataAlign,dScaleFactor),
        m_eDatasetID(eDatasetID) {}

void litiv::Image::Segm::DatasetInfo::WriteEvalResults(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) const {
    if( m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test)
        BSDS500BoundaryEvaluator::WriteEvalResults(*this,vpGroups);
    else
        throw std::logic_error(cv::format("litiv::Image::Segm::DatasetInfo::WriteEvalResults: missing dataset evaluator impl, cannot write results"));
}

std::shared_ptr<litiv::Image::Segm::DatasetInfo> litiv::Image::Segm::GetDatasetInfo(const eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign) {
    std::shared_ptr<DatasetInfo> pInfo;
    if(eDatasetID==eDataset_BSDS500_segm_train || eDatasetID==eDataset_BSDS500_edge_train) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "BSDS500 Training set";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/BSDS500/data/images/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/BSDS500/BSR/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_vsWorkBatchPaths           = {"train"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDatasetID;
    }
    else if(eDatasetID==eDataset_BSDS500_segm_train_valid || eDatasetID==eDataset_BSDS500_edge_train_valid) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "BSDS500 Training+Validation set";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/BSDS500/data/images/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/BSDS500/BSR/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_vsWorkBatchPaths           = {"train","val"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDatasetID;
    }
    else if(eDatasetID==eDataset_BSDS500_segm_train_valid_test || eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "BSDS500 Training+Validation+Test set";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/BSDS500/data/images/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/BSDS500/BSR/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_vsWorkBatchPaths           = {"train","val","test"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDatasetID;
    }
    else if(eDatasetID==eDataset_Custom)
        throw std::logic_error(cv::format("litiv::Image::Segm::GetDatasetInfo: custom dataset info struct (eDataset_Custom) can only be filled manually"));
    else
        throw std::logic_error(cv::format("litiv::Image::Segm::GetDatasetInfo: unknown dataset type, cannot use predefined dataset info struct"));
    return pInfo;
}

litiv::Image::Segm::Set::Set(const std::string& sSetName, const DatasetInfo& oDatasetInfo, const std::string& sRelativePath) :
        WorkBatch(sSetName,oDatasetInfo,sRelativePath),
        m_eDatasetID(oDatasetInfo.m_eDatasetID),
        m_dExpectedLoad(0),
        m_nTotImageCount(0),
        m_bIsConstantSize(false) {
    if(m_eDatasetID==eDataset_BSDS500_segm_train || m_eDatasetID==eDataset_BSDS500_segm_train_valid || m_eDatasetID==eDataset_BSDS500_segm_train_valid_test ||
       m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
        PlatformUtils::GetFilesFromDir(m_sDatasetPath,m_vsInputImagePaths);
        PlatformUtils::FilterFilePaths(m_vsInputImagePaths,{},{".jpg"});
        if(m_vsInputImagePaths.empty())
            throw std::runtime_error(cv::format("Image set '%s' did not possess any image file",sSetName.c_str()));
        m_oMaxSize = cv::Size(481,321);
        m_nTotImageCount = m_vsInputImagePaths.size();
        m_dExpectedLoad = (double)m_oMaxSize.area()*m_nTotImageCount*(int(!m_bForcingGrayscale)+1);
        if(m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
            PlatformUtils::GetSubDirsFromDir(oDatasetInfo.m_sDatasetRootPath+"/../groundTruth_bdry_images/"+sRelativePath,m_vsGTImagePaths);
            if(m_vsGTImagePaths.empty())
                throw std::runtime_error(cv::format("Image set '%s' did not possess any groundtruth image folders",sSetName.c_str()));
            else if(m_vsGTImagePaths.size()!=m_vsInputImagePaths.size())
                throw std::runtime_error(cv::format("Image set '%s' input/groundtruth count mismatch",sSetName.c_str()));
            // make sure folders are non-empty, and folders & images are similarliy ordered
            std::vector<std::string> vsTempPaths;
            for(size_t nImageIdx=0; nImageIdx<m_vsGTImagePaths.size(); ++nImageIdx) {
                PlatformUtils::GetFilesFromDir(m_vsGTImagePaths[nImageIdx],vsTempPaths);
                CV_Assert(!vsTempPaths.empty());
                const size_t nLastInputSlashPos = m_vsInputImagePaths[nImageIdx].find_last_of("/\\");
                const std::string sInputImageFullName = nLastInputSlashPos==std::string::npos?m_vsInputImagePaths[nImageIdx]:m_vsInputImagePaths[nImageIdx].substr(nLastInputSlashPos+1);
                const size_t nLastGTSlashPos = m_vsGTImagePaths[nImageIdx].find_last_of("/\\");
                CV_Assert(sInputImageFullName.find(nLastGTSlashPos==std::string::npos?m_vsGTImagePaths[nImageIdx]:m_vsGTImagePaths[nImageIdx].substr(nLastGTSlashPos+1))!=std::string::npos);
            }
            m_pEvaluator = std::shared_ptr<EvaluatorBase>(new BSDS500BoundaryEvaluator());
        }
        else { //m_eDatasetID==eDataset_BSDS500_segm_train || m_eDatasetID==eDataset_BSDS500_segm_train_valid || m_eDatasetID==eDataset_BSDS500_segm_train_valid_test
            // current impl cannot parse GT/evaluate (matlab files only)
            CV_Error(0,"missing impl");
        }
    }
    else if(m_eDatasetID==eDataset_Custom) {
        PlatformUtils::GetFilesFromDir(m_sDatasetPath,m_vsInputImagePaths);
        PlatformUtils::FilterFilePaths(m_vsInputImagePaths,{},{".jpg"});
        if(m_vsInputImagePaths.empty())
            throw std::runtime_error(cv::format("Image set '%s' did not possess any jpg image file",sSetName.c_str()));
        for(size_t n=0; n<m_vsInputImagePaths.size(); ++n) {
            cv::Mat oCurrInput = cv::imread(m_vsInputImagePaths[n]);
            if(m_oMaxSize.width<oCurrInput.cols)
                m_oMaxSize.width = oCurrInput.cols;
            if(m_oMaxSize.height<oCurrInput.rows)
                m_oMaxSize.height = oCurrInput.rows;
        }
        m_nTotImageCount = m_vsInputImagePaths.size();
        m_dExpectedLoad = (double)m_oMaxSize.area()*m_nTotImageCount*(int(!m_bForcingGrayscale)+1);
    }
    else
        throw std::logic_error(cv::format("Image set '%s': unknown dataset type, cannot use any known parsing strategy",sSetName.c_str()));
    m_voOrigImageSizes.resize(m_nTotImageCount);
    m_vsOrigImageNames.resize(m_nTotImageCount);
    for(size_t nImageIdx=0; nImageIdx<m_vsInputImagePaths.size(); ++nImageIdx) {
        const size_t nLastSlashPos = m_vsInputImagePaths[nImageIdx].find_last_of("/\\");
        const std::string sImageFullName = nLastSlashPos==std::string::npos?m_vsInputImagePaths[nImageIdx]:m_vsInputImagePaths[nImageIdx].substr(nLastSlashPos+1);
        const size_t nLastDotPos = sImageFullName.find_last_of(".");
        const std::string sImageName = nLastSlashPos==std::string::npos?sImageFullName:sImageFullName.substr(0,nLastDotPos);
        m_vsOrigImageNames[nImageIdx] = sImageName;
    }
}

cv::Mat litiv::Image::Segm::Set::GetInputFromIndex_external(size_t nImageIdx) {
    cv::Mat oImage;
    oImage = cv::imread(m_vsInputImagePaths[nImageIdx],m_bForcingGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    CV_Assert(!oImage.empty());
    CV_Assert(m_voOrigImageSizes[nImageIdx]==cv::Size() || m_voOrigImageSizes[nImageIdx]==oImage.size());
    m_voOrigImageSizes[nImageIdx] = oImage.size();
    if(m_eDatasetID==eDataset_BSDS500_segm_train || m_eDatasetID==eDataset_BSDS500_segm_train_valid || m_eDatasetID==eDataset_BSDS500_segm_train_valid_test ||
       m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
        CV_Assert(oImage.size()==cv::Size(481,321) || oImage.size()==cv::Size(321,481));
        if(oImage.size()==cv::Size(321,481))
            cv::transpose(oImage,oImage);
    }
    CV_Assert(oImage.cols<=m_oMaxSize.width && oImage.rows<=m_oMaxSize.height);
    if(m_bForcing4ByteDataAlign && oImage.channels()==3)
        cv::cvtColor(oImage,oImage,cv::COLOR_BGR2BGRA);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Image #" << nImageIdx;
    WriteOnImage(oImage,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return oImage;
}

cv::Mat litiv::Image::Segm::Set::GetGTFromIndex_external(size_t nImageIdx) {
    cv::Mat oImage;
    if(m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
        if(m_vsGTImagePaths.size()>nImageIdx) {
            std::vector<std::string> vsTempPaths;
            PlatformUtils::GetFilesFromDir(m_vsGTImagePaths[nImageIdx],vsTempPaths);
            CV_Assert(!vsTempPaths.empty());
            cv::Mat oTempRefGTImage = cv::imread(vsTempPaths[0],cv::IMREAD_GRAYSCALE);
            CV_Assert(!oTempRefGTImage.empty());
            CV_Assert(m_voOrigImageSizes[nImageIdx]==cv::Size() || m_voOrigImageSizes[nImageIdx]==oTempRefGTImage.size());
            CV_Assert(oTempRefGTImage.size()==cv::Size(481,321) || oTempRefGTImage.size()==cv::Size(321,481));
            m_voOrigImageSizes[nImageIdx] = oTempRefGTImage.size();
            if(oTempRefGTImage.size()==cv::Size(321,481))
                cv::transpose(oTempRefGTImage,oTempRefGTImage);
            oImage.create(int(oTempRefGTImage.rows*vsTempPaths.size()),oTempRefGTImage.cols,CV_8UC1);
            for(size_t nGTImageIdx=0; nGTImageIdx<vsTempPaths.size(); ++nGTImageIdx) {
                cv::Mat oTempGTImage = cv::imread(vsTempPaths[nGTImageIdx],cv::IMREAD_GRAYSCALE);
                CV_Assert(!oTempGTImage.empty() && (oTempGTImage.size()==cv::Size(481,321) || oTempGTImage.size()==cv::Size(321,481)));
                if(oTempGTImage.size()==cv::Size(321,481))
                    cv::transpose(oTempGTImage,oTempGTImage);
                oTempGTImage.copyTo(cv::Mat(oImage,cv::Rect(0,int(oTempGTImage.rows*nGTImageIdx),oTempGTImage.cols,oTempGTImage.rows)));
            }
            if(m_bForcing4ByteDataAlign && oImage.channels()==3)
                cv::cvtColor(oImage,oImage,cv::COLOR_BGR2BGRA);
        }
    }
    if(oImage.empty())
        oImage = cv::Mat(m_oMaxSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL));
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Image #" << nImageIdx;
    WriteOnImage(oImage,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return oImage;
}

cv::Mat litiv::Image::Segm::Set::ReadResult(size_t nImageIdx) {
    CV_Assert(m_vsOrigImageNames[nImageIdx]!=std::string());
    CV_Assert(!m_sResultNameSuffix.empty());
    std::stringstream sResultFilePath;
    sResultFilePath << m_sResultsPath << m_sResultNamePrefix << m_vsOrigImageNames[nImageIdx] << m_sResultNameSuffix;
    cv::Mat oImage = cv::imread(sResultFilePath.str(),m_bForcingGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    if(m_eDatasetID==eDataset_BSDS500_segm_train || m_eDatasetID==eDataset_BSDS500_segm_train_valid || m_eDatasetID==eDataset_BSDS500_segm_train_valid_test ||
       m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
        CV_Assert(oImage.size()==cv::Size(481,321) || oImage.size()==cv::Size(321,481));
        CV_Assert(m_voOrigImageSizes[nImageIdx]==cv::Size() || m_voOrigImageSizes[nImageIdx]==oImage.size());
        m_voOrigImageSizes[nImageIdx] = oImage.size();
        if(oImage.size()==cv::Size(321,481))
            cv::transpose(oImage,oImage);
    }
    return oImage;
}

void litiv::Image::Segm::Set::WriteResult(size_t nImageIdx, const cv::Mat& oResult) {
    CV_Assert(m_vsOrigImageNames[nImageIdx]!=std::string());
    CV_Assert(!m_sResultNameSuffix.empty());
    cv::Mat oImage = oResult;
    if(m_eDatasetID==eDataset_BSDS500_segm_train || m_eDatasetID==eDataset_BSDS500_segm_train_valid || m_eDatasetID==eDataset_BSDS500_segm_train_valid_test ||
       m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
        CV_Assert(oImage.size()==cv::Size(481,321) || oImage.size()==cv::Size(321,481));
        CV_Assert(m_voOrigImageSizes[nImageIdx]==cv::Size(481,321) || m_voOrigImageSizes[nImageIdx]==cv::Size(321,481));
        if(m_voOrigImageSizes[nImageIdx]==cv::Size(321,481))
            cv::transpose(oImage,oImage);
    }
    std::stringstream sResultFilePath;
    sResultFilePath << m_sResultsPath << m_sResultNamePrefix << m_vsOrigImageNames[nImageIdx] << m_sResultNameSuffix;
    const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
    cv::imwrite(sResultFilePath.str(),oImage,vnComprParams);
}

bool litiv::Image::Segm::Set::StartPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return WorkBatch::StartPrecaching(bUsingGT,m_oMaxSize.height*m_oMaxSize.width*(m_nTotImageCount+1)*(m_bForcingGrayscale?1:m_bForcing4ByteDataAlign?4:3));
}

#endif
