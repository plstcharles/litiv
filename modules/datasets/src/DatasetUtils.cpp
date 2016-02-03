
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

bool litiv::IDataHandler::compare(const IDataHandler* i, const IDataHandler* j) {
    return PlatformUtils::compare_lowercase(i->getName(),j->getName());
}

bool litiv::IDataHandler::compare_load(const IDataHandler* i, const IDataHandler* j) {
    return i->getExpectedLoad()<j->getExpectedLoad();
}

bool litiv::IDataHandler::compare(const IDataHandler& i, const IDataHandler& j) {
    return PlatformUtils::compare_lowercase(i.getName(),j.getName());
}

bool litiv::IDataHandler::compare_load(const IDataHandler& i, const IDataHandler& j) {
    return i.getExpectedLoad()<j.getExpectedLoad();
}

litiv::DataPrecacher::DataPrecacher(std::function<const cv::Mat&(size_t)> lDataLoaderCallback) :
        m_lCallback(lDataLoaderCallback) {
    CV_Assert(m_lCallback);
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

void litiv::IDataLoader_<litiv::eNotGroup>::startPrecaching(bool bUsingGT, size_t nSuggestedBufferSize) {
    CV_Assert(m_oInputPrecacher.startPrecaching(getTotPackets(),nSuggestedBufferSize));
    CV_Assert(!bUsingGT || m_oGTPrecacher.startPrecaching(getTotPackets(),nSuggestedBufferSize));
}

void litiv::IDataLoader_<litiv::eNotGroup>::stopPrecaching() {
    m_oInputPrecacher.stopPrecaching();
    m_oGTPrecacher.stopPrecaching();
}

litiv::IDataLoader_<litiv::eNotGroup>::IDataLoader_() :
        m_oInputPrecacher(std::bind(&IDataLoader_<eNotGroup>::_getInputPacket_redirect,this,std::placeholders::_1)),
        m_oGTPrecacher(std::bind(&IDataLoader_<eNotGroup>::_getGTPacket_redirect,this,std::placeholders::_1)) {}

const cv::Mat& litiv::IDataLoader_<litiv::eNotGroup>::_getInputPacket_redirect(size_t nIdx) {
    CV_Assert(nIdx<getTotPackets());
    m_oLatestInputPacket = _getInputPacket_impl(nIdx);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    writeOnImage(m_oLatestInputPacket,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return m_oLatestInputPacket;
}

const cv::Mat& litiv::IDataLoader_<litiv::eNotGroup>::_getGTPacket_redirect(size_t nIdx) {
    CV_Assert(nIdx<getTotPackets());
    m_oLatestGTPacket = _getGTPacket_impl(nIdx);
#if HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    writeOnImage(m_oLatestGTPacket,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_FRAME_INDEX
    return m_oLatestGTPacket;
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eGroup>::getInputFrame(size_t nFrameIdx) {
    return dynamic_cast<IDataReader_<eDatasetType_VideoSegm>&>(*getBatch(nFrameIdx)).getInputFrame(nFrameIdx);
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eGroup>::getGTFrame(size_t nFrameIdx) {
    return dynamic_cast<IDataReader_<eDatasetType_VideoSegm>&>(*getBatch(nFrameIdx)).getGTFrame(nFrameIdx);
}

double litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::getExpectedLoad() const {
    return m_oROI.empty()?0.0:(double)cv::countNonZero(m_oROI)*m_nFrameCount*(int(!isGrayscale())+1);
}

size_t litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::getTotPackets() const {
    return m_nFrameCount;
}

void litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::startPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return IDataLoader_<eNotGroup>::startPrecaching(bUsingGT,m_oSize.area()*(m_nFrameCount+1)*(isGrayscale()?1:getDatasetInfo()->is4ByteAligned()?4:3));
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::getInputFrame(size_t nFrameIdx) {
    return m_oInputPrecacher.getPacket(nFrameIdx);
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::getGTFrame(size_t nFrameIdx) {
    return m_oGTPrecacher.getPacket(nFrameIdx);
}

void litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::parseData() {
    cv::Mat oTempImg;
    m_voVideoReader.open(getDataPath());
    if(!m_voVideoReader.isOpened()) {
        PlatformUtils::GetFilesFromDir(getDataPath(),m_vsInputFramePaths);
        if(!m_vsInputFramePaths.empty()) {
            oTempImg = cv::imread(m_vsInputFramePaths[0]);
            m_nFrameCount = m_vsInputFramePaths.size();
        }
    }
    else {
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        m_voVideoReader >> oTempImg;
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        m_nFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
    }
    if(oTempImg.empty())
        lvErrorExt("Sequence '%s': video could not be opened via VideoReader or imread (you might need to implement your own DataProducer_ interface)",getName().c_str());
    m_oOrigSize = oTempImg.size();
    const double dScale = getDatasetInfo()->getScaleFactor();
    if(dScale!=1.0)
        cv::resize(oTempImg,oTempImg,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
    m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
    m_oSize = oTempImg.size();
    m_nNextExpectedVideoReaderFrameIdx = 0;
    CV_Assert(m_nFrameCount>0);
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::_getInputPacket_impl(size_t nIdx) {
    cv::Mat oFrame;
    if(!m_voVideoReader.isOpened())
        oFrame = cv::imread(m_vsInputFramePaths[nIdx],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    else {
        if(m_nNextExpectedVideoReaderFrameIdx!=nIdx) {
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,(double)nIdx);
            m_nNextExpectedVideoReaderFrameIdx = nIdx+1;
        }
        else
            ++m_nNextExpectedVideoReaderFrameIdx;
        m_voVideoReader >> oFrame;
        if(isGrayscale() && oFrame.channels()>1)
            cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2GRAY);
    }
    if(getDatasetInfo()->is4ByteAligned() && oFrame.channels()==3)
        cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2BGRA);
    if(oFrame.size()!=m_oSize)
        cv::resize(oFrame,oFrame,m_oSize,0,0,cv::INTER_NEAREST);
    return oFrame;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::_getGTPacket_impl(size_t) {
    return cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL));
}

size_t litiv::IDataCounter_<litiv::eNotGroup>::getProcessedPacketsCountPromise() {
    return m_nProcessedPacketsPromise.get_future().get();
}

size_t litiv::IDataCounter_<litiv::eNotGroup>::getProcessedPacketsCount() const {
    return m_nProcessedPackets;
}

size_t litiv::IDataCounter_<litiv::eGroup>::getProcessedPacketsCountPromise() {
    return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCountPromise();});
}

size_t litiv::IDataCounter_<litiv::eGroup>::getProcessedPacketsCount() const {
    return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCount();});
}

void litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm,litiv::eGroup>::pushSegmMask(const cv::Mat& oSegm, size_t nIdx) {
    dynamic_cast<IDataRecorder_<eDatasetType_VideoSegm>&>(*getBatch(nIdx)).pushSegmMask(oSegm,nIdx);
}

cv::Mat litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm,litiv::eGroup>::readSegmMask(size_t nIdx) const {
    return dynamic_cast<const IDataRecorder_<eDatasetType_VideoSegm>&>(*getBatch(nIdx)).readSegmMask(nIdx);
}

void litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm,litiv::eGroup>::writeSegmMask(const cv::Mat& oSegm, size_t nIdx) const {
    dynamic_cast<const IDataRecorder_<eDatasetType_VideoSegm>&>(*getBatch(nIdx)).writeSegmMask(oSegm,nIdx);
}

void litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::pushSegmMask(const cv::Mat& oSegm, size_t nIdx) {
    processPacket();
    if(getDatasetInfo()->isSavingOutput())
        writeSegmMask(oSegm,nIdx);
}

cv::Mat litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::readSegmMask(size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << acBuffer.data() << getDatasetInfo()->getOutputNameSuffix();
    return cv::imread(sOutputFilePath.str(),isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
}

void litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::writeSegmMask(const cv::Mat& oSegm, size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << acBuffer.data() << getDatasetInfo()->getOutputNameSuffix();
    const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
    auto pProducer = std::dynamic_pointer_cast<const IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(shared_from_this());
    CV_Assert(pProducer);
    const cv::Mat& oROI = pProducer->getROI();
    cv::Mat oOutputSegm;
    if(!oROI.empty())
        cv::bitwise_or(oSegm,UCHAR_MAX/2,oOutputSegm,oROI==0);
    else
        oOutputSegm = oSegm;
    cv::imwrite(sOutputFilePath.str(),oOutputSegm,vnComprParams);
}

litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::IAsyncDataConsumer_(const std::shared_ptr<ParallelUtils::IParallelAlgo_GLSL>& pAlgo, const IDataHandlerPtr& pSequence) :
        m_pAlgo(pAlgo),
        m_pDisplayHelper(pAlgo->m_pDisplayHelper),
        m_pProducer(std::dynamic_pointer_cast<IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(pSequence)),
m_pConsumer(std::dynamic_pointer_cast<IDataConsumer_<eDatasetType_VideoSegm,eNotGroup>>(pSequence)),
m_bPreserveInputs(pAlgo->m_pDisplayHelper),
m_nLastIdx(0),
m_nCurrIdx(0),
m_nNextIdx(1),
m_nFrameCount(0) {
CV_Assert(pAlgo && m_pProducer && m_pConsumer);
CV_Assert(m_pProducer->getFrameCount()>1);
}

cv::Size litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::getIdealGLWindowSize() {
    cv::Size oFrameSize = m_pProducer->getFrameSize();
    oFrameSize.width *= int(m_pAlgo->m_nSxSDisplayCount);
    return oFrameSize;
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::pre_initialize_gl() {
    m_oNextInput = m_pProducer->getInputFrame(m_nNextIdx).clone();
    m_oCurrInput = m_pProducer->getInputFrame(m_nCurrIdx).clone();
    m_oLastInput = m_oCurrInput.clone();
    CV_Assert(!m_oCurrInput.empty());
    CV_Assert(m_oCurrInput.isContinuous());
    glAssert(m_oCurrInput.channels()==1 || m_oCurrInput.channels()==4);
    m_nFrameCount= m_pProducer->getFrameCount();
    if(m_pProducer->getDatasetInfo()->isSavingOutput() || m_pDisplayHelper)
        m_pAlgo->setOutputFetching(true);
    if(m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
        m_pAlgo->setDebugFetching(true);
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::post_apply_gl() {
    if(m_bPreserveInputs) {
        m_oCurrInput.copyTo(m_oLastInput);
        m_oNextInput.copyTo(m_oCurrInput);
    }
    if(m_nNextIdx<m_nFrameCount)
        m_oNextInput = m_pProducer->getInputFrame(m_nNextIdx);
    m_pConsumer->processPacket();
    if(m_pConsumer->getDatasetInfo()->isSavingOutput() || m_pDisplayHelper) {
        cv::Mat oLastOutput,oLastDebug;
        m_pAlgo->fetchLastOutput(oLastOutput);
        if(m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
            m_pAlgo->fetchLastDebug(oLastDebug);
        else
            oLastDebug = oLastOutput;
        if(m_pConsumer->getDatasetInfo()->isSavingOutput())
            m_pConsumer->writeSegmMask(oLastOutput,m_nLastIdx);
        if(m_pDisplayHelper) {
            const cv::Mat& oROI = m_pProducer->getROI();
            if(!oROI.empty()) {
                cv::bitwise_or(oLastOutput,UCHAR_MAX/2,oLastOutput,oROI==0);
                cv::bitwise_or(oLastDebug,UCHAR_MAX/2,oLastDebug,oROI==0);
            }
            m_pDisplayHelper->display(m_oLastInput,oLastDebug,oLastOutput,m_nLastIdx);
        }
    }
}


#if 0

litiv::Video::Segm::DatasetInfo::DatasetInfo() : DatasetInfoBase(), m_eDatasetID(eDataset_Custom), m_nOutputIdxOffset(0) {}

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
