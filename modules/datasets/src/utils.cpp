
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

#include "litiv/datasets/utils.hpp"

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

#if HAVE_GLSL

litiv::AsyncEvaluationWrapper_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::AsyncEvaluationWrapper_(const std::shared_ptr<ParallelUtils::IParallelAlgo_GLSL>& pAlgo, const IDataHandlerPtr& pSequence) :
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

cv::Size litiv::AsyncEvaluationWrapper_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::getIdealGLWindowSize() {
    glAssert(m_pAlgo->getIsGLInitialized());
    cv::Size oFrameSize = m_pProducer->getFrameSize();
    oFrameSize.width *= int(m_pAlgo->m_nSxSDisplayCount);
    return oFrameSize;
}

void litiv::AsyncEvaluationWrapper_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::pre_initialize_gl() {
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

void litiv::AsyncEvaluationWrapper_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::post_apply_gl() {
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

#endif //HAVE_GLSL
