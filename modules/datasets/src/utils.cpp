
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

#include "litiv/datasets.hpp"

#define HARDCODE_IMAGE_PACKET_INDEX        0 // for sync debug only! will corrupt data for non-image packets
#define PRECACHE_REQUEST_TIMEOUT_MS        1
#define PRECACHE_QUERY_TIMEOUT_MS          10
#define PRECACHE_QUERY_END_TIMEOUT_MS      500
#define PRECACHE_REFILL_TIMEOUT_MS         10000
#if (!(defined(_M_X64) || defined(__amd64__)) && CACHE_MAX_SIZE_GB>2)
#error "Cache max size exceeds system limit (x86)."
#endif //(!(defined(_M_X64) || defined(__amd64__)) && CACHE_MAX_SIZE_GB>2)
#define CACHE_MAX_SIZE size_t(((CACHE_MAX_SIZE_GB*1024)*1024)*1024)
#define CACHE_MIN_SIZE size_t(((10)*1024)*1024) // 10mb

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

std::string lv::IDataHandler::getInputName(size_t nPacketIdx) const {
    std::array<char,32> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),getInputCount()<1e7?"%06zu":"%09zu",nPacketIdx);
    return std::string(acBuffer.data());
}

std::string lv::IDataHandler::getOutputName(size_t nPacketIdx) const {
    const auto pProducer = shared_from_this_cast<const IDataProducer_<DatasetSource_Image>>();
    if(pProducer && pProducer->getIOMappingType()<=IndexMapping)
        return getInputName(nPacketIdx); // will reuse input image file name as output name
    std::array<char,32> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),getExpectedOutputCount()<1e7?"%06zu":"%09zu",nPacketIdx);
    return std::string(acBuffer.data());
}

std::string lv::IDataHandler::getFeaturesName(size_t nPacketIdx) const {
    const auto pProducer = shared_from_this_cast<const IDataProducer_<DatasetSource_Image>>();
    if(pProducer && pProducer->getIOMappingType()<=IndexMapping)
        return getInputName(nPacketIdx); // will reuse input image file name as features name
    std::array<char,32> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),nPacketIdx<size_t(1e7)?"%06zu":"%09zu",nPacketIdx);
    return std::string(acBuffer.data());
}

bool lv::IDataHandler::compare(const IDataHandler* i, const IDataHandler* j) {
    return lv::compare_lowercase(i->getName(),j->getName());
}

bool lv::IDataHandler::compare_load(const IDataHandler* i, const IDataHandler* j) {
    return i->getExpectedLoadSize()<j->getExpectedLoadSize();
}

bool lv::IDataHandler::compare(const IDataHandler& i, const IDataHandler& j) {
    return lv::compare_lowercase(i.getName(),j.getName());
}

bool lv::IDataHandler::compare_load(const IDataHandler& i, const IDataHandler& j) {
    return i.getExpectedLoadSize()<j.getExpectedLoadSize();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string& lv::DataHandler::getName() const {
    return m_sBatchName;
}

const std::string& lv::DataHandler::getDataPath() const {
    return m_sDataPath;
}

const std::string& lv::DataHandler::getOutputPath() const {
    return m_sOutputPath;
}

const std::string& lv::DataHandler::getFeaturesPath() const {
    return m_sFeaturesPath;
}

const std::string& lv::DataHandler::getRelativePath() const {
    return m_sRelativePath;
}

const std::vector<std::string>& lv::DataHandler::getSkippedDirTokens() const {
    return m_oParent.getSkippedDirTokens();
}

double lv::DataHandler::getScaleFactor() const {
    return m_oParent.getScaleFactor();
}

lv::IDataHandlerConstPtr lv::DataHandler::getRoot() const {
    return m_oRoot.shared_from_this();
}

lv::IDataHandlerConstPtr lv::DataHandler::getParent() const {
    return m_oParent.shared_from_this();
}

bool lv::DataHandler::isRoot() const {
    return false;
}

bool lv::DataHandler::is4ByteAligned() const {
    return m_oParent.is4ByteAligned();
}

bool lv::DataHandler::isSavingOutput() const {
    return m_oParent.isSavingOutput();
}

bool lv::DataHandler::isEvaluating() const {
    return m_oParent.isEvaluating();
}

inline const lv::IDataHandler& getRootNodeHelper(const lv::IDataHandler& oStart) {
    lvDbgExceptionWatch;
    lv::IDataHandlerConstPtr p=oStart.shared_from_this();
    while(!p->isRoot())
        p = p->getParent();
    return *p.get();
}

lv::DataHandler::DataHandler(const std::string& sBatchName, const std::string& sRelativePath, const IDataHandler& oParent) :
        m_sBatchName(sBatchName),
        m_sRelativePath(lv::addDirSlashIfMissing(sRelativePath)),
        m_sDataPath(getRootNodeHelper(oParent).getDataPath()+lv::addDirSlashIfMissing(sRelativePath)),
        m_sOutputPath(getRootNodeHelper(oParent).getOutputPath()+lv::addDirSlashIfMissing(sRelativePath)),
        m_sFeaturesPath(getRootNodeHelper(oParent).getOutputPath()+lv::addDirSlashIfMissing(sRelativePath)+"precomp/"),
        m_oParent(oParent),
        m_oRoot(getRootNodeHelper(oParent)) {
    if(!m_sOutputPath.empty())
        lv::createDirIfNotExist(m_sOutputPath);
    lv::createDirIfNotExist(m_sFeaturesPath);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

size_t lv::DataGroupHandler::getExpectedLoadSize() const {
    return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getExpectedLoadSize();});
}

size_t lv::DataGroupHandler::getInputCount() const {
    return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getInputCount();});
}

size_t lv::DataGroupHandler::getGTCount() const {
    return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getGTCount();});
}

size_t lv::DataGroupHandler::getExpectedOutputCount() const {
    return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getExpectedOutputCount();});
}

size_t lv::DataGroupHandler::getCurrentOutputCount() const {
    return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getCurrentOutputCount();});
}

size_t lv::DataGroupHandler::getFinalOutputCount() {
    return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getFinalOutputCount();});
}

double lv::DataGroupHandler::getCurrentProcessTime() const {
    return lv::accumulateMembers<double,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getCurrentProcessTime();});
}

double lv::DataGroupHandler::getFinalProcessTime() {
    return lv::accumulateMembers<double,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getFinalProcessTime();});
}

void lv::DataGroupHandler::resetMetrics() {
    for(auto& pBatch : getBatches(true))
        pBatch->resetMetrics();
}

bool lv::DataGroupHandler::isProcessing() const {
    for(const auto& pBatch : getBatches(true))
        if(pBatch->isProcessing())
            return true;
    return false;
}

bool lv::DataGroupHandler::isBare() const {
    return m_bIsBare;
}

bool lv::DataGroupHandler::isGroup() const {
    return true;
}

lv::IDataHandlerPtrArray lv::DataGroupHandler::getBatches(bool bWithHierarchy) const {
    if(bWithHierarchy)
        return m_vpBatches;
    IDataHandlerPtrArray vpBatches;
    std::function<void(const IDataHandlerPtr&)> lPushBatches = [&](const IDataHandlerPtr& pBatch) {
        if(pBatch->isGroup())
            for(const auto& pSubBatch : pBatch->getBatches(true))
                lPushBatches(pSubBatch);
        else
            vpBatches.push_back(pBatch);
    };
    for(const auto& pBatch : getBatches(true))
        lPushBatches(pBatch);
    return vpBatches;
}

void lv::DataGroupHandler::startPrecaching(bool bPrecacheInputOnly, size_t nSuggestedBufferSize) {
    for(const auto& pBatch : getBatches(true))
        pBatch->startPrecaching(bPrecacheInputOnly,nSuggestedBufferSize);
}

void lv::DataGroupHandler::stopPrecaching() {
    for(const auto& pBatch : getBatches(true))
        pBatch->stopPrecaching();
}

void lv::DataGroupHandler::parseData() {
    lvDbgExceptionWatch;
    m_vpBatches.clear();
    m_bIsBare = true;
    if(!lv::string_contains_token(getName(),getSkippedDirTokens())) {
        if(datasets::getParserVerbosity()>0)
            std::cout << "\tParsing directory '" << getDataPath() << "' for work group '" << getName() << "'..." << std::endl;
        // by default, all subdirs are considered work batch directories (if none, the category directory itself is a batch, and 'bare')
        const std::vector<std::string> vsWorkBatchPaths = lv::getSubDirsFromDir(getDataPath());
        if(vsWorkBatchPaths.empty())
            m_vpBatches.push_back(createWorkBatch(getName(),getRelativePath()));
        else {
            m_bIsBare = false;
            for(const auto& sPathIter : vsWorkBatchPaths) {
                const size_t nLastSlashPos = sPathIter.find_last_of("/\\");
                const std::string sNewBatchName = nLastSlashPos==std::string::npos?sPathIter:sPathIter.substr(nLastSlashPos+1);
                if(!lv::string_contains_token(sNewBatchName,getSkippedDirTokens()))
                    m_vpBatches.push_back(createWorkBatch(sNewBatchName,getRelativePath()+lv::addDirSlashIfMissing(sNewBatchName)));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

lv::DataPrecacher::DataPrecacher(std::function<const cv::Mat&(size_t)> lDataLoaderCallback) :
        m_lCallback(lDataLoaderCallback) {
    lvAssert_(m_lCallback,"invalid data precacher callback");
    m_bIsActive = false;
    m_pWorkerException = nullptr;
    m_nAnswIdx = m_nReqIdx = m_nLastReqIdx = size_t(-1);
}

lv::DataPrecacher::~DataPrecacher() {
    stopAsyncPrecaching();
}

const cv::Mat& lv::DataPrecacher::getPacket(size_t nIdx) {
    lvDbgExceptionWatch;
    if(nIdx==m_nLastReqIdx)
        return m_oLastReqPacket;
    else if(!m_bIsActive) {
        m_oLastReqPacket = m_lCallback(nIdx);
        m_nLastReqIdx = nIdx;
        return m_oLastReqPacket;
    }
    lv::mutex_unique_lock sync_lock(m_oSyncMutex);
    m_nReqIdx = nIdx;
    std::cv_status res;
    size_t nAnswIdx;
    do {
        m_oReqCondVar.notify_one();
        res = m_oSyncCondVar.wait_for(sync_lock,std::chrono::milliseconds(PRECACHE_REQUEST_TIMEOUT_MS));
        nAnswIdx = m_nAnswIdx.load();
        if(datasets::getParserVerbosity()>2 && res==std::cv_status::timeout && nAnswIdx!=m_nReqIdx)
            std::cout << "data precacher [" << uintptr_t(this) << "] retrying request for packet #" << nIdx << "..." << std::endl;
    } while(res==std::cv_status::timeout && nAnswIdx!=m_nReqIdx && !m_pWorkerException);
    if(m_pWorkerException) {
        m_bIsActive = false;
        m_hWorker.join();
        std::rethrow_exception(m_pWorkerException);
    }
    m_oLastReqPacket = m_oReqPacket;
    m_nLastReqIdx = nAnswIdx;
    return m_oLastReqPacket;
}

bool lv::DataPrecacher::startAsyncPrecaching(size_t nSuggestedBufferSize) {
    static_assert(PRECACHE_REQUEST_TIMEOUT_MS>0,"Precache request timeout must be a positive value");
    static_assert(PRECACHE_QUERY_TIMEOUT_MS>0,"Precache query timeout must be a positive value");
    static_assert(PRECACHE_QUERY_END_TIMEOUT_MS>0,"Precache query post-end timeout must be a positive value");
    static_assert(PRECACHE_REFILL_TIMEOUT_MS>0,"Precache refill timeout must be a positive value");
    stopAsyncPrecaching();
    if(nSuggestedBufferSize>0) {
        m_bIsActive = true;
        m_pWorkerException = nullptr;
        m_nAnswIdx = m_nReqIdx = size_t(-1);
        m_hWorker = std::thread(&DataPrecacher::entry,this,std::max(std::min(nSuggestedBufferSize,CACHE_MAX_SIZE),CACHE_MIN_SIZE));
    }
    return m_bIsActive;
}

void lv::DataPrecacher::stopAsyncPrecaching() {
    if(m_bIsActive) {
        m_bIsActive = false;
        m_hWorker.join();
    }
    if(m_pWorkerException)
        std::rethrow_exception(m_pWorkerException);
}

void lv::DataPrecacher::entry(const size_t nBufferSize) {
    lvDbgExceptionWatch;
    lv::mutex_unique_lock sync_lock(m_oSyncMutex);
    try {
        if(datasets::getParserVerbosity()>1)
            std::cout << "data precacher [" << uintptr_t(this) << "] init w/ buffer size = " << (nBufferSize/1024)/1024 << " mb" << std::endl;
        std::queue<cv::Mat> qoCache;
        std::vector<uchar> vcBuffer(nBufferSize);
        size_t nNextExpectedReqIdx = 0;
        size_t nNextPrecacheIdx = 0;
        size_t nFirstBufferIdx = size_t(-1);
        size_t nNextBufferIdx = size_t(-1);
        bool bReachedEnd = false;
        const auto lCacheNextPacket = [&]() -> size_t {
            const cv::Mat& oNextPacket = m_lCallback(nNextPrecacheIdx);
            const size_t nNextPacketSize = oNextPacket.total()*oNextPacket.elemSize();
            if(nNextPacketSize==0) {
                bReachedEnd = true;
                return 0;
            }
            else if(nFirstBufferIdx<=nNextBufferIdx) {
                bReachedEnd = false;
                if(nNextBufferIdx==size_t(-1) || (nNextBufferIdx+nNextPacketSize>=nBufferSize)) {
                    if((nFirstBufferIdx!=size_t(-1) && nNextPacketSize>=nFirstBufferIdx) || nNextPacketSize>=nBufferSize)
                        return 0;
                    cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),vcBuffer.data());
                    oNextPacket.copyTo(oNextPacket_cache);
                    qoCache.push(oNextPacket_cache);
                    nNextBufferIdx = nNextPacketSize;
                    if(nFirstBufferIdx==size_t(-1))
                        nFirstBufferIdx = 0;
                }
                else { // nNextBufferIdx+nNextPacketSize<m_nBufferSize
                    cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),vcBuffer.data()+nNextBufferIdx);
                    oNextPacket.copyTo(oNextPacket_cache);
                    qoCache.push(oNextPacket_cache);
                    nNextBufferIdx += nNextPacketSize;
                }
            }
            else if(nNextBufferIdx+nNextPacketSize<nFirstBufferIdx) {
                cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),vcBuffer.data()+nNextBufferIdx);
                oNextPacket.copyTo(oNextPacket_cache);
                qoCache.push(oNextPacket_cache);
                nNextBufferIdx += nNextPacketSize;
            }
            else // nNextBufferIdx+nNextPacketSize>=nFirstBufferIdx
                return 0;
            ++nNextPrecacheIdx;
            if(datasets::getParserVerbosity()>3)
                std::cout << "data precacher [" << uintptr_t(this) << "] filled one packet w/ size = " << nNextPacketSize/1024 << " kb" << std::endl;
            return nNextPacketSize;
        };
        const std::chrono::time_point<std::chrono::high_resolution_clock> nPrefillTick = std::chrono::high_resolution_clock::now();
        while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-nPrefillTick).count()<PRECACHE_REFILL_TIMEOUT_MS && lCacheNextPacket());
        while(m_bIsActive) {
            if(m_oReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(bReachedEnd?PRECACHE_QUERY_END_TIMEOUT_MS:PRECACHE_QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
                if(m_nReqIdx!=nNextExpectedReqIdx-1) {
                    if(!qoCache.empty()) {
                        if(m_nReqIdx<nNextPrecacheIdx && m_nReqIdx>=nNextExpectedReqIdx) {
                            if(datasets::getParserVerbosity()>2 && m_nReqIdx>nNextExpectedReqIdx)
                                std::cout << "data precacher [" << uintptr_t(this) << "] popping " << m_nReqIdx-nNextExpectedReqIdx << " extra packet(s) from cache" << std::endl;
                            while(m_nReqIdx-nNextExpectedReqIdx+1>0) {
                                m_oReqPacket = qoCache.front();
                                m_nAnswIdx = m_nReqIdx;
                                nFirstBufferIdx = (size_t)(m_oReqPacket.data-vcBuffer.data());
                                qoCache.pop();
                                ++nNextExpectedReqIdx;
                            }
                        }
                        else {
                            if(datasets::getParserVerbosity()>2)
                                std::cout << "data precacher [" << uintptr_t(this) << "] out-of-order request, destroying cache" << std::endl;
                            qoCache = std::queue<cv::Mat>();
                            m_oReqPacket = m_lCallback(m_nReqIdx);
                            m_nAnswIdx = m_nReqIdx;
                            nFirstBufferIdx = nNextBufferIdx = size_t(-1);
                            nNextExpectedReqIdx = nNextPrecacheIdx = m_nReqIdx+1;
                            bReachedEnd = false;
                        }
                    }
                    else {
                        if(datasets::getParserVerbosity()>2)
                            std::cout << "data precacher [" << uintptr_t(this) << "] answering request manually, precaching is falling behind" << std::endl;
                        m_oReqPacket = m_lCallback(m_nReqIdx);
                        m_nAnswIdx = m_nReqIdx;
                        nFirstBufferIdx = nNextBufferIdx = size_t(-1);
                        nNextExpectedReqIdx = nNextPrecacheIdx = m_nReqIdx+1;
                    }
                }
                else if(datasets::getParserVerbosity()>2)
                    std::cout << "data precacher [" << uintptr_t(this) << "] answering request using last packet" << std::endl;
                m_oSyncCondVar.notify_one();
                lCacheNextPacket();
            }
            else if(!bReachedEnd) {
                const size_t nUsedBufferSize = nFirstBufferIdx==size_t(-1)?0:(nFirstBufferIdx<nNextBufferIdx?nNextBufferIdx-nFirstBufferIdx:nBufferSize-nFirstBufferIdx+nNextBufferIdx);
                if(nUsedBufferSize<nBufferSize/4) {
                    if(datasets::getParserVerbosity()>2)
                        std::cout << "data precacher [" << uintptr_t(this) << "] force refilling precache buffer... (current size = " << (nUsedBufferSize/1024)/1024 << " mb)" << std::endl;
                    size_t nFillCount = 0;
                    const std::chrono::time_point<std::chrono::high_resolution_clock> nRefillTick = std::chrono::high_resolution_clock::now();
                    while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-nRefillTick).count()<PRECACHE_REFILL_TIMEOUT_MS && nFillCount++<10 && lCacheNextPacket());
                }
            }
        }
    }
    catch(...) {
        m_pWorkerException = std::current_exception();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IIDataLoader::startPrecaching(bool bPrecacheInputOnly, size_t nSuggestedBufferSize) {
    lvDbgExceptionWatch;
    if(nSuggestedBufferSize==SIZE_MAX)
        nSuggestedBufferSize = getExpectedLoadSize();
    lvAssert_(m_oInputPrecacher.startAsyncPrecaching(nSuggestedBufferSize),"could not start precaching input packets");
    if(!bPrecacheInputOnly) {
        lvAssert_(m_oGTPrecacher.startAsyncPrecaching(nSuggestedBufferSize),"could not start precaching gt packets");
        lvAssert_(m_oFeaturesPrecacher.startAsyncPrecaching(nSuggestedBufferSize),"could not start precaching feature packets");
    }
}

void lv::IIDataLoader::stopPrecaching() {
    m_oInputPrecacher.stopAsyncPrecaching();
    m_oGTPrecacher.stopAsyncPrecaching();
    m_oFeaturesPrecacher.stopAsyncPrecaching();
}

const cv::Mat& lv::IIDataLoader::getInput(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    return m_oInputPrecacher.getPacket(nPacketIdx);
}

const cv::Mat& lv::IIDataLoader::getGT(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    return m_oGTPrecacher.getPacket(nPacketIdx);
}

const cv::Mat& lv::IIDataLoader::loadFeatures(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    return m_oFeaturesPrecacher.getPacket(nPacketIdx);
}

void lv::IIDataLoader::saveFeatures(size_t nPacketIdx, const cv::Mat& oFeatures) const {
    lvDbgExceptionWatch;
    if(!oFeatures.empty()) {
        // could use a datawriter here for REALLY big features (but its unlikely that they can be produced faster than saved)
        std::stringstream ssFeatsFilePath;
        ssFeatsFilePath << getFeaturesPath() << getFeaturesName(nPacketIdx) << ".bin";
        lv::write(ssFeatsFilePath.str(),oFeatures);
    }
}

const cv::Mat& lv::IIDataLoader::getInputROI(size_t /*nPacketIdx*/) const {
    return lv::emptyMat();
}

const cv::Mat& lv::IIDataLoader::getGTROI(size_t /*nPacketIdx*/) const {
    return lv::emptyMat();
}

lv::IIDataLoader::IIDataLoader(PacketPolicy eInputType, PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType) :
        m_oInputPrecacher(std::bind(&IIDataLoader::getInput_redirect,this,std::placeholders::_1)),
        m_oGTPrecacher(std::bind(&IIDataLoader::getGT_redirect,this,std::placeholders::_1)),
        m_oFeaturesPrecacher(std::bind(&IIDataLoader::loadRawFeatures,this,std::placeholders::_1)),
        m_eInputType(eInputType),m_eGTType(eGTType),m_eOutputType(eOutputType),m_eGTMappingType(eGTMappingType),m_eIOMappingType(eIOMappingType) {}

const cv::Mat& lv::IIDataLoader::loadRawFeatures(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    std::stringstream ssFeatsFilePath;
    ssFeatsFilePath << getFeaturesPath() << getFeaturesName(nPacketIdx) << ".bin";
    // all features are user-defined, so we keep no mapping information, and offer no default transformations
    if(lv::checkIfExists(ssFeatsFilePath.str()))
        m_oLatestFeatures = lv::read(ssFeatsFilePath.str());
    else
        m_oLatestFeatures = cv::Mat();
    return m_oLatestFeatures;
}

namespace {

    cv::Mat transformImagePacket(size_t nPacketIdx, const cv::Mat& oPacket, const lv::MatInfo& oInfo) {
        lvDbgExceptionWatch;
        lvDbgAssert(!oPacket.empty());
        lvDbgAssert(!oInfo.size.empty());
        lvDbgAssert(oInfo.size.dims()<=2);
        lvDbgAssert(oInfo.type()>=0);
    #if HARDCODE_IMAGE_PACKET_INDEX
        std::stringstream sstr;
        sstr << "Packet #" << nPacketIdx;
        lv::putText(m_oLatestInput,sstr.str(),cv::Scalar_<uchar>::all(255));
    #else //!HARDCODE_IMAGE_PACKET_INDEX
        UNUSED(nPacketIdx);
    #endif //!HARDCODE_IMAGE_PACKET_INDEX
        cv::Mat oCvtOutput;
        if(oInfo.type.depth()==oPacket.depth() && oInfo.type.channels()!=oPacket.channels()) {
            if(oInfo.type.channels()==4 && oPacket.channels()==3)
                cv::cvtColor(oPacket,oCvtOutput,cv::COLOR_BGR2BGRA);
            else if(oInfo.type.channels()==1 && oPacket.channels()==3)
                cv::cvtColor(oPacket,oCvtOutput,cv::COLOR_BGR2GRAY);
            else
                oCvtOutput = oPacket; // dont know how to handle this here; need override of 'redirect'
        }
        else
            oCvtOutput = oPacket;
        cv::Mat oResizeOutput;
        if(oInfo.size!=oCvtOutput.size())
            cv::resize(oPacket,oResizeOutput,oInfo.size(),0,0,cv::INTER_NEAREST);
        else
            oResizeOutput = oCvtOutput;
        return oResizeOutput;
    }

} // anonymous namespace

const cv::Mat& lv::IIDataLoader::getInput_redirect(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    auto pNotArrayLoader = shared_from_this_cast<IDataLoader_<NotArray>>();
    if(pNotArrayLoader) {
        lvDbgExceptionWatch;
        const lv::MatInfo& oPacketInfo = getInputInfo(nPacketIdx);
        m_oLatestInput = pNotArrayLoader->getRawInput(nPacketIdx);
        if(!m_oLatestInput.empty()) {
            if(m_eInputType==ImagePacket) {
                lvAssert__(m_oLatestInput.dims<=2 && oPacketInfo.size.dims()<=2,"bad raw image formatting (packet = %s)",getInputName(nPacketIdx).c_str());
                m_oLatestInput = transformImagePacket(nPacketIdx,m_oLatestInput,oPacketInfo);
            }
            else
                lvAssert_(m_eInputType==UnspecifiedPacket,"unexpected packet type for not-array loader");
        }
        else
            lvAssert__(oPacketInfo.size.empty(),"unexpected empty raw image (packet = %s)",getInputName(nPacketIdx).c_str());
    }
    else {
        auto pArrayLoader = shared_from_this_cast<IDataLoader_<Array>>(true);
        if(pArrayLoader) {
            lvDbgExceptionWatch;
            std::vector<cv::Mat> vLatestInputs = pArrayLoader->getRawInputArray(nPacketIdx);
            lvAssert__(vLatestInputs.size()==pArrayLoader->getInputStreamCount(),"unexpected raw input array size (packet = %s)",getInputName(nPacketIdx).c_str());
            const std::vector<lv::MatInfo>& vStreamInfos = pArrayLoader->getInputInfoArray(nPacketIdx);
            lvAssert__(vStreamInfos.size()==pArrayLoader->getInputStreamCount(),"unexpected raw input info array size (packet = %s)",getInputName(nPacketIdx).c_str());
            for(size_t nStreamIdx=0; nStreamIdx<vLatestInputs.size(); ++nStreamIdx) {
                if(!vLatestInputs[nStreamIdx].empty()) {
                    if(m_eInputType==ImageArrayPacket) {
                        lvAssert__(vLatestInputs[nStreamIdx].dims<=2 && vStreamInfos[nStreamIdx].size.dims()<=2,"bad raw image formatting (stream = %s, packet = %s)",pArrayLoader->getInputStreamName(nStreamIdx).c_str(),getInputName(nPacketIdx).c_str());
                        vLatestInputs[nStreamIdx] = transformImagePacket(nPacketIdx,vLatestInputs[nStreamIdx],vStreamInfos[nStreamIdx]);
                    }
                    else
                        lvAssert_(m_eInputType==UnspecifiedPacket,"unexpected packet type for not-array loader");
                    lvAssert__(vLatestInputs[nStreamIdx].type()==vStreamInfos[nStreamIdx].type() && vLatestInputs[nStreamIdx].size==vStreamInfos[nStreamIdx].size,
                               "unexpected stream size/type --- need redirect override (stream = %s, packet = %s)",pArrayLoader->getInputStreamName(nStreamIdx).c_str(),getInputName(nPacketIdx).c_str());
                }
                else
                    lvAssert__(vStreamInfos[nStreamIdx].size.empty(),"unexpected empty raw stream (stream = %s, packet = %s)",pArrayLoader->getInputStreamName(nStreamIdx).c_str(),getInputName(nPacketIdx).c_str());
            }
            m_oLatestInput = lv::packData(vLatestInputs);
            if(pArrayLoader->m_vPackedInputInfos.size()!=getInputCount())
                pArrayLoader->m_vPackedInputInfos.resize(getInputCount());
            pArrayLoader->m_vPackedInputInfos[nPacketIdx] = lv::MatInfo{m_oLatestInput.size,m_oLatestInput.type()};
        }
        else
            lvError("unexpected data loader type");
    }
    const lv::MatInfo& oPacketInfo = getInputInfo(nPacketIdx);
    if(!m_oLatestInput.empty() || !oPacketInfo.size.empty())
        lvAssert__(m_oLatestInput.type()==oPacketInfo.type() && m_oLatestInput.size==oPacketInfo.size,"unexpected output packet size/type --- need redirect override (packet = %s)",getInputName(nPacketIdx).c_str());
    return m_oLatestInput;
}

const cv::Mat& lv::IIDataLoader::getGT_redirect(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    auto pNotArrayLoader = shared_from_this_cast<IDataLoader_<NotArray>>();
    if(pNotArrayLoader) {
        lvDbgExceptionWatch;
        const lv::MatInfo& oPacketInfo = getGTInfo(nPacketIdx);
        m_oLatestGT = pNotArrayLoader->getRawGT(nPacketIdx);
        if(!m_oLatestGT.empty()) {
            if(m_eGTType==ImagePacket) {
                lvAssert__(m_oLatestGT.dims<=2 && oPacketInfo.size.dims()<=2,"bad raw image formatting (gt packet #%d)",(int)nPacketIdx);
                m_oLatestGT = transformImagePacket(nPacketIdx,m_oLatestGT,oPacketInfo);
            }
            else
                lvAssert_(m_eGTType==UnspecifiedPacket,"unexpected packet type for not-array loader");
        }
        else
            lvAssert__(oPacketInfo.size.empty(),"unexpected empty raw image (gt packet #%d)",(int)nPacketIdx);
    }
    else {
        auto pArrayLoader = shared_from_this_cast<IDataLoader_<Array>>(true);
        if(pArrayLoader) {
            lvDbgExceptionWatch;
            std::vector<cv::Mat> vLatestGTs = pArrayLoader->getRawGTArray(nPacketIdx);
            lvAssert__(vLatestGTs.size()==pArrayLoader->getGTStreamCount(),"unexpected raw GT array size (gt packet #%d)",(int)nPacketIdx);
            const std::vector<lv::MatInfo>& vStreamInfos = pArrayLoader->getGTInfoArray(nPacketIdx);
            lvAssert__(vStreamInfos.size()==pArrayLoader->getGTStreamCount(),"unexpected raw GT info array size (gt packet #%d)",(int)nPacketIdx);
            for(size_t nStreamIdx=0; nStreamIdx<vLatestGTs.size(); ++nStreamIdx) {
                if(!vLatestGTs[nStreamIdx].empty()) {
                    if(m_eGTType==ImageArrayPacket) {
                        lvAssert__(vLatestGTs[nStreamIdx].dims<=2 && vStreamInfos[nStreamIdx].size.dims()<=2,"bad raw image formatting (stream = %s, gt packet #%d)",pArrayLoader->getGTStreamName(nStreamIdx).c_str(),(int)nPacketIdx);
                        vLatestGTs[nStreamIdx] = transformImagePacket(nPacketIdx,vLatestGTs[nStreamIdx],vStreamInfos[nStreamIdx]);
                    }
                    else
                        lvAssert_(m_eGTType==UnspecifiedPacket,"unexpected packet type for not-array loader");
                    lvAssert__(vLatestGTs[nStreamIdx].type()==vStreamInfos[nStreamIdx].type() && vLatestGTs[nStreamIdx].size==vStreamInfos[nStreamIdx].size,
                               "unexpected stream size/type --- need redirect override (stream = %s, gt packet #%d)",pArrayLoader->getGTStreamName(nStreamIdx).c_str(),(int)nPacketIdx);
                }
                else
                    lvAssert__(vStreamInfos[nStreamIdx].size.empty(),"unexpected empty raw stream (stream = %s, gt packet #%d)",pArrayLoader->getGTStreamName(nStreamIdx).c_str(),(int)nPacketIdx);
            }
            m_oLatestGT = lv::packData(vLatestGTs);
            if(pArrayLoader->m_vPackedGTInfos.size()!=getGTCount())
                pArrayLoader->m_vPackedGTInfos.resize(getGTCount());
            pArrayLoader->m_vPackedGTInfos[nPacketIdx] = lv::MatInfo{m_oLatestGT.size,m_oLatestGT.type()};
        }
        else
            lvError("unexpected data loader type");
    }
    const lv::MatInfo& oPacketInfo = getGTInfo(nPacketIdx);
    if(!m_oLatestGT.empty() || !oPacketInfo.size.empty())
        lvAssert__(m_oLatestGT.type()==oPacketInfo.type() && m_oLatestGT.size==oPacketInfo.size,"unexpected output packet size/type --- need redirect override (gt packet #%d)",(int)nPacketIdx);
    return m_oLatestGT;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t lv::IDataLoader_<lv::Array>::getGTStreamCount() const {
    return 0;
}

std::string lv::IDataLoader_<lv::Array>::getInputStreamName(size_t nStreamIdx) const {
    return cv::format("in%02d",(int)nStreamIdx);
}

std::string lv::IDataLoader_<lv::Array>::getGTStreamName(size_t nStreamIdx) const {
    return cv::format("gt%02d",(int)nStreamIdx);
}

const std::vector<cv::Mat>& lv::IDataLoader_<lv::Array>::getInputArray(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    if(getInputStreamCount()==0)
        m_vLatestUnpackedInput.resize(0);
    else if(m_oInputPrecacher.getLastReqIdx()!=nPacketIdx) {
        const std::vector<lv::MatInfo>& vPackInfo = getInputInfoArray(nPacketIdx);
        lvAssert_(vPackInfo.size()==getInputStreamCount(),"unexpected stream pack info array size");
        // no need to clone if getInput does not allow reentrancy
        const cv::Mat& oPacket = getInput(nPacketIdx)/*.clone()*/;
        // output mats in the vector will stay valid without copy for as long as oPacket is valid (typically until next call)
        m_vLatestUnpackedInput = lv::unpackData(oPacket,vPackInfo);
    }
    return m_vLatestUnpackedInput;
}

const std::vector<cv::Mat>& lv::IDataLoader_<lv::Array>::getGTArray(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    if(getGTStreamCount()==0)
        m_vLatestUnpackedGT.resize(0);
    else if(m_oGTPrecacher.getLastReqIdx()!=nPacketIdx) {
        const std::vector<lv::MatInfo>& vPackInfo = getGTInfoArray(nPacketIdx);
        lvAssert_(vPackInfo.size()==getGTStreamCount(),"unexpected stream pack info array size");
        // no need to clone if getter does not allow reentrancy
        const cv::Mat& oPacket = getGT(nPacketIdx)/*.clone()*/;
        // output mats in the vector will stay valid without copy for as long as oPacket is valid (typically until next call)
        m_vLatestUnpackedGT = lv::unpackData(oPacket,vPackInfo);
    }
    return m_vLatestUnpackedGT;
}

const std::vector<cv::Mat>& lv::IDataLoader_<lv::Array>::loadFeaturesArray(size_t nPacketIdx, const std::vector<lv::MatInfo>& vPackingInfo) {
    lvDbgExceptionWatch;
    if(m_oFeaturesPrecacher.getLastReqIdx()!=nPacketIdx) {
        // no need to clone from packed data if loadFeatures does not allow reentrancy
        const cv::Mat& oFeatures = loadFeatures(nPacketIdx)/*.clone()*/;
        // output mats in the vector will stay valid without copy for as long as oFeatures is valid (typically until next loadFeatures call)
        m_vLatestUnpackedFeatures = lv::unpackData(oFeatures,vPackingInfo);
    }
    return m_vLatestUnpackedFeatures;
}

void lv::IDataLoader_<lv::Array>::saveFeaturesArray(size_t nPacketIdx, const std::vector<cv::Mat>& oFeatures, std::vector<lv::MatInfo>* pvOutputPackingInfo) const {
    lvDbgExceptionWatch;
    saveFeatures(nPacketIdx,lv::packData(oFeatures,pvOutputPackingInfo));
}

const std::vector<cv::Mat>& lv::IDataLoader_<lv::Array>::getInputROIArray(size_t /*nPacketIdx*/) const {
    m_vEmptyInputROIArray.resize(getInputStreamCount());
    return m_vEmptyInputROIArray;
}

const std::vector<cv::Mat>& lv::IDataLoader_<lv::Array>::getGTROIArray(size_t /*nPacketIdx*/) const {
    m_vEmptyGTROIArray.resize(getGTStreamCount());
    return m_vEmptyGTROIArray;
}

lv::MatInfo lv::IDataLoader_<lv::Array>::getInputInfo(size_t nPacketIdx) const {
    if(nPacketIdx>=m_vPackedInputInfos.size())
        return lv::MatInfo();
    return m_vPackedInputInfos[nPacketIdx];
}

lv::MatInfo lv::IDataLoader_<lv::Array>::getGTInfo(size_t nPacketIdx) const {
    if(nPacketIdx>=m_vPackedGTInfos.size())
        return lv::MatInfo();
    return m_vPackedGTInfos[nPacketIdx];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

const cv::Mat& lv::IDataProducer_<lv::DatasetSource_Video>::getFrameROI() const {
    return m_oInputROI;
}

lv::MatSize lv::IDataProducer_<lv::DatasetSource_Video>::getFrameSize() const {
    return m_oInputInfo.size;
}

lv::MatInfo lv::IDataProducer_<lv::DatasetSource_Video>::getInputInfo() const {
    return m_oInputInfo;
}

lv::MatInfo lv::IDataProducer_<lv::DatasetSource_Video>::getGTInfo() const {
    return m_oGTInfo;
}

size_t lv::IDataProducer_<lv::DatasetSource_Video>::getInputCount() const {
    return m_nFrameCount;
}

size_t lv::IDataProducer_<lv::DatasetSource_Video>::getGTCount() const {
    return m_mGTIndexLUT.size();
}

size_t lv::IDataProducer_<lv::DatasetSource_Video>::getExpectedLoadSize() const {
    const cv::Mat& oROI = getFrameROI();
    const lv::MatInfo& oMatInfo = getInputInfo();
    return (oROI.empty()?oMatInfo.size.total():(size_t)cv::countNonZero(oROI))*oMatInfo.type.elemSize()*getFrameCount();
}

lv::IDataProducer_<lv::DatasetSource_Video>::IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType) :
        IDataLoader_<NotArray>(ImagePacket,eGTType,eOutputType,eGTMappingType,eIOMappingType),m_nFrameCount(0),m_nNextExpectedVideoReaderFrameIdx(size_t(-1)) {}

const cv::Mat& lv::IDataProducer_<lv::DatasetSource_Video>::getInputROI(size_t /*nPacketIdx*/) const {
    return m_oInputROI;
}

const cv::Mat& lv::IDataProducer_<lv::DatasetSource_Video>::getGTROI(size_t /*nPacketIdx*/) const {
    return m_oGTROI;
}

lv::MatInfo lv::IDataProducer_<lv::DatasetSource_Video>::getInputInfo(size_t /*nPacketIdx*/) const {
    return m_oInputInfo;
}

lv::MatInfo lv::IDataProducer_<lv::DatasetSource_Video>::getGTInfo(size_t /*nPacketIdx*/) const {
    return m_oGTInfo;
}

bool lv::IDataProducer_<lv::DatasetSource_Video>::isInputInfoConst() const {
    return true;
}

bool lv::IDataProducer_<lv::DatasetSource_Video>::isGTInfoConst() const {
    return true;
}

cv::Mat lv::IDataProducer_<lv::DatasetSource_Video>::getRawInput(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    cv::Mat oFrame;
    if(!m_voVideoReader.isOpened() && nPacketIdx<m_vsInputPaths.size())
        oFrame = cv::imread(m_vsInputPaths[nPacketIdx],cv::IMREAD_UNCHANGED);
    else {
        if(m_nNextExpectedVideoReaderFrameIdx!=nPacketIdx) {
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,(double)nPacketIdx);
            m_nNextExpectedVideoReaderFrameIdx = nPacketIdx+1;
        }
        else
            ++m_nNextExpectedVideoReaderFrameIdx;
        m_voVideoReader >> oFrame;
    }
    return oFrame;
}

cv::Mat lv::IDataProducer_<lv::DatasetSource_Video>::getRawGT(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    lvAssert_(getGTPacketType()==ImagePacket,"default impl only works for image gt packets");
    if(m_mGTIndexLUT.count(nPacketIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nPacketIdx];
        if(nGTIdx<m_vsGTPaths.size())
            return cv::imread(m_vsGTPaths[nGTIdx],cv::IMREAD_UNCHANGED);
    }
    return cv::Mat();
}

void lv::IDataProducer_<lv::DatasetSource_Video>::parseData() {
    lvDbgExceptionWatch;
    m_nFrameCount = 0;
    m_mGTIndexLUT.clear();
    m_vsInputPaths.clear();
    m_vsGTPaths.clear();
    m_voVideoReader.release();
    m_nNextExpectedVideoReaderFrameIdx = 0;
    m_oInputROI = cv::Mat();
    m_oGTROI = cv::Mat();
    m_oInputInfo = lv::MatInfo();
    m_oGTInfo = lv::MatInfo();
    cv::Mat oTempImg;
    m_voVideoReader.open(getDataPath());
    if(!m_voVideoReader.isOpened()) {
        m_vsInputPaths = lv::getFilesFromDir(getDataPath());
        if(m_vsInputPaths.size()>1) {
            oTempImg = cv::imread(m_vsInputPaths[0],cv::IMREAD_UNCHANGED);
            m_nFrameCount = m_vsInputPaths.size();
        }
        else if(m_vsInputPaths.size()==1)
            m_voVideoReader.open(m_vsInputPaths[0]);
    }
    if(m_voVideoReader.isOpened()) {
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        m_voVideoReader >> oTempImg;
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        m_nFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
    }
    if(oTempImg.empty())
        lvError_("Sequence '%s': video could not be opened via VideoReader or imread (you might need to implement your own DataProducer_ interface)",getName().c_str());
    const double dScale = getScaleFactor();
    if(dScale!=1.0)
        cv::resize(oTempImg,oTempImg,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
    m_oInputROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
    m_oInputInfo.size = oTempImg.size();
    m_oInputInfo.type = (oTempImg.channels()==3&&is4ByteAligned())?CV_MAKE_TYPE(oTempImg.depth(),4):oTempImg.type();
    lvAssert__(m_nFrameCount>0,"could not find any input frames at data root '%s'",getDataPath().c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::vector<cv::Mat>& lv::IDataProducer_<lv::DatasetSource_VideoArray>::getFrameROIArray() const {
    return m_vInputROIs;
}

std::vector<cv::Size> lv::IDataProducer_<lv::DatasetSource_VideoArray>::getFrameSizeArray() const {
    std::vector<cv::Size> vSizes(m_vInputInfos.size());
    for(size_t nSizeIdx=0; nSizeIdx<m_vInputInfos.size(); ++nSizeIdx)
        vSizes[nSizeIdx] = m_vInputInfos[nSizeIdx].size;
    return vSizes;
}

std::vector<lv::MatInfo> lv::IDataProducer_<lv::DatasetSource_VideoArray>::getInputInfoArray() const {
    return m_vInputInfos;
}

std::vector<lv::MatInfo> lv::IDataProducer_<lv::DatasetSource_VideoArray>::getGTInfoArray() const {
    return m_vGTInfos;
}

size_t lv::IDataProducer_<lv::DatasetSource_VideoArray>::getInputCount() const {
    return m_vvsInputPaths.size();
}

size_t lv::IDataProducer_<lv::DatasetSource_VideoArray>::getGTCount() const {
    return m_mGTIndexLUT.size();
}

size_t lv::IDataProducer_<lv::DatasetSource_VideoArray>::getExpectedLoadSize() const {
    const std::vector<cv::Mat>& vROIArray = getFrameROIArray();
    const std::vector<lv::MatInfo>& vMatInfos = getInputInfoArray();
    lvAssert_(vROIArray.size()==vMatInfos.size(),"internal array sizes mismatch");
    size_t nLoad = size_t(0);
    for(size_t nStreamIdx=0; nStreamIdx<vMatInfos.size(); ++nStreamIdx)
        nLoad += (vROIArray[nStreamIdx].empty()?vMatInfos[nStreamIdx].size.total():(size_t)cv::countNonZero(vROIArray[nStreamIdx]))*vMatInfos[nStreamIdx].type.elemSize()*getFrameCount();
    return nLoad;
}

lv::IDataProducer_<lv::DatasetSource_VideoArray>::IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType) :
        IDataLoader_<Array>(ImageArrayPacket,eGTType,eOutputType,eGTMappingType,eIOMappingType) {}

const std::vector<cv::Mat>& lv::IDataProducer_<lv::DatasetSource_VideoArray>::getInputROIArray(size_t /*nPacketIdx*/) const {
    return m_vInputROIs;
}

const std::vector<cv::Mat>& lv::IDataProducer_<lv::DatasetSource_VideoArray>::getGTROIArray(size_t /*nPacketIdx*/) const {
    return m_vGTROIs;
}

std::vector<lv::MatInfo> lv::IDataProducer_<lv::DatasetSource_VideoArray>::getInputInfoArray(size_t /*nPacketIdx*/) const {
    return m_vInputInfos;
}

std::vector<lv::MatInfo> lv::IDataProducer_<lv::DatasetSource_VideoArray>::getGTInfoArray(size_t /*nPacketIdx*/) const {
    return m_vGTInfos;
}

bool lv::IDataProducer_<lv::DatasetSource_VideoArray>::isInputInfoConst() const {
    return true;
}

bool lv::IDataProducer_<lv::DatasetSource_VideoArray>::isGTInfoConst() const {
    return true;
}

std::vector<cv::Mat> lv::IDataProducer_<lv::DatasetSource_VideoArray>::getRawInputArray(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    if(nPacketIdx>=m_vvsInputPaths.size())
        return std::vector<cv::Mat>(getInputStreamCount());
    const std::vector<std::string>& vsInputPaths = m_vvsInputPaths[nPacketIdx];
    lvAssert_(vsInputPaths.size()==getInputStreamCount(),"input path count did not match stream count");
    std::vector<cv::Mat> vInputs(vsInputPaths.size());
    for(size_t nStreamIdx=0; nStreamIdx<vsInputPaths.size(); ++nStreamIdx)
        vInputs[nStreamIdx] = cv::imread(vsInputPaths[nStreamIdx],cv::IMREAD_UNCHANGED);
    return vInputs;
}

std::vector<cv::Mat> lv::IDataProducer_<lv::DatasetSource_VideoArray>::getRawGTArray(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    lvAssert_(getGTPacketType()==ImageArrayPacket,"default impl only works for image array packets");
    if(m_mGTIndexLUT.count(nPacketIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nPacketIdx];
        if(nGTIdx<m_vvsGTPaths.size()) {
            const std::vector<std::string>& vsGTPaths = m_vvsGTPaths[nGTIdx];
            lvAssert_(vsGTPaths.size()==getGTStreamCount(),"GT path count did not match stream count");
            std::vector<cv::Mat> vGTs(vsGTPaths.size());
            for(size_t nStreamIdx=0; nStreamIdx<vsGTPaths.size(); ++nStreamIdx)
                vGTs[nStreamIdx] = cv::imread(vsGTPaths[nStreamIdx],cv::IMREAD_UNCHANGED);
            return vGTs;
        }
    }
    return std::vector<cv::Mat>(getGTStreamCount());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t lv::IDataProducer_<lv::DatasetSource_Image>::getInputCount() const {
    return m_vsInputPaths.size();
}

size_t lv::IDataProducer_<lv::DatasetSource_Image>::getGTCount() const {
    return m_mGTIndexLUT.size();
}

size_t lv::IDataProducer_<lv::DatasetSource_Image>::getExpectedLoadSize() const {
    size_t nLoad = size_t(0);
    for(size_t nPacketIdx=0; nPacketIdx<getInputCount(); ++nPacketIdx) {
        const cv::Mat& oROI = getInputROI(nPacketIdx);
        const lv::MatInfo& oMatInfo = getInputInfo(nPacketIdx);
        nLoad += (oROI.empty()?oMatInfo.size.total():(size_t)cv::countNonZero(oROI))*oMatInfo.type.elemSize();
    }
    return nLoad;
}

lv::MatInfo lv::IDataProducer_<lv::DatasetSource_Image>::getInputInfo(size_t nPacketIdx) const {
    if(nPacketIdx>=m_vInputInfos.size())
        return lv::MatInfo();
    return m_vInputInfos[nPacketIdx];
}

lv::MatInfo lv::IDataProducer_<lv::DatasetSource_Image>::getGTInfo(size_t nPacketIdx) const {
    if(m_mGTIndexLUT.count(nPacketIdx)) {
        const size_t& nGTIdx = m_mGTIndexLUT.at(nPacketIdx);
        if(nGTIdx<m_vGTInfos.size())
            return m_vGTInfos[nGTIdx];
    }
    return lv::MatInfo();
}

bool lv::IDataProducer_<lv::DatasetSource_Image>::isInputInfoConst() const {
    return m_bIsInputInfoConst;
}

bool lv::IDataProducer_<lv::DatasetSource_Image>::isGTInfoConst() const {
    return m_bIsGTInfoConst;
}

std::string lv::IDataProducer_<lv::DatasetSource_Image>::getInputName(size_t nPacketIdx) const {
    if(nPacketIdx>=m_vsInputPaths.size())
        return IDataHandler::getInputName(nPacketIdx);
    const size_t nLastSlashPos = m_vsInputPaths[nPacketIdx].find_last_of("/\\");
    std::string sFileName = (nLastSlashPos==std::string::npos)?m_vsInputPaths[nPacketIdx]:m_vsInputPaths[nPacketIdx].substr(nLastSlashPos+1);
    return sFileName.substr(0,sFileName.find_last_of("."));
}

lv::IDataProducer_<lv::DatasetSource_Image>::IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType) :
        IDataLoader_<NotArray>(ImagePacket,eGTType,eOutputType,eGTMappingType,eIOMappingType) {}

cv::Mat lv::IDataProducer_<lv::DatasetSource_Image>::getRawInput(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    if(nPacketIdx>=m_vsInputPaths.size())
        return cv::Mat();
    return cv::imread(m_vsInputPaths[nPacketIdx],cv::IMREAD_UNCHANGED);
}

cv::Mat lv::IDataProducer_<lv::DatasetSource_Image>::getRawGT(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    lvAssert_(getGTPacketType()==ImagePacket,"default impl only works for image gt packets");
    if(m_mGTIndexLUT.count(nPacketIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nPacketIdx];
        if(nGTIdx<m_vsGTPaths.size())
            return cv::imread(m_vsGTPaths[nGTIdx],cv::IMREAD_UNCHANGED);
    }
    return cv::Mat();
}

void lv::IDataProducer_<lv::DatasetSource_Image>::parseData() {
    lvDbgExceptionWatch;
    m_mGTIndexLUT.clear();
    m_vsGTPaths.clear();
    m_vInputInfos.clear();
    m_vGTInfos.clear();
    m_bIsInputInfoConst = true;
    m_bIsGTInfoConst = true;
    m_vsInputPaths = lv::getFilesFromDir(getDataPath());
    lv::filterFilePaths(m_vsInputPaths,{},{".jpg",".png",".bmp"});
    if(m_vsInputPaths.empty())
        lvError_("Set '%s' did not possess any jpg/png/bmp image files",getName().c_str());
    m_vInputInfos.reserve(m_vsInputPaths.size());
    lv::MatInfo oLastInfo;
    const double dScale = getScaleFactor();
    for(size_t n = 0; n<m_vsInputPaths.size(); ++n) {
        cv::Mat oCurrInput = cv::imread(m_vsInputPaths[n],cv::IMREAD_UNCHANGED);
        while(oCurrInput.empty()) {
            m_vsInputPaths.erase(m_vsInputPaths.begin()+n);
            if(n>=m_vsInputPaths.size())
                break;
            oCurrInput = cv::imread(m_vsInputPaths[n],cv::IMREAD_UNCHANGED);
        }
        if(oCurrInput.empty())
            break;
        if(dScale!=1.0)
            cv::resize(oCurrInput,oCurrInput,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
        m_vInputInfos.push_back(lv::MatInfo{oCurrInput.size(),((oCurrInput.channels()==3&&is4ByteAligned())?CV_MAKE_TYPE(oCurrInput.depth(),4):oCurrInput.type())});
        if(!oLastInfo.size.empty() && oLastInfo!=m_vInputInfos.back())
            m_bIsInputInfoConst = false;
        oLastInfo = m_vInputInfos.back();
    }
    lvAssert__(!m_vInputInfos.empty(),"could not find any input images at data root '%s'",getDataPath().c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t lv::IDataProducer_<lv::DatasetSource_ImageArray>::getInputCount() const {
    return m_vvsInputPaths.size();
}

size_t lv::IDataProducer_<lv::DatasetSource_ImageArray>::getGTCount() const {
    return m_mGTIndexLUT.size();
}

size_t lv::IDataProducer_<lv::DatasetSource_ImageArray>::getExpectedLoadSize() const {
    size_t nLoad = size_t(0);
    for(size_t nPacketIdx=0; nPacketIdx<getInputCount(); ++nPacketIdx) {
        const std::vector<cv::Mat>& vROIArray = getInputROIArray(nPacketIdx);
        const std::vector<lv::MatInfo>& vMatInfos = getInputInfoArray(nPacketIdx);
        lvAssert_(vROIArray.size()==vMatInfos.size(),"internal array sizes mismatch");
        for(size_t nStreamIdx=0; nStreamIdx<vMatInfos.size(); ++nStreamIdx)
            nLoad += (vROIArray[nStreamIdx].empty()?vMatInfos[nStreamIdx].size.total():(size_t)cv::countNonZero(vROIArray[nStreamIdx]))*vMatInfos[nStreamIdx].type.elemSize();
    }
    return nLoad;
}

std::vector<lv::MatInfo> lv::IDataProducer_<lv::DatasetSource_ImageArray>::getInputInfoArray(size_t nPacketIdx) const {
    if(nPacketIdx>=m_vvInputInfos.size())
        return std::vector<lv::MatInfo>();
    return m_vvInputInfos[nPacketIdx];
}

std::vector<lv::MatInfo> lv::IDataProducer_<lv::DatasetSource_ImageArray>::getGTInfoArray(size_t nPacketIdx) const {
    if(m_mGTIndexLUT.count(nPacketIdx)) {
        const size_t& nGTIdx = m_mGTIndexLUT.at(nPacketIdx);
        if(nGTIdx<m_vvGTInfos.size())
            return m_vvGTInfos[nGTIdx];
    }
    return std::vector<lv::MatInfo>();
}

bool lv::IDataProducer_<lv::DatasetSource_ImageArray>::isInputInfoConst() const {
    return m_bIsInputInfoConst;
}

bool lv::IDataProducer_<lv::DatasetSource_ImageArray>::isGTInfoConst() const {
    return m_bIsGTInfoConst;
}

lv::IDataProducer_<lv::DatasetSource_ImageArray>::IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType) :
        IDataLoader_<Array>(ImageArrayPacket,eGTType,eOutputType,eGTMappingType,eIOMappingType) {}

std::vector<cv::Mat> lv::IDataProducer_<lv::DatasetSource_ImageArray>::getRawInputArray(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    if(nPacketIdx>=m_vvsInputPaths.size())
        return std::vector<cv::Mat>(getInputStreamCount());
    const std::vector<std::string>& vsInputPaths = m_vvsInputPaths[nPacketIdx];
    lvAssert_(vsInputPaths.size()==getInputStreamCount(),"input path count did not match stream count");
    std::vector<cv::Mat> vInputs(vsInputPaths.size());
    for(size_t nStreamIdx=0; nStreamIdx<vsInputPaths.size(); ++nStreamIdx)
        vInputs[nStreamIdx] = cv::imread(vsInputPaths[nStreamIdx],cv::IMREAD_UNCHANGED);
    return vInputs;
}

std::vector<cv::Mat> lv::IDataProducer_<lv::DatasetSource_ImageArray>::getRawGTArray(size_t nPacketIdx) {
    lvDbgExceptionWatch;
    lvAssert_(getGTPacketType()==ImageArrayPacket,"default impl only works for image array packets");
    if(m_mGTIndexLUT.count(nPacketIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nPacketIdx];
        if(nGTIdx<m_vvsGTPaths.size()) {
            const std::vector<std::string>& vsGTPaths = m_vvsGTPaths[nGTIdx];
            lvAssert_(vsGTPaths.size()==getGTStreamCount(),"GT path count did not match stream count");
            std::vector<cv::Mat> vGTs(vsGTPaths.size());
            for(size_t nStreamIdx=0; nStreamIdx<vsGTPaths.size(); ++nStreamIdx)
                vGTs[nStreamIdx] = cv::imread(vsGTPaths[nStreamIdx],cv::IMREAD_UNCHANGED);
            return vGTs;
        }
    }
    return std::vector<cv::Mat>(getGTStreamCount());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

size_t lv::IDataCounter::getCurrentOutputCount() const {
    return m_mProcessedPackets.size();
}

size_t lv::IDataCounter::getFinalOutputCount() {
    return m_nPacketCountFuture.valid()?(m_nFinalPacketCount=m_nPacketCountFuture.get()):m_nFinalPacketCount;
}

void lv::IDataCounter::countOutput(size_t nPacketIdx) {
    m_mProcessedPackets.insert(nPacketIdx);
}

void lv::IDataCounter::setOutputCountPromise() {
    m_nPacketCountPromise.set_value(m_mProcessedPackets.size());
}

void lv::IDataCounter::resetOutputCount() {
    m_mProcessedPackets.clear();
    m_nPacketCountPromise = std::promise<size_t>();
    m_nPacketCountFuture = m_nPacketCountPromise.get_future();
    m_nFinalPacketCount = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

lv::DataWriter::DataWriter(std::function<size_t(const cv::Mat&,size_t)> lDataArchiverCallback) :
        m_lCallback(lDataArchiverCallback) {
    lvAssert_(m_lCallback,"invalid data writer callback");
    m_bIsActive = false;
    m_bAllowPacketDrop = false;
    m_nQueueSize = 0;
    m_nQueueCount = 0;
}

lv::DataWriter::~DataWriter() {
    stopAsyncWriting();
}

size_t lv::DataWriter::queue(const cv::Mat& oPacket, size_t nIdx) {
    lvDbgExceptionWatch;
    if(!m_bIsActive)
        return m_lCallback(oPacket,nIdx);
    const size_t nPacketSize = oPacket.total()*oPacket.elemSize();
    if(nPacketSize>m_nQueueMaxSize)
        return m_lCallback(oPacket,nIdx);
    cv::Mat oPacketCopy = oPacket.clone();
    size_t nPacketPosition;
    {
        lv::mutex_unique_lock sync_lock(m_oSyncMutex);
        if(!m_bAllowPacketDrop)
            m_oClearCondVar.wait(sync_lock,[&]{return m_nQueueSize+nPacketSize<=m_nQueueMaxSize;});
        if(m_nQueueSize+nPacketSize<=m_nQueueMaxSize) {
            m_mQueue[nIdx] = std::move(oPacketCopy);
            m_nQueueSize += nPacketSize;
            // @@@ could cut a find operation here using C++17's map::insert_or_assign above
            nPacketPosition = std::distance(m_mQueue.begin(),m_mQueue.find(nIdx));
            ++m_nQueueCount;
            m_oQueueCondVar.notify_one();
        }
        else {
            if(datasets::getParserVerbosity()>2)
                std::cout << "data writer [" << uintptr_t(this) << "] dropping packet #" << nIdx << std::endl;
            nPacketPosition = SIZE_MAX; // packet dropped
        }
    }
    if(datasets::getParserVerbosity()>2 && (nIdx%50)==0)
        std::cout << "data writer [" << uintptr_t(this) << "] queue @ " << (int)(((float)m_nQueueSize*100)/m_nQueueMaxSize) << "% capacity" << std::endl;
    return nPacketPosition;
}

bool lv::DataWriter::startAsyncWriting(size_t nSuggestedQueueSize, bool bDropPacketsIfFull, size_t nWorkers) {
    stopAsyncWriting();
    if(nSuggestedQueueSize>0) {
        m_bIsActive = true;
        m_bAllowPacketDrop = bDropPacketsIfFull;
        m_nQueueMaxSize = std::max(std::min(nSuggestedQueueSize,CACHE_MAX_SIZE),CACHE_MIN_SIZE);
        m_nQueueSize = 0;
        m_nQueueCount = 0;
        m_mQueue.clear();
        m_vhWorkers.clear();
        for(size_t n=0; n<nWorkers; ++n)
            m_vhWorkers.emplace_back(std::bind(&DataWriter::entry,this));
    }
    return m_bIsActive;
}

void lv::DataWriter::stopAsyncWriting() {
    if(m_bIsActive) {
        {
            lv::mutex_unique_lock sync_lock(m_oSyncMutex);
            m_bIsActive = false;
            m_oQueueCondVar.notify_all();
        }
        for(std::thread& oWorker : m_vhWorkers)
            oWorker.join();
    }
    while(!m_vWorkerExceptions.empty()) {
        std::exception_ptr pLatestException = m_vWorkerExceptions.top().first; // add packet idx to exception...? somewhow?
        m_vWorkerExceptions.pop();
        std::rethrow_exception(pLatestException);
    }
}

void lv::DataWriter::entry() {
    lvDbgExceptionWatch;
    lv::mutex_unique_lock sync_lock(m_oSyncMutex);
    if(datasets::getParserVerbosity()>1)
        std::cout << "data writer [" << uintptr_t(this) << "] init w/ max buffer size = " << (m_nQueueMaxSize/1024)/1024 << " mb" << std::endl;
    while(m_bIsActive || m_nQueueCount>0) {
        m_oQueueCondVar.wait(sync_lock,[&](){return !m_bIsActive || m_nQueueCount>0;});
        if(m_nQueueCount>0) {
            auto pCurrPacket = m_mQueue.begin();
            if(pCurrPacket!=m_mQueue.end()) {
                const size_t nPacketSize = pCurrPacket->second.total()*pCurrPacket->second.elemSize();
                if(nPacketSize<=m_nQueueSize) {
                    try {
                        lv::unlock_guard<lv::mutex_unique_lock> oUnlock(sync_lock);
                        m_lCallback(pCurrPacket->second,pCurrPacket->first);
                    }
                    catch(...) {
                        m_vWorkerExceptions.push(std::make_pair(std::current_exception(),pCurrPacket->first));
                    }
                    m_nQueueSize -= nPacketSize;
                    m_mQueue.erase(pCurrPacket);
                    --m_nQueueCount;
                    m_oClearCondVar.notify_all();
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

cv::Mat lv::IDataArchiver_<lv::NotArray>::loadOutput(size_t nIdx, int nFlags) {
    lvDbgExceptionWatch;
    const auto pLoader = shared_from_this_cast<const IIDataLoader>(true);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getOutputName(nIdx);
    if(pLoader->getOutputPacketType()==ImagePacket) {
        sOutputFilePath << ".png";
        if(nFlags==-1)
            return cv::imread(sOutputFilePath.str(),cv::IMREAD_UNCHANGED);
        else
            return cv::imread(sOutputFilePath.str(),nFlags);
    }
    else {
        sOutputFilePath << ".bin";
        return lv::read(sOutputFilePath.str());
    }
}

void lv::IDataArchiver_<lv::NotArray>::saveOutput(const cv::Mat& _oOutput, size_t nIdx, int /*nFlags*/) {
    lvDbgExceptionWatch;
    const auto pLoader = shared_from_this_cast<const IIDataLoader>(true);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getOutputName(nIdx);
    if(pLoader->getOutputPacketType()==ImagePacket) {
        cv::Mat oOutput = _oOutput;
        // automatically gray-out zones outside ROI if output is binary image mask with 1:1 mapping (e.g. segmentation)
        if(pLoader->getGTPacketType()==ImagePacket && pLoader->getGTMappingType()==ElemMapping &&
           _oOutput.type()==CV_8UC1 && (cv::countNonZero(_oOutput==UCHAR_MAX)+cv::countNonZero(_oOutput==0))==_oOutput.size().area()) {
            const cv::Mat& oROI = pLoader->getGTROI(nIdx);
            if(!oROI.empty() && oROI.size()==_oOutput.size()) {
                oOutput = _oOutput.clone();
                cv::bitwise_or(oOutput,UCHAR_MAX/2,oOutput,oROI==0);
                cv::bitwise_and(oOutput,UCHAR_MAX/2,oOutput,oROI==0);
            }
        }
        const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
        sOutputFilePath << ".png";
        cv::imwrite(sOutputFilePath.str(),oOutput,vnComprParams);
    }
    else {
        sOutputFilePath << ".bin";
        lv::write(sOutputFilePath.str(),_oOutput);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<cv::Mat> lv::IDataArchiver_<lv::Array>::loadOutputArray(size_t nIdx, int nFlags) {
    lvDbgExceptionWatch;
    const auto pLoader = shared_from_this_cast<const IIDataLoader>(true);
    std::vector<cv::Mat> vOutput(getOutputStreamCount());
    for(size_t nStreamIdx=0; nStreamIdx<vOutput.size(); ++nStreamIdx) {
        std::stringstream sOutputFilePath;
        sOutputFilePath << getOutputPath() << getOutputName(nIdx);
        if(vOutput.size()>size_t(1))
            sOutputFilePath << "_" << nStreamIdx;
        if(pLoader->getOutputPacketType()==ImagePacket || pLoader->getOutputPacketType()==ImageArrayPacket) {
            sOutputFilePath << ".png";
            if(nFlags==-1)
                vOutput[nStreamIdx] = cv::imread(sOutputFilePath.str(),cv::IMREAD_UNCHANGED);
            else
                vOutput[nStreamIdx] = cv::imread(sOutputFilePath.str(),nFlags);
        }
        else {
            sOutputFilePath << ".bin";
            vOutput[nStreamIdx] = lv::read(sOutputFilePath.str());
        }
    }
    return vOutput;
}

size_t lv::IDataArchiver_<lv::Array>::getOutputStreamCount() const {
    auto pLoader = shared_from_this_cast<IDataLoader_<Array>>();
    if(pLoader) {
        if(pLoader->getIOMappingType()<=ArrayMapping) {
            // if you catch this error, whoever developed your dataset specialization is in trouble
            lvAssert__(pLoader->getInputStreamCount()>=1,"input stream count (%d) for dataset is bad, need at least one stream",(int)pLoader->getInputStreamCount());
            return pLoader->getInputStreamCount();
        }
        else if(pLoader->getGTMappingType()<=ArrayMapping) {
            // if you catch this error, whoever developed your dataset specialization is in trouble
            lvAssert__(pLoader->getGTStreamCount()>=1,"gt stream count (%d) for dataset is bad, need at least one stream",(int)pLoader->getGTStreamCount());
            return pLoader->getGTStreamCount();
        }
    }
    // if you catch this error, whoever developed your dataset specialization is in trouble
    lvError("output stream count not specified, and cannot be deduced from data loader mappings");
}

void lv::IDataArchiver_<lv::Array>::saveOutputArray(const std::vector<cv::Mat>& vOutput, size_t nIdx, int /*nFlags*/) {
    lvDbgExceptionWatch;
    const size_t nStreamCount = getOutputStreamCount();
    lvAssert__(vOutput.size()==nStreamCount,"expected output vector to have %d elements, had %d",(int)nStreamCount,(int)vOutput.size());
    if(nStreamCount==0)
        return;
    const auto pLoader = shared_from_this_cast<const IIDataLoader>(true);
    if(nStreamCount==size_t(1)) {
        const cv::Mat& _oOutput = vOutput[0];
        std::stringstream sOutputFilePath;
        sOutputFilePath << getOutputPath() << getOutputName(nIdx);
        if(pLoader->getOutputPacketType()==ImagePacket || pLoader->getOutputPacketType()==ImageArrayPacket) {
            cv::Mat oOutput = _oOutput;
            // automatically gray-out zones outside ROI if output is binary image mask with 1:1 mapping (e.g. segmentation)
            if((pLoader->getGTPacketType()==ImagePacket || pLoader->getGTPacketType()==ImageArrayPacket) && pLoader->getGTMappingType()==ElemMapping &&
               _oOutput.type()==CV_8UC1 && (cv::countNonZero(_oOutput==UCHAR_MAX)+cv::countNonZero(_oOutput==0))==_oOutput.size().area()) {
                const cv::Mat& oROI = pLoader->getGTROI(nIdx);
                if(!oROI.empty() && oROI.size()==_oOutput.size()) {
                    oOutput = _oOutput.clone();
                    cv::bitwise_or(oOutput,UCHAR_MAX/2,oOutput,oROI==0);
                    cv::bitwise_and(oOutput,UCHAR_MAX/2,oOutput,oROI==0);
                }
            }
            const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
            sOutputFilePath << ".png";
            cv::imwrite(sOutputFilePath.str(),oOutput,vnComprParams);
        }
        else {
            sOutputFilePath << ".bin";
            lv::write(sOutputFilePath.str(),_oOutput);
        }
    }
    else { // nStreamCount>size_t(1)
        for(size_t nStreamIdx=0; nStreamIdx<nStreamCount; ++nStreamIdx) {
            const cv::Mat& _oOutput = vOutput[nStreamIdx];
            std::stringstream sOutputFilePath;
            sOutputFilePath << getOutputPath() << getOutputName(nIdx) << "_" << nStreamIdx;
            lvDbgAssert(pLoader->getOutputPacketType()!=ImagePacket);
            if(pLoader->getOutputPacketType()==ImageArrayPacket) {
                cv::Mat oOutput = _oOutput;
                // automatically gray-out zones outside ROI if output is binary image mask with 1:1 mapping (e.g. segmentation)
                if(pLoader->getGTPacketType()==ImageArrayPacket && pLoader->getGTMappingType()==ElemMapping &&
                   _oOutput.type()==CV_8UC1 && (cv::countNonZero(_oOutput==UCHAR_MAX)+cv::countNonZero(_oOutput==0))==_oOutput.size().area()) {
                    const cv::Mat& oROI = pLoader->getGTROI(nIdx);
                    if(!oROI.empty() && oROI.size()==_oOutput.size()) {
                        oOutput = _oOutput.clone();
                        cv::bitwise_or(oOutput,UCHAR_MAX/2,oOutput,oROI==0);
                        cv::bitwise_and(oOutput,UCHAR_MAX/2,oOutput,oROI==0);
                    }
                }
                const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
                sOutputFilePath << ".png";
                cv::imwrite(sOutputFilePath.str(),oOutput,vnComprParams);
            }
            else {
                sOutputFilePath << ".bin";
                lv::write(sOutputFilePath.str(),_oOutput);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#if HAVE_GLSL

cv::Size lv::IAsyncDataConsumer_<lv::DatasetEval_BinaryClassifier,lv::GLSL>::getIdealGLWindowSize() const {
    lvAssert_(getExpectedOutputCount()>1,"async data consumer requires work batch to expect more than one output packet");
    auto pLoader = shared_from_this_cast<const IDataLoader_<NotArray>>(true);
    lvAssert_(pLoader->isInputInfoConst(),"async data consumer requires input data to be constant size/type");
    cv::Size oWindowSize = pLoader->getInputInfo(0).size;
    lvAssert_(oWindowSize.area(),"input size must be non-null");
    if(m_pEvalAlgo) {
        lvAssert_(m_pEvalAlgo->getIsGLInitialized(),"evaluator algo must be initialized first");
        oWindowSize.width *= int(m_pEvalAlgo->m_nSxSDisplayCount);
    }
    else if(m_pAlgo) {
        lvAssert_(m_pAlgo->getIsGLInitialized(),"algo must be initialized first");
        oWindowSize.width *= int(m_pAlgo->m_nSxSDisplayCount);
    }
    return oWindowSize;
}

lv::IAsyncDataConsumer_<lv::DatasetEval_BinaryClassifier,lv::GLSL>::IAsyncDataConsumer_() :
        m_nLastIdx(0),
        m_nCurrIdx(0),
        m_nNextIdx(1) {}

void lv::IAsyncDataConsumer_<lv::DatasetEval_BinaryClassifier,lv::GLSL>::pre_initialize_gl() {
    lvAssert_(getExpectedOutputCount()>1,"async data consumer requires work batch to expect more than one output packet");
    m_pLoader = shared_from_this_cast<IIDataLoader>(true);
    lvAssert_(m_pLoader->getInputPacketType()==ImagePacket && m_pLoader->getOutputPacketType()==ImagePacket && m_pLoader->getIOMappingType()==ElemMapping,"async data consumer only defined to work with image packets under 1:1 mapping");
    lvAssert_(m_pAlgo,"invalid algo given to async data consumer");
    m_oCurrInput = m_pLoader->getInput(m_nCurrIdx).clone();
    m_oNextInput = m_pLoader->getInput(m_nNextIdx).clone();
    m_oLastInput = m_oCurrInput.clone();
    lvAssert_(!m_oCurrInput.empty() && m_oCurrInput.isContinuous(),"invalid input fetched from loader");
    lvAssert_(m_oCurrInput.channels()==1 || m_oCurrInput.channels()==4,"loaded data must be 1ch or 4ch to avoid alignment problems");
    if(isSavingOutput() || m_pAlgo->m_pDisplayHelper)
        m_pAlgo->setOutputFetching(true);
    if(m_pAlgo->m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
        m_pAlgo->setDebugFetching(true);
    if(isEvaluating()) {
        lvAssert_(m_pLoader->getGTPacketType()==ImagePacket && m_pLoader->getGTMappingType()==ElemMapping,"async data consumer only defined to work with gt image packets under 1:1 mapping");
        m_oCurrGT = m_pLoader->getGT(m_nCurrIdx).clone();
        m_oNextGT = m_pLoader->getGT(m_nNextIdx).clone();
        m_oLastGT = m_oCurrGT.clone();
        lvAssert_(!m_oCurrGT.empty() && m_oCurrGT.isContinuous(),"invalid gt fetched from loader");
        lvAssert_(m_oCurrGT.channels()==1 || m_oCurrGT.channels()==4,"gt data must be 1ch or 4ch to avoid alignment problems");
    }
}

void lv::IAsyncDataConsumer_<lv::DatasetEval_BinaryClassifier,lv::GLSL>::post_initialize_gl() {
    lvDbgAssert(m_pAlgo);
}

void lv::IAsyncDataConsumer_<lv::DatasetEval_BinaryClassifier,lv::GLSL>::pre_apply_gl(size_t nNextIdx, bool bRebindAll) {
    UNUSED(bRebindAll);
    lvDbgAssert_(m_pLoader,"invalid data loader given to async data consumer");
    lvDbgAssert_(m_pAlgo,"invalid algo given to async data consumer");
    if(nNextIdx!=m_nNextIdx)
        m_oNextInput = m_pLoader->getInput(nNextIdx);
    if(isEvaluating() && nNextIdx!=m_nNextIdx)
        m_oNextGT = m_pLoader->getGT(nNextIdx);
}

void lv::IAsyncDataConsumer_<lv::DatasetEval_BinaryClassifier,lv::GLSL>::post_apply_gl(size_t nNextIdx, bool bRebindAll) {
    lvDbgAssert(m_pLoader && m_pAlgo);
    if(m_pEvalAlgo && isEvaluating())
        m_pEvalAlgo->apply_gl(m_oNextGT,bRebindAll);
    m_nLastIdx = m_nCurrIdx;
    m_nCurrIdx = nNextIdx;
    m_nNextIdx = nNextIdx+1;
    if(m_pAlgo->m_pDisplayHelper || m_lDataCallback) {
        m_oCurrInput.copyTo(m_oLastInput);
        m_oNextInput.copyTo(m_oCurrInput);
        if(isEvaluating()) {
            m_oCurrGT.copyTo(m_oLastGT);
            m_oNextGT.copyTo(m_oCurrGT);
        }
    }
    if(m_nNextIdx<getInputCount()) {
        m_oNextInput = m_pLoader->getInput(m_nNextIdx);
        if(isEvaluating())
            m_oNextGT = m_pLoader->getGT(m_nNextIdx);
    }
    if(isSavingOutput() || m_pAlgo->m_pDisplayHelper || m_lDataCallback) {
        cv::Mat oLastOutput,oLastDebug;
        m_pAlgo->fetchLastOutput(oLastOutput);
        if(m_pAlgo->m_pDisplayHelper && m_pEvalAlgo && m_pEvalAlgo->m_bUsingDebug)
            m_pEvalAlgo->fetchLastDebug(oLastDebug);
        else if(m_pAlgo->m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
            m_pAlgo->fetchLastDebug(oLastDebug);
        else
            oLastDebug = oLastOutput.clone();
        countOutput(m_nLastIdx);
        if(m_lDataCallback)
            m_lDataCallback(m_oLastInput,oLastDebug,oLastOutput,m_oLastGT,m_pLoader->getGTROI(m_nLastIdx),m_nLastIdx);
        if(isSavingOutput() && !oLastOutput.empty())
            saveOutput(oLastOutput,m_nLastIdx);
        if(m_pAlgo->m_pDisplayHelper && m_pLoader->getGTPacketType()==ImagePacket && m_pLoader->getGTMappingType()==ElemMapping) {
            getColoredMasks(oLastOutput,oLastDebug,m_oLastGT,m_pLoader->getGTROI(m_nLastIdx));
            m_pAlgo->m_pDisplayHelper->display(m_oLastInput,oLastDebug,oLastOutput,m_nLastIdx);
        }
    }
}

void lv::IAsyncDataConsumer_<lv::DatasetEval_BinaryClassifier,lv::GLSL>::getColoredMasks(cv::Mat& oOutput, cv::Mat& oDebug, const cv::Mat& /*oGT*/, const cv::Mat& oGTROI) {
    if(!oGTROI.empty()) {
        lvAssert_(oOutput.size()==oGTROI.size(),"output mat size must match gt ROI size");
        cv::bitwise_or(oOutput,UCHAR_MAX/2,oOutput,oGTROI==0);
        if(!oDebug.empty()) {
            lvAssert_(oDebug.size()==oGTROI.size(),"debug mat size must match gt ROI size");
            cv::bitwise_or(oDebug,UCHAR_MAX/2,oDebug,oGTROI==0);
        }
    }
}

#endif //HAVE_GLSL

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string& lv::DatasetHandler::getName() const {
    return m_sDatasetName;
}

const std::string& lv::DatasetHandler::getDataPath() const {
    return m_sDatasetPath;
}

const std::string& lv::DatasetHandler::getOutputPath() const {
    return m_sOutputPath;
}

const std::string& lv::DatasetHandler::getFeaturesPath() const {
    return m_sFeaturesPath;
}

const std::string& lv::DatasetHandler::getRelativePath() const {
    return m_sRelativePath;
}

const std::vector<std::string>& lv::DatasetHandler::getWorkBatchDirs() const {
    return m_vsWorkBatchDirs;
}

const std::vector<std::string>& lv::DatasetHandler::getSkippedDirTokens() const {
    return m_vsSkippedDirTokens;
}

double lv::DatasetHandler::getScaleFactor() const {
    return m_dScaleFactor;
}

lv::IDataHandlerConstPtr lv::DatasetHandler::getRoot() const {
    return shared_from_this();
}

lv::IDataHandlerConstPtr lv::DatasetHandler::getParent() const {
    return IDataHandlerConstPtr();
}

bool lv::DatasetHandler::isRoot() const {
    return true;
}

bool lv::DatasetHandler::is4ByteAligned() const {
    return m_bForce4ByteDataAlign;
}

bool lv::DatasetHandler::isSavingOutput() const {
    return m_bSavingOutput;
}

bool lv::DatasetHandler::isEvaluating() const {
    return m_bUsingEvaluator;
}

lv::DatasetHandler::DatasetHandler(const std::string& sDatasetName, const std::string& sDatasetDirPath, const std::string& sOutputDirPath,
                                   const std::vector<std::string>& vsWorkBatchDirs, const std::vector<std::string>& vsSkippedDirTokens,
                                   bool bSaveOutput, bool bUseEvaluator, bool bForce4ByteDataAlign, double dScaleFactor) :
        m_sDatasetName(sDatasetName),
        m_sDatasetPath(lv::addDirSlashIfMissing(sDatasetDirPath)),
        m_sOutputPath(lv::addDirSlashIfMissing(sOutputDirPath)),
        m_sFeaturesPath(lv::addDirSlashIfMissing(sOutputDirPath)+"precomp/"),
        m_vsWorkBatchDirs(vsWorkBatchDirs),
        m_vsSkippedDirTokens(vsSkippedDirTokens),
        m_bSavingOutput(bSaveOutput),
        m_bUsingEvaluator(bUseEvaluator),
        m_bForce4ByteDataAlign(bForce4ByteDataAlign),
        m_dScaleFactor(dScaleFactor) {
    if(!m_sOutputPath.empty())
        lv::createDirIfNotExist(m_sOutputPath);
    lv::createDirIfNotExist(m_sFeaturesPath);
}
