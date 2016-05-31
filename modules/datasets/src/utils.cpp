
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

#define HARDCODE_IMAGE_PACKET_INDEX        0 // for sync debug only! will corrupt data for non-image packets
#define CONSOLE_DEBUG                      0
#define PRECACHE_REQUEST_TIMEOUT_MS        1
#define PRECACHE_QUERY_TIMEOUT_MS          10
#define PRECACHE_PREFILL_TIMEOUT_MS        5000
#if (!(defined(_M_X64) || defined(__amd64__)) && CACHE_MAX_SIZE_GB>2)
#error "Cache max size exceeds system limit (x86)."
#endif //(!(defined(_M_X64) || defined(__amd64__)) && CACHE_MAX_SIZE_GB>2)
#define CACHE_MAX_SIZE size_t(((CACHE_MAX_SIZE_GB*1024)*1024)*1024)

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

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

std::string litiv::IDataHandler::getPacketName(size_t nPacketIdx) const {
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),getTotPackets()<1e7?"%06zu":"%09zu",nPacketIdx);
    return std::string(acBuffer.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

litiv::DataPrecacher::DataPrecacher(std::function<const cv::Mat&(size_t)> lDataLoaderCallback) :
        m_lCallback(lDataLoaderCallback) {
    CV_Assert(m_lCallback);
    m_bIsActive = false;
    m_nReqIdx = m_nLastReqIdx = size_t(-1);
}

litiv::DataPrecacher::~DataPrecacher() {
    stopAsyncPrecaching();
}

const cv::Mat& litiv::DataPrecacher::getPacket(size_t nIdx) {
    if(nIdx==m_nLastReqIdx)
        return m_oLastReqPacket;
    else if(!m_bIsActive) {
        m_oLastReqPacket = m_lCallback(nIdx);
        m_nLastReqIdx = nIdx;
        return m_oLastReqPacket;
    }
    std::unique_lock<std::mutex> sync_lock(m_oSyncMutex);
    m_nReqIdx = nIdx;
    std::cv_status res;
    do {
        m_oReqCondVar.notify_one();
        res = m_oSyncCondVar.wait_for(sync_lock,std::chrono::milliseconds(PRECACHE_REQUEST_TIMEOUT_MS));
#if CONSOLE_DEBUG
        if(res==std::cv_status::timeout)
            std::cout << " # retrying request..." << std::endl;
#endif //CONSOLE_DEBUG
    } while(res==std::cv_status::timeout);
    m_oLastReqPacket = m_oReqPacket;
    m_nLastReqIdx = nIdx;
    return m_oLastReqPacket;
}

bool litiv::DataPrecacher::startAsyncPrecaching(size_t nSuggestedBufferSize) {
    static_assert(PRECACHE_REQUEST_TIMEOUT_MS>0,"Precache request timeout must be a positive value");
    static_assert(PRECACHE_QUERY_TIMEOUT_MS>0,"Precache query timeout must be a positive value");
    static_assert(PRECACHE_PREFILL_TIMEOUT_MS>0,"Precache prefill timeout must be a positive value");
    if(m_bIsActive)
        stopAsyncPrecaching();
    if(nSuggestedBufferSize>0) {
        m_bIsActive = true;
        m_nReqIdx = size_t(-1);
        m_hWorker = std::thread(&DataPrecacher::entry,this,(nSuggestedBufferSize>CACHE_MAX_SIZE)?(CACHE_MAX_SIZE):nSuggestedBufferSize);
    }
    return m_bIsActive;
}

void litiv::DataPrecacher::stopAsyncPrecaching() {
    if(m_bIsActive) {
        m_bIsActive = false;
        m_hWorker.join();
    }
}

void litiv::DataPrecacher::entry(const size_t nBufferSize) {
    std::unique_lock<std::mutex> sync_lock(m_oSyncMutex);
    std::queue<cv::Mat> qoCache;
    std::vector<uchar> vcBuffer(nBufferSize);
    size_t nNextExpectedReqIdx = 0;
    size_t nNextPrecacheIdx = 0;
    size_t nFirstBufferIdx = 0;
    size_t nNextBufferIdx = 0;
#if CONSOLE_DEBUG
    std::cout << " @ initializing precaching with buffer size = " << (nBufferSize/1024)/1024 << " mb" << std::endl;
#endif //CONSOLE_DEBUG
    const std::chrono::time_point<std::chrono::high_resolution_clock> nPrefillTick = std::chrono::high_resolution_clock::now();
    while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-nPrefillTick).count()<PRECACHE_PREFILL_TIMEOUT_MS) {
        const cv::Mat& oNextPacket = m_lCallback(nNextPrecacheIdx);
        const size_t nNextPacketSize = oNextPacket.total()*oNextPacket.elemSize();
        if(nNextPacketSize>0 && nNextBufferIdx+nNextPacketSize<nBufferSize) {
            cv::Mat oNextPacket_cache(oNextPacket.size(),oNextPacket.type(),vcBuffer.data()+nNextBufferIdx);
            oNextPacket.copyTo(oNextPacket_cache);
            qoCache.push(oNextPacket_cache);
            nNextBufferIdx += nNextPacketSize;
            ++nNextPrecacheIdx;
        }
        else break;
    }
    while(m_bIsActive) {
        if(m_oReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(PRECACHE_QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
            if(m_nReqIdx!=nNextExpectedReqIdx-1) {
                if(!qoCache.empty()) {
                    if(m_nReqIdx<nNextPrecacheIdx && m_nReqIdx>=nNextExpectedReqIdx) {
//#if CONSOLE_DEBUG
//                        std::cout << " -- popping " << m_nReqIdx-nNextExpectedReqIdx+1 << " Packet(s) from cache" << std::endl;
//#endif //CONSOLE_DEBUG
                        while(m_nReqIdx-nNextExpectedReqIdx+1>0) {
                            m_oReqPacket = qoCache.front();
                            nFirstBufferIdx = (size_t)(m_oReqPacket.data-vcBuffer.data());
                            qoCache.pop();
                            ++nNextExpectedReqIdx;
                        }
                    }
                    else {
#if CONSOLE_DEBUG
                        std::cout << " -- out-of-order request, destroying cache" << std::endl;
#endif //CONSOLE_DEBUG
                        qoCache = std::queue<cv::Mat>();
                        m_oReqPacket = m_lCallback(m_nReqIdx);
                        nFirstBufferIdx = nNextBufferIdx = size_t(-1);
                        nNextExpectedReqIdx = nNextPrecacheIdx = m_nReqIdx+1;
                    }
                }
                else {
#if CONSOLE_DEBUG
                    std::cout << " @ answering request manually, precaching is falling behind" << std::endl;
#endif //CONSOLE_DEBUG
                    m_oReqPacket = m_lCallback(m_nReqIdx);
                    nFirstBufferIdx = nNextBufferIdx = size_t(-1);
                    nNextExpectedReqIdx = nNextPrecacheIdx = m_nReqIdx+1;
                }
            }
#if CONSOLE_DEBUG
            else
                std::cout << " @ answering request using last Packet" << std::endl;
#endif //CONSOLE_DEBUG
            m_oSyncCondVar.notify_one();
        }
        else {
            size_t nUsedBufferSize = nFirstBufferIdx==size_t(-1)?0:(nFirstBufferIdx<nNextBufferIdx?nNextBufferIdx-nFirstBufferIdx:nBufferSize-nFirstBufferIdx+nNextBufferIdx);
            if(nUsedBufferSize<nBufferSize/4) {
#if CONSOLE_DEBUG
                std::cout << " @ filling precache buffer... (current size = " << (nUsedBufferSize/1024)/1024 << " mb)" << std::endl;
#endif //CONSOLE_DEBUG
                size_t nFillCount = 0;
                while(nUsedBufferSize<nBufferSize && nFillCount<10) {
                    const cv::Mat& oNextPacket = m_lCallback(nNextPrecacheIdx);
                    const size_t nNextPacketSize = oNextPacket.total()*oNextPacket.elemSize();
                    if(nNextPacketSize==0)
                        break;
                    else if(nFirstBufferIdx<=nNextBufferIdx) {
                        if(nNextBufferIdx==size_t(-1) || (nNextBufferIdx+nNextPacketSize>=nBufferSize)) {
                            if((nFirstBufferIdx!=size_t(-1) && nNextPacketSize>=nFirstBufferIdx) || nNextPacketSize>=nBufferSize)
                                break;
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
                        break;
                    nUsedBufferSize += nNextPacketSize;
                    ++nNextPrecacheIdx;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void litiv::IDataLoader::startAsyncPrecaching(bool bUsingGT, size_t nSuggestedBufferSize) {
    CV_Assert(m_oInputPrecacher.startAsyncPrecaching(nSuggestedBufferSize));
    CV_Assert(!bUsingGT || m_oGTPrecacher.startAsyncPrecaching(nSuggestedBufferSize));
}

void litiv::IDataLoader::stopAsyncPrecaching() {
    m_oInputPrecacher.stopAsyncPrecaching();
    m_oGTPrecacher.stopAsyncPrecaching();
}

litiv::IDataLoader::IDataLoader(ePacketPolicy eInputType, ePacketPolicy eOutputType, eMappingPolicy eGTMappingType, eMappingPolicy eIOMappingType) :
        m_oInputPrecacher(std::bind(&IDataLoader::_getInputPacket_redirect,this,std::placeholders::_1)),
        m_oGTPrecacher(std::bind(&IDataLoader::_getGTPacket_redirect,this,std::placeholders::_1)),
        m_eInputType(eInputType),m_eOutputType(eOutputType),m_eGTMappingType(eGTMappingType),m_eIOMappingType(eIOMappingType) {}

const cv::Mat& litiv::IDataLoader::_getInputPacket_redirect(size_t nIdx) {
    if(nIdx>=getTotPackets())
        return cv::emptyMat();
    m_oLatestInputPacket = _getInputPacket_impl(nIdx);
    if(!m_oLatestInputPacket.empty()) {
        CV_Assert(getInputOrigSize(nIdx)==m_oLatestInputPacket.size());
        if(m_eInputType==eImagePacket) {
            if(isInputTransposed(nIdx))
                cv::transpose(m_oLatestInputPacket,m_oLatestInputPacket);
#if HARDCODE_IMAGE_PACKET_INDEX
            std::stringstream sstr;
            sstr << "Packet #" << nIdx;
            writeOnImage(m_oLatestInputPacket,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_IMAGE_PACKET_INDEX
            if(getDatasetInfo()->is4ByteAligned() && m_oLatestInputPacket.channels()==3)
                cv::cvtColor(m_oLatestInputPacket,m_oLatestInputPacket,cv::COLOR_BGR2BGRA);
            const cv::Size& oPacketSize = getInputSize(nIdx);
            if(oPacketSize.area()>0 && m_oLatestInputPacket.size()!=oPacketSize)
                cv::resize(m_oLatestInputPacket,m_oLatestInputPacket,oPacketSize,0,0,cv::INTER_NEAREST);
        }
    }
    return m_oLatestInputPacket;
}

const cv::Mat& litiv::IDataLoader::_getGTPacket_redirect(size_t nIdx) {
    if(nIdx>=getTotPackets())
        return cv::emptyMat();
    m_oLatestGTPacket = _getGTPacket_impl(nIdx);
    if(!m_oLatestGTPacket.empty()) {
        CV_Assert(getGTOrigSize(nIdx)==m_oLatestGTPacket.size());
        if(m_eGTMappingType==ePixelMapping && m_eInputType==eImagePacket) {
            if(isGTTransposed(nIdx))
                cv::transpose(m_oLatestGTPacket,m_oLatestGTPacket);
#if HARDCODE_IMAGE_PACKET_INDEX
            std::stringstream sstr;
            sstr << "Packet #" << nIdx;
            writeOnImage(m_oLatestGTPacket,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //HARDCODE_IMAGE_PACKET_INDEX
            if(getDatasetInfo()->is4ByteAligned() && m_oLatestGTPacket.channels()==3)
                cv::cvtColor(m_oLatestGTPacket,m_oLatestGTPacket,cv::COLOR_BGR2BGRA);
            const cv::Size& oPacketSize = getGTSize(nIdx);
            if(oPacketSize.area()>0 && m_oLatestGTPacket.size()!=oPacketSize)
                cv::resize(m_oLatestGTPacket,m_oLatestGTPacket,oPacketSize,0,0,cv::INTER_NEAREST);
        }
    }
    return m_oLatestGTPacket;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

double litiv::IDataProducer_<litiv::eDatasetSource_Video>::getExpectedLoad() const {
    return m_oROI.empty()?0.0:(double)cv::countNonZero(m_oROI)*m_nFrameCount*(int(!isGrayscale())+1);
}

void litiv::IDataProducer_<litiv::eDatasetSource_Video>::startAsyncPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return IDataLoader::startAsyncPrecaching(bUsingGT,m_oSize.area()*(m_nFrameCount+1)*(isGrayscale()?1:getDatasetInfo()->is4ByteAligned()?4:3));
}

litiv::IDataProducer_<litiv::eDatasetSource_Video>::IDataProducer_(ePacketPolicy eOutputType, eMappingPolicy eGTMappingType, eMappingPolicy eIOMappingType) :
        IDataLoader(eImagePacket,eOutputType,eGTMappingType,eIOMappingType),m_nFrameCount(0),m_nNextExpectedVideoReaderFrameIdx(size_t(-1)),m_bTransposeFrames(false) {}

size_t litiv::IDataProducer_<litiv::eDatasetSource_Video>::getTotPackets() const {
    return m_nFrameCount;
}

bool litiv::IDataProducer_<litiv::eDatasetSource_Video>::isInputTransposed(size_t /*nPacketIdx*/) const {
    return m_bTransposeFrames;
}

bool litiv::IDataProducer_<litiv::eDatasetSource_Video>::isGTTransposed(size_t nPacketIdx) const {
    return getGTMappingType()==ePixelMapping?isInputTransposed(nPacketIdx):false;
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getInputROI(size_t /*nPacketIdx*/) const {
    return getROI();
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getGTROI(size_t nPacketIdx) const {
    return getGTMappingType()==ePixelMapping?getInputROI(nPacketIdx):cv::emptyMat();
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getInputSize(size_t /*nPacketIdx*/) const {
    return getFrameSize();
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getGTSize(size_t nPacketIdx) const {
    return getGTMappingType()==ePixelMapping?getInputSize(nPacketIdx):cv::emptySize();
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getInputOrigSize(size_t /*nPacketIdx*/) const {
    return getFrameOrigSize();
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getGTOrigSize(size_t nPacketIdx) const {
    return getGTMappingType()==ePixelMapping?getInputOrigSize(nPacketIdx):cv::emptySize();
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getInputMaxSize() const {
    return getFrameSize();
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Video>::getGTMaxSize() const {
    return getGTMappingType()==ePixelMapping?getFrameSize():cv::emptySize();
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Video>::_getInputPacket_impl(size_t nFrameIdx) {
    lvDbgAssert(getInputPacketType()==eImagePacket);
    lvDbgAssert(nFrameIdx<getTotPackets());
    cv::Mat oFrame;
    if(!m_voVideoReader.isOpened())
        oFrame = cv::imread(m_vsInputPaths[nFrameIdx],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    else {
        if(m_nNextExpectedVideoReaderFrameIdx!=nFrameIdx) {
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,(double)nFrameIdx);
            m_nNextExpectedVideoReaderFrameIdx = nFrameIdx+1;
        }
        else
            ++m_nNextExpectedVideoReaderFrameIdx;
        m_voVideoReader >> oFrame;
    }
    return oFrame;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Video>::_getGTPacket_impl(size_t nFrameIdx) {
    lvDbgAssert(nFrameIdx<getTotPackets());
    if(m_mGTIndexLUT.count(nFrameIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nFrameIdx];
        if(m_vsGTPaths.size()>nGTIdx) {
            lvAssert(getGTMappingType()==ePixelMapping); // loading as an image wouldnt make sense otherwise
            return cv::imread(m_vsGTPaths[nGTIdx],cv::IMREAD_GRAYSCALE); // @@@@ expose grayscale flag in class member?
        }
    }
    return cv::Mat();
}

void litiv::IDataProducer_<litiv::eDatasetSource_Video>::parseData() {
    lvAssert(getInputPacketType()==eImagePacket);
    cv::Mat oTempImg;
    m_voVideoReader.open(getDataPath());
    if(!m_voVideoReader.isOpened()) {
        PlatformUtils::GetFilesFromDir(getDataPath(),m_vsInputPaths);
        if(!m_vsInputPaths.empty()) {
            oTempImg = cv::imread(m_vsInputPaths[0]);
            m_nFrameCount = m_vsInputPaths.size();
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

double litiv::IDataProducer_<litiv::eDatasetSource_Image>::getExpectedLoad() const {
    return (double)getInputMaxSize().area()*m_nImageCount*(int(!isGrayscale())+1);
}

void litiv::IDataProducer_<litiv::eDatasetSource_Image>::startAsyncPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return IDataLoader::startAsyncPrecaching(bUsingGT,getInputMaxSize().area()*(m_nImageCount+1)*(isGrayscale()?1:getDatasetInfo()->is4ByteAligned()?4:3));
}

bool litiv::IDataProducer_<litiv::eDatasetSource_Image>::isInputConstantSize() const {
    return m_bIsInputConstantSize;
}

bool litiv::IDataProducer_<litiv::eDatasetSource_Image>::isGTConstantSize() const {
    return m_bIsGTConstantSize;
}

bool litiv::IDataProducer_<litiv::eDatasetSource_Image>::isInputTransposed(size_t nPacketIdx) const {
    lvAssert(nPacketIdx<m_nImageCount);
    return m_vbInputTransposed[nPacketIdx];
}

bool litiv::IDataProducer_<litiv::eDatasetSource_Image>::isGTTransposed(size_t nPacketIdx) const {
    lvAssert(getGTMappingType()<=eIdxMapping);
    lvAssert(nPacketIdx<m_nImageCount);
    return m_vbGTTransposed[nPacketIdx];
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getInputROI(size_t /*nPacketIdx*/) const {
    return cv::emptyMat();
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getGTROI(size_t /*nPacketIdx*/) const {
    return cv::emptyMat();
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getInputSize(size_t nPacketIdx) const {
    if(nPacketIdx>=m_nImageCount)
        return cv::emptySize();
    return m_voInputSizes[nPacketIdx];
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getGTSize(size_t nPacketIdx) const {
    lvAssert(getGTMappingType()<=eIdxMapping);
    if(nPacketIdx>=m_nImageCount)
        return cv::emptySize();
    return m_voGTSizes[nPacketIdx];
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getInputOrigSize(size_t nPacketIdx) const {
    if(nPacketIdx>=m_nImageCount)
        return cv::emptySize();
    return m_voInputOrigSizes[nPacketIdx];
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getGTOrigSize(size_t nPacketIdx) const {
    lvAssert(getGTMappingType()<=eIdxMapping);
    if(nPacketIdx>=m_nImageCount)
        return cv::emptySize();
    return m_voGTOrigSizes[nPacketIdx];
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getInputMaxSize() const {
    return m_oInputMaxSize;
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getGTMaxSize() const {
    return m_oGTMaxSize;
}

std::string litiv::IDataProducer_<litiv::eDatasetSource_Image>::getPacketName(size_t nPacketIdx) const {
    lvAssert(nPacketIdx<m_nImageCount);
    const size_t nLastSlashPos = m_vsInputPaths[nPacketIdx].find_last_of("/\\");
    std::string sFileName = (nLastSlashPos==std::string::npos)?m_vsInputPaths[nPacketIdx]:m_vsInputPaths[nPacketIdx].substr(nLastSlashPos+1);
    return sFileName.substr(0,sFileName.find_last_of("."));
}

litiv::IDataProducer_<litiv::eDatasetSource_Image>::IDataProducer_(ePacketPolicy eOutputType, eMappingPolicy eGTMappingType, eMappingPolicy eIOMappingType) :
        IDataLoader(eImagePacket,eOutputType,eGTMappingType,eIOMappingType),m_nImageCount(0) {}

size_t litiv::IDataProducer_<litiv::eDatasetSource_Image>::getTotPackets() const {
    return m_nImageCount;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Image>::_getInputPacket_impl(size_t nImageIdx) {
    lvDbgAssert(getInputPacketType()==eImagePacket);
    lvDbgAssert(nImageIdx<getTotPackets());
    return cv::imread(m_vsInputPaths[nImageIdx],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Image>::_getGTPacket_impl(size_t nImageIdx) {
    lvDbgAssert(nImageIdx<getTotPackets());
    if(m_mGTIndexLUT.count(nImageIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nImageIdx];
        if(m_vsGTPaths.size()>nGTIdx) {
            lvAssert(getGTMappingType()==ePixelMapping); // loading as an image wouldnt make sense otherwise
            return cv::imread(m_vsGTPaths[m_mGTIndexLUT[nGTIdx]],cv::IMREAD_GRAYSCALE); // @@@@ expose grayscale flag in class member?
        }
    }
    return cv::Mat();
}

void litiv::IDataProducer_<litiv::eDatasetSource_Image>::parseData() {
    lvAssert(getInputPacketType()==eImagePacket);
    PlatformUtils::GetFilesFromDir(getDataPath(),m_vsInputPaths);
    PlatformUtils::FilterFilePaths(m_vsInputPaths,{},{".jpg",".png",".bmp"});
    if(m_vsInputPaths.empty())
        lvErrorExt("Set '%s' did not possess any jpg/png/bmp image file",getName().c_str());
    m_bIsInputConstantSize = m_bIsGTConstantSize = true;
    m_oInputMaxSize = m_oGTMaxSize = cv::Size(0,0);
    m_voInputSizes.clear();
    m_voInputSizes.reserve(m_vsInputPaths.size());
    m_voGTSizes.clear();
    m_voGTSizes.reserve(m_vsInputPaths.size());
    m_voInputOrigSizes.clear();
    m_voInputOrigSizes.reserve(m_vsInputPaths.size());
    m_voGTOrigSizes.clear();
    m_voGTOrigSizes.reserve(m_vsInputPaths.size());
    m_vbInputTransposed.clear();
    m_vbInputTransposed.reserve(m_vsInputPaths.size());
    m_vbGTTransposed.clear();
    m_vbGTTransposed.reserve(m_vsInputPaths.size());
    cv::Mat oLastInput;
    const double dScale = getDatasetInfo()->getScaleFactor();
    for(size_t n = 0; n<m_vsInputPaths.size(); ++n) {
        cv::Mat oCurrInput = cv::imread(m_vsInputPaths[n],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
        while(oCurrInput.empty()) {
            m_vsInputPaths.erase(m_vsInputPaths.begin()+n);
            if(n>=m_vsInputPaths.size())
                break;
            oCurrInput = cv::imread(m_vsInputPaths[n],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
        }
        if(oCurrInput.empty())
            break;
        m_voInputOrigSizes.push_back(oCurrInput.size());
        if(dScale!=1.0)
            cv::resize(oCurrInput,oCurrInput,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
        m_voInputSizes.push_back(oCurrInput.size());
        if(m_oInputMaxSize.width<oCurrInput.cols)
            m_oInputMaxSize.width = oCurrInput.cols;
        if(m_oInputMaxSize.height<oCurrInput.rows)
            m_oInputMaxSize.height = oCurrInput.rows;
        if(!oLastInput.empty() && oCurrInput.size()!=oLastInput.size())
            m_bIsInputConstantSize = false;
        oLastInput = oCurrInput;
        m_vbInputTransposed.push_back(false);
    }
    m_nImageCount = m_vsInputPaths.size();
    CV_Assert(m_nImageCount>0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

size_t litiv::DataCounter_<litiv::eNotGroup>::getProcessedPacketsCountPromise() {
    return m_nProcessedPacketsPromise.get_future().get();
}

size_t litiv::DataCounter_<litiv::eNotGroup>::getProcessedPacketsCount() const {
    return m_nProcessedPackets;
}

size_t litiv::DataCounter_<litiv::eGroup>::getProcessedPacketsCountPromise() {
    return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCountPromise();});
}

size_t litiv::DataCounter_<litiv::eGroup>::getProcessedPacketsCount() const {
    return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCount();});
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

litiv::DataWriter::DataWriter(std::function<size_t(const cv::Mat&,size_t)> lDataArchiverCallback) :
        m_lCallback(lDataArchiverCallback) {
    CV_Assert(m_lCallback);
    m_bIsActive = false;
    m_nQueueSize = 0;
    m_nQueueCount = 0;
}

litiv::DataWriter::~DataWriter() {
    stopAsyncWriting();
}

size_t litiv::DataWriter::queue(const cv::Mat& oPacket, size_t nIdx) {
    if(!m_bIsActive)
        return m_lCallback(oPacket,nIdx);
    cv::Mat oPacketCopy = oPacket.clone();
    const size_t nPacketSize = oPacket.total()*oPacket.elemSize();
    size_t nPacketPosition;
    {
        std::unique_lock<std::mutex> sync_lock(m_oSyncMutex);
        // @@@ could cut a find operation here using C++17's map::insert_or_assign
        m_mQueue[nIdx] = std::move(oPacketCopy);
        m_nQueueSize += nPacketSize;
        nPacketPosition = std::distance(m_mQueue.begin(),m_mQueue.find(nIdx));
        ++m_nQueueCount;
    }
    m_oQueueCondVar.notify_one();
    return nPacketPosition;
}

bool litiv::DataWriter::startAsyncWriting(size_t nSuggestedQueueSize, size_t nWorkers) {
    if(m_bIsActive)
        stopAsyncWriting();
    if(nSuggestedQueueSize>0) {
        m_bIsActive = true;
        m_nQueueSize = 0;
        m_nQueueCount = 0;
        m_mQueue.clear();
        for(size_t n=0; n<nWorkers; ++n)
            m_vhWorkers.emplace_back(std::bind(&DataWriter::entry,this,(nSuggestedQueueSize>CACHE_MAX_SIZE)?(CACHE_MAX_SIZE):nSuggestedQueueSize));
    }
    return m_bIsActive;
}

void litiv::DataWriter::stopAsyncWriting() {
    if(m_bIsActive) {
        m_bIsActive = false;
        m_oQueueCondVar.notify_all();
        for(std::thread& oWorker : m_vhWorkers)
            oWorker.join();
    }
}

void litiv::DataWriter::entry(const size_t nMaxQueueSize) {
    std::unique_lock<std::mutex> sync_lock(m_oSyncMutex);
#if CONSOLE_DEBUG
    std::cout << " @ initializing writing with max buffer size = " << (nMaxQueueSize/1024)/1024 << " mb" << std::endl;
#endif //CONSOLE_DEBUG
    while(m_bIsActive || m_nQueueCount>0) {
        if(m_nQueueCount==0)
            m_oQueueCondVar.wait(sync_lock);
        if(m_nQueueCount>0) {
            cv::Mat oPacketData;
            size_t nPacketIdx;
            do {
                auto pCurrPacket = m_mQueue.begin();
                oPacketData = std::move(pCurrPacket->second);
                nPacketIdx = pCurrPacket->first;
                const size_t nPacketSize = oPacketData.total()*oPacketData.elemSize();
                lvDbgAssert(nPacketSize<m_nQueueSize);
                m_nQueueSize -= nPacketSize;
                m_mQueue.erase(pCurrPacket);
                --m_nQueueCount;
            } while(m_nQueueSize>=nMaxQueueSize);
            CxxUtils::unlock_guard<std::unique_lock<std::mutex>> oUnlock(sync_lock);
            m_lCallback(oPacketData,nPacketIdx);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

size_t litiv::IDataArchiver::save(const cv::Mat& oOutput, size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << getPacketName(nIdx) << getDatasetInfo()->getOutputNameSuffix();
    const auto pLoader = shared_from_this_cast<const IDataLoader>(true);
    if(pLoader->getIOMappingType()==ePixelMapping && pLoader->getOutputPacketType()==eImagePacket) {
        const cv::Mat& oROI = pLoader->getInputROI(nIdx);
        cv::Mat oOutputClone = oOutput.clone();
        if(!oROI.empty() && oROI.size()==oOutputClone.size())
            cv::bitwise_or(oOutputClone,DATASETUTILS_UNKNOWN_VAL,oOutputClone,oROI==0);
        if(pLoader->isInputTransposed(nIdx))
            cv::transpose(oOutputClone,oOutputClone);
        if(pLoader->getInputOrigSize(nIdx).area()>0 && oOutputClone.size()!=pLoader->getInputOrigSize(nIdx))
            cv::resize(oOutputClone,oOutputClone,pLoader->getInputOrigSize(nIdx),0,0,cv::INTER_NEAREST);
        const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
        cv::imwrite(sOutputFilePath.str(),oOutputClone,vnComprParams);
    }
    else {
        // @@@@ save to YML
        lvError("Missing impl");
    }
    return 0;
}

cv::Mat litiv::IDataArchiver::load(size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << getPacketName(nIdx) << getDatasetInfo()->getOutputNameSuffix();
    const auto pLoader = shared_from_this_cast<const IDataLoader>(true);
    if(pLoader->getIOMappingType()==ePixelMapping && pLoader->getOutputPacketType()==eImagePacket) {
        cv::Mat oOutput = cv::imread(sOutputFilePath.str(),isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
        if(pLoader->isInputTransposed(nIdx))
            cv::transpose(oOutput,oOutput);
        if(getDatasetInfo()->is4ByteAligned() && oOutput.channels()==3)
            cv::cvtColor(oOutput,oOutput,cv::COLOR_BGR2BGRA);
        if(pLoader->getInputSize(nIdx).area()>0 && oOutput.size()!=pLoader->getInputSize(nIdx))
            cv::resize(oOutput,oOutput,pLoader->getInputSize(nIdx),0,0,cv::INTER_NEAREST);
        return oOutput;
    }
    else {
        // @@@@ read from YML
        lvError("Missing impl");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#if HAVE_GLSL

cv::Size litiv::IAsyncDataConsumer_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::getIdealGLWindowSize() const {
    glAssert(getTotPackets()>1);
    cv::Size oWindowSize = shared_from_this_cast<const IDataLoader>(true)->getInputMaxSize();
    if(m_pEvalAlgo) {
        glAssert(m_pEvalAlgo->getIsGLInitialized());
        oWindowSize.width *= int(m_pEvalAlgo->m_nSxSDisplayCount);
    }
    else if(m_pAlgo) {
        glAssert(m_pAlgo->getIsGLInitialized());
        oWindowSize.width *= int(m_pAlgo->m_nSxSDisplayCount);
    }
    return oWindowSize;
}

litiv::IAsyncDataConsumer_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::IAsyncDataConsumer_() :
        m_nLastIdx(0),
        m_nCurrIdx(0),
        m_nNextIdx(1) {}

void litiv::IAsyncDataConsumer_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::pre_initialize_gl() {
    m_pLoader = shared_from_this_cast<IDataLoader>(true);
    glAssert(m_pLoader->getTotPackets()>1);
    glDbgAssert(m_pAlgo);
    m_oCurrInput = m_pLoader->getInput(m_nCurrIdx).clone();
    m_oNextInput = m_pLoader->getInput(m_nNextIdx).clone();
    m_oLastInput = m_oCurrInput.clone();
    CV_Assert(!m_oCurrInput.empty());
    CV_Assert(m_oCurrInput.isContinuous());
    glAssert(m_oCurrInput.channels()==1 || m_oCurrInput.channels()==4);
    if(getDatasetInfo()->isSavingOutput() || m_pAlgo->m_pDisplayHelper)
        m_pAlgo->setOutputFetching(true);
    if(m_pAlgo->m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
        m_pAlgo->setDebugFetching(true);
    if(getDatasetInfo()->isUsingEvaluator()) {
        m_oCurrGT = m_pLoader->getGT(m_nCurrIdx).clone();
        m_oNextGT = m_pLoader->getGT(m_nNextIdx).clone();
        m_oLastGT = m_oCurrGT.clone();
        CV_Assert(!m_oCurrGT.empty());
        CV_Assert(m_oCurrGT.isContinuous());
        glAssert(m_oCurrGT.channels()==1 || m_oCurrGT.channels()==4);
    }
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::post_initialize_gl() {
    glDbgAssert(m_pAlgo);
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::pre_apply_gl(size_t nNextIdx, bool bRebindAll) {
    UNUSED(bRebindAll);
    glDbgAssert(m_pLoader);
    glDbgAssert(m_pAlgo);
    if(nNextIdx!=m_nNextIdx)
        m_oNextInput = m_pLoader->getInput(nNextIdx);
    if(getDatasetInfo()->isUsingEvaluator() && nNextIdx!=m_nNextIdx)
        m_oNextGT = m_pLoader->getGT(nNextIdx);
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::post_apply_gl(size_t nNextIdx, bool bRebindAll) {
    glDbgAssert(m_pLoader);
    glDbgAssert(m_pAlgo);
    if(m_pEvalAlgo && getDatasetInfo()->isUsingEvaluator())
        m_pEvalAlgo->apply_gl(m_oNextGT,bRebindAll);
    m_nLastIdx = m_nCurrIdx;
    m_nCurrIdx = nNextIdx;
    m_nNextIdx = nNextIdx+1;
    if(m_pAlgo->m_pDisplayHelper || m_lDataCallback) {
        m_oCurrInput.copyTo(m_oLastInput);
        m_oNextInput.copyTo(m_oCurrInput);
        if(getDatasetInfo()->isUsingEvaluator()) {
            m_oCurrGT.copyTo(m_oLastGT);
            m_oNextGT.copyTo(m_oCurrGT);
        }
    }
    if(m_nNextIdx<getTotPackets()) {
        m_oNextInput = m_pLoader->getInput(m_nNextIdx);
        if(getDatasetInfo()->isUsingEvaluator())
            m_oNextGT = m_pLoader->getGT(m_nNextIdx);
    }
    processPacket();
    if(getDatasetInfo()->isSavingOutput() || m_pAlgo->m_pDisplayHelper || m_lDataCallback) {
        cv::Mat oLastOutput,oLastDebug;
        m_pAlgo->fetchLastOutput(oLastOutput);
        if(m_pAlgo->m_pDisplayHelper && m_pEvalAlgo && m_pEvalAlgo->m_bUsingDebug)
            m_pEvalAlgo->fetchLastDebug(oLastDebug);
        else if(m_pAlgo->m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
            m_pAlgo->fetchLastDebug(oLastDebug);
        else
            oLastDebug = oLastOutput.clone();
        if(getDatasetInfo()->isSavingOutput())
            save(oLastOutput,m_nLastIdx);
        if(m_lDataCallback)
            m_lDataCallback(m_oLastInput,oLastDebug,oLastOutput,m_oLastGT,m_pLoader->getInputROI(m_nLastIdx),m_nLastIdx);
        if(m_pAlgo->m_pDisplayHelper) {
            getColoredMasks(oLastOutput,oLastDebug,m_oLastGT,m_pLoader->getInputROI(m_nLastIdx));
            m_pAlgo->m_pDisplayHelper->display(m_oLastInput,oLastDebug,oLastOutput,m_nLastIdx);
        }
    }
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::getColoredMasks(cv::Mat& oOutput, cv::Mat& oDebug, const cv::Mat& /*oGT*/, const cv::Mat& oROI) {
    if(!oROI.empty()) {
        lvAssert(oOutput.size()==oROI.size());
        lvAssert(oDebug.size()==oROI.size());
        cv::bitwise_or(oOutput,UCHAR_MAX/2,oOutput,oROI==0);
        cv::bitwise_or(oDebug,UCHAR_MAX/2,oDebug,oROI==0);
    }
}

#endif //HAVE_GLSL
