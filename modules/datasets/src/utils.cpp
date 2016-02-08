
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
    if(getDatasetInfo()->is4ByteAligned() && m_oLatestInputPacket.channels()==3)
        cv::cvtColor(m_oLatestInputPacket,m_oLatestInputPacket,cv::COLOR_BGR2BGRA);
    const double dScale = getDatasetInfo()->getScaleFactor();
    if(dScale!=1.0)
        cv::resize(m_oLatestInputPacket,m_oLatestInputPacket,cv::Size(),dScale,dScale,cv::INTER_CUBIC);
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
    if(getDatasetInfo()->is4ByteAligned() && m_oLatestInputPacket.channels()==3)
        cv::cvtColor(m_oLatestGTPacket,m_oLatestGTPacket,cv::COLOR_BGR2BGRA);
    const double dScale = getDatasetInfo()->getScaleFactor();
    if(dScale!=1.0)
        cv::resize(m_oLatestGTPacket,m_oLatestGTPacket,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
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
    }
    return oFrame;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::_getGTPacket_impl(size_t) {
    return cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL));
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

bool litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eGroup>::isConstantSize() const {
    for(const auto& pBatch : getBatches())
        if(!dynamic_cast<const IDataProducer_<eDatasetType_ImageEdgDet,eNotGroup>&>(*pBatch).isConstantSize())
            return false;
    return true;
}

cv::Size litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eGroup>::getMaxImageSize() const {
    cv::Size oMaxSize(0,0);
    for(const auto& pBatch : getBatches()) {
        const cv::Size oCurrSize = dynamic_cast<const IDataProducer_<eDatasetType_ImageEdgDet,eNotGroup>&>(*pBatch).getMaxImageSize();
        if(oCurrSize.width>oMaxSize.width)
            oMaxSize.width = oCurrSize.width;
        if(oCurrSize.height>oMaxSize.height)
            oMaxSize.height = oCurrSize.height;
    }
    return oMaxSize;
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eGroup>::getInputImage(size_t nImageIdx) {
    return dynamic_cast<IDataReader_<eDatasetType_ImageEdgDet>&>(*getBatch(nImageIdx)).getInputImage(nImageIdx);
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eGroup>::getGTMask(size_t nImageIdx) {
    return dynamic_cast<IDataReader_<eDatasetType_ImageEdgDet>&>(*getBatch(nImageIdx)).getGTMask(nImageIdx);
}

double litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::getExpectedLoad() const {
    return getMaxImageSize().area()*m_nImageCount*(int(!isGrayscale())+1);
}

size_t litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::getTotPackets() const {
    return m_nImageCount;
}

void litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::startPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return IDataLoader_<eNotGroup>::startPrecaching(bUsingGT,getMaxImageSize().area()*(m_nImageCount+1)*(isGrayscale()?1:getDatasetInfo()->is4ByteAligned()?4:3));
}

bool litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::isConstantSize() const {
    return m_bIsConstantSize;
}

cv::Size litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::getMaxImageSize() const {
    return m_oMaxSize;
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::getInputImage(size_t nImageIdx) {
    return m_oInputPrecacher.getPacket(nImageIdx);
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::getGTMask(size_t nImageIdx) {
    return m_oGTPrecacher.getPacket(nImageIdx);
}

std::string litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::getInputImageName(size_t nImageIdx, bool bStripExt) const {
    lvAssert(nImageIdx<m_nImageCount);
    const size_t nLastSlashPos = m_vsInputImagePaths[nImageIdx].find_last_of("/\\");
    std::string sFileName = (nLastSlashPos==std::string::npos)?m_vsInputImagePaths[nImageIdx]:m_vsInputImagePaths[nImageIdx].substr(nLastSlashPos+1);
    if(bStripExt)
        sFileName = sFileName.substr(0,sFileName.find_last_of("."));
    return sFileName;
}

cv::Size litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::getInputImageSize(size_t nImageIdx) const {
    lvAssert(nImageIdx<m_nImageCount);
    return m_voOrigImageSizes[nImageIdx];
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::_getInputPacket_impl(size_t nImageIdx) {
    cv::Mat oImage = cv::imread(m_vsInputImagePaths[nImageIdx],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    CV_Assert(!oImage.empty());
    CV_Assert(m_voOrigImageSizes[nImageIdx]==oImage.size());
    return oImage;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::_getGTPacket_impl(size_t nImageIdx) {
    return cv::Mat(m_voOrigImageSizes[nImageIdx],CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL));
}

void litiv::IDataProducer_<litiv::eDatasetType_ImageEdgDet,litiv::eNotGroup>::parseData() {
    PlatformUtils::GetFilesFromDir(getDataPath(),m_vsInputImagePaths);
    PlatformUtils::FilterFilePaths(m_vsInputImagePaths,{},{".jpg",".png",".bmp"});
    if(m_vsInputImagePaths.empty())
        lvErrorExt("Set '%s' did not possess any jpg/png/bmp image file",getName().c_str());
    m_bIsConstantSize = true;
    m_oMaxSize = cv::Size(0,0);
    cv::Mat oLastInput;
    m_voOrigImageSizes.clear();
    m_voOrigImageSizes.reserve(m_vsInputImagePaths.size());
    for(size_t n = 0; n<m_vsInputImagePaths.size(); ++n) {
        cv::Mat oCurrInput = cv::imread(m_vsInputImagePaths[n],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
        while(oCurrInput.empty()) {
            m_vsInputImagePaths.erase(m_vsInputImagePaths.begin()+n);
            if(n>=m_vsInputImagePaths.size())
                break;
            oCurrInput = cv::imread(m_vsInputImagePaths[n],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
        }
        if(oCurrInput.empty())
            break;
        if(m_oMaxSize.width<oCurrInput.cols)
            m_oMaxSize.width = oCurrInput.cols;
        if(m_oMaxSize.height<oCurrInput.rows)
            m_oMaxSize.height = oCurrInput.rows;
        if(!oLastInput.empty() && oCurrInput.size()!=oLastInput.size())
            m_bIsConstantSize = false;
        m_voOrigImageSizes.push_back(oCurrInput.size());
        oLastInput = oCurrInput;
    }
    m_nImageCount = m_vsInputImagePaths.size();
    const double dScale = getDatasetInfo()->getScaleFactor();
    if(dScale!=1.0) {
        oLastInput.create(m_oMaxSize,CV_8UC1);
        cv::resize(oLastInput,oLastInput,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
        m_oMaxSize = oLastInput.size(); // helps make sure we have the same scale rounding
    }
    CV_Assert(m_nImageCount>0);
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

void litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm>::pushSegmMask(const cv::Mat& oSegm, size_t nIdx) {
    processPacket();
    if(getDatasetInfo()->isSavingOutput())
        writeSegmMask(oSegm,nIdx);
}

void litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm>::writeSegmMask(const cv::Mat& oSegm, size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << acBuffer.data() << getDatasetInfo()->getOutputNameSuffix();
    const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
    auto pProducer = shared_from_this_cast<const IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(true);
    const cv::Mat& oROI = pProducer->getROI();
    cv::Mat oOutputSegm;
    if(!oROI.empty())
        cv::bitwise_or(oSegm,UCHAR_MAX/2,oOutputSegm,oROI==0);
    else
        oOutputSegm = oSegm;
    if(oOutputSegm.size()!=pProducer->getOrigFrameSize())
        cv::resize(oOutputSegm,oOutputSegm,pProducer->getOrigFrameSize(),0,0,cv::INTER_NEAREST);
    cv::imwrite(sOutputFilePath.str(),oOutputSegm,vnComprParams);
}

cv::Mat litiv::IDataConsumer_<litiv::eDatasetType_VideoSegm>::readSegmMask(size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << acBuffer.data() << getDatasetInfo()->getOutputNameSuffix();
    cv::Mat oSegm = cv::imread(sOutputFilePath.str(),isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    if(getDatasetInfo()->is4ByteAligned() && oSegm.channels()==3)
        cv::cvtColor(oSegm,oSegm,cv::COLOR_BGR2BGRA);
    const double dScale = getDatasetInfo()->getScaleFactor();
    if(dScale!=1.0)
        cv::resize(oSegm,oSegm,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
    return oSegm;
}

void litiv::IDataConsumer_<litiv::eDatasetType_ImageEdgDet>::pushEdgeMask(const cv::Mat& oEdges, size_t nIdx) {
    processPacket();
    if(getDatasetInfo()->isSavingOutput())
        writeEdgeMask(oEdges,nIdx);
}

void litiv::IDataConsumer_<litiv::eDatasetType_ImageEdgDet>::writeEdgeMask(const cv::Mat& oEdges, size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    auto pProducer = shared_from_this_cast<const IDataProducer_<eDatasetType_ImageEdgDet,eNotGroup>>(true);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << pProducer->getInputImageName(nIdx) << getDatasetInfo()->getOutputNameSuffix();
    const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
    cv::Mat oResizedEdges = oEdges;
    if(oEdges.size()!=pProducer->getInputImageSize(nIdx))
        cv::resize(oResizedEdges,oResizedEdges,pProducer->getInputImageSize(nIdx),0,0,cv::INTER_NEAREST);
    cv::imwrite(sOutputFilePath.str(),oResizedEdges,vnComprParams);
}

cv::Mat litiv::IDataConsumer_<litiv::eDatasetType_ImageEdgDet>::readEdgeMask(size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    auto pProducer = shared_from_this_cast<const IDataProducer_<eDatasetType_ImageEdgDet,eNotGroup>>(true);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << pProducer->getInputImageName(nIdx) << getDatasetInfo()->getOutputNameSuffix();
    cv::Mat oEdges = cv::imread(sOutputFilePath.str(),isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    if(getDatasetInfo()->is4ByteAligned() && oEdges.channels()==3)
        cv::cvtColor(oEdges,oEdges,cv::COLOR_BGR2BGRA);
    const double dScale = getDatasetInfo()->getScaleFactor();
    if(dScale!=1.0)
        cv::resize(oEdges,oEdges,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
    return oEdges;
}

#if HAVE_GLSL

cv::Size litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::getIdealGLWindowSize() const {
    glAssert(m_pAlgo && m_pAlgo->getIsGLInitialized());
    auto pProducer = shared_from_this_cast<const IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(true);
    glAssert(pProducer->getFrameCount()>1);
    cv::Size oFrameSize = pProducer->getFrameSize();
    oFrameSize.width *= int(m_pAlgo->m_nSxSDisplayCount);
    return oFrameSize;
}

litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::IAsyncDataConsumer_() :
        m_nLastIdx(0),
        m_nCurrIdx(0),
        m_nNextIdx(1),
        m_nFrameCount(0) {}

void litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::pre_initialize_gl() {
    m_pProducer = shared_from_this_cast<IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(true);
    glAssert(m_pProducer->getFrameCount()>1);
    glDbgAssert(m_pAlgo);
    m_oNextInput = m_pProducer->getInputFrame(m_nNextIdx).clone();
    m_oCurrInput = m_pProducer->getInputFrame(m_nCurrIdx).clone();
    m_oLastInput = m_oCurrInput.clone();
    CV_Assert(!m_oCurrInput.empty());
    CV_Assert(m_oCurrInput.isContinuous());
    glAssert(m_oCurrInput.channels()==1 || m_oCurrInput.channels()==4);
    m_nFrameCount= m_pProducer->getFrameCount();
    if(getDatasetInfo()->isSavingOutput() || m_pAlgo->m_pDisplayHelper)
        m_pAlgo->setOutputFetching(true);
    if(m_pAlgo->m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
        m_pAlgo->setDebugFetching(true);
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::post_initialize_gl() {
    glDbgAssert(m_pAlgo);
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::pre_apply_gl(size_t nNextIdx, bool bRebindAll) {
    UNUSED(bRebindAll);
    glDbgAssert(m_pProducer);
    glDbgAssert(m_pAlgo);
    if(nNextIdx!=m_nNextIdx)
        m_oNextInput = m_pProducer->getInputFrame(nNextIdx);
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::post_apply_gl(size_t nNextIdx, bool bRebindAll) {
    UNUSED(bRebindAll);
    glDbgAssert(m_pProducer);
    glDbgAssert(m_pAlgo);
    m_nLastIdx = m_nCurrIdx;
    m_nCurrIdx = nNextIdx;
    m_nNextIdx = nNextIdx+1;
    if(m_pAlgo->m_pDisplayHelper) {
        m_oCurrInput.copyTo(m_oLastInput);
        m_oNextInput.copyTo(m_oCurrInput);
    }
    if(m_nNextIdx<m_nFrameCount)
        m_oNextInput = m_pProducer->getInputFrame(m_nNextIdx);
    processPacket();
    if(getDatasetInfo()->isSavingOutput() || m_pAlgo->m_pDisplayHelper) {
        cv::Mat oLastOutput,oLastDebug;
        m_pAlgo->fetchLastOutput(oLastOutput);
        if(m_pAlgo->m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
            m_pAlgo->fetchLastDebug(oLastDebug);
        else
            oLastDebug = oLastOutput;
        if(getDatasetInfo()->isSavingOutput())
            writeSegmMask(oLastOutput,m_nLastIdx);
        if(m_pAlgo->m_pDisplayHelper) {
            const cv::Mat& oROI = m_pProducer->getROI();
            if(!oROI.empty()) {
                cv::bitwise_or(oLastOutput,UCHAR_MAX/2,oLastOutput,oROI==0);
                cv::bitwise_or(oLastDebug,UCHAR_MAX/2,oLastDebug,oROI==0);
            }
            m_pAlgo->m_pDisplayHelper->display(m_oLastInput,oLastDebug,oLastOutput,m_nLastIdx);
        }
    }
}

void litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::writeSegmMask(const cv::Mat& oSegm,size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << acBuffer.data() << getDatasetInfo()->getOutputNameSuffix();
    const std::vector<int> vnComprParams ={cv::IMWRITE_PNG_COMPRESSION,9};
    auto pProducer = shared_from_this_cast<const IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(true);
    const cv::Mat& oROI = pProducer->getROI();
    cv::Mat oOutputSegm;
    if(!oROI.empty())
        cv::bitwise_or(oSegm,UCHAR_MAX/2,oOutputSegm,oROI==0);
    else
        oOutputSegm = oSegm;
    cv::imwrite(sOutputFilePath.str(),oOutputSegm,vnComprParams);
}

cv::Mat litiv::IAsyncDataConsumer_<litiv::eDatasetType_VideoSegm,ParallelUtils::eGLSL>::readSegmMask(size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << acBuffer.data() << getDatasetInfo()->getOutputNameSuffix();
    return cv::imread(sOutputFilePath.str(),isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
}

#endif //HAVE_GLSL
