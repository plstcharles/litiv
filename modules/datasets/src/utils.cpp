
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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void litiv::IIDataLoader::startPrecaching(bool bUsingGT, size_t nSuggestedBufferSize) {
    CV_Assert(m_oInputPrecacher.startPrecaching(getTotPackets(),nSuggestedBufferSize));
    CV_Assert(!bUsingGT || m_oGTPrecacher.startPrecaching(getTotPackets(),nSuggestedBufferSize));
}

void litiv::IIDataLoader::stopPrecaching() {
    m_oInputPrecacher.stopPrecaching();
    m_oGTPrecacher.stopPrecaching();
}

litiv::IIDataLoader::IIDataLoader(ePacketPolicy ePacket) :
        m_oInputPrecacher(std::bind(&IIDataLoader::_getInputPacket_redirect,this,std::placeholders::_1)),
        m_oGTPrecacher(std::bind(&IIDataLoader::_getGTPacket_redirect,this,std::placeholders::_1)),
        m_ePacketType(ePacket) {}

const cv::Mat& litiv::IIDataLoader::_getInputPacket_redirect(size_t nIdx) {
    CV_Assert(nIdx<getTotPackets());
    m_oLatestInputPacket = _getInputPacket_impl(nIdx);
    if(m_ePacketType==eImagePacket) {
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
    }
    return m_oLatestInputPacket;
}

const cv::Mat& litiv::IIDataLoader::_getGTPacket_redirect(size_t nIdx) {
    CV_Assert(nIdx<getTotPackets());
    m_oLatestGTPacket = _getGTPacket_impl(nIdx);
    if(m_ePacketType==eImagePacket) {
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
    }
    return m_oLatestGTPacket;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

double litiv::IDataProducer_<litiv::eDatasetSource_Video>::getExpectedLoad() const {
    return m_oROI.empty()?0.0:(double)cv::countNonZero(m_oROI)*m_nFrameCount*(int(!isGrayscale())+1);
}

void litiv::IDataProducer_<litiv::eDatasetSource_Video>::startPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return IIDataLoader::startPrecaching(bUsingGT,m_oSize.area()*(m_nFrameCount+1)*(isGrayscale()?1:getDatasetInfo()->is4ByteAligned()?4:3));
}

litiv::IDataProducer_<litiv::eDatasetSource_Video>::IDataProducer_() :
        m_nFrameCount(0),m_nNextExpectedVideoReaderFrameIdx(size_t(-1)),m_bTransposeFrames(false) {}

size_t litiv::IDataProducer_<litiv::eDatasetSource_Video>::getTotPackets() const {
    return m_nFrameCount;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Video>::_getInputPacket_impl(size_t nFrameIdx) {
    cv::Mat oFrame;
    if(!m_voVideoReader.isOpened())
        oFrame = cv::imread(m_vsInputFramePaths[nFrameIdx],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    else {
        if(m_nNextExpectedVideoReaderFrameIdx!=nFrameIdx) {
            m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,(double)nFrameIdx);
            m_nNextExpectedVideoReaderFrameIdx = nFrameIdx+1;
        }
        else
            ++m_nNextExpectedVideoReaderFrameIdx;
        m_voVideoReader >> oFrame;
    }
    CV_Assert(getPacketOrigSize(nFrameIdx)==oFrame.size());
    if(isPacketTransposed(nFrameIdx))
        cv::transpose(oFrame,oFrame);
    if(getDatasetInfo()->is4ByteAligned() && oFrame.channels()==3)
        cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2BGRA);
    if(getPacketSize(nFrameIdx).area()>0 && oFrame.size()!=getPacketSize(nFrameIdx))
        cv::resize(oFrame,oFrame,getPacketSize(nFrameIdx),0,0,cv::INTER_NEAREST);
    return oFrame;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Video>::_getGTPacket_impl(size_t nFrameIdx) {
    if(m_mGTIndexLUT.count(nFrameIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nFrameIdx];
        if(m_vsGTFramePaths.size()>nGTIdx) {
            cv::Mat oFrame = cv::imread(m_vsGTFramePaths[nGTIdx],cv::IMREAD_GRAYSCALE);
            if(!oFrame.empty()) {
                lvAssert(oFrame.size()==getPacketOrigSize(nFrameIdx));
                if(isPacketTransposed(nFrameIdx))
                    cv::transpose(oFrame,oFrame);
                if(getPacketSize(nFrameIdx).area()>0 && oFrame.size()!=getPacketSize(nFrameIdx))
                    cv::resize(oFrame,oFrame,getPacketSize(nFrameIdx),0,0,cv::INTER_NEAREST);
                return oFrame;
            }
        }
    }
    return cv::Mat(getPacketSize(nFrameIdx),CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUTOFSCOPE_VAL));
}

void litiv::IDataProducer_<litiv::eDatasetSource_Video>::parseData() {
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

double litiv::IDataProducer_<litiv::eDatasetSource_Image>::getExpectedLoad() const {
    return (double)getPacketMaxSize().area()*m_nImageCount*(int(!isGrayscale())+1);
}

void litiv::IDataProducer_<litiv::eDatasetSource_Image>::startPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return IIDataLoader::startPrecaching(bUsingGT,getPacketMaxSize().area()*(m_nImageCount+1)*(isGrayscale()?1:getDatasetInfo()->is4ByteAligned()?4:3));
}

bool litiv::IDataProducer_<litiv::eDatasetSource_Image>::isPacketTransposed(size_t nPacketIdx) const {
    lvAssert(nPacketIdx<m_nImageCount);
    return m_vbImageTransposed[nPacketIdx];
}

const cv::Mat& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getPacketROI(size_t /*nPacketIdx*/) const {
    return m_oDefaultEmptyROI;
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getPacketSize(size_t nPacketIdx) const {
    lvAssert(nPacketIdx<m_nImageCount);
    return m_voImageSizes[nPacketIdx];
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getPacketOrigSize(size_t nPacketIdx) const {
    lvAssert(nPacketIdx<m_nImageCount);
    return m_voImageOrigSizes[nPacketIdx];
}

const cv::Size& litiv::IDataProducer_<litiv::eDatasetSource_Image>::getPacketMaxSize() const {
    return m_oMaxSize;
}

std::string litiv::IDataProducer_<litiv::eDatasetSource_Image>::getPacketName(size_t nPacketIdx) const {
    lvAssert(nPacketIdx<m_nImageCount);
    const size_t nLastSlashPos = m_vsInputImagePaths[nPacketIdx].find_last_of("/\\");
    std::string sFileName = (nLastSlashPos==std::string::npos)?m_vsInputImagePaths[nPacketIdx]:m_vsInputImagePaths[nPacketIdx].substr(nLastSlashPos+1);
    return sFileName.substr(0,sFileName.find_last_of("."));
}

litiv::IDataProducer_<litiv::eDatasetSource_Image>::IDataProducer_() :
        m_nImageCount(0) {}

size_t litiv::IDataProducer_<litiv::eDatasetSource_Image>::getTotPackets() const {
    return m_nImageCount;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Image>::_getInputPacket_impl(size_t nImageIdx) {
    cv::Mat oImage = cv::imread(m_vsInputImagePaths[nImageIdx],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    CV_Assert(getPacketOrigSize(nImageIdx)==oImage.size());
    if(isPacketTransposed(nImageIdx))
        cv::transpose(oImage,oImage);
    if(getDatasetInfo()->is4ByteAligned() && oImage.channels()==3)
        cv::cvtColor(oImage,oImage,cv::COLOR_BGR2BGRA);
    if(getPacketSize(nImageIdx).area()>0 && oImage.size()!=getPacketSize(nImageIdx))
        cv::resize(oImage,oImage,getPacketSize(nImageIdx),0,0,cv::INTER_NEAREST);
    return oImage;
}

cv::Mat litiv::IDataProducer_<litiv::eDatasetSource_Image>::_getGTPacket_impl(size_t nImageIdx) {
    if(m_mGTIndexLUT.count(nImageIdx)) {
        const size_t nGTIdx = m_mGTIndexLUT[nImageIdx];
        if(m_vsGTImagePaths.size()>nGTIdx) {
            cv::Mat oFrame = cv::imread(m_vsGTImagePaths[nGTIdx],cv::IMREAD_GRAYSCALE);
            if(!oFrame.empty()) {
                lvAssert(oFrame.size()==getPacketOrigSize(nImageIdx));
                if(isPacketTransposed(nImageIdx))
                    cv::transpose(oFrame,oFrame);
                if(getPacketSize(nImageIdx).area()>0 && oFrame.size()!=getPacketSize(nImageIdx))
                    cv::resize(oFrame,oFrame,getPacketSize(nImageIdx),0,0,cv::INTER_NEAREST);
                return oFrame;
            }
        }
    }
    return cv::Mat(getPacketSize(nImageIdx),CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUTOFSCOPE_VAL));
}

void litiv::IDataProducer_<litiv::eDatasetSource_Image>::parseData() {
    PlatformUtils::GetFilesFromDir(getDataPath(),m_vsInputImagePaths);
    PlatformUtils::FilterFilePaths(m_vsInputImagePaths,{},{".jpg",".png",".bmp"});
    if(m_vsInputImagePaths.empty())
        lvErrorExt("Set '%s' did not possess any jpg/png/bmp image file",getName().c_str());
    m_bIsConstantSize = true;
    m_oMaxSize = cv::Size(0,0);
    cv::Mat oLastInput;
    m_voImageSizes.clear();
    m_voImageOrigSizes.clear();
    m_vbImageTransposed.clear();
    m_voImageSizes.reserve(m_vsInputImagePaths.size());
    m_voImageOrigSizes.reserve(m_vsInputImagePaths.size());
    m_vbImageTransposed.reserve(m_vsInputImagePaths.size());
    const double dScale = getDatasetInfo()->getScaleFactor();
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
        m_voImageOrigSizes.push_back(oCurrInput.size());
        if(dScale!=1.0)
            cv::resize(oCurrInput,oCurrInput,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
        m_voImageSizes.push_back(oCurrInput.size());
        if(m_oMaxSize.width<oCurrInput.cols)
            m_oMaxSize.width = oCurrInput.cols;
        if(m_oMaxSize.height<oCurrInput.rows)
            m_oMaxSize.height = oCurrInput.rows;
        if(!oLastInput.empty() && oCurrInput.size()!=oLastInput.size())
            m_bIsConstantSize = false;
        oLastInput = oCurrInput;
        m_vbImageTransposed.push_back(false);
    }
    m_nImageCount = m_vsInputImagePaths.size();
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
    return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCountPromise();});
}

size_t litiv::DataCounter_<litiv::eGroup>::getProcessedPacketsCount() const {
    return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCount();});
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


void litiv::IDataArchiver::save(const cv::Mat& oClassif, size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << getPacketName(nIdx) << getDatasetInfo()->getOutputNameSuffix();
    if(shared_from_this_cast<const IIDataLoader>(true)->getPacketType()==eImagePacket) {
        auto pLoader = shared_from_this_cast<const IDataLoader_<eImagePacket>>(true);
        cv::Mat oOutput = oClassif;
        const cv::Mat& oROI = pLoader->getPacketROI(nIdx);
        if(!oROI.empty() && oROI.size()==oOutput.size())
            cv::bitwise_or(oOutput,DATASETUTILS_UNKNOWN_VAL,oOutput,oROI==0);
        if(pLoader->isPacketTransposed(nIdx))
            cv::transpose(oOutput,oOutput);
        if(pLoader->getPacketOrigSize(nIdx).area()>0 && oOutput.size()!=pLoader->getPacketOrigSize(nIdx))
            cv::resize(oOutput,oOutput,pLoader->getPacketOrigSize(nIdx),0,0,cv::INTER_NEAREST);
        const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
        cv::imwrite(sOutputFilePath.str(),oOutput,vnComprParams);
    }
    else {
        // @@@@ save to YML
        lvError("Missing impl");
    }
}

cv::Mat litiv::IDataArchiver::load(size_t nIdx) const {
    CV_Assert(!getDatasetInfo()->getOutputNameSuffix().empty());
    std::stringstream sOutputFilePath;
    sOutputFilePath << getOutputPath() << getDatasetInfo()->getOutputNamePrefix() << getPacketName(nIdx) << getDatasetInfo()->getOutputNameSuffix();
    if(shared_from_this_cast<const IIDataLoader>(true)->getPacketType()==eImagePacket) {
        auto pLoader = shared_from_this_cast<const IDataLoader_<eImagePacket>>(true);
        cv::Mat oOutput = cv::imread(sOutputFilePath.str(),isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
        if(pLoader->isPacketTransposed(nIdx))
            cv::transpose(oOutput,oOutput);
        if(getDatasetInfo()->is4ByteAligned() && oOutput.channels()==3)
            cv::cvtColor(oOutput,oOutput,cv::COLOR_BGR2BGRA);
        if(pLoader->getPacketSize(nIdx).area()>0 && oOutput.size()!=pLoader->getPacketSize(nIdx))
            cv::resize(oOutput,oOutput,pLoader->getPacketSize(nIdx),0,0,cv::INTER_NEAREST);
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
    cv::Size oWindowSize = shared_from_this_cast<const IDataLoader_<eImagePacket>>(true)->getPacketMaxSize();
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
    m_pLoader = shared_from_this_cast<IDataLoader_<eImagePacket>>(true);
    glAssert(m_pLoader->getTotPackets()>1);
    glDbgAssert(m_pAlgo);
    m_oNextInput = m_pLoader->getInput(m_nNextIdx).clone();
    m_oCurrInput = m_pLoader->getInput(m_nCurrIdx).clone();
    m_oLastInput = m_oCurrInput.clone();
    CV_Assert(!m_oCurrInput.empty());
    CV_Assert(m_oCurrInput.isContinuous());
    glAssert(m_oCurrInput.channels()==1 || m_oCurrInput.channels()==4);
    if(getDatasetInfo()->isSavingOutput() || m_pAlgo->m_pDisplayHelper)
        m_pAlgo->setOutputFetching(true);
    if(m_pAlgo->m_pDisplayHelper && m_pAlgo->m_bUsingDebug)
        m_pAlgo->setDebugFetching(true);
    if(getDatasetInfo()->isUsingEvaluator()) {
        m_oNextGT = m_pLoader->getGT(m_nNextIdx).clone();
        m_oCurrGT = m_pLoader->getGT(m_nCurrIdx).clone();
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
            m_lDataCallback(m_oLastInput,oLastDebug,oLastOutput,m_oLastGT,m_pLoader->getPacketROI(m_nLastIdx),m_nLastIdx);
        if(m_pAlgo->m_pDisplayHelper) {
            getColoredMasks(oLastOutput,oLastDebug,m_oLastGT,m_pLoader->getPacketROI(m_nLastIdx));
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
