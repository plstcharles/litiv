#include "litiv/utils/DatasetUtils.hpp"

#define PRECACHE_CONSOLE_DEBUG             0
#define PRECACHE_REQUEST_TIMEOUT_MS        1
#define PRECACHE_QUERY_TIMEOUT_MS          10
#define PRECACHE_MAX_CACHE_SIZE_GB         6L
#define PRECACHE_MAX_CACHE_SIZE            (((PRECACHE_MAX_CACHE_SIZE_GB*1024)*1024)*1024)
#if (!(defined(_M_X64) || defined(__amd64__)) && PRECACHE_MAX_CACHE_SIZE_GB>2)
#error "Cache max size exceeds system limit (x86)."
#endif //(!(defined(_M_X64) || defined(__amd64__)) && PRECACHE_MAX_CACHE_SIZE_GB>2)

void DatasetUtils::WriteOnImage(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom) {
    cv::putText(oImg,sText,cv::Point(4,bBottom?(oImg.rows-15):15),cv::FONT_HERSHEY_PLAIN,1.2,vColor,2,cv::LINE_AA);
}

void DatasetUtils::ValidateKeyPoints(const cv::Mat& oROI, std::vector<cv::KeyPoint>& voKPs) {
    std::vector<cv::KeyPoint> voNewKPs;
    for(size_t k=0; k<voKPs.size(); ++k) {
        if( voKPs[k].pt.x>=0 && voKPs[k].pt.x<oROI.cols &&
            voKPs[k].pt.y>=0 && voKPs[k].pt.y<oROI.rows &&
            oROI.at<uchar>(voKPs[k].pt)>0)
            voNewKPs.push_back(voKPs[k]);
    }
    voKPs = voNewKPs;
}

DatasetUtils::SegmEval::BasicMetrics::BasicMetrics()
    :   nTP(0),nTN(0),nFP(0),nFN(0),nSE(0),dTimeElapsed_sec(0) {}

DatasetUtils::SegmEval::BasicMetrics DatasetUtils::SegmEval::BasicMetrics::operator+(const BasicMetrics& m) const {
    BasicMetrics res(m);
    res.nTP += this->nTP;
    res.nTN += this->nTN;
    res.nFP += this->nFP;
    res.nFN += this->nFN;
    res.nSE += this->nSE;
    res.dTimeElapsed_sec += this->dTimeElapsed_sec;
    return res;
}

DatasetUtils::SegmEval::BasicMetrics& DatasetUtils::SegmEval::BasicMetrics::operator+=(const BasicMetrics& m) {
    this->nTP += m.nTP;
    this->nTN += m.nTN;
    this->nFP += m.nFP;
    this->nFN += m.nFN;
    this->nSE += m.nSE;
    this->dTimeElapsed_sec += m.dTimeElapsed_sec;
    return *this;
}

DatasetUtils::SegmEval::SegmMetrics::SegmMetrics(const DatasetUtils::SegmEval::BasicMetrics& m)
    :    dRecall(DatasetUtils::SegmEval::CalcRecall(m))
        ,dSpecificity(DatasetUtils::SegmEval::CalcSpecificity(m))
        ,dFPR(DatasetUtils::SegmEval::CalcFalsePositiveRate(m))
        ,dFNR(DatasetUtils::SegmEval::CalcFalseNegativeRate(m))
        ,dPBC(DatasetUtils::SegmEval::CalcPercentBadClassifs(m))
        ,dPrecision(DatasetUtils::SegmEval::CalcPrecision(m))
        ,dFMeasure(DatasetUtils::SegmEval::CalcFMeasure(m))
        ,dMCC(DatasetUtils::SegmEval::CalcMatthewsCorrCoeff(m))
        ,dTimeElapsed_sec(m.dTimeElapsed_sec)
        ,nWeight(1) {}

DatasetUtils::SegmEval::SegmMetrics DatasetUtils::SegmEval::SegmMetrics::operator+(const DatasetUtils::SegmEval::BasicMetrics& m) const {
    DatasetUtils::SegmEval::SegmMetrics tmp(m);
    return (*this)+tmp;
}

DatasetUtils::SegmEval::SegmMetrics& DatasetUtils::SegmEval::SegmMetrics::operator+=(const DatasetUtils::SegmEval::BasicMetrics& m) {
    DatasetUtils::SegmEval::SegmMetrics tmp(m);
    (*this) += tmp;
    return *this;
}

DatasetUtils::SegmEval::SegmMetrics DatasetUtils::SegmEval::SegmMetrics::operator+(const DatasetUtils::SegmEval::SegmMetrics& m) const {
    DatasetUtils::SegmEval::SegmMetrics res(m);
    const size_t nTotWeight = this->nWeight+res.nWeight;
    res.dRecall = (res.dRecall*res.nWeight + this->dRecall*this->nWeight)/nTotWeight;
    res.dSpecificity = (res.dSpecificity*res.nWeight + this->dSpecificity*this->nWeight)/nTotWeight;
    res.dFPR = (res.dFPR*res.nWeight + this->dFPR*this->nWeight)/nTotWeight;
    res.dFNR = (res.dFNR*res.nWeight + this->dFNR*this->nWeight)/nTotWeight;
    res.dPBC = (res.dPBC*res.nWeight + this->dPBC*this->nWeight)/nTotWeight;
    res.dPrecision = (res.dPrecision*res.nWeight + this->dPrecision*this->nWeight)/nTotWeight;
    res.dFMeasure = (res.dFMeasure*res.nWeight + this->dFMeasure*this->nWeight)/nTotWeight;
    res.dMCC = (res.dMCC*res.nWeight + this->dMCC*this->nWeight)/nTotWeight;
    res.dTimeElapsed_sec += this->dTimeElapsed_sec;
    res.nWeight = nTotWeight;
    return res;
}

DatasetUtils::SegmEval::SegmMetrics& DatasetUtils::SegmEval::SegmMetrics::operator+=(const DatasetUtils::SegmEval::SegmMetrics& m) {
    const size_t nTotWeight = this->nWeight+m.nWeight;
    this->dRecall = (m.dRecall*m.nWeight + this->dRecall*this->nWeight)/nTotWeight;
    this->dSpecificity = (m.dSpecificity*m.nWeight + this->dSpecificity*this->nWeight)/nTotWeight;
    this->dFPR = (m.dFPR*m.nWeight + this->dFPR*this->nWeight)/nTotWeight;
    this->dFNR = (m.dFNR*m.nWeight + this->dFNR*this->nWeight)/nTotWeight;
    this->dPBC = (m.dPBC*m.nWeight + this->dPBC*this->nWeight)/nTotWeight;
    this->dPrecision = (m.dPrecision*m.nWeight + this->dPrecision*this->nWeight)/nTotWeight;
    this->dFMeasure = (m.dFMeasure*m.nWeight + this->dFMeasure*this->nWeight)/nTotWeight;
    this->dMCC = (m.dMCC*m.nWeight + this->dMCC*this->nWeight)/nTotWeight;
    this->dTimeElapsed_sec += m.dTimeElapsed_sec;
    this->nWeight = nTotWeight;
    return *this;
}

double DatasetUtils::SegmEval::CalcFMeasure(const DatasetUtils::SegmEval::BasicMetrics& m) {
    const double dRecall = DatasetUtils::SegmEval::CalcRecall(m);
    const double dPrecision = DatasetUtils::SegmEval::CalcPrecision(m);
    return (2.0*(dRecall*dPrecision)/(dRecall+dPrecision));
}
double DatasetUtils::SegmEval::CalcRecall(const DatasetUtils::SegmEval::BasicMetrics& m) {return ((double)m.nTP/(m.nTP+m.nFN));}
double DatasetUtils::SegmEval::CalcPrecision(const DatasetUtils::SegmEval::BasicMetrics& m) {return ((double)m.nTP/(m.nTP+m.nFP));}
double DatasetUtils::SegmEval::CalcSpecificity(const DatasetUtils::SegmEval::BasicMetrics& m) {return ((double)m.nTN/(m.nTN+m.nFP));}
double DatasetUtils::SegmEval::CalcFalsePositiveRate(const DatasetUtils::SegmEval::BasicMetrics& m) {return ((double)m.nFP/(m.nFP+m.nTN));}
double DatasetUtils::SegmEval::CalcFalseNegativeRate(const DatasetUtils::SegmEval::BasicMetrics& m) {return ((double)m.nFN/(m.nTP+m.nFN));}
double DatasetUtils::SegmEval::CalcPercentBadClassifs(const DatasetUtils::SegmEval::BasicMetrics& m) {return (100.0*(m.nFN+m.nFP)/(m.nTP+m.nFP+m.nFN+m.nTN));}
double DatasetUtils::SegmEval::CalcMatthewsCorrCoeff(const DatasetUtils::SegmEval::BasicMetrics& m) {return ((((double)m.nTP*m.nTN)-(m.nFP*m.nFN))/sqrt(((double)m.nTP+m.nFP)*(m.nTP+m.nFN)*(m.nTN+m.nFP)*(m.nTN+m.nFN)));}

DatasetUtils::ImagePrecacher::ImagePrecacher(ImageQueryByIndexFunc pCallback) {
    CV_Assert(pCallback);
    m_pCallback = pCallback;
    m_bIsPrecaching = false;
    m_nLastReqIdx = size_t(-1);
}

DatasetUtils::ImagePrecacher::~ImagePrecacher() {
    StopPrecaching();
}

const cv::Mat& DatasetUtils::ImagePrecacher::GetImageFromIndex(size_t nIdx) {
    if(!m_bIsPrecaching)
        return GetImageFromIndex_internal(nIdx);
    CV_Assert(nIdx<m_nTotImageCount);
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
    return m_oReqImage;
}

bool DatasetUtils::ImagePrecacher::StartPrecaching(size_t nTotImageCount, size_t nSuggestedBufferSize) {
    static_assert(PRECACHE_REQUEST_TIMEOUT_MS>0,"Precache request timeout must be a positive value");
    static_assert(PRECACHE_QUERY_TIMEOUT_MS>0,"Precache query timeout must be a positive value");
    static_assert(PRECACHE_MAX_CACHE_SIZE>=0,"Precache size must be a non-negative value");
    CV_Assert(nTotImageCount);
    m_nTotImageCount = nTotImageCount;
    if(m_bIsPrecaching)
        StopPrecaching();
    if(nSuggestedBufferSize>0) {
        m_bIsPrecaching = true;
        m_nBufferSize = (nSuggestedBufferSize>PRECACHE_MAX_CACHE_SIZE)?(PRECACHE_MAX_CACHE_SIZE):nSuggestedBufferSize;
        m_qoCache.clear();
        m_vcBuffer.resize(m_nBufferSize);
        m_nNextExpectedReqIdx = 0;
        m_nNextPrecacheIdx = 0;
        m_nReqIdx = m_nLastReqIdx = size_t(-1);
        m_hPrecacher = std::thread(&DatasetUtils::ImagePrecacher::Precache,this);
    }
    return m_bIsPrecaching;
}

void DatasetUtils::ImagePrecacher::StopPrecaching() {
    if(m_bIsPrecaching) {
        m_bIsPrecaching = false;
        m_hPrecacher.join();
    }
}

void DatasetUtils::ImagePrecacher::Precache() {
    std::unique_lock<std::mutex> sync_lock(m_oSyncMutex);
#if PRECACHE_CONSOLE_DEBUG
    std::cout << " @ initializing precaching with buffer size = " << (m_nBufferSize/1024)/1024 << " mb" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
    m_nFirstBufferIdx = m_nNextBufferIdx = 0;
    while(m_nNextPrecacheIdx<m_nTotImageCount) {
        const cv::Mat& oNextImage = GetImageFromIndex_internal(m_nNextPrecacheIdx);
        const size_t nNextImageSize = oNextImage.total()*oNextImage.elemSize();
        if(m_nNextBufferIdx+nNextImageSize<m_nBufferSize) {
            cv::Mat oNextImage_cache(oNextImage.size(),oNextImage.type(),m_vcBuffer.data()+m_nNextBufferIdx);
            oNextImage.copyTo(oNextImage_cache);
            m_qoCache.push_back(oNextImage_cache);
            m_nNextBufferIdx += nNextImageSize;
            ++m_nNextPrecacheIdx;
        }
        else break;
    }
    while(m_bIsPrecaching) {
        if(m_oReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(m_nNextPrecacheIdx==m_nTotImageCount?PRECACHE_QUERY_TIMEOUT_MS*32:PRECACHE_QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
            if(m_nReqIdx!=m_nNextExpectedReqIdx-1) {
                if(!m_qoCache.empty()) {
                    if(m_nReqIdx<m_nNextPrecacheIdx && m_nReqIdx>=m_nNextExpectedReqIdx) {
//#if PRECACHE_CONSOLE_DEBUG
//                        std::cout << " -- popping " << m_nReqIdx-m_nNextExpectedReqIdx+1 << " image(s) from cache" << std::endl;
//#endif //PRECACHE_CONSOLE_DEBUG
                        while(m_nReqIdx-m_nNextExpectedReqIdx+1>0) {
                            m_oReqImage = m_qoCache.front();
                            m_nFirstBufferIdx = (size_t)(m_oReqImage.data-m_vcBuffer.data());
                            m_qoCache.pop_front();
                            ++m_nNextExpectedReqIdx;
                        }
                    }
                    else {
#if PRECACHE_CONSOLE_DEBUG
                        std::cout << " -- out-of-order request, destroying cache" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
                        m_qoCache.clear();
                        m_oReqImage = GetImageFromIndex_internal(m_nReqIdx);
                        m_nFirstBufferIdx = m_nNextBufferIdx = size_t(-1);
                        m_nNextExpectedReqIdx = m_nNextPrecacheIdx = m_nReqIdx+1;
                    }
                }
                else {
#if PRECACHE_CONSOLE_DEBUG
                    std::cout << " @ answering request manually, precaching is falling behind" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
                    m_oReqImage = GetImageFromIndex_internal(m_nReqIdx);
                    m_nFirstBufferIdx = m_nNextBufferIdx = size_t(-1);
                    m_nNextExpectedReqIdx = m_nNextPrecacheIdx = m_nReqIdx+1;
                }
            }
#if PRECACHE_CONSOLE_DEBUG
            else
                std::cout << " @ answering request using last image" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
            m_oSyncCondVar.notify_one();
        }
        else {
            size_t nUsedBufferSize = m_nFirstBufferIdx==size_t(-1)?0:(m_nFirstBufferIdx<m_nNextBufferIdx?m_nNextBufferIdx-m_nFirstBufferIdx:m_nBufferSize-m_nFirstBufferIdx+m_nNextBufferIdx);
            if(nUsedBufferSize<m_nBufferSize/4 && m_nNextPrecacheIdx<m_nTotImageCount) {
#if PRECACHE_CONSOLE_DEBUG
                std::cout << " @ filling precache buffer... (current size = " << (nUsedBufferSize/1024)/1024 << " mb)" << std::endl;
#endif //PRECACHE_CONSOLE_DEBUG
                size_t nFillCount = 0;
                while(nUsedBufferSize<m_nBufferSize && m_nNextPrecacheIdx<m_nTotImageCount && nFillCount<10) {
                    const cv::Mat& oNextImage = GetImageFromIndex_internal(m_nNextPrecacheIdx);
                    const size_t nNextImageSize = (oNextImage.total()*oNextImage.elemSize());
                    if(m_nFirstBufferIdx<=m_nNextBufferIdx) {
                        if(m_nNextBufferIdx==size_t(-1) || (m_nNextBufferIdx+nNextImageSize>=m_nBufferSize)) {
                            if((m_nFirstBufferIdx!=size_t(-1) && nNextImageSize>=m_nFirstBufferIdx) || nNextImageSize>=m_nBufferSize)
                                break;
                            cv::Mat oNextImage_cache(oNextImage.size(),oNextImage.type(),m_vcBuffer.data());
                            oNextImage.copyTo(oNextImage_cache);
                            m_qoCache.push_back(oNextImage_cache);
                            m_nNextBufferIdx = nNextImageSize;
                            if(m_nFirstBufferIdx==size_t(-1))
                                m_nFirstBufferIdx = 0;
                        }
                        else { // m_nNextBufferIdx+nNextImageSize<m_nBufferSize
                            cv::Mat oNextImage_cache(oNextImage.size(),oNextImage.type(),m_vcBuffer.data()+m_nNextBufferIdx);
                            oNextImage.copyTo(oNextImage_cache);
                            m_qoCache.push_back(oNextImage_cache);
                            m_nNextBufferIdx += nNextImageSize;
                        }
                    }
                    else if(m_nNextBufferIdx+nNextImageSize<m_nFirstBufferIdx) {
                        cv::Mat oNextImage_cache(oNextImage.size(),oNextImage.type(),m_vcBuffer.data()+m_nNextBufferIdx);
                        oNextImage.copyTo(oNextImage_cache);
                        m_qoCache.push_back(oNextImage_cache);
                        m_nNextBufferIdx += nNextImageSize;
                    }
                    else // m_nNextBufferIdx+nNextImageSize>=m_nFirstBufferIdx
                        break;
                    nUsedBufferSize += nNextImageSize;
                    ++m_nNextPrecacheIdx;
                }
            }
        }
    }
}

const cv::Mat& DatasetUtils::ImagePrecacher::GetImageFromIndex_internal(size_t nIdx) {
    if(m_nLastReqIdx!=nIdx) {
        m_oLastReqImage = m_pCallback(nIdx);
        m_nLastReqIdx = nIdx;
    }
    return m_oLastReqImage;
}

DatasetUtils::WorkBatch::WorkBatch(const std::string& sName, const std::string& sPath, bool bForceGrayscale, bool bUse4chAlign)
    :    m_sName(sName)
        ,m_sPath(sPath)
        ,m_nIMReadInputFlags(bForceGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR)
        ,m_bForcingGrayscale(bForceGrayscale)
        ,m_bUsing4chAlignment(bUse4chAlign)
        ,m_oInputPrecacher(std::bind(&DatasetUtils::WorkBatch::GetInputFromIndex_internal,this,std::placeholders::_1))
        ,m_oGTPrecacher(std::bind(&DatasetUtils::WorkBatch::GetGTFromIndex_internal,this,std::placeholders::_1)) {}

bool DatasetUtils::WorkBatch::StartPrecaching(size_t nSuggestedBufferSize) {
    return m_oInputPrecacher.StartPrecaching(GetTotalImageCount(),nSuggestedBufferSize) &&
           m_oGTPrecacher.StartPrecaching(GetTotalImageCount(),nSuggestedBufferSize);
}

void DatasetUtils::WorkBatch::StopPrecaching() {
    m_oInputPrecacher.StopPrecaching();
    m_oGTPrecacher.StopPrecaching();
}

const cv::Mat& DatasetUtils::WorkBatch::GetInputFromIndex_internal(size_t nIdx) {
    CV_Assert(nIdx<GetTotalImageCount());
    m_oLatestInputImage = GetInputFromIndex_external(nIdx);
    return m_oLatestInputImage;
}

const cv::Mat& DatasetUtils::WorkBatch::GetGTFromIndex_internal(size_t nIdx) {
    CV_Assert(nIdx<GetTotalImageCount());
    m_oLatestGTMask = GetGTFromIndex_external(nIdx);
    return m_oLatestGTMask;
}

DatasetUtils::VideoSegm::DatasetInfo DatasetUtils::VideoSegm::GetDatasetInfo(const DatasetUtils::VideoSegm::eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirPath) {
    if(eDatasetID==DatasetUtils::VideoSegm::eDataset_CDnet2012)
        return DatasetUtils::VideoSegm::DatasetInfo {
            eDatasetID,
            sDatasetRootDirPath+"/CDNet/dataset/",
            sDatasetRootDirPath+"/CDNet/"+sResultsDirPath+"/",
            "bin",
            ".png",
            {"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"},
            {"thermal"},
            {},
            1,
        };
    else if(eDatasetID==DatasetUtils::VideoSegm::eDataset_CDnet2014)
        return DatasetUtils::VideoSegm::DatasetInfo {
            eDatasetID,
            sDatasetRootDirPath+"/CDNet2014/dataset/",
            sDatasetRootDirPath+"/CDNet2014/"+sResultsDirPath+"/",
            "bin",
            ".png",
            {"baseline_highway"},//{"shadow_cubicle"},//{"dynamicBackground_fall"},//{"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"},
            {"thermal","turbulence"},//{"baseline_highway"},//
            {},
            1,
        };
    else if(eDatasetID==DatasetUtils::VideoSegm::eDataset_Wallflower)
        return DatasetUtils::VideoSegm::DatasetInfo {
            eDatasetID,
            sDatasetRootDirPath+"/Wallflower/dataset/",
            sDatasetRootDirPath+"/Wallflower/"+sResultsDirPath+"/",
            "bin",
            ".png",
            {"global"},
            {},
            {},
            0,
        };
    else if(eDatasetID==DatasetUtils::VideoSegm::eDataset_PETS2001_D3TC1)
        return DatasetUtils::VideoSegm::DatasetInfo {
            eDatasetID,
            sDatasetRootDirPath+"/PETS2001/DATASET3/",
            sDatasetRootDirPath+"/PETS2001/DATASET3/"+sResultsDirPath+"/",
            "bin",
            ".png",
            {"TESTING"},
            {},
            {},
            0,
        };
    else if(eDatasetID==DatasetUtils::VideoSegm::eDataset_LITIV2012)
        return DatasetUtils::VideoSegm::DatasetInfo {
                eDatasetID,
                sDatasetRootDirPath+"/litiv/litiv2012_dataset/",
                sDatasetRootDirPath+"/litiv/litiv2012_dataset/"+sResultsDirPath+"/",
                "bin",
                ".png",
                {"SEQUENCE1","SEQUENCE2","SEQUENCE3","SEQUENCE4","SEQUENCE5","SEQUENCE6","SEQUENCE7","SEQUENCE8","SEQUENCE9"},//{"vid1","vid2/cut1","vid2/cut2","vid3"},
                {"THERMAL"},
                {},//{"1Person","2Person","3Person","4Person","5Person"},
                0,
        };
    else if(eDatasetID==DatasetUtils::VideoSegm::eDataset_GenericTest)
        // @@@@@ remove from this func, set only in main where/when needed?
        return DatasetUtils::VideoSegm::DatasetInfo {
            eDatasetID,
            sDatasetRootDirPath+"/avitest/",                        // HARDCODED
            sDatasetRootDirPath+"/avitest/"+sResultsDirPath+"/",    // HARDCODED
            "",                                                     // HARDCODED
            ".png",                                                 // HARDCODED
            {"inf6803_tp1"},                                        // HARDCODED
            {},                                                     // HARDCODED
            {},                                                     // HARDCODED
            0,                                                      // HARDCODED
        };
    else
        throw std::runtime_error(std::string("Unknown dataset type, cannot use predefined info struct"));
}

cv::Mat DatasetUtils::VideoSegm::ReadResult( const std::string& sResultsPath, const std::string& sCatName, const std::string& sSeqName,
                                             const std::string& sResultPrefix, size_t nFrameIdx, const std::string& sResultSuffix, int nFlags) {
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nFrameIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << acBuffer.data() << sResultSuffix;
    return cv::imread(sResultFilePath.str(),nFlags);
}

void DatasetUtils::VideoSegm::WriteResult( const std::string& sResultsPath, const std::string& sCatName, const std::string& sSeqName, const std::string& sResultPrefix,
                                           size_t nFrameIdx, const std::string& sResultSuffix, const cv::Mat& oResult, const std::vector<int>& vnComprParams) {
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nFrameIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << acBuffer.data() << sResultSuffix;
    cv::imwrite(sResultFilePath.str(),oResult,vnComprParams);
}

DatasetUtils::VideoSegm::CategoryInfo::CategoryInfo(const std::string& sName, const std::string& sDirectoryPath, DatasetUtils::VideoSegm::eDatasetList eDatasetID,
                                                    const std::vector<std::string>& vsGrayscaleNameTokens, const std::vector<std::string>& vsSkippedNameTokens,
                                                    bool bUse4chAlign)
    :    DatasetUtils::WorkBatch(sName,sDirectoryPath,PlatformUtils::string_contains_token(sName,vsGrayscaleNameTokens),bUse4chAlign)
        ,m_eDatasetID(eDatasetID)
        ,m_dExpectedLoad(0)
        ,m_nTotFrameCount(0) {
    std::cout<<"\tParsing dir '"<<sDirectoryPath<<"' for category '"<<m_sName<<"'; ";
    std::vector<std::string> vsSequencePaths;
    if(m_eDatasetID == DatasetUtils::VideoSegm::eDataset_CDnet2012 ||
       m_eDatasetID == DatasetUtils::VideoSegm::eDataset_CDnet2014 ||
       m_eDatasetID == DatasetUtils::VideoSegm::eDataset_Wallflower ||
       m_eDatasetID == DatasetUtils::VideoSegm::eDataset_PETS2001_D3TC1) {
        // all subdirs are considered sequence directories
        PlatformUtils::GetSubDirsFromDir(sDirectoryPath,vsSequencePaths);
        std::cout<<vsSequencePaths.size()<<" potential sequence(s)"<<std::endl;
    }
    else if(m_eDatasetID == DatasetUtils::VideoSegm::eDataset_LITIV2012) {
        // all subdirs should contain individual video tracks in separate modalities
        PlatformUtils::GetSubDirsFromDir(sDirectoryPath,vsSequencePaths);
        std::cout<<vsSequencePaths.size()<<" potential track(s)"<<std::endl;
    }
    else if(m_eDatasetID == DatasetUtils::VideoSegm::eDataset_GenericTest) {
        // all files are considered sequences
        PlatformUtils::GetFilesFromDir(sDirectoryPath,vsSequencePaths);
        std::cout<<vsSequencePaths.size()<<" potential sequence(s)"<<std::endl;
    }
    else
        throw std::runtime_error(std::string("Unknown dataset type, cannot use any known parsing strategy"));
    if(!PlatformUtils::string_contains_token(sName,vsSkippedNameTokens)) {
        for(auto oSeqPathIter=vsSequencePaths.begin(); oSeqPathIter!=vsSequencePaths.end(); ++oSeqPathIter) {
            const size_t idx = oSeqPathIter->find_last_of("/\\");
            const std::string sSeqName = idx==std::string::npos?*oSeqPathIter:oSeqPathIter->substr(idx+1);
            if(!PlatformUtils::string_contains_token(sSeqName,vsSkippedNameTokens)) {
                bool bForceGrayscale = m_bForcingGrayscale || PlatformUtils::string_contains_token(sSeqName,vsGrayscaleNameTokens);
                m_vpSequences.push_back(std::make_shared<SequenceInfo>(sSeqName,m_sName,*oSeqPathIter,m_eDatasetID,bForceGrayscale,bUse4chAlign));
                m_dExpectedLoad += m_vpSequences.back()->GetExpectedLoad();
                m_nTotFrameCount += m_vpSequences.back()->GetTotalImageCount();
            }
        }
    }
}

DatasetUtils::SegmEval::SegmMetrics DatasetUtils::VideoSegm::CategoryInfo::CalcMetricsFromCategory(const DatasetUtils::VideoSegm::CategoryInfo& oCat, bool bAverage) {
    if(!bAverage) {
        DatasetUtils::SegmEval::BasicMetrics oCumulBasicMetrics;
        for(auto oSeqIter=oCat.m_vpSequences.begin(); oSeqIter!=oCat.m_vpSequences.end(); ++oSeqIter)
            oCumulBasicMetrics += (*oSeqIter)->m_oMetrics;
        return DatasetUtils::SegmEval::SegmMetrics(oCumulBasicMetrics);
    }
    else {
        CV_Assert(!oCat.m_vpSequences.empty());
        DatasetUtils::SegmEval::SegmMetrics tmp(oCat.m_vpSequences[0]->m_oMetrics);
        for(auto oSeqIter=oCat.m_vpSequences.begin()+1; oSeqIter!=oCat.m_vpSequences.end(); ++oSeqIter)
            tmp += (*oSeqIter)->m_oMetrics;
        return tmp;
    }
}

DatasetUtils::SegmEval::SegmMetrics DatasetUtils::VideoSegm::CategoryInfo::CalcMetricsFromCategories(const std::vector<std::shared_ptr<DatasetUtils::VideoSegm::CategoryInfo>>& vpCat, bool bAverage) {
    if(!bAverage) {
        DatasetUtils::SegmEval::BasicMetrics oCumulBasicMetrics;
        for(auto oCatIter=vpCat.begin(); oCatIter!=vpCat.end(); ++oCatIter)
            for(auto oSeqIter=(*oCatIter)->m_vpSequences.begin(); oSeqIter!=(*oCatIter)->m_vpSequences.end(); ++oSeqIter)
                oCumulBasicMetrics += (*oSeqIter)->m_oMetrics;
        return DatasetUtils::SegmEval::SegmMetrics(oCumulBasicMetrics);
    }
    else {
        CV_Assert(!vpCat.empty());
        DatasetUtils::SegmEval::SegmMetrics res = DatasetUtils::VideoSegm::CategoryInfo::CalcMetricsFromCategory(*(vpCat[0]),bAverage);
        res.nWeight = 1;
        for(auto oCatIter=vpCat.begin()+1; oCatIter!=vpCat.end(); ++oCatIter) {
            if(!(*oCatIter)->m_vpSequences.empty()) {
                DatasetUtils::SegmEval::SegmMetrics tmp = DatasetUtils::VideoSegm::CategoryInfo::CalcMetricsFromCategory(**oCatIter,bAverage);
                tmp.nWeight = 1;
                res += tmp;
            }
        }
        return res;
    }
}

void DatasetUtils::VideoSegm::CategoryInfo::WriteMetrics(const std::string& sResultsFilePath, const DatasetUtils::VideoSegm::CategoryInfo& oCat) {
    std::ofstream oMetricsOutput(sResultsFilePath);
    oMetricsOutput << "Results for category '" << oCat.m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Sequence Metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    for(auto oSeqIter=oCat.m_vpSequences.begin(); oSeqIter!=oCat.m_vpSequences.end(); ++oSeqIter) {
        DatasetUtils::SegmEval::SegmMetrics tmp((*oSeqIter)->m_oMetrics);
        std::string sName = (*oSeqIter)->m_sName;
        if(sName.size()>10)
            sName = sName.substr(0,10);
        else if(sName.size()<10)
            sName += std::string(10-sName.size(),' ');
        oMetricsOutput << sName << " " << tmp.dRecall << " " << tmp.dSpecificity << " " << tmp.dFPR << " " << tmp.dFNR << " " << tmp.dPBC << " " << tmp.dPrecision << " " << tmp.dFMeasure << " " << tmp.dMCC << std::endl;
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    DatasetUtils::SegmEval::SegmMetrics all(DatasetUtils::VideoSegm::CategoryInfo::CalcMetricsFromCategory(oCat,DATASETUTILS_USE_AVERAGE_EVAL_METRICS));
    const std::string sCurrCatName = oCat.m_sName.size()>12?oCat.m_sName.substr(0,12):oCat.m_sName;
    std::cout << "\t" << std::setfill(' ') << std::setw(12) << sCurrCatName << " : Rcl=" << std::fixed << std::setprecision(4) << all.dRecall << " Prc=" << all.dPrecision << " FM=" << all.dFMeasure << " MCC=" << all.dMCC << std::endl;
    oMetricsOutput << std::string(DATASETUTILS_USE_AVERAGE_EVAL_METRICS?"averaged   ":"cumulative ") << all.dRecall << " " << all.dSpecificity << " " << all.dFPR << " " << all.dFNR << " " << all.dPBC << " " << all.dPrecision << " " << all.dFMeasure << " " << all.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Category FPS: " << all.dTimeElapsed_sec/oCat.GetTotalImageCount() << std::endl;
    oMetricsOutput.close();
}

void DatasetUtils::VideoSegm::CategoryInfo::WriteMetrics(const std::string& sResultsFilePath, const std::vector<std::shared_ptr<DatasetUtils::VideoSegm::CategoryInfo>>& vpCat) {
    std::ofstream oMetricsOutput(sResultsFilePath);
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Overall results :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << std::string(DATASETUTILS_USE_AVERAGE_EVAL_METRICS?"Averaged":"Cumulative") << " category metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    size_t nOverallFrameCount = 0;
    for(auto oCatIter=vpCat.begin(); oCatIter!=vpCat.end(); ++oCatIter) {
        if(!(*oCatIter)->m_vpSequences.empty()) {
            DatasetUtils::SegmEval::SegmMetrics tmp(DatasetUtils::VideoSegm::CategoryInfo::CalcMetricsFromCategory(**oCatIter,DATASETUTILS_USE_AVERAGE_EVAL_METRICS));
            std::string sName = (*oCatIter)->m_sName;
            if(sName.size()>10)
                sName = sName.substr(0,10);
            else if(sName.size()<10)
                sName += std::string(10-sName.size(),' ');
            oMetricsOutput << sName << " " << tmp.dRecall << " " << tmp.dSpecificity << " " << tmp.dFPR << " " << tmp.dFNR << " " << tmp.dPBC << " " << tmp.dPrecision << " " << tmp.dFMeasure << " " << tmp.dMCC << std::endl;
            nOverallFrameCount += (*oCatIter)->GetTotalImageCount();
        }
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    DatasetUtils::SegmEval::SegmMetrics all(DatasetUtils::VideoSegm::CategoryInfo::CalcMetricsFromCategories(vpCat,DATASETUTILS_USE_AVERAGE_EVAL_METRICS));
    oMetricsOutput << "overall    " << all.dRecall << " " << all.dSpecificity << " " << all.dFPR << " " << all.dFNR << " " << all.dPBC << " " << all.dPrecision << " " << all.dFMeasure << " " << all.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Overall FPS: " << all.dTimeElapsed_sec/nOverallFrameCount << std::endl;
    oMetricsOutput.close();
}

cv::Mat DatasetUtils::VideoSegm::CategoryInfo::GetInputFromIndex_external(size_t nFrameIdx) {
    size_t nCumulFrameIdx = 0;
    size_t nSeqIdx = 0;
    do {
        nCumulFrameIdx += m_vpSequences[nSeqIdx++]->GetTotalImageCount();
    } while(nCumulFrameIdx<nFrameIdx);
    return m_vpSequences[nSeqIdx-1]->GetInputFromIndex_external(nFrameIdx-(nCumulFrameIdx-m_vpSequences[nSeqIdx-1]->GetTotalImageCount()));
}

cv::Mat DatasetUtils::VideoSegm::CategoryInfo::GetGTFromIndex_external(size_t nFrameIdx) {
    size_t nCumulFrameIdx = 0;
    size_t nSeqIdx = 0;
    do {
        nCumulFrameIdx += m_vpSequences[nSeqIdx++]->GetTotalImageCount();
    } while(nCumulFrameIdx<nFrameIdx);
    return m_vpSequences[nSeqIdx-1]->GetGTFromIndex_external(nFrameIdx-(nCumulFrameIdx-m_vpSequences[nSeqIdx-1]->GetTotalImageCount()));
}

DatasetUtils::VideoSegm::SequenceInfo::SequenceInfo(const std::string& sName, const std::string& sParentName, const std::string& sPath, DatasetUtils::VideoSegm::eDatasetList eDatasetID, bool bForceGrayscale, bool bUse4chAlign)
    :    WorkBatch(sName,sPath,bForceGrayscale,bUse4chAlign)
        ,m_eDatasetID(eDatasetID)
        ,m_sParentName(sParentName)
        ,m_dExpectedLoad(0)
        ,m_nTotFrameCount(0)
        ,m_nNextExpectedVideoReaderFrameIdx(0) {
    if(m_eDatasetID==DatasetUtils::VideoSegm::eDataset_CDnet2012 || m_eDatasetID==DatasetUtils::VideoSegm::eDataset_CDnet2014) {
        std::vector<std::string> vsSubDirs;
        PlatformUtils::GetSubDirsFromDir(m_sPath,vsSubDirs);
        auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),m_sPath+"/groundtruth");
        auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),m_sPath+"/input");
        if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess the required groundtruth and input directories");
        PlatformUtils::GetFilesFromDir(*inputDir,m_vsInputFramePaths);
        PlatformUtils::GetFilesFromDir(*gtDir,m_vsGTFramePaths);
        if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess same amount of GT & input frames");
        m_oROI = cv::imread(m_sPath+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
        if(m_oROI.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess a ROI.bmp file");
        m_oROI = m_oROI>0;
        m_oSize = m_oROI.size();
        m_nTotFrameCount = m_vsInputFramePaths.size();
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)cv::countNonZero(m_oROI)*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
        // note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
    }
    else if(m_eDatasetID==DatasetUtils::VideoSegm::eDataset_Wallflower) {
        std::vector<std::string> vsImgPaths;
        PlatformUtils::GetFilesFromDir(m_sPath,vsImgPaths);
        bool bFoundScript=false, bFoundGTFile=false;
        const std::string sGTFilePrefix("hand_segmented_");
        const size_t nInputFileNbDecimals = 5;
        const std::string sInputFileSuffix(".bmp");
        for(auto iter=vsImgPaths.begin(); iter!=vsImgPaths.end(); ++iter) {
            if(*iter==m_sPath+"/script.txt")
                bFoundScript = true;
            else if(iter->find(sGTFilePrefix)!=std::string::npos) {
                m_mTestGTIndexes.insert(std::pair<size_t,size_t>(atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),m_vsGTFramePaths.size()));
                m_vsGTFramePaths.push_back(*iter);
                bFoundGTFile = true;
            }
            else {
                if(iter->find(sInputFileSuffix)!=iter->size()-sInputFileSuffix.size())
                    throw std::runtime_error(std::string("Sequence directory at ") + m_sPath + " contained an unknown file ('" + *iter + "')");
                m_vsInputFramePaths.push_back(*iter);
            }
        }
        if(!bFoundGTFile || !bFoundScript || m_vsInputFramePaths.empty() || m_vsGTFramePaths.size()!=1)
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess the required groundtruth and input files");
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess a valid GT file");
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(255));
        m_oSize = oTempImg.size();
        m_nTotFrameCount = m_vsInputFramePaths.size();
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
    }
    else if(m_eDatasetID==DatasetUtils::VideoSegm::eDataset_PETS2001_D3TC1) {
        std::vector<std::string> vsVideoSeqPaths;
        PlatformUtils::GetFilesFromDir(m_sPath,vsVideoSeqPaths);
        if(vsVideoSeqPaths.size()!=1)
            throw std::runtime_error(std::string("Bad subdirectory ('")+m_sPath+std::string("') for PETS2001 parsing (should contain only one video sequence file)"));
        std::vector<std::string> vsGTSubdirPaths;
        PlatformUtils::GetSubDirsFromDir(m_sPath,vsGTSubdirPaths);
        if(vsGTSubdirPaths.size()!=1)
            throw std::runtime_error(std::string("Bad subdirectory ('")+m_sPath+std::string("') for PETS2001 parsing (should contain only one GT subdir)"));
        m_voVideoReader.open(vsVideoSeqPaths[0]);
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(std::string("Bad video file ('")+vsVideoSeqPaths[0]+std::string("'), could not be opened"));
        PlatformUtils::GetFilesFromDir(vsGTSubdirPaths[0],m_vsGTFramePaths);
        if(m_vsGTFramePaths.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess any valid GT frames");
        const std::string sGTFilePrefix("image_");
        const size_t nInputFileNbDecimals = 4;
        for(auto iter=m_vsGTFramePaths.begin(); iter!=m_vsGTFramePaths.end(); ++iter)
            m_mTestGTIndexes.insert(std::pair<size_t,size_t>((size_t)atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),iter-m_vsGTFramePaths.begin()));
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess valid GT file(s)");
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(255));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        m_nTotFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
    }
    else if(m_eDatasetID==DatasetUtils::VideoSegm::eDataset_LITIV2012) {
        PlatformUtils::GetFilesFromDir(m_sPath+"/input/",m_vsInputFramePaths);
        if(m_vsInputFramePaths.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess any parsable input images");
        cv::Mat oTempImg = cv::imread(m_vsInputFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Bad image file ('")+m_vsInputFramePaths[0]+"'), could not be read");
        /*m_voVideoReader.open(m_sPath+"/input/in%06d.jpg");
        if(!m_voVideoReader.isOpened())
            m_voVideoReader.open(m_sPath+"/"+m_sName+".avi");
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be opened"));
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        cv::Mat oTempImg;
        m_voVideoReader >> oTempImg;
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be read"));*/
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(255));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = (size_t)-1;
        //m_nTotFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        m_nTotFrameCount = m_vsInputFramePaths.size();
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
    }
    else if(m_eDatasetID==DatasetUtils::VideoSegm::eDataset_GenericTest) {
        m_voVideoReader.open(m_sPath);
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be opened"));
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        cv::Mat oTempImg;
        m_voVideoReader >> oTempImg;
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be read"));
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(255));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        m_nTotFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
    }
    else
        throw std::runtime_error(std::string("Unknown dataset type, cannot use any known parsing strategy"));
}

void DatasetUtils::VideoSegm::SequenceInfo::WriteMetrics(const std::string& sResultsFilePath, const DatasetUtils::VideoSegm::SequenceInfo& oSeq) {
    std::ofstream oMetricsOutput(sResultsFilePath);
    DatasetUtils::SegmEval::SegmMetrics tmp(oSeq.m_oMetrics);
    const std::string sCurrSeqName = oSeq.m_sName.size()>12?oSeq.m_sName.substr(0,12):oSeq.m_sName;
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " : Rcl=" << std::fixed << std::setprecision(4) << tmp.dRecall << " Prc=" << tmp.dPrecision << " FM=" << tmp.dFMeasure << " MCC=" << tmp.dMCC << std::endl;
    oMetricsOutput << "Results for sequence '" << oSeq.m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << "nTP nFP nFN nTN nSE nTot" << std::endl; // order similar to the files saved by the CDNet analysis script
    oMetricsOutput << oSeq.m_oMetrics.nTP << " " << oSeq.m_oMetrics.nFP << " " << oSeq.m_oMetrics.nFN << " " << oSeq.m_oMetrics.nTN << " " << oSeq.m_oMetrics.nSE << " " << oSeq.m_oMetrics.total() << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Cumulative metrics :" << std::endl;
    oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    oMetricsOutput << tmp.dRecall << " " << tmp.dSpecificity << " " << tmp.dFPR << " " << tmp.dFNR << " " << tmp.dPBC << " " << tmp.dPrecision << " " << tmp.dFMeasure << " " << tmp.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Sequence FPS: " << oSeq.m_oMetrics.dTimeElapsed_sec/oSeq.GetTotalImageCount() << std::endl;
    oMetricsOutput.close();
}

cv::Mat DatasetUtils::VideoSegm::SequenceInfo::GetInputFromIndex_external(size_t nFrameIdx) {
    cv::Mat oFrame;
    if( m_eDatasetID==DatasetUtils::VideoSegm::eDataset_CDnet2012 ||
        m_eDatasetID==DatasetUtils::VideoSegm::eDataset_CDnet2014 ||
        m_eDatasetID==DatasetUtils::VideoSegm::eDataset_Wallflower ||
        m_eDatasetID==DatasetUtils::VideoSegm::eDataset_LITIV2012)
        oFrame = cv::imread(m_vsInputFramePaths[nFrameIdx],m_nIMReadInputFlags);
    else if( m_eDatasetID==DatasetUtils::VideoSegm::eDataset_PETS2001_D3TC1 ||
            /*m_eDatasetID==DatasetUtils::VideoSegm::eDataset_LITIV2012 || */
            m_eDatasetID==DatasetUtils::VideoSegm::eDataset_GenericTest) {
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
    if(m_bUsing4chAlignment && oFrame.channels()==3)
        cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2BGRA);
    CV_Assert(oFrame.size()==m_oSize);
#if DATASETUTILS_HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    DatasetUtils::WriteOnImage(oFrame,sstr.str(),cv::Scalar::all(255));
#endif //DATASETUTILS_HARDCODE_FRAME_INDEX
    return oFrame;
}

cv::Mat DatasetUtils::VideoSegm::SequenceInfo::GetGTFromIndex_external(size_t nFrameIdx) {
    cv::Mat oFrame;
    if(m_eDatasetID == DatasetUtils::VideoSegm::eDataset_CDnet2012 ||
       m_eDatasetID == DatasetUtils::VideoSegm::eDataset_CDnet2014)
        oFrame = cv::imread(m_vsGTFramePaths[nFrameIdx],cv::IMREAD_GRAYSCALE);
    else if(m_eDatasetID == DatasetUtils::VideoSegm::eDataset_Wallflower ||
            m_eDatasetID == DatasetUtils::VideoSegm::eDataset_PETS2001_D3TC1) {
        auto res = m_mTestGTIndexes.find(nFrameIdx);
        if(res != m_mTestGTIndexes.end())
            oFrame = cv::imread(m_vsGTFramePaths[res->second],cv::IMREAD_GRAYSCALE);
    }
#if DATASETUTILS_HARDCODE_FRAME_INDEX
    if(!oFrame.empty()) {
        std::stringstream sstr;
        sstr << "Frame #" << nFrameIdx;
        DatasetUtils::WriteOnImage(oFrame,sstr.str(),cv::Scalar::all(255));
    }
#endif //DATASETUTILS_HARDCODE_FRAME_INDEX
    return oFrame;
}

void DatasetUtils::VideoSegm::CDnet::AccumulateMetricsFromResult(const cv::Mat& oSegmResFrame, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, DatasetUtils::SegmEval::BasicMetrics& m) {
    CV_DbgAssert(oSegmResFrame.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmResFrame.size()==oGTSegmMask.size() && oSegmResFrame.size()==oROI.size());
    const size_t step_row = oSegmResFrame.step.p[0];
    for(size_t i=0; i<(size_t)oSegmResFrame.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmResFrame.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j=0; j<oSegmResFrame.cols; ++j) {
            if( gt_step_ptr[j]!=DatasetUtils::VideoSegm::CDnet::g_nSegmOutOfScope &&
                gt_step_ptr[j]!=DatasetUtils::VideoSegm::CDnet::g_nSegmUnknown &&
                roi_step_ptr[j]!=DatasetUtils::VideoSegm::CDnet::g_nSegmNegative ) {
                if(input_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmPositive) {
                    if(gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmPositive)
                        ++m.nTP;
                    else // gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmNegative
                        ++m.nFP;
                }
                else { // input_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmNegative
                    if(gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmPositive)
                        ++m.nFN;
                    else // gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmNegative
                        ++m.nTN;
                }
                if(gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmShadow) {
                    if(input_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmPositive)
                        ++m.nSE;
                }
            }
        }
    }
}

cv::Mat DatasetUtils::VideoSegm::CDnet::GetDebugDisplayFrame(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, size_t nFrame, cv::Point oDbgPt) {
    cv::Mat oInputImgBYTE3, oDebugImgBYTE3, oSegmMaskBYTE3;
    CV_Assert(!oInputImg.empty() && (oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3 || oInputImg.type()==CV_8UC4));
    CV_Assert(!oDebugImg.empty() && (oDebugImg.type()==CV_8UC1 || oDebugImg.type()==CV_8UC3 || oDebugImg.type()==CV_8UC4));
    CV_Assert(!oSegmMask.empty() && oSegmMask.type()==CV_8UC1);
    CV_Assert(!oGTSegmMask.empty() && oGTSegmMask.type()==CV_8UC1);
    CV_Assert(!oROI.empty() && oROI.type()==CV_8UC1);
    if(oInputImg.channels()==1) {
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_GRAY2RGB);
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_GRAY2RGB);
    }
    else if(oInputImg.channels()==4) {
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_RGBA2RGB);
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_RGBA2RGB);
    }
    else {
        oInputImgBYTE3 = oInputImg;
        oDebugImgBYTE3 = oDebugImg;
    }
    oSegmMaskBYTE3 = DatasetUtils::VideoSegm::CDnet::GetColoredSegmFrameFromResult(oSegmMask,oGTSegmMask,oROI);
    if(oDbgPt!=cv::Point(-1,-1)) {
        cv::circle(oInputImgBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
        cv::circle(oSegmMaskBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
    }
    cv::Mat displayH,displayV1,displayV2;
    cv::resize(oInputImgBYTE3,oInputImgBYTE3,cv::Size(320,240));
    cv::resize(oDebugImgBYTE3,oDebugImgBYTE3,cv::Size(320,240));
    cv::resize(oSegmMaskBYTE3,oSegmMaskBYTE3,cv::Size(320,240));

    std::stringstream sstr;
    sstr << "Frame #" << nFrame;
    DatasetUtils::WriteOnImage(oInputImgBYTE3,sstr.str(),cv::Scalar_<uchar>(0,0,255));
    DatasetUtils::WriteOnImage(oDebugImgBYTE3,"BG Reference",cv::Scalar_<uchar>(0,0,255));
    DatasetUtils::WriteOnImage(oSegmMaskBYTE3,"Segmentation Result",cv::Scalar_<uchar>(0,0,255));

    cv::hconcat(oInputImgBYTE3,oDebugImgBYTE3,displayH);
    cv::hconcat(displayH,oSegmMaskBYTE3,displayH);
    return displayH;
}

cv::Mat DatasetUtils::VideoSegm::CDnet::GetColoredSegmFrameFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) {
    CV_DbgAssert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmMask.size()==oGTSegmMask.size() && oSegmMask.size()==oROI.size());
    cv::Mat oResult(oSegmMask.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if( gt_step_ptr[j]!=DatasetUtils::VideoSegm::CDnet::g_nSegmOutOfScope &&
                gt_step_ptr[j]!=DatasetUtils::VideoSegm::CDnet::g_nSegmUnknown &&
                roi_step_ptr[j]!=DatasetUtils::VideoSegm::CDnet::g_nSegmNegative ) {
                if(input_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmPositive) {
                    if(gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmPositive)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmNegative)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmShadow) {
                        res_step_ptr[j*3+1] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                    else {
                        for(size_t c=0; c<3; ++c)
                            res_step_ptr[j*3+c] = UCHAR_MAX/3;
                    }
                }
                else { // input_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmNegative
                    if(gt_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmPositive) {
                        res_step_ptr[j*3] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                }
            }
            else if(roi_step_ptr[j]==DatasetUtils::VideoSegm::CDnet::g_nSegmNegative) {
                for(size_t c=0; c<3; ++c)
                    res_step_ptr[j*3+c] = UCHAR_MAX/2;
            }
            else {
                for(size_t c=0; c<3; ++c)
                    res_step_ptr[j*3+c] = input_step_ptr[j];
            }
        }
    }
    return oResult;
}

#if HAVE_GLSL

DatasetUtils::VideoSegm::CDnet::Evaluator::Evaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount)
    :    GLImageProcEvaluatorAlgo(pParent,nTotFrameCount,eCDNetEvalCountersCount,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

std::string DatasetUtils::VideoSegm::CDnet::Evaluator::getComputeShaderSource(size_t nStage) const {
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "#define VAL_POSITIVE     " << (uint)g_nSegmPositive << "\n"
             "#define VAL_NEGATIVE     " << (uint)g_nSegmNegative << "\n"
             "#define VAL_OUTOFSCOPE   " << (uint)g_nSegmOutOfScope << "\n"
             "#define VAL_UNKNOWN      " << (uint)g_nSegmUnknown << "\n"
             "#define VAL_SHADOW       " << (uint)g_nSegmShadow << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
             "layout(binding=" << GLImageProcAlgo::eImage_ROIBinding << ", r8ui) readonly uniform uimage2D imgROI;\n"
             "layout(binding=" << GLImageProcAlgo::eImage_OutputBinding << ", r8ui) readonly uniform uimage2D imgInput;\n"
             "layout(binding=" << GLImageProcAlgo::eImage_GTBinding << ", r8ui) readonly uniform uimage2D imgGT;\n";
    if(m_bUsingDebug) ssSrc <<
             "layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ") writeonly uniform uimage2D imgDebug;\n";
    ssSrc << "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eCDNetEvalCounter_TP*4 << ") uniform atomic_uint nTP;\n"
             "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eCDNetEvalCounter_TN*4 << ") uniform atomic_uint nTN;\n"
             "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eCDNetEvalCounter_FP*4 << ") uniform atomic_uint nFP;\n"
             "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eCDNetEvalCounter_FN*4 << ") uniform atomic_uint nFN;\n"
             "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eCDNetEvalCounter_SE*4 << ") uniform atomic_uint nSE;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    ivec2 imgCoord = ivec2(gl_GlobalInvocationID.xy);\n"
             "    uint nInputSegmVal = imageLoad(imgInput,imgCoord).r;\n"
             "    uint nGTSegmVal = imageLoad(imgGT,imgCoord).r;\n"
             "    uint nROIVal = imageLoad(imgROI,imgCoord).r;\n"
             "    if(nROIVal!=VAL_NEGATIVE) {\n"
             "        if(nGTSegmVal!=VAL_OUTOFSCOPE && nGTSegmVal!=VAL_UNKNOWN) {\n"
             "            if(nInputSegmVal==VAL_POSITIVE) {\n"
             "                if(nGTSegmVal==VAL_POSITIVE) {\n"
             "                    atomicCounterIncrement(nTP);\n"
             "                }\n"
             "                else { // nGTSegmVal==VAL_NEGATIVE\n"
             "                    atomicCounterIncrement(nFP);\n"
             "                }\n"
             "            }\n"
             "            else { // nInputSegmVal==VAL_NEGATIVE\n"
             "                if(nGTSegmVal==VAL_POSITIVE) {\n"
             "                    atomicCounterIncrement(nFN);\n"
             "                }\n"
             "                else { // nGTSegmVal==VAL_NEGATIVE\n"
             "                    atomicCounterIncrement(nTN);\n"
             "                }\n"
             "            }\n"
             "            if(nGTSegmVal==VAL_SHADOW) {\n"
             "                if(nInputSegmVal==VAL_POSITIVE) {\n"
             "                   atomicCounterIncrement(nSE);\n"
             "                }\n"
             "            }\n"
             "        }\n"
             "    }\n";
    if(m_bUsingDebug) { ssSrc <<
             "    uvec4 out_color = uvec4(0,0,0,255);\n"
             "    if(nGTSegmVal!=VAL_OUTOFSCOPE && nGTSegmVal!=VAL_UNKNOWN && nROIVal!=VAL_NEGATIVE) {\n"
             "        if(nInputSegmVal==VAL_POSITIVE) {\n"
             "            if(nGTSegmVal==VAL_POSITIVE) {\n"
             "                out_color.g = uint(255);\n"
             "            }\n"
             "            else if(nGTSegmVal==VAL_NEGATIVE) {\n"
             "                out_color.r = uint(255);\n"
             "            }\n"
             "            else if(nGTSegmVal==VAL_SHADOW) {\n"
             "                out_color.rg = uvec2(255,128);\n"
             "            }\n"
             "            else {\n"
             "                out_color.rgb = uvec3(85);\n"
             "            }\n"
             "        }\n"
             "        else { // nInputSegmVal==VAL_NEGATIVE\n"
             "            if(nGTSegmVal==VAL_POSITIVE) {\n"
             "                out_color.rb = uvec2(255,128);\n"
             "            }\n"
             "        }\n"
             "    }\n"
             "    else if(nROIVal==VAL_NEGATIVE) {\n"
             "        out_color.rgb = uvec3(128);\n"
             "    }\n"
             "    else if(nInputSegmVal==VAL_POSITIVE) {\n"
             "        out_color.rgb = uvec3(255);\n"
             "    }\n"
             "    else if(nInputSegmVal==VAL_NEGATIVE) {\n"
             "        out_color.rgb = uvec3(0);\n"
             "    }\n"
             "    imageStore(imgDebug,imgCoord,out_color);\n";
    }
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

void DatasetUtils::VideoSegm::CDnet::Evaluator::getCumulativeCounts(DatasetUtils::SegmEval::BasicMetrics& m) {
    const cv::Mat& oAtomicCountersQueryBuffer = this->getEvaluationAtomicCounterBuffer();
    m.nTP=0; m.nTN=0; m.nFP=0; m.nFN=0; m.nSE=0;
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        m.nTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,VideoSegm::CDnet::Evaluator::eCDNetEvalCounter_TP);
        m.nTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,VideoSegm::CDnet::Evaluator::eCDNetEvalCounter_TN);
        m.nFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,VideoSegm::CDnet::Evaluator::eCDNetEvalCounter_FP);
        m.nFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,VideoSegm::CDnet::Evaluator::eCDNetEvalCounter_FN);
        m.nSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,VideoSegm::CDnet::Evaluator::eCDNetEvalCounter_SE);
    }
}

#endif //HAVE_GLSL
