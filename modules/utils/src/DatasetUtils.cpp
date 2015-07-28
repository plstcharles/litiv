#include "litiv/utils/DatasetUtils.hpp"
#include "litiv/utils/DatasetEvalUtils.hpp"

#define PRECACHE_CONSOLE_DEBUG             0
#define PRECACHE_REQUEST_TIMEOUT_MS        1
#define PRECACHE_QUERY_TIMEOUT_MS          10
#define PRECACHE_MAX_CACHE_SIZE_GB         6L
#define PRECACHE_MAX_CACHE_SIZE            (((PRECACHE_MAX_CACHE_SIZE_GB*1024)*1024)*1024)
#if (!(defined(_M_X64) || defined(__amd64__)) && PRECACHE_MAX_CACHE_SIZE_GB>2)
#error "Cache max size exceeds system limit (x86)."
#endif //(!(defined(_M_X64) || defined(__amd64__)) && PRECACHE_MAX_CACHE_SIZE_GB>2)

// @@@@ can remove type scope for member function args

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

std::vector<std::shared_ptr<DatasetUtils::WorkGroup>> DatasetUtils::DatasetInfoBase::ParseDataset(const DatasetUtils::DatasetInfoBase& oInfo) {
    if(!oInfo.m_sResultsRootPath.empty())
        PlatformUtils::CreateDirIfNotExist(oInfo.m_sResultsRootPath);
    std::vector<std::shared_ptr<DatasetUtils::WorkGroup>> vpGroups;
    for(auto psPathIter=oInfo.m_vsWorkBatchPaths.begin(); psPathIter!=oInfo.m_vsWorkBatchPaths.end(); ++psPathIter)
        vpGroups.push_back(std::make_shared<DatasetUtils::WorkGroup>(*psPathIter,oInfo));
    return vpGroups;
}

DatasetUtils::ImagePrecacher::ImagePrecacher(std::function<const cv::Mat&(size_t)> pCallback) {
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
                std::cout << " @ filling precache buffer... (current size = " << (nUsedBufferSize/1024)/1024 << " mb, " << m_nTotImageCount-m_nNextPrecacheIdx << " todo)" << std::endl;
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

DatasetUtils::WorkBatch::WorkBatch(const std::string& sBatchName, const DatasetInfoBase& oDatasetInfo, const std::string& sRelativePath) :
        m_sName(sBatchName),
        m_sRelativePath(sRelativePath),
        m_sDatasetPath(oDatasetInfo.m_sDatasetRootPath+sRelativePath),
        m_sResultsPath(oDatasetInfo.m_sResultsRootPath+sRelativePath),
        m_sResultNamePrefix(oDatasetInfo.m_sResultNamePrefix),
        m_sResultNameSuffix(oDatasetInfo.m_sResultNameSuffix),
        m_bHasGroundTruth(oDatasetInfo.m_pEvaluator!=nullptr),
        m_bForcingGrayscale(PlatformUtils::string_contains_token(sBatchName,oDatasetInfo.m_vsGrayscaleNameTokens)),
        m_bForcing4ByteDataAlign(oDatasetInfo.m_bForce4ByteDataAlign),
        m_oInputPrecacher(std::bind(&DatasetUtils::WorkBatch::GetInputFromIndex_internal,this,std::placeholders::_1)),
        m_oGTPrecacher(std::bind(&DatasetUtils::WorkBatch::GetGTFromIndex_internal,this,std::placeholders::_1)) {
    PlatformUtils::CreateDirIfNotExist(m_sResultsPath);
}

cv::Mat DatasetUtils::WorkBatch::ReadResult(size_t nIdx) {
    CV_Assert(!m_sResultNameSuffix.empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << m_sResultsPath << m_sResultNamePrefix << acBuffer.data() << m_sResultNameSuffix;
    return cv::imread(sResultFilePath.str(),m_bForcingGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
}

void DatasetUtils::WorkBatch::WriteResult(size_t nIdx, const cv::Mat& oResult) {
    CV_Assert(!m_sResultNameSuffix.empty());
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << m_sResultsPath << m_sResultNamePrefix << acBuffer.data() << m_sResultNameSuffix;
    const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
    cv::imwrite(sResultFilePath.str(),oResult,vnComprParams);
}

bool DatasetUtils::WorkBatch::StartPrecaching(bool bUsingGT, size_t nSuggestedBufferSize) {
    return m_oInputPrecacher.StartPrecaching(GetTotalImageCount(),nSuggestedBufferSize) &&
           (!bUsingGT || !m_bHasGroundTruth || m_oGTPrecacher.StartPrecaching(GetTotalImageCount(),nSuggestedBufferSize));
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

DatasetUtils::WorkGroup::WorkGroup(const std::string& sGroupName, const DatasetInfoBase& oDatasetInfo, const std::string& sRelativePath) :
        WorkBatch(sGroupName,oDatasetInfo,sRelativePath+"/"+sGroupName+"/"),
        m_dExpectedLoad(0),
        m_nTotImageCount(0) {
    PlatformUtils::CreateDirIfNotExist(m_sResultsPath);
    if(!PlatformUtils::string_contains_token(m_sName,oDatasetInfo.m_vsSkippedNameTokens)) {
        std::cout << "[" << oDatasetInfo.m_sDatasetName << "] -- Parsing directory '" << oDatasetInfo.m_sDatasetRootPath+sRelativePath << "' for work group '" << m_sName << "'..." << std::endl;
        std::vector<std::string> vsWorkBatchPaths;
        // all subdirs are considered work batch directories (if none, the category directory itself is a batch)
        PlatformUtils::GetSubDirsFromDir(m_sDatasetPath,vsWorkBatchPaths);
        if(vsWorkBatchPaths.empty()) {
            if(oDatasetInfo.GetType()==eDatasetType_Segm_Video)
                m_vpBatches.push_back(std::make_shared<Segm::Video::Sequence>(m_sName,dynamic_cast<const Segm::Video::DatasetInfo&>(oDatasetInfo),m_sRelativePath));
            else if(oDatasetInfo.GetType()==eDatasetType_Segm_Image)
                m_vpBatches.push_back(std::make_shared<Segm::Image::Set>(m_sName,dynamic_cast<const Segm::Image::DatasetInfo&>(oDatasetInfo),m_sRelativePath));
            else
                throw std::logic_error(cv::format("Workgroup '%s': bad dataset type bassed in dataset info struct",m_sName.c_str()));
            m_dExpectedLoad += m_vpBatches.back()->GetExpectedLoad();
            m_nTotImageCount += m_vpBatches.back()->GetTotalImageCount();
        }
        else {
            for(auto psPathIter = vsWorkBatchPaths.begin(); psPathIter!=vsWorkBatchPaths.end(); ++psPathIter) {
                const size_t nLastSlashPos = psPathIter->find_last_of("/\\");
                const std::string sBatchName = nLastSlashPos==std::string::npos?*psPathIter:psPathIter->substr(nLastSlashPos+1);
                if(!PlatformUtils::string_contains_token(sBatchName,oDatasetInfo.m_vsSkippedNameTokens)) {
                    if(oDatasetInfo.GetType()==eDatasetType_Segm_Video)
                        m_vpBatches.push_back(std::make_shared<Segm::Video::Sequence>(sBatchName,dynamic_cast<const Segm::Video::DatasetInfo&>(oDatasetInfo),m_sRelativePath+"/"+sBatchName+"/"));
                    else if(oDatasetInfo.GetType()==eDatasetType_Segm_Image)
                        m_vpBatches.push_back(std::make_shared<Segm::Image::Set>(sBatchName,dynamic_cast<const Segm::Image::DatasetInfo&>(oDatasetInfo),m_sRelativePath+"/"+sBatchName+"/"));
                    else
                        throw std::logic_error(cv::format("Workgroup '%s': bad dataset type bassed in dataset info struct",m_sName.c_str()));
                    m_dExpectedLoad += m_vpBatches.back()->GetExpectedLoad();
                    m_nTotImageCount += m_vpBatches.back()->GetTotalImageCount();
                }
            }
        }
    }
}

cv::Mat DatasetUtils::WorkGroup::GetInputFromIndex_external(size_t nIdx) {
    if(m_vpBatches.size()==1)
        return m_vpBatches.front()->GetInputFromIndex_external(nIdx);
    size_t nCumulIdx = 0;
    size_t nBatchIdx = 0;
    do {
        nCumulIdx += m_vpBatches[nBatchIdx++]->GetTotalImageCount();
    } while(nCumulIdx<nIdx);
    return m_vpBatches[nBatchIdx-1]->GetInputFromIndex_external(nIdx-(nCumulIdx-m_vpBatches[nBatchIdx-1]->GetTotalImageCount()));
}

cv::Mat DatasetUtils::WorkGroup::GetGTFromIndex_external(size_t nIdx) {
    if(m_vpBatches.size()==1)
        return m_vpBatches.front()->GetGTFromIndex_external(nIdx);
    size_t nCumulIdx = 0;
    size_t nBatchIdx = 0;
    do {
        nCumulIdx += m_vpBatches[nBatchIdx++]->GetTotalImageCount();
    } while(nCumulIdx<nIdx);
    return m_vpBatches[nBatchIdx-1]->GetGTFromIndex_external(nIdx-(nCumulIdx-m_vpBatches[nBatchIdx-1]->GetTotalImageCount()));
}

DatasetUtils::Segm::BasicMetrics::BasicMetrics() :
        nTP(0),nTN(0),nFP(0),nFN(0),nSE(0),dTimeElapsed_sec(0) {}

DatasetUtils::Segm::BasicMetrics DatasetUtils::Segm::BasicMetrics::operator+(const BasicMetrics& m) const {
    BasicMetrics res(m);
    res.nTP += this->nTP;
    res.nTN += this->nTN;
    res.nFP += this->nFP;
    res.nFN += this->nFN;
    res.nSE += this->nSE;
    res.dTimeElapsed_sec += this->dTimeElapsed_sec;
    return res;
}

DatasetUtils::Segm::BasicMetrics& DatasetUtils::Segm::BasicMetrics::operator+=(const BasicMetrics& m) {
    this->nTP += m.nTP;
    this->nTN += m.nTN;
    this->nFP += m.nFP;
    this->nFN += m.nFN;
    this->nSE += m.nSE;
    this->dTimeElapsed_sec += m.dTimeElapsed_sec;
    return *this;
}

DatasetUtils::Segm::Metrics::Metrics(const DatasetUtils::Segm::BasicMetrics& m) :
        dRecall(CalcRecall(m)),
        dSpecificity(CalcSpecificity(m)),
        dFPR(CalcFalsePositiveRate(m)),
        dFNR(CalcFalseNegativeRate(m)),
        dPBC(CalcPercentBadClassifs(m)),
        dPrecision(CalcPrecision(m)),
        dFMeasure(CalcFMeasure(m)),
        dMCC(CalcMatthewsCorrCoeff(m)),
        dTimeElapsed_sec(m.dTimeElapsed_sec),
        nWeight(1) {}

DatasetUtils::Segm::Metrics DatasetUtils::Segm::Metrics::operator+(const DatasetUtils::Segm::BasicMetrics& m) const {
    Metrics tmp(m);
    return (*this)+tmp;
}

DatasetUtils::Segm::Metrics& DatasetUtils::Segm::Metrics::operator+=(const DatasetUtils::Segm::BasicMetrics& m) {
    Metrics tmp(m);
    (*this) += tmp;
    return *this;
}

DatasetUtils::Segm::Metrics DatasetUtils::Segm::Metrics::operator+(const DatasetUtils::Segm::Metrics& m) const {
    Metrics res(m);
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

DatasetUtils::Segm::Metrics& DatasetUtils::Segm::Metrics::operator+=(const DatasetUtils::Segm::Metrics& m) {
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

double DatasetUtils::Segm::Metrics::CalcFMeasure(const DatasetUtils::Segm::BasicMetrics& m) {
    const double dRecall = CalcRecall(m);
    const double dPrecision = CalcPrecision(m);
    return (2.0*(dRecall*dPrecision)/(dRecall+dPrecision));
}
double DatasetUtils::Segm::Metrics::CalcRecall(const DatasetUtils::Segm::BasicMetrics& m) {return ((double)m.nTP/(m.nTP+m.nFN));}
double DatasetUtils::Segm::Metrics::CalcPrecision(const DatasetUtils::Segm::BasicMetrics& m) {return ((double)m.nTP/(m.nTP+m.nFP));}
double DatasetUtils::Segm::Metrics::CalcSpecificity(const DatasetUtils::Segm::BasicMetrics& m) {return ((double)m.nTN/(m.nTN+m.nFP));}
double DatasetUtils::Segm::Metrics::CalcFalsePositiveRate(const DatasetUtils::Segm::BasicMetrics& m) {return ((double)m.nFP/(m.nFP+m.nTN));}
double DatasetUtils::Segm::Metrics::CalcFalseNegativeRate(const DatasetUtils::Segm::BasicMetrics& m) {return ((double)m.nFN/(m.nTP+m.nFN));}
double DatasetUtils::Segm::Metrics::CalcPercentBadClassifs(const DatasetUtils::Segm::BasicMetrics& m) {return (100.0*(m.nFN+m.nFP)/(m.nTP+m.nFP+m.nFN+m.nTN));}
double DatasetUtils::Segm::Metrics::CalcMatthewsCorrCoeff(const DatasetUtils::Segm::BasicMetrics& m) {return ((((double)m.nTP*m.nTN)-(m.nFP*m.nFN))/sqrt(((double)m.nTP+m.nFP)*(m.nTP+m.nFN)*(m.nTN+m.nFP)*(m.nTN+m.nFN)));}

DatasetUtils::Segm::SegmWorkBatch::SegmWorkBatch(const std::string& sBatchName, const DatasetInfoBase& oDatasetInfo, const std::string& sRelativePath) :
        WorkBatch(sBatchName,oDatasetInfo,sRelativePath) {}

std::shared_ptr<DatasetUtils::Segm::Video::DatasetInfo> DatasetUtils::Segm::Video::GetDatasetInfo(const DatasetUtils::Segm::Video::eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign) {
    std::shared_ptr<DatasetUtils::Segm::Video::DatasetInfo> pInfo;
    if(eDatasetID==eDataset_CDnet2012) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "CDnet 2012";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/CDNet/dataset/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/CDNet/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "bin";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_pEvaluator                 = std::shared_ptr<SegmEvaluator>(new CDnetEvaluator);
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
        pInfo->m_pEvaluator                 = std::shared_ptr<SegmEvaluator>(new CDnetEvaluator);
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
        pInfo->m_pEvaluator                 = std::shared_ptr<SegmEvaluator>(new BinarySegmEvaluator);
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
        pInfo->m_pEvaluator                 = std::shared_ptr<SegmEvaluator>(new BinarySegmEvaluator);
        pInfo->m_vsWorkBatchPaths           = {"TESTING"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_PETS2001_D3TC1;
        pInfo->m_nResultIdxOffset           = 0;
    }
    else if(eDatasetID==eDataset_Custom)
        throw std::logic_error(cv::format("DatasetUtils::Segm::Video::GetDatasetInfo: custom dataset info struct (eDataset_Custom) can only be filled manually"));
    else
        throw std::logic_error(cv::format("DatasetUtils::Segm::Video::GetDatasetInfo: unknown dataset type, cannot use predefined dataset info struct"));
    return pInfo;
}

DatasetUtils::Segm::Video::Sequence::Sequence(const std::string& sSeqName, const DatasetInfo& oDatasetInfo, const std::string& sRelativePath) :
        SegmWorkBatch(sSeqName,oDatasetInfo,sRelativePath),
        m_eDatasetID(oDatasetInfo.m_eDatasetID),
        m_nResultIdxOffset(oDatasetInfo.m_nResultIdxOffset),
        m_dExpectedLoad(0),
        m_nTotFrameCount(0),
        m_nNextExpectedVideoReaderFrameIdx(0) {
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014) {
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
        // note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
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
    }
    else if(m_eDatasetID==eDataset_Custom) {
        m_voVideoReader.open(m_sDatasetPath);
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(cv::format("Sequence '%s': video could not be opened",sSeqName.c_str()));
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        cv::Mat oTempImg;
        m_voVideoReader >> oTempImg;
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        if(oTempImg.empty())
            throw std::runtime_error(cv::format("Sequence '%s': video could not be read",sSeqName.c_str()));
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        m_nTotFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
    }
    else
        throw std::logic_error(cv::format("Sequence '%s': unknown dataset type, cannot use any known parsing strategy",sSeqName.c_str()));
}

void DatasetUtils::Segm::Video::Sequence::WriteResult(size_t nIdx, const cv::Mat& oResult) {
    WorkBatch::WriteResult(nIdx+m_nResultIdxOffset,oResult);
}

bool DatasetUtils::Segm::Video::Sequence::StartPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return WorkBatch::StartPrecaching(bUsingGT,m_oSize.height*m_oSize.width*(m_nTotFrameCount+1)*(m_bForcingGrayscale?1:m_bForcing4ByteDataAlign?4:3));
}

cv::Mat DatasetUtils::Segm::Video::Sequence::GetInputFromIndex_external(size_t nFrameIdx) {
    cv::Mat oFrame;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014 || m_eDatasetID==eDataset_Wallflower)
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
    CV_Assert(oFrame.size()==m_oSize);
#if DATASETUTILS_HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    WriteOnImage(oFrame,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //DATASETUTILS_HARDCODE_FRAME_INDEX
    return oFrame;
}

cv::Mat DatasetUtils::Segm::Video::Sequence::GetGTFromIndex_external(size_t nFrameIdx) {
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
    CV_Assert(oFrame.size()==m_oSize);
#if DATASETUTILS_HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    WriteOnImage(oFrame,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //DATASETUTILS_HARDCODE_FRAME_INDEX
    return oFrame;
}

std::shared_ptr<DatasetUtils::Segm::Image::DatasetInfo> DatasetUtils::Segm::Image::GetDatasetInfo(const DatasetUtils::Segm::Image::eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign) {
    std::shared_ptr<DatasetUtils::Segm::Image::DatasetInfo> pInfo;
    if(eDatasetID==eDataset_BSDS500_train) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "BSDS500 Training set";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/BSDS500/data/images/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/BSDS500/BSR/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_pEvaluator                 = nullptr; // external only, still not impl
        pInfo->m_vsWorkBatchPaths           = {"train"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_BSDS500_train;
    }
    else if(eDatasetID==eDataset_BSDS500_train_valid) {
        pInfo = std::make_shared<DatasetInfo>();
        pInfo->m_sDatasetName               = "BSDS500 Training+Validation set";
        pInfo->m_sDatasetRootPath           = sDatasetRootDirPath+"/BSDS500/data/images/";
        pInfo->m_sResultsRootPath           = sDatasetRootDirPath+"/BSDS500/BSR/"+sResultsDirName+"/";
        pInfo->m_sResultNamePrefix          = "";
        pInfo->m_sResultNameSuffix          = ".png";
        pInfo->m_pEvaluator                 = nullptr; // external only, still not impl
        pInfo->m_vsWorkBatchPaths           = {"train","val"};
        pInfo->m_vsSkippedNameTokens        = {};
        pInfo->m_vsGrayscaleNameTokens      = {};
        pInfo->m_bForce4ByteDataAlign       = bForce4ByteDataAlign;
        pInfo->m_eDatasetID                 = eDataset_BSDS500_train;
    }
    else if(eDatasetID==eDataset_Custom)
        throw std::logic_error(cv::format("DatasetUtils::Segm::Image::GetDatasetInfo: custom dataset info struct (eDataset_Custom) can only be filled manually"));
    else
        throw std::logic_error(cv::format("DatasetUtils::Segm::Image::GetDatasetInfo: unknown dataset type, cannot use predefined dataset info struct"));
    return pInfo;
}

DatasetUtils::Segm::Image::Set::Set(const std::string& sSetName, const DatasetInfo& oDatasetInfo, const std::string& sRelativePath) :
        SegmWorkBatch(sSetName,oDatasetInfo,sRelativePath),
        m_eDatasetID(oDatasetInfo.m_eDatasetID),
        m_dExpectedLoad(0),
        m_nTotImageCount(0),
        m_bIsConstantSize(false) {
    if(m_eDatasetID==eDataset_BSDS500_train || m_eDatasetID==eDataset_BSDS500_train_valid) {
        // current impl cannot parse GT (matlab files only)
        PlatformUtils::GetFilesFromDir(m_sDatasetPath,m_vsInputImagePaths);
        PlatformUtils::FilterFilePaths(m_vsInputImagePaths,{},{".jpg"});
        if(m_vsInputImagePaths.empty())
            throw std::runtime_error(cv::format("Image set '%s' did not possess any image file",sSetName.c_str()));
        m_oMaxSize = cv::Size(481,321);
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

cv::Mat DatasetUtils::Segm::Image::Set::GetInputFromIndex_external(size_t nImageIdx) {
    cv::Mat oImage;
    oImage = cv::imread(m_vsInputImagePaths[nImageIdx],m_bForcingGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    CV_Assert(!oImage.empty());
    CV_Assert(m_voOrigImageSizes[nImageIdx]==cv::Size() || m_voOrigImageSizes[nImageIdx]==oImage.size());
    m_voOrigImageSizes[nImageIdx] = oImage.size();
    if(m_eDatasetID==eDataset_BSDS500_train || m_eDatasetID==eDataset_BSDS500_train_valid) {
        CV_Assert(oImage.size()==cv::Size(481,321) || oImage.size()==cv::Size(321,481));
        if(oImage.size()==cv::Size(321,481))
            cv::transpose(oImage,oImage);
    }
    CV_Assert(oImage.cols<=m_oMaxSize.width && oImage.rows<=m_oMaxSize.height);
    if(m_bForcing4ByteDataAlign && oImage.channels()==3)
        cv::cvtColor(oImage,oImage,cv::COLOR_BGR2BGRA);
#if DATASETUTILS_HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Image #" << nImageIdx;
    WriteOnImage(oImage,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //DATASETUTILS_HARDCODE_FRAME_INDEX
    return oImage;
}

cv::Mat DatasetUtils::Segm::Image::Set::GetGTFromIndex_external(size_t nImageIdx) {
    cv::Mat oImage;
    if(m_vsGTImagePaths.size()>nImageIdx)
        oImage = cv::imread(m_vsGTImagePaths[nImageIdx],cv::IMREAD_GRAYSCALE);
    if(!oImage.empty()) {
        CV_Assert(m_voOrigImageSizes[nImageIdx]==cv::Size() || m_voOrigImageSizes[nImageIdx]==oImage.size());
        m_voOrigImageSizes[nImageIdx] = oImage.size();
        if(m_eDatasetID==eDataset_BSDS500_train || m_eDatasetID==eDataset_BSDS500_train_valid) {
            CV_Assert(oImage.size()==cv::Size(481,321) || oImage.size()==cv::Size(321,481));
            if(oImage.size()==cv::Size(321,481))
                cv::transpose(oImage,oImage);
        }
        CV_Assert(oImage.cols<=m_oMaxSize.width && oImage.rows<=m_oMaxSize.height);
        if(m_bForcing4ByteDataAlign && oImage.channels()==3)
            cv::cvtColor(oImage,oImage,cv::COLOR_BGR2BGRA);
    }
    if(oImage.empty())
        oImage = cv::Mat(m_oMaxSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL));
#if DATASETUTILS_HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Image #" << nImageIdx;
    WriteOnImage(oImage,sstr.str(),cv::Scalar_<uchar>::all(255);
#endif //DATASETUTILS_HARDCODE_FRAME_INDEX
    return oImage;
}

cv::Mat DatasetUtils::Segm::Image::Set::ReadResult(size_t nImageIdx) {
    CV_Assert(m_vsOrigImageNames[nImageIdx]!=std::string());
    CV_Assert(!m_sResultNameSuffix.empty());
    std::stringstream sResultFilePath;
    sResultFilePath << m_sResultsPath << m_sResultNamePrefix << m_vsOrigImageNames[nImageIdx] << m_sResultNameSuffix;
    cv::Mat oImage = cv::imread(sResultFilePath.str(),m_bForcingGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
    if(m_eDatasetID==eDataset_BSDS500_train || m_eDatasetID==eDataset_BSDS500_train_valid) {
        CV_Assert(oImage.size()==cv::Size(481,321) || oImage.size()==cv::Size(321,481));
        CV_Assert(m_voOrigImageSizes[nImageIdx]==cv::Size() || m_voOrigImageSizes[nImageIdx]==oImage.size());
        m_voOrigImageSizes[nImageIdx] = oImage.size();
        if(oImage.size()==cv::Size(321,481))
            cv::transpose(oImage,oImage);
    }
    return oImage;
}

void DatasetUtils::Segm::Image::Set::WriteResult(size_t nImageIdx, const cv::Mat& oResult) {
    CV_Assert(m_vsOrigImageNames[nImageIdx]!=std::string());
    CV_Assert(!m_sResultNameSuffix.empty());
    cv::Mat oImage = oResult;
    if(m_eDatasetID==eDataset_BSDS500_train || m_eDatasetID==eDataset_BSDS500_train_valid) {
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

bool DatasetUtils::Segm::Image::Set::StartPrecaching(bool bUsingGT, size_t /*nUnused*/) {
    return WorkBatch::StartPrecaching(bUsingGT,m_oMaxSize.height*m_oMaxSize.width*(m_nTotImageCount+1)*(m_bForcingGrayscale?1:m_bForcing4ByteDataAlign?4:3));
}
