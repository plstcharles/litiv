#pragma once

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <unordered_map>
#include <deque>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "PlatformUtils.h"

#define USE_AVERAGE_METRICS      1
#define USE_BROKEN_FNR_FUNCTION  0
#define USE_PRECACHED_IO         0

#define CDNET_DB_NAME            "CDNet"
#define WALLFLOWER_DB_NAME       "WALLFLOWER"
#define PETS2001_D3TC1_DB_NAME   "PETS2001_D3TC1"
#define SINGLE_AVI_TEST_NAME     "AVI_TEST"

// as defined in the 2012 CDNet scripts/dataset
#define VAL_POSITIVE     255
#define VAL_NEGATIVE     0
#define VAL_OUTOFSCOPE   85
#define VAL_UNKNOWN      170
#define VAL_SHADOW       50

#define METRIC_RECALL(TP,TN,FP,FN)       ((double)TP/(TP+FN))
#define METRIC_PRECISION(TP,TN,FP,FN)    ((double)TP/(TP+FP))
#define METRIC_SPECIFICITY(TP,TN,FP,FN)  ((double)TN/(TN+FP))
#define METRIC_FALSEPOSRATE(TP,TN,FP,FN) ((double)FP/(FP+TN))
#define METRIC_FALSENEGRATE(TP,TN,FP,FN) ((double)FN/(USE_BROKEN_FNR_FUNCTION?(TN+FP):(TP+FN)))
#define METRIC_PERCENTBADCL(TP,TN,FP,FN) (100.0*(FN+FP)/(TP+FP+FN+TN))
#define METRIC_FMEASURE(TP,TN,FP,FN)     (2.0*(METRIC_RECALL(TP,TN,FP,FN)*METRIC_PRECISION(TP,TN,FP,FN))/(METRIC_RECALL(TP,TN,FP,FN)+METRIC_PRECISION(TP,TN,FP,FN)))
#define METRIC_MATTCORRCOEF(TP,TN,FP,FN) ((((double)TP*TN)-(FP*FN))/sqrt(((double)TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

#if USE_PRECACHED_IO
#define MAX_NB_PRECACHED_FRAMES   100
#define PRECACHE_REFILL_THRESHOLD (MAX_NB_PRECACHED_FRAMES/4)
#define REQUEST_TIMEOUT_MS        1
#define QUERY_TIMEOUT_MS          10
#endif //USE_PRECACHED_IO

class SequenceInfo;

static inline bool compare_lowercase(const std::string& i, const std::string& j) {
    std::string i_lower(i), j_lower(j);
    std::transform(i_lower.begin(),i_lower.end(),i_lower.begin(),tolower);
    std::transform(j_lower.begin(),j_lower.end(),j_lower.begin(),tolower);
    return i_lower<j_lower;
}

template<typename T> int decimal_integer_digit_count(T number) {
    int digits = number<0?1:0;
    while(std::abs(number)>=1) {
        number /= 10;
        digits++;
    }
    return digits;
}

class CategoryInfo {
public:
    CategoryInfo(const std::string& name, const std::string& dir, const std::string& dbname, bool forceGrayscale=false);
    ~CategoryInfo();
    const std::string m_sName;
    const std::string m_sDBName;
    std::vector<SequenceInfo*> m_vpSequences;
    uint64_t nTP, nTN, nFP, nFN, nSE;
    double m_dAvgFPS;
    static inline bool compare(const CategoryInfo* i, const CategoryInfo* j) {return compare_lowercase(i->m_sName,j->m_sName);}
private:
#if PLATFORM_SUPPORTS_CPP11
    CategoryInfo& operator=(const CategoryInfo&) = delete;
    CategoryInfo(const CategoryInfo&) = delete;
#else //!PLATFORM_SUPPORTS_CPP11
    CategoryInfo& operator=(const CategoryInfo&);
    CategoryInfo(const CategoryInfo&);
#endif //!PLATFORM_SUPPORTS_CPP11
};

class SequenceInfo {
public:
    SequenceInfo(const std::string& name, const std::string& dir, const std::string& dbname, CategoryInfo* parent, bool forceGrayscale=false);
    ~SequenceInfo();
    const cv::Mat& GetInputFrameFromIndex(size_t idx);
    const cv::Mat& GetGTFrameFromIndex(size_t idx);
    size_t GetNbInputFrames() const;
    size_t GetNbGTFrames() const;
    cv::Size GetFrameSize() const;
    const cv::Mat& GetSequenceROI() const;
    std::vector<cv::KeyPoint> GetKeyPointsFromROI() const;
    void ValidateKeyPoints(std::vector<cv::KeyPoint>& voKPs) const;
    const std::string m_sName;
    const std::string m_sDBName;
    uint64_t nTP, nTN, nFP, nFN, nSE;
    double m_dAvgFPS;
    double m_dExpectedLoad;
    double m_dExpectedROILoad;
    CategoryInfo* m_pParent;
    cv::Size GetSize() {return m_oSize;}
    static inline bool compare(const SequenceInfo* i, const SequenceInfo* j) {return compare_lowercase(i->m_sName,j->m_sName);}
#if USE_PRECACHED_IO
    void StartPrecaching();
    void StopPrecaching();
private:
    void PrecacheInputFrames();
    void PrecacheGTFrames();
#if PLATFORM_SUPPORTS_CPP11
    std::thread m_hInputFramePrecacher,m_hGTFramePrecacher;
    std::mutex m_oInputFrameSyncMutex,m_oGTFrameSyncMutex;
    std::condition_variable m_oInputFrameReqCondVar,m_oGTFrameReqCondVar;
    std::condition_variable m_oInputFrameSyncCondVar,m_oGTFrameSyncCondVar;
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
    HANDLE m_hInputFramePrecacher,m_hGTFramePrecacher;
    static DWORD WINAPI PrecacheInputFramesEntryPoint(LPVOID lpParam) {try{((SequenceInfo*)lpParam)->PrecacheInputFrames();}catch(...){return-1;}return 0;}
    static DWORD WINAPI PrecacheGTFramesEntryPoint(LPVOID lpParam) {try{((SequenceInfo*)lpParam)->PrecacheGTFrames();}catch(...){return-1;} return 0;}
    CRITICAL_SECTION m_oInputFrameSyncMutex,m_oGTFrameSyncMutex;
    CONDITION_VARIABLE m_oInputFrameReqCondVar,m_oGTFrameReqCondVar;
    CONDITION_VARIABLE m_oInputFrameSyncCondVar,m_oGTFrameSyncCondVar;
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for semaphores on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
    bool m_bIsPrecaching;
    size_t m_nRequestInputFrameIndex,m_nRequestGTFrameIndex;
    std::deque<cv::Mat> m_qoInputFrameCache,m_qoGTFrameCache;
    size_t m_nNextExpectedInputFrameIdx,m_nNextExpectedGTFrameIdx;
    size_t m_nNextPrecachedInputFrameIdx,m_nNextPrecachedGTFrameIdx;
    cv::Mat m_oReqInputFrame,m_oReqGTFrame;
#else //!USE_PRECACHED_IO
private:
    size_t m_nLastReqInputFrameIndex,m_nLastReqGTFrameIndex;
    cv::Mat m_oLastReqInputFrame,m_oLastReqGTFrame;
#endif //!USE_PRECACHED_IO
    std::vector<std::string> m_vsInputFramePaths;
    std::vector<std::string> m_vsGTFramePaths;
    cv::VideoCapture m_voVideoReader;
    size_t m_nNextExpectedVideoReaderFrameIdx;
    size_t m_nTotalNbFrames;
    cv::Mat m_oROI;
    cv::Size m_oSize;
    const int m_nIMReadInputFlags;
    std::unordered_map<size_t,size_t> m_mTestGTIndexes;
    cv::Mat GetInputFrameFromIndex_Internal(size_t idx);
    cv::Mat GetGTFrameFromIndex_Internal(size_t idx);
#if PLATFORM_SUPPORTS_CPP11
    SequenceInfo& operator=(const SequenceInfo&) = delete;
    SequenceInfo(const CategoryInfo&) = delete;
#else //!PLATFORM_SUPPORTS_CPP11
    SequenceInfo& operator=(const SequenceInfo&);
    SequenceInfo(const SequenceInfo&);
#endif //!PLATFORM_SUPPORTS_CPP11
};

class AdvancedMetrics {
public:
    AdvancedMetrics(uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN, uint64_t nSE);
    AdvancedMetrics(const SequenceInfo* pSeq);
    AdvancedMetrics(const CategoryInfo* pCat, bool bAverage=false);
    AdvancedMetrics(const std::vector<CategoryInfo*>& vpCat, bool bAverage=false);
    double dRecall;
    double dSpecficity;
    double dFPR;
    double dFNR;
    double dPBC;
    double dPrecision;
    double dFMeasure;
    double dMCC;
    double dFPS; // note: always averaged
    bool bAveraged;
};

static inline void WriteOnImage(cv::Mat& oImg, const std::string& sText, bool bBottom=false) {
    cv::putText(oImg,sText,cv::Point(10,bBottom?(oImg.rows-15):15),cv::FONT_HERSHEY_PLAIN,1.0,cv::Scalar_<uchar>(0,0,255),1,CV_AA);
}

static inline void WriteResult( const std::string& sResultsPath,
                                const std::string& sCatName,
                                const std::string& sSeqName,
                                const std::string& sResultPrefix,
                                size_t framenum,
                                const std::string& sResultSuffix,
                                const cv::Mat& res,
                                const std::vector<int>& vnComprParams) {
    char buffer[10];
    sprintf(buffer,"%06lu",framenum);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << buffer << sResultSuffix;
    cv::imwrite(sResultFilePath.str(), res, vnComprParams);
}

static inline cv::Mat ReadResult( const std::string& sResultsPath,
                                  const std::string& sCatName,
                                  const std::string& sSeqName,
                                  const std::string& sResultPrefix,
                                  size_t framenum,
                                  const std::string& sResultSuffix) {
    char buffer[10];
    sprintf(buffer,"%06lu",framenum);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << buffer << sResultSuffix;
    return cv::imread(sResultFilePath.str(),cv::IMREAD_GRAYSCALE);
}

static inline void WriteMetrics(const std::string sResultsFileName, const SequenceInfo* pSeq) {
    std::ofstream oMetricsOutput(sResultsFileName);
    AdvancedMetrics temp(pSeq);
    const std::string sCurrSeqName = pSeq->m_sName.size()>12?pSeq->m_sName.substr(0,12):pSeq->m_sName;
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " : Rcl=" << std::fixed << std::setprecision(4) << temp.dRecall << " Prc=" << temp.dPrecision << " FM=" << temp.dFMeasure << " MCC=" << temp.dMCC << std::endl;
    oMetricsOutput << "Results for sequence '" << pSeq->m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
    oMetricsOutput << pSeq->nTP << " " << pSeq->nFP << " " << pSeq->nFN << " " << pSeq->nTN << " " << pSeq->nSE << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Cumulative metrics :" << std::endl;
    oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    oMetricsOutput << temp.dRecall << " " << temp.dSpecficity << " " << temp.dFPR << " " << temp.dFNR << " " << temp.dPBC << " " << temp.dPrecision << " " << temp.dFMeasure << " " << temp.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Sequence FPS: " << pSeq->m_dAvgFPS << std::endl;
    oMetricsOutput.close();
}

static inline void WriteMetrics(const std::string sResultsFileName, CategoryInfo* pCat) {
    std::ofstream oMetricsOutput(sResultsFileName);
    std::sort(pCat->m_vpSequences.begin(),pCat->m_vpSequences.end(),&SequenceInfo::compare);
    AdvancedMetrics met(pCat, USE_AVERAGE_METRICS);
    const std::string sCurrCatName = pCat->m_sName.size()>12?pCat->m_sName.substr(0,12):pCat->m_sName;
    std::cout << "\t" << std::setfill(' ') << std::setw(12) << sCurrCatName << " : Rcl=" << std::fixed << std::setprecision(4) << met.dRecall << " Prc=" << met.dPrecision << " FM=" << met.dFMeasure << " MCC=" << met.dMCC << std::endl;
    oMetricsOutput << "Results for category '" << pCat->m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
    oMetricsOutput << pCat->nTP << " " << pCat->nFP << " " << pCat->nFN << " " << pCat->nTN << " " << pCat->nSE << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Sequence Metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    for(size_t i=0; i<pCat->m_vpSequences.size(); ++i) {
        AdvancedMetrics temp_seqmetrics(pCat->m_vpSequences[i]);
        std::string sName = pCat->m_vpSequences[i]->m_sName;
        if(sName.size()>10)
            sName = sName.substr(0,10);
        else if(sName.size()<10)
            sName += std::string(10-sName.size(),' ');
        oMetricsOutput << sName << " " << temp_seqmetrics.dRecall << " " << temp_seqmetrics.dSpecficity << " " << temp_seqmetrics.dFPR << " " << temp_seqmetrics.dFNR << " " << temp_seqmetrics.dPBC << " " << temp_seqmetrics.dPrecision << " " << temp_seqmetrics.dFMeasure << " " << temp_seqmetrics.dMCC << std::endl;
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    oMetricsOutput << std::string(USE_AVERAGE_METRICS?"averaged   ":"cumulative ") << met.dRecall << " " << met.dSpecficity << " " << met.dFPR << " " << met.dFNR << " " << met.dPBC << " " << met.dPrecision << " " << met.dFMeasure << " " << met.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "All Sequences Average FPS: " << met.dFPS << std::endl;
    oMetricsOutput.close();
}

static inline void WriteMetrics(const std::string sResultsFileName, std::vector<CategoryInfo*>& vpCat, double dTotalFPS) {
    std::ofstream oMetricsOutput(sResultsFileName);
    std::sort(vpCat.begin(),vpCat.end(),&CategoryInfo::compare);
    AdvancedMetrics met(vpCat,USE_AVERAGE_METRICS);
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Overall results :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << std::string(USE_AVERAGE_METRICS?"Averaged":"Cumulative") << " metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    for(size_t i=0; i<vpCat.size(); ++i) {
        if(!vpCat[i]->m_vpSequences.empty()) {
            AdvancedMetrics temp_met(vpCat[i],USE_AVERAGE_METRICS);
            std::string sName = vpCat[i]->m_sName;
            if(sName.size()>10)
                sName = sName.substr(0,10);
            else if(sName.size()<10)
                sName += std::string(10-sName.size(),' ');
            oMetricsOutput << sName << " " << temp_met.dRecall << " " << temp_met.dSpecficity << " " << temp_met.dFPR << " " << temp_met.dFNR << " " << temp_met.dPBC << " " << temp_met.dPrecision << " " << temp_met.dFMeasure << " " << temp_met.dMCC << std::endl;
        }
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    oMetricsOutput << "overall    " << met.dRecall << " " << met.dSpecficity << " " << met.dFPR << " " << met.dFNR << " " << met.dPBC << " " << met.dPrecision << " " << met.dFMeasure << " " << met.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "All Sequences Average FPS: " << met.dFPS << std::endl;
    oMetricsOutput << "Total FPS: " << dTotalFPS << std::endl;
    oMetricsOutput.close();
}

static inline void CalcMetricsFromResult(const cv::Mat& oSegmResFrame, const cv::Mat& oGTFrame, const cv::Mat& oROI, uint64_t& nTP, uint64_t& nTN, uint64_t& nFP, uint64_t& nFN, uint64_t& nSE) {
    CV_DbgAssert(oSegmResFrame.type()==CV_8UC1 && oGTFrame.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmResFrame.size()==oGTFrame.size() && oSegmResFrame.size()==oROI.size());
    const size_t step_row = oSegmResFrame.step.p[0];
    for(size_t i=0; i<(size_t)oSegmResFrame.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmResFrame.data+idx_nstep;
        const uchar* gt_step_ptr = oGTFrame.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j=0; j<oSegmResFrame.cols; ++j) {
            if( gt_step_ptr[j]!=VAL_OUTOFSCOPE &&
                gt_step_ptr[j]!=VAL_UNKNOWN &&
                roi_step_ptr[j]!=VAL_NEGATIVE ) {
                if(input_step_ptr[j]==VAL_POSITIVE) {
                    if(gt_step_ptr[j]==VAL_POSITIVE)
                        ++nTP;
                    else // gt_step_ptr[j]==VAL_NEGATIVE
                        ++nFP;
                }
                else { // input_step_ptr[j]==VAL_NEGATIVE
                    if(gt_step_ptr[j]==VAL_POSITIVE)
                        ++nFN;
                    else // gt_step_ptr[j]==VAL_NEGATIVE
                        ++nTN;
                }
                if(gt_step_ptr[j]==VAL_SHADOW) {
                    if(input_step_ptr[j]==VAL_POSITIVE)
                        ++nSE;
                }
            }
        }
    }
}

static inline cv::Mat GetColoredSegmFrameFromResult(const cv::Mat& oSegmResFrame, const cv::Mat& oGTFrame, const cv::Mat& oROI) {
    CV_DbgAssert(oSegmResFrame.type()==CV_8UC1 && oGTFrame.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmResFrame.size()==oGTFrame.size() && oSegmResFrame.size()==oROI.size());
    cv::Mat oResult(oSegmResFrame.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oSegmResFrame.step.p[0];
    for(size_t i=0; i<(size_t)oSegmResFrame.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmResFrame.data+idx_nstep;
        const uchar* gt_step_ptr = oGTFrame.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oSegmResFrame.cols; ++j) {
            if( gt_step_ptr[j]!=VAL_OUTOFSCOPE &&
                gt_step_ptr[j]!=VAL_UNKNOWN &&
                roi_step_ptr[j]!=VAL_NEGATIVE ) {
                if(input_step_ptr[j]==VAL_POSITIVE) {
                    if(gt_step_ptr[j]==VAL_POSITIVE)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==VAL_NEGATIVE)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==VAL_SHADOW) {
                        res_step_ptr[j*3+1] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                    else {
                        for(size_t c=0; c<3; ++c)
                            res_step_ptr[j*3+c] = UCHAR_MAX/3;
                    }
                }
                else { // input_step_ptr[j]==VAL_NEGATIVE
                    if(gt_step_ptr[j]==VAL_POSITIVE) {
                        res_step_ptr[j*3] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                }
            }
            else if(roi_step_ptr[j]==VAL_NEGATIVE) {
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
