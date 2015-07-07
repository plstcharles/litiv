#include "DatasetUtils.h"

#if DATASETUTILS_USE_PRECACHED_IO
#define CONSOLE_DEBUG             0
#define REQUEST_TIMEOUT_MS        1
#define QUERY_TIMEOUT_MS          10
#define MAX_CACHE_SIZE_GB         6L
#define MAX_CACHE_SIZE            (((MAX_CACHE_SIZE_GB*1024)*1024)*1024)
#if (!(defined(_M_X64) || defined(__amd64__)) && MAX_CACHE_SIZE_GB>2)
#error "Cache max size exceeds system limit (x86)."
#endif //(!(defined(_M_X64) || defined(__amd64__)) && MAX_CACHE_SIZE_GB>2)
#endif //DATASETUTILS_USE_PRECACHED_IO

DatasetUtils::DatasetInfo DatasetUtils::GetDatasetInfo(const DatasetUtils::eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirPath) {
    if(eDatasetID==DatasetUtils::eDataset_CDnet2012) {
        const DatasetUtils::DatasetInfo oCDnet2012DatasetInfo = {
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
        return oCDnet2012DatasetInfo;
    }
    else if(eDatasetID==DatasetUtils::eDataset_CDnet2014) {
        const DatasetUtils::DatasetInfo oCDnet2014DatasetInfo = {
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
        return oCDnet2014DatasetInfo;
    }
    else if(eDatasetID==DatasetUtils::eDataset_Wallflower) {
        const DatasetUtils::DatasetInfo oWallflowerDatasetInfo = {
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
        return oWallflowerDatasetInfo;
    }
    else if(eDatasetID==DatasetUtils::eDataset_PETS2001_D3TC1) {
        const DatasetUtils::DatasetInfo oPETS2001D3TC1DatasetInfo = {
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
        return oPETS2001D3TC1DatasetInfo;
    }
    else if(eDatasetID==DatasetUtils::eDataset_GenericTest) {
        const DatasetUtils::DatasetInfo oGenericTestDatasetInfo = {
            eDatasetID,
            sDatasetRootDirPath+"/avitest/",
            sDatasetRootDirPath+"/avitest/"+sResultsDirPath+"/",
            "",
            ".png",
            {"inf6803_tp1"},
            {},
            {},
            0,
        };
        return oGenericTestDatasetInfo;
    }
    else if(eDatasetID==DatasetUtils::eDataset_LITIV2012) {
        const DatasetUtils::DatasetInfo oLITIV2012DatasetInfo = {
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
        return oLITIV2012DatasetInfo;
    }
    else
        throw std::runtime_error(std::string("Unknown dataset type, cannot use predefined info struct."));
}

double DatasetUtils::CalcMetric_FMeasure(uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN) {
    const double dRecall = DatasetUtils::CalcMetric_Recall(nTP,nTN,nFP,nFN);
    const double dPrecision = DatasetUtils::CalcMetric_Precision(nTP,nTN,nFP,nFN);
    return (2.0*(dRecall*dPrecision)/(dRecall+dPrecision));
}
double DatasetUtils::CalcMetric_Recall(uint64_t nTP, uint64_t /*nTN*/, uint64_t /*nFP*/, uint64_t nFN) {return ((double)nTP/(nTP+nFN));}
double DatasetUtils::CalcMetric_Precision(uint64_t nTP, uint64_t /*nTN*/, uint64_t nFP, uint64_t /*nFN*/) {return ((double)nTP/(nTP+nFP));}
double DatasetUtils::CalcMetric_Specificity(uint64_t /*nTP*/, uint64_t nTN, uint64_t nFP, uint64_t /*nFN*/) {return ((double)nTN/(nTN+nFP));}
double DatasetUtils::CalcMetric_FalsePositiveRate(uint64_t /*nTP*/, uint64_t nTN, uint64_t nFP, uint64_t /*nFN*/) {return ((double)nFP/(nFP+nTN));}
double DatasetUtils::CalcMetric_FalseNegativeRate(uint64_t nTP, uint64_t /*nTN*/, uint64_t /*nFP*/, uint64_t nFN) {return ((double)nFN/(nTP+nFN));}
double DatasetUtils::CalcMetric_PercentBadClassifs(uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN) {return (100.0*(nFN+nFP)/(nTP+nFP+nFN+nTN));}
double DatasetUtils::CalcMetric_MatthewsCorrCoeff(uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN) {return ((((double)nTP*nTN)-(nFP*nFN))/sqrt(((double)nTP+nFP)*(nTP+nFN)*(nTN+nFP)*(nTN+nFN)));}

cv::Mat DatasetUtils::ReadResult( const std::string& sResultsPath,
                                  const std::string& sCatName,
                                  const std::string& sSeqName,
                                  const std::string& sResultPrefix,
                                  size_t nFrameIdx,
                                  const std::string& sResultSuffix) {
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nFrameIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << acBuffer.data() << sResultSuffix;
    return cv::imread(sResultFilePath.str(),cv::IMREAD_GRAYSCALE);
}

void DatasetUtils::WriteResult( const std::string& sResultsPath,
                                const std::string& sCatName,
                                const std::string& sSeqName,
                                const std::string& sResultPrefix,
                                size_t nFrameIdx,
                                const std::string& sResultSuffix,
                                const cv::Mat& oResult,
                                const std::vector<int>& vnComprParams) {
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nFrameIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << acBuffer.data() << sResultSuffix;
    cv::imwrite(sResultFilePath.str(),oResult,vnComprParams);
}

void DatasetUtils::WriteOnImage(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom) {
    cv::putText(oImg,sText,cv::Point(4,bBottom?(oImg.rows-15):15),cv::FONT_HERSHEY_PLAIN,1.2,vColor,2,cv::LINE_AA);
}

void DatasetUtils::WriteMetrics(const std::string sResultsFileName, const SequenceInfo& oSeq) {
    std::ofstream oMetricsOutput(sResultsFileName);
    MetricsCalculator tmp(oSeq);
    const std::string sCurrSeqName = oSeq.m_sName.size()>12?oSeq.m_sName.substr(0,12):oSeq.m_sName;
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " : Rcl=" << std::fixed << std::setprecision(4) << tmp.m_oMetrics.dRecall << " Prc=" << tmp.m_oMetrics.dPrecision << " FM=" << tmp.m_oMetrics.dFMeasure << " MCC=" << tmp.m_oMetrics.dMCC << std::endl;
    oMetricsOutput << "Results for sequence '" << oSeq.m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
    oMetricsOutput << oSeq.nTP << " " << oSeq.nFP << " " << oSeq.nFN << " " << oSeq.nTN << " " << oSeq.nSE << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Cumulative metrics :" << std::endl;
    oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    oMetricsOutput << tmp.m_oMetrics.dRecall << " " << tmp.m_oMetrics.dSpecficity << " " << tmp.m_oMetrics.dFPR << " " << tmp.m_oMetrics.dFNR << " " << tmp.m_oMetrics.dPBC << " " << tmp.m_oMetrics.dPrecision << " " << tmp.m_oMetrics.dFMeasure << " " << tmp.m_oMetrics.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Sequence FPS: " << oSeq.m_dAvgFPS << std::endl;
    oMetricsOutput.close();
}

void DatasetUtils::WriteMetrics(const std::string sResultsFileName, const CategoryInfo& oCat) {
    std::ofstream oMetricsOutput(sResultsFileName);
    oMetricsOutput << "Results for category '" << oCat.m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
    oMetricsOutput << oCat.nTP << " " << oCat.nFP << " " << oCat.nFN << " " << oCat.nTN << " " << oCat.nSE << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Sequence Metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    for(size_t i=0; i<oCat.m_vpSequences.size(); ++i) {
        MetricsCalculator tmp(*oCat.m_vpSequences[i]);
        std::string sName = oCat.m_vpSequences[i]->m_sName;
        if(sName.size()>10)
            sName = sName.substr(0,10);
        else if(sName.size()<10)
            sName += std::string(10-sName.size(),' ');
        oMetricsOutput << sName << " " << tmp.m_oMetrics.dRecall << " " << tmp.m_oMetrics.dSpecficity << " " << tmp.m_oMetrics.dFPR << " " << tmp.m_oMetrics.dFNR << " " << tmp.m_oMetrics.dPBC << " " << tmp.m_oMetrics.dPrecision << " " << tmp.m_oMetrics.dFMeasure << " " << tmp.m_oMetrics.dMCC << std::endl;
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    MetricsCalculator all(oCat,DATASETUTILS_USE_AVERAGE_EVAL_METRICS);
    const std::string sCurrCatName = oCat.m_sName.size()>12?oCat.m_sName.substr(0,12):oCat.m_sName;
    std::cout << "\t" << std::setfill(' ') << std::setw(12) << sCurrCatName << " : Rcl=" << std::fixed << std::setprecision(4) << all.m_oMetrics.dRecall << " Prc=" << all.m_oMetrics.dPrecision << " FM=" << all.m_oMetrics.dFMeasure << " MCC=" << all.m_oMetrics.dMCC << std::endl;
    oMetricsOutput << std::string(DATASETUTILS_USE_AVERAGE_EVAL_METRICS?"averaged   ":"cumulative ") << all.m_oMetrics.dRecall << " " << all.m_oMetrics.dSpecficity << " " << all.m_oMetrics.dFPR << " " << all.m_oMetrics.dFNR << " " << all.m_oMetrics.dPBC << " " << all.m_oMetrics.dPrecision << " " << all.m_oMetrics.dFMeasure << " " << all.m_oMetrics.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "All Sequences Average FPS: " << all.m_oMetrics.dFPS << std::endl;
    oMetricsOutput.close();
}

void DatasetUtils::WriteMetrics(const std::string sResultsFileName, const std::vector<std::shared_ptr<CategoryInfo>>& vpCat, double dTotalFPS) {
    std::ofstream oMetricsOutput(sResultsFileName);
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Overall results :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << std::string(DATASETUTILS_USE_AVERAGE_EVAL_METRICS?"Averaged":"Cumulative") << " metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    for(size_t i=0; i<vpCat.size(); ++i) {
        if(!vpCat[i]->m_vpSequences.empty()) {
            MetricsCalculator tmp(*vpCat[i],DATASETUTILS_USE_AVERAGE_EVAL_METRICS);
            std::string sName = vpCat[i]->m_sName;
            if(sName.size()>10)
                sName = sName.substr(0,10);
            else if(sName.size()<10)
                sName += std::string(10-sName.size(),' ');
            oMetricsOutput << sName << " " << tmp.m_oMetrics.dRecall << " " << tmp.m_oMetrics.dSpecficity << " " << tmp.m_oMetrics.dFPR << " " << tmp.m_oMetrics.dFNR << " " << tmp.m_oMetrics.dPBC << " " << tmp.m_oMetrics.dPrecision << " " << tmp.m_oMetrics.dFMeasure << " " << tmp.m_oMetrics.dMCC << std::endl;
        }
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    MetricsCalculator all(vpCat,DATASETUTILS_USE_AVERAGE_EVAL_METRICS);
    oMetricsOutput << "overall    " << all.m_oMetrics.dRecall << " " << all.m_oMetrics.dSpecficity << " " << all.m_oMetrics.dFPR << " " << all.m_oMetrics.dFNR << " " << all.m_oMetrics.dPBC << " " << all.m_oMetrics.dPrecision << " " << all.m_oMetrics.dFMeasure << " " << all.m_oMetrics.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "All Sequences Average FPS: " << all.m_oMetrics.dFPS << std::endl;
    oMetricsOutput << "Total FPS: " << dTotalFPS << std::endl;
    oMetricsOutput.close();
}

void DatasetUtils::CalcMetricsFromResult(const cv::Mat& oSegmResFrame, const cv::Mat& oGTFrame, const cv::Mat& oROI, uint64_t& nTP, uint64_t& nTN, uint64_t& nFP, uint64_t& nFN, uint64_t& nSE) {
    CV_DbgAssert(oSegmResFrame.type()==CV_8UC1 && oGTFrame.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmResFrame.size()==oGTFrame.size() && oSegmResFrame.size()==oROI.size());
    const size_t step_row = oSegmResFrame.step.p[0];
    for(size_t i=0; i<(size_t)oSegmResFrame.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmResFrame.data+idx_nstep;
        const uchar* gt_step_ptr = oGTFrame.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j=0; j<oSegmResFrame.cols; ++j) {
            if( gt_step_ptr[j]!=g_nCDnetOutOfScope &&
                gt_step_ptr[j]!=g_nCDnetUnknown &&
                roi_step_ptr[j]!=g_nCDnetNegative ) {
                if(input_step_ptr[j]==g_nCDnetPositive) {
                    if(gt_step_ptr[j]==g_nCDnetPositive)
                        ++nTP;
                    else // gt_step_ptr[j]==g_nCDnetNegative
                        ++nFP;
                }
                else { // input_step_ptr[j]==g_nCDnetNegative
                    if(gt_step_ptr[j]==g_nCDnetPositive)
                        ++nFN;
                    else // gt_step_ptr[j]==g_nCDnetNegative
                        ++nTN;
                }
                if(gt_step_ptr[j]==g_nCDnetShadow) {
                    if(input_step_ptr[j]==g_nCDnetPositive)
                        ++nSE;
                }
            }
        }
    }
}

inline DatasetUtils::CommonMetrics CalcMetricsFromCounts(uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN, uint64_t /*nSE*/, double dFPS) {
    DatasetUtils::CommonMetrics res;
    res.dRecall = DatasetUtils::CalcMetric_Recall(nTP,nTN,nFP,nFN);
    res.dSpecficity = DatasetUtils::CalcMetric_Specificity(nTP,nTN,nFP,nFN);
    res.dFPR = DatasetUtils::CalcMetric_FalsePositiveRate(nTP,nTN,nFP,nFN);
    res.dFNR = DatasetUtils::CalcMetric_FalseNegativeRate(nTP,nTN,nFP,nFN);
    res.dPBC = DatasetUtils::CalcMetric_PercentBadClassifs(nTP,nTN,nFP,nFN);
    res.dPrecision = DatasetUtils::CalcMetric_Precision(nTP,nTN,nFP,nFN);
    res.dFMeasure = DatasetUtils::CalcMetric_FMeasure(nTP,nTN,nFP,nFN);
    res.dMCC = DatasetUtils::CalcMetric_MatthewsCorrCoeff(nTP,nTN,nFP,nFN);
    res.dFPS = dFPS;
    return res;
}

inline DatasetUtils::CommonMetrics CalcMetricsFromCategory(const DatasetUtils::CategoryInfo& oCat, bool bAverage) {
    DatasetUtils::CommonMetrics res;
    if(!bAverage) {
        res.dRecall = DatasetUtils::CalcMetric_Recall(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dSpecficity = DatasetUtils::CalcMetric_Specificity(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dFPR = DatasetUtils::CalcMetric_FalsePositiveRate(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dFNR = DatasetUtils::CalcMetric_FalseNegativeRate(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dPBC = DatasetUtils::CalcMetric_PercentBadClassifs(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dPrecision = DatasetUtils::CalcMetric_Precision(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dFMeasure = DatasetUtils::CalcMetric_FMeasure(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dMCC = DatasetUtils::CalcMetric_MatthewsCorrCoeff(oCat.nTP,oCat.nTN,oCat.nFP,oCat.nFN);
        res.dFPS = oCat.m_dAvgFPS;
    }
    else {
        res.dRecall = 0;
        res.dSpecficity = 0;
        res.dFPR = 0;
        res.dFNR = 0;
        res.dPBC = 0;
        res.dPrecision = 0;
        res.dFMeasure = 0;
        res.dMCC = 0;
        res.dFPS = 0;
        const size_t nSeq = oCat.m_vpSequences.size();
        for(size_t i=0; i<nSeq; ++i) {
            const DatasetUtils::SequenceInfo& oCurrSeq = *oCat.m_vpSequences[i];
            res.dRecall += DatasetUtils::CalcMetric_Recall(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dSpecficity += DatasetUtils::CalcMetric_Specificity(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dFPR += DatasetUtils::CalcMetric_FalsePositiveRate(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dFNR += DatasetUtils::CalcMetric_FalseNegativeRate(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dPBC += DatasetUtils::CalcMetric_PercentBadClassifs(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dPrecision += DatasetUtils::CalcMetric_Precision(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dFMeasure += DatasetUtils::CalcMetric_FMeasure(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dMCC += DatasetUtils::CalcMetric_MatthewsCorrCoeff(oCurrSeq.nTP,oCurrSeq.nTN,oCurrSeq.nFP,oCurrSeq.nFN);
            res.dFPS += oCurrSeq.m_dAvgFPS;
        }
        res.dRecall /= nSeq;
        res.dSpecficity /= nSeq;
        res.dFPR /= nSeq;
        res.dFNR /= nSeq;
        res.dPBC /= nSeq;
        res.dPrecision /= nSeq;
        res.dFMeasure /= nSeq;
        res.dMCC /= nSeq;
        res.dFPS /= nSeq;
    }
    return res;
}

inline DatasetUtils::CommonMetrics CalcMetricsFromCategories(const std::vector<std::shared_ptr<DatasetUtils::CategoryInfo>>& vpCat, bool bAverage) {
    DatasetUtils::CommonMetrics res;
    const size_t nCat = vpCat.size();
    size_t nBadCat = 0;
    if(!bAverage) {
        uint64_t nGlobalTP=0, nGlobalTN=0, nGlobalFP=0, nGlobalFN=0, nGlobalSE=0;
        res.dFPS=0;
        for(size_t i=0; i<nCat; ++i) {
            if(vpCat[i]->m_vpSequences.empty()) {
                ++nBadCat;
            }
            else {
                nGlobalTP += vpCat[i]->nTP;
                nGlobalTN += vpCat[i]->nTN;
                nGlobalFP += vpCat[i]->nFP;
                nGlobalFN += vpCat[i]->nFN;
                nGlobalSE += vpCat[i]->nSE;
                res.dFPS += vpCat[i]->m_dAvgFPS;
            }
        }
        CV_Assert(nBadCat<nCat);
        res.dRecall = DatasetUtils::CalcMetric_Recall(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dSpecficity = DatasetUtils::CalcMetric_Specificity(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dFPR = DatasetUtils::CalcMetric_FalsePositiveRate(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dFNR = DatasetUtils::CalcMetric_FalseNegativeRate(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dPBC = DatasetUtils::CalcMetric_PercentBadClassifs(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dPrecision = DatasetUtils::CalcMetric_Precision(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dFMeasure = DatasetUtils::CalcMetric_FMeasure(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dMCC = DatasetUtils::CalcMetric_MatthewsCorrCoeff(nGlobalTP,nGlobalTN,nGlobalFP,nGlobalFN);
        res.dFPS /= (nCat-nBadCat);
    }
    else {
        res.dRecall = 0;
        res.dSpecficity = 0;
        res.dFPR = 0;
        res.dFNR = 0;
        res.dPBC = 0;
        res.dPrecision = 0;
        res.dFMeasure = 0;
        res.dMCC = 0;
        res.dFPS = 0;
        for(size_t i=0; i<nCat; ++i) {
            if(vpCat[i]->m_vpSequences.empty())
                ++nBadCat;
            else {
                DatasetUtils::CommonMetrics curr = CalcMetricsFromCategory(*vpCat[i],true);
                res.dRecall += curr.dRecall;
                res.dSpecficity += curr.dSpecficity;
                res.dFPR += curr.dFPR;
                res.dFNR += curr.dFNR;
                res.dPBC += curr.dPBC;
                res.dPrecision += curr.dPrecision;
                res.dFMeasure += curr.dFMeasure;
                res.dMCC += curr.dMCC;
                res.dFPS += curr.dFPS;
            }
        }
        CV_Assert(nBadCat<nCat);
        res.dRecall /= (nCat-nBadCat);
        res.dSpecficity /= (nCat-nBadCat);
        res.dFPR /= (nCat-nBadCat);
        res.dFNR /= (nCat-nBadCat);
        res.dPBC /= (nCat-nBadCat);
        res.dPrecision /= (nCat-nBadCat);
        res.dFMeasure /= (nCat-nBadCat);
        res.dMCC /= (nCat-nBadCat);
        res.dFPS /= (nCat-nBadCat);
    }
    return res;
}

cv::Mat DatasetUtils::GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oFGMask, const cv::Mat& oGTFGMask, const cv::Mat& oROI, size_t nFrame, cv::Point oDbgPt) {
    cv::Mat oInputImgBYTE3, oBGImgBYTE3, oFGMaskBYTE3;
    CV_Assert(!oInputImg.empty() && (oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3 || oInputImg.type()==CV_8UC4));
    CV_Assert(!oBGImg.empty() && (oBGImg.type()==CV_8UC1 || oBGImg.type()==CV_8UC3 || oBGImg.type()==CV_8UC4));
    CV_Assert(!oFGMask.empty() && oFGMask.type()==CV_8UC1);
    CV_Assert(!oGTFGMask.empty() && oGTFGMask.type()==CV_8UC1);
    CV_Assert(!oROI.empty() && oROI.type()==CV_8UC1);
    if(oInputImg.channels()==1) {
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_GRAY2RGB);
        cv::cvtColor(oBGImg,oBGImgBYTE3,cv::COLOR_GRAY2RGB);
    }
    else if(oInputImg.channels()==4) {
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_RGBA2RGB);
        cv::cvtColor(oBGImg,oBGImgBYTE3,cv::COLOR_RGBA2RGB);
    }
    else {
        oInputImgBYTE3 = oInputImg;
        oBGImgBYTE3 = oBGImg;
    }
    oFGMaskBYTE3 = DatasetUtils::GetColoredSegmFrameFromResult(oFGMask,oGTFGMask,oROI);
    if(oDbgPt!=cv::Point(-1,-1)) {
        cv::circle(oInputImgBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
        cv::circle(oFGMaskBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
    }
    cv::Mat displayH,displayV1,displayV2;
    cv::resize(oInputImgBYTE3,oInputImgBYTE3,cv::Size(320,240));
    cv::resize(oBGImgBYTE3,oBGImgBYTE3,cv::Size(320,240));
    cv::resize(oFGMaskBYTE3,oFGMaskBYTE3,cv::Size(320,240));

    std::stringstream sstr;
    sstr << "Frame #" << nFrame;
    DatasetUtils::WriteOnImage(oInputImgBYTE3,sstr.str(),cv::Scalar_<uchar>(0,0,255));
    DatasetUtils::WriteOnImage(oBGImgBYTE3,"BG Reference",cv::Scalar_<uchar>(0,0,255));
    DatasetUtils::WriteOnImage(oFGMaskBYTE3,"Segmentation Result",cv::Scalar_<uchar>(0,0,255));

    cv::hconcat(oInputImgBYTE3,oBGImgBYTE3,displayH);
    cv::hconcat(displayH,oFGMaskBYTE3,displayH);
    return displayH;
}

cv::Mat DatasetUtils::GetColoredSegmFrameFromResult(const cv::Mat& oSegmResFrame, const cv::Mat& oGTFrame, const cv::Mat& oROI) {
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
            if( gt_step_ptr[j]!=g_nCDnetOutOfScope &&
                gt_step_ptr[j]!=g_nCDnetUnknown &&
                roi_step_ptr[j]!=g_nCDnetNegative ) {
                if(input_step_ptr[j]==g_nCDnetPositive) {
                    if(gt_step_ptr[j]==g_nCDnetPositive)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==g_nCDnetNegative)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==g_nCDnetShadow) {
                        res_step_ptr[j*3+1] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                    else {
                        for(size_t c=0; c<3; ++c)
                            res_step_ptr[j*3+c] = UCHAR_MAX/3;
                    }
                }
                else { // input_step_ptr[j]==g_nCDnetNegative
                    if(gt_step_ptr[j]==g_nCDnetPositive) {
                        res_step_ptr[j*3] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                }
            }
            else if(roi_step_ptr[j]==g_nCDnetNegative) {
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

DatasetUtils::MetricsCalculator::MetricsCalculator(uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN, uint64_t nSE)
    :    m_oMetrics(CalcMetricsFromCounts(nTP,nTN,nFP,nFN,nSE,0)),m_bAveraged(false) {}

DatasetUtils::MetricsCalculator::MetricsCalculator(const SequenceInfo& oSeq)
    :    m_oMetrics(CalcMetricsFromCounts(oSeq.nTP,oSeq.nTN,oSeq.nFP,oSeq.nFN,oSeq.nSE,oSeq.m_dAvgFPS)),m_bAveraged(false) {}

DatasetUtils::MetricsCalculator::MetricsCalculator(const CategoryInfo& oCat, bool bAverage)
    :    m_oMetrics(CalcMetricsFromCategory(oCat,bAverage)),m_bAveraged(bAverage) {CV_Assert(!oCat.m_vpSequences.empty());}

DatasetUtils::MetricsCalculator::MetricsCalculator(const std::vector<std::shared_ptr<CategoryInfo>>& vpCat, bool bAverage)
    :    m_oMetrics(CalcMetricsFromCategories(vpCat,bAverage)),m_bAveraged(bAverage) {CV_Assert(!vpCat.empty());}

DatasetUtils::CategoryInfo::CategoryInfo(const std::string& sName, const std::string& sDirectoryPath,
                                         DatasetUtils::eDatasetList eDatasetID,
                                         const std::vector<std::string>& vsGrayscaleDirNameTokens,
                                         const std::vector<std::string>& vsSkippedDirNameTokens,
                                         bool bUse4chAlign)
    :    m_sName(sName)
        ,m_eDatasetID(eDatasetID)
        ,nTP(0),nTN(0),nFP(0),nFN(0),nSE(0)
        ,m_dAvgFPS(-1) {
    std::cout << "\tParsing dir '" << sDirectoryPath << "' for category '" << m_sName << "'; ";
    std::vector<std::string> vsSequencePaths;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014 || m_eDatasetID==eDataset_Wallflower || m_eDatasetID==eDataset_PETS2001_D3TC1) {
        // all subdirs are considered sequence directories
        PlatformUtils::GetSubDirsFromDir(sDirectoryPath,vsSequencePaths);
        std::cout << vsSequencePaths.size() << " potential sequence(s)" << std::endl;
    }
    else if(m_eDatasetID==eDataset_LITIV2012) {
        // all subdirs should contain individual video tracks in separate modalities
        PlatformUtils::GetSubDirsFromDir(sDirectoryPath,vsSequencePaths);
        std::cout << vsSequencePaths.size() << " potential track(s)" << std::endl;
    }
    else if(m_eDatasetID==eDataset_GenericTest) {
        // all files are considered sequences
        PlatformUtils::GetFilesFromDir(sDirectoryPath,vsSequencePaths);
        std::cout << vsSequencePaths.size() << " potential sequence(s)" << std::endl;
    }
    else
        throw std::runtime_error(std::string("Unknown dataset type, cannot use any known parsing strategy."));
    for(auto iter=vsSequencePaths.begin(); iter!=vsSequencePaths.end(); ++iter) {
        bool bForceGrayscale = false, bSkip = false;
        for(size_t i=0; i<vsGrayscaleDirNameTokens.size() && !bForceGrayscale; ++i)
            bForceGrayscale = iter->find(vsGrayscaleDirNameTokens[i])!=std::string::npos;
        for(size_t i=0; i<vsSkippedDirNameTokens.size() && !bSkip; ++i)
            bSkip = iter->find(vsSkippedDirNameTokens[i])!=std::string::npos;
        if(!bSkip) {
            const size_t pos = iter->find_last_of("/\\");
            if(pos==std::string::npos)
                m_vpSequences.push_back(std::make_shared<SequenceInfo>(*iter,*iter,this,bForceGrayscale,bUse4chAlign));
            else
                m_vpSequences.push_back(std::make_shared<SequenceInfo>(iter->substr(pos+1),*iter,this,bForceGrayscale,bUse4chAlign));
        }
    }
}

DatasetUtils::SequenceInfo::SequenceInfo(const std::string& sName, const std::string& sPath, CategoryInfo* pParent, bool bForceGrayscale, bool bUse4chAlign)
    :    m_sName(sName)
        ,m_sPath(sPath)
        ,m_eDatasetID(pParent?pParent->m_eDatasetID:DatasetUtils::eDataset_GenericTest)
        ,nTP(0),nTN(0),nFP(0),nFN(0),nSE(0)
        ,m_dAvgFPS(-1)
        ,m_dExpectedLoad(0)
        ,m_dExpectedROILoad(0)
        ,m_pParent(pParent)
#if DATASETUTILS_USE_PRECACHED_IO
        ,m_bIsPrecaching(false)
        ,m_nInputFrameSize(0)
        ,m_nGTFrameSize(0)
        ,m_nInputBufferSize(0)
        ,m_nGTBufferSize(0)
        ,m_nInputPrecacheSize(0)
        ,m_nGTPrecacheSize(0)
        ,m_nInputBufferFrameCount(0)
        ,m_nGTBufferFrameCount(0)
        ,m_nRequestInputFrameIndex(SIZE_MAX)
        ,m_nRequestGTFrameIndex(SIZE_MAX)
        ,m_nNextInputBufferIdx(0)
        ,m_nNextGTBufferIdx(0)
        ,m_nNextExpectedInputFrameIdx(0)
        ,m_nNextExpectedGTFrameIdx(0)
        ,m_nNextPrecachedInputFrameIdx(0)
        ,m_nNextPrecachedGTFrameIdx(0)
#else //!DATASETUTILS_USE_PRECACHED_IO
        ,m_nLastReqInputFrameIndex(UINT_MAX)
        ,m_nLastReqGTFrameIndex(UINT_MAX)
#endif //!DATASETUTILS_USE_PRECACHED_IO
        ,m_nNextExpectedVideoReaderFrameIdx(0)
        ,m_nTotalNbFrames(0)
        ,m_bForcingGrayscale(bForceGrayscale)
        ,m_bUsing4chAlignment(bUse4chAlign)
        ,m_nIMReadInputFlags(bForceGrayscale?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR) {
#if DATASETUTILS_USE_PRECACHED_IO
    CV_Assert(MAX_CACHE_SIZE>0);
    CV_Assert(REQUEST_TIMEOUT_MS>0);
    CV_Assert(QUERY_TIMEOUT_MS>0);
#endif //DATASETUTILS_USE_PRECACHED_IO
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014) {
        std::vector<std::string> vsSubDirs;
        PlatformUtils::GetSubDirsFromDir(m_sPath,vsSubDirs);
        auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),m_sPath+"/groundtruth");
        auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),m_sPath+"/input");
        if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess the required groundtruth and input directories.");
        PlatformUtils::GetFilesFromDir(*inputDir,m_vsInputFramePaths);
        PlatformUtils::GetFilesFromDir(*gtDir,m_vsGTFramePaths);
        if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess same amount of GT & input frames.");
        m_oROI = cv::imread(m_sPath+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
        if(m_oROI.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess a ROI.bmp file.");
        m_oROI = m_oROI>0;
        m_oSize = m_oROI.size();
        m_nTotalNbFrames = m_vsInputFramePaths.size();
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotalNbFrames*(int(!m_bForcingGrayscale)+1);
        m_dExpectedROILoad = (double)cv::countNonZero(m_oROI)*m_nTotalNbFrames*(int(!m_bForcingGrayscale)+1);
        // note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
    }
    else if(m_eDatasetID==eDataset_Wallflower) {
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
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess the required groundtruth and input files.");
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess a valid GT file.");
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(g_nCDnetPositive));
        m_oSize = oTempImg.size();
        m_nTotalNbFrames = m_vsInputFramePaths.size();
        m_dExpectedLoad = m_dExpectedROILoad = (double)m_oSize.height*m_oSize.width*m_nTotalNbFrames*(int(!m_bForcingGrayscale)+1);
    }
    else if(m_eDatasetID==eDataset_PETS2001_D3TC1) {
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
            throw std::runtime_error(std::string("Bad video file ('")+vsVideoSeqPaths[0]+std::string("'), could not be opened."));
        PlatformUtils::GetFilesFromDir(vsGTSubdirPaths[0],m_vsGTFramePaths);
        if(m_vsGTFramePaths.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess any valid GT frames.");
        const std::string sGTFilePrefix("image_");
        const size_t nInputFileNbDecimals = 4;
        for(auto iter=m_vsGTFramePaths.begin(); iter!=m_vsGTFramePaths.end(); ++iter)
            m_mTestGTIndexes.insert(std::pair<size_t,size_t>(atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),iter-m_vsGTFramePaths.begin()));
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess valid GT file(s).");
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(g_nCDnetPositive));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        m_nTotalNbFrames = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        CV_Assert(m_nTotalNbFrames>0);
        m_dExpectedLoad = m_dExpectedROILoad = (double)m_oSize.height*m_oSize.width*m_nTotalNbFrames*(int(!m_bForcingGrayscale)+1);
    }
    else if(m_eDatasetID==eDataset_LITIV2012) {
        PlatformUtils::GetFilesFromDir(m_sPath+"/input/",m_vsInputFramePaths);
        if(m_vsInputFramePaths.empty())
            throw std::runtime_error(std::string("Sequence at ") + m_sPath + " did not possess any parsable input images.");
        cv::Mat oTempImg = cv::imread(m_vsInputFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Bad image file ('")+m_vsInputFramePaths[0]+"'), could not be read.");
        /*m_voVideoReader.open(m_sPath+"/input/in%06d.jpg");
        if(!m_voVideoReader.isOpened())
            m_voVideoReader.open(m_sPath+"/"+m_sName+".avi");
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be opened."));
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        cv::Mat oTempImg;
        m_voVideoReader >> oTempImg;
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be read."));*/
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(g_nCDnetPositive));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = -1;
        //m_nTotalNbFrames = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        m_nTotalNbFrames = m_vsInputFramePaths.size();
        m_dExpectedLoad = m_dExpectedROILoad = (double)m_oSize.height*m_oSize.width*m_nTotalNbFrames*(int(!m_bForcingGrayscale)+1);
    }
    else if(m_eDatasetID==eDataset_GenericTest) {
        m_voVideoReader.open(m_sPath);
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be opened."));
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        cv::Mat oTempImg;
        m_voVideoReader >> oTempImg;
        m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
        if(oTempImg.empty())
            throw std::runtime_error(std::string("Bad video file ('")+m_sPath+std::string("'), could not be read."));
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar(g_nCDnetPositive));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        m_nTotalNbFrames = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        CV_Assert(m_nTotalNbFrames>0);
        m_dExpectedLoad = m_dExpectedROILoad = (double)m_oSize.height*m_oSize.width*m_nTotalNbFrames*(int(!m_bForcingGrayscale)+1);
    }
    else
        throw std::runtime_error(std::string("Unknown dataset type, cannot use any known parsing strategy."));
}

DatasetUtils::SequenceInfo::~SequenceInfo() {
#if DATASETUTILS_USE_PRECACHED_IO
    StopPrecaching();
#endif //DATASETUTILS_USE_PRECACHED_IO
}

size_t DatasetUtils::SequenceInfo::GetNbInputFrames() const {
    return m_nTotalNbFrames;
}

size_t DatasetUtils::SequenceInfo::GetNbGTFrames() const {
    return m_mTestGTIndexes.empty()?m_vsGTFramePaths.size():m_mTestGTIndexes.size();
}

cv::Size DatasetUtils::SequenceInfo::GetFrameSize() const {
    return m_oSize;
}

const cv::Mat& DatasetUtils::SequenceInfo::GetSequenceROI() const {
    return m_oROI;
}

void DatasetUtils::SequenceInfo::ValidateKeyPoints(std::vector<cv::KeyPoint>& voKPs) const {
    std::vector<cv::KeyPoint> voNewKPs;
    for(size_t k=0; k<voKPs.size(); ++k) {
        if(m_oROI.at<uchar>(voKPs[k].pt)>0)
            voNewKPs.push_back(voKPs[k]);
    }
    voKPs = voNewKPs;
}

cv::Mat DatasetUtils::SequenceInfo::GetInputFrameFromIndex_Internal(size_t nFrameIdx) {
    CV_Assert(nFrameIdx<m_nTotalNbFrames);
    cv::Mat oFrame;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014 || m_eDatasetID==eDataset_Wallflower || m_eDatasetID==eDataset_LITIV2012)
        oFrame = cv::imread(m_vsInputFramePaths[nFrameIdx],m_nIMReadInputFlags);
    else if(m_eDatasetID==eDataset_PETS2001_D3TC1 || /*m_eDatasetID==eDataset_LITIV2012 || */m_eDatasetID==eDataset_GenericTest) {
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

cv::Mat DatasetUtils::SequenceInfo::GetGTFrameFromIndex_Internal(size_t nFrameIdx) {
    CV_Assert(nFrameIdx<m_nTotalNbFrames);
    cv::Mat oFrame;
    if(m_eDatasetID==eDataset_CDnet2012 || m_eDatasetID==eDataset_CDnet2014)
        oFrame = cv::imread(m_vsGTFramePaths[nFrameIdx],cv::IMREAD_GRAYSCALE);
    else if(m_eDatasetID==eDataset_Wallflower || m_eDatasetID==eDataset_PETS2001_D3TC1) {
        auto res = m_mTestGTIndexes.find(nFrameIdx);
        if(res!=m_mTestGTIndexes.end())
            oFrame = cv::imread(m_vsGTFramePaths[res->second],cv::IMREAD_GRAYSCALE);
        else
            oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar(g_nCDnetOutOfScope));
    }
    else if(m_eDatasetID==eDataset_LITIV2012 || m_eDatasetID==eDataset_GenericTest) {
        oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar(g_nCDnetOutOfScope));
    }
    CV_Assert(oFrame.size()==m_oSize);
#if DATASETUTILS_HARDCODE_FRAME_INDEX
    std::stringstream sstr;
    sstr << "Frame #" << nFrameIdx;
    DatasetUtils::WriteOnImage(oFrame,sstr.str(),cv::Scalar::all(255));
#endif //DATASETUTILS_HARDCODE_FRAME_INDEX
    return oFrame;
}

const cv::Mat& DatasetUtils::SequenceInfo::GetInputFrameFromIndex(size_t nFrameIdx) {
#if DATASETUTILS_USE_PRECACHED_IO
    if(!m_bIsPrecaching)
        throw std::runtime_error(m_sName + " [SequenceInfo] : Error, queried a frame before precaching was activated.");
    std::unique_lock<std::mutex> sync_lock(m_oInputFrameSyncMutex);
    m_nRequestInputFrameIndex = nFrameIdx;
    std::cv_status res;
    do {
        m_oInputFrameReqCondVar.notify_one();
        res = m_oInputFrameSyncCondVar.wait_for(sync_lock,std::chrono::milliseconds(REQUEST_TIMEOUT_MS));
#if CONSOLE_DEBUG
        if(res==std::cv_status::timeout)
            std::cout << " # retrying request..." << std::endl;
#endif //CONSOLE_DEBUG
    } while(res==std::cv_status::timeout);
    return m_oReqInputFrame;
#else //!DATASETUTILS_USE_PRECACHED_IO
    if(m_nLastReqInputFrameIndex!=nFrameIdx) {
        m_oLastReqInputFrame = GetInputFrameFromIndex_Internal(nFrameIdx);
        m_nLastReqInputFrameIndex = nFrameIdx;
    }
    return m_oLastReqInputFrame;
#endif //!DATASETUTILS_USE_PRECACHED_IO
}

const cv::Mat& DatasetUtils::SequenceInfo::GetGTFrameFromIndex(size_t nFrameIdx) {
#if DATASETUTILS_USE_PRECACHED_IO
    if(!m_bIsPrecaching)
        throw std::runtime_error(m_sName + " [SequenceInfo] : Error, queried a frame before precaching was activated.");
    std::unique_lock<std::mutex> sync_lock(m_oGTFrameSyncMutex);
    m_nRequestGTFrameIndex = nFrameIdx;
    std::cv_status res;
    do {
        m_oGTFrameReqCondVar.notify_one();
        res = m_oGTFrameSyncCondVar.wait_for(sync_lock,std::chrono::milliseconds(REQUEST_TIMEOUT_MS));
#if CONSOLE_DEBUG
        if(res==std::cv_status::timeout)
            std::cout << " # retrying request..." << std::endl;
#endif //CONSOLE_DEBUG
    } while(res==std::cv_status::timeout);
    return m_oReqGTFrame;
#else //!DATASETUTILS_USE_PRECACHED_IO
    if(m_nLastReqGTFrameIndex!=nFrameIdx) {
        m_oLastReqGTFrame = GetGTFrameFromIndex_Internal(nFrameIdx);
        m_nLastReqGTFrameIndex = nFrameIdx;
    }
    return m_oLastReqGTFrame;
#endif //!DATASETUTILS_USE_PRECACHED_IO
}

#if DATASETUTILS_USE_PRECACHED_IO

void DatasetUtils::SequenceInfo::PrecacheInputFrames() {
    std::unique_lock<std::mutex> sync_lock(m_oInputFrameSyncMutex);
#if CONSOLE_DEBUG
    std::cout << " @ initializing precaching with " << m_nInputBufferFrameCount << " frames " << std::endl;
#endif //CONSOLE_DEBUG
    while(m_qoInputFrameCache.size()<m_nInputBufferFrameCount && m_nNextPrecachedInputFrameIdx<m_nTotalNbFrames) {
        cv::Mat oNextInputFrame = GetInputFrameFromIndex_Internal(m_nNextPrecachedInputFrameIdx++);
        cv::Mat oNextInputFrame_precached(m_oSize,m_bUsing4chAlignment?CV_8UC4:m_bForcingGrayscale?CV_8UC1:CV_8UC3,m_vcInputBuffer.data()+m_nNextInputBufferIdx);
        // @@@@@@@@ try to fetch without copy to?
        oNextInputFrame.copyTo(oNextInputFrame_precached);
        m_qoInputFrameCache.push_back(oNextInputFrame_precached);
        m_nNextInputBufferIdx += m_nInputFrameSize;
        m_nNextInputBufferIdx %= m_nInputBufferSize;
    }
    while(m_bIsPrecaching) {
        if(m_oInputFrameReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(m_nNextPrecachedInputFrameIdx==m_nTotalNbFrames?QUERY_TIMEOUT_MS*32:QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
            CV_DbgAssert(m_nRequestInputFrameIndex<m_nTotalNbFrames);
            if(m_nRequestInputFrameIndex!=m_nNextExpectedInputFrameIdx-1) {
                if(!m_qoInputFrameCache.empty() && m_nRequestInputFrameIndex==m_nNextExpectedInputFrameIdx) {
                    m_oReqInputFrame = m_qoInputFrameCache.front();
                    m_qoInputFrameCache.pop_front();
                }
                else {
                    if(!m_qoInputFrameCache.empty()) {
#if CONSOLE_DEBUG
                        std::cout << " @ answering request manually, out of order (req=" << m_nRequestInputFrameIndex << ", expected=" << m_nNextExpectedInputFrameIdx <<") ";
#endif //CONSOLE_DEBUG
                        CV_DbgAssert((m_nNextPrecachedInputFrameIdx-m_qoInputFrameCache.size())==m_nNextExpectedInputFrameIdx);
                        if(m_nRequestInputFrameIndex<m_nNextPrecachedInputFrameIdx && m_nRequestInputFrameIndex>m_nNextExpectedInputFrameIdx) {
#if CONSOLE_DEBUG
                            std::cout << " -- popping " << m_nRequestInputFrameIndex-m_nNextExpectedInputFrameIdx << " item(s) from cache" << std::endl;
#endif //CONSOLE_DEBUG
                            while(m_nRequestInputFrameIndex-m_nNextExpectedInputFrameIdx>0) {
                                m_qoInputFrameCache.pop_front();
                                ++m_nNextExpectedInputFrameIdx;
                            }
                            m_oReqInputFrame = m_qoInputFrameCache.front();
                            m_qoInputFrameCache.pop_front();
                        }
                        else {
#if CONSOLE_DEBUG
                            std::cout << " -- destroying cache" << std::endl;
#endif //CONSOLE_DEBUG
                            m_qoInputFrameCache.clear();
                            m_oReqInputFrame = GetInputFrameFromIndex_Internal(m_nRequestInputFrameIndex);
                            m_nNextPrecachedInputFrameIdx = m_nRequestInputFrameIndex+1;
                        }
                    }
                    else {
#if CONSOLE_DEBUG
                        std::cout << " @ answering request manually, precaching is falling behind" << std::endl;
#endif //CONSOLE_DEBUG
                        m_oReqInputFrame = GetInputFrameFromIndex_Internal(m_nRequestInputFrameIndex);
                        m_nNextPrecachedInputFrameIdx = m_nRequestInputFrameIndex+1;
                    }
                }
            }
#if CONSOLE_DEBUG
            else
                std::cout << " @ answering request using last frame" << std::endl;
#endif //CONSOLE_DEBUG
            m_nNextExpectedInputFrameIdx = m_nRequestInputFrameIndex+1;
            m_oInputFrameSyncCondVar.notify_one();
        }
        else {
            CV_DbgAssert((m_nNextPrecachedInputFrameIdx-m_nNextExpectedInputFrameIdx)==m_qoInputFrameCache.size());
            if(m_qoInputFrameCache.size()<m_nInputBufferFrameCount/4 && m_nNextPrecachedInputFrameIdx<m_nTotalNbFrames) {
#if CONSOLE_DEBUG
                std::cout << " @ filling precache buffer... (" << m_nInputBufferFrameCount-m_qoInputFrameCache.size() << " frames)" << std::endl;
#endif //CONSOLE_DEBUG
                size_t nFillCount = 0;
                while(m_qoInputFrameCache.size()<m_nInputBufferFrameCount && m_nNextPrecachedInputFrameIdx<m_nTotalNbFrames && nFillCount<10) {
                   cv::Mat oNextInputFrame = GetInputFrameFromIndex_Internal(m_nNextPrecachedInputFrameIdx++);
                   cv::Mat oNextInputFrame_precached(m_oSize,m_bUsing4chAlignment?CV_8UC4:m_bForcingGrayscale?CV_8UC1:CV_8UC3,m_vcInputBuffer.data()+m_nNextInputBufferIdx);
                   // @@@@@@@@ try to fetch without copy to?
                   oNextInputFrame.copyTo(oNextInputFrame_precached);
                   m_qoInputFrameCache.push_back(oNextInputFrame_precached);
                   m_nNextInputBufferIdx += m_nInputFrameSize;
                   m_nNextInputBufferIdx %= m_nInputBufferSize;
               }
            }
        }
    }
}

void DatasetUtils::SequenceInfo::PrecacheGTFrames() {
    std::unique_lock<std::mutex> sync_lock(m_oGTFrameSyncMutex);
#if CONSOLE_DEBUG
    std::cout << " @ initializing precaching with " << m_nGTBufferFrameCount << " frames " << std::endl;
#endif //CONSOLE_DEBUG
    while(m_qoGTFrameCache.size()<m_nGTBufferFrameCount && m_nNextPrecachedGTFrameIdx<m_nTotalNbFrames) {
        cv::Mat oNextGTFrame = GetGTFrameFromIndex_Internal(m_nNextPrecachedGTFrameIdx++);
        cv::Mat oNextGTFrame_precached(m_oSize,CV_8UC1,m_vcGTBuffer.data()+m_nNextGTBufferIdx);
        // @@@@@@@@ try to fetch without copy to?
        oNextGTFrame.copyTo(oNextGTFrame_precached);
        m_qoGTFrameCache.push_back(oNextGTFrame_precached);
        m_nNextGTBufferIdx += m_nGTFrameSize;
        m_nNextGTBufferIdx %= m_nGTBufferSize;
    }
    while(m_bIsPrecaching) {
        if(m_oGTFrameReqCondVar.wait_for(sync_lock,std::chrono::milliseconds(m_nNextPrecachedGTFrameIdx==m_nTotalNbFrames?QUERY_TIMEOUT_MS*32:QUERY_TIMEOUT_MS))!=std::cv_status::timeout) {
            CV_DbgAssert(m_nRequestGTFrameIndex<m_nTotalNbFrames);
            if(m_nRequestGTFrameIndex!=m_nNextExpectedGTFrameIdx-1) {
                if(!m_qoGTFrameCache.empty() && m_nRequestGTFrameIndex==m_nNextExpectedGTFrameIdx) {
                    m_oReqGTFrame = m_qoGTFrameCache.front();
                    m_qoGTFrameCache.pop_front();
                }
                else {
                    if(!m_qoGTFrameCache.empty()) {
#if CONSOLE_DEBUG
                        std::cout << " @ answering request manually, out of order (req=" << m_nRequestGTFrameIndex << ", expected=" << m_nNextExpectedGTFrameIdx <<") ";
#endif //CONSOLE_DEBUG
                        CV_DbgAssert((m_nNextPrecachedGTFrameIdx-m_qoGTFrameCache.size())==m_nNextExpectedGTFrameIdx);
                        if(m_nRequestGTFrameIndex<m_nNextPrecachedGTFrameIdx && m_nRequestGTFrameIndex>m_nNextExpectedGTFrameIdx) {
#if CONSOLE_DEBUG
                            std::cout << " -- popping " << m_nRequestGTFrameIndex-m_nNextExpectedGTFrameIdx << " item(s) from cache" << std::endl;
#endif //CONSOLE_DEBUG
                            while(m_nRequestGTFrameIndex-m_nNextExpectedGTFrameIdx>0) {
                                m_qoGTFrameCache.pop_front();
                                ++m_nNextExpectedGTFrameIdx;
                            }
                            m_oReqGTFrame = m_qoGTFrameCache.front();
                            m_qoGTFrameCache.pop_front();
                        }
                        else {
#if CONSOLE_DEBUG
                            std::cout << " -- destroying cache" << std::endl;
#endif //CONSOLE_DEBUG
                            m_qoGTFrameCache.clear();
                            m_oReqGTFrame = GetGTFrameFromIndex_Internal(m_nRequestGTFrameIndex);
                            m_nNextPrecachedGTFrameIdx = m_nRequestGTFrameIndex+1;
                        }
                    }
                    else {
#if CONSOLE_DEBUG
                        std::cout << " @ answering request manually, precaching is falling behind" << std::endl;
#endif //CONSOLE_DEBUG
                        m_oReqGTFrame = GetGTFrameFromIndex_Internal(m_nRequestGTFrameIndex);
                        m_nNextPrecachedGTFrameIdx = m_nRequestGTFrameIndex+1;
                    }
                }
            }
#if CONSOLE_DEBUG
            else
                std::cout << " @ answering request using last frame" << std::endl;
#endif //CONSOLE_DEBUG
            m_nNextExpectedGTFrameIdx = m_nRequestGTFrameIndex+1;
            m_oGTFrameSyncCondVar.notify_one();
        }
        else {
            CV_DbgAssert((m_nNextPrecachedGTFrameIdx-m_nNextExpectedGTFrameIdx)==m_qoGTFrameCache.size());
            if(m_qoGTFrameCache.size()<m_nGTBufferFrameCount/4 && m_nNextPrecachedGTFrameIdx<m_nTotalNbFrames) {
#if CONSOLE_DEBUG
                std::cout << " @ filling precache buffer... (" << m_nGTBufferFrameCount-m_qoGTFrameCache.size() << " frames)" << std::endl;
#endif //CONSOLE_DEBUG
                size_t nFillCount = 0;
                while(m_qoGTFrameCache.size()<m_nGTBufferFrameCount && m_nNextPrecachedGTFrameIdx<m_nTotalNbFrames && nFillCount<10) {
                    cv::Mat oNextGTFrame = GetGTFrameFromIndex_Internal(m_nNextPrecachedGTFrameIdx++);
                    cv::Mat oNextGTFrame_precached(m_oSize,CV_8UC1,m_vcGTBuffer.data()+m_nNextGTBufferIdx);
                    // @@@@@@@@ try to fetch without copy to?
                    oNextGTFrame.copyTo(oNextGTFrame_precached);
                    m_qoGTFrameCache.push_back(oNextGTFrame_precached);
                    m_nNextGTBufferIdx += m_nGTFrameSize;
                    m_nNextGTBufferIdx %= m_nGTBufferSize;
                }
            }
        }
    }
}

void DatasetUtils::SequenceInfo::StartPrecaching() {
    if(!m_bIsPrecaching) {
        m_bIsPrecaching = true;
        m_nGTFrameSize = (size_t)(m_oSize.height*m_oSize.width);
        m_nGTPrecacheSize = m_nGTFrameSize*m_nTotalNbFrames;
        m_nInputFrameSize = m_nGTFrameSize*(m_bForcingGrayscale?1:m_bUsing4chAlignment?4:3);
        m_nInputPrecacheSize = m_nInputFrameSize*m_nTotalNbFrames;
        CV_Assert(m_nInputFrameSize>0 && m_nInputPrecacheSize);
        m_nInputBufferFrameCount = (m_nInputPrecacheSize>MAX_CACHE_SIZE)?(MAX_CACHE_SIZE/m_nInputFrameSize):m_nTotalNbFrames;
        m_nGTBufferFrameCount = (m_nGTPrecacheSize>MAX_CACHE_SIZE)?(MAX_CACHE_SIZE/m_nGTFrameSize):m_nTotalNbFrames;
        m_nInputBufferSize = m_nInputBufferFrameCount*m_nInputFrameSize;
        m_nGTBufferSize = m_nGTBufferFrameCount*m_nGTFrameSize;
        m_vcInputBuffer.resize(m_nInputBufferSize);
        m_vcGTBuffer.resize(m_nGTBufferSize);
        m_nNextInputBufferIdx = 0; m_nNextGTBufferIdx = 0;
        m_hInputFramePrecacher = std::thread(&DatasetUtils::SequenceInfo::PrecacheInputFrames,this);
        m_hGTFramePrecacher = std::thread(&DatasetUtils::SequenceInfo::PrecacheGTFrames,this);
    }
}

void DatasetUtils::SequenceInfo::StopPrecaching() {
    if(m_bIsPrecaching) {
        m_bIsPrecaching = false;
        m_hInputFramePrecacher.join();
        m_hGTFramePrecacher.join();
    }
}

#endif //DATASETUTILS_USE_PRECACHED_IO

#if HAVE_GLSL

DatasetUtils::CDNetEvaluator::CDNetEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount)
    :    GLEvaluatorAlgo(pParent,nTotFrameCount,eCDNetEvalCountersCount,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

std::string DatasetUtils::CDNetEvaluator::getComputeShaderSource(size_t nStage) const {
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "#define VAL_POSITIVE     " << (uint)g_nCDnetPositive << "\n"
             "#define VAL_NEGATIVE     " << (uint)g_nCDnetNegative << "\n"
             "#define VAL_OUTOFSCOPE   " << (uint)g_nCDnetOutOfScope << "\n"
             "#define VAL_UNKNOWN      " << (uint)g_nCDnetUnknown << "\n"
             "#define VAL_SHADOW       " << (uint)g_nCDnetShadow << "\n";
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

void DatasetUtils::CDNetEvaluator::getCumulativeCounts(uint64_t& nTotTP, uint64_t& nTotTN, uint64_t& nTotFP, uint64_t& nTotFN, uint64_t& nTotSE) {
    const cv::Mat& oAtomicCountersQueryBuffer = this->getEvaluationAtomicCounterBuffer();
    nTotTP=0; nTotTN=0; nTotFP=0; nTotFN=0; nTotSE=0;
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        nTotTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,CDNetEvaluator::eCDNetEvalCounter_TP);
        nTotTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,CDNetEvaluator::eCDNetEvalCounter_TN);
        nTotFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,CDNetEvaluator::eCDNetEvalCounter_FP);
        nTotFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,CDNetEvaluator::eCDNetEvalCounter_FN);
        nTotSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,CDNetEvaluator::eCDNetEvalCounter_SE);
    }
}

#endif //HAVE_GLSL
