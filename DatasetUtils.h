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

#define CDNET_DB_NAME 			"CDNet"
#define WALLFLOWER_DB_NAME 		"WALLFLOWER"
#define PETS2001_D3TC1_DB_NAME	"PETS2001_D3TC1"

// as defined in the CDNet scripts/dataset
#define VAL_POSITIVE 	255
#define VAL_NEGATIVE 	0
#define VAL_OUTOFSCOPE	85
#define VAL_UNKNOWN		170
#define VAL_SHADOW		50

#define USE_BROKEN_FNR_FUNCTION 1
#define USE_PRECACHED_IO 0
#if USE_PRECACHED_IO
#define MAX_NB_PRECACHED_FRAMES 100
#define PRECACHE_REFILL_THRESHOLD (MAX_NB_PRECACHED_FRAMES/4)
#define REQUEST_TIMEOUT_MS 1
#define QUERY_TIMEOUT_MS 10
#endif //USE_PRECACHED_IO

class SequenceInfo;

class CategoryInfo {
public:
	CategoryInfo(const std::string& name, const std::string& dir, const std::string& dbname, bool forceGrayscale=false);
	~CategoryInfo();
	const std::string m_sName;
	const std::string m_sDBName;
	std::vector<SequenceInfo*> m_vpSequences;
	uint64_t nTP, nTN, nFP, nFN, nSE;
	double m_dAvgFPS;
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
	cv::Mat GetSequenceROI() const;
	void ValidateKeyPoints(std::vector<cv::KeyPoint>& voKPs) const;
	const std::string m_sName;
	const std::string m_sDBName;
	uint64_t nTP, nTN, nFP, nFN, nSE;
	double m_dAvgFPS;
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
	CategoryInfo* m_pParent;
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
	double dFPS; // note: always averaged
	bool bAveraged;
};

static inline void WriteOnImage(cv::Mat& oImg, const std::string& sText, bool bBottom=false) {
	cv::putText(oImg,sText,cv::Point(10,bBottom?(oImg.rows-15):15),cv::FONT_HERSHEY_PLAIN,1.0,cv::Scalar_<uchar>(0,0,255),1,CV_AA);
}

static inline void WriteResult(	const std::string& sResultsPath,
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

static inline void WriteMetrics(const std::string sResultsFileName, const SequenceInfo* pSeq) {
	std::ofstream oMetricsOutput(sResultsFileName);
	AdvancedMetrics temp(pSeq);
	std::cout << "\t\t" << std::setw(12) << pSeq->m_sName << ":  Rcl=" << temp.dRecall << ", Prc=" << temp.dPrecision << ", FMs=" << temp.dFMeasure << std::endl;
	oMetricsOutput << "Results for sequence '" << pSeq->m_sName << "' :" << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
	oMetricsOutput << pSeq->nTP << " " << pSeq->nFP << " " << pSeq->nFN << " " << pSeq->nTN << " " << pSeq->nSE << std::endl;
	oMetricsOutput << std::endl << std::endl;
	oMetricsOutput << std::fixed << std::setprecision(8);
	oMetricsOutput << "Cumulative metrics :" << std::endl;
	oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	oMetricsOutput << temp.dRecall << " " << temp.dSpecficity << " " << temp.dFPR << " " << temp.dFNR << " " << temp.dPBC << " " << temp.dPrecision << " " << temp.dFMeasure << std::endl;
	oMetricsOutput << std::endl << std::endl;
	oMetricsOutput << "Sequence FPS: " << pSeq->m_dAvgFPS << std::endl;
	oMetricsOutput.close();
}

static inline void WriteMetrics(const std::string sResultsFileName, const CategoryInfo* pCat) {
	std::ofstream oMetricsOutput(sResultsFileName);
	AdvancedMetrics cumulative(pCat, false);
	AdvancedMetrics averaged(pCat, true);
	std::cout << "\t" << std::setw(12) << pCat->m_sName << ":  Rcl=" << averaged.dRecall << ", Prc=" << averaged.dPrecision << ", FMs=" << averaged.dFMeasure << std::endl;
	oMetricsOutput << "Results for category '" << pCat->m_sName << "' :" << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
	oMetricsOutput << pCat->nTP << " " << pCat->nFP << " " << pCat->nFN << " " << pCat->nTN << " " << pCat->nSE << std::endl;
	oMetricsOutput << std::endl << std::endl;
	oMetricsOutput << std::fixed << std::setprecision(8);
	oMetricsOutput << "Cumulative metrics :" << std::endl;
	oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	oMetricsOutput << cumulative.dRecall << " " << cumulative.dSpecficity << " " << cumulative.dFPR << " " << cumulative.dFNR << " " << cumulative.dPBC << " " << cumulative.dPrecision << " " << cumulative.dFMeasure << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "Averaged metrics :" << std::endl;
	oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	oMetricsOutput << averaged.dRecall << " " << averaged.dSpecficity << " " << averaged.dFPR << " " << averaged.dFNR << " " << averaged.dPBC << " " << averaged.dPrecision << " " << averaged.dFMeasure << std::endl;
	oMetricsOutput << std::endl << std::endl;
	oMetricsOutput << "All Sequences Average FPS: " << averaged.dFPS << std::endl;
	oMetricsOutput.close();
}

static inline void WriteMetrics(const std::string sResultsFileName, const std::vector<CategoryInfo*>& vpCat, double dTotalFPS) {
	std::ofstream oMetricsOutput(sResultsFileName);
	AdvancedMetrics cumulative(vpCat, false);
	AdvancedMetrics averaged(vpCat, true);
	oMetricsOutput << std::fixed << std::setprecision(8);
	oMetricsOutput << "Overall results :" << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "Cumulative metrics :" << std::endl;
	oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	for(size_t i=0; i<vpCat.size(); ++i) {
		AdvancedMetrics temp_cumulative(vpCat[i],false);
		std::string sName = vpCat[i]->m_sName;
		if(sName.size()>10)
			sName = sName.substr(0,10);
		else if(sName.size()<10)
			sName += std::string(10-sName.size(),' ');
		oMetricsOutput << sName << " " << temp_cumulative.dRecall << " " << temp_cumulative.dSpecficity << " " << temp_cumulative.dFPR << " " << temp_cumulative.dFNR << " " << temp_cumulative.dPBC << " " << temp_cumulative.dPrecision << " " << temp_cumulative.dFMeasure << std::endl;
	}
	oMetricsOutput << "---------------------------------------------------------------------------------------" << std::endl;
	oMetricsOutput << "overall    " << cumulative.dRecall << " " << cumulative.dSpecficity << " " << cumulative.dFPR << " " << cumulative.dFNR << " " << cumulative.dPBC << " " << cumulative.dPrecision << " " << cumulative.dFMeasure << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "Averaged metrics :" << std::endl;
	oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	for(size_t i=0; i<vpCat.size(); ++i) {
		AdvancedMetrics temp_averaged(vpCat[i],true);
		std::string sName = vpCat[i]->m_sName;
		if(sName.size()>10)
			sName = sName.substr(0,10);
		else if(sName.size()<10)
			sName += std::string(10-sName.size(),' ');
		oMetricsOutput << sName << " " << temp_averaged.dRecall << " " << temp_averaged.dSpecficity << " " << temp_averaged.dFPR << " " << temp_averaged.dFNR << " " << temp_averaged.dPBC << " " << temp_averaged.dPrecision << " " << temp_averaged.dFMeasure << std::endl;
	}
	oMetricsOutput << "---------------------------------------------------------------------------------------" << std::endl;
	oMetricsOutput << "overall    " << averaged.dRecall << " " << averaged.dSpecficity << " " << averaged.dFPR << " " << averaged.dFNR << " " << averaged.dPBC << " " << averaged.dPrecision << " " << averaged.dFMeasure << std::endl;
	oMetricsOutput << std::endl << std::endl;
	oMetricsOutput << "All Sequences Average FPS: " << averaged.dFPS << std::endl;
	oMetricsOutput << "Total FPS: " << dTotalFPS << std::endl;
	oMetricsOutput.close();
}

static inline void CalcMetricsFromResult(const cv::Mat& oInputFrame, const cv::Mat& oGTFrame, const cv::Mat& oROI, uint64_t& nTP, uint64_t& nTN, uint64_t& nFP, uint64_t& nFN, uint64_t& nSE) {
	CV_DbgAssert(oInputFrame.type()==CV_8UC1 && oGTFrame.type()==CV_8UC1 && oROI.type()==CV_8UC1);
	CV_DbgAssert(oInputFrame.size()==oGTFrame.size() && oInputFrame.size()==oROI.size());
	const size_t step_row = oInputFrame.step.p[0];
	for(size_t i=0; i<(size_t)oInputFrame.rows; ++i) {
		const size_t idx_nstep = step_row*i;
		const uchar* input_step_ptr = oInputFrame.data+idx_nstep;
		const uchar* gt_step_ptr = oGTFrame.data+idx_nstep;
		const uchar* roi_step_ptr = oROI.data+idx_nstep;
		for(int j=0; j<oInputFrame.cols; ++j) {
			if(	gt_step_ptr[j]!=VAL_OUTOFSCOPE &&
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
