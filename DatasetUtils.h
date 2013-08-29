#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef WIN32
#include <windows.h>
#include <stdint.h>
#else
#define sprintf_s sprintf
#include <dirent.h>
#include <sys/stat.h>
#endif

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

class SequenceInfo;

class CategoryInfo {
public:
	CategoryInfo(const std::string& name, const std::string& dir, const std::string& dbname, bool forceGrayscale=false);
	~CategoryInfo();
	const std::string m_sName;
	const std::string m_sDBName;
	std::vector<SequenceInfo*> m_vpSequences;
	uint64_t nTP, nTN, nFP, nFN, nSE;
};

class SequenceInfo {
public:
	SequenceInfo(const std::string& name, const std::string& dir, const std::string& dbname, CategoryInfo* parent, bool forceGrayscale=false);
	cv::Mat GetInputFrameFromIndex(size_t idx);
	cv::Mat GetGTFrameFromIndex(size_t idx);
	size_t GetNbInputFrames() const;
	size_t GetNbGTFrames() const;
	cv::Size GetFrameSize() const;
	cv::Mat GetSequenceROI() const;
	const std::string m_sName;
	const std::string m_sDBName;
	uint64_t nTP, nTN, nFP, nFN, nSE;
private:
	std::vector<std::string> m_vsInputFramePaths;
	std::vector<std::string> m_vsGTFramePaths;
	cv::VideoCapture m_voVideoReader;
	size_t m_nNextFrame;
	size_t m_nTotalNbFrames;
	cv::Mat m_oROI;
	cv::Size m_oSize;
	CategoryInfo* m_pParent;
	const int m_nIMReadInputFlags;
	std::unordered_map<size_t,size_t> m_mTestGTIndexes;
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
	bool bAveraged;
};

static inline void WriteOnImage(cv::Mat& oImg, const std::string& sText, bool bBottom=false) {
	cv::putText(oImg,sText,cv::Point(10,bBottom?(oImg.rows-15):15),cv::FONT_HERSHEY_PLAIN,1.0,cv::Scalar_<uchar>(0,0,255),1,CV_AA);
}

static inline void WriteResult(	const std::string& sResultsPath,
								const std::string& sCatName,
								const std::string& sSeqName,
								const std::string& sResultPrefix,
								int framenum,
								const std::string& sResultSuffix,
								const cv::Mat& res,
								const std::vector<int>& vnComprParams) {
	char buffer[10];
	sprintf_s(buffer,"%06d",framenum);
	std::stringstream sResultFilePath;
	sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << buffer << sResultSuffix;
	cv::imwrite(sResultFilePath.str(), res, vnComprParams);
}

static inline void WriteMetrics(const std::string sResultsFileName, const SequenceInfo* pSeq) {
	std::ofstream oMetricsOutput(sResultsFileName);
	AdvancedMetrics temp(pSeq);
	oMetricsOutput << "Results for sequence '" << pSeq->m_sName << "' :" << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
	oMetricsOutput << pSeq->nTP << " " << pSeq->nFP << " " << pSeq->nFN << " " << pSeq->nTN << " " << pSeq->nSE << std::endl;
	oMetricsOutput << std::endl << std::endl;
	oMetricsOutput << std::fixed << std::setprecision(8);
	oMetricsOutput << "Precise metrics :" << std::endl;
	oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	oMetricsOutput << temp.dRecall << " " << temp.dSpecficity << " " << temp.dFPR << " " << temp.dFNR << " " << temp.dPBC << " " << temp.dPrecision << " " << temp.dFMeasure << std::endl;
	oMetricsOutput.close();
}

static inline void WriteMetrics(const std::string sResultsFileName, const CategoryInfo* pCat) {
	std::ofstream oMetricsOutput(sResultsFileName);
	AdvancedMetrics precise(pCat, false);
	AdvancedMetrics averaged(pCat, true);
	oMetricsOutput << "Results for category '" << pCat->m_sName << "' :" << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "nTP nFP nFN nTN nSE" << std::endl; // order similar to the files saved by the CDNet analysis script
	oMetricsOutput << pCat->nTP << " " << pCat->nFP << " " << pCat->nFN << " " << pCat->nTN << " " << pCat->nSE << std::endl;
	oMetricsOutput << std::endl << std::endl;
	oMetricsOutput << std::fixed << std::setprecision(8);
	oMetricsOutput << "Precise metrics :" << std::endl;
	oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	oMetricsOutput << precise.dRecall << " " << precise.dSpecficity << " " << precise.dFPR << " " << precise.dFNR << " " << precise.dPBC << " " << precise.dPrecision << " " << precise.dFMeasure << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "Averaged metrics :" << std::endl;
	oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	oMetricsOutput << averaged.dRecall << " " << averaged.dSpecficity << " " << averaged.dFPR << " " << averaged.dFNR << " " << averaged.dPBC << " " << averaged.dPrecision << " " << averaged.dFMeasure << std::endl;
	oMetricsOutput.close();
}

static inline void WriteMetrics(const std::string sResultsFileName, const std::vector<CategoryInfo*>& vpCat) {
	std::ofstream oMetricsOutput(sResultsFileName);
	AdvancedMetrics precise(vpCat, false);
	AdvancedMetrics averaged(vpCat, true);
	oMetricsOutput << std::fixed << std::setprecision(8);
	oMetricsOutput << "Overall results :" << std::endl;
	oMetricsOutput << std::endl;
	oMetricsOutput << "Precise metrics :" << std::endl;
	oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FMs       " << std::endl;
	for(size_t i=0; i<vpCat.size(); ++i) {
		AdvancedMetrics temp_precise(vpCat[i],false);
		std::string sName = vpCat[i]->m_sName;
		if(sName.size()>10)
			sName = sName.substr(0,10);
		else if(sName.size()<10)
			sName += std::string(10-sName.size(),' ');
		oMetricsOutput << sName << " " << temp_precise.dRecall << " " << temp_precise.dSpecficity << " " << temp_precise.dFPR << " " << temp_precise.dFNR << " " << temp_precise.dPBC << " " << temp_precise.dPrecision << " " << temp_precise.dFMeasure << std::endl;
	}
	oMetricsOutput << "---------------------------------------------------------------------------------------" << std::endl;
	oMetricsOutput << "overall    " << precise.dRecall << " " << precise.dSpecficity << " " << precise.dFPR << " " << precise.dFNR << " " << precise.dPBC << " " << precise.dPrecision << " " << precise.dFMeasure << std::endl;
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
	oMetricsOutput.close();
}

static inline void GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths) {
	vsFilePaths.clear();
#ifdef WIN32
	WIN32_FIND_DATA ffd;
	std::wstring dir(sDirPath.begin(),sDirPath.end());
	dir += L"/*";
	BOOL ret = TRUE;
	HANDLE h;
	h = FindFirstFile(dir.c_str(),&ffd);
	if(h!=INVALID_HANDLE_VALUE) {
		size_t nFiles=0;
		while(ret) {
			nFiles++;
			ret = FindNextFile(h, &ffd);
		}
		if(nFiles>0) {
			vsFilePaths.reserve(nFiles);
			h = FindFirstFile(dir.c_str(),&ffd);
			assert(h!=INVALID_HANDLE_VALUE);
			ret = TRUE;
			while(ret) {
				if(!(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
					std::wstring file(ffd.cFileName);
					vsFilePaths.push_back(sDirPath + "/" + std::string(file.begin(),file.end()));
				}
				ret = FindNextFile(h, &ffd);
			}
		}
	}
#else
	DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(sDirPath.c_str()))!=NULL) {
		size_t nFiles=0;
		while((dirp = readdir(dp)) != NULL)
			nFiles++;
		if(nFiles>0) {
			vsFilePaths.reserve(nFiles);
			rewinddir(dp);
			while((dirp = readdir(dp)) != NULL) {
				struct stat sb;
				std::string sFullPath = sDirPath + "/" + dirp->d_name;
				int ret = stat(sFullPath.c_str(),&sb);
				if(!ret && S_ISREG(sb.st_mode))
					vsFilePaths.push_back(sFullPath);
			}
		}
		closedir(dp);
	}
#endif
}

static inline void GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths) {
	vsSubDirPaths.clear();
#ifdef WIN32
	WIN32_FIND_DATA ffd;
	std::wstring dir(sDirPath.begin(),sDirPath.end());
	dir += L"/*";
	BOOL ret = TRUE;
	HANDLE h;
	h = FindFirstFile(dir.c_str(),&ffd);
	if(h!=INVALID_HANDLE_VALUE) {
		size_t nFiles=0;
		while(ret) {
			nFiles++;
			ret = FindNextFile(h, &ffd);
		}
		if(nFiles>0) {
			vsSubDirPaths.reserve(nFiles);
			h = FindFirstFile(dir.c_str(),&ffd);
			assert(h!=INVALID_HANDLE_VALUE);
			ret = TRUE;
			while(ret) {
				if(ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
					std::wstring subdir(ffd.cFileName);
					if(subdir!=L"." && subdir!=L"..")
						vsSubDirPaths.push_back(sDirPath + "/" + std::string(subdir.begin(),subdir.end()));
				}
				ret = FindNextFile(h, &ffd);
			}
		}
	}
#else
	DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(sDirPath.c_str()))!=NULL) {
		size_t nFiles=0;
		while((dirp = readdir(dp)) != NULL)
			nFiles++;
		if(nFiles>0) {
			vsSubDirPaths.reserve(nFiles);
			rewinddir(dp);
			while((dirp = readdir(dp)) != NULL) {
				struct stat sb;
				std::string sFullPath = sDirPath + "/" + dirp->d_name;
				int ret = stat(sFullPath.c_str(),&sb);
				if(!ret && S_ISDIR(sb.st_mode)
						&& strcmp(dirp->d_name,".")
						&& strcmp(dirp->d_name,".."))
					vsSubDirPaths.push_back(sFullPath);
			}
		}
		closedir(dp);
	}
#endif
}

static inline void CalcMetricsFromResult(const cv::Mat& oInputFrame, const cv::Mat& oGTFrame, const cv::Mat& oROI, uint64_t& nTP, uint64_t& nTN, uint64_t& nFP, uint64_t& nFN, uint64_t& nSE) {
	CV_DbgAssert(oInputFrame.type()==CV_8UC1 && oGTFrame.type()==CV_8UC1 && oROI.type()==CV_8UC1);
	CV_DbgAssert(oInputFrame.size()==oGTFrame.size() && oInputFrame.size()==oROI.size());
	const int step_row = oInputFrame.step.p[0];
	for(int i=0; i<oInputFrame.rows; ++i) {
		const int step_idx = step_row*i;
		const uchar* input_step_ptr = oInputFrame.data+step_idx;
		const uchar* gt_step_ptr = oGTFrame.data+step_idx;
		const uchar* roi_step_ptr = oROI.data+step_idx;
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
