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

class SequenceInfo;

class CategoryInfo {
public:
	CategoryInfo(const std::string& name, const std::string& dir, const std::string& dbname);
	~CategoryInfo();
	const std::string m_sName;
	const std::string m_sDBName;
	std::vector<SequenceInfo*> m_vpSequences;
	uint64_t nTP, nTN, nFP, nFN;
};

class SequenceInfo {
public:
	SequenceInfo(const std::string& name, const std::string& dir, const std::string& dbname, CategoryInfo* parent);
	cv::Mat GetInputFrameFromIndex(size_t idx);
	cv::Mat GetGTFrameFromIndex(size_t idx);
	size_t GetNbInputFrames() const;
	size_t GetNbGTFrames() const;
	cv::Size GetFrameSize() const;
	cv::Mat GetSequenceROI() const;
	const std::string m_sName;
	const std::string m_sDBName;
	uint64_t nTP, nTN, nFP, nFN;
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

static inline void WriteMetrics(const std::string sResultsFileName, uint64_t nTP, uint64_t nTN, uint64_t nFP, uint64_t nFN) {
	std::ofstream oMetricsOutput(sResultsFileName);
	oMetricsOutput << nTP << " " << nFP << " " << nFN << " " << nTN << std::endl; // order similar to the files saved by the CDNet analysis script
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

static inline void CalcMetricsFromResult(const cv::Mat& oInputFrame, const cv::Mat& oGTFrame, const cv::Mat& oROI, uint64_t& nTP, uint64_t& nTN, uint64_t& nFP, uint64_t& nFN) {
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
				// ADD SUPPORT FOR SHADOW ERRORS? @@@@@@@@@@@@@@@@
			}
		}
	}
}
