#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

static inline void WriteOnImage(cv::Mat& oImg, const std::string& sText, bool bBottom=false) {
	cv::putText(oImg,sText,cv::Point(10,bBottom?(oImg.rows-15):15),cv::FONT_HERSHEY_PLAIN,1.0,cv::Scalar(0,0,255),1,CV_AA);
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
	sprintf(buffer,"%06d",framenum);
	std::stringstream sResultFilePath;
	sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << buffer << sResultSuffix;
	cv::imwrite(sResultFilePath.str(), res, vnComprParams);
}

static inline cv::Mat GetDisplayResult(	const cv::Mat& oInputImg,
										const cv::Mat& oBGImg,
										const cv::Mat& oBGDesc,
										const cv::Mat& oFGMask,
										std::vector<cv::KeyPoint> voKeyPoints,
										size_t nFrame) {
	// note: this function is definitely NOT efficient in any way; it is only intended for debug purposes.
	cv::Mat oInputImgBYTE3, oBGImgBYTE3, oBGDescBYTE, oBGDescBYTE3, oFGMaskBYTE3;
	cv::Mat oInputDesc, oInputDescBYTE, oInputDescBYTE3;
	cv::Mat oDescDiff, oDescDiffBYTE, oDescDiffBYTE3;
	LBSP oExtractor;
	oExtractor.setReference(oBGImg);
	oExtractor.compute2(oInputImg,voKeyPoints,oInputDesc);
	LBSP::calcDescImgDiff(oInputDesc,oBGDesc,oDescDiff);
	oInputDesc.convertTo(oInputDescBYTE,CV_8U);
	oBGDesc.convertTo(oBGDescBYTE,CV_8U);
	oDescDiff.convertTo(oDescDiffBYTE,CV_8U);
	cv::cvtColor(oFGMask,oFGMaskBYTE3,CV_GRAY2RGB);
	if(oInputImg.channels()!=3) {
		cv::cvtColor(oInputImg,oInputImgBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oBGImg,oBGImgBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oInputDescBYTE,oInputDescBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oBGDescBYTE,oBGDescBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oDescDiffBYTE,oDescDiffBYTE3,CV_GRAY2RGB);
	}
	else {
		oInputImgBYTE3 = oInputImg;
		oBGImgBYTE3 = oBGImg;
		oInputDescBYTE3 = oInputDescBYTE;
		oBGDescBYTE3 = oBGDescBYTE;
		oDescDiffBYTE3 = oDescDiffBYTE;
	}
	cv::Mat display1H,display2H,display3H;
	std::stringstream sstr;
	sstr << "Input Img #" << nFrame;
	WriteOnImage(oInputImgBYTE3,sstr.str());
	WriteOnImage(oBGImgBYTE3,"BGModel Img");
	WriteOnImage(oInputDescBYTE3,"Input Desc");
	WriteOnImage(oBGDescBYTE3,"BGModel Desc");
	WriteOnImage(oFGMaskBYTE3,"Detection Result");
	WriteOnImage(oDescDiffBYTE3,"BGModel-Input Desc Diff");
	cv::hconcat(oInputImgBYTE3,oBGImgBYTE3,display1H);
	cv::hconcat(oInputDescBYTE3,oBGDescBYTE3,display2H);
	cv::hconcat(oFGMaskBYTE3,oDescDiffBYTE3,display3H);
	cv::Mat display;
	cv::vconcat(display1H,display2H,display);
	cv::vconcat(display,display3H,display);
	return display;
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

struct SequenceInfo {
	SequenceInfo(const std::string& name) {
		sName = name;
	}
	SequenceInfo(const std::string& name, const std::string& dir) {
		sName = name;

		// amongst possible subdirs at this level, we expect a 'groundtruth' and an 'input' directory (throws if not found)
		std::vector<std::string> vsSubDirs;
		GetSubDirsFromDir(dir,vsSubDirs);
		auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),dir+"/groundtruth");
		auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),dir+"/input");
		if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess the required groundtruth and input directories.");
		GetFilesFromDir(*inputDir,vsInputFramePaths);
		GetFilesFromDir(*gtDir,vsGTFramePaths);
		if(vsGTFramePaths.size()!=vsInputFramePaths.size())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess same amount of GT & input frames.");

		// amongst possible files at this level, we expect a 'ROI.bmp' file to specify the sequence's region of interest (throws if not found)
		oROI = cv::imread(dir+"/ROI.bmp");
		if(oROI.empty())
			throw std::runtime_error(std::string("Sequence at ") + dir + " did not possess a ROI.bmp file.");
	}
	SequenceInfo(const std::string& name, const std::vector<std::string>& inputframes, const std::vector<std::string>& gtframes, const cv::Mat& roi) {
		sName = name;
		vsInputFramePaths = inputframes;
		vsGTFramePaths = gtframes;
		oROI = roi;
	}
	std::string sName;
	std::vector<std::string> vsInputFramePaths;
	std::vector<std::string> vsGTFramePaths;
	cv::Mat oROI;
};

struct CategoryInfo {
	CategoryInfo(const std::string& name) {
		sName = name;
	}
	CategoryInfo(const std::string& name, const std::string& dir) {
		sName = name;
		// all subdirs are considered sequence directories for this category; no parsing is done for files at this level.
		std::vector<std::string> vsSequencePaths;
		GetSubDirsFromDir(dir,vsSequencePaths);
		for(size_t i=0; i<vsSequencePaths.size(); i++) {
			size_t pos = vsSequencePaths[i].find_last_of("/\\");
			if(pos==std::string::npos)
				vpSequences.push_back(new SequenceInfo(vsSequencePaths[i],vsSequencePaths[i]));
			else
				vpSequences.push_back(new SequenceInfo(vsSequencePaths[i].substr(pos+1),vsSequencePaths[i]));
		}
	}
	CategoryInfo(const std::string& name, const std::vector<SequenceInfo*>& sequences) {
		sName = name;
		vpSequences = sequences;
	}
	~CategoryInfo() {
		for(size_t i=0; i<vpSequences.size(); i++)
			delete vpSequences[i];
	}
	std::string sName;
	std::vector<SequenceInfo*> vpSequences;
};
