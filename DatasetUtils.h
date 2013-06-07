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
