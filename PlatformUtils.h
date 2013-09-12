#pragma once

// note: PLATFORM_SUPPORTS_CPP11 assumes that if compiling on linux and __cplusplus is 'bugged' (i.e. GCC<4.7), we can use C++11 if GCC>4.6 (which is definitely not true unless compiled with -std=c++0x or -std=c++11)
#define PLATFORM_SUPPORTS_CPP11 ((defined(WIN32) && defined(_MSC_VER) && _MSC_VER > 1600) || __cplusplus>199711L || (defined(__GNUC__) && __GNUC__>=4 && __GNUC_MINOR__>=6))
#define PLATFORM_USES_WIN32API (WIN32 && !__MINGW32__)

#if PLATFORM_USES_WIN32API
#include <windows.h>
#include <stdint.h>
#define sprintf sprintf_s
#else //!PLATFORM_USES_WIN32API
#include <dirent.h>
#include <sys/stat.h>
#endif //!PLATFORM_USES_WIN32API
#if PLATFORM_SUPPORTS_CPP11
#include <mutex>
#include <condition_variable>
#endif //PLATFORM_USES_WIN32API

#include <string>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

class FrameSemaphore {
public:
	FrameSemaphore(int nCountMax);
	~FrameSemaphore();
	void notify();
	void wait();
	bool try_wait();
private:
#if PLATFORM_SUPPORTS_CPP11
	std::mutex m_oMutex;
    std::condition_variable m_oCondVar;
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
	HANDLE m_oSemaphore;
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for semaphores on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
	size_t m_nCount;
	const size_t m_nCountMax;
};

static inline void GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths) {
	vsFilePaths.clear();
#if PLATFORM_USES_WIN32API
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
#else //!PLATFORM_USES_WIN32API
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
#endif //!PLATFORM_USES_WIN32API
}

static inline void GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths) {
	vsSubDirPaths.clear();
#if PLATFORM_USES_WIN32API
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
#else //!PLATFORM_USES_WIN32API
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
#endif //!PLATFORM_USES_WIN32API
}
