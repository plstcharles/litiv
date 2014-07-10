#include "PlatformUtils.h"

void GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths) {
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
	if((dp  = opendir(sDirPath.c_str()))!=nullptr) {
		size_t nFiles=0;
		while((dirp = readdir(dp))!=nullptr)
			nFiles++;
		if(nFiles>0) {
			vsFilePaths.reserve(nFiles);
			rewinddir(dp);
			while((dirp = readdir(dp))!=nullptr) {
				struct stat sb;
				std::string sFullPath = sDirPath + "/" + dirp->d_name;
				int ret = stat(sFullPath.c_str(),&sb);
				if(!ret && S_ISREG(sb.st_mode))
					vsFilePaths.push_back(sFullPath);
			}
			std::sort(vsFilePaths.begin(),vsFilePaths.end());
		}
		closedir(dp);
	}
#endif //!PLATFORM_USES_WIN32API
}

void GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths) {
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
	if((dp  = opendir(sDirPath.c_str()))!=nullptr) {
		size_t nFiles=0;
		while((dirp = readdir(dp))!=nullptr)
			nFiles++;
		if(nFiles>0) {
			vsSubDirPaths.reserve(nFiles);
			rewinddir(dp);
			while((dirp = readdir(dp))!=nullptr) {
				struct stat sb;
				std::string sFullPath = sDirPath + "/" + dirp->d_name;
				int ret = stat(sFullPath.c_str(),&sb);
				if(!ret && S_ISDIR(sb.st_mode)
						&& strcmp(dirp->d_name,".")
						&& strcmp(dirp->d_name,".."))
					vsSubDirPaths.push_back(sFullPath);
			}
			std::sort(vsSubDirPaths.begin(),vsSubDirPaths.end());
		}
		closedir(dp);
	}
#endif //!PLATFORM_USES_WIN32API
}

bool CreateDirIfNotExist(const std::string& sDirPath) {
#if PLATFORM_USES_WIN32API
	std::wstring dir(sDirPath.begin(),sDirPath.end());
	return CreateDirectory(dir.c_str(),NULL)!=ERROR_PATH_NOT_FOUND;
#else //!PLATFORM_USES_WIN32API
	struct stat st;
	if(stat(sDirPath.c_str(),&st)==-1)
		return !mkdir(sDirPath.c_str(),0777);
	else
		return (stat(sDirPath.c_str(),&st)==0 && S_ISDIR(st.st_mode));
#endif //!PLATFORM_USES_WIN32API
}
