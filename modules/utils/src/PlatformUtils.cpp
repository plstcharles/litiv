
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litiv/utils/PlatformUtils.hpp"

void PlatformUtils::GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths) {
    vsFilePaths.clear();
#if defined(_MSC_VER)
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
#else //!defined(_MSC_VER)
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
                if(!ret && S_ISREG(sb.st_mode)
                        && strcmp(dirp->d_name,"Thumbs.db"))
                    vsFilePaths.push_back(sFullPath);
            }
            std::sort(vsFilePaths.begin(),vsFilePaths.end());
        }
        closedir(dp);
    }
#endif //!defined(_MSC_VER)
}

void PlatformUtils::GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths) {
    vsSubDirPaths.clear();
#if defined(_MSC_VER)
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
#else //!defined(_MSC_VER)
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
                        && strcmp(dirp->d_name,"..")) // @@@ also ignore all hidden folders/files + system folders/files?
                    vsSubDirPaths.push_back(sFullPath);
            }
            std::sort(vsSubDirPaths.begin(),vsSubDirPaths.end());
        }
        closedir(dp);
    }
#endif //!defined(_MSC_VER)
}

void PlatformUtils::FilterFilePaths(std::vector<std::string>& vsFilePaths, const std::vector<std::string>& vsRemoveTokens, const std::vector<std::string>& vsKeepTokens) {
    // note: remove tokens take precedence over keep tokens, and no keep tokens means all are kept by default
    std::vector<std::string> vsResultFilePaths;
    vsResultFilePaths.reserve(vsFilePaths.size());
    for(auto pPathIter=vsFilePaths.begin(); pPathIter!=vsFilePaths.end(); ++pPathIter) {
        if(!vsRemoveTokens.empty() && string_contains_token(*pPathIter,vsRemoveTokens))
            continue;
        else if(vsKeepTokens.empty() || string_contains_token(*pPathIter,vsKeepTokens))
            vsResultFilePaths.push_back(*pPathIter);
    }
    vsFilePaths = vsResultFilePaths;
}

bool PlatformUtils::CreateDirIfNotExist(const std::string& sDirPath) {
#if defined(_MSC_VER)
    std::wstring dir(sDirPath.begin(),sDirPath.end());
    return CreateDirectory(dir.c_str(),NULL)!=ERROR_PATH_NOT_FOUND;
#else //!defined(_MSC_VER)
    struct stat st;
    if(stat(sDirPath.c_str(),&st)==-1)
        return !mkdir(sDirPath.c_str(),0777);
    else
        return (stat(sDirPath.c_str(),&st)==0 && S_ISDIR(st.st_mode));
#endif //!defined(_MSC_VER)
}

#if defined(_MSC_VER)
// SetConsoleWindowSize(...) : derived from http://www.cplusplus.com/forum/windows/121444/
void PlatformUtils::SetConsoleWindowSize(int x, int y, int buffer_lines) {
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    if(h==INVALID_HANDLE_VALUE)
        throw std::runtime_error("SetConsoleWindowSize(...): Unable to get stdout handle");
    COORD largestSize = GetLargestConsoleWindowSize(h);
    if(x>largestSize.X)
        x = largestSize.X;
    if(y>largestSize.Y)
        y = largestSize.Y;
    if(buffer_lines<=0)
        buffer_lines = y;
    CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
    if(!GetConsoleScreenBufferInfo(h,&bufferInfo))
        throw std::runtime_error("SetConsoleWindowSize(...): Unable to retrieve screen buffer info");
    SMALL_RECT& winInfo = bufferInfo.srWindow;
    COORD windowSize = {winInfo.Right-winInfo.Left+1,winInfo.Bottom-winInfo.Top+1};
    if(windowSize.X>x || windowSize.Y>y) {
        SMALL_RECT info = {0,0,SHORT((x<windowSize.X)?(x-1):(windowSize.X-1)),SHORT((y<windowSize.Y)?(y-1):(windowSize.Y-1))};
        if(!SetConsoleWindowInfo(h,TRUE,&info))
            throw std::runtime_error("SetConsoleWindowSize(...): Unable to resize window before resizing buffer");
    }
    COORD size = {SHORT(x),SHORT(y)};
    if(!SetConsoleScreenBufferSize(h,size))
        throw std::runtime_error("SetConsoleWindowSize(...): Unable to resize screen buffer");
    SMALL_RECT info = {0,0,SHORT(x-1),SHORT(y-1)};
    if(!SetConsoleWindowInfo(h, TRUE, &info))
        throw std::runtime_error("SetConsoleWindowSize(...): Unable to resize window after resizing buffer");
}
#endif //defined(_MSC_VER)
