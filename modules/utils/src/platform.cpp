
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

#include "litiv/utils/platform.hpp"

std::string lv::GetCurrentWorkDirPath() {
    static std::array<char,FILENAME_MAX> s_acCurrentPath = {};
#if defined(_MSC_VER)
    if(!_getcwd(s_acCurrentPath.data(),int(s_acCurrentPath.size()-1)))
#else //(!defined(_MSC_VER))
    if(!getcwd(s_acCurrentPath.data(),s_acCurrentPath.size()-1))
#endif //(!defined(_MSC_VER))
        return std::string();
    return std::string(s_acCurrentPath.data());
}

std::string lv::AddDirSlashIfMissing(const std::string& sDirPath) {
    if(sDirPath.empty())
        return std::string();
    if(sDirPath=="." || sDirPath=="..")
        return sDirPath+"/";
    const char cLastCharacter = sDirPath[sDirPath.size()-1];
    if(cLastCharacter!='/' && cLastCharacter!='\\')
        return sDirPath+"/";
    return sDirPath;
}

void lv::GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths) {
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
                    if(file!=L"Thumbs.db")
                        vsFilePaths.push_back(AddDirSlashIfMissing(sDirPath)+std::string(file.begin(),file.end()));
                }
                ret = FindNextFile(h, &ffd);
            }
        }
    }
#else //(!defined(_MSC_VER))
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
                std::string sFullPath = AddDirSlashIfMissing(sDirPath)+dirp->d_name;
                int ret = stat(sFullPath.c_str(),&sb);
                if(!ret && S_ISREG(sb.st_mode)
                        && strcmp(dirp->d_name,"Thumbs.db"))
                    vsFilePaths.push_back(sFullPath);
            }
            std::sort(vsFilePaths.begin(),vsFilePaths.end());
        }
        closedir(dp);
    }
#endif //(!defined(_MSC_VER))
}

void lv::GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths) {
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
                        vsSubDirPaths.push_back(AddDirSlashIfMissing(sDirPath)+std::string(subdir.begin(),subdir.end()));
                }
                ret = FindNextFile(h, &ffd);
            }
        }
    }
#else //(!defined(_MSC_VER))
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
                std::string sFullPath = AddDirSlashIfMissing(sDirPath)+dirp->d_name;
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
#endif //(!defined(_MSC_VER))
}

void lv::FilterFilePaths(std::vector<std::string>& vsFilePaths, const std::vector<std::string>& vsRemoveTokens, const std::vector<std::string>& vsKeepTokens) {
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

bool lv::CreateDirIfNotExist(const std::string& sDirPath) {
#if defined(_MSC_VER)
    std::wstring dir(sDirPath.begin(),sDirPath.end());
    return CreateDirectory(dir.c_str(),NULL)!=ERROR_PATH_NOT_FOUND;
#else //(!defined(_MSC_VER))
    struct stat st;
    if(stat(sDirPath.c_str(),&st)==-1)
        return !mkdir(sDirPath.c_str(),0777);
    else
        return (stat(sDirPath.c_str(),&st)==0 && S_ISDIR(st.st_mode));
#endif //(!defined(_MSC_VER))
}

std::fstream lv::CreateBinFileWithPrealloc(const std::string & sFilePath, size_t nPreallocBytes, bool bZeroInit) {
    std::fstream ssFile(sFilePath,std::ios::out|std::ios::in|std::ios::ate|std::ios::binary);
    if(!ssFile.is_open())
        ssFile.open(sFilePath,std::ios::out|std::ios::binary);
    lvAssert__(ssFile.is_open(),"could not create file at '%s'",sFilePath.c_str());
    const std::streampos nInitFileSize = ssFile.tellp();
    if(nInitFileSize<(std::streampos)nPreallocBytes) {
        size_t nBytesToWrite = nPreallocBytes-size_t(nInitFileSize);
        std::vector<char> aPreallocBuff;
        while(nBytesToWrite>0) {
            const size_t nBufferSize = TARGET_PLATFORM_x64?nBytesToWrite:std::min(size_t(250*1024*1024)/*250MB*/,nBytesToWrite);
            if(bZeroInit)
                aPreallocBuff.resize(nBufferSize,0);
            else
                aPreallocBuff.resize(nBufferSize);
            ssFile.write(aPreallocBuff.data(),nBufferSize);
            nBytesToWrite -= nBufferSize;
        }
    }
    lvAssert_(ssFile.seekp(0),"could not return seek pointer to beg of file");
    return ssFile;
}

void lv::RegisterAllConsoleSignals(void(*lHandler)(int)) {
    signal(SIGINT,lHandler);
    signal(SIGTERM,lHandler);
#ifdef SIGBREAK
    signal(SIGBREAK,lHandler);
#endif //def(SIGBREAK)
}

size_t lv::GetCurrentPhysMemBytesUsed() {
#if defined(_MSC_VER)
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(),&info,sizeof(info));
    return (size_t)info.WorkingSetSize;
#else //ndef(_MSC_VER)
    FILE* fp = nullptr;
    if(!(fp=fopen("/proc/self/statm","r")))
        return size_t(0);
    long nMemUsed = 0L;
    if(fscanf(fp,"%*s%ld",&nMemUsed)!=1) {
        fclose(fp);
        return size_t(0);
    }
    fclose(fp);
    return size_t(nMemUsed*sysconf(_SC_PAGESIZE));
#endif //ndef(_MSC_VER)
}
