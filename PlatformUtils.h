#pragma once

#include <queue>
#include <string>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#define PLATFORM_SUPPORTS_CPP11 ((_MSC_VER > 1600) || (__GNUC__>=4 && __GNUC_MINOR__>=6))
#if (defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64)
#define NOMINMAX
#include <windows.h>
#define PLATFORM_USES_WIN32API (WINVER>0x0599)
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#ifndef DBG_NEW
#define DBG_NEW new (_NORMAL_BLOCK , __FILE__ , __LINE__)
#define new DBG_NEW
#endif //!DBG_NEW
#endif //_DEBUG
#endif //WIN32
#if PLATFORM_USES_WIN32API
#include <stdint.h>
#include <process.h>
#define TIMER_INIT \
    LARGE_INTEGER frequency; \
    LARGE_INTEGER t1,t2; \
    QueryPerformanceFrequency(&frequency);
#define TIMER_START QueryPerformanceCounter(&t1);
#define TIMER_STOP \
    QueryPerformanceCounter(&t2); \
    std::cout << (float)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart << " sec" << std::endl;
#define __func__ __FUNCTION__
#else //!PLATFORM_USES_WIN32API
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif //!PLATFORM_USES_WIN32API
#if PLATFORM_SUPPORTS_CPP11
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <condition_variable>
#endif //PLATFORM_USES_WIN32API

namespace PlatformUtils {
    void GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths);
    void GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths);
    bool CreateDirIfNotExist(const std::string& sDirPath);

    inline bool compare_lowercase(const std::string& i, const std::string& j) {
        std::string i_lower(i), j_lower(j);
        std::transform(i_lower.begin(),i_lower.end(),i_lower.begin(),tolower);
        std::transform(j_lower.begin(),j_lower.end(),j_lower.begin(),tolower);
        return i_lower<j_lower;
    }

    template<typename T> inline int decimal_integer_digit_count(T number) {
        int digits = number<0?1:0;
        while(std::abs(number)>=1) {
            number /= 10;
            digits++;
        }
        return digits;
    }

#if PLATFORM_USES_WIN32API
    void SetConsoleWindowSize(int x, int y, int buffer_lines=-1);
#endif //PLATFORM_USES_WIN32API
}; //namespace PlatformUtils
