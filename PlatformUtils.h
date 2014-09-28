#pragma once

#include <queue>
#include <string>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
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
    double elapsedTime; \
    QueryPerformanceFrequency(&frequency);
#define TIMER_START QueryPerformanceCounter(&t1);
#define TIMER_STOP \
    QueryPerformanceCounter(&t2); \
    elapsedTime=(float)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart; \
    std::cout << elapsedTime << " sec" << std::endl;
void SetConsoleWindowSize(int x, int y, int buffer_lines=-1);
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

void GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths);
void GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths);
bool CreateDirIfNotExist(const std::string& sDirPath);
