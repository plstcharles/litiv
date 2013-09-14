#pragma once

// note: PLATFORM_SUPPORTS_CPP11 assumes that if compiling on linux and __cplusplus is 'bugged' (i.e. GCC<4.7), we can use C++11 if GCC>4.6 (which is definitely not true unless compiled with -std=c++0x or -std=c++11)
#define PLATFORM_SUPPORTS_CPP11 ((defined(WIN32) && defined(_MSC_VER) && _MSC_VER > 1600) || __cplusplus>199711L || (defined(__GNUC__) && __GNUC__>=4 && __GNUC_MINOR__>=6))
#define PLATFORM_USES_WIN32API (WIN32 && !__MINGW32__)

#include <queue>
#include <string>
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if PLATFORM_USES_WIN32API
#include <windows.h>
#include <stdint.h>
#include <process.h>
#else //!PLATFORM_USES_WIN32API
#include <dirent.h>
#include <sys/stat.h>
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
