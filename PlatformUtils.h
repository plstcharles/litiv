#pragma once

#define PLATFORM_SUPPORTS_CPP11 ((_MSC_VER > 1600) || (__GNUC__>=4 && __GNUC_MINOR__>=6))
#define PLATFORM_USES_WIN32API (WINVER>0 && WINVER>=_WIN32_WINNT_VISTA)

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
