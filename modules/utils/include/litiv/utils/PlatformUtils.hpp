#pragma once

#include "litiv/utils/DefineUtils.hpp"
#include "litiv/utils/CxxUtils.hpp"
#include <queue>
#include <string>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <unordered_map>
#include <deque>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#if defined(_MSC_VER)
#define PLATFORM_USES_WIN32API 1
#define NOMINMAX
#include <windows.h>
#include <stdint.h>
#define __func__ __FUNCTION__
#else //!defined(_MSC_VER)
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif //!defined(_MSC_VER)

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
