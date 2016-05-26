
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

#pragma once

// includes here really need cleanup @@@@

#include <opencv2/core.hpp>
#include "litiv/utils/DistanceUtils.hpp"
#include "litiv/utils/CxxUtils.hpp"
#include <queue>
#include <string>
#include <algorithm>
#include <vector>
#include <set>
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
#include <inttypes.h>
#if defined(_MSC_VER)
#include <windows.h>
#include <winerror.h>
#include <comdef.h>
#include <stdint.h>
#include <direct.h>
template<class T>
void SafeRelease(T **ppT) {if(*ppT) {(*ppT)->Release();*ppT = nullptr;}}
#else //(!defined(_MSC_VER))
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif //(!defined(_MSC_VER))

namespace PlatformUtils {

    std::string GetCurrentWorkDirPath();
    std::string AddDirSlashIfMissing(const std::string& sDirPath);
    void GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths);
    void GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths);
    void FilterFilePaths(std::vector<std::string>& vsFilePaths, const std::vector<std::string>& vsRemoveTokens, const std::vector<std::string>& vsKeepTokens);
    bool CreateDirIfNotExist(const std::string& sDirPath);

    inline bool compare_lowercase(const std::string& i, const std::string& j) {
        std::string i_lower(i), j_lower(j);
        std::transform(i_lower.begin(),i_lower.end(),i_lower.begin(),tolower);
        std::transform(j_lower.begin(),j_lower.end(),j_lower.begin(),tolower);
        return i_lower<j_lower;
    }

    template<typename T>
    inline int decimal_integer_digit_count(T number) {
        int digits = number<0?1:0;
        while(std::abs(number)>=1) {
            number /= 10;
            digits++;
        }
        return digits;
    }

    inline bool string_contains_token(const std::string& s, const std::vector<std::string>& tokens) {
        for(size_t i=0; i<tokens.size(); ++i)
            if(s.find(tokens[i])!=std::string::npos)
                return true;
        return false;
    }

    template<typename T>
    inline std::vector<size_t> sort_indexes(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndexes(voVals.size());
        for(size_t n=0; n<voVals.size(); ++n)
            vnIndexes[n] = n;
        std::sort(vnIndexes.begin(),vnIndexes.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]<voVals[n2];
        });
        return vnIndexes;
    }

    template<typename T, typename P>
    inline std::vector<size_t> sort_indexes(const std::vector<T>& voVals, P oSortFunctor) {
        std::vector<size_t> vnIndexes(voVals.size());
        for(size_t n=0; n<voVals.size(); ++n)
            vnIndexes[n] = n;
        std::sort(vnIndexes.begin(),vnIndexes.end(),oSortFunctor);
        return vnIndexes;
    }

    template<typename T>
    inline std::vector<size_t> unique_indexes(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndexes = sort_indexes(voVals);
        auto pLastIdxIter = std::unique(vnIndexes.begin(),vnIndexes.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]==voVals[n2];
        });
        return std::vector<size_t>(vnIndexes.begin(),pLastIdxIter);
    }

    template<typename T, typename P1, typename P2>
    inline std::vector<size_t> unique_indexes(const std::vector<T>& voVals, P1 oSortFunctor, P2 oCompareFunctor) {
        std::vector<size_t> vnIndexes = sort_indexes(voVals,oSortFunctor);
        auto pLastIdxIter = std::unique(vnIndexes.begin(),vnIndexes.end(),oCompareFunctor);
        return std::vector<size_t>(vnIndexes.begin(),pLastIdxIter);
    }

    inline std::vector<uchar> unique_8uc1_values(const cv::Mat& oMat) {
        CV_Assert(!oMat.empty() && oMat.type()==CV_8UC1);
        std::array<bool,UCHAR_MAX+1> anUniqueLUT{0};
        for(int i=0; i<oMat.rows; ++i)
            for(int j=0; j<oMat.cols; ++j)
                anUniqueLUT[oMat.at<uchar>(i,j)] = true;
        std::vector<uchar> vuVals;
        vuVals.reserve(UCHAR_MAX+1);
        for(size_t n=0; n<UCHAR_MAX+1; ++n)
            if(anUniqueLUT[n])
                vuVals.push_back((uchar)n);
        return vuVals;
    }

    template<typename T>
    size_t find_nn_index(T oReqVal, const std::vector<T>& voRefVals) {
        decltype(DistanceUtils::L1dist(T(0),T(0))) oMinDist = std::numeric_limits<decltype(DistanceUtils::L1dist(T(0),T(0)))>::max();
        size_t nIdx = size_t(-1);
        for(size_t n=0; n<voRefVals.size(); ++n) {
            auto oCurrDist = DistanceUtils::L1dist(oReqVal,voRefVals[n]);
            if(nIdx==size_t(-1) || oCurrDist<oMinDist) {
                oMinDist = oCurrDist;
                nIdx = n;
            }
        }
        return nIdx;
    }

    template<typename Tx, typename Ty>
    std::vector<Ty> interp1(const std::vector<Tx>& vX, const std::vector<Ty>& vY, const std::vector<Tx>& vXReq) {
        // assumes that all vectors are sorted
        CV_Assert(vX.size()==vY.size());
        CV_Assert(vX.size()>1);
        std::vector<Tx> vDX;
        vDX.reserve(vX.size());
        std::vector<Ty> vDY, vSlope, vIntercept;
        vDY.reserve(vX.size());
        vSlope.reserve(vX.size());
        vIntercept.reserve(vX.size());
        for(size_t i=0; i<vX.size(); ++i) {
            if(i<vX.size()-1) {
                vDX.push_back(vX[i+1]-vX[i]);
                vDY.push_back(vY[i+1]-vY[i]);
                vSlope.push_back(Ty(vDY[i]/vDX[i]));
                vIntercept.push_back(vY[i]-Ty(vX[i]*vSlope[i]));
            }
            else {
                vDX.push_back(vDX[i-1]);
                vDY.push_back(vDY[i-1]);
                vSlope.push_back(vSlope[i-1]);
                vIntercept.push_back(vIntercept[i-1]);
            }
        }
        std::vector<Ty> vYReq;
        vYReq.reserve(vXReq.size());
        for(size_t i=0; i<vXReq.size(); ++i) {
            if(vXReq[i]>=vX.front() && vXReq[i]<=vX.back()) {
                size_t nNNIdx = find_nn_index(vXReq[i],vX);
                vYReq.push_back(vSlope[nNNIdx]*vXReq[i]+vIntercept[nNNIdx]);
            }
        }
        return vYReq;
    }

    template<typename T>
    inline typename std::enable_if<std::is_integral<T>::value,std::vector<T>>::type linspace(T a, T b, size_t steps, bool bIncludeInitVal=true) {
        if(steps==0)
            return std::vector<T>();
        else if(steps==1)
            return std::vector<T>(1,b);
        std::vector<T> vnResult(steps);
        const double dStep = double(b-a)/(steps-int(bIncludeInitVal));
        if(bIncludeInitVal)
            for(size_t nStepIter = 0; nStepIter<steps; ++nStepIter)
                vnResult[nStepIter] = a+T(dStep*nStepIter);
        else
            for(size_t nStepIter = 1; nStepIter<=steps; ++nStepIter)
                vnResult[nStepIter-1] = a+T(dStep*nStepIter);
        return vnResult;
    }

    template<typename T>
    inline typename std::enable_if<std::is_floating_point<T>::value,std::vector<T>>::type linspace(T a, T b, size_t steps, bool bIncludeInitVal=true) {
        if(steps==0)
            return std::vector<T>();
        else if(steps==1)
            return std::vector<T>(1,b);
        std::vector<T> vfResult(steps);
        const T fStep = (b-a)/(steps-int(bIncludeInitVal));
        if(bIncludeInitVal)
            for(size_t nStepIter = 0; nStepIter<steps; ++nStepIter)
                vfResult[nStepIter] = a+fStep*T(nStepIter);
        else
            for(size_t nStepIter = 1; nStepIter<=steps; ++nStepIter)
                vfResult[nStepIter] = a+fStep*T(nStepIter);
        return vfResult;
    }

} //namespace PlatformUtils
