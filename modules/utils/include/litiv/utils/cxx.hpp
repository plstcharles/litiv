
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

// includes here really need cleanup @@@@ split between platform & cxx?

#include "litiv/utils/defines.hpp"
#include <set>
#include <map>
#include <cmath>
#include <mutex>
#include <array>
#include <queue>
#include <deque>
#include <stack>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <future>
#include <memory>
#include <inttypes.h>
#include <cstdarg>
#include <cstdint>
#include <csignal>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <cassert>
#include <ctime>
#include <numeric>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <exception>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <condition_variable>
#if USE_CVCORE_WITH_UTILS
#include <opencv2/core.hpp>
#endif //USE_CVCORE_WITH_UTILS

// @@@ cleanup, move impls down as inline?
// @@@ replace some vector args by templated begin/end iterators? (stl-like)

namespace lv {

    /// helper struct used for compile-time integer expr printing via error; just write "_<expr> test;"
    template<int>
    struct _;

    /// vsnprintf wrapper for std::string output (avoids using two buffers via C++11 string contiguous memory access)
    inline std::string putf(const char* acFormat, ...) {
        va_list vArgs;
        va_start(vArgs,acFormat);
        std::string vBuffer(1024,'\0');
#ifdef _DEBUG
        if(((&vBuffer[0])+vBuffer.size()-1)!=&vBuffer[vBuffer.size()-1])
            throw std::runtime_error("basic_string should have contiguous memory (need C++11!)");
#endif //defined(_DEBUG)
        const int nWritten = vsnprintf(&vBuffer[0],(int)vBuffer.size(),acFormat,vArgs);
        va_end(vArgs);
        if(nWritten<0)
            throw std::runtime_error("putf failed (1)");
        if((size_t)nWritten<=vBuffer.size()) {
            vBuffer.resize((size_t)nWritten);
            return vBuffer;
        }
        vBuffer.resize((size_t)nWritten+1);
        va_list vArgs2;
        va_start(vArgs2,acFormat);
        const int nWritten2 = vsnprintf(&vBuffer[0],(int)vBuffer.size(),acFormat,vArgs2);
        va_end(vArgs2);
        if(nWritten2<0 || (size_t)nWritten2>vBuffer.size())
            throw std::runtime_error("putf failed (2)");
        vBuffer.resize((size_t)nWritten2);
        return vBuffer;
    }

    /// debug helper class which prints status messages in console during stack unwinding (used through lvExceptionWatch macro)
    struct UncaughtExceptionLogger {
        UncaughtExceptionLogger(const char* sFunc, const char* sFile, int nLine) :
                m_sFunc(sFunc),m_sFile(sFile),m_nLine(nLine) {}
        const char* const m_sFunc;
        const char* const m_sFile;
        const int m_nLine;
        ~UncaughtExceptionLogger() {
            if(std::uncaught_exception())
                std::cerr << lv::putf("Unwinding at function '%s' from %s(%d) due to uncaught exception\n",m_sFunc,m_sFile,m_nLine);
        }
    };

    /// high-level (costly) exception class with extra mesagge + info on error location (used through lvAssert and lvError macros)
    struct Exception : public std::runtime_error {
        template<typename... Targs>
        Exception(const std::string& sErrMsg, const char* sFunc, const char* sFile, int nLine, Targs&&... args) :
                std::runtime_error(lv::putf((std::string("Exception in function '%s' from %s(%d) : \n")+sErrMsg).c_str(),sFunc,sFile,nLine,std::forward<Targs>(args)...)),
                m_acFuncName(sFunc),
                m_acFileName(sFile),
                m_nLineNumber(nLine) {
            std::cerr << this->what() << std::endl;
        }
        const char* const m_acFuncName;
        const char* const m_acFileName;
        const int m_nLineNumber;
    };

    /// unique pointer static cast helper utility (moves ownership to the returned pointer)
    template<typename TDerived, typename TBase, typename TDeleter>
    inline std::unique_ptr<TDerived,TDeleter> static_unique_ptr_cast(std::unique_ptr<TBase,TDeleter>&& p) {
        auto d = static_cast<TDerived*>(p.release()); // release does not actually call the object's deleter... (unlike OpenCV's)
        return std::unique_ptr<TDerived,TDeleter>(d,std::move(p.get_deleter()));
    }

    /// unique pointer dynamic cast helper utility (moves ownership to the returned pointer on success only)
    template<typename TDerived, typename TBase, typename TDeleter>
    inline std::unique_ptr<TDerived,TDeleter> dynamic_unique_ptr_cast(std::unique_ptr<TBase,TDeleter>&& p) {
        if(TDerived* result = dynamic_cast<TDerived*>(p.get())) {
            p.release(); // release does not actually call the object's deleter... (unlike OpenCV's)
            return std::unique_ptr<TDerived,TDeleter>(result,std::move(p.get_deleter()));
        }
        return std::unique_ptr<TDerived,TDeleter>(nullptr,p.get_deleter());
    }

    /// explicit loop unroller helper function (specialization for null iter count)
    template<size_t n, typename TFunc>
    constexpr std::enable_if_t<n==0> unroll(const TFunc&) {}

    /// explicit loop unroller helper function (specialization for non-null iter count)
    template<size_t n, typename TFunc>
    constexpr std::enable_if_t<(n>0)> unroll(const TFunc& f) {
        unroll<n-1>(f);
        f(n-1);
    }

    /// returns the comparison of two strings, ignoring character case
    inline bool compare_lowercase(const std::string& i, const std::string& j) {
        std::string i_lower(i), j_lower(j);
        std::transform(i_lower.begin(),i_lower.end(),i_lower.begin(),tolower);
        std::transform(j_lower.begin(),j_lower.end(),j_lower.begin(),tolower);
        return i_lower<j_lower;
    }

    /// returns the number of decimal digits required to display the non-fractional part of a given number (counts sign as extra digit if negative)
    template<typename T>
    inline int digit_count(T number) {
        if(std::isnan((float)number))
            return 3;
        int digits = number<0?2:1;
        while(std::abs(number)>=10) {
            number /= 10;
            digits++;
        }
        return digits;
    }

    /// returns whether the input string contains any of the given tokens
    inline bool string_contains_token(const std::string& s, const std::vector<std::string>& tokens) {
        for(size_t i=0; i<tokens.size(); ++i)
            if(s.find(tokens[i])!=std::string::npos)
                return true;
        return false;
    }

    /// returns the index of all provided values in the reference array (uses std::find to match objects)
    template<typename T>
    inline std::vector<size_t> indices_of(const std::vector<T>& voVals, const std::vector<T>& voRefs) {
        if(voRefs.empty())
            return std::vector<size_t>(voVals.size(),1); // all out-of-range indices
        std::vector<size_t> vnIndices(voVals.size());
        size_t nIdx = 0;
        std::for_each(voVals.begin(),voVals.end(),[&](const T& oVal){
            vnIndices[nIdx++] = std::distance(voRefs.begin(),std::find(voRefs.begin(),voRefs.end(),oVal));
        });
        return vnIndices;
    }

    /// returns the indices mapping for the sorted value array
    template<typename T>
    inline std::vector<size_t> sort_indices(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndices(voVals.size());
        std::iota(vnIndices.begin(),vnIndices.end(),0);
        std::sort(vnIndices.begin(),vnIndices.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]<voVals[n2];
        });
        return vnIndices;
    }

    /// returns the indices mapping for the sorted value array using a custom functor
    template<typename T, typename P>
    inline std::vector<size_t> sort_indices(const std::vector<T>& voVals, P oSortFunctor) {
        std::vector<size_t> vnIndices(voVals.size());
        std::iota(vnIndices.begin(),vnIndices.end(),0);
        std::sort(vnIndices.begin(),vnIndices.end(),oSortFunctor);
        return vnIndices;
    }

    /// returns the indices of the first occurrence of each unique value in the provided array
    template<typename T>
    inline std::vector<size_t> unique_indices(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndices = sort_indices(voVals);
        auto pLastIdxIter = std::unique(vnIndices.begin(),vnIndices.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]==voVals[n2];
        });
        return std::vector<size_t>(vnIndices.begin(),pLastIdxIter);
    }

    /// returns the indices of the first occurrence of each unique value in the provided array using custom sorting/comparison functors
    template<typename T, typename P1, typename P2>
    inline std::vector<size_t> unique_indices(const std::vector<T>& voVals, P1 oSortFunctor, P2 oCompareFunctor) {
        std::vector<size_t> vnIndices = sort_indices(voVals,oSortFunctor);
        auto pLastIdxIter = std::unique(vnIndices.begin(),vnIndices.end(),oCompareFunctor);
        return std::vector<size_t>(vnIndices.begin(),pLastIdxIter);
    }

    /// returns a sorted array of unique values in the iterator range [begin,end[
    template<typename Titer>
    inline std::vector<typename std::iterator_traits<Titer>::value_type> unique(Titer begin, Titer end) {
        const std::set<typename std::iterator_traits<Titer>::value_type> mMap(begin,end);
        return std::vector<typename std::iterator_traits<Titer>::value_type>(mMap.begin(),mMap.end());
    }

    /// returns the index of the nearest neighbor of the requested value in the reference value array, using a custom distance functor
    template<typename Tval, typename Tcomp>
    inline size_t find_nn_index(const Tval& oReqVal, const std::vector<Tval>& vRefVals, Tcomp lCompFunc) {
        if(vRefVals.empty())
            return size_t(-1);
        decltype(lCompFunc(oReqVal,oReqVal)) oMinDist = std::numeric_limits<decltype(lCompFunc(oReqVal,oReqVal))>::max();
        size_t nIdx = size_t(-1);
        for(size_t n=0; n<vRefVals.size(); ++n) {
            auto oCurrDist = lCompFunc(oReqVal,vRefVals[n]);
            if(nIdx==size_t(-1) || oCurrDist<oMinDist) {
                oMinDist = oCurrDist;
                nIdx = n;
            }
        }
        return nIdx;
    }

    /// given a pair vx,vy of arrays describing a 1D function x->y (assumed sorted based on vx), linearly interpolates the outputs (y) for a new set of inputs (x)
    template<typename Tx, typename Ty>
    inline std::vector<Ty> interp1(const std::vector<Tx>& vX, const std::vector<Ty>& vY, const std::vector<Tx>& vXReq) {
        const size_t nSize = std::min(vY.size(),vX.size()); // input vector sizes *should* be identical (use min overlap otherwise)
        if(nSize==0)
            return std::vector<Ty>();
        std::vector<Ty> vSlope(nSize),vOffset(nSize);
        Ty tSlope = Ty();
        for(size_t i=0; i<nSize; ++i) {
            if(i<nSize-1) {
                Tx dX = vX[i+1]-vX[i];
#ifdef _DEBUG
                if(dX<0)
                    throw std::invalid_argument("input domain vector must be sorted");
#endif //defined(_DEBUG)
                Ty dY = vY[i+1]-vY[i];
                tSlope = Ty(dY/dX);
            }
            vSlope[i] = tSlope;
            vOffset[i] = vY[i]-Ty(tSlope*vX[i]);
        }
        const size_t nReqSize = vXReq.size();
        std::vector<Ty> vYReq(nReqSize);
        for(size_t i=0; i<nReqSize; ++i) {
            if(vXReq[i]>=vX.front() && vXReq[i]<=vX.back()) {
                const size_t nNNIdx = lv::find_nn_index(vXReq[i],vX,[](const Tx& a,const Tx& b){return std::abs(b-a);});
                vYReq[i] = vSlope[nNNIdx]*vXReq[i]+vOffset[nNNIdx];
            }
            else
                throw std::domain_error("extrapolation not supported");
        }
        return vYReq;
    }

    /// returns a linearly spaced array of n points in [a,b]
    template<typename T>
    inline std::vector<T> linspace(T a, T b, size_t n, bool bIncludeInitVal=true) {
        if(n==0)
            return std::vector<T>();
        else if(n==1)
            return std::vector<T>(1,b);
        std::vector<T> vResult(n);
        const double dStep = double(b-a)/(n-size_t(bIncludeInitVal));
        if(bIncludeInitVal)
            for(size_t nStepIter = 0; nStepIter<n; ++nStepIter)
                vResult[nStepIter] = T(a+dStep*nStepIter);
        else
            for(size_t nStepIter = 1; nStepIter<=n; ++nStepIter)
                vResult[nStepIter-1] = T(a+dStep*nStepIter);
        return vResult;
    }

    /// implements a work thread pool used to process packaged tasks asynchronously
    template<size_t nWorkers>
    struct WorkerPool {
        static_assert(nWorkers>0,"Worker pool must have at least one work thread");
        /// default constructor; creates 'nWorkers' threads to process queued tasks
        WorkerPool();
        /// default destructor; will block until all queued tasks have been processed
        ~WorkerPool();
        /// queues a task to be processed by the pool, and returns a future tied to its result
        template<typename Tfunc, typename... Targs>
        std::future<std::result_of_t<Tfunc(Targs...)>> queueTask(Tfunc&& lTaskEntryPoint, Targs&&... args);
    protected:
        std::queue<std::function<void()>> m_qTasks;
        std::vector<std::thread> m_vhWorkers;
        std::mutex m_oSyncMutex;
        std::condition_variable m_oSyncVar;
        std::atomic_bool m_bIsActive;
    private:
        void entry();
        WorkerPool(const WorkerPool&) = delete;
        WorkerPool& operator=(const WorkerPool&) = delete;
    };

    /// bitfield expansion function (specialized for unit word size)
    template<size_t nWordBitSize, typename Tr>
    constexpr std::enable_if_t<nWordBitSize==1,Tr> expand_bits(const Tr& nBits, size_t=0) {
        return nBits;
    }

    /// bitfield expansion function (specialized for non-unit word size)
    template<size_t nWordBitSize, typename Tr>
    constexpr std::enable_if_t<(nWordBitSize>1),Tr> expand_bits(const Tr& nBits, size_t n=((sizeof(Tr)*8)/nWordBitSize)-1) {
        static_assert(std::is_integral<Tr>::value,"nBits type must be integral");
        // only the first [(sizeof(Tr)*8)/nWordBitSize] bits are kept (otherwise overflow/ignored)
        return (Tr)(bool((nBits&(1<<n))!=0)<<(n*nWordBitSize)) + ((n>=1)?expand_bits<nWordBitSize,Tr>(nBits,n-1):(Tr)0);
    }

    /// stopwatch/chrono helper class; relies on std::chrono::high_resolution_clock internally
    struct StopWatch {
        StopWatch() {tick();}
        inline void tick() {m_nTick = std::chrono::high_resolution_clock::now();}
        inline double elapsed() const {
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-m_nTick).count();
        }
        inline double tock() {
            const std::chrono::high_resolution_clock::time_point nNow = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> dElapsed_sec = nNow-m_nTick;
            m_nTick = nNow;
            return dElapsed_sec.count();
        }
    private:
        std::chrono::high_resolution_clock::time_point m_nTick;
    };

    /// returns the string of the current 'localtime' output (useful for log tagging)
    inline std::string getTimeStamp() {
        std::time_t tNow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char acBuffer[128];
        std::strftime(acBuffer,sizeof(acBuffer),"%F %T",std::localtime(&tNow)); // std::put_time missing w/ GCC<5.0
        return std::string(acBuffer);
    }

    /// returns the string of the current framework version and version control hash (useful for log tagging)
    inline std::string getVersionStamp() {
        return "LITIV Framework v" LITIV_VERSION_STR " (SHA1=" LITIV_VERSION_SHA1 ")";
    }

    /// returns a combined version of 'getVersionStamp()' and 'getTimeStamp()' for inline use by loggers
    inline std::string getLogStamp() {
        return std::string("\n")+lv::getVersionStamp()+"\n["+lv::getTimeStamp()+"]\n";
    }

    /// clamps a given string to a specific size, padding with a given character if the string is too small
    inline std::string clampString(const std::string& sInput, size_t nSize, char cPadding=' ') {
        return sInput.size()>nSize?sInput.substr(0,nSize):std::string(nSize-sInput.size(),cPadding)+sInput;
    }

    /// accumulates the values of some object members in the given array
    template<typename TSum, typename TObj>
    inline TSum accumulateMembers(const std::vector<TObj>& vObjArray, const std::function<TSum(const TObj&)>& lFunc, TSum tInit=TSum(0)) {
        return std::accumulate(vObjArray.begin(),vObjArray.end(),tInit,[&](TSum tSum, const TObj& p) {
            return tSum + lFunc(p);
        });
    }

    /// concatenates the given vectors (useful for inline constructor where performance doesn't matter too much)
    template<typename To, typename Ta, typename Tb>
    inline std::vector<To> concat(const std::vector<Ta>& a, const std::vector<Tb>& b) {
        std::vector<To> v;
        v.reserve(v.size()+b.size());
        v.insert(v.end(),a.begin(),a.end());
        v.insert(v.end(),b.begin(),b.end());
        return v;
    }

    /// will return all elements of vVals which were not matched to an element in vTokens
    template<typename T>
    inline std::vector<T> filter_out(const std::vector<T>& vVals, const std::vector<T>& vTokens) {
        std::vector<T> vRet;
        vRet.reserve(vVals.size());
        std::copy_if(vVals.begin(),vVals.end(),std::back_inserter(vRet),[&](const T& o){return std::find(vTokens.begin(),vTokens.end(),o)==vTokens.end();});
        return vRet;
    }

    /// will return all elements of vVals which were matched to an element in vTokens
    template<typename T>
    inline std::vector<T> filter_in(const std::vector<T>& vVals, const std::vector<T>& vTokens) {
        std::vector<T> vRet;
        vRet.reserve(vVals.size());
        std::copy_if(vVals.begin(),vVals.end(),std::back_inserter(vRet),[&](const T& o){return std::find(vTokens.begin(),vTokens.end(),o)!=vTokens.end();});
        return vRet;
    }

    /// helper struct for std::enable_shared_from_this to allow easier dynamic casting
    template<typename T>
    struct enable_shared_from_this : public std::enable_shared_from_this<T> {
        /// cast helper function (const version)
        template<typename Tcast>
        inline std::shared_ptr<const Tcast> shared_from_this_cast(bool bThrowIfFail=false) const {
            auto pCast = std::dynamic_pointer_cast<const Tcast>(this->shared_from_this());
            if(bThrowIfFail && !pCast)
                throw std::bad_cast();
            return pCast;
        }
        /// cast helper function (non-const version)
        template<typename Tcast>
        inline std::shared_ptr<Tcast> shared_from_this_cast(bool bThrowIfFail=false) {
            return std::const_pointer_cast<Tcast>(static_cast<const T*>(this)->template shared_from_this_cast<Tcast>(bThrowIfFail));
        }
    };

    /// type traits helper to check if a class possesses a const iterator
    template<template<typename,typename...> class TContainer>
    struct has_const_iterator {
    private:
        template<typename T>
        static char test(typename T::const_iterator*) {return char(0);}
        template<typename T>
        static int test(...) {return int(0);}
    public:
        enum {value=(sizeof(test<TContainer>(0))==sizeof(char))};
    };

    /// helper function to apply a functor to all members of a tuple/array (impl)
    template<typename TTuple, typename TFunc, size_t... anIndices>
    inline void _for_each(TTuple&& t, TFunc f, std::index_sequence<anIndices...>) {
        auto l = {(f(std::get<anIndices>(t)),0)...};
    }

    /// helper function to apply a functor to all members of a tuple
    template<typename... TTupleTypes, typename TFunc>
    inline void for_each_in_tuple(const std::tuple<TTupleTypes...>& t, TFunc f) {
        _for_each(t,f,std::make_index_sequence<sizeof...(TTupleTypes)>{});
    }

    /// computes a 2D array -> 1D array transformation with constexpr support (impl)
    template<typename TValue, size_t nArraySize, typename TFunc, size_t... anIndices>
    constexpr auto _static_transform(const std::array<TValue,nArraySize>& a, const std::array<TValue,nArraySize>& b, TFunc lOp, std::index_sequence<anIndices...>) -> std::array<decltype(lOp(a[0],b[0])),nArraySize> {
        return {lOp(a[anIndices],b[anIndices])...};
    }

    /// computes a 2D array -> 1D array transformation with constexpr support
    template<typename TValue, size_t nArraySize, typename TFunc>
    constexpr auto static_transform(const std::array<TValue,nArraySize>& a, const std::array<TValue,nArraySize>& b, TFunc lOp) -> decltype(_static_transform(a,b,lOp,std::make_index_sequence<nArraySize>{})) {
        return _static_transform(a,b,lOp,std::make_index_sequence<nArraySize>{});
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (iterator-based version)
    template<typename TValue, typename TFunc>
    constexpr auto static_reduce(const TValue* begin, const TValue* end, TFunc lOp) -> decltype(lOp(*begin,*begin)) {
        return (begin>=end)?TValue{}:(begin+1)==end?*begin:lOp(*begin,static_reduce(begin+1,end,lOp));
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (array-based version, impl, specialization for last array value)
    template<size_t nArrayIdx, typename TValue, size_t nArraySize, typename TFunc>
    constexpr std::enable_if_t<nArrayIdx==0,TValue> _static_reduce_impl(const std::array<TValue,nArraySize>& a, TFunc) {
        return std::get<nArrayIdx>(a);
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (array-based version, impl, specialization for non-last array value)
    template<size_t nArrayIdx, typename TValue, size_t nArraySize, typename TFunc>
    constexpr std::enable_if_t<(nArrayIdx>0),std::result_of_t<TFunc(const TValue&,const TValue&)>> _static_reduce_impl(const std::array<TValue,nArraySize>& a, TFunc lOp) {
        return lOp(std::get<nArrayIdx>(a),_static_reduce_impl<nArrayIdx-1>(a,lOp));
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (array-based version)
    template<typename TValue, size_t nArraySize, typename TFunc>
    constexpr auto static_reduce(const std::array<TValue,nArraySize>& a, TFunc lOp) -> decltype(lOp(std::get<0>(a),std::get<0>(a))) {
        static_assert(nArraySize>0,"need non-empty array for reduction");
        return _static_reduce_impl<nArraySize-1>(a,lOp);
    }

    /// helper structure to create lookup tables with generic functors (also exposes multiple lookup interfaces)
    template<typename Tx, typename Ty, size_t nBins, size_t nSafety=0>
    struct LUT {
        static_assert(nBins>1 && (nBins%2)==1,"LUT bin count must be at least two and odd");
        /// default constructor; will automatically fill the LUT array for lFunc([tMinLookup,tMaxLookup])
        template<typename TFunc>
        LUT(Tx tMinLookup, Tx tMaxLookup, TFunc lFunc) :
                m_tMin(std::min(tMinLookup,tMaxLookup)),m_tMax(std::max(tMinLookup,tMaxLookup)),
                m_tMidOffset((m_tMax+m_tMin)/2),m_tLowOffset(m_tMin),
                m_fScale(float(nBins-1)/(m_tMax-m_tMin)),m_vLUT(init(m_tMin,m_tMax,lFunc)),
                m_pMid(m_vLUT.data()+nBins/2+nSafety),m_pLow(m_vLUT.data()+nSafety) {}
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer after offsetting x (i.e. assuming tOffset!=0)
        inline Ty eval_mid(Tx x) const {
            lvDbgAssert(ptrdiff_t((x-m_tMidOffset)*m_fScale)>=-ptrdiff_t(nBins/2+nSafety) && ptrdiff_t((x-m_tMidOffset)*m_fScale)<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[ptrdiff_t((x-m_tMidOffset)*m_fScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer without offsetting x (i.e. assuming tOffset==0)
        inline Ty eval_mid_noffset(Tx x) const {
            lvDbgAssert(ptrdiff_t(x*m_fScale)>=-ptrdiff_t(nBins/2+nSafety) && ptrdiff_t(x*m_fScale)<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[ptrdiff_t(x*m_fScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer with x as a direct index value
        inline Ty eval_mid_raw(ptrdiff_t x) const {
            lvDbgAssert(x>=-ptrdiff_t(nBins/2+nSafety) && x<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[x];
        }
        /// returns the evaluation result of lFunc(x) using the LUT low pointer after offsetting x (i.e. assuming tOffset!=0)
        inline Ty eval(Tx x) const {
            lvDbgAssert(ptrdiff_t((x-m_tLowOffset)*m_fScale)>=-ptrdiff_t(nSafety) && ptrdiff_t((x-m_tLowOffset)*m_fScale)<ptrdiff_t(nBins+nSafety));
            return m_pLow[ptrdiff_t((x-m_tLowOffset)*m_fScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT low pointer without offsetting x (i.e. assuming tOffset==0)
        inline Ty eval_noffset(Tx x) const {
            lvDbgAssert(ptrdiff_t(x*m_fScale)>=-ptrdiff_t(nSafety) && ptrdiff_t(x*m_fScale)<ptrdiff_t(nBins+nSafety));
            return m_pLow[ptrdiff_t(x*m_fScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT low pointer with x as a direct index value
        inline Ty eval_raw(ptrdiff_t x) const {
            lvDbgAssert(x>=-ptrdiff_t(nSafety) && x<ptrdiff_t(nBins+nSafety));
            return m_pLow[x];
        }
        /// min/max lookup values passed to the constructor (LUT bounds)
        const Tx m_tMin,m_tMax;
        /// input value offsets for lookup
        const Tx m_tMidOffset,m_tLowOffset;
        /// scale coefficient for lookup
        const float m_fScale;
        /// functor lookup table
        const std::vector<Ty> m_vLUT;
        /// base LUT pointers for lookup
        const Ty* m_pMid,*m_pLow;
    private:
        template<typename TFunc>
        static std::vector<Ty> init(Tx tMin, Tx tMax, TFunc lFunc) {
            std::vector<Ty> vLUT(nBins+nSafety*2);
            for(size_t n=0; n<=nSafety; ++n)
                vLUT[n] = lFunc(tMin);
            const Tx tStep = (tMax-tMin)/(nBins-1);
            for(size_t n=nSafety+1; n<nBins+nSafety-1; ++n)
                vLUT[n] = lFunc(tMin+(Tx)(n*tStep));
            for(size_t n=nBins+nSafety-1; n<nBins+nSafety*2; ++n)
                vLUT[n] = lFunc(tMax);
            return vLUT;
        }
    };

    /// helper class used to unlock several mutexes in the current scope (logical inverse of lock_guard)
    template<typename... TMutexes>
    struct unlock_guard {
        /// unlocks all given mutexes, one at a time, until destruction
        explicit unlock_guard(TMutexes&... aMutexes) noexcept :
                m_aMutexes(aMutexes...) {
            for_each_in_tuple(m_aMutexes,[](auto& oMutex) noexcept {oMutex.unlock();});
        }
        /// relocks all mutexes simultaneously
        ~unlock_guard() {
            std::lock(m_aMutexes);
        }
    private:
        std::tuple<TMutexes&...> m_aMutexes;
        unlock_guard(const unlock_guard&) = delete;
        unlock_guard& operator=(const unlock_guard&) = delete;
    };

    /// helper class used to unlock a mutex in the current scope (logical inverse of lock_guard)
    template<typename TMutex>
    struct unlock_guard<TMutex> {
        /// unlocks the given mutex until destruction
        explicit unlock_guard(TMutex& oMutex) noexcept :
                m_oMutex(oMutex) {
            m_oMutex.unlock();
        }
        /// relocks the initially provided mutex
        ~unlock_guard() {
            m_oMutex.lock();
        }
    private:
        TMutex& m_oMutex;
        unlock_guard(const unlock_guard&) = delete;
        unlock_guard& operator=(const unlock_guard&) = delete;
    };

    /// simple semaphore implementation based on STL's conditional variable/mutex combo
    struct Semaphore {
        /// initializes internal resource count to 'nInitCount'
        inline explicit Semaphore(size_t nInitCount) :
                m_nCount(nInitCount) {}
        /// notifies one waiter that a resource is now available
        inline void notify() {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            ++m_nCount;
            m_oCondVar.notify_one();
        }
        /// blocks and waits for a resource to become available (returning instantly if one already is)
        inline void wait() {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            m_oCondVar.wait(oLock,[&]{return m_nCount>0;});
            --m_nCount;
        }
        /// checks if a resource is available, capturing it if so, and always returns immediately
        inline bool try_wait() {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            if(m_nCount>0) {
                --m_nCount;
                return true;
            }
            return false;
        }
        /// blocks and waits for a resource to become available (with timeout support)
        template<typename TRep, typename TPeriod=std::ratio<1>>
        inline bool wait_for(const std::chrono::duration<TRep,TPeriod>& nDuration) {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            bool bFinished = m_oCondVar.wait_for(oLock,nDuration,[&]{return m_nCount>0;});
            if(bFinished)
                --m_nCount;
            return bFinished;
        }
        /// blocks and waits for a resource to become available (with time limit support)
        template<typename TClock, typename TDuration=typename TClock::duration>
        inline bool wait_until(const std::chrono::time_point<TClock,TDuration>& nTimePoint) {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            bool bFinished = m_oCondVar.wait_until(oLock,nTimePoint,[&]{return m_nCount>0;});
            if(bFinished)
                --m_nCount;
            return bFinished;
        }
        using native_handle_type = std::condition_variable::native_handle_type;
        /// returns the os-native handle to the internal condition variable
        inline native_handle_type native_handle() {
            return m_oCondVar.native_handle();
        }
    private:
        std::condition_variable m_oCondVar;
        std::mutex m_oMutex;
        size_t m_nCount;
        Semaphore(const Semaphore&) = delete;
        Semaphore& operator=(const Semaphore&) = delete;
    };

} // namespace lv

namespace std { // extending std

    template<typename T>
    using unlock_guard = lv::unlock_guard<T>;
    using semaphore = lv::Semaphore;
    using mutex_lock_guard = lock_guard<mutex>;
    using mutex_unique_lock = unique_lock<mutex>;

} // namespace std

template<size_t nWorkers>
lv::WorkerPool<nWorkers>::WorkerPool() : m_bIsActive(true) {
    lv::unroll<nWorkers>([this](size_t){m_vhWorkers.emplace_back(std::bind(&WorkerPool::entry,this));});
}

template<size_t nWorkers>
lv::WorkerPool<nWorkers>::~WorkerPool() {
    m_bIsActive = false;
    m_oSyncVar.notify_all();
    for(std::thread& oWorker : m_vhWorkers)
        oWorker.join();
}

template<size_t nWorkers>
template<typename Tfunc, typename... Targs>
std::future<std::result_of_t<Tfunc(Targs...)>> lv::WorkerPool<nWorkers>::queueTask(Tfunc&& lTaskEntryPoint, Targs&&... args) {
    if(!m_bIsActive)
        throw std::runtime_error("cannot queue task, destruction in progress");
    using task_return_t = std::result_of_t<Tfunc(Targs...)>;
    using task_t = std::packaged_task<task_return_t()>;
    // http://stackoverflow.com/questions/28179817/how-can-i-store-generic-packaged-tasks-in-a-container
    std::shared_ptr<task_t> pSharableTask = std::make_shared<task_t>(std::bind(std::forward<Tfunc>(lTaskEntryPoint),std::forward<Targs>(args)...));
    std::future<task_return_t> oTaskRes = pSharableTask->get_future();
    {
        std::mutex_lock_guard sync_lock(m_oSyncMutex);
        m_qTasks.emplace([pSharableTask](){(*pSharableTask)();}); // lambda keeps a copy of the task in the queue
    }
    m_oSyncVar.notify_one();
    return oTaskRes;
}

template<size_t nWorkers>
void lv::WorkerPool<nWorkers>::entry() {
    std::mutex_unique_lock sync_lock(m_oSyncMutex);
    while(m_bIsActive || !m_qTasks.empty()) {
        if(m_qTasks.empty())
            m_oSyncVar.wait(sync_lock);
        if(!m_qTasks.empty()) {
            std::function<void()> task = std::move(m_qTasks.front());
            m_qTasks.pop();
            std::unlock_guard<std::mutex_unique_lock> oUnlock(sync_lock);
            task(); // if the execution throws, the exception will be contained in the shared state returned on queue
        }
    }
}
