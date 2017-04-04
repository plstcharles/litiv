
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

#include "litiv/utils/defines.hpp"
#include <set>
#include <cmath>
#include <mutex>
#include <array>
#include <queue>
#include <tuple>
#include <utility>
#include <type_traits>
#include <iterator>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <future>
#include <memory>
#include <inttypes.h>
#include <cstddef>
#include <cstdarg>
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <numeric>
#include <cstdio>
#include <ratio>
#include <exception>
#include <algorithm>
#include <cctype>
#include <clocale>
#include <functional>
#include <condition_variable>
#if USE_CVCORE_WITH_UTILS
#include <opencv2/core.hpp>
#endif //USE_CVCORE_WITH_UTILS

// @@@ cleanup, move impls down as inline?
// @@@ replace some vector args by templated begin/end iterators? (stl-like)

#define NEXT_BIGGER_INTEGER(curr, next) \
    template<> \
    struct get_bigger_integer<curr> {\
        typedef next type; \
    }

namespace lv {

    /// vsnprintf wrapper for std::string output (avoids using two buffers via C++11 string contiguous memory access)
    std::string putf(const char* acFormat, ...);
    /// returns the comparison of two strings, ignoring character case
    bool compare_lowercase(const std::string& i, const std::string& j);
    /// returns whether the input string contains any of the given tokens
    bool string_contains_token(const std::string& s, const std::vector<std::string>& tokens);
    /// clamps a given string to a specific size, padding with a given character if the string is too small
    std::string clampString(const std::string& sInput, size_t nSize, char cPadding=' ');
    /// returns the string of the current 'localtime' output (useful for log tagging)
    std::string getTimeStamp();
    /// returns the string of the current framework version and version control hash (useful for log tagging)
    std::string getVersionStamp();
    /// returns a combined version of 'getVersionStamp()' and 'getTimeStamp()' for inline use by loggers
    std::string getLogStamp();
    /// returns the global verbosity level (greater = more verbose, default = 1)
    int getVerbosity();
    /// sets the global verbosity level (greater = more verbose, default = 1)
    void setVerbosity(int nLevel);
    /// prevents a data block given by a char pointer from being optimized away
    void doNotOptimizeCharPointer(char const volatile*);

    /// prevents a value/expression from being optimized away (see https://youtu.be/nXaxk27zwlk?t=2441)
    template<typename T>
    inline void doNotOptimize(const T& v) {
#if defined(__GNUC__)
        asm volatile("" : : "g"(v) : "memory");
#else //ndef(__GNUC__)
        doNotOptimizeCharPointer(&reinterpret_cast<char const volatile&>(v));
#endif //ndef(__GNUC__)
    }

    /// helper struct used for compile-time integer expr printing via error; just write "IntegerPrinter<expr> test;"
    template<int>
    struct IntegerPrinter;

    /// helper struct used for compile-time typename printing via error; just write "TypePrinter<type> test;"
    template<typename T>
    struct TypePrinter;

#ifndef LV_UNCAUGHT_EXCEPT_LOGGER_DECL
#define LV_UNCAUGHT_EXCEPT_LOGGER_DECL
    /// debug helper class which prints status messages in console during stack unwinding (used through lvExceptionWatch macro)
    struct UncaughtExceptionLogger {
        inline UncaughtExceptionLogger(const char* sFunc, const char* sFile, int nLine) :
                m_sFunc(sFunc),m_sFile(sFile),m_nLine(nLine) {}
        const char* const m_sFunc;
        const char* const m_sFile;
        const int m_nLine;
        inline ~UncaughtExceptionLogger() {
            if(std::uncaught_exception() && lv::getVerbosity()>=1)
                std::cerr << lv::putf("Unwinding due to uncaught exception at function '%s'\n\t... from %s(%d)\n",m_sFunc,m_sFile,m_nLine);
        }
    };
#endif //LV_UNCAUGHT_EXCEPT_LOGGER_DECL

#ifndef LV_EXCEPTION_DECL
#define LV_EXCEPTION_DECL
    /// high-level (costly) exception class with extra mesagge + info on error location (used through lvAssert and lvError macros)
    struct Exception : public std::runtime_error {
        /// default lv exception constructor; all exception should be created via macros
        template<typename... Targs>
        inline Exception(const std::string& sErrMsg, const char* sFunc, const char* sFile, int nLine, Targs&&... args) :
                std::runtime_error(lv::putf((std::string("Exception in function '%s'\n\t... from %s(%d)\n\t... what = ")+sErrMsg).c_str(),sFunc,sFile,nLine,std::forward<Targs>(args)...)),
                m_acFuncName(sFunc),m_acFileName(sFile),m_nLineNumber(nLine) {
            if(lv::getVerbosity()>=1)
                std::cerr << this->what() << std::endl;
        }
        /// name of the function the exception originated from
        const char* const m_acFuncName;
        /// name of the file the exception originated from
        const char* const m_acFileName;
        /// line number the exception originated from
        const int m_nLineNumber;
    };
#endif //LV_EXCEPTION_DECL

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
    std::enable_if_t<n==0> unroll(const TFunc&) {}

    /// explicit loop unroller helper function (specialization for non-null iter count)
    template<size_t n, typename TFunc>
    std::enable_if_t<(n>0)> unroll(const TFunc& f) {
        unroll<n-1>(f);
        f(n-1);
    }

    /// returns the number of decimal digits required to display the non-fractional part of a given number (counts sign as extra digit if negative)
    template<typename T>
    inline int digit_count(T number) {
        if(std::isnan((float)number))
            return 3;
        int digits = number<0?2:1;
        while(std::abs((int)number)>=10) {
            number /= 10;
            digits++;
        }
        return digits;
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

    /// copies an array of objects to a vector (objects must be default-constructible)
    template<size_t nSize, typename T>
    inline void copyArrayToVector(const std::array<T,nSize>& aDataIn, std::vector<T>& vDataOut) {
        vDataOut.resize(aDataIn.size());
        std::copy_n(aDataIn.begin(),nSize,vDataOut.begin());
    }

    /// converts an array of objects to a vector (objects must be default-constructible)
    template<size_t nSize, typename T>
    inline std::vector<T> convertArrayToVector(const std::array<T,nSize>& aDataIn) {
        std::vector<T> vDataOut;
        copyArrayToVector(aDataIn,vDataOut);
        return vDataOut;
    }

    /// copies a vector of objects to an array (must specify size in template, and objects must be default-constructible)
    template<size_t nSize, typename T>
    inline void copyVectorToArray(const std::vector<T>& vDataIn, std::array<T,nSize>& aDataOut) {
        lvAssert_(vDataIn.size()==nSize,"bad input vector size");
        std::copy_n(vDataIn.begin(),nSize,aDataOut.begin());
    }

    /// converts a vector of objects to an array (must specify size in template, and objects must be default-constructible)
    template<size_t nSize, typename T>
    inline std::array<T,nSize> convertVectorToArray(const std::vector<T>& vDataIn) {
        lvAssert_(vDataIn.size()==nSize,"bad input vector size");
        std::array<T,nSize> aDataOut;
        copyVectorToArray(vDataIn,aDataOut);
        return aDataOut;
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

    /// accumulates the values of some object members in the given array
    template<typename TSum, typename TObj, typename TFunc>
    inline TSum accumulateMembers(const std::vector<TObj>& vObjArray, TFunc lObjEval, TSum tInit=TSum(0)) {
        return std::accumulate(vObjArray.begin(),vObjArray.end(),tInit,[&](TSum tSum, const TObj& p) {
            return tSum + lObjEval(p);
        });
    }

    /// returns the index of all provided values in the reference array (uses std::find to match objects)
    template<typename Tin, typename Tout=size_t>
    inline std::vector<Tout> indices_of(const std::vector<Tin>& voVals, const std::vector<Tin>& voRefs) {
        if(voRefs.empty())
            return std::vector<Tout>(voVals.size(),1); // all out-of-range indices
        std::vector<Tout> vnIndices(voVals.size());
        size_t nIdx = 0;
        std::for_each(voVals.begin(),voVals.end(),[&](const Tin& oVal){
            vnIndices[nIdx++] = (Tout)std::distance(voRefs.begin(),std::find(voRefs.begin(),voRefs.end(),oVal));
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

    /// returns the vector of all integer values in the [a,b] range (empty if b<a), with optional step size
    template<typename Tinteger>
    inline std::vector<Tinteger> make_range(Tinteger a, Tinteger b, Tinteger nStep=Tinteger(1)) {
        static_assert(std::is_integral<Tinteger>::value,"input type must be integral");
        if(b<a)
            return std::vector<Tinteger>{};
        if(nStep==Tinteger(1)) {
            std::vector<Tinteger> vRet(size_t(b-a)+1);
            std::iota(vRet.begin(),vRet.end(),a);
            return vRet;
        }
        lvAssert_(nStep>0,"specified step size must be strictly positive");
        lvAssert_((size_t(b-a)%nStep)==size_t(0),"interval size must be a multiple of integer step size");
        std::vector<Tinteger> vRet(size_t(b-a)/nStep+1);
        Tinteger nVal = (vRet[0]=a);
        std::generate(vRet.begin()+1,vRet.end(),[&nVal,&nStep](){return nVal+=nStep;});
        return vRet;
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

    /// stopwatch/chrono helper class; relies on std::chrono::high_resolution_clock internally
    struct StopWatch {
        /// default constructor; calls 'tick' for member initialization
        inline StopWatch() {
            tick();
        }
        /// updates the internal clock tick with the current tick
        inline void tick() {
            m_nTick = std::chrono::high_resolution_clock::now();
        }
        /// returns the elapsed time (in seconds) between the last 'tick' call and now
        inline double elapsed() const {
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-m_nTick).count();
        }
        /// returns the elapsed time (in seconds) between the last 'tick' call and now, and sets 'tick' as now
        inline double tock() {
            const std::chrono::high_resolution_clock::time_point nNow = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> dElapsed_sec = nNow-m_nTick;
            m_nTick = nNow;
            return dElapsed_sec.count();
        }
    private:
        std::chrono::high_resolution_clock::time_point m_nTick;
    };

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
            auto pCast = std::dynamic_pointer_cast<Tcast>(this->shared_from_this());
            if(bThrowIfFail && !pCast)
                throw std::bad_cast();
            return pCast;
        }
    };

    /// type traits helper to check if a class possesses a const iterator
    template<typename TContainer>
    struct has_const_iterator {
    private:
        template<typename T>
        static char test(typename T::const_iterator*) {return char(0);}
        template<typename T>
        static int test(...) {return int(0);}
    public:
        enum {value=(sizeof(test<TContainer>(0))==sizeof(char))};
    };

    /// type traits helper which provides the smallest integer type that is bigger than Tint
    template<typename Tint>
    struct get_bigger_integer {
        static_assert(!std::is_same<uint64_t,Tint>::value,"missing uint128_t impl");
        static_assert(!std::is_same<int64_t,Tint>::value,"missing int128_t impl");
    };
    NEXT_BIGGER_INTEGER(char,int16_t);
    NEXT_BIGGER_INTEGER(uint8_t,uint16_t);
    NEXT_BIGGER_INTEGER(uint16_t,uint32_t);
    NEXT_BIGGER_INTEGER(uint32_t,uint64_t);
    NEXT_BIGGER_INTEGER(int8_t,int16_t);
    NEXT_BIGGER_INTEGER(int16_t,int32_t);
    NEXT_BIGGER_INTEGER(int32_t,int64_t);

    /// helper function to apply a functor to all members of a tuple/array (impl)
    template<typename TFunc, typename TTuple, size_t... anIndices>
    inline void _for_each(TTuple&& t, TFunc f, std::index_sequence<anIndices...>) {
        auto l = {(f(std::get<anIndices>(t)),0)...}; UNUSED(l);
    }

    /// helper function to apply a functor to all members of a tuple
    template<typename TFunc, typename... TTupleTypes>
    inline void for_each(const std::tuple<TTupleTypes...>& t, TFunc f) {
        _for_each<TFunc>(t,f,std::make_index_sequence<sizeof...(TTupleTypes)>{});
    }

    /// helper function to apply a functor to all members of an array
    template<typename TFunc, typename TArrayType, size_t nArraySize>
    inline void for_each(const std::array<TArrayType,nArraySize>& a, TFunc f) {
        _for_each<TFunc>(a,f,std::make_index_sequence<nArraySize>{});
    }

    /// helper function to apply a functor to all members of a tuple/array (impl, also passes index to functor)
    template<typename TFunc, typename TTuple, size_t... anIndices>
    inline void _for_each_w_idx(TTuple&& t, TFunc f, std::index_sequence<anIndices...>) {
        auto l = {(f(std::get<anIndices>(t),anIndices),0)...}; UNUSED(l);
    }

    /// helper function to apply a functor to all members of a tuple (also passes index to functor)
    template<typename TFunc, typename... TTupleTypes>
    inline void for_each_w_idx(const std::tuple<TTupleTypes...>& t, TFunc f) {
        _for_each_w_idx<TFunc>(t,f,std::make_index_sequence<sizeof...(TTupleTypes)>{});
    }

    /// helper function to apply a functor to all members of an array (also passes index to functor)
    template<typename TFunc, typename TArrayType, size_t nArraySize>
    inline void for_each_w_idx(const std::array<TArrayType,nArraySize>& a, TFunc f) {
        _for_each_w_idx<TFunc>(a,f,std::make_index_sequence<nArraySize>{});
    }

    /// helper function to unpack a tuple/array into function arguments (impl)
    template<typename TFunc, typename TTuple, size_t... anIndices>
    inline auto _unpack_and_call(const TTuple& t, TFunc f, std::index_sequence<anIndices...>) {
        return f(std::get<anIndices>(t)...);
    }

    /// helper function to unpack tuple into function arguments
    template<typename TFunc, typename... TTupleTypes>
    inline auto unpack_and_call(const std::tuple<TTupleTypes...>& t, TFunc f) {
        return _unpack_and_call(t,f,std::make_index_sequence<sizeof...(TTupleTypes)>{});
    }

    /// helper function to unpack array into function arguments
    template<typename TFunc, typename TArrayType, size_t nArraySize>
    inline auto unpack_and_call(const std::array<TArrayType,nArraySize>& a, TFunc f) {
        return _unpack_and_call(a,f,std::make_index_sequence<nArraySize>{});
    }

    /// computes a 1D array -> 1D array transformation with constexpr support (impl)
    template<typename TFunc, typename TValue, size_t nArraySize, size_t... anIndices>
    constexpr auto _static_transform(const std::array<TValue,nArraySize>& a, TFunc lOp, std::index_sequence<anIndices...>) -> std::array<decltype(lOp(a[0])),nArraySize> {
        return {lOp(a[anIndices])...};
    }

    /// computes a 1D array -> 1D array transformation with constexpr support
    template<typename TFunc, typename TValue, size_t nArraySize>
    constexpr auto static_transform(const std::array<TValue,nArraySize>& a, TFunc lOp) -> decltype(_static_transform(a,lOp,std::make_index_sequence<nArraySize>{})) {
        return _static_transform(a,lOp,std::make_index_sequence<nArraySize>{});
    }

    /// computes a 2D array -> 1D array transformation with constexpr support (impl)
    template<typename TFunc, typename TValue, size_t nArraySize, size_t... anIndices>
    constexpr auto _static_transform(const std::array<TValue,nArraySize>& a, const std::array<TValue,nArraySize>& b, TFunc lOp, std::index_sequence<anIndices...>) -> std::array<decltype(lOp(a[0],b[0])),nArraySize> {
        return {lOp(a[anIndices],b[anIndices])...};
    }

    /// computes a 2D array -> 1D array transformation with constexpr support
    template<typename TFunc, typename TValue, size_t nArraySize>
    constexpr auto static_transform(const std::array<TValue,nArraySize>& a, const std::array<TValue,nArraySize>& b, TFunc lOp) -> decltype(_static_transform(a,b,lOp,std::make_index_sequence<nArraySize>{})) {
        return _static_transform(a,b,lOp,std::make_index_sequence<nArraySize>{});
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (iterator-based version)
    template<typename TFunc, typename TValue>
    constexpr auto static_reduce(const TValue* begin, const TValue* end, TFunc lOp) -> decltype(lOp(*begin,*begin)) {
        return (begin>=end)?throw std::runtime_error("bad iters"):(begin+1)==end?*begin:lOp(*begin,static_reduce(begin+1,end,lOp));
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (array-based version, impl, specialization for last array value)
    template<size_t nArrayIdx, typename TFunc, typename TValue, size_t nArraySize>
    constexpr std::enable_if_t<nArrayIdx==0,TValue> _static_reduce_impl(const std::array<TValue,nArraySize>& a, TFunc) {
        return std::get<nArrayIdx>(a);
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (array-based version, impl, specialization for non-last array value)
    template<size_t nArrayIdx, typename TFunc, typename TValue, size_t nArraySize>
    constexpr std::enable_if_t<(nArrayIdx>0),std::result_of_t<TFunc(const TValue&,const TValue&)>> _static_reduce_impl(const std::array<TValue,nArraySize>& a, TFunc lOp) {
        return lOp(std::get<nArrayIdx>(a),_static_reduce_impl<nArrayIdx-1>(a,lOp));
    }

    /// computes a 1D array -> 1D scalar reduction with constexpr support (array-based version)
    template<typename TFunc, typename TValue, size_t nArraySize>
    constexpr auto static_reduce(const std::array<TValue,nArraySize>& a, TFunc lOp) -> decltype(lOp(std::get<0>(a),std::get<0>(a))) {
        static_assert(nArraySize>0,"need non-empty array for reduction");
        return _static_reduce_impl<nArraySize-1>(a,lOp);
    }

    /// helper constexpr 'logical and' folding expression for often-used static array reduction
    constexpr bool static_reduce_and(bool a, bool b) {
        return a&&b;
    }

    /// helper constexpr addition folding expression for often-used static array reduction
    template<typename T>
    constexpr T static_reduce_add(T a, T b) {
        return a+b;
    }

    /// defines an stl-friendly aligned+default-init memory allocator to be used in container classes
    template<typename T, std::size_t nByteAlign, bool bDefaultInit=true>
    struct AlignedMemAllocator {
        static_assert(nByteAlign>0,"byte alignment must be a non-null value");
        static_assert(!std::is_const<T>::value,"ill-formed: container of const elements forbidden");
        typedef T value_type;
        typedef T* pointer;
        typedef T& reference;
        typedef const T* const_pointer;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::true_type propagate_on_container_move_assignment;
        typedef AlignedMemAllocator<T,nByteAlign,bDefaultInit> this_type;
        template<typename T2>
        using similar_type = AlignedMemAllocator<T2,nByteAlign,bDefaultInit>;
        template<typename T2>
        struct rebind {typedef similar_type<T2> other;};
        inline AlignedMemAllocator() noexcept {}
        template<typename T2>
        inline AlignedMemAllocator(const similar_type<T2>&) noexcept {}
        template<typename T2>
        inline this_type& operator=(const similar_type<T2>&) noexcept {return *this;}
        template<typename T2>
        inline AlignedMemAllocator(similar_type<T2>&&) noexcept {}
        template<typename T2>
        inline this_type& operator=(similar_type<T2>&&) noexcept {return *this;}
        inline ~AlignedMemAllocator() noexcept {}
        static inline pointer address(reference r) noexcept {return std::addressof(r);}
        static inline const_pointer address(const_reference r) noexcept {return std::addressof(r);}
#ifdef _MSC_VER
        static inline pointer allocate(size_type n) {
            const size_type alignment = static_cast<size_type>(nByteAlign);
            size_t alloc_size = n*sizeof(value_type);
            if((alloc_size%alignment)!=0)
                alloc_size += alignment - alloc_size%alignment;
            void* ptr = _aligned_malloc(alloc_size,nByteAlign);
            if(ptr==nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }
        static inline void deallocate(pointer p, size_type) noexcept {_aligned_free(p);}
        static inline void destroy(pointer p) {p->~value_type();UNUSED(p);}
#else //(!def(_MSC_VER))
        static inline pointer allocate(size_type n) {
            const size_type alignment = static_cast<size_type>(nByteAlign);
            size_t alloc_size = n*sizeof(value_type);
            if((alloc_size%alignment)!=0)
                alloc_size += alignment - alloc_size%alignment;
#if HAVE_STL_ALIGNED_ALLOC
            void* ptr = aligned_alloc(alignment,alloc_size);
#elif HAVE_POSIX_ALIGNED_ALLOC
            void* ptr;
            if(posix_memalign(&ptr,alignment,alloc_size)!=0)
                throw std::bad_alloc();
#else //HAVE_..._ALIGNED_ALLOC
#error "Missing aligned mem allocator"
#endif //HAVE_..._ALIGNED_ALLOC
            if(ptr==nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }
        static inline void deallocate(pointer p, size_type) noexcept {free(p);}
        static inline void destroy(pointer p) {p->~value_type();}
#endif //(!def(_MSC_VER))
        template<typename T2, typename... TArgs>
        static inline void construct(T2* p, TArgs&&... args) {
            ::new((void*)p) T2(std::forward<TArgs>(args)...);
        }
        static_assert(!bDefaultInit || std::is_default_constructible<value_type>::value,"object type not default-constructible");
        template<typename T2, typename... TDummy, bool _bDefaultInit=bDefaultInit>
        static inline std::enable_if_t<_bDefaultInit> construct(T2* p) noexcept(std::is_nothrow_default_constructible<T2>::value) {
            static_assert(sizeof...(TDummy)==0,"template args not needed");
            ::new((void*)p) T2;
        }
        static inline size_type max_size() noexcept {return (size_type(~0)-size_type(nByteAlign))/sizeof(value_type);}
        bool operator!=(const this_type& other) const noexcept {return !(*this==other);}
        bool operator==(const this_type&) const noexcept {return true;}
    };

    /// defines an stl-friendly default-init memory allocator to be used in container classes
    template<typename T, typename TAlloc=std::allocator<T>>
    struct DefaultInitAllocator : TAlloc {
        using TAlloc::TAlloc;
        template<typename T2> struct rebind {
            typedef DefaultInitAllocator<T2,typename std::allocator_traits<TAlloc>::template rebind_alloc<T2>> other;
        };
        template<typename T2, typename... TArgs>
        void construct(T2* p, TArgs&&... args) {
            std::allocator_traits<TAlloc>::construct(static_cast<TAlloc&>(*this),p,std::forward<TArgs>(args)...);
        }
        template<typename T2>
        void construct(T2* p) noexcept(std::is_nothrow_default_constructible<T2>::value) {
            ::new((void*)p) T2;
        }
    };

    /// helper alias; std-friendly version of vector with N-byte aligned memory allocator
    template<typename T, std::size_t nByteAlign, bool bDefaultInit=false> // vectors are typically value-init
    using aligned_vector = std::vector<T,lv::AlignedMemAllocator<T,nByteAlign,bDefaultInit>>;

    /// helper structure to create lookup tables with generic functors (also exposes multiple lookup interfaces)
    template<typename Tx, typename Ty, size_t nBins, size_t nSafety=0, typename TStep=float>
    struct LUT {
        static_assert(nBins>1 && (nBins%2)==1,"LUT bin count must be at least two and odd");
        /// default constructor; will automatically fill the LUT array for lFunc([tMinLookup,tMaxLookup])
        template<typename TFunc>
        LUT(Tx tMinLookup, Tx tMaxLookup, TFunc lFunc) :
                m_tMin(std::min(tMinLookup,tMaxLookup)),m_tMax(std::max(tMinLookup,tMaxLookup)),
                m_tMidOffset((m_tMax+m_tMin)/2),m_tLowOffset(m_tMin),
                m_tScale(TStep(nBins-1)/(m_tMax-m_tMin)),
                m_tStep(TStep(m_tMax-m_tMin)/(nBins-1)),
                m_vLUT(init(m_tMin,m_tMax,m_tStep,lFunc)),
                m_pMid(m_vLUT.data()+nBins/2+nSafety),
                m_pLow(m_vLUT.data()+nSafety) {}
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer after offsetting and scaling x (i.e. assuming tOffset!=0)
        inline Ty eval_mid(Tx x) const {
            lvDbgAssert(ptrdiff_t((x-m_tMidOffset)*m_tScale)>=-ptrdiff_t(nBins/2+nSafety) && ptrdiff_t((x-m_tMidOffset)*m_tScale)<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[ptrdiff_t((x-m_tMidOffset)*m_tScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer after offsetting, scaling, and rounding x (i.e. assuming tOffset!=0)
        inline Ty eval_mid_round(Tx x) const {
            lvDbgAssert((ptrdiff_t)std::llround((x-m_tMidOffset)*m_tScale)>=-ptrdiff_t(nBins/2+nSafety) && (ptrdiff_t)std::llround((x-m_tMidOffset)*m_tScale)<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[(ptrdiff_t)std::llround((x-m_tMidOffset)*m_tScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer after scaling x (i.e. assuming tOffset==0)
        inline Ty eval_mid_noffset(Tx x) const {
            lvDbgAssert(ptrdiff_t(x*m_tScale)>=-ptrdiff_t(nBins/2+nSafety) && ptrdiff_t(x*m_tScale)<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[ptrdiff_t(x*m_tScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer after scaling and rounding x (i.e. assuming tOffset==0)
        inline Ty eval_mid_noffset_round(Tx x) const {
            lvDbgAssert((ptrdiff_t)std::llround(x*m_tScale)>=-ptrdiff_t(nBins/2+nSafety) && (ptrdiff_t)std::llround(x*m_tScale)<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[(ptrdiff_t)std::llround(x*m_tScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT mid pointer with x as a direct index value
        inline Ty eval_mid_raw(ptrdiff_t x) const {
            lvDbgAssert(x>=-ptrdiff_t(nBins/2+nSafety) && x<=ptrdiff_t(nBins/2+nSafety));
            return m_pMid[x];
        }
        /// returns the evaluation result of lFunc(x) using the LUT low pointer after offsetting and scaling x (i.e. assuming tOffset!=0)
        inline Ty eval(Tx x) const {
            lvDbgAssert(ptrdiff_t((x-m_tLowOffset)*m_tScale)>=-ptrdiff_t(nSafety) && ptrdiff_t((x-m_tLowOffset)*m_tScale)<ptrdiff_t(nBins+nSafety));
            return m_pLow[ptrdiff_t((x-m_tLowOffset)*m_tScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT low pointer after offsetting, scaling, and rounding x (i.e. assuming tOffset!=0)
        inline Ty eval_round(Tx x) const {
            lvDbgAssert((ptrdiff_t)std::llround((x-m_tLowOffset)*m_tScale)>=-ptrdiff_t(nSafety) && (ptrdiff_t)std::llround((x-m_tLowOffset)*m_tScale)<ptrdiff_t(nBins+nSafety));
            return m_pLow[(ptrdiff_t)std::llround((x-m_tLowOffset)*m_tScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT low pointer after scaling x (i.e. assuming tOffset==0)
        inline Ty eval_noffset(Tx x) const {
            lvDbgAssert(ptrdiff_t(x*m_tScale)>=-ptrdiff_t(nSafety) && ptrdiff_t(x*m_tScale)<ptrdiff_t(nBins+nSafety));
            return m_pLow[ptrdiff_t(x*m_tScale)];
        }
        /// returns the evaluation result of lFunc(x) using the LUT low pointer after scaling and rounding x (i.e. assuming tOffset==0)
        inline Ty eval_noffset_round(Tx x) const {
            lvDbgAssert((ptrdiff_t)std::llround(x*m_tScale)>=-ptrdiff_t(nSafety) && (ptrdiff_t)std::llround(x*m_tScale)<ptrdiff_t(nBins+nSafety));
            return m_pLow[(ptrdiff_t)std::llround(x*m_tScale)];
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
        /// scale coefficient & step for lookup
        const TStep m_tScale,m_tStep;
        /// functor lookup table
        const std::vector<Ty> m_vLUT;
        /// base LUT pointers for lookup
        const Ty* m_pMid,*m_pLow;
    private:
        template<typename TFunc>
        static std::vector<Ty> init(Tx tMin, Tx tMax, TStep tStep, TFunc lFunc) {
            std::vector<Ty> vLUT(nBins+nSafety*2);
            for(size_t n=0; n<=nSafety; ++n)
                vLUT[n] = lFunc(tMin);
            for(size_t n=nSafety+1; n<nBins+nSafety-1; ++n)
                vLUT[n] = lFunc(tMin+Tx((n-nSafety)*tStep));
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
            lv::for_each(m_aMutexes,[](auto& oMutex) noexcept {oMutex.unlock();});
        }
        /// relocks all mutexes simultaneously
        ~unlock_guard() {
            lv::unpack_and_call<void(TMutexes&...)>(m_aMutexes,std::lock);
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
        /// returns the current internal resource count
        inline size_t count() {
            return m_nCount;
        }
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

    using mutex_lock_guard = std::lock_guard<std::mutex>;
    using mutex_unique_lock = std::unique_lock<std::mutex>;

} // namespace lv

template<size_t nWorkers>
lv::WorkerPool<nWorkers>::WorkerPool() : m_bIsActive(true) {
    lv::unroll<nWorkers>([this](size_t){m_vhWorkers.emplace_back(std::bind(&WorkerPool::entry,this));});
}

template<size_t nWorkers>
lv::WorkerPool<nWorkers>::~WorkerPool() {
    {
        std::unique_lock<std::mutex> oLock(m_oSyncMutex);
        m_bIsActive = false;
        m_oSyncVar.notify_all();
    }
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
        lv::mutex_lock_guard sync_lock(m_oSyncMutex);
        m_qTasks.emplace([pSharableTask](){(*pSharableTask)();}); // lambda keeps a copy of the task in the queue
    }
    m_oSyncVar.notify_one();
    return oTaskRes;
}

template<size_t nWorkers>
void lv::WorkerPool<nWorkers>::entry() {
    lv::mutex_unique_lock sync_lock(m_oSyncMutex);
    while(m_bIsActive || !m_qTasks.empty()) {
        m_oSyncVar.wait(sync_lock,[&](){return !m_bIsActive || !m_qTasks.empty();});
        if(!m_qTasks.empty()) {
            std::function<void()> task = std::move(m_qTasks.front());
            m_qTasks.pop();
            lv::unlock_guard<lv::mutex_unique_lock> oUnlock(sync_lock);
            task(); // if the execution throws, the exception will be contained in the shared state returned on queue
        }
    }
}