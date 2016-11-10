
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
#include <opencv2/core.hpp>
#include "litiv/utils/defines.hpp"

// @@@ cleanup, move impls down as inline?
// @@@ replace some vector args by templated begin/end iterators? (stl-like)

namespace lv {

    template<int> struct _; // used for compile-time integer expr printing via error; just write "_<expr> __;"

    struct UncaughtExceptionLogger {
        UncaughtExceptionLogger(const char* sFunc, const char* sFile, int nLine) :
                m_sFunc(sFunc),m_sFile(sFile),m_nLine(nLine) {}
        const char* const m_sFunc;
        const char* const m_sFile;
        const int m_nLine;
        ~UncaughtExceptionLogger() {
            if(std::uncaught_exception())
                std::cerr << cv::format("Unwinding at function '%s' from %s(%d) due to uncaught exception\n",m_sFunc,m_sFile,m_nLine);
        }
    };

    struct Exception : public std::runtime_error {
        template<typename... Targs>
        Exception(const std::string& sErrMsg, const char* sFunc, const char* sFile, int nLine, Targs&&... args) :
                std::runtime_error(cv::format((std::string("Exception in function '%s' from %s(%d) : \n")+sErrMsg).c_str(),sFunc,sFile,nLine,std::forward<Targs>(args)...)),
                m_acFuncName(sFunc),
                m_acFileName(sFile),
                m_nLineNumber(nLine) {
            std::cerr << this->what() << std::endl;
        }
        const char* const m_acFuncName;
        const char* const m_acFileName;
        const int m_nLineNumber;
    };

    template<typename TDerived, typename TBase, typename TDeleter>
    inline std::unique_ptr<TDerived,TDeleter> static_unique_ptr_cast(std::unique_ptr<TBase,TDeleter>&& p) {
        auto d = static_cast<TDerived*>(p.release());
        return std::unique_ptr<TDerived,TDeleter>(d,std::move(p.get_deleter()));
    }

    template<typename TDerived, typename TBase, typename TDeleter>
    inline std::unique_ptr<TDerived,TDeleter> dynamic_unique_ptr_cast(std::unique_ptr<TBase,TDeleter>&& p) {
        // note: returned ptr deleter will be the original ptr's
        if(TDerived* result = dynamic_cast<TDerived*>(p.get())) {
            p.release();
            return std::unique_ptr<TDerived,TDeleter>(result,std::move(p.get_deleter()));
        }
        return std::unique_ptr<TDerived,TDeleter>(nullptr,p.get_deleter());
    }

    template<size_t n, typename F>
    constexpr inline std::enable_if_t<n==0> unroll(const F&) {}

    template<size_t n, typename F>
    constexpr inline std::enable_if_t<(n>0)> unroll(const F& f) {
        unroll<n-1>(f);
        f(n-1);
    }

    inline bool compare_lowercase(const std::string& i, const std::string& j) {
        std::string i_lower(i), j_lower(j);
        std::transform(i_lower.begin(),i_lower.end(),i_lower.begin(),tolower);
        std::transform(j_lower.begin(),j_lower.end(),j_lower.begin(),tolower);
        return i_lower<j_lower;
    }

    template<typename T>
    inline int digit_count(T number) {
        // counts sign as extra digit if negative
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
    inline std::vector<size_t> indices_of(const std::vector<T>& voVals, const std::vector<T>& voRefs) {
        std::vector<size_t> vnIndices(voVals.size());
        size_t nIdx = 0;
        std::for_each(voVals.begin(),voVals.end(),[&](const T& oVal){
            vnIndices[nIdx++] = std::distance(voRefs.begin(),std::find(voRefs.begin(),voRefs.end(),oVal));
        });
        return vnIndices;
    }

    template<typename T>
    inline std::vector<size_t> sort_indices(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndices(voVals.size());
        std::iota(vnIndices.begin(),vnIndices.end(),0);
        std::sort(vnIndices.begin(),vnIndices.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]<voVals[n2];
        });
        return vnIndices;
    }

    template<typename T, typename P>
    inline std::vector<size_t> sort_indices(const std::vector<T>& voVals, P oSortFunctor) {
        std::vector<size_t> vnIndices(voVals.size());
        std::iota(vnIndices.begin(),vnIndices.end(),0);
        std::sort(vnIndices.begin(),vnIndices.end(),oSortFunctor);
        return vnIndices;
    }

    template<typename T>
    inline std::vector<size_t> unique_indices(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndices = sort_indices(voVals);
        auto pLastIdxIter = std::unique(vnIndices.begin(),vnIndices.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]==voVals[n2];
        });
        return std::vector<size_t>(vnIndices.begin(),pLastIdxIter);
    }

    template<typename T, typename P1, typename P2>
    inline std::vector<size_t> unique_indices(const std::vector<T>& voVals, P1 oSortFunctor, P2 oCompareFunctor) {
        std::vector<size_t> vnIndices = sort_indices(voVals,oSortFunctor);
        auto pLastIdxIter = std::unique(vnIndices.begin(),vnIndices.end(),oCompareFunctor);
        return std::vector<size_t>(vnIndices.begin(),pLastIdxIter);
    }

    template<typename Titer>
    inline std::vector<typename std::iterator_traits<Titer>::value_type> unique(Titer begin, Titer end) {
        const std::set<typename std::iterator_traits<Titer>::value_type> mMap(begin,end);
        return std::vector<typename std::iterator_traits<Titer>::value_type>(mMap.begin(),mMap.end());
    }

    template<typename T>
    inline std::vector<T> unique(const cv::Mat_<T>& oMat) {
        const std::set<T> mMap(oMat.begin(),oMat.end());
        return std::vector<T>(mMap.begin(),mMap.end());
    }

    template<typename Tval, typename Tcomp>
    size_t find_nn_index(Tval oReqVal, const std::vector<Tval>& voRefVals, const Tcomp& lCompFunc) {
        decltype(lCompFunc(oReqVal,oReqVal)) oMinDist = std::numeric_limits<decltype(lCompFunc(oReqVal,oReqVal))>::max();
        size_t nIdx = size_t(-1);
        for(size_t n=0; n<voRefVals.size(); ++n) {
            auto oCurrDist = lCompFunc(oReqVal,voRefVals[n]);
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
        lvAssert_(vX.size()==vY.size(),"size of input X and Y vectors must be identical");
        lvAssert_(vX.size()>1,"input vectors must contain at least one element");
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
                size_t nNNIdx = lv::find_nn_index(vXReq[i],vX,[](const Tx& a,const Tx& b){return std::abs(b-a);});
                vYReq.push_back(vSlope[nNNIdx]*vXReq[i]+vIntercept[nNNIdx]);
            }
        }
        return vYReq;
    }

    template<typename T>
    inline std::enable_if_t<std::is_integral<T>::value,std::vector<T>> linspace(T a, T b, size_t steps, bool bIncludeInitVal=true) {
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
    inline std::enable_if_t<std::is_floating_point<T>::value,std::vector<T>> linspace(T a, T b, size_t steps, bool bIncludeInitVal=true) {
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

    template<size_t nWorkers>
    struct WorkerPool {
        static_assert(nWorkers>0,"Worker pool must have at least one work thread");
        WorkerPool();
        ~WorkerPool();
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

    template<size_t nWordBitSize, typename Tr>
    constexpr inline std::enable_if_t<nWordBitSize==1,Tr> expand_bits(const Tr& nBits, int=0) {
        return nBits;
    }

    template<size_t nWordBitSize, typename Tr>
    constexpr inline std::enable_if_t<(nWordBitSize>1),Tr> expand_bits(const Tr& nBits, int n=((sizeof(Tr)*8)/nWordBitSize)-1) {
        static_assert(std::is_integral<Tr>::value,"nBits type must be integral");
        // only the first [(sizeof(Tr)*8)/nWordBitSize] bits are kept (otherwise overflow/ignored)
        return (Tr)(bool(nBits&(1<<n))<<(n*nWordBitSize)) + ((n>=1)?expand_bits<nWordBitSize,Tr>(nBits,n-1):(Tr)0);
    }

    template<int... anIndices>
    struct MetaIdxConcat {};

    template<int N, int... anIndices>
    struct MetaIdxConcatenator : MetaIdxConcatenator<N - 1, N - 1, anIndices...> {};

    template<int... anIndices>
    struct MetaIdxConcatenator<0, anIndices...> : MetaIdxConcat<anIndices...> {};

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

    inline std::string getTimeStamp() {
        std::time_t tNow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char acBuffer[128];
        std::strftime(acBuffer,sizeof(acBuffer),"%F %T",std::localtime(&tNow)); // std::put_time missing w/ GCC<5.0
        return std::string(acBuffer);
    }

    inline std::string getVersionStamp() {
        return "LITIV Framework v" LITIV_VERSION_STR " (SHA1=" LITIV_VERSION_SHA1 ")";
    }

    inline std::string getLogStamp() {
        return std::string("\n")+lv::getVersionStamp()+"\n["+lv::getTimeStamp()+"]\n";
    }

    inline std::string clampString(const std::string& sInput, size_t nSize, char cPadding=' ') {
        return sInput.size()>nSize?sInput.substr(0,nSize):std::string(nSize-sInput.size(),cPadding)+sInput;
    }

    template<typename TSum, typename TObj>
    inline TSum accumulateMembers(const std::vector<TObj>& vObjArray, const std::function<TSum(const TObj&)>& lFunc, TSum tInit=TSum(0)) {
        return std::accumulate(vObjArray.begin(),vObjArray.end(),tInit,[&](TSum tSum, const TObj& p) {
            return tSum + lFunc(p);
        });
    }

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

    template<typename T>
    struct enable_shared_from_this : public std::enable_shared_from_this<T> {
        template<typename Tcast>
        inline std::shared_ptr<const Tcast> shared_from_this_cast(bool bThrowIfFail=false) const {
            auto pCast = std::dynamic_pointer_cast<const Tcast>(this->shared_from_this());
            if(bThrowIfFail && !pCast)
                lvError_("Failed shared_from_this_cast from type '%s' to type '%s'",typeid(T).name(),typeid(Tcast).name());
            return pCast;
        }
        template<typename Tcast>
        inline std::shared_ptr<Tcast> shared_from_this_cast(bool bThrowIfFail=false) {
            return std::const_pointer_cast<Tcast>(static_cast<const T*>(this)->template shared_from_this_cast<Tcast>(bThrowIfFail));
        }
    };

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

    template<typename TTuple, typename TFunc, int... anIndices>
    inline void for_each(TTuple&& t, TFunc f, MetaIdxConcat<anIndices...>) {
        auto l = { (f(std::get<anIndices>(t)),0)... };
    }

    template<typename... TTupleTypes, typename TFunc>
    inline void for_each_in_tuple(const std::tuple<TTupleTypes...>& t, TFunc f) {
        for_each(t, f, MetaIdxConcatenator<sizeof...(TTupleTypes)>());
    }

    template<typename... TMutexes>
    struct unlock_guard {
        explicit unlock_guard(TMutexes&... aMutexes) noexcept :
                m_aMutexes(aMutexes...) {
            for_each_in_tuple(m_aMutexes,[](auto& oMutex) noexcept {oMutex.unlock();});
        }
        ~unlock_guard() {
            std::lock(m_aMutexes);
        }
        unlock_guard(const unlock_guard&) = delete;
        unlock_guard& operator=(const unlock_guard&) = delete;
    private:
        std::tuple<TMutexes&...> m_aMutexes;
    };

    template<typename TMutex>
    struct unlock_guard<TMutex> {
        explicit unlock_guard(TMutex& oMutex) noexcept :
                m_oMutex(oMutex) {
            m_oMutex.unlock();
        }
        ~unlock_guard() {
            m_oMutex.lock();
        }
        unlock_guard(const unlock_guard&) = delete;
        unlock_guard& operator=(const unlock_guard&) = delete;
    private:
        TMutex& m_oMutex;
    };

    struct Semaphore {
        using native_handle_type = std::condition_variable::native_handle_type;
        inline explicit Semaphore(size_t nInitCount) :
                m_nCount(nInitCount) {}
        inline void notify() {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            ++m_nCount;
            m_oCondVar.notify_one();
        }
        inline void wait() {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            m_oCondVar.wait(oLock,[&]{return m_nCount>0;});
            --m_nCount;
        }
        inline bool try_wait() {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            if(m_nCount>0) {
                --m_nCount;
                return true;
            }
            return false;
        }
        template<typename TRep, typename TPeriod=std::ratio<1>>
        inline bool wait_for(const std::chrono::duration<TRep,TPeriod>& nDuration) {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            bool bFinished = m_oCondVar.wait_for(oLock,nDuration,[&]{return m_nCount>0;});
            if(bFinished)
                --m_nCount;
            return bFinished;
        }
        template<typename TClock, typename TDuration=typename TClock::duration>
        inline bool wait_until(const std::chrono::time_point<TClock,TDuration>& nTimePoint) {
            std::unique_lock<std::mutex> oLock(m_oMutex);
            bool bFinished = m_oCondVar.wait_until(oLock,nTimePoint,[&]{return m_nCount>0;});
            if(bFinished)
                --m_nCount;
            return bFinished;
        }
        inline native_handle_type native_handle() {
            return m_oCondVar.native_handle();
        }
        Semaphore(const Semaphore&) = delete;
        Semaphore& operator=(const Semaphore&) = delete;
    private:
        std::condition_variable m_oCondVar;
        std::mutex m_oMutex;
        size_t m_nCount;
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
