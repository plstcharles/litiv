
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

#include <cmath>
#include <mutex>
#include <array>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <future>
#include <memory>
#include <numeric>
#include <iomanip>
#include <iostream>
#include <functional>
#include <type_traits>
#include <condition_variable>
#include <opencv2/core.hpp>
#include "litiv/utils/DefineUtils.hpp"

namespace CxxUtils {

    template<int> struct _; // used for compile-time integer expr printing via error; just write "_<expr> __;"

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

    template<typename T, std::size_t nByteAlign>
    class AlignAllocator {
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef T& reference;
        typedef const T* const_pointer;
        typedef const T& const_reference;
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;
        typedef std::true_type propagate_on_container_move_assignment;
        template<typename T2> struct rebind {typedef AlignAllocator<T2,nByteAlign> other;};
    public:
        inline AlignAllocator() noexcept {}
        template<typename T2> inline AlignAllocator(const AlignAllocator<T2,nByteAlign>&) noexcept {}
        inline ~AlignAllocator() throw() {}
        inline pointer address(reference r) {return std::addressof(r);}
        inline const_pointer address(const_reference r) const noexcept {return std::addressof(r);}
#ifdef _MSC_VER
        inline pointer allocate(size_type n) {
            const size_type alignment = static_cast<size_type>(nByteAlign);
            size_t alloc_size = n*sizeof(value_type);
            if((alloc_size%alignment)!=0) {
                alloc_size += alignment - alloc_size%alignment;
                CV_DbgAssert((alloc_size%alignment)==0);
            }
            void* ptr = _aligned_malloc(alloc_size,nByteAlign);
            if(ptr==nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }
        inline void deallocate(pointer p, size_type) noexcept {_aligned_free(p);}
        inline void destroy(pointer p) {p->~value_type();p;}
#else //(!def(_MSC_VER))
        inline pointer allocate(size_type n) {
            const size_type alignment = static_cast<size_type>(nByteAlign);
            size_t alloc_size = n*sizeof(value_type);
            if((alloc_size%alignment)!=0) {
                alloc_size += alignment - alloc_size%alignment;
                CV_DbgAssert((alloc_size%alignment)==0);
            }
            void* ptr = aligned_alloc(alignment,alloc_size);
            if(ptr==nullptr)
                throw std::bad_alloc();
            return reinterpret_cast<pointer>(ptr);
        }
        inline void deallocate(pointer p, size_type) noexcept {free(p);}
        inline void destroy(pointer p) {p->~value_type();}
#endif //(!def(_MSC_VER))
        template<typename T2, typename... Targs> inline void construct(T2* p, Targs&&... args) {::new(reinterpret_cast<void*>(p)) T2(std::forward<Targs>(args)...);}
        inline void construct(pointer p, const value_type& wert) {new(p) value_type(wert);}
        inline size_type max_size() const noexcept {return (size_type(~0)-size_type(nByteAlign))/sizeof(value_type);}
        bool operator!=(const AlignAllocator<T,nByteAlign>& other) const {return !(*this==other);}
        bool operator==(const AlignAllocator<T,nByteAlign>& other) const {return true;}
    };

    template<int... anIndices>
    struct MetaIdxConcat {};

    template<int N, int... anIndices>
    struct MetaIdxConcatenator : MetaIdxConcatenator<N - 1, N - 1, anIndices...> {};

    template<int... anIndices>
    struct MetaIdxConcatenator<0, anIndices...> : MetaIdxConcat<anIndices...> {};

    struct UncaughtExceptionLogger {
        UncaughtExceptionLogger(const char* sFunc, const char* sFile, int nLine) :
                m_sMsg(cv::format("Unwinding at function '%s' from %s(%d) due to uncaught exception\n",sFunc,sFile,nLine)) {}
        const std::string m_sMsg;
        ~UncaughtExceptionLogger() {
            if(std::uncaught_exception())
                std::cerr << m_sMsg;
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

    template<typename T>
    inline bool isnan(T dVal) {
#ifdef _MSC_VER // needed for portability...
        return _isnan((double)dVal)!=0;
#else //(!def(_MSC_VER))
        return std::isnan(dVal);
#endif //(!def(_MSC_VER))
    }

    struct StopWatch {
        StopWatch() {tick();}
        inline void tick() {m_nTick = std::chrono::high_resolution_clock::now();}
        inline double tock(bool bReset=true) {
            const std::chrono::high_resolution_clock::time_point nNow = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> dElapsed_sec = nNow-m_nTick;
            if(bReset)
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
        return std::string("\n")+CxxUtils::getVersionStamp()+"\n["+CxxUtils::getTimeStamp()+"]\n";
    }

    inline std::string clampString(const std::string& sInput, size_t nSize, char cPadding=' ') {
        return sInput.size()>nSize?sInput.substr(0,nSize):std::string(nSize-sInput.size(),cPadding)+sInput;
    }

    template<typename TSum, typename TObj>
    inline TSum accumulateMembers(const std::vector<TObj>& vObjArray, const std::function<TSum(const TObj&)>& lFunc) {
        return std::accumulate(vObjArray.begin(),vObjArray.end(),TSum(0),[&](TSum tSum, const TObj& p) {
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

    template<typename T>
    inline std::vector<T> filter_out(const std::vector<T>& vVals, const std::vector<T>& vTokens) {
        std::vector<T> vRet = vVals; // will return all values not found in token list
        vRet.erase(std::remove_if(vRet.begin(),vRet.end(),[&](const T& o){return std::find(vTokens.begin(),vTokens.end(),o)!=vTokens.end();}),vRet.end());
        return vRet;
    }

    template<typename T>
    inline std::vector<T> filter_in(const std::vector<T>& vVals, const std::vector<T>& vTokens) {
        std::vector<T> vRet = vVals; // will returns all values found in token list
        vRet.erase(std::remove_if(vRet.begin(),vRet.end(),[&](const T& o){return std::find(vTokens.begin(),vTokens.end(),o)==vTokens.end();}),vRet.end());
        return vRet;
    }

    template<typename T>
    struct enable_shared_from_this : public std::enable_shared_from_this<T> {
        template<typename Tcast>
        inline std::shared_ptr<const Tcast> shared_from_this_cast(bool bThrowIfFail=false) const {
            auto pCast = std::dynamic_pointer_cast<const Tcast>(this->shared_from_this());
            if(bThrowIfFail && !pCast)
                lvErrorExt("Failed shared_from_this_cast from type '%s' to type '%s'",typeid(T).name(),typeid(Tcast).name());
            return pCast;
        }
        template<typename Tcast>
        inline std::shared_ptr<Tcast> shared_from_this_cast(bool bThrowIfFail=false) {
            return std::const_pointer_cast<Tcast>(static_cast<const T*>(this)->shared_from_this_cast<Tcast>(bThrowIfFail));
        }
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

} //namespace CxxUtils

namespace std { // extending std

    template<typename T, size_t N>
    using aligned_vector = vector<T,CxxUtils::AlignAllocator<T,N>>;
    template<typename T>
    using unlock_guard = CxxUtils::unlock_guard<T>;
    using semaphore = CxxUtils::Semaphore;
    using mutex_lock_guard = lock_guard<mutex>;
    using mutex_unique_lock = unique_lock<mutex>;

#if !defined(_MSC_VER) && __cplusplus<=201103L // make_unique is missing from C++11 (at least on GCC)
    template<typename T, typename... Targs>
    inline std::enable_if_t<!std::is_array<T>::value,std::unique_ptr<T>> make_unique(Targs&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Targs>(args)...));
    }
    template<typename T>
    inline std::enable_if_t<(std::is_array<T>::value && !std::extent<T>::value),std::unique_ptr<T>> make_unique(size_t nSize) {
        using ElemType = std::remove_extent_t<T>;
        return std::unique_ptr<T>(new ElemType[nSize]());
    }
    template<typename T, typename... Targs>
    std::enable_if_t<(std::extent<T>::value!=0)> make_unique(Targs&&...) = delete;
#endif //(!defined(_MSC_VER) && __cplusplus<=201103L)

} //namespace std
