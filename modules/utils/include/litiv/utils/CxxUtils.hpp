
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

#ifdef _MSC_VER
#if _MSC_VER<1900 // requires at least MSVC 2015 toolchain for constexpr support
#error "This project requires C++11 support w/ constexpr -- MSVC 2015 minimum."
#endif //_MSC_VER<...
#elif __cplusplus<201103L
#error "This project requires C++11 support."
#endif //__cplusplus<=201103L

#include <cmath>
#include <mutex>
#include <array>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <future>
#include <iostream>
#include <functional>
#include <type_traits>
#include <condition_variable>
#include <opencv2/core.hpp>

#define D2R(d) ((d)*(M_PI/180.0))
#define R2D(r) ((r)*(180.0/M_PI))
#define isnan(f) std::isnan(f)

#define lvError(msg) throw CxxUtils::Exception(msg,__PRETTY_FUNCTION__,__FILE__,__LINE__)
#define lvErrorExt(msg,...) throw CxxUtils::Exception(msg,__PRETTY_FUNCTION__,__FILE__,__LINE__,__VA_ARGS__)
#define lvAssert(expr) {if(!!(expr)); else lvError("assertion failed ("#expr")");}
#ifdef _DEBUG
#define lvDbgAssert(expr) lvAssert(expr)
#define lvDbgExceptionWatch CxxUtils::UncaughtExceptionLogger __logger(__PRETTY_FUNCTION__,__FILE__,__LINE__)
#else //!defined(_DEBUG)
#define lvDbgAssert(expr)
#define lvDbgExceptionWatch
#endif //!defined(_DEBUG)

namespace CxxUtils {

    template<typename Derived,typename Base,typename Del>
    std::unique_ptr<Derived,Del> static_unique_ptr_cast(std::unique_ptr<Base,Del>&& p) {
        auto d = static_cast<Derived*>(p.release());
        return std::unique_ptr<Derived,Del>(d,std::move(p.get_deleter()));
    }

    template<typename Derived,typename Base,typename Del>
    std::unique_ptr<Derived,Del>dynamic_unique_ptr_cast(std::unique_ptr<Base,Del>&& p) {
        if(Derived* result = dynamic_cast<Derived*>(p.get())) {
            p.release();
            return std::unique_ptr<Derived,Del>(result,std::move(p.get_deleter()));
        }
        return std::unique_ptr<Derived,Del>(nullptr,p.get_deleter());
    }

    template<size_t n, typename F>
    inline typename std::enable_if<n==0>::type unroll(const F& f) {
        f(0);
    }

    template<size_t n, typename F>
    inline typename std::enable_if<(n>0)>::type unroll(const F& f) {
        unroll<n-1>(f);
        f(n);
    }

    template<typename T,std::size_t nByteAlign>
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
#else //!def(_MSC_VER)
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
#endif //!def(_MSC_VER)
        template<class T2, class ...Args> inline void construct(T2* p, Args&&... args) {::new(reinterpret_cast<void*>(p)) T2(std::forward<Args>(args)...);}
        inline void construct(pointer p, const value_type& wert) {new(p) value_type(wert);}
        inline size_type max_size() const noexcept {return (size_type(~0)-size_type(nByteAlign))/sizeof(value_type);}
        bool operator!=(const AlignAllocator<T,nByteAlign>& other) const {return !(*this==other);}
        bool operator==(const AlignAllocator<T,nByteAlign>& other) const {return true;}
    };

#ifndef _MSC_VER // meta-str-concat below does not compile properly w/ MSVC 2015 toolchain (last tested Jan. 2015)
    template<char... str> struct MetaStr {
        static constexpr char value[] = {str...};
    };
    template<char... str>
    constexpr char MetaStr<str...>::value[];

    template<typename, typename>
    struct MetaStrConcat;
    template<char... str1, char... str2>
    struct MetaStrConcat<MetaStr<str1...>, MetaStr<str2...>> {
        using type = MetaStr<str1..., str2...>;
    };

    template<typename...>
    struct MetaStrConcatenator;
    template<>
    struct MetaStrConcatenator<> {
        using type = MetaStr<>;
    };
    template<typename str, typename... vstr>
    struct MetaStrConcatenator<str, vstr...> {
        using type = typename MetaStrConcat<str, typename MetaStrConcatenator<vstr...>::type>::type;
    };

    template<size_t N>
    struct MetaITOA {
        using type = typename MetaStrConcat<typename std::conditional<(N>=10),typename MetaITOA<(N/10)>::type,MetaStr<>>::type,MetaStr<'0'+(N%10)>>::type;
    };
    template<>
    struct MetaITOA<0> {
        using type = MetaStr<'0'>;
    };
#endif //!def(_MSC_VER)

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
        template<typename... VALIST>
        Exception(const std::string& sErrMsg, const char* sFunc, const char* sFile, int nLine, VALIST... vArgs) :
                std::runtime_error(cv::format((std::string("Exception in function '%s' from %s(%d) : \n")+sErrMsg).c_str(),sFunc,sFile,nLine,vArgs...)),
                m_acFuncName(sFunc),
                m_acFileName(sFile),
                m_nLineNumber(nLine) {}
        const char* const m_acFuncName;
        const char* const m_acFileName;
        const int m_nLineNumber;
    };

} //namespace CxxUtils

namespace std {
    template<typename T, size_t N>
    using aligned_vector = vector<T,CxxUtils::AlignAllocator<T,N>>;
} //namespace std
