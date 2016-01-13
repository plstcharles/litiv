
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
#include <iostream>
#include <functional>
#include <type_traits>
#include <condition_variable>
#include <opencv2/core.hpp>
#include "litiv/utils/DefineUtils.hpp"

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

    //! returns pixel coordinates clamped to the given image & border size
    static inline void clampImageCoords(int& nSampleCoord_X,int& nSampleCoord_Y,const int nBorderSize,const cv::Size& oImageSize) {
        if(nSampleCoord_X<nBorderSize)
            nSampleCoord_X = nBorderSize;
        else if(nSampleCoord_X>=oImageSize.width-nBorderSize)
            nSampleCoord_X = oImageSize.width-nBorderSize-1;
        if(nSampleCoord_Y<nBorderSize)
            nSampleCoord_Y = nBorderSize;
        else if(nSampleCoord_Y>=oImageSize.height-nBorderSize)
            nSampleCoord_Y = oImageSize.height-nBorderSize-1;
    }

    //! returns a random init/sampling position for the specified pixel position, given a predefined kernel; also guards against out-of-bounds values via image/border size check.
    template<int nKernelHeight,int nKernelWidth>
    static inline void getRandSamplePosition(const std::array<std::array<int,nKernelWidth>,nKernelHeight>& anSamplesInitPattern,
                                             const int nSamplesInitPatternTot,int& nSampleCoord_X,int& nSampleCoord_Y,
                                             const int nOrigCoord_X,const int nOrigCoord_Y,const int nBorderSize,const cv::Size& oImageSize) {
        int r = 1+rand()%nSamplesInitPatternTot;
        for(nSampleCoord_X=0; nSampleCoord_X<nKernelWidth; ++nSampleCoord_X) {
            for(nSampleCoord_Y=0; nSampleCoord_Y<nKernelHeight; ++nSampleCoord_Y) {
                r -= anSamplesInitPattern[nSampleCoord_Y][nSampleCoord_X];
                if(r<=0)
                    goto stop;
            }
        }
        stop:
        nSampleCoord_X += nOrigCoord_X-nKernelWidth/2;
        nSampleCoord_Y += nOrigCoord_Y-nKernelHeight/2;
        clampImageCoords(nSampleCoord_X,nSampleCoord_Y,nBorderSize,oImageSize);
    }

    //! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    static inline void getRandSamplePosition_3x3_std1(int& nSampleCoord_X,int& nSampleCoord_Y,const int nOrigCoord_X,const int nOrigCoord_Y,const int nBorderSize,const cv::Size& oImageSize) {
        // based on 'floor(fspecial('gaussian',3,1)*256)'
        static_assert(sizeof(std::array<int,3>)==sizeof(int)*3,"bad std::array stl impl");
        static const int s_nSamplesInitPatternTot = 256;
        static const std::array<std::array<int,3>,3> s_anSamplesInitPattern ={
            std::array<int,3>{19,32,19,},
            std::array<int,3>{32,52,32,},
            std::array<int,3>{19,32,19,},
        };
        getRandSamplePosition<3,3>(s_anSamplesInitPattern,s_nSamplesInitPatternTot,nSampleCoord_X,nSampleCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    //! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    static inline void getRandSamplePosition_7x7_std2(int& nSampleCoord_X,int& nSampleCoord_Y,const int nOrigCoord_X,const int nOrigCoord_Y,const int nBorderSize,const cv::Size& oImageSize) {
        // based on 'floor(fspecial('gaussian',7,2)*512)'
        static_assert(sizeof(std::array<int,7>)==sizeof(int)*7,"bad std::array stl impl");
        static const int s_nSamplesInitPatternTot = 512;
        static const std::array<std::array<int,7>,7> s_anSamplesInitPattern ={
            std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
            std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
            std::array<int,7>{ 6,12,21,25,21,12, 6,},
            std::array<int,7>{ 7,14,25,28,25,14, 7,},
            std::array<int,7>{ 6,12,21,25,21,12, 6,},
            std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
            std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
        };
        getRandSamplePosition<7,7>(s_anSamplesInitPattern,s_nSamplesInitPatternTot,nSampleCoord_X,nSampleCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    //! returns a random neighbor position for the specified pixel position, given a predefined neighborhood; also guards against out-of-bounds values via image/border size check.
    template<int nNeighborCount>
    static inline void getRandNeighborPosition(const std::array<std::array<int,2>,nNeighborCount>& anNeighborPattern,
                                               int& nNeighborCoord_X,int& nNeighborCoord_Y,
                                               const int nOrigCoord_X,const int nOrigCoord_Y,
                                               const int nBorderSize,const cv::Size& oImageSize) {
        int r = rand()%nNeighborCount;
        nNeighborCoord_X = nOrigCoord_X+anNeighborPattern[r][0];
        nNeighborCoord_Y = nOrigCoord_Y+anNeighborPattern[r][1];
        clampImageCoords(nNeighborCoord_X,nNeighborCoord_Y,nBorderSize,oImageSize);
    }

    //! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    static inline void getRandNeighborPosition_3x3(int& nNeighborCoord_X,int& nNeighborCoord_Y,const int nOrigCoord_X,const int nOrigCoord_Y,const int nBorderSize,const cv::Size& oImageSize) {
        typedef std::array<int,2> Nb;
        static const std::array<std::array<int,2>,8> s_anNeighborPattern ={
            Nb{-1, 1},Nb{0, 1},Nb{1, 1},
            Nb{-1, 0},         Nb{1, 0},
            Nb{-1,-1},Nb{0,-1},Nb{1,-1},
        };
        getRandNeighborPosition<8>(s_anNeighborPattern,nNeighborCoord_X,nNeighborCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    //! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    static inline void getRandNeighborPosition_5x5(int& nNeighborCoord_X,int& nNeighborCoord_Y,const int nOrigCoord_X,const int nOrigCoord_Y,const int nBorderSize,const cv::Size& oImageSize) {
        typedef std::array<int,2> Nb;
        static const std::array<std::array<int,2>,24> s_anNeighborPattern ={
            Nb{-2, 2},Nb{-1, 2},Nb{0, 2},Nb{1, 2},Nb{2, 2},
            Nb{-2, 1},Nb{-1, 1},Nb{0, 1},Nb{1, 1},Nb{2, 1},
            Nb{-2, 0},Nb{-1, 0},         Nb{1, 0},Nb{2, 0},
            Nb{-2,-1},Nb{-1,-1},Nb{0,-1},Nb{1,-1},Nb{2,-1},
            Nb{-2,-2},Nb{-1,-2},Nb{0,-2},Nb{1,-2},Nb{2,-2},
        };
        getRandNeighborPosition<24>(s_anNeighborPattern,nNeighborCoord_X,nNeighborCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

} //namespace CxxUtils

namespace std {
    template<typename T, size_t N>
    using aligned_vector = vector<T,CxxUtils::AlignAllocator<T,N>>;
} //namespace std
