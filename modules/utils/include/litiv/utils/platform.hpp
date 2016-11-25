
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
#if defined(_MSC_VER)
#include <windows.h>
#include <winerror.h>
#include <comdef.h>
#include <stdint.h>
#include <direct.h>
#include <psapi.h>
template<class T>
void SafeRelease(T** ppT) {if(*ppT) {(*ppT)->Release(); *ppT = nullptr;}}
#if !USE_KINECTSDK_STANDALONE
#include <Kinect.h>
#endif //(!USE_KINECTSDK_STANDALONE)
#else //(!defined(_MSC_VER))
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <stdio.h>
#endif //(!defined(_MSC_VER))
#include "litiv/utils/cxx.hpp"

namespace lv {

    /// returns the executable's current working directory path; relies on getcwd, and may return an empty string
    std::string getCurrentWorkDirPath();
    /// adds a forward slash to the given directory path if it ends without one, with handling for special cases (useful for path concatenation)
    std::string addDirSlashIfMissing(const std::string& sDirPath);
    /// returns a sorted list of all files located at a given directory path
    std::vector<std::string> getFilesFromDir(const std::string& sDirPath);
    /// returns a sorted list of all subdirectories located at a given directory path
    std::vector<std::string> getSubDirsFromDir(const std::string& sDirPath);
    /// filters a list of paths using string tokens; if a token is found in a path, it is removed/kept from the list
    void filterFilePaths(std::vector<std::string>& vsFilePaths, const std::vector<std::string>& vsRemoveTokens, const std::vector<std::string>& vsKeepTokens);
    /// creates a local directory at the given path if one does not already exist (does not work recursively)
    bool createDirIfNotExist(const std::string& sDirPath);
    /// creates a binary file at the specified location, and fills it with unspecified/zero data bytes (useful for critical/real-time stream writing without continuous reallocation)
    std::fstream createBinFileWithPrealloc(const std::string& sFilePath, size_t nPreallocBytes, bool bZeroInit=false);
    /// registers the SIGINT, SIGTERM, and SIGBREAK (if available) console signals to the given handler
    void registerAllConsoleSignals(void(*lHandler)(int));
    /// returns the amount of physical memory currently used on the system
    size_t getCurrentPhysMemBytesUsed();

    /// defines an stl-friendly aligned memory allocator to be used in container classes
    template<typename T, std::size_t nByteAlign, bool bDefaultInit=false>
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

    /// returns whether a value is NaN (required due to non-portable msvc signature)
    template<typename T>
    inline bool isnan(T dVal) {
#ifdef _MSC_VER // needed for portability...
        return _isnan((double)dVal)!=0;
#else //(!def(_MSC_VER))
        return std::isnan(dVal);
#endif //(!def(_MSC_VER))
    }

#if USE_KINECTSDK_STANDALONE
#ifndef BODY_COUNT
#define BODY_COUNT 6
#endif //ndef(BODY_COUNT)
    using TIMESPAN = int64_t;
#if !defined(_MSC_VER)
    using BOOLEAN = uchar;
    using DWORD = unsigned long;
    using UINT64 = unsigned long long;
#endif //(!defined(_MSC_VER))
#ifndef _Vector4_
#define _Vector4_
    using Vector4 = struct _Vector4 {
        float x,y,z,w;
    };
#endif //ndef(_Vector4_)
#ifndef _PointF_
#define _PointF_
    using PointF = struct _PointF {
        float X,Y;
    };
#endif //ndef(_PointF_)
#ifndef _FrameEdges_
#define _FrameEdges_
    using FrameEdges = enum _FrameEdges {
        FrameEdge_None   = 0x0,
        FrameEdge_Right  = 0x1,
        FrameEdge_Left   = 0x2,
        FrameEdge_Top    = 0x4,
        FrameEdge_Bottom = 0x8,
    };
#endif //ndef(_FrameEdges_)
#ifndef _TrackingState_
#define _TrackingState_
    using TrackingState = enum _TrackingState {
        TrackingState_NotTracked = 0,
        TrackingState_Inferred   = 1,
        TrackingState_Tracked    = 2,
    };
#endif //ndef(_TrackingState_)
#ifndef _HandState_
#define _HandState_
    using HandState = enum _HandState {
        HandState_Unknown    = 0,
        HandState_NotTracked = 1,
        HandState_Open       = 2,
        HandState_Closed     = 3,
        HandState_Lasso      = 4,
    };
#endif //ndef(_HandState_)
#ifndef _TrackingConfidence_
#define _TrackingConfidence_
    using TrackingConfidence = enum _TrackingConfidence {
        TrackingConfidence_Low  = 0,
        TrackingConfidence_High = 1,
    };
#endif //ndef(_TrackingConfidence_)
#ifndef _CameraSpacePoint_
#define _CameraSpacePoint_
    using CameraSpacePoint = struct _CameraSpacePoint {
        float X,Y,Z;
    };
#endif //ndef(_CameraSpacePoint_)
#ifndef _JointType_
#define _JointType_
    using JointType = enum _JointType {
        JointType_SpineBase     = 0,
        JointType_SpineMid      = 1,
        JointType_Neck          = 2,
        JointType_Head          = 3,
        JointType_ShoulderLeft  = 4,
        JointType_ElbowLeft     = 5,
        JointType_WristLeft     = 6,
        JointType_HandLeft      = 7,
        JointType_ShoulderRight = 8,
        JointType_ElbowRight    = 9,
        JointType_WristRight    = 10,
        JointType_HandRight     = 11,
        JointType_HipLeft       = 12,
        JointType_KneeLeft      = 13,
        JointType_AnkleLeft     = 14,
        JointType_FootLeft      = 15,
        JointType_HipRight      = 16,
        JointType_KneeRight     = 17,
        JointType_AnkleRight    = 18,
        JointType_FootRight     = 19,
        JointType_SpineShoulder = 20,
        JointType_HandTipLeft   = 21,
        JointType_ThumbLeft     = 22,
        JointType_HandTipRight  = 23,
        JointType_ThumbRight    = 24,
        JointType_Count         = 25,
    };
#endif //ndef(_JointType_)
#ifndef _Joint_
#define _Joint_
    using Joint = struct _Joint {
        _JointType JointType;
        _CameraSpacePoint Position;
        _TrackingState TrackingState;
    };
#endif //ndef(_Joint_)
#ifndef _JointOrientation_
#define _JointOrientation_
    using JointOrientation = struct _JointOrientation {
        _JointType JointType;
        _Vector4 Orientation;
    };
#endif //ndef(_JointOrientation_)
#endif //USE_KINECTSDK_STANDALONE

    /// portable structure containing kinect body data for one frame
    struct KinectBodyFrame {
        bool bIsValid;
        TIMESPAN nTimeStamp;
        Vector4 vFloorClipPlane;
        size_t nFrameIdx;
        struct BodyData {
            BOOLEAN bIsTracked;
            BOOLEAN bIsRestricted;
            DWORD nClippedEdges;
            UINT64 nTrackingID;
            PointF vLean;
            TrackingState eLeanTrackState;
            HandState eLeftHandState;
            HandState eRightHandState;
            TrackingConfidence eLeftHandStateConfidence;
            TrackingConfidence eRightHandStateConfidence;
            std::array<Joint,JointType::JointType_Count> aJointData;
            std::array<JointOrientation,JointType::JointType_Count> aJointOrientationData;
        } aBodyData[BODY_COUNT];
    };

} // namespace lv

namespace std { // extending std

    /// helper alias; std-friendly version of vector with N-byte aligned memory allocator
    template<typename T, size_t N>
    using aligned_vector = vector<T,lv::AlignedMemAllocator<T,N>>;
    /// helper alias; std-friendly version of vector with 16-byte aligned memory allocator
    template<typename T>
    using vec16a = vector<T,lv::AlignedMemAllocator<T,16>>;
    /// helper alias; std-friendly version of vector with 32-byte aligned memory allocator
    template<typename T>
    using vec32a = vector<T,lv::AlignedMemAllocator<T,32>>;

#if !defined(_MSC_VER) && __cplusplus<=201103L // make_unique is missing from C++11 (at least on GCC)
    template<typename T, typename... TArgs>
    inline std::enable_if_t<!std::is_array<T>::value,std::unique_ptr<T>> make_unique(TArgs&&... args) {
        return std::unique_ptr<T>(new T(std::forward<TArgs>(args)...));
    }
    template<typename T>
    inline std::enable_if_t<(std::is_array<T>::value && !std::extent<T>::value),std::unique_ptr<T>> make_unique(size_t nSize) {
        using ElemType = std::remove_extent_t<T>;
        return std::unique_ptr<T>(new ElemType[nSize]());
    }
    template<typename T, typename... TArgs>
    std::enable_if_t<(std::extent<T>::value!=0)> make_unique(TArgs&&...) = delete;
#endif //(!defined(_MSC_VER) && __cplusplus<=201103L)

} // namespace std
