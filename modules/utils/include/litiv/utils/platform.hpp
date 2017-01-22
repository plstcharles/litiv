
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

    /// helper alias; std-friendly version of vector with N-byte aligned memory allocator
    template<typename T, size_t N>
    using aligned_vector = std::vector<T,lv::AlignedMemAllocator<T,N>>;

} // namespace lv

#if defined(_MSC_VER)
namespace {
    template<class T>
    inline void SafeRelease(T** ppT) {
        if(*ppT) {
            (*ppT)->Release();
            *ppT = nullptr;
        }
    }
}
#endif //defined(_MSC_VER)