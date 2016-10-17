
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

#include "litiv/utils/opencv.hpp"
#include "litiv/utils/platform.hpp"
#include <type_traits>

#if HAVE_GLSL
#include "litiv/utils/opengl-imgproc.hpp"
#endif //HAVE_GLSL
#if HAVE_CUDA
// ...
#endif //HAVE_CUDA
#if HAVE_OPENCL
// ...
#endif //HAVE_OPENCL
#if defined(_MSC_VER)
#include <intrin.h>
#if TARGET_PLATFORM_x64 && HAVE_MMX
// msvc does not define mmx intrinsics for x64 targets (even though it could...)
#undef HAVE_MMX
#define HAVE_MMX 0
#endif //TARGET_PLATFORM_x64 && HAVE_MMX
#else //(!defined(_MSC_VER))
#include <x86intrin.h>
#endif //(!defined(_MSC_VER))

namespace lv {

    enum ParallelAlgoType {
#if HAVE_GLSL
        GLSL,
#endif //HAVE_GLSL
#if HAVE_CUDA
        CUDA,
#endif //HAVE_CUDA
#if HAVE_OPENCL
        OpenCL,
#endif //HAVE_OPENCL
        NonParallel
    };

    struct IIParallelAlgo {
        /// returns whether the algorithm is implemented for parallel processing or not
        virtual bool isParallel() = 0;
        /// returns which type of parallel implementation is used in this algo
        virtual ParallelAlgoType getParallelAlgoType() = 0;
    public:
        // #### for debug purposes only ####
        cv::DisplayHelperPtr m_pDisplayHelper;
    };

    template<ParallelAlgoType eImpl>
    struct IParallelAlgo_;

#if HAVE_GLSL
    template<>
    struct IParallelAlgo_<GLSL> : public GLImageProcAlgo, public IIParallelAlgo {
        IParallelAlgo_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures, int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
            GLImageProcAlgo(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nOutputType,nDebugType,bUseInput,bUseDisplay,bUseTimers,bUseIntegralFormat) {}
        virtual bool isParallel() {return true;}
        virtual ParallelAlgoType getParallelAlgoType() {return GLSL;}
    };
    using IParallelAlgo_GLSL = IParallelAlgo_<GLSL>;
#endif //(!HAVE_GLSL)

#if HAVE_CUDA
    template<>
    struct IParallelAlgo_<CUDA> : /*public CUDAImageProcAlgo,*/ public IIParallelAlgo {
        static_assert(false,"Missing CUDA impl");
        virtual bool isParallel() {return true;}
        virtual ParallelAlgoType getParallelAlgoType() {return CUDA;}
    };
    using IParallelAlgo_CUDA = IParallelAlgo_<CUDA>;
#endif //HAVE_CUDA

#if HAVE_CUDA
    template<>
    struct IParallelAlgo_<OpenCL> : /*public OpenCLImageProcAlgo,*/ public IIParallelAlgo {
        static_assert(false,"Missing OpenCL impl");
        virtual bool isParallel() {return true;}
        virtual ParallelAlgoType getParallelAlgoType() {return OpenCL;}
    };
    using IParallelAlgo_OpenCL = IParallelAlgo_<OpenCL>;
#endif //HAVE_CUDA

    template<>
    struct IParallelAlgo_<NonParallel> : public IIParallelAlgo {
        virtual bool isParallel() {return false;}
        virtual ParallelAlgoType getParallelAlgoType() {return NonParallel;}
    };
    using NonParallelAlgo = IParallelAlgo_<NonParallel>;

#if HAVE_MMX
    /// returns the (horizontal) sum of the provided 8-unsigned-byte array
    inline uint hsum_8ub(const __m64& anBuffer) {
        __m64 _anRes = _mm_sad_pu8(anBuffer,_mm_set1_pi8(char(0)));
        return uint(_mm_cvtsi64_si32(_anRes));
    }
#endif //HAVE_MMX

#if HAVE_SSE2
    /// returns the (horizontal) sum of the provided 16-unsigned-byte array
    inline uint hsum_16ub(const __m128i& anBuffer) {
        __m128i _anRes = _mm_sad_epu8(anBuffer,_mm_set1_epi8(char(0)));
        return uint(_mm_cvtsi128_si32(_mm_add_epi64(_mm_srli_si128(_anRes,8),_anRes)));
    }

    /// fills the provided 16-byte array with a constant
    inline void copy_16ub(__m128i* anBuffer, uchar nVal) {
        _mm_store_si128(anBuffer,_mm_set1_epi8((char)nVal));
    }

    inline __m128i mult_32si(const __m128i& a, const __m128i& b) {
#if HAVE_SSE4_1
        return _mm_mullo_epi32(a, b);
#else //(!HAVE_SSE4_1)
        return _mm_unpacklo_epi32(_mm_shuffle_epi32(_mm_mul_epu32(a,b),_MM_SHUFFLE(0,0,2,0)),_mm_shuffle_epi32(_mm_mul_epu32(_mm_srli_si128(a,4),_mm_srli_si128(b,4)),_MM_SHUFFLE(0,0,2,0)));
#endif //(!HAVE_SSE4_1)
    }

    template<int nPos>
    inline int extract_32si(const __m128i& anBuffer) {
        static_assert(nPos>=0 && nPos<4,"Integer position out of bounds");
#if HAVE_SSE4_1
        return _mm_extract_epi32(anBuffer,nPos);
#else //(!HAVE_SSE4_1)
        return _mm_extract_epi16(anBuffer,nPos*2)+_mm_extract_epi16(anBuffer,nPos*2+1)<<16;
#endif //(!HAVE_SSE4_1)
    }
#endif //HAVE_SSE2

#if HAVE_SSE4_1
    /// returns the maximum value of the provided 16-unsigned-byte array
    inline uchar hmax_16ub(const __m128i& anBuffer) {
        __m128i _anTmp = _mm_sub_epi8(_mm_set1_epi8(char(CHAR_MAX)),anBuffer);
        return uchar(char(CHAR_MAX)-_mm_cvtsi128_si32(_mm_minpos_epu16(_mm_min_epu8(_anTmp,_mm_srli_epi16(_anTmp,8)))));
    }

    /// returns the minimum value of the provided 16-unsigned-byte array
    inline uchar hmin_16ub(const __m128i& anBuffer) {
        return uchar(_mm_cvtsi128_si32(_mm_minpos_epu16(_mm_min_epu8(anBuffer,_mm_srli_epi16(anBuffer,8)))));
    }
#endif //HAVE_SSE4_1

} // namespace lv
