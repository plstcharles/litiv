
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/utils/platform.hpp"
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

#if HAVE_MMX

    /// returns the (horizontal) sum of the provided unsigned byte array
    inline uint32_t hsum_8ui(const __m64& anBuffer) {
        const __m64 _anRes = _mm_sad_pu8(anBuffer,_mm_set1_pi8(int8_t(0)));
        return uint32_t(_mm_cvtsi64_si32(_anRes));
    }

#endif //HAVE_MMX

#if HAVE_SSE2

    /// returns whether the 128-bit array contains all zero bits or not
    inline bool cmp_zero_128i(const __m128i& anBuffer) {
    #if HAVE_SSE4_1
        return _mm_testz_si128(anBuffer,anBuffer)!=0;
    #else //!HAVE_SSE4_1
        static constexpr __m128i anZeroBuffer = {0}; // compare to _mm_setzero_si128?
        return _mm_movemask_epi8(_mm_cmpeq_epi32(anBuffer,anZeroBuffer))==0xFFFF;
    #endif //!HAVE_SSE4_1
    }

    /// returns whether the two 128-bit arrays are identical or not
    inline bool cmp_eq_128i(const __m128i& a, const __m128i& b) {
        static constexpr __m128i anZeroBuffer = {0}; // compare to _mm_setzero_si128?
    #if HAVE_SSE4_1
        return _mm_testc_si128(anZeroBuffer,_mm_xor_si128(a, b))!=0;
    #else //!HAVE_SSE4_1
        return _mm_movemask_epi8(_mm_cmpeq_epi32(a,b))==0xFFFF;
    #endif //!HAVE_SSE4_1
    }

    /// returns the (horizontal) sum of the provided unsigned byte array
    inline uint32_t hsum_8ui(const __m128i& anBuffer) {
        const __m128i _anRes = _mm_sad_epu8(anBuffer,_mm_set1_epi8(int8_t(0)));
        return uint32_t(_mm_cvtsi128_si32(_mm_add_epi64(_mm_srli_si128(_anRes,8),_anRes)));
    }

    /// returns the (horizontal) sum of the provided signed integer array
    inline int32_t hsum_32i(const __m128i& anBuffer) {
        const __m128i anSum64 = _mm_add_epi32(anBuffer,_mm_shuffle_epi32(anBuffer,_MM_SHUFFLE(1,0,3,2)));
        const __m128i anSum32 = _mm_add_epi32(_mm_shufflelo_epi16(anSum64,_MM_SHUFFLE(1,0,3,2)),anSum64);
        return _mm_cvtsi128_si32(anSum32);
    }

    /// fills the provided 16-byte array with a uint8_t constant
    inline void store1_8ui(__m128i* anBuffer, uint8_t nVal) {
        lvDbgAssert_(((uintptr_t)anBuffer%16)==0,"buffer must be 16-byte aligned");
        _mm_store_si128(anBuffer,_mm_set1_epi8((char)nVal));
    }

    /// fills 'nCount' consecutive 16-byte arrays with a uint8_t constant (i.e. fast fill_n)
    inline void store_8ui(__m128i* anBuffer, size_t nCount, uint8_t nVal) {
        lvDbgAssert_(((uintptr_t)anBuffer%16)==0,"buffer must be 16-byte aligned");
        const __m128i vVal = _mm_set1_epi8((char)nVal);
        for(size_t nIter=0; nIter<nCount; ++nIter)
            _mm_store_si128(anBuffer+nIter,vVal);
    }

    /// multiplies two sets of four 32-bit signed integers
    inline __m128i mult_32si(const __m128i& a, const __m128i& b) {
    #if HAVE_SSE4_1
        return _mm_mullo_epi32(a, b);
    #else //(!HAVE_SSE4_1)
        return _mm_unpacklo_epi32(_mm_shuffle_epi32(_mm_mul_epu32(a,b),_MM_SHUFFLE(0,0,2,0)),_mm_shuffle_epi32(_mm_mul_epu32(_mm_srli_si128(a,4),_mm_srli_si128(b,4)),_MM_SHUFFLE(0,0,2,0)));
    #endif //(!HAVE_SSE4_1)
    }

    /// extracts a single 32-bit signed integer value at position 'nPos' from the given array
    template<int nPos>
    inline int32_t extract_32si(const __m128i& anBuffer) {
        static_assert(nPos>=0 && nPos<4,"Integer position out of bounds");
    #if HAVE_SSE4_1
        return _mm_extract_epi32(anBuffer,nPos);
    #else //(!HAVE_SSE4_1)
        return _mm_extract_epi16(anBuffer,nPos*2)+_mm_extract_epi16(anBuffer,nPos*2+1)<<16;
    #endif //(!HAVE_SSE4_1)
    }

#endif //HAVE_SSE2

#if HAVE_SSE4_1

    /// returns the minimum value of the provided 16-unsigned-byte array
    inline uint8_t hmin_8ui(const __m128i& anBuffer) {
        return uint8_t(_mm_cvtsi128_si32(_mm_minpos_epu16(_mm_min_epu8(anBuffer,_mm_srli_epi16(anBuffer,8)))));
    }

    /// returns the maximum value of the provided 16-unsigned-byte array
    inline uint8_t hmax_8ui(const __m128i& anBuffer) {
        return uint8_t(255)-lv::hmin_8ui(_mm_sub_epi8(_mm_set1_epi8(int8_t(-1)),anBuffer));
    }

    /// returns the maximum value of the provided signed integer array
    inline int32_t hmax_32si(const __m128i& anBuffer) {
        const __m128i anHighMax = _mm_max_epi32(anBuffer,_mm_shuffle_epi32(anBuffer,_MM_SHUFFLE(0,0,3,2)));
        return _mm_cvtsi128_si32(_mm_max_epi32(anHighMax,_mm_shuffle_epi32(anHighMax,_MM_SHUFFLE(0,0,0,1))));
    }

#endif //HAVE_SSE4_1

} // namespace lv
