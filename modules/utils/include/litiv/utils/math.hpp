
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

#include "litiv/utils/simd.hpp"

namespace lv {

    /// returns whether a value is NaN (required due to non-portable msvc signature)
    template<typename T>
    inline bool isnan(T dVal) {
#ifdef _MSC_VER // needed for portability...
        return _isnan((double)dVal)!=0;
#else //(!def(_MSC_VER))
        return std::isnan(dVal);
#endif //(!def(_MSC_VER))
    }

    /// returns whether an integer is a power of two
    template<typename T>
    inline bool ispow2(T nVal) {
        static_assert(std::is_integral<T>::value,"ispow2 only excepts integer types");
        return ((nVal&(nVal-1))==0);
    }

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-aliasing"
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif //defined(...GCC)

    /// computes the fast absolute value of a floating point value (relies on sign-extended shift)
    template<typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
    inline T abs_fast(T x) {
        static_assert(std::is_same<T,float>::value||std::is_same<T,double>::value,"abs_fast not defined for long double or non-ieee fp types");
        static_assert(sizeof(T)==4||sizeof(T)==8,"unexpected fp type size");
        using Tint = std::conditional_t<sizeof(T)==4,int32_t,int64_t>;
        Tint& nCast = reinterpret_cast<Tint&>(x);
        nCast &= std::numeric_limits<Tint>::max();
        return x;
    }

    /// computes the fast inverse of a floating point value -- http://bits.stephan-brumme.com/inverse.html
    inline float inv_fast(float x) {
        uint32_t& i = reinterpret_cast<uint32_t&>(x);
        i = 0x7F000000 - i;
        return x;
    }

    /// computes the fast inverse square root of a floating point value -- http://bits.stephan-brumme.com/invSquareRoot.html
    inline float invsqrt_fastest(float x) {
        lvDbgAssert_(x>=0.0f,"function undefined for negative values");
        uint32_t& i = reinterpret_cast<uint32_t&>(x);
        i = 0x5F375A86 - (i>>1);
        return x;
    }

    /// computes the fast inverse square root of a floating point value w/ optional newton iterations optimization -- http://bits.stephan-brumme.com/invSquareRoot.html
    template<size_t nNewtonIters=1>
    inline float invsqrt_fast(float x) {
        lvDbgAssert_(x>=0.0f,"function undefined for negative values");
        static_assert(nNewtonIters>=1,"use invsqrt_fastest to skip Newton iterations");
        const float fHalf = 0.5f*x;
        float fRes = invsqrt_fastest(x);
        for(size_t i=0; i<nNewtonIters; ++i)
            fRes = fRes*(1.5f-fHalf*fRes*fRes);
        return fRes;
    }

    /// computes the fast square root of a floating point value -- http://bits.stephan-brumme.com/squareRoot.html
    inline float sqrt_fast(float x) {
        lvDbgAssert_(x>=0.0f,"function undefined for negative values");
        uint32_t& i = reinterpret_cast<uint32_t&>(x);
        i += 127<<23;
        i >>= 1;
        return x;
    }

#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#endif //defined(...GCC)

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /// computes the L1 distance between two integer values; returns an unsigned type of the same size as the input type
    template<typename T, typename=std::enable_if_t<std::is_integral<T>::value>>
    inline auto L1dist(T a, T b) {
        static_assert(!std::is_same<T,bool>::value,"L1dist not specialized for boolean types");
        return (std::make_unsigned_t<T>)std::abs(((std::make_signed_t<typename lv::get_bigger_integer<T>::type>)a)-b);
    }

    /// computes the L1 distance between two floating point values (with bit trick)
    template<typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
    inline T _L1dist_cheat(T a, T b) {
        return abs_fast(a-b);
    }

    /// computes the L1 distance between two floating point values (without bit trick)
    template<typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
    inline T _L1dist_nocheat(T a, T b) {
        return std::abs(a-b);
    }

    /// computes the L1 distance between two floating point values (without bit trick)
    template<typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
    inline T L1dist(T a, T b) {
#if USE_SIGNEXT_SHIFT_TRICK
        return _L1dist_cheat(a,b);
#else //!USE_SIGNEXT_SHIFT_TRICK
        return _L1dist_nocheat(a,b);
#endif //!USE_SIGNEXT_SHIFT_TRICK
    }

    /// computes the L1 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L1dist(Tin(),Tin()))>
    inline Tout L1dist(const Tin* a, const Tin* b) {
        Tout tResult = 0;
        for(size_t c=0; c<nChannels; ++c)
            tResult += (Tout)L1dist(a[c],b[c]);
        return tResult;
    }

    /// computes the L1 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L1dist(Tin(),Tin()))>
    inline Tout L1dist(const std::array<Tin,nChannels>& a, const std::array<Tin,nChannels>& b) {
        return L1dist<nChannels,Tin,Tout>(a.data(),b.data());
    }

    /// computes the L1 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L1dist(Tin(),Tin()))>
    inline Tout L1dist(const std::array<Tin,nChannels>& a, const Tin* b) {
        return L1dist<nChannels,Tin,Tout>(a.data(),b);
    }

    /// computes the L1 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L1dist(Tin(),Tin()))>
    inline Tout L1dist(const Tin* a, const std::array<Tin,nChannels>& b) {
        return L1dist<nChannels,Tin,Tout>(a,b.data());
    }

    /// computes the L1 distance between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<size_t nChannels, typename Tin, typename Tout=std::conditional_t<std::is_integral<Tin>::value,size_t,float>>
    inline Tout L1dist(const Tin* a, const Tin* b, size_t nElements, const uint8_t* m=nullptr) {
        Tout tResult = 0;
        const size_t nTotElements = nElements*nChannels;
        if(m) {
            for(size_t n=0,i=0; n<nTotElements; n+=nChannels,++i)
                if(m[i])
                    tResult += L1dist<nChannels,Tin,Tout>(a+n,b+n);
        }
        else {
            for(size_t n=0; n<nTotElements; n+=nChannels)
                tResult += L1dist<nChannels,Tin,Tout>(a+n,b+n);
        }
        return tResult;
    }

    /// computes the L1 distance between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<typename Tin, typename Tout=std::conditional_t<std::is_integral<Tin>::value,size_t,float>>
    inline Tout L1dist(const Tin* a, const Tin* b, size_t nElements, size_t nChannels, const uint8_t* m=nullptr) {
        lvAssert_(nChannels>0 && nChannels<=4,"untemplated distance function only defined for 1 to 4 channels");
        switch(nChannels) {
            case 1: return L1dist<1,Tin,Tout>(a,b,nElements,m);
            case 2: return L1dist<2,Tin,Tout>(a,b,nElements,m);
            case 3: return L1dist<3,Tin,Tout>(a,b,nElements,m);
            case 4: return L1dist<4,Tin,Tout>(a,b,nElements,m);
            default: return (Tout)0;
        }
    }

#if USE_CVCORE_WITH_UTILS

    /// computes the L1 distance between two opencv vectors
    template<int nChannels, typename Tin, typename Tout=decltype(L1dist(Tin(),Tin()))>
    inline Tout L1dist(const cv::Vec<Tin,nChannels>& a, const cv::Vec<Tin,nChannels>& b) {
        Tin a_array[nChannels], b_array[nChannels];
        for(int c=0; c<nChannels; ++c) {
            a_array[c] = a[c];
            b_array[c] = b[c];
        }
        return L1dist<nChannels,Tin,Tout>(a_array,b_array);
    }

#endif //USE_CVCORE_WITH_UTILS

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /// computes the squared L2 distance between two integer values (i.e. == squared L1dist); returns an unsigned type of twice the size of the input type
    template<typename T, typename=std::enable_if_t<std::is_integral<T>::value>>
    inline auto L2sqrdist(T a, T b) {
        typedef std::make_signed_t<typename lv::get_bigger_integer<T>::type> Tintern;
        const Tintern tResult = Tintern(a)-Tintern(b);
        return std::make_unsigned_t<Tintern>(tResult*tResult);
    }

    /// computes the squared L2 distance between two floating point values (i.e. == squared L1dist)
    template<typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
    inline T L2sqrdist(T a, T b) {
        const T tResult = a-b;
        return tResult*tResult;
    }

    /// computes the squared L2 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L2sqrdist(Tin(),Tin()))>
    inline Tout L2sqrdist(const Tin* a, const Tin* b) {
        Tout tResult = 0;
        for(size_t c=0; c<nChannels; ++c)
            tResult += (Tout)L2sqrdist(a[c],b[c]);
        return tResult;
    }

    /// computes the squared L2 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L2sqrdist(Tin(),Tin()))>
    inline Tout L2sqrdist(const std::array<Tin,nChannels>& a, const std::array<Tin,nChannels>& b) {
        return L2sqrdist<nChannels,Tin,Tout>(a.data(),b.data());
    }

    /// computes the squared L2 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L2sqrdist(Tin(),Tin()))>
    inline Tout L2sqrdist(const std::array<Tin,nChannels>& a, const Tin* b) {
        return L2sqrdist<nChannels,Tin,Tout>(a.data(),b);
    }

    /// computes the squared L2 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L2sqrdist(Tin(),Tin()))>
    inline Tout L2sqrdist(const Tin* a, const std::array<Tin,nChannels>& b) {
        return L2sqrdist<nChannels,Tin,Tout>(a,b.data());
    }

    /// computes the squared L2 distance between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<size_t nChannels, typename Tin, typename Tout=std::conditional_t<std::is_integral<Tin>::value,size_t,float>>
    inline Tout L2sqrdist(const Tin* a, const Tin* b, size_t nElements, const uint8_t* m=nullptr) {
        Tout tResult = 0;
        const size_t nTotElements = nElements*nChannels;
        if(m) {
            for(size_t n=0,i=0; n<nTotElements; n+=nChannels,++i)
                if(m[i])
                    tResult += L2sqrdist<nChannels,Tin,Tout>(a+n,b+n);
        }
        else {
            for(size_t n=0; n<nTotElements; n+=nChannels)
                tResult += L2sqrdist<nChannels,Tin,Tout>(a+n,b+n);
        }
        return tResult;
    }

    /// computes the squared L2 distance between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<typename Tin, typename Tout=std::conditional_t<std::is_integral<Tin>::value,size_t,float>>
    inline Tout L2sqrdist(const Tin* a, const Tin* b, size_t nElements, size_t nChannels, const uint8_t* m=nullptr) {
        lvAssert_(nChannels>0 && nChannels<=4,"untemplated distance function only defined for 1 to 4 channels");
        switch(nChannels) {
            case 1: return L2sqrdist<1,Tin,Tout>(a,b,nElements,m);
            case 2: return L2sqrdist<2,Tin,Tout>(a,b,nElements,m);
            case 3: return L2sqrdist<3,Tin,Tout>(a,b,nElements,m);
            case 4: return L2sqrdist<4,Tin,Tout>(a,b,nElements,m);
            default: return (Tout)0;
        }
    }

#if USE_CVCORE_WITH_UTILS

    /// computes the squared L2 distance between two opencv vectors
    template<int nChannels, typename Tin, typename Tout=decltype(L2sqrdist(Tin(),Tin()))>
    inline Tout L2sqrdist(const cv::Vec<Tin,nChannels>& a, const cv::Vec<Tin,nChannels>& b) {
        Tin a_array[nChannels], b_array[nChannels];
        for(int c=0; c<nChannels; ++c) {
            a_array[c] = a[(int)c];
            b_array[c] = b[(int)c];
        }
        return L2sqrdist<nChannels,Tin,Tout>(a_array,b_array);
    }

#endif //USE_CVCORE_WITH_UTILS

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /// computes the L2 distance between two integer arrays
    template<size_t nChannels, typename Tin, typename Tout=float, typename=std::enable_if_t<std::is_integral<Tin>::value>>
    inline Tout L2dist(const Tin* a, const Tin* b) {
        decltype(L2sqrdist(Tin(),Tin())) tResult = 0;
        for(size_t c=0; c<nChannels; ++c)
            tResult += L2sqrdist(a[c],b[c]);
        return (Tout)std::sqrt(tResult);
    }

    /// computes the L2 distance between two floating point arrays
    template<size_t nChannels, typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
    inline T L2dist(const T* a, const T* b) {
        T tResult = 0;
        for(size_t c=0; c<nChannels; ++c)
            tResult += L2sqrdist(a[c],b[c]);
        return std::sqrt(tResult);
    }

    /// computes the L2 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L2dist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout L2dist(const std::array<Tin,nChannels>& a, const std::array<Tin,nChannels>& b) {
        return (Tout)L2dist<nChannels>(a.data(),b.data());
    }

    /// computes the L2 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L2dist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout L2dist(const std::array<Tin,nChannels>& a, const Tin* b) {
        return (Tout)L2dist<nChannels>(a.data(),b);
    }

    /// computes the L2 distance between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(L2dist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout L2dist(const Tin* a, const std::array<Tin,nChannels>& b) {
        return (Tout)L2dist<nChannels>(a,b.data());
    }

    /// computes the L2 distance between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<size_t nChannels, typename Tin, typename Tout=float, typename Tintern=std::conditional_t<std::is_integral<Tin>::value,size_t,float>>
    inline Tout L2dist(const Tin* a, const Tin* b, size_t nElements, const uint8_t* m=nullptr) {
        Tintern tResult = 0;
        const size_t nTotElements = nElements*nChannels;
        if(m) {
            for(size_t n=0,i=0; n<nTotElements; n+=nChannels,++i)
                if(m[i])
                    tResult += L2sqrdist<nChannels,Tin,Tintern>(a+n,b+n);
        }
        else {
            for(size_t n=0; n<nTotElements; n+=nChannels)
                tResult += L2sqrdist<nChannels,Tin,Tintern>(a+n,b+n);
        }
        return (Tout)std::sqrt(tResult);
    }

    /// computes the squared L2 distance between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<typename Tin, typename Tout=decltype(L2dist<3>((Tin*)0,(Tin*)0))>
    inline Tout L2dist(const Tin* a, const Tin* b, size_t nElements, size_t nChannels, const uint8_t* m=nullptr) {
        lvAssert_(nChannels>0 && nChannels<=4,"untemplated distance function only defined for 1 to 4 channels");
        switch(nChannels) {
            case 1: return L2dist<1,Tin,Tout>(a,b,nElements,m);
            case 2: return L2dist<2,Tin,Tout>(a,b,nElements,m);
            case 3: return L2dist<3,Tin,Tout>(a,b,nElements,m);
            case 4: return L2dist<4,Tin,Tout>(a,b,nElements,m);
            default: return (Tout)0;
        }
    }

#if USE_CVCORE_WITH_UTILS

    /// computes the L2 distance between two opencv vectors
    template<int nChannels, typename Tin, typename Tout=decltype(L2dist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout L2dist(const cv::Vec<Tin,nChannels>& a, const cv::Vec<Tin,nChannels>& b) {
        Tin a_array[nChannels], b_array[nChannels];
        for(int c=0; c<nChannels; ++c) {
            a_array[c] = a[(int)c];
            b_array[c] = b[(int)c];
        }
        return (Tout)L2dist<nChannels>(a_array,b_array);
    }

#endif //USE_CVCORE_WITH_UTILS

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /// computes the color distortion between two unsigned integer arrays
    template<size_t nChannels, typename T, typename=std::enable_if_t<std::is_integral<T>::value>>
    inline size_t cdist(const T* curr, const T* bg) {
        static_assert(std::is_unsigned<T>::value,"cdist does not support negative values");
        static_assert(nChannels>1,"vectors should have more than one channel");
        bool bNonConstDist = false;
        bool bNonNullDist = (curr[0]!=bg[0]);
        for(size_t c=1; c<nChannels; ++c) {
            bNonConstDist |= (curr[c]!=curr[c-1]) || (bg[c]!=bg[c-1]);
            bNonNullDist |= (curr[c]!=bg[c]);
        }
        if(!bNonConstDist || !bNonNullDist)
            return 0;
        uint64_t curr_sqr = 0;
        uint64_t bg_sqr = 0;
        uint64_t mix = 0;
        for(size_t c=0; c<nChannels; ++c) {
            curr_sqr += uint64_t(curr[c]*curr[c]);
            bg_sqr += uint64_t(bg[c]*bg[c]);
            mix += uint64_t(curr[c]*bg[c]);
        }
        const float fSqrDistort = (float)(curr_sqr-(mix*mix)/std::max(bg_sqr,uint64_t(1)));
        return (size_t)std::sqrt(fSqrDistort); // will already be well optimized for integer output
    }

    /// computes the color distortion between two floating point arrays
    template<size_t nChannels, typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
    inline T cdist(const T* curr, const T* bg) {
        static_assert(nChannels>1,"vectors should have more than one channel");
        lvDbgAssert_(curr[0]>=0.0f && bg[0]>=0.0f,"cdist does not support negative values");
        bool bNonConstDist = false;
        bool bNonNullDist = (curr[0]!=bg[0]);
        for(size_t c=1; c<nChannels; ++c) {
            lvDbgAssert_(curr[c]>=0.0f && bg[c]>=0.0f,"cdist does not support negative values");
            bNonConstDist |= (curr[c]!=curr[c-1]) || (bg[c]!=bg[c-1]);
            bNonNullDist |= (curr[c]!=bg[c]);
        }
        if(!bNonConstDist || !bNonNullDist)
            return (T)0;
        T curr_sqr = 0;
        T bg_sqr = 0;
        T mix = 0;
        for(size_t c=0; c<nChannels; ++c) {
            curr_sqr += curr[c]*curr[c];
            bg_sqr += bg[c]*bg[c];
            mix += curr[c]*bg[c];
        }
        bg_sqr += std::numeric_limits<T>::epsilon();
        if(curr_sqr<=(mix*mix)/bg_sqr)
            return (T)0;
        else {
            const float fSqrDistort = (float)(curr_sqr-(mix*mix)/bg_sqr);
#if USE_FAST_SQRT_FOR_CDIST
            return (T)(lv::invsqrt_fast(fSqrDistort)*fSqrDistort);
#else //!USE_FAST_SQRT_FOR_CDIST
            return (T)std::sqrt(fSqrDistort);
#endif //!USE_FAST_SQRT_FOR_CDIST
        }
    }

    /// computes the color distortion between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(cdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout cdist(const std::array<Tin,nChannels>& a, const std::array<Tin,nChannels>& b) {
        return (Tout)cdist<nChannels>(a.data(),b.data());
    }

    /// computes the color distortion between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(cdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout cdist(const std::array<Tin,nChannels>& a, const Tin* b) {
        return (Tout)cdist<nChannels>(a.data(),b);
    }

    /// computes the color distortion between two generic arrays
    template<size_t nChannels, typename Tin, typename Tout=decltype(cdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout cdist(const Tin* a, const std::array<Tin,nChannels>& b) {
        return (Tout)cdist<nChannels>(a,b.data());
    }

    /// computes the color distortion between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<size_t nChannels, typename Tin, typename Tout=decltype(cdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout cdist(const Tin* a, const Tin* b, size_t nElements, const uint8_t* m=nullptr) {
        Tout tResult = 0;
        const size_t nTotElements = nElements*nChannels;
        if(m) {
            for(size_t n=0,i=0; n<nTotElements; n+=nChannels,++i)
                if(m[i])
                    tResult += cdist<nChannels>(a+n,b+n);
        }
        else {
            for(size_t n=0; n<nTotElements; n+=nChannels)
                tResult += cdist<nChannels>(a+n,b+n);
        }
        return tResult;
    }

    /// computes the color distortion between two generic arrays (note: for very large arrays, using ocv matrix ops will be faster)
    template<typename Tin, typename Tout=decltype(cdist<3>((Tin*)0,(Tin*)0))>
    inline Tout cdist(const Tin* a, const Tin* b, size_t nElements, size_t nChannels, const uint8_t* m=nullptr) {
        lvAssert_(nChannels>1 && nChannels<=4,"untemplated distance function only defined for 2 to 4 channels");
        switch(nChannels) {
            case 2: return cdist<2,Tin,Tout>(a,b,nElements,m);
            case 3: return cdist<3,Tin,Tout>(a,b,nElements,m);
            case 4: return cdist<4,Tin,Tout>(a,b,nElements,m);
            default: return (Tout)0;
        }
    }

#if USE_CVCORE_WITH_UTILS

    /// computes the color distortion between two opencv vectors
    template<int nChannels, typename Tin, typename Tout=decltype(cdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout cdist(const cv::Vec<Tin,nChannels>& a, const cv::Vec<Tin,nChannels>& b) {
        Tin a_array[nChannels], b_array[nChannels];
        for(int c=0; c<nChannels; ++c) {
            a_array[c] = a[(int)c];
            b_array[c] = b[(int)c];
        }
        return (Tout)cdist<nChannels>(a_array,b_array);
    }

#endif //USE_CVCORE_WITH_UTILS

    /// computes a color distortion-distance mix using two generic distances
    template<typename TL1Dist, typename TCDist>
    inline auto cmixdist(TL1Dist tL1Distance, TCDist tCDistortion) {
        return (tL1Distance/2+tCDistortion*4);
    }

    /// computes a color distortion-distance mix using two generic arrays
    template<size_t nChannels, typename T>
    inline auto cmixdist(const T* curr, const T* bg) {
        return cmixdist(L1dist<nChannels>(curr,bg),cdist<nChannels>(curr,bg));
    }

    /// computes a color distortion-distance mix using two generic arrays
    template<size_t nChannels, typename T>
    inline auto cmixdist(const std::array<T,nChannels>& a, const std::array<T,nChannels>& b) {
        return cmixdist<nChannels>(a.data(),b.data());
    }

    /// computes a color distortion-distance mix using two generic arrays
    template<size_t nChannels, typename T>
    inline auto cmixdist(const std::array<T,nChannels>& a, const T* b) {
        return cmixdist<nChannels>(a.data(),b);
    }

    /// computes a color distortion-distance mix using two generic arrays
    template<size_t nChannels, typename T>
    inline auto cmixdist(const T* a, const std::array<T,nChannels>& b) {
        return cmixdist<nChannels>(a,b.data());
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    /// computes the population count of an 8-bit vector using an 8-bit popcount LUT
    template<typename Tin, typename Tout=uint8_t>
    inline std::enable_if_t<sizeof(Tin)==1,Tout> popcount(const Tin x) {
        static_assert(std::is_integral<Tin>::value,"type must be integral");
        static_assert(std::numeric_limits<unsigned char>::digits==8,"lots of stuff is going to break...");
        static constexpr Tout s_anPopcntLUT8[256] = {
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
            4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
        };
        return s_anPopcntLUT8[reinterpret_cast<const uint8_t&>(x)];
    }

#if HAVE_POPCNT

    /// computes the population count of a 2- or 4-byte vector using 32-bit popcnt instruction
    template<typename Tin, typename Tout=uint8_t>
    inline std::enable_if_t<(sizeof(Tin)==2 || sizeof(Tin)==4),Tout> popcount(const Tin x) {
        static_assert(std::is_integral<Tin>::value,"type must be integral");
        return (Tout)_mm_popcnt_u32((uint)reinterpret_cast<std::make_unsigned_t<const Tin>&>(x));
    }

#if TARGET_PLATFORM_x64

    /// computes the population count of an 8-byte vector using 64-bit popcnt instruction
    template<typename Tin, typename Tout=uint8_t>
    inline std::enable_if_t<sizeof(Tin)==8,Tout> popcount(const Tin x) {
        static_assert(std::is_integral<Tin>::value,"type must be integral");
        return (Tout)_mm_popcnt_u64(reinterpret_cast<const uint64_t&>(x));
    }

#endif //TARGET_PLATFORM_x64

#else //(!HAVE_POPCNT)

    /// computes the population count of an N-byte vector using an 8-bit popcount LUT
    template<typename Tin, typename Tout=uint8_t>
    inline std::enable_if_t<(sizeof(Tin)>1),Tout> popcount(const Tin x) {
        static_assert(std::is_integral<Tin>::value,"type must be integral");
        Tout tResult = 0;
        for(size_t l=0; l<sizeof(Tin); ++l)
            tResult += popcount<uint8_t,Tout>((uint8_t)(x>>(l*8)));
        return tResult;
    }

#endif //(!HAVE_POPCNT)

    /// computes the population count of a (nChannels*N)-byte vector
    template<size_t nChannels, typename Tin, typename Tout=decltype(popcount(Tin()))>
    inline Tout popcount(const Tin* x) {
        Tout tResult = 0;
        for(size_t c=0; c<nChannels; ++c)
            tResult += popcount<Tin,Tout>(x[c]);
        return tResult;
    }

    /// computes the population count of a (nChannels*N)-byte vector
    template<size_t nChannels, typename Tin, typename Tout=decltype(popcount(Tin()))>
    inline Tout popcount(const std::array<Tin,nChannels>& x) {
        return popcount<nChannels,Tin,Tout>(x.data());
    }

    /// computes the hamming distance between two N-byte vectors
    template<typename Tin, typename Tout=decltype(popcount(Tin()))>
    inline Tout hdist(Tin a, Tin b) {
        static_assert(std::is_integral<Tin>::value,"type must be integral");
        return popcount<Tin,Tout>(a^b);
    }

    /// computes the hamming distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename Tin, typename Tout=decltype(popcount(Tin()))>
    inline Tout hdist(const Tin* a, const Tin* b) {
        static_assert(std::is_integral<Tin>::value,"type must be integral");
        Tout tResult = 0;
        for(size_t c=0; c<nChannels; ++c)
            tResult += popcount<Tin,Tout>(a[c]^b[c]);
        return tResult;
    }

    /// computes the hamming distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename Tin, typename Tout=decltype(hdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout hdist(const std::array<Tin,nChannels>& a, const std::array<Tin,nChannels>& b) {
        return hdist<nChannels,Tin,Tout>(a.data(),b.data());
    }

    /// computes the hamming distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename Tin, typename Tout=decltype(hdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout hdist(const std::array<Tin,nChannels>& a, const Tin* b) {
        return hdist<nChannels,Tin,Tout>(a.data(),b);
    }

    /// computes the hamming distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename Tin, typename Tout=decltype(hdist<nChannels>((Tin*)0,(Tin*)0))>
    inline Tout hdist(const Tin* a, const std::array<Tin,nChannels>& b) {
        return hdist<nChannels,Tin,Tout>(a,b.data());
    }

    /// computes the gradient magnitude distance between two N-byte vectors
    template<typename T>
    inline auto gdist(T a, T b) {
        return L1dist(popcount(a),popcount(b));
    }

    /// computes the gradient magnitude distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename T>
    inline auto gdist(const T* a, const T* b) {
        return L1dist(popcount<nChannels>(a),popcount<nChannels>(b));
    }

    /// computes the gradient magnitude distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename T>
    inline auto gdist(const std::array<T,nChannels>& a, const std::array<T,nChannels>& b) {
        return gdist<nChannels>(a.data(),b.data());
    }

    /// computes the gradient magnitude distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename T>
    inline auto gdist(const std::array<T,nChannels>& a, const T* b) {
        return gdist<nChannels>(a.data(),b);
    }

    /// computes the gradient magnitude distance between two (nChannels*N)-byte vectors
    template<size_t nChannels, typename T>
    inline auto gdist(const T* a, const std::array<T,nChannels>& b) {
        return gdist<nChannels>(a,b.data());
    }

#if HAVE_GLSL

    inline std::string getShaderFunctionSource_absdiff(bool bUseBuiltinDistance) {
        // @@@@ test with/without built-in in final impl
        std::stringstream ssSrc;
        ssSrc << "uvec3 absdiff(in uvec3 a, in uvec3 b) {\n"
                 "    return uvec3(abs(ivec3(a)-ivec3(b)));\n"
                 "}\n"
                 "uint absdiff(in uint a, in uint b) {\n"
                 "    return uint(" << (bUseBuiltinDistance?"distance(a,b)":"abs((int)a-(int)b)") << ");\n"
                 "}\n";
        return ssSrc.str();
    }

    inline std::string getShaderFunctionSource_L1dist() {
        std::stringstream ssSrc;
        ssSrc << "uint L1dist(in uvec3 a, in uvec3 b) {\n"
                 "    ivec3 absdiffs = abs(ivec3(a)-ivec3(b));\n"
                 "    return uint(absdiffs.b+absdiffs.g+absdiffs.r);\n"
                 "}\n";
        return ssSrc.str();
    }

    inline std::string getShaderFunctionSource_L2dist(bool bUseBuiltinDistance) {
        std::stringstream ssSrc;
        ssSrc << "uint L2dist(in uvec3 a, in uvec3 b) {\n"
                 "    return uint(" << (bUseBuiltinDistance?"distance(a,b)":"sqrt(dot(ivec3(a)-ivec3(b)))") << ");\n"
                 "}\n";
        return ssSrc.str();
    }

    inline std::string getShaderFunctionSource_hdist() {
        std::stringstream ssSrc;
        ssSrc << "uvec3 hdist(in uvec3 a, in uvec3 b) {\n"
                 "    return bitCount(a^b);\n"
                 "}\n"
                 "uint hdist(in uint a, in uint b) {\n"
                 "    return bitCount(a^b);\n"
                 "}\n";
        return ssSrc.str();
    }

#endif //HAVE_GLSL

    /// returns the L1-EMD (earth mover's distance) between two normalized vectors of the same size
    template<typename TVal, typename TDist=double>
    inline TDist EMDL1dist(const std::vector<TVal>& vArr1, const std::vector<TVal>& vArr2) {
        // special EMD case described by Rubner et al. in "The Earth Mover’s Distance as a Metric for Image Retrieval" (IJCV2000)
        static_assert(std::is_floating_point<TVal>::value,"input must be floating point & normalized");
        lvDbgAssert_(std::abs(std::accumulate(vArr1.begin(),vArr1.end(),0.0)-1.0)<FLT_EPSILON*vArr1.size(),"input array is not normalized");
        lvDbgAssert_(std::abs(std::accumulate(vArr2.begin(),vArr2.end(),0.0)-1.0)<FLT_EPSILON*vArr2.size(),"input array is not normalized");
        lvDbgAssert_(vArr1.size()==vArr2.size(),"input arrays must be same size");
        const std::vector<TDist> vSums1 = lv::cumulativeSum<TVal,TDist>(vArr1);
        const std::vector<TDist> vSums2 = lv::cumulativeSum<TVal,TDist>(vArr2);
        return lv::L1dist<1,TDist,TDist>(vSums1.data(),vSums2.data(),vSums1.size());
    }

    /// returns the L1-CEMD (circular earth mover's distance) between two normalized vectors of the same size
    template<typename TVal, typename TDist=double>
    inline TDist CEMDL1dist(const std::vector<TVal>& vArr1, const std::vector<TVal>& vArr2) {
        // 'circular' variant of EMDL1 described by Rabin et al. in "Circular Earth Mover’s Distance for the Comparison of Local Features" (ICPR2008)
        static_assert(std::is_floating_point<TVal>::value,"input must be floating point & normalized");
        lvDbgAssert_(std::abs(std::accumulate(vArr1.begin(),vArr1.end(),0.0)-1.0)<FLT_EPSILON*vArr1.size(),"input array is not normalized");
        lvDbgAssert_(std::abs(std::accumulate(vArr2.begin(),vArr2.end(),0.0)-1.0)<FLT_EPSILON*vArr2.size(),"input array is not normalized");
        lvDbgAssert_(vArr1.size()==vArr2.size(),"input arrays must be same size");
        std::vector<TDist> vCumSumDiffs(1,TDist(0));
        vCumSumDiffs.reserve(vArr1.size()+1);
        std::transform(vArr1.begin(),vArr1.end(),vArr2.begin(),std::back_inserter(vCumSumDiffs),[&](const TVal& a, const TVal& b) {
            return vCumSumDiffs.back()+TDist(a-b);
        });
        std::nth_element(vCumSumDiffs.begin()+1,vCumSumDiffs.begin()+1+vArr1.size()/2,vCumSumDiffs.end());
        const TDist tMedian = vCumSumDiffs[1+vArr1.size()/2];
        TDist tResult = TDist(0);
        for(size_t nIdx=1; nIdx<vCumSumDiffs.size(); ++nIdx)
            tResult += std::abs(vCumSumDiffs[nIdx]-tMedian);
        return tResult;
    }

    /// performs 'root-sift'-like descriptor value adjustment via Hellinger kernel to improve matching performance
    template<typename TVal, bool bUseL2Norm=false, bool bNegativeCompat=false>
    inline void rootSIFT(TVal* aDesc, size_t nDescSize) {
        // the original strategy consist of three steps: L1-normalization, per-elem square root, and L2-normalization.
        // for some descriptors (e.g. SIFT), the final step (L2-norm) is not required (check bUseL2Norm appropriately)
        // for more info, see Arandjelovic and Zisserman's "Three things everyone should know to improve object retrieval" (CVPR2012)
        const double dL1Norm = std::accumulate(aDesc,aDesc+nDescSize,0.0,[](double dSum, TVal tVal){
            lvDbgAssert_(bNegativeCompat || double(tVal)>=0.0,"bad function config, cannot handle negative descriptor bins");
            return dSum+(bNegativeCompat?std::abs(tVal):tVal);
        }) + DBL_EPSILON;
        std::transform(aDesc,aDesc+nDescSize,aDesc,[&dL1Norm](TVal tVal){
            if(bNegativeCompat && double(tVal)<0.0)
                return (TVal)-std::sqrt(-tVal/dL1Norm);
            else
                return (TVal)std::sqrt(tVal/dL1Norm);
        });
        if(bUseL2Norm) {
            const double dL2Norm = std::sqrt(std::accumulate(aDesc,aDesc+nDescSize,0.0,[](double dSum, TVal tVal){
                return dSum+(tVal*tVal);
            })) + DBL_EPSILON;
            std::transform(aDesc,aDesc+nDescSize,aDesc,[&dL2Norm](TVal tVal){
                return tVal/dL2Norm;
            });
        }
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
        typedef decltype(lv::L1dist<Tx>((Tx)0,(Tx)0)) DistType;
        DistType (&rlDist)(Tx,Tx) = lv::L1dist<Tx>;
        for(size_t i=0; i<nReqSize; ++i) {
            if(vXReq[i]>=vX.front() && vXReq[i]<=vX.back()) {
                const size_t nNNIdx = lv::find_nn_index(vXReq[i],vX,rlDist);
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

    /// helper global constexpr variable for endianness testing
    constexpr union {uint32_t i;uint8_t c[4];} s_oEndianessTest = {0x01020304};

    /// returns whether the machine uses a big endian architecture or not
    constexpr bool is_big_endian() {
        return s_oEndianessTest.c[0]==1;
    }

    /// bitfield linear unpacking function w/ almost linear scaling & 0-0/max-max mapping
    template<typename T>
    constexpr T extend_bits(T nValue, int32_t nOrigBitCount, int32_t nExtendedBitCount) {
        static_assert(std::is_unsigned<T>::value,"input value type must be unsigned integer");
        return (nValue<<(nExtendedBitCount-nOrigBitCount))|(nValue>>std::max(nOrigBitCount-(nExtendedBitCount-nOrigBitCount),0));
    }

    /// bitfield expansion function (specialized for unit word size)
    template<size_t nWordBitSize, typename T>
    constexpr std::enable_if_t<nWordBitSize==1,T> expand_bits(const T& nBits, size_t=0) {
        return nBits;
    }

    /// bitfield expansion function (specialized for non-unit word size)
    template<size_t nWordBitSize, typename T>
    constexpr std::enable_if_t<(nWordBitSize>1),T> expand_bits(const T& nBits, size_t n=((sizeof(T)*8)/nWordBitSize)-1) {
        static_assert(std::is_integral<T>::value,"input value type must be integer");
        // only the first [(sizeof(T)*8)/nWordBitSize] bits are kept (otherwise overflow/ignored)
        return (T)(bool((nBits&(1<<n))!=0)<<(n*nWordBitSize)) + ((n>=1)?expand_bits<nWordBitSize,T>(nBits,n-1):(T)0);
    }

} // namespace lv