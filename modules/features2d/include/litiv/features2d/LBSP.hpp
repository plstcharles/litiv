
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

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "litiv/utils/distances.hpp"
/**
    Local Binary Similarity Pattern (LBSP) feature extractor

    Note 1: both grayscale and RGB/BGR images may be used with this extractor.
    Note 2: using LBSP::compute2(...) is logically equivalent to using LBSP::compute(...) followed by LBSP::reshapeDesc(...).

    For more details on the different parameters, see G.-A. Bilodeau et al, "Change Detection in Feature Space Using Local
    Binary Similarity Patterns", in CRV 2013.
*/
class LBSP : public cv::Feature2D {
public:
    /// constructor 1, threshold = absolute intensity 'similarity' threshold used when computing comparisons
    LBSP(size_t nThreshold);
    /// constructor 2, threshold = relative intensity 'similarity' threshold used when computing comparisons
    LBSP(float fRelThreshold, size_t nThresholdOffset=0);
    /// default destructor
    virtual ~LBSP();
    /// loads extractor params from the specified file node @@@@ not impl
    virtual void read(const cv::FileNode&);
    /// writes extractor params to the specified file storage @@@@ not impl
    virtual void write(cv::FileStorage&) const;
    /// sets the 'reference' image to be used for inter-frame comparisons (note: if no image is set or if the image is empty, the algorithm will default back to intra-frame comparisons)
    virtual void setReference(const cv::Mat&);
    /// returns the current descriptor size, in bytes
    virtual int descriptorSize() const;
    /// returns the current descriptor data type
    virtual int descriptorType() const;
    /// returns whether this extractor is using a relative threshold or not
    virtual bool isUsingRelThreshold() const;
    /// returns the current relative threshold used for comparisons (-1 = invalid/not used)
    virtual float getRelThreshold() const;
    /// returns the current absolute threshold used for comparisons (-1 = invalid/not used)
    virtual size_t getAbsThreshold() const;

    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix
    void compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;
    /// batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat>& voDescCollection) const;

    /// utility function, used to reshape a descriptors matrix to its input image size via their keypoint locations
    static void reshapeDesc(cv::Size oSize, const std::vector<cv::KeyPoint>& voKeypoints, const cv::Mat& oDescriptors, cv::Mat& oOutput);
    /// utility function, used to illustrate the difference between two descriptor images
    static void calcDescImgDiff(const cv::Mat& oDesc1, const cv::Mat& oDesc2, cv::Mat& oOutput, bool bForceMergeChannels=false);
    /// utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
    static void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize);
    /// utility function, used to filter out bad pixels in a ROI that would trigger out of bounds error because they're too close to the image border
    static void validateROI(cv::Mat& oROI);
#if HAVE_GLSL
    /// utility function, returns the glsl source code required to describe an LBSP descriptor based on the image load store
    static std::string getShaderFunctionSource(size_t nChannels, bool bUseSharedDataPreload, const glm::uvec2& vWorkGroupSize);
#endif //HAVE_GLSL

    /// utility, specifies the integer type used to store descriptors
    typedef ushort desc_t;
    /// utility, specifies the pixel size of the pattern used (width and height)
    static constexpr size_t PATCH_SIZE = 5;
    /// utility, specifies the number of bytes per descriptor (should be the same as calling 'descriptorSize()')
    static constexpr size_t DESC_SIZE = 2;
    /// utility, specifies the number of bits per descriptor
    static constexpr size_t DESC_SIZE_BITS = DESC_SIZE*8;
    /// utility, specifies the maximum gradient magnitude value that can be returned by computeDescriptor_gradient
    static constexpr size_t MAX_GRAD_MAG = DESC_SIZE_BITS;

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup, single-channel array thresholding)
    template<size_t nChannels>
    static inline void computeDescriptor(const cv::Mat& oInputImg, const uchar nRef, const int _x, const int _y, const size_t _c, const uchar nThreshold, desc_t& nDesc) {
        alignas(16) std::array<uchar,LBSP::DESC_SIZE_BITS> anVals;
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,_c,anVals);
        nDesc = LBSP::computeDescriptor_threshold(anVals.data(),nRef,nThreshold);
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels>
    static inline void computeDescriptor(const cv::Mat& oInputImg, const std::array<uchar,nChannels>& anRefs, const int _x, const int _y, const std::array<uchar,nChannels>& anThresholds, std::array<desc_t,nChannels>& anDesc) {
        LBSP::computeDescriptor<nChannels>(oInputImg,anRefs.data(),_x,_y,anThresholds.data(),anDesc.data());
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels>
    static inline void computeDescriptor(const cv::Mat& oInputImg, const uchar* const anRefs, const int _x, const int _y, const std::array<uchar,nChannels>& anThresholds, std::array<desc_t,nChannels>& anDesc) {
        LBSP::computeDescriptor<nChannels>(oInputImg,anRefs,_x,_y,anThresholds.data(),anDesc.data());
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels>
    static inline void computeDescriptor(const cv::Mat& oInputImg, const uchar* const anRefs, const int _x, const int _y, const std::array<uchar,nChannels>& anThresholds, desc_t* anDesc) {
        LBSP::computeDescriptor<nChannels>(oInputImg,anRefs,_x,_y,anThresholds.data(),anDesc);
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels>
    static inline void computeDescriptor(const cv::Mat& oInputImg, const uchar* const anRefs, const int _x, const int _y, const uchar* const anThresholds, desc_t* anDesc) {
        alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,nChannels> aanVals;
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,aanVals);
        lv::unroll<nChannels>([&](int _c) {
            anDesc[_c] = LBSP::computeDescriptor_threshold(aanVals[_c].data(),anRefs[_c],anThresholds[_c]);
        });
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup only)
    template<size_t nChannels>
    static inline void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _c, std::array<uchar,DESC_SIZE_BITS>& anVals) {
        static_assert(sizeof(std::array<uchar,DESC_SIZE_BITS>)==sizeof(uchar)*DESC_SIZE_BITS,"terrible impl of std::array right here");
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,_c,anVals.data());
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup only)
    template<size_t nChannels>
    static inline void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>& aanVals) {
        static_assert(sizeof(std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>)==sizeof(uchar)*DESC_SIZE_BITS*nChannels,"terrible impl of std::array right here");
        lvDbgAssert_((void*)aanVals.data()==(void*)aanVals[0].data(),"bad indexing in array-of-array impl");
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,aanVals[0].data());
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup only)
    template<size_t nChannels>
    static inline void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _c, uchar* anVals) {
        static_assert(nChannels>0,"need at least one image channel");
        lvDbgAssert_(anVals,"need to provide a valid pixel pointer");
        lvDbgAssert__(!oInputImg.empty() && oInputImg.type()==CV_8UC(nChannels) && _c<nChannels,"need to provide a non-empty matrix of %d channels, with _c<%d",(int)nChannels,(int)nChannels);
        lvDbgAssert__(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2,"descriptor center needs to be at least %d pixels from image borders",(int)LBSP::PATCH_SIZE/2);
        lvDbgAssert__(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2,"descriptor center needs to be at least %d pixels from image borders",(int)LBSP::PATCH_SIZE/2);
        const size_t nRowStep = oInputImg.step.p[0];
        const size_t nColStep = oInputImg.step.p[1];
        const uchar* const anData = oInputImg.data+_y*nRowStep+_x*nColStep+_c;
        LBSP::lookup_16bits_dbcross<nChannels>(anData,nRowStep,nColStep,anVals);
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup only)
    template<size_t nChannels>
    static inline void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, uchar* aanVals) {
        static_assert(nChannels>0,"need at least one image channel");
        lvDbgAssert_(aanVals,"need to provide a valid pixel pointer");
        lvDbgAssert__(!oInputImg.empty() && oInputImg.type()==CV_8UC(nChannels),"need to provide a non-empty matrix of %d channels",(int)nChannels);
        lvDbgAssert__(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2,"descriptor center needs to be at least %d pixels from image borders",(int)LBSP::PATCH_SIZE/2);
        lvDbgAssert__(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2,"descriptor center needs to be at least %d pixels from image borders",(int)LBSP::PATCH_SIZE/2);
        const size_t nRowStep = oInputImg.step.p[0];
        const size_t nColStep = oInputImg.step.p[1];
        lv::unroll<nChannels>([&](int _c) {
            const uchar* const anData = oInputImg.data+_y*nRowStep+_x*nColStep+_c;
            LBSP::lookup_16bits_dbcross<nChannels>(anData,nRowStep,nColStep,aanVals+_c*LBSP::DESC_SIZE_BITS);
        });
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (array thresholding only)
    static inline desc_t computeDescriptor_threshold(const std::array<uchar,LBSP::DESC_SIZE_BITS>& anVals, const uchar nRef, const uchar nThreshold) {
        static_assert(sizeof(std::array<uchar,LBSP::DESC_SIZE_BITS>)==sizeof(uchar)*LBSP::DESC_SIZE_BITS,"terrible impl of std::array right here");
        return LBSP::computeDescriptor_threshold(anVals.data(),nRef,nThreshold);
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (array thresholding only)
    static inline desc_t computeDescriptor_threshold(const uchar* const anVals, const uchar nRef, const uchar nThreshold) {
        // note: this function is used to threshold an LBSP pattern based on a predefined lookup array (see LBSP_16bits_dbcross_lookup for more information)
        // @@@ todo: use array template to unroll loops & allow any descriptor size here
        lvDbgAssert_(anVals,"need to provide a valid pixel pointer");
#if (!HAVE_SSE4_1 && !HAVE_SSE2)
        desc_t nDesc = 0;
        lv::unroll<LBSP::DESC_SIZE_BITS>([&](int n) {
            nDesc |= (lv::L1dist(anVals[n],nRef) > nThreshold) << n;
        });
        return nDesc;
#else //(HAVE_SSE4_1 || HAVE_SSE2)
        static_assert(LBSP::DESC_SIZE_BITS==16,"current sse impl can only manage 16-byte chunks");
        // @@@@ send <16byte back to non-sse, and >16 to loop via template enableif?
        lvDbgAssert_(((uintptr_t)(&anVals[0])&15)==0,"pixel pointer must be 16-byte aligned");
        __m128i _anInputVals = _mm_load_si128((__m128i*)&anVals[0]); // @@@@@ load? or just cast?
        __m128i _anRefVals = _mm_set1_epi8(nRef);
#if HAVE_SSE4_1
        __m128i _anDistVals = _mm_sub_epi8(_mm_max_epu8(_anInputVals,_anRefVals),_mm_min_epu8(_anInputVals,_anRefVals));
        __m128i _abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_mm_set1_epi8(uchar(0x80))),_mm_set1_epi8(uchar(nThreshold^0x80)));
#else //HAVE_SSE2
        __m128i _anBitFlipper = _mm_set1_epi8(uchar(0x80));
        __m128i _anDistVals = _mm_xor_si128(_anInputVals,_anBitFlipper);
        _anRefVals = _mm_xor_si128(_anRefVals,_anBitFlipper);
        __m128i _abCmpRes = _mm_cmpgt_epi8(_anDistVals,_anRefVals);
        __m128i _anDistVals1 = _mm_sub_epi8(_anDistVals,_anRefVals);
        __m128i _anDistVals2 = _mm_sub_epi8(_anRefVals,_anDistVals);
        _anDistVals = _mm_or_si128(_mm_and_si128(_abCmpRes,_anDistVals1),_mm_andnot_si128(_abCmpRes,_anDistVals2));
        _abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_anBitFlipper),_mm_set1_epi8(uchar(nThreshold^0x80)));
#endif //HAVE_SSE2
        return (desc_t)_mm_movemask_epi8(_abCmpRes);
#endif //(HAVE_SSE4_1 || HAVE_SSE2)
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP gradient computation function (mixes rel+abs, returns max-channel only)
    template<size_t nChannels, size_t nAbsOffset=20, size_t nRelShift=2, typename Tr1=int, typename Tr2=uint>
    static inline void computeDescriptor_gradient(const std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>& aanVals, const std::array<uchar,nChannels>& anRefs, Tr1& nGradX, Tr1& nGradY, Tr2& nGradMag) {
        static_assert(sizeof(std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>)==sizeof(uchar)*DESC_SIZE_BITS*nChannels,"terrible impl of std::array right here");
        static_assert(sizeof(std::array<uchar,nChannels>)==sizeof(uchar)*nChannels,"terrible impl of std::array right here");
        LBSP::computeDescriptor_gradient<nChannels,nAbsOffset,nRelShift>(aanVals.data(),anRefs.data(),nGradX,nGradY,nGradMag);
    }

    /// utility function, shortcut/lightweight/direct single-point LBSP gradient estimation function (mixes rel+abs, returns max-channel only)
    template<size_t nChannels, size_t nAbsOffset=20, size_t nRelShift=2, typename Tr1=char, typename Tr2=uchar>
    static inline void computeDescriptor_gradient(const uchar* const aanVals, const uchar* const anRefs, Tr1& nGradX, Tr1& nGradY, Tr2& nGradMag) {
        // note: this function is used to threshold a multi-channel LBSP pattern based on a predefined lookup array (see LBSP_16bits_dbcross_lookup for more information)
        // @@@ todo: use array template to unroll loops & allow any descriptor size here
        static_assert(std::numeric_limits<Tr1>::max()>=4*LBSP::DESC_SIZE_BITS,"output size is too small for descriptor config");
        static_assert(nChannels>0,"need at least one image channel");
        lvDbgAssert_(aanVals,"need to provide a valid pixel pointer");
        lvDbgAssert_(anRefs,"need to provide a valid ref pixel pointer");
        desc_t nTempDesc = computeDescriptor_threshold(aanVals+(nChannels-1)*LBSP::DESC_SIZE_BITS,anRefs[nChannels-1],((anRefs[nChannels-1]>>nRelShift)+nAbsOffset)/2);
        nGradMag = (Tr2)lv::popcount(nTempDesc);
        lv::unroll<nChannels-1>([&](int cn) {
            desc_t nNewTempDesc = computeDescriptor_threshold(aanVals+cn*LBSP::DESC_SIZE_BITS,anRefs[cn],((anRefs[cn]>>nRelShift)+nAbsOffset)/2);
            const Tr2 nNewGradMag = (Tr2)lv::popcount(nNewTempDesc);
            if(nGradMag<nNewGradMag) {
                nGradMag = nNewGradMag;
                nTempDesc = nNewTempDesc;
            }
        });
        nGradX = (Tr1)lv::popcount(nTempDesc&s_nDesc_16bitdbcross_GradX_Pos)-(Tr1)lv::popcount(nTempDesc&s_nDesc_16bitdbcross_GradX_Neg);
        nGradY = (Tr1)lv::popcount(nTempDesc&s_nDesc_16bitdbcross_GradY_Pos)-(Tr1)lv::popcount(nTempDesc&s_nDesc_16bitdbcross_GradY_Neg);
        lvDbgAssert(nGradMag<=MAX_GRAD_MAG);
    }

protected:
    /// classic 'compute' implementation, based on the regular DescriptorExtractor::computeImpl arguments & expected output
    virtual void computeImpl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;
    const bool m_bOnlyUsingAbsThreshold;
    const float m_fRelThreshold;
    const size_t m_nThreshold;
    cv::Mat m_oRefImage;

    // arrays below do not rely on std::array to avoid multi-dim init problems w/ static constexpr in header files

    /// LBSP 16 bit double-cross indiv RGB/RGBA pattern as used in the original article by G.-A. Bilodeau et al.
    static constexpr int s_anIdxLUT_16bitdbcross[16][2] = {
            //  O   O   O        4 ..  3 ..  6
            //    O O O         .. 15  8 13 ..
            //  O O X O O   =>   0  9  X 11  1
            //    O O O         .. 12 10 14 ..
            //  O   O   O        7 ..  2 ..  5
            {-2, 0}, { 2, 0}, { 0,-2}, { 0, 2},
            {-2, 2}, { 2,-2}, { 2, 2}, {-2,-2},
            { 0, 1}, {-1, 0}, { 0,-1}, { 1, 0},
            {-1,-1}, { 1, 1}, { 1,-1}, {-1, 1},
    };//                                                   # { 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15};
    static constexpr int s_anIdxLUT_16bitdbcross_GradX[16] = { 1,-1, 0, 0, 1,-1,-1, 1, 0, 1, 0,-1, 1,-1,-1, 1};
    static constexpr int s_anIdxLUT_16bitdbcross_GradY[16] = { 0, 0,-1, 1, 1,-1, 1,-1, 1, 0,-1, 0,-1, 1,-1, 1};
    static constexpr desc_t s_nDesc_16bitdbcross_GradX_Pos = ((1<<0)+(1<<4)+(1<<7)+(1<<9)+(1<<12)+(1<<15));
    static constexpr desc_t s_nDesc_16bitdbcross_GradX_Neg = ((1<<1)+(1<<5)+(1<<6)+(1<<11)+(1<<13)+(1<<14));
    static constexpr desc_t s_nDesc_16bitdbcross_GradY_Pos = ((1<<3)+(1<<4)+(1<<6)+(1<<8)+(1<<13)+(1<<15));
    static constexpr desc_t s_nDesc_16bitdbcross_GradY_Neg = ((1<<2)+(1<<5)+(1<<7)+(1<<10)+(1<<12)+(1<<14));
#if HAVE_SSE2
    static constexpr union IdxLUTOffsetArray {int anOffsets[16]; __m128i vnOffsets[4];}
#else //(!HAVE_SSE2)
    static constexpr union IdxLUTOffsetArray {int anOffsets[16];}
#endif //(!HAVE_SSE2)
            s_oIdxLUT_16bitdbcross_x = {{-2, 2, 0, 0,  -2, 2, 2,-2,   0,-1, 0, 1,  -1, 1, 1,-1}},
            s_oIdxLUT_16bitdbcross_y = {{ 0, 0,-2, 2,   2,-2, 2,-2,   1, 0,-1, 0,  -1, 1,-1, 1}};

    template<size_t nChannels, typename Tv>
    static inline void lookup_16bits_dbcross(const Tv* const anData, const size_t nRowStep, const size_t nColStep, Tv* const anVals) {
/*#if !HAVE_SSE2
        // SSE2 version is not faster than fully optimized version
        lvDbgAssert(nRowStep*2+nColStep*2<INT32_MAX); // map index offsets will be stored in signed 32-bit integers
        const __m128i vnColStep = _mm_set1_epi32((int)nColStep);
        const __m128i vnRowStep = _mm_set1_epi32((int)nRowStep);
        lv::unroll<4>([&](int n) {
            __m128i vnMapOffsets = _mm_add_epi32(lv::mult_32si(vnColStep,s_oIdxLUT_16bitdbcross_x.vnOffsets[n]),lv::mult_32si(vnRowStep,s_oIdxLUT_16bitdbcross_y.vnOffsets[n]));
            anVals[n*4+3] = anData[lv::extract_32si<3>(vnMapOffsets)];
            anVals[n*4+2] = anData[lv::extract_32si<2>(vnMapOffsets)];
            anVals[n*4+1] = anData[lv::extract_32si<1>(vnMapOffsets)];
            anVals[n*4+0] = anData[lv::extract_32si<0>(vnMapOffsets)];
        });
#else //(!HAVE_SSE2)*/
        lv::unroll<16>([&](int n) {
            anVals[n] = anData[nRowStep*s_oIdxLUT_16bitdbcross_y.anOffsets[n]+nColStep*s_oIdxLUT_16bitdbcross_x.anOffsets[n]];
        });
//#endif //(!HAVE_SSE2)
    }
};
