
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
#include "litiv/utils/ParallelUtils.hpp"
#include "litiv/utils/DistanceUtils.hpp"
#include "litiv/utils/CxxUtils.hpp"

/*!
    Local Binary Similarity Pattern (LBSP) feature extractor

    Note 1: both grayscale and RGB/BGR images may be used with this extractor.
    Note 2: using LBSP::compute2(...) is logically equivalent to using LBSP::compute(...) followed by LBSP::reshapeDesc(...).

    For more details on the different parameters, see G.-A. Bilodeau et al, "Change Detection in Feature Space Using Local
    Binary Similarity Patterns", in CRV 2013.
 */
class LBSP : public cv::Feature2D {
public:
    //! constructor 1, threshold = absolute intensity 'similarity' threshold used when computing comparisons
    LBSP(size_t nThreshold);
    //! constructor 2, threshold = relative intensity 'similarity' threshold used when computing comparisons
    LBSP(float fRelThreshold, size_t nThresholdOffset=0);
    //! default destructor
    virtual ~LBSP();
    //! loads extractor params from the specified file node @@@@ not impl
    virtual void read(const cv::FileNode&);
    //! writes extractor params to the specified file storage @@@@ not impl
    virtual void write(cv::FileStorage&) const;
    //! sets the 'reference' image to be used for inter-frame comparisons (note: if no image is set or if the image is empty, the algorithm will default back to intra-frame comparisons)
    virtual void setReference(const cv::Mat&);
    //! returns the current descriptor size, in bytes
    virtual int descriptorSize() const;
    //! returns the current descriptor data type
    virtual int descriptorType() const;
    //! returns whether this extractor is using a relative threshold or not
    virtual bool isUsingRelThreshold() const;
    //! returns the current relative threshold used for comparisons (-1 = invalid/not used)
    virtual float getRelThreshold() const;
    //! returns the current absolute threshold used for comparisons (-1 = invalid/not used)
    virtual size_t getAbsThreshold() const;

    //! similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix
    void compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;
    //! batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat>& voDescCollection) const;

    //! utility function, used to reshape a descriptors matrix to its input image size via their keypoint locations
    static void reshapeDesc(cv::Size oSize, const std::vector<cv::KeyPoint>& voKeypoints, const cv::Mat& oDescriptors, cv::Mat& oOutput);
    //! utility function, used to illustrate the difference between two descriptor images
    static void calcDescImgDiff(const cv::Mat& oDesc1, const cv::Mat& oDesc2, cv::Mat& oOutput, bool bForceMergeChannels=false);
    //! utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
    static void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize);
    //! utility function, used to filter out bad pixels in a ROI that would trigger out of bounds error because they're too close to the image border
    static void validateROI(cv::Mat& oROI);
    //! utility, specifies the pixel size of the pattern used (width and height)
    static const size_t PATCH_SIZE = 5;
    //! utility, specifies the number of bytes per descriptor (should be the same as calling 'descriptorSize()')
    static const size_t DESC_SIZE = 2;
    //! utility, specifies the number of bits per descriptor
    static const size_t DESC_SIZE_BITS = DESC_SIZE*8;
#if HAVE_GLSL
    //! utility function, returns the glsl source code required to describe an LBSP descriptor based on the image load store
    static std::string getShaderFunctionSource(size_t nChannels, bool bUseSharedDataPreload, const glm::uvec2& vWorkGroupSize);
#endif //HAVE_GLSL

//! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels, typename Tt, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const uchar* const _ref, const int _x, const int _y, const std::array<Tt,nChannels>& _t, std::array<Tr,nChannels>& _res) {
        LBSP::computeDescriptor(oInputImg,_ref,_x,_y,_t,_res.data());
    }
    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels, typename Tt, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const std::array<uchar,nChannels>& _ref, const int _x, const int _y, const std::array<Tt,nChannels>& _t, std::array<Tr,nChannels>& _res) {
        LBSP::computeDescriptor(oInputImg,_ref.data(),_x,_y,_t,_res.data());
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup, single-channel array thresholding)
    template<size_t nChannels, typename Tt, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const uchar _ref, const int _x, const int _y, const size_t _c, const Tt _t, Tr& _res) {
        alignas(16) std::array<uchar,LBSP::DESC_SIZE_BITS> _anVals;
        computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,_c,_anVals);
        computeDescriptor_threshold(_anVals,_ref,_t,_res);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels, typename Tt, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const uchar* const _ref, const int _x, const int _y, const std::array<Tt,nChannels>& _t, Tr _res) {
        alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,nChannels> _aanVals;
        computeDescriptor_lookup(oInputImg,_x,_y,_aanVals);
        for(size_t _c=0; _c<nChannels; ++_c)
            computeDescriptor_threshold(_aanVals[_c],_ref[_c],_t[_c],_res[_c]);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _c, uchar* _anVals) {
        static_assert(nChannels>0,"need at least one image channel");
        CV_DbgAssert(_anVals);
        CV_DbgAssert(!oInputImg.empty());
        CV_DbgAssert(oInputImg.type()==CV_8UC(nChannels) && _c<nChannels);
        CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
        CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
        const size_t _step_row = oInputImg.step.p[0];
        const uchar* const _data = oInputImg.data;
        LBSP::lookup_16bits_dbcross<nChannels>(_data,_x,_y,_c,_step_row,_anVals);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _c, std::array<uchar,DESC_SIZE_BITS>& _anVals) {
        static_assert(nChannels>0,"need at least one image channel");
        CV_DbgAssert(!oInputImg.empty());
        CV_DbgAssert(oInputImg.type()==CV_8UC(nChannels) && _c<nChannels);
        CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
        CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
        const size_t _step_row = oInputImg.step.p[0];
        const uchar* const _data = oInputImg.data;
        LBSP::lookup_16bits_dbcross<nChannels>(_data,_x,_y,_c,_step_row,_anVals);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, uchar* _aanVals) {
        static_assert(nChannels>0,"need at least one image channel");
        CV_DbgAssert(_aanVals);
        CV_DbgAssert(!oInputImg.empty());
        CV_DbgAssert(oInputImg.type()==CV_8UC(nChannels));
        CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
        CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
        const size_t _step_row = oInputImg.step.p[0];
        const uchar* const _data = oInputImg.data;
        for(size_t _c=0; _c<nChannels; ++_c) {
            uchar* _anVals = _aanVals+_c*LBSP::DESC_SIZE_BITS;
            LBSP::lookup_16bits_dbcross<nChannels>(_data,_x,_y,_c,_step_row,_anVals);
        }
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>& _aanVals) {
        static_assert(nChannels>0,"need at least one image channel");
        CV_DbgAssert(!oInputImg.empty());
        CV_DbgAssert(oInputImg.type()==CV_8UC(nChannels));
        CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
        CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
        const size_t _step_row = oInputImg.step.p[0];
        const uchar* const _data = oInputImg.data;
        for(size_t _c=0; _c<nChannels; ++_c) {
            std::array<uchar,LBSP::DESC_SIZE_BITS>& _anVals = _aanVals[_c];
            LBSP::lookup_16bits_dbcross<nChannels>(_data,_x,_y,_c,_step_row,_anVals);
        }
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (array thresholding only)
    template<typename Tv, typename Tt, typename Tr>
    inline static void computeDescriptor_threshold(const Tv& _anVals, const uchar _ref, const Tt _t, Tr& _anResVals) {
        static_assert(std::is_integral<Tt>::value,"internal threshold type must be integral");
        static_assert(sizeof(Tr)>=LBSP::DESC_SIZE,"result type size is too small");
        LBSP::threshold_internal(_anVals,_t,_ref,_anResVals);
    }

protected:
    //! classic 'compute' implementation, based on the regular DescriptorExtractor::computeImpl arguments & expected output
    virtual void computeImpl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;
    const bool m_bOnlyUsingAbsThreshold;
    const float m_fRelThreshold;
    const size_t m_nThreshold;
    cv::Mat m_oRefImage;

    template<size_t nChannels, typename Tv, typename Tr>
    inline static void lookup_16bits_dbcross(const Tv& _data, const int _x, const int _y, const size_t _c, const size_t _step_row, Tr& _anVals) {
        // note: this is the LBSP 16 bit double-cross indiv RGB/RGBA pattern as used in the original article by G.-A. Bilodeau et al.
        //
        //  O   O   O          4 ..  3 ..  6
        //    O O O           .. 15  8 13 ..
        //  O O X O O    =>    0  9  X 11  1
        //    O O O           .. 12 10 14 ..
        //  O   O   O          7 ..  2 ..  5
        //
        const int _anIdxLUT[16][2] = {
            {-2, 0},{ 2, 0},{ 0,-2},{ 0, 2},
            {-2, 2},{ 2,-2},{ 2, 2},{-2,-2},
            { 0, 1},{-1, 0},{ 0,-1},{ 1, 0},
            {-1,-1},{ 1, 1},{ 1,-1},{-1, 1},
        };
        auto _idx = [&](int n) {
            return _step_row*(_anIdxLUT[n][1]+_y)+nChannels*(_anIdxLUT[n][0]+_x)+_c;
        };
        auto _f = [&](int n) {
            _anVals[n] = _data[_idx(n)];
        };
        CxxUtils::unroll<15>(_f);
    }

    template<typename Tv, typename Tt, typename Tr>
    inline static void threshold_internal(const Tv& _anVals, const Tt _t, const uchar _ref, Tr& _anResVal) {
        // note: this function is used to threshold an LBSP pattern based on a predefined lookup array (see LBSP_16bits_dbcross_lookup for more information).
        // @@@ todo: use array template to unroll loops & allow any descriptor size here
#if (!HAVE_SSE4_1 && !HAVE_SSE2)
        _anResVal = 0;
        auto _f = [&](int n) {
            _anResVal |= (DistanceUtils::L1dist(_anVals[n],_ref) > _t) << n;
        };
        CxxUtils::unroll<LBSP::DESC_SIZE_BITS-1>(_f);
#else //(HAVE_SSE4_1 || HAVE_SSE2)
        static_assert(LBSP::DESC_SIZE_BITS==16,"Current sse impl can only manage 16-byte chunks");
        // @@@@ send <16byte back to non-sse, and >16 to loop via template enableif?
        CV_DbgAssert(((uintptr_t)(&_anVals[0])&15)==0);
        __m128i _anMMXVals = _mm_load_si128((__m128i*)&_anVals[0]);
        __m128i _anRefVals = _mm_set1_epi8(_ref);
#if HAVE_SSE4_1
        __m128i _anDistVals = _mm_sub_epi8(_mm_max_epu8(_anMMXVals,_anRefVals),_mm_min_epu8(_anMMXVals,_anRefVals));
        __m128i _abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_mm_set1_epi8(uchar(0x80))),_mm_set1_epi8(uchar(_t^0x80)));
#else //HAVE_SSE2
        __m128i _anBitFlipper = _mm_set1_epi8(char(0x80));
        __m128i _anDistVals = _mm_xor_si128(_anMMXVals,_anBitFlipper);
        _anRefVals = _mm_xor_si128(_anRefVals,_anBitFlipper);
        __m128i _abCmpRes = _mm_cmpgt_epi8(_anDistVals,_anRefVals);
        __m128i _anDistVals1 = _mm_sub_epi8(_anDistVals,_anRefVals);
        __m128i _anDistVals2 = _mm_sub_epi8(_anRefVals,_anDistVals);
        _anDistVals = _mm_or_si128(_mm_and_si128(_abCmpRes,_anDistVals1),_mm_andnot_si128(_abCmpRes,_anDistVals2));
        _abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_anBitFlipper),_mm_set1_epi8(uchar(_t^0x80)));
#endif //HAVE_SSE20x80u
        _anResVal = (Tr)_mm_movemask_epi8(_abCmpRes);
#endif //(HAVE_SSE4_1 || HAVE_SSE2)
    }
};
