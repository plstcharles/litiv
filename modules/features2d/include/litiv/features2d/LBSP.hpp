
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
    static constexpr size_t PATCH_SIZE = 5;
    //! utility, specifies the number of bytes per descriptor (should be the same as calling 'descriptorSize()')
    static constexpr size_t DESC_SIZE = 2;
    //! utility, specifies the number of bits per descriptor
    static constexpr size_t DESC_SIZE_BITS = DESC_SIZE*8;
    //! utility, specifies the maximum gradient magnitude value that can be returned by computeDescriptor_orientation
    static constexpr size_t MAX_GRAD_MAG = DESC_SIZE_BITS*2;
#if HAVE_GLSL
    //! utility function, returns the glsl source code required to describe an LBSP descriptor based on the image load store
    static std::string getShaderFunctionSource(size_t nChannels, bool bUseSharedDataPreload, const glm::uvec2& vWorkGroupSize);
#endif //HAVE_GLSL

    //! gradient orientations used as the output of the computeDescriptor_orientation function (same as Canny's)
    enum eGradientOrientation {
        //      2   4   8
        //       *  *  *
        //        * * *
        //      1*******1
        //        * * *
        //       *  *  *
        //      8   4   2
        eGradientOrientation_None=0,
        eGradientOrientation_Horizontal=1,
        eGradientOrientation_Diagonal=2,
        eGradientOrientation_Vertical=4,
        eGradientOrientation_DiagonalInv=8,
        eGradientOrientation_Point=0xF
    };

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup, single-channel array thresholding)
    template<size_t nChannels, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const uchar nRef, const int _x, const int _y, const size_t _c, const uchar nThreshold, Tr& nDesc) {
        alignas(16) std::array<uchar,LBSP::DESC_SIZE_BITS> anVals;
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,_c,anVals);
        LBSP::computeDescriptor_threshold(anVals.data(),nRef,nThreshold,nDesc);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const std::array<uchar,nChannels>& anRefs, const int _x, const int _y, const std::array<uchar,nChannels>& anThresholds, std::array<Tr,nChannels>& anDesc) {
        LBSP::computeDescriptor<nChannels>(oInputImg,anRefs.data(),_x,_y,anThresholds.data(),anDesc.data());
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const uchar* const anRefs, const int _x, const int _y, const std::array<uchar,nChannels>& anThresholds, std::array<Tr,nChannels>& anDesc) {
        LBSP::computeDescriptor<nChannels>(oInputImg,anRefs,_x,_y,anThresholds.data(),anDesc.data());
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const uchar* const anRefs, const int _x, const int _y, const std::array<uchar,nChannels>& anThresholds, Tr* anDesc) {
        LBSP::computeDescriptor<nChannels>(oInputImg,anRefs,_x,_y,anThresholds.data(),anDesc);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup, multi-channel array thresholding)
    template<size_t nChannels, typename Tr>
    inline static void computeDescriptor(const cv::Mat& oInputImg, const uchar* const anRefs, const int _x, const int _y, const uchar* const anThresholds, Tr* anDesc) {
        alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,nChannels> aanVals;
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,aanVals);
        CxxUtils::unroll<nChannels>([&](int _c) {
            LBSP::computeDescriptor_threshold(aanVals[_c].data(),anRefs[_c],anThresholds[_c],anDesc[_c]);
        });
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _c, std::array<uchar,DESC_SIZE_BITS>& anVals) {
        static_assert(sizeof(std::array<uchar,DESC_SIZE_BITS>)==sizeof(uchar)*DESC_SIZE_BITS,"terrible impl of std::array right here");
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,_c,anVals.data());
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>& aanVals) {
        static_assert(sizeof(std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>)==sizeof(uchar)*DESC_SIZE_BITS*nChannels,"terrible impl of std::array right here");
        CV_DbgAssert((void*)aanVals.data()==(void*)aanVals[0].data());
        LBSP::computeDescriptor_lookup<nChannels>(oInputImg,_x,_y,aanVals[0].data());
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (single-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, const size_t _c, uchar* anVals) {
        static_assert(nChannels>0,"need at least one image channel");
        CV_DbgAssert(anVals);
        CV_DbgAssert(!oInputImg.empty());
        CV_DbgAssert(oInputImg.type()==CV_8UC(nChannels) && _c<nChannels);
        CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
        CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
        const size_t _step_row = oInputImg.step.p[0];
        const uchar* const _data = oInputImg.data;
        LBSP::lookup_16bits_dbcross<nChannels>(_data,_x,_y,_c,_step_row,anVals);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (multi-channel lookup only)
    template<size_t nChannels>
    inline static void computeDescriptor_lookup(const cv::Mat& oInputImg, const int _x, const int _y, uchar* aanVals) {
        static_assert(nChannels>0,"need at least one image channel");
        CV_DbgAssert(aanVals);
        CV_DbgAssert(!oInputImg.empty());
        CV_DbgAssert(oInputImg.type()==CV_8UC(nChannels));
        CV_DbgAssert(_x>=(int)LBSP::PATCH_SIZE/2 && _y>=(int)LBSP::PATCH_SIZE/2);
        CV_DbgAssert(_x<oInputImg.cols-(int)LBSP::PATCH_SIZE/2 && _y<oInputImg.rows-(int)LBSP::PATCH_SIZE/2);
        const size_t _step_row = oInputImg.step.p[0];
        const uchar* const _data = oInputImg.data;
        CxxUtils::unroll<nChannels>([&](int _c) {
            LBSP::lookup_16bits_dbcross<nChannels>(_data,_x,_y,_c,_step_row,aanVals+_c*LBSP::DESC_SIZE_BITS);
        });
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP orientation estimation function (w/ optional bitmask gaps, useful for counting)
    template<typename To=LBSP::eGradientOrientation, size_t nBitMaskWordSize=1, typename Tr>
    inline static To computeDescriptor_orientation(const Tr& nDesc) {
        // simple example of counter init: To=ushort, nBitMaskWordSize=4 => 4x bit counters with [0,15] range
        static_assert(nBitMaskWordSize>0,"bit mask word size needs to be larger or equal to one");
        static_assert(nBitMaskWordSize*4<=sizeof(To)*8,"bit mask word size is too large for output type");
//#if (!HAVE_SSE4_1 && !HAVE_SSE2)
        const uint nBitCount = DistanceUtils::popcount(nDesc);
        if(nBitCount<=1)
            return (To)LBSP::eGradientOrientation_None;
        else if(nBitCount>=LBSP::DESC_SIZE_BITS-1) {
            constexpr To nRet = CxxUtils::expand_bits<nBitMaskWordSize>((To)LBSP::eGradientOrientation_Point);
            return nRet;
        }
        std::array<uint,4> anOrientMag = {0,0,0,0};
        CxxUtils::unroll<LBSP::DESC_SIZE_BITS/4>([&](int n) {
            anOrientMag[0] += bool(nDesc&s_anIdxLUT_16bitdbcross_Horiz_Bits[n]);
        });
        CxxUtils::unroll<LBSP::DESC_SIZE_BITS/4>([&](int n) {
            anOrientMag[1] += bool(nDesc&s_anIdxLUT_16bitdbcross_Diag_Bits[n]);
        });
        CxxUtils::unroll<LBSP::DESC_SIZE_BITS/4>([&](int n) {
            anOrientMag[2] += bool(nDesc&s_anIdxLUT_16bitdbcross_Vert_Bits[n]);
        });
        CxxUtils::unroll<LBSP::DESC_SIZE_BITS/4>([&](int n) {
            anOrientMag[3] += bool(nDesc&s_anIdxLUT_16bitdbcross_DiagInv_Bits[n]);
        });
        return (To)(1<<((std::max_element(anOrientMag.begin(),anOrientMag.end())-anOrientMag.begin())*nBitMaskWordSize));
//#else //(HAVE_SSE4_1 || HAVE_SSE2)
// @@@@ TODO
//#endif //(HAVE_SSE4_1 || HAVE_SSE2)
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (array thresholding only)
    template<typename Tr>
    inline static void computeDescriptor_threshold(const std::array<uchar,LBSP::DESC_SIZE_BITS>& anVals, const uchar nRef, const uchar nThreshold, Tr& nDesc) {
        static_assert(sizeof(std::array<uchar,LBSP::DESC_SIZE_BITS>)==sizeof(uchar)*LBSP::DESC_SIZE_BITS,"terrible impl of std::array right here");
        LBSP::computeDescriptor_threshold(anVals.data(),nRef,nThreshold,nDesc);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP computation function for extra flexibility (array thresholding only)
    template<typename Tr>
    inline static void computeDescriptor_threshold(const uchar* const anVals, const uchar nRef, const uchar nThreshold, Tr& nDesc) {
        // note: this function is used to threshold an LBSP pattern based on a predefined lookup array (see LBSP_16bits_dbcross_lookup for more information)
        // @@@ todo: use array template to unroll loops & allow any descriptor size here
        static_assert(sizeof(Tr)>=LBSP::DESC_SIZE,"output size is too small for descriptor config");
        CV_DbgAssert(anVals);
#if (!HAVE_SSE4_1 && !HAVE_SSE2)
        nDesc = 0;
        CxxUtils::unroll<LBSP::DESC_SIZE_BITS-1>([&](int n) {
            nDesc |= (DistanceUtils::L1dist(anVals[n],nRef) > nThreshold) << n;
        });
#else //(HAVE_SSE4_1 || HAVE_SSE2)
        static_assert(LBSP::DESC_SIZE_BITS==16,"current sse impl can only manage 16-byte chunks");
        // @@@@ send <16byte back to non-sse, and >16 to loop via template enableif?
        CV_DbgAssert(((uintptr_t)(&anVals[0])&15)==0);
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
        nDesc = (Tr)_mm_movemask_epi8(_abCmpRes);
#endif //(HAVE_SSE4_1 || HAVE_SSE2)
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP gradient computation function (mixes rel+abs, returns max-channel only)
    template<size_t nChannels, size_t nShift=1, typename Tr1=short, typename Tr2=ushort>
    inline static void computeDescriptor_gradient(const std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>& aanVals, const std::array<uchar,nChannels>& anRefs, const uchar nThreshold, Tr1& nGradX, Tr1& nGradY, Tr2& nGradMag) {
        static_assert(sizeof(std::array<std::array<uchar,DESC_SIZE_BITS>,nChannels>)==sizeof(uchar)*DESC_SIZE_BITS*nChannels,"terrible impl of std::array right here");
        static_assert(sizeof(std::array<uchar,nChannels>)==sizeof(uchar)*nChannels,"terrible impl of std::array right here");
        LBSP::computeDescriptor_gradient<nChannels,nShift>(aanVals.data(),anRefs.data(),nThreshold,nGradX,nGradY,nGradMag);
    }

    //! utility function, shortcut/lightweight/direct single-point LBSP gradient computation function (mixes rel+abs, returns max-channel only)
    template<size_t nChannels, size_t nShift=1, typename Tr1=short, typename Tr2=ushort>
    inline static void computeDescriptor_gradient(const uchar* const aanVals, const uchar* const anRefs, const uchar nThreshold, Tr1& nGradX, Tr1& nGradY, Tr2& nGradMag) {
        // note: this function is used to threshold a multi-channel LBSP pattern based on a predefined lookup array (see LBSP_16bits_dbcross_lookup for more information)
        // @@@ todo: use array template to unroll loops & allow any descriptor size here
        static_assert(std::numeric_limits<Tr1>::max()>=4*LBSP::DESC_SIZE_BITS,"output size is too small for descriptor config");
        static_assert(nChannels>0,"need at least one image channel");
        CV_DbgAssert(aanVals);
        CV_DbgAssert(anRefs);
//#if (!HAVE_SSE4_1 && !HAVE_SSE2)
        nGradX = 0;
        nGradY = 0;
        nGradMag = 0;
        CxxUtils::unroll<LBSP::DESC_SIZE_BITS>([&](int n) {
            const Tr2 nCurrDist = (Tr2)DistanceUtils::L1dist(aanVals[(nChannels-1)*LBSP::DESC_SIZE_BITS+n],anRefs[(nChannels-1)]);
            const Tr2 nCurrGradMag = (Tr2)bool(nCurrDist>anRefs[(nChannels-1)]>>nShift) + (Tr2)bool(nCurrDist>nThreshold);
            nGradX += (Tr1)(s_anIdxLUT_16bitdbcross_GradX[n]*nCurrGradMag); // range: [-4,4]x16
            nGradY += (Tr1)(s_anIdxLUT_16bitdbcross_GradY[n]*nCurrGradMag);
            //nGradMag += nCurrGradMag*nCurrGradMag; // range: [0-4]x16
            //nGradMag += nCurrGradMag*nCurrGradMag*(std::abs(s_anIdxLUT_16bitdbcross_GradX[n])+std::abs(s_anIdxLUT_16bitdbcross_GradY[n])); // range: [0-4]x16
            nGradMag += nCurrGradMag; // range: [0-2]x16
        });
        //nGradMag = (Tr2)std::abs(nGradX)+(Tr2)std::abs(nGradY); // range: [0,8]x16
        //nGradMag = (Tr2)(int(nGradX)*nGradX+int(nGradY)*nGradY); // range: [0,16]x16
        CxxUtils::unroll<nChannels-1>([&](int cn) {
            Tr1 nNewGradX = 0;
            Tr1 nNewGradY = 0;
            Tr2 nNewGradMag = 0;
            CxxUtils::unroll<LBSP::DESC_SIZE_BITS>([&](int n) {
                const Tr2 nCurrDist = (Tr2)DistanceUtils::L1dist(aanVals[cn*LBSP::DESC_SIZE_BITS+n],anRefs[cn]);
                const Tr2 nCurrGradMag = (Tr2)bool(nCurrDist>anRefs[cn]>>nShift) + (Tr2)bool(nCurrDist>nThreshold);
                nNewGradX += (Tr1)(s_anIdxLUT_16bitdbcross_GradX[n]*nCurrGradMag);
                nNewGradY += (Tr1)(s_anIdxLUT_16bitdbcross_GradY[n]*nCurrGradMag);
                //nNewGradMag += nCurrGradMag*nCurrGradMag;
                //nNewGradMag += nCurrGradMag*nCurrGradMag*(std::abs(s_anIdxLUT_16bitdbcross_GradX[n])+std::abs(s_anIdxLUT_16bitdbcross_GradY[n])); // range: [0-4]x16
                nNewGradMag += nCurrGradMag;
            });
            //const Tr2 nNewGradMag = (Tr2)std::abs(nNewGradX)+(Tr2)std::abs(nNewGradY);
            //const Tr2 nNewGradMag = (Tr2)(int(nNewGradX)*nNewGradX+int(nNewGradY)*nNewGradY); // range: [0,16]x16
            if(nGradMag<nNewGradMag) {
                nGradX = nNewGradX;
                nGradY = nNewGradY;
                nGradMag = nNewGradMag;
            }
            CV_DbgAssert(nGradMag<=MAX_GRAD_MAG);
        });
//#else //(HAVE_SSE4_1 || HAVE_SSE2)
// @@@@ TODO
//#endif //(HAVE_SSE4_1 || HAVE_SSE2)*/
    }

protected:
    //! classic 'compute' implementation, based on the regular DescriptorExtractor::computeImpl arguments & expected output
    virtual void computeImpl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const;
    const bool m_bOnlyUsingAbsThreshold;
    const float m_fRelThreshold;
    const size_t m_nThreshold;
    cv::Mat m_oRefImage;

    //! note: this is the LBSP 16 bit double-cross indiv RGB/RGBA pattern as used in the original article by G.-A. Bilodeau et al.
    //  O   O   O          4 ..  3 ..  6
    //    O O O           .. 15  8 13 ..
    //  O O X O O    =>    0  9  X 11  1
    //    O O O           .. 12 10 14 ..
    //  O   O   O          7 ..  2 ..  5
    static const std::array<std::array<int,2>,16> s_anIdxLUT_16bitdbcross;
    static const std::array<uint,4> s_anIdxLUT_16bitdbcross_Horiz_Bits;
    static const std::array<uint,4> s_anIdxLUT_16bitdbcross_Diag_Bits;
    static const std::array<uint,4> s_anIdxLUT_16bitdbcross_Vert_Bits;
    static const std::array<uint,4> s_anIdxLUT_16bitdbcross_DiagInv_Bits;
    static const std::array<uint,12> s_anIdxLUT_16bitdbcross_GradX_Idxs;
    static const std::array<uint,12> s_anIdxLUT_16bitdbcross_GradY_Idxs;
    static const std::array<int,16> s_anIdxLUT_16bitdbcross_GradX;
    static const std::array<int,16> s_anIdxLUT_16bitdbcross_GradY;

    template<size_t nChannels, typename Tv>
    static inline void lookup_16bits_dbcross(const Tv* const _data, const int _x, const int _y, const size_t _c, const size_t _step_row, Tv* const anVals) {
        auto _idx = [&](int n) {
            return _step_row*(s_anIdxLUT_16bitdbcross[n][1]+_y)+nChannels*(s_anIdxLUT_16bitdbcross[n][0]+_x)+_c;
        };
        CxxUtils::unroll<16>([&](int n) {
            anVals[n] = _data[_idx(n)];
        });
    }
};
