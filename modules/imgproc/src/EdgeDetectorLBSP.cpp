
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

#include <bitset>
#include "litiv/imgproc/EdgeDetectorLBSP.hpp"
#include "litiv/features2d/LBSP.hpp"

#define USE_CANNY_NMS_HYST_STEPS  1
#define USE_DOLLAR_STR_RAND_FOR   0

#if USE_CANNY_NMS_HYST_STEPS
#define USE_5x5_NON_MAX_SUPP      1
#define USE_MIN_GRAD_ORIENT       1
#define USE_3_AXIS_ORIENT         1
#endif //USE_CANNY_NMS_HYST_STEPS
#if USE_DOLLAR_STR_RAND_FOR

#endif //USE_DOLLAR_STR_RAND_FOR
#if (USE_CANNY_NMS_HYST_STEPS+USE_DOLLAR_STR_RAND_FOR)!=1
#error "edge detector lbsp internal cfg error"
#endif //(USE_...+...)!=1

EdgeDetectorLBSP::EdgeDetectorLBSP(size_t nLevels, double dHystLowThrshFactor, bool bNormalizeOutput) :
        EdgeDetector(LBSP::PATCH_SIZE/2),
        m_nLevels(nLevels),
        m_dHystLowThrshFactor(dHystLowThrshFactor),
        m_dGaussianKernelSigma(0),
        m_bNormalizeOutput(bNormalizeOutput),
        m_vvuInputPyrMaps(std::max(nLevels,size_t(1))-1),
        m_vvuLBSPLookupMaps(nLevels),
        m_voMapSizeList(nLevels) {
    CV_Assert(m_nLevels>0);
}

template<size_t nHalfWinSize, typename Tr>
inline bool isLocalMaximum_Horizontal(const Tr* const anGradPos, const size_t nGradMapColStep, const size_t /*nGradMapRowStep*/) {
    static_assert(nHalfWinSize>=1,"need win size >= 3x3");
    const Tr nGradMag = *anGradPos;
    bool bRes = true;
    CxxUtils::unroll<nHalfWinSize>([&](int n){
        bRes &= nGradMag>anGradPos[(-n-1)*nGradMapColStep];
    });
    CxxUtils::unroll<nHalfWinSize>([&](int n){
        bRes &= nGradMag>=anGradPos[(n+1)*nGradMapColStep];
    });
    return bRes;
}

template<size_t nHalfWinSize, typename Tr>
inline bool isLocalMaximum_Vertical(const Tr* const anGradPos, const size_t /*nGradMapColStep*/, const size_t nGradMapRowStep) {
    static_assert(nHalfWinSize>=1,"need win size >= 3x3");
    const Tr nGradMag = *anGradPos;
    bool bRes = true;
    CxxUtils::unroll<nHalfWinSize>([&](int n){
        bRes &= nGradMag>anGradPos[(-n-1)*nGradMapRowStep];
    });
    CxxUtils::unroll<nHalfWinSize>([&](int n){
        bRes &= nGradMag>=anGradPos[(n+1)*nGradMapRowStep];
    });
    return bRes;
}

template<size_t nHalfWinSize, typename Tr>
inline bool isLocalMaximum_Diagonal(const Tr* const anGradPos, const size_t nGradMapColStep, const size_t nGradMapRowStep, bool bInv) {
    static_assert(nHalfWinSize>=1,"need win size >= 3x3");
    const Tr nGradMag = *anGradPos;
    bool bRes = true;
    CxxUtils::unroll<nHalfWinSize>([&](int n){
        bRes &= nGradMag>anGradPos[(bInv?-1:1)*(-n-1)*nGradMapColStep+(-n-1)*nGradMapRowStep];
    });
    CxxUtils::unroll<nHalfWinSize>([&](int n){
        bRes &= nGradMag>=anGradPos[(bInv?-1:1)*(n+1)*nGradMapColStep+(n+1)*nGradMapRowStep];
    });
    return bRes;
}

template<size_t nChannels>
void EdgeDetectorLBSP::apply_threshold_internal(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nDetThreshold, uchar nLBSPThreshold) {
    //cv::Mat oNewInput = oInputImg;
    //oNewInput = cv::Scalar_<uchar>::all(0);
    //cv::circle(oNewInput,cv::Point(oInputImg.cols/2,oInputImg.rows/2),100,cv::Scalar_<uchar>::all(255),20);
    CV_DbgAssert(!oInputImg.empty());
    CV_DbgAssert(oInputImg.isContinuous());
    CV_DbgAssert(!oEdgeMask.empty());
    CV_DbgAssert(oEdgeMask.isContinuous());
    const int nOrigType = CV_8UC(int(nChannels));
    const size_t nColLUTStep = LBSP::DESC_SIZE_BITS*nChannels;
    size_t nNextScaleRows = size_t(oInputImg.rows);
    size_t nNextScaleCols = size_t(oInputImg.cols);
    size_t nNextRowLUTStep = nColLUTStep*nNextScaleCols;
    cv::Size oNextScaleSize((int)nNextScaleCols,(int)nNextScaleRows);
    size_t nNextScaleMapSize = oNextScaleSize.area();
    m_vvuLBSPLookupMaps[0].resize(nNextScaleMapSize*nChannels*LBSP::DESC_SIZE_BITS);
    m_voMapSizeList[0] = oNextScaleSize;
    cv::Mat oNextPyrInputMap = oInputImg;
    for(size_t nLevelIter=0; nLevelIter<m_nLevels; ++nLevelIter) {
        if(!nNextScaleMapSize)
            break;
        const size_t nCurrScaleRows = nNextScaleRows;
        const size_t nCurrScaleCols = nNextScaleCols;
        const size_t nCurrRowLUTStep = nNextRowLUTStep;
        const cv::Mat oCurrPyrInputMap = oNextPyrInputMap;
        nNextScaleRows = (nCurrScaleRows+1)/2;
        nNextScaleCols = (nCurrScaleCols+1)/2;
        nNextRowLUTStep = nColLUTStep*nNextScaleCols;
        oNextScaleSize = cv::Size((int)nNextScaleCols,(int)nNextScaleRows);
        nNextScaleMapSize = nLevelIter+1<m_nLevels?oNextScaleSize.area():0;
        if(nNextScaleMapSize) {
            m_vvuInputPyrMaps[nLevelIter].resize(nNextScaleMapSize*nChannels);
            m_vvuLBSPLookupMaps[nLevelIter+1].resize(nNextScaleMapSize*nChannels*LBSP::DESC_SIZE_BITS);
            m_voMapSizeList[nLevelIter+1] = oNextScaleSize;
            oNextPyrInputMap = cv::Mat(oNextScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter].data());
            CV_DbgAssert(size_t(oNextScaleSize.area()*nChannels)==m_vvuInputPyrMaps[nLevelIter].size());
        }
        //std::cout << "LUT L=" << nLevelIter << "; [" << nCurrScaleCols << "," << nCurrScaleRows << "]" << std::endl;
        for(size_t nRowIter = 0; nRowIter<nCurrScaleRows; ++nRowIter) {
            const size_t nCurrRowLUTIdx = nRowIter*nCurrRowLUTStep;
            if(nRowIter<m_nROIBorderSize || nRowIter>=nCurrScaleRows-m_nROIBorderSize) {
//#if HAVE_SSE2
//                ...
//#else //!HAVE_SSE2
                for(size_t nColIter = 0; nColIter<nCurrScaleCols; ++nColIter) {
                    const size_t nCurrColLUTIdx = nCurrRowLUTIdx+nColIter*nColLUTStep;
                    uchar* aanCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nCurrColLUTIdx;
                    const uchar* aanCurrImg = oCurrPyrInputMap.data+nCurrColLUTIdx/LBSP::DESC_SIZE_BITS;
                    CV_DbgAssert(nCurrColLUTIdx<m_vvuLBSPLookupMaps[nLevelIter].size() && (nCurrColLUTIdx%LBSP::DESC_SIZE_BITS)==0);
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter)
                        std::fill_n(aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS,LBSP::DESC_SIZE_BITS,*(aanCurrImg+nChIter));
                    if(nNextScaleMapSize && !(nRowIter%2) && !(nColIter%2)) {
                        const size_t nNextColLUTIdx = (nRowIter/2)*nNextRowLUTStep + (nColIter/2)*nColLUTStep;
                        for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                            const size_t nNextPyrImgIdx = nNextColLUTIdx/LBSP::DESC_SIZE_BITS + nChIter;
                            CV_DbgAssert(nNextPyrImgIdx<size_t(oNextPyrInputMap.dataend-oNextPyrInputMap.datastart));
                            *(oNextPyrInputMap.data+nNextPyrImgIdx) = *(aanCurrImg+nChIter);
                        }
                    }
                }
//#endif //!HAVE_SSE2
                continue;
            }
            for(size_t nColIter = 0; nColIter<nCurrScaleCols; ++nColIter) {
                if(nColIter<m_nROIBorderSize || nColIter>=nCurrScaleCols-m_nROIBorderSize) {
//#if HAVE_SSE2
//                    ...
//#else //!HAVE_SSE2
                    for(size_t nBorderIter = 0; nBorderIter<m_nROIBorderSize; ++nBorderIter) {
                        const size_t nCurrColLUTIdx = nCurrRowLUTIdx+nColIter*nColLUTStep;
                        uchar* aanCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nCurrColLUTIdx;
                        const uchar* aanCurrImg = oCurrPyrInputMap.data+nCurrColLUTIdx/LBSP::DESC_SIZE_BITS;
                        CV_DbgAssert(nCurrColLUTIdx<m_vvuLBSPLookupMaps[nLevelIter].size() && (nCurrColLUTIdx%LBSP::DESC_SIZE_BITS)==0);
                        for(size_t nChIter = 0; nChIter<nChannels; ++nChIter)
                            std::fill_n(aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS,LBSP::DESC_SIZE_BITS,*(aanCurrImg+nChIter));
                        if(nNextScaleMapSize && !(nRowIter%2) && !(nColIter%2)) {
                            const size_t nNextColLUTIdx = (nRowIter/2)*nNextRowLUTStep + (nColIter/2)*nColLUTStep;
                            for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                                const size_t nNextPyrImgIdx = nNextColLUTIdx/LBSP::DESC_SIZE_BITS + nChIter;
                                CV_DbgAssert(nNextPyrImgIdx<size_t(oNextPyrInputMap.dataend-oNextPyrInputMap.datastart));
                                *(oNextPyrInputMap.data+nNextPyrImgIdx) = *(aanCurrImg+nChIter);
                            }
                        }
                        if(nBorderIter<m_nROIBorderSize-1)
                            ++nColIter;
                    }
//#endif //!HAVE_SSE2
                    continue;
                }
                const size_t nCurrColLUTIdx = nCurrRowLUTIdx+nColIter*nColLUTStep;
                uchar* aanCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nCurrColLUTIdx;
                CV_DbgAssert(nCurrColLUTIdx<m_vvuLBSPLookupMaps[nLevelIter].size() && (nCurrColLUTIdx%LBSP::DESC_SIZE_BITS)==0);
                LBSP::computeDescriptor_lookup<nChannels>(oCurrPyrInputMap,int(nColIter),int(nRowIter),aanCurrLUT);
                if(nNextScaleMapSize && !(nRowIter%2) && !(nColIter%2)) {
                    const size_t nNextColLUTIdx = (nRowIter/2)*nNextRowLUTStep + (nColIter/2)*nColLUTStep;
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
#if HAVE_SSE2
                        CV_DbgAssert(LBSP::DESC_SIZE_BITS==16);
                        __m128i _anInputVals = _mm_load_si128((__m128i*)(aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS)); // @@@@@ load? or just cast?
                        size_t nLUTSum = (size_t)ParallelUtils::hsum_16ub(_anInputVals);
#else //!HAVE_SSE2
                        uchar* anCurrChLUT = aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS;
                        size_t nLUTSum = 0;
                        for(size_t nLUTIter = 0; nLUTIter<LBSP::DESC_SIZE_BITS; ++nLUTIter)
                            nLUTSum += anCurrChLUT[nLUTIter];
#endif //!HAVE_SSE2
                        const size_t nNextPyrImgIdx = nNextColLUTIdx/LBSP::DESC_SIZE_BITS + nChIter;
                        CV_DbgAssert(nNextPyrImgIdx<size_t(oNextPyrInputMap.dataend-oNextPyrInputMap.datastart));
                        *(oNextPyrInputMap.data+nNextPyrImgIdx) = uchar(nLUTSum/LBSP::DESC_SIZE_BITS);
                    }
                }
            }
        }
    }
#if USE_CANNY_NMS_HYST_STEPS
    const uchar nAbsLBSPThreshold = cv::saturate_cast<uchar>(nLBSPThreshold);
    const uchar nHystHighThreshold = nDetThreshold;
    const uchar nHystLowThreshold = (uchar)(nDetThreshold*m_dHystLowThrshFactor);
    constexpr size_t nNMSWinSize = USE_5x5_NON_MAX_SUPP?LBSP::PATCH_SIZE:3;
    constexpr size_t nNMSHalfWinSize = nNMSWinSize>>1;
    const cv::Size oMapSize(oInputImg.cols+nNMSHalfWinSize*2,oInputImg.rows+nNMSHalfWinSize*2);
    const size_t nGradMapColStep = 3;
    const size_t nGradMapRowStep = oMapSize.width*nGradMapColStep;
    const size_t nEdgeMapColStep = 1;
    const size_t nEdgeMapRowStep = oMapSize.width*nEdgeMapColStep;
    std::aligned_vector<uchar,32> vuLBSPGradMapData(oMapSize.height*nGradMapRowStep);
    std::aligned_vector<uchar,32> vuEdgeTempMaskData(oMapSize.height*nEdgeMapRowStep);
    cv::Mat oLBSPGradMap(oMapSize,CV_8UC3,vuLBSPGradMapData.data()); // 3ch, X-Y-MAG
    cv::Mat oEdgeTempMask(oMapSize,CV_8UC1,vuEdgeTempMaskData.data());
    std::fill(vuLBSPGradMapData.data(),vuLBSPGradMapData.data()+nGradMapRowStep*nNMSHalfWinSize,0);
    std::fill(vuLBSPGradMapData.data()+(oMapSize.height-nNMSHalfWinSize)*nGradMapRowStep,vuLBSPGradMapData.data()+oMapSize.height*nGradMapRowStep,0);
    std::fill(vuEdgeTempMaskData.data(),vuEdgeTempMaskData.data()+nEdgeMapRowStep*nNMSHalfWinSize,1);
    std::fill(vuEdgeTempMaskData.data()+(oMapSize.height-nNMSHalfWinSize)*nEdgeMapRowStep,vuEdgeTempMaskData.data()+oMapSize.height*nEdgeMapRowStep,1);
#if USE_MIN_GRAD_ORIENT
    oLBSPGradMap(cv::Rect(nNMSHalfWinSize,nNMSHalfWinSize,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = cv::Scalar_<uchar>(CHAR_MAX,CHAR_MAX,UCHAR_MAX); // optm? @@@@ try full?
#else //!USE_MIN_GRAD_ORIENT
    oLBSPGradMap(cv::Rect(nNMSHalfWinSize,nNMSHalfWinSize,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = cv::Scalar_<uchar>(0,0,UCHAR_MAX); // optm? @@@@ try full?
#endif //!USE_MIN_GRAD_ORIENT
    size_t nCurrHystStackSize = std::max((size_t)1<<10,(size_t)oMapSize.area()/8);
    std::vector<uchar*> vuHystStack(nCurrHystStackSize);
    uchar** pauHystStack_top = &vuHystStack[0];
    uchar** pauHystStack_bottom = &vuHystStack[0];
    auto stack_push = [&](uchar* pAddr) {
        CV_DbgAssert(pAddr>=oEdgeTempMask.datastart+nEdgeMapRowStep*nNMSHalfWinSize);
        CV_DbgAssert(pAddr<oEdgeTempMask.dataend-nEdgeMapRowStep*nNMSHalfWinSize);
        *pAddr = 2, *pauHystStack_top++ = pAddr;
    };
    auto stack_pop = [&]() -> uchar* {
        CV_DbgAssert(pauHystStack_top>pauHystStack_bottom);
        return *--pauHystStack_top;
    };
    auto stack_check_size = [&](size_t nPotentialSize) {
        if(ptrdiff_t(pauHystStack_top-pauHystStack_bottom)+nPotentialSize>nCurrHystStackSize) {
            const ptrdiff_t nUsedHystStackSize = pauHystStack_top-pauHystStack_bottom;
            nCurrHystStackSize = std::max(nCurrHystStackSize*2,nUsedHystStackSize+nPotentialSize);
            vuHystStack.resize(nCurrHystStackSize);
            pauHystStack_bottom = &vuHystStack[0];
            pauHystStack_top = pauHystStack_bottom+nUsedHystStackSize;
        }
    };
    for(int nLevelIter = (int)m_nLevels-1; nLevelIter>=0; --nLevelIter) {
        const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
        cv::Mat oPyrMap = (!nLevelIter)?oInputImg:cv::Mat(oCurrScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter-1].data());
        const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
        //std::cout << "GRAD L=" << nLevelIter << "; [" << oCurrScaleSize.width << "," << oCurrScaleSize.height << "]" << std::endl;
        for(int nRowIter = oCurrScaleSize.height-1; nRowIter>=-(int)nNMSHalfWinSize; --nRowIter) {
            uchar* anGradRow = oLBSPGradMap.data+(nRowIter+nNMSHalfWinSize)*nGradMapRowStep+nGradMapColStep*nNMSHalfWinSize;
            CV_DbgAssert(anGradRow>oLBSPGradMap.datastart && anGradRow<oLBSPGradMap.dataend);
            if(nRowIter>=0) {
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = (size_t)oCurrScaleSize.width-1; nColIter!=size_t(-1); --nColIter) {
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    const uchar* const anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nColLUTIdx;
                    const uchar* const auRefColor = (oPyrMap.data+nColLUTIdx/LBSP::DESC_SIZE_BITS);
                    char nGradX, nGradY;
                    uchar nGradMag;
                    LBSP::computeDescriptor_gradient<nChannels>(anCurrLUT,auRefColor,nAbsLBSPThreshold,nGradX,nGradY,nGradMag);
#if USE_MIN_GRAD_ORIENT
                    const auto lAbsComp = [](char a, char b){return std::abs(a)<std::abs(b);}; // @@@ retest w/ fixed char sign (done)
                    (char&)(anGradRow[nColIter*nGradMapColStep]) = std::min(nGradX,char(anGradRow[nColIter*nGradMapColStep]),lAbsComp);
                    (char&)(anGradRow[nColIter*nGradMapColStep+1]) = std::min(nGradY,char(anGradRow[nColIter*nGradMapColStep+1]),lAbsComp);
#else //!USE_MIN_GRAD_ORIENT
                    CV_DbgAssert((nGradX+(char)(anGradRow[nColIter*nGradMapColStep]*2))/2<=UCHAR_MAX);
                    CV_DbgAssert((nGradY+(char)(anGradRow[nColIter*nGradMapColStep+1]*2))/2<=UCHAR_MAX);
                    (char&)(anGradRow[nColIter*nGradMapColStep]) = ((nGradX+(char)(anGradRow[nColIter*nGradMapColStep]*2))/2);
                    (char&)(anGradRow[nColIter*nGradMapColStep+1]) = ((nGradY+(char)(anGradRow[nColIter*nGradMapColStep+1]*2))/2);
#endif //!USE_MIN_GRAD_ORIENT
                    anGradRow[nColIter*nGradMapColStep+2] = std::min(nGradMag,anGradRow[nColIter*nGradMapColStep+2]);
                    if(nLevelIter>0) {
                        const size_t nRowIter_base = nRowIter << 1;
                        const size_t nColIter_base = nColIter << 1;
                        CV_DbgAssert((nRowIter_base+1)<(oMapSize.height));
                        CV_DbgAssert((nColIter_base+1)<(oMapSize.width));
                        for(size_t nRowIterOffset = nRowIter_base; nRowIterOffset<nRowIter_base+2; ++nRowIterOffset) {
                            uchar* anNextScaleGradRow = oLBSPGradMap.data+(nRowIterOffset+nNMSHalfWinSize)*nGradMapRowStep+nGradMapColStep*nNMSHalfWinSize;
                            CV_DbgAssert(anNextScaleGradRow<oLBSPGradMap.dataend);
                            for(size_t nColIterOffset = nColIter_base; nColIterOffset<nColIter_base+2; ++nColIterOffset) {
                                anNextScaleGradRow[nColIterOffset*nGradMapColStep] = anGradRow[nColIter*nGradMapColStep];
                                anNextScaleGradRow[nColIterOffset*nGradMapColStep+1] = anGradRow[nColIter*nGradMapColStep+1];
                                anNextScaleGradRow[nColIterOffset*nGradMapColStep+2] = anGradRow[nColIter*nGradMapColStep+2];
                            }
                        }
                    }
                }
            }
            if(nLevelIter==0) {
                std::fill(anGradRow-nGradMapColStep*nNMSHalfWinSize,anGradRow,0); // remove if init'd at top
                std::fill(anGradRow+oInputImg.cols*nGradMapColStep,anGradRow+(oInputImg.cols+nNMSHalfWinSize)*nGradMapColStep,0);
                if(nRowIter<oCurrScaleSize.height-nNMSHalfWinSize) {
                    anGradRow += nGradMapRowStep*nNMSHalfWinSize; // offset by nNMSHalfWinSize rows
                    uchar* anEdgeMapRow = oEdgeTempMask.ptr<uchar>(nRowIter+nNMSHalfWinSize)+nNMSHalfWinSize*nEdgeMapColStep;
                    CV_DbgAssert(anEdgeMapRow>=oEdgeTempMask.datastart+nEdgeMapRowStep*nNMSHalfWinSize && anEdgeMapRow<oEdgeTempMask.dataend-nEdgeMapRowStep*nNMSHalfWinSize);
                    std::fill(anEdgeMapRow-nNMSHalfWinSize*nGradMapColStep,anEdgeMapRow,1);
                    std::fill(anEdgeMapRow+oInputImg.cols*nGradMapColStep,anEdgeMapRow+(oInputImg.cols+nNMSHalfWinSize)*nGradMapColStep,1);
                    stack_check_size(oInputImg.cols);
                    bool nNeighbMax = false;
                    for(size_t nColIter = 0; nColIter<(size_t)oInputImg.cols; ++nColIter) {
                        // make sure all 'quick-idx' lookups are at the right positions...
                        CV_DbgAssert(anGradRow[nColIter*nGradMapColStep]==oLBSPGradMap.at<cv::Vec3b>(int(nRowIter+(nNMSHalfWinSize*2)),int(nColIter+nNMSHalfWinSize))[0]);
                        CV_DbgAssert(anGradRow[nColIter*nGradMapColStep+1]==oLBSPGradMap.at<cv::Vec3b>(int(nRowIter+(nNMSHalfWinSize*2)),int(nColIter+nNMSHalfWinSize))[1]);
                        for(int nNMSWinIter=-(int)nNMSHalfWinSize; nNMSWinIter<=(int)nNMSHalfWinSize; ++nNMSWinIter)
                            CV_DbgAssert(anGradRow[nColIter*nGradMapColStep+nGradMapRowStep*nNMSWinIter+2]==oLBSPGradMap.at<cv::Vec3b>(int(nRowIter+nNMSWinIter+(nNMSHalfWinSize*2)),int(nColIter+nNMSHalfWinSize))[2]);
                        const uchar nGradMag = anGradRow[nColIter*nGradMapColStep+2];
                        if(nGradMag>=nHystLowThreshold) {
#if USE_3_AXIS_ORIENT
                            const char nGradX = anGradRow[nColIter*nGradMapColStep];
                            const char nGradY = anGradRow[nColIter*nGradMapColStep+1];
                            const uint nShift_FPA = 15;
                            constexpr uint nTG22deg_FPA = (int)(0.4142135623730950488016887242097*(1<<nShift_FPA)+0.5); // == tan(pi/8)
                            const uint nGradX_abs = (uint)std::abs(nGradX);
                            const uint nGradY_abs = (uint)std::abs(nGradY)<<nShift_FPA;
                            uint nTG22GradX_FPA = nGradX_abs*nTG22deg_FPA; // == 0.4142135623730950488016887242097*nGradX_abs
                            if(nGradY_abs<nTG22GradX_FPA) { // if(nGradX_abs<0.4142135623730950488016887242097*nGradX_abs) == flat gradient (sector 0)
                                if(isLocalMaximum_Horizontal<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep))
                                    goto _edge_good; // push as 'edge'
                            }
                            else { // else(nGradX_abs>=0.4142135623730950488016887242097*nGradX_abs) == not a flat gradient (sectors 1, 2 or 3)
                                uint nTG67GradX_FPA = nTG22GradX_FPA+(nGradX_abs<<(nShift_FPA+1)); // == 2.4142135623730950488016887242097*nGradX_abs == tan(3*pi/8)*nGradX_abs
                                if(nGradY_abs>nTG67GradX_FPA) { // if(nGradX_abs>2.4142135623730950488016887242097*nGradX_abs == vertical gradient (sector 2)
                                    if(isLocalMaximum_Vertical<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep))
                                        goto _edge_good;
                                }
                                else { // else(nGradX_abs<=2.4142135623730950488016887242097*nGradX_abs == diagonal gradient (sector 1 or 3, depending on grad sign diff)
                                    if(isLocalMaximum_Diagonal<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep,(nGradX^nGradY)<0))
                                        goto _edge_good;
                                }
                            }
#else //!USE_3_AXIS_ORIENT
                            const uint nGradX_abs = (uint)std::abs(anGradRow[nColIter*nGradMapColStep]);
                            const uint nGradY_abs = (uint)std::abs(anGradRow[nColIter*nGradMapColStep+1]);
                            if((nGradY_abs<=nGradX_abs && isLocalMaximum_Horizontal<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep)) ||
                               (nGradY_abs>nGradX_abs && isLocalMaximum_Vertical<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep)))
                                goto _edge_good;
#endif //!USE_3_AXIS_ORIENT
                        }
                        nNeighbMax = false;
                        anEdgeMapRow[nColIter] = 1; // not an edge
                        continue;
                        _edge_good:
                        // if not neighbor to previously identified edge, and gradmag above max threshold
                        if(!nNeighbMax && nGradMag>=nHystHighThreshold && anEdgeMapRow[nColIter+nEdgeMapRowStep]!=2) {
                            stack_push(anEdgeMapRow+nColIter);
                            nNeighbMax = true;
                            continue;
                        }
                        //_edge_maybe:
                        anEdgeMapRow[nColIter] = 0; // might belong to an edge
                    }
                }
            }
        }
    }
    CV_DbgAssert(oEdgeTempMask.step.p[0]==nEdgeMapRowStep);
    while(pauHystStack_top>pauHystStack_bottom) {
        stack_check_size(8);
        uchar* pEdgeAddr = stack_pop();
        if(!pEdgeAddr[-1*nEdgeMapColStep])
            stack_push(pEdgeAddr-nEdgeMapColStep);
        if(!pEdgeAddr[nEdgeMapColStep])
            stack_push(pEdgeAddr+nEdgeMapColStep);
        if(!pEdgeAddr[-1*nEdgeMapRowStep-nEdgeMapColStep])
            stack_push(pEdgeAddr-nEdgeMapRowStep-nEdgeMapColStep);
        if(!pEdgeAddr[-1*nEdgeMapRowStep])
            stack_push(pEdgeAddr-nEdgeMapRowStep);
        if(!pEdgeAddr[-1*nEdgeMapRowStep+nEdgeMapColStep])
            stack_push(pEdgeAddr-nEdgeMapRowStep+nEdgeMapColStep);
        if(!pEdgeAddr[nEdgeMapRowStep-nEdgeMapColStep])
            stack_push(pEdgeAddr+nEdgeMapRowStep-nEdgeMapColStep);
        if(!pEdgeAddr[nEdgeMapRowStep])
            stack_push(pEdgeAddr+nEdgeMapRowStep);
        if(!pEdgeAddr[nEdgeMapRowStep+nEdgeMapColStep])
            stack_push(pEdgeAddr+nEdgeMapRowStep+nEdgeMapColStep);
    }
    const uchar* anEdgeTempMaskData = oEdgeTempMask.data+nEdgeMapRowStep*nNMSHalfWinSize+nEdgeMapColStep*nNMSHalfWinSize;
    uchar* oEdgeMaskData = oEdgeMask.data;
    for(size_t nRowIter=0; nRowIter<oInputImg.rows; ++nRowIter, anEdgeTempMaskData+=nEdgeMapRowStep, oEdgeMaskData+=oEdgeMask.step)
        for(size_t nColIter=0; nColIter<oInputImg.cols; ++nColIter)
            oEdgeMaskData[nColIter] = (uchar)-(*(anEdgeTempMaskData+nColIter*nEdgeMapColStep)>>1);

    /*std::vector<cv::Mat> voLBSPGradMap;
    cv::split(oLBSPGradMap(cv::Rect(nNMSHalfWinSize,nNMSHalfWinSize,oInputImg.cols,oInputImg.rows)).clone(),voLBSPGradMap); // @@@@ clone?
    cv::Mat oGradX(voLBSPGradMap[0].size(),CV_8SC1,voLBSPGradMap[0].data);
    cv::Mat oGradX_norm; cv::normalize(oGradX,oGradX_norm,0,255,cv::NORM_MINMAX,CV_8U);
    cv::Mat oGradY(voLBSPGradMap[1].size(),CV_8SC1,voLBSPGradMap[1].data);
    cv::Mat oGradY_norm; cv::normalize(oGradY,oGradY_norm,0,255,cv::NORM_MINMAX,CV_8U);
    cv::Mat oGradMag(voLBSPGradMap[2].size(),CV_8UC1,voLBSPGradMap[2].data);
    cv::Mat oGradMag_norm; cv::normalize(oGradMag,oGradMag_norm,0,255,cv::NORM_MINMAX,CV_8U);
    cv::Mat concat = oGradX_norm.clone();
    cv::vconcat(concat,oGradY_norm,concat);
    cv::vconcat(concat,oGradMag_norm,concat);
    cv::imshow("concat grad 0-1-2",concat);*/
    //cv::imshow("oEdgeMask",oEdgeMask);
    //cv::Mat test_canny_out;
    //litiv::cv_canny<3,true>(oInputImg,test_canny_out,double(nHystLowThreshold),double(nHystHighThreshold));
    //cv::imshow("test_canny_out",test_canny_out);
    //cv::waitKey(0);

#elif USE_DOLLAR_STR_RAND_FOR

    //@@@@ WiP

#endif //USE_DOLLAR_STR_RAND_FOR
}

template void EdgeDetectorLBSP::apply_threshold_internal<1>(const cv::Mat&, cv::Mat&, uchar, uchar);
template void EdgeDetectorLBSP::apply_threshold_internal<2>(const cv::Mat&, cv::Mat&, uchar, uchar);
template void EdgeDetectorLBSP::apply_threshold_internal<3>(const cv::Mat&, cv::Mat&, uchar, uchar);
template void EdgeDetectorLBSP::apply_threshold_internal<4>(const cv::Mat&, cv::Mat&, uchar, uchar);

void EdgeDetectorLBSP::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dDetThreshold) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.isContinuous());
    if(m_dGaussianKernelSigma>0) {
        const int nDefaultKernelSize = int(8*ceil(m_dGaussianKernelSigma));
        const int nRealKernelSize = nDefaultKernelSize%2==0?nDefaultKernelSize+1:nDefaultKernelSize;
        oInputImg = oInputImg.clone();
        cv::GaussianBlur(oInputImg,oInputImg,cv::Size(nRealKernelSize,nRealKernelSize),m_dGaussianKernelSigma,m_dGaussianKernelSigma);
    }
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    if(dDetThreshold<0||dDetThreshold>1)
        dDetThreshold = getDefaultThreshold();
    const uchar nDetThreshold = (uchar)(dDetThreshold*LBSP::MAX_GRAD_MAG);
    const int nChannels = oInputImg.channels();
    if(nChannels==1)
        apply_threshold_internal<1>(oInputImg,oEdgeMask,nDetThreshold,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else if(nChannels==2)
        apply_threshold_internal<2>(oInputImg,oEdgeMask,nDetThreshold,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else if(nChannels==3)
        apply_threshold_internal<3>(oInputImg,oEdgeMask,nDetThreshold,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else if(nChannels==4)
        apply_threshold_internal<4>(oInputImg,oEdgeMask,nDetThreshold,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else
        CV_Error(0,"Unexpected channel count");
}

void EdgeDetectorLBSP::apply(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.channels()>=1 && oInputImg.channels()<=4);
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    oEdgeMask = cv::Scalar_<uchar>(0);
    cv::Mat oTempEdgeMask = oEdgeMask.clone();
    for(size_t nCurrThreshold=0; nCurrThreshold<LBSP::MAX_GRAD_MAG; ++nCurrThreshold) {
        // @@@ reimpl so it shares LUT and only diverges for thresholding
        apply_threshold(oInputImg,oTempEdgeMask,double(nCurrThreshold)/LBSP::MAX_GRAD_MAG);
        oEdgeMask += oTempEdgeMask/LBSP::MAX_GRAD_MAG;
    }
    if(m_bNormalizeOutput)
        cv::normalize(oEdgeMask,oEdgeMask,0,UCHAR_MAX,cv::NORM_MINMAX);
}
