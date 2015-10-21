
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

#define USE_PREPROC_PYR_DISPLAY   1
#if USE_PREPROC_PYR_DISPLAY
#define USE_REL_LBSP                  0
#define DEFAULT_LBSP_REL_THRESHOLD    0.50f
#endif //USE_PREPROC_PYR_DISPLAY
#define USE_NMS_HYST_CANNY        0
#if (USE_PREPROC_PYR_DISPLAY+USE_NMS_HYST_CANNY)!=1
#error "edge detector lbsp internal cfg error"
#endif //(USE_...+...)!=1

EdgeDetectorLBSP::EdgeDetectorLBSP(size_t nLevels, bool bNormalizeOutput) :
        EdgeDetector(LBSP::PATCH_SIZE/2),
        m_nLevels(nLevels),
        m_bNormalizeOutput(bNormalizeOutput),
        m_vvuInputPyrMaps(std::max(nLevels,size_t(1))-1),
        m_vvuLBSPLookupMaps(nLevels),
        m_voMapSizeList(nLevels),
        m_dHystLowThrshFactor(0),
        m_dGaussianKernelSigma(0),
        m_bUsingL2GradientNorm(0) {
    CV_Assert(m_nLevels>0);
}

template<size_t nChannels>
void EdgeDetectorLBSP::apply_threshold_internal(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nThreshold, bool bNormalize) {
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
        std::cout << "L=" << nLevelIter << "; [" << nCurrScaleCols << "," << nCurrScaleRows << "]" << std::endl;
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
                        size_t nLUTSum = (size_t)ParallelUtils::hsum_16bytes(_anInputVals);
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
#if USE_PREPROC_PYR_DISPLAY
    const uchar nAbsLBSPThreshold = cv::saturate_cast<uchar>(nThreshold);
    const cv::Size oMapSize(oInputImg.cols+(oInputImg.cols%2),oInputImg.rows+(oInputImg.rows%2));
    std::aligned_vector<ushort,32> vuLBSPDescGradMagSumData(oMapSize.area());
    std::aligned_vector<ushort,32> vuLBSPDescGradOrientSumData(oMapSize.area());
    cv::Mat oLBSPDescGradMagSum(oMapSize,CV_16UC1,vuLBSPDescGradMagSumData.data());
    cv::Mat oLBSPDescGradOrientSum(oMapSize,CV_16UC1,vuLBSPDescGradOrientSumData.data());
    oLBSPDescGradMagSum(cv::Rect(0,0,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = 0; // optm? @@@@
    oLBSPDescGradOrientSum(cv::Rect(0,0,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = 0;
    CV_DbgAssert(m_nLevels*UCHAR_MAX*2<USHRT_MAX); // make sure grad sum map will not overflow (adding 2x uchar gradients, stored using 8 bits, 'nLevels' times)
    CV_DbgAssert(m_nLevels*2<0xF); // make sure orient sum map will not overflow (adding 2x binary orientations, stored using 4 bits, 'nLevels' times)
    const size_t nRowMapStep = (size_t)(LBSP::DESC_SIZE*oMapSize.width);
    for(size_t nLevelIter = m_nLevels-1; nLevelIter!=size_t(-1); --nLevelIter) {
        const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
        cv::Mat oPyrMap = (!nLevelIter)?oInputImg:cv::Mat(oCurrScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter-1].data());
        const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
        for(size_t nRowIter = (size_t)oCurrScaleSize.height-1; nRowIter!=size_t(-1); --nRowIter) {
            const size_t nRowMapIdx = nRowIter*nRowMapStep;
            const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
            for(size_t nColIter = (size_t)oCurrScaleSize.width-1; nColIter!=size_t(-1); --nColIter) {
                const size_t nColMapIdx = nRowMapIdx+nColIter*LBSP::DESC_SIZE;
                const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                const uchar* const anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nColLUTIdx;
                const uchar* const auRefColor = (oPyrMap.data+nColLUTIdx/LBSP::DESC_SIZE_BITS);
                ushort nAbsLBSPDesc; LBSP::computeDescriptor_threshold_max<nChannels>(anCurrLUT,auRefColor,nAbsLBSPThreshold,nAbsLBSPDesc);
                ushort nRelLBSPDesc; LBSP::computeDescriptor_threshold_max_rel<2,nChannels>(anCurrLUT,auRefColor,nRelLBSPDesc);
                const size_t nAbsLBSPDescGradMag = (DistanceUtils::popcount(nAbsLBSPDesc)*UCHAR_MAX)/LBSP::DESC_SIZE_BITS;
                const size_t nRelLBSPDescGradMag = (DistanceUtils::popcount(nRelLBSPDesc)*UCHAR_MAX)/LBSP::DESC_SIZE_BITS;
                const ushort nAbsLBSPDescOrient = LBSP::computeDescriptor_orientation<ushort,4>(nAbsLBSPDesc);
                const ushort nRelLBSPDescOrient = LBSP::computeDescriptor_orientation<ushort,4>(nRelLBSPDesc);
                CV_DbgAssert(nColMapIdx<size_t(oLBSPDescGradMagSum.dataend-oLBSPDescGradMagSum.datastart));
                CV_DbgAssert(nColMapIdx<size_t(oLBSPDescGradOrientSum.dataend-oLBSPDescGradOrientSum.datastart));
                CV_DbgAssert((size_t)nAbsLBSPDescGradMag+nRelLBSPDescGradMag+*(ushort*)(oLBSPDescGradMagSum.data+nColMapIdx)<USHRT_MAX);
                CV_DbgAssert((size_t)(nAbsLBSPDescOrient&0xF)+(nRelLBSPDescOrient&0xF)+(*(ushort*)(oLBSPDescGradOrientSum.data+nColMapIdx)&0xF)<0xF);
                CV_DbgAssert((size_t)nAbsLBSPDescOrient+nRelLBSPDescOrient+*(ushort*)(oLBSPDescGradOrientSum.data+nColMapIdx)<USHRT_MAX);
                *(ushort*)(oLBSPDescGradMagSum.data+nColMapIdx) += nAbsLBSPDescGradMag + nRelLBSPDescGradMag;
                *(ushort*)(oLBSPDescGradOrientSum.data+nColMapIdx) += nAbsLBSPDescOrient + nRelLBSPDescOrient;
                //MAJORITY VOTE ON ORIENTATION FROM EACH LEVEL HERE
                if(nLevelIter>0) {
                    const size_t nRowIdx_base = nRowIter<<1;
                    const size_t nColIdx_base = nColIter<<1;
                    CV_DbgAssert((nRowIdx_base+1)<=(oMapSize.height));
                    CV_DbgAssert((nColIdx_base+1)<=(oMapSize.width));
                    for(size_t nRowIterOffset = 0; nRowIterOffset<2; ++nRowIterOffset) {
                        const size_t nRowMapIdx_base = (nRowIdx_base+nRowIterOffset)*nRowMapStep;
                        for(size_t nColIterOffset = 0; nColIterOffset<2; ++nColIterOffset) {
                            const size_t nColMapIdx_base = nRowMapIdx_base+(nColIdx_base+nColIterOffset)*LBSP::DESC_SIZE;
                            *(ushort*)(oLBSPDescGradMagSum.data+nColMapIdx_base) = *(ushort*)(oLBSPDescGradMagSum.data+nColMapIdx);
                            *(ushort*)(oLBSPDescGradOrientSum.data+nColMapIdx_base) = *(ushort*)(oLBSPDescGradOrientSum.data+nColMapIdx);
                        }
                    }
                }
                else {
                    const ushort nGradOrient = *(ushort*)(oLBSPDescGradOrientSum.data+nColMapIdx);
                    const std::array<std::bitset<4>,4> avbBins{std::bitset<4>(nGradOrient),std::bitset<4>(nGradOrient>>4),std::bitset<4>(nGradOrient>>8),std::bitset<4>(nGradOrient>>12)};
                    const auto cmp = [](const std::bitset<4>& a, const std::bitset<4>& b) {return a.to_ulong()<b.to_ulong();};
                    *(ushort*)(oLBSPDescGradOrientSum.data+nColMapIdx) = (LBSP::eGradientOrientation)(1<<(std::max_element(avbBins.begin(),avbBins.end(),cmp)-avbBins.begin()));
                }
            }
        }
    }
    cv::Mat oLBSPDescGradMagSumNorm;
    cv::normalize(oLBSPDescGradMagSum/m_nLevels,oLBSPDescGradMagSumNorm,0,255,cv::NORM_MINMAX,CV_8U);
    cv::imshow("oLBSPDescGradMagSumNorm",oLBSPDescGradMagSumNorm);
    cv::imshow("oLBSPDescGradOrient_Horiz",oLBSPDescGradOrientSum==LBSP::eGradientOrientation_Horizontal);

    /*std::vector<cv::Mat> test;
    cv::split(oDisplayLBSPDescGradSumNorm,test);
    cv::Mat oDisplayLBSPDescGradSumNorm_NMS;
    litiv::nonMaxSuppression<5>(test[0],oDisplayLBSPDescGradSumNorm_NMS);
    cv::imshow("GradSumNorm abs/rel/mix NMS",oDisplayLBSPDescGradSumNorm_NMS);*/
    cv::waitKey(0);
#elif USE_NMS_HYST_CANNY

    //@@@@ WiP

#endif //USE_NMS_HYST_CANNY
    if(bNormalize)
        cv::normalize(oEdgeMask,oEdgeMask,0,UCHAR_MAX,cv::NORM_MINMAX);
}

template void EdgeDetectorLBSP::apply_threshold_internal<1>(const cv::Mat&, cv::Mat&, uchar, bool);
template void EdgeDetectorLBSP::apply_threshold_internal<2>(const cv::Mat&, cv::Mat&, uchar, bool);
template void EdgeDetectorLBSP::apply_threshold_internal<3>(const cv::Mat&, cv::Mat&, uchar, bool);
template void EdgeDetectorLBSP::apply_threshold_internal<4>(const cv::Mat&, cv::Mat&, uchar, bool);

void EdgeDetectorLBSP::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dThreshold) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.isContinuous());
    if(m_dGaussianKernelSigma>0) {
        // @@@ clone if overwrite image?
        const int nDefaultKernelSize = int(8*ceil(m_dGaussianKernelSigma));
        const int nRealKernelSize = nDefaultKernelSize%2==0?nDefaultKernelSize+1:nDefaultKernelSize;
        cv::GaussianBlur(oInputImg,oInputImg,cv::Size(nRealKernelSize,nRealKernelSize),m_dGaussianKernelSigma,m_dGaussianKernelSigma);
    }
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    //oEdgeMask = cv::Scalar_<uchar>(0);
    if(dThreshold<0||dThreshold>1)
        dThreshold = getDefaultThreshold();
    const uchar nThreshold = (uchar)(dThreshold*UCHAR_MAX);
    const int nChannels = oInputImg.channels();
    if(nChannels==1)
        apply_threshold_internal<1>(oInputImg,oEdgeMask,nThreshold,m_bNormalizeOutput);
    else if(nChannels==2)
        apply_threshold_internal<2>(oInputImg,oEdgeMask,nThreshold,m_bNormalizeOutput);
    else if(nChannels==3)
        apply_threshold_internal<3>(oInputImg,oEdgeMask,nThreshold,m_bNormalizeOutput);
    else if(nChannels==4)
        apply_threshold_internal<4>(oInputImg,oEdgeMask,nThreshold,m_bNormalizeOutput);
    else
        CV_Error(0,"Unexpected channel count");
}

void EdgeDetectorLBSP::apply(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.isContinuous());
    if(m_dGaussianKernelSigma>0) {
        const int nDefaultKernelSize = int(8*ceil(m_dGaussianKernelSigma));
        const int nRealKernelSize = nDefaultKernelSize%2==0?nDefaultKernelSize+1:nDefaultKernelSize;
        cv::GaussianBlur(oInputImg,oInputImg,cv::Size(nRealKernelSize,nRealKernelSize),m_dGaussianKernelSigma,m_dGaussianKernelSigma);
    }
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    //oEdgeMask = cv::Scalar_<uchar>(0);
    // @@@@ add threshold loop like canny's
    const int nChannels = oInputImg.channels();
    if(nChannels==1)
        apply_threshold_internal<1>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_INT_THRESHOLD,m_bNormalizeOutput);
    else if(nChannels==2)
        apply_threshold_internal<2>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_INT_THRESHOLD,m_bNormalizeOutput);
    else if(nChannels==3)
        apply_threshold_internal<3>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_INT_THRESHOLD,m_bNormalizeOutput);
    else if(nChannels==4)
        apply_threshold_internal<4>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_INT_THRESHOLD,m_bNormalizeOutput);
    else
        CV_Error(0,"Unexpected channel count");

}
