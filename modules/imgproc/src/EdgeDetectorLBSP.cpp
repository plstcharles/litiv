
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
#endif //USE_PREPROC_PYR_DISPLAY
#define USE_NMS_HYST_CANNY        0
#if (USE_PREPROC_PYR_DISPLAY+USE_NMS_HYST_CANNY)!=1
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

template<typename Tr>
inline bool isLocalMaximum(const Tr* pCentralValue, const Tr* anTopRow, const Tr* anCenterRow, const Tr* anBottomRow) {
//#if HAVE_SSE2
//                ...
//#else //!HAVE_SSE2
    const ptrdiff_t nColOffset = pCentralValue-anCenterRow;
    const std::array<Tr,8> anLocalVals{   anTopRow[nColOffset-1], anTopRow[nColOffset],  anTopRow[nColOffset+1],
                                       anCenterRow[nColOffset-1],                        anCenterRow[nColOffset+1],
                                       anBottomRow[nColOffset-1],anBottomRow[nColOffset],anBottomRow[nColOffset+1]};
    return (*pCentralValue)>(*std::max_element(anLocalVals.begin(),anLocalVals.end()));
//#endif //!HAVE_SSE2
}

template<size_t nChannels>
void EdgeDetectorLBSP::apply_threshold_internal(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nDetThreshold, uchar nLBSPThreshold) {
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
    const uchar nAbsLBSPThreshold = cv::saturate_cast<uchar>(nLBSPThreshold);
    const uchar nHystHighThreshold = nDetThreshold;
    const uchar nHystLowThreshold = (uchar)(nDetThreshold*m_dHystLowThrshFactor);
    const cv::Size oMapSize(oInputImg.cols+2,oInputImg.rows+2);
    std::aligned_vector<ushort,32> vuLBSPDescGradMagSumData(oMapSize.area());
    std::aligned_vector<uchar,32> vuEdgeTempMaskData(oMapSize.area());
    cv::Mat oLBSPDescGradMagSum(oMapSize,CV_16UC1,vuLBSPDescGradMagSumData.data());
    cv::Mat oEdgeTempMask(oMapSize,CV_8UC1,vuEdgeTempMaskData.data());
    std::fill(vuLBSPDescGradMagSumData.data(),vuLBSPDescGradMagSumData.data()+oMapSize.width,0);
    std::fill(vuLBSPDescGradMagSumData.data()+oMapSize.area()-oMapSize.width,vuLBSPDescGradMagSumData.data()+oMapSize.area(),0);
    std::fill(vuEdgeTempMaskData.data(),vuEdgeTempMaskData.data()+oMapSize.width,1);
    std::fill(vuEdgeTempMaskData.data()+oMapSize.area()-oMapSize.width,vuEdgeTempMaskData.data()+oMapSize.area(),1);
    oLBSPDescGradMagSum(cv::Rect(1,1,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = 0; // optm? @@@@ try full?
    CV_DbgAssert(m_nLevels*UCHAR_MAX*2<USHRT_MAX); // make sure grad sum map will not overflow (adding 2x uchar gradients, stored using 8 bits, 'nLevels' times)
    size_t nMaxHystStackSize = std::max((size_t)1<<10,(size_t)oMapSize.area()/10);
    std::vector<uchar*> vuHystStack(nMaxHystStackSize);
    uchar** pauHystStack_top = &vuHystStack[0];
    uchar** pauHystStack_bottom = &vuHystStack[0];
    auto stack_push = [&](uchar* pAddr) {
        CV_DbgAssert(pAddr>=oEdgeTempMask.datastart+oMapSize.width);
        CV_DbgAssert(pAddr<oEdgeTempMask.dataend-oMapSize.width);
        *pAddr = 2, *pauHystStack_top++ = pAddr;
    };
    auto stack_pop = [&]() -> uchar* {
        CV_DbgAssert(pauHystStack_top>pauHystStack_bottom);
        return *--pauHystStack_top;
    };
    auto stack_check_size = [&](size_t nPotentialSize) {
        if(ptrdiff_t(pauHystStack_top-pauHystStack_bottom)+nPotentialSize>nMaxHystStackSize) {
            const ptrdiff_t nCurrHystStackSize = pauHystStack_top-pauHystStack_bottom;
            nMaxHystStackSize = std::max(nMaxHystStackSize*2,nCurrHystStackSize+nPotentialSize);
            vuHystStack.resize(nMaxHystStackSize);
            pauHystStack_bottom = &vuHystStack[0];
            pauHystStack_top = pauHystStack_bottom+nCurrHystStackSize;
        }
    };
    for(size_t nLevelIter = m_nLevels-1; nLevelIter!=size_t(-1); --nLevelIter) {
        const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
        cv::Mat oPyrMap = (!nLevelIter)?oInputImg:cv::Mat(oCurrScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter-1].data());
        const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
        std::array<ushort*,3> aanGradMagRingBuffer = {vuLBSPDescGradMagSumData.data(),vuLBSPDescGradMagSumData.data()}; // pointers to the 3 lines buffers (i.e. ring buffer)
        for(size_t nRowIter = size_t(oCurrScaleSize.height-1); nRowIter!=size_t(-2); --nRowIter) {
            if(nRowIter!=size_t(-1)) {
                aanGradMagRingBuffer[2] = oLBSPDescGradMagSum.ptr<ushort>(nRowIter+1)+1;
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = (size_t)oCurrScaleSize.width-1; nColIter!=size_t(-1); --nColIter) {
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    const uchar* const anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nColLUTIdx;
                    const uchar* const auRefColor = (oPyrMap.data+nColLUTIdx/LBSP::DESC_SIZE_BITS);
                    ushort nAbsLBSPDesc; LBSP::computeDescriptor_threshold_max<nChannels>(anCurrLUT,auRefColor,nAbsLBSPThreshold,nAbsLBSPDesc);
                    ushort nRelLBSPDesc; LBSP::computeDescriptor_threshold_max_rel<2,nChannels>(anCurrLUT,auRefColor,nRelLBSPDesc);
                    const size_t nAbsLBSPDescGradMag = (DistanceUtils::popcount(nAbsLBSPDesc)*UCHAR_MAX)/LBSP::DESC_SIZE_BITS;
                    const size_t nRelLBSPDescGradMag = (DistanceUtils::popcount(nRelLBSPDesc)*UCHAR_MAX)/LBSP::DESC_SIZE_BITS;
                    CV_DbgAssert((size_t)nAbsLBSPDescGradMag+nRelLBSPDescGradMag+aanGradMagRingBuffer[2][nColIter]<USHRT_MAX);
                    aanGradMagRingBuffer[2][nColIter] += nAbsLBSPDescGradMag+nRelLBSPDescGradMag;
                    if(nLevelIter>0) {
                        const size_t nRowIter_base = nRowIter << 1;
                        const size_t nColIter_base = nColIter << 1;
                        CV_DbgAssert((nRowIter_base+1)<(oMapSize.height));
                        CV_DbgAssert((nColIter_base+1)<(oMapSize.width));
                        for(size_t nRowIterOffset = nRowIter_base; nRowIterOffset<nRowIter_base+2; ++nRowIterOffset) {
                            ushort* anGradMagNextScaleBuffer = oLBSPDescGradMagSum.ptr<ushort>(nRowIterOffset+1)+1;
                            for(size_t nColIterOffset = nColIter_base; nColIterOffset<nColIter_base+2; ++nColIterOffset)
                                anGradMagNextScaleBuffer[nColIterOffset] = aanGradMagRingBuffer[2][nColIter];
                        }
                    }
                }
            }
            if(nLevelIter==0) {
                aanGradMagRingBuffer[2][-1] = aanGradMagRingBuffer[2][oInputImg.cols] = 0; // remove if init'd at top
                if(nRowIter<size_t(oCurrScaleSize.height-1)) {
                    // if not first iter, then mag_buf[0] is before-last, mag_buf[1] is last, and mag_buf[2] was just filled
                    // orientation analysis is done for magnitude values contained in mag_buf[1] (= defined 3x3 neighborhood)
                    uchar* anEdgeMapRow = oEdgeTempMask.ptr<uchar>(nRowIter+1)+1;
                    CV_DbgAssert(oEdgeTempMask.step.p[0]==oMapSize.width);
                    CV_DbgAssert(anEdgeMapRow>=oEdgeTempMask.datastart+oMapSize.width && anEdgeMapRow<oEdgeTempMask.dataend-oMapSize.width);
                    anEdgeMapRow[-1] = anEdgeMapRow[oInputImg.cols] = 1;
                    const ushort* anGradMagTopRow = aanGradMagRingBuffer[2];
                    const ushort* anGradMagCenterRow = aanGradMagRingBuffer[1];
                    const ushort* anGradMagBottomRow = aanGradMagRingBuffer[0];
                    stack_check_size(oInputImg.cols);
                    bool nNeighbMax = false;
                    for(size_t nColIter = 0; nColIter<(size_t)oInputImg.cols; ++nColIter) {
                        const ushort nGradMag = anGradMagCenterRow[nColIter];
                        if(nGradMag>nHystLowThreshold && isLocalMaximum(anGradMagCenterRow+nColIter,anGradMagTopRow,anGradMagCenterRow,anGradMagBottomRow)) {
                            if(!nNeighbMax && nGradMag>nHystHighThreshold && anEdgeMapRow[nColIter-oMapSize.width]!=2) {
                                CV_DbgAssert(anEdgeMapRow+nColIter>=oEdgeTempMask.datastart+oMapSize.width && anEdgeMapRow+nColIter<oEdgeTempMask.dataend-oMapSize.width);
                                stack_push(anEdgeMapRow+nColIter);
                                nNeighbMax = true;
                            }
                            else
                                anEdgeMapRow[nColIter] = 0; // might belong to an edge
                        }
                        else {
                            nNeighbMax = false;
                            anEdgeMapRow[nColIter] = 1; // not an edge
                        }
                    }
                }
                aanGradMagRingBuffer[0] = aanGradMagRingBuffer[1];
                aanGradMagRingBuffer[1] = aanGradMagRingBuffer[2];
            }
        }
    }
    CV_DbgAssert(oEdgeTempMask.step.p[0]==oMapSize.width);
    while(pauHystStack_top>pauHystStack_bottom) {
        stack_check_size(8);
        uchar* pEdgeAddr = stack_pop();;
        if(!pEdgeAddr[-1])
            stack_push(pEdgeAddr-1);
        if(!pEdgeAddr[1])
            stack_push(pEdgeAddr+1);
        if(!pEdgeAddr[-oMapSize.width-1])
            stack_push(pEdgeAddr-oMapSize.width-1);
        if(!pEdgeAddr[-oMapSize.width])
            stack_push(pEdgeAddr-oMapSize.width);
        if(!pEdgeAddr[-oMapSize.width+1])
            stack_push(pEdgeAddr-oMapSize.width+1);
        if(!pEdgeAddr[oMapSize.width-1])
            stack_push(pEdgeAddr+oMapSize.width-1);
        if(!pEdgeAddr[oMapSize.width])
            stack_push(pEdgeAddr+oMapSize.width);
        if(!pEdgeAddr[oMapSize.width+1])
            stack_push(pEdgeAddr+oMapSize.width+1);
    }
    const uchar* oEdgeTempMaskData = oEdgeTempMask.data+oMapSize.width+1;
    uchar* oEdgeMaskData = oEdgeMask.data;
    for(size_t nRowIter=0; nRowIter<oInputImg.rows; ++nRowIter, oEdgeTempMaskData+=oMapSize.width, oEdgeMaskData+=oEdgeMask.step)
        for(size_t nColIter=0; nColIter<oInputImg.cols; ++nColIter)
            oEdgeMaskData[nColIter] = (uchar)-(oEdgeTempMaskData[nColIter] >> 1);

    cv::Mat oLBSPDescGradMagSumNorm;
    cv::normalize(oLBSPDescGradMagSum/m_nLevels,oLBSPDescGradMagSumNorm,0,255,cv::NORM_MINMAX,CV_8U);
    cv::imshow("oLBSPDescGradMagSumNorm",oLBSPDescGradMagSumNorm);
    cv::imshow("oEdgeMask",oEdgeMask);
    cv::Mat tmp_canny;
    litiv::cv_canny<3,true>(oInputImg,tmp_canny,double(nHystLowThreshold),double(nHystHighThreshold));
    cv::imshow("tmp_canny",tmp_canny);
    cv::waitKey(0);

#elif USE_NMS_HYST_CANNY

    //@@@@ WiP

#endif //USE_NMS_HYST_CANNY
}

template void EdgeDetectorLBSP::apply_threshold_internal<1>(const cv::Mat&, cv::Mat&, uchar, uchar);
template void EdgeDetectorLBSP::apply_threshold_internal<2>(const cv::Mat&, cv::Mat&, uchar, uchar);
template void EdgeDetectorLBSP::apply_threshold_internal<3>(const cv::Mat&, cv::Mat&, uchar, uchar);
template void EdgeDetectorLBSP::apply_threshold_internal<4>(const cv::Mat&, cv::Mat&, uchar, uchar);

void EdgeDetectorLBSP::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dDetThreshold, uchar nLBSPThreshold) {
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
    if(dDetThreshold<0||dDetThreshold>1)
        dDetThreshold = getDefaultThreshold();
    const uchar nDetThreshold = (uchar)(dDetThreshold*UCHAR_MAX);
    const int nChannels = oInputImg.channels();
    if(nChannels==1)
        apply_threshold_internal<1>(oInputImg,oEdgeMask,nDetThreshold,nLBSPThreshold);
    else if(nChannels==2)
        apply_threshold_internal<2>(oInputImg,oEdgeMask,nDetThreshold,nLBSPThreshold);
    else if(nChannels==3)
        apply_threshold_internal<3>(oInputImg,oEdgeMask,nDetThreshold,nLBSPThreshold);
    else if(nChannels==4)
        apply_threshold_internal<4>(oInputImg,oEdgeMask,nDetThreshold,nLBSPThreshold);
    else
        CV_Error(0,"Unexpected channel count");
}

void EdgeDetectorLBSP::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dDetThreshold) {
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
    if(dDetThreshold<0||dDetThreshold>1)
        dDetThreshold = getDefaultThreshold();
    const uchar nDetThreshold = (uchar)(dDetThreshold*UCHAR_MAX);
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
        apply_threshold_internal<1>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else if(nChannels==2)
        apply_threshold_internal<2>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else if(nChannels==3)
        apply_threshold_internal<3>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else if(nChannels==4)
        apply_threshold_internal<4>(oInputImg,oEdgeMask,EDGLBSP_DEFAULT_DET_THRESHOLD_INTEGER,EDGLBSP_DEFAULT_LBSP_THRESHOLD_INTEGER);
    else
        CV_Error(0,"Unexpected channel count");
    if(m_bNormalizeOutput)
        cv::normalize(oEdgeMask,oEdgeMask,0,UCHAR_MAX,cv::NORM_MINMAX);
}
