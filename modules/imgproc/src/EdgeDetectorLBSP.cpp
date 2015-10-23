
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

#define USE_3_AXIS_ORIENT         1
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
        //std::cout << "L=" << nLevelIter << "; [" << nCurrScaleCols << "," << nCurrScaleRows << "]" << std::endl;
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
    //const size_t nNMSWinSize = (1<<(m_nLevels-1))+1; // @@@ need?
    //const size_t nNMSHalfWinSize = nNMSWinSize>>1;
    const cv::Size oMapSize(oInputImg.cols+2,oInputImg.rows+2);
    const size_t nGradMapStep = oMapSize.width*3;
    const size_t nEdgeMapStep = oMapSize.width;
    std::aligned_vector<ushort,32> vuLBSPDescGradSumData(oMapSize.height*nGradMapStep);
    std::aligned_vector<uchar,32> vuEdgeTempMaskData(oMapSize.height*nEdgeMapStep);
    cv::Mat oLBSPGradMap(oMapSize,CV_16UC3,vuLBSPDescGradSumData.data()); // 3ch, X-Y-MAG
    cv::Mat oEdgeTempMask(oMapSize,CV_8UC1,vuEdgeTempMaskData.data());
    std::fill(vuLBSPDescGradSumData.data(),vuLBSPDescGradSumData.data()+nGradMapStep,0);
    std::fill(vuLBSPDescGradSumData.data()+(oMapSize.height-1)*nGradMapStep,vuLBSPDescGradSumData.data()+oMapSize.height*nGradMapStep,0);
    std::fill(vuEdgeTempMaskData.data(),vuEdgeTempMaskData.data()+nEdgeMapStep,1);
    std::fill(vuEdgeTempMaskData.data()+(oMapSize.height-1)*nEdgeMapStep,vuEdgeTempMaskData.data()+oMapSize.height*nEdgeMapStep,1);
    oLBSPGradMap(cv::Rect(1,1,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = cv::Scalar_<ushort>(0,0,SHRT_MAX); // optm? @@@@ try full?
    //oLBSPGradMap(cv::Rect(1,1,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = cv::Scalar_<ushort>(SHRT_MAX,SHRT_MAX,SHRT_MAX); // optm? @@@@ try full?
    CV_DbgAssert(m_nLevels*2*16<USHRT_MAX); // rewrite @@@@@ make sure grad orientation maps will not overflow (adding 2x 16x uchar gradients 'nLevels' times into ushort)
    CV_DbgAssert(m_nLevels*2<USHRT_MAX); // rewrite @@@@@ make sure grad sum map will not overflow (adding 2x uchar gradients 'nLevels' times into ushort)
    size_t nCurrHystStackSize = std::max((size_t)1<<10,(size_t)oMapSize.area()/10);
    std::vector<uchar*> vuHystStack(nCurrHystStackSize);
    uchar** pauHystStack_top = &vuHystStack[0];
    uchar** pauHystStack_bottom = &vuHystStack[0];
    auto stack_push = [&](uchar* pAddr) {
        CV_DbgAssert(pAddr>=oEdgeTempMask.datastart+nEdgeMapStep);
        CV_DbgAssert(pAddr<oEdgeTempMask.dataend-nEdgeMapStep);
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
    for(size_t nLevelIter = m_nLevels-1; nLevelIter!=size_t(-1); --nLevelIter) {
        const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
        cv::Mat oPyrMap = (!nLevelIter)?oInputImg:cv::Mat(oCurrScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter-1].data());
        const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
        for(size_t nRowIter = size_t(oCurrScaleSize.height-1); nRowIter!=size_t(-2); --nRowIter) {
            CV_DbgAssert(oLBSPGradMap.step.p[0]==nGradMapStep*2 && oLBSPGradMap.step.p[1]==6);
            ushort* anGradRow = ((ushort*)oLBSPGradMap.data)+(nRowIter+1)*nGradMapStep+3;
            CV_DbgAssert(anGradRow>(ushort*)oLBSPGradMap.datastart && anGradRow<(ushort*)oLBSPGradMap.dataend);
            if(nRowIter!=size_t(-1)) {
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = (size_t)oCurrScaleSize.width-1; nColIter!=size_t(-1); --nColIter) {
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    const uchar* const anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nColLUTIdx;
                    const uchar* const auRefColor = (oPyrMap.data+nColLUTIdx/LBSP::DESC_SIZE_BITS);
                    short nGradX, nGradY;
                    ushort nGradMag;
                    LBSP::computeDescriptor_gradient<nChannels>(anCurrLUT,auRefColor,nAbsLBSPThreshold,nGradX,nGradY,nGradMag);
                    //const auto lAbsComp = [](short a, short b){return std::abs(a)<std::abs(b);};
                    //(short&)(anGradRow[nColIter*3]) = std::min(nGradX,(short)(anGradRow[nColIter*3]),lAbsComp);
                    //(short&)(anGradRow[nColIter*3+1]) = std::min(nGradY,(short)(anGradRow[nColIter*3+1]),lAbsComp);
                    (short&)(anGradRow[nColIter*3]) = (nGradX+(short)(anGradRow[nColIter*3]*2))/2;
                    (short&)(anGradRow[nColIter*3+1]) = (nGradY+(short)(anGradRow[nColIter*3+1]*2))/2;
                    anGradRow[nColIter*3+2] = std::min(nGradMag,anGradRow[nColIter*3+2]);
                    if(nLevelIter>0) {
                        const size_t nRowIter_base = nRowIter << 1;
                        const size_t nColIter_base = nColIter << 1;
                        CV_DbgAssert((nRowIter_base+1)<(oMapSize.height));
                        CV_DbgAssert((nColIter_base+1)<(oMapSize.width));
                        for(size_t nRowIterOffset = nRowIter_base; nRowIterOffset<nRowIter_base+2; ++nRowIterOffset) {
                            CV_DbgAssert(oLBSPGradMap.step.p[0]==nGradMapStep*2 && oLBSPGradMap.step.p[1]==6);
                            ushort* anNextScaleGradRow = ((ushort*)oLBSPGradMap.data)+(nRowIterOffset+1)*nGradMapStep+3;
                            CV_DbgAssert(anNextScaleGradRow<(ushort*)oLBSPGradMap.dataend);
                            for(size_t nColIterOffset = nColIter_base; nColIterOffset<nColIter_base+2; ++nColIterOffset) {
                                anNextScaleGradRow[nColIterOffset*3] = anGradRow[nColIter*3];
                                anNextScaleGradRow[nColIterOffset*3+1] = anGradRow[nColIter*3+1];
                                anNextScaleGradRow[nColIterOffset*3+2] = anGradRow[nColIter*3+2];
                            }
                        }
                    }
                }
            }
            if(nLevelIter==0) {
                std::fill(anGradRow-3,anGradRow,0); // remove if init'd at top
                std::fill(anGradRow+oInputImg.cols*3,anGradRow+(oInputImg.cols+1)*3,0);
                if(nRowIter<size_t(oCurrScaleSize.height-1)) {
                    anGradRow += nGradMapStep; // offset by one row (nms is post grad comp since 3x3 neighborhood)
                    uchar* anEdgeMapRow = oEdgeTempMask.ptr<uchar>(nRowIter+1)+1;
                    CV_DbgAssert(oEdgeTempMask.step.p[0]==nEdgeMapStep);
                    CV_DbgAssert(anEdgeMapRow>=oEdgeTempMask.datastart+nEdgeMapStep && anEdgeMapRow<oEdgeTempMask.dataend-nEdgeMapStep);
                    anEdgeMapRow[-1] = anEdgeMapRow[oInputImg.cols] = 1;
                    stack_check_size(oInputImg.cols);
                    bool nNeighbMax = false;
                    for(size_t nColIter = 0; nColIter<(size_t)oInputImg.cols; ++nColIter) {
                        // make sure all 'quick-idx' lookups are at the right positions...
                        CV_DbgAssert(anGradRow[nColIter*3]==oLBSPGradMap.at<cv::Vec3w>(nRowIter+2,nColIter+1)[0]);
                        CV_DbgAssert(anGradRow[nColIter*3+1]==oLBSPGradMap.at<cv::Vec3w>(nRowIter+2,nColIter+1)[1]);
                        CV_DbgAssert(anGradRow[nColIter*3+2]==oLBSPGradMap.at<cv::Vec3w>(nRowIter+2,nColIter+1)[2]);
                        CV_DbgAssert(anGradRow[nColIter*3-nGradMapStep+2]==oLBSPGradMap.at<cv::Vec3w>(nRowIter+1,nColIter+1)[2]);
                        CV_DbgAssert(anGradRow[nColIter*3+nGradMapStep+2]==oLBSPGradMap.at<cv::Vec3w>(nRowIter+3,nColIter+1)[2]);
                        const ushort nGradMag = anGradRow[nColIter*3+2];
                        if(nGradMag>=nHystLowThreshold) {
#if USE_3_AXIS_ORIENT
                            const short nGradX = anGradRow[nColIter*3];
                            const short nGradY = anGradRow[nColIter*3+1];
                            const uint nShift_FPA = 15;
                            constexpr uint nTG22deg_FPA = (int)(0.4142135623730950488016887242097*(1<<nShift_FPA)+0.5); // == tan(pi/8)
                            const uint nGradX_abs = (ushort)std::abs(nGradX);
                            const uint nGradY_abs = (ushort)std::abs(nGradY)<<nShift_FPA;
                            uint nTG22GradX_FPA = nGradX_abs*nTG22deg_FPA; // == 0.4142135623730950488016887242097*nGradX_abs
                            if(nGradY_abs<nTG22GradX_FPA) { // if(nGradX_abs<0.4142135623730950488016887242097*nGradX_abs) == flat gradient (sector 0)
                                if(nGradMag>anGradRow[(nColIter-1)*3+2] && nGradMag>=anGradRow[(nColIter+1)*3+2]) // if current magnitude is a peak among neighbors
                                    goto _edge_good; // push as 'edge'
                            }
                            else { // else(nGradX_abs>=0.4142135623730950488016887242097*nGradX_abs) == not a flat gradient (sectors 1, 2 or 3)
                                uint nTG67GradX_FPA = nTG22GradX_FPA+(nGradX_abs<<(nShift_FPA+1)); // == 2.4142135623730950488016887242097*nGradX_abs == tan(3*pi/8)*nGradX_abs
                                if(nGradY_abs>nTG67GradX_FPA) { // if(nGradX_abs>2.4142135623730950488016887242097*nGradX_abs == vertical gradient (sector 2)
                                    if(nGradMag>anGradRow[nColIter*3-nGradMapStep+2] && nGradMag>=anGradRow[nColIter*3+nGradMapStep+2])
                                        goto _edge_good;
                                }
                                else { // else(nGradX_abs<=2.4142135623730950488016887242097*nGradX_abs == diagonal gradient (sector 1 or 3, depending on grad sign diff)
                                    const int nSignOffset = (std::signbit(nGradX)!=std::signbit(nGradY))?1:-1;
                                    if(nGradMag>anGradRow[(nColIter-nSignOffset)*3-nGradMapStep+2] && nGradMag>anGradRow[(nColIter+nSignOffset)*3+nGradMapStep+2])
                                        goto _edge_good;
                                }
                            }
#else //!USE_3_AXIS_ORIENT
                            const uint nGradX_abs = (ushort)std::abs(anGradRow[nColIter*3]);
                            const uint nGradY_abs = (ushort)std::abs(anGradRow[nColIter*3+1]);
                            if((nGradY_abs<=nGradX_abs && nGradMag>anGradRow[(nColIter-1)*3+2] && nGradMag>=anGradRow[(nColIter+1)*3+2]) ||
                               (nGradY_abs>nGradX_abs && nGradMag>anGradRow[nColIter*3-nGradMapStep+2] && nGradMag>=anGradRow[nColIter*3+nGradMapStep+2]))
                                goto _edge_good;
#endif //!USE_3_AXIS_ORIENT
                        }
                        nNeighbMax = false;
                        anEdgeMapRow[nColIter] = 1; // not an edge
                        continue;
                        _edge_good:
                        // if not neighbor to previously identified edge, and gradmag above max threshold
                        if(!nNeighbMax && nGradMag>=nHystHighThreshold && anEdgeMapRow[nColIter+nEdgeMapStep]!=2) {
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
    CV_DbgAssert(oEdgeTempMask.step.p[0]==nEdgeMapStep);
    while(pauHystStack_top>pauHystStack_bottom) {
        stack_check_size(8);
        uchar* pEdgeAddr = stack_pop();
        if(!pEdgeAddr[-1])
            stack_push(pEdgeAddr-1);
        if(!pEdgeAddr[1])
            stack_push(pEdgeAddr+1);
        if(!pEdgeAddr[-nEdgeMapStep-1])
            stack_push(pEdgeAddr-nEdgeMapStep-1);
        if(!pEdgeAddr[-nEdgeMapStep])
            stack_push(pEdgeAddr-nEdgeMapStep);
        if(!pEdgeAddr[-nEdgeMapStep+1])
            stack_push(pEdgeAddr-nEdgeMapStep+1);
        if(!pEdgeAddr[nEdgeMapStep-1])
            stack_push(pEdgeAddr+nEdgeMapStep-1);
        if(!pEdgeAddr[nEdgeMapStep])
            stack_push(pEdgeAddr+nEdgeMapStep);
        if(!pEdgeAddr[nEdgeMapStep+1])
            stack_push(pEdgeAddr+nEdgeMapStep+1);
    }
    const uchar* oEdgeTempMaskData = oEdgeTempMask.data+nEdgeMapStep+1;
    uchar* oEdgeMaskData = oEdgeMask.data;
    for(size_t nRowIter=0; nRowIter<oInputImg.rows; ++nRowIter, oEdgeTempMaskData+=nEdgeMapStep, oEdgeMaskData+=oEdgeMask.step)
        for(size_t nColIter=0; nColIter<oInputImg.cols; ++nColIter)
            oEdgeMaskData[nColIter] = (uchar)-(oEdgeTempMaskData[nColIter] >> 1);

    std::vector<cv::Mat> voLBSPGradMap;
    cv::split(oLBSPGradMap(cv::Rect(1,1,oInputImg.cols,oInputImg.rows)).clone(),voLBSPGradMap);
    cv::Mat oGradX(voLBSPGradMap[0].size(),CV_16SC1,voLBSPGradMap[0].data);
    cv::Mat oGradX_norm; cv::normalize(oGradX,oGradX_norm,0,255,cv::NORM_MINMAX,CV_8U);
    cv::Mat oGradY(voLBSPGradMap[1].size(),CV_16SC1,voLBSPGradMap[1].data);
    cv::Mat oGradY_norm; cv::normalize(oGradY,oGradY_norm,0,255,cv::NORM_MINMAX,CV_8U);
    cv::Mat oGradMag(voLBSPGradMap[2].size(),CV_16UC1,voLBSPGradMap[2].data);
    cv::Mat oGradMag_norm; cv::normalize(oGradMag,oGradMag_norm,0,255,cv::NORM_MINMAX,CV_8U);

    cv::Mat concat = oGradX_norm.clone();
    cv::vconcat(concat,oGradY_norm,concat);
    cv::vconcat(concat,oGradMag_norm,concat);
    cv::imshow("concat grad 0-1-2",concat);
    //cv::imshow("oEdgeMask",oEdgeMask);
    //cv::Mat test_canny_out;
    //litiv::cv_canny<3,true>(oInputImg,test_canny_out,double(nHystLowThreshold),double(nHystHighThreshold));
    //cv::imshow("test_canny_out",test_canny_out);
    //cv::waitKey(0);

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
