
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

#include "litiv/imgproc/EdgeDetectorLBSP.hpp"
#include "litiv/imgproc.hpp"
#include <bitset>

#define USE_5x5_NON_MAX_SUPP      1
#define USE_MIN_GRAD_ORIENT       1
#define USE_3_AXIS_ORIENT         1

EdgeDetectorLBSP::EdgeDetectorLBSP(size_t nLevels, double dHystLowThrshFactor, bool bNormalizeOutput) :
        m_nLevels(nLevels),
        m_dHystLowThrshFactor(dHystLowThrshFactor),
        m_dGaussianKernelSigma(0),
        m_bNormalizeOutput(bNormalizeOutput),
        m_vvuInputPyrMaps(std::max(nLevels,size_t(1))-1),
        m_vvuLBSPLookupMaps(nLevels),
        m_voMapSizeList(nLevels) {
    lvAssert_(m_dHystLowThrshFactor>0 && m_dHystLowThrshFactor<1,"lower hysteresis threshold factor must be between 0 and 1");
    lvAssert_(m_dGaussianKernelSigma>=0,"gaussian smoothing kernel sigma must be non-negative");
    m_nROIBorderSize = LBSP::PATCH_SIZE/2;
    lvAssert_(m_nLevels>0,"number of pyramid levels must be positive");
}

template<size_t nChannels>
void EdgeDetectorLBSP::apply_internal_lookup(const cv::Mat& oInputImg) {
    lvAssert_(!oInputImg.empty() && oInputImg.isContinuous(),"input image must be non-empty and continuous");
    const int nOrigType = CV_8UC(int(nChannels));
    lvDbgAssert(m_nROIBorderSize==LBSP::PATCH_SIZE/2);
    constexpr size_t nROIBorderSize = LBSP::PATCH_SIZE/2;
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
            lvDbgAssert(size_t(oNextScaleSize.area()*nChannels)==m_vvuInputPyrMaps[nLevelIter].size());
        }
        const auto lBorderColLookup = [&](size_t nRowIter, size_t nCurrRowLUTIdx, size_t nColIter){
            const size_t nCurrColLUTIdx = nCurrRowLUTIdx+nColIter*nColLUTStep;
            uchar* aanCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nCurrColLUTIdx;
            const uchar* aanCurrImg = oCurrPyrInputMap.data+nCurrColLUTIdx/LBSP::DESC_SIZE_BITS;
            lvDbgAssert(nCurrColLUTIdx<m_vvuLBSPLookupMaps[nLevelIter].size() && (nCurrColLUTIdx%LBSP::DESC_SIZE_BITS)==0);
#if HAVE_SSE2
            // no slower than fill_n if fill_n is implemented with SSE
            static_assert(LBSP::DESC_SIZE_BITS==16,"all channels should already be 16-byte-aligned");
            lv::unroll<nChannels>([&](size_t nChIter){
                lv::copy_16ub((__m128i*)(aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS),*(aanCurrImg+nChIter));
            });
#else //(!HAVE_SSE2)
            lv::unroll<nChannels>([&](size_t nChIter){
                std::fill_n(aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS,LBSP::DESC_SIZE_BITS,*(aanCurrImg+nChIter));
            });
#endif //(!HAVE_SSE2)
            if(nNextScaleMapSize && !(nRowIter%2) && !(nColIter%2)) {
                const size_t nNextColLUTIdx = (nRowIter/2)*nNextRowLUTStep + (nColIter/2)*nColLUTStep;
                for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                    const size_t nNextPyrImgIdx = nNextColLUTIdx/LBSP::DESC_SIZE_BITS + nChIter;
                    lvDbgAssert(nNextPyrImgIdx<size_t(oNextPyrInputMap.dataend-oNextPyrInputMap.datastart));
                    *(oNextPyrInputMap.data+nNextPyrImgIdx) = *(aanCurrImg+nChIter);
                }
            }
        };
        const auto lBorderRowLookup = [&](size_t nRowIter){
            const size_t nCurrRowLUTIdx = nRowIter*nCurrRowLUTStep;
            for(size_t nColIter = 0; nColIter<nCurrScaleCols; ++nColIter)
                lBorderColLookup(nRowIter,nCurrRowLUTIdx,nColIter);
        };
        size_t nRowIter = 0;
        for(; nRowIter<nROIBorderSize; ++nRowIter)
            lBorderRowLookup(nRowIter);
        for(; nRowIter<nCurrScaleRows-nROIBorderSize; ++nRowIter) {
            const size_t nCurrRowLUTIdx = nRowIter*nCurrRowLUTStep;
            size_t nColIter = 0;
            for(; nColIter<nROIBorderSize; ++nColIter)
                lBorderColLookup(nRowIter,nCurrRowLUTIdx,nColIter);
            for(; nColIter<nCurrScaleCols-nROIBorderSize; ++nColIter) {
                const size_t nCurrColLUTIdx = nCurrRowLUTIdx+nColIter*nColLUTStep;
                uchar* aanCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nCurrColLUTIdx;
                lvDbgAssert(nCurrColLUTIdx<m_vvuLBSPLookupMaps[nLevelIter].size() && (nCurrColLUTIdx%LBSP::DESC_SIZE_BITS)==0);
                LBSP::computeDescriptor_lookup<nChannels>(oCurrPyrInputMap,int(nColIter),int(nRowIter),aanCurrLUT);
                if(nNextScaleMapSize && !(nRowIter%2) && !(nColIter%2)) {
                    const size_t nNextColLUTIdx = (nRowIter/2)*nNextRowLUTStep + (nColIter/2)*nColLUTStep;
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
#if HAVE_SSE2
                        static_assert(LBSP::DESC_SIZE_BITS==16,"all channels should already be 16-byte-aligned");
                        __m128i _anInputVals = _mm_load_si128((__m128i*)(aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS));
                        size_t nLUTSum = (size_t)lv::hsum_16ub(_anInputVals);
#else //(!HAVE_SSE2)
                        uchar* anCurrChLUT = aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS;
                        size_t nLUTSum = 0;
                        lv::unroll<LBSP::DESC_SIZE_BITS>([&](size_t nLUTIter){
                            nLUTSum += anCurrChLUT[nLUTIter];
                        });
#endif //(!HAVE_SSE2)
                        const size_t nNextPyrImgIdx = nNextColLUTIdx/LBSP::DESC_SIZE_BITS + nChIter;
                        lvDbgAssert(nNextPyrImgIdx<size_t(oNextPyrInputMap.dataend-oNextPyrInputMap.datastart));
                        *(oNextPyrInputMap.data+nNextPyrImgIdx) = uchar(nLUTSum/LBSP::DESC_SIZE_BITS);
                    }
                }
            }
            for(; nColIter<nCurrScaleCols; ++nColIter)
                lBorderColLookup(nRowIter,nCurrRowLUTIdx,nColIter);
        }
        for(; nRowIter<nCurrScaleRows; ++nRowIter)
            lBorderRowLookup(nRowIter);
    }
}

template void EdgeDetectorLBSP::apply_internal_lookup<1>(const cv::Mat&);
template void EdgeDetectorLBSP::apply_internal_lookup<2>(const cv::Mat&);
template void EdgeDetectorLBSP::apply_internal_lookup<3>(const cv::Mat&);
template void EdgeDetectorLBSP::apply_internal_lookup<4>(const cv::Mat&);

void EdgeDetectorLBSP::apply_internal_lookup(const cv::Mat& oInputImg, size_t nChannels) {
    if(nChannels==1)
        apply_internal_lookup<1>(oInputImg);
    else if(nChannels==2)
        apply_internal_lookup<2>(oInputImg);
    else if(nChannels==3)
        apply_internal_lookup<3>(oInputImg);
    else if(nChannels==4)
        apply_internal_lookup<4>(oInputImg);
    else
        CV_Error(-1,"Unexpected channel count");
}

template<size_t nChannels>
void EdgeDetectorLBSP::apply_internal_threshold(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nDetThreshold) {
    lvAssert_(!oInputImg.empty() && oInputImg.isContinuous(),"input image must be non-empty and continuous");
    lvAssert_(!oEdgeMask.empty() && oEdgeMask.isContinuous(),"output mask must be non-empty and continuous");
    const int nOrigType = CV_8UC(int(nChannels));
    const size_t nColLUTStep = LBSP::DESC_SIZE_BITS*nChannels;
    const uchar nHystHighThreshold = nDetThreshold;
    const uchar nHystLowThreshold = (uchar)(nDetThreshold*m_dHystLowThrshFactor);
    constexpr size_t nNMSWinSize = USE_5x5_NON_MAX_SUPP?LBSP::PATCH_SIZE:3;
    constexpr size_t nNMSHalfWinSize = nNMSWinSize>>1;
    const cv::Size oMapSize(oInputImg.cols+nNMSHalfWinSize*2,oInputImg.rows+nNMSHalfWinSize*2);
    constexpr size_t nGradMapColStep = 4; // 4ch (gradx, grady, gradmag, 'dont care')
    const size_t nGradMapRowStep = oMapSize.width*nGradMapColStep;
    constexpr size_t nEdgeMapColStep = 1; // 1ch (label)
    const size_t nEdgeMapRowStep = oMapSize.width*nEdgeMapColStep;
    m_vuLBSPGradMapData.resize(oMapSize.height*nGradMapRowStep);
    m_vuEdgeTempMaskData.resize(oMapSize.height*nEdgeMapRowStep);
    cv::Mat oGradMap(oMapSize,CV_8UC4,m_vuLBSPGradMapData.data());
    cv::Mat oEdgeTempMask(oMapSize,CV_8UC1,m_vuEdgeTempMaskData.data());
    std::fill(m_vuLBSPGradMapData.data(),m_vuLBSPGradMapData.data()+nGradMapRowStep*nNMSHalfWinSize,0);
    std::fill(m_vuLBSPGradMapData.data()+(oMapSize.height-nNMSHalfWinSize)*nGradMapRowStep,m_vuLBSPGradMapData.data()+oMapSize.height*nGradMapRowStep,0);
    std::fill(m_vuEdgeTempMaskData.data(),m_vuEdgeTempMaskData.data()+nEdgeMapRowStep*nNMSHalfWinSize,1);
    std::fill(m_vuEdgeTempMaskData.data()+(oMapSize.height-nNMSHalfWinSize)*nEdgeMapRowStep,m_vuEdgeTempMaskData.data()+oMapSize.height*nEdgeMapRowStep,1);
#if USE_MIN_GRAD_ORIENT
    static_assert(nGradMapColStep==4,"Need 32-bit chunks to copy (see lines with uint32_t)");
    constexpr uint32_t nDefaultGradMapVal4Ch = (CHAR_MAX<<24)|(CHAR_MAX<<16)|(UCHAR_MAX)<<8;
    std::fill((uint32_t*)(m_vuLBSPGradMapData.data()+nGradMapRowStep*nNMSHalfWinSize+nGradMapColStep*nNMSHalfWinSize),(uint32_t*)(m_vuLBSPGradMapData.data()+(oMapSize.height-nNMSHalfWinSize)*nGradMapRowStep-nNMSHalfWinSize*nGradMapColStep),nDefaultGradMapVal4Ch);
    const auto lAbsCharComp = [](char a, char b){return std::abs(a)<std::abs(b);};
#else //(!USE_MIN_GRAD_ORIENT)
    oGradMap(cv::Rect(nNMSHalfWinSize,nNMSHalfWinSize,m_voMapSizeList.back().width,m_voMapSizeList.back().height)) = cv::Scalar_<uchar>(0,0,UCHAR_MAX,0);
#endif //(!USE_MIN_GRAD_ORIENT)
    size_t nCurrHystStackSize = std::max(std::max((size_t)1<<10,(size_t)oMapSize.area()/8),m_vuHystStack.size());
    m_vuHystStack.resize(nCurrHystStackSize);
    uchar** pauHystStack_top = &m_vuHystStack[0];
    uchar** pauHystStack_bottom = &m_vuHystStack[0];
    auto stack_push = [&](uchar* pAddr) {
        lvDbgAssert(pAddr>=oEdgeTempMask.datastart+nEdgeMapRowStep*nNMSHalfWinSize);
        lvDbgAssert(pAddr<oEdgeTempMask.dataend-nEdgeMapRowStep*nNMSHalfWinSize);
        *pAddr = 2, *pauHystStack_top++ = pAddr;
    };
    auto stack_pop = [&]() -> uchar* {
        lvDbgAssert(pauHystStack_top>pauHystStack_bottom);
        return *--pauHystStack_top;
    };
    auto stack_check_size = [&](size_t nPotentialSize) {
        if(ptrdiff_t(pauHystStack_top-pauHystStack_bottom)+nPotentialSize>nCurrHystStackSize) {
            const ptrdiff_t nUsedHystStackSize = pauHystStack_top-pauHystStack_bottom;
            nCurrHystStackSize = std::max(nCurrHystStackSize*2,nUsedHystStackSize+nPotentialSize);
            m_vuHystStack.resize(nCurrHystStackSize);
            pauHystStack_bottom = &m_vuHystStack[0];
            pauHystStack_top = pauHystStack_bottom+nUsedHystStackSize;
        }
    };
    for(int nLevelIter = (int)m_nLevels-1; nLevelIter>=0; --nLevelIter) {
        const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
        const cv::Mat& oPyrMap = (!nLevelIter)?oInputImg:cv::Mat(oCurrScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter-1].data());
        const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
        for(int nRowIter = oCurrScaleSize.height-1; nRowIter>=-(int)nNMSHalfWinSize; --nRowIter) {
            uchar* anGradRow = oGradMap.data+(nRowIter+nNMSHalfWinSize)*nGradMapRowStep+nGradMapColStep*nNMSHalfWinSize;
            lvDbgAssert(anGradRow>oGradMap.datastart && anGradRow<oGradMap.dataend);
            if(nRowIter>=0) {
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = (size_t)oCurrScaleSize.width-1; nColIter!=size_t(-1); --nColIter) {
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    const uchar* const anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nColLUTIdx;
                    const uchar* const auRefColor = (oPyrMap.data+nColLUTIdx/LBSP::DESC_SIZE_BITS);
                    char nGradX, nGradY;
                    uchar nGradMag;
                    LBSP::computeDescriptor_gradient<nChannels>(anCurrLUT,auRefColor,nGradX,nGradY,nGradMag);
#if USE_MIN_GRAD_ORIENT
                    (char&)(anGradRow[nColIter*nGradMapColStep]) = std::min(nGradX,char(anGradRow[nColIter*nGradMapColStep]),lAbsCharComp);
                    (char&)(anGradRow[nColIter*nGradMapColStep+1]) = std::min(nGradY,char(anGradRow[nColIter*nGradMapColStep+1]),lAbsCharComp);
#else //(!USE_MIN_GRAD_ORIENT)
                    lvDbgAssert((nGradX+(char)(anGradRow[nColIter*nGradMapColStep]*2))/2<=UCHAR_MAX);
                    lvDbgAssert((nGradY+(char)(anGradRow[nColIter*nGradMapColStep+1]*2))/2<=UCHAR_MAX);
                    (char&)(anGradRow[nColIter*nGradMapColStep]) = ((nGradX+(char)(anGradRow[nColIter*nGradMapColStep]*2))/2);
                    (char&)(anGradRow[nColIter*nGradMapColStep+1]) = ((nGradY+(char)(anGradRow[nColIter*nGradMapColStep+1]*2))/2);
#endif //(!USE_MIN_GRAD_ORIENT)
                    anGradRow[nColIter*nGradMapColStep+2] = std::min(nGradMag,anGradRow[nColIter*nGradMapColStep+2]);
                    if(nLevelIter>0) {
                        const size_t nRowIter_base = nRowIter << 1;
                        const size_t nColIter_base = nColIter << 1;
                        lvDbgAssert((nRowIter_base+1)<size_t(oMapSize.height));
                        lvDbgAssert((nColIter_base+1)<size_t(oMapSize.width));
                        lv::unroll<2>([&](int nRowIterOffset){
                            uchar* anNextScaleGradRow = oGradMap.data+(nRowIter_base+nRowIterOffset+nNMSHalfWinSize)*nGradMapRowStep+nGradMapColStep*nNMSHalfWinSize;
                            lvDbgAssert(anNextScaleGradRow<oGradMap.dataend);
                            lv::unroll<2>([&](int nColIterOffset){ // 4ch x 8ub = 32-bit chunks to copy
                                *(uint32_t*)(anNextScaleGradRow+(nColIter_base+nColIterOffset)*nGradMapColStep) = *(uint32_t*)(anGradRow+nColIter*nGradMapColStep);
                            });
                        });
                    }
                }
            }
            if(nLevelIter==0) {
                std::fill(anGradRow-nGradMapColStep*nNMSHalfWinSize,anGradRow,0); // remove if init'd at top
                std::fill(anGradRow+oInputImg.cols*nGradMapColStep,anGradRow+(oInputImg.cols+nNMSHalfWinSize)*nGradMapColStep,0);
                if(nRowIter<oCurrScaleSize.height-int(nNMSHalfWinSize)) {
                    anGradRow += nGradMapRowStep*nNMSHalfWinSize; // offset by nNMSHalfWinSize rows
                    uchar* anEdgeMapRow = oEdgeTempMask.ptr<uchar>(nRowIter+nNMSHalfWinSize)+nNMSHalfWinSize*nEdgeMapColStep;
                    std::fill(anEdgeMapRow-nEdgeMapColStep*nNMSHalfWinSize,anEdgeMapRow,1);
                    std::fill(anEdgeMapRow+oInputImg.cols*nEdgeMapColStep,anEdgeMapRow+(oInputImg.cols+nNMSHalfWinSize)*nEdgeMapColStep,1);
                    stack_check_size(oInputImg.cols);
                    bool nNeighbMax = false;
                    for(size_t nColIter = 0; nColIter<(size_t)oInputImg.cols; ++nColIter) {
                        // make sure all 'quick-idx' lookups are at the right positions...
                        lvDbgAssert(anGradRow[nColIter*nGradMapColStep]==oGradMap.at<cv::Vec4b>(int(nRowIter+(nNMSHalfWinSize*2)),int(nColIter+nNMSHalfWinSize))[0]);
                        lvDbgAssert(anGradRow[nColIter*nGradMapColStep+1]==oGradMap.at<cv::Vec4b>(int(nRowIter+(nNMSHalfWinSize*2)),int(nColIter+nNMSHalfWinSize))[1]);
                        for(int nNMSWinIter=-(int)nNMSHalfWinSize; nNMSWinIter<=(int)nNMSHalfWinSize; ++nNMSWinIter)
                            lvDbgAssert(anGradRow[nColIter*nGradMapColStep+nGradMapRowStep*nNMSWinIter+2]==oGradMap.at<cv::Vec4b>(int(nRowIter+nNMSWinIter+(nNMSHalfWinSize*2)),int(nColIter+nNMSHalfWinSize))[2]);
                        const uchar nGradMag = anGradRow[nColIter*nGradMapColStep+2];
                        if(nGradMag>=nHystLowThreshold) {
#if USE_3_AXIS_ORIENT
                            const char nGradX = ((char*)anGradRow)[nColIter*nGradMapColStep];
                            const char nGradY = ((char*)anGradRow)[nColIter*nGradMapColStep+1];
                            const uint nShift_FPA = 15;
                            constexpr uint nTG22deg_FPA = (int)(0.4142135623730950488016887242097*(1<<nShift_FPA)+0.5); // == tan(pi/8)
                            const uint nGradX_abs = (uint)std::abs(nGradX);
                            const uint nGradY_abs = (uint)std::abs(nGradY)<<nShift_FPA;
                            uint nTG22GradX_FPA = nGradX_abs*nTG22deg_FPA; // == 0.4142135623730950488016887242097*nGradX_abs
                            if(nGradY_abs<nTG22GradX_FPA) { // if(nGradX_abs<0.4142135623730950488016887242097*nGradX_abs) == flat gradient (sector 0)
                                if(lv::isLocalMaximum_Horizontal<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep))
                                    goto _edge_good; // push as 'edge'
                            }
                            else { // else(nGradX_abs>=0.4142135623730950488016887242097*nGradX_abs) == not a flat gradient (sectors 1, 2 or 3)
                                uint nTG67GradX_FPA = nTG22GradX_FPA+(nGradX_abs<<(nShift_FPA+1)); // == 2.4142135623730950488016887242097*nGradX_abs == tan(3*pi/8)*nGradX_abs
                                if(nGradY_abs>nTG67GradX_FPA) { // if(nGradX_abs>2.4142135623730950488016887242097*nGradX_abs == vertical gradient (sector 2)
                                    if(lv::isLocalMaximum_Vertical<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep))
                                        goto _edge_good;
                                }
                                else { // else(nGradX_abs<=2.4142135623730950488016887242097*nGradX_abs == diagonal gradient (sector 1 or 3, depending on grad sign diff)
                                    if(nGradX || nGradY) {
                                        if(lv::isLocalMaximum_Diagonal<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep,(nGradX^nGradY)>=0))
                                            goto _edge_good;
                                    }
                                    else {
                                        if(lv::isLocalMaximum_Diagonal<nNMSHalfWinSize,true>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep) ||
                                           lv::isLocalMaximum_Diagonal<nNMSHalfWinSize,false>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep))
                                            goto _edge_good;
                                    }
                                }
                            }
#else //(!USE_3_AXIS_ORIENT)
                            const uint nGradX_abs = (uint)std::abs(anGradRow[nColIter*nGradMapColStep]);
                            const uint nGradY_abs = (uint)std::abs(anGradRow[nColIter*nGradMapColStep+1]);
                            if((nGradY_abs<=nGradX_abs && isLocalMaximum_Horizontal<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep)) ||
                               (nGradY_abs>nGradX_abs && isLocalMaximum_Vertical<nNMSHalfWinSize>(anGradRow+nColIter*nGradMapColStep+2,nGradMapColStep,nGradMapRowStep)))
                                goto _edge_good;
#endif //(!USE_3_AXIS_ORIENT)
                        }
                        nNeighbMax = false;
                        anEdgeMapRow[nColIter*nEdgeMapColStep] = 1; // not an edge
                        continue;
                        _edge_good:
                        // if not neighbor to previously identified edge, and gradmag above max threshold
                        if(!nNeighbMax && nGradMag>=nHystHighThreshold && anEdgeMapRow[nColIter*nEdgeMapColStep+nEdgeMapRowStep]!=2) {
                            stack_push(anEdgeMapRow+nColIter);
                            nNeighbMax = true;
                            continue;
                        }
                        //_edge_maybe:
                        anEdgeMapRow[nColIter*nEdgeMapColStep] = 0; // might belong to an edge
                    }
                }
            }
        }
    }
    lvDbgAssert(oEdgeTempMask.step.p[0]==nEdgeMapRowStep);
    lvDbgAssert(oEdgeTempMask.step.p[1]==nEdgeMapColStep);
    while(pauHystStack_top>pauHystStack_bottom) {
        stack_check_size(8);
        uchar* pEdgeAddr = stack_pop();
        if(!pEdgeAddr[-intptr_t(nEdgeMapColStep)])
            stack_push(pEdgeAddr-nEdgeMapColStep);
        if(!pEdgeAddr[nEdgeMapColStep])
            stack_push(pEdgeAddr+nEdgeMapColStep);
        if(!pEdgeAddr[-intptr_t(nEdgeMapRowStep)-nEdgeMapColStep])
            stack_push(pEdgeAddr-nEdgeMapRowStep-nEdgeMapColStep);
        if(!pEdgeAddr[-intptr_t(nEdgeMapRowStep)])
            stack_push(pEdgeAddr-nEdgeMapRowStep);
        if(!pEdgeAddr[-intptr_t(nEdgeMapRowStep)+nEdgeMapColStep])
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
    for(int nRowIter=0; nRowIter<oInputImg.rows; ++nRowIter, anEdgeTempMaskData+=nEdgeMapRowStep, oEdgeMaskData+=oEdgeMask.step)
        for(int nColIter=0; nColIter<oInputImg.cols; ++nColIter)
            oEdgeMaskData[nColIter] = (uchar)-(*(anEdgeTempMaskData+nColIter*nEdgeMapColStep)>>1);
}

template void EdgeDetectorLBSP::apply_internal_threshold<1>(const cv::Mat&, cv::Mat&, uchar);
template void EdgeDetectorLBSP::apply_internal_threshold<2>(const cv::Mat&, cv::Mat&, uchar);
template void EdgeDetectorLBSP::apply_internal_threshold<3>(const cv::Mat&, cv::Mat&, uchar);
template void EdgeDetectorLBSP::apply_internal_threshold<4>(const cv::Mat&, cv::Mat&, uchar);

void EdgeDetectorLBSP::apply_internal_threshold(const cv::Mat& oInputImg, cv::Mat& oEdgeMask, uchar nDetThreshold, size_t nChannels) {
    if(nChannels==1)
        apply_internal_threshold<1>(oInputImg,oEdgeMask,nDetThreshold);
    else if(nChannels==2)
        apply_internal_threshold<2>(oInputImg,oEdgeMask,nDetThreshold);
    else if(nChannels==3)
        apply_internal_threshold<3>(oInputImg,oEdgeMask,nDetThreshold);
    else if(nChannels==4)
        apply_internal_threshold<4>(oInputImg,oEdgeMask,nDetThreshold);
    else
        CV_Error(-1,"Unexpected channel count");
}

void EdgeDetectorLBSP::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dDetThreshold) {
    cv::Mat oInputImg = _oInputImage.getMat();
    lvAssert_(!oInputImg.empty() && oInputImg.isContinuous(),"input image must be non-empty and continuous");
    lvAssert_(oInputImg.depth()==CV_8U,"input image depth must be 8U")
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
    apply_internal_lookup(oInputImg,oInputImg.channels());
    apply_internal_threshold(oInputImg,oEdgeMask,nDetThreshold,oInputImg.channels());
}

void EdgeDetectorLBSP::apply(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask) {
    cv::Mat oInputImg = _oInputImage.getMat();
    lvAssert_(!oInputImg.empty() && oInputImg.isContinuous(),"input image must be non-empty and continuous");
    lvAssert_(oInputImg.depth()==CV_8U,"input image depth must be 8U")
    if(m_dGaussianKernelSigma>0) {
        const int nDefaultKernelSize = int(8*ceil(m_dGaussianKernelSigma));
        const int nRealKernelSize = nDefaultKernelSize%2==0?nDefaultKernelSize+1:nDefaultKernelSize;
        oInputImg = oInputImg.clone();
        cv::GaussianBlur(oInputImg,oInputImg,cv::Size(nRealKernelSize,nRealKernelSize),m_dGaussianKernelSigma,m_dGaussianKernelSigma);
    }
    apply_internal_lookup(oInputImg,oInputImg.channels());
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    oEdgeMask = cv::Scalar_<uchar>(0);
    cv::Mat oTempEdgeMask = oEdgeMask.clone();
    for(size_t nCurrThreshold=0; nCurrThreshold<LBSP::MAX_GRAD_MAG; ++nCurrThreshold) {
        apply_internal_threshold(oInputImg,oTempEdgeMask,uchar(nCurrThreshold),oInputImg.channels());
        oEdgeMask += oTempEdgeMask/double(LBSP::MAX_GRAD_MAG);
    }
    if(m_bNormalizeOutput)
        cv::normalize(oEdgeMask,oEdgeMask,0,UCHAR_MAX,cv::NORM_MINMAX);
}
