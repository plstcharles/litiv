
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
#include "litiv/features2d/LBSP.hpp"

#define USE_SINGLE_SCALE          0
#if USE_SINGLE_SCALE

#else //!USE_SINGLE_SCALE
#define USE_PREPROC_PYR_DISPLAY   1
#if USE_PREPROC_PYR_DISPLAY
#define USE_REL_LBSP                  0
#define DEFAULT_LBSP_REL_THRESHOLD    0.50f
#endif //USE_PREPROC_PYR_DISPLAY
#define USE_NMS_HYST_CANNY        0
#if (USE_PREPROC_PYR_DISPLAY+USE_NMS_HYST_CANNY)!=1
#error "edge detector lbsp internal cfg error"
#endif //(USE_...+...)!=1
#endif //!USE_SINGLE_SCALE

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
    const cv::Size oOrigInputSize = oInputImg.size();
    const int nOrigType = CV_8UC(int(nChannels));
    const size_t nColLUTStep = LBSP::DESC_SIZE_BITS*nChannels;
#if USE_SINGLE_SCALE
    const size_t nColDescStep = nChDescStep*nChannels;
    const size_t nOrigMapSize = oOrigInputSize.area()*nChannels;
    const size_t nRows = size_t(oInputImg.rows);
    const size_t nCols = size_t(oInputImg.cols);
    const size_t nRowDescStep = nColDescStep*nCols;
    const size_t nRowLUTStep = nColLUTStep*nCols;
    const cv::Size oSize((int)nCols,(int)nRows);
    const size_t nMapSize = oSize.area()*nChannels;
    m_vvuLBSPLookupMaps[0].resize(nMapSize*nChannels*LBSP::DESC_SIZE_BITS);
    m_voMapSizeList[0] = oSize;
    const cv::Mat oInputMap = oInputImg;
    for(size_t nRowIter = 0; nRowIter<nRows; ++nRowIter) {
        const size_t nRowDescIdx = nRowIter*nRowDescStep;
        const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
        for(size_t nColIter = 0; nColIter<nCols; ++nColIter) {
            const size_t nColDescIdx = nRowDescIdx+nColIter*nColDescStep;
            const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
            uchar* aanCurrLUT = m_vvuLBSPLookupMaps[0].data()+nColLUTIdx;
            if( nRowIter<m_nROIBorderSize || nRowIter>=nRows-m_nROIBorderSize ||
                nColIter<m_nROIBorderSize || nColIter>=nCols-m_nROIBorderSize) {
                CV_DbgAssert(nColDescIdx/2<size_t(oInputMap.dataend-oInputMap.datastart));
                const uchar* const auInputRef = oInputMap.data+nColDescIdx/2;
                for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                    uchar* anCurrChLUT = aanCurrLUT+nChIter*LBSP::DESC_SIZE_BITS;
                    memset(anCurrChLUT,auInputRef[nChIter],LBSP::DESC_SIZE_BITS);
                }
            }
            else {
                CV_DbgAssert(nColLUTIdx<m_vvuLBSPLookupMaps[0].size() && (nColLUTIdx%LBSP::DESC_SIZE_BITS)==0);
                LBSP::computeDescriptor_lookup<nChannels>(oInputMap,int(nColIter),int(nRowIter),aanCurrLUT);
            }
        }
    }
    std::aligned_vector<uchar,32> vuDisplayPairsData(nMapSize);
    std::aligned_vector<float,32> vuAbsLBSPDescGradSumData(nMapSize);
    std::aligned_vector<float,32> vuRelLBSPDescGradSumData(nMapSize);
    std::aligned_vector<ushort,32> vuAbsLBSPDescMapData(nMapSize);
    std::aligned_vector<ushort,32> vuRelLBSPDescMapData(nMapSize);
    std::aligned_vector<uchar,32> vuAbsLBSPDescGradMagMapData(nMapSize);
    std::aligned_vector<uchar,32> vuRelLBSPDescGradMagMapData(nMapSize);
    cv::Mat oTempInputCopyMat(oOrigInputSize,nOrigType);
    cv::Mat oDisplayPair;
    cv::Mat oAbsLBSPDescGradSum(oOrigInputSize,CV_32FC(nChannels),vuAbsLBSPDescGradSumData.data());
    cv::Mat oRelLBSPDescGradSum(oOrigInputSize,CV_32FC(nChannels),vuRelLBSPDescGradSumData.data());
    oAbsLBSPDescGradSum = cv::Scalar_<float>::all(0.0f);
    oRelLBSPDescGradSum = cv::Scalar_<float>::all(0.0f);
    cv::Mat oAbsLBSPDescMap(oSize,CV_16UC(int(nChannels)),vuAbsLBSPDescMapData.data());
    cv::Mat oRelLBSPDescMap(oSize,CV_16UC(int(nChannels)),vuRelLBSPDescMapData.data());
    cv::Mat oAbsLBSPDescGradMagMap(oSize,nOrigType,vuAbsLBSPDescGradMagMapData.data());
    cv::Mat oRelLBSPDescGradMagMap(oSize,nOrigType,vuRelLBSPDescGradMagMapData.data());
    for(size_t nRowIter = 0; nRowIter<nRows; ++nRowIter) {
        const size_t nRowDescIdx = nRowIter*nRowDescStep;
        const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
        for(size_t nColIter = 0; nColIter<nCols; ++nColIter) {
            const size_t nColDescIdx = nRowDescIdx+nColIter*nColDescStep;
            const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
            for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                const size_t nChDescIdx = nColDescIdx+nChIter*nChDescStep;
                const uchar* const anCurrLUT = m_vvuLBSPLookupMaps[0].data()+nColLUTIdx;
                CV_DbgAssert(nChDescIdx<size_t(oAbsLBSPDescMap.dataend-oAbsLBSPDescMap.datastart));
                ushort& nAbsLBSPDesc = *(ushort*)(oAbsLBSPDescMap.data+nChDescIdx);
                ushort& nRelLBSPDesc = *(ushort*)(oRelLBSPDescMap.data+nChDescIdx);
                const uchar* const auRefColor = (oInputMap.data+nColDescIdx/2);
                LBSP::computeDescriptor_threshold_max<nChannels>(anCurrLUT,auRefColor,nThreshold,nAbsLBSPDesc);
                LBSP::computeDescriptor_threshold_max<1,nChannels>(anCurrLUT,auRefColor,nRelLBSPDesc);
            }
            for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                const size_t nChDescIdx = nColDescIdx+nChIter*nChDescStep;
                CV_DbgAssert(nChDescIdx<size_t(oAbsLBSPDescMap.dataend-oAbsLBSPDescMap.datastart));
                ushort& nAbsLBSPDesc = *(ushort*)(oAbsLBSPDescMap.data+nChDescIdx);
                ushort& nRelLBSPDesc = *(ushort*)(oRelLBSPDescMap.data+nChDescIdx);
                CV_DbgAssert(nChDescIdx/2<size_t(oAbsLBSPDescGradMagMap.dataend-oAbsLBSPDescGradMagMap.datastart));
                *(oAbsLBSPDescGradMagMap.data+nChDescIdx/2) = (uchar)(DistanceUtils::popcount(nAbsLBSPDesc)*float(UCHAR_MAX)/LBSP::DESC_SIZE_BITS);
                *(oRelLBSPDescGradMagMap.data+nChDescIdx/2) = (uchar)(DistanceUtils::popcount(nRelLBSPDesc)*float(UCHAR_MAX)/LBSP::DESC_SIZE_BITS);
            }
        }
    }
    cv::add(oAbsLBSPDescGradSum,oAbsLBSPDescGradMagMap,oAbsLBSPDescGradSum,cv::Mat(),CV_32F);
    cv::add(oRelLBSPDescGradSum,oRelLBSPDescGradMagMap,oRelLBSPDescGradSum,cv::Mat(),CV_32F);
    oDisplayPair = cv::Mat(oSize,nOrigType,vuDisplayPairsData.data());
    cv::hconcat(oInputMap,oAbsLBSPDescGradMagMap,oDisplayPair);
    cv::hconcat(oDisplayPair,oRelLBSPDescGradMagMap,oDisplayPair);
    cv::resize(oDisplayPair,oDisplayPair,cv::Size(oInputImg.cols*3,oInputImg.rows),0,0,cv::INTER_NEAREST);

    cv::Mat oAbsLBSPDescGradSumNorm,oRelLBSPDescGradSumNorm;
    cv::normalize(oAbsLBSPDescGradSum/m_nLevels,oAbsLBSPDescGradSumNorm,0,255,cv::NORM_MINMAX,CV_8U);//oAbsLBSPDescGradSum.convertTo(oAbsLBSPDescGradSumNorm,CV_8U,1.0/m_nLevels);//
    cv::normalize(oRelLBSPDescGradSum/m_nLevels,oRelLBSPDescGradSumNorm,0,255,cv::NORM_MINMAX,CV_8U);//oRelLBSPDescGradSum.convertTo(oRelLBSPDescGradSumNorm,CV_8U,1.0/m_nLevels);//
    cv::Mat oMixLBSPDescGradSumNorm = (oAbsLBSPDescGradSumNorm+oRelLBSPDescGradSumNorm)/2;
    cv::Mat oDisplayLBSPDescGradSumNorm;
    cv::hconcat(oAbsLBSPDescGradSumNorm,oRelLBSPDescGradSumNorm,oDisplayLBSPDescGradSumNorm);
    cv::hconcat(oDisplayLBSPDescGradSumNorm,oMixLBSPDescGradSumNorm,oDisplayLBSPDescGradSumNorm);
    cv::imshow("GradSumNorm abs/rel/mix",oDisplayLBSPDescGradSumNorm);

    cv::Mat dx(oSize,CV_16SC3);
    cv::Mat dy(oSize,CV_16SC3);
    const int winsize = 3;
    cv::Sobel(oInputMap, dx, CV_16S, 1, 0, winsize, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(oInputMap, dy, CV_16S, 0, 1, winsize, 1, 0, cv::BORDER_REPLICATE);
    cv::Mat norm;
    cv::add(cv::abs(dx),cv::abs(dy),norm,cv::noArray(),CV_32SC1);
    std::vector<cv::Mat> vchn(3);
    cv::split(norm,vchn);
    cv::Mat final = cv::max(cv::max(vchn[0],vchn[1]),vchn[2]);
    cv::Mat final_byte;
    cv::normalize(final,final_byte,0,255,cv::NORM_MINMAX,CV_8UC1);
    cv::imshow("final_byte",final_byte);
    cv::waitKey(0);
#else //!USE_SINGLE_SCALE
    size_t nNextScaleRows = size_t(oInputImg.rows);
    size_t nNextScaleCols = size_t(oInputImg.cols);
    size_t nNextRowDescStep = LBSP::DESC_SIZE*nNextScaleCols;
    size_t nNextRowLUTStep = nColLUTStep*nNextScaleCols;
    CV_DbgAssert((nNextRowLUTStep/((LBSP::DESC_SIZE_BITS/LBSP::DESC_SIZE)*nChannels))==nNextRowDescStep);
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
        nNextRowDescStep = LBSP::DESC_SIZE*nNextScaleCols;
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
    bool bUsingRelLBSP = USE_REL_LBSP;
    float fRelLBSPThreshold = DEFAULT_LBSP_REL_THRESHOLD;
    uchar nAbsLBSPThreshold = cv::saturate_cast<uchar>(nThreshold);
    const size_t nOrigMapSize = oOrigInputSize.area();
    std::vector<std::aligned_vector<uchar,32>> vvuDisplayPairsData(m_nLevels,std::aligned_vector<uchar,32>(nOrigMapSize));
    std::aligned_vector<ushort,32> vuAbsLBSPDescGradSumData(nOrigMapSize);
    std::aligned_vector<ushort,32> vuRelLBSPDescGradSumData(nOrigMapSize);
    std::aligned_vector<ushort,32> vuAbsLBSPDescMapData(nOrigMapSize);
    std::aligned_vector<ushort,32> vuRelLBSPDescMapData(nOrigMapSize);
    std::aligned_vector<uchar,32> vuAbsLBSPDescGradMagMapData(nOrigMapSize);
    std::aligned_vector<uchar,32> vuRelLBSPDescGradMagMapData(nOrigMapSize);
    std::aligned_vector<uchar,32> vuAbsLBSPDescGradOrientMapData(nOrigMapSize);
    std::aligned_vector<uchar,32> vuRelLBSPDescGradOrientMapData(nOrigMapSize);
    cv::Mat oTempInputCopyMat(oOrigInputSize,nOrigType);
    int nKeyPressed = 0;
    while((char)nKeyPressed!=27) { // escape breaks
        if(bUsingRelLBSP) {
            if((char)nKeyPressed=='u' && fRelLBSPThreshold<1.0f) {
                fRelLBSPThreshold = std::min(fRelLBSPThreshold+0.01f,1.0f);
                std::cout << "fRelLBSPThreshold = " << fRelLBSPThreshold << std::endl;
            }
            else if((char)nKeyPressed=='d' && fRelLBSPThreshold>0.0f) {
                fRelLBSPThreshold = std::max(fRelLBSPThreshold-0.01f,0.0f);
                std::cout << "fRelLBSPThreshold = " << fRelLBSPThreshold << std::endl;
            }
        }
        else {
            if((char)nKeyPressed=='u' && nAbsLBSPThreshold<UCHAR_MAX)
                std::cout << "nAbsLBSPThreshold = " << ++nAbsLBSPThreshold << std::endl;
            else if((char)nKeyPressed=='d' && nAbsLBSPThreshold>0)
                std::cout << "nAbsLBSPThreshold = " << --nAbsLBSPThreshold << std::endl;
        }
        if((char)nKeyPressed=='b') {
            bUsingRelLBSP = !bUsingRelLBSP;
            std::cout << "bUsingRelLBSP = " << bUsingRelLBSP << std::endl;
        }
        std::vector<cv::Mat> voDisplayPairs(m_nLevels);
        cv::Mat oAbsLBSPDescGradSum(oOrigInputSize,CV_16UC1,vuAbsLBSPDescGradSumData.data());
        cv::Mat oRelLBSPDescGradSum(oOrigInputSize,CV_16UC1,vuRelLBSPDescGradSumData.data());
        for(size_t nLevelIter=m_nLevels-1; nLevelIter!=size_t(-1); --nLevelIter) {
            const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
            cv::Mat oAbsLBSPDescMap(oCurrScaleSize,CV_16UC1,vuAbsLBSPDescMapData.data());
            cv::Mat oRelLBSPDescMap(oCurrScaleSize,CV_16UC1,vuRelLBSPDescMapData.data());
            cv::Mat oAbsLBSPDescGradMagMap(oCurrScaleSize,CV_8UC1,vuAbsLBSPDescGradMagMapData.data());
            cv::Mat oRelLBSPDescGradMagMap(oCurrScaleSize,CV_8UC1,vuRelLBSPDescGradMagMapData.data());
            cv::Mat oAbsLBSPDescGradOrientMap(oCurrScaleSize,CV_8UC1,vuAbsLBSPDescGradOrientMapData.data());
            cv::Mat oRelLBSPDescGradOrientMap(oCurrScaleSize,CV_8UC1,vuRelLBSPDescGradOrientMapData.data());
            cv::Mat oPyrMap = !nLevelIter?oInputImg:cv::Mat(oCurrScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter-1].data());
            const size_t nRowDescStep = LBSP::DESC_SIZE*(size_t)oCurrScaleSize.width;
            const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
            for(size_t nRowIter = 0; nRowIter<(size_t)oCurrScaleSize.height; ++nRowIter) {
                const size_t nRowDescIdx = nRowIter*nRowDescStep;
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = 0; nColIter<(size_t)oCurrScaleSize.width; ++nColIter) {
                    const size_t nColDescIdx = nRowDescIdx+nColIter*LBSP::DESC_SIZE;
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    const uchar* const anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nColLUTIdx;
                    CV_DbgAssert(nColDescIdx<size_t(oAbsLBSPDescMap.dataend-oAbsLBSPDescMap.datastart));
                    ushort& nAbsLBSPDesc = *(ushort*)(oAbsLBSPDescMap.data+nColDescIdx);
                    ushort& nRelLBSPDesc = *(ushort*)(oRelLBSPDescMap.data+nColDescIdx);
                    const uchar* const auRefColor = (oPyrMap.data+nColLUTIdx/LBSP::DESC_SIZE_BITS);
                    LBSP::computeDescriptor_threshold_max<nChannels>(anCurrLUT,auRefColor,nAbsLBSPThreshold,nAbsLBSPDesc);
                    LBSP::computeDescriptor_threshold_max_rel<2,nChannels>(anCurrLUT,auRefColor,nRelLBSPDesc);
                    //MAJORITY VOTE ON ORIENTATION FROM EACH LEVEL HERE
                    LBSP::computeDescriptor_orientation(nAbsLBSPDesc);
                    LBSP::computeDescriptor_orientation(nRelLBSPDesc);

                    CV_DbgAssert(UCHAR_MAX>LBSP::DESC_SIZE_BITS); // make sure mag map precision is greater than popcount precision
                    *(oAbsLBSPDescGradMagMap.data+nColDescIdx/2) = (DistanceUtils::popcount(nAbsLBSPDesc)*UCHAR_MAX)/LBSP::DESC_SIZE_BITS;
                    *(oRelLBSPDescGradMagMap.data+nColDescIdx/2) = (DistanceUtils::popcount(nRelLBSPDesc)*UCHAR_MAX)/LBSP::DESC_SIZE_BITS;
                }
            }
            if(nLevelIter) {
                cv::resize(oAbsLBSPDescGradMagMap,oTempInputCopyMat,oOrigInputSize,0,0,cv::INTER_NEAREST);
                if(nLevelIter==m_nLevels-1)
                    oTempInputCopyMat.convertTo(oAbsLBSPDescGradSum,CV_16U);
                else
                    cv::add(oAbsLBSPDescGradSum,oTempInputCopyMat,oAbsLBSPDescGradSum,cv::Mat(),CV_16UC1);
                cv::resize(oRelLBSPDescGradMagMap,oTempInputCopyMat,oOrigInputSize,0,0,cv::INTER_NEAREST);
                if(nLevelIter==m_nLevels-1)
                    oTempInputCopyMat.convertTo(oRelLBSPDescGradSum,CV_16U);
                else
                    cv::add(oRelLBSPDescGradSum,oTempInputCopyMat,oRelLBSPDescGradSum,cv::Mat(),CV_16UC1);
            }
            else {
                cv::add(oAbsLBSPDescGradSum,oAbsLBSPDescGradMagMap,oAbsLBSPDescGradSum,cv::Mat(),CV_16UC1);
                cv::add(oRelLBSPDescGradSum,oRelLBSPDescGradMagMap,oRelLBSPDescGradSum,cv::Mat(),CV_16UC1);
            }
            // display purposes below only
            voDisplayPairs[nLevelIter] = cv::Mat(oCurrScaleSize,CV_8UC1,vvuDisplayPairsData[nLevelIter].data());
            cv::Mat oAbsLBSPDescGradMagMap_conv, oRelLBSPDescGradMagMap_conv;
            if(oPyrMap.channels()==1) {
                oAbsLBSPDescGradMagMap_conv = oAbsLBSPDescGradMagMap;
                oRelLBSPDescGradMagMap_conv = oRelLBSPDescGradMagMap;
            }
            else if(oPyrMap.channels()==2) {
                cv::merge(std::vector<cv::Mat>{oAbsLBSPDescGradMagMap,oAbsLBSPDescGradMagMap},oAbsLBSPDescGradMagMap_conv);
                cv::merge(std::vector<cv::Mat>{oRelLBSPDescGradMagMap,oRelLBSPDescGradMagMap},oRelLBSPDescGradMagMap_conv);
            }
            else if(oPyrMap.channels()==3) {
                cv::cvtColor(oAbsLBSPDescGradMagMap,oAbsLBSPDescGradMagMap_conv,cv::COLOR_GRAY2BGR);
                cv::cvtColor(oRelLBSPDescGradMagMap,oRelLBSPDescGradMagMap_conv,cv::COLOR_GRAY2BGR);
            }
            else if(oPyrMap.channels()==4) {
                cv::cvtColor(oAbsLBSPDescGradMagMap,oAbsLBSPDescGradMagMap_conv,cv::COLOR_GRAY2BGRA);
                cv::cvtColor(oRelLBSPDescGradMagMap,oRelLBSPDescGradMagMap_conv,cv::COLOR_GRAY2BGRA);
            }
            cv::hconcat(oPyrMap,oAbsLBSPDescGradMagMap_conv,voDisplayPairs[nLevelIter]);
            cv::hconcat(voDisplayPairs[nLevelIter],oRelLBSPDescGradMagMap_conv,voDisplayPairs[nLevelIter]);
            cv::resize(voDisplayPairs[nLevelIter],voDisplayPairs[nLevelIter],cv::Size(oInputImg.cols*3,oInputImg.rows),0,0,cv::INTER_NEAREST);
            //std::stringstream ssWinName;
            //ssWinName << "L[" << nLevelIter << "] in/abs/rel";
            //cv::imshow(ssWinName.str(),voDisplayPairs[nLevelIter]);

        }
        cv::Mat oAbsLBSPDescGradSumNorm,oRelLBSPDescGradSumNorm;
        cv::normalize(oAbsLBSPDescGradSum/m_nLevels,oAbsLBSPDescGradSumNorm,0,255,cv::NORM_MINMAX,CV_8U);//oAbsLBSPDescGradSum.convertTo(oAbsLBSPDescGradSumNorm,CV_8U,1.0/m_nLevels);//
        cv::normalize(oRelLBSPDescGradSum/m_nLevels,oRelLBSPDescGradSumNorm,0,255,cv::NORM_MINMAX,CV_8U);//oRelLBSPDescGradSum.convertTo(oRelLBSPDescGradSumNorm,CV_8U,1.0/m_nLevels);//
        cv::Mat oMixLBSPDescGradSumNorm = (oAbsLBSPDescGradSumNorm+oRelLBSPDescGradSumNorm)/2;
        cv::Mat oDisplayLBSPDescGradSumNorm;
        cv::hconcat(oAbsLBSPDescGradSumNorm,oRelLBSPDescGradSumNorm,oDisplayLBSPDescGradSumNorm);
        cv::hconcat(oDisplayLBSPDescGradSumNorm,oMixLBSPDescGradSumNorm,oDisplayLBSPDescGradSumNorm);
        cv::imshow("GradSumNorm abs/rel/mix",oDisplayLBSPDescGradSumNorm);
        /*std::vector<cv::Mat> test;
        cv::split(oDisplayLBSPDescGradSumNorm,test);
        cv::Mat oDisplayLBSPDescGradSumNorm_NMS;
        litiv::nonMaxSuppression<5>(test[0],oDisplayLBSPDescGradSumNorm_NMS);
        cv::imshow("GradSumNorm abs/rel/mix NMS",oDisplayLBSPDescGradSumNorm_NMS);*/
        nKeyPressed = cv::waitKey(0);
    }
#elif USE_NMS_HYST_CANNY

    //@@@@ WiP

#endif //USE_NMS_HYST_CANNY
#endif //!USE_SINGLE_SCALE
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
