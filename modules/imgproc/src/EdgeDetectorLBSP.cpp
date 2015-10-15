
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
#include <opencv2/highgui.hpp>

#define NB_THRESHOLD_BINS             UCHAR_MAX
#define USE_REL_LBSP                  0
#define DEFAULT_LBSP_ABS_ThRESHOLD    30
#define DEFAULT_LBSP_REL_ThRESHOLD    0.450f

#define USE_PREPROC_PYR_DISPLAY   1
#define USE_NMS_MJVOTE            0
#define USE_MBLUR_MJVOTE_CFG      0

#if (USE_PREPROC_PYR_DISPLAY+USE_MBLUR_MJVOTE_CFG+USE_NMS_MJVOTE)!=1
#error "edge detector lbsp internal cfg error"
#endif //(USE_...+...)!=1

EdgeDetectorLBSP::EdgeDetectorLBSP(size_t nLevels) :
        EdgeDetector(LBSP::PATCH_SIZE/2),
        m_nLevels(nLevels),
        m_vvuInputPyrMaps(std::max(nLevels,size_t(1))-1),
        m_vvuLBSPLookupMaps(nLevels),
        m_voMapSizeList(nLevels),
        m_dHystLowThrshFactor(0),
        m_dGaussianKernelSigma(0),
        m_bUsingL2GradientNorm(0) {
    CV_Assert(m_nLevels>0);
}

void EdgeDetectorLBSP::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dThreshold) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.channels()==1 || oInputImg.channels()==3 || oInputImg.channels()==4);
    cv::Mat oInputImage_gray;
    if(oInputImg.channels()==3)
        cv::cvtColor(oInputImg,oInputImage_gray,cv::COLOR_BGR2GRAY);
    else if(oInputImg.channels()==4)
        cv::cvtColor(oInputImg,oInputImage_gray,cv::COLOR_BGRA2GRAY);
    else
        oInputImage_gray = oInputImg;
    if(m_dGaussianKernelSigma>0) {
        const int nDefaultKernelSize = int(8*ceil(m_dGaussianKernelSigma));
        const int nRealKernelSize = nDefaultKernelSize%2==0?nDefaultKernelSize+1:nDefaultKernelSize;
        cv::GaussianBlur(oInputImage_gray,oInputImage_gray,cv::Size(nRealKernelSize,nRealKernelSize),m_dGaussianKernelSigma,m_dGaussianKernelSigma);
    }
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    if(dThreshold<0||dThreshold>1)
        dThreshold = getDefaultThreshold();
//    const size_t nCurrBaseHystThreshold = (size_t)(dThreshold*UCHAR_MAX);
//    cv::Canny(oInputImage_gray,oEdgeMask,nCurrBaseHystThreshold*m_dHystLowThrshFactor,(double)nCurrBaseHystThreshold,3,m_bUsingL2GradientNorm);
}

void EdgeDetectorLBSP::apply(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.channels()==1 || oInputImg.channels()==3 || oInputImg.channels()==4);
    CV_Assert(oInputImg.isContinuous());
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    //oEdgeMask = cv::Scalar_<uchar>(0);

    const size_t nBorderSize = LBSP::PATCH_SIZE/2;
    const size_t nChannels = (size_t)oInputImg.channels();
    const cv::Size oOrigInputSize = oInputImg.size();
    const size_t nOrigMapSize = oOrigInputSize.area()*nChannels;
    const int nOrigType = CV_8UC(int(nChannels));
    const size_t nChDescStep = LBSP::DESC_SIZE;
    const size_t nColDescStep = nChDescStep*nChannels;
    const size_t nChLUTStep = LBSP::DESC_SIZE_BITS;
    const size_t nColLUTStep = nChLUTStep*nChannels;

    CV_Assert(nChannels==3); // cheat @@@@
    size_t nNextScaleRows = size_t(oInputImg.rows);
    size_t nNextScaleCols = size_t(oInputImg.cols);
    size_t nNextRowDescStep = nColDescStep*nNextScaleCols;
    size_t nNextRowLUTStep = nColLUTStep*nNextScaleCols;
    cv::Size oNextScaleSize((int)nNextScaleCols,(int)nNextScaleRows);
    size_t nNextScaleMapSize = oNextScaleSize.area()*nChannels;
    m_vvuLBSPLookupMaps[0].resize(nNextScaleMapSize*LBSP::DESC_SIZE_BITS);
    m_voMapSizeList[0] = oNextScaleSize;
    cv::Mat oNextPyrInputMap = oInputImg;
    for(size_t nLevelIter=0; nLevelIter<m_nLevels; ++nLevelIter) {
        if(!nNextScaleMapSize)
            break;
        const size_t nCurrScaleRows = nNextScaleRows;
        const size_t nCurrScaleCols = nNextScaleCols;
        const size_t nCurrRowDescStep = nNextRowDescStep;
        const size_t nCurrRowLUTStep = nNextRowLUTStep;
        const cv::Mat oCurrPyrInputMap = oNextPyrInputMap;
        nNextScaleRows = (nCurrScaleRows+1)/2;
        nNextScaleCols = (nCurrScaleCols+1)/2;
        nNextRowDescStep = nColDescStep*nNextScaleCols;
        nNextRowLUTStep = nColLUTStep*nNextScaleCols;
        oNextScaleSize = cv::Size((int)nNextScaleCols,(int)nNextScaleRows);
        nNextScaleMapSize = oNextScaleSize.area()*nChannels;
        if(nLevelIter+1<m_nLevels && nNextScaleMapSize) {
            m_vvuInputPyrMaps[nLevelIter].resize(nNextScaleMapSize);
            m_vvuLBSPLookupMaps[nLevelIter+1].resize(nNextScaleMapSize*LBSP::DESC_SIZE_BITS);
            m_voMapSizeList[nLevelIter+1] = oNextScaleSize;
            oNextPyrInputMap = cv::Mat(oNextScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter].data());
            CV_DbgAssert(size_t(oNextScaleSize.area()*nChannels)==m_vvuInputPyrMaps[nLevelIter].size());
        }
        std::cout << "L=" << nLevelIter << "; [" << nCurrScaleCols << "," << nCurrScaleRows << "]" << std::endl;
        for(size_t nRowIter = 0; nRowIter<nCurrScaleRows; ++nRowIter) {
            const size_t nCurrRowDescIdx = nRowIter*nCurrRowDescStep;
            const size_t nCurrRowLUTIdx = nRowIter*nCurrRowLUTStep;
            for(size_t nColIter = 0; nColIter<nCurrScaleCols; ++nColIter) {
                const size_t nCurrColDescIdx = nCurrRowDescIdx+nColIter*nColDescStep;
                const size_t nCurrColLUTIdx = nCurrRowLUTIdx+nColIter*nColLUTStep;
                uchar* aanCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nCurrColLUTIdx;
                if( nRowIter<nBorderSize || nRowIter>=nCurrScaleRows-LBSP::PATCH_SIZE/2 ||
                    nColIter<nBorderSize || nColIter>=nCurrScaleCols-LBSP::PATCH_SIZE/2) {
                    CV_DbgAssert(nCurrColDescIdx/2<size_t(oCurrPyrInputMap.dataend-oCurrPyrInputMap.datastart));
                    const uchar* const auInputRef = oCurrPyrInputMap.data+nCurrColDescIdx/2;
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                        uchar* anCurrChLUT = aanCurrLUT+nChIter*nChLUTStep;
                        memset(anCurrChLUT,auInputRef[nChIter],nChLUTStep);
                    }
                }
                else {
                    CV_DbgAssert(nCurrColLUTIdx<m_vvuLBSPLookupMaps[nLevelIter].size() && (nCurrColLUTIdx%LBSP::DESC_SIZE_BITS)==0);
                    LBSP::computeDescriptor_lookup<3>(oCurrPyrInputMap,int(nColIter),int(nRowIter),aanCurrLUT);
                }
                if(nLevelIter+1<m_nLevels && nNextScaleMapSize && !(nRowIter%2) && !(nColIter%2)) {
                    const size_t nNextColDescIdx = (nRowIter/2)*nNextRowDescStep + (nColIter/2)*nColDescStep;
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                        uchar* anCurrChLUT = aanCurrLUT+nChIter*nChLUTStep;
                        size_t nLUTSum = 0;
                        for(size_t nLUTIter = 0; nLUTIter<LBSP::DESC_SIZE_BITS; ++nLUTIter)
                            nLUTSum += anCurrChLUT[nLUTIter];
                        const size_t nNextChDescIdx = nNextColDescIdx + nChIter*nChDescStep;
                        CV_DbgAssert(nNextChDescIdx/2<size_t(oNextPyrInputMap.dataend-oNextPyrInputMap.datastart));
                        *(oNextPyrInputMap.data+nNextChDescIdx/2) = uchar(nLUTSum/LBSP::DESC_SIZE_BITS);
                    }
                }
            }
        }
    }
#if USE_PREPROC_PYR_DISPLAY
    bool bUsingRelLBSP = USE_REL_LBSP;
    float fRelLBSPThreshold = DEFAULT_LBSP_REL_ThRESHOLD;
    size_t nAbsLBSPThreshold = DEFAULT_LBSP_ABS_ThRESHOLD;
    std::vector<std::aligned_vector<uchar,32>> vvuDisplayPairsData(m_nLevels,std::aligned_vector<uchar,32>(nOrigMapSize));
    std::aligned_vector<float,32> vuAbsLBSPDescGradSumData(nOrigMapSize);
    std::aligned_vector<float,32> vuRelLBSPDescGradSumData(nOrigMapSize);
    std::aligned_vector<ushort,32> vuAbsLBSPDescMapData(nOrigMapSize);
    std::aligned_vector<ushort,32> vuRelLBSPDescMapData(nOrigMapSize);
    std::aligned_vector<uchar,32> vuAbsLBSPDescGradMapData(nOrigMapSize);
    std::aligned_vector<uchar,32> vuRelLBSPDescGradMapData(nOrigMapSize);
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
            if((char)nKeyPressed=='u' && nAbsLBSPThreshold<255)
                std::cout << "nAbsLBSPThreshold = " << ++nAbsLBSPThreshold << std::endl;
            else if((char)nKeyPressed=='d' && nAbsLBSPThreshold>0)
                std::cout << "nAbsLBSPThreshold = " << --nAbsLBSPThreshold << std::endl;
        }
        if((char)nKeyPressed=='b') {
            bUsingRelLBSP = !bUsingRelLBSP;
            std::cout << "bUsingRelLBSP = " << bUsingRelLBSP << std::endl;
        }
        std::vector<cv::Mat> voDisplayPairs(m_nLevels);
        cv::Mat oAbsLBSPDescGradSum(oOrigInputSize,CV_32FC(nChannels),vuAbsLBSPDescGradSumData.data());
        cv::Mat oRelLBSPDescGradSum(oOrigInputSize,CV_32FC(nChannels),vuRelLBSPDescGradSumData.data());
        oAbsLBSPDescGradSum = cv::Scalar_<float>::all(0.0f);
        oRelLBSPDescGradSum = cv::Scalar_<float>::all(0.0f);
        for(size_t nLevelIter=m_nLevels-1; nLevelIter!=size_t(-1); --nLevelIter) {
            const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
            cv::Mat oAbsLBSPDescMap(oCurrScaleSize,CV_16UC(int(nChannels)),vuAbsLBSPDescMapData.data());
            cv::Mat oRelLBSPDescMap(oCurrScaleSize,CV_16UC(int(nChannels)),vuRelLBSPDescMapData.data());
            cv::Mat oAbsLBSPDescGradMap(oCurrScaleSize,nOrigType,vuAbsLBSPDescGradMapData.data());
            cv::Mat oRelLBSPDescGradMap(oCurrScaleSize,nOrigType,vuRelLBSPDescGradMapData.data());
            cv::Mat oPyrMap = !nLevelIter?oInputImg:cv::Mat(oCurrScaleSize,nOrigType,m_vvuInputPyrMaps[nLevelIter-1].data());
            const size_t nRowDescStep = nColDescStep*(size_t)oCurrScaleSize.width;
            const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
            for(size_t nRowIter = 0; nRowIter<(size_t)oCurrScaleSize.height; ++nRowIter) {
                const size_t nRowDescIdx = nRowIter*nRowDescStep;
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = 0; nColIter<(size_t)oCurrScaleSize.width; ++nColIter) {
                    const size_t nColDescIdx = nRowDescIdx+nColIter*nColDescStep;
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                        const size_t nChDescIdx = nColDescIdx+nChIter*nChDescStep;
                        const size_t nChLUTIdx = nColLUTIdx+nChIter*nChLUTStep;
                        uchar* anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nChLUTIdx;
                        CV_DbgAssert(nChDescIdx<size_t(oAbsLBSPDescMap.dataend-oAbsLBSPDescMap.datastart));
                        ushort& nAbsLBSPDesc = *(ushort*)(oAbsLBSPDescMap.data+nChDescIdx);
                        ushort& nRelLBSPDesc = *(ushort*)(oRelLBSPDescMap.data+nChDescIdx);
                        const uchar uRefColor = *(oPyrMap.data+nChDescIdx/2);
                        LBSP::computeDescriptor_threshold(anCurrLUT,uRefColor,nAbsLBSPThreshold,nAbsLBSPDesc);
                        LBSP::computeDescriptor_threshold(anCurrLUT,uRefColor,(size_t)(uRefColor*fRelLBSPThreshold),nRelLBSPDesc);
                        CV_DbgAssert(nChDescIdx/2<size_t(oAbsLBSPDescGradMap.dataend-oAbsLBSPDescGradMap.datastart));
                        *(oAbsLBSPDescGradMap.data+nChDescIdx/2) = (uchar)(DistanceUtils::popcount(nAbsLBSPDesc)*float(UCHAR_MAX)/LBSP::DESC_SIZE_BITS);
                        *(oRelLBSPDescGradMap.data+nChDescIdx/2) = (uchar)(DistanceUtils::popcount(nRelLBSPDesc)*float(UCHAR_MAX)/LBSP::DESC_SIZE_BITS);
                    }
                }
            }
            if(nLevelIter) {
                cv::resize(oAbsLBSPDescGradMap,oTempInputCopyMat,oOrigInputSize,0,0,cv::INTER_NEAREST);
                cv::add(oAbsLBSPDescGradSum,oTempInputCopyMat,oAbsLBSPDescGradSum,cv::Mat(),CV_32F);
                cv::resize(oRelLBSPDescGradMap,oTempInputCopyMat,oOrigInputSize,0,0,cv::INTER_NEAREST);
                cv::add(oRelLBSPDescGradSum,oTempInputCopyMat,oRelLBSPDescGradSum,cv::Mat(),CV_32F);
            }
            else {
                cv::add(oAbsLBSPDescGradSum,oAbsLBSPDescGradMap,oAbsLBSPDescGradSum,cv::Mat(),CV_32F);
                cv::add(oRelLBSPDescGradSum,oRelLBSPDescGradMap,oRelLBSPDescGradSum,cv::Mat(),CV_32F);
            }
            voDisplayPairs[nLevelIter] = cv::Mat(oCurrScaleSize,nOrigType,vvuDisplayPairsData[nLevelIter].data());
            cv::hconcat(oPyrMap,oAbsLBSPDescGradMap,voDisplayPairs[nLevelIter]);
            cv::hconcat(voDisplayPairs[nLevelIter],oRelLBSPDescGradMap,voDisplayPairs[nLevelIter]);
            cv::resize(voDisplayPairs[nLevelIter],voDisplayPairs[nLevelIter],cv::Size(oInputImg.cols*3,oInputImg.rows),0,0,cv::INTER_NEAREST);
            std::stringstream ssWinName;
            ssWinName << "L[" << nLevelIter << "] in/abs/rel";
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
        /*cv::Mat oDisplayLBSPDescGradSumNorm_NMS;
        litiv::nonMaxSuppression<5>(oDisplayLBSPDescGradSumNorm,oDisplayLBSPDescGradSumNorm_NMS);
        cv::imshow("GradSumNorm abs/rel/mix NMS",oDisplayLBSPDescGradSumNorm_NMS);*/
        nKeyPressed = cv::waitKey(0);
    }
#elif USE_NMS_MJVOTE

    @@@@ everything below is WiP/broken/placeholder

    for(size_t nThresholdIter=0; nThresholdIter<NB_THRESHOLD_BINS; ++nThresholdIter) {
        std::vector<cv::Mat> voDisplayPairs(m_nLevels);
        cv::Mat oDescMap,oDescGradMap,oPyrInputMap=oInputImg;
        for(size_t nLevelIter=0; nLevelIter<m_nLevels; ++nLevelIter) {
            const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
            oDescMap.create(oCurrScaleSize,CV_16UC(int(nChannels)));
            oDescGradMap.create(oCurrScaleSize,nOrigType);
            const size_t nRowDescStep = nColDescStep*(size_t)oCurrScaleSize.width;
            const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
            for(size_t nRowIter = 0; nRowIter<(size_t)oCurrScaleSize.height; ++nRowIter) {
                const size_t nRowDescIdx = nRowIter*nRowDescStep;
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = 0; nColIter<(size_t)oCurrScaleSize.width; ++nColIter) {
                    const size_t nColDescIdx = nRowDescIdx+nColIter*nColDescStep;
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                        const size_t nChDescIdx = nColDescIdx+nChIter*nChDescStep;
                        const size_t nChLUTIdx = nColLUTIdx+nChIter*nChLUTStep;
                        uchar* anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nChLUTIdx;
                        CV_DbgAssert(nChDescIdx<size_t(oDescMap.dataend-oDescMap.datastart));
                        ushort& nChDesc = *(ushort*)(oDescMap.data+nChDescIdx);
                        LBSP::computeDescriptor_threshold(anCurrLUT,*(oPyrInputMap.data+nChDescIdx/2),nThres,nChDesc);
                        CV_DbgAssert(nChDescIdx/2<size_t(oDescGradMap.dataend-oDescGradMap.datastart));
                        *(oDescGradMap.data+nChDescIdx/2) = (uchar)(DistanceUtils::popcount(nChDesc)*float(UCHAR_MAX)/LBSP::DESC_SIZE_BITS);
                    }
                }
            }
            cv::hconcat(oPyrInputMap,oDescGradMap,voDisplayPairs[nLevelIter]);
            cv::resize(voDisplayPairs[nLevelIter],voDisplayPairs[nLevelIter],cv::Size(oInputImg.cols*2,oInputImg.rows),0,0,cv::INTER_NEAREST);
            std::stringstream ssWinName;
            ssWinName << "L[" << nLevelIter << "]";
            cv::imshow(ssWinName.str(),voDisplayPairs[nLevelIter]);
            if(nLevelIter+1<m_nLevels)
                oPyrInputMap = cv::Mat(m_voMapSizeList[nLevelIter+1],nOrigType,m_vvuInputPyrMaps[nLevelIter].data());
        }
    }
#elif USE_MBLUR_MJVOTE_CFG

    @@@@ everything below is WiP/broken/placeholder

    for(size_t nThresholdIter=0; nThresholdIter<NB_THRESHOLD_BINS; ++nThresholdIter) {
        std::vector<cv::Mat> voDisplayPairs(m_nLevels);
        cv::Mat oDescMap,oDescGradMap,oPyrInputMap=oInputImg;
        for(size_t nLevelIter=0; nLevelIter<m_nLevels; ++nLevelIter) {
            const cv::Size& oCurrScaleSize = m_voMapSizeList[nLevelIter];
            oDescMap.create(oCurrScaleSize,CV_16UC(int(nChannels)));
            oDescGradMap.create(oCurrScaleSize,nOrigType);
            const size_t nRowDescStep = nColDescStep*(size_t)oCurrScaleSize.width;
            const size_t nRowLUTStep = nColLUTStep*(size_t)oCurrScaleSize.width;
            for(size_t nRowIter = 0; nRowIter<(size_t)oCurrScaleSize.height; ++nRowIter) {
                const size_t nRowDescIdx = nRowIter*nRowDescStep;
                const size_t nRowLUTIdx = nRowIter*nRowLUTStep;
                for(size_t nColIter = 0; nColIter<(size_t)oCurrScaleSize.width; ++nColIter) {
                    const size_t nColDescIdx = nRowDescIdx+nColIter*nColDescStep;
                    const size_t nColLUTIdx = nRowLUTIdx+nColIter*nColLUTStep;
                    for(size_t nChIter = 0; nChIter<nChannels; ++nChIter) {
                        const size_t nChDescIdx = nColDescIdx+nChIter*nChDescStep;
                        const size_t nChLUTIdx = nColLUTIdx+nChIter*nChLUTStep;
                        uchar* anCurrLUT = m_vvuLBSPLookupMaps[nLevelIter].data()+nChLUTIdx;
                        CV_DbgAssert(nChDescIdx<size_t(oDescMap.dataend-oDescMap.datastart));
                        ushort& nChDesc = *(ushort*)(oDescMap.data+nChDescIdx);
                        LBSP::computeDescriptor_threshold(anCurrLUT,*(oPyrInputMap.data+nChDescIdx/2),nThres,nChDesc);
                        CV_DbgAssert(nChDescIdx/2<size_t(oDescGradMap.dataend-oDescGradMap.datastart));
                        *(oDescGradMap.data+nChDescIdx/2) = (uchar)(DistanceUtils::popcount(nChDesc)*float(UCHAR_MAX)/LBSP::DESC_SIZE_BITS);
                    }
                }
            }
            cv::hconcat(oPyrInputMap,oDescGradMap,voDisplayPairs[nLevelIter]);
            cv::resize(voDisplayPairs[nLevelIter],voDisplayPairs[nLevelIter],cv::Size(oInputImg.cols*2,oInputImg.rows),0,0,cv::INTER_NEAREST);
            std::stringstream ssWinName;
            ssWinName << "L[" << nLevelIter << "]";
            cv::imshow(ssWinName.str(),voDisplayPairs[nLevelIter]);
            if(nLevelIter+1<m_nLevels)
                oPyrInputMap = cv::Mat(m_voMapSizeList[nLevelIter+1],nOrigType,m_vvuInputPyrMaps[nLevelIter].data());
        }
    }
#endif //USE_...
    cv::normalize(oEdgeMask,oEdgeMask,0,UCHAR_MAX,cv::NORM_MINMAX);
}
