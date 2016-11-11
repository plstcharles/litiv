
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/features2d/DASC.hpp"

#define DASC_SIZE 128

constexpr int static_diff(int a, int b) {return b-a;}

namespace pretrained { // obtained via middlebury dataset (imported here from original mat archives)

    constexpr std::array<int,DASC_SIZE*2> anRP1 = {
        5,-9,-14,-5,10,0,-12,4,-4,-6,-13,-7,-6,-11,5,14,3,9,10,11,10,-2,15,3,6,5,14,5,14,-5,-9,-5,
        8,10,-8,-10,-3,-15,-11,6,12,4,9,3,-7,13,3,-15,-15,3,13,0,4,-12,-6,-11,12,-2,-4,-12,-5,14,-3,-15,
        10,-11,11,10,-10,0,-11,-10,-9,5,8,6,-12,-4,13,7,15,0,-15,3,5,14,-11,10,5,-6,-13,-7,3,-15,-9,-5,
        -6,-8,0,13,-7,13,-9,3,0,15,11,-6,10,11,5,14,-11,10,14,-5,8,10,0,15,-5,14,11,6,8,-6,-7,13,
        -13,0,-11,-6,-8,-13,2,12,-7,13,4,6,-10,-11,-6,-11,-10,-11,5,-9,-14,-5,-3,-15,-4,12,10,-2,13,7,-15,0,
        10,2,5,14,-5,-14,-13,0,5,-14,0,5,10,8,-12,4,-12,4,-10,8,-15,-3,14,5,-10,-8,15,0,-2,-10,-7,1,
        0,15,14,-5,3,-9,-13,0,0,15,2,12,-10,8,13,7,-5,-6,5,14,15,3,-8,-13,-9,3,-5,9,5,-9,-0,-13,
        6,-11,15,3,-0,-15,13,7,6,11,-2,10,-15,-3,6,-8,11,-6,-14,5,-5,-6,3,-7,-15,0,6,-8,14,5,0,13,
    };

    constexpr std::array<int,DASC_SIZE*2> anRP2 = {
        -0,-13,1,7,12,2,-10,-11,-3,-9,0,15,-11,10,-15,-3,-2,-12,-15,3,5,14,-12,2,15,0,-11,-6,-8,-6,-2,10,
        -0,-8,-8,6,-4,-12,-9,-5,10,-8,5,-6,11,10,-13,0,12,2,-8,-13,-11,10,10,8,-10,11,-0,-8,4,12,11,10,
        11,-10,6,-11,-3,9,-11,6,-14,5,8,13,-0,-15,4,12,14,5,5,14,-10,-2,13,0,-15,-3,-3,15,7,-3,-14,5,
        13,7,4,-6,13,-8,-15,3,0,15,-14,-5,-14,-5,14,-5,5,-9,11,-6,-3,15,-10,-8,-3,15,14,5,10,-2,5,14,
        -3,-15,6,4,0,15,-5,14,11,6,-3,-7,-5,14,12,-4,8,-13,9,-3,7,1,-14,-5,-8,-10,8,10,0,15,-5,-14,
        -0,-15,-8,6,-7,-3,8,6,-0,-10,-7,-3,-11,-10,8,10,10,11,-15,0,-11,6,-2,10,-12,2,15,0,-10,8,10,-11,
        3,-7,-6,-8,9,3,5,-2,8,10,10,-2,14,-5,-10,-11,-13,-7,0,15,-6,11,-10,8,5,9,-3,15,-8,-6,-5,-14,
        3,15,12,-2,-4,-12,3,-4,8,-13,-0,-13,-0,-10,10,11,-5,-6,-13,7,-8,-10,-11,-10,5,14,8,-13,8,10,-15,-3,
    };

    constexpr std::array<int,DASC_SIZE*2> anRPDiff = lv::static_transform(anRP1,anRP2,static_diff);

} // namespace pretrained

// RF VERSION

template<size_t nRowOffset, size_t nColOffset>
inline void diff(const cv::Mat_<float>& oImage, cv::Mat_<float>& oLocalDiff) {
    lvDbgAssert(!oImage.empty() && (nColOffset>0 || nRowOffset>0));
    oLocalDiff.create(oImage.size());
    for(int nRowIdx=int(nRowOffset); nRowIdx<oImage.rows; ++nRowIdx)
        for(int nColIdx=int(nColOffset); nColIdx<oImage.cols; ++nColIdx)
            oLocalDiff(nRowIdx,nColIdx) = oImage(nRowIdx-nRowOffset,nColIdx-nColOffset)-oImage(nRowIdx,nColIdx);
    lv::unroll<nRowOffset>([&](size_t nRowIdx){
        for(int nColIdx=0; nColIdx<oImage.cols; ++nColIdx)
            oLocalDiff((int)nRowIdx,nColIdx) = 0.0f;
    });
    lv::unroll<nColOffset>([&](size_t nColIdx){
        for(int nRowIdx=nRowOffset; nRowIdx<oImage.rows; ++nRowIdx)
            oLocalDiff(nRowIdx,(int)nColIdx) = 0.0f;
    });
}

inline void domaintransform_runfilter(const cv::Mat_<float>& oImage, const cv::Mat_<float>& oRef_V_dHdx, const cv::Mat_<float>& oRef_V_dVdy_t, cv::Mat_<float>& oOutput, size_t nIters) {
    lvDbgAssert(!oImage.empty() && !oRef_V_dHdx.empty() && !oRef_V_dHdx.empty() && nIters>0 && oImage.dims==2 && oRef_V_dHdx.dims==3 && oRef_V_dVdy_t.dims==3);
    lvDbgAssert(oImage.rows==oRef_V_dHdx.size[1] && oImage.rows==oRef_V_dVdy_t.size[2] && oImage.cols==oRef_V_dHdx.size[2] && oImage.cols==oRef_V_dVdy_t.size[1]);
    lvDbgAssert(oRef_V_dHdx.size[0]==(int)nIters && oRef_V_dVdy_t.size[0]==(int)nIters);
    oImage.copyTo(oOutput);
    cv::Mat_<float> oOutput_t;
    const auto lTransfDomRecursFilter_H = [](cv::Mat_<float>& _oImage, const cv::Mat_<float>& oRef, int nIterIdx) {
        lvDbgAssert(!_oImage.empty() && _oImage.dims==2 && !oRef.empty() && oRef.dims==3);
        lvDbgAssert(_oImage.rows==oRef.size[1] && _oImage.cols==oRef.size[2]);
        for(int nRowIdx=0; nRowIdx<_oImage.rows; ++nRowIdx) {
            for(int nColIdx=1; nColIdx<_oImage.cols; ++nColIdx)
                _oImage(nRowIdx,nColIdx) += oRef(nIterIdx,nRowIdx,nColIdx)*(_oImage(nRowIdx,nColIdx-1)-_oImage(nRowIdx,nColIdx));
            for(int nColIdx=_oImage.cols-2; nColIdx>=0; --nColIdx)
                _oImage(nRowIdx,nColIdx) += oRef(nIterIdx,nRowIdx,nColIdx+1)*(_oImage(nRowIdx,nColIdx+1)-_oImage(nRowIdx,nColIdx));
        }
    };
    for(int nIterIdx=0; nIterIdx<(int)nIters; ++nIterIdx) {
        lTransfDomRecursFilter_H(oOutput,oRef_V_dHdx,nIterIdx);
        cv::transpose(oOutput,oOutput_t);
        lTransfDomRecursFilter_H(oOutput_t,oRef_V_dVdy_t,nIterIdx);
        cv::transpose(oOutput_t,oOutput);
    }
}

//M_half = 15;
//N_half = 2;
//epsil = 0.09;
//downSize = 1;

//cv::Mat dasc(const cv::Mat& _oImage, float fSigma_s=2.0f, float fSigma_r=0.2f, size_t nIters=1, bool bPrefilter=true) {
cv::Mat dasc(const cv::Mat& _oImage, float fSigma_s, float fSigma_r, size_t nIters, bool bPrefilter) {
    lvAssert(!_oImage.empty() && fSigma_s>0.0f && fSigma_r>0.0f && nIters>0);
    lvAssert((_oImage.channels()==1 || _oImage.channels()==3) && (_oImage.depth()==CV_32F || _oImage.depth()==CV_8U));
    cv::Mat oImageTemp;
    if(_oImage.depth()==CV_8U)
        _oImage.convertTo(oImageTemp,CV_32F,1.0/UCHAR_MAX);
    else
        oImageTemp = _oImage;
    if(oImageTemp.channels()==3)
        cv::cvtColor(oImageTemp,oImageTemp,cv::COLOR_BGR2GRAY);
    lvDbgAssert(cv::countNonZero((oImageTemp>1.0f)|(oImageTemp<0.0f))==0);
    cv::Mat_<float> oImage = oImageTemp;
    if(bPrefilter)
        cv::GaussianBlur(oImage,oImage,cv::Size(7,7),1.0);
    const cv::Size oImageSize = oImage.size();
    const int nRows = oImageSize.height;
    const int nCols = oImageSize.width;
    cv::Mat_<float> oImageLocalDiff_Y,oImageLocalDiff_X;
    diff<1,0>(oImage,oImageLocalDiff_Y);
    diff<0,1>(oImage,oImageLocalDiff_X);
    cv::Mat_<float> oRef_dVdy = 1.0f + fSigma_s/fSigma_r*cv::abs(oImageLocalDiff_Y);
    cv::Mat_<float> oRef_dHdx = 1.0f + fSigma_s/fSigma_r*cv::abs(oImageLocalDiff_X);
    const std::array<int,3> anRefDims = {(int)nIters,nRows,nCols};
    cv::Mat_<float> oRef_V_dHdx(3,anRefDims.data());
    const std::array<int,3> anRefDims_t = {(int)nIters,nCols,nRows};
    cv::Mat_<float> oRef_V_dVdy_t(3,anRefDims_t.data());
    for(int nIterIdx=0; nIterIdx<(int)nIters; ++nIterIdx) {
        const float fBase = exp(-sqrt(2.0f)/(fSigma_s*sqrt(3.0f)*(float)pow(2.0f,(int)nIters-(nIterIdx+1))/sqrt((float)pow(4.0f,(int)nIters)-1)));
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                oRef_V_dHdx(nIterIdx,nRowIdx,nColIdx) = pow(fBase,oRef_dHdx(nRowIdx,nColIdx));
                oRef_V_dVdy_t(nIterIdx,nColIdx,nRowIdx) = pow(fBase,oRef_dVdy(nRowIdx,nColIdx));
            }
        }
    }
    cv::Mat_<float> oImage_AdaptiveMean,oImage_AdaptiveMeanSqr;
    domaintransform_runfilter(oImage,oRef_V_dHdx,oRef_V_dVdy_t,oImage_AdaptiveMean,nIters);
    domaintransform_runfilter(oImage.mul(oImage),oRef_V_dHdx,oRef_V_dVdy_t,oImage_AdaptiveMeanSqr,nIters);
    cv::Mat_<float> oLookupImage(oImageSize),oLookupImage_Sqr(oImageSize),oLookupImage_Mix(oImageSize);
    cv::Mat_<float> oLookupImage_AdaptiveMean,oLookupImage_AdaptiveMeanSqr,oLookupImage_AdaptiveMeanMix;
    const std::array<int,3> anDescDims = {nRows,nCols,DASC_SIZE};
    cv::Mat_<float> oDesc(3,anDescDims.data(),CV_32FC1);
    for(int nLUTIdx=0; nLUTIdx<DASC_SIZE; nLUTIdx++) {
        const int nRowOffset = pretrained::anRPDiff[nLUTIdx*2];
        const int nColOffset = pretrained::anRPDiff[nLUTIdx*2+1];
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                if(nRowIdx+nRowOffset>=0 && nRowIdx+nRowOffset<nRows && nColIdx+nColOffset>=0 && nColIdx+nColOffset<nCols) {
                    oLookupImage(nRowIdx,nColIdx) = oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                    oLookupImage_Sqr(nRowIdx,nColIdx) = oImage(nRowIdx+nRowOffset,nColIdx+nColOffset)*oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                    oLookupImage_Mix(nRowIdx,nColIdx) = oImage(nRowIdx,nColIdx)*oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                }
                else
                    oLookupImage(nRowIdx,nColIdx) = oLookupImage_Sqr(nRowIdx,nColIdx) = oLookupImage_Mix(nRowIdx,nColIdx) = 0.0f;
            }
        }
        domaintransform_runfilter(oLookupImage,oRef_V_dHdx,oRef_V_dVdy_t,oLookupImage_AdaptiveMean,nIters);
        domaintransform_runfilter(oLookupImage_Sqr,oRef_V_dHdx,oRef_V_dVdy_t,oLookupImage_AdaptiveMeanSqr,nIters);
        domaintransform_runfilter(oLookupImage_Mix,oRef_V_dHdx,oRef_V_dVdy_t,oLookupImage_AdaptiveMeanMix,nIters);
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx = 0; nColIdx<nCols; ++nColIdx) {
                const int nOffsetRowIdx = nRowIdx+pretrained::anRP1[nLUTIdx*2];
                const int nOffsetColIdx = nColIdx+pretrained::anRP1[nLUTIdx*2+1];
                if(nOffsetRowIdx>0 && nOffsetRowIdx<nRows && nOffsetColIdx>0 && nOffsetColIdx<nCols) {
                    const float fCorrSurfDenom = sqrt((oImage_AdaptiveMeanSqr(nOffsetRowIdx,nOffsetColIdx)-oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)) * (oLookupImage_AdaptiveMeanSqr(nOffsetRowIdx,nOffsetColIdx)-oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)));
                    if(fCorrSurfDenom>1e-10)
                        oDesc(nRowIdx,nColIdx,nLUTIdx) = exp(-(1-(oLookupImage_AdaptiveMeanMix(nOffsetRowIdx,nOffsetColIdx)-oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx))/fCorrSurfDenom)/0.5f);
                    else
                        oDesc(nRowIdx,nColIdx,nLUTIdx) = 1.0f;
                }
                else
                    oDesc(nRowIdx,nColIdx,nLUTIdx) = 0.0f;
            }
        }
    }
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float fNorm = 0;
            for(int nLUTIdx=0; nLUTIdx<DASC_SIZE; ++nLUTIdx)
                fNorm += oDesc(nRowIdx,nColIdx,nLUTIdx)*oDesc(nRowIdx,nColIdx,nLUTIdx);
            const float fNormSqrt = sqrt(fNorm);
            for(int nLUTIdx=0; nLUTIdx<DASC_SIZE; ++nLUTIdx)
                oDesc(nRowIdx,nColIdx,nLUTIdx) = (fNormSqrt>1e-10)?oDesc(nRowIdx,nColIdx,nLUTIdx)/fNormSqrt:0.0f;
        }
    }
    return oDesc;
}