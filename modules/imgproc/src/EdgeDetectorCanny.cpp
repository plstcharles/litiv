
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

#include "litiv/imgproc/EdgeDetectorCanny.hpp"
#if DEBUG
#include "litiv/imgproc.hpp"
#endif //DEBUG

EdgeDetectorCanny::EdgeDetectorCanny(double dHystLowThrshFactor, double dGaussianKernelSigma) :
        m_dHystLowThrshFactor(dHystLowThrshFactor),
        m_dGaussianKernelSigma(dGaussianKernelSigma) {
    CV_Assert(m_dHystLowThrshFactor>0 && m_dHystLowThrshFactor<1);
    CV_Assert(m_dGaussianKernelSigma>=0);
}

void EdgeDetectorCanny::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dThreshold) {
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
    const size_t nCurrBaseHystThreshold = (size_t)(dThreshold*UCHAR_MAX);
    static const int nWindowSize = EDGCANNY_DEFAULT_NMS_WINDOW_SIZE;
    static const bool bUseL2Gradient = EDGCANNY_DEFAULT_USE_L2_GRADIENT_NORM;
    litiv::cv_canny<nWindowSize,bUseL2Gradient>(oInputImage_gray,oEdgeMask,nCurrBaseHystThreshold*m_dHystLowThrshFactor,(double)nCurrBaseHystThreshold);
#if DEBUG
    cv::Mat tmp(oEdgeMask.size(),oEdgeMask.type());
    cv::Canny(oInputImage_gray,tmp,nCurrBaseHystThreshold*m_dHystLowThrshFactor,(double)nCurrBaseHystThreshold,nWindowSize,bUseL2Gradient);
    CV_Assert(cv::countNonZero(tmp!=oEdgeMask)==0);
#endif //DEBUG
}

void EdgeDetectorCanny::apply(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.channels()==1 || oInputImg.channels()==3 || oInputImg.channels()==4);
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    oEdgeMask = cv::Scalar_<uchar>(0);
    cv::Mat oTempEdgeMask = oEdgeMask.clone();
    for(size_t nCurrThreshold=0; nCurrThreshold<UCHAR_MAX; ++nCurrThreshold) {
        apply_threshold(oInputImg,oTempEdgeMask,double(nCurrThreshold)/UCHAR_MAX);
        oEdgeMask += oTempEdgeMask/UCHAR_MAX;
    }
    cv::normalize(oEdgeMask,oEdgeMask,0,UCHAR_MAX,cv::NORM_MINMAX);
}
