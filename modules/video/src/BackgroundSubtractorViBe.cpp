
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

#include "litiv/video/BackgroundSubtractorViBe.hpp"
#include "litiv/utils/distances.hpp"
#include "litiv/utils/opencv.hpp"

BackgroundSubtractorViBe::BackgroundSubtractorViBe(size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples) :
        m_nBGSamples(nBGSamples),
        m_nRequiredBGSamples(nRequiredBGSamples),
        m_voBGImg(nBGSamples),
        m_nColorDistThreshold(nColorDistThreshold),
        m_bInitialized(false) {
    lvAssert(m_nBGSamples>0 && m_nRequiredBGSamples<=m_nBGSamples);
}

BackgroundSubtractorViBe::~BackgroundSubtractorViBe() {}

void BackgroundSubtractorViBe::getBackgroundImage(cv::OutputArray backgroundImage) const {
    lvAssert(m_bInitialized);
    cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC(m_voBGImg[0].channels()));
    for(size_t n=0; n<m_nBGSamples; ++n) {
        for(int y=0; y<m_oImgSize.height; ++y) {
            for(int x=0; x<m_oImgSize.width; ++x) {
                const size_t idx_uchar = m_voBGImg[n].step.p[0]*y + m_voBGImg[n].step.p[1]*x;
                const size_t idx_flt32 = idx_uchar*4;
                float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+idx_flt32);
                const uchar* const oBGImgPtr = m_voBGImg[n].data+idx_uchar;
                for(size_t c=0; c<(size_t)m_voBGImg[n].channels(); ++c)
                    oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
            }
        }
    }
    oAvgBGImg.convertTo(backgroundImage,CV_8U);
}

BackgroundSubtractorViBe_1ch::BackgroundSubtractorViBe_1ch(size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples) :
        BackgroundSubtractorViBe(nColorDistThreshold,nBGSamples,nRequiredBGSamples) {}

BackgroundSubtractorViBe_1ch::~BackgroundSubtractorViBe_1ch() {}

void BackgroundSubtractorViBe_1ch::initialize(const cv::Mat& oInitImg) {
    lvAssert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
    lvAssert(oInitImg.isContinuous());
    lvAssert(oInitImg.type()==CV_8UC1);
    m_oImgSize = oInitImg.size();
    lvAssert(m_voBGImg.size()==(size_t)m_nBGSamples);
    for(size_t s=0; s<m_nBGSamples; s++) {
        m_voBGImg[s].create(m_oImgSize,CV_8UC1);
        m_voBGImg[s] = cv::Scalar(0);
        for(int y_orig=0; y_orig<m_oImgSize.height; y_orig++) {
            for(int x_orig=0; x_orig<m_oImgSize.width; x_orig++) {
                int y_sample, x_sample;
                cv::getRandSamplePosition_7x7_std2(x_sample,y_sample,x_orig,y_orig,0,m_oImgSize);
                m_voBGImg[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
            }
        }
    }
    m_bInitialized = true;
}

void BackgroundSubtractorViBe_1ch::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
    lvAssert(m_bInitialized);
    lvAssert(learningRate>0);
    cv::Mat oInputImg = _image.getMat();
    lvAssert(oInputImg.type()==CV_8UC1 && oInputImg.size()==m_oImgSize);
    _fgmask.create(m_oImgSize,CV_8UC1);
    cv::Mat oFGMask = _fgmask.getMat();
    oFGMask = cv::Scalar_<uchar>(0);
    const size_t nLearningRate = (size_t)ceil(learningRate);
    for(int y=0; y<m_oImgSize.height; y++) {
        for(int x=0; x<m_oImgSize.width; x++) {
            size_t nGoodSamplesCount=0, nSampleIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
                if(lv::L1dist(oInputImg.at<uchar>(y,x),m_voBGImg[nSampleIdx].at<uchar>(y,x))<m_nColorDistThreshold)
                    nGoodSamplesCount++;
                nSampleIdx++;
            }
            if(nGoodSamplesCount<m_nRequiredBGSamples)
                oFGMask.at<uchar>(y,x) = UCHAR_MAX;
            else {
                if((rand()%nLearningRate)==0)
                    m_voBGImg[rand()%m_nBGSamples].at<uchar>(y,x)=oInputImg.at<uchar>(y,x);
                if((rand()%nLearningRate)==0) {
                    int x_rand,y_rand;
                    cv::getRandNeighborPosition_3x3(x_rand,y_rand,x,y,0,m_oImgSize);
                    m_voBGImg[rand()%m_nBGSamples].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y,x);
                }
            }
        }
    }
}

BackgroundSubtractorViBe_3ch::BackgroundSubtractorViBe_3ch(size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples) :
        BackgroundSubtractorViBe(nColorDistThreshold,nBGSamples,nRequiredBGSamples) {}

BackgroundSubtractorViBe_3ch::~BackgroundSubtractorViBe_3ch() {}

void BackgroundSubtractorViBe_3ch::initialize(const cv::Mat& oInitImg) {
    lvAssert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
    lvAssert(oInitImg.isContinuous());
    lvAssert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
    cv::Mat oInitImgRGB;
    if(oInitImg.type()==CV_8UC3)
        oInitImgRGB = oInitImg;
    else
        cv::cvtColor(oInitImg,oInitImgRGB,cv::COLOR_GRAY2BGR);
    m_oImgSize = oInitImgRGB.size();
    lvAssert(m_voBGImg.size()==(size_t)m_nBGSamples);
    int y_sample, x_sample;
    for(size_t s=0; s<m_nBGSamples; s++) {
        m_voBGImg[s].create(m_oImgSize,CV_8UC3);
        m_voBGImg[s] = cv::Scalar(0,0,0);
        for(int y_orig=0; y_orig<m_oImgSize.height; y_orig++) {
            for(int x_orig=0; x_orig<m_oImgSize.width; x_orig++) {
                cv::getRandSamplePosition_7x7_std2(x_sample,y_sample,x_orig,y_orig,0,m_oImgSize);
                m_voBGImg[s].at<cv::Vec3b>(y_orig,x_orig) = oInitImgRGB.at<cv::Vec3b>(y_sample,x_sample);
            }
        }
    }
    m_bInitialized = true;
}

void BackgroundSubtractorViBe_3ch::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
    lvAssert(m_bInitialized);
    lvAssert(learningRate>0);
    cv::Mat oInputImg = _image.getMat();
    lvAssert((oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3) && oInputImg.size()==m_oImgSize);
    cv::Mat oInputImgRGB;
    if(oInputImg.type()==CV_8UC3)
        oInputImgRGB = oInputImg;
    else
        cv::cvtColor(oInputImg,oInputImgRGB,cv::COLOR_GRAY2BGR);
    _fgmask.create(m_oImgSize,CV_8UC1);
    cv::Mat oFGMask = _fgmask.getMat();
    oFGMask = cv::Scalar_<uchar>(0);
    const size_t nLearningRate = (size_t)ceil(learningRate);
    for(int y=0; y<m_oImgSize.height; y++) {
        for(int x=0; x<m_oImgSize.width; x++) {
#if BGSVIBE_USE_SC_THRS_VALIDATION
            const size_t nCurrSCColorDistThreshold = (size_t)(m_nColorDistThreshold*BGSVIBE_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR)/3;
#endif //BGSVIBE_USE_SC_THRS_VALIDATION
            size_t nGoodSamplesCount=0, nSampleIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
                const cv::Vec3b& in = oInputImgRGB.at<cv::Vec3b>(y,x);
                const cv::Vec3b& bg = m_voBGImg[nSampleIdx].at<cv::Vec3b>(y,x);
#if BGSVIBE_USE_SC_THRS_VALIDATION
                for(size_t c=0; c<3; c++)
                    if(lv::L1dist(in[c],bg[c])>nCurrSCColorDistThreshold)
                        goto skip;
#endif //BGSVIBE_USE_SC_THRS_VALIDATION
#if BGSVIBE_USE_L1_DISTANCE_CHECK
                if(lv::L1dist(in,bg)<m_nColorDistThreshold*3)
#else //(!BGSVIBE_USE_L1_DISTANCE_CHECK)
                if(lv::L2dist(in,bg)<m_nColorDistThreshold*3)
#endif //(!BGSVIBE_USE_L1_DISTANCE_CHECK)
                    nGoodSamplesCount++;
#if BGSVIBE_USE_SC_THRS_VALIDATION
                skip:
#endif //BGSVIBE_USE_SC_THRS_VALIDATION
                nSampleIdx++;
            }
            if(nGoodSamplesCount<m_nRequiredBGSamples)
                oFGMask.at<uchar>(y,x) = UCHAR_MAX;
            else {
                if((rand()%nLearningRate)==0)
                    m_voBGImg[rand()%m_nBGSamples].at<cv::Vec3b>(y,x)=oInputImgRGB.at<cv::Vec3b>(y,x);
                if((rand()%nLearningRate)==0) {
                    int x_rand,y_rand;
                    cv::getRandNeighborPosition_3x3(x_rand,y_rand,x,y,0,m_oImgSize);
                    const size_t s_rand = rand()%m_nBGSamples;
                    m_voBGImg[s_rand].at<cv::Vec3b>(y_rand,x_rand) = oInputImgRGB.at<cv::Vec3b>(y,x);
                }
            }
        }
    }
}
