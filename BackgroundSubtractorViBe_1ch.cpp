#include "BackgroundSubtractorViBe_1ch.h"
#include "DistanceUtils.h"
#include "RandUtils.h"

BackgroundSubtractorViBe_1ch::BackgroundSubtractorViBe_1ch(  size_t nColorDistThreshold
                                                            ,size_t nBGSamples
                                                            ,size_t nRequiredBGSamples)
    :    BackgroundSubtractorViBe(nColorDistThreshold,nBGSamples,nRequiredBGSamples) {}

BackgroundSubtractorViBe_1ch::~BackgroundSubtractorViBe_1ch() {}

void BackgroundSubtractorViBe_1ch::initialize(const cv::Mat& oInitImg) {
    CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
    CV_Assert(oInitImg.isContinuous());
    CV_Assert(oInitImg.type()==CV_8UC1);
    m_oImgSize = oInitImg.size();
    CV_Assert(m_voBGImg.size()==(size_t)m_nBGSamples);
    for(size_t s=0; s<m_nBGSamples; s++) {
        m_voBGImg[s].create(m_oImgSize,CV_8UC1);
        m_voBGImg[s] = cv::Scalar(0);
        for(int y_orig=0; y_orig<m_oImgSize.height; y_orig++) {
            for(int x_orig=0; x_orig<m_oImgSize.width; x_orig++) {
                int y_sample, x_sample;
                getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,0,m_oImgSize);
                m_voBGImg[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
            }
        }
    }

    m_bInitialized = true;
}

void BackgroundSubtractorViBe_1ch::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
    CV_Assert(m_bInitialized);
    CV_Assert(learningRate>0);
    cv::Mat oInputImg = _image.getMat();
    CV_Assert(oInputImg.type()==CV_8UC1 && oInputImg.size()==m_oImgSize);
    _fgmask.create(m_oImgSize,CV_8UC1);
    cv::Mat oFGMask = _fgmask.getMat();
    oFGMask = cv::Scalar_<uchar>(0);
    const size_t nLearningRate = (size_t)ceil(learningRate);
    for(int y=0; y<m_oImgSize.height; y++) {
        for(int x=0; x<m_oImgSize.width; x++) {
            size_t nGoodSamplesCount=0, nSampleIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
                if(L1dist(oInputImg.at<uchar>(y,x),m_voBGImg[nSampleIdx].at<uchar>(y,x))<m_nColorDistThreshold)
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
                    getRandNeighborPosition_3x3(x_rand,y_rand,x,y,0,m_oImgSize);
                    m_voBGImg[rand()%m_nBGSamples].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y,x);
                }
            }
        }
    }
}
