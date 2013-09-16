#include "BackgroundSubtractorViBe_1ch.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

BackgroundSubtractorViBe_1ch::BackgroundSubtractorViBe_1ch(	 int nColorDistThreshold
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_nColorDistThreshold(nColorDistThreshold)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
}

BackgroundSubtractorViBe_1ch::~BackgroundSubtractorViBe_1ch() {}

void BackgroundSubtractorViBe_1ch::initialize(const cv::Mat& oInitImg) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC1);
	m_oImgSize = oInitImg.size();
	CV_Assert(m_voBGImg.size()==(size_t)m_nBGSamples);
	for(int s=0; s<m_nBGSamples; s++) {
		m_voBGImg[s].create(m_oImgSize,CV_8UC1);
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

void BackgroundSubtractorViBe_1ch::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	CV_DbgAssert(m_bInitialized);
	CV_DbgAssert(learningRate>0);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert(oInputImg.type()==CV_8UC1 && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);
	const int nLearningRate = (int)learningRate;
	for(int y=0; y<m_oImgSize.height; y++) {
		for(int x=0; x<m_oImgSize.width; x++) {
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				// unlike what was stated in their 2011 paper, the real vibe algorithm uses L1 (abs diff) distance instead of L2 (euclidean)
				//if(cv::norm(in,bg)<m_nColorDistThreshold*3)
				if(L1dist_uchar(oInputImg.at<uchar>(y,x),m_voBGImg[nSampleIdx].at<uchar>(y,x))<m_nColorDistThreshold*3)
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
					getRandNeighborPosition(x_rand,y_rand,x,y,0,m_oImgSize);
					m_voBGImg[rand()%m_nBGSamples].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y,x);
				}
			}
		}
	}
}

cv::AlgorithmInfo* BackgroundSubtractorViBe_1ch::info() const {
	CV_Assert(false); // NOT IMPL @@@@@
	return NULL;
}

void BackgroundSubtractorViBe_1ch::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_DbgAssert(!m_voBGImg.empty());
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC1);
	for(int n=0; n<m_nBGSamples; ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				oAvgBGImg.at<float>(y,x) += ((float)m_voBGImg[n].at<uchar>(y,x))/m_nBGSamples;
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}
