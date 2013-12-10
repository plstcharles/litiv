#include "BackgroundSubtractorViBe_3ch.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

BackgroundSubtractorViBe_3ch::BackgroundSubtractorViBe_3ch(	 size_t nColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples)
	:	BackgroundSubtractorViBe(nColorDistThreshold,nBGSamples,nRequiredBGSamples) {}

BackgroundSubtractorViBe_3ch::~BackgroundSubtractorViBe_3ch() {}

void BackgroundSubtractorViBe_3ch::initialize(const cv::Mat& oInitImg) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	cv::Mat oInitImgRGB;
	if(oInitImg.type()==CV_8UC3)
		oInitImgRGB = oInitImg;
	else
		cv::cvtColor(oInitImg,oInitImgRGB,CV_GRAY2RGB);
	m_oImgSize = oInitImgRGB.size();
	CV_Assert(m_voBGImg.size()==(size_t)m_nBGSamples);
	int y_sample, x_sample;
	for(size_t s=0; s<m_nBGSamples; s++) {
		m_voBGImg[s].create(m_oImgSize,CV_8UC3);
		m_voBGImg[s] = cv::Scalar(0,0,0);
		for(int y_orig=0; y_orig<m_oImgSize.height; y_orig++) {
			for(int x_orig=0; x_orig<m_oImgSize.width; x_orig++) {
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,0,m_oImgSize);
				m_voBGImg[s].at<cv::Vec3b>(y_orig,x_orig) = oInitImgRGB.at<cv::Vec3b>(y_sample,x_sample);
			}
		}
	}

	m_bInitialized = true;
}

void BackgroundSubtractorViBe_3ch::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	CV_DbgAssert(m_bInitialized);
	CV_DbgAssert(learningRate>0);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert((oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3) && oInputImg.size()==m_oImgSize);
	cv::Mat oInputImgRGB;
	if(oInputImg.type()==CV_8UC3)
		oInputImgRGB = oInputImg;
	else
		cv::cvtColor(oInputImg,oInputImgRGB,CV_GRAY2RGB);
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
					if(absdiff_uchar(in[c],bg[c])>nCurrSCColorDistThreshold)
						goto skip;
#endif //BGSVIBE_USE_SC_THRS_VALIDATION
				// unlike what was stated in their 2011 paper, the real vibe algorithm uses L1 (abs diff) distance instead of L2 (euclidean)
				//if(cv::norm(in,bg)<m_nColorDistThreshold*3)
				if(L1dist_uchar(in,bg)<m_nColorDistThreshold*3)
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
					getRandNeighborPosition(x_rand,y_rand,x,y,0,m_oImgSize);
					const size_t s_rand = rand()%m_nBGSamples;
					m_voBGImg[s_rand].at<cv::Vec3b>(y_rand,x_rand) = oInputImgRGB.at<cv::Vec3b>(y,x);
				}
			}
		}
	}
}
