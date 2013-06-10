#include "BackgroundSubtractorLBSP.h"
#include "LBSP.h"
#include "HammingDist.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP()
	:	 m_nBGSamples(BGSLBSP_DEFAULT_BG_SAMPLES)
		,m_nRequiredBGSamples(BGSLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES)
		,m_voBGImg(BGSLBSP_DEFAULT_BG_SAMPLES)
		,m_voBGDesc(BGSLBSP_DEFAULT_BG_SAMPLES)
	 	,m_nFGThreshold(BGSLBSP_DEFAULT_FG_THRESHOLD)
		,m_nFGSCThreshold(BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD)
		,m_bInitialized(false)
		,m_oExtractor(LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
	CV_Assert(m_nBGSamples>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(  int nDescThreshold
													,int nFGThreshold
													,int nFGSCThreshold
													,int nBGSamples
													,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_voBGDesc(nBGSamples)
		,m_nFGThreshold(nFGThreshold)
		,m_nFGSCThreshold(nFGSCThreshold)
		,m_bInitialized(false)
		,m_oExtractor(nDescThreshold) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
	CV_Assert(m_nBGSamples>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(	 float fDescThreshold
													,int nFGThreshold
													,int nFGSCThreshold
													,int nBGSamples
													,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_voBGDesc(nBGSamples)
		,m_nFGThreshold(nFGThreshold)
		,m_nFGSCThreshold(nFGSCThreshold)
		,m_bInitialized(false)
		,m_oExtractor(fDescThreshold) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
	CV_Assert(m_nBGSamples>0);
}

BackgroundSubtractorLBSP::~BackgroundSubtractorLBSP() {}

void BackgroundSubtractorLBSP::initialize(const cv::Mat& oInitImg) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3);
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nCurrFGThreshold = m_nFGThreshold*m_nImgChannels;

	// init keypoints used for the extractor :
	cv::DenseFeatureDetector oKPDDetector(1.f, 1, 1.f, 1, 0, true, false);
	if(m_voKeyPoints.capacity()<(size_t)(m_oImgSize.width*m_oImgSize.height))
		m_voKeyPoints.reserve(m_oImgSize.width*m_oImgSize.height);
	oKPDDetector.detect(cv::Mat(m_oImgSize,m_nImgType), m_voKeyPoints);
	LBSP::validateKeyPoints(m_voKeyPoints,m_oImgSize);
	CV_Assert(!m_voKeyPoints.empty());

	// init bg model samples :
	cv::Mat oInitDesc;
	m_oExtractor.compute2(oInitImg,m_voKeyPoints,oInitDesc);
	CV_Assert(m_voBGImg.size()==(size_t)m_nBGSamples);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	int y_sample, x_sample;
	if(m_nImgChannels==1) {
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,m_nImgType);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC1);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::DESC_SIZE/2,m_oImgSize);
				m_voBGImg[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
				m_voBGDesc[s].at<unsigned short>(y_orig,x_orig) = oInitDesc.at<unsigned short>(y_sample,x_sample);
			}
		}
	}
	else { //m_nImgChannels==3
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,m_nImgType);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC3);
			m_voBGImg[s] = cv::Scalar(0,0,0);
			m_voBGDesc[s] = cv::Scalar(0,0,0);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::DESC_SIZE/2,m_oImgSize);
				const int idx_orig_img = oInitImg.step.p[0]*y_orig + oInitImg.step.p[1]*x_orig;
				const int idx_orig_desc = oInitDesc.step.p[0]*y_orig + oInitDesc.step.p[1]*x_orig;
				const int idx_rand_img = oInitImg.step.p[0]*y_sample + oInitImg.step.p[1]*x_sample;
				const int idx_rand_desc = oInitDesc.step.p[0]*y_sample + oInitDesc.step.p[1]*x_sample;
				uchar* bgimg_ptr = m_voBGImg[s].data+idx_orig_img;
				const uchar* initimg_ptr = oInitImg.data+idx_rand_img;
				unsigned short* bgdesc_ptr = (unsigned short*)(m_voBGDesc[s].data+idx_orig_desc);
				const unsigned short* initdesc_ptr = (unsigned short*)(oInitDesc.data+idx_rand_desc);
				for(int n=0;n<3; ++n) {
					bgimg_ptr[n] = initimg_ptr[n];
					bgdesc_ptr[n] = initdesc_ptr[n];
				}
			}
		}
	}
	m_bInitialized = true;
}

void BackgroundSubtractorLBSP::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	CV_DbgAssert(m_bInitialized);
	CV_DbgAssert(learningRate>0);
	cv::Mat oInputImg = _image.getMat(), oInputDesc;
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	const int nDescThreshold = m_oExtractor.getAbsThreshold();
	const int nLearningRate = (int)learningRate;
	if(m_nImgChannels==1) {
		unsigned short inputdesc;
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				LBSP::computeSingle(oInputImg,m_voBGImg[nSampleIdx],x,y,nDescThreshold,inputdesc);
				if(hdist_ushort_8bitLUT(inputdesc,m_voBGDesc[nSampleIdx].at<unsigned short>(y,x))<=m_nCurrFGThreshold)
					nGoodSamplesCount++;
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.at<uchar>(m_voKeyPoints[k].pt) = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					LBSP::computeSingle(oInputImg,cv::Mat(),x,y,nDescThreshold,inputdesc);
					m_voBGDesc[s_rand].at<unsigned short>(y,x) = inputdesc;
					m_voBGImg[s_rand].at<uchar>(y,x) = oInputImg.at<uchar>(y,x);
				}
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::DESC_SIZE/2,m_oImgSize);
					LBSP::computeSingle(oInputImg,cv::Mat(),x,y,nDescThreshold,inputdesc);
					m_voBGDesc[s_rand].at<unsigned short>(y_rand,x_rand) = inputdesc;
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y,x);
				}
			}
		}
	}
	else { //m_nImgChannels==3
		unsigned short inputdesc[3];
		int hdist[3];
		const int desc_row_step = m_voBGDesc[0].step.p[0];
		CV_DbgAssert(m_voBGDesc[0].step.p[1]==6);
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int idx_desc = desc_row_step*y + 6*x;
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				LBSP::computeSingle(oInputImg,m_voBGImg[nSampleIdx],x,y,nDescThreshold,inputdesc);
				const unsigned short* bgdesc_ptr = (unsigned short*)(m_voBGDesc[nSampleIdx].data+idx_desc);
				for(int n=0;n<3; ++n) {
					hdist[n] = hdist_ushort_8bitLUT(inputdesc[n],bgdesc_ptr[n]);
					if(hdist[n]>m_nFGSCThreshold)
						goto skip;
				}
				if(hdist[0]+hdist[1]+hdist[2]<=m_nCurrFGThreshold)
					goto count;
				goto skip;
				count:
				nGoodSamplesCount++;
				skip:
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.at<uchar>(m_voKeyPoints[k].pt) = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					unsigned short* bgdesc_ptr = ((unsigned short*)(m_voBGDesc[s_rand].data + desc_row_step*y + 6*x));
					LBSP::computeSingle(oInputImg,cv::Mat(),x,y,nDescThreshold,bgdesc_ptr);
					const int img_row_step = m_voBGImg[0].step.p[0];
					for(int n=0; n<3; ++n)
						*(m_voBGImg[s_rand].data + img_row_step*y + 3*x + n) = *(oInputImg.data + img_row_step*y + 3*x + n);
					CV_DbgAssert(m_voBGImg[s_rand].at<cv::Vec3b>(y,x)==oInputImg.at<cv::Vec3b>(y,x));
				}
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::DESC_SIZE/2,m_oImgSize);
					unsigned short* bgdesc_ptr = ((unsigned short*)(m_voBGDesc[s_rand].data + desc_row_step*y_rand + 6*x_rand));
					LBSP::computeSingle(oInputImg,cv::Mat(),x,y,nDescThreshold,bgdesc_ptr);
					const int img_row_step = m_voBGImg[0].step.p[0];
					for(int n=0; n<3; ++n)
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + n) = *(oInputImg.data + img_row_step*y + 3*x + n);
					CV_DbgAssert(m_voBGImg[s_rand].at<cv::Vec3b>(y_rand,x_rand)==oInputImg.at<cv::Vec3b>(y,x));
				}
			}
		}
	}
	cv::medianBlur(oFGMask,oFGMask,5);
}

cv::AlgorithmInfo* BackgroundSubtractorLBSP::info() const {
	CV_Assert(false); // NOT IMPL @@@@@
	return NULL;
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGImage() const {
	return m_voBGImg[0].clone();
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGDescriptors() const {
	return m_voBGDesc[0].clone();
}

std::vector<cv::KeyPoint> BackgroundSubtractorLBSP::getBGKeyPoints() const {
	return m_voKeyPoints;
}

void BackgroundSubtractorLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	m_oExtractor.validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	// @@@@ NOT IMPL
	CV_Assert(false);
	// need to reinit sample buffers...
}
