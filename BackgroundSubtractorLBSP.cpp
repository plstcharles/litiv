#include "BackgroundSubtractorLBSP.h"
#include "LBSP.h"
#include "HammingDist.h"
#include <iostream>

#define N_SAMPLES_NEEDED_FOR_BG 2

// floor(fspecial('gaussian', 5, 0.7)*512)
static const int nSamplesInitPatternWidth = 5;
static const int nSamplesInitPatternHeight = 5;
static const int nSamplesInitPatternTot = 502;
static const int anSamplesInitPattern[25] = {
	0,     1,     2,     1,     0,
	1,    21,    59,    21,     1,
	2,    59,   166,    59,     2,
	1,    21,    59,    21,     1,
	0,     1,     2,     1,     0,
};

static inline void getRandSamplePosition(int& x_sample, int& y_sample, const int x_orig, const int y_orig, const int border, const cv::Size& size) {
	int r = 1+rand()%nSamplesInitPatternTot;
	for(x_sample=0; x_sample<nSamplesInitPatternWidth; ++x_sample) {
		for(y_sample=0; y_sample<nSamplesInitPatternHeight; ++y_sample) {
			r -= anSamplesInitPattern[x_sample*nSamplesInitPatternWidth + y_sample];
			if(r<=0)
				goto stop;
		}
	}
	stop:
	x_sample += x_orig-nSamplesInitPatternWidth/2;
	y_sample += y_orig-nSamplesInitPatternHeight/2;
	if(x_sample<border)
		x_sample = border;
	else if(x_sample>=size.width-border)
		x_sample = size.width-border-1;
	if(y_sample<border)
		y_sample = border;
	else if(y_sample>=size.height-border)
		y_sample = size.height-border-1;
	//printf("orig: [%d,%d]\t sample: [%d,%d]\n",x_orig,y_orig,x_sample,y_sample);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP()
	:	 m_nBGSamples(BGSLBSP_DEFAULT_BG_SAMPLES)
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
													,int nBGSamples )
	:	 m_nBGSamples(nBGSamples)
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
													,int nBGSamples )
	:	 m_nBGSamples(nBGSamples)
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
	int _y_sample, _x_sample;
	if(m_nImgChannels==1) {
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,m_nImgType);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC1);
			for(int k=0; k<nKeyPoints; ++k) {
				const int _y_orig = (int)m_voKeyPoints[k].pt.y;
				const int _x_orig = (int)m_voKeyPoints[k].pt.x;
				getRandSamplePosition(_x_sample,_y_sample,_x_orig,_y_orig,LBSP::DESC_SIZE/2,m_oImgSize);
				m_voBGImg[s].at<uchar>(_y_orig,_x_orig) = oInitImg.at<uchar>(_y_sample,_x_sample);
				m_voBGDesc[s].at<unsigned short>(_y_orig,_x_orig) = oInitDesc.at<unsigned short>(_y_sample,_x_sample);
			}
		}
	}
	else { //m_nImgChannels==3
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,m_nImgType);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC3);
			for(int k=0; k<nKeyPoints; ++k) {
				const int _y_orig = (int)m_voKeyPoints[k].pt.y;
				const int _x_orig = (int)m_voKeyPoints[k].pt.x;
				getRandSamplePosition(_x_sample,_y_sample,_x_orig,_y_orig,LBSP::DESC_SIZE/2,m_oImgSize);
				const int idx_orig_img = oInitImg.step.p[0]*_y_orig + oInitImg.step.p[1]*_x_orig;
				const int idx_orig_desc = oInitDesc.step.p[0]*_y_orig + oInitDesc.step.p[1]*_x_orig;
				const int idx_rand_img = oInitImg.step.p[0]*_y_sample + oInitImg.step.p[1]*_x_sample;
				const int idx_rand_desc = oInitDesc.step.p[0]*_y_sample + oInitDesc.step.p[1]*_x_sample;
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
	cv::Mat oInputImg = _image.getMat(), oInputDesc;
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);

	const int nKeyPoints = (int)m_voKeyPoints.size();
	const int nDescThreshold = m_oExtractor.getAbsThreshold();
	if(m_nImgChannels==1) {
		unsigned short inputdesc;
		for(int k=0; k<nKeyPoints; ++k) {
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<N_SAMPLES_NEEDED_FOR_BG && nSampleIdx<m_nBGSamples) {
				LBSP::computeSingle(oInputImg,m_voBGImg[nSampleIdx],m_voKeyPoints[k],nDescThreshold,inputdesc);
				if(hdist_ushort_8bitLUT(inputdesc,m_voBGDesc[nSampleIdx].at<unsigned short>(m_voKeyPoints[k].pt))<=m_nCurrFGThreshold)
					nGoodSamplesCount++;
				nSampleIdx++;
			}
			if(nGoodSamplesCount<N_SAMPLES_NEEDED_FOR_BG)
				oFGMask.at<uchar>(m_voKeyPoints[k].pt) = UCHAR_MAX;
			else {
				// @@@@@ randomly update BG model
			}
		}
	}
	else { //m_nImgChannels==3
		unsigned short inputdesc[3];
		int hdist[3];
		for(int k=0; k<nKeyPoints; ++k) {
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<N_SAMPLES_NEEDED_FOR_BG && nSampleIdx<m_nBGSamples) {
				CV_DbgAssert(m_voBGDesc[nSampleIdx].step.p[1]==6);
				const unsigned short* bgdesc_ptr = (unsigned short*)(m_voBGDesc[nSampleIdx].data + m_voBGDesc[nSampleIdx].step.p[0]*(int)m_voKeyPoints[k].pt.y + 6*(int)m_voKeyPoints[k].pt.x);
				LBSP::computeSingle(oInputImg,m_voBGImg[nSampleIdx],m_voKeyPoints[k],nDescThreshold,inputdesc);
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
			if(nGoodSamplesCount<N_SAMPLES_NEEDED_FOR_BG)
				oFGMask.at<uchar>(m_voKeyPoints[k].pt) = UCHAR_MAX;
			else {
				// @@@@@ randomly update BG model
			}
		}
	}
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
