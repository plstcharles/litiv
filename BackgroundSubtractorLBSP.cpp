#include "BackgroundSubtractorLBSP.h"
#include "LBSP.h"
#include "HammingDist.h"
#include <iostream>

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP()
	:	 m_nFGThreshold(BGSLBSP_DEFAULT_FG_THRESHOLD)
		,m_nFGSCThreshold(BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD)
		,m_bInitialized(false)
		,m_oExtractor(LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(	int nDescThreshold,
													int nFGThreshold,
													int nFGSCThreshold
													)
	:	 m_nFGThreshold(nFGThreshold)
		,m_nFGSCThreshold(nFGSCThreshold)
		,m_bInitialized(false)
		,m_oExtractor(nDescThreshold) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(	float fDescThreshold,
													int nFGThreshold,
													int nFGSCThreshold
													)
	:	 m_nFGThreshold(nFGThreshold)
		,m_nFGSCThreshold(nFGSCThreshold)
		,m_bInitialized(false)
		,m_oExtractor(fDescThreshold) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
}

BackgroundSubtractorLBSP::~BackgroundSubtractorLBSP() {}

void BackgroundSubtractorLBSP::initialize(const cv::Mat& oInitImg) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3);
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_nCurrFGThreshold = m_nFGThreshold*m_nImgChannels;
	cv::DenseFeatureDetector oKPDDetector(1.f, 1, 1.f, 1, 0, true, false);
	if(m_voBGKeyPoints.capacity()<(size_t)(m_oImgSize.width*m_oImgSize.height))
		m_voBGKeyPoints.reserve(m_oImgSize.width*m_oImgSize.height);
	oKPDDetector.detect(cv::Mat(m_oImgSize,m_nImgType), m_voBGKeyPoints);
	LBSP::validateKeyPoints(m_voBGKeyPoints,m_oImgSize);
	CV_Assert(!m_voBGKeyPoints.empty());
	CV_Assert(LBSP::DESC_SIZE==2);
	oInitImg.copyTo(m_oBGImg);
	m_oExtractor.compute(m_oBGImg,m_voBGKeyPoints,m_oBGDesc);
	m_oExtractor.setReference(m_oBGImg);
	m_bInitialized = true;
}

void BackgroundSubtractorLBSP::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	CV_DbgAssert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat(), oInputDesc;
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	m_oExtractor.compute(oInputImg,m_voBGKeyPoints,oInputDesc);
	CV_DbgAssert(oInputDesc.size()==m_oBGDesc.size() && oInputDesc.type()==m_oBGDesc.type());
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);
	const int nKeyPoints = (int)m_voBGKeyPoints.size();
	if(m_nImgChannels==1) {
		CV_DbgAssert(oInputDesc.step.p[0]==m_oBGDesc.step.p[0] && oInputDesc.step.p[1]==m_oBGDesc.step.p[1] && oInputDesc.step.p[0]==oInputDesc.step.p[1] && oInputDesc.step.p[1]==2);
		for(int k=0; k<nKeyPoints; ++k) {
			const int idx = oInputDesc.step.p[0]*k; // should be the same steps for both mats... (asserted above)
			if(hdist_ushort_8bitLUT(*((unsigned short*)(oInputDesc.data+idx)),*((unsigned short*)(m_oBGDesc.data+idx)))>m_nCurrFGThreshold)
				oFGMask.at<uchar>(m_voBGKeyPoints[k].pt) = UCHAR_MAX;
		}
	}
	else { //m_nImgChannels==3
		CV_DbgAssert(oInputDesc.step.p[0]==m_oBGDesc.step.p[0] && oInputDesc.step.p[1]==m_oBGDesc.step.p[1] && oInputDesc.step.p[0]==oInputDesc.step.p[1] && oInputDesc.step.p[1]==6);
		int hdist[3];
		for(int k=0; k<nKeyPoints; ++k) {
			const int idx = oInputDesc.step.p[0]*k; // should be the same steps for both mats... (asserted above)
			for(int n=0;n<3; ++n) {
				hdist[n] = hdist_ushort_8bitLUT(((unsigned short*)(oInputDesc.data+idx))[n],((unsigned short*)(m_oBGDesc.data+idx))[n]);
				if(hdist[n]>m_nFGSCThreshold)
					goto foreground;
			}
			if(hdist[0]+hdist[1]+hdist[2]>m_nCurrFGThreshold)
				goto foreground;
			continue;
			foreground:
			oFGMask.at<uchar>(m_voBGKeyPoints[k].pt) = UCHAR_MAX;
		}
	}
}

cv::AlgorithmInfo* BackgroundSubtractorLBSP::info() const {
	CV_Assert(false); // NOT IMPL @@@@@
	return NULL;
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGImage() const {
	return m_oBGImg.clone();
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGDescriptors() const {
	return m_oBGDesc.clone();
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGDescriptorsImage() const {
	cv::Mat oCurrBGDescImg;
	LBSP::recreateDescImage(m_oImgSize,m_voBGKeyPoints,m_oBGDesc,oCurrBGDescImg);
	return oCurrBGDescImg;
}

std::vector<cv::KeyPoint> BackgroundSubtractorLBSP::getBGKeyPoints() const {
	return m_voBGKeyPoints;
}

void BackgroundSubtractorLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	m_oExtractor.validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voBGKeyPoints = keypoints;
}
