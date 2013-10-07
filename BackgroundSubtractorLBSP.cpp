#include "BackgroundSubtractorLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>
#include <exception>

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP()
	:	 nDebugCoordX(0),nDebugCoordY(0)
		,m_nDescDistThreshold(BGSLBSP_DEFAULT_DESC_DIST_THRESHOLD)
		,m_bLBSPUsingRelThreshold(false)
		,m_nLBSPThreshold(LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD)
		,m_fLBSPThreshold(-1)
		,m_bInitialized(false) {
	CV_Assert(m_nDescDistThreshold>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(uchar nLBSPThreshold, int nDescDistThreshold)
	:	 nDebugCoordX(0),nDebugCoordY(0)
		,m_nDescDistThreshold(nDescDistThreshold)
		,m_bLBSPUsingRelThreshold(false)
		,m_nLBSPThreshold(nLBSPThreshold)
		,m_fLBSPThreshold(-1)
		,m_bInitialized(false) {
	CV_Assert(m_nDescDistThreshold>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(float fLBSPThreshold, int nDescDistThreshold)
	:	 nDebugCoordX(0),nDebugCoordY(0)
		,m_nDescDistThreshold(nDescDistThreshold)
		,m_bLBSPUsingRelThreshold(true)
		,m_nLBSPThreshold(-1)
		,m_fLBSPThreshold(fLBSPThreshold)
		,m_bInitialized(false) {
	CV_Assert(m_nDescDistThreshold>0);
	CV_Assert(m_fLBSPThreshold>=0);
}

BackgroundSubtractorLBSP::~BackgroundSubtractorLBSP() {}

void BackgroundSubtractorLBSP::initialize(const cv::Mat& oInitImg) {
	this->initialize(oInitImg,std::vector<cv::KeyPoint>());
}

cv::AlgorithmInfo* BackgroundSubtractorLBSP::info() const {
	CV_DbgAssert(false); // NOT IMPL @@@@@
	return NULL;
}

void BackgroundSubtractorLBSP::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	CV_DbgAssert(LBSP::DESC_SIZE==2);
	CV_DbgAssert(m_bInitialized);
	cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC(m_nImgChannels));
	for(size_t n=0; n<m_voBGDesc.size(); ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				int desc_idx = m_voBGDesc[n].step.p[0]*y + m_voBGDesc[n].step.p[1]*x;
				int flt32_idx = desc_idx*2;
				float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+flt32_idx);
				ushort* oBGDescPtr = (ushort*)(m_voBGDesc[n].data+desc_idx);
				for(int c=0; c<m_nImgChannels; ++c)
					oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDesc.size();
			}
		}
	}
	oAvgBGDesc.convertTo(backgroundDescImage,CV_16U);
}

std::vector<cv::KeyPoint> BackgroundSubtractorLBSP::getBGKeyPoints() const {
	return m_voKeyPoints;
}

void BackgroundSubtractorLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
}
