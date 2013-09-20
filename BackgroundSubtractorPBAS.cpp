#include "BackgroundSubtractorPBAS.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

BackgroundSubtractorPBAS::BackgroundSubtractorPBAS(	 int nInitColorDistThreshold
													,float fInitUpdateRate
													,int nBGSamples
													,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_voBGGrad(nBGSamples)
		,m_nDefaultColorDistThreshold(nInitColorDistThreshold)
		,m_fDefaultUpdateRate(fInitUpdateRate)
		,m_fFormerMeanGradDist(20)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDefaultColorDistThreshold>0);
	CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
}

BackgroundSubtractorPBAS::~BackgroundSubtractorPBAS() {}

cv::AlgorithmInfo* BackgroundSubtractorPBAS::info() const {
	CV_DbgAssert(false); // NOT IMPL @@@@@
	return NULL;
}

void BackgroundSubtractorPBAS::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_DbgAssert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC(m_voBGImg[0].channels()));
	for(size_t n=0; n<m_voBGImg.size(); ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				int uchar_idx = m_voBGImg[n].step.p[0]*y + m_voBGImg[n].step.p[1]*x;
				int flt32_idx = uchar_idx*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+flt32_idx);
				uchar* oBGImgPtr = m_voBGImg[n].data+uchar_idx;
				for(int c=0; c<m_voBGImg[n].channels(); ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_voBGImg.size();
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}
