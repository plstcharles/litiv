#include "BackgroundSubtractorViBeLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

BackgroundSubtractorViBeLBSP::BackgroundSubtractorViBeLBSP()
	:	 m_nBGSamples(BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES)
		,m_nRequiredBGSamples(BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES)
		,m_voBGImg(BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES)
		,m_voBGDesc(BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES)
		,m_nDescDistThreshold(BGSVIBELBSP_DEFAULT_DESC_DIST_THRESHOLD)
		,m_nColorDistThreshold(BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD)
		,m_bLBSPUsingRelThreshold(false)
		,m_nLBSPThreshold(LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD)
		,m_fLBSPThreshold(-1)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDescDistThreshold>0);
	CV_Assert(m_nColorDistThreshold>0);
	CV_Assert(m_nLBSPThreshold>0);
}

BackgroundSubtractorViBeLBSP::BackgroundSubtractorViBeLBSP(  int nLBSPThreshold
															,int nDescDistThreshold
															,int nColorDistThreshold
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_voBGDesc(nBGSamples)
		,m_nDescDistThreshold(nDescDistThreshold)
		,m_nColorDistThreshold(nColorDistThreshold)
		,m_bLBSPUsingRelThreshold(false)
		,m_nLBSPThreshold(nLBSPThreshold)
		,m_fLBSPThreshold(-1)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDescDistThreshold>0);
	CV_Assert(m_nColorDistThreshold>0);
	CV_Assert(m_nLBSPThreshold>0);
}

BackgroundSubtractorViBeLBSP::BackgroundSubtractorViBeLBSP(	 float fLBSPThreshold
															,int nDescDistThreshold
															,int nColorDistThreshold
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_voBGDesc(nBGSamples)
		,m_nDescDistThreshold(nDescDistThreshold)
		,m_nColorDistThreshold(nColorDistThreshold)
		,m_bLBSPUsingRelThreshold(true)
		,m_nLBSPThreshold(-1)
		,m_fLBSPThreshold(fLBSPThreshold)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDescDistThreshold>0);
	CV_Assert(m_nColorDistThreshold>0);
	CV_Assert(m_fLBSPThreshold>0);
}

BackgroundSubtractorViBeLBSP::~BackgroundSubtractorViBeLBSP() {}

void BackgroundSubtractorViBeLBSP::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3);
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();

	if(voKeyPoints.empty()) {
		// init keypoints used for the extractor :
		cv::DenseFeatureDetector oKPDDetector(1.f, 1, 1.f, 1, 0, true, false);
		if(m_voKeyPoints.capacity()<(size_t)(m_oImgSize.width*m_oImgSize.height))
			m_voKeyPoints.reserve(m_oImgSize.width*m_oImgSize.height);
		oKPDDetector.detect(cv::Mat(m_oImgSize,m_nImgType), m_voKeyPoints);
	}
	else
		m_voKeyPoints = voKeyPoints;
	LBSP::validateKeyPoints(m_voKeyPoints,m_oImgSize);
	CV_Assert(!m_voKeyPoints.empty());

	// init bg model samples :
	cv::Mat oInitDesc;
	// create an extractor this one time, for a batch job
	if(m_bLBSPUsingRelThreshold) {
		LBSP oExtractor(m_fLBSPThreshold);
		oExtractor.compute2(oInitImg,m_voKeyPoints,oInitDesc);
	}
	else {
		LBSP oExtractor(m_nLBSPThreshold);
		oExtractor.compute2(oInitImg,m_voKeyPoints,oInitDesc);
	}
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
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				m_voBGImg[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
				m_voBGDesc[s].at<unsigned short>(y_orig,x_orig) = oInitDesc.at<unsigned short>(y_sample,x_sample);
			}
		}
	}
	else { //m_nImgChannels==3
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,m_nImgType);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC3);
			m_voBGImg[s] = cv::Scalar_<uchar>(0,0,0);
			m_voBGDesc[s] = cv::Scalar_<ushort>(0,0,0);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
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

void BackgroundSubtractorViBeLBSP::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	CV_DbgAssert(m_bInitialized);
	CV_DbgAssert(learningRate>0);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	const int nLearningRate = (int)learningRate;
	if(m_nImgChannels==1) {
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = oInputImg.step.p[0]*y + x;
			const int ushrt_idx = uchar_idx*2;
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
				if(absdiff_uchar(oInputImg.data[uchar_idx],m_voBGImg[nSampleIdx].data[uchar_idx])<=m_nColorDistThreshold*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT) {
#else //!BGSVIBELBSP_USE_COLOR_COMPLEMENT
				{
#endif //!BGSVIBELBSP_USE_COLOR_COMPLEMENT
					unsigned short nCurrInputDesc;
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_fLBSPThreshold,nCurrInputDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_nLBSPThreshold,nCurrInputDesc);
					if(hdist_ushort_8bitLUT(nCurrInputDesc,*((unsigned short*)(m_voBGDesc[nSampleIdx].data+ushrt_idx)))<=m_nDescDistThreshold)
							nGoodSamplesCount++;
				}
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.data[uchar_idx] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					unsigned short& bgdesc = m_voBGDesc[s_rand].at<unsigned short>(y,x);
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bgdesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bgdesc);
					m_voBGImg[s_rand].data[uchar_idx] = oInputImg.data[uchar_idx];
				}
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					unsigned short& bgdesc = m_voBGDesc[s_rand].at<unsigned short>(y_rand,x_rand);
#if BGSVIBELBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,bgdesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,bgdesc);
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y_rand,x_rand);
#else //!BGSVIBELBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bgdesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bgdesc);
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.data[uchar_idx];
#endif //!BGSVIBELBSP_USE_SELF_DIFFUSION
				}
			}
		}
	}
	else { //m_nImgChannels==3
		const int nCurrDescDistThreshold = m_nDescDistThreshold*3;
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const int nCurrColorDistThreshold = m_nColorDistThreshold*3;
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#if BGSVIBELBSP_USE_SC_THRS_VALIDATION
		const int nCurrSCDescDistThreshold = (int)(m_nDescDistThreshold*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const int nCurrSCColorDistThreshold = (int)(m_nColorDistThreshold*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#endif //BGSVIBELBSP_USE_SC_THRS_VALIDATION
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const int desc_row_step = m_voBGDesc[0].step.p[0];
		const int img_row_step = m_voBGImg[0].step.p[0];
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int base_idx = oInputImg.cols*y + x;
			const int rgbimg_idx = base_idx*3;
			const int descimg_idx = base_idx*6;
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const unsigned short* bgdesc_ptr = (unsigned short*)(m_voBGDesc[nSampleIdx].data+descimg_idx);
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
				const uchar* inputimg_ptr = oInputImg.data+rgbimg_idx;
				const uchar* bgimg_ptr = m_voBGImg[nSampleIdx].data+rgbimg_idx;
				int nTotColorDist = 0;
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
				int nTotDescDist = 0;
				for(int c=0;c<3; ++c) {
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
					int nColorDist = absdiff_uchar(inputimg_ptr[c],bgimg_ptr[c]);
#if BGSVIBELBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto skip;
#endif //BGSVIBELBSP_USE_SC_THRS_VALIDATION
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
					unsigned short nCurrInputDesc;
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeSingleRGBRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_fLBSPThreshold,nCurrInputDesc);
					else
						LBSP::computeSingleRGBAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_nLBSPThreshold,nCurrInputDesc);
					int nDescDist = hdist_ushort_8bitLUT(nCurrInputDesc,bgdesc_ptr[c]);
#if BGSVIBELBSP_USE_SC_THRS_VALIDATION
					if(nDescDist>nCurrSCDescDistThreshold)
						goto skip;
#endif //BGSVIBELBSP_USE_SC_THRS_VALIDATION
					nTotColorDist += nColorDist;
					nTotDescDist += nDescDist;
				}
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
				if(nTotDescDist<=nCurrDescDistThreshold && nTotColorDist<=nCurrColorDistThreshold)
#else //!BGSVIBELBSP_USE_COLOR_COMPLEMENT
				if(nTotDescDist<=nCurrDescDistThreshold)
#endif //!BGSVIBELBSP_USE_COLOR_COMPLEMENT
					nGoodSamplesCount++;
#if BGSVIBELBSP_USE_SC_THRS_VALIDATION
				skip:
#endif //BGSVIBELBSP_USE_SC_THRS_VALIDATION
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.data[base_idx] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					unsigned short* bgdesc_ptr = ((unsigned short*)(m_voBGDesc[s_rand].data+descimg_idx));
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bgdesc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bgdesc_ptr);
					for(int n=0; n<3; ++n)
						*(m_voBGImg[s_rand].data + img_row_step*y + 3*x + n) = *(oInputImg.data+rgbimg_idx+n);
					CV_DbgAssert(m_voBGImg[s_rand].at<cv::Vec3b>(y,x)==oInputImg.at<cv::Vec3b>(y,x));
				}
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					unsigned short* bgdesc_ptr = ((unsigned short*)(m_voBGDesc[s_rand].data + desc_row_step*y_rand + 6*x_rand));
#if BGSVIBELBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,bgdesc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,bgdesc_ptr);
					for(int n=0; n<3; ++n)
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + n) = *(oInputImg.data + img_row_step*y_rand + 3*x_rand + n);
#else //!BGSVIBELBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bgdesc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bgdesc_ptr);
					for(int n=0; n<3; ++n)
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + n) = *(oInputImg.data+rgbimg_idx+n);
#endif //!BGSVIBELBSP_USE_SELF_DIFFUSION
				}
			}
		}
	}
	cv::medianBlur(oFGMask,oFGMask,9); // give user access to mblur params... @@@@@
}

cv::AlgorithmInfo* BackgroundSubtractorViBeLBSP::info() const {
	CV_Assert(false); // NOT IMPL @@@@@
	return NULL;
}

void BackgroundSubtractorViBeLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	m_voBGImg[0].copyTo(backgroundImage);
}

void BackgroundSubtractorViBeLBSP::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	m_voBGDesc[0].copyTo(backgroundDescImage);
}

std::vector<cv::KeyPoint> BackgroundSubtractorViBeLBSP::getBGKeyPoints() const {
	return m_voKeyPoints;
}

void BackgroundSubtractorViBeLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
}
