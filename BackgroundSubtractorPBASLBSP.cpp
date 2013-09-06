#include "BackgroundSubtractorPBASLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

#define R_OFFST (0.1000f)
#define R_SCALE (1.7500f)
#define R_INCR  (1.0750f)
#define R_DECR  (0.9750f)
#define R_LOWER (0.6500f)
#define R_UPPER (2.0000f)

#define R2_OFFST (0.075f)
#define R2_INCR  (0.005f)
#define R2_DECR  (0.001f)
#define R2_LOWER (0.950f)
#define R2_UPPER (1.050f)

#define T_OFFST (0.0001f)
#define T_SCALE (1.0000f)
#define T_DECR  (0.0500f)
#define T_INCR  (1.0000f)
#define T_LOWER (2.0000f)
#define T_UPPER (200.00f)

#define N_SAMPLES_FOR_MEAN (m_nBGSamples)

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP()
	:	 m_nBGSamples(BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES)
		,m_nRequiredBGSamples(BGSPBASLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES)
		,m_voBGImg(BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES)
		,m_voBGDesc(BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES)
		,m_nDefaultDescDistThreshold(BGSPBASLBSP_DEFAULT_DESC_DIST_THRESHOLD)
		,m_nDefaultColorDistThreshold(BGSPBASLBSP_DEFAULT_COLOR_DIST_THRESHOLD)
		,m_fDefaultUpdateRate(BGSPBASLBSP_DEFAULT_LEARNING_RATE)
		,m_bLBSPUsingRelThreshold(false)
		,m_nLBSPThreshold(LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD)
		,m_fLBSPThreshold(-1)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDefaultDescDistThreshold>0);
	CV_Assert(m_nDefaultColorDistThreshold>0);
	CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
	CV_Assert(m_nLBSPThreshold>0);
}

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(	 int nLBSPThreshold
															,int nInitDescDistThreshold
															,int nInitColorDistThreshold
															,float fInitUpdateRate
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_voBGDesc(nBGSamples)
		,m_nDefaultDescDistThreshold(nInitDescDistThreshold)
		,m_nDefaultColorDistThreshold(nInitColorDistThreshold)
		,m_fDefaultUpdateRate(fInitUpdateRate)
		,m_bLBSPUsingRelThreshold(false)
		,m_nLBSPThreshold(nLBSPThreshold)
		,m_fLBSPThreshold(-1)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDefaultDescDistThreshold>0);
	CV_Assert(m_nDefaultColorDistThreshold>0);
	CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
	CV_Assert(m_nLBSPThreshold>0);
}

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(	 float fLBSPThreshold
															,int nInitDescDistThreshold
															,int nInitColorDistThreshold
															,float fInitUpdateRate
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_voBGImg(nBGSamples)
		,m_voBGDesc(nBGSamples)
		,m_nDefaultDescDistThreshold(nInitDescDistThreshold)
		,m_nDefaultColorDistThreshold(nInitColorDistThreshold)
		,m_fDefaultUpdateRate(fInitUpdateRate)
		,m_bLBSPUsingRelThreshold(true)
		,m_nLBSPThreshold(-1)
		,m_fLBSPThreshold(fLBSPThreshold)
		,m_bInitialized(false) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDefaultDescDistThreshold>0);
	CV_Assert(m_nDefaultColorDistThreshold>0);
	CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
	CV_Assert(m_fLBSPThreshold>0);
}

BackgroundSubtractorPBASLBSP::~BackgroundSubtractorPBASLBSP() {}

void BackgroundSubtractorPBASLBSP::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	m_oImgSize = oInitImg.size();
	m_nImgType = oInitImg.type();
	m_nImgChannels = oInitImg.channels();
	m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(1.0f);
	m_oDistThresholdVariationFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdVariationFrame = cv::Scalar(R2_LOWER);
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(m_fDefaultUpdateRate);
	m_oMeanMinDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame = cv::Scalar(0.0f);
	m_oLastFGMask.create(m_oImgSize,CV_8UC1);
	m_oLastFGMask = cv::Scalar(0);
	m_oFloodedFGMask.create(m_oImgSize,CV_8UC1);
	m_oFloodedFGMask = cv::Scalar(0);

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

void BackgroundSubtractorPBASLBSP::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
	CV_DbgAssert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	static const int nChannelSize = UCHAR_MAX;
	static const int nDescSize = LBSP::DESC_SIZE*8;
	if(m_nImgChannels==1) {
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = oInputImg.step.p[0]*y + x;
			const int ushrt_idx = uchar_idx*2;
			const int flt32_idx = uchar_idx*4;
			int nMinColorDist=nChannelSize;
			int nMinDescDist=nDescSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			const int nCurrColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDefaultColorDistThreshold);
			const int nCurrDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDefaultDescDistThreshold);;
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				int nColorDist = absdiff_uchar(oInputImg.data[uchar_idx],m_voBGImg[nSampleIdx].data[uchar_idx]);
				if(nColorDist<=nCurrColorDistThreshold*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT) {
					unsigned short nCurrInputDesc;
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_fLBSPThreshold,nCurrInputDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_nLBSPThreshold,nCurrInputDesc);
					int nDescDist = hdist_ushort_8bitLUT(nCurrInputDesc,*((unsigned short*)(m_voBGDesc[nSampleIdx].data+ushrt_idx)));
					if(nDescDist<=nCurrDescDistThreshold) {
						if(nMinColorDist>nColorDist)
							nMinColorDist = nColorDist;
						if(nMinDescDist>nDescDist)
							nMinDescDist = nDescDist;
						nGoodSamplesCount++;
					}
				}
				nSampleIdx++;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(N_SAMPLES_FOR_MEAN-1) + (((float)nMinColorDist/nChannelSize)+((float)nMinDescDist/nDescSize))/2)/N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrLearningRate += T_INCR/((*pfCurrMeanMinDist)*T_SCALE+T_OFFST);
				if((*pfCurrLearningRate)>T_UPPER)
					*pfCurrLearningRate = T_UPPER;
			}
			else {
				const uchar nLearningRate = learningRateOverride>0?(uchar)learningRateOverride:(uchar)(*pfCurrLearningRate);
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
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,bgdesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,bgdesc);
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y_rand,x_rand);
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bgdesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bgdesc);
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.data[uchar_idx];
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
				*pfCurrLearningRate -= T_DECR/((*pfCurrMeanMinDist)*T_SCALE+T_OFFST);
				if((*pfCurrLearningRate)<T_LOWER)
					*pfCurrLearningRate = T_LOWER;
			}
			if((*pfCurrMeanMinDist)>R2_OFFST && (oFGMask.data[uchar_idx]!=m_oLastFGMask.data[uchar_idx])) {
				if((*pfCurrDistThresholdVariationFactor)<R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) -= R2_DECR;
			}
			if((*pfCurrDistThresholdFactor)<R_LOWER+(*pfCurrMeanMinDist)*R_SCALE+R_OFFST) {
				if((*pfCurrDistThresholdFactor)<R_UPPER)
					(*pfCurrDistThresholdFactor) *= R_INCR*(*pfCurrDistThresholdVariationFactor);
			}
			else if((*pfCurrDistThresholdFactor)>R_LOWER)
				(*pfCurrDistThresholdFactor) *= R_DECR*(*pfCurrDistThresholdVariationFactor);
		}
	}
	else { //m_nImgChannels==3
		const int desc_row_step = m_voBGDesc[0].step.p[0];
		const int img_row_step = m_voBGImg[0].step.p[0];
		static const int nTotChannelSize = nChannelSize*3;
		static const int nTotDescSize = nDescSize*3;
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = oInputImg.cols*y + x;
			const int rgbimg_idx = uchar_idx*3;
			const int descimg_idx = uchar_idx*6;
			const int flt32_idx = uchar_idx*4;
			int nMinTotColorDist=nTotChannelSize;
			int nMinTotDescDist=nTotDescSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			const int nCurrTotColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDefaultColorDistThreshold*3);
			const int nCurrTotDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDefaultDescDistThreshold*3);
#if BGSPBASLBSP_USE_SC_THRS_VALIDATION
			const int nCurrSCColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDefaultColorDistThreshold*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const int nCurrSCDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDefaultDescDistThreshold*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSPBASLBSP_USE_SC_THRS_VALIDATION
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const unsigned short* bgdesc_ptr = (unsigned short*)(m_voBGDesc[nSampleIdx].data+descimg_idx);
				const uchar* inputimg_ptr = oInputImg.data+rgbimg_idx;
				const uchar* bgimg_ptr = m_voBGImg[nSampleIdx].data+rgbimg_idx;
				int nTotColorDist = 0;
				int nTotDescDist = 0;
				for(int c=0;c<3; ++c) {
					int nColorDist = absdiff_uchar(inputimg_ptr[c],bgimg_ptr[c]);
#if BGSPBASLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto skip;
#endif //BGSPBASLBSP_USE_SC_THRS_VALIDATION
					unsigned short nCurrInputDesc;
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeSingleRGBRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_fLBSPThreshold,nCurrInputDesc);
					else
						LBSP::computeSingleRGBAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_nLBSPThreshold,nCurrInputDesc);
					int nDescDist = hdist_ushort_8bitLUT(nCurrInputDesc,bgdesc_ptr[c]);
#if BGSPBASLBSP_USE_SC_THRS_VALIDATION
					if(nDescDist>nCurrSCDescDistThreshold)
						goto skip;
#endif //BGSPBASLBSP_USE_SC_THRS_VALIDATION
					nTotColorDist += nColorDist;
					nTotDescDist += nDescDist;
				}
				if(nTotDescDist<=nCurrTotDescDistThreshold && nTotColorDist<=nCurrTotColorDistThreshold) {
					if(nMinTotColorDist>nTotColorDist)
						nMinTotColorDist = nTotColorDist;
					if(nMinTotDescDist>nTotDescDist)
						nMinTotDescDist = nTotDescDist;
					nGoodSamplesCount++;
				}
#if BGSPBASLBSP_USE_SC_THRS_VALIDATION
				skip:
#endif //BGSPBASLBSP_USE_SC_THRS_VALIDATION
				nSampleIdx++;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(N_SAMPLES_FOR_MEAN-1) + (((float)nMinTotColorDist/nTotChannelSize)+((float)nMinTotDescDist/nTotDescSize))/2)/N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrLearningRate += T_INCR/((*pfCurrMeanMinDist)*T_SCALE+T_OFFST);
				if((*pfCurrLearningRate)>T_UPPER)
					*pfCurrLearningRate = T_UPPER;
			}
			else {
				const uchar nLearningRate = learningRateOverride>0?(uchar)learningRateOverride:(uchar)(*pfCurrLearningRate);
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
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,bgdesc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,bgdesc_ptr);
					const int img_row_step = m_voBGImg[0].step.p[0];
					for(int n=0; n<3; ++n)
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + n) = *(oInputImg.data + img_row_step*y_rand + 3*x_rand + n);
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bgdesc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bgdesc_ptr);
					for(int n=0; n<3; ++n)
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + n) = *(oInputImg.data+rgbimg_idx+n);
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
				*pfCurrLearningRate -= T_DECR/((*pfCurrMeanMinDist)*T_SCALE+T_OFFST);
				if((*pfCurrLearningRate)<T_LOWER)
					*pfCurrLearningRate = T_LOWER;
			}
			if((*pfCurrMeanMinDist)>R2_OFFST && (oFGMask.data[uchar_idx]!=m_oLastFGMask.data[uchar_idx])) {
				if((*pfCurrDistThresholdVariationFactor)<R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) -= R2_DECR;
			}
			if((*pfCurrDistThresholdFactor)<R_LOWER+(*pfCurrMeanMinDist)*R_SCALE+R_OFFST) {
				if((*pfCurrDistThresholdFactor)<R_UPPER)
					(*pfCurrDistThresholdFactor) *= R_INCR*(*pfCurrDistThresholdVariationFactor);
			}
			else if((*pfCurrDistThresholdFactor)>R_LOWER)
				(*pfCurrDistThresholdFactor) *= R_DECR*(*pfCurrDistThresholdVariationFactor);
		}
	}
	/*cv::Mat oMeanMinDistFrameNormalized, oDistThresholdFrameNormalized, oDistThresholdVariationFrameNormalized, oUpdateRateFrameNormalized;
	oMeanMinDistFrameNormalized = m_oMeanMinDistFrame;
	oDistThresholdFrameNormalized = (m_oDistThresholdFrame-cv::Scalar(R_LOWER))/(R_UPPER-R_LOWER);
	oDistThresholdVariationFrameNormalized = (m_oDistThresholdVariationFrame-cv::Scalar(R2_LOWER))/(R2_UPPER-R2_LOWER);
	oUpdateRateFrameNormalized = (m_oUpdateRateFrame-cv::Scalar(T_LOWER))/(T_UPPER-T_LOWER);
	cv::Point dbg1(60,40), dbg2(218,132);
	cv::circle(oMeanMinDistFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oMeanMinDistFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::circle(oDistThresholdFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oDistThresholdFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::circle(oDistThresholdVariationFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oDistThresholdVariationFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::circle(oUpdateRateFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oUpdateRateFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::imshow("m(x)",oMeanMinDistFrameNormalized);
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " m(" << dbg1 << ") = " << m_oMeanMinDistFrame.at<float>(dbg1) << "  ,  m(" << dbg2 << ") = " << m_oMeanMinDistFrame.at<float>(dbg2) << std::endl;
	std::cout << std::fixed << std::setprecision(5) << " r(" << dbg1 << ") = " << m_oDistThresholdFrame.at<float>(dbg1) << "  ,  r(" << dbg2 << ") = " << m_oDistThresholdFrame.at<float>(dbg2) << std::endl;
	std::cout << std::fixed << std::setprecision(5) << "r2(" << dbg1 << ") = " << m_oDistThresholdVariationFrame.at<float>(dbg1) << "  , r2(" << dbg2 << ") = " << m_oDistThresholdVariationFrame.at<float>(dbg2) << std::endl;
	cv::waitKey(1);*/
	oFGMask.copyTo(m_oLastFGMask);
	//cv::imshow("pure seg",oFGMask);
	cv::medianBlur(oFGMask,oFGMask,3);
	//cv::imshow("median3",oFGMask);
	oFGMask.copyTo(m_oFloodedFGMask);
	cv::dilate(m_oFloodedFGMask,m_oFloodedFGMask,cv::Mat());
	//cv::imshow("median3 + dilate3",m_oFloodedFGMask);
	cv::erode(m_oFloodedFGMask,m_oFloodedFGMask,cv::Mat());
	//cv::imshow("median3 + dilate3 + erode3",m_oFloodedFGMask);
	cv::floodFill(m_oFloodedFGMask,cv::Point(0,0),255);
	cv::bitwise_not(m_oFloodedFGMask,m_oFloodedFGMask);
	//cv::imshow("median3 de3 fill region",m_oFloodedFGMask);
	cv::bitwise_or(m_oFloodedFGMask,m_oLastFGMask,oFGMask);
	//cv::imshow("median3 post-fill",oFGMask);
	cv::medianBlur(oFGMask,oFGMask,9);
	//cv::imshow("median3 post-fill, +median9 ",oFGMask);
	//cv::waitKey(0);
}

cv::AlgorithmInfo* BackgroundSubtractorPBASLBSP::info() const {
	CV_Assert(false); // NOT IMPL @@@@@
	return NULL;
}

void BackgroundSubtractorPBASLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_DbgAssert(!m_voBGImg.empty());
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_8UC(m_nImgChannels));
	for(int n=0; n<m_nBGSamples; ++n) {
		for(int i=0; i<m_oImgSize.height; ++i) {
			for(int j=0; j<m_oImgSize.width; ++j) {
				for(int c=0; c<m_nImgChannels; ++c) {
					oAvgBGImg.data[oAvgBGImg.step.p[0]*i+oAvgBGImg.step.p[1]*j+c] += m_voBGImg[n].data[oAvgBGImg.step.p[0]*i+oAvgBGImg.step.p[1]*j+c]/m_nBGSamples;
				}
			}
		}
	}
	oAvgBGImg.copyTo(backgroundImage);
}

void BackgroundSubtractorPBASLBSP::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
	CV_DbgAssert(!m_voBGImg.empty());
	CV_DbgAssert(LBSP::DESC_SIZE==2);
	cv::Mat oAvgBGDescImg = cv::Mat::zeros(m_oImgSize,CV_16UC(m_nImgChannels));
	for(int n=0; n<m_nBGSamples; ++n) {
		for(int i=0; i<m_oImgSize.height; ++i) {
			for(int j=0; j<m_oImgSize.width; ++j) {
				unsigned short* avg = (unsigned short*)(oAvgBGDescImg.data+oAvgBGDescImg.step.p[0]*i+oAvgBGDescImg.step.p[1]*j);
				unsigned short* sample = (unsigned short*)(m_voBGDesc[n].data+oAvgBGDescImg.step.p[0]*i+oAvgBGDescImg.step.p[1]*j);
				for(int c=0; c<m_nImgChannels; ++c) {
					avg[c] += sample[c]/m_nBGSamples;
				}
			}
		}
	}
	oAvgBGDescImg.copyTo(backgroundDescImage);
}

std::vector<cv::KeyPoint> BackgroundSubtractorPBASLBSP::getBGKeyPoints() const {
	return m_voKeyPoints;
}

void BackgroundSubtractorPBASLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	LBSP::validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voKeyPoints = keypoints;
}
