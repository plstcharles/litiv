#include "BackgroundSubtractorPBASLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP()
	:	 BackgroundSubtractorLBSP()
		,m_nBGSamples(BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES)
		,m_nRequiredBGSamples(BGSPBASLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES)
		,m_nColorDistThreshold(BGSPBASLBSP_DEFAULT_COLOR_DIST_THRESHOLD)
		,m_fDefaultUpdateRate(BGSPBASLBSP_DEFAULT_LEARNING_RATE)
		,m_fFormerMeanGradDist(20) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDescDistThreshold>0);
	CV_Assert(m_nColorDistThreshold>0);
	CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
	CV_Assert(m_nLBSPThreshold>0);
}

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(	 int nLBSPThreshold
															,int nInitDescDistThreshold
															,int nInitColorDistThreshold
															,float fInitUpdateRate
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(nLBSPThreshold,nInitDescDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_fDefaultUpdateRate(fInitUpdateRate)
		,m_fFormerMeanGradDist(20) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDescDistThreshold>0);
	CV_Assert(m_nColorDistThreshold>0);
	CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
	CV_Assert(m_nLBSPThreshold>0);
}

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(	 float fLBSPThreshold
															,int nInitDescDistThreshold
															,int nInitColorDistThreshold
															,float fInitUpdateRate
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_fDefaultUpdateRate(fInitUpdateRate)
		,m_fFormerMeanGradDist(20) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nDescDistThreshold>0);
	CV_Assert(m_nColorDistThreshold>0);
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
#if BGSPBASLBSP_USE_R2_ACCELERATION
	m_oDistThresholdVariationFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdVariationFrame = cv::Scalar(BGSPBASLBSP_R2_LOWER);
#endif //BGSPBASLBSP_USE_R2_ACCELERATION
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(m_fDefaultUpdateRate);
	m_oMeanMinDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame = cv::Scalar(0.0f);
	m_oLastFGMask.create(m_oImgSize,CV_8UC1);
	m_oLastFGMask = cv::Scalar(0);
	m_oFloodedFGMask.create(m_oImgSize,CV_8UC1);
	m_oFloodedFGMask = cv::Scalar(0);
	cv::Mat oBlurredInitImg;
	cv::GaussianBlur(oInitImg,oBlurredInitImg,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInitImg_GradX, oBlurredInitImg_GradY;
	cv::Scharr(oBlurredInitImg,oBlurredInitImg_GradX,CV_16S,1,0,1,0,cv::BORDER_DEFAULT);
	cv::Scharr(oBlurredInitImg,oBlurredInitImg_GradY,CV_16S,0,1,1,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInitImg_AbsGradX, oBlurredInitImg_AbsGradY;
	cv::convertScaleAbs(oBlurredInitImg_GradX,oBlurredInitImg_AbsGradX);
	cv::convertScaleAbs(oBlurredInitImg_GradY,oBlurredInitImg_AbsGradY);
	cv::Mat oBlurredInitImg_AbsGrad;
	cv::addWeighted(oBlurredInitImg_AbsGradX,0.5,oBlurredInitImg_AbsGradY,0.5,0,oBlurredInitImg_AbsGrad);
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
	m_voBGImg.resize(m_nBGSamples);
	m_voBGGrad.resize(m_nBGSamples);
	m_voBGDesc.resize(m_nBGSamples);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,CV_8UC1);
			m_voBGImg[s] = cv::Scalar_<uchar>(0);
			m_voBGGrad[s].create(m_oImgSize,CV_8UC1);
			m_voBGGrad[s] = cv::Scalar_<uchar>(0);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC1);
			m_voBGDesc[s] = cv::Scalar_<ushort>(0);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				m_voBGImg[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
				m_voBGGrad[s].at<uchar>(y_orig,x_orig) = oBlurredInitImg_AbsGrad.at<uchar>(y_sample,x_sample);
				m_voBGDesc[s].at<ushort>(y_orig,x_orig) = oInitDesc.at<ushort>(y_sample,x_sample);
			}
		}
	}
	else { //m_nImgChannels==3
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,CV_8UC3);
			m_voBGImg[s] = cv::Scalar_<uchar>(0,0,0);
			m_voBGGrad[s].create(m_oImgSize,CV_8UC3);
			m_voBGGrad[s] = cv::Scalar_<uchar>(0,0,0);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC3);
			m_voBGDesc[s] = cv::Scalar_<ushort>(0,0,0);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const int idx_orig_img = oInitImg.step.p[0]*y_orig + oInitImg.step.p[1]*x_orig;
				const int idx_orig_desc = oInitDesc.step.p[0]*y_orig + oInitDesc.step.p[1]*x_orig;
				const int idx_rand_img = oInitImg.step.p[0]*y_sample + oInitImg.step.p[1]*x_sample;
				const int idx_rand_desc = oInitDesc.step.p[0]*y_sample + oInitDesc.step.p[1]*x_sample;
				uchar* bg_img_ptr = m_voBGImg[s].data+idx_orig_img;
				uchar* bg_grad_ptr = m_voBGGrad[s].data+idx_orig_img;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDesc[s].data+idx_orig_desc);
				const uchar* init_img_ptr = oInitImg.data+idx_rand_img;
				const uchar* init_grad_ptr = oBlurredInitImg_AbsGrad.data+idx_rand_img;
				const ushort* init_desc_ptr = (ushort*)(oInitDesc.data+idx_rand_desc);
				for(int n=0;n<3; ++n) {
					bg_img_ptr[n] = init_img_ptr[n];
					bg_grad_ptr[n] = init_grad_ptr[n];
					bg_desc_ptr[n] = init_desc_ptr[n];
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
	cv::Mat oBlurredInputImg;
	cv::GaussianBlur(oInputImg,oBlurredInputImg,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInputImg_GradX, oBlurredInputImg_GradY;
	cv::Scharr(oBlurredInputImg,oBlurredInputImg_GradX,CV_16S,1,0,1,0,cv::BORDER_DEFAULT);
	cv::Scharr(oBlurredInputImg,oBlurredInputImg_GradY,CV_16S,0,1,1,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInputImg_AbsGradX, oBlurredInputImg_AbsGradY;
	cv::convertScaleAbs(oBlurredInputImg_GradX,oBlurredInputImg_AbsGradX);
	cv::convertScaleAbs(oBlurredInputImg_GradY,oBlurredInputImg_AbsGradY);
	cv::Mat oBlurredInputImg_AbsGrad;
	cv::addWeighted(oBlurredInputImg_AbsGradX,0.5,oBlurredInputImg_AbsGradY,0.5,0,oBlurredInputImg_AbsGrad);
	//cv::imshow("input grad mag",oBlurredInputImg_AbsGrad);
	//cv::Mat oSampleColorAbsDiff,oSampleGradAbsDiff;
	//cv::absdiff(oInputImg,m_voBGImg[0],oSampleColorAbsDiff);
	//cv::absdiff(oBlurredInputImg_AbsGrad,m_voBGGrad[0],oSampleGradAbsDiff);
	//cv::imshow("oSampleColorAbsDiff",oSampleColorAbsDiff);
	//cv::imshow("oSampleGradAbsDiff",oSampleGradAbsDiff);
	//cv::Mat oNewDistImg;
	//cv::addWeighted(oSampleGradAbsDiff,GRAD_WEIGHT_ALPHA/m_fFormerMeanGradDist,oSampleColorAbsDiff,1.0,0,oNewDistImg);
	//cv::imshow("oNewDistImg",oNewDistImg);
	//cv::Mat oNewDistImgDiff;
	//cv::absdiff(oNewDistImg,oSampleColorAbsDiff,oNewDistImgDiff);
	//cv::imshow("diff oNewDistImg",oNewDistImgDiff);
	int nFrameTotGradDist=0;
	int nFrameTotBadSamplesCount=1;
	if(m_nImgChannels==1) {
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = oInputImg.step.p[0]*y + x;
			const int ushrt_idx = uchar_idx*2;
			const int flt32_idx = uchar_idx*4;
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			float fMinSumDist=(float)nChannelSize;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			int nMinColorDist=nChannelSize;
			int nMinGradDist=nChannelSize;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR)
			int nMinDescDist=nDescSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const float fCurrSumDistThreshold = (*pfCurrDistThresholdFactor)*m_nColorDistThreshold*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const int nCurrColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const int nCurrGradDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold/*m_nGradDistThreshold@@@@*/*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const int nCurrDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold);
			int nGoodSamplesCount=0, nSampleIdx=0;
			int nColorDist,nGradDist,nDescDist;
			ushort nCurrInputDesc;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				nGradDist = absdiff_uchar(oBlurredInputImg_AbsGrad.data[uchar_idx],m_voBGGrad[nSampleIdx].data[uchar_idx]);
				nColorDist = absdiff_uchar(oInputImg.data[uchar_idx],m_voBGImg[nSampleIdx].data[uchar_idx]);
				const float fSumDist = std::min(((BGSPBASLBSP_GRAD_WEIGHT_ALPHA/m_fFormerMeanGradDist)*nGradDist)+nColorDist,(float)nChannelSize);
				if(fSumDist>fCurrSumDistThreshold)
					goto failedcheck1ch;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				nGradDist = absdiff_uchar(oBlurredInputImg_AbsGrad.data[uchar_idx],m_voBGGrad[nSampleIdx].data[uchar_idx]);
				if(nGradDist>nCurrGradDistThreshold)
					goto failedcheck1ch;
				nColorDist = absdiff_uchar(oInputImg.data[uchar_idx],m_voBGImg[nSampleIdx].data[uchar_idx]);
				if(nColorDist>nCurrColorDistThreshold)
					goto failedcheck1ch;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				if(m_bLBSPUsingRelThreshold)
					LBSP::computeGrayscaleRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_fLBSPThreshold,nCurrInputDesc);
				else
					LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_nLBSPThreshold,nCurrInputDesc);
				nDescDist = hdist_ushort_8bitLUT(nCurrInputDesc,*((ushort*)(m_voBGDesc[nSampleIdx].data+ushrt_idx)));
				if(nDescDist>nCurrDescDistThreshold)
					goto failedcheck1ch;
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				if(fMinSumDist>fSumDist)
					fMinSumDist = fSumDist;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				if(nMinColorDist>nColorDist)
					nMinColorDist = nColorDist;
				if(nMinGradDist>nGradDist)
					nMinGradDist = nGradDist;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				if(nMinDescDist>nDescDist)
					nMinDescDist = nDescDist;
				nGoodSamplesCount++;
				goto endcheck1ch;
				failedcheck1ch:
				nFrameTotGradDist += nGradDist;
				nFrameTotBadSamplesCount++;
				endcheck1ch:
				nSampleIdx++;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((fMinSumDist/nChannelSize)+((float)nMinDescDist/nDescSize))/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + (((float)nMinColorDist/nChannelSize)+((float)nMinGradDist/nChannelSize)+((float)nMinDescDist/nDescSize))/3)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
			}
			else {
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					ushort* bg_desc_ptr = ((ushort*)(m_voBGDesc[s_rand].data+ushrt_idx));
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,*bg_desc_ptr);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,*bg_desc_ptr);
					m_voBGGrad[s_rand].data[uchar_idx] = oBlurredInputImg_AbsGrad.data[uchar_idx];
					m_voBGImg[s_rand].data[uchar_idx] = oInputImg.data[uchar_idx];
				}
				if((rand()%nLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					int s_rand = rand()%m_nBGSamples;
					ushort& nRandInputDesc = m_voBGDesc[s_rand].at<ushort>(y_rand,x_rand);
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,nRandInputDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,nRandInputDesc);
					m_voBGGrad[s_rand].at<uchar>(y_rand,x_rand) = oBlurredInputImg_AbsGrad.at<uchar>(y_rand,x_rand);
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y_rand,x_rand);
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,nRandInputDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,nRandInputDesc);
					m_voBGGrad[s_rand].at<uchar>(y_rand,x_rand) = oBlurredInputImg_AbsGrad.data[uchar_idx];
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.data[uchar_idx];
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
				*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
			}
#if BGSPBASLBSP_USE_R2_ACCELERATION
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && (oFGMask.data[uchar_idx]!=m_oLastFGMask.data[uchar_idx])) {
				if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_DECR*(*pfCurrDistThresholdVariationFactor);
#else //!BGSPBASLBSP_USE_R2_ACCELERATION
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_INCR;
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_DECR;
#endif //!BGSPBASLBSP_USE_R2_ACCELERATION
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
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			float fMinTotSumDist=(float)nTotChannelSize;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			int nMinTotColorDist=nTotChannelSize;
			int nMinTotGradDist=nTotChannelSize;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR)
			int nMinTotDescDist=nTotDescSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const float fCurrTotSumDistThreshold = ((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const int nCurrTotColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
			const int nCurrTotGradDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold/*m_nGradDistThreshold@@@@*/*3);
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const int nCurrTotDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
#if BGSLBSP_USE_SC_THRS_VALIDATION
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const float fCurrSCTotSumDistThreshold = ((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const int nCurrSCColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const int nCurrSCGradDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold/*m_nGradDistThreshold@@@@*/*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			const int nCurrSCDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
			int nGoodSamplesCount=0, nSampleIdx=0;
			ushort anCurrInputDesc[3];
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* bg_desc_ptr = (ushort*)(m_voBGDesc[nSampleIdx].data+descimg_idx);
				const uchar* bg_grad_ptr = m_voBGGrad[nSampleIdx].data+rgbimg_idx;
				const uchar* input_grad_ptr = oBlurredInputImg_AbsGrad.data+rgbimg_idx;
				const uchar* bg_img_ptr = m_voBGImg[nSampleIdx].data+rgbimg_idx;
				const uchar* input_img_ptr = oInputImg.data+rgbimg_idx;
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				float fTotSumDist = 0.0f;
				int nTotGradDist = 0;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				int nTotColorDist = 0;
				int nTotGradDist = 0;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				int nTotDescDist = 0;
				for(int c=0;c<3; ++c) {
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
					const int nColorDist = absdiff_uchar(input_img_ptr[c],bg_img_ptr[c]);
					const int nGradDist = absdiff_uchar(input_grad_ptr[c],bg_grad_ptr[c]);
					const float fSumDist = std::min(((BGSPBASLBSP_GRAD_WEIGHT_ALPHA/m_fFormerMeanGradDist)*nGradDist)+nColorDist,(float)nChannelSize);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(fSumDist>fCurrSCTotSumDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
					const int nColorDist = absdiff_uchar(input_img_ptr[c],bg_img_ptr[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					const int nGradDist = absdiff_uchar(input_grad_ptr[c],bg_grad_ptr[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nGradDist>nCurrSCGradDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeSingleRGBRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_fLBSPThreshold,anCurrInputDesc[c]);
					else
						LBSP::computeSingleRGBAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_nLBSPThreshold,anCurrInputDesc[c]);
					const int nDescDist = hdist_ushort_8bitLUT(anCurrInputDesc[c],bg_desc_ptr[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
					fTotSumDist += fSumDist;
					nTotGradDist += nGradDist;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
					nTotColorDist += nColorDist;
					nTotGradDist += nGradDist;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
					nTotDescDist += nDescDist;
				}
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				if(nTotDescDist>nCurrTotDescDistThreshold || fTotSumDist>fCurrTotSumDistThreshold)
					goto failedcheck3ch;
				if(fMinTotSumDist>fTotSumDist)
					fMinTotSumDist = fTotSumDist;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				if(nTotDescDist>nCurrTotDescDistThreshold || nTotColorDist>nCurrTotColorDistThreshold || nTotGradDist>nCurrTotGradDistThreshold)
					goto failedcheck3ch;
				if(nMinTotColorDist>nTotColorDist)
					nMinTotColorDist = nTotColorDist;
				if(nMinTotGradDist>nTotGradDist)
					nMinTotGradDist = nTotGradDist;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
				if(nMinTotDescDist>nTotDescDist)
					nMinTotDescDist = nTotDescDist;
				nGoodSamplesCount++;
				goto endcheck3ch;
				failedcheck3ch:
				nFrameTotGradDist += nTotGradDist;
				nFrameTotBadSamplesCount++;
				endcheck3ch:
				nSampleIdx++;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
#if BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((fMinTotSumDist/nTotChannelSize)+((float)nMinTotDescDist/nTotDescSize))/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
#else //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + (((float)nMinTotColorDist/nTotChannelSize)+((float)nMinTotGradDist/nTotChannelSize)+((float)nMinTotDescDist/nTotDescSize))/3)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
#endif //!BGSPBASLBSP_MIX_GRADIENT_WITH_COLOR
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
			}
			else {
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					ushort* bg_desc_ptr = ((ushort*)(m_voBGDesc[s_rand].data+descimg_idx));
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bg_desc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bg_desc_ptr);
					for(int c=0; c<3; ++c) {
						*(m_voBGImg[s_rand].data+rgbimg_idx+c) = *(oInputImg.data+rgbimg_idx+c);
						*(m_voBGGrad[s_rand].data+rgbimg_idx+c) = *(oBlurredInputImg_AbsGrad.data+rgbimg_idx+c);
					}
				}
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					ushort* bg_desc_ptr = ((ushort*)(m_voBGDesc[s_rand].data + desc_row_step*y_rand + 6*x_rand));
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,bg_desc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,bg_desc_ptr);
					const int img_row_step = m_voBGImg[0].step.p[0];
					for(int c=0; c<3; ++c) {
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oInputImg.data + img_row_step*y_rand + 3*x_rand + c);
						*(m_voBGGrad[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oBlurredInputImg_AbsGrad.data + img_row_step*y_rand + 3*x_rand + c);
					}
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bg_desc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bg_desc_ptr);
					for(int c=0; c<3; ++c) {
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oInputImg.data+rgbimg_idx+c);
						*(m_voBGGrad[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oBlurredInputImg_AbsGrad.data+rgbimg_idx+c);
					}
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
				*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
			}
#if BGSPBASLBSP_USE_R2_ACCELERATION
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && (oFGMask.data[uchar_idx]!=m_oLastFGMask.data[uchar_idx])) {
				if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_DECR*(*pfCurrDistThresholdVariationFactor);
#else //!BGSPBASLBSP_USE_R2_ACCELERATION
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_INCR;
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) *= BGSPBASLBSP_R_DECR;
#endif //!BGSPBASLBSP_USE_R2_ACCELERATION
		}
	}
	m_fFormerMeanGradDist = std::max(((float)nFrameTotGradDist)/nFrameTotBadSamplesCount,20.0f);
	/*cv::Point dbg1(60,40), dbg2(218,132);
	cv::Mat oMeanMinDistFrameNormalized = m_oMeanMinDistFrame;
	cv::circle(oMeanMinDistFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oMeanMinDistFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::imshow("m(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " m(" << dbg1 << ") = " << m_oMeanMinDistFrame.at<float>(dbg1) << "  ,  m(" << dbg2 << ") = " << m_oMeanMinDistFrame.at<float>(dbg2) << std::endl;
	cv::Mat oDistThresholdFrameNormalized = (m_oDistThresholdFrame-cv::Scalar(R_LOWER))/(R_UPPER-R_LOWER);
	cv::circle(oDistThresholdFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oDistThresholdFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " r(" << dbg1 << ") = " << m_oDistThresholdFrame.at<float>(dbg1) << "  ,  r(" << dbg2 << ") = " << m_oDistThresholdFrame.at<float>(dbg2) << std::endl;
#if BGSPBASLBSP_USE_R2_ACCELERATION
	cv::Mat oDistThresholdVariationFrameNormalized = (m_oDistThresholdVariationFrame-cv::Scalar(R2_LOWER))/(R2_UPPER-R2_LOWER);
	cv::circle(oDistThresholdVariationFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oDistThresholdVariationFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "r2(" << dbg1 << ") = " << m_oDistThresholdVariationFrame.at<float>(dbg1) << "  , r2(" << dbg2 << ") = " << m_oDistThresholdVariationFrame.at<float>(dbg2) << std::endl;
#endif //BGSPBASLBSP_USE_R2_ACCELERATION
	cv::Mat oUpdateRateFrameNormalized = (m_oUpdateRateFrame-cv::Scalar(T_LOWER))/(T_UPPER-T_LOWER);
	cv::circle(oUpdateRateFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oUpdateRateFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " t(" << dbg1 << ") = " << m_oUpdateRateFrame.at<float>(dbg1) << "  ,  t(" << dbg2 << ") = " << m_oUpdateRateFrame.at<float>(dbg2) << std::endl;
	cv::waitKey(1);*/
#if BGSLBSP_USE_ADVANCED_MORPH_OPS || BGSPBASLBSP_USE_R2_ACCELERATION
	oFGMask.copyTo(m_oLastFGMask);
#endif //BGSLBSP_USE_ADVANCED_MORPH_OPS || BGSPBASLBSP_USE_R2_ACCELERATION
#if BGSLBSP_USE_ADVANCED_MORPH_OPS
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
#else //!BGSLBSP_USE_ADVANCED_MORPH_OPS
	cv::medianBlur(oFGMask,oFGMask,9);
#endif //!BGSLBSP_USE_ADVANCED_MORPH_OPS
}

void BackgroundSubtractorPBASLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_DbgAssert(m_bInitialized);
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC(m_nImgChannels));
	for(int n=0; n<m_nBGSamples; ++n) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				int img_idx = m_voBGImg[n].step.p[0]*y + m_voBGImg[n].step.p[1]*x;
				int flt32_idx = img_idx*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+flt32_idx);
				uchar* oBGImgPtr = m_voBGImg[n].data+img_idx;
				for(int c=0; c<m_nImgChannels; ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}
