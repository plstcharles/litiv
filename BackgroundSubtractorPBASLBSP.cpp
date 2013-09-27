#include "BackgroundSubtractorPBASLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

#define OVERLOAD_GRAD_PROP (0.50f)

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(bool bDelayedAnalysis)
	:	 BackgroundSubtractorLBSP()
		,m_bDelayAnalysis(bDelayedAnalysis)
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
															,bool bDelayedAnalysis
															,int nInitDescDistThreshold
															,int nInitColorDistThreshold
															,float fInitUpdateRate
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(nLBSPThreshold,nInitDescDistThreshold)
		,m_bDelayAnalysis(bDelayedAnalysis)
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
															,bool bDelayedAnalysis
															,int nInitDescDistThreshold
															,int nInitColorDistThreshold
															,float fInitUpdateRate
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold)
		,m_bDelayAnalysis(bDelayedAnalysis)
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
	m_oDistThresholdVariationFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdVariationFrame = cv::Scalar(1.0f);
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(m_fDefaultUpdateRate);
	m_oMeanMinDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame = cv::Scalar(0.0f);

	m_oMeanNbBlinksFrame.create(m_oImgSize,CV_32FC1); // @@@@@@@@@@
	m_oMeanNbBlinksFrame = cv::Scalar(0.0f); // @@@@@@@@@@
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1); // @@@@@@@@@@
	m_oBlinksFrame = cv::Scalar(0); // @@@@@@@@@@

	m_oTempFGMask.create(m_oImgSize,CV_8UC1);
	m_oTempFGMask = cv::Scalar(0);
	m_oPureFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGMask_last = cv::Scalar(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar(0);
	if(m_bDelayAnalysis) {
		m_oPureFGMask_old.create(m_oImgSize,CV_8UC1);
		m_oPureFGMask_old = cv::Scalar(0);
		m_oFGMask_old.create(m_oImgSize,CV_8UC1);
		m_oFGMask_old = cv::Scalar(0);
		m_oFGMask_old_dilated.create(m_oImgSize,CV_8UC1);
		m_oFGMask_old_dilated = cv::Scalar(0);
	}

	/*cv::Mat oBlurredInitImg;
	cv::GaussianBlur(oInitImg,oBlurredInitImg,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInitImg_GradX, oBlurredInitImg_GradY;
	cv::Scharr(oBlurredInitImg,oBlurredInitImg_GradX,CV_16S,1,0,1,0,cv::BORDER_DEFAULT);
	cv::Scharr(oBlurredInitImg,oBlurredInitImg_GradY,CV_16S,0,1,1,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInitImg_AbsGradX, oBlurredInitImg_AbsGradY;
	cv::convertScaleAbs(oBlurredInitImg_GradX,oBlurredInitImg_AbsGradX);
	cv::convertScaleAbs(oBlurredInitImg_GradY,oBlurredInitImg_AbsGradY);
	cv::Mat oBlurredInitImg_AbsGrad;
	cv::addWeighted(oBlurredInitImg_AbsGradX,0.5,oBlurredInitImg_AbsGradY,0.5,0,oBlurredInitImg_AbsGrad);*/

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
	// create an extractor, this one time, for a batch job
	if(m_bLBSPUsingRelThreshold) {
		LBSP oExtractor(m_fLBSPThreshold);
		oExtractor.compute2(oInitImg,m_voKeyPoints,oInitDesc);
	}
	else {
		LBSP oExtractor(m_nLBSPThreshold);
		oExtractor.compute2(oInitImg,m_voKeyPoints,oInitDesc);
	}
	m_voBGImg.resize(m_nBGSamples);
	//m_voBGGrad.resize(m_nBGSamples);
	m_voBGDesc.resize(m_nBGSamples);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,CV_8UC1);
			m_voBGImg[s] = cv::Scalar_<uchar>(0);
			//m_voBGGrad[s].create(m_oImgSize,CV_8UC1);
			//m_voBGGrad[s] = cv::Scalar_<uchar>(0);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC1);
			m_voBGDesc[s] = cv::Scalar_<ushort>(0);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				m_voBGImg[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
				//m_voBGGrad[s].at<uchar>(y_orig,x_orig) = oBlurredInitImg_AbsGrad.at<uchar>(y_sample,x_sample);
				m_voBGDesc[s].at<ushort>(y_orig,x_orig) = oInitDesc.at<ushort>(y_sample,x_sample);
			}
		}
	}
	else { //m_nImgChannels==3
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,CV_8UC3);
			m_voBGImg[s] = cv::Scalar_<uchar>(0,0,0);
			//m_voBGGrad[s].create(m_oImgSize,CV_8UC3);
			//m_voBGGrad[s] = cv::Scalar_<uchar>(0,0,0);
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
				//uchar* bg_grad_ptr = m_voBGGrad[s].data+idx_orig_img;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDesc[s].data+idx_orig_desc);
				const uchar* init_img_ptr = oInitImg.data+idx_rand_img;
				//const uchar* init_grad_ptr = oBlurredInitImg_AbsGrad.data+idx_rand_img;
				const ushort* init_desc_ptr = (ushort*)(oInitDesc.data+idx_rand_desc);
				for(int n=0;n<3; ++n) {
					bg_img_ptr[n] = init_img_ptr[n];
					//bg_grad_ptr[n] = init_grad_ptr[n];
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
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	static const int nChannelSize = UCHAR_MAX;
	static const int nDescSize = LBSP::DESC_SIZE*8;
	/*cv::Mat oBlurredInputImg;
	cv::GaussianBlur(oInputImg,oBlurredInputImg,cv::Size(3,3),0,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInputImg_GradX, oBlurredInputImg_GradY;
	cv::Scharr(oBlurredInputImg,oBlurredInputImg_GradX,CV_16S,1,0,1,0,cv::BORDER_DEFAULT);
	cv::Scharr(oBlurredInputImg,oBlurredInputImg_GradY,CV_16S,0,1,1,0,cv::BORDER_DEFAULT);
	cv::Mat oBlurredInputImg_AbsGradX, oBlurredInputImg_AbsGradY;
	cv::convertScaleAbs(oBlurredInputImg_GradX,oBlurredInputImg_AbsGradX);
	cv::convertScaleAbs(oBlurredInputImg_GradY,oBlurredInputImg_AbsGradY);
	cv::Mat oBlurredInputImg_AbsGrad;
	cv::addWeighted(oBlurredInputImg_AbsGradX,0.5,oBlurredInputImg_AbsGradY,0.5,0,oBlurredInputImg_AbsGrad);
	cv::Mat oInputDesc, oSampleDescHammDiff;
	if(m_bLBSPUsingRelThreshold) {
		LBSP oExtractor(m_fLBSPThreshold);
		//oExtractor.setReference(m_voBGImg[0]);
		oExtractor.compute2(oInputImg,m_voKeyPoints,oInputDesc);
		oExtractor.calcDescImgDiff(oInputDesc,m_voBGDesc[0],oSampleDescHammDiff);
	}
	else {
		LBSP oExtractor(m_nLBSPThreshold);
		//oExtractor.setReference(m_voBGImg[0]);
		oExtractor.compute2(oInputImg,m_voKeyPoints,oInputDesc);
		oExtractor.calcDescImgDiff(oInputDesc,m_voBGDesc[0],oSampleDescHammDiff);
	}
	cv::imshow("oInputDesc",oInputDesc);
	cv::imshow("oBlurredInputImg_AbsGrad",oBlurredInputImg_AbsGrad);
	cv::Mat oSampleColorAbsDiff,oSampleGradAbsDiff;
	cv::absdiff(oInputImg,m_voBGImg[0],oSampleColorAbsDiff);
	cv::absdiff(oBlurredInputImg_AbsGrad,m_voBGGrad[0],oSampleGradAbsDiff);
	cv::imshow("oSampleColorAbsDiff",oSampleColorAbsDiff);
	cv::imshow("oSampleGradAbsDiff",oSampleGradAbsDiff);
	cv::imshow("oSampleDescHammDiff",oSampleDescHammDiff);
	cv::waitKey();*/
	/*cv::Mat oNewBOOSTEDDistImg;
	cv::addWeighted(oSampleGradAbsDiff,OVERLOAD_GRAD_PROP,oSampleColorAbsDiff,1.0,0,oNewBOOSTEDDistImg);
	cv::imshow("oNewBOOSTEDDistImg",oNewBOOSTEDDistImg);
	cv::Mat oDistImgDiff;
	cv::absdiff(oNewBOOSTEDDistImg,oSampleColorAbsDiff,oDistImgDiff);
	cv::imshow("diff between regular and boosted",oDistImgDiff);*/
	if(m_nImgChannels==1) {
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = oInputImg.cols*y + x;
			const int descimg_idx = uchar_idx*2;
			const int flt32_idx = uchar_idx*4;
			int nMinSumDist=nChannelSize;
			int nMinDescDist=nDescSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			const int nCurrColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			//const int nCurrDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold);
			ushort nCurrInterDesc, nCurrIntraDesc;
			if(m_bLBSPUsingRelThreshold)
				LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,nCurrIntraDesc);
			else
				LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,nCurrIntraDesc);
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				{
					const int nColorDist = absdiff_uchar(oInputImg.data[uchar_idx],m_voBGImg[nSampleIdx].data[uchar_idx]);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nBGIntraDesc = *((ushort*)(m_voBGDesc[nSampleIdx].data+descimg_idx));
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_fLBSPThreshold,nCurrInterDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_nLBSPThreshold,nCurrInterDesc);
					const int nDescDist = (hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc)+hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc))/2;
					/*if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;*/
					const int nSumDist = std::min((int)(OVERLOAD_GRAD_PROP*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT*nDescDist)*(nChannelSize/nDescSize)+nColorDist,nChannelSize);
					if(nSumDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					if(nMinDescDist>nDescDist)
						nMinDescDist = nDescDist;
					if(nMinSumDist>nSumDist)
						nMinSumDist = nSumDist;
					nGoodSamplesCount++;
				}
				failedcheck1ch:
				nSampleIdx++;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinSumDist/nChannelSize+(float)nMinDescDist/nDescSize)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrMeanNbBlinks = ((float*)(m_oMeanNbBlinksFrame.data+flt32_idx));
			*pfCurrMeanNbBlinks = ((*pfCurrMeanNbBlinks)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + (float)m_oBlinksFrame.data[uchar_idx]/nChannelSize)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
			}
			else {
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDesc[s_rand].data+descimg_idx)) = nCurrIntraDesc;
					m_voBGImg[s_rand].data[uchar_idx] = oInputImg.data[uchar_idx];
				}
				if((rand()%nLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					int s_rand = rand()%m_nBGSamples;
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					ushort& nRandIntraDesc = m_voBGDesc[s_rand].at<ushort>(y_rand,x_rand);
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,nRandIntraDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,nRandIntraDesc);
					//m_voBGGrad[s_rand].at<uchar>(y_rand,x_rand) = oBlurredInputImg_AbsGrad.at<uchar>(y_rand,x_rand);
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y_rand,x_rand);
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					m_voBGDesc[s_rand].at<ushort>(y_rand,x_rand) = nCurrIntraDesc;
					//m_voBGGrad[s_rand].at<uchar>(y_rand,x_rand) = oBlurredInputImg_AbsGrad.data[uchar_idx];
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.data[uchar_idx];
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
				*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
			}
#if BGSPBASLBSP_USE_R2_ACCELERATION
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
#else //!BGSPBASLBSP_USE_R2_ACCELERATION
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR;
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR;
#endif //!BGSPBASLBSP_USE_R2_ACCELERATION
			if(m_bDelayAnalysis)
				m_oBlinksFrame.data[uchar_idx] = ((m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx]||m_oPureFGMask_last.data[uchar_idx]!=m_oPureFGMask_old.data[uchar_idx]) && m_oFGMask_old_dilated.data[uchar_idx]==0)?UCHAR_MAX:0;
			else
				m_oBlinksFrame.data[uchar_idx] = (m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx])?UCHAR_MAX:0;
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
			const uchar* anCurrColorInt = oInputImg.data+rgbimg_idx;
			int nMinTotDescDist=nTotDescSize;
			int nMinTotSumDist=nTotChannelSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			const int nCurrTotColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
			//const int nCurrTotDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
#if BGSLBSP_USE_SC_THRS_VALIDATION
			const int nCurrSCColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			//const int nCurrSCDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			if(m_bLBSPUsingRelThreshold)
				LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,anCurrIntraDesc);
			else
				LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,anCurrIntraDesc);
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* anBGIntraDesc = (ushort*)(m_voBGDesc[nSampleIdx].data+descimg_idx);
				const uchar* anBGColorInt = m_voBGImg[nSampleIdx].data+rgbimg_idx;
				int nTotDescDist = 0;
				int nTotSumDist = 0;
				for(int c=0;c<3; ++c) {
					const int nColorDist = absdiff_uchar(anCurrColorInt[c],anBGColorInt[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeSingleRGBRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_fLBSPThreshold,anCurrInterDesc[c]);
					else
						LBSP::computeSingleRGBAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_nLBSPThreshold,anCurrInterDesc[c]);
					const int nDescDist = (hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c])+hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]))/2;
#if BGSLBSP_USE_SC_THRS_VALIDATION
					/*if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;*/
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					const int nSumDist = std::min((int)(OVERLOAD_GRAD_PROP*nDescDist)*(nChannelSize/nDescSize)+nColorDist,nChannelSize);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
				}
				if(/*nTotDescDist>nCurrTotDescDistThreshold || */nTotSumDist>nCurrTotColorDistThreshold)
					goto failedcheck3ch;
				if(nMinTotDescDist>nTotDescDist)
					nMinTotDescDist = nTotDescDist;
				if(nMinTotSumDist>nTotSumDist)
					nMinTotSumDist = nTotSumDist;
				nGoodSamplesCount++;
				failedcheck3ch:
				nSampleIdx++;
			}
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+flt32_idx));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinTotSumDist/nTotChannelSize+(float)nMinTotDescDist/nTotDescSize)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrMeanNbBlinks = ((float*)(m_oMeanNbBlinksFrame.data+flt32_idx));
			*pfCurrMeanNbBlinks = ((*pfCurrMeanNbBlinks)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + (float)m_oBlinksFrame.data[uchar_idx]/nChannelSize)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
			}
			else {
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDesc[s_rand].data+descimg_idx+2*c)) = anCurrIntraDesc[c];
						//*(m_voBGGrad[s_rand].data+rgbimg_idx+c) = *(oBlurredInputImg_AbsGrad.data+rgbimg_idx+c);
						*(m_voBGImg[s_rand].data+rgbimg_idx+c) = *(oInputImg.data+rgbimg_idx+c);
					}
				}
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					ushort* anRandIntraDesc = ((ushort*)(m_voBGDesc[s_rand].data + desc_row_step*y_rand + 6*x_rand));
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,anRandIntraDesc);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,anRandIntraDesc);
					const int img_row_step = m_voBGImg[0].step.p[0];
					for(int c=0; c<3; ++c) {
						//*(m_voBGGrad[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oBlurredInputImg_AbsGrad.data + img_row_step*y_rand + 3*x_rand + c);
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oInputImg.data + img_row_step*y_rand + 3*x_rand + c);
					}
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDesc[s_rand].data + desc_row_step*y_rand + 6*x_rand + 2*c)) = anCurrIntraDesc[c];
						//*(m_voBGGrad[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oBlurredInputImg_AbsGrad.data+rgbimg_idx+c);
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oInputImg.data+rgbimg_idx+c);
					}
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
				*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
			}
#if BGSPBASLBSP_USE_R2_ACCELERATION
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER)
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER)
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
#else //!BGSPBASLBSP_USE_R2_ACCELERATION
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER)
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR;
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER)
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR;
#endif //!BGSPBASLBSP_USE_R2_ACCELERATION
			if(m_bDelayAnalysis)
				m_oBlinksFrame.data[uchar_idx] = ((m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx]||m_oPureFGMask_last.data[uchar_idx]!=m_oPureFGMask_old.data[uchar_idx]) && m_oFGMask_old_dilated.data[uchar_idx]==0)?UCHAR_MAX:0;
			else
				m_oBlinksFrame.data[uchar_idx] = (m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx])?UCHAR_MAX:0;
		}
	}
	//m_fFormerMeanGradDist = std::max(((float)nFrameTotGradDist)/nFrameTotBadSamplesCount,20.0f);
	//std::cout << "#### m_fFormerMeanGradDist = " << m_fFormerMeanGradDist << std::endl;
	/*std::cout << std::endl;
	cv::Point dbg1(70,64), dbg2(126,130), dbg3(218,132);
	cv::Mat oMeanMinDistFrameNormalized = m_oMeanMinDistFrame;
	cv::circle(oMeanMinDistFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oMeanMinDistFrameNormalized,dbg2,5,cv::Scalar(1.0f));cv::circle(oMeanMinDistFrameNormalized,dbg3,5,cv::Scalar(1.0f));
	cv::imshow("m(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " m(" << dbg1 << ") = " << m_oMeanMinDistFrame.at<float>(dbg1) << "  ,  m(" << dbg2 << ") = " << m_oMeanMinDistFrame.at<float>(dbg2) << "  ,  m(" << dbg3 << ") = " << m_oMeanMinDistFrame.at<float>(dbg3) << std::endl;
	cv::Mat oMeanNbBlinksFrameNormalized = m_oMeanNbBlinksFrame;
	cv::circle(oMeanNbBlinksFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oMeanNbBlinksFrameNormalized,dbg2,5,cv::Scalar(1.0f));cv::circle(oMeanNbBlinksFrameNormalized,dbg3,5,cv::Scalar(1.0f));
	cv::imshow("b(x)",oMeanNbBlinksFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " b(" << dbg1 << ") = " << oMeanNbBlinksFrameNormalized.at<float>(dbg1) << "  ,  b(" << dbg2 << ") = " << oMeanNbBlinksFrameNormalized.at<float>(dbg2) << "  ,  b(" << dbg3 << ") = " << oMeanNbBlinksFrameNormalized.at<float>(dbg3) << std::endl;
	cv::Mat oDistThresholdFrameNormalized = (m_oDistThresholdFrame-cv::Scalar(BGSPBASLBSP_R_LOWER))/(BGSPBASLBSP_R_UPPER-BGSPBASLBSP_R_LOWER);
	cv::circle(oDistThresholdFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oDistThresholdFrameNormalized,dbg2,5,cv::Scalar(1.0f));cv::circle(oDistThresholdFrameNormalized,dbg3,5,cv::Scalar(1.0f));
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " r(" << dbg1 << ") = " << m_oDistThresholdFrame.at<float>(dbg1) << "  ,  r(" << dbg2 << ") = " << m_oDistThresholdFrame.at<float>(dbg2) << "  ,  r(" << dbg3 << ") = " << m_oDistThresholdFrame.at<float>(dbg3) << std::endl;
#if BGSPBASLBSP_USE_R2_ACCELERATION
	cv::Mat oDistThresholdVariationFrameNormalized = (m_oDistThresholdVariationFrame-cv::Scalar(BGSPBASLBSP_R2_LOWER))/(BGSPBASLBSP_R2_UPPER-BGSPBASLBSP_R2_LOWER);
	cv::circle(oDistThresholdVariationFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oDistThresholdVariationFrameNormalized,dbg2,5,cv::Scalar(1.0f));cv::circle(oDistThresholdVariationFrameNormalized,dbg3,5,cv::Scalar(1.0f));
	cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "r2(" << dbg1 << ") = " << m_oDistThresholdVariationFrame.at<float>(dbg1) << "  , r2(" << dbg2 << ") = " << m_oDistThresholdVariationFrame.at<float>(dbg2) << "  , r2(" << dbg3 << ") = " << m_oDistThresholdVariationFrame.at<float>(dbg3) << std::endl;
#endif //BGSPBASLBSP_USE_R2_ACCELERATION
	cv::Mat oUpdateRateFrameNormalized = (m_oUpdateRateFrame-cv::Scalar(BGSPBASLBSP_T_LOWER))/(BGSPBASLBSP_T_UPPER-BGSPBASLBSP_T_LOWER);
	cv::circle(oUpdateRateFrameNormalized,dbg1,5,cv::Scalar(1.0f));cv::circle(oUpdateRateFrameNormalized,dbg2,5,cv::Scalar(1.0f));
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " t(" << dbg1 << ") = " << m_oUpdateRateFrame.at<float>(dbg1) << "  ,  t(" << dbg2 << ") = " << m_oUpdateRateFrame.at<float>(dbg2) << std::endl;*/
	/*cv::Mat oPureFGMaskBlinks;
	cv::bitwise_xor(oPureFGMask,m_oPureFGMask_last,oPureFGMaskBlinks);
	cv::imshow("oPureFGMaskBlinks",oPureFGMaskBlinks);
	cv::Mat oFGMask_last_noblobs;
	cv::bitwise_not(m_oFGMask_last,oFGMask_last_noblobs);
	cv::imshow("oFGMask_last_noblobs",oFGMask_last_noblobs);
	cv::Mat oFGMaskBlinks_confirmed;
	cv::bitwise_and(oPureFGMaskBlinks,oFGMask_last_noblobs,oFGMaskBlinks_confirmed);
	cv::imshow("oFGMaskBlinks_confirmed (1)",oFGMaskBlinks_confirmed);
	if(m_bDelaying) {
		cv::Mat oFGMask_old_noblobs;
		cv::bitwise_not(m_oFGMask_old,oFGMask_old_noblobs);
		cv::imshow("oFGMask_old_noblobs",oFGMask_old_noblobs);
		cv::bitwise_and(oFGMaskBlinks_confirmed,oFGMask_old_noblobs,oFGMaskBlinks_confirmed);
		cv::imshow("oFGMaskBlinks_confirmed (2)",oFGMaskBlinks_confirmed);
	}
	cv::imshow("m_oBlinksFrame",m_oBlinksFrame);
	cv::imshow("m_oMeanNbBlinksFrame",m_oMeanNbBlinksFrame);*/
#if BGSLBSP_USE_ADVANCED_MORPH_OPS || BGSPBASLBSP_USE_R2_ACCELERATION
	m_oPureFGMask_last.copyTo(m_oPureFGMask_old); // @@@@@@@@
	oCurrFGMask.copyTo(m_oPureFGMask_last);
#endif //BGSLBSP_USE_ADVANCED_MORPH_OPS || BGSPBASLBSP_USE_R2_ACCELERATION
#if BGSLBSP_USE_ADVANCED_MORPH_OPS
	cv::medianBlur(oCurrFGMask,m_oTempFGMask,3);
	cv::morphologyEx(m_oTempFGMask,m_oTempFGMask,cv::MORPH_CLOSE,cv::Mat());
	cv::floodFill(m_oTempFGMask,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oTempFGMask,m_oTempFGMask);
	cv::bitwise_or(m_oTempFGMask,m_oPureFGMask_last,oCurrFGMask);
#endif //BGSLBSP_USE_ADVANCED_MORPH_OPS
	if(m_bDelayAnalysis) {
		m_oFGMask_last.copyTo(m_oFGMask_old);
		m_oFGMask_last_dilated.copyTo(m_oFGMask_old_dilated);
	}
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,9);
	if(m_bDelayAnalysis) {
		cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat());
		cv::bitwise_not(m_oFGMask_last_dilated,m_oTempFGMask);
		cv::bitwise_and(m_oTempFGMask,m_oBlinksFrame,m_oBlinksFrame);
		m_oFGMask_old.copyTo(oCurrFGMask);
	}
	else
		m_oFGMask_last.copyTo(oCurrFGMask);
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
