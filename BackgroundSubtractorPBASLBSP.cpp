#include "BackgroundSubtractorPBASLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

// local define used for debug purposes only
#define DISPLAY_PBASLBSP_DEBUG_FRAMES 0
// local define for the gradient proportion value used in color+grad distance calculations
#define OVERLOAD_GRAD_PROP ((1.0f-std::pow(((*pfCurrDistThresholdFactor)-BGSPBASLBSP_R_LOWER)/(BGSPBASLBSP_R_UPPER-BGSPBASLBSP_R_LOWER),2))*0.5f)
// local defines used when not using cutoff configuration
#if !BGSPBASLBSP_USE_LBSP_TYPE_CUTOFF
#define bUseRelLBSP true
#endif //!BGSPBASLBSP_USE_LBSP_TYPE_CUTOFF

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(	 bool bDelayedAnalysis
															,float fLBSPThreshold
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
	CV_Assert(m_nColorDistThreshold>0);
	CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
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
	m_oRelLBSPThresFrame.create(m_oImgSize,CV_8UC1); // @@@@@@@@@@
	m_oRelLBSPThresFrame = cv::Scalar(UCHAR_MAX/2); // @@@@@@@@@@

	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1); // @@@@@@@@@@
	m_oMeanLastDistFrame = cv::Scalar(0.0f); // @@@@@@@@@@

	m_oMeanSegmResFrame.create(m_oImgSize,CV_32FC1); // @@@@@@@@@@
	m_oMeanSegmResFrame = cv::Scalar(0.0f); // @@@@@@@@@@

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
	oInitImg.copyTo(m_oLastColorFrame);
	// create an extractor, this one time, for a batch job
	LBSP oExtractor(m_nImgChannels==3?m_fLBSPThreshold:(m_fLBSPThreshold*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT));
	oExtractor.compute2(m_oLastColorFrame,m_voKeyPoints,m_oLastDescFrame);

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
				m_voBGImg[s].at<uchar>(y_orig,x_orig) = m_oLastColorFrame.at<uchar>(y_sample,x_sample);
				//m_voBGGrad[s].at<uchar>(y_orig,x_orig) = oBlurredInitImg_AbsGrad.at<uchar>(y_sample,x_sample);
				m_voBGDesc[s].at<ushort>(y_orig,x_orig) = m_oLastDescFrame.at<ushort>(y_sample,x_sample);
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
				const int idx_orig_img = m_oLastColorFrame.step.p[0]*y_orig + m_oLastColorFrame.step.p[1]*x_orig;
				const int idx_orig_desc = m_oLastDescFrame.step.p[0]*y_orig + m_oLastDescFrame.step.p[1]*x_orig;
				const int idx_rand_img = m_oLastColorFrame.step.p[0]*y_sample + m_oLastColorFrame.step.p[1]*x_sample;
				const int idx_rand_desc = m_oLastDescFrame.step.p[0]*y_sample + m_oLastDescFrame.step.p[1]*x_sample;
				uchar* bg_img_ptr = m_voBGImg[s].data+idx_orig_img;
				//uchar* bg_grad_ptr = m_voBGGrad[s].data+idx_orig_img;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDesc[s].data+idx_orig_desc);
				const uchar* const init_img_ptr = m_oLastColorFrame.data+idx_rand_img;
				//const uchar* const init_grad_ptr = oBlurredInitImg_AbsGrad.data+idx_rand_img;
				const ushort* const init_desc_ptr = (ushort*)(m_oLastDescFrame.data+idx_rand_desc);
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
			const int uchar_idx = m_oImgSize.width*y + x;
			const int ushrt_idx = uchar_idx*2;
			const int flt32_idx = uchar_idx*4;
			const uchar nCurrColorInt = oInputImg.data[uchar_idx];
			int nMinSumDist=nChannelSize;
			int nMinDescDist=nDescSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			const int nCurrColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
#if BGSPBASLBSP_USE_DESC_DIST_CHECKS
			const int nCurrDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold);
#endif //BGSPBASLBSP_USE_DESC_DIST_CHECKS
#if BGSPBASLBSP_USE_LBSP_TYPE_CUTOFF
			const bool bUseRelLBSP = (m_oRelLBSPThresFrame.data[uchar_idx]>0)?((*pfCurrDistThresholdFactor)<BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL):((*pfCurrDistThresholdFactor)<BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL-BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL_BUFFER);
			m_oRelLBSPThresFrame.data[uchar_idx] = bUseRelLBSP?UCHAR_MAX:0;
#endif //BGSPBASLBSP_USE_LBSP_TYPE_CUTOFF
			ushort nCurrInterDesc, nCurrIntraDesc;
			const uchar nCurrLBSPThreshold = (uchar)((bUseRelLBSP?(m_fLBSPThreshold*nCurrColorInt):(BGSPBASLBSP_DEF_ABS_LBSP_THRES))*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColorInt,x,y,nCurrLBSPThreshold,nCurrIntraDesc);
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar nBGColorInt = m_voBGImg[nSampleIdx].data[uchar_idx];
				{
					const int nColorDist = absdiff_uchar(nCurrColorInt,nBGColorInt);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nBGIntraDesc = *((ushort*)(m_voBGDesc[nSampleIdx].data+ushrt_idx));
					const uchar nBGLBSPThreshold = (uchar)((bUseRelLBSP?(m_fLBSPThreshold*nBGColorInt):(BGSPBASLBSP_DEF_ABS_LBSP_THRES))*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColorInt,x,y,nBGLBSPThreshold,nCurrInterDesc);
					const int nDescDist = (hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc)+hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc))/2;
#if BGSPBASLBSP_USE_DESC_DIST_CHECKS
					if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;
#endif //BGSPBASLBSP_USE_DESC_DIST_CHECKS
					const int nSumDist = std::min((int)(OVERLOAD_GRAD_PROP*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT*nDescDist)*(nChannelSize/nDescSize)+nColorDist,nChannelSize);
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
			//float* pfCurrMeanNbBlinks = ((float*)(m_oMeanNbBlinksFrame.data+flt32_idx));
			//*pfCurrMeanNbBlinks = ((*pfCurrMeanNbBlinks)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + (float)m_oBlinksFrame.data[uchar_idx]/nChannelSize)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+ushrt_idx));
			uchar& nLastColorInt = m_oLastColorFrame.data[uchar_idx];
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+flt32_idx));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(absdiff_uchar(nLastColorInt,nCurrColorInt))/nChannelSize+(float)(hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc))/nDescSize)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				*pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
			}
			else {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1))/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDesc[s_rand].data+ushrt_idx)) = nCurrIntraDesc;
					m_voBGImg[s_rand].data[uchar_idx] = nCurrColorInt;
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				int n_rand = rand();
				const int uchar_randidx = m_oImgSize.width*y_rand + x_rand;
				const int flt32_randidx = uchar_randidx*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+flt32_randidx));
				const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+flt32_randidx));
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSPBASLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSPBASLBSP_GHOST_DETECTION_D_SPREAD_MAX*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT && (n_rand%4)==0)) {
					const int ushrt_randidx = uchar_randidx*2;
					int s_rand = rand()%m_nBGSamples;
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					ushort& nRandIntraDesc = *((ushort*)(m_voBGDesc[s_rand].data+ushrt_randidx));
					const uchar nRandColorInt = oInputImg.data[uchar_randidx];
					const uchar nRandLBSPThreshold = (uchar)((bUseRelLBSP?(m_fLBSPThreshold*nRandColorInt):(BGSPBASLBSP_DEF_ABS_LBSP_THRES))*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
					LBSP::computeGrayscaleDescriptor(oInputImg,nRandColorInt,x_rand,y_rand,nRandLBSPThreshold,nRandIntraDesc);
					//m_voBGGrad[s_rand].data[uchar_randidx] = oBlurredInputImg_AbsGrad.data[uchar_randidx];
					m_voBGImg[s_rand].data[uchar_randidx] = nRandColorInt;
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					*((ushort*)(m_voBGDesc[s_rand].data+ushrt_randidx)) = nCurrIntraDesc;
					//m_voBGGrad[s_rand].data[uchar_randidx] = oBlurredInputImg_AbsGrad.data[uchar_idx];
					m_voBGImg[s_rand].data[uchar_randidx] = nCurrColorInt;
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
			}
#if BGSPBASLBSP_USE_R2_ACCELERATION
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER) {
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
					if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_UPPER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_UPPER;
				}
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER) {
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_LOWER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_LOWER;
				}
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
					if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER) {
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_LOWER;
			}
#else //!BGSPBASLBSP_USE_R2_ACCELERATION
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR;
					if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER) {
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR;
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_LOWER;
			}
#endif //!BGSPBASLBSP_USE_R2_ACCELERATION
			if(m_bDelayAnalysis)
				m_oBlinksFrame.data[uchar_idx] = ((m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx]||m_oPureFGMask_last.data[uchar_idx]!=m_oPureFGMask_old.data[uchar_idx]) && m_oFGMask_old_dilated.data[uchar_idx]==0)?UCHAR_MAX:0;
			else
				m_oBlinksFrame.data[uchar_idx] = (m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx])?UCHAR_MAX:0;
			nLastIntraDesc = nCurrIntraDesc;
			nLastColorInt = nCurrColorInt;
		}
	}
	else { //m_nImgChannels==3
		static const int nTotChannelSize = nChannelSize*3;
		static const int nTotDescSize = nDescSize*3;
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = m_oImgSize.width*y + x;
			const int flt32_idx = uchar_idx*4;
			const int uchar_rgb_idx = uchar_idx*3;
			const int ushrt_rgb_idx = uchar_rgb_idx*2;
			const uchar* const anCurrColorInt = oInputImg.data+uchar_rgb_idx;
			int nMinTotDescDist=nTotDescSize;
			int nMinTotSumDist=nTotChannelSize;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+flt32_idx);
			const int nCurrTotColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
#if BGSPBASLBSP_USE_DESC_DIST_CHECKS
			const int nCurrTotDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
#endif //BGSPBASLBSP_USE_DESC_DIST_CHECKS
#if BGSLBSP_USE_SC_THRS_VALIDATION
			const int nCurrSCColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#if BGSPBASLBSP_USE_DESC_DIST_CHECKS
			const int nCurrSCDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSPBASLBSP_USE_DESC_DIST_CHECKS
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#if BGSPBASLBSP_USE_LBSP_TYPE_CUTOFF
			const bool bUseRelLBSP = (m_oRelLBSPThresFrame.data[uchar_idx]>0)?((*pfCurrDistThresholdFactor)<BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL):((*pfCurrDistThresholdFactor)<BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL-BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL_BUFFER);
			m_oRelLBSPThresFrame.data[uchar_idx] = bUseRelLBSP?UCHAR_MAX:0;
#endif //BGSPBASLBSP_USE_LBSP_TYPE_CUTOFF
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			if(bUseRelLBSP) {
				const uchar anCurrIntraLBSPThresholds[3] = {(uchar)(anCurrColorInt[0]*m_fLBSPThreshold),(uchar)(anCurrColorInt[1]*m_fLBSPThreshold),(uchar)(anCurrColorInt[2]*m_fLBSPThreshold)};
				LBSP::computeRGBDescriptor(oInputImg,anCurrColorInt,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			}
			else
				LBSP::computeRGBDescriptor(oInputImg,anCurrColorInt,x,y,BGSPBASLBSP_DEF_ABS_LBSP_THRES,anCurrIntraDesc);
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGIntraDesc = (ushort*)(m_voBGDesc[nSampleIdx].data+ushrt_rgb_idx);
				const uchar* const anBGColorInt = m_voBGImg[nSampleIdx].data+uchar_rgb_idx;
				int nTotDescDist = 0;
				int nTotSumDist = 0;
				for(int c=0;c<3; ++c) {
					const int nColorDist = absdiff_uchar(anCurrColorInt[c],anBGColorInt[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					const uchar nBGLBSPThreshold = bUseRelLBSP?((uchar)(m_fLBSPThreshold*anBGColorInt[c])):BGSPBASLBSP_DEF_ABS_LBSP_THRES;
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColorInt[c],x,y,c,nBGLBSPThreshold,anCurrInterDesc[c]);
					const int nDescDist = (hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c])+hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]))/2;
#if BGSLBSP_USE_SC_THRS_VALIDATION && BGSPBASLBSP_USE_DESC_DIST_CHECKS
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION && BGSPBASLBSP_USE_DESC_DIST_CHECKS
					const int nSumDist = std::min((int)(OVERLOAD_GRAD_PROP*nDescDist)*(nChannelSize/nDescSize)+nColorDist,nChannelSize);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
				}
#if BGSPBASLBSP_USE_DESC_DIST_CHECKS
				if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
#else //!BGSPBASLBSP_USE_DESC_DIST_CHECKS
				if(nTotSumDist>nCurrTotColorDistThreshold)
#endif //!BGSPBASLBSP_USE_DESC_DIST_CHECKS
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
			//float* pfCurrMeanNbBlinks = ((float*)(m_oMeanNbBlinksFrame.data+flt32_idx));
			//*pfCurrMeanNbBlinks = ((*pfCurrMeanNbBlinks)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + (float)m_oBlinksFrame.data[uchar_idx]/nChannelSize)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+ushrt_rgb_idx));
			uchar* anLastColorInt = m_oLastColorFrame.data+uchar_rgb_idx;
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+flt32_idx));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(L1dist_uchar(anLastColorInt,anCurrColorInt))/nTotChannelSize+(float)(hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc))/nTotDescSize)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				*pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
			}
			else {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1))/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDesc[s_rand].data+ushrt_rgb_idx+2*c)) = anCurrIntraDesc[c];
						//*(m_voBGGrad[s_rand].data+rgbimg_idx+c) = *(oBlurredInputImg_AbsGrad.data+rgbimg_idx+c);
						*(m_voBGImg[s_rand].data+uchar_rgb_idx+c) = anCurrColorInt[c];
					}
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				int n_rand = rand();
				const int uchar_randidx = m_oImgSize.width*y_rand + x_rand;
				const int flt32_randidx = uchar_randidx*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+flt32_randidx));
				const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+flt32_randidx));
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSPBASLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSPBASLBSP_GHOST_DETECTION_D_SPREAD_MAX && (n_rand%4)==0)) {
					const int ushrt_rgb_randidx = uchar_randidx*6;
					const int uchar_rgb_randidx = uchar_randidx*3;
					int s_rand = rand()%m_nBGSamples;
#if BGSPBASLBSP_USE_SELF_DIFFUSION
					ushort* anRandIntraDesc = ((ushort*)(m_voBGDesc[s_rand].data + ushrt_rgb_randidx));
					const uchar* const anRandIntraLBSPRef = oInputImg.data+uchar_rgb_randidx;
					if(bUseRelLBSP) {
						const uchar anRandIntraLBSPThresholds[3] = {(uchar)(anRandIntraLBSPRef[0]*m_fLBSPThreshold),(uchar)(anRandIntraLBSPRef[1]*m_fLBSPThreshold),(uchar)(anRandIntraLBSPRef[2]*m_fLBSPThreshold)};
						LBSP::computeRGBDescriptor(oInputImg,anRandIntraLBSPRef,x_rand,y_rand,anRandIntraLBSPThresholds,anRandIntraDesc);
					}
					else
						LBSP::computeRGBDescriptor(oInputImg,anRandIntraLBSPRef,x_rand,y_rand,BGSPBASLBSP_DEF_ABS_LBSP_THRES,anRandIntraDesc);
					for(int c=0; c<3; ++c) {
						//*(m_voBGGrad[s_rand].data+uchar_rgb_randidx+c) = *(oBlurredInputImg_AbsGrad.data+uchar_rgb_randidx+c);
						*(m_voBGImg[s_rand].data+uchar_rgb_randidx+c) = anRandIntraLBSPRef[c];
					}
#else //!BGSPBASLBSP_USE_SELF_DIFFUSION
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDesc[s_rand].data+ushrt_rgb_randidx+2*c)) = anCurrIntraDesc[c];
						//*(m_voBGGrad[s_rand].data+uchar_rgb_randidx+c) = *(oBlurredInputImg_AbsGrad.data+rgbimg_idx+c);
						*(m_voBGImg[s_rand].data+uchar_rgb_randidx+c) = anCurrColorInt[c];
					}
#endif //!BGSPBASLBSP_USE_SELF_DIFFUSION
				}
			}
#if BGSPBASLBSP_USE_R2_ACCELERATION
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER) {
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
					if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_UPPER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_UPPER;
				}
			}
			else {
				if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER) {
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_LOWER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_LOWER;
				}
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR*(*pfCurrDistThresholdVariationFactor);
					if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER) {
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR/(*pfCurrDistThresholdVariationFactor);
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_LOWER;
			}
#else //!BGSPBASLBSP_USE_R2_ACCELERATION
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE+BGSPBASLBSP_R_OFFST) {
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_UPPER) {
					(*pfCurrDistThresholdFactor) += BGSPBASLBSP_R_INCR;
					if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_UPPER)
						(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_UPPER;
				}
			}
			else if((*pfCurrDistThresholdFactor)>BGSPBASLBSP_R_LOWER) {
				(*pfCurrDistThresholdFactor) -= BGSPBASLBSP_R_DECR;
				if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER)
					(*pfCurrDistThresholdFactor) = BGSPBASLBSP_R_LOWER;
			}
#endif //!BGSPBASLBSP_USE_R2_ACCELERATION
			if(m_bDelayAnalysis)
				m_oBlinksFrame.data[uchar_idx] = ((m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx]||m_oPureFGMask_last.data[uchar_idx]!=m_oPureFGMask_old.data[uchar_idx]) && m_oFGMask_old_dilated.data[uchar_idx]==0)?UCHAR_MAX:0;
			else
				m_oBlinksFrame.data[uchar_idx] = (m_oPureFGMask_last.data[uchar_idx]!=oCurrFGMask.data[uchar_idx])?UCHAR_MAX:0;
			for(int c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColorInt[c] = anCurrColorInt[c];
			}
		}
	}
	//m_fFormerMeanGradDist = std::max(((float)nFrameTotGradDist)/nFrameTotBadSamplesCount,20.0f);
	//std::cout << "#### m_fFormerMeanGradDist = " << m_fFormerMeanGradDist << std::endl;
#if DISPLAY_PBASLBSP_DEBUG_FRAMES
	std::cout << std::endl;
	cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
	cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame.copyTo(oMeanMinDistFrameNormalized);
	cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::imshow("m(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " m(" << dbgpt << ") = " << m_oMeanMinDistFrame.at<float>(dbgpt) << std::endl;
	//cv::Mat oMeanNbBlinksFrameNormalized; m_oMeanNbBlinksFrame.copyTo(oMeanNbBlinksFrameNormalized);
	//cv::circle(oMeanNbBlinksFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	//cv::imshow("b(x)",oMeanNbBlinksFrameNormalized);
	//std::cout << std::fixed << std::setprecision(5) << " b(" << dbgpt << ") = " << m_oMeanNbBlinksFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
	cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::imshow("d(x)",oMeanLastDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanSegmResFrameNormalized; m_oMeanSegmResFrame.copyTo(oMeanSegmResFrameNormalized);
	cv::circle(oMeanSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::imshow("s(x)",oMeanSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " s(" << dbgpt << ") = " << m_oMeanSegmResFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,1.0f/BGSPBASLBSP_R_UPPER,-BGSPBASLBSP_R_LOWER/BGSPBASLBSP_R_UPPER);
	cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
#if BGSPBASLBSP_USE_R2_ACCELERATION
	cv::Mat oDistThresholdVariationFrameNormalized; m_oDistThresholdVariationFrame.convertTo(oDistThresholdVariationFrameNormalized,CV_32FC1,1.0f/BGSPBASLBSP_R2_UPPER,-BGSPBASLBSP_R2_LOWER/BGSPBASLBSP_R2_UPPER);
	cv::circle(oDistThresholdVariationFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "r2(" << dbgpt << ") = " << m_oDistThresholdVariationFrame.at<float>(dbgpt) << std::endl;
#endif //BGSPBASLBSP_USE_R2_ACCELERATION
	cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/BGSPBASLBSP_T_UPPER,-BGSPBASLBSP_T_LOWER/BGSPBASLBSP_T_UPPER);
	cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
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
#endif //DISPLAY_PBASLBSP_DEBUG_FRAMES
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
