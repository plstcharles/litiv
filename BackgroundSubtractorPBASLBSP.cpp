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

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(	 float fLBSPThreshold
															,int nInitDescDistThreshold
															,int nInitColorDistThreshold
															,int nBGSamples
															,int nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples) {
	CV_Assert(m_nBGSamples>0);
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples && m_nRequiredBGSamples>0);
	CV_Assert(m_nColorDistThreshold>0);
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
	m_oUpdateRateFrame = cv::Scalar(BGSPBASLBSP_T_LOWER);
	m_oMeanMinDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame = cv::Scalar(0.0f);
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar(0);
	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.0f);
	m_oMeanSegmResFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanSegmResFrame = cv::Scalar(0.0f);
	m_oTempFGMask.create(m_oImgSize,CV_8UC1);
	m_oTempFGMask = cv::Scalar(0);
	m_oPureFGBlinkMask_curr.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_curr = cv::Scalar(0);
	m_oPureFGBlinkMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_last = cv::Scalar(0);
	m_oPureFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGMask_last = cv::Scalar(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar(0);
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
	m_voBGDesc.resize(m_nBGSamples);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,CV_8UC1);
			m_voBGImg[s] = cv::Scalar_<uchar>(0);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC1);
			m_voBGDesc[s] = cv::Scalar_<ushort>(0);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				m_voBGImg[s].at<uchar>(y_orig,x_orig) = m_oLastColorFrame.at<uchar>(y_sample,x_sample);
				m_voBGDesc[s].at<ushort>(y_orig,x_orig) = m_oLastDescFrame.at<ushort>(y_sample,x_sample);
			}
		}
		for(int t=0; t<=UCHAR_MAX; ++t) {
			int nCurrLBSPThreshold = (int)(t*m_fLBSPThreshold*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			m_nLBSPThreshold_8bitLUT[t]=nCurrLBSPThreshold>UCHAR_MAX?UCHAR_MAX:(uchar)nCurrLBSPThreshold;
		}
	}
	else { //m_nImgChannels==3
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,CV_8UC3);
			m_voBGImg[s] = cv::Scalar_<uchar>(0,0,0);
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
				ushort* bg_desc_ptr = (ushort*)(m_voBGDesc[s].data+idx_orig_desc);
				const uchar* const init_img_ptr = m_oLastColorFrame.data+idx_rand_img;
				const ushort* const init_desc_ptr = (ushort*)(m_oLastDescFrame.data+idx_rand_desc);
				for(int n=0;n<3; ++n) {
					bg_img_ptr[n] = init_img_ptr[n];
					bg_desc_ptr[n] = init_desc_ptr[n];
				}
			}
		}
		for(int t=0; t<=UCHAR_MAX; ++t) {
			int nCurrLBSPThreshold = (int)(t*m_fLBSPThreshold);
			m_nLBSPThreshold_8bitLUT[t]=nCurrLBSPThreshold>UCHAR_MAX?UCHAR_MAX:(uchar)nCurrLBSPThreshold;
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
			const int nCurrDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold); // not adjusted like ^^, the internal LBSP thresholds are instead
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColorInt,x,y,m_nLBSPThreshold_8bitLUT[nCurrColorInt],nCurrIntraDesc);
			int nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar nBGColorInt = m_voBGImg[nSampleIdx].data[uchar_idx];
				{
					const int nColorDist = absdiff_uchar(nCurrColorInt,nBGColorInt);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nBGIntraDesc = *((ushort*)(m_voBGDesc[nSampleIdx].data+ushrt_idx));
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColorInt,x,y,m_nLBSPThreshold_8bitLUT[nBGColorInt],nCurrInterDesc);
					const int nDescDist = (hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc)+hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc))/2;
					if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;
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
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+ushrt_idx));
			uchar& nLastColorInt = m_oLastColorFrame.data[uchar_idx];
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+flt32_idx));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(absdiff_uchar(nLastColorInt,nCurrColorInt))/nChannelSize+(float)(hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc))/nDescSize)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				/**pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;*/
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_UPPER) {
					*pfCurrLearningRate += BGSPBASLBSP_T_INCR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
						*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
				}
			}
			else {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1))/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				/**pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;*/
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_LOWER) {
					*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
						*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
				}
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
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSPBASLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSPBASLBSP_GHOST_DETECTION_D_MAX*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT && (n_rand%4)==0)) {
					const int ushrt_randidx = uchar_randidx*2;
					int s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDesc[s_rand].data+ushrt_randidx)) = nCurrIntraDesc;
					m_voBGImg[s_rand].data[uchar_randidx] = nCurrColorInt;
				}
			}
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				/*if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER) {
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
					if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_UPPER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_UPPER;
				}*/
				(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			//else if(/*(*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && */(*pfCurrMeanSegmRes)>BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN) {
			else if(((*pfCurrMeanSegmRes)>BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN)
				|| ((*pfCurrMeanSegmRes)>BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN2)) {
				//(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR/2;
				(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			else {
				/*if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER) {
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_LOWER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_LOWER;
				}*/
				if((*pfCurrDistThresholdVariationFactor)>0) {
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<0)
						(*pfCurrDistThresholdVariationFactor) = 0;
				}
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE) {
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
			const int nCurrTotDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
#if BGSLBSP_USE_SC_THRS_VALIDATION
			const int nCurrSCColorDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const int nCurrSCDescDistThreshold = (int)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const uchar anCurrIntraLBSPThresholds[3] = {m_nLBSPThreshold_8bitLUT[anCurrColorInt[0]],m_nLBSPThreshold_8bitLUT[anCurrColorInt[1]],m_nLBSPThreshold_8bitLUT[anCurrColorInt[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColorInt,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
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
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColorInt[c],x,y,c,m_nLBSPThreshold_8bitLUT[anBGColorInt[c]],anCurrInterDesc[c]);
					const int nDescDist = (hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c])+hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]))/2;
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					const int nSumDist = std::min((int)(OVERLOAD_GRAD_PROP*nDescDist)*(nChannelSize/nDescSize)+nColorDist,nChannelSize);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					nTotDescDist += nDescDist;
					nTotSumDist += nSumDist;
				}
				if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
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
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+ushrt_rgb_idx));
			uchar* anLastColorInt = m_oLastColorFrame.data+uchar_rgb_idx;
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+flt32_idx));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(L1dist_uchar(anLastColorInt,anCurrColorInt))/nTotChannelSize+(float)(hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc))/nTotDescSize)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+flt32_idx));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+flt32_idx));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[uchar_idx] = UCHAR_MAX;
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + 1.0f)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				/**pfCurrLearningRate += BGSPBASLBSP_T_INCR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
					*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;*/
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_UPPER) {
					*pfCurrLearningRate += BGSPBASLBSP_T_INCR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)>BGSPBASLBSP_T_UPPER)
						*pfCurrLearningRate = BGSPBASLBSP_T_UPPER;
				}
			}
			else {
				*pfCurrMeanSegmRes = ((*pfCurrMeanSegmRes)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1))/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
				/**pfCurrLearningRate -= BGSPBASLBSP_T_DECR/((*pfCurrMeanMinDist)*BGSPBASLBSP_T_SCALE+BGSPBASLBSP_T_OFFST);
				if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
					*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;*/
				if((*pfCurrLearningRate)>BGSPBASLBSP_T_LOWER) {
					*pfCurrLearningRate -= BGSPBASLBSP_T_DECR/(*pfCurrMeanMinDist);
					if((*pfCurrLearningRate)<BGSPBASLBSP_T_LOWER)
						*pfCurrLearningRate = BGSPBASLBSP_T_LOWER;
				}
				const int nLearningRate = learningRateOverride>0?(int)ceil(learningRateOverride):(int)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDesc[s_rand].data+ushrt_rgb_idx+2*c)) = anCurrIntraDesc[c];
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
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSPBASLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSPBASLBSP_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) {
					const int ushrt_rgb_randidx = uchar_randidx*6;
					const int uchar_rgb_randidx = uchar_randidx*3;
					int s_rand = rand()%m_nBGSamples;
					for(int c=0; c<3; ++c) {
						*((ushort*)(m_voBGDesc[s_rand].data+ushrt_rgb_randidx+2*c)) = anCurrIntraDesc[c];
						*(m_voBGImg[s_rand].data+uchar_rgb_randidx+c) = anCurrColorInt[c];
					}
				}
			}
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+flt32_idx);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[uchar_idx]>0) {
				/*if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_UPPER) {
					(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
					if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_UPPER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_UPPER;
				}*/
				(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			//else if(/*(*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && */(*pfCurrMeanSegmRes)>BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN) {
			else if(((*pfCurrMeanSegmRes)>BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist)>BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN)
				|| ((*pfCurrMeanSegmRes)>BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist)>BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN2)) {
				//(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR/2;
				(*pfCurrDistThresholdVariationFactor) += BGSPBASLBSP_R2_INCR;
			}
			else {
				/*if((*pfCurrDistThresholdVariationFactor)>BGSPBASLBSP_R2_LOWER) {
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<BGSPBASLBSP_R2_LOWER)
						(*pfCurrDistThresholdVariationFactor) = BGSPBASLBSP_R2_LOWER;
				}*/
				if((*pfCurrDistThresholdVariationFactor)>0) {
					(*pfCurrDistThresholdVariationFactor) -= BGSPBASLBSP_R2_DECR;
					if((*pfCurrDistThresholdVariationFactor)<0)
						(*pfCurrDistThresholdVariationFactor) = 0;
				}
			}
			if((*pfCurrDistThresholdFactor)<BGSPBASLBSP_R_LOWER+(*pfCurrMeanMinDist)*BGSPBASLBSP_R_SCALE) {
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
			for(int c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColorInt[c] = anCurrColorInt[c];
			}
		}
	}
#if DISPLAY_PBASLBSP_DEBUG_FRAMES
	std::cout << std::endl;
	cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
	cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame.copyTo(oMeanMinDistFrameNormalized);
	cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,cv::Size(320,240));
	cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
	cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,cv::Size(320,240));
	cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanSegmResFrameNormalized; m_oMeanSegmResFrame.copyTo(oMeanSegmResFrameNormalized);
	cv::circle(oMeanSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanSegmResFrameNormalized,oMeanSegmResFrameNormalized,cv::Size(320,240));
	cv::imshow("s(x)",oMeanSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " s(" << dbgpt << ") = " << m_oMeanSegmResFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,1.0f/BGSPBASLBSP_R_UPPER,-BGSPBASLBSP_R_LOWER/BGSPBASLBSP_R_UPPER);
	cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,cv::Size(320,240));
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdVariationFrameNormalized; cv::normalize(m_oDistThresholdVariationFrame,oDistThresholdVariationFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::circle(oDistThresholdVariationFrameNormalized,dbgpt,5,cv::Scalar(255));
	cv::resize(oDistThresholdVariationFrameNormalized,oDistThresholdVariationFrameNormalized,cv::Size(320,240));
	cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "r2(" << dbgpt << ") = " << m_oDistThresholdVariationFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/BGSPBASLBSP_T_UPPER,-BGSPBASLBSP_T_LOWER/BGSPBASLBSP_T_UPPER);
	cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,cv::Size(320,240));
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
#endif //DISPLAY_PBASLBSP_DEBUG_FRAMES
	cv::bitwise_xor(oCurrFGMask,m_oPureFGMask_last,m_oPureFGBlinkMask_curr);
	cv::bitwise_or(m_oPureFGBlinkMask_curr,m_oPureFGBlinkMask_last,m_oBlinksFrame);
	cv::bitwise_not(m_oFGMask_last_dilated,m_oTempFGMask);
	cv::bitwise_and(m_oBlinksFrame,m_oTempFGMask,m_oBlinksFrame);
	m_oPureFGBlinkMask_curr.copyTo(m_oPureFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oPureFGMask_last);
	//cv::medianBlur(oCurrFGMask,m_oTempFGMask,3);
	//cv::morphologyEx(m_oTempFGMask,m_oTempFGMask,cv::MORPH_CLOSE,cv::Mat());
	cv::morphologyEx(oCurrFGMask,m_oTempFGMask,cv::MORPH_CLOSE,cv::Mat());
	cv::floodFill(m_oTempFGMask,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oTempFGMask,m_oTempFGMask);
	cv::bitwise_or(oCurrFGMask,m_oTempFGMask,oCurrFGMask);
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,9);
	cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat());
	cv::bitwise_not(m_oFGMask_last_dilated,m_oTempFGMask);
	cv::bitwise_and(m_oBlinksFrame,m_oTempFGMask,m_oBlinksFrame);
	m_oFGMask_last.copyTo(oCurrFGMask);
}

void BackgroundSubtractorPBASLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_Assert(m_bInitialized);
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
