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

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorPBASLBSP::BackgroundSubtractorPBASLBSP(	 float fLBSPThreshold
															,size_t nLBSPThresholdOffset
															,size_t nInitDescDistThreshold
															,size_t nInitColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nInitDescDistThreshold,nLBSPThresholdOffset)
		,m_nColorDistThreshold(nInitColorDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples) {
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples);
}

BackgroundSubtractorPBASLBSP::~BackgroundSubtractorPBASLBSP() {}

void BackgroundSubtractorPBASLBSP::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	std::vector<cv::KeyPoint> voNewKeyPoints;
	if(voKeyPoints.empty()) {
		cv::DenseFeatureDetector oKPDDetector(1.f, 1, 1.f, 1, 0, true, false);
		voNewKeyPoints.reserve(oInitImg.rows*oInitImg.cols);
		oKPDDetector.detect(cv::Mat(oInitImg.size(),oInitImg.type()),voNewKeyPoints);
	}
	else
		voNewKeyPoints = voKeyPoints;
	LBSP::validateKeyPoints(voNewKeyPoints,oInitImg.size());
	CV_Assert(!voNewKeyPoints.empty());
	m_voKeyPoints = voNewKeyPoints;
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
	m_oBlinksFrame = cv::Scalar_<uchar>(0);
	m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame = cv::Scalar(0.0f);
	m_oMeanSegmResFrame.create(m_oImgSize,CV_32FC1);
	m_oMeanSegmResFrame = cv::Scalar(0.0f);
	m_oTempFGMask.create(m_oImgSize,CV_8UC1);
	m_oTempFGMask = cv::Scalar_<uchar>(0);
	m_oPureFGBlinkMask_curr.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_curr = cv::Scalar_<uchar>(0);
	m_oPureFGBlinkMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGBlinkMask_last = cv::Scalar_<uchar>(0);
	m_oPureFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oPureFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar_<uchar>(0);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_voBGColorSamples.resize(m_nBGSamples);
	m_voBGDescSamples.resize(m_nBGSamples);
	for(size_t s=0; s<m_nBGSamples; ++s) {
		m_voBGColorSamples[s].create(m_oImgSize,CV_8UC((int)m_nImgChannels));
		m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);
		m_voBGDescSamples[s].create(m_oImgSize,CV_16UC((int)m_nImgChannels));
		m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
	}
	const size_t nKeyPoints = m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_nLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT+m_nAbsLBSPThreshold);
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			m_oLastColorFrame.data[idx_color] = oInitImg.data[idx_color];
			LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_nLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_orig_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=0; s<m_nBGSamples; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = m_oLastColorFrame.cols*y_sample + x_sample;
				const size_t idx_sample_desc = idx_sample_color*2;
				m_voBGColorSamples[s].data[idx_orig_color] = m_oLastColorFrame.data[idx_sample_color];
				*((ushort*)(m_voBGDescSamples[s].data+idx_orig_desc)) = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_nLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nAbsLBSPThreshold);
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_color = 3*(m_oLastColorFrame.cols*y_orig + x_orig);
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			for(size_t c=0; c<3; ++c) {
				const uchar nCurrBGInitColor = oInitImg.data[idx_color+c];
				m_oLastColorFrame.data[idx_color+c] = nCurrBGInitColor;
				LBSP::computeSingleRGBDescriptor(oInitImg,nCurrBGInitColor,x_orig,y_orig,c,m_nLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);
			}
		}
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_orig_color = 3*(m_oLastColorFrame.cols*y_orig + x_orig);
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=0; s<m_nBGSamples; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = 3*(m_oLastColorFrame.cols*y_sample + x_sample);
				const size_t idx_sample_desc = idx_sample_color*2;
				uchar* bg_color_ptr = m_voBGColorSamples[s].data+idx_orig_color;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDescSamples[s].data+idx_orig_desc);
				const uchar* const init_color_ptr = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const init_desc_ptr = (ushort*)(m_oLastDescFrame.data+idx_sample_desc);
				for(size_t c=0; c<3; ++c) {
					bg_color_ptr[c] = init_color_ptr[c];
					bg_desc_ptr[c] = init_desc_ptr[c];
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
	const size_t nKeyPoints = m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_ushrt = idx_uchar*2;
			const size_t idx_flt32 = idx_uchar*4;
			const uchar nCurrColor = oInputImg.data[idx_uchar];
			size_t nMinDescDist=s_nDescMaxDataRange_1ch;
			size_t nMinSumDist=s_nColorMaxDataRange_1ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			const size_t nCurrColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			const size_t nCurrDescDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold); // not adjusted like ^^, the internal LBSP thresholds are instead
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_nLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[idx_uchar];
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,nBGColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nBGIntraDesc = *((ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt));
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,x,y,m_nLBSPThreshold_8bitLUT[nBGColor],nCurrInterDesc);
					const size_t nDescDist = (hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc)+hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc))/2;
					if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;
					const size_t nSumDist = std::min((size_t)(OVERLOAD_GRAD_PROP*BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
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
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(absdiff_uchar(nLastColor,nCurrColor))/s_nColorMaxDataRange_1ch+(float)(hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc))/s_nDescMaxDataRange_1ch)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+idx_flt32));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
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
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+idx_rand_flt32));
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSPBASLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSPBASLBSP_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) {
					const size_t idx_rand_ushrt = idx_rand_uchar*2;
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
				}
			}
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) {
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
			nLastColor = nCurrColor;
		}
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_flt32 = idx_uchar*4;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			const uchar* const anCurrColor = oInputImg.data+idx_uchar_rgb;
			size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
			size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			const size_t nCurrTotColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*3);
			const size_t nCurrTotDescDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*3);
#if BGSLBSP_USE_SC_THRS_VALIDATION
			const size_t nCurrSCColorDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
			const size_t nCurrSCDescDistThreshold = (size_t)((*pfCurrDistThresholdFactor)*m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_nLBSPThreshold_8bitLUT[anCurrColor[0]],m_nLBSPThreshold_8bitLUT[anCurrColor[1]],m_nLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_nLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					const size_t nDescDist = (hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c])+hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]))/2;
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					const size_t nSumDist = std::min((size_t)(OVERLOAD_GRAD_PROP*nDescDist)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
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
			float* pfCurrMeanMinDist = ((float*)(m_oMeanMinDistFrame.data+idx_flt32));
			*pfCurrMeanMinDist = ((*pfCurrMeanMinDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+idx_flt32));
			*pfCurrMeanLastDist = ((*pfCurrMeanLastDist)*(BGSPBASLBSP_N_SAMPLES_FOR_MEAN-1) + ((float)(L1dist_uchar(anLastColor,anCurrColor))/s_nColorMaxDataRange_3ch+(float)(hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc))/s_nDescMaxDataRange_3ch)/2)/BGSPBASLBSP_N_SAMPLES_FOR_MEAN;
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanSegmRes = ((float*)(m_oMeanSegmResFrame.data+idx_flt32));
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
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
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil((*pfCurrLearningRate));
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
					}
				}
				int x_rand,y_rand;
				getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
				const float fRandMeanSegmRes = *((float*)(m_oMeanSegmResFrame.data+idx_rand_flt32));
				if((n_rand%(nLearningRate)==0) || (fRandMeanSegmRes>BGSPBASLBSP_GHOST_DETECTION_S_MIN && fRandMeanLastDist<BGSPBASLBSP_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) {
					const size_t idx_rand_uchar_rgb = idx_rand_uchar*3;
					const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb*2;
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_rand_uchar_rgb+c) = anCurrColor[c];
					}
				}
			}
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			if((*pfCurrMeanMinDist)>BGSPBASLBSP_R2_OFFST && m_oBlinksFrame.data[idx_uchar]>0) {
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
			for(size_t c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColor[c] = anCurrColor[c];
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
	cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
	for(size_t s=0; s<m_nBGSamples; ++s) {
		for(int y=0; y<m_oImgSize.height; ++y) {
			for(int x=0; x<m_oImgSize.width; ++x) {
				const size_t idx_nimg = m_voBGColorSamples[s].step.p[0]*y + m_voBGColorSamples[s].step.p[1]*x;
				const size_t idx_flt32 = idx_nimg*4;
				float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+idx_flt32);
				const uchar* const oBGImgPtr = m_voBGColorSamples[s].data+idx_nimg;
				for(size_t c=0; c<m_nImgChannels; ++c)
					oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
			}
		}
	}
	oAvgBGImg.convertTo(backgroundImage,CV_8U);
}
