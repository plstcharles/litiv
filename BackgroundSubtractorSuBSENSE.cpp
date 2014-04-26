#include "BackgroundSubtractorSuBSENSE.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

static int test_reset=0;

#define USE_MEAN_MIN_DIST_WITH_INTERS_COUNT 1
#define NONZERO_DESC_POPCOUNT_MIN 2
#define NONZERO_ADJUSTMENT_RATIO_MIN 0.050f
#define NONZERO_ADJUSTMENT_RATIO_MAX 0.400f
#define SPREAD_3X3_MIN_R_DIST_FACT 1.25f
#define COLOR_DIFF_RATIO_RESET_THRESHOLD 18

// local define used for debug purposes only
#define DISPLAY_SUBSENSE_DEBUG_INFO 0
// local define used to specify the debug window size
#define DEBUG_WINDOW_SIZE cv::Size(320,240)
// local define used to specify the color dist threshold offset used for unstable regions
#define STAB_COLOR_DIST_OFFSET 6
// local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET 3
// local define used to determine the default image ROI size
#define DEFAULT_IMG_ROI_SIZE (320*240)
// local define used to determine the median blur kernel size
#define DEFAULT_MEDIAN_BLUR_KERNEL_SIZE (9)
// local define used to determine which model regions are 'super-stable'
#define SUPER_STABLE_REGION_DMIN 0.025f
// local define used to set the sub-min R threshold
#define SUPER_STABLE_REGION_RMIN 0.750f

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorSuBSENSE::BackgroundSubtractorSuBSENSE(	 float fRelLBSPThreshold
															,size_t nLBSPThresholdOffset
															,size_t nMinDescDistThreshold
															,size_t nMinColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fRelLBSPThreshold,nMinDescDistThreshold,nLBSPThresholdOffset)
		,m_bInitializedInternalStructs(false)
		,m_nMinColorDistThreshold(nMinColorDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nFrameIndex(SIZE_MAX)
		,m_fLastColorDiffRatio(0.0f)
		,m_nModelResetCooldown(0)
		,m_nCurrLearningRateLowerCap(BGSSUBSENSE_T_LOWER)
		,m_nCurrLearningRateUpperCap(BGSSUBSENSE_T_UPPER)
		,m_nMedianBlurKernelSize(3)
		,m_bUse3x3Spread(true) {
	CV_Assert(m_nBGSamples>0 && m_nRequiredBGSamples<=m_nBGSamples);
	CV_Assert(m_nMinColorDistThreshold>=STAB_COLOR_DIST_OFFSET);
}

BackgroundSubtractorSuBSENSE::~BackgroundSubtractorSuBSENSE() {}

void BackgroundSubtractorSuBSENSE::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	// == init
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC3 || oInitImg.type()==CV_8UC1);
	if(oInitImg.type()==CV_8UC3) {
		std::vector<cv::Mat> voInitImgChannels;
		cv::split(oInitImg,voInitImgChannels);
		bool eq = std::equal(voInitImgChannels[0].begin<uchar>(), voInitImgChannels[0].end<uchar>(), voInitImgChannels[1].begin<uchar>())
				&& std::equal(voInitImgChannels[1].begin<uchar>(), voInitImgChannels[1].end<uchar>(), voInitImgChannels[2].begin<uchar>());
		if(eq)
			std::cout << std::endl << "\tBackgroundSubtractorSuBSENSE : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
	}
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
	m_nFrameIndex = 0;
	m_fLastColorDiffRatio = 0.0f;
	m_nModelResetCooldown = 0;
	m_nCurrLearningRateLowerCap = BGSSUBSENSE_T_LOWER;
	m_nCurrLearningRateUpperCap = BGSSUBSENSE_T_UPPER;
	if(m_oImgSize.height*m_oImgSize.width>DEFAULT_IMG_ROI_SIZE) {
		const int nRawMedianBlurKernelSize = std::min((int)floor((float)(m_oImgSize.height*m_oImgSize.width)/DEFAULT_IMG_ROI_SIZE+0.5f)+DEFAULT_MEDIAN_BLUR_KERNEL_SIZE,14);
		if((nRawMedianBlurKernelSize%2)==1)
			m_nMedianBlurKernelSize = nRawMedianBlurKernelSize;
		else
			m_nMedianBlurKernelSize = nRawMedianBlurKernelSize-1;
	}
	else
		m_nMedianBlurKernelSize = DEFAULT_MEDIAN_BLUR_KERNEL_SIZE;
	m_bUse3x3Spread = (m_oImgSize.height*m_oImgSize.width)>DEFAULT_IMG_ROI_SIZE*2?false:true;
	std::cout << m_oImgSize << " => m_nMedianBlurKernelSize=" << m_nMedianBlurKernelSize << ", with 3x3Spread=" << m_bUse3x3Spread << std::endl;
	m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
	m_oUpdateRateFrame = cv::Scalar(BGSSUBSENSE_T_LOWER);
	m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdFrame = cv::Scalar(1.0f);
	m_oDistThresholdVariationFrame.create(m_oImgSize,CV_32FC1);
	m_oDistThresholdVariationFrame = cv::Scalar(10.0f);
	m_oMeanMinDistFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanMinDistFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanMinDistFrame_ST = cv::Scalar(0.0f);
	m_oMeanLastDistFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanLastDistFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanLastDistFrame_ST = cv::Scalar(0.0f);
	m_oMeanDownSampledLastDistFrame_LT.create(cv::Size(m_oImgSize.width/8,m_oImgSize.height/8),CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_LT = cv::Scalar(0.0f);
	m_oMeanDownSampledLastDistFrame_ST.create(cv::Size(m_oImgSize.width/8,m_oImgSize.height/8),CV_32FC((int)m_nImgChannels));
	m_oMeanDownSampledLastDistFrame_ST = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanRawSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanRawSegmResFrame_ST = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_LT = cv::Scalar(0.0f);
	m_oMeanFinalSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
	m_oMeanFinalSegmResFrame_ST = cv::Scalar(0.0f);
	m_oUnstableRegionMask.create(m_oImgSize,CV_8UC1);
	m_oUnstableRegionMask = cv::Scalar_<uchar>(0);
	m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
	m_oBlinksFrame = cv::Scalar_<uchar>(0);
	m_oDownSampledColorFrame.create(cv::Size(m_oImgSize.width/8,m_oImgSize.height/8),CV_8UC((int)m_nImgChannels));
	m_oDownSampledColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastColorFrame.create(m_oImgSize,CV_8UC((int)m_nImgChannels));
	m_oLastColorFrame = cv::Scalar_<uchar>::all(0);
	m_oLastDescFrame.create(m_oImgSize,CV_16UC((int)m_nImgChannels));
	m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
	m_oRawFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oRawFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated = cv::Scalar_<uchar>(0);
	m_oFGMask_last_dilated_inverted.create(m_oImgSize,CV_8UC1);
	m_oFGMask_last_dilated_inverted = cv::Scalar_<uchar>(0);
	m_oFGMask_FloodedHoles.create(m_oImgSize,CV_8UC1);
	m_oFGMask_FloodedHoles = cv::Scalar_<uchar>(0);
	m_oFGMask_PreFlood.create(m_oImgSize,CV_8UC1);
	m_oFGMask_PreFlood = cv::Scalar_<uchar>(0);
	m_oRawFGBlinkMask_curr.create(m_oImgSize,CV_8UC1);
	m_oRawFGBlinkMask_curr = cv::Scalar_<uchar>(0);
	m_oRawFGBlinkMask_last.create(m_oImgSize,CV_8UC1);
	m_oRawFGBlinkMask_last = cv::Scalar_<uchar>(0);
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
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset)/2);
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			m_oLastColorFrame.data[idx_color] = oInitImg.data[idx_color];
			LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(m_oLastDescFrame.data+idx_desc)));
		}
	}
	else { //m_nImgChannels==3
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset);
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
				LBSP::computeSingleRGBDescriptor(oInitImg,nCurrBGInitColor,x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(m_oLastDescFrame.data+idx_desc))[c]);
			}
		}
	}
	m_bInitializedInternalStructs = true;
	refreshModel(1.0f);
	m_bInitialized = true;
}

void BackgroundSubtractorSuBSENSE::refreshModel(float fSamplesRefreshFrac) {
	// == refresh
	CV_Assert(m_bInitializedInternalStructs);
	CV_Assert(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f);
	const size_t nKeyPoints = m_voKeyPoints.size();
	const size_t nBGSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
	const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
	if(m_nImgChannels==1) {
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols && m_oLastColorFrame.step.p[1]==1);
			const size_t idx_orig_color = m_oLastColorFrame.cols*y_orig + x_orig;
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = m_oLastColorFrame.cols*y_sample + x_sample;
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;
				m_voBGColorSamples[idx_sample].data[idx_orig_color] = m_oLastColorFrame.data[idx_sample_color];
				*((ushort*)(m_voBGDescSamples[idx_sample].data+idx_orig_desc)) = *((ushort*)(m_oLastDescFrame.data+idx_sample_desc));
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(m_oLastColorFrame.step.p[0]==(size_t)m_oLastColorFrame.cols*3 && m_oLastColorFrame.step.p[1]==3);
			const size_t idx_orig_color = 3*(m_oLastColorFrame.cols*y_orig + x_orig);
			CV_DbgAssert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=nRefreshStartPos; s<nRefreshStartPos+nBGSamplesToRefresh; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = 3*(m_oLastColorFrame.cols*y_sample + x_sample);
				const size_t idx_sample_desc = idx_sample_color*2;
				const size_t idx_sample = s%m_nBGSamples;
				uchar* bg_color_ptr = m_voBGColorSamples[idx_sample].data+idx_orig_color;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDescSamples[idx_sample].data+idx_orig_desc);
				const uchar* const init_color_ptr = m_oLastColorFrame.data+idx_sample_color;
				const ushort* const init_desc_ptr = (ushort*)(m_oLastDescFrame.data+idx_sample_desc);
				for(size_t c=0; c<3; ++c) {
					bg_color_ptr[c] = init_color_ptr[c];
					bg_desc_ptr[c] = init_desc_ptr[c];
				}
			}
		}
	}
}

void BackgroundSubtractorSuBSENSE::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
	// == process
	CV_DbgAssert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat();
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oCurrFGMask = _fgmask.getMat();
	memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
	size_t nNonZeroDescCount = 0;
	const size_t nKeyPoints = m_voKeyPoints.size();
	const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIndex,(size_t)BGSSUBSENSE_N_SAMPLES_FOR_LT_MVAVGS);
	const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,(size_t)BGSSUBSENSE_N_SAMPLES_FOR_ST_MVAVGS);
	if(m_nImgChannels==1) {
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_ushrt = idx_uchar*2;
			const size_t idx_flt32 = idx_uchar*4;
			const uchar nCurrColor = oInputImg.data[idx_uchar];
			size_t nMinDescDist = s_nDescMaxDataRange_1ch;
			size_t nMinSumDist = s_nColorMaxDataRange_1ch;
			float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+idx_flt32);
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+idx_flt32));
			float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+idx_flt32));
			float* pfCurrMeanLastDist_LT = ((float*)(m_oMeanLastDistFrame_LT.data+idx_flt32));
			float* pfCurrMeanLastDist_ST = ((float*)(m_oMeanLastDistFrame_ST.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+idx_flt32));
			ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+idx_ushrt));
			uchar& nLastColor = m_oLastColorFrame.data[idx_uchar];
			const size_t nCurrColorDistThreshold = (size_t)((((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET))/2);
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			ushort nCurrInterDesc, nCurrIntraDesc;
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
			m_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>BGSSUBSENSE_INSTBLTY_DETECTION_MIN_R_VAL || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>BGSSUBSENSE_INSTBLTY_DETECTION_SEGM_DIFF || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>BGSSUBSENSE_INSTBLTY_DETECTION_SEGM_DIFF)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[idx_uchar];
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,nBGColor);
					if(nColorDist>nCurrColorDistThreshold)
						goto failedcheck1ch;
					const ushort& nBGIntraDesc = *((ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt));
					const size_t nIntraDescDist = hdist_ushort_8bitLUT(nCurrIntraDesc,nBGIntraDesc);
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,x,y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInterDesc);
					const size_t nInterDescDist = hdist_ushort_8bitLUT(nCurrInterDesc,nBGIntraDesc);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					if(nDescDist>nCurrDescDistThreshold)
						goto failedcheck1ch;
					const size_t nSumDist = std::min((nDescDist/4)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
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
			const float fNormalizedLastDist = ((float)absdiff_uchar(nLastColor,nCurrColor)/s_nColorMaxDataRange_1ch+(float)hdist_ushort_8bitLUT(nLastIntraDesc,nCurrIntraDesc)/s_nDescMaxDataRange_1ch)/2;
			*pfCurrMeanLastDist_LT = (*pfCurrMeanLastDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedLastDist*fRollAvgFactor_LT;
			*pfCurrMeanLastDist_ST = (*pfCurrMeanLastDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
#if !USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
			const float fNormalizedMinDist = ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2;
			*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
#endif //!USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
#if USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
#endif //USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%2)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
			}
			else {
				// == background
#if USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				const float fNormalizedMinDist = ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2;
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
#endif //USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
				int x_rand,y_rand;
				//const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && (!m_oUnstableRegionMask.data[idx_uchar] || (*pfCurrDistThresholdFactor)<SPREAD_3X3_MIN_R_DIST_FACT);
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist_ST = *((float*)(m_oMeanLastDistFrame_ST.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes_ST = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				//if((n_rand%(bCurrUsing3x3Spread?nLearningRate:std::max(nLearningRate/2,(size_t)BGSSUBSENSE_T_LOWER)))==0
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes_ST>BGSSUBSENSE_GHOST_DETECTION_S_MIN && fRandMeanLastDist_ST<BGSSUBSENSE_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) {
					const size_t idx_rand_ushrt = idx_rand_uchar*2;
					const size_t s_rand = rand()%m_nBGSamples;
					*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt)) = nCurrIntraDesc;
					m_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
				}
			}
			if((*pfCurrLearningRate)<BGSSUBSENSE_T_UPPER && m_oFGMask_last.data[idx_uchar]) {
				*pfCurrLearningRate += BGSSUBSENSE_T_INCR/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
				if((*pfCurrLearningRate)>m_nCurrLearningRateUpperCap)
					*pfCurrLearningRate = m_nCurrLearningRateUpperCap;
			}
			else if((*pfCurrLearningRate)>m_nCurrLearningRateLowerCap) {
				*pfCurrLearningRate -= BGSSUBSENSE_T_DECR/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
				if((*pfCurrLearningRate)<m_nCurrLearningRateLowerCap)
					*pfCurrLearningRate = m_nCurrLearningRateLowerCap;
			}
			if((std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>0.1f && m_oBlinksFrame.data[idx_uchar])
				|| ((*pfCurrMeanRawSegmRes_ST)>BGSSUBSENSE_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist_LT)>BGSSUBSENSE_HIGH_VAR_DETECTION_D_MIN)
				|| ((*pfCurrMeanRawSegmRes_ST)>BGSSUBSENSE_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist_LT)>BGSSUBSENSE_HIGH_VAR_DETECTION_D_MIN2)) {
				(*pfCurrDistThresholdVariationFactor) += BGSSUBSENSE_R2_INCR;
			}
			else if((*pfCurrDistThresholdVariationFactor)>BGSSUBSENSE_R2_DECR) {
				//(*pfCurrDistThresholdVariationFactor) -= m_oFGMask_last.data[idx_uchar]?BGSSUBSENSE_R2_DECR/8:m_oUnstableRegionMask.data[idx_uchar]?BGSSUBSENSE_R2_DECR/4:BGSSUBSENSE_R2_DECR;
				(*pfCurrDistThresholdVariationFactor) -= m_oFGMask_last.data[idx_uchar]?BGSSUBSENSE_R2_DECR/4:m_oUnstableRegionMask.data[idx_uchar]?BGSSUBSENSE_R2_DECR/2:BGSSUBSENSE_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<BGSSUBSENSE_R2_DECR)
					(*pfCurrDistThresholdVariationFactor) = BGSSUBSENSE_R2_DECR;
			}
			if((*pfCurrMeanMinDist_LT)<SUPER_STABLE_REGION_DMIN && (*pfCurrDistThresholdFactor)>SUPER_STABLE_REGION_RMIN) {
				(*pfCurrDistThresholdFactor) -= BGSSUBSENSE_R_VAR/8;
				if((*pfCurrDistThresholdFactor)<SUPER_STABLE_REGION_RMIN)
					(*pfCurrDistThresholdFactor) = SUPER_STABLE_REGION_RMIN;
			}
			else if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2)) {
				(*pfCurrDistThresholdFactor) += BGSSUBSENSE_R_VAR*(*pfCurrDistThresholdVariationFactor-BGSSUBSENSE_R2_DECR);
			}
			else if((*pfCurrDistThresholdFactor)>1.0f) {
				(*pfCurrDistThresholdFactor) -= BGSSUBSENSE_R_VAR/(*pfCurrDistThresholdVariationFactor);
				if((*pfCurrDistThresholdFactor)<1.0f)
					(*pfCurrDistThresholdFactor) = 1.0f;
			}
			if(popcount_ushort_8bitsLUT(nCurrIntraDesc)>=s_nDescMaxDataRange_1ch/8-1)
				++nNonZeroDescCount;
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
			float* pfCurrDistThresholdVariationFactor = (float*)(m_oDistThresholdVariationFrame.data+idx_flt32);
			float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+idx_flt32));
			float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+idx_flt32));
			float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+idx_flt32));
			float* pfCurrMeanLastDist_LT = ((float*)(m_oMeanLastDistFrame_LT.data+idx_flt32));
			float* pfCurrMeanLastDist_ST = ((float*)(m_oMeanLastDistFrame_ST.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+idx_flt32));
			float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+idx_flt32));
			ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+idx_ushrt_rgb));
			uchar* anLastColor = m_oLastColorFrame.data+idx_uchar_rgb;
			const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[idx_uchar])*STAB_COLOR_DIST_OFFSET));
			const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThreshold+(m_oUnstableRegionMask.data[idx_uchar]*UNSTAB_DESC_DIST_OFFSET);
			const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
			const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
			const size_t nCurrSCColorDistThreshold = (3*nCurrTotColorDistThreshold)/4;
			//const size_t nCurrSCDescDistThreshold = (3*nCurrTotDescDistThreshold)/4;
			ushort anCurrInterDesc[3], anCurrIntraDesc[3];
			const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
			LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
			m_oUnstableRegionMask.data[idx_uchar] = ((*pfCurrDistThresholdFactor)>BGSSUBSENSE_INSTBLTY_DETECTION_MIN_R_VAL || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>BGSSUBSENSE_INSTBLTY_DETECTION_SEGM_DIFF || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>BGSSUBSENSE_INSTBLTY_DETECTION_SEGM_DIFF)?1:0;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
				size_t nTotDescDist = 0;
				size_t nTotSumDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					size_t nIntraDescDist = hdist_ushort_8bitLUT(anCurrIntraDesc[c],anBGIntraDesc[c]);
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
					size_t nInterDescDist = hdist_ushort_8bitLUT(anCurrInterDesc[c],anBGIntraDesc[c]);
					const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
					//if(nDescDist>nCurrSCDescDistThreshold)
					//	goto failedcheck3ch;
					const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
					if(nSumDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
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
			const float fNormalizedLastDist = ((float)L1dist_uchar(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist_ushort_8bitLUT(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
			*pfCurrMeanLastDist_LT = (*pfCurrMeanLastDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedLastDist*fRollAvgFactor_LT;
			*pfCurrMeanLastDist_ST = (*pfCurrMeanLastDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
#if !USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
			const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
			*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
			*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
#endif //!USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
			if(nGoodSamplesCount<m_nRequiredBGSamples) {
				// == foreground
#if USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
#endif //USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
				oCurrFGMask.data[idx_uchar] = UCHAR_MAX;
				if(m_nModelResetCooldown && (rand()%2)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
					}
				}
			}
			else {
				// == background
#if USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
				*pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
				*pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
#endif //USE_MEAN_MIN_DIST_WITH_INTERS_COUNT
				*pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
				*pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
				const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
					}
				}
				int x_rand,y_rand;
				//const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[idx_uchar];
				const bool bCurrUsing3x3Spread = m_bUse3x3Spread && (!m_oUnstableRegionMask.data[idx_uchar] || (*pfCurrDistThresholdFactor)<SPREAD_3X3_MIN_R_DIST_FACT);
				if(bCurrUsing3x3Spread)
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				else
					getRandNeighborPosition_5x5(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t n_rand = rand();
				const size_t idx_rand_uchar = m_oImgSize.width*y_rand + x_rand;
				const size_t idx_rand_flt32 = idx_rand_uchar*4;
				const float fRandMeanLastDist_ST = *((float*)(m_oMeanLastDistFrame_ST.data+idx_rand_flt32));
				const float fRandMeanRawSegmRes_ST = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
				//if((n_rand%(bCurrUsing3x3Spread?nLearningRate:std::max(nLearningRate/2,(size_t)BGSSUBSENSE_T_LOWER)))==0
				if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
					|| (fRandMeanRawSegmRes_ST>BGSSUBSENSE_GHOST_DETECTION_S_MIN && fRandMeanLastDist_ST<BGSSUBSENSE_GHOST_DETECTION_D_MAX && (n_rand%4)==0)) {
					const size_t idx_rand_uchar_rgb = idx_rand_uchar*3;
					const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb*2;
					const size_t s_rand = rand()%m_nBGSamples;
					for(size_t c=0; c<3; ++c) {
						*((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
						*(m_voBGColorSamples[s_rand].data+idx_rand_uchar_rgb+c) = anCurrColor[c];
					}
				}
			}
			if((*pfCurrLearningRate)<BGSSUBSENSE_T_UPPER && m_oFGMask_last.data[idx_uchar]) {
				*pfCurrLearningRate += BGSSUBSENSE_T_INCR/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
				if((*pfCurrLearningRate)>m_nCurrLearningRateUpperCap)
					*pfCurrLearningRate = m_nCurrLearningRateUpperCap;
			}
			else if((*pfCurrLearningRate)>m_nCurrLearningRateLowerCap) {
				*pfCurrLearningRate -= BGSSUBSENSE_T_DECR/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
				if((*pfCurrLearningRate)<m_nCurrLearningRateLowerCap)
					*pfCurrLearningRate = m_nCurrLearningRateLowerCap;
			}
			if((std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>0.1f && m_oBlinksFrame.data[idx_uchar])
				|| ((*pfCurrMeanRawSegmRes_ST)>BGSSUBSENSE_HIGH_VAR_DETECTION_S_MIN && (*pfCurrMeanLastDist_LT)>BGSSUBSENSE_HIGH_VAR_DETECTION_D_MIN)
				|| ((*pfCurrMeanRawSegmRes_ST)>BGSSUBSENSE_HIGH_VAR_DETECTION_S_MIN2 && (*pfCurrMeanLastDist_LT)>BGSSUBSENSE_HIGH_VAR_DETECTION_D_MIN2)) {
				(*pfCurrDistThresholdVariationFactor) += BGSSUBSENSE_R2_INCR;
			}
			else if((*pfCurrDistThresholdVariationFactor)>BGSSUBSENSE_R2_DECR) {
				//(*pfCurrDistThresholdVariationFactor) -= m_oFGMask_last.data[idx_uchar]?BGSSUBSENSE_R2_DECR/8:m_oUnstableRegionMask.data[idx_uchar]?BGSSUBSENSE_R2_DECR/4:BGSSUBSENSE_R2_DECR;
				(*pfCurrDistThresholdVariationFactor) -= m_oFGMask_last.data[idx_uchar]?BGSSUBSENSE_R2_DECR/4:m_oUnstableRegionMask.data[idx_uchar]?BGSSUBSENSE_R2_DECR/2:BGSSUBSENSE_R2_DECR;
				if((*pfCurrDistThresholdVariationFactor)<BGSSUBSENSE_R2_DECR)
					(*pfCurrDistThresholdVariationFactor) = BGSSUBSENSE_R2_DECR;
			}
			if((*pfCurrMeanMinDist_LT)<SUPER_STABLE_REGION_DMIN && (*pfCurrDistThresholdFactor)>SUPER_STABLE_REGION_RMIN) {
				(*pfCurrDistThresholdFactor) -= BGSSUBSENSE_R_VAR/8;
				if((*pfCurrDistThresholdFactor)<SUPER_STABLE_REGION_RMIN)
					(*pfCurrDistThresholdFactor) = SUPER_STABLE_REGION_RMIN;
			}
			else if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2)) {
				(*pfCurrDistThresholdFactor) += BGSSUBSENSE_R_VAR*(*pfCurrDistThresholdVariationFactor-BGSSUBSENSE_R2_DECR);
			}
			else if((*pfCurrDistThresholdFactor)>1.0f) {
				(*pfCurrDistThresholdFactor) -= BGSSUBSENSE_R_VAR/(*pfCurrDistThresholdVariationFactor);
				if((*pfCurrDistThresholdFactor)<1.0f)
					(*pfCurrDistThresholdFactor) = 1.0f;
			}
			if(popcount_ushort_8bitsLUT(anCurrIntraDesc)>s_nDescMaxDataRange_3ch/8-1)
				++nNonZeroDescCount;
			for(size_t c=0; c<3; ++c) {
				anLastIntraDesc[c] = anCurrIntraDesc[c];
				anLastColor[c] = anCurrColor[c];
			}
		}
	}
#if DISPLAY_SUBSENSE_DEBUG_INFO
	/*cv::Mat oMeanMinDistFrameNormalized_ST; m_oMeanMinDistFrame_ST.copyTo(oMeanMinDistFrameNormalized_ST);
	cv::resize(oMeanMinDistFrameNormalized_ST,oMeanMinDistFrameNormalized_ST,DEBUG_WINDOW_SIZE);
	cv::imshow("d_min_st(x)",oMeanMinDistFrameNormalized_ST);
	cv::Mat oMeanMinDistFrameNormalized_LT; m_oMeanMinDistFrame_LT.copyTo(oMeanMinDistFrameNormalized_LT);
	cv::resize(oMeanMinDistFrameNormalized_LT,oMeanMinDistFrameNormalized_LT,DEBUG_WINDOW_SIZE);
	cv::imshow("d_min_lt(x)",oMeanMinDistFrameNormalized_LT);
	cv::Mat oMeanLastDistFrameNormalized_ST; m_oMeanLastDistFrame_ST.copyTo(oMeanLastDistFrameNormalized_ST);
	cv::resize(oMeanLastDistFrameNormalized_ST,oMeanLastDistFrameNormalized_ST,DEBUG_WINDOW_SIZE);
	cv::imshow("d_last_st(x)",oMeanLastDistFrameNormalized_ST);
	cv::Mat oMeanLastDistFrameNormalized_LT; m_oMeanLastDistFrame_LT.copyTo(oMeanLastDistFrameNormalized_LT);
	cv::resize(oMeanLastDistFrameNormalized_LT,oMeanLastDistFrameNormalized_LT,DEBUG_WINDOW_SIZE);
	cv::imshow("d_last_lt(x)",oMeanLastDistFrameNormalized_LT);*/
	std::cout << std::endl;
	cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
	cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame_ST.copyTo(oMeanMinDistFrameNormalized);
	cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,DEBUG_WINDOW_SIZE);
	cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame_ST.copyTo(oMeanLastDistFrameNormalized);
	cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,DEBUG_WINDOW_SIZE);
	cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanRawSegmResFrameNormalized; m_oMeanRawSegmResFrame_ST.copyTo(oMeanRawSegmResFrameNormalized);
	cv::circle(oMeanRawSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanRawSegmResFrameNormalized,oMeanRawSegmResFrameNormalized,DEBUG_WINDOW_SIZE);
	cv::imshow("s_avg(x)",oMeanRawSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  s_avg(" << dbgpt << ") = " << m_oMeanRawSegmResFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oMeanFinalSegmResFrameNormalized; m_oMeanFinalSegmResFrame_ST.copyTo(oMeanFinalSegmResFrameNormalized);
	cv::circle(oMeanFinalSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oMeanFinalSegmResFrameNormalized,oMeanFinalSegmResFrameNormalized,DEBUG_WINDOW_SIZE);
	cv::imshow("z_avg(x)",oMeanFinalSegmResFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "  z_avg(" << dbgpt << ") = " << m_oMeanFinalSegmResFrame_ST.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,0.25f,-0.25f);
	cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEBUG_WINDOW_SIZE);
	cv::imshow("r(x)",oDistThresholdFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oDistThresholdVariationFrameNormalized; cv::normalize(m_oDistThresholdVariationFrame,oDistThresholdVariationFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
	cv::circle(oDistThresholdVariationFrameNormalized,dbgpt,5,cv::Scalar(255));
	cv::resize(oDistThresholdVariationFrameNormalized,oDistThresholdVariationFrameNormalized,DEBUG_WINDOW_SIZE);
	cv::imshow("r2(x)",oDistThresholdVariationFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "     r2(" << dbgpt << ") = " << m_oDistThresholdVariationFrame.at<float>(dbgpt) << std::endl;
	cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/BGSSUBSENSE_T_UPPER,-BGSSUBSENSE_T_LOWER/BGSSUBSENSE_T_UPPER);
	cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
	cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,DEBUG_WINDOW_SIZE);
	cv::imshow("t(x)",oUpdateRateFrameNormalized);
	std::cout << std::fixed << std::setprecision(5) << "      t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
#endif //DISPLAY_SUBSENSE_DEBUG_INFO
	cv::bitwise_xor(oCurrFGMask,m_oRawFGMask_last,m_oRawFGBlinkMask_curr);
	cv::bitwise_or(m_oRawFGBlinkMask_curr,m_oRawFGBlinkMask_last,m_oBlinksFrame);
	m_oRawFGBlinkMask_curr.copyTo(m_oRawFGBlinkMask_last);
	oCurrFGMask.copyTo(m_oRawFGMask_last);
	//cv::imshow("orig raw",oCurrFGMask);
	cv::morphologyEx(oCurrFGMask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
	//cv::imshow("post-1close",m_oFGMask_PreFlood);
	m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
	cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
	cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
	//cv::imshow("post-1close, flooded+inverted",m_oFGMask_FloodedHoles);
	cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),2);
	//cv::imshow("post-1close, 2eroded",m_oFGMask_PreFlood);
	cv::bitwise_or(oCurrFGMask,m_oFGMask_FloodedHoles,oCurrFGMask);
	cv::bitwise_or(oCurrFGMask,m_oFGMask_PreFlood,oCurrFGMask);
	cv::medianBlur(oCurrFGMask,m_oFGMask_last,m_nMedianBlurKernelSize);
	//cv::imshow("result",m_oFGMask_last);
	cv::dilate(m_oFGMask_last,m_oFGMask_last_dilated,cv::Mat(),cv::Point(-1,-1),3);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	cv::bitwise_not(m_oFGMask_last_dilated,m_oFGMask_last_dilated_inverted);
	cv::bitwise_and(m_oBlinksFrame,m_oFGMask_last_dilated_inverted,m_oBlinksFrame);
	m_oFGMask_last.copyTo(oCurrFGMask);
	cv::resize(oInputImg,m_oDownSampledColorFrame,cv::Size(m_oImgSize.width/8,m_oImgSize.height/8),0,0,cv::INTER_AREA);
	cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_LT,fRollAvgFactor_LT);
	cv::accumulateWeighted(m_oDownSampledColorFrame,m_oMeanDownSampledLastDistFrame_ST,fRollAvgFactor_ST);
	cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
	cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oFGMask_last,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);
	const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/nKeyPoints;
	std::cout << "\t nzdesc = " << (int)(fCurrNonZeroDescRatio*100) << "%" << std::endl;
	if(fCurrNonZeroDescRatio<NONZERO_ADJUSTMENT_RATIO_MIN) {
		std::cout << "\t LBSP-- : [";
	    for(size_t t=0; t<=UCHAR_MAX; ++t) {
	        if(m_anLBSPThreshold_8bitLUT[t]>m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4))
	            --m_anLBSPThreshold_8bitLUT[t];
	        std::cout << m_anLBSPThreshold_8bitLUT[t] << " ";
	    }
	    std::cout << "];" << std::endl;
	}
	else if(fCurrNonZeroDescRatio>NONZERO_ADJUSTMENT_RATIO_MAX) {
		std::cout << "\t LBSP++ : [";
	    for(size_t t=0; t<=UCHAR_MAX; ++t) {
	        if(m_anLBSPThreshold_8bitLUT[t]<(UCHAR_MAX*m_fRelLBSPThreshold))
	            ++m_anLBSPThreshold_8bitLUT[t];
	        std::cout << m_anLBSPThreshold_8bitLUT[t] << " ";
	    }
	    std::cout << "];" << std::endl;
	}
	size_t nTotColorDiff = 0;
	for(int i=0; i<m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
		const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0]*i;
		for(int j=0; j<m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
			const size_t idx2 = idx1+m_oMeanDownSampledLastDistFrame_ST.step.p[1]*j;
			if(m_nImgChannels==1)
				nTotColorDiff += (size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2)));
			else //m_nImgChannels==3
				nTotColorDiff += std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2))),
								std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+4))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+4))),
										(size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+8))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+8)))));
		}
	}
	m_fLastColorDiffRatio = (float)nTotColorDiff/(m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
	std::cout << "\t totcolordiff = " << (int)(m_fLastColorDiffRatio) << "%" << std::endl;
	if(m_fLastColorDiffRatio>=COLOR_DIFF_RATIO_RESET_THRESHOLD && m_nModelResetCooldown==0) {
		refreshModel(0.50f);
		m_nModelResetCooldown = BGSSUBSENSE_N_SAMPLES_FOR_ST_MVAVGS;
		m_oUpdateRateFrame = cv::Scalar(1.0f);
		m_oDistThresholdFrame = cv::Scalar(1.0f);
		m_oDistThresholdVariationFrame = cv::Scalar(2.0f);
		m_oMeanMinDistFrame_LT = cv::Scalar(0.0f);
		m_oMeanMinDistFrame_ST = cv::Scalar(0.0f);
		m_oUnstableRegionMask = cv::Scalar_<uchar>(0);
		m_oBlinksFrame = cv::Scalar_<uchar>(0);
		int test_reset_local = ++test_reset;
		std::stringstream sstr;
		sstr << "reset_n" << test_reset_local << "_f" << m_nFrameIndex << ".png";
		std::cout << sstr.str() << std::endl;
		cv::imwrite(sstr.str(),oInputImg);
	}
	m_nCurrLearningRateLowerCap = (size_t)(m_fLastColorDiffRatio>=COLOR_DIFF_RATIO_RESET_THRESHOLD/2?BGSSUBSENSE_T_LOWER/2:BGSSUBSENSE_T_LOWER);
	m_nCurrLearningRateUpperCap = (size_t)(m_fLastColorDiffRatio>=COLOR_DIFF_RATIO_RESET_THRESHOLD/3?std::max((int)BGSSUBSENSE_T_UPPER>>(int)(m_fLastColorDiffRatio/2),1):BGSSUBSENSE_T_UPPER);
	std::cout << "\t [lower,upper] caps = [" << m_nCurrLearningRateLowerCap << "," << m_nCurrLearningRateUpperCap << "]" << std::endl;
	if(m_nModelResetCooldown>0)
		--m_nModelResetCooldown;
}

void BackgroundSubtractorSuBSENSE::getBackgroundImage(cv::OutputArray backgroundImage) const {
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
