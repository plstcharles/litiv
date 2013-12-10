#include "BackgroundSubtractorViBeLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

BackgroundSubtractorViBeLBSP::BackgroundSubtractorViBeLBSP(  size_t nLBSPThreshold
															,size_t nDescDistThreshold
															,size_t nColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(nLBSPThreshold,nDescDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nColorDistThreshold(nColorDistThreshold) {
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples);
}

BackgroundSubtractorViBeLBSP::BackgroundSubtractorViBeLBSP(	 float fLBSPThreshold
															,size_t nDescDistThreshold
															,size_t nColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nDescDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nColorDistThreshold(nColorDistThreshold) {
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples);
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
		LBSP oExtractor(m_nImgChannels==3?m_fLBSPThreshold:(m_fLBSPThreshold*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT));
		oExtractor.compute2(oInitImg,m_voKeyPoints,oInitDesc);
	}
	else {
		LBSP oExtractor(m_nImgChannels==3?m_nLBSPThreshold:(size_t)(m_nLBSPThreshold*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT));
		oExtractor.compute2(oInitImg,m_voKeyPoints,oInitDesc);
	}
	m_voBGColorSamples.resize(m_nBGSamples);
	m_voBGDescSamples.resize(m_nBGSamples);
	const size_t nKeyPoints = m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(size_t s=0; s<m_nBGSamples; s++) {
			m_voBGColorSamples[s].create(m_oImgSize,CV_8UC1);
			m_voBGDescSamples[s].create(m_oImgSize,CV_16UC1);
			for(size_t k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				m_voBGColorSamples[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
				m_voBGDescSamples[s].at<ushort>(y_orig,x_orig) = oInitDesc.at<ushort>(y_sample,x_sample);
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t s=0; s<m_nBGSamples; s++) {
			m_voBGColorSamples[s].create(m_oImgSize,CV_8UC3);
			m_voBGColorSamples[s] = cv::Scalar_<uchar>(0,0,0);
			m_voBGDescSamples[s].create(m_oImgSize,CV_16UC3);
			m_voBGDescSamples[s] = cv::Scalar_<ushort>(0,0,0);
			for(size_t k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_orig_color = oInitImg.step.p[0]*y_orig + oInitImg.step.p[1]*x_orig;
				const size_t idx_orig_desc = oInitDesc.step.p[0]*y_orig + oInitDesc.step.p[1]*x_orig;
				const size_t idx_rand_color = oInitImg.step.p[0]*y_sample + oInitImg.step.p[1]*x_sample;
				const size_t idx_rand_desc = oInitDesc.step.p[0]*y_sample + oInitDesc.step.p[1]*x_sample;
				uchar* bg_color_ptr = m_voBGColorSamples[s].data+idx_orig_color;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDescSamples[s].data+idx_orig_desc);
				const uchar* const init_color_ptr = oInitImg.data+idx_rand_color;
				const ushort* const init_desc_ptr = (ushort*)(oInitDesc.data+idx_rand_desc);
				for(size_t c=0; c<3; ++c) {
					bg_color_ptr[c] = init_color_ptr[c];
					bg_desc_ptr[c] = init_desc_ptr[c];
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
	const size_t nKeyPoints = m_voKeyPoints.size();
	const size_t nLearningRate = (size_t)ceil(learningRate);
	if(m_nImgChannels==1) {
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = oInputImg.step.p[0]*y + x;
			const size_t idx_ushrt = idx_uchar*2;
			const uchar nCurrColor = oInputImg.data[idx_uchar];
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			ushort nCurrInputDesc;
#if !BGSLBSP_EXTRACT_INTER_LBSP
			const size_t nCurrLBSPThreshold = (size_t)((m_bLBSPUsingRelThreshold?(m_fLBSPThreshold*nCurrColor):(m_nLBSPThreshold))*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
			LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,nCurrLBSPThreshold,nCurrInputDesc);
#endif //!BGSLBSP_EXTRACT_INTER_LBSP
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar nBGColor = m_voBGColorSamples[nSampleIdx].data[idx_uchar];
				{
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
					const size_t nColorDist = absdiff_uchar(nCurrColor,nBGColor);
					if(nColorDist>m_nColorDistThreshold*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT)
						goto failedcheck1ch;
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#if BGSLBSP_EXTRACT_INTER_LBSP
					const size_t nBGLBSPThreshold = (size_t)((m_bLBSPUsingRelThreshold?(m_fLBSPThreshold*nBGColor):(m_nLBSPThreshold))*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,x,y,nBGLBSPThreshold,nCurrInputDesc);
#endif //BGSLBSP_EXTRACT_INTER_LBSP
					const size_t nDescDist = hdist_ushort_8bitLUT(nCurrInputDesc,*((ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt)));
					if(nDescDist>m_nDescDistThreshold)
						goto failedcheck1ch;
					nGoodSamplesCount++;
				}
				failedcheck1ch:
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.data[idx_uchar] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					ushort& nRandInputDesc = *((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt));
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					nRandInputDesc = nCurrInputDesc;
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					const size_t nCurrLBSPThreshold = (size_t)((m_bLBSPUsingRelThreshold?(m_fLBSPThreshold*nCurrColor):(m_nLBSPThreshold))*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
					LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,nCurrLBSPThreshold,nRandInputDesc);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
					m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
				if((rand()%nLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t s_rand = rand()%m_nBGSamples;
					ushort& nRandInputDesc = m_voBGDescSamples[s_rand].at<ushort>(y_rand,x_rand);
#if BGSVIBELBSP_USE_SELF_DIFFUSION
#if BGSLBSP_MODEL_INTER_LBSP
					const uchar nRandBGColor = m_voBGColorSamples[nSampleIdx].at<uchar>(y_rand,x_rand);
					const size_t nRandBGLBSPThreshold = (size_t)((m_bLBSPUsingRelThreshold?(m_fLBSPThreshold*nRandBGColor):(m_nLBSPThreshold))*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
					LBSP::computeGrayscaleDescriptor(oInputImg,nRandBGColor,x_rand,y_rand,nRandBGLBSPThreshold,nRandInputDesc);
					m_voBGColorSamples[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y_rand,x_rand);
#else //!BGSLBSP_MODEL_INTER_LBSP
					const uchar nRandColor = oInputImg.at<uchar>(y_rand,x_rand);
					const size_t nRandLBSPThreshold = (size_t)((m_bLBSPUsingRelThreshold?(m_fLBSPThreshold*nRandColor):(m_nLBSPThreshold))*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
					LBSP::computeGrayscaleDescriptor(oInputImg,nRandColor,x_rand,y_rand,nRandLBSPThreshold,nRandInputDesc);
					m_voBGColorSamples[s_rand].at<uchar>(y_rand,x_rand) = nRandColor;
#endif //!BGSLBSP_MODEL_INTER_LBSP
#else //!BGSVIBELBSP_USE_SELF_DIFFUSION
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					nRandInputDesc = nCurrInputDesc;
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					const size_t nCurrLBSPThreshold = (size_t)((m_bLBSPUsingRelThreshold?(m_fLBSPThreshold*nCurrColor):(m_nLBSPThreshold))*BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT);
					LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,nCurrLBSPThreshold,nRandInputDesc);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
					m_voBGColorSamples[s_rand].at<uchar>(y_rand,x_rand) = nCurrColor;
#endif //!BGSVIBELBSP_USE_SELF_DIFFUSION
				}
			}
		}
	}
	else { //m_nImgChannels==3
		const size_t nCurrDescDistThreshold = m_nDescDistThreshold*3;
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const size_t nCurrColorDistThreshold = m_nColorDistThreshold*3;
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#if BGSLBSP_USE_SC_THRS_VALIDATION
		const size_t nCurrSCDescDistThreshold = (size_t)(m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const size_t nCurrSCColorDistThreshold = (size_t)(m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const size_t desc_row_step = m_voBGDescSamples[0].step.p[0];
		const size_t img_row_step = m_voBGColorSamples[0].step.p[0];
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const size_t idx_uchar = m_oImgSize.width*y + x;
			const size_t idx_uchar_rgb = idx_uchar*3;
			const size_t idx_ushrt_rgb = idx_uchar_rgb*2;
			const uchar* const anCurrColor = oInputImg.data+idx_uchar_rgb;
			size_t nGoodSamplesCount=0, nSampleIdx=0;
			ushort anCurrInputDesc[3];
#if !BGSLBSP_EXTRACT_INTER_LBSP
			if(m_bLBSPUsingRelThreshold) {
				const size_t anCurrIntraLBSPThresholds[3] = {(size_t)(anCurrColor[0]*m_fLBSPThreshold),(size_t)(anCurrColor[1]*m_fLBSPThreshold),(size_t)(anCurrColor[2]*m_fLBSPThreshold)};
				LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anCurrInputDesc);
			}
			else
				LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,m_nLBSPThreshold,anCurrInputDesc);
#endif //!BGSLBSP_EXTRACT_INTER_LBSP
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
				size_t nTotColorDist = 0;
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
				size_t nTotDescDist = 0;
				for(size_t c=0;c<3; ++c) {
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#if BGSLBSP_EXTRACT_INTER_LBSP
					const size_t nBGLBSPThreshold = m_bLBSPUsingRelThreshold?((size_t)(m_fLBSPThreshold*anBGColor[c])):m_nLBSPThreshold;
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,nBGLBSPThreshold,anCurrInputDesc[c]);
#endif //BGSLBSP_EXTRACT_INTER_LBSP
					const size_t nDescDist = hdist_ushort_8bitLUT(anCurrInputDesc[c],anBGDesc[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
					nTotColorDist += nColorDist;
					nTotDescDist += nDescDist;
				}
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
				if(nTotDescDist<=nCurrDescDistThreshold && nTotColorDist<=nCurrColorDistThreshold)
#else //!BGSVIBELBSP_USE_COLOR_COMPLEMENT
				if(nTotDescDist<=nCurrDescDistThreshold)
#endif //!BGSVIBELBSP_USE_COLOR_COMPLEMENT
					nGoodSamplesCount++;
#if BGSLBSP_USE_SC_THRS_VALIDATION
				failedcheck3ch:
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.data[idx_uchar] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb));
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					for(size_t c=0; c<3; ++c)
						anRandInputDesc[c] = anCurrInputDesc[c];
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					if(m_bLBSPUsingRelThreshold) {
						const size_t anCurrIntraLBSPThresholds[3] = {(size_t)(anCurrColor[0]*m_fLBSPThreshold),(size_t)(anCurrColor[1]*m_fLBSPThreshold),(size_t)(anCurrColor[2]*m_fLBSPThreshold)};
						LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anRandInputDesc);
					}
					else
						LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,m_nLBSPThreshold,anRandInputDesc);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
					for(size_t c=0; c<3; ++c)
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
				}
				if((rand()%nLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t s_rand = rand()%m_nBGSamples;
					ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[s_rand].data + desc_row_step*y_rand + 6*x_rand));
#if BGSVIBELBSP_USE_SELF_DIFFUSION
#if BGSLBSP_MODEL_INTER_LBSP
					const uchar* const anRandInterLBSPRef = m_voBGColorSamples[nSampleIdx].data+img_row_step*y_rand+3*x_rand;
					if(m_bLBSPUsingRelThreshold) {
						const size_t anRandInterLBSPThresholds[3] = {(size_t)(anRandInterLBSPRef[0]*m_fLBSPThreshold),(size_t)(anRandInterLBSPRef[1]*m_fLBSPThreshold),(size_t)(anRandInterLBSPRef[2]*m_fLBSPThreshold)};
						LBSP::computeRGBDescriptor(oInputImg,anRandInterLBSPRef,x_rand,y_rand,anRandInterLBSPThresholds,anRandInputDesc);
					}
					else
						LBSP::computeRGBDescriptor(oInputImg,anRandInterLBSPRef,x_rand,y_rand,m_nLBSPThreshold,anRandInputDesc);
#else //!BGSLBSP_MODEL_INTER_LBSP
					const uchar* const anRandIntraLBSPRef = oInputImg.data+img_row_step*y_rand+3*x_rand;
					if(m_bLBSPUsingRelThreshold) {
						const size_t anRandIntraLBSPThresholds[3] = {(size_t)(anRandIntraLBSPRef[0]*m_fLBSPThreshold),(size_t)(anRandIntraLBSPRef[1]*m_fLBSPThreshold),(size_t)(anRandIntraLBSPRef[2]*m_fLBSPThreshold)};
						LBSP::computeRGBDescriptor(oInputImg,anRandIntraLBSPRef,x_rand,y_rand,anRandIntraLBSPThresholds,anRandInputDesc);
					}
					else
						LBSP::computeRGBDescriptor(oInputImg,anRandIntraLBSPRef,x_rand,y_rand,m_nLBSPThreshold,anRandInputDesc);
#endif //!BGSLBSP_MODEL_INTER_LBSP
					for(size_t c=0; c<3; ++c)
						*(m_voBGColorSamples[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oInputImg.data + img_row_step*y_rand + 3*x_rand + c);
#else //!BGSVIBELBSP_USE_SELF_DIFFUSION
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					for(size_t c=0; c<3; ++c)
						anRandInputDesc[c] = anCurrInputDesc[c];
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					if(m_bLBSPUsingRelThreshold) {
						const size_t anCurrIntraLBSPThresholds[3] = {(size_t)(anCurrColor[0]*m_fLBSPThreshold),(size_t)(anCurrColor[1]*m_fLBSPThreshold),(size_t)(anCurrColor[2]*m_fLBSPThreshold)};
						LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anRandInputDesc);
					}
					else
						LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,m_nLBSPThreshold,anRandInputDesc);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
					for(size_t c=0; c<3; ++c)
						*(m_voBGColorSamples[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = anCurrColor[c];
#endif //!BGSVIBELBSP_USE_SELF_DIFFUSION
				}
			}
		}
	}
	cv::medianBlur(oFGMask,oFGMask,9); // give user access to mblur params... @@@@@
}

void BackgroundSubtractorViBeLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
	CV_DbgAssert(m_bInitialized);
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
