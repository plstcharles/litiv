#include "BackgroundSubtractorViBeLBSP.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

BackgroundSubtractorViBeLBSP::BackgroundSubtractorViBeLBSP()
	:	 BackgroundSubtractorLBSP()
	 	,m_nBGSamples(BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES)
		,m_nRequiredBGSamples(BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES)
		,m_nColorDistThreshold(BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD) {
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
	:	 BackgroundSubtractorLBSP(nLBSPThreshold,nDescDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nColorDistThreshold(nColorDistThreshold) {
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
	:	 BackgroundSubtractorLBSP(fLBSPThreshold,nDescDistThreshold)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nColorDistThreshold(nColorDistThreshold) {
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
	m_voBGImg.resize(m_nBGSamples);
	m_voBGDesc.resize(m_nBGSamples);
	const int nKeyPoints = (int)m_voKeyPoints.size();
	if(m_nImgChannels==1) {
		for(int s=0; s<m_nBGSamples; s++) {
			m_voBGImg[s].create(m_oImgSize,CV_8UC1);
			m_voBGDesc[s].create(m_oImgSize,CV_16UC1);
			for(int k=0; k<nKeyPoints; ++k) {
				const int y_orig = (int)m_voKeyPoints[k].pt.y;
				const int x_orig = (int)m_voKeyPoints[k].pt.x;
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				m_voBGImg[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
				m_voBGDesc[s].at<ushort>(y_orig,x_orig) = oInitDesc.at<ushort>(y_sample,x_sample);
			}
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
				const int idx_orig_img = oInitImg.step.p[0]*y_orig + oInitImg.step.p[1]*x_orig;
				const int idx_orig_desc = oInitDesc.step.p[0]*y_orig + oInitDesc.step.p[1]*x_orig;
				const int idx_rand_img = oInitImg.step.p[0]*y_sample + oInitImg.step.p[1]*x_sample;
				const int idx_rand_desc = oInitDesc.step.p[0]*y_sample + oInitDesc.step.p[1]*x_sample;
				uchar* bg_img_ptr = m_voBGImg[s].data+idx_orig_img;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDesc[s].data+idx_orig_desc);
				const uchar* init_img_ptr = oInitImg.data+idx_rand_img;
				const ushort* init_desc_ptr = (ushort*)(oInitDesc.data+idx_rand_desc);
				for(int n=0;n<3; ++n) {
					bg_img_ptr[n] = init_img_ptr[n];
					bg_desc_ptr[n] = init_desc_ptr[n];
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
	const int nLearningRate = (int)ceil(learningRate);
	if(m_nImgChannels==1) {
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = oInputImg.step.p[0]*y + x;
			const int ushrt_idx = uchar_idx*2;
			int nGoodSamplesCount=0, nSampleIdx=0;
			int nColorDist, nDescDist;
			ushort nCurrInputDesc;
#if !BGSLBSP_EXTRACT_INTER_LBSP
			if(m_bLBSPUsingRelThreshold)
				LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,nCurrInputDesc);
			else
				LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,nCurrInputDesc);
#endif //!BGSLBSP_EXTRACT_INTER_LBSP
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
				nColorDist = absdiff_uchar(oInputImg.data[uchar_idx],m_voBGImg[nSampleIdx].data[uchar_idx]);
				if(nColorDist>m_nColorDistThreshold*LBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT)
					goto failedcheck1ch;
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#if BGSLBSP_EXTRACT_INTER_LBSP
				if(m_bLBSPUsingRelThreshold)
					LBSP::computeGrayscaleRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_fLBSPThreshold,nCurrInputDesc);
				else
					LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,m_nLBSPThreshold,nCurrInputDesc);
#endif //BGSLBSP_EXTRACT_INTER_LBSP
				nDescDist = hdist_ushort_8bitLUT(nCurrInputDesc,*((ushort*)(m_voBGDesc[nSampleIdx].data+ushrt_idx)));
				if(nDescDist>m_nDescDistThreshold)
					goto failedcheck1ch;
				nGoodSamplesCount++;
				failedcheck1ch:
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.data[uchar_idx] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					ushort* bg_desc_ptr = ((ushort*)(m_voBGDesc[s_rand].data+ushrt_idx));
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					*bg_desc_ptr = nCurrInputDesc;
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,*bg_desc_ptr);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,*bg_desc_ptr);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
					m_voBGImg[s_rand].data[uchar_idx] = oInputImg.data[uchar_idx];
				}
				if((rand()%nLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					int s_rand = rand()%m_nBGSamples;
					ushort& nRandInputDesc = m_voBGDesc[s_rand].at<ushort>(y_rand,x_rand);
#if BGSVIBELBSP_USE_SELF_DIFFUSION
#if BGSLBSP_MODEL_INTER_LBSP
					CV_DbgAssert(nSampleIdx<m_nBGSamples);
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x_rand,y_rand,m_fLBSPThreshold,nRandInputDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x_rand,y_rand,m_nLBSPThreshold,nRandInputDesc);
#else //!BGSLBSP_MODEL_INTER_LBSP
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,nRandInputDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,nRandInputDesc);
#endif //!BGSLBSP_MODEL_INTER_LBSP
					m_voBGImg[s_rand].at<uchar>(y_rand,x_rand) = oInputImg.at<uchar>(y_rand,x_rand);
#else //!BGSVIBELBSP_USE_SELF_DIFFUSION
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					nRandInputDesc = nCurrInputDesc;
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeGrayscaleRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,nRandInputDesc);
					else
						LBSP::computeGrayscaleAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,nRandInputDesc);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
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
#if BGSLBSP_USE_SC_THRS_VALIDATION
		const int nCurrSCDescDistThreshold = (int)(m_nDescDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const int nCurrSCColorDistThreshold = (int)(m_nColorDistThreshold*BGSLBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR);
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
		const int desc_row_step = m_voBGDesc[0].step.p[0];
		const int img_row_step = m_voBGImg[0].step.p[0];
		for(int k=0; k<nKeyPoints; ++k) {
			const int x = (int)m_voKeyPoints[k].pt.x;
			const int y = (int)m_voKeyPoints[k].pt.y;
			const int uchar_idx = oInputImg.cols*y + x;
			const int rgbimg_idx = uchar_idx*3;
			const int descimg_idx = uchar_idx*6;
			int nGoodSamplesCount=0, nSampleIdx=0;
			ushort anCurrInputDesc[3];
#if !BGSLBSP_EXTRACT_INTER_LBSP
			if(m_bLBSPUsingRelThreshold)
				LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,anCurrInputDesc);
			else
				LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,anCurrInputDesc);
#endif //!BGSLBSP_EXTRACT_INTER_LBSP
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* bg_desc_ptr = (ushort*)(m_voBGDesc[nSampleIdx].data+descimg_idx);
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
				const uchar* bg_img_ptr = m_voBGImg[nSampleIdx].data+rgbimg_idx;
				const uchar* input_img_ptr = oInputImg.data+rgbimg_idx;
				int nTotColorDist = 0;
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
				int nTotDescDist = 0;
				for(int c=0;c<3; ++c) {
#if BGSVIBELBSP_USE_COLOR_COMPLEMENT
					const int nColorDist = absdiff_uchar(input_img_ptr[c],bg_img_ptr[c]);
#if BGSLBSP_USE_SC_THRS_VALIDATION
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
#endif //BGSLBSP_USE_SC_THRS_VALIDATION
#endif //BGSVIBELBSP_USE_COLOR_COMPLEMENT
#if BGSLBSP_EXTRACT_INTER_LBSP
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeSingleRGBRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_fLBSPThreshold,anCurrInputDesc[c]);
					else
						LBSP::computeSingleRGBAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x,y,c,m_nLBSPThreshold,anCurrInputDesc[c]);
#endif //BGSLBSP_EXTRACT_INTER_LBSP
					const int nDescDist = hdist_ushort_8bitLUT(anCurrInputDesc[c],bg_desc_ptr[c]);
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
				oFGMask.data[uchar_idx] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					ushort* bg_desc_ptr = ((ushort*)(m_voBGDesc[s_rand].data+descimg_idx));
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					for(int c=0; c<3; ++c)
						bg_desc_ptr[c] = anCurrInputDesc[c];
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bg_desc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bg_desc_ptr);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
					for(int c=0; c<3; ++c)
						*(m_voBGImg[s_rand].data+rgbimg_idx+c) = *(oInputImg.data+rgbimg_idx+c);
					CV_DbgAssert(m_voBGImg[s_rand].at<cv::Vec3b>(y,x)==oInputImg.at<cv::Vec3b>(y,x));
				}
				if((rand()%nLearningRate)==0) {
					int s_rand = rand()%m_nBGSamples;
					int x_rand,y_rand;
					getRandNeighborPosition(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					ushort* bg_desc_ptr = ((ushort*)(m_voBGDesc[s_rand].data + desc_row_step*y_rand + 6*x_rand));
#if BGSVIBELBSP_USE_SELF_DIFFUSION
#if BGSLBSP_MODEL_INTER_LBSP
					CV_DbgAssert(nSampleIdx<m_nBGSamples);
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,m_voBGImg[nSampleIdx],x_rand,y_rand,m_fLBSPThreshold,bg_desc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,m_voBGImg[nSampleIdx],x_rand,y_rand,m_nLBSPThreshold,bg_desc_ptr);
#else //!BGSLBSP_MODEL_INTER_LBSP
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_fLBSPThreshold,bg_desc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x_rand,y_rand,m_nLBSPThreshold,bg_desc_ptr);
#endif //!BGSLBSP_MODEL_INTER_LBSP
					for(int c=0; c<3; ++c)
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oInputImg.data + img_row_step*y_rand + 3*x_rand + c);
#else //!BGSVIBELBSP_USE_SELF_DIFFUSION
#if (!BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP) || (BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP)
					for(int c=0; c<3; ++c)
						bg_desc_ptr[c] = anCurrInputDesc[c];
#elif !BGSLBSP_MODEL_INTER_LBSP && BGSLBSP_EXTRACT_INTER_LBSP
					if(m_bLBSPUsingRelThreshold)
						LBSP::computeRGBRelativeDescriptor(oInputImg,cv::Mat(),x,y,m_fLBSPThreshold,bg_desc_ptr);
					else
						LBSP::computeRGBAbsoluteDescriptor(oInputImg,cv::Mat(),x,y,m_nLBSPThreshold,bg_desc_ptr);
#else //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
#error "Illogical model desc <-> extracted desc association."
#endif //BGSLBSP_MODEL_INTER_LBSP && !BGSLBSP_EXTRACT_INTER_LBSP
					for(int c=0; c<3; ++c)
						*(m_voBGImg[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = *(oInputImg.data+rgbimg_idx+c);
#endif //!BGSVIBELBSP_USE_SELF_DIFFUSION
				}
			}
		}
	}
	cv::medianBlur(oFGMask,oFGMask,9); // give user access to mblur params... @@@@@
}

void BackgroundSubtractorViBeLBSP::getBackgroundImage(cv::OutputArray backgroundImage) const {
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
