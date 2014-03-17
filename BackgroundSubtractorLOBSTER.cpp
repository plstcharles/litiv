#include "BackgroundSubtractorLOBSTER.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

BackgroundSubtractorLOBSTER::BackgroundSubtractorLOBSTER(	 float fRelLBSPThreshold
															,size_t nLBSPThresholdOffset
															,size_t nDescDistThreshold
															,size_t nColorDistThreshold
															,size_t nBGSamples
															,size_t nRequiredBGSamples)
	:	 BackgroundSubtractorLBSP(fRelLBSPThreshold,nDescDistThreshold,nLBSPThresholdOffset)
		,m_nBGSamples(nBGSamples)
		,m_nRequiredBGSamples(nRequiredBGSamples)
		,m_nColorDistThreshold(nColorDistThreshold) {
	CV_Assert(m_nRequiredBGSamples<=m_nBGSamples);
}

BackgroundSubtractorLOBSTER::~BackgroundSubtractorLOBSTER() {}

void BackgroundSubtractorLOBSTER::initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints) {
	CV_Assert(!oInitImg.empty() && oInitImg.cols>0 && oInitImg.rows>0);
	CV_Assert(oInitImg.type()==CV_8UC1 || oInitImg.type()==CV_8UC3);
	if(oInitImg.type()==CV_8UC3) {
		std::vector<cv::Mat> voInitImgChannels;
		cv::split(oInitImg,voInitImgChannels);
		bool eq = std::equal(voInitImgChannels[0].begin<uchar>(), voInitImgChannels[0].end<uchar>(), voInitImgChannels[1].begin<uchar>())
				&& std::equal(voInitImgChannels[1].begin<uchar>(), voInitImgChannels[1].end<uchar>(), voInitImgChannels[2].begin<uchar>());
		if(eq)
			std::cout << std::endl << "\tBackgroundSubtractorLOBSTER : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance." << std::endl;
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
	cv::Mat oInitDesc(m_oImgSize,CV_16UC((int)m_nImgChannels),cv::Scalar_<ushort>::all(0));
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
			CV_DbgAssert(oInitImg.step.p[0]==(size_t)oInitImg.cols && oInitImg.step.p[1]==1);
			const size_t idx_color = oInitImg.cols*y_orig + x_orig;
			CV_DbgAssert(oInitDesc.step.p[0]==oInitImg.step.p[0]*2 && oInitDesc.step.p[1]==oInitImg.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[idx_color],x_orig,y_orig,m_anLBSPThreshold_8bitLUT[oInitImg.data[idx_color]],*((ushort*)(oInitDesc.data+idx_desc)));
		}
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			for(size_t s=0; s<m_nBGSamples; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				m_voBGColorSamples[s].at<uchar>(y_orig,x_orig) = oInitImg.at<uchar>(y_sample,x_sample);
				m_voBGDescSamples[s].at<ushort>(y_orig,x_orig) = oInitDesc.at<ushort>(y_sample,x_sample);
			}
		}
	}
	else { //m_nImgChannels==3
		for(size_t t=0; t<=UCHAR_MAX; ++t)
			m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset);
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(oInitImg.step.p[0]==(size_t)oInitImg.cols*3 && oInitImg.step.p[1]==3);
			const size_t idx_color = 3*(oInitImg.cols*y_orig + x_orig);
			CV_DbgAssert(oInitDesc.step.p[0]==oInitImg.step.p[0]*2 && oInitDesc.step.p[1]==oInitImg.step.p[1]*2);
			const size_t idx_desc = idx_color*2;
			for(size_t c=0; c<3; ++c) {
				const uchar nCurrBGInitColor = oInitImg.data[idx_color+c];
				LBSP::computeSingleRGBDescriptor(oInitImg,nCurrBGInitColor,x_orig,y_orig,c,m_anLBSPThreshold_8bitLUT[nCurrBGInitColor],((ushort*)(oInitDesc.data+idx_desc))[c]);
			}
		}
		for(size_t k=0; k<nKeyPoints; ++k) {
			const int y_orig = (int)m_voKeyPoints[k].pt.y;
			const int x_orig = (int)m_voKeyPoints[k].pt.x;
			CV_DbgAssert(oInitImg.step.p[0]==(size_t)oInitImg.cols*3 && oInitImg.step.p[1]==3);
			const size_t idx_orig_color = 3*(oInitImg.cols*y_orig + x_orig);
			CV_DbgAssert(oInitDesc.step.p[0]==oInitImg.step.p[0]*2 && oInitDesc.step.p[1]==oInitImg.step.p[1]*2);
			const size_t idx_orig_desc = idx_orig_color*2;
			for(size_t s=0; s<m_nBGSamples; ++s) {
				int y_sample, x_sample;
				getRandSamplePosition(x_sample,y_sample,x_orig,y_orig,LBSP::PATCH_SIZE/2,m_oImgSize);
				const size_t idx_sample_color = oInitImg.step.p[0]*y_sample + oInitImg.step.p[1]*x_sample;
				const size_t idx_sample_desc = oInitDesc.step.p[0]*y_sample + oInitDesc.step.p[1]*x_sample;
				uchar* bg_color_ptr = m_voBGColorSamples[s].data+idx_orig_color;
				ushort* bg_desc_ptr = (ushort*)(m_voBGDescSamples[s].data+idx_orig_desc);
				const uchar* const init_color_ptr = oInitImg.data+idx_sample_color;
				const ushort* const init_desc_ptr = (ushort*)(oInitDesc.data+idx_sample_desc);
				for(size_t c=0; c<3; ++c) {
					bg_color_ptr[c] = init_color_ptr[c];
					bg_desc_ptr[c] = init_desc_ptr[c];
				}
			}
		}
	}
	m_bInitialized = true;
}

void BackgroundSubtractorLOBSTER::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
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
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const uchar nBGColor = m_voBGColorSamples[nSampleIdx].data[idx_uchar];
				{
					const size_t nColorDist = absdiff_uchar(nCurrColor,nBGColor);
					if(nColorDist>m_nColorDistThreshold/2)
						goto failedcheck1ch;
					LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,x,y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInputDesc);
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
					LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nRandInputDesc);
					m_voBGColorSamples[s_rand].data[idx_uchar] = nCurrColor;
				}
				if((rand()%nLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t s_rand = rand()%m_nBGSamples;
					ushort& nRandInputDesc = m_voBGDescSamples[s_rand].at<ushort>(y_rand,x_rand);
					LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,x,y,m_anLBSPThreshold_8bitLUT[nCurrColor],nRandInputDesc);
					m_voBGColorSamples[s_rand].at<uchar>(y_rand,x_rand) = nCurrColor;
				}
			}
		}
	}
	else { //m_nImgChannels==3
		const size_t nCurrDescDistThreshold = m_nDescDistThreshold*3;
		const size_t nCurrColorDistThreshold = m_nColorDistThreshold*3;
		const size_t nCurrSCDescDistThreshold = nCurrDescDistThreshold/2;
		const size_t nCurrSCColorDistThreshold = nCurrColorDistThreshold/2;
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
			while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
				const ushort* const anBGDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+idx_ushrt_rgb);
				const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+idx_uchar_rgb;
				size_t nTotColorDist = 0;
				size_t nTotDescDist = 0;
				for(size_t c=0;c<3; ++c) {
					const size_t nColorDist = absdiff_uchar(anCurrColor[c],anBGColor[c]);
					if(nColorDist>nCurrSCColorDistThreshold)
						goto failedcheck3ch;
					LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],x,y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInputDesc[c]);
					const size_t nDescDist = hdist_ushort_8bitLUT(anCurrInputDesc[c],anBGDesc[c]);
					if(nDescDist>nCurrSCDescDistThreshold)
						goto failedcheck3ch;
					nTotColorDist += nColorDist;
					nTotDescDist += nDescDist;
				}
				if(nTotDescDist<=nCurrDescDistThreshold && nTotColorDist<=nCurrColorDistThreshold)
					nGoodSamplesCount++;
				failedcheck3ch:
				nSampleIdx++;
			}
			if(nGoodSamplesCount<m_nRequiredBGSamples)
				oFGMask.data[idx_uchar] = UCHAR_MAX;
			else {
				if((rand()%nLearningRate)==0) {
					const size_t s_rand = rand()%m_nBGSamples;
					ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[s_rand].data+idx_ushrt_rgb));
					const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
					LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anRandInputDesc);
					for(size_t c=0; c<3; ++c)
						*(m_voBGColorSamples[s_rand].data+idx_uchar_rgb+c) = anCurrColor[c];
				}
				if((rand()%nLearningRate)==0) {
					int x_rand,y_rand;
					getRandNeighborPosition_3x3(x_rand,y_rand,x,y,LBSP::PATCH_SIZE/2,m_oImgSize);
					const size_t s_rand = rand()%m_nBGSamples;
					ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[s_rand].data + desc_row_step*y_rand + 6*x_rand));
					const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
					LBSP::computeRGBDescriptor(oInputImg,anCurrColor,x,y,anCurrIntraLBSPThresholds,anRandInputDesc);
					for(size_t c=0; c<3; ++c)
						*(m_voBGColorSamples[s_rand].data + img_row_step*y_rand + 3*x_rand + c) = anCurrColor[c];
				}
			}
		}
	}
	cv::medianBlur(oFGMask,oFGMask,9);
}

void BackgroundSubtractorLOBSTER::getBackgroundImage(cv::OutputArray backgroundImage) const {
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
