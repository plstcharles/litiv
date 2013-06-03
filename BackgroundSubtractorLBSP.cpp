#include "BackgroundSubtractorLBSP.h"
#include "LBSP.h"
#include "HammingDist.h"
#include <iostream>

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP()
	:	 m_nFGThreshold(BGSLBSP_DEFAULT_FG_THRESHOLD)
		,m_nFGSCThreshold(BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD)
		,m_bInitialized(false)
		,m_oExtractor(LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(	int nDescThreshold,
													int nFGThreshold,
													int nFGSCThreshold
													)
	:	 m_nFGThreshold(nFGThreshold)
		,m_nFGSCThreshold(nFGSCThreshold)
		,m_bInitialized(false)
		,m_oExtractor(nDescThreshold) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
}

BackgroundSubtractorLBSP::BackgroundSubtractorLBSP(	float fDescThreshold,
													int nFGThreshold,
													int nFGSCThreshold
													)
	:	 m_nFGThreshold(nFGThreshold)
		,m_nFGSCThreshold(nFGSCThreshold)
		,m_bInitialized(false)
		,m_oExtractor(fDescThreshold) {
	CV_Assert(m_nFGThreshold>0 && m_nFGSCThreshold>0);
}

BackgroundSubtractorLBSP::~BackgroundSubtractorLBSP() {}

void BackgroundSubtractorLBSP::initialize(const cv::Size& oFrameSize, int nFrameType) {
	CV_Assert(oFrameSize.width>0 && oFrameSize.height>0);
	CV_Assert(nFrameType==CV_8UC1 || nFrameType==CV_8UC3);
	m_oImgSize = oFrameSize;
	m_nImgType = nFrameType;
	m_nImgChannels = CV_MAT_CN(nFrameType);
	m_oBGImg.create(m_oImgSize,m_nImgType);
	m_nCurrFGThreshold = m_nFGThreshold*m_nImgChannels;
	cv::DenseFeatureDetector oKPDDetector(	1.0f,	// init feature scale
											1,		// feature scale levels
											1.0f,	// feature scale mult
											1,		// init xy step
											0,		// init img bound
											true,	// var xy step with scale
											false	// var img bound with scale
											);		// note: the extractor will remove keypoints that are out of bounds itself
	if(m_voBGKeyPoints.capacity()<(size_t)(m_oImgSize.width*m_oImgSize.height))
		m_voBGKeyPoints.reserve(m_oImgSize.width*m_oImgSize.height);
	oKPDDetector.detect(cv::Mat(m_oImgSize,m_nImgType), m_voBGKeyPoints);
	LBSP::validateKeyPoints(m_voBGKeyPoints,m_oImgSize);
	CV_Assert(!m_voBGKeyPoints.empty());
	CV_Assert(LBSP::DESC_SIZE==2);
	m_oBGDesc.create(m_voBGKeyPoints.size(),1,CV_16UC(m_nImgChannels));
	m_bInitialized = true;
	gotbgmodel=false;
}

void BackgroundSubtractorLBSP::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate) {
	CV_DbgAssert(m_bInitialized);
	cv::Mat oInputImg = _image.getMat(), oInputDesc;
	CV_DbgAssert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
	m_oExtractor.compute(oInputImg,m_voBGKeyPoints,oInputDesc);
	CV_DbgAssert(oInputDesc.size()==m_oBGDesc.size() && oInputDesc.type()==m_oBGDesc.type());
	_fgmask.create(m_oImgSize,CV_8UC1);
	cv::Mat oFGMask = _fgmask.getMat();
	oFGMask = cv::Scalar_<uchar>(0);

	// @@@@@@@@@
	if(!gotbgmodel) {
		gotbgmodel = true;
		oInputDesc.copyTo(m_oBGDesc);
		oInputImg.copyTo(m_oBGImg);
		m_oExtractor.setReference(m_oBGImg);
	}

	const int nKeyPoints = (int)m_voBGKeyPoints.size();
	if(m_nImgChannels==1) {
		CV_DbgAssert(oInputDesc.step.p[0]==m_oBGDesc.step.p[0] && oInputDesc.step.p[1]==m_oBGDesc.step.p[1] && oInputDesc.step.p[0]==oInputDesc.step.p[1] && oInputDesc.step.p[1]==2);
		for(int k=0; k<nKeyPoints; ++k) {
			const int idx = oInputDesc.step.p[0]*k; // should be the same steps for both mats... (asserted above)
			if(hdist_ushort_8bitLUT(*((unsigned short*)(oInputDesc.data+idx)),*((unsigned short*)(m_oBGDesc.data+idx)))>m_nCurrFGThreshold)
				oFGMask.at<uchar>(m_voBGKeyPoints[k].pt) = UCHAR_MAX;
		}
	}
	else { //m_nImgChannels==3
		CV_DbgAssert(oInputDesc.step.p[0]==m_oBGDesc.step.p[0] && oInputDesc.step.p[1]==m_oBGDesc.step.p[1] && oInputDesc.step.p[0]==oInputDesc.step.p[1] && oInputDesc.step.p[1]==6);
		int hdist[3];
		for(int k=0; k<nKeyPoints; ++k) {
			const int idx = oInputDesc.step.p[0]*k; // should be the same steps for both mats... (asserted above)
			for(int n=0;n<3; ++n) {
				hdist[n] = hdist_ushort_8bitLUT(((unsigned short*)(oInputDesc.data+idx))[n],((unsigned short*)(m_oBGDesc.data+idx))[n]);
				if(hdist[n]>m_nFGSCThreshold)
					goto foreground;
			}
			if(hdist[0]+hdist[1]+hdist[2]>m_nCurrFGThreshold)
				goto foreground;
			continue;
			foreground:
			oFGMask.at<uchar>(m_voBGKeyPoints[k].pt) = UCHAR_MAX;
		}
	}
}

/*bool BackgroundSubtractorLBSP::trainandcompute(const cv::Mat &descImg1, const cv::Mat &descImg2, const cv::Mat &descImg3, const cv::Mat &origImg, cv::Mat &fgMask) {
	CV_DbgAssert(descImg1.type()==m_oBGDesc.type() && descImg1.size==m_oBGDesc.size && descImg1.channels()==m_oBGDesc.channels());
	CV_DbgAssert(descImg2.type()==m_oBGDesc.type() && descImg2.size==m_oBGDesc.size && descImg2.channels()==m_oBGDesc.channels());
	CV_DbgAssert(descImg3.type()==m_oBGDesc.type() && descImg3.size==m_oBGDesc.size && descImg3.channels()==m_oBGDesc.channels());
	CV_DbgAssert(origImg.type()==CV_8UC1 || origImg.type()==CV_8UC3);
	CV_DbgAssert(origImg.type()==m_oBGImg.type() && origImg.size==m_oBGImg.size && origImg.channels()==m_oBGImg.channels());
	fgMask = cv::Mat::zeros(descImg1.rows,descImg1.cols,CV_8UC1);
	std::vector<cv::Mat> planesDESC, planesDESC2, planesDESC3, planesBGM2, planesBGM,planesBGI,planesBGI2,planesIM;
	split(descImg1,planesDESC);
	split(descImg2,planesDESC2);
	split(descImg3,planesDESC3);
	split(m_oBGDesc,planesBGM);
	split(m_oBGDesc2,planesBGM2);
	split(m_oBGImg,planesBGI);
	split(m_oBGImg2,planesBGI2);
	split(origImg, planesIM);
	int diff, total;
	int nChannels = descImg1.channels();
	for(int i=0; i<descImg1.rows; ++i) {
		for(int j=0; j<descImg1.cols; ++j) {
			if(nChannels>1) {
				total = 0;
				for(int n=0; n<nChannels; ++n) {
					diff = cv::normHamming(&planesDESC[n].data[planesDESC[n].step[0]*i+planesDESC[n].step[1]*j],
											&planesBGM[n].data[planesBGM[n].step[0]*i+planesBGM[n].step[1]*j],
											LBSP::LBSP_DESC_SIZE);
					total += (diff>=m_nFGSCThreshold)?m_nFGThreshold:diff; // @@@@ guarantee bust when single channel is enough; NOT OPTIMIZED...
				}
			}
			else {
				total = 3*cv::normHamming(&planesDESC[0].data[planesDESC[0].step[0]*i+planesDESC[0].step[1]*j],
										&planesBGM[0].data[planesBGM[0].step[0]*i+planesBGM[0].step[1]*j],
										LBSP::LBSP_DESC_SIZE);
			}
			if(total>=m_nFGThreshold) {
				fgMask.at<uchar>(i,j) = UCHAR_MAX;
				if(m_oFGFreq.at<uchar>(i,j)<UCHAR_MAX)
					m_oFGFreq.at<uchar>(i,j)++;
				if(m_oStable.at<uchar>(i,j)!=1 && m_oFGFreq.at<uchar>(i,j)>m_nStreak) {
					for(int n=0; n<nChannels; ++n) {
						planesBGM[n].at<unsigned short>(i,j)=planesDESC3[n].at<unsigned short>(i,j);
						planesBGI[n].at<uchar>(i,j)=planesIM[n].at<uchar>(i,j);
					}
					m_oStable.at<uchar>(i,j) = 1;
					m_nTotElems++;
				}
			}
			else {
				m_oFGFreq.at<uchar>(i,j) = 0;
			}

			// Learning stable code
			if(m_oStable.at<uchar>(i,j)!=1) {
				// Case 1: labeled as background
				if(total<m_nFGThreshold) {
					if(m_oTrainingFreq.at<uchar>(i,j)<UCHAR_MAX)
						m_oTrainingFreq.at<uchar>(i,j)++;
					if(m_oTrainingFreq.at<uchar>(i,j)>=m_nTrainingStreak) {
						//for(int n=0; n<nChannels; ++n) {
						//	planesBGM[0].at<unsigned short>(i,j)=planesBGM2[0].at<unsigned short>(i,j);
						//	planesBGM[1].at<unsigned short>(i,j)=planesBGM2[1].at<unsigned short>(i,j);
						//	planesBGM[2].at<unsigned short>(i,j)=planesBGM2[2].at<unsigned short>(i,j);
						//	planesBGI2[0].at<uchar>(i,j)=255;//planesIM[0].at<uchar>(i,j);
						//	planesBGI2[1].at<uchar>(i,j)=0;//planesIM[1].at<uchar>(i,j);
						//	planesBGI2[2].at<uchar>(i,j)=0;//planesIM[2].at<uchar>(i,j);
						//	 == ? @@@@
						//	planesBGM[n].at<unsigned short>(i,j)=planesBGM2[n].at<unsigned short>(i,j);
						//}
						m_oStable.at<uchar>(i,j) = 1;
						m_nTotElems++;
					}
				}
				// Case 2: labeled as foreground
				else { //total>=m_nTotalThreshold
					if(nChannels>1) {
						total = 0;
						for(int n=0; n<nChannels; ++n) {
							diff = cv::normHamming(&planesDESC2[n].data[planesDESC2[n].step[0]*i+planesDESC2[n].step[1]*j],
													&planesBGM2[n].data[planesBGM2[n].step[0]*i+planesBGM2[n].step[1]*j],
													LBSP::LBSP_DESC_SIZE);
							total += (diff>=m_nFGSCThreshold)?m_nFGThreshold:diff; // @@@@ guarantee bust when single channel is enough; NOT OPTIMIZED...
						}
					}
					else {
						total = 3*cv::normHamming(&planesDESC2[0].data[planesDESC2[0].step[0]*i+planesDESC2[0].step[1]*j],
												&planesBGM2[0].data[planesBGM2[0].step[0]*i+planesBGM2[0].step[1]*j],
												LBSP::LBSP_DESC_SIZE);
					}
					if(total<m_nFGThreshold)	{
						if(m_oTrainingFreq.at<uchar>(i,j)<UCHAR_MAX)
							m_oTrainingFreq.at<uchar>(i,j)++;
						if(m_oTrainingFreq.at<uchar>(i,j)>=m_nTrainingStreak) {
							for(int n=0; n<nChannels; ++n) {
								planesBGM[n].at<unsigned short>(i,j)=planesBGM2[n].at<unsigned short>(i,j);
								planesBGI[n].at<uchar>(i,j)=planesBGI2[n].at<uchar>(i,j);
							}
							m_oStable.at<uchar>(i,j) = 1;
							m_nTotElems++;
						}
					}
					else {
						m_oTrainingFreq.at<uchar>(i,j) = 0;
						for(int n=0; n<nChannels; ++n) {
							planesBGM2[n].at<unsigned short>(i,j)=planesDESC3[n].at<unsigned short>(i,j);
							planesBGI2[n].at<uchar>(i,j)=planesIM[n].at<uchar>(i,j);
						}
					}
				}
			}
		}
	}
	merge(planesBGM, m_oBGDesc);
	merge(planesBGM2, m_oBGDesc2);
	merge(planesBGI, m_oBGImg);
	merge(planesBGI2, m_oBGImg2);
	return m_nTotElems!=m_nUntrainedElems;
}*/

cv::AlgorithmInfo* BackgroundSubtractorLBSP::info() const {
	CV_Assert(false); // NOT IMPL @@@@@
	return NULL;
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGImage() const {
	return m_oBGImg.clone();
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGDescriptors() const {
	return m_oBGDesc.clone();
}

cv::Mat BackgroundSubtractorLBSP::getCurrentBGDescriptorsImage() const {
	cv::Mat oCurrBGDescImg;
	LBSP::recreateDescImage(m_oImgSize,m_voBGKeyPoints,m_oBGDesc,oCurrBGDescImg);
	return oCurrBGDescImg;
}

std::vector<cv::KeyPoint> BackgroundSubtractorLBSP::getBGKeyPoints() const {
	return m_voBGKeyPoints;
}

void BackgroundSubtractorLBSP::setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints) {
	m_oExtractor.validateKeyPoints(keypoints,m_oImgSize);
	CV_Assert(!keypoints.empty());
	m_voBGKeyPoints = keypoints;
}
