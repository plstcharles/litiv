#include "DetectChange.h"
#include "LBSP.h"
#include <iostream>

DetectChange::DetectChange(int threshold, int streak)
	:	 m_nThreshold(threshold)
		,m_nStreak(streak)
		,m_nUntrainedElems(-1)
		,m_nTotElems(-1) {
	CV_Assert(threshold>0 && m_nStreak>0 && m_nStreak<UCHAR_MAX);
}

void DetectChange::setBGModel(const cv::Mat &descImg, const cv::Mat &bgImg) {
	m_oBGDesc = descImg.clone();
	m_oBGDesc2 = descImg.clone();
	m_oBGImg = bgImg.clone();
	m_oBGImg2 = bgImg.clone();
	m_oFreq = cv::Mat::zeros(descImg.rows,descImg.cols,CV_8UC1);
	m_oFGFreq = cv::Mat::zeros(descImg.rows,descImg.cols,CV_8UC1);
	m_oStable = cv::Mat::zeros(descImg.rows,descImg.cols,CV_8UC1);
	m_nUntrainedElems = descImg.rows*descImg.cols;
	m_nTotElems = 0;
}

void DetectChange::compute_Hamming(const cv::Mat &descImg, cv::Mat &fgMask) const {
	CV_Assert(descImg.type()==m_oBGDesc.type() && descImg.size==m_oBGDesc.size && descImg.channels()==m_oBGDesc.channels());
	fgMask = cv::Mat::zeros(descImg.rows,descImg.cols,CV_8UC1);
	std::vector<cv::Mat> planes1, planes2;
	cv::split(descImg,planes1);
	cv::split(m_oBGDesc,planes2);
	for(int i=0; i<descImg.rows; ++i) {
		for(int j=0; j<descImg.cols; ++j) {
			int total=0;
			for(int n=0;n<descImg.channels();++n)
				total += cv::normHamming(&planes1[n].data[planes1[n].step[0]*i+planes1[n].step[1]*j],
										&planes2[n].data[planes2[n].step[0]*i+planes2[n].step[1]*j],
										LBSP::LBSP_DESC_SIZE);
			if(total>=m_nThreshold)
				fgMask.at<uchar>(i,j) = UCHAR_MAX;
		}
	}
}

void DetectChange::compute_L1(const cv::Mat &desc, cv::Mat &fgMask) const {
	CV_Assert(desc.type()==m_oBGDesc.type() && desc.size==m_oBGDesc.size && desc.channels()==m_oBGDesc.channels());
	fgMask = cv::Mat::zeros(desc.rows,desc.cols,CV_8UC1);
	std::vector<cv::Mat> planes1, planes2;
	split(desc,planes1);
	split(m_oBGDesc,planes2);
	for(int i=0; i<desc.rows; ++i) {
		for(int j=0; j<desc.cols; ++j) {
			int total=0;
			for(int n=0;n<desc.channels();++n)
				total += LBSP::absdiff(planes1[n].data[planes1[n].step[0]*i+planes1[n].step[1]*j],
										planes2[n].data[planes2[n].step[0]*i+planes2[n].step[1]*j]);
			if(total>=m_nThreshold)
				fgMask.at<uchar>(i,j) = UCHAR_MAX;
		}
	}
}

bool DetectChange::trainandcompute(const cv::Mat &descImg1, const cv::Mat &descImg2, const cv::Mat &descImg3, const cv::Mat &origImg, cv::Mat &fgMask) {
	CV_Assert(descImg1.type()==m_oBGDesc.type() && descImg1.size==m_oBGDesc.size && descImg1.channels()==m_oBGDesc.channels());
	CV_Assert(descImg2.type()==m_oBGDesc.type() && descImg2.size==m_oBGDesc.size && descImg2.channels()==m_oBGDesc.channels());
	CV_Assert(descImg3.type()==m_oBGDesc.type() && descImg3.size==m_oBGDesc.size && descImg3.channels()==m_oBGDesc.channels());
	CV_Assert(origImg.type()==CV_8UC1 || origImg.type()==CV_8UC3);
	CV_Assert(origImg.type()==m_oBGImg.type() && origImg.size==m_oBGImg.size && origImg.channels()==m_oBGImg.channels());
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
	for(int i=0; i<descImg1.rows; ++i) {
		for(int j=0; j<descImg1.cols; ++j) {
			int total1=0, total2=0;
			for(int n=0;n<descImg1.channels();++n) {
				total1 += cv::normHamming(&planesDESC[n].data[planesDESC[n].step[0]*i+planesDESC[n].step[1]*j],
											&planesBGM[n].data[planesBGM[n].step[0]*i+planesBGM[n].step[1]*j],
											LBSP::LBSP_DESC_SIZE);
				total2 += cv::normHamming(&planesDESC2[n].data[planesDESC2[n].step[0]*i+planesDESC2[n].step[1]*j],
											&planesBGM2[n].data[planesBGM2[n].step[0]*i+planesBGM2[n].step[1]*j],
											LBSP::LBSP_DESC_SIZE);
			}
			if(total1>=m_nThreshold) {
				fgMask.at<uchar>(i,j) = UCHAR_MAX;
				if(m_oFGFreq.at<uchar>(i,j)<UCHAR_MAX)
					m_oFGFreq.at<uchar>(i,j)++;
				if(m_oStable.at<uchar>(i,j)!=1 && m_oFGFreq.at<uchar>(i,j)>=m_nStreak) {
					for(int n=0;n<descImg1.channels();++n) {
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
				if(total1<m_nThreshold) {
					if(m_oFreq.at<uchar>(i,j)<UCHAR_MAX)
						m_oFreq.at<uchar>(i,j)++;
					if(m_oFreq.at<uchar>(i,j)>=m_nStreak) {
						for(int n=0;n<descImg1.channels();++n) {
							//planesBGM[0].at<unsigned short>(i,j)=planesBGM2[0].at<unsigned short>(i,j);
							//planesBGM[1].at<unsigned short>(i,j)=planesBGM2[1].at<unsigned short>(i,j);
							//planesBGM[2].at<unsigned short>(i,j)=planesBGM2[2].at<unsigned short>(i,j);
							//planesBGI2[0].at<uchar>(i,j)=255;//planesIM[0].at<uchar>(i,j);
							//planesBGI2[1].at<uchar>(i,j)=0;//planesIM[1].at<uchar>(i,j);
							//planesBGI2[2].at<uchar>(i,j)=0;//planesIM[2].at<uchar>(i,j);
							// == ? @@@@
							//planesBGM[n].at<unsigned short>(i,j)=planesBGM2[n].at<unsigned short>(i,j);
						}
						m_oStable.at<uchar>(i,j) = 1;
						m_nTotElems++;
					}
				}
				// Case 2: labeled as foreground
				else { //total1>=(int)BGThreshold_.at<uchar>(i,j)
					if(total2<m_nThreshold)	{
						if(m_oFreq.at<uchar>(i,j)<UCHAR_MAX)
							m_oFreq.at<uchar>(i,j)++;
						if(m_oFreq.at<uchar>(i,j)>=m_nStreak) {
							for(int n=0;n<descImg1.channels();++n) {
								planesBGM[n].at<unsigned short>(i,j)=planesBGM2[n].at<unsigned short>(i,j);
								planesBGI[n].at<uchar>(i,j)=planesBGI2[n].at<uchar>(i,j);
							}
							m_oStable.at<uchar>(i,j) = 1;
							m_nTotElems++;
						}
					}
					else {
						m_oFreq.at<uchar>(i,j) = 0;
						for(int n=0;n<descImg1.channels();++n) {
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
}

const cv::Mat& DetectChange::getBGImage() const
{
	return m_oBGImg;
}

const cv::Mat& DetectChange::getBGImage2() const
{
	return m_oBGImg2;
}

const cv::Mat& DetectChange::getBGDesc() const
{
	return m_oBGDesc;
}

const cv::Mat& DetectChange::getBGDesc2() const
{
	return m_oBGDesc2;
}
