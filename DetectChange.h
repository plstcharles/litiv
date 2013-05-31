#pragma once

#include <opencv2/opencv.hpp>

class DetectChange
{
public:
	DetectChange(int threshold=1, int streak=30);
	void setBGModel(const cv::Mat& descImg, const cv::Mat& bgImg);
	void compute(const cv::Mat &descImg, cv::Mat &fgMask) const;
	bool trainandcompute(const cv::Mat &descImg1, const cv::Mat &descImg2, const cv::Mat &descImg3, const cv::Mat &origImg, cv::Mat &fgMask);

	const cv::Mat& getBGImage() const;
	const cv::Mat& getBGImage2() const;
	const cv::Mat& getBGDesc() const;
	const cv::Mat& getBGDesc2() const;

private:
	cv::Mat m_oBGDesc, m_oBGDesc2;
	cv::Mat m_oFreq;
	cv::Mat m_oFGFreq;
	cv::Mat m_oStable;
	cv::Mat m_oBGImg, m_oBGImg2;
	const int m_nTotalThreshold;
	const int m_nSingleChannelThreshold;
	const int m_nStreak;
	int m_nUntrainedElems;
	int m_nTotElems;
};

