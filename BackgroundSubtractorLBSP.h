#pragma once

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include "LBSP.h"

//! defines the default value for BackgroundSubtractorLBSP::m_nFGThreshold
#define BGSLBSP_DEFAULT_FG_THRESHOLD (5)
//! defines the default difference factor between BackgroundSubtractorLBSP::m_nFGThreshold and BackgroundSubtractorLBSP::m_nFGSCThreshold
#define BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR (1.6)
//! defines the default value for BackgroundSubtractorLBSP::m_nFGSCThreshold
#define BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD (BGSLBSP_DEFAULT_FG_THRESHOLD*BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR)
//! defines the default value for BackgroundSubtractorLBSP::m_nBGSamples
#define BGSLBSP_DEFAULT_BG_SAMPLES (20)
//! defines the default value for BackgroundSubtractorLBSP::m_nRequiredBGSamples
#define BGSLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorLBSP::operator()
#define BGSLBSP_DEFAULT_LEARNING_RATE (5)


/*!
	Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters
	are adjusted automatically). When processing grayscale images, only m_nFGThreshold is
	used to determine if a pixel is foreground/background.

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorLBSP : public cv::BackgroundSubtractor {
public:
	//! default constructor (also using default LBSP descriptor extractor params)
	BackgroundSubtractorLBSP();
	//! full constructor based on the absolute LBSP extractor, with algorithm parameters passed as arguments
	BackgroundSubtractorLBSP(	int nDescThreshold,
								int nFGThreshold=BGSLBSP_DEFAULT_FG_THRESHOLD,
								int nFGSCThreshold=BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD,
								int nBGSamples=BGSLBSP_DEFAULT_BG_SAMPLES,
								int nRequiredBGSamples=BGSLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! full constructor based on the relative LBSP extractor, with algorithm parameters passed as arguments
	BackgroundSubtractorLBSP(	float fDescThreshold,
								int nFGThreshold=BGSLBSP_DEFAULT_FG_THRESHOLD,
								int nFGSCThreshold=BGSLBSP_DEFAULT_FG_SINGLECHANNEL_THRESHOLD,
								int nBGSamples=BGSLBSP_DEFAULT_BG_SAMPLES,
								int nRequiredBGSamples=BGSLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg);
	//! primary model update function; the learning param is reinterpreted as an integer and should be >= 0 (smaller values == faster adaptation)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=32);
	//! @@@@@@@@@@@@ ????
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background image
	cv::Mat getCurrentBGImage() const;
	//! returns a copy of the latest background descriptors image
	cv::Mat getCurrentBGDescriptors() const;
	//! returns the keypoints list used for descriptor extraction (note: by default, they are generated from the DenseFeatureDetector class, but the border points are removed)
	std::vector<cv::KeyPoint> getBGKeyPoints() const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border keypoints, and reinits the model's samples buffers)
	void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);

private:
	//! number of different samples per pixel/block to be taken from input frames to build the background model
	const int m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background'
	const int m_nRequiredBGSamples;
	//! background model pixel samples used as references for LBSP computations
	std::vector<cv::Mat> m_voBGImg;
	//! background model descriptors samples used as references for change detection (tied to m_voKeyPoints but shaped like the input frames)
	std::vector<cv::Mat> m_voBGDesc;
	//! background model keypoints used for LBSP descriptor extraction (specific to the input image size)
	std::vector<cv::KeyPoint> m_voKeyPoints;
	//! input image size
	cv::Size m_oImgSize;
	//! input image channel size
	int m_nImgChannels;
	//! input image type
	int m_nImgType;
	//! absolute per-channel hamming distance threshold, used to calculate m_nCurrFGThreshold based on the number of channels in the input image
	const int m_nFGThreshold;
	//! absolute single-channel hamming distance override threshold, used for change detection along with m_nCurrFGThreshold only when m_nImgChannels>1
	const int m_nFGSCThreshold;
	//! absolute total hamming distance threshold, used for change detection
	int m_nCurrFGThreshold;
	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
	//! LBSP feature extractor
	LBSP m_oExtractor;
};

