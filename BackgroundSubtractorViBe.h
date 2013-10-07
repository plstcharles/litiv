#pragma once

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>

//! defines the default value for BackgroundSubtractorViBe::m_nColorDistThreshold
#define BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD (20)
//! defines the default value for BackgroundSubtractorViBe::m_nBGSamples
#define BGSVIBE_DEFAULT_NB_BG_SAMPLES (20)
//! defines the default value for BackgroundSubtractorViBe::m_nRequiredBGSamples
#define BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorViBe::operator() (the 'subsampling' factor in the original ViBe paper)
#define BGSVIBE_DEFAULT_LEARNING_RATE (16)

/*!
	ViBe foreground-background segmentation algorithm (abstract version).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorViBe : public cv::BackgroundSubtractor {
public:
	//! full constructor
	BackgroundSubtractorViBe(	int nColorDistThreshold=BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
								int nBGSamples=BGSVIBE_DEFAULT_NB_BG_SAMPLES,
								int nRequiredBGSamples=BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorViBe();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg)=0;
	//! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBE_DEFAULT_LEARNING_RATE)=0;
	//! @@@@@@@@@@@@ ????
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background image
	void getBackgroundImage(cv::OutputArray backgroundImage) const;

protected:
	//! number of different samples per pixel/block to be taken from input frames to build the background model ('N' in the original ViBe paper)
	const int m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background' ('#_min' in the original ViBe paper)
	const int m_nRequiredBGSamples;
	//! background model pixel intensity samples
	std::vector<cv::Mat> m_voBGImg;
	//! input image size
	cv::Size m_oImgSize;
	//! absolute color distance threshold ('R' or 'radius' in the original ViBe paper)
	const int m_nColorDistThreshold;
	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
};
