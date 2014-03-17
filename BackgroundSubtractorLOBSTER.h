#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines the default value for BackgroundSubtractorLBSP::m_fRelLBSPThreshold
#define BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.365f)
//! defines the default value for BackgroundSubtractorLBSP::m_nLBSPThresholdOffset
#define BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD (0)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD (4)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nColorDistThreshold
#define BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nBGSamples
#define BGSLOBSTER_DEFAULT_NB_BG_SAMPLES (35)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nRequiredBGSamples
#define BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorLOBSTER::operator()
#define BGSLOBSTER_DEFAULT_LEARNING_RATE (16)

/*!
	LOcal Binary Similarity segmenTER (LOBSTER) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).

	For more details on the different parameters or on the algorithm itself, see P.-L. St-Charles and
	G.-A. Bilodeau, "Improving Background Subtraction using Local Binary Similarity Patterns", in WACV 2014.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorLOBSTER : public BackgroundSubtractorLBSP {
public:
	//! full constructor
	BackgroundSubtractorLOBSTER(float fRelLBSPThreshold=BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
								size_t nLBSPThresholdOffset=BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
								size_t nDescDistThreshold=BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
								size_t nColorDistThreshold=BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
								size_t nBGSamples=BGSLOBSTER_DEFAULT_NB_BG_SAMPLES,
								size_t nRequiredBGSamples=BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorLOBSTER();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSLOBSTER_DEFAULT_LEARNING_RATE);
	//! returns a copy of the latest reconstructed background image
	void getBackgroundImage(cv::OutputArray backgroundImage) const;

protected:
	//! number of different samples per pixel/block to be taken from input frames to build the background model
	const size_t m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background'
	const size_t m_nRequiredBGSamples;
	//! background model pixel intensity samples
	std::vector<cv::Mat> m_voBGColorSamples;
	//! absolute per-channel color distance threshold (based on the provided LBSP threshold)
	const size_t m_nColorDistThreshold;
};

