#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines whether we should use the selected pixel's characteristics during diffusion or not (1=PBAS-like diffusion, 0=ViBe-like)
#define BGSLOBSTER_USE_SELF_DIFFUSION 0
//! defines whether we should complement the LBSP core component using color or not
#define BGSLOBSTER_USE_COLOR_COMPLEMENT 1

//! defines the default value for BackgroundSubtractorLBSP::m_nLBSPThreshold (if needed)
#define BGSLOBSTER_DEFAULT_LBSP_ABS_SIMILARITY_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorLBSP::m_fLBSPThreshold (if needed)
#define BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.365f)
//! defines the default offset LBSP threshold value (only used along with a relative threshold)
#define BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD (0)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nColorDistThreshold
#define BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD (4)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nBGSamples
#define BGSLOBSTER_DEFAULT_NB_BG_SAMPLES (35)
//! defines the default value for BackgroundSubtractorLOBSTER::m_nRequiredBGSamples
#define BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorLOBSTER::operator()
#define BGSLOBSTER_DEFAULT_LEARNING_RATE (16)
//! defines the internal threshold adjustment factor to use when treating single channel images (based on the assumption that grayscale images have less noise per channel...)
#define BGSLOBSTER_SINGLECHANNEL_THRESHOLD_MODULATION_FACT (0.350f)

/*!
	LOcal Binary Similarity segmenTER (LOBSTER) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).

	For more details on the different parameters or on the algorithm itself, see P.-L. St-Charles and
	G.-A. Bilodeau, "Improving Background Subtraction using Local Binary Similarity Patterns", in WACV 2014.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorLOBSTER : public BackgroundSubtractorLBSP {
public:
	//! full constructor used to intialize an 'absolute' LBSP-based background subtractor
	explicit BackgroundSubtractorLOBSTER(	size_t nLBSPThreshold/*=BGSLOBSTER_DEFAULT_LBSP_ABS_SIMILARITY_THRESHOLD*/,
											size_t nDescDistThreshold=BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
											size_t nColorDistThreshold=BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
											size_t nBGSamples=BGSLOBSTER_DEFAULT_NB_BG_SAMPLES,
											size_t nRequiredBGSamples=BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! full constructor used to intialize a 'relative' LBSP-based background subtractor
	explicit BackgroundSubtractorLOBSTER(	float fLBSPThreshold/*=BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD*/,
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

