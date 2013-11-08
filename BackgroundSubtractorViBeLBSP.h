#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines whether we should use the selected pixel's characteristics during diffusion or not (1=PBAS-like diffusion, 0=ViBe-like)
#define BGSVIBELBSP_USE_SELF_DIFFUSION 0
//! defines whether we should complement the LBSP core component using color or not
#define BGSVIBELBSP_USE_COLOR_COMPLEMENT 1

//! defines the default value for BackgroundSubtractorViBe::m_nLBSPThreshold (if needed)
#define BGSVIBELBSP_DEFAULT_LBSP_ABS_SIMILARITY_THRESHOLD ((uchar)30)
//! defines the default value for BackgroundSubtractorViBe::m_fLBSPThreshold (if needed)
#define BGSVIBELBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.365f)
//! defines the default value for BackgroundSubtractorViBeLBSP::m_nColorDistThreshold
#define BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSVIBELBSP_DEFAULT_DESC_DIST_THRESHOLD (4)
//! defines the default value for BackgroundSubtractorViBeLBSP::m_nBGSamples
#define BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES (35)
//! defines the default value for BackgroundSubtractorViBeLBSP::m_nRequiredBGSamples
#define BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorViBeLBSP::operator()
#define BGSVIBELBSP_DEFAULT_LEARNING_RATE (16)
//! defines the internal threshold adjustment factor to use when treating single channel images (based on the assumption that grayscale images have less noise per channel...)
#define BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT (0.350f)

/*!
	ViBe-Based Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorViBeLBSP : public BackgroundSubtractorLBSP {
public:
	//! full constructor used to intialize an 'absolute' LBSP-based background subtractor
	explicit BackgroundSubtractorViBeLBSP(	uchar nLBSPThreshold/*=BGSVIBELBSP_DEFAULT_LBSP_ABS_SIMILARITY_THRESHOLD*/,
											int nDescDistThreshold=BGSVIBELBSP_DEFAULT_DESC_DIST_THRESHOLD,
											int nColorDistThreshold=BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD,
											int nBGSamples=BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES,
											int nRequiredBGSamples=BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! full constructor used to intialize a 'relative' LBSP-based background subtractor
	explicit BackgroundSubtractorViBeLBSP(	float fLBSPThreshold/*=BGSVIBELBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD*/,
											int nDescDistThreshold=BGSVIBELBSP_DEFAULT_DESC_DIST_THRESHOLD,
											int nColorDistThreshold=BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD,
											int nBGSamples=BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES,
											int nRequiredBGSamples=BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorViBeLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBELBSP_DEFAULT_LEARNING_RATE);
	//! returns a copy of the latest reconstructed background image
	void getBackgroundImage(cv::OutputArray backgroundImage) const;

protected:
	//! number of different samples per pixel/block to be taken from input frames to build the background model
	const int m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background'
	const int m_nRequiredBGSamples;
	//! background model pixel intensity samples
	std::vector<cv::Mat> m_voBGColorSamples;
	//! absolute per-channel color distance threshold (based on the provided LBSP threshold)
	const int m_nColorDistThreshold;
};

