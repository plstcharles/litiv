#pragma once

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/background_segm.hpp>
#include "LBSP.h"

//! defines the default value for BackgroundSubtractorViBeLBSP::m_nDescDistThreshold
#define BGSVIBELBSP_DEFAULT_DESC_DIST_THRESHOLD (4)
//! defines the default value for BackgroundSubtractorViBeLBSP::m_nColorDistThreshold
#define BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD (LBSP_DEFAULT_ABS_SIMILARITY_THRESHOLD)
//! defines the default value for BackgroundSubtractorViBeLBSP::m_nBGSamples
#define BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES (35)
//! defines the default value for BackgroundSubtractorViBeLBSP::m_nRequiredBGSamples
#define BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorViBeLBSP::operator()
#define BGSVIBELBSP_DEFAULT_LEARNING_RATE (16)
//! defines the internal threshold adjustment factor to use when determining if the variation of a single channel is enough to declare the pixel as foreground
#define BGSVIBELBSP_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR (1.50f)
//! defines whether we should use the selected pixel's characteristics during diffusion or not (1=PBAS-like diffusion, 0=ViBe-like)
#define BGSVIBELBSP_USE_SELF_DIFFUSION 0
//! defines whether we should use single channel variation checks for fg/bg segmentation validation or not
#define BGSVIBELBSP_USE_SC_THRS_VALIDATION 1
//! defines whether we should complement the LBSP core component using color or not
#define BGSVIBELBSP_USE_COLOR_COMPLEMENT 1

/*!
	ViBe-Based Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters
	are adjusted automatically).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorViBeLBSP : public cv::BackgroundSubtractor {
public:
	//! default constructor (also uses the default LBSP descriptor extractor constructor & params)
	BackgroundSubtractorViBeLBSP();
	//! full constructor used to intialize an 'absolute' LBSP-based background subtractor
	BackgroundSubtractorViBeLBSP(	int nLBSPThreshold,
									int nDescDistThreshold=BGSVIBELBSP_DEFAULT_DESC_DIST_THRESHOLD,
									int nColorDistThreshold=BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD,
									int nBGSamples=BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES,
									int nRequiredBGSamples=BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! full constructor used to intialize a 'relative' LBSP-based background subtractor
	BackgroundSubtractorViBeLBSP(	float fLBSPThreshold,
									int nDescDistThreshold=BGSVIBELBSP_DEFAULT_DESC_DIST_THRESHOLD,
									int nColorDistThreshold=BGSVIBELBSP_DEFAULT_COLOR_DIST_THRESHOLD,
									int nBGSamples=BGSVIBELBSP_DEFAULT_NB_BG_SAMPLES,
									int nRequiredBGSamples=BGSVIBELBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorViBeLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints=std::vector<cv::KeyPoint>());
	//! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBELBSP_DEFAULT_LEARNING_RATE);
	//! @@@@@@@@@@@@ ????
	virtual cv::AlgorithmInfo* info() const;
	//! returns a copy of the latest reconstructed background image (in this case, since we're using a ViBe-like method, it returns the first sample set)
	void getBackgroundImage(cv::OutputArray backgroundImage) const;
	//! returns a copy of the latest background descriptors image (in this case, since we're using a ViBe-like method, it returns the first sample set)
	void getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const;
	//! returns the keypoints list used for descriptor extraction (note: by default, these are generated from the DenseFeatureDetector class, and the border points are removed)
	std::vector<cv::KeyPoint> getBGKeyPoints() const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border keypoints)
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
	//! absolute per-channel descriptor hamming distance threshold
	const int m_nDescDistThreshold;
	//! absolute per-channel color distance threshold (based on the provided LBSP threshold)
	const int m_nColorDistThreshold;
	//! defines if we're using a relative threshold when extracting LBSP features (kept here since we don't keep an LBSP object)
	const bool m_bLBSPUsingRelThreshold;
	//! LBSP absolute internal threshold (kept here since we don't keep an LBSP object)
	const int m_nLBSPThreshold;
	//! LBSP relative internal threshold (kept here since we don't keep an LBSP object)
	const float m_fLBSPThreshold;
	//! defines whether or not the subtractor is fully initialized
	bool m_bInitialized;
};

