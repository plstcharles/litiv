#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines whether we should use the selected pixel's characteristics during diffusion or not (1=PBAS-like diffusion, 0=ViBe-like)
#define BGSPBASLBSP_USE_SELF_DIFFUSION 0
//! defines whether to use or not the R2 acceleration thresholds to modulate R(x) variations
#define BGSPBASLBSP_USE_R2_ACCELERATION 1
//! defines whether to use or not the R(x)-based rel/abs LBSP cutoff configuration
#define BGSPBASLBSP_USE_LBSP_TYPE_CUTOFF 0
//! defines whether to use or not the LBSP-desc-dist-based threshold checks & skips
#define BGSPBASLBSP_USE_DESC_DIST_CHECKS 0

//! defines the default value for BackgroundSubtractorViBe::m_fLBSPThreshold
#define BGSPBASLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.300f)
//! defines the default value for BackgroundSubtractorPBASLBSP::m_nDefaultColorDistThreshold
#define BGSPBASLBSP_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSPBASLBSP_DEFAULT_DESC_DIST_THRESHOLD (6)
//! defines the default value for BackgroundSubtractorPBASLBSP::m_nBGSamples
#define BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES (35)
//! defines the default value for BackgroundSubtractorPBASLBSP::m_nRequiredBGSamples
#define BGSPBASLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for BackgroundSubtractorPBASLBSP::m_fDefaultUpdateRate
#define BGSPBASLBSP_DEFAULT_LEARNING_RATE (16.0f)
//! defines the default value for the learning rate passed to BackgroundSubtractorViBeLBSP::operator()
#define BGSPBASLBSP_DEFAULT_LEARNING_RATE_OVERRIDE (-1.0)
//! defines the internal threshold adjustment factor to use when treating single channel images (based on the assumption that grayscale images have less noise per channel...)
#define BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT (0.350f)
//! parameters used for dynamic threshold adjustments
#define BGSPBASLBSP_R_OFFST (0.0000f)
#define BGSPBASLBSP_R_SCALE (3.5000f)
#define BGSPBASLBSP_R_INCR  (0.0750f)
#define BGSPBASLBSP_R_DECR  (0.0250f)
#define BGSPBASLBSP_R_LOWER (0.8000f)
#define BGSPBASLBSP_R_UPPER (3.5000f)
//! parameters used for adjusting the variation speed of dynamic thresholds
#define BGSPBASLBSP_R2_OFFST (0.085f)
#define BGSPBASLBSP_R2_INCR  (0.200f)
#define BGSPBASLBSP_R2_DECR  (0.025f)
#define BGSPBASLBSP_R2_LOWER (0.001f)
#define BGSPBASLBSP_R2_UPPER (15.00f)
//! parameters used for dynamic learning rate adjustments
#define BGSPBASLBSP_T_OFFST (0.0001f)
#define BGSPBASLBSP_T_SCALE (1.0000f)
#define BGSPBASLBSP_T_DECR  (0.0500f)
#define BGSPBASLBSP_T_INCR  (1.0000f)
#define BGSPBASLBSP_T_LOWER (2.0000f)
#define BGSPBASLBSP_T_UPPER (200.00f)
//! weight ratio attributed to gradient intensities when mixing with color
#define BGSPBASLBSP_GRAD_WEIGHT_ALPHA (10.0f)
//! number of samples used to create running averages for model variation computations
#define BGSPBASLBSP_N_SAMPLES_FOR_MEAN (25)
//! defines the default cutoff values to switch between the relative and absolute LBSP approaches
#define BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL (1.500f)
#define BGSPBASLBSP_DEFAULT_REL_LBSP_CUTOFF_R_VAL_BUFFER (0.650f)
#define BGSPBASLBSP_DEF_ABS_LBSP_THRES ((uchar)25)

#define BGSPBASLBSP_D_SPREAD_MAX (0.004f)

/*!
	PBAS-Based Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters
	are adjusted automatically).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorPBASLBSP : public BackgroundSubtractorLBSP {
public:
	//! full constructor
	BackgroundSubtractorPBASLBSP(	bool bDelayedAnalysis=true,
									float fLBSPThreshold=BGSPBASLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									int nInitDescDistThreshold=BGSPBASLBSP_DEFAULT_DESC_DIST_THRESHOLD,
									int nInitColorDistThreshold=BGSPBASLBSP_DEFAULT_COLOR_DIST_THRESHOLD,
									float fInitUpdateRate=BGSPBASLBSP_DEFAULT_LEARNING_RATE,
									int nBGSamples=BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES,
									int nRequiredBGSamples=BGSPBASLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorPBASLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=BGSPBASLBSP_DEFAULT_LEARNING_RATE_OVERRIDE);
	//! returns a copy of the latest reconstructed background image
	void getBackgroundImage(cv::OutputArray backgroundImage) const;

protected:
	//! defines whether we should use a delayed output approach to get better spatiotemporal blink analysis or not (note: delay is always limited to 1 frame)
	const bool m_bDelayAnalysis;
	//! number of different samples per pixel/block to be taken from input frames to build the background model ('N' in the original ViBe/PBAS papers)
	const int m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background' ('#_min' in the original ViBe/PBAS papers)
	const int m_nRequiredBGSamples;
	//! background model pixel intensity samples
	std::vector<cv::Mat> m_voBGImg;
	//! background model pixel gradient samples
	//std::vector<cv::Mat> m_voBGGrad;
	//! absolute color distance threshold ('R' or 'radius' in the original ViBe paper, and the default 'R(x)' value in the original PBAS paper)
	const int m_nColorDistThreshold;
	//! per-pixel distance thresholds ('R(x)' in the original PBAS paper)
	cv::Mat m_oDistThresholdFrame;
	//! per-pixel distance thresholds variation
	cv::Mat m_oDistThresholdVariationFrame;
	//! per-pixel mean minimal decision distances ('D(x)' in the original PBAS paper)
	cv::Mat m_oMeanMinDistFrame;
	// @@@@@@@@@@@@
	cv::Mat m_oMeanNbBlinksFrame; // @@@@@@@@
	// @@@@@@@@@@@@
	cv::Mat m_oBlinksFrame; // @@@@@@@@
	// @@@@@@@@@@@@
	cv::Mat m_oRelLBSPThresFrame; // @@@@@@@@
	// @@@@@@@@@@@@
	cv::Mat m_oLastColorFrame; // @@@@@@@@
	// @@@@@@@@@@@@
	cv::Mat m_oLastDescFrame; // @@@@@@@@
	// @@@@@@@@@@@@
	cv::Mat m_oMeanLastDistFrame; // @@@@@@@@
	//! absolute default update rate threshold (the default 'T(x)' value in the original PBAS paper)
	const float m_fDefaultUpdateRate;
	//! mean gradient magnitude distance over the past frame
	float m_fFormerMeanGradDist;
	//! per-pixel update rate ('T(x)' in the original PBAS paper)
	cv::Mat m_oUpdateRateFrame;

	//! the foreground mask generated by the method at [t-1] (without post-proc)
	cv::Mat m_oPureFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	cv::Mat m_oFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc + dilatation)
	cv::Mat m_oFGMask_last_dilated;
	//! the foreground mask generated by the method at [t-2] (without post-proc) --- used only with delayed analysis
	cv::Mat m_oPureFGMask_old;
	//! the foreground mask generated by the method at [t-2] (with post-proc) --- used only with delayed analysis
	cv::Mat m_oFGMask_old;
	//! the foreground mask generated by the method at [t-2] (with post-proc + dilatation) --- used only with delayed analysis
	cv::Mat m_oFGMask_old_dilated;

	//! pre-allocated CV_8UC1 matrix used to speed up morph ops
	cv::Mat m_oTempFGMask;
};

