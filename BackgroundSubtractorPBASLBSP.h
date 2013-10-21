#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines the default value for BackgroundSubtractorLBSP::m_fLBSPThreshold
#define BGSPBASLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.300f)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSPBASLBSP_DEFAULT_DESC_DIST_THRESHOLD (7)
//! defines the default value for BackgroundSubtractorPBASLBSP::m_nDefaultColorDistThreshold
#define BGSPBASLBSP_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorPBASLBSP::m_nBGSamples
#define BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES (35)
//! defines the default value for BackgroundSubtractorPBASLBSP::m_nRequiredBGSamples
#define BGSPBASLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the number of samples to use when computing running averages
#define BGSPBASLBSP_N_SAMPLES_FOR_MEAN (25)
//! defines the threshold values used to detect long-term ghosting and trigger a fast edge-based absorption in the model
#define BGSPBASLBSP_GHOST_DETECTION_D_MAX (0.01f)
#define BGSPBASLBSP_GHOST_DETECTION_S_MIN (0.995f)
//! defines the threshold values used to detect high variation regions that are often labelled as foreground and trigger a local, gradual change in distance thresholds
#define BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN (0.850f)
#define BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN (0.175f)
#define BGSPBASLBSP_HIGH_VAR_DETECTION_S_MIN2 (0.100f)
#define BGSPBASLBSP_HIGH_VAR_DETECTION_D_MIN2 (0.225f)
//! defines the internal threshold adjustment factor to use when treating single channel images
#define BGSPBASLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT (0.350f)
//! parameters used for dynamic distance threshold adjustments ('R(x)')
#define BGSPBASLBSP_R_SCALE (3.5000f)
#define BGSPBASLBSP_R_INCR  (0.0750f)
#define BGSPBASLBSP_R_DECR  (0.0250f)
#define BGSPBASLBSP_R_LOWER (0.8000f)
#define BGSPBASLBSP_R_UPPER (3.5000f)
//! parameters used for adjusting the variation speed of dynamic distance thresholds  ('R2(x)')
#define BGSPBASLBSP_R2_OFFST (0.070f)
#define BGSPBASLBSP_R2_INCR  (0.200f)
#define BGSPBASLBSP_R2_DECR  (0.025f)
//! parameters used for dynamic learning rates adjustments  ('T(x)')
#define BGSPBASLBSP_T_DECR  (0.0500f)
#define BGSPBASLBSP_T_INCR  (0.5000f)
#define BGSPBASLBSP_T_LOWER (2.0000f)
#define BGSPBASLBSP_T_UPPER (256.00f)

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
	BackgroundSubtractorPBASLBSP(	float fLBSPThreshold=BGSPBASLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									int nInitDescDistThreshold=BGSPBASLBSP_DEFAULT_DESC_DIST_THRESHOLD,
									int nInitColorDistThreshold=BGSPBASLBSP_DEFAULT_COLOR_DIST_THRESHOLD,
									int nBGSamples=BGSPBASLBSP_DEFAULT_NB_BG_SAMPLES,
									int nRequiredBGSamples=BGSPBASLBSP_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorPBASLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! primary model update function; the learning param is used to override the internal learning thresholds (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
	//! returns a copy of the latest reconstructed background image
	void getBackgroundImage(cv::OutputArray backgroundImage) const;

protected:
	//! absolute color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here, paired with BackgroundSubtractorLBSP::m_nDescDistThreshold)
	const int m_nColorDistThreshold;
	//! number of different samples per pixel/block to be taken from input frames to build the background model (same as 'N' in ViBe/PBAS)
	const int m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background' (same as '#_min' in ViBe/PBAS)
	const int m_nRequiredBGSamples;

	//! background model pixel intensity samples (equivalent to 'B(x)' in PBAS, but also paired with BackgroundSubtractorLBSP::m_voBGDesc to create our complete model)
	std::vector<cv::Mat> m_voBGImg;

	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	cv::Mat m_oDistThresholdFrame;
	//! per-pixel distance threshold variation modulators ('R2(x)', relative value used to modulate 'R(x)' variations)
	cv::Mat m_oDistThresholdVariationFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat m_oMeanMinDistFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanLastDistFrame;
	//! per-pixel mean segmentation results ('S(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanSegmResFrame;
	//! per-pixel blink detection results ('Z(x)', used to determine which frame regions should be assigned stronger 'R(x)' variations)
	cv::Mat m_oBlinksFrame;
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	cv::Mat m_oUpdateRateFrame;
	//! copy of previously used pixel intensities used to calculate 'D_last(x)'
	cv::Mat m_oLastColorFrame;
	//! copy of previously used descriptors used to calculate 'D_last(x)'
	cv::Mat m_oLastDescFrame;
	//! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat m_oPureFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	cv::Mat m_oFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc + dilatation, used for blinking px validation)
	cv::Mat m_oFGMask_last_dilated;

	//! pre-allocated CV_8UC1 matrix used to speed up morph ops
	cv::Mat m_oTempFGMask;
	cv::Mat m_oPureFGBlinkMask_curr;
	cv::Mat m_oPureFGBlinkMask_last;
};

