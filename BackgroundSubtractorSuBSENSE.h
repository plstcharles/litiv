#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines the default value for BackgroundSubtractorLBSP::m_fRelLBSPThreshold
#define BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.333f)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD (3)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nMinColorDistThreshold
#define BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nBGSamples
#define BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES (50)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nRequiredBGSamples
#define BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for BackgroundSubtractorSuBSENSE::m_nSamplesForMovingAvgs
#define BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS (25)

/*!
	Self-Balanced Sensitivity segmenTER (SuBSENSE) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).
	For optimal grayscale results, use CV_8UC1 frames instead of CV_8UC3.

	For more details on the different parametersor on the algorithm itself, see P.-L. St-Charles et al.,
	"Flexible Background Subtraction With Self-Balanced Local Sensitivity", in CVPRW 2014.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorSuBSENSE : public BackgroundSubtractorLBSP {
public:
	//! full constructor
	BackgroundSubtractorSuBSENSE(	float fRelLBSPThreshold=BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
									size_t nMinDescDistThreshold=BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD,
									size_t nMinColorDistThreshold=BGSSUBSENSE_DEFAULT_COLOR_DIST_THRESHOLD,
									size_t nBGSamples=BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
									size_t nRequiredBGSamples=BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES,
									size_t nSamplesForMovingAvgs=BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS);
	//! default destructor
	virtual ~BackgroundSubtractorSuBSENSE();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! refreshes all samples based on the last analyzed frame
	virtual void refreshModel(float fSamplesRefreshFrac);
	//! primary model update function; the learning param is used to override the internal learning thresholds (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
	//! returns a copy of the latest reconstructed background image
	void getBackgroundImage(cv::OutputArray backgroundImage) const;
	//! turns automatic model reset on or off
	void setAutomaticModelReset(bool);

protected:
	//! indicates whether internal structures have already been initialized (LBSP lookup tables, samples, etc.)
	bool m_bInitializedInternalStructs;
	//! absolute minimal color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here, paired with BackgroundSubtractorLBSP::m_nDescDistThreshold)
	const size_t m_nMinColorDistThreshold;
	//! number of different samples per pixel/block to be taken from input frames to build the background model (same as 'N' in ViBe/PBAS)
	const size_t m_nBGSamples;
	//! number of similar samples needed to consider the current pixel/block as 'background' (same as '#_min' in ViBe/PBAS)
	const size_t m_nRequiredBGSamples;
	//! number of samples to use to compute the learning rate of moving averages
	const size_t m_nSamplesForMovingAvgs;
	//! current frame index, frame count since last model reset & model reset cooldown counters
	size_t m_nFrameIndex, m_nFramesSinceLastReset, m_nModelResetCooldown;
	//! last calculated non-zero desc ratio
	float m_fLastNonZeroDescRatio;
	//! specifies whether automatic model reset is enabled or not
	bool m_bAutoModelResetEnabled;
	//! specifies whether Tmin/Tmax scaling is enabled or not
	bool m_bLearningRateScalingEnabled;
	//! current learning rate caps
	float m_fCurrLearningRateLowerCap, m_fCurrLearningRateUpperCap;
	//! current kernel size for median blur post-proc filtering
	int m_nMedianBlurKernelSize;
	//! specifies the px update spread range
	bool m_bUse3x3Spread;
	//! specifies the downsampled frame size (used for cam motion analysis)
	cv::Size m_oDownSampledFrameSize;

	//! background model pixel color intensity samples (equivalent to 'B(x)' in PBAS, but also paired with BackgroundSubtractorLBSP::m_voBGDescSamples to create our complete model)
	std::vector<cv::Mat> m_voBGColorSamples;

	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	cv::Mat m_oUpdateRateFrame;
	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	cv::Mat m_oDistThresholdFrame;
	//! per-pixel distance variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
	cv::Mat m_oVariationModulatorFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanLastDistFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat m_oMeanMinDistFrame_LT, m_oMeanMinDistFrame_ST;
	//! per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and control max learning rates globally)
	cv::Mat m_oMeanDownSampledLastDistFrame_LT, m_oMeanDownSampledLastDistFrame_ST;
	//! per-pixel mean raw segmentation results
	cv::Mat m_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_ST;
	//! per-pixel mean final segmentation results
	cv::Mat m_oMeanFinalSegmResFrame_LT, m_oMeanFinalSegmResFrame_ST;
	//! a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
	cv::Mat m_oUnstableRegionMask;
	//! per-pixel blink detection results ('Z(x)')
	cv::Mat m_oBlinksFrame;
	//! pre-allocated matrix used to downsample (1/8) the input frame when needed
	cv::Mat m_oDownSampledColorFrame;
	//! copy of previously used pixel intensities used to calculate 'D_last(x)'
	cv::Mat m_oLastColorFrame;
	//! copy of previously used descriptors used to calculate 'D_last(x)'
	cv::Mat m_oLastDescFrame;
	//! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	cv::Mat m_oRawFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	cv::Mat m_oFGMask_last;

	//! pre-allocated CV_8UC1 matrices used to speed up morph ops
	cv::Mat m_oFGMask_PreFlood;
	cv::Mat m_oFGMask_FloodedHoles;
	cv::Mat m_oFGMask_last_dilated;
	cv::Mat m_oFGMask_last_dilated_inverted;
	cv::Mat m_oRawFGBlinkMask_curr;
	cv::Mat m_oRawFGBlinkMask_last;
};

