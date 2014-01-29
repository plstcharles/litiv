#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines the default value for BackgroundSubtractorLBSP::m_fLBSPThreshold
#define BGSCBLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.250f)
//! defines the default offset LBSP threshold value
#define BGSCBLBSP_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD (3)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSCBLBSP_DEFAULT_DESC_DIST_THRESHOLD (6)
//! defines the default value for BackgroundSubtractorCBLBSP::m_nColorDistThreshold
#define BGSCBLBSP_DEFAULT_COLOR_DIST_THRESHOLD (20)
//! defines the default value for BackgroundSubtractorCBLBSP::m_fLocalWordsPerChannel
#define BGSCBLBSP_DEFAULT_NB_LOCAL_WORDS_PER_CH (8.0f)
//! defines the default value for BackgroundSubtractorCBLBSP::m_fGlobalWordsPerPixelChannel
#define BGSCBLBSP_DEFAULT_NB_GLOBAL_WORDS_PER_CH (10.0f)
//! defines the number of samples to use when computing running averages
#define BGSCBLBSP_N_SAMPLES_FOR_MEAN (100)
//! defines the threshold values used to detect long-term ghosting and trigger a fast edge-based absorption in the model
#define BGSCBLBSP_GHOST_DETECTION_SAVG_MIN (0.8500f)
#define BGSCBLBSP_GHOST_DETECTION_ZAVG_MIN (0.8500f)
#define BGSCBLBSP_GHOST_DETECTION_DMIX_MAX (0.6000f)
#define BGSCBLBSP_GHOST_DETECTION_DLST_MAX (0.0075f)
//! defines the threshold values used to detect high variation regions that are often labelled as foreground and trigger a local, gradual change in distance thresholds
#define BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN (0.625f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_ZAVG_MIN (0.625f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_DMIX_MIN (0.625f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN (0.150f)
//! defines the internal threshold adjustment factor to use when treating single channel images
#define BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT (0.350f) // or (0.500f) for final version? ... more consistent across categories
//! parameters used for dynamic distance threshold adjustments ('R(x)')
#define BGSCBLBSP_R_SCALE (6.0000f)
#define BGSCBLBSP_R_INCR  (0.0300f)
#define BGSCBLBSP_R_DECR  (0.0150f)
#define BGSCBLBSP_R_LOWER (0.6000f)
#define BGSCBLBSP_R_UPPER (3.0000f)
//! parameters used for adjusting the variation speed of dynamic distance thresholds  ('R2(x)')
#define BGSCBLBSP_R2_OFFST (0.100f)
#define BGSCBLBSP_R2_INCR  (1.000f)
#define BGSCBLBSP_R2_DECR  (0.100f)
#define BGSCBLBSP_R2_UPPER (5.000f)
//! parameters used for dynamic learning rates adjustments  ('T(x)')
#define BGSCBLBSP_T_DECR  (0.0002f)
#define BGSCBLBSP_T_INCR  (0.0040f)
#define BGSCBLBSP_T_LOWER (2.0000f)
#define BGSCBLBSP_T_UPPER (64.000f)

/*!
	CB-Based Local Binary Similarity Pattern (LBSP) foreground-background segmentation algorithm.

	Note: both grayscale and RGB/BGR images may be used with this extractor (parameters
	are adjusted automatically).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorCBLBSP : public BackgroundSubtractorLBSP {
public:
	//! full constructor
	BackgroundSubtractorCBLBSP(	float fLBSPThreshold=BGSCBLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
								size_t nLBSPThresholdOffset=BGSCBLBSP_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
								size_t nInitDescDistThreshold=BGSCBLBSP_DEFAULT_DESC_DIST_THRESHOLD,
								size_t nInitColorDistThreshold=BGSCBLBSP_DEFAULT_COLOR_DIST_THRESHOLD,
								float fLocalWordsPerChannel=BGSCBLBSP_DEFAULT_NB_LOCAL_WORDS_PER_CH,
								float fGlobalWordsPerChannel=BGSCBLBSP_DEFAULT_NB_GLOBAL_WORDS_PER_CH);
	//! default destructor
	virtual ~BackgroundSubtractorCBLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints); // @@@@@@@@@@ currently always reinits internal memory structs without reusing old words, if KPs match
	//! refreshes all local (+ global) dictionaries based on the last analyzed frame
	virtual void refreshModel(size_t nBaseOccCount, size_t nOverallMatchOccIncr, size_t nUniversalOccDecr);
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
	//! returns a copy of the latest reconstructed background image
	virtual void getBackgroundImage(cv::OutputArray backgroundImage) const;
	//! returns a copy of the latest reconstructed background descriptors image
	virtual void getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (@@@@@@ note: cannot be used once model is initialized)
	virtual void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);

protected:
	struct LocalWord {
		size_t nFirstOcc;
		size_t nLastOcc;
		size_t nOccurrences;
	protected:
		~LocalWord(); // only used to prevent internal polymorphism (keeps algo cost down)
	};
	struct LocalWord_1ch : LocalWord {
		uchar nColor;
		uchar nDescBITS;
		ushort nDesc;
	};
	struct LocalWord_3ch : LocalWord {
		uchar anColor[3];
		uchar anDescBITS[3];
		uchar nDescBITS;
		ushort anDesc[3];
	};
	struct GlobalWord {
		float fLatestWeight;
		cv::Mat oSpatioOccMap;
	protected:
		~GlobalWord(); // only used to prevent internal polymorphism (keeps algo cost down)
	};
	struct GlobalWord_1ch : GlobalWord {
		uchar nColor;
		uchar nDescBITS; // 'smoothness' indicator
	};
	struct GlobalWord_3ch : GlobalWord {
		uchar anColor[3];
		uchar nDescBITS; // 'smoothness' indicator
	};
	//! indicates whether internal structures have already been initialized (LBSP lookup tables, word lists, etc.)
	bool m_bInitializedInternalStructs;
	//! absolute color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here, paired with BackgroundSubtractorLBSP::m_nDescDistThreshold)
	const size_t m_nColorDistThreshold;
	//! number of local words per channel used to build a background submodel (for a single pixel)
	const float m_fLocalWordsPerChannel;
	//! number of local words used to build a background submodel (for a single pixel, similar to 'N' in ViBe/PBAS -- only usable after initialisation)
	size_t m_nLocalWords;
	//! number of global words per pixel, per channel used to build a complete background model
	const float m_fGlobalWordsPerChannel;
	//! number of global words used to build a complete background model (only usable after initialisation)
	size_t m_nGlobalWords;
	//! total maximum number of local dictionaries (depends on the input frame size)
	size_t m_nMaxLocalDictionaries;
	//! current frame index, used to keep track of word occurrence information
	size_t m_nFrameIndex;

	//! background model local word list & dictionaries
	LocalWord** m_aapLocalDicts;
	LocalWord_1ch* m_apLocalWordList_1ch, *m_apLocalWordListIter_1ch;
	LocalWord_3ch* m_apLocalWordList_3ch, *m_apLocalWordListIter_3ch;
	//! background model global word list & dictionary (1x dictionary for the whole model)
	GlobalWord** m_apGlobalDict;
	GlobalWord_1ch* m_apGlobalWordList_1ch, *m_apGlobalWordListIter_1ch;
	GlobalWord_3ch* m_apGlobalWordList_3ch, *m_apGlobalWordListIter_3ch;
	GlobalWord** m_apGlobalWordLookupTable;

	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	cv::Mat m_oDistThresholdFrame;
	//! per-pixel distance threshold variation modulators ('R2(x)', relative value used to modulate 'R(x)' variations)
	cv::Mat m_oDistThresholdVariationFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	cv::Mat m_oMeanMinDistFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanLastDistFrame;
	//! per-pixel mean raw segmentation results ('Savg(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanRawSegmResFrame;
	//! per-pixel mean final segmentation results ('Zavg(x)', used to detect ghosts and high variation regions in the sequence)
	cv::Mat m_oMeanFinalSegmResFrame;
	//! per-pixel blink detection results (used to determine which frame regions should be assigned stronger 'R(x)' variations via 'R2(x)')
	cv::Mat m_oBlinksFrame;
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	cv::Mat m_oUpdateRateFrame;
	//! per-pixel word weight thresholds @@@@@@@@ curr used for debug
	//cv::Mat m_oWeightThresholdFrame;
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
	//! a lookup mat used to keep track of high variation regions (for neighbor-spread ops)
	cv::Mat m_oHighVarRegionMask;
	//! a lookup mat used to keep track of ghost regions (for neighbor-spread ops)
	cv::Mat m_oGhostRegionMask;

	//! pre-allocated CV_8UC1 matrix used to speed up morph ops
	cv::Mat m_oTempFGMask;
	cv::Mat m_oTempFGMask2;
	cv::Mat m_oFGMask_last_dilated_inverted;
	cv::Mat m_oPureFGBlinkMask_curr;
	cv::Mat m_oPureFGBlinkMask_last;

	//! pre-allocated internal LBSP threshold values for all possible 8-bit intensity values
	size_t m_anLBSPThreshold_8bitLUT[256];

	//! internal cleanup function for the dictionary structures
	void CleanupDictionaries();
	//! internal weight lookup function for local words
	static float GetLocalWordWeight(const LocalWord* w, size_t nCurrFrame);
	//! internal weight lookup function for global words
	static float GetGlobalWordWeight(const GlobalWord* w);
};
