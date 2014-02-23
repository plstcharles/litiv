#pragma once

#include "BackgroundSubtractorLBSP.h"


 // @@@@@ RETEST BURST FOR HIGHVAR / GHOSTS...
 // @@@@@ TEST HIGHVAR WiTH UNSTABLE AS REQUIREMENT

//! defines the default value for BackgroundSubtractorLBSP::m_fLBSPThreshold
#define BGSCBLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.300f)
//! defines the default offset LBSP threshold value
#define BGSCBLBSP_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD (10)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSCBLBSP_DEFAULT_DESC_DIST_THRESHOLD (1)
//! defines the default value for BackgroundSubtractorCBLBSP::m_nColorDistThreshold
#define BGSCBLBSP_DEFAULT_COLOR_DIST_THRESHOLD (25)
//! defines the default value for BackgroundSubtractorCBLBSP::m_fLocalWordsPerChannel
#define BGSCBLBSP_DEFAULT_NB_LOCAL_WORDS_PER_CH (7.0f)
//! defines the default value for BackgroundSubtractorCBLBSP::m_fGlobalWordsPerPixelChannel
#define BGSCBLBSP_DEFAULT_NB_GLOBAL_WORDS_PER_CH (10.0f)
//! defines the number of samples to use when computing running averages
#define BGSCBLBSP_N_SAMPLES_FOR_MEAN (100)
//! defines the threshold values used to detect long-term ghosting
#define BGSCBLBSP_GHOST_DETECTION_SAVG_MIN1 (9.995f)
#define BGSCBLBSP_GHOST_DETECTION_DLST_MAX1 (-0.005f)
#define BGSCBLBSP_GHOST_DETECTION_SAVG_MIN2 (9.99999f)
//! defines the threshold values used to detect high variation regions
#define BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN1 (1.975f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN1 (-0.125f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN2 (1.900f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN2 (-0.150f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN3 (1.150f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN3 (-0.200f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_SAVG_MIN4 (1.125f)
#define BGSCBLBSP_HIGH_VAR_DETECTION_DLST_MIN4 (-0.225f)
//! defines the threshold values used to detect burst variation regions
#define BGSCBLBSP_BURST_VAR_DETECTION_SAVG_MIN1 (1.950f)
#define BGSCBLBSP_BURST_VAR_DETECTION_DLST_MIN1 (-0.235f)
#define BGSCBLBSP_BURST_VAR_DETECTION_SAVG_MIN2 (1.100f)
#define BGSCBLBSP_BURST_VAR_DETECTION_DLST_MIN2 (-0.260f)
//! defines the threshold values used to detect unstable regions and edges
#define BGSCBLBSP_INSTBLTY_DETECTION_SEGM_DIFF (0.200f)
#define BGSCBLBSP_INSTBLTY_DETECTION_MIN_R_VAL (3.000f)
//! defines the internal threshold adjustment factor to use when treating single channel images
#define BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT (0.400f)
//! parameters used for dynamic distance threshold adjustments ('R(x)')
#define BGSCBLBSP_R_INCR  (0.0100f)
#define BGSCBLBSP_R_DECR  (0.0100f)
#define BGSCBLBSP_R_LOWER (1.0000f)
#define BGSCBLBSP_R_UPPER (4.0000f)
//! parameters used for adjusting the variation speed of dynamic distance thresholds  ('R2(x)')
#define BGSCBLBSP_R2_OFFST (0.100f)
#define BGSCBLBSP_R2_INCR  (1.000f)
#define BGSCBLBSP_R2_DECR  (0.100f)
#define BGSCBLBSP_R2_LOWER (0.100f)
#define BGSCBLBSP_R2_UPPER (1000.f)
//! parameters used for dynamic learning rates adjustments  ('T(x)')
#define BGSCBLBSP_T_DECR  (0.2500f)
#define BGSCBLBSP_T_INCR  (0.5000f)
#define BGSCBLBSP_T_LOWER (2.0000f)
#define BGSCBLBSP_T_UPPER (256.00f)

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
	virtual void refreshModel(size_t nBaseOccCount, size_t nOverallMatchOccIncr, float fOccDecrFrac, bool bForceFGUpdate=false);
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
		ushort nDesc;
	};
	struct LocalWord_3ch : LocalWord {
		uchar anColor[3];
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
	//! current number of bad continuous frames detected (used to reset the model when it gets high)
	size_t m_nModelResetFrameCount;

	//! background model local word list & dictionaries
	LocalWord** m_aapLocalDicts;
	LocalWord_1ch* m_apLocalWordList_1ch, *m_apLocalWordListIter_1ch;
	LocalWord_3ch* m_apLocalWordList_3ch, *m_apLocalWordListIter_3ch;
	//! background model global word list & dictionary (1x dictionary for the whole model)
	GlobalWord** m_apGlobalDict;
	GlobalWord_1ch* m_apGlobalWordList_1ch, *m_apGlobalWordListIter_1ch;
	GlobalWord_3ch* m_apGlobalWordList_3ch, *m_apGlobalWordListIter_3ch;
	GlobalWord** m_apGlobalWordLookupTable_BG;
	GlobalWord** m_apGlobalWordLookupTable_FG;

	//! per-pixel distance thresholds ('R(x)')
	cv::Mat m_oDistThresholdFrame;
	//! per-pixel distance threshold variation modulators ('R2(x)')
	cv::Mat m_oDistThresholdVariationFrame;
	//! per-pixel mean minimal model distances ('D_min(x)')
	cv::Mat m_oMeanMinDistFrame;
	//! per-pixel mean minimal model distances (burst version -- 'D_min_burst(x)')
	cv::Mat m_oMeanMinDistFrame_burst;
	//! per-pixel mean distances between consecutive frames ('D_last(x)')
	cv::Mat m_oMeanLastDistFrame;
	//! per-pixel mean distances between consecutive frames (burst version -- 'D_last_burst(x)')
	cv::Mat m_oMeanLastDistFrame_burst;
	//! per-pixel mean raw segmentation results ('S_avg(x)')
	cv::Mat m_oMeanRawSegmResFrame;
	//! per-pixel mean raw segmentation results (burst version -- 'S_avg_burst(x)')
	cv::Mat m_oMeanRawSegmResFrame_burst;
	//! per-pixel mean final segmentation results ('Z_avg(x)')
	cv::Mat m_oMeanFinalSegmResFrame;
	//! per-pixel blink detection results (@@@@@ unnamed)
	cv::Mat m_oBlinksFrame;
	//! per-pixel update rates ('T(x)')
	cv::Mat m_oUpdateRateFrame;
	//! per-pixel word weight thresholds (@@@@@ unnamed)
	//cv::Mat m_oWeightThresholdFrame;
	//! copy of previously used pixel intensities (used to calculate 'D_last(x)')
	cv::Mat m_oLastColorFrame;
	//! copy of previously used descriptors (used to calculate 'D_last(x)')
	cv::Mat m_oLastDescFrame;
	//! the raw foreground mask generated by the method at [t-1]
	cv::Mat m_oPureFGMask_last;
	//! the final foreground masks generated by the method at [t-1 & t-2]
	cv::Mat m_oFGMask_last, m_oFGMask_last2;
	//! the dilated final foreground mask generated by the method at [t-1]
	cv::Mat m_oFGMask_last_dilated;
	//! a lookup map used to keep track of high variation regions
	cv::Mat m_oHighVarRegionMask;
	//! a lookup map used to keep track of ghost regions
	cv::Mat m_oGhostRegionMask;
	//! a lookup map used to keep track of unstable regions
	cv::Mat m_oUnstableRegionMask;
	//! a lookup-map used to keep track of illumination updates
	cv::Mat m_oIllumUpdtRegionMask;

	//! pre-allocated matrices used to speed up morph ops and keep by-products
	cv::Mat m_oFGMask_PreFlood;
	cv::Mat m_oFGMask_FloodedHoles;
	cv::Mat m_oFGMask_last_dilated_inverted;
	cv::Mat m_oPureFGBlinkMask_curr;
	cv::Mat m_oPureFGBlinkMask_last;
	cv::Mat m_oTempGlobalWordWeightDiffFactor;

	//! pre-allocated internal LBSP threshold values for all possible 8-bit intensity values
	size_t m_anLBSPThreshold_8bitLUT[256];

	//! internal cleanup function for the dictionary structures
	void CleanupDictionaries();
	//! internal weight lookup function for local words
	static float GetLocalWordWeight(const LocalWord* w, size_t nCurrFrame);
	//! internal weight lookup function for global words
	static float GetGlobalWordWeight(const GlobalWord* w);
};
