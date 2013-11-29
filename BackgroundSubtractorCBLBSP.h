#pragma once

#include "BackgroundSubtractorLBSP.h"

//! defines the default value for BackgroundSubtractorLBSP::m_fLBSPThreshold
#define BGSCBLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD (0.300f)
//! defines the default value for BackgroundSubtractorLBSP::m_nDescDistThreshold
#define BGSCBLBSP_DEFAULT_DESC_DIST_THRESHOLD (7)
//! defines the default value for BackgroundSubtractorCBLBSP::m_nColorDistThreshold
#define BGSCBLBSP_DEFAULT_COLOR_DIST_THRESHOLD (30)
//! defines the default value for BackgroundSubtractorCBLBSP::m_nLocalWords
#define BGSCBLBSP_DEFAULT_NB_LOCAL_WORDS (35)
//! defines the default value for BackgroundSubtractorCBLBSP::m_nGlobalWords
#define BGSCBLBSP_DEFAULT_NB_GLOBAL_WORDS (200)
//! defines the number of samples to use when computing running averages
//#define BGSCBLBSP_N_SAMPLES_FOR_MEAN (25)
//! defines the threshold values used to detect long-term ghosting and trigger a fast edge-based absorption in the model
//#define BGSCBLBSP_GHOST_DETECTION_D_MAX (0.01f)
//#define BGSCBLBSP_GHOST_DETECTION_S_MIN (0.995f)
//! defines the threshold values used to detect high variation regions that are often labelled as foreground and trigger a local, gradual change in distance thresholds
//#define BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN (0.850f)
//#define BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN (0.175f)
//#define BGSCBLBSP_HIGH_VAR_DETECTION_S_MIN2 (0.100f)
//#define BGSCBLBSP_HIGH_VAR_DETECTION_D_MIN2 (0.225f)
//! defines the internal threshold adjustment factor to use when treating single channel images
#define BGSCBLBSP_SINGLECHANNEL_THRESHOLD_MODULATION_FACT (0.350f) // or (0.500f) for final version? ... more consistent across categories
//! parameters used for dynamic distance threshold adjustments ('R(x)')
//#define BGSCBLBSP_R_SCALE (3.5000f)
//#define BGSCBLBSP_R_INCR  (0.0850f)
//#define BGSCBLBSP_R_DECR  (0.0300f)
//#define BGSCBLBSP_R_LOWER (0.8000f)
//#define BGSCBLBSP_R_UPPER (3.5000f)
//! parameters used for adjusting the variation speed of dynamic distance thresholds  ('R2(x)')
//#define BGSCBLBSP_R2_OFFST (0.100f)
//#define BGSCBLBSP_R2_INCR  (0.800f)
//#define BGSCBLBSP_R2_DECR  (0.100f)
//! parameters used for dynamic learning rates adjustments  ('T(x)')
//#define BGSCBLBSP_T_DECR  (0.0250f)
//#define BGSCBLBSP_T_INCR  (0.2500f)
//#define BGSCBLBSP_T_LOWER (2.0000f)
//#define BGSCBLBSP_T_UPPER (64.000f)

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
								int nInitDescDistThreshold=BGSCBLBSP_DEFAULT_DESC_DIST_THRESHOLD,
								int nInitColorDistThreshold=BGSCBLBSP_DEFAULT_COLOR_DIST_THRESHOLD,
								int nLocalWords=BGSCBLBSP_DEFAULT_NB_LOCAL_WORDS,
								int nGlobalWords=BGSCBLBSP_DEFAULT_NB_GLOBAL_WORDS);
	//! default destructor
	virtual ~BackgroundSubtractorCBLBSP();
	//! (re)initiaization method; needs to be called before starting background subtraction (note: also reinitializes the keypoints vector)
	virtual void initialize(const cv::Mat& oInitImg, const std::vector<cv::KeyPoint>& voKeyPoints);
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
	//! returns a copy of the latest reconstructed background image
	virtual void getBackgroundImage(cv::OutputArray backgroundImage) const;
	//! returns a copy of the latest reconstructed background descriptors image
	virtual void getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const;
	//! sets the keypoints to be used for descriptor extraction, effectively setting the BGModel ROI (note: this function will remove all border keypoints and allocate new words where needed)
	virtual void setBGKeyPoints(std::vector<cv::KeyPoint>& keypoints);

protected:
	class LocalWord {
	public:
		LocalWord(int& nWIDSeed);
		virtual float distance(LocalWord* w)=0;
		virtual float weight(int nCurrFrame);
		int nWID;
		int nFirstOcc;
		int nLastOcc;
		int nOccurrences;
	};
	class GlobalWord {
	public:
		GlobalWord(cv::Size oFrameSize);
		cv::Mat oSpatioOccMap;
	};
	class LocalWord_1ch : public LocalWord {
	public:
		LocalWord_1ch(int& nWIDSeed);
		virtual float distance(LocalWord* w);
		virtual float distance(uchar color, ushort desc);
		uchar nColor;
		ushort nDesc;
	};
	class LocalWord_3ch : public LocalWord {
	public:
		LocalWord_3ch(int& nWIDSeed);
		virtual float distance(LocalWord* w);
		virtual float distance(const uchar* color, const ushort* desc);
		uchar anColor[3];
		ushort anDesc[3];
	};
	class GlobalWord_1ch : public LocalWord_1ch, public GlobalWord {
	public:
		GlobalWord_1ch(int& nWIDSeed, cv::Size oFrameSize);
		virtual float distance(LocalWord* w);
	};
	class GlobalWord_3ch : public LocalWord_3ch, public GlobalWord {
	public:
		GlobalWord_3ch(int& nWIDSeed, cv::Size oFrameSize);
		virtual float distance(LocalWord* w);
	};
	//! absolute color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here, paired with BackgroundSubtractorLBSP::m_nDescDistThreshold)
	const int m_nColorDistThreshold;
	//! number of different local words per pixel/block to be taken from input frames to build the background model (similar to 'N' in ViBe/PBAS)
	const int m_nLocalWords;
	//! number of local words (offset via index from the end) that can be randomly updated when needed
	const int m_nLastLocalWordReplaceableIdxs;
	//! number of different global words to be taken from input frames to build the background model
	const int m_nGlobalWords;
	//! total number of local dictionaries (depends on the input frame size and nb of channels)
	int m_nLocalDictionaries;
	//! WID seed (if implemented as atomic, might be useful later for threaded operations)
	int m_nCurrWIDSeed;
	//! current frame index, used to keep track of word occurrence information
	int m_nFrameIndex;

	//! background model local words dictionaries (1x dictionary/subregion)
	LocalWord*** m_aapLocalWords; // @@@@@@@@ IMPLEMENT A SIMPLE BUBBLE SORT FOR 1-SWAP-PASS/CYCLE ? @@@@@@ create map to keep track of wid => dictionary index?
	//! background model global words dictionary
	GlobalWord** m_apGlobalWords; // @@@@@@@@ IMPLEMENT A SIMPLE BUBBLE SORT FOR 1-SWAP-PASS/CYCLE ?

	//! per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
	//cv::Mat m_oDistThresholdFrame;
	//! per-pixel distance threshold variation modulators ('R2(x)', relative value used to modulate 'R(x)' variations)
	//cv::Mat m_oDistThresholdVariationFrame;
	//! per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
	//cv::Mat m_oMeanMinDistFrame;
	//! per-pixel mean distances between consecutive frames ('D_last(x)', used to detect ghosts and high variation regions in the sequence)
	//cv::Mat m_oMeanLastDistFrame;
	//! per-pixel mean segmentation results ('S(x)', used to detect ghosts and high variation regions in the sequence)
	//cv::Mat m_oMeanSegmResFrame;
	//! per-pixel blink detection results ('Z(x)', used to determine which frame regions should be assigned stronger 'R(x)' variations)
	//cv::Mat m_oBlinksFrame;
	//! per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
	//cv::Mat m_oUpdateRateFrame;
	//! per-pixel word weight thresholds @@@@@@@@ curr used for debug
	//cv::Mat m_oWeightThresholdFrame;
	//! copy of previously used pixel intensities used to calculate 'D_last(x)'
	cv::Mat m_oLastColorFrame;
	//! copy of previously used descriptors used to calculate 'D_last(x)'
	cv::Mat m_oLastDescFrame;
	//! the foreground mask generated by the method at [t-1] (without post-proc, used for blinking px detection)
	//cv::Mat m_oPureFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc)
	cv::Mat m_oFGMask_last;
	//! the foreground mask generated by the method at [t-1] (with post-proc + dilatation, used for blinking px validation)
	//cv::Mat m_oFGMask_last_dilated;

	//! pre-allocated CV_8UC1 matrix used to speed up morph ops
	//cv::Mat m_oTempFGMask;
	//cv::Mat m_oPureFGBlinkMask_curr;
	//cv::Mat m_oPureFGBlinkMask_last;

	//! pre-allocated internal LBSP threshold values for all possible 8-bit intensity values
	uchar m_nLBSPThreshold_8bitLUT[256];

	//! background model pixel color intensity samples (UNUSED, LEFT FOR DEBUG PURPOSES ONLY, SAME AS BackgroundSubtractorLBSP::m_voBGDescSamples @@@@ )
	//std::vector<cv::Mat> m_voBGColorSamples;

	//! internal cleanup function for the dictionary structures
	void CleanupDictionaries();
};

