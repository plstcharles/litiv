#pragma once

#include "BackgroundSubtractorViBe.h"

//! defines the internal threshold adjustment factor to use when determining if the variation of a single channel is enough to declare the pixel as foreground
#define BGSVIBE_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR (1.60f)
//! defines whether we should use single channel variation checks for fg/bg segmentation validation or not
#define BGSVIBE_USE_SC_THRS_VALIDATION 0

/*!
	ViBe foreground-background segmentation algorithm (3ch/RGB version).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorViBe_3ch : public BackgroundSubtractorViBe {
public:
	//! full constructor
	BackgroundSubtractorViBe_3ch(	int nColorDistThreshold=BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
									int nBGSamples=BGSVIBE_DEFAULT_NB_BG_SAMPLES,
									int nRequiredBGSamples=BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorViBe_3ch();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg);
	//! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBE_DEFAULT_LEARNING_RATE);
};
