#pragma once

#include "BackgroundSubtractorPBAS.h"

//! defines the internal threshold adjustment factor to use when determining if the variation of a single channel is enough to declare the pixel as foreground
#define BGSPBAS_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR (1.60f)
//! defines whether we should use single channel variation checks for fg/bg segmentation validation or not
#define BGSPBAS_USE_SC_THRS_VALIDATION 0

/*!
	PBAS foreground-background segmentation algorithm (3ch/RGB version).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorPBAS_3ch : public BackgroundSubtractorPBAS {
public:
	//! full constructor
	BackgroundSubtractorPBAS_3ch(	int nInitColorDistThreshold=BGSPBAS_DEFAULT_COLOR_DIST_THRESHOLD,
									float fInitUpdateRate=BGSPBAS_DEFAULT_LEARNING_RATE,
									int nBGSamples=BGSPBAS_DEFAULT_NB_BG_SAMPLES,
									int nRequiredBGSamples=BGSPBAS_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorPBAS_3ch();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg);
	//! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
	virtual void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE);
};
