#pragma once

#include "BackgroundSubtractorViBe.h"

/*!
	ViBe foreground-background segmentation algorithm (1ch/grayscale version).

	For more details on the different parameters, go to @@@@@@@@@@@@@@.

	This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorViBe_1ch : public BackgroundSubtractorViBe {
public:
	//! full constructor
	BackgroundSubtractorViBe_1ch(	size_t nColorDistThreshold=BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
									size_t nBGSamples=BGSVIBE_DEFAULT_NB_BG_SAMPLES,
									size_t nRequiredBGSamples=BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
	//! default destructor
	virtual ~BackgroundSubtractorViBe_1ch();
	//! (re)initiaization method; needs to be called before starting background subtraction
	virtual void initialize(const cv::Mat& oInitImg);
	//! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
	virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBE_DEFAULT_LEARNING_RATE);
};
