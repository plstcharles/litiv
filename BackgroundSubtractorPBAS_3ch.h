#pragma once

#include "BackgroundSubtractorPBAS.h"

/*!
    PBAS foreground-background segmentation algorithm (3ch/RGB version).

    This algorithm is currently NOT thread-safe.
 */
class BackgroundSubtractorPBAS_3ch : public BackgroundSubtractorPBAS {
public:
    //! full constructor
    BackgroundSubtractorPBAS_3ch(   size_t nInitColorDistThreshold=BGSPBAS_DEFAULT_COLOR_DIST_THRESHOLD,
                                    float fInitUpdateRate=BGSPBAS_DEFAULT_LEARNING_RATE,
                                    size_t nBGSamples=BGSPBAS_DEFAULT_NB_BG_SAMPLES,
                                    size_t nRequiredBGSamples=BGSPBAS_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    //! default destructor
    virtual ~BackgroundSubtractorPBAS_3ch();
    //! (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg);
    //! primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
    virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE);
};
