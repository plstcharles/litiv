#pragma once

#include "litiv/imgproc/EdgeDetectionUtils.hpp"

//! defines the default value for EdgeDetectorCanny::m_nDefaultBaseHystThreshold
#define EDGCANNY_DEFAULT_BASE_HYSTERESIS_THRESHOLD (75)
//! defines the default value for EdgeDetectorCanny::m_dHystThresholdMultiplier
#define EDGCANNY_DEFAULT_HYSTERESIS_THRESHOLD_MULT (3)
//! defines the default value for EdgeDetectorCanny::m_nKernelSize
#define EDGCANNY_DEFAULT_KERNEL_SIZE (3)
//! defines the default value for EdgeDetectorCanny::m_bUsingL2GradientNorm
#define EDGCANNY_DEFAULT_USE_L2_GRADIENT_NORM (false)
//! defines the default value for the threshold passed to EdgeDetectorCanny::apply
#define EDGCANNY_DEFAULT_THRESHOLD (double(EDGCANNY_DEFAULT_BASE_HYSTERESIS_THRESHOLD)/UCHAR_MAX)

/*!
    Canny edge detection algorithm (wraps the OpenCV implementation).

    Note: converts all RGB/RGBA images to grayscale internally.
 */
class EdgeDetectorCanny : public EdgeDetectorImpl {
public:
    //! full constructor
    EdgeDetectorCanny(size_t nDefaultBaseHystThreshold=EDGCANNY_DEFAULT_BASE_HYSTERESIS_THRESHOLD,
                      double dHystThresholdMultiplier=EDGCANNY_DEFAULT_HYSTERESIS_THRESHOLD_MULT,
                      size_t nKernelSize=EDGCANNY_DEFAULT_KERNEL_SIZE,
                      bool bUseL2GradientNorm=EDGCANNY_DEFAULT_USE_L2_GRADIENT_NORM);
    //! returns the default threshold value used in 'apply'
    virtual double getDefaultThreshold() const {return double(m_nDefaultBaseHystThreshold)/UCHAR_MAX;}
    //! thresholded edge detection function; the threshold should be between 0 and 1 (will use default otherwise), and sets the base hysteresis threshold
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dThreshold=EDGCANNY_DEFAULT_THRESHOLD);
    //! edge detection function; returns a confidence edge mask (0-255) instead of a thresholded/binary edge mask
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask);

protected:
    //! default base hysteresis lower/base threshold
    const size_t m_nDefaultBaseHystThreshold;
    //! base threshold multiplier used to compute the upper hysteresis threshold
    const double m_dHystThresholdMultiplier;
    //! gaussian blur/sobel gradient kernel size
    const size_t m_nKernelSize;
    //! specifies whether to use the accurate L2 norm for gradient magnitude calculations or simply the L1 norm
    const bool m_bUsingL2GradientNorm;
};
