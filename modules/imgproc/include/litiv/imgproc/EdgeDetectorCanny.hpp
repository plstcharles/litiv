#pragma once

#include "litiv/imgproc/EdgeDetectionUtils.hpp"

// note: all default parameters mimic matlab's implementation

//! defines the default value for EdgeDetectorCanny::m_dHystLowThrshFactor
#define EDGCANNY_DEFAULT_HYST_LOW_THRSH_FACT (0.4)
//! defines the default value for EdgeDetectorCanny::m_dGaussianKernelSigma
#define EDGCANNY_DEFAULT_GAUSSIAN_KERNEL_SIGMA (2)
//! defines the default value for EdgeDetectorCanny::m_bUsingL2GradientNorm
#define EDGCANNY_DEFAULT_USE_L2_GRADIENT_NORM (true)
//! defines the default value for the threshold passed to EdgeDetectorCanny::apply
#define EDGCANNY_DEFAULT_THRESHOLD (75.0/UCHAR_MAX)

/*!
    Canny edge detection algorithm (wraps the OpenCV implementation).

    Note: converts all RGB/RGBA images to grayscale internally.
 */
class EdgeDetectorCanny : public EdgeDetectorImpl {
public:
    //! full constructor
    EdgeDetectorCanny(double dHystLowThrshFactor=EDGCANNY_DEFAULT_HYST_LOW_THRSH_FACT,
                      double dGaussianKernelSigma=EDGCANNY_DEFAULT_GAUSSIAN_KERNEL_SIGMA,
                      bool bUseL2GradientNorm=EDGCANNY_DEFAULT_USE_L2_GRADIENT_NORM);
    //! returns the default threshold value used in 'apply'
    virtual double getDefaultThreshold() const {return EDGCANNY_DEFAULT_THRESHOLD;}
    //! thresholded edge detection function; the threshold should be between 0 and 1 (will use default otherwise), and sets the base hysteresis threshold
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dThreshold=EDGCANNY_DEFAULT_THRESHOLD);
    //! edge detection function; returns a confidence edge mask (0-255) instead of a thresholded/binary edge mask
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask);

protected:
    //! base threshold multiplier used to compute the upper hysteresis threshold
    const double m_dHystLowThrshFactor;
    //! gaussian blur kernel sigma value
    const double m_dGaussianKernelSigma;
    //! specifies whether to use the accurate L2 norm for gradient magnitude calculations or simply the L1 norm
    const bool m_bUsingL2GradientNorm;
};
