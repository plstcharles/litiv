#pragma once

#include "litiv/utils/CxxUtils.hpp"
#include "litiv/imgproc.hpp"

//! defines the default value for EdgeDetectorLBSP::m_nLevels
#define EDGLBSP_DEFAULT_LEVEL_COUNT (4)
//! defines the default value for the threshold passed to EdgeDetectorLBSP::apply
#define EDGLBSP_DEFAULT_THRESHOLD (75.0/UCHAR_MAX)

class EdgeDetectorLBSP : public EdgeDetector {
public:
    //! full constructor
    EdgeDetectorLBSP(size_t nLevels=EDGLBSP_DEFAULT_LEVEL_COUNT);
    //! returns the default threshold value used in 'apply'
    virtual double getDefaultThreshold() const {return EDGLBSP_DEFAULT_THRESHOLD;}
    //! thresholded edge detection function; the threshold should be between 0 and 1 (will use default otherwise), and sets the base hysteresis threshold
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dThreshold=EDGLBSP_DEFAULT_THRESHOLD);
    //! edge detection function; returns a confidence edge mask (0-255) instead of a thresholded/binary edge mask
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask);

protected:

    //! number of pyramid levels to analyze
    const size_t m_nLevels;
    //! pre-allocated image pyramid maps for multi-scale LBSP lookup
    std::vector<std::aligned_vector<uchar,32>> m_vvuInputPyrMaps;
    //! pre-allocated image pyramid LUT maps for multi-scale LBSP computation
    std::vector<std::aligned_vector<uchar,32>> m_vvuLBSPLookupMaps;
    //! multi-level image map size lookup list
    std::vector<cv::Size> m_voMapSizeList;
    //! base threshold multiplier used to compute the upper hysteresis threshold
    const double m_dHystLowThrshFactor;
    //! gaussian blur kernel sigma value
    const double m_dGaussianKernelSigma;
    //! specifies whether to use the accurate L2 norm for gradient magnitude calculations or simply the L1 norm
    const bool m_bUsingL2GradientNorm;
};
