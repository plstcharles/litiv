
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "litiv/imgproc/EdgeDetectionUtils.hpp"

// note: all default parameters mimic matlab's implementation

/// defines the default value for the norm type used to compute gradient magnitudes
#define EDGCANNY_USE_L2_GRADIENT_NORM (true)
/// defines the default value for the aperture (or window) size used for Sobel gradient estimation
#define EDGCANNY_SOBEL_KERNEL_SIZE (3)
/// defines the default value for the threshold passed to EdgeDetectorCanny::apply_threshold
#define EDGCANNY_DEFAULT_THRESHOLD (100<<std::max(EDGCANNY_SOBEL_KERNEL_SIZE-3,0)*2)
/// defines the default value for EdgeDetectorCanny::m_dHystLowThrshFactor
#define EDGCANNY_DEFAULT_HYST_LOW_THRSH_FACT (0.4)
/// defines the default value for EdgeDetectorCanny::m_dGaussianKernelSigma
#define EDGCANNY_DEFAULT_GAUSSIAN_KERNEL_SIGMA (sqrt(2.0))

/**
    Canny edge detection algorithm (wraps the OpenCV implementation).

    Only available in non-parallel version.

    Note: converts all RGB/RGBA images to grayscale internally.
*/
struct EdgeDetectorCanny : public IEdgeDetector {
    /// full constructor
    EdgeDetectorCanny(double dHystLowThrshFactor=EDGCANNY_DEFAULT_HYST_LOW_THRSH_FACT,
                      double dGaussianKernelSigma=EDGCANNY_DEFAULT_GAUSSIAN_KERNEL_SIGMA);
    /// returns the default threshold value used in 'apply'
    virtual double getDefaultThreshold() const {return EDGCANNY_DEFAULT_THRESHOLD;}
    /// thresholded edge detection function; the threshold should be between 0 and 1 (will use default otherwise), and sets the base hysteresis threshold
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dThreshold=EDGCANNY_DEFAULT_THRESHOLD);
    /// edge detection function; returns a confidence edge mask (0-255) instead of a thresholded/binary edge mask
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask);

protected:
    /// base threshold multiplier used to compute the upper hysteresis threshold
    const double m_dHystLowThrshFactor;
    /// gaussian blur kernel sigma value
    const double m_dGaussianKernelSigma;
};
