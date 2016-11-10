
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

#include "litiv/utils/parallel.hpp"
#include "litiv/utils/opencv.hpp"

/// super-interface for edge detection algos which exposes common interface functions
struct IIEdgeDetector : public cv::Algorithm {

    /// returns the default threshold value used in 'apply'
    virtual double getDefaultThreshold() const = 0;
    /// edge detection function; the threshold should be between 0 and 1 (returns a binary edge map)
    virtual void apply_threshold(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dThreshold) = 0;
    /// edge detection function; performs a full sensitivty sweep and returns a non-binary (grayscale confidence) edge map
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask) = 0;
    /// required for derived class destruction from this interface
    virtual ~IIEdgeDetector() = default;

protected:
    /// default impl constructor (for common parameters only -- none must be const to avoid constructor hell when deriving)
    IIEdgeDetector();
    /// ROI border size to be ignored, useful for descriptor-based methods
    size_t m_nROIBorderSize;
private:
    IIEdgeDetector& operator=(const IIEdgeDetector&) = delete;
    IIEdgeDetector(const IIEdgeDetector&) = delete;
};

template<lv::ParallelAlgoType eImpl>
struct IEdgeDetector_;

#if HAVE_GLSL
template<>
struct IEdgeDetector_<lv::GLSL> :
        public lv::IParallelAlgo_GLSL,
        public IIEdgeDetector {

    /// required for derived class destruction from this interface
    virtual ~IEdgeDetector_() {}
    /// returns a copy of the latest edge mask
    void getLatestEdgeMask(cv::OutputArray oLastEdgeMask);

    // @@@@ also add init funcs here?

    /// edge detection function (asynchronous version w/ gl interface); the threshold should be between 0 and 1, or -1 for sweep
    void apply_gl(cv::InputArray oNextImage, bool bRebindAll, double dThreshold);
    /// edge detection function (asynchronous version w/ gl interface); the threshold should be between 0 and 1, or -1 for sweep
    void apply_gl(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, bool bRebindAll, double dThreshold);
    /// overloads 'apply_threshold' from IIEdgeDetector and redirects it to apply_gl
    virtual void apply_threshold(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, double dThreshold) override final;
    /// overloads 'apply' from IIEdgeDetector and redirects it to apply_gl (with threshold = -1)
    virtual void apply(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask) override final;

protected:
    /// glsl impl constructor
    IEdgeDetector_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                   size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                   bool bUseTimers, bool bUseIntegralFormat);
    /// used to pass threshold parameter to overridden dispatch call, if needed
    double m_dCurrThreshold;
    /// eliminates 'hides overloaded virtual func' warning on some platforms
    using lv::IParallelAlgo_GLSL::apply_gl;
};

using IEdgeDetector_GLSL = IEdgeDetector_<lv::GLSL>;
#endif //HAVE_GLSL

#if HAVE_CUDA
// IEdgeDetector_<lv::CUDA> will not compile here, missing impl
#endif //HAVE_CUDA

#if HAVE_OPENCL
// IEdgeDetector_<lv::OpenCL> will not compile here, missing impl
#endif //HAVE_OPENCL

template<>
struct IEdgeDetector_<lv::NonParallel> :
        public lv::NonParallelAlgo,
        public IIEdgeDetector {
    /// required for derived class destruction from this interface
    virtual ~IEdgeDetector_() {}
};

using IEdgeDetector = IEdgeDetector_<lv::NonParallel>;
