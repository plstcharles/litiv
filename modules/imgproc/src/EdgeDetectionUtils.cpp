
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

#include "litiv/imgproc/EdgeDetectionUtils.hpp"
#include <iostream>

#if HAVE_GLSL

template class IEdgeDetector<ParallelUtils::eGLSL>;

template<>
EdgeDetector_<ParallelUtils::eGLSL>::EdgeDetector_( size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                                                    size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                                                    bool bUseTimers, bool bUseIntegralFormat, size_t nROIBorderSize) :
        IEdgeDetector<ParallelUtils::eGLSL>(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nDebugType,bUseDisplay,bUseTimers,bUseIntegralFormat,nROIBorderSize),
        m_dCurrThreshold(-1) {}

template<>
void EdgeDetector_<ParallelUtils::eGLSL>::getLatestEdgeMask(cv::OutputArray _oLastEdgeMask) {
    _oLastEdgeMask.create(m_oFrameSize,CV_8UC1);
    cv::Mat oLastEdgeMask = _oLastEdgeMask.getMat();
    if(!GLImageProcAlgo::m_bFetchingOutput)
    glAssert(GLImageProcAlgo::setOutputFetching(true))
    GLImageProcAlgo::fetchLastOutput(oLastEdgeMask);
}

template<>
void EdgeDetector_<ParallelUtils::eGLSL>::apply_async_glimpl(cv::InputArray _oNextImage, bool bRebindAll, double dThreshold) {
    m_dCurrThreshold = dThreshold;
    cv::Mat oNextInputImg = _oNextImage.getMat();
    CV_Assert(oNextInputImg.size()==m_oFrameSize);
    CV_Assert(oNextInputImg.isContinuous());
    GLImageProcAlgo::apply_async(oNextInputImg,bRebindAll);
}

template<>
void EdgeDetector_<ParallelUtils::eGLSL>::apply_async(cv::InputArray oNextImage, double dThreshold) {
    apply_async_glimpl(oNextImage,false,dThreshold);
}

template<>
void EdgeDetector_<ParallelUtils::eGLSL>::apply_async(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, double dThreshold) {
    apply_async(oNextImage,dThreshold);
    getLatestEdgeMask(oLastEdgeMask);
}

template<>
void EdgeDetector_<ParallelUtils::eGLSL>::apply_threshold(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, double dThreshold) {
    CV_Assert(dThreshold>=0 && dThreshold<=1);
    apply_async(oNextImage,oLastEdgeMask,dThreshold);
}

template<>
void EdgeDetector_<ParallelUtils::eGLSL>::apply(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask) {
    apply_async(oNextImage,oLastEdgeMask,-1);
}

template class EdgeDetector_<ParallelUtils::eGLSL>;
#endif //HAVE_GLSL

#if HAVE_CUDA
template class IEdgeDetector<ParallelUtils::eCUDA>;
// ... @@@ add impl later
template class EdgeDetector_<ParallelUtils::eCUDA>;
#endif //HAVE_CUDA

#if HAVE_OPENCL
template class IEdgeDetector<ParallelUtils::eOpenCL>;
// ... @@@ add impl later
template class EdgeDetector_<ParallelUtils::eOpenCL>;
#endif //HAVE_OPENCL

template class IEdgeDetector<ParallelUtils::eNonParallel>;

template<>
EdgeDetector_<ParallelUtils::eNonParallel>::EdgeDetector_(size_t nROIBorderSize) :
        IEdgeDetector<ParallelUtils::eNonParallel>(nROIBorderSize) {}

template class EdgeDetector_<ParallelUtils::eNonParallel>;
