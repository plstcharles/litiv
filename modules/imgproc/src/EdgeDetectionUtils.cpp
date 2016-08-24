
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

IIEdgeDetector::IIEdgeDetector() :
        m_nROIBorderSize(0) {}

#if HAVE_GLSL

void IEdgeDetector_GLSL::getLatestEdgeMask(cv::OutputArray _oLastEdgeMask) {
    lvAssert_(GLImageProcAlgo::m_bGLInitialized,"algo must be initialized first");
    _oLastEdgeMask.create(GLImageProcAlgo::m_oFrameSize,CV_8UC1);
    cv::Mat oLastEdgeMask = _oLastEdgeMask.getMat();
    lvAssert_(GLImageProcAlgo::m_bFetchingOutput || GLImageProcAlgo::setOutputFetching(true),"algo not initialized with mat output support")
    if(GLImageProcAlgo::m_nInternalFrameIdx>0)
        GLImageProcAlgo::fetchLastOutput(oLastEdgeMask);
    else
        oLastEdgeMask = cv::Scalar_<uchar>(0);
}

void IEdgeDetector_GLSL::apply_gl(cv::InputArray _oNextImage, bool bRebindAll, double dThreshold) {
    lvAssert_(GLImageProcAlgo::m_bGLInitialized,"algo must be initialized first");
    m_dCurrThreshold = dThreshold;
    cv::Mat oNextInputImg = _oNextImage.getMat();
    lvAssert_(oNextInputImg.size()==GLImageProcAlgo::m_oFrameSize,"input image size must match initialization size");
    lvAssert_(oNextInputImg.isContinuous(),"input image must be continuous");
    GLImageProcAlgo::apply_gl(oNextInputImg,bRebindAll);
}

void IEdgeDetector_GLSL::apply_gl(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, bool bRebindAll, double dThreshold) {
    apply_gl(oNextImage,bRebindAll,dThreshold);
    getLatestEdgeMask(oLastEdgeMask);
}

void IEdgeDetector_GLSL::apply_threshold(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, double dThreshold) {
    apply_gl(oNextImage,oLastEdgeMask,false,dThreshold);
}

void IEdgeDetector_GLSL::apply(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask) {
    apply_gl(oNextImage,oLastEdgeMask,false,-1);
}

IEdgeDetector_GLSL::IEdgeDetector_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                                   size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                                   bool bUseTimers, bool bUseIntegralFormat) :
        lv::IParallelAlgo_GLSL(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,CV_8UC1,nDebugType,true,bUseDisplay,bUseTimers,bUseIntegralFormat),
        m_dCurrThreshold(-1) {}

#endif //HAVE_GLSL
