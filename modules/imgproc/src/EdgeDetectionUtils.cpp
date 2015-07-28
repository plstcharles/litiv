#include "litiv/imgproc/EdgeDetectionUtils.hpp"
#include <iostream>

cv::AlgorithmInfo* EdgeDetectorImpl::info() const {
    return nullptr;
}

#if HAVE_GLSL

template<>
EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::EdgeDetectorParallelImpl( size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs,
                                                                                       size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures,
                                                                                       int nDebugType, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
    ParallelUtils::ParallelImpl_GLSL(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,CV_8UC1,nDebugType,true,bUseDisplay,bUseTimers,bUseIntegralFormat),
    m_dCurrThreshold(-1) {}

template<>
void EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::getLatestEdgeMask(cv::OutputArray _oLastEdgeMask) {
    _oLastEdgeMask.create(m_oFrameSize,CV_8UC1);
    cv::Mat oLastEdgeMask = _oLastEdgeMask.getMat();
    if(!GLImageProcAlgo::m_bFetchingOutput)
    glAssert(GLImageProcAlgo::setOutputFetching(true))
    GLImageProcAlgo::fetchLastOutput(oLastEdgeMask);
};

template<>
void EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply_async_glimpl(cv::InputArray _oNextImage, bool bRebindAll, double dThreshold) {
    m_dCurrThreshold = dThreshold;
    cv::Mat oNextInputImg = _oNextImage.getMat();
    CV_Assert(oNextInputImg.size()==m_oFrameSize);
    CV_Assert(oNextInputImg.isContinuous());
    GLImageProcAlgo::apply_async(oNextInputImg,bRebindAll);
};

template<>
void EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply_async(cv::InputArray oNextImage, double dThreshold) {
    apply_async_glimpl(oNextImage,false,dThreshold);
};

template<>
void EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply_async(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, double dThreshold) {
    apply_async(oNextImage,dThreshold);
    getLatestEdgeMask(oLastEdgeMask);
};

template<>
void EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL>::apply(cv::InputArray oNextImage, cv::OutputArray oLastEdgeMask, double dThreshold) {
    apply_async(oNextImage,oLastEdgeMask,dThreshold);
}

template class EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL>;
#endif //HAVE_GLSL

#if HAVE_CUDA
// ... @@@ add impl later
//template class EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_CUDA>;
#endif //HAVE_CUDA

#if HAVE_OPENCL
// ... @@@ add impl later
//template class EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_OpenCL>;
#endif //HAVE_OPENCL

template<>
EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_None>::EdgeDetectorParallelImpl() {}

template class EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_None>;
