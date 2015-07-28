#pragma once

#include "litiv/utils/ParallelUtils.hpp"
#include <opencv2/imgproc.hpp>

class EdgeDetectorImpl :
        public cv::Algorithm {
public:
    //! unused, always returns nullptr
    virtual cv::AlgorithmInfo* info() const;
    //! default edge detection function; the threshold should be between 0 and 1 (will use default otherwise)
    virtual void apply(cv::InputArray oInputImage, cv::OutputArray oEdgeMask, double dThreshold=-1) = 0;

    // #### for debug purposes only ####
    std::string m_sDebugName;
    cv::FileStorage* m_pDebugFS;
};

template<ParallelUtils::eParallelImplType eImpl=ParallelUtils::eParallelImpl_None, typename enable=void>
class EdgeDetectorParallelImpl;

#if HAVE_GLSL
template<ParallelUtils::eParallelImplType eImpl>
class EdgeDetectorParallelImpl<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_GLSL>::type> :
        public ParallelUtils::ParallelImpl_GLSL,
        public EdgeDetectorImpl {
public:
    //! glsl impl constructor
    EdgeDetectorParallelImpl(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs,
                             size_t nExtraImages, size_t nExtraTextures, int nDebugType, bool bUseDisplay,
                             bool bUseTimers, bool bUseIntegralFormat);

    //! returns a copy of the latest edge mask
    void getLatestEdgeMask(cv::OutputArray _oLastEdgeMask);
    //! edge detection function (asynchronous version, glsl interface); the threshold should be between 0 and 1 (will use default otherwise)
    void apply_async_glimpl(cv::InputArray _oNextImage, bool bRebindAll, double dThreshold=-1);
    //! edge detection function (asynchronous version); the threshold should be between 0 and 1 (will use default otherwise)
    void apply_async(cv::InputArray oNextImage, double dThreshold=-1);
    //! edge detection function (asynchronous version); the threshold should be between 0 and 1 (will use default otherwise)
    void apply_async(cv::InputArray oNextImage, cv::OutputArray oEdgeMask, double dThreshold=-1);
    //! overloads 'apply' from EdgeDetectorImpl and redirects it to apply_async
    virtual void apply(cv::InputArray oNextImage, cv::OutputArray oEdgeMask, double dThreshold=-1);

protected:
    //! used to pass 'apply' threshold parameter to overloaded dispatch call, if needed
    double m_dCurrThreshold;
};
typedef EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_GLSL> EdgeDetectorImpl_GLSL;
#endif //!HAVE_GLSL

#if HAVE_CUDA
template<ParallelUtils::eParallelImplType eImpl>
class EdgeDetectorParallelImpl<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_CUDA>::type> :
        public ParallelUtils::ParallelImpl_CUDA,
        public EdgeDetectorImpl {
        static_assert(false,"Missing CUDA impl");
};
typedef EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_CUDA> EdgeDetectorImpl_CUDA;
#endif //HAVE_CUDA

#if HAVE_OPENCL
template<ParallelUtils::eParallelImplType eImpl>
class EdgeDetectorParallelImpl<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_OpenCL>::type> :
        public ParallelUtils::ParallelImpl_OpenCL,
        public EdgeDetectorImpl {
        static_assert(false,"Missing OpenCL impl");
};
typedef EdgeDetectorParallelImpl<ParallelUtils::eParallelImpl_OpenCL> EdgeDetectorImpl_OpenCL;
#endif //HAVE_OPENCL

template<ParallelUtils::eParallelImplType eImpl>
class EdgeDetectorParallelImpl<eImpl, typename std::enable_if<eImpl==ParallelUtils::eParallelImpl_None>::type> :
        public ParallelUtils::NoParallelImpl,
        public EdgeDetectorImpl {};
