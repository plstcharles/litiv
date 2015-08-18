#pragma once

#include "litiv/utils/DefineUtils.hpp"
#include <type_traits>

#if HAVE_GLSL
#include "litiv/utils/GLImageProcUtils.hpp"
#endif //HAVE_GLSL
#if HAVE_CUDA
#include @@@@@
#endif //HAVE_CUDA
#if HAVE_OPENCL
#include @@@@@
#endif //HAVE_OPENCL

#if HAVE_SIMD_SUPPORT
#if defined(_MSC_VER)
#include <intrin.h>
#else //!defined(_MSC_VER)
#include <x86intrin.h>
#endif //!defined(_MSC_VER)
#endif //HAVE_SIMD_SUPPORT

#if DEFAULT_NB_THREADS<1
#error "Bad default number of threads specified."
#endif //DEFAULT_NB_THREADS<1

namespace ParallelUtils {

    enum eParallelAlgoType {
#if HAVE_GLSL
        eGLSL,
#endif //HAVE_GLSL
#if HAVE_CUDA
        eCUDA,
#endif //HAVE_CUDA
#if HAVE_OPENCL
        eOpenCL,
#endif //HAVE_OPENCL
        eNonParallel
    };

    template<eParallelAlgoType eImpl=eNonParallel, typename enable=void>
    struct ParallelAlgo_;

    struct IParallelAlgo {
        //! returns whether the algorithm is implemented for parallel processing or not
        virtual bool isParallel() = 0;
        //! returns which type of parallel implementation is used in this algo
        virtual eParallelAlgoType getParallelAlgoType() = 0;
    };

#if HAVE_GLSL
    template<eParallelAlgoType eImpl>
    struct ParallelAlgo_<eImpl, typename std::enable_if<eImpl==eGLSL>::type> : public GLImageProcAlgo, public IParallelAlgo {
        ParallelAlgo_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures, int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
            GLImageProcAlgo(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nOutputType,nDebugType,bUseInput,bUseDisplay,bUseTimers,bUseIntegralFormat) {}
        virtual bool isParallel() {return true;}
        virtual eParallelAlgoType getParallelAlgoType() {return eGLSL;}
    };
    typedef ParallelAlgo_<eGLSL> GLSLAlgo;
#endif //!HAVE_GLSL

#if HAVE_CUDA
    template<eParallelAlgoType eImpl>
    struct ParallelAlgo_<eImpl, typename std::enable_if<eImpl==eCUDA>::type> : public IParallelAlgo {
        static_assert(false,"Missing CUDA impl");
        virtual bool isParallel() {return true;}
        virtual eParallelAlgoType getParallelAlgoType() {return eCUDA;}
    };
    typedef ParallelAlgo_<eCUDA> CUDAAlgo;
#endif //HAVE_CUDA

#if HAVE_CUDA
    template<eParallelAlgoType eImpl>
    struct ParallelAlgo_<eImpl, typename std::enable_if<eImpl==eOpenCL>::type> : public IParallelAlgo {
        static_assert(false,"Missing OpenCL impl");
        virtual bool isParallel() {return true;}
        virtual eParallelAlgoType getParallelAlgoType() {return eOpenCL;}
    };
    typedef ParallelAlgo_<eOpenCL> OpenCLAlgo;
#endif //HAVE_CUDA

    template<eParallelAlgoType eImpl>
    struct ParallelAlgo_<eImpl, typename std::enable_if<eImpl==eNonParallel>::type> : public IParallelAlgo {
        ParallelAlgo_() {}
        virtual bool isParallel() {return false;}
        virtual eParallelAlgoType getParallelAlgoType() {return eNonParallel;}
    };
    typedef ParallelAlgo_<eNonParallel> NonParallelAlgo;

} //namespace ParallelUtils
