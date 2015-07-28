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

    enum eParallelImplType {
#if HAVE_GLSL
        eParallelImpl_GLSL,
#endif //HAVE_GLSL
#if HAVE_CUDA
        eParallelImpl_CUDA,
#endif //HAVE_CUDA
#if HAVE_OPENCL
        eParallelImpl_OpenCL,
#endif //HAVE_OPENCL
        eParallelImpl_None
    };

    template<eParallelImplType eImpl=eParallelImpl_None, typename enable=void>
    struct ParallelImpl;

#if HAVE_GLSL
    template<eParallelImplType eImpl>
    struct ParallelImpl<eImpl, typename std::enable_if<eImpl==eParallelImpl_GLSL>::type> : public GLImageProcAlgo {
        ParallelImpl(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures, int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
            GLImageProcAlgo(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nOutputType,nDebugType,bUseInput,bUseDisplay,bUseTimers,bUseIntegralFormat) {}
        static constexpr bool hasParallelImpl() {return true;}
        static eParallelImplType getParallelImplType() {return eParallelImpl_GLSL;}
    };
    typedef ParallelImpl<eParallelImpl_GLSL> ParallelImpl_GLSL;
#endif //!HAVE_GLSL

#if HAVE_CUDA
    template<eParallelImplType eImpl>
    struct ParallelImpl<eImpl, typename std::enable_if<eImpl==eParallelImpl_CUDA>::type> {
        static_assert(false,"Missing CUDA impl");
        static constexpr bool hasParallelImpl() {return true;}
        static eParallelImplType getParallelImplType() {return eParallelImpl_CUDA;}
    };
    typedef ParallelImpl<eParallelImpl_CUDA> ParallelImpl_CUDA;
#endif //HAVE_CUDA

#if HAVE_CUDA
    template<eParallelImplType eImpl>
    struct ParallelImpl<eImpl, typename std::enable_if<eImpl==eParallelImpl_OpenCL>::type> {
        static_assert(false,"Missing OpenCL impl");
        static constexpr bool hasParallelImpl() {return true;}
        static eParallelImplType getParallelImplType() {return eParallelImpl_OpenCL;}
    };
    typedef ParallelImpl<eParallelImpl_OpenCL> ParallelImpl_OpenCL;
#endif //HAVE_CUDA

    template<eParallelImplType eImpl>
    struct ParallelImpl<eImpl, typename std::enable_if<eImpl==eParallelImpl_None>::type> {
        static constexpr bool hasParallelImpl() {return false;}
        static eParallelImplType getParallelImplType() {return eParallelImpl_None;}
    };
    typedef ParallelImpl<eParallelImpl_None> NoParallelImpl;

}; //namespace ParallelUtils
