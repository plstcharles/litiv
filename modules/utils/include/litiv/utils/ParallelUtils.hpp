
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

#include "litiv/utils/DefineUtils.hpp"
#include "litiv/utils/OpenCVUtils.hpp"
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
#else //(!defined(_MSC_VER))
#include <x86intrin.h>
#endif //(!defined(_MSC_VER))
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

    struct IIParallelAlgo {
        //! returns whether the algorithm is implemented for parallel processing or not
        virtual bool isParallel() = 0;
        //! returns which type of parallel implementation is used in this algo
        virtual eParallelAlgoType getParallelAlgoType() = 0;
    public:
        // #### for debug purposes only ####
        cv::DisplayHelperPtr m_pDisplayHelper;
    };

    template<eParallelAlgoType eImpl>
    struct IParallelAlgo_;

#if HAVE_GLSL
    template<>
    struct IParallelAlgo_<eGLSL> : public GLImageProcAlgo, public IIParallelAlgo {
        IParallelAlgo_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures, int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
            GLImageProcAlgo(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nOutputType,nDebugType,bUseInput,bUseDisplay,bUseTimers,bUseIntegralFormat) {}
        virtual bool isParallel() {return true;}
        virtual eParallelAlgoType getParallelAlgoType() {return eGLSL;}
    };
    using IParallelAlgo_GLSL = IParallelAlgo_<eGLSL>;
#endif //(!HAVE_GLSL)

#if HAVE_CUDA
    template<>
    struct IParallelAlgo_<eCUDA> : /*public CUDAImageProcAlgo,*/ public IIParallelAlgo {
        static_assert(false,"Missing CUDA impl");
        virtual bool isParallel() {return true;}
        virtual eParallelAlgoType getParallelAlgoType() {return eCUDA;}
    };
    using IParallelAlgo_CUDA = IParallelAlgo_<eCUDA>;
#endif //HAVE_CUDA

#if HAVE_CUDA
    template<>
    struct IParallelAlgo_<eOpenCL> : /*public OpenCLImageProcAlgo,*/ public IIParallelAlgo {
        static_assert(false,"Missing OpenCL impl");
        virtual bool isParallel() {return true;}
        virtual eParallelAlgoType getParallelAlgoType() {return eOpenCL;}
    };
    using IParallelAlgo_OpenCL = IParallelAlgo_<eOpenCL>;
#endif //HAVE_CUDA

    template<>
    struct IParallelAlgo_<eNonParallel> : public IIParallelAlgo {
        IParallelAlgo_() {}
        virtual bool isParallel() {return false;}
        virtual eParallelAlgoType getParallelAlgoType() {return eNonParallel;}
    };
    using NonParallelAlgo = IParallelAlgo_<eNonParallel>;

} //namespace ParallelUtils
