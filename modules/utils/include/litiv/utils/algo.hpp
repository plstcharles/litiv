
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

#include "litiv/utils/math.hpp"
#if USE_CVCORE_WITH_UTILS
#include "litiv/utils/opencv.hpp"
#if HAVE_GLSL
#include "litiv/utils/opengl-imgproc.hpp"
#endif //HAVE_GLSL
#if HAVE_CUDA
// ...
#endif //HAVE_CUDA
#if HAVE_OPENCL
// ...
#endif //HAVE_OPENCL
#endif //USE_CVCORE_WITH_UTILS

namespace lv {

    enum ParallelAlgoType {
#if HAVE_GLSL
        GLSL,
#endif //HAVE_GLSL
#if HAVE_CUDA
        CUDA,
#endif //HAVE_CUDA
#if HAVE_OPENCL
        OpenCL,
#endif //HAVE_OPENCL
        NonParallel
    };

    struct IIParallelAlgo {
        /// returns whether the algorithm is implemented for parallel processing or not
        virtual bool isParallel() = 0;
        /// returns which type of parallel implementation is used in this algo
        virtual ParallelAlgoType getParallelAlgoType() = 0;
#if USE_CVCORE_WITH_UTILS
        // #### for debug purposes only ####
        cv::DisplayHelperPtr m_pDisplayHelper;
#endif //USE_CVCORE_WITH_UTILS
    };

    template<ParallelAlgoType eImpl>
    struct IParallelAlgo_;

#if USE_CVCORE_WITH_UTILS

#if HAVE_GLSL
    template<>
    struct IParallelAlgo_<GLSL> : public GLImageProcAlgo, public IIParallelAlgo {
        IParallelAlgo_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures, int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
            GLImageProcAlgo(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nOutputType,nDebugType,bUseInput,bUseDisplay,bUseTimers,bUseIntegralFormat) {}
        virtual bool isParallel() {return true;}
        virtual ParallelAlgoType getParallelAlgoType() {return GLSL;}
    };
    using IParallelAlgo_GLSL = IParallelAlgo_<GLSL>;
#endif //(!HAVE_GLSL)

#if HAVE_CUDA
    template<>
    struct IParallelAlgo_<CUDA> : /*public CUDAImageProcAlgo,*/ public IIParallelAlgo {
        static_assert(false,"Missing CUDA impl");
        virtual bool isParallel() {return true;}
        virtual ParallelAlgoType getParallelAlgoType() {return CUDA;}
    };
    using IParallelAlgo_CUDA = IParallelAlgo_<CUDA>;
#endif //HAVE_CUDA

#if HAVE_CUDA
    template<>
    struct IParallelAlgo_<OpenCL> : /*public OpenCLImageProcAlgo,*/ public IIParallelAlgo {
        static_assert(false,"Missing OpenCL impl");
        virtual bool isParallel() {return true;}
        virtual ParallelAlgoType getParallelAlgoType() {return OpenCL;}
    };
    using IParallelAlgo_OpenCL = IParallelAlgo_<OpenCL>;
#endif //HAVE_CUDA

#endif //USE_CVCORE_WITH_UTILS

    template<>
    struct IParallelAlgo_<NonParallel> : public IIParallelAlgo {
        virtual bool isParallel() {return false;}
        virtual ParallelAlgoType getParallelAlgoType() {return NonParallel;}
    };
    using NonParallelAlgo = IParallelAlgo_<NonParallel>;

} // namespace lv
