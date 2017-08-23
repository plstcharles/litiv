
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
#include "litiv/utils/cuda.hpp"
#endif //HAVE_CUDA
#endif //USE_CVCORE_WITH_UTILS

namespace lv {

    /// parallel algo type list; used for class template specialization
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

    /// abstract parallel algo interface; will expose some common low-level utils
    struct IIParallelAlgo {
        /// returns whether the algorithm is implemented for parallel processing or not
        virtual bool isParallel() const = 0;
        /// returns which type of parallel implementation is used in this algo
        virtual ParallelAlgoType getParallelAlgoType() const = 0;
#if USE_CVCORE_WITH_UTILS
        // #### for debug purposes only ####
        lv::DisplayHelperPtr m_pDisplayHelper;
#endif //USE_CVCORE_WITH_UTILS
    };

    /// parallel algo interface specialization forward declaration
    template<ParallelAlgoType eImpl>
    struct IParallelAlgo_;

#if USE_CVCORE_WITH_UTILS

#if HAVE_GLSL
    /// GLSL algo interface specialization; exposes forwarding constructor to GLImageProcAlgo class to initialize its const members
    template<>
    struct IParallelAlgo_<GLSL> : public GLImageProcAlgo, public IIParallelAlgo {
        /// default constructor; receives all requires members for the GLImageProcAlgo interface (all of which will be constant in the impl)
        IParallelAlgo_(size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures, int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
                GLImageProcAlgo(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nOutputType,nDebugType,bUseInput,bUseDisplay,bUseTimers,bUseIntegralFormat) {}
        /// returns whether the algorithm is implemented for parallel processing or not
        virtual bool isParallel() const override {return true;}
        /// returns which type of parallel implementation is used in this algo
        virtual ParallelAlgoType getParallelAlgoType() const override {return GLSL;}
    };
    using IParallelAlgo_GLSL = IParallelAlgo_<GLSL>;
#endif //(!HAVE_GLSL)

#if HAVE_CUDA
    /// CUDA algo interface specialization; exposes some common API utilities and on/off toggle
    template<>
    struct IParallelAlgo_<CUDA> : /*public CUDAImageProcAlgo @@@ TODO?,*/ public IIParallelAlgo {
        // add GPU selection util here? @@@
        /// default constructor; provides the initial state for the CUDA toggle flag (default = on)
        IParallelAlgo_(bool bInitUseCUDA=true, int nDeviceID=lv::cuda::getDefaultDeviceID()) :
                m_bUseCUDA(bInitUseCUDA),m_bCUDAInitialized(false),m_nDeviceID(-1) {
            if(m_bUseCUDA)
                tryInitEnableCUDA(nDeviceID);
        }
        /// attempts to initialize a CUDA context with the given device ID, returning whether successful or not
        bool tryInitEnableCUDA(int nDeviceID=lv::cuda::getDefaultDeviceID()) {
            try {
                lv::cuda::init(nDeviceID);
                m_pDeviceInfo = std::make_unique<cv::cuda::DeviceInfo>();
                m_nDeviceID = nDeviceID;
                lvDbgAssert(m_pDeviceInfo->deviceID()==nDeviceID);
                return (m_bCUDAInitialized=(m_bUseCUDA=true));
            } catch(...) {
                lvWarn("CUDA init failed, algo impl will not use GPU");
                m_pDeviceInfo = nullptr;
                m_nDeviceID = -1;
                return (m_bCUDAInitialized=(m_bUseCUDA=false));
            }
        }
        /// sets whether CUDA should be enabled for this algo or not (must be initialized prior to this call)
        void enableCUDA(bool bVal) {
            if(bVal && !m_bUseCUDA) {
                lvAssert_(m_bCUDAInitialized,"CUDA must have been initialized already before enabling algo");
                m_bUseCUDA = true;
            }
            else if(!bVal && m_bUseCUDA)
                m_bUseCUDA = false;
        }
        /// returns whether the algorithm is implemented and enabled for parallel processing or not
        virtual bool isParallel() const override {return m_bUseCUDA;}
        /// returns which type of parallel implementation is used in this algo
        virtual ParallelAlgoType getParallelAlgoType() const override {return m_bUseCUDA?CUDA:NonParallel;}
    protected:
        /// defines whether the CUDA impl should be used internally or not (internal toggle)
        bool m_bUseCUDA;
        /// defines whether the CUDA context was already successfully initialized or not
        bool m_bCUDAInitialized;
        /// defines the device used to initialize the current cuda context for this object
        int m_nDeviceID;
        /// holds other device information utilities tied to the 'm_nDeviceID' device context
        std::unique_ptr<cv::cuda::DeviceInfo> m_pDeviceInfo;
    };
    using IParallelAlgo_CUDA = IParallelAlgo_<CUDA>;
#endif //HAVE_CUDA

#if HAVE_OPENCL
    /// todo @@@
    template<>
    struct IParallelAlgo_<OpenCL> : /*public OpenCLImageProcAlgo,*/ public IIParallelAlgo {
        static_assert(false,"Missing OpenCL impl");
        virtual bool isParallel() const override {return true;}
        virtual ParallelAlgoType getParallelAlgoType() const override {return OpenCL;}
    };
    using IParallelAlgo_OpenCL = IParallelAlgo_<OpenCL>;
#endif //HAVE_OPENCL

#endif //USE_CVCORE_WITH_UTILS

    /// default (non-parallel) algo interface specialization; overrides virtual pure functions from base class only
    template<>
    struct IParallelAlgo_<NonParallel> : public IIParallelAlgo {
        /// returns whether the algorithm is implemented for parallel processing or not
        virtual bool isParallel() const override final {return false;}
        /// returns which type of parallel implementation is used in this algo
        virtual ParallelAlgoType getParallelAlgoType() const override final {return NonParallel;}
#if HAVE_CUDA
    protected:
        /// defines whether the CUDA impl should be used internally or not (always false in this case)
        static constexpr bool m_bUseCUDA = false;
#endif //HAVE_CUDA
    };
    using NonParallelAlgo = IParallelAlgo_<NonParallel>;

} // namespace lv
