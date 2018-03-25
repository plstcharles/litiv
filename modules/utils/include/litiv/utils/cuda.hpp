
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

#ifdef __CUDACC__
#include "litiv/utils/cudev/common.hpp"
#include "litiv/utils/cudev/vec_traits.hpp"
#include <curand_kernel.h>
#else //ndef(__CUDACC__)
#include "litiv/utils/cxx.hpp"
#if !HAVE_CUDA
#error "cuda util header included without cuda support in framework"
#endif //!HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <curand.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include "litiv/utils/opencv.hpp"
#endif //ndef(__CUDACC__)

namespace lv {

    // notes:
    //   warp leader selection code for divergent code flow: (requires sm_20)
    //     int mask = __ballot(1);  // mask of active lanes
    //     int leader = __ffs(mask) - 1;  // -1 for 0-based indexing

    namespace cuda {

        /// container for cuda kernel execution configuration parameters
        struct KernelParams {
            KernelParams() :
                    vGridSize(0),vBlockSize(0),nSharedMemSize(0),pStream(nullptr) {}
            KernelParams(dim3 vGridSize_, dim3 vBlockSize_, size_t nSharedMemSize_=0, cudaStream_t pStream_=nullptr) :
                    vGridSize(std::move(vGridSize_)),vBlockSize(std::move(vBlockSize_)),nSharedMemSize(nSharedMemSize_),pStream(pStream_) {} // NOLINT
            dim3 vGridSize; ///< kernel grid size (i.e. number of block instantiations in x,y,z dimensions)
            dim3 vBlockSize; ///< kernel block size (i.e. number of threads in x,y,z dimensions)
            size_t nSharedMemSize; ///< size of the dynamic shared memory used by the kernel (in bytes)
            cudaStream_t pStream; ///< kernel execution stream context (allows async processing if non-null)
            /// is-equal test operator for other KernelParams structs
            bool operator==(const KernelParams& o) const {
                return
                    vGridSize.x==o.vGridSize.x && vGridSize.y==o.vGridSize.y && vGridSize.z==o.vGridSize.z &&
                    vBlockSize.x==o.vBlockSize.x && vBlockSize.y==o.vBlockSize.y && vBlockSize.z==o.vBlockSize.z &&
                    nSharedMemSize==o.nSharedMemSize &&
                    pStream==o.pStream;
            }
            /// is-not-equal test operator for other KernelParams structs
            bool operator!=(const KernelParams& o) const {
                return !(*this==o);
            }
            /// implicit conversion op to string (for printing/debug purposes only)
            operator std::string() const {
                std::stringstream ssStr;
                ssStr << "{ ";
                ssStr << "grid=[" << vGridSize.x << "," << vGridSize.y << "," << vGridSize.z << "], ";
                ssStr << "block=[" << vBlockSize.x << "," << vBlockSize.y << "," << vBlockSize.z << "], ";
                ssStr << "shmem=" << nSharedMemSize << ", stream=" << (void*)pStream;
                ssStr << " }";
                return ssStr.str();
            }
            /// returns the result of the implicit std::string cast (for printing/debug purposes only)
            std::string str() const {
                return (std::string)*this;
            }
        };

#ifndef __CUDACC__

        /// used to set a global default device ID to use in CUDA-based impls
        void setDefaultDeviceID(int nDeviceID);
        /// used to query the global default device ID to use in CUDA-based impls
        int getDefaultDeviceID();
        /// used to launch a trivial kernel to test if device connection & compute arch are good
        void test_kernel(int nVerbosity);
        /// initializes cuda on the given device id for the current thread, and checks for compute compatibility
        void init(int nDeviceID=getDefaultDeviceID(), bool bReset=false);

#endif //ndef(__CUDACC__)

    } // namespace cuda

} // namespace lv
