
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
#include <array>
#include <cstdio>
#include <sstream>
#include <cassert>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/limits.hpp>
#include <opencv2/cudev.hpp>
#ifdef CUDA_EXIT_ON_ERROR
#define CUDA_ERROR_HANDLER(errn,msg) do { printf("%s",msg); std::exit(errn); } while(false)
#else //ndef(CUDA_EXIT_ON_ERROR)
#define CUDA_ERROR_HANDLER(errn,msg) do { (void)errn; throw std::runtime_error(msg); } while(false)
#endif //ndef(CUDA_..._ON_ERROR)
#define cudaKernelWrap(func,kparams,...) do { \
        impl::func<<<kparams.vGridSize,kparams.vBlockSize,kparams.nSharedMemSize,kparams.nStream>>>(__VA_ARGS__); \
        const cudaError_t __errn = cudaGetLastError(); \
        if(__errn!=cudaSuccess) { \
            std::array<char,1024> acBuffer; \
            snprintf(acBuffer.data(),acBuffer.size(),"cuda kernel '" #func "' execution failed [code=%d, msg=%s]\n\t... in function '%s'\n\t... from %s(%d)\n\t... with kernel params = %s\n", \
                     (int)__errn,cudaGetErrorString(__errn),__PRETTY_FUNCTION__,__FILE__,__LINE__,kparams.str().c_str()); \
            CUDA_ERROR_HANDLER((int)__errn,acBuffer.data()); \
        } \
    } while(false)
#else //ndef(__CUDACC__)
#include "litiv/utils/cxx.hpp"
#if !HAVE_CUDA
#error "cuda util header included without cuda support in framework"
#endif //!HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <npp.h>
#include "litiv/utils/opencv.hpp"
#endif //ndef(__CUDACC__)

namespace lv {

    // @@@@ TODO, ship common cuda utils here

    // warp leader selection code for divergent code flow: (requires sm_20)
    //     int mask = __ballot(1);  // mask of active lanes
    //     int leader = __ffs(mask) - 1;  // -1 for 0-based indexing

    namespace cuda {

        /// container for cuda kernel execution configuration parameters
        struct KernelParams {
            KernelParams() :
                    vGridSize(0),vBlockSize(0),nSharedMemSize(0),nStream(0) {}
            KernelParams(const dim3& _vGridSize, const dim3& _vBlockSize, size_t _nSharedMemSize=0, cudaStream_t _nStream=0) :
                    vGridSize(_vGridSize),vBlockSize(_vBlockSize),nSharedMemSize(_nSharedMemSize),nStream(_nStream) {}
            dim3 vGridSize;
            dim3 vBlockSize;
            size_t nSharedMemSize;
            cudaStream_t nStream;
            /// is-equal test operator for other KernelParams structs
            bool operator==(const KernelParams& o) const {
                return
                    vGridSize.x==o.vGridSize.x && vGridSize.y==o.vGridSize.y && vGridSize.z==o.vGridSize.z &&
                    vBlockSize.x==o.vBlockSize.x && vBlockSize.y==o.vBlockSize.y && vBlockSize.z==o.vBlockSize.z &&
                    nSharedMemSize==o.nSharedMemSize &&
                    nStream==o.nStream;
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
                ssStr << "shmem=" << nSharedMemSize << ", stream=" << (void*)nStream;
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
        void init(int nDeviceID=getDefaultDeviceID());

#endif //ndef(__CUDACC__)

    } // namespace cuda

} // namespace lv
