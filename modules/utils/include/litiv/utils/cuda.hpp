
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
#include <cstdio>
#define cudaKernelWrap(func,kparams,...) do { \
    device::func<<<kparams.vGridSize,kparams.vBlockSize,kparams.nSharedMemSize,kparams.nStream>>>(__VA_ARGS__); \
    const cudaError_t __errn = cudaGetLastError(); \
    if(__errn!=cudaSuccess) { \
        printf("cuda kernel '" #func "' execution failed [code=%d, msg=%s]\n\t... in function '%s'\n\t... from %s(%d)\n",(int)__errn,cudaGetErrorString(__errn),__PRETTY_FUNCTION__,__FILE__,__LINE__); \
        std::exit((int)__errn); \
    } \
} while(0)
#endif //def(__CUDACC__)
#include <cuda_runtime.h>

namespace lv {

    // @@@@ TODO, ship common cuda utils here

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
        };

#ifndef __CUDACC__

        /// used to launch a trivial kernel to test if device connection & compute arch are good
        extern void test(const lv::cuda::KernelParams& oKParams=lv::cuda::KernelParams(dim3(1),dim3(1)), int n=1);

#endif //ndef(__CUDACC__)

    } // namespace cuda

} // namespace lv
