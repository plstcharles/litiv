
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2017 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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
//

#include "test.cuh"

__global__ void device::test(int n) {
    const int px = blockIdx.x*blockDim.x + threadIdx.x;
    const int py = blockIdx.y*blockDim.y + threadIdx.y;
    const int pz = blockIdx.z*blockDim.z + threadIdx.z;
    printf("cuda test kernel @ px = %d, py = %d, pz = %d, with n = %d\n",px,py,pz,n);
}

void host::test(const lv::cuda::KernelParams& oKParams, int n) {
    cudaKernelWrap(test,oKParams,n);
}

// for use via extern in litiv/utils/cuda.hpp
namespace lv {
    namespace cuda {
        void test(const lv::cuda::KernelParams& oKParams, int n) {
            host::test(oKParams,n);
        }
    }
}