
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
#include <vector>

namespace impl {

    __global__ void test_kernel(uchar* pTestData, int nVerbosity) {
        if(nVerbosity>=3)
            printf("internal warp size = %d, n = %d\n",warpSize,nVerbosity);
        assert(pTestData[0]==1);
        assert(pTestData[13]==0);
        pTestData[13] = 1;
    }

} // namespace impl

void device::test_kernel(const lv::cuda::KernelParams& oKParams, uchar* pTestData, int nVerbosity) {
    cudaKernelWrap(test_kernel,oKParams,pTestData,nVerbosity);
}