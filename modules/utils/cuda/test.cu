
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

    __global__ void test(uchar* pTestData, int nVerbosity) {
        if(nVerbosity>=3)
            printf("internal warp size = %d, n = %d\n",warpSize,nVerbosity);
        assert(pTestData[0]==1);
        assert(pTestData[13]==0);
        pTestData[13] = 1;
    }

} // namespace impl

void device::test(const lv::cuda::KernelParams& oKParams, uchar* pTestData, int nVerbosity) {
    cudaKernelWrap(test,oKParams,pTestData,nVerbosity);
}

// for use via extern in litiv/utils/cuda.hpp
namespace lv {
    namespace cuda {
        void test(int nVerbosity) {
            if(nVerbosity>=2)
                printf("running cuda test kernel for device warmup...\n");
            const size_t nTestSize = size_t(10000);
            uchar* pTest_dev;
            cudaMalloc(&pTest_dev,nTestSize);
            cudaMemset(pTest_dev,1,nTestSize);
            std::vector<uchar> pTest_host(nTestSize);
            cudaMemcpy(pTest_host.data(),pTest_dev,nTestSize,cudaMemcpyDeviceToHost);
            pTest_host[13] = 0;
            cudaMemcpy(pTest_dev,pTest_host.data(),nTestSize,cudaMemcpyHostToDevice);
            device::test(lv::cuda::KernelParams(dim3(1),dim3(1)),pTest_dev,nVerbosity);
            cudaMemcpy(pTest_host.data(),pTest_dev,nTestSize,cudaMemcpyDeviceToHost);
            assert(pTest_host[13]==1);
            cudaFree(pTest_dev);
        }
    }
}