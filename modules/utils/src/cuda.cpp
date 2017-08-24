
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

#include "litiv/utils/cuda.hpp"
#include "test.cuh"

int g_nDefaultCUDADeviceID = 0;

void lv::cuda::setDefaultDeviceID(int nDeviceID) {
    g_nDefaultCUDADeviceID = nDeviceID;
}

int lv::cuda::getDefaultDeviceID() {
    return g_nDefaultCUDADeviceID;
}

void lv::cuda::test_kernel(int nVerbosity) {
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
    device::test_kernel(lv::cuda::KernelParams(dim3(1),dim3(1)),pTest_dev,nVerbosity);
    cudaMemcpy(pTest_host.data(),pTest_dev,nTestSize,cudaMemcpyDeviceToHost);
    assert(pTest_host[13]==1);
    cudaFree(pTest_dev);
}

void lv::cuda::init(int nDeviceID) {
    const int nDeviceCount = cv::cuda::getCudaEnabledDeviceCount();
    lvAssert_(nDeviceCount>0,"no valid cuda-enabled device found on system");
    lvAssert__(nDeviceCount>nDeviceID,"provided device ID out of range (device count=%d)",nDeviceCount);
    cv::cuda::setDevice(nDeviceID);
    lvAssert(cv::cuda::getDevice()==nDeviceID);
    lvAssert_(cv::cuda::deviceSupports(LITIV_CUDA_MIN_COMPUTE_CAP),"device does not support min compute capabilities required by framework");
    if(lv::getVerbosity()>=2) {
        lvCout << "Initialized CUDA-enabled device w/ id=" << nDeviceID << std::endl;
        cv::cuda::printShortCudaDeviceInfo(nDeviceID);
        const cv::cuda::DeviceInfo oInfo;
        lvCout << "warp size = " << oInfo.warpSize() << ", async eng count = " << oInfo.asyncEngineCount() << ", concurrent kernels = " << oInfo.concurrentKernels() << std::endl;
    }
    lv::cuda::test_kernel(lv::getVerbosity()); // warm-up kernel; should always succeed
}