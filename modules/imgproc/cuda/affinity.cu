
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

#include "affinity.cuh"

__constant__ int g_anDispRange[AFF_MAP_DISP_RANGE_MAX];

namespace impl {

    __device__ inline void l2dist_lu_exec(device::DistCalcLUT* aDistCalcLUTs, float* aDescCalcLUT, float* aAffArray, int nOffsets, int nDescSize) {
        const int nThreads = blockDim.x;
        const int nThreadIdx = threadIdx.x;
        const int nStepPerDesc = __float2int_ru(float(nDescSize)/nThreads);
        assert(nStepPerDesc>0);
        const int nDescSize_LUT = nStepPerDesc*nThreads;
        assert(nDescSize_LUT>=nDescSize);
        for(int nOffsetIdx=0; nOffsetIdx<nOffsets; ++nOffsetIdx) {
            device::DistCalcLUT* pDistCalcLUT = aDistCalcLUTs+nOffsetIdx;
            if(pDistCalcLUT->aDesc1) {
                for(int nStep=0; nStep<nStepPerDesc; ++nStep) {
                    const int nDescBinIdx = nThreads*nStep + nThreadIdx;
                    assert(nDescBinIdx<nDescSize_LUT);
                    if(nDescBinIdx<nDescSize) {
                        const float fDescBinDiff = pDistCalcLUT->aDesc1[nDescBinIdx]-pDistCalcLUT->aDesc2[nDescBinIdx];
                        aDescCalcLUT[nDescBinIdx] = fDescBinDiff*fDescBinDiff;
                    }
                    else
                        aDescCalcLUT[nDescBinIdx] = 0.0f;
                }
                __syncthreads();
                if(nDescSize_LUT>nThreads) {
                    assert((nDescSize_LUT%nThreads)==0);
                    for(int nStep=nDescSize_LUT-nThreads; nStep>=nThreads; nStep-=nThreads)
                        aDescCalcLUT[nThreadIdx+(nStep-nThreads)] += aDescCalcLUT[nThreadIdx+nStep];
                    for(int nStep=nThreads/2; nStep>0; nStep>>=1)
                        aDescCalcLUT[nThreadIdx] += aDescCalcLUT[nThreadIdx+nStep];
                }
                else {
                    assert(nDescSize_LUT==nThreads);
                    for(int nStep=nThreads/2; nStep>0; nStep>>=1) {
                        if(nThreadIdx+nStep<nDescSize_LUT)
                            aDescCalcLUT[nThreadIdx] += aDescCalcLUT[nThreadIdx+nStep];
                        __syncthreads();
                    }
                }
                if(nThreadIdx==0)
                    aAffArray[nOffsetIdx] = sqrtf(aDescCalcLUT[0]);
            }
        }
    }

    __global__ void compute_desc_affinity_l2(const cv::cuda::PtrStep<float> oDescMap1,
                                             const cv::cuda::PtrStep<float> oDescMap2,
                                             cv::cuda::PtrStep<float> oAffinityMap,
                                             int nOffsets, int nDescSize) {
        assert((blockDim.x%warpSize)==0 && blockDim.y==1 && blockDim.z==1 && gridDim.z==1);
        assert(nDescSize>0 && nOffsets>0 && nOffsets<=blockDim.x && nOffsets<=AFF_MAP_DISP_RANGE_MAX);
        const int nCols = gridDim.x;
        const int nRowIdx = blockIdx.y;
        const int nColIdx = blockIdx.x;
        const int nThreads = blockDim.x;
        const int nThreadIdx = threadIdx.x;
        const int nStepPerPixel = __float2int_ru(float(nOffsets)/nThreads);
        assert(nStepPerPixel>0);
        const int nOffsets_LUT = nStepPerPixel*nThreads;
        assert(nOffsets_LUT>=nOffsets);
        float* aAffArray = oAffinityMap.ptr(nRowIdx*nCols+nColIdx);
        extern __shared__ int aTmpCommon_l2[];
        device::DistCalcLUT* aDistCalcLUTs = (device::DistCalcLUT*)aTmpCommon_l2;
        for(int nStep=0; nStep<nStepPerPixel; ++nStep) {
            const int nOffsetIdx = nThreads*nStep + nThreadIdx;
            assert(nOffsetIdx<nOffsets_LUT);
            device::DistCalcLUT* pDistCalcLUT = aDistCalcLUTs+nOffsetIdx;
            pDistCalcLUT->aDesc1 = nullptr;
            if(nOffsetIdx<nOffsets) {
                const int nOffsetColIdx = nColIdx+g_anDispRange[nOffsetIdx];
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) {
                    pDistCalcLUT->aDesc1 = oDescMap1.ptr(nRowIdx*nCols+nColIdx);
                    pDistCalcLUT->aDesc2 = oDescMap2.ptr(nRowIdx*nCols+nOffsetColIdx);
                }
                else
                    aAffArray[nOffsetIdx] = -1.0f;
            }
        }
        __syncthreads();
        float* aDescCalcLUT = (float*)(aDistCalcLUTs+nOffsets_LUT);
        l2dist_lu_exec(aDistCalcLUTs,aDescCalcLUT,aAffArray,nOffsets,nDescSize);
    }

    __global__ void compute_desc_affinity_l2_roi(const cv::cuda::PtrStep<float> oDescMap1,
                                                 const cv::cuda::PtrStep<uchar> oROI1,
                                                 const cv::cuda::PtrStep<float> oDescMap2,
                                                 const cv::cuda::PtrStep<uchar> oROI2,
                                                 cv::cuda::PtrStep<float> oAffinityMap,
                                                 int nOffsets, int nDescSize) {
        assert((blockDim.x%warpSize)==0 && blockDim.y==1 && blockDim.z==1 && gridDim.z==1);
        assert(nDescSize>0 && nOffsets>0 && nOffsets<=blockDim.x && nOffsets<=AFF_MAP_DISP_RANGE_MAX);
        const int nCols = gridDim.x;
        const int nRowIdx = blockIdx.y;
        const int nColIdx = blockIdx.x;
        const int nThreads = blockDim.x;
        const int nThreadIdx = threadIdx.x;
        const int nStepPerPixel = __float2int_ru(float(nOffsets)/nThreads);
        assert(nStepPerPixel>0);
        const int nOffsets_LUT = nStepPerPixel*nThreads;
        assert(nOffsets_LUT>=nOffsets);
        float* aAffArray = oAffinityMap.ptr(nRowIdx*nCols+nColIdx);
        if(oROI1(nRowIdx,nColIdx)==0) {
            for(int nStep=0; nStep<nStepPerPixel; ++nStep) {
                const int nOffsetIdx = nThreads*nStep+nThreadIdx;
                assert(nOffsetIdx<nOffsets_LUT);
                if(nOffsetIdx<nOffsets)
                    aAffArray[nOffsetIdx] = -1.0f;
            }
            return;
        }
        const uchar* aROI2Array = oROI2.ptr(nRowIdx);
        extern __shared__ int aTmpCommon_l2_roi[];
        device::DistCalcLUT* aDistCalcLUTs = (device::DistCalcLUT*)aTmpCommon_l2_roi;
        for(int nStep=0; nStep<nStepPerPixel; ++nStep) {
            const int nOffsetIdx = nThreads*nStep + nThreadIdx;
            assert(nOffsetIdx<nOffsets_LUT);
            device::DistCalcLUT* pDistCalcLUT = aDistCalcLUTs+nOffsetIdx;
            pDistCalcLUT->aDesc1 = nullptr;
            if(nOffsetIdx<nOffsets) {
                const int nOffsetColIdx = nColIdx+g_anDispRange[nOffsetIdx];
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && aROI2Array[nOffsetColIdx]!=0) {
                    pDistCalcLUT->aDesc1 = oDescMap1.ptr(nRowIdx*nCols+nColIdx);
                    pDistCalcLUT->aDesc2 = oDescMap2.ptr(nRowIdx*nCols+nOffsetColIdx);
                }
                else
                    aAffArray[nOffsetIdx] = -1.0f;
            }
        }
        __syncthreads();
        float* aDescCalcLUT = (float*)(aDistCalcLUTs+nOffsets_LUT);
        l2dist_lu_exec(aDistCalcLUTs,aDescCalcLUT,aAffArray,nOffsets,nDescSize);
    }

} // namespace impl

/////////////////////////////////////////////////////////////////////////

void device::compute_desc_affinity_l2(const lv::cuda::KernelParams& oKParams, const cv::cuda::PtrStep<float> oDescMap1, const cv::cuda::PtrStep<float> oDescMap2, cv::cuda::PtrStep<float> oAffinityMap, int nOffsets, int nDescSize) {
    cudaKernelWrap(compute_desc_affinity_l2,oKParams,oDescMap1,oDescMap2,oAffinityMap,nOffsets,nDescSize);
}

void device::compute_desc_affinity_l2_roi(const lv::cuda::KernelParams& oKParams, const cv::cuda::PtrStep<float> oDescMap1, const cv::cuda::PtrStep<uchar> oROI1, const cv::cuda::PtrStep<float> oDescMap2, const cv::cuda::PtrStep<uchar> oROI2, cv::cuda::PtrStep<float> oAffinityMap, int nOffsets, int nDescSize) {
    cudaKernelWrap(compute_desc_affinity_l2_roi,oKParams,oDescMap1,oROI1,oDescMap2,oROI2,oAffinityMap,nOffsets,nDescSize);
}

void device::setDisparityRange(const std::array<int,AFF_MAP_DISP_RANGE_MAX>& aDispRange) {
    cudaErrorCheck(cudaMemcpyToSymbol(g_anDispRange,aDispRange.data(),sizeof(int)*aDispRange.size()));
}