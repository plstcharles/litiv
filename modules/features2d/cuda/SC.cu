
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

#include "SC.cuh"

__global__ void device::scdesc_fill_desc_direct(const cv::cuda::PtrStep<cv::Point2f> oKeyPts,
                                                const cv::cuda::PtrStepSz<cv::Point2f> oContourPts,
                                                const cv::cuda::PtrStep<uchar> oDistMask,
                                                const cv::cuda::PtrStepSzi oDescLUMask,
                                                cv::cuda::PtrStepSzf oDescs, bool bNonZeroInitBins,
                                                bool bGenDescMap, bool bNormalizeBins) {
    assert(oContourPts.cols==1);
    assert(blockDim.x==warpSize);
    assert(oDescLUMask.rows==oDescLUMask.cols);
    assert((oDescLUMask.rows%2)==1);
    int2 vKeyPt_i;
    float2 vKeyPt_f;
    const int nDescSize = oDescs.cols;
    extern __shared__ float aTmpDesc[];
    float* aDesc;
    if(bGenDescMap) {
        vKeyPt_i = make_int2(blockIdx.x,blockIdx.y);
        vKeyPt_f = make_float2((float)blockIdx.x,(float)blockIdx.y);
        aDesc = oDescs.ptr(blockIdx.y*gridDim.x+blockIdx.x);
    }
    else {
        const cv::Point2f& oKeyPt = oKeyPts(blockIdx.x,0);
        vKeyPt_i = make_int2(__float2int_rn(oKeyPt.x),__float2int_rn(oKeyPt.y));
        vKeyPt_f = make_float2(oKeyPt.x,oKeyPt.y);
        aDesc = oDescs.ptr(blockIdx.x);
    }
    int nBaseDescIdx = 0;
    while(nBaseDescIdx<nDescSize) {
        int nDescIdx = nBaseDescIdx+threadIdx.x;
        aTmpDesc[nDescIdx] = bNonZeroInitBins?max(10.0f/nDescSize,0.5f):0.0f;
        nBaseDescIdx += blockDim.x;
    }
    if(oDistMask(vKeyPt_i.y,vKeyPt_i.x)) {
        const int nContourPts = oContourPts.rows;
        const int nMaskSize = oDescLUMask.rows;
        const int nHalfMaskSize = nMaskSize/2;
        int nContourPtIdx = threadIdx.x;
        while(nContourPtIdx<nContourPts) {
            const cv::Point2f& oContourPt = oContourPts(nContourPtIdx,0);
            const int nLookupRow = __float2int_rn(oContourPt.y-vKeyPt_f.y)+nHalfMaskSize;
            const int nLookupCol = __float2int_rn(oContourPt.x-vKeyPt_f.x)+nHalfMaskSize;
            if(nLookupRow>=0 && nLookupRow<nMaskSize && nLookupCol>=0 && nLookupCol<nMaskSize) {
                const int nDescBinIdx = oDescLUMask(nLookupRow,nLookupCol);
                if(nDescBinIdx>=0)
                    atomicAdd(aDesc+nDescBinIdx,1.0f);
            }
            nContourPtIdx += blockDim.x;
        }
    }
    /*@@@
    if(bNormalizeBins) {
        nBaseDescIdx = 0;
        while(nBaseDescIdx<nDescSize) {
            int nDescIdx = nBaseDescIdx+threadIdx.x;
            aTmpDesc[nDescIdx] = aDesc[nDescIdx]*aDesc[nDescIdx];
            nBaseDescIdx += blockDim.x;
        }
        for(int d=nBaseDescIdx>>1; d>=1; d>>=1) {
            __syncthreads();
            int nDescIdx = threadIdx.x;
            while(nDescIdx<d) {
                aTmpDesc[nDescIdx] += aTmpDesc[nDescIdx+d];
                nDescIdx += blockDim.x;
            }
        }
        float fInvNorm;
        if(threadIdx.x==0)
            fInvNorm = rsqrt(aTmpDesc[0]);
        __syncthreads();
        __shfl(fInvNorm,0);
        nBaseDescIdx = 0;
        while(nBaseDescIdx<nDescSize) {
            int nDescIdx = nBaseDescIdx+threadIdx.x;
            aDesc[nDescIdx] = aTmpDesc[nDescIdx]*fInvNorm;
            nBaseDescIdx += blockDim.x;
        }
    }*/
}

/////////////////////////////////////////////////////////////////////////

void host::scdesc_fill_desc_direct(const lv::cuda::KernelParams& oKParams,
                                   const cv::cuda::PtrStep<cv::Point2f> oKeyPts,
                                   const cv::cuda::PtrStepSz<cv::Point2f> oContourPts,
                                   const cv::cuda::PtrStep<uchar> oDistMask,
                                   const cv::cuda::PtrStepSzi oDescLUMask,
                                   cv::cuda::PtrStepSzf oDescs, bool bNonZeroInitBins,
                                   bool bGenDescMap, bool bNormalizeBins) {
    cudaKernelWrap(scdesc_fill_desc_direct,oKParams,oKeyPts,oContourPts,oDistMask,oDescLUMask,oDescs,bNonZeroInitBins,bGenDescMap,bNormalizeBins);
}