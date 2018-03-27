
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

namespace impl {

    // note: as of 2017/08 with cuda 8.0 and msvc2015, nvcc fails to compile the kernel below
    // with the proper values for 'bGenDescMap' via template parameter on release builds
    //
    // ...no matter what template parameter value is given in the device call, the value in
    //    the kernel always evaluates to 'false' (wtf)
    //
    // ...current workaround with a regular parameter might be slightly slower

    __global__ void scdesc_fill_desc_direct(const cv::cuda::PtrStep<cv::Point2f> oKeyPts,
                                            const cv::cuda::PtrStepSz<cv::Point2f> oContourPts,
                                            const cv::cuda::PtrStep<uchar> oDistMask,
                                            const cudaTextureObject_t pDescLUMask_tex, int nMaskSize,
                                            cv::cuda::PtrStepSzf oDescs, bool bGenDescMap,
                                            bool bNonZeroInitBins, bool bNormalizeBins) {
        assert((oContourPts.cols==0 && oContourPts.rows==0) || oContourPts.cols==1);
        assert((nMaskSize%2)==1);
        assert((blockDim.x%warpSize)==0 && blockDim.y==1 && blockDim.z==1);
        const int nDescSize = oDescs.cols;
        assert(nDescSize>0);
        const int nStepPerDesc = __float2int_ru(float(nDescSize)/blockDim.x);
        assert(nStepPerDesc>=1);
        const int nLUTSize = nStepPerDesc*blockDim.x;
        assert(blockDim.x<=nLUTSize);
        extern __shared__ volatile float aTmpCommon[];
        volatile float* aTmpDesc = aTmpCommon;
        int2 vKeyPt_i;
        float2 vKeyPt_f;
        float* aOutputDesc;
        if(bGenDescMap) {
            vKeyPt_i = make_int2(blockIdx.x,blockIdx.y);
            vKeyPt_f = make_float2((float)blockIdx.x,(float)blockIdx.y);
            aOutputDesc = oDescs.ptr(blockIdx.y*gridDim.x+blockIdx.x);
        }
        else {
            const cv::Point2f& oKeyPt = oKeyPts(blockIdx.x,0);
            vKeyPt_i = make_int2(__float2int_rn(oKeyPt.x),__float2int_rn(oKeyPt.y));
            vKeyPt_f = make_float2(oKeyPt.x,oKeyPt.y);
            aOutputDesc = oDescs.ptr(blockIdx.x);
        }
        const float fInitVal = bNonZeroInitBins?max(10.0f/nDescSize,0.5f):0.0f;
        for(int nStep=0; nStep<nStepPerDesc; ++nStep) {
            const int nDescIdx = blockDim.x*nStep + threadIdx.x;
            aTmpDesc[nDescIdx] = (nDescIdx<nDescSize)?fInitVal:0.0f;
        }
        __syncthreads();
        if(oDistMask(vKeyPt_i.y,vKeyPt_i.x)) {
            const int nContourPts = oContourPts.rows;
            const int nHalfMaskSize = nMaskSize/2;
            int nContourPtIdx = threadIdx.x;
            while(nContourPtIdx<nContourPts) {
                const cv::Point2f& oContourPt = oContourPts(nContourPtIdx,0);
                const int nLookupRow = __float2int_rn(oContourPt.y-vKeyPt_f.y)+nHalfMaskSize;
                const int nLookupCol = __float2int_rn(oContourPt.x-vKeyPt_f.x)+nHalfMaskSize;
                if(nLookupRow>=0 && nLookupRow<nMaskSize && nLookupCol>=0 && nLookupCol<nMaskSize) {
                    const int nDescBinIdx = tex2D<int>(pDescLUMask_tex,nLookupCol,nLookupRow);
                    if(nDescBinIdx>=0)
                        atomicAdd((float*)aTmpDesc+nDescBinIdx,1.0f);
                }
                nContourPtIdx += blockDim.x;
            }
            __syncthreads();
        }
        if(bNormalizeBins) {
            float fSum;
            if(nLUTSize==32 && blockDim.x==32) {
                assert(warpSize==32);
                float fVal = aTmpDesc[threadIdx.x]*aTmpDesc[threadIdx.x];
                fVal += __shfl_down(fVal,16);
                fVal += __shfl_down(fVal,8);
                fVal += __shfl_down(fVal,4);
                fVal += __shfl_down(fVal,2);
                fVal += __shfl_down(fVal,1);
                fSum = __shfl(fVal,0);
            }
            else {
                volatile float* aTmpLUT = aTmpCommon+nLUTSize;
                for(int nStep=0; nStep<nStepPerDesc; ++nStep) {
                    const int nDescIdx = blockDim.x*nStep + threadIdx.x;
                    aTmpLUT[nDescIdx] = aTmpDesc[nDescIdx]*aTmpDesc[nDescIdx];
                }
                if(blockDim.x==32) {
                    assert(warpSize==32 && nLUTSize>32);
                    for(int nStep=nLUTSize-32; nStep>32; nStep-=32)
                        aTmpLUT[threadIdx.x + (nStep-32)] += aTmpLUT[threadIdx.x + nStep];
                    aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x + 32];
                    aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x + 16];
                    aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x +  8];
                    aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x +  4];
                    aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x +  2];
                    aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x +  1];
                }
                else {
                    assert(lv::isPow2(blockDim.x));
                    if(nLUTSize>blockDim.x) {
                        assert(nLUTSize>=blockDim.x*2);
                        for(int nStep=nLUTSize-blockDim.x; nStep>=blockDim.x; nStep-=blockDim.x)
                            aTmpLUT[threadIdx.x + (nStep-blockDim.x)] += aTmpLUT[threadIdx.x + nStep];
                        for(int nStep=blockDim.x/2; nStep>0; nStep>>=1)
                            aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x + nStep];
                    }
                    else {
                        assert(nLUTSize==blockDim.x);
                        for(int nStep=blockDim.x/2; nStep>0; nStep>>=1) {
                            if(threadIdx.x<nStep)
                                aTmpLUT[threadIdx.x] += aTmpLUT[threadIdx.x+nStep];
                            __syncthreads();
                        }
                    }
                }
                fSum = aTmpLUT[0];
            }
            const float fInvNorm = rsqrt(fSum);
            int nDescIdx = threadIdx.x;
            while(nDescIdx<nDescSize) {
                aOutputDesc[nDescIdx] = aTmpDesc[nDescIdx]*fInvNorm;
                nDescIdx += blockDim.x;
            }
        }
        else {
            int nDescIdx = threadIdx.x;
            while(nDescIdx<nDescSize) {
                aOutputDesc[nDescIdx] = aTmpDesc[nDescIdx];
                nDescIdx += blockDim.x;
            }
        }
    }

} // namespace impl

/////////////////////////////////////////////////////////////////////////

void device::scdesc_fill_desc_direct(const lv::cuda::KernelParams& oKParams,
                                     const cv::cuda::PtrStep<cv::Point2f> oKeyPts,
                                     const cv::cuda::PtrStepSz<cv::Point2f> oContourPts,
                                     const cv::cuda::PtrStep<uchar> oDistMask,
                                     const cudaTextureObject_t pDescLUMask_tex, int nMaskSize,
                                     cv::cuda::PtrStepSzf oDescs, bool bNonZeroInitBins,
                                     bool bGenDescMap, bool bNormalizeBins) {
    cudaKernelWrap(scdesc_fill_desc_direct,oKParams,oKeyPts,oContourPts,oDistMask,pDescLUMask_tex,nMaskSize,oDescs,bGenDescMap,bNonZeroInitBins,bNormalizeBins);
}