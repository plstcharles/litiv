
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
// //////////////////////////////////////////////////////////////////////////
//
//               SLIC Superpixel Oversegmentation Algorithm
//       CUDA implementation of Achanta et al.'s method (TPAMI 2012)
//
// Note: requires CUDA compute architecture >= 3.0
// Author: Francois-Xavier Derue
// Contact: francois.xavier.derue@gmail.com
// Source: https://github.com/fderue/SLIC_CUDA
//
// Copyright (c) 2016 fderue
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#include "SLIC.cuh"

__device__ inline float2 operator-(const float2 & a, const float2 & b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ inline float3 operator-(const float3 & a, const float3 & b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline int2 operator+(const int2 & a, const int2 & b) {
    return make_int2(a.x + b.x, a.y + b.y);
}

__device__ inline float computeDistance(float2 c_p_xy, float3 c_p_Lab, float areaSpx, float wc2) {
    float ds2 = pow(c_p_xy.x, 2) + pow(c_p_xy.y, 2);
    float dc2 = pow(c_p_Lab.x, 2) + pow(c_p_Lab.y, 2) + pow(c_p_Lab.z, 2);
    float dist = sqrt(dc2 + ds2/areaSpx*wc2);
    return dist;
}

__device__ inline int convertIdx(int2 wg, int lc_idx, int nBloc_per_row) {
    int2 relPos2D = make_int2(lc_idx % 5 - 2, lc_idx / 5 - 2);
    int2 glPos2D = wg + relPos2D;
    return glPos2D.y*nBloc_per_row + glPos2D.x;
}

/////////////////////////////////////////////////////////////////////////

__global__ void device::kRgb2CIELab(const cudaTextureObject_t texFrameBGRA, cudaSurfaceObject_t surfFrameLab, int width, int height) {

    int px = blockIdx.x*blockDim.x + threadIdx.x;
    int py = blockIdx.y*blockDim.y + threadIdx.y;

    if (px<width && py<height) {
        const uchar4 nPixel = tex2D<uchar4>(texFrameBGRA, px, py);//inputImg[offset];

        const float _b = nPixel.x/255.0f;
        const float _g = nPixel.y/255.0f;
        const float _r = nPixel.z/255.0f;

        float x = _r * 0.412453f + _g * 0.357580f + _b * 0.180423f;
        float y = _r * 0.212671f + _g * 0.715160f + _b * 0.072169f;
        float z = _r * 0.019334f + _g * 0.119193f + _b * 0.950227f;

        x /= 0.950456f;
        float y3 = exp(log(y)/3.0f);
        z /= 1.088754f;

        float l, a, b;

        x = x > 0.008856f ? exp(log(x)/3.0f) : (7.787f * x + 0.13793f);
        y = y > 0.008856f ? y3 : 7.787f * y + 0.13793f;
        z = z > 0.008856f ? z /= exp(log(z)/3.0f) : (7.787f * z + 0.13793f);

        l = y > 0.008856f ? (116.0f * y3 - 16.0f) : 903.3f * y;
        a = (x - y) * 500.0f;
        b = (y - z) * 200.0f;

        float4 fPixel;
        fPixel.x = l;
        fPixel.y = a;
        fPixel.z = b;
        fPixel.w = 0;

        surf2Dwrite(fPixel, surfFrameLab, px * 16, py);
    }
}

__global__ void device::kInitClusters(const cudaSurfaceObject_t surfFrameLab, float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol) {
    int centroidIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int nSpx = nSpxPerCol*nSpxPerRow;

    if (centroidIdx<nSpx){
        int wSpx = width / nSpxPerRow;
        int hSpx = height / nSpxPerCol;

        int i = centroidIdx / nSpxPerRow;
        int j = centroidIdx%nSpxPerRow;

        int x = j*wSpx + wSpx / 2;
        int y = i*hSpx + hSpx / 2;

        float4 color;
        surf2Dread(&color, surfFrameLab, x * 16, y);
        clusters[centroidIdx] = color.x;
        clusters[centroidIdx + nSpx] = color.y;
        clusters[centroidIdx + 2 * nSpx] = color.z;
        clusters[centroidIdx + 3 * nSpx] = x;
        clusters[centroidIdx + 4 * nSpx] = y;
    }
}

__global__ void device::kAssignment(const cudaSurfaceObject_t surfFrameLab, const float* clusters, const int width, const int height, const int wSpx, const int hSpx, const float wc2, cudaSurfaceObject_t surfLabels, float* accAtt_g) {
    // gather NNEIGH surrounding clusters
    const int NNEIGH = 3;
    __shared__ float4 sharedLab[NNEIGH][NNEIGH];
    __shared__ float2 sharedXY[NNEIGH][NNEIGH];

    int nClustPerRow = width / wSpx;
    int nn2 = NNEIGH / 2;

    int nbSpx = width / wSpx * height / hSpx;

    if (threadIdx.x<NNEIGH && threadIdx.y<NNEIGH){
        int id_x = threadIdx.x - nn2;
        int id_y = threadIdx.y - nn2;

        int clustLinIdx = blockIdx.x + id_y*nClustPerRow + id_x;
        if (clustLinIdx >= 0 && clustLinIdx<gridDim.x){
            sharedLab[threadIdx.y][threadIdx.x].x = clusters[clustLinIdx];
            sharedLab[threadIdx.y][threadIdx.x].y = clusters[clustLinIdx + nbSpx];
            sharedLab[threadIdx.y][threadIdx.x].z = clusters[clustLinIdx + 2 * nbSpx];

            sharedXY[threadIdx.y][threadIdx.x].x = clusters[clustLinIdx + 3 * nbSpx];
            sharedXY[threadIdx.y][threadIdx.x].y = clusters[clustLinIdx + 4 * nbSpx];
        }
        else{
            sharedLab[threadIdx.y][threadIdx.x].x = -1;
        }
    }

    __syncthreads();

    // Find nearest neighbour
    float areaSpx = wSpx*hSpx;
    float distanceMin = 100000;
    float labelMin = -1;

    int px_in_grid = blockIdx.x*blockDim.x + threadIdx.x;
    int py_in_grid = blockIdx.y*blockDim.y + threadIdx.y;

    int px = px_in_grid%width;

    if (py_in_grid<hSpx && px<width){
        int py = py_in_grid + px_in_grid / width*hSpx;

        float4 color;
        surf2Dread(&color, surfFrameLab, px * 16, py);
        float3 px_Lab = make_float3(color.x, color.y, color.z);
        float2 px_xy = make_float2(px, py);
        for (int i = 0; i<NNEIGH; i++){
            for (int j = 0; j<NNEIGH; j++){
                if (sharedLab[i][j].x != -1){
                    float2 cluster_xy = make_float2(sharedXY[i][j].x, sharedXY[i][j].y);
                    float3 cluster_Lab = make_float3(sharedLab[i][j].x, sharedLab[i][j].y, sharedLab[i][j].z);

                    float2 px_c_xy = px_xy - cluster_xy;
                    float3 px_c_Lab = px_Lab - cluster_Lab;

                    float distTmp = fminf(computeDistance(px_c_xy, px_c_Lab, areaSpx, wc2), distanceMin);

                    if (distTmp != distanceMin){
                        distanceMin = distTmp;
                        labelMin = blockIdx.x + (i - nn2)*nClustPerRow + (j - nn2);
                    }
                }
            }
        }
        surf2Dwrite(labelMin, surfLabels, px * 4, py);

        int iLabelMin = int(labelMin);
        atomicAdd(&accAtt_g[iLabelMin], px_Lab.x);
        atomicAdd(&accAtt_g[iLabelMin + nbSpx], px_Lab.y);
        atomicAdd(&accAtt_g[iLabelMin + 2 * nbSpx], px_Lab.z);
        atomicAdd(&accAtt_g[iLabelMin + 3 * nbSpx], px);
        atomicAdd(&accAtt_g[iLabelMin + 4 * nbSpx], py);
        atomicAdd(&accAtt_g[iLabelMin + 5 * nbSpx], 1); //counter*/
    }
}

__global__ void device::kUpdate(int nbSpx, float* clusters, float* accAtt_g) {
    int cluster_idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (cluster_idx<nbSpx){
        int nbSpx2 = nbSpx * 2;
        int nbSpx3 = nbSpx * 3;
        int nbSpx4 = nbSpx * 4;
        int nbSpx5 = nbSpx * 5;
        int counter = accAtt_g[cluster_idx + nbSpx5];
        if (counter != 0){
            clusters[cluster_idx] = accAtt_g[cluster_idx] / counter;
            clusters[cluster_idx + nbSpx] = accAtt_g[cluster_idx + nbSpx] / counter;
            clusters[cluster_idx + nbSpx2] = accAtt_g[cluster_idx + nbSpx2] / counter;
            clusters[cluster_idx + nbSpx3] = accAtt_g[cluster_idx + nbSpx3] / counter;
            clusters[cluster_idx + nbSpx4] = accAtt_g[cluster_idx + nbSpx4] / counter;

            //reset accumulator
            accAtt_g[cluster_idx] = 0;
            accAtt_g[cluster_idx + nbSpx] = 0;
            accAtt_g[cluster_idx + nbSpx2] = 0;
            accAtt_g[cluster_idx + nbSpx3] = 0;
            accAtt_g[cluster_idx + nbSpx4] = 0;
            accAtt_g[cluster_idx + nbSpx5] = 0;
        }
    }
}

/////////////////////////////////////////////////////////////////////////

void host::kRgb2CIELab(const lv::cuda::KernelParams& oKParams, const cudaTextureObject_t inputImg, cudaSurfaceObject_t outputImg, int width, int height) {
    cudaKernelWrap(kRgb2CIELab,oKParams,inputImg,outputImg,width,height);
}

void host::kInitClusters(const lv::cuda::KernelParams& oKParams, const cudaSurfaceObject_t frameLab, float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol) {
    cudaKernelWrap(kInitClusters,oKParams,frameLab,clusters,width,height,nSpxPerRow,nSpxPerCol);
}

void host::kAssignment(const lv::cuda::KernelParams& oKParams, const cudaSurfaceObject_t frameLab, const float* clusters, const int width, const int height, const int wSpx, const int hSpx, const float wc2, cudaSurfaceObject_t labels, float* accAtt_g) {
    cudaKernelWrap(kAssignment,oKParams,frameLab,clusters,width,height,wSpx,hSpx,wc2,labels,accAtt_g);
}

void host::kUpdate(const lv::cuda::KernelParams& oKParams, int nbSpx, float* clusters, float* accAtt_g) {
    cudaKernelWrap(kUpdate,oKParams,nbSpx,clusters,accAtt_g);
}