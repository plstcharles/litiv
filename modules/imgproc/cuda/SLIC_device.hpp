
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

#pragma once

__global__ void kRgb2CIELab(const cudaTextureObject_t inputImg,
                            cudaSurfaceObject_t outputImg,
                            int width,
                            int height);

__global__ void kInitClusters(const cudaSurfaceObject_t frameLab,
                              float* clusters,
                              int width,
                              int height,
                              int nSpxPerRow,
                              int nSpxPerCol);


__global__ void kAssignment(const cudaSurfaceObject_t frameLab,
                            const float* clusters,
                            const int width,
                            const int height,
                            const int wSpx,
                            const int hSpx,
                            const float wc2,
                            cudaSurfaceObject_t labels,
                            float* accAtt_g);

__global__ void kUpdate(int nbSpx, float* clusters, float* accAtt_g);

__device__ inline float2 operator-(const float2 & a, const float2 & b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ inline float3 operator-(const float3 & a, const float3 & b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline int2 operator+(const int2 & a, const int2 & b) { return make_int2(a.x + b.x, a.y + b.y); }

__device__ inline float computeDistance(float2 c_p_xy, float3 c_p_Lab, float areaSpx, float wc2){
    float ds2 = pow(c_p_xy.x, 2) + pow(c_p_xy.y, 2);
    float dc2 = pow(c_p_Lab.x, 2) + pow(c_p_Lab.y, 2) + pow(c_p_Lab.z, 2);
    float dist = sqrt(dc2 + ds2 / areaSpx*wc2);
    return dist;
}

__device__ inline int convertIdx(int2 wg, int lc_idx, int nBloc_per_row){
    int2 relPos2D = make_int2(lc_idx % 5 - 2, lc_idx / 5 - 2);
    int2 glPos2D = wg + relPos2D;

    return glPos2D.y*nBloc_per_row + glPos2D.x;
}