
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

#include "litiv/utils/cuda.hpp"

namespace device {

    __global__ void kRgb2CIELab(const cudaTextureObject_t inputImg, cudaSurfaceObject_t outputImg, int width, int height);
    __global__ void kInitClusters(const cudaSurfaceObject_t frameLab, float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol, float nbSpxD2);
    __global__ void kAssignment(const cudaSurfaceObject_t surfFrameLab, const float* clusters, const int width, const int height, const int nClustPerRow, const int nbSpx, const int diamSpx, const float wc2, cudaSurfaceObject_t surfLabels, float* accAtt_g);
    __global__ void kUpdate(int nbSpx, float* clusters, float* accAtt_g);

} // namespace device

namespace host {

    void kRgb2CIELab(const lv::cuda::KernelParams& oKParams, const cudaTextureObject_t inputImg, cudaSurfaceObject_t outputImg, int width, int height);
    void kInitClusters(const lv::cuda::KernelParams& oKParams, const cudaSurfaceObject_t frameLab, float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol,float diamSpxD2);
    void kAssignment(const lv::cuda::KernelParams& oKParams, const cudaSurfaceObject_t surfFrameLab, const float* clusters, const int width, const int height, const int nClustPerRow, const int nbSpx, const int diamSpx, const float wc2, cudaSurfaceObject_t surfLabels, float* accAtt_g);
    void kUpdate(const lv::cuda::KernelParams& oKParams, int nbSpx, float* clusters, float* accAtt_g);

} // namespace host