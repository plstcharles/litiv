
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

#pragma once

#include "litiv/utils/cuda.hpp"

#define AFF_MAP_DISP_RANGE_MAX UCHAR_MAX-1 // 8-bit masks w 0xFF reserved for 'dont care'

namespace device {

    struct DistCalcLUT {
        const float* aDesc1;
        const float* aDesc2;
    };

    void compute_desc_affinity_l2(const lv::cuda::KernelParams& oKParams,
                                  const cv::cuda::PtrStep<float> oDescMap1,
                                  const cv::cuda::PtrStep<float> oDescMap2,
                                  cv::cuda::PtrStep<float> oAffinityMap,
                                  int nOffsets, int nDescSize);

    void compute_desc_affinity_l2_roi(const lv::cuda::KernelParams& oKParams,
                                      const cv::cuda::PtrStep<float> oDescMap1,
                                      const cv::cuda::PtrStep<uchar> oROI1,
                                      const cv::cuda::PtrStep<float> oDescMap2,
                                      const cv::cuda::PtrStep<uchar> oROI2,
                                      cv::cuda::PtrStep<float> oAffinityMap,
                                      int nOffsets, int nDescSize);

    void compute_desc_affinity_patch(const lv::cuda::KernelParams& oKParams,
                                     const cv::cuda::PtrStep<float> oRawAffinityMap,
                                     cv::cuda::PtrStep<float> oAffinityMap, int nPatchSize);

    void setDisparityRange(const std::array<int,AFF_MAP_DISP_RANGE_MAX>& aDispRange);

} // namespace device