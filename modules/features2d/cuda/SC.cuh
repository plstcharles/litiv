
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

namespace device {

    __global__ void scdesc_fill_desc_direct(const cv::cuda::PtrStep<cv::Point2f> oKeyPts,
                                            const cv::cuda::PtrStepSz<cv::Point2f> oContourPts,
                                            const cv::cuda::PtrStep<uchar> oDistMask,
                                            const cv::cuda::PtrStepSzi oDescLUMask,
                                            cv::cuda::PtrStepSzf oDescs, bool bNonZeroInitBins,
                                            bool bGenDescMap, bool bNormalizeBins);

} // namespace device

namespace host {

    void scdesc_fill_desc_direct(const lv::cuda::KernelParams& oKParams,
                                 const cv::cuda::PtrStep<cv::Point2f> oKeyPts,
                                 const cv::cuda::PtrStepSz<cv::Point2f> oContourPts,
                                 const cv::cuda::PtrStep<uchar> oDistMask,
                                 const cv::cuda::PtrStepSzi oDescLUMask,
                                 cv::cuda::PtrStepSzf oDescs, bool bNonZeroInitBins,
                                 bool bGenDescMap, bool bNormalizeBins);

} // namespace host
