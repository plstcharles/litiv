
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

#pragma once

#include "litiv/features2d/DASC.hpp"
#include "litiv/features2d/LBSP.hpp"
#include "litiv/features2d/LSS.hpp"

namespace lv {

    // the functions below are utils for feature descriptors which may better fit
    // into the imgproc module, but are kept here to avoid circular dependencies

    /// computes a local intensity difference map for a given image
    template<size_t nRowOffset, size_t nColOffset, typename TValue>
    inline void localDiff(const cv::Mat_<TValue>& oImage, cv::Mat_<TValue>& oLocalDiff) {
        lvDbgAssert(!oImage.empty() && (nColOffset>0 || nRowOffset>0));
        oLocalDiff.create(oImage.size());
        for(int nRowIdx=int(nRowOffset); nRowIdx<oImage.rows; ++nRowIdx)
            for(int nColIdx=int(nColOffset); nColIdx<oImage.cols; ++nColIdx)
                oLocalDiff(nRowIdx,nColIdx) = oImage(nRowIdx-nRowOffset,nColIdx-nColOffset)-oImage(nRowIdx,nColIdx);
        lv::unroll<nRowOffset>([&](size_t nRowIdx){
            for(int nColIdx=0; nColIdx<oImage.cols; ++nColIdx)
                oLocalDiff((int)nRowIdx,nColIdx) = (TValue)0;
        });
        lv::unroll<nColOffset>([&](size_t nColIdx){
            for(int nRowIdx=nRowOffset; nRowIdx<oImage.rows; ++nRowIdx)
                oLocalDiff(nRowIdx,(int)nColIdx) = (TValue)0;
        });
    }

} // namespace lv