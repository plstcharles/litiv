
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

#include "litiv/imgproc/EdgeDetectorCanny.hpp"
#include "litiv/imgproc/EdgeDetectorLBSP.hpp"

namespace lv {

    /// possible implementation modes for lv::thinning
    enum ThinningMode {
        ThinningMode_ZhangSuen=0,
        ThinningMode_LamLeeSuen
    };

    /// 'thins' the provided image (currently only works on 1ch 8UC1 images, treated as binary)
    void thinning(const cv::Mat& oInput, cv::Mat& oOutput, ThinningMode eMode=ThinningMode_LamLeeSuen);

    /// performs non-maximum suppression on the input image, with a (nWinSize)x(nWinSize) window
    template<int nWinSize>
    void nonMaxSuppression(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oMask=cv::Mat());

    /// determines if '*anMap' is a local maximum on the horizontal axis, given 'nMapColStep' spacing between horizontal elements in 'anMap'
    template<size_t nHalfWinSize, typename Tr>
    inline bool isLocalMaximum_Horizontal(const Tr* const anMap, const size_t nMapColStep, const size_t /*nMapRowStep*/) {
        static_assert(nHalfWinSize>=1,"Window size needs to be at least 3x3");
        const Tr nVal = *anMap;
        bool bRes = true;
        lv::unroll<nHalfWinSize>([&](int n){
            bRes &= nVal>anMap[(-n-1)*nMapColStep];
        });
        lv::unroll<nHalfWinSize>([&](int n){
            bRes &= nVal>=anMap[(n+1)*nMapColStep];
        });
        return bRes;
    }

    /// determines if '*anMap' is a local maximum on the vertical axis, given 'nMapRowStep' spacing between vertical elements in 'anMap'
    template<size_t nHalfWinSize, typename Tr>
    inline bool isLocalMaximum_Vertical(const Tr* const anMap, const size_t /*nMapColStep*/, const size_t nMapRowStep) {
        static_assert(nHalfWinSize>=1,"Window size needs to be at least 3x3");
        const Tr nVal = *anMap;
        bool bRes = true;
        lv::unroll<nHalfWinSize>([&](int n){
            bRes &= nVal>anMap[(-n-1)*nMapRowStep];
        });
        lv::unroll<nHalfWinSize>([&](int n){
            bRes &= nVal>=anMap[(n+1)*nMapRowStep];
        });
        return bRes;
    }

    /// determines if '*anMap' is a local maximum on the diagonal, given 'nMapColStep'/'nMapColStep' spacing between horizontal/vertical elements in 'anMap'
    template<size_t nHalfWinSize, bool bInvDiag, typename Tr>
    inline bool isLocalMaximum_Diagonal(const Tr* const anMap, const size_t nMapColStep, const size_t nMapRowStep) {
        static_assert(nHalfWinSize>=1,"Window size needs to be at least 3x3");
        const Tr nVal = *anMap;
        bool bRes = true;
        lv::unroll<nHalfWinSize>([&](int n){
            bRes &= nVal>anMap[(bInvDiag?-1:1)*(-n-1)*nMapColStep+(-n-1)*nMapRowStep];
        });
        lv::unroll<nHalfWinSize>([&](int n){
            bRes &= nVal>=anMap[(bInvDiag?-1:1)*(n+1)*nMapColStep+(n+1)*nMapRowStep];
        });
        return bRes;
    }

    /// determines if '*anMap' is a local maximum on the diagonal, given 'nMapColStep'/'nMapColStep' spacing between horizontal/vertical elements in 'anMap'
    template<size_t nHalfWinSize, typename Tr>
    inline bool isLocalMaximum_Diagonal(const Tr* const anMap, const size_t nMapColStep, const size_t nMapRowStep, bool bInvDiag) {
        if(bInvDiag)
            return isLocalMaximum_Diagonal<nHalfWinSize,true>(anMap,nMapColStep,nMapRowStep);
        else
            return isLocalMaximum_Diagonal<nHalfWinSize,false>(anMap,nMapColStep,nMapRowStep);
    }

} // namespace lv

template<int nWinSize>
void lv::nonMaxSuppression(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oMask) {
    //  http://code.opencv.org/attachments/994/nms.cpp
    //  Copyright (c) 2012, Willow Garage, Inc.
    //  All rights reserved.
    //
    //  Redistribution and use in source and binary forms, with or without
    //  modification, are permitted provided that the following conditions
    //  are met:
    //
    //   * Redistributions of source code must retain the above copyright
    //     notice, this list of conditions and the following disclaimer.
    //   * Redistributions in binary form must reproduce the above
    //     copyright notice, this list of conditions and the following
    //     disclaimer in the documentation and/or other materials provided
    //     with the distribution.
    //   * Neither the name of Willow Garage, Inc. nor the names of its
    //     contributors may be used to endorse or promote products derived
    //     from this software without specific prior written permission.
    //
    //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    //  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    //  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    //  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    //  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    //  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    //  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    //  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    //  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    //  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    //  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    //  POSSIBILITY OF SUCH DAMAGE.
    lvAssert(oInput.channels()==1);
    // initialise the block oMask and destination
    const int M = oInput.rows;
    const int N = oInput.cols;
    const bool masked = !oMask.empty();
    cv::Mat block = 255*cv::Mat_<uint8_t>::ones(cv::Size(2*nWinSize+1,2*nWinSize+1));
    oOutput = cv::Mat_<uint8_t>::zeros(oInput.size());
    // iterate over image blocks
    for(int m = 0; m < M; m+=nWinSize+1) {
        for(int n = 0; n < N; n+=nWinSize+1) {
            cv::Point  ijmax;
            double vcmax, vnmax;
            // get the maximal candidate within the block
            cv::Range ic(m,std::min(m+nWinSize+1,M));
            cv::Range jc(n,std::min(n+nWinSize+1,N));
            cv::minMaxLoc(oInput(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? oMask(ic,jc) : cv::noArray());
            cv::Point cc = ijmax + cv::Point(jc.start,ic.start);
            // search the neighbours centered around the candidate for the true maxima
            cv::Range in(std::max(cc.y-nWinSize,0),std::min(cc.y+nWinSize+1,M));
            cv::Range jn(std::max(cc.x-nWinSize,0),std::min(cc.x+nWinSize+1,N));
            // mask out the block whose maxima we already know
            cv::Mat_<uint8_t> blockmask;
            block(cv::Range(0,in.size()),cv::Range(0,jn.size())).copyTo(blockmask);
            cv::Range iis(ic.start-in.start,std::min(ic.start-in.start+nWinSize+1, in.size()));
            cv::Range jis(jc.start-jn.start,std::min(jc.start-jn.start+nWinSize+1, jn.size()));
            blockmask(iis, jis) = cv::Mat_<uint8_t>::zeros(cv::Size(jis.size(),iis.size()));
            cv::minMaxLoc(oInput(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? oMask(in,jn).mul(blockmask) : blockmask);
            //cv::Point cn = ijmax + cv::Point(jn.start, in.start);
            // if the block centre is also the neighbour centre, then it's a local maxima
            if(vcmax > vnmax)
                oOutput.at<uint8_t>(cc.y, cc.x) = 255;
        }
    }
}
