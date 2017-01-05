
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
    template<size_t nRowOffset, size_t nColOffset, typename T>
    inline void localDiff(const cv::Mat_<T>& oImage, cv::Mat_<T>& oLocalDiff) {
        static_assert(nRowOffset>0 || nColOffset>0,"at least one offset must be non-null");
        lvDbgAssert(!oImage.empty());
        oLocalDiff.create(oImage.size());
        for(int nRowIdx=int(nRowOffset); nRowIdx<oImage.rows; ++nRowIdx)
            for(int nColIdx=int(nColOffset); nColIdx<oImage.cols; ++nColIdx)
                oLocalDiff(nRowIdx,nColIdx) = oImage(nRowIdx-nRowOffset,nColIdx-nColOffset)-oImage(nRowIdx,nColIdx);
        lv::unroll<nRowOffset>([&](size_t nRowIdx){
            for(int nColIdx=0; nColIdx<oImage.cols; ++nColIdx)
                oLocalDiff((int)nRowIdx,nColIdx) = (T)0;
        });
        lv::unroll<nColOffset>([&](size_t nColIdx){
            for(int nRowIdx=int(nRowOffset); nRowIdx<oImage.rows; ++nRowIdx)
                oLocalDiff(nRowIdx,(int)nColIdx) = (T)0;
        });
    }

    /// helper struct containing joint & marginal probability histograms
    template<typename... TMatTypes>
    struct JointProbData {
        static constexpr size_t nDims = sizeof...(TMatTypes);
        static_assert(nDims>1,"must provide at least two matrix types in template pack");
        static constexpr bool bAllTypesIntegral = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(std::is_integral<TMatTypes>::value)...},lv::static_reduce_and);
        static constexpr bool bAllTypesNonUINT = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(!(std::is_same<TMatTypes,uint>::value))...},lv::static_reduce_and);
        static constexpr bool bAllTypesSmall = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(sizeof(TMatTypes)<=sizeof(int))...},lv::static_reduce_and);
        static constexpr bool bAllTypesByte = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(sizeof(TMatTypes)==1)...},lv::static_reduce_and);
        static_assert(bAllTypesIntegral,"matrix types must be integral & ocv-arithmetic-compat");
        static_assert(bAllTypesNonUINT,"matrix types must be integral & ocv-arithmetic-compat");
        static_assert(bAllTypesSmall,"matrix types must be integral & ocv-arithmetic-compat");
        static_assert(std::is_same<std::common_type_t<int,TMatTypes...>,int>::value,"matrix types must be ocv-arithmetic-compat");
        std::array<int,nDims> aMinVals,aMaxVals;
        std::array<size_t,nDims> aStates;
        std::array<cv::Mat_<float>,nDims> aMargHists;
        cv::Mat_<float> oJointHist;
        size_t nJointStates;
    };

    /// computes the joint & marginal probability histograms for a given array of matrices
    template<size_t nQuantifStep=1, bool bUseFullRange=false, bool bUseFracSum=true, typename... TMatTypes>
    inline void calcJointProbHist(const std::tuple<cv::Mat_<TMatTypes>...>& aInputs, JointProbData<TMatTypes...>& oOutput) {
        typedef JointProbData<TMatTypes...> HistData;
        static_assert(!bUseFullRange || HistData::bAllTypesByte,"very unlikely to have enough memory for full-range w/ large ints...");
        static_assert(nQuantifStep>0,"quantification step must be greater than zero (used in division)");
        std::array<int,HistData::nDims> aJointHistDims;
        lv::for_each_w_idx(aInputs,[&](auto oInput, size_t nInputIdx) {
            lvAssert_(!oInput.empty() && oInput.size==std::get<0>(aInputs).size,"input matrices must all have the same size");
            if(bUseFullRange) {
                oOutput.aMinVals[nInputIdx] = int(std::numeric_limits<decltype(oInput)::value_type>::min());
                oOutput.aMaxVals[nInputIdx] = int(std::numeric_limits<decltype(oInput)::value_type>::max());
            }
            else {
                double dMin,dMax;
                cv::minMaxIdx(oInput,&dMin,&dMax);
                oOutput.aMinVals[nInputIdx] = int(dMin);
                oOutput.aMaxVals[nInputIdx] = int(dMax);
            }
            const size_t nCurrStates = size_t(oOutput.aMaxVals[nInputIdx]-oOutput.aMinVals[nInputIdx])/nQuantifStep;
            lvAssert_(nCurrStates<size_t(std::numeric_limits<int>::max()),"element type too big to fit in matrix histograms due to int-indexing");
            aJointHistDims[nInputIdx] = int(oOutput.aStates[nInputIdx] = nCurrStates);
            oOutput.aMargHists[nInputIdx].create(1,aJointHistDims[nInputIdx]);
            oOutput.aMargHists[nInputIdx] = 0.0f;
        });
        oOutput.oJointHist.create(int(HistData::nDims),aJointHistDims.data());
        oOutput.oJointHist = 0.0f;
        oOutput.nJointStates = oOutput.oJointHist.total();
        const size_t nElemCount = std::get<0>(aInputs).total();
        std::array<int,HistData::nDims> aCurrHistIndxs;
        const float fCountIncr = 1.0f/nElemCount;
        for(size_t nElemIdx=0; nElemIdx<nElemCount; ++nElemIdx) {
            lv::for_each_w_idx(aInputs,[&](auto oInput, size_t nInputIdx) {
                const int nCurrElem = int(oInput(int(nElemIdx)));
                aCurrHistIndxs[nInputIdx] = nQuantifStep>1?nCurrElem/int(nQuantifStep):nCurrElem;
                oOutput.aMargHists[nInputIdx](aCurrHistIndxs[nInputIdx]) += bUseFracSum?fCountIncr:1.0f;
            });
            oOutput.oJointHist(aJointHistDims.data()) += bUseFracSum?fCountIncr:1.0f;
        }
        if(!bUseFracSum) {
            for(size_t nInputIdx=0; nInputIdx<HistData::nDims; ++nInputIdx)
                oOutput.aMargHists[nInputIdx] *= fCountIncr;
            oOutput.oJointHist *= fCountIncr;
        }
    }

    /// computes the joint & marginal probability histograms for a given array of matrices (inline return version)
    template<size_t nQuantifStep=1, bool bUseFullRange=false, bool bUseFracSum=true, typename... TMatTypes>
    inline JointProbData<TMatTypes...> calcJointProbHist(const std::tuple<cv::Mat_<TMatTypes>...>& aInputs) {
        JointProbData<TMatTypes...> oOutput;
        calcJointProbHist<nQuantifStep,bUseFullRange,bUseFracSum,TMatTypes...>(aInputs,oOutput);
        return oOutput;
    }

} // namespace lv