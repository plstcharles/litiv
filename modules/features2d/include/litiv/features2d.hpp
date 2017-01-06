
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
#include "litiv/utils/opencv.hpp"

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

    /// helper struct containing joint & marginal probability histograms (dense/full-range version)
    template<bool bUseSparseMats, typename... TMatTypes>
    struct JointHistData {
        static constexpr inline size_t dims() {return sizeof...(TMatTypes);}
        static_assert(dims()>1,"must provide at least two matrix types in template pack");
        static constexpr bool bAllTypesIntegral = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(std::is_integral<TMatTypes>::value)...},lv::static_reduce_and);
        static constexpr bool bAllTypesNonUINT = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(!(std::is_same<TMatTypes,uint>::value))...},lv::static_reduce_and);
        static constexpr bool bAllTypesSmall = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(sizeof(TMatTypes)<=sizeof(int))...},lv::static_reduce_and);
        static constexpr bool bAllTypesByte = lv::static_reduce(std::array<bool,sizeof...(TMatTypes)>{(sizeof(TMatTypes)==1)...},lv::static_reduce_and);
        static_assert(bAllTypesIntegral && bAllTypesNonUINT && bAllTypesSmall,"matrix types must be integral & ocv-arithmetic-compat");
        static_assert(bUseSparseMats || bAllTypesByte,"very unlikely to have enough memory for full-range w/ large ints...");
        static_assert(std::is_same<std::common_type_t<int,TMatTypes...>,int>::value,"matrix types must be ocv-arithmetic-compat");
        typedef std::conditional_t<bUseSparseMats,cv::SparseMat_<float>,cv::Mat_<float>> HistMat;
        typedef std::conditional_t<bUseSparseMats,cv::SparseMat_<int>,cv::Mat_<int>> CountMat;
        std::array<int,dims()> aMinVals,aMaxVals;
        std::array<size_t,dims()> aStates;
        size_t nJointStates;
        std::array<CountMat,dims()> aMargCounts;
        std::array<HistMat,dims()> aMargHists;
        CountMat oJointCount;
        HistMat oJointHist;
    };

    /// computes the joint & marginal probability histograms for a given array of matrices
    template<size_t nQuantifStep=1, bool bUseSparseMats=true, bool bFastNumApprox=false, bool bSkipMinMax=false, typename... TMatTypes>
    inline void calcJointProbHist(const std::tuple<cv::Mat_<TMatTypes>...>& aInputs, JointHistData<bUseSparseMats,TMatTypes...>& oOutput) {
        static_assert(nQuantifStep>0,"quantification step must be greater than zero (used in division)");
        typedef JointHistData<bUseSparseMats,TMatTypes...> HistData;
        std::array<int,HistData::dims()> aJointHistMaxStates;
        lv::for_each_w_idx(aInputs,[&](auto oInput, size_t nInputIdx) {
            lvAssert_(!oInput.empty() && oInput.size==std::get<0>(aInputs).size,"input matrices must all have the same size");
            typedef typename decltype(oInput)::value_type InputElemType;
            constexpr bool bSkippingMinMax = (bSkipMinMax && !std::is_same<InputElemType,int>::value);
            if(bSkippingMinMax) {
                oOutput.aMinVals[nInputIdx] = int(std::numeric_limits<InputElemType>::min());
                oOutput.aMaxVals[nInputIdx] = int(std::numeric_limits<InputElemType>::max());
            }
            else {
                double dMin,dMax;
                cv::minMaxIdx(oInput,&dMin,&dMax);
                oOutput.aMinVals[nInputIdx] = int(dMin);
                oOutput.aMaxVals[nInputIdx] = int(dMax);
            }
            const size_t nMaxStates = size_t(oOutput.aMaxVals[nInputIdx]-oOutput.aMinVals[nInputIdx])/nQuantifStep+size_t(1);
            lvAssert_(nMaxStates<size_t(std::numeric_limits<int>::max()),"element type too big to fit in matrix histograms due to int-indexing");
            const std::array<int,2> aMargHistMaxStates = {(aJointHistMaxStates[nInputIdx]=int(nMaxStates)),1};
            oOutput.aMargHists[nInputIdx].create(bUseSparseMats?1:2,aMargHistMaxStates.data());
            cv::zeroMat(oOutput.aMargHists[nInputIdx]);
            if(!bFastNumApprox) {
                oOutput.aMargCounts[nInputIdx].create(bUseSparseMats?1:2,aMargHistMaxStates.data());
                cv::zeroMat(oOutput.aMargCounts[nInputIdx]);
            }
        });
        oOutput.oJointHist.create(int(HistData::dims()),aJointHistMaxStates.data());
        cv::zeroMat(oOutput.oJointHist);
        if(!bFastNumApprox) {
            oOutput.oJointCount.create(int(HistData::dims()),aJointHistMaxStates.data());
            cv::zeroMat(oOutput.oJointCount);
        }
        const size_t nElemCount = std::get<0>(aInputs).total();
        std::array<int,HistData::dims()> aCurrJointHistIdxs;
        const float fCountIncr = 1.0f/nElemCount;
        for(size_t nElemIdx=0; nElemIdx<nElemCount; ++nElemIdx) {
            lv::for_each_w_idx(aInputs,[&](auto oInput, size_t nInputIdx) {
                typedef typename decltype(oInput)::value_type InputElemType;
                constexpr bool bSkippingMinMax = (bSkipMinMax && !std::is_same<InputElemType,int>::value);
                const int nCurrElem = int(oInput(int(nElemIdx)))-(bSkippingMinMax?int(std::numeric_limits<InputElemType>::min()):oOutput.aMinVals[nInputIdx]);
                lvDbgAssert(nCurrElem>=0);
                const std::array<int,2> aMargHistIdxs = {(aCurrJointHistIdxs[nInputIdx]=nQuantifStep>1?nCurrElem/int(nQuantifStep):nCurrElem),0};
                if(bFastNumApprox)
                    cv::getElem(oOutput.aMargHists[nInputIdx],aMargHistIdxs.data()) += fCountIncr;
                else
                    ++cv::getElem(oOutput.aMargCounts[nInputIdx],aMargHistIdxs.data());
            });
            if(bFastNumApprox)
                cv::getElem(oOutput.oJointHist,aCurrJointHistIdxs.data()) += fCountIncr;
            else
                ++cv::getElem(oOutput.oJointCount,aCurrJointHistIdxs.data());
        }
        if(!bFastNumApprox) {
            for(size_t nInputIdx=0; nInputIdx<HistData::dims(); ++nInputIdx) {
                std::array<int,bUseSparseMats?1:2> anIterPos;
                for(auto pIter=oOutput.aMargCounts[nInputIdx].begin(); pIter!=oOutput.aMargCounts[nInputIdx].end(); ++pIter)
                    cv::getElem(oOutput.aMargHists[nInputIdx],cv::getIterPos(pIter,anIterPos)) = (*pIter)*fCountIncr;
            }
            std::array<int,HistData::dims()> anIterPos;
            for(auto pIter=oOutput.oJointCount.begin(); pIter!=oOutput.oJointCount.end(); ++pIter)
                cv::getElem(oOutput.oJointHist,cv::getIterPos(pIter,anIterPos)) = (*pIter)*fCountIncr;
        }
        if(bUseSparseMats) {
            oOutput.nJointStates = cv::getElemCount(oOutput.oJointHist);
            for(size_t nInputIdx=0; nInputIdx<HistData::dims(); ++nInputIdx)
                oOutput.aStates[nInputIdx] = cv::getElemCount(oOutput.aMargHists[nInputIdx]);
        }
        else {
            oOutput.nJointStates = size_t(lv::static_reduce(aJointHistMaxStates,[](int a, int b){return a*b;}));
            for(size_t nInputIdx=0; nInputIdx<HistData::dims(); ++nInputIdx)
                oOutput.aStates[nInputIdx] = size_t(aJointHistMaxStates[nInputIdx]);
        }
    }

    /// computes the joint & marginal probability histograms for a given array of matrices (inline return version)
    template<size_t nQuantifStep=1, bool bUseSparseMats=true, bool bFastNumApprox=false, bool bSkipMinMax=false, typename... TMatTypes>
    inline JointHistData<bUseSparseMats,TMatTypes...> calcJointProbHist(const std::tuple<cv::Mat_<TMatTypes>...>& aInputs) {
        JointHistData<bUseSparseMats,TMatTypes...> oOutput;
        calcJointProbHist<nQuantifStep,bUseSparseMats,bFastNumApprox,bSkipMinMax,TMatTypes...>(aInputs,oOutput);
        return oOutput;
    }

} // namespace lv