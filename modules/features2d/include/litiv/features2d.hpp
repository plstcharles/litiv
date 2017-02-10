
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

#include "litiv/utils/opencv.hpp"

// feature descriptor impls headers are included below

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
        static constexpr size_t nByteSum = lv::static_reduce(std::array<size_t,sizeof...(TMatTypes)>{(sizeof(TMatTypes))...},lv::static_reduce_add<size_t>);
        static_assert(bAllTypesIntegral && bAllTypesNonUINT && bAllTypesSmall,"matrix types must be integral & ocv-arithmetic-compat");
        static_assert(bUseSparseMats || (nByteSum<=3),"very unlikely to have enough memory for dense histograms w/ large integers...");
        static_assert(std::is_same<typename std::common_type<int,TMatTypes...>::type,int>::value,"matrix types must be ocv-arithmetic-compat");
        typedef typename std::conditional<bUseSparseMats,cv::SparseMat_<float>,cv::Mat_<float>>::type HistMat;
        typedef typename std::conditional<bUseSparseMats,cv::SparseMat_<int>,cv::Mat_<int>>::type CountMat;
        std::array<int,sizeof...(TMatTypes)> aMinVals,aMaxVals;
        std::array<CountMat,sizeof...(TMatTypes)> aMargCounts;
        std::array<HistMat,sizeof...(TMatTypes)> aMargHists;
        std::array<size_t,sizeof...(TMatTypes)> aStates;
        CountMat oJointCount;
        HistMat oJointHist;
        size_t nJointStates;
    };

#if __cplusplus>=201402L

    template<typename... TMatTypes>
    using JointSparseHistData = JointHistData<true,TMatTypes...>;
    template<typename... TMatTypes>
    using JointDenseHistData = JointHistData<false,TMatTypes...>;

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

    /// computes the mutual information score for a given pair of matrices
    template<size_t nHistQuantifStep=1, bool bUseSparseHistMats=true, bool bFastNumApprox=false, bool bSkipMinMax=false, typename T1, typename T2>
    inline double calcMutualInfo(const cv::Mat_<T1>& oInput1, const cv::Mat_<T2>& oInput2, JointHistData<bUseSparseHistMats,T1,T2>* pJointProbHistOutput=nullptr) {
        std::unique_ptr<JointHistData<bUseSparseHistMats,T1,T2>> pNewJointProbHistOutput;
        if(!pJointProbHistOutput) {
            pNewJointProbHistOutput = std::make_unique<JointHistData<bUseSparseHistMats,T1,T2>>();
            pJointProbHistOutput = pNewJointProbHistOutput.get();
        }
        calcJointProbHist<nHistQuantifStep,bUseSparseHistMats,bFastNumApprox,bSkipMinMax>(std::make_tuple(oInput1,oInput2),*pJointProbHistOutput);
        lvDbgAssert(pJointProbHistOutput->aStates[0]>0 && pJointProbHistOutput->aStates[1]>0 && pJointProbHistOutput->nJointStates>0);
        lvDbgAssert(cv::getElemCount(pJointProbHistOutput->aMargHists[0])>0 && cv::getElemCount(pJointProbHistOutput->aMargHists[1])>0 && cv::getElemCount(pJointProbHistOutput->oJointHist)>0);
        double dMutualInfoScore = 0.0f;
        std::array<int,2> anIterPos;
        for(auto pIter=pJointProbHistOutput->oJointCount.begin(); pIter!=pJointProbHistOutput->oJointCount.end(); ++pIter) {
            const int* anPos = cv::getIterPos(pIter,anIterPos);
            const float& fCurrElemJointProb = cv::getElem(pJointProbHistOutput->oJointHist,anPos);
            if(fCurrElemJointProb>0) {
                const float& fCurrElemMargProb1 = cv::getElem(pJointProbHistOutput->aMargHists[0],std::array<int,2>{anPos[0],0}.data());
                const float& fCurrElemMargProb2 = cv::getElem(pJointProbHistOutput->aMargHists[1],std::array<int,2>{anPos[1],0}.data());
                if(fCurrElemMargProb1>0 && fCurrElemMargProb2>0)
                    dMutualInfoScore += double(fCurrElemJointProb*std::log(fCurrElemJointProb/fCurrElemMargProb1/fCurrElemMargProb2)); // @@@@ check precision/speed for mult instead of double-div? set as FastNumApprox?
            }
        }
        dMutualInfoScore /= std::log(2.0);
        return dMutualInfoScore;
    }

#endif //__cplusplus>=201402L

} // namespace lv

#include "litiv/features2d/DASC.hpp"
#include "litiv/features2d/LBSP.hpp"
#include "litiv/features2d/LSS.hpp"
#include "litiv/features2d/MI.hpp"