
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include <stdlib.h>

#ifndef __LITIV_FGSTEREOM_HPP__
#error "Cannot include .inl.hpp headers directly!"
#endif //ndef(__LITIV_FGSTEREOM_HPP__)
#pragma once
#if (FGSTEREOMATCH_CONFIG_USE_DASCGF_FEATS+FGSTEREOMATCH_CONFIG_USE_DASCRF_FEATS+FGSTEREOMATCH_CONFIG_USE_LSS_FEATS/*+...*/)!=1
#error "Must specify only one feature type to use."
#endif //(features config ...)!=1

inline FGStereoMatcher::FGStereoMatcher(const cv::Size& oImageSize, size_t nMinDispOffset, size_t nMaxDispOffset, size_t nDispStep) :
        m_oImageSize(oImageSize) {
    static_assert(getInputStreamCount()==4 && getOutputStreamCount()==4 && getCameraCount()==2,"i/o stream must be two image-mask pairs");
    lvAssert_(m_oImageSize.area()>1,"graph grid must be 2D and have at least two nodes");
    lvAssert_(nDispStep>0,"specified disparity offset step size must be strictly positive");
    if(nMaxDispOffset<nMinDispOffset)
        std::swap(nMaxDispOffset,nMinDispOffset);
    lvAssert_(nMaxDispOffset<size_t(s_nStereoOccludedLabel),"using reserved disparity integer label value");
    const size_t nMaxAllowedDispLabelCount = size_t(std::numeric_limits<StereoLabelType>::max()-2);
    const size_t nExpectedDispLabelCount = ((nMaxDispOffset-nMinDispOffset)/nDispStep)+1;
    lvAssert__(nMaxAllowedDispLabelCount>=nExpectedDispLabelCount,"internal stereo label type too small for given disparity range (max = %d)",(int)nMaxAllowedDispLabelCount);
    const std::vector<OutputLabelType> vStereoLabels = lv::make_range((OutputLabelType)nMinDispOffset,(OutputLabelType)nMaxDispOffset,(OutputLabelType)nDispStep);
    lvDbgAssert(nExpectedDispLabelCount==vStereoLabels.size());
    lvAssert_(vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
    const std::vector<OutputLabelType> vReservedStereoLabels = {s_nStereoDontCareLabel,s_nStereoOccludedLabel};
    m_pModelData = std::make_unique<GraphModelData>(m_oImageSize,lv::concat<OutputLabelType>(vStereoLabels,vReservedStereoLabels),nDispStep);
    m_pStereoInf = std::make_unique<StereoGraphInference>(*m_pModelData);
    //@@@@@ m_pResegmInf = std::make_unique<ResegmGraphInference>(*m_pModelData,m_pModelData->m_oResegmLabeling);
}

inline void FGStereoMatcher::apply(const MatArrayIn& aInputs, MatArrayOut& aOutputs) {
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx) {
        lvAssert__(aInputs[nInputIdx].dims==2 && m_oImageSize==aInputs[nInputIdx].size(),"input in array at index=%d had the wrong size",(int)nInputIdx);
        lvAssert_((nInputIdx%2)==0 || aInputs[nInputIdx].type()==CV_8UC1,"unexpected input mask type");
        aInputs[nInputIdx].copyTo(m_pModelData->m_aInputs[nInputIdx]);
    }
    lvDbgAssert(m_pModelData && m_pStereoInf /*&& m_pResegmInf@@@@@*/);
    m_pModelData->updateModels(aInputs);
    if(lv::getVerbosity()>=2) {
        StereoGraphInference::VerboseVisitorType oVisitor;
        m_pStereoInf->infer(oVisitor);
    }
    else {
        StereoGraphInference::EmptyVisitorType oVisitor;
        m_pStereoInf->infer(oVisitor);
    }
    // @@@@@ fill missing outputs
    for(size_t nOutputIdx=0; nOutputIdx<aOutputs.size(); ++nOutputIdx) {
        if(nOutputIdx==0)
            m_pStereoInf->getOutput(aOutputs[nOutputIdx]);
        else
            cv::Mat_<OutputLabelType>(m_oImageSize,OutputLabelType(0)).copyTo(aOutputs[nOutputIdx]);
        aOutputs[nOutputIdx].copyTo(m_pModelData->m_aOutputs[nOutputIdx]);
    }
}

inline void FGStereoMatcher::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    m_pModelData->calcFeatures(aInputs,pFeatsPacket);
}

inline void FGStereoMatcher::setNextFeatures(const cv::Mat& oPackedFeats) {
    m_pModelData->setNextFeatures(oPackedFeats);
}

inline std::string FGStereoMatcher::getFeatureExtractorName() const {
#if FGSTEREOMATCH_CONFIG_USE_DASCGF_FEATS
    return "sc-dasc-gf";
#elif FGSTEREOMATCH_CONFIG_USE_DASCRF_FEATS
    return "sc-dasc-rf";
#elif FGSTEREOMATCH_CONFIG_USE_LSS_FEATS
    return "sc-lss";
#endif //FGSTEREOMATCH_CONFIG_USE_..._FEATS
}

inline size_t FGStereoMatcher::getMaxLabelCount() const {
    lvDbgAssert(m_pModelData);
    return m_pModelData->m_vStereoLabels.size();
}

inline const std::vector<FGStereoMatcher::OutputLabelType>& FGStereoMatcher::getLabels() const {
    lvDbgAssert(m_pModelData);
    return m_pModelData->m_vStereoLabels;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline FGStereoMatcher::GraphModelData::GraphModelData(const cv::Size& oImageSize, std::vector<OutputLabelType>&& vStereoLabels, size_t nStereoLabelStep) :
        m_nMaxIterCount(FGSTEREOMATCH_DEFAULT_MAXITERCOUNT),
        m_eStereoLabelInitType(LabelInit_LocalOptim),
        m_eResegmLabelInitType(LabelInit_LocalOptim),
        m_eStereoLabelOrderType(LabelOrder_Default),
        m_eResegmLabelOrderType(LabelOrder_Default),
        m_nStereoLabelingRandomSeed(size_t(0)),
        m_nResegmLabelingRandomSeed(size_t(0)),
        m_nStereoLabelOrderRandomSeed(size_t(0)),
        m_nResegmLabelOrderRandomSeed(size_t(0)),
        m_oGridSize(oImageSize),
        m_vStereoLabels(vStereoLabels),
        m_nDispOffsetStep(nStereoLabelStep),
        m_nMinDispOffset(size_t(m_vStereoLabels[0])),
        m_nMaxDispOffset(size_t(m_vStereoLabels.size()>3?m_vStereoLabels[m_vStereoLabels.size()-3]:m_vStereoLabels.back())),
        m_nStereoDontCareLabelIdx(StereoLabelType(m_vStereoLabels.size()-2)),
        m_nStereoOccludedLabelIdx(StereoLabelType(m_vStereoLabels.size()-1)),
        m_bUsePrecalcFeatsNext(false),m_bModelUpToDate(false) {
    lvAssert_(m_nMaxIterCount>0,"max iter count must be positive");
    lvDbgAssert_(m_oGridSize.dims()==2 && m_oGridSize.total()>size_t(1),"graph grid must be 2D and have at least two nodes");
    lvAssert_(m_vStereoLabels.size()>3,"graph must have at least two possible output stereo labels, beyond reserved ones");
    lvAssert_(m_vStereoLabels.size()<=size_t(std::numeric_limits<StereoLabelType>::max()),"too many labels for internal type");
    lvDbgAssert(m_vStereoLabels[m_nStereoDontCareLabelIdx]==s_nStereoDontCareLabel && m_vStereoLabels[m_nStereoOccludedLabelIdx]==s_nStereoOccludedLabel);
    lvDbgAssert(std::min_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==m_vStereoLabels.begin() && m_vStereoLabels[0]>=OutputLabelType(0));
    lvDbgAssert(std::max_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==(m_vStereoLabels.begin()+m_vStereoLabels.size()-3));
    lvDbgAssert(std::equal(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2,lv::unique(m_vStereoLabels.begin(),m_vStereoLabels.end()).begin()+1));
    lvAssert_(m_nDispOffsetStep>0,"label step size must be positive");
    lvAssert_(m_oGridSize[1]>m_nMinDispOffset,"row length too small for smallest disp");
    lvAssert_(m_nMinDispOffset<m_nMaxDispOffset,"min/max disp offsets mismatch");
    lvDbgAssert_(std::numeric_limits<AssocCountType>::max()>m_oGridSize[1],"grid width is too large for association counter type");
#if FGSTEREOMATCH_CONFIG_USE_DASCGF_FEATS
    m_pVisDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS,DASC_DEFAULT_GF_SUBSPL,DASC_DEFAULT_PREPROCESS);
#elif FGSTEREOMATCH_CONFIG_USE_DASCRF_FEATS
    m_pVisDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR,DASC_DEFAULT_RF_ITERS,DASC_DEFAULT_PREPROCESS);
#elif FGSTEREOMATCH_CONFIG_USE_LSS_FEATS
    @@@
#endif //FGSTEREOMATCH_CONFIG_USE_..._FEATS
    const size_t nShapeContextInnerRadius=2, nShapeContextOuterRadius=FGSTEREOMATCH_DEFAULT_SHAPEDESC_RAD;
    m_pShpDescExtractor = std::make_unique<ShapeContext>(nShapeContextInnerRadius,nShapeContextOuterRadius,SHAPECONTEXT_DEFAULT_ANG_BINS,SHAPECONTEXT_DEFAULT_RAD_BINS);
    const cv::Size oMinWindowSize = m_pVisDescExtractor->windowSize();
    lvAssert__(oMinWindowSize.width<=oImageSize.width && oMinWindowSize.height<=oImageSize.height,"image is too small to compute descriptors with current pattern size -- need at least (%d,%d) and got (%d,%d)",oMinWindowSize.width,oMinWindowSize.height,oImageSize.width,oImageSize.height);
    m_nGridBorderSize = (size_t)std::max(m_pVisDescExtractor->borderSize(0),m_pVisDescExtractor->borderSize(1));
    lvDbgAssert(m_nGridBorderSize<m_oGridSize[0] && m_nGridBorderSize<m_oGridSize[1]);
    lvDbgAssert((size_t)std::max(m_pShpDescExtractor->borderSize(0),m_pShpDescExtractor->borderSize(1))<=m_nGridBorderSize);
    lvDbgAssert(m_aAssocCostAddLUT.size()==m_aAssocCostSumLUT.size() && m_aAssocCostRemLUT.size()==m_aAssocCostSumLUT.size());
    lvDbgAssert_(m_nMaxDispOffset<m_aAssocCostSumLUT.size(),"assoc cost lut size might not be large enough");
    lvDbgAssert(FGSTEREOMATCH_UNIQUE_COST_ZERO_COUNT<m_aAssocCostSumLUT.size());
    lvDbgAssert(FGSTEREOMATCH_UNIQUE_COST_INCR_REL(0)==0.0f);
    std::fill_n(m_aAssocCostAddLUT.begin(),FGSTEREOMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    std::fill_n(m_aAssocCostRemLUT.begin(),FGSTEREOMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    std::fill_n(m_aAssocCostSumLUT.begin(),FGSTEREOMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    for(size_t n=FGSTEREOMATCH_UNIQUE_COST_ZERO_COUNT; n<m_aAssocCostAddLUT.size(); ++n) {
        m_aAssocCostAddLUT[n] = ValueType(FGSTEREOMATCH_UNIQUE_COST_INCR_REL(n+1-FGSTEREOMATCH_UNIQUE_COST_ZERO_COUNT)*FGSTEREOMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRemLUT[n] = -ValueType(FGSTEREOMATCH_UNIQUE_COST_INCR_REL(n-FGSTEREOMATCH_UNIQUE_COST_ZERO_COUNT)*FGSTEREOMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostSumLUT[n] = (n==size_t(0)?ValueType(0):(m_aAssocCostSumLUT[n-1]+m_aAssocCostAddLUT[n-1]));
    }
#if FGSTEREOMATCH_LBLSIM_USE_EXP_GRADPIVOT
    m_aLabelSimCostGradFactLUT.init(0,255,[](int nLocalGrad){
        return (float)std::exp(float(FGSTEREOMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/FGSTEREOMATCH_LBLSIM_COST_GRADRAW_SCALE);
    });
#else //!FGSTEREOMATCH_LBLSIM_USE_EXP_GRADPIVOT
    m_aLabelSimCostGradFactLUT.init(0,255,[](int nLocalGrad){
        const float fGradPivotFact = 1.0f+(float(FGSTEREOMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/((nLocalGrad>=FGSTEREOMATCH_LBLSIM_COST_GRADPIVOT_CST)?(255-FGSTEREOMATCH_LBLSIM_COST_GRADPIVOT_CST):FGSTEREOMATCH_LBLSIM_COST_GRADPIVOT_CST));
        const float fGradScaleFact = FGSTEREOMATCH_LBLSIM_COST_GRADRAW_SCALE*fGradPivotFact*fGradPivotFact;
        lvDbgAssert(fGradScaleFact>=0.0f && fGradScaleFact<=4.0f*FGSTEREOMATCH_LBLSIM_COST_GRADRAW_SCALE);
        return fGradScaleFact;
    });
#endif //!FGSTEREOMATCH_LBLSIM_USE_EXP_GRADPIVOT
    lvDbgAssert(m_aLabelSimCostGradFactLUT.size()==size_t(256) && m_aLabelSimCostGradFactLUT.domain_offset_low()==0);
    lvDbgAssert(m_aLabelSimCostGradFactLUT.domain_index_step()==1.0 && m_aLabelSimCostGradFactLUT.domain_index_scale()==1.0);
    ////////////////////////////////////////////// put in new func, dupe for resegm model
    const size_t nLabels = vStereoLabels.size();
    const size_t nRealLabels = vStereoLabels.size()-2;
    const size_t nRows = m_oGridSize(0);
    const size_t nCols = m_oGridSize(1);
    const size_t nNodes = m_oGridSize.total();
    const size_t nStereoVisSimUnaryFuncDataSize = nNodes*nLabels;
    const size_t nStereoSmoothPairwFuncDataSize = nNodes*2*(nLabels*nLabels);
    const size_t nStereoFuncDataSize = nStereoVisSimUnaryFuncDataSize+nStereoSmoothPairwFuncDataSize/*+...@@@@*/;
    const size_t nModelSize = (nStereoFuncDataSize/*+...@@@@*/)*sizeof(ValueType)/*+...@@@@*/;
    lvLog_(1,"Expecting model size = %zu mb",nModelSize/1024/1024);
    lvAssert_(nModelSize<(CACHE_MAX_SIZE_GB*1024*1024*1024),"too many nodes/labels; model is unlikely to fit in memory");
    lvLog(1,"Constructing graphical model for stereo matching...");
    lv::StopWatch oLocalTimer;
    const size_t nFactorsPerNode = 3;
    const size_t nFunctions = nNodes*nFactorsPerNode;
    m_pStereoModel = std::make_unique<StereoModelType>(StereoSpaceType(nNodes,(StereoLabelType)vStereoLabels.size()),nFactorsPerNode);
    m_pStereoModel->reserveFunctions<StereoExplicitFunction>(nFunctions);
    const std::array<int,2> anAssocCountsDims{int(m_oGridSize[0]/m_nDispOffsetStep),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep)};
    m_oAssocCounts.create(2,anAssocCountsDims.data());
    const std::array<int,3> anAssocMapDims{int(m_oGridSize[0]/m_nDispOffsetStep),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep),int(nRealLabels*m_nDispOffsetStep)};
    m_oAssocMap.create(3,anAssocMapDims.data());
    m_oAssocCosts.create(m_oGridSize);
    m_oUnaryCosts.create(m_oGridSize);
    m_oPairwCosts.create(m_oGridSize);
    m_vNodeInfos.resize(nNodes);
    for(int nRowIdx = 0; nRowIdx<(int)nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<(int)nCols; ++nColIdx) {
            const size_t nNodeIdx = nRowIdx*nCols+nColIdx;
            m_vNodeInfos[nNodeIdx].nRowIdx = nRowIdx;
            m_vNodeInfos[nNodeIdx].nColIdx = nColIdx;
            // the LUT members below will be properly initialized in the following sections
            m_vNodeInfos[nNodeIdx].nStereoVisSimUnaryFactID = SIZE_MAX;
            m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc = nullptr;
            m_vNodeInfos[nNodeIdx].anPairwNodeIdxs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nNodeIdx].anStereoSmoothPairwFactIDs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nNodeIdx].apStereoSmoothPairwFuncs = std::array<StereoFunc*,2>{nullptr,nullptr};
        }
    }
    m_vStereoVisSimUnaryFuncs.reserve(nNodes);
    m_avStereoSmoothPairwFuncs[0].reserve(nNodes);
    m_avStereoSmoothPairwFuncs[1].reserve(nNodes);
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
    m_pStereoVisSimUnaryFuncsDataBase = nullptr;
    m_pStereoSmoothPairwFuncsDataBase = nullptr;
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
    m_aStereoFuncsData = std::make_unique<ValueType[]>(nStereoFuncDataSize);
    m_pStereoVisSimUnaryFuncsDataBase = m_aStereoFuncsData.get();
    m_pStereoSmoothPairwFuncsDataBase = m_pStereoVisSimUnaryFuncsDataBase+nStereoVisSimUnaryFuncDataSize;
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
    {
        lvLog(1,"\tadding visual similarity factor for each grid node...");
        // (unary costs will depend on input data of both images, so each node function is likely unique)
        const std::array<size_t,1> aUnaryFuncDims = {nLabels};
        for(size_t nNodeIdx = 0; nNodeIdx<nNodes; ++nNodeIdx) {
            m_vStereoVisSimUnaryFuncs.push_back(m_pStereoModel->addFunctionWithRefReturn(StereoExplicitFunction()));
            StereoFunc& oFunc = m_vStereoVisSimUnaryFuncs.back();
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
            oFunc.second.resize(aUnaryFuncDims.begin(),aUnaryFuncDims.end());
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
            oFunc.second.assign(aUnaryFuncDims.begin(),aUnaryFuncDims.end(),m_pStereoVisSimUnaryFuncsDataBase+(nNodeIdx*nLabels));
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
            lvDbgAssert(oFunc.second.strides(0)==1); // expect no padding
            const std::array<size_t,1> aNodeIndices = {nNodeIdx};
            m_vNodeInfos[nNodeIdx].nStereoVisSimUnaryFactID = m_pStereoModel->addFactorNonFinalized(oFunc.first,aNodeIndices.begin(),aNodeIndices.end());
            m_vNodeInfos[nNodeIdx].pStereoVisSimUnaryFunc = &oFunc;
        }
    }
    {
        lvLog(1,"\tadding label similarity factor for each grid node pair...");
        // note: current def w/ explicit function will require too much memory if using >>50 labels
        const std::array<size_t,2> aPairwiseFuncDims = {nLabels,nLabels};
        std::array<size_t,2> aNodeIndices;
        for(size_t nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(size_t nColIdx=0; nColIdx<nCols; ++nColIdx) {
                aNodeIndices[0] = nRowIdx*nCols+nColIdx;
                if(nRowIdx+1<nRows) { // vertical pair
                    aNodeIndices[1] = (nRowIdx+1)*nCols+nColIdx;
                    m_avStereoSmoothPairwFuncs[0].push_back(m_pStereoModel->addFunctionWithRefReturn(StereoExplicitFunction()));
                    StereoFunc& oFunc = m_avStereoSmoothPairwFuncs[0].back();
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.resize(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end());
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.assign(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end(),m_pStereoSmoothPairwFuncsDataBase+((aNodeIndices[0]*2)*nLabels*nLabels));
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oFunc.second.strides(0)==1 && oFunc.second.strides(1)==nLabels); // expect last-idx-major
                    m_vNodeInfos[aNodeIndices[0]].anStereoSmoothPairwFactIDs[0] = m_pStereoModel->addFactorNonFinalized(oFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].anPairwNodeIdxs[0] = aNodeIndices[1];
                    m_vNodeInfos[aNodeIndices[0]].apStereoSmoothPairwFuncs[0] = &oFunc;
                }
                if(nColIdx+1<nCols) { // horizontal pair
                    aNodeIndices[1] = nRowIdx*nCols+nColIdx+1;
                    m_avStereoSmoothPairwFuncs[1].push_back(m_pStereoModel->addFunctionWithRefReturn(StereoExplicitFunction()));
                    StereoFunc& oFunc = m_avStereoSmoothPairwFuncs[1].back();
#if FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.resize(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end());
#else //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    oFunc.second.assign(aPairwiseFuncDims.begin(),aPairwiseFuncDims.end(),m_pStereoSmoothPairwFuncsDataBase+((aNodeIndices[0]*2+1)*nLabels*nLabels));
#endif //!FGSTEREOMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oFunc.second.strides(0)==1 && oFunc.second.strides(1)==nLabels); // expect last-idx-major
                    m_vNodeInfos[aNodeIndices[0]].anStereoSmoothPairwFactIDs[1] = m_pStereoModel->addFactorNonFinalized(oFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].anPairwNodeIdxs[1] = aNodeIndices[1];
                    m_vNodeInfos[aNodeIndices[0]].apStereoSmoothPairwFuncs[1] = &oFunc;
                }
            }
        }
    }/*{
     // add 3rd order function and factors to the model (test)
        const std::array<LabelType@,3> aHOEFuncDims = {nLabels,nLabels,nLabels};
        ExplicitFunction vHOEFunc(aHOEFuncDims.begin(),aHOEFuncDims.end(),0.5f);
        FunctionID nFID = m_pGM->addFunction(vHOEFunc);
        for(size_t nLabelIdx1=0; nLabelIdx1<nRealLabels; ++nLabelIdx1) {
            for(size_t nLabelIdx2 = 0; nLabelIdx2<nRealLabels; ++nLabelIdx2) {
                for(size_t nLabelIdx3 = 0; nLabelIdx3<nRealLabels; ++nLabelIdx3) {
                    ...
                }
            }
        }
        for(size_t nNodeIdx=nPxGridNodes; nNodeIdx<vNodeLabelCounts.size(); ++nNodeIdx) {
            const size_t nRandNodeIdx1 = ((rand()%nRows)*nCols+rand()%nCols);
            const size_t nRandNodeIdx2 = ((rand()%nRows)*nCols+rand()%nCols);
            const std::array<size_t,3> aNodeIndices = {nRandNodeIdx1<nRandNodeIdx2?nRandNodeIdx1:nRandNodeIdx2,nRandNodeIdx1<nRandNodeIdx2?nRandNodeIdx2:nRandNodeIdx1,nNodeIdx};
            m_pGM->addFactorNonFinalized(nFID,aNodeIndices.begin(),aNodeIndices.end());
        }
    }*/
    m_pStereoModel->finalize();
    lvLog_(1,"Stereo model constructed in %f second(s).",oLocalTimer.tock());
    lv::gm::printModelInfo(*m_pStereoModel);

    // @@@@ add resegm struct init here
    // @@@@@@@ RETEST RESEGM GRAPH STABILITY FROM SANDBOX W/ NEW STRUCTURE

}

inline void FGStereoMatcher::GraphModelData::resetLabelings() {
    const StereoModelType& oGM = *m_pStereoModel;
    const size_t nLabels = m_vStereoLabels.size();
    const size_t nRealLabels = m_vStereoLabels.size()-2;
    const size_t nNodes = m_oGridSize.total();
    lvDbgAssert(nLabels>3 && nNodes==m_vNodeInfos.size());
    lvIgnore(oGM);
    if(m_eStereoLabelInitType==LabelInit_Default) {
        m_oStereoLabeling.create(m_oGridSize);
        std::fill(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),(StereoLabelType)0);
    }
    else if(m_eStereoLabelInitType==LabelInit_Random) {
        m_oStereoLabeling.create(m_oGridSize);
        std::mt19937 oGen(m_nStereoLabelingRandomSeed);
        std::generate(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),[&](){return StereoLabelType(oGen()%nRealLabels);});
    }
    else if(m_eStereoLabelInitType==LabelInit_LocalOptim) {
        m_oStereoLabeling.create(m_oGridSize);
        std::fill(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),(StereoLabelType)0);
        for(size_t nNodeIdx=0; nNodeIdx<nNodes; ++nNodeIdx) {
            const NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
            if(oNode.nStereoVisSimUnaryFactID!=SIZE_MAX) {
                lvDbgAssert(oNode.nStereoVisSimUnaryFactID<oGM.numberOfFactors());
                lvDbgAssert(oGM.numberOfLabels(oNode.nStereoVisSimUnaryFactID)==nLabels);
                lvDbgAssert(StereoFuncID(oGM[oNode.nStereoVisSimUnaryFactID].functionIndex(),oGM[oNode.nStereoVisSimUnaryFactID].functionType())==oNode.pStereoVisSimUnaryFunc->first);
                StereoLabelType nEvalLabel = 0; // == value already assigned via std::fill
                ValueType fOptimalEnergy = oNode.pStereoVisSimUnaryFunc->second(&nEvalLabel);
                for(nEvalLabel = 1; nEvalLabel<nLabels; ++nEvalLabel) {
                    const ValueType fCurrEnergy = oNode.pStereoVisSimUnaryFunc->second(&nEvalLabel);
                    if(fOptimalEnergy>fCurrEnergy) {
                        m_oStereoLabeling(oNode.nRowIdx,oNode.nColIdx) = nEvalLabel;
                        fOptimalEnergy = fCurrEnergy;
                    }
                }
            }
        }
    }
    else /*if(m_eStereoLabelInitType==LabelInit_Explicit)*/ {
        lvAssert_(m_oStereoLabeling.isContinuous(),"internal labeling must be continuous for easier assignments");
        lvAssert_(m_oGridSize==m_oStereoLabeling.size && m_oStereoLabeling.total()==m_pStereoModel->numberOfVariables(),"internal labeling mat size & gm node count mismatch");
    }
    m_oAssocCounts = (AssocCountType)0;
    m_oAssocMap = (AssocIdxType)-1;
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<(int)m_oGridSize[1]; ++nColIdx) {
            const StereoLabelType nLabel = m_oStereoLabeling(nRowIdx,nColIdx);
            if(nLabel<m_nStereoDontCareLabelIdx) // both special labels avoided here
                addAssoc(nRowIdx,nColIdx,nLabel);
        }
    }
    if(m_eStereoLabelOrderType==LabelOrder_Default) {
        m_vStereoLabelOrdering.resize(nLabels);
        std::iota(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),0);
    }
    else if(m_eStereoLabelOrderType==LabelOrder_Random) {
        m_vStereoLabelOrdering.resize(nLabels);
        std::iota(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),0);
        std::mt19937 oGen(m_nStereoLabelOrderRandomSeed);
        std::shuffle(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),oGen);
    }
    else /*if(m_eStereoLabelOrderType==LabelOrder_Explicit)*/ {
        lvAssert_(m_vStereoLabelOrdering.size()==nLabels,"label order array did not contain all labels");
        lvAssert_(lv::unique(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end())==lv::make_range(StereoLabelType(0),StereoLabelType(nLabels-1)),"label order array did not contain all labels");
    }
    // @@@@@@@@@@ resegm todo
}

inline void FGStereoMatcher::GraphModelData::updateModels(const MatArrayIn& aInputs) {
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx)
        lvDbgAssert__(aInputs[nInputIdx].dims==2 && m_oGridSize==aInputs[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvDbgAssert_(aInputs[nCamIdx*2+1].type()==CV_8UC1,"unexpected input mask type");
        /*const cv::Point2i oDisplayOffset = {-(int)m_nGridBorderSize,-(int)m_nGridBorderSize};
        lvCout << "input" << nCamIdx << " = " << lv::to_string(aInputs[nCamIdx*2],oDisplayOffset) << std::endl;*/
        cv::imshow(std::string("input")+std::to_string(nCamIdx),aInputs[nCamIdx*2]);
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    const size_t nLabels = m_vStereoLabels.size();
    lvDbgAssert(nLabels>3);
    const size_t nRealLabels = nLabels-2;
    const int nMinDescColIdx = (int)m_nGridBorderSize;
    const int nMinDescRowIdx = (int)m_nGridBorderSize;
    const int nMaxDescColIdx = nCols-(int)m_nGridBorderSize-1;
    const int nMaxDescRowIdx = nRows-(int)m_nGridBorderSize-1;
    const auto lHasValidDesc = [&nMinDescRowIdx,&nMaxDescRowIdx,&nMinDescColIdx,&nMaxDescColIdx](int nRowIdx, int nColIdx) {
        return nRowIdx>=nMinDescRowIdx && nRowIdx<=nMaxDescRowIdx && nColIdx>=nMinDescColIdx && nColIdx<=nMaxDescColIdx;
    };
    lvAssert_(!m_bUsePrecalcFeatsNext || m_vNextFeats.size()==FeatPackCount*getCameraCount(),"unexpected precalculated features vec size");
    if(!m_bUsePrecalcFeatsNext)
        calcFeatures(aInputs);
    else
        m_bUsePrecalcFeatsNext = false;
    static_assert(getCameraCount()==2,"lots of stuff hardcoded below for 2-cam stereo");
    const cv::Mat_<float> oVisDistMap = m_vNextFeats[FeatPack_VisDist];
    const cv::Mat_<float> oShpDistMap = m_vNextFeats[FeatPack_ShpDist];
    lvDbgAssert(oVisDistMap.dims==3 && oVisDistMap.size[0]==nRows && oVisDistMap.size[1]==nCols && oVisDistMap.size[2]==(int)nRealLabels);
    lvDbgAssert(oShpDistMap.dims==3 && oShpDistMap.size[0]==nRows && oShpDistMap.size[1]==nCols && oShpDistMap.size[2]==(int)nRealLabels);
    const std::array<cv::Mat_<float>,2> aFGDistMaps = {m_vNextFeats[FeatPack_FGDist],m_vNextFeats[FeatPackCount+FeatPack_FGDist]};
    const std::array<cv::Mat_<float>,2> aBGDistMaps = {m_vNextFeats[FeatPack_BGDist],m_vNextFeats[FeatPackCount+FeatPack_BGDist]};
    lvDbgAssert(lv::MatInfo(aFGDistMaps[0])==lv::MatInfo(aFGDistMaps[1]) && m_oGridSize==aFGDistMaps[0].size);
    lvDbgAssert(lv::MatInfo(aBGDistMaps[0])==lv::MatInfo(aBGDistMaps[1]) && m_oGridSize==aBGDistMaps[0].size);
    const std::array<cv::Mat_<float>,2> aFGSimMaps = {m_vNextFeats[FeatPack_FGSim],m_vNextFeats[FeatPackCount+FeatPack_FGSim]};
    const std::array<cv::Mat_<float>,2> aBGSimMaps = {m_vNextFeats[FeatPack_BGSim],m_vNextFeats[FeatPackCount+FeatPack_BGSim]};
    lvDbgAssert(lv::MatInfo(aFGSimMaps[0])==lv::MatInfo(aFGSimMaps[1]) && m_oGridSize==aFGSimMaps[0].size);
    lvDbgAssert(lv::MatInfo(aBGSimMaps[0])==lv::MatInfo(aBGSimMaps[1]) && m_oGridSize==aBGSimMaps[0].size);
    /*for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::Rect test(0,128,256,1);
        lvCout << "input" << nCamIdx*2 << " = " << lv::to_string(aInputs[nCamIdx*2](test)) << std::endl;
        lvCout << "aFGDist" << nCamIdx << " = " << lv::to_string(aFGDist[nCamIdx](test)) << std::endl;
        double max;
        cv::minMaxIdx(aFGDist[nCamIdx],nullptr,&max);
        cv::imshow(std::string("aFGDist")+std::to_string(nCamIdx),aFGDist[nCamIdx]/max);
        lvCout << "aBGDist" << nCamIdx << " = " << lv::to_string(aBGDist[nCamIdx](test)) << std::endl;
        cv::minMaxIdx(aBGDist[nCamIdx],nullptr,&max);
        cv::imshow(std::string("aBGDist")+std::to_string(nCamIdx),aBGDist[nCamIdx]/max);
    }
    cv::waitKey(0);*/
    const std::array<cv::Mat_<uchar>,2> aGradYMaps = {m_vNextFeats[FeatPack_GradY],m_vNextFeats[FeatPackCount+FeatPack_GradY]};
    const std::array<cv::Mat_<uchar>,2> aGradXMaps = {m_vNextFeats[FeatPack_GradX],m_vNextFeats[FeatPackCount+FeatPack_GradX]};
    const std::array<cv::Mat_<uchar>,2> aGradMagMaps = {m_vNextFeats[FeatPack_GradMag],m_vNextFeats[FeatPackCount+FeatPack_GradMag]};
    lvDbgAssert(lv::MatInfo(aGradYMaps[0])==lv::MatInfo(aGradYMaps[1]) && m_oGridSize==aGradYMaps[0].size);
    lvDbgAssert(lv::MatInfo(aGradXMaps[0])==lv::MatInfo(aGradXMaps[1]) && m_oGridSize==aGradXMaps[0].size);
    lvDbgAssert(lv::MatInfo(aGradMagMaps[0])==lv::MatInfo(aGradMagMaps[1]) && m_oGridSize==aGradMagMaps[0].size);
    /*for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Point2i oDisplayOffset = {-(int)m_nGridBorderSize,-(int)m_nGridBorderSize};
        lvCout << "gradY" << nCamIdx << " = " << lv::to_string(aGradY[nCamIdx],oDisplayOffset) << std::endl;
        lvCout << "gradX" << nCamIdx << " = " << lv::to_string(aGradX[nCamIdx],oDisplayOffset) << std::endl;
        lvCout << "gradM" << nCamIdx << " = " << lv::to_string(aGradMag[nCamIdx],oDisplayOffset) << std::endl;
    }*/
    lvLog(1,"Updating graph models energy terms based on input data...");
    lv::StopWatch oLocalTimer;
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::mutex oPrintMutex;
    #if USING_OPENMP
    #pragma omp parallel for collapse(2)
    #endif //USING_OPENMP
    for(int nRowIdx = 0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<nCols; ++nColIdx) {
            const size_t nNodeIdx = size_t(nRowIdx*nCols+nColIdx);
            const GraphModelData::NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
            lvDbgAssert(oNode.nRowIdx==nRowIdx && oNode.nColIdx==nColIdx);
            //const bool bValidDescNode = lHasValidDesc(nRowIdx,nColIdx);

            // update stereo unary term for each grid node
            lvDbgAssert(oNode.pStereoVisSimUnaryFunc);
            lvDbgAssert(oNode.nStereoVisSimUnaryFactID!=SIZE_MAX);
            lvDbgAssert((&m_pStereoModel->getFunction<StereoExplicitFunction>(oNode.pStereoVisSimUnaryFunc->first))==(&oNode.pStereoVisSimUnaryFunc->second));
            StereoExplicitFunction& vUnaryFunc = oNode.pStereoVisSimUnaryFunc->second;
            lvDbgAssert(vUnaryFunc.dimension()==1 && vUnaryFunc.size()==nLabels);
            lvDbgAssert(m_pVisDescExtractor->defaultNorm()==cv::NORM_L2);
            const float* pVisDistPtr = oVisDistMap.ptr<float>(nRowIdx,nColIdx);
            const float* pShpDistPtr = oShpDistMap.ptr<float>(nRowIdx,nColIdx);
            for(StereoLabelType nLabelIdx=0; nLabelIdx<nRealLabels; ++nLabelIdx) {
                const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                const int nOffsetColIdx = nColIdx-nRealLabel;
                //const bool bValidDescOffsetNode = lHasValidDesc(nRowIdx,nOffsetColIdx); // @@@ update to use roi?
                vUnaryFunc(nLabelIdx) = ValueType(0);
                const float& fVisDist = pVisDistPtr[nLabelIdx];
                const float& fShpDist = pShpDistPtr[nLabelIdx];
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    if(!(fVisDist>=0.0f && fShpDist>=0.0f))
                        lvPrint(fVisDist) << "nRowIdx="<<nRowIdx << ", nColIdx=" << nColIdx <<", nLabelIdx="<<(int)nLabelIdx<<"\n";
                    lvAssert(fVisDist>=0.0f && fShpDist>=0.0f);
                    vUnaryFunc(nLabelIdx) += ValueType(fVisDist*FGSTEREOMATCH_VISSIM_COST_DESC_SCALE);
                    vUnaryFunc(nLabelIdx) += ValueType(fShpDist*FGSTEREOMATCH_SHPSIM_COST_DESC_SCALE);
                }
                else {
                    lvAssert(fVisDist<1.0f && fShpDist<0.0f);
                    vUnaryFunc(nLabelIdx) += FGSTEREOMATCH_UNARY_COST_OOB_CST;
                }
                vUnaryFunc(nLabelIdx) = std::min(FGSTEREOMATCH_UNARY_COST_MAXTRUNC_CST,vUnaryFunc(nLabelIdx));
            }
            // @@@@ add discriminativeness factor --- vect-append all vis sim vals, sort, setup discrim costs
            vUnaryFunc(m_nStereoDontCareLabelIdx) = ValueType(100000); // @@@@ check roi, if dc set to 0, otherwise set to inf
            vUnaryFunc(m_nStereoOccludedLabelIdx) = ValueType(100000);//FGSTEREOMATCH_VISSIM_COST_OCCLUDED_CST;

            // update stereo smoothness pairwise terms for each grid node
            for(size_t nOrientIdx=0; nOrientIdx<oNode.anStereoSmoothPairwFactIDs.size(); ++nOrientIdx) {
                if(oNode.anStereoSmoothPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                    lvDbgAssert(oNode.apStereoSmoothPairwFuncs[nOrientIdx]);
                    lvDbgAssert((&m_pStereoModel->getFunction<StereoExplicitFunction>(oNode.apStereoSmoothPairwFuncs[nOrientIdx]->first))==(&oNode.apStereoSmoothPairwFuncs[nOrientIdx]->second));
                    StereoExplicitFunction& vPairwiseFunc = oNode.apStereoSmoothPairwFuncs[nOrientIdx]->second;
                    lvDbgAssert(vPairwiseFunc.dimension()==2 && vPairwiseFunc.size()==nLabels*nLabels);
                    for(StereoLabelType nLabelIdx1=0; nLabelIdx1<nRealLabels; ++nLabelIdx1) {
                        for(StereoLabelType nLabelIdx2=0; nLabelIdx2<nRealLabels; ++nLabelIdx2) {
                            const OutputLabelType nRealLabel1 = getRealLabel(nLabelIdx1);
                            const OutputLabelType nRealLabel2 = getRealLabel(nLabelIdx2);
                            const int nRealLabelDiff = std::min(std::abs((int)nRealLabel1-(int)nRealLabel2),FGSTEREOMATCH_LBLSIM_COST_MAXDIFF_CST);
                            const int nLocalGrad = (int)((nOrientIdx==0)?aGradYMaps[0]:(nOrientIdx==1)?aGradXMaps[0]:aGradMagMaps[0])(nRowIdx,nColIdx);
                            const float fGradScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                            lvDbgAssert(fGradScaleFact==(float)std::exp(float(FGSTEREOMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/FGSTEREOMATCH_LBLSIM_COST_GRADRAW_SCALE));
                            vPairwiseFunc(nLabelIdx1,nLabelIdx2) = ValueType((nRealLabelDiff*nRealLabelDiff)*fGradScaleFact);
                            const bool bValidDescOffsetNode1 = lHasValidDesc(nRowIdx,nColIdx-(int)nRealLabel1);
                            const bool bValidDescOffsetNode2 = lHasValidDesc(nRowIdx,nColIdx-(int)nRealLabel2);
                            if(!bValidDescOffsetNode1 || !bValidDescOffsetNode2)
                                vPairwiseFunc(nLabelIdx1,nLabelIdx2) *= 2; // incr smoothness cost for node pairs in border regions to outweight bad unaries
                        }
                    }
                    for(size_t nLabelIdx=0; nLabelIdx<nRealLabels; ++nLabelIdx) {
                        // @@@ change later for vis-data-dependent or roi-dependent energies?
                        // @@@@ outside-roi-dc to inside-roi-something is ok (0 cost)
                        vPairwiseFunc(m_nStereoDontCareLabelIdx,nLabelIdx) = ValueType(100000);
                        vPairwiseFunc(m_nStereoOccludedLabelIdx,nLabelIdx) = ValueType(100000); // @@@@ FGSTEREOMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                        vPairwiseFunc(nLabelIdx,m_nStereoDontCareLabelIdx) = ValueType(100000);
                        vPairwiseFunc(nLabelIdx,m_nStereoOccludedLabelIdx) = ValueType(100000); // @@@@ FGSTEREOMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                    }
                    vPairwiseFunc(m_nStereoDontCareLabelIdx,m_nStereoDontCareLabelIdx) = ValueType(0);
                    vPairwiseFunc(m_nStereoOccludedLabelIdx,m_nStereoOccludedLabelIdx) = ValueType(0);
                }
            }
            const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
            }
        }
    }
    lv::cleanConsoleRow();
    lvLog_(1,"Graph energy terms update completed in %f second(s).",oLocalTimer.tock());
    resetLabelings();
    m_bModelUpToDate = true;
}

inline void FGStereoMatcher::GraphModelData::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    static_assert(getCameraCount()==2,"lots of hardcoded indices below");
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvAssert__(aInputs[nCamIdx*2].dims==2 && m_oGridSize==aInputs[nCamIdx*2].size(),"input image in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert__(aInputs[nCamIdx*2+1].dims==2 && m_oGridSize==aInputs[nCamIdx*2+1].size(),"input mask in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(aInputs[nCamIdx*2].type()==CV_8UC1 || aInputs[nCamIdx*2].type()==CV_8UC3,"unexpected input image type");
        lvAssert_(aInputs[nCamIdx*2+1].type()==CV_8UC1,"unexpected input mask type");
    }
    m_vNextFeats.resize(FeatPackCount*getCameraCount());
    lvLog(1,"Calculating features maps...");
    lv::StopWatch oLocalTimer;
    #if USING_OPENMP
    #pragma omp parallel for num_threads(getCameraCount())
    #endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvDbgAssert(m_pVisDescExtractor && m_pShpDescExtractor);
        m_pVisDescExtractor->compute2(aInputs[nCamIdx*2],m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_VisDescs]);
        m_pShpDescExtractor->compute2(aInputs[nCamIdx*2+1],m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_ShpDescs]);
        cv::distanceTransform(aInputs[nCamIdx*2+1]==0,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_FGDist],cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
        cv::exp(FGSTEREOMATCH_DEFAULT_DISTTRANSF_SCALE*m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_FGDist],m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_FGSim]);
        //lvPrint(cv::Mat_<float>(m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_FGSim](cv::Rect(0,128,256,1))));
        cv::divide(1.0,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_FGSim]+FGSTEREOMATCH_DEFAULT_DISTTRANSF_OFFSET,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_FGDist]);
        //lvPrint(cv::Mat_<float>(m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_FGDist](cv::Rect(0,128,256,1))));
        cv::distanceTransform(aInputs[nCamIdx*2+1]>0,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_BGDist],cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
        cv::exp(FGSTEREOMATCH_DEFAULT_DISTTRANSF_SCALE*m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_BGDist],m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_BGSim]);
        //lvPrint(cv::Mat_<float>(m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_BGSim](cv::Rect(0,128,256,1))));
        cv::divide(1.0,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_BGSim]+FGSTEREOMATCH_DEFAULT_DISTTRANSF_OFFSET,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_BGDist]);
        //lvPrint(cv::Mat_<float>(m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_BGDist](cv::Rect(0,128,256,1))));
        cv::Mat oBlurredInput;
        cv::GaussianBlur(aInputs[nCamIdx*2],oBlurredInput,cv::Size(3,3),0);
        cv::Mat oBlurredGrayInput;
        if(oBlurredInput.channels()==3)
            cv::cvtColor(oBlurredInput,oBlurredGrayInput,cv::COLOR_BGR2GRAY);
        else
            oBlurredGrayInput = oBlurredInput;
        cv::Mat oGradInput_X,oGradInput_Y;
        cv::Sobel(oBlurredGrayInput,oGradInput_Y,CV_16S,0,1,FGSTEREOMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::Sobel(oBlurredGrayInput,oGradInput_X,CV_16S,1,0,FGSTEREOMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::convertScaleAbs(oGradInput_Y,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_GradY]);
        cv::convertScaleAbs(oGradInput_X,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_GradX]);
        cv::addWeighted(m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_GradY],0.5,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_GradX],0.5,0,m_vNextFeats[nCamIdx*FeatPackCount+FeatPack_GradMag]);
    }
    lvLog_(1,"Features maps computed in %f second(s).",oLocalTimer.tock());
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    const size_t nLabels = m_vStereoLabels.size();
    lvDbgAssert(nLabels>3);
    const size_t nRealLabels = nLabels-2;
    const int nMinDescColIdx = (int)m_nGridBorderSize;
    const int nMinDescRowIdx = (int)m_nGridBorderSize;
    const int nMaxDescColIdx = nCols-(int)m_nGridBorderSize-1;
    const int nMaxDescRowIdx = nRows-(int)m_nGridBorderSize-1;
    const auto lHasValidDesc = [&nMinDescRowIdx,&nMaxDescRowIdx,&nMinDescColIdx,&nMaxDescColIdx](int nRowIdx, int nColIdx) {
        return nRowIdx>=nMinDescRowIdx && nRowIdx<=nMaxDescRowIdx && nColIdx>=nMinDescColIdx && nColIdx<=nMaxDescColIdx;
    };
    const std::array<cv::Mat_<float>,2> aVisDescs = {m_vNextFeats[FeatPack_VisDescs],m_vNextFeats[FeatPackCount+FeatPack_VisDescs]};
    const std::array<cv::Mat_<float>,2> aShpDescs = {m_vNextFeats[FeatPack_ShpDescs],m_vNextFeats[FeatPackCount+FeatPack_ShpDescs]};
    lvDbgAssert(lv::MatInfo(aVisDescs[0])==lv::MatInfo(aVisDescs[1]) && aVisDescs[0].dims==3);
    lvDbgAssert(lv::MatInfo(aShpDescs[0])==lv::MatInfo(aShpDescs[1]) && aShpDescs[0].dims==3);
    lvDbgAssert(aVisDescs[0].size[0]==nRows && aVisDescs[0].size[1]==nCols);
    lvDbgAssert(aShpDescs[0].size[0]==nRows && aShpDescs[0].size[1]==nCols);
    const std::array<int,3> anDistMapDims = {nRows,nCols,(int)nRealLabels};
    m_vNextFeats[FeatPack_VisDist].create(3,anDistMapDims.data(),CV_32FC1);
    m_vNextFeats[FeatPack_ShpDist].create(3,anDistMapDims.data(),CV_32FC1);
    m_vNextFeats[FeatPack_VisDist] = -1.0f; // all OOB by default
    m_vNextFeats[FeatPack_ShpDist] = -1.0f; // all OOB by default
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::mutex oPrintMutex;
    lvLog(1,"Calculating affinity maps...");
    #if USING_OPENMP
    #pragma omp parallel for collapse(2)
    #endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float* pVisDistPtr = m_vNextFeats[FeatPack_VisDist].ptr<float>(nRowIdx,nColIdx);
            float* pShpDistPtr = m_vNextFeats[FeatPack_ShpDist].ptr<float>(nRowIdx,nColIdx);
            const bool bValidDescNode = lHasValidDesc(nRowIdx,nColIdx);
            lvDbgAssert(m_pVisDescExtractor->defaultNorm()==cv::NORM_L2);
            for(StereoLabelType nLabelIdx=0; nLabelIdx<nRealLabels; ++nLabelIdx) {
                const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                const int nOffsetColIdx = nColIdx-nRealLabel;
                const bool bValidDescOffsetNode = lHasValidDesc(nRowIdx,nOffsetColIdx); // @@@ update to use roi?
                // calc inter-spectral visual descriptor distance
                if(bValidDescNode && bValidDescOffsetNode) {
                    // test w/ root-sift transform here? @@@@@
                    pVisDistPtr[nLabelIdx] = (float)m_pVisDescExtractor->calcDistance(aVisDescs[0].ptr<float>(nRowIdx,nColIdx),aVisDescs[1].ptr<float>(nRowIdx,nOffsetColIdx));
                    // @@@@ check if descs are really normalized by impl
                }
                else if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    lvDbgAssert(aInputs[0].type()==CV_8UC1 && aInputs[2].type()==CV_8UC1);
                    const int nCurrVisDiff = (int)aInputs[0].at<uchar>(nRowIdx,nColIdx)-(int)aInputs[2].at<uchar>(nRowIdx,nOffsetColIdx);
                    pVisDistPtr[nLabelIdx] = (std::abs(nCurrVisDiff)*FGSTEREOMATCH_VISSIM_COST_RAW_SCALE);
                }
                // calc inter-spectral shape descriptor distance
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    // test w/ root-sift transform here? @@@@@
                    pShpDistPtr[nLabelIdx] = (float)m_pShpDescExtractor->calcDistance(aShpDescs[0].ptr<float>(nRowIdx,nColIdx),aShpDescs[1].ptr<float>(nRowIdx,nOffsetColIdx));
                }
            }
            const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
            }
        }
    }
    lv::cleanConsoleRow();
    // same affinity maps for 2nd camera, but using negative disparity offsets indices inside
    m_vNextFeats[FeatPack_VisDist].copyTo(m_vNextFeats[FeatPackCount+FeatPack_VisDist]);
    m_vNextFeats[FeatPack_ShpDist].copyTo(m_vNextFeats[FeatPackCount+FeatPack_ShpDist]);
    lvLog_(1,"Affinity maps computed in %f second(s).",oLocalTimer.tock());
    if(pFeatsPacket)
        *pFeatsPacket = lv::packData(m_vNextFeats,&m_vNextFeatPackInfo);
    else {
        m_vNextFeatPackInfo.resize(m_vNextFeats.size());
        for(size_t nFeatMapIdx=0; nFeatMapIdx<m_vNextFeats.size(); ++nFeatMapIdx)
            m_vNextFeatPackInfo[nFeatMapIdx] = lv::MatInfo(m_vNextFeats[nFeatMapIdx]);
    }
    if(m_vExpectedFeatPackInfo.empty())
        m_vExpectedFeatPackInfo = m_vNextFeatPackInfo;
    lvAssert_(m_vNextFeatPackInfo==m_vExpectedFeatPackInfo,"packed features info mismatch (should stay constant for all inputs)");
    m_bModelUpToDate = false;
}

inline void FGStereoMatcher::GraphModelData::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvAssert_(!oPackedFeats.empty(),"features packet must be non-empty");
    lvDbgExceptionWatch;
    if(m_vExpectedFeatPackInfo.empty()) {
        const int nRows = (int)m_oGridSize(0);
        const int nCols = (int)m_oGridSize(1);
        const int nRealLabels = (int)m_vStereoLabels.size()-2;
        lvDbgAssert(nRealLabels>1);
        m_vExpectedFeatPackInfo.resize(FeatPackCount*getCameraCount());
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_VisDescs] = m_pVisDescExtractor->getOutputInfo(lv::MatInfo(m_oGridSize,CV_8UC3));
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_ShpDescs] = m_pShpDescExtractor->getOutputInfo(lv::MatInfo(m_oGridSize,CV_8UC1));
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_VisDist] = lv::MatInfo(std::array<int,3>{nRows,nCols,nRealLabels},CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_ShpDist] = lv::MatInfo(std::array<int,3>{nRows,nCols,nRealLabels},CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_FGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_BGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_FGSim] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_BGSim] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_GradY] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_GradX] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackCount+FeatPack_GradMag] = lv::MatInfo(m_oGridSize,CV_8UC1);
        }
    }
    m_oNextPackedFeats = oPackedFeats; // get pointer to data instead of copy; provider should not overwrite until next 'apply' call!
    m_vNextFeats = lv::unpackData(m_oNextPackedFeats,m_vExpectedFeatPackInfo);
    m_bUsePrecalcFeatsNext = true;
    m_bModelUpToDate = false;
}

inline FGStereoMatcher::OutputLabelType FGStereoMatcher::GraphModelData::getRealLabel(StereoLabelType nLabel) const {
    lvDbgAssert(nLabel<m_vStereoLabels.size());
    return m_vStereoLabels[nLabel];
}

inline FGStereoMatcher::StereoLabelType FGStereoMatcher::GraphModelData::getInternalLabel(OutputLabelType nRealLabel) const {
    lvDbgAssert(nRealLabel==s_nStereoOccludedLabel || nRealLabel==s_nStereoDontCareLabel || (nRealLabel>=(OutputLabelType)m_nMinDispOffset && nRealLabel<=(OutputLabelType)m_nMaxDispOffset));
    return (StereoLabelType)std::distance(m_vStereoLabels.begin(),std::find(m_vStereoLabels.begin(),m_vStereoLabels.end(),nRealLabel));
}

inline FGStereoMatcher::AssocCountType FGStereoMatcher::GraphModelData::getAssocCount(int nRowIdx, int nColIdx) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(nColIdx>=-(int)m_nMaxDispOffset && nColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    return ((AssocCountType*)m_oAssocCounts.data)[(nRowIdx/m_nDispOffsetStep)*m_oAssocCounts.cols + (nColIdx+m_nMaxDispOffset)/m_nDispOffsetStep];
}

inline void FGStereoMatcher::GraphModelData::addAssoc(int nRowIdx, int nColIdx, StereoLabelType nLabel) const {
    lvDbgAssert(nLabel<m_nStereoDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
    lvDbgAssert(nAssocColIdx<=nColIdx && nAssocColIdx>=-(int)m_nMaxDispOffset && nAssocColIdx<(int)(m_oGridSize[1]-m_nMinDispOffset));
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_oAssocMap.data)+((nRowIdx/m_nDispOffsetStep)*m_oAssocMap.size[1] + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)*m_oAssocMap.size[2];
    lvDbgAssert(pAssocList==m_oAssocMap.ptr<AssocIdxType>(nRowIdx/m_nDispOffsetStep,(nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_oAssocMap.size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(-1));
    pAssocList[nListOffset] = AssocIdxType(nColIdx);
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[(nRowIdx/m_nDispOffsetStep)*m_oAssocCounts.cols + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(int(nRowIdx/m_nDispOffsetStep),int((nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount<std::numeric_limits<AssocCountType>::max());
    ++nAssocCount;
}

inline void FGStereoMatcher::GraphModelData::removeAssoc(int nRowIdx, int nColIdx, StereoLabelType nLabel) const {
    lvDbgAssert(nLabel<m_nStereoDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
    lvDbgAssert(nAssocColIdx<=nColIdx && nAssocColIdx>=-(int)m_nMaxDispOffset && nAssocColIdx<(int)(m_oGridSize[1]-m_nMinDispOffset));
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_oAssocMap.data)+((nRowIdx/m_nDispOffsetStep)*m_oAssocMap.size[1] + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)*m_oAssocMap.size[2];
    lvDbgAssert(pAssocList==m_oAssocMap.ptr<AssocIdxType>(nRowIdx/m_nDispOffsetStep,(nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_oAssocMap.size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(nColIdx));
    pAssocList[nListOffset] = AssocIdxType(-1);
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[(nRowIdx/m_nDispOffsetStep)*m_oAssocCounts.cols + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(int(nRowIdx/m_nDispOffsetStep),int((nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount>AssocCountType(0));
    --nAssocCount;
}

inline FGStereoMatcher::ValueType FGStereoMatcher::GraphModelData::calcAddAssocCost(int nRowIdx, int nColIdx, StereoLabelType nLabel) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nStereoDontCareLabelIdx) {
        const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
        lvDbgAssert(nAssocColIdx<=nColIdx && nAssocColIdx>=-(int)m_nMaxDispOffset && nAssocColIdx<(int)(m_oGridSize[1]-m_nMinDispOffset));
        lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
        lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
        const AssocCountType nAssocCount = getAssocCount(nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount<m_aAssocCostAddLUT.size());
        // 'true' cost of adding one assoc to target pixel (target can only have one new assoc per iter, due to single label move)
        return m_aAssocCostAddLUT[nAssocCount];
    }
    return ValueType(100000); // @@@@ dirty
}

inline FGStereoMatcher::ValueType FGStereoMatcher::GraphModelData::calcRemoveAssocCost(int nRowIdx, int nColIdx, StereoLabelType nLabel) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nStereoDontCareLabelIdx) {
        const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
        lvDbgAssert(nAssocColIdx<=nColIdx && nAssocColIdx>=-(int)m_nMaxDispOffset && nAssocColIdx<(int)(m_oGridSize[1]-m_nMinDispOffset));
        lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
        lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
        const AssocCountType nAssocCount = getAssocCount(nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount>0); // cannot be zero, must have at least an association in order to remove it
        // 'average' cost of removing one assoc from target pixel (not 'true' energy, but easy-to-optimize worse case)
        return -m_aAssocCostSumLUT[nAssocCount]/nAssocCount;
    }
    return -ValueType(100000); // @@@@ dirty
}

inline FGStereoMatcher::ValueType FGStereoMatcher::GraphModelData::calcTotalAssocCost() const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
    ValueType tEnergy = ValueType(0);
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx)
        for(int nColIdx=-(int)m_nMaxDispOffset; nColIdx<(int)(m_oGridSize[1]-m_nMinDispOffset); ++nColIdx)
            tEnergy += m_aAssocCostSumLUT[getAssocCount(nRowIdx,nColIdx)];
    // @@@@ really needed?
    const size_t nTotNodeCount = m_oGridSize.total();
    for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
        const StereoLabelType nCurrLabel = ((StereoLabelType*)m_oStereoLabeling.data)[nNodeIdx];
        if(nCurrLabel>=m_nStereoDontCareLabelIdx) // both special labels treated here
            tEnergy += ValueType(100000); // @@@@ dirty
    }
    lvDbgAssert(tEnergy>=ValueType(0));
    return tEnergy;
}

inline void FGStereoMatcher::GraphModelData::calcMoveCosts(StereoLabelType nNewLabel) const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1);
    lvDbgAssert(m_oGridSize==m_oAssocCosts.size && m_oGridSize==m_oUnaryCosts.size && m_oGridSize==m_oPairwCosts.size);
    const size_t nTotNodeCount = m_oGridSize.total();
    // @@@@@ openmp here?
    for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
        const GraphModelData::NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
        const StereoLabelType& nInitLabel = ((StereoLabelType*)m_oStereoLabeling.data)[nNodeIdx];
        lvDbgAssert(&nInitLabel==&m_oStereoLabeling(oNode.nRowIdx,oNode.nColIdx));
        ValueType& tAssocCost = ((ValueType*)m_oAssocCosts.data)[nNodeIdx];
        lvDbgAssert(&tAssocCost==&m_oAssocCosts(oNode.nRowIdx,oNode.nColIdx));
        ValueType& tUnaryCost = ((ValueType*)m_oUnaryCosts.data)[nNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_oUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
        if(nInitLabel!=nNewLabel) {
            const ValueType tAssocEnergyCost = calcRemoveAssocCost(oNode.nRowIdx,oNode.nColIdx,nInitLabel)+calcAddAssocCost(oNode.nRowIdx,oNode.nColIdx,nNewLabel);
            /*const float fAssocScaleFact = float(m_oData.getAssocCount(oNode.nRowIdx,oNode.nColIdx)+1);
            const ValueType tVisSimEnergyInit = ValueType(oNode.pStereoVisSimUnaryFunc->second(&nInitLabel)/fAssocScaleFact);
            const ValueType tVisSimEnergyModif = ValueType(oNode.pStereoVisSimUnaryFunc->second(&nAlphaLabel)/fAssocScaleFact);*/
            const ValueType tVisSimEnergyInit = oNode.pStereoVisSimUnaryFunc->second(&nInitLabel);
            const ValueType tVisSimEnergyModif = oNode.pStereoVisSimUnaryFunc->second(&nNewLabel);
            const ValueType tVisSimEnergyCost = tVisSimEnergyModif-tVisSimEnergyInit;
            tUnaryCost = /*@@@cleanup tAssocEnergyCost+*/tVisSimEnergyCost;
            tAssocCost = tAssocEnergyCost;
        }
        else
            tAssocCost = tUnaryCost = ValueType(0);
        ValueType& tPairwCost = ((ValueType*)m_oPairwCosts.data)[nNodeIdx];
        lvDbgAssert(&tPairwCost==&m_oPairwCosts(oNode.nRowIdx,oNode.nColIdx));
        tPairwCost = ValueType(0);
        for(size_t nOrientIdx=0; nOrientIdx<m_vNodeInfos[nNodeIdx].anStereoSmoothPairwFactIDs.size(); ++nOrientIdx) {
            if(oNode.anStereoSmoothPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                lvDbgAssert(oNode.apStereoSmoothPairwFuncs[nOrientIdx] && oNode.anPairwNodeIdxs[nOrientIdx]<m_oGridSize.total());
                std::array<StereoLabelType,2> aLabels = {nInitLabel,((StereoLabelType*)m_oStereoLabeling.data)[oNode.anPairwNodeIdxs[nOrientIdx]]};
                const ValueType tPairwEnergyInit = oNode.apStereoSmoothPairwFuncs[nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyInit>=ValueType(0));
                aLabels[0] = nNewLabel;
                const ValueType tPairwEnergyModif = oNode.apStereoSmoothPairwFuncs[nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyModif>=ValueType(0));
                tPairwCost += tPairwEnergyModif-tPairwEnergyInit;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline FGStereoMatcher::StereoGraphInference::StereoGraphInference(const GraphModelData& oData) :
        m_oData(oData) {
    lvDbgAssert_(m_oData.m_pStereoModel && m_oData.m_pStereoModel->numberOfFactors()>0,"invalid graph");
    const StereoModelType& oGM = *m_oData.m_pStereoModel;
    for(size_t nFactIdx=0; nFactIdx<m_oData.m_pStereoModel->numberOfFactors(); ++nFactIdx)
        lvDbgAssert__(oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvDbgAssert_(m_oData.m_pStereoModel->numberOfVariables()>0 && m_oData.m_pStereoModel->numberOfVariables()==(IndexType)m_oData.m_oGridSize.total(),"graph node count must match grid size");
    for(size_t nNodeIdx=0; nNodeIdx<m_oData.m_pStereoModel->numberOfVariables(); ++nNodeIdx)
        lvDbgAssert_(m_oData.m_pStereoModel->numberOfLabels(nNodeIdx)==m_oData.m_vStereoLabels.size(),"graph nodes must all have the same number of labels");
    lvIgnore(oGM);
}

inline std::string FGStereoMatcher::StereoGraphInference::name() const {
    return std::string("litiv-stereo-matcher");
}

inline const FGStereoMatcher::StereoModelType& FGStereoMatcher::StereoGraphInference::graphicalModel() const {
    return *m_oData.m_pStereoModel;
}

inline opengm::InferenceTermination FGStereoMatcher::StereoGraphInference::infer() {
    EmptyVisitorType visitor;
    return infer(visitor);
}

inline void FGStereoMatcher::StereoGraphInference::setStartingPoint(typename std::vector<StereoLabelType>::const_iterator begin) {
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_pStereoModel->numberOfVariables()),lv::make_range(StereoLabelType(0),StereoLabelType(m_oData.m_vStereoLabels.size()-1))).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_pStereoModel->numberOfVariables(),m_oData.m_oStereoLabeling.begin());
}

inline void FGStereoMatcher::StereoGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),m_oData.m_vStereoLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_oStereoLabeling.begin(),[&](const OutputLabelType& nRealLabel){return m_oData.getInternalLabel(nRealLabel);});
}

inline opengm::InferenceTermination FGStereoMatcher::StereoGraphInference::arg(std::vector<StereoLabelType>& oLabeling, const size_t n) const {
    if(n==1) {
        lvDbgAssert_(m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_oStereoLabeling.total());
        std::copy(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones!
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void FGStereoMatcher::StereoGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_oStereoLabeling.size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin(),[&](const StereoLabelType& nLabel){return m_oData.getRealLabel(nLabel);});
}

inline FGStereoMatcher::ValueType FGStereoMatcher::StereoGraphInference::value() const {
    lvDbgAssert_(m_oData.m_pStereoModel->numberOfVariables()==m_oData.m_oStereoLabeling.total(),"graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oStereoLabeling.size,"output labeling must be a 2d grid");
    lvAssert_(m_oData.m_bModelUpToDate,"model must be up-to-date with new data before inference can begin");
    return m_oData.m_pStereoModel->evaluate((StereoLabelType*)m_oData.m_oStereoLabeling.data)+m_oData.calcTotalAssocCost();
}

template<typename TVisitor> inline
opengm::InferenceTermination FGStereoMatcher::StereoGraphInference::infer(TVisitor& oVisitor) {
    lvDbgAssert_(m_oData.m_pStereoModel->numberOfVariables()==m_oData.m_oStereoLabeling.total(),"graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oStereoLabeling.size,"output labeling must be a 2d grid");
    lvAssert_(m_oData.m_bModelUpToDate,"model must be up-to-date with new data before inference can begin");
    // @@@@@@ use one gm labeling/output to infer the stereo result for another camera?
    const StereoModelType& oGM = *m_oData.m_pStereoModel;
    const size_t nTotNodeCount = oGM.numberOfVariables();
    lvDbgAssert(nTotNodeCount==m_oData.m_oGridSize.total());
    lvDbgAssert(nTotNodeCount==m_oData.m_vNodeInfos.size());
    const size_t nLabels = m_oData.m_vStereoLabels.size();
    const lv::MatSize& oGridSize = m_oData.m_oGridSize;
    const size_t nGridBorderSize = m_oData.m_nGridBorderSize;
    const cv::Point2i oDisplayOffset = {-(int)m_oData.m_nGridBorderSize,-(int)m_oData.m_nGridBorderSize};
    lvIgnore(oDisplayOffset); // @@@@@@
    const cv::Rect oAssocCountsROI((int)m_oData.m_nMaxDispOffset,0,(int)m_oData.m_oGridSize[1],(int)m_oData.m_oGridSize[0]);
    lvIgnore(oAssocCountsROI); // @@@@@@
    const cv::Rect oFeatROI((int)nGridBorderSize,(int)nGridBorderSize,(int)(oGridSize[1]-nGridBorderSize*2),(int)(oGridSize[0]-nGridBorderSize*2));
    lvIgnore(oFeatROI); // @@@@@@
    // @@@@@@ use member-alloc QPBO here instead of stack
    kolmogorov::qpbo::QPBO<ValueType> oBinaryEnergyMinimizer((int)nTotNodeCount,0); // @@@@@@@ preset max edge count using max clique size times node count
    // @@@@@@ THIS IS A DYNAMIC MRF OPTIMIZATION PROBLEM; MUST TRY TO REUSE MRF STRUCTURE
    std::array<ValueType,s_nMaxCliqueAssign> aCliqueCoeffs;
    std::array<StereoLabelType,s_nMaxOrder> aCliqueLabels;
    std::array<typename HigherOrderEnergy<ValueType,s_nMaxOrder>::VarId,s_nMaxOrder> aTermEnergyLUT;
    cv::Mat_<StereoLabelType>& oLabeling = m_oData.m_oStereoLabeling;
    size_t nMoveIter = 0, nConsecUnchangedLabels = 0, nOrderingIdx = 0;
    lvDbgAssert(m_oData.m_vStereoLabelOrdering.size()==m_oData.m_vStereoLabels.size());
    StereoLabelType nAlphaLabel = m_oData.m_vStereoLabelOrdering[nOrderingIdx];
    lv::StopWatch oLocalTimer;
    ValueType tLastEnergy = value();
    lvIgnore(tLastEnergy);
    oVisitor.begin(*this);
    // each iter below is an alpha-exp move based on A. Fix's primal-dual energy minimization method for higher-order MRFs
    // see "A Primal-Dual Algorithm for Higher-Order Multilabel Markov Random Fields" in CVPR2014 for more info (doi = 10.1109/CVPR.2014.149)
    while(++nMoveIter<=m_oData.m_nMaxIterCount && nConsecUnchangedLabels<nLabels) {
        m_oData.calcMoveCosts(nAlphaLabel);


        if(lv::getVerbosity()>=3) {

            /*lvCout << "\n\n\n\n\n\n@@@ next label idx = " << (int)nAlphaLabel << ",  real = " << (int)m_oData.getRealLabel(nAlphaLabel) << "\n\n" << std::endl;

            lvCout << "disp = " << lv::to_string(m_oData.m_oStereoLabeling,oDisplayOffset) << std::endl;
            lvCout << "assoc_counts = " << lv::to_string(m_oData.m_oAssocCounts(oAssocCountsROI),oDisplayOffset) << std::endl;
            lvCout << "assoc_cost = " << lv::to_string(m_oData.m_oAssocCosts,oDisplayOffset) << std::endl;
            lvCout << "unary = " << lv::to_string(m_oData.m_oUnaryCosts,oDisplayOffset) << std::endl;
            lvCout << "pairw = " << lv::to_string(m_oData.m_oPairwCosts,oDisplayOffset) << std::endl;

            lvCout << "disp = " << lv::to_string(m_oData.m_oStereoLabeling(oFeatROI)) << std::endl;
            lvCout << "assoc_counts = " << lv::to_string(m_oData.m_oAssocCounts(oFeatROI+cv::Point2i((int)m_oData.m_nMaxDispOffset,0))) << std::endl;
            lvCout << "assoc_cost = " << lv::to_string(m_oData.m_oAssocCosts(oFeatROI)) << std::endl;
            lvCout << "unary = " << lv::to_string(m_oData.m_oUnaryCosts(oFeatROI)) << std::endl;
            lvCout << "pairw = " << lv::to_string(m_oData.m_oPairwCosts(oFeatROI)) << std::endl;*/

            const cv::Rect oLineROI(50,128,200,1);
            lvCout << "-----\n\n\n";
            lvCout << "input0 = " << lv::to_string(m_oData.m_aInputs[0](oLineROI)) << '\n';
            lvCout << "input1 = " << lv::to_string(m_oData.m_aInputs[2](oLineROI)) << '\n';
            lvCout << "disp = " << lv::to_string(m_oData.m_oStereoLabeling(oLineROI)) << '\n';
            lvCout << "assoc_counts = " << lv::to_string(m_oData.m_oAssocCounts(oLineROI+cv::Point2i((int)m_oData.m_nMaxDispOffset,0))) << '\n';
            lvCout << "assoc_cost = " << lv::to_string(m_oData.m_oAssocCosts(oLineROI)) << '\n';
            lvCout << "unary = " << lv::to_string(m_oData.m_oUnaryCosts(oLineROI)) << '\n';
            lvCout << "pairw = " << lv::to_string(m_oData.m_oPairwCosts(oLineROI)) << '\n';

            lvCout << "next label = " << (int)m_oData.getRealLabel(nAlphaLabel) << '\n';
            cv::Mat oCurrAssocCountsDisplay = FGStereoMatcher::getAssocCountsMapDisplay(m_oData);
            cv::resize(oCurrAssocCountsDisplay,oCurrAssocCountsDisplay,cv::Size(),4,4,cv::INTER_NEAREST);
            cv::imshow("assoc_counts",oCurrAssocCountsDisplay);
            cv::Mat oCurrLabelingDisplay = FGStereoMatcher::getStereoMapDisplay(m_oData);
            cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),4,4,cv::INTER_NEAREST);
            cv::imshow("disp",oCurrLabelingDisplay);
            cv::waitKey(0);
        }
        HigherOrderEnergy<ValueType,s_nMaxOrder> oHigherOrderEnergyReducer; // @@@@@@ check if HigherOrderEnergy::Clear() can be used instead of realloc
        oHigherOrderEnergyReducer.AddVars((int)nTotNodeCount);
        const auto lHigherOrderFactorAdder = [&](size_t nFactIdx, size_t nCurrFactOrder) {
            lvDbgAssert(oGM[nFactIdx].numberOfVariables()==nCurrFactOrder);
            const size_t nAssignCount = 1UL<<nCurrFactOrder;
            std::fill_n(aCliqueCoeffs.begin(),nAssignCount,(ValueType)0);
            for(size_t nAssignIdx=0; nAssignIdx<nAssignCount; ++nAssignIdx) {
                for(size_t nVarIdx=0; nVarIdx<nCurrFactOrder; ++nVarIdx)
                    aCliqueLabels[nVarIdx] = (nAssignIdx&(1<<nVarIdx))?nAlphaLabel:oLabeling((int)oGM[nFactIdx].variableIndex(nVarIdx));
                for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                    if(!(nAssignIdx&~nAssignSubsetIdx)) {
                        int nParityBit = 0;
                        for(size_t nVarIdx=0; nVarIdx<nCurrFactOrder; ++nVarIdx)
                            nParityBit ^= (((nAssignIdx^nAssignSubsetIdx)&(1<<nVarIdx))!=0);
                        const ValueType fCurrAssignEnergy = oGM[nFactIdx](aCliqueLabels.begin());
                        aCliqueCoeffs[nAssignSubsetIdx] += nParityBit?-fCurrAssignEnergy:fCurrAssignEnergy;
                    }
                }
            }
            for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                int nCurrTermDegree = 0;
                for(size_t nVarIdx=0; nVarIdx<nCurrFactOrder; ++nVarIdx)
                    if(nAssignSubsetIdx&(1<<nVarIdx))
                        aTermEnergyLUT[nCurrTermDegree++] = (typename HigherOrderEnergy<ValueType,s_nMaxOrder>::VarId)oGM[nFactIdx].variableIndex(nVarIdx);
                std::sort(aTermEnergyLUT.begin(),aTermEnergyLUT.begin()+nCurrTermDegree);
                oHigherOrderEnergyReducer.AddTerm(aCliqueCoeffs[nAssignSubsetIdx],nCurrTermDegree,aTermEnergyLUT.data());
            }
        };
        ValueType tMinEnergy = ValueType(0);
        for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
            const GraphModelData::NodeInfo& oNode = m_oData.m_vNodeInfos[nNodeIdx];
            // manually add 1st order factors while evaluating new assoc energy
            if(oNode.nStereoVisSimUnaryFactID!=SIZE_MAX) {
                const ValueType& tAssocCost = ((ValueType*)m_oData.m_oAssocCosts.data)[nNodeIdx];
                lvDbgAssert(&tAssocCost==&m_oData.m_oAssocCosts(oNode.nRowIdx,oNode.nColIdx));
                const ValueType& tVisSimCost = ((ValueType*)m_oData.m_oUnaryCosts.data)[nNodeIdx];
                lvDbgAssert(&tVisSimCost==&m_oData.m_oUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
                oHigherOrderEnergyReducer.AddUnaryTerm((int)nNodeIdx,tAssocCost+tVisSimCost);
            }
            // now add 2nd order & higher order factors via lambda
            if(oNode.anStereoSmoothPairwFactIDs[0]!=SIZE_MAX)
                lHigherOrderFactorAdder(oNode.anStereoSmoothPairwFactIDs[0],2);
            if(oNode.anStereoSmoothPairwFactIDs[1]!=SIZE_MAX)
                lHigherOrderFactorAdder(oNode.anStereoSmoothPairwFactIDs[1],2);
            lvIgnore(lHigherOrderFactorAdder);
        }
        oBinaryEnergyMinimizer.Reset();
        oHigherOrderEnergyReducer.ToQuadratic(oBinaryEnergyMinimizer); // @@@@@ internally might call addNodes/edges to QPBO object; optim tbd
        //@@@@@@ oBinaryEnergyMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11); @@@ can still add more if needed
        oBinaryEnergyMinimizer.Solve();
        oBinaryEnergyMinimizer.ComputeWeakPersistencies();
        size_t nChangedLabelings = 0;
        //cv::Mat_<uchar> SWAPS(m_oData.m_oGridSize(),0);
        for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
            const int nMoveLabel = oBinaryEnergyMinimizer.GetLabel((int)nNodeIdx);
            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
            if(nMoveLabel==1) { // node label changed to alpha
                const int nRowIdx = m_oData.m_vNodeInfos[nNodeIdx].nRowIdx;
                const int nColIdx = m_oData.m_vNodeInfos[nNodeIdx].nColIdx;
                //SWAPS(nRowIdx,nColIdx) = 255;
                const StereoLabelType nOldLabel = oLabeling(nRowIdx,nColIdx);
                if(nOldLabel<m_oData.m_nStereoDontCareLabelIdx)
                    m_oData.removeAssoc(nRowIdx,nColIdx,nOldLabel);
                oLabeling(nRowIdx,nColIdx) = nAlphaLabel;
                if(nAlphaLabel<m_oData.m_nStereoDontCareLabelIdx)
                    m_oData.addAssoc(nRowIdx,nColIdx,nAlphaLabel);
                ++nChangedLabelings;
            }
        }
        /*lvCout << "\n\n\n\n" << std::endl;
        lvCout << "swaps (tot=" << (size_t)cv::sum(SWAPS)[0]/255 << ") = " << lv::to_string(SWAPS,oDisplayOffset) << std::endl;
        lvCout << "swaps (tot=" << (size_t)cv::sum(SWAPS(oFeatROI))[0]/255 << ") = " << lv::to_string(SWAPS(oFeatROI)) << std::endl;
        lvCout << "\n\n\n\n" << std::endl;*/
        lvDbgAssert__(tLastEnergy>=value() && (tLastEnergy=value())>=ValueType(0),"not minimizing! curr=%f",(float)value());
        nConsecUnchangedLabels = (nChangedLabelings>0)?0:nConsecUnchangedLabels+1;
        // @@@@ order of future moves can be influenced by labels that cause the most changes? (but only late, to avoid bad local minima?)
        nAlphaLabel = m_oData.m_vStereoLabelOrdering[(++nOrderingIdx%=nLabels)];
        if(oVisitor(*this)!=opengm::visitors::VisitorReturnFlag::ContinueInf)
            break;
    }
    oVisitor.end(*this);
    lvLog_(1,"Inference completed in %f second(s).",oLocalTimer.tock());
    return opengm::InferenceTermination::NORMAL;
}

inline cv::Mat FGStereoMatcher::getStereoMapDisplay(const GraphModelData& oData) {
    lvAssert(!oData.m_oStereoLabeling.empty());
    lvDbgAssert(oData.m_oGridSize==oData.m_oStereoLabeling.size);
    const float fRescaleFact = float(UCHAR_MAX)/(oData.m_nMaxDispOffset-oData.m_nMinDispOffset+1);
    cv::Mat oOutput(oData.m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<oData.m_oStereoLabeling.rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<oData.m_oStereoLabeling.cols; ++nColIdx) {
            const OutputLabelType nRealLabel = oData.getRealLabel(oData.m_oStereoLabeling(nRowIdx,nColIdx));
            if(nRealLabel==s_nStereoDontCareLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,128,128);
            else if(nRealLabel==s_nStereoOccludedLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(128,0,128);
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b::all(uchar((nRealLabel-oData.m_nMinDispOffset)*fRescaleFact));
        }
    }
    return oOutput;
}

inline cv::Mat FGStereoMatcher::getAssocCountsMapDisplay(const GraphModelData& oData) {
    lvAssert(!oData.m_oAssocCounts.empty());
    lvDbgAssert(oData.m_oAssocCounts.rows==(int)oData.m_oGridSize[0]);
    lvDbgAssert(oData.m_oAssocCounts.cols==(int)(oData.m_oGridSize[1]+(oData.m_nMaxDispOffset-oData.m_nMinDispOffset)));
    double dMax;
    cv::minMaxIdx(oData.m_oAssocCounts,nullptr,&dMax);
    const float fRescaleFact = float(UCHAR_MAX)/(int(dMax)+1);
    cv::Mat oOutput(oData.m_oAssocCounts.size(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<(int)oData.m_oGridSize[0]; ++nRowIdx) {
        for(int nColIdx=-(int)oData.m_nMaxDispOffset; nColIdx<(int)(oData.m_oGridSize[1]-oData.m_nMinDispOffset); ++nColIdx) {
            const AssocCountType nCount = oData.getAssocCount(nRowIdx,nColIdx);
            if(nColIdx<0 || nColIdx>=(int)oData.m_oGridSize[1])
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx+(int)oData.m_nMaxDispOffset) = cv::Vec3b(0,0,uchar(nCount*fRescaleFact));
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx+(int)oData.m_nMaxDispOffset) = cv::Vec3b::all(uchar(nCount*fRescaleFact));
        }
    }
    return oOutput;
}