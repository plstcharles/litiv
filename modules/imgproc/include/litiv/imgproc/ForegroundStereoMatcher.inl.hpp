
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

#ifndef __LITIV_FGSTEREOM_HPP__
#error "Cannot include .inl.hpp headers directly!"
#endif //ndef(__LITIV_FGSTEREOM_HPP__)
#pragma once
#if (STEREOSEGMATCH_CONFIG_USE_DASCGF_FEATS+STEREOSEGMATCH_CONFIG_USE_DASCRF_FEATS+STEREOSEGMATCH_CONFIG_USE_LSS_FEATS/*+...*/)!=1
#error "Must specify only one feature type to use."
#endif //(features config ...)!=1

inline StereoSegmMatcher::StereoSegmMatcher(const cv::Size& oImageSize, size_t nMinDispOffset, size_t nMaxDispOffset, size_t nDispStep) :
        m_oImageSize(oImageSize) {
    static_assert(getInputStreamCount()==4 && getOutputStreamCount()==4 && getCameraCount()==2,"i/o stream must be two image-mask pairs");
    static_assert(getInputStreamCount()==InputPackSize && getOutputStreamCount()==OutputPackSize,"bad i/o internal enum mapping");
    lvAssert_(m_oImageSize.area()>1,"graph grid must be 2D and have at least two nodes");
    lvAssert_(nDispStep>0,"specified disparity offset step size must be strictly positive");
    if(nMaxDispOffset<nMinDispOffset)
        std::swap(nMaxDispOffset,nMinDispOffset);
    lvAssert_(nMaxDispOffset<size_t(s_nStereoOccludedLabel),"using reserved disparity integer label value");
    const size_t nMaxAllowedDispLabelCount = size_t(std::numeric_limits<InternalLabelType>::max()-2);
    const size_t nExpectedDispLabelCount = ((nMaxDispOffset-nMinDispOffset)/nDispStep)+1;
    lvAssert__(nMaxAllowedDispLabelCount>=nExpectedDispLabelCount,"internal stereo label type too small for given disparity range (max = %d)",(int)nMaxAllowedDispLabelCount);
    const std::vector<OutputLabelType> vStereoLabels = lv::make_range((OutputLabelType)nMinDispOffset,(OutputLabelType)nMaxDispOffset,(OutputLabelType)nDispStep);
    lvDbgAssert(nExpectedDispLabelCount==vStereoLabels.size());
    lvAssert_(vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
    const std::vector<OutputLabelType> vReservedStereoLabels = {s_nStereoDontCareLabel,s_nStereoOccludedLabel};
    m_pModelData = std::make_unique<GraphModelData>(m_oImageSize,lv::concat<OutputLabelType>(vStereoLabels,vReservedStereoLabels),nDispStep);
}

inline void StereoSegmMatcher::apply(const MatArrayIn& aInputs, MatArrayOut& aOutputs) {
    lvDbgAssert(m_pModelData);
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx) {
        lvAssert__(aInputs[nInputIdx].dims==2 && m_oImageSize==aInputs[nInputIdx].size(),"input in array at index=%d had the wrong size",(int)nInputIdx);
        lvAssert_((nInputIdx%InputPackOffset)==InputPackOffset_Img || aInputs[nInputIdx].type()==CV_8UC1,"unexpected input mask type");
        aInputs[nInputIdx].copyTo(m_pModelData->m_aInputs[nInputIdx]);
    }
    m_pModelData->updateModels(aInputs);
    m_pModelData->infer();
    m_pModelData->m_oStereoLabeling.copyTo(aOutputs[OutputPack_LeftDisp]);
    m_pModelData->m_oResegmLabeling.copyTo(aOutputs[OutputPack_LeftMask]);
    // @@@@@ fill missing outputs
    cv::Mat_<OutputLabelType>(m_oImageSize,OutputLabelType(0)).copyTo(aOutputs[OutputPack_RightDisp]);
    cv::Mat_<OutputLabelType>(m_oImageSize,OutputLabelType(0)).copyTo(aOutputs[OutputPack_RightMask]);
    for(size_t nOutputIdx=0; nOutputIdx<aOutputs.size(); ++nOutputIdx)
        aOutputs[nOutputIdx].copyTo(m_pModelData->m_aOutputs[nOutputIdx]); // @@@@ copy for temporal stuff later
}

inline void StereoSegmMatcher::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    m_pModelData->calcFeatures(aInputs,pFeatsPacket);
}

inline void StereoSegmMatcher::setNextFeatures(const cv::Mat& oPackedFeats) {
    m_pModelData->setNextFeatures(oPackedFeats);
}

inline std::string StereoSegmMatcher::getFeatureExtractorName() const {
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_FEATS
    return "sc-dasc-gf";
#elif STEREOSEGMATCH_CONFIG_USE_DASCRF_FEATS
    return "sc-dasc-rf";
#elif STEREOSEGMATCH_CONFIG_USE_LSS_FEATS
    return "sc-lss";
#endif //STEREOSEGMATCH_CONFIG_USE_..._FEATS
}

inline size_t StereoSegmMatcher::getMaxLabelCount() const {
    lvDbgAssert(m_pModelData);
    return m_pModelData->m_vStereoLabels.size();
}

inline const std::vector<StereoSegmMatcher::OutputLabelType>& StereoSegmMatcher::getLabels() const {
    lvDbgAssert(m_pModelData);
    return m_pModelData->m_vStereoLabels;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::GraphModelData::GraphModelData(const cv::Size& oImageSize, std::vector<OutputLabelType>&& vStereoLabels, size_t nStereoLabelStep) :
        m_nMaxStereoIterCount(STEREOSEGMATCH_DEFAULT_STEREO_ITER),
        m_nMaxResegmIterCount(STEREOSEGMATCH_DEFAULT_RESEGM_ITER),
        m_eStereoLabelInitType(LabelInit_LocalOptim),
        m_eStereoLabelOrderType(LabelOrder_Default),
        m_nStereoLabelOrderRandomSeed(size_t(0)),
        m_nStereoLabelingRandomSeed(size_t(0)),
        m_oGridSize(oImageSize),
        m_vStereoLabels(vStereoLabels),
        m_nDispOffsetStep(nStereoLabelStep),
        m_nMinDispOffset(size_t(m_vStereoLabels[0])),
        m_nMaxDispOffset(size_t(m_vStereoLabels.size()>3?m_vStereoLabels[m_vStereoLabels.size()-3]:m_vStereoLabels.back())),
        m_nStereoDontCareLabelIdx(InternalLabelType(m_vStereoLabels.size()-2)),
        m_nStereoOccludedLabelIdx(InternalLabelType(m_vStereoLabels.size()-1)),
        m_bUsePrecalcFeatsNext(false),m_bModelUpToDate(false) {
    lvAssert_(m_nMaxStereoIterCount>0 && m_nMaxResegmIterCount>0,"max iter counts must be positive");
    lvDbgAssert_(m_oGridSize.dims()==2 && m_oGridSize.total()>size_t(1),"graph grid must be 2D and have at least two nodes");
    lvAssert_(m_vStereoLabels.size()>3,"graph must have at least two possible output stereo labels, beyond reserved ones");
    lvAssert_(m_vStereoLabels.size()<=size_t(std::numeric_limits<InternalLabelType>::max()),"too many labels for internal type");
    lvDbgAssert(m_vStereoLabels[m_nStereoDontCareLabelIdx]==s_nStereoDontCareLabel && m_vStereoLabels[m_nStereoOccludedLabelIdx]==s_nStereoOccludedLabel);
    lvDbgAssert(std::min_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==m_vStereoLabels.begin() && m_vStereoLabels[0]>=OutputLabelType(0));
    lvDbgAssert(std::max_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==(m_vStereoLabels.begin()+m_vStereoLabels.size()-3));
    lvDbgAssert(std::equal(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2,lv::unique(m_vStereoLabels.begin(),m_vStereoLabels.end()).begin()+1));
    lvAssert_(m_nDispOffsetStep>0,"label step size must be positive");
    lvAssert_(m_oGridSize[1]>m_nMinDispOffset,"row length too small for smallest disp");
    lvAssert_(m_nMinDispOffset<m_nMaxDispOffset,"min/max disp offsets mismatch");
    lvDbgAssert_(std::numeric_limits<AssocCountType>::max()>m_oGridSize[1],"grid width is too large for association counter type");
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_FEATS
    m_pImgDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS,DASC_DEFAULT_GF_SUBSPL,DASC_DEFAULT_PREPROCESS);
#elif STEREOSEGMATCH_CONFIG_USE_DASCRF_FEATS
    m_pImgDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR,DASC_DEFAULT_RF_ITERS,DASC_DEFAULT_PREPROCESS);
#elif STEREOSEGMATCH_CONFIG_USE_LSS_FEATS
    m_pImgDescExtractor = std::make_unique<LSS>();
#endif //STEREOSEGMATCH_CONFIG_USE_..._FEATS
    const size_t nShapeContextInnerRadius=2, nShapeContextOuterRadius=STEREOSEGMATCH_DEFAULT_SHAPEDESC_RAD;
    m_pShpDescExtractor = std::make_unique<ShapeContext>(nShapeContextInnerRadius,nShapeContextOuterRadius,SHAPECONTEXT_DEFAULT_ANG_BINS,SHAPECONTEXT_DEFAULT_RAD_BINS);
    const cv::Size oMinWindowSize = m_pImgDescExtractor->windowSize();
    lvAssert__(oMinWindowSize.width<=oImageSize.width && oMinWindowSize.height<=oImageSize.height,"image is too small to compute descriptors with current pattern size -- need at least (%d,%d) and got (%d,%d)",oMinWindowSize.width,oMinWindowSize.height,oImageSize.width,oImageSize.height);
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
    lvDbgAssert(m_nGridBorderSize<m_oGridSize[0] && m_nGridBorderSize<m_oGridSize[1]);
    lvDbgAssert((size_t)std::max(m_pShpDescExtractor->borderSize(0),m_pShpDescExtractor->borderSize(1))<=m_nGridBorderSize);
    lvDbgAssert(m_aAssocCostAddLUT.size()==m_aAssocCostSumLUT.size() && m_aAssocCostRemLUT.size()==m_aAssocCostSumLUT.size());
    lvDbgAssert_(m_nMaxDispOffset<m_aAssocCostSumLUT.size(),"assoc cost lut size might not be large enough");
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT<m_aAssocCostSumLUT.size());
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(0)==0.0f);
    std::fill_n(m_aAssocCostAddLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    std::fill_n(m_aAssocCostRemLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    std::fill_n(m_aAssocCostSumLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    for(size_t n=STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT; n<m_aAssocCostAddLUT.size(); ++n) {
        m_aAssocCostAddLUT[n] = ValueType(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(n+1-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRemLUT[n] = -ValueType(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(n-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostSumLUT[n] = (n==size_t(0)?ValueType(0):(m_aAssocCostSumLUT[n-1]+m_aAssocCostAddLUT[n-1]));
    }
#if STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
    m_aLabelSimCostGradFactLUT.init(0,255,[](int nLocalGrad){
        return (float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE);
    });
#else //!STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
    m_aLabelSimCostGradFactLUT.init(0,255,[](int nLocalGrad){
        const float fGradPivotFact = 1.0f+(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/((nLocalGrad>=STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST)?(255-STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST):STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST));
        const float fGradScaleFact = STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE*fGradPivotFact*fGradPivotFact;
        lvDbgAssert(fGradScaleFact>=0.0f && fGradScaleFact<=4.0f*STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE);
        return fGradScaleFact;
    });
#endif //!STEREOSEGMATCH_LBLSIM_USE_EXP_GRADPIVOT
    lvDbgAssert(m_aLabelSimCostGradFactLUT.size()==size_t(256) && m_aLabelSimCostGradFactLUT.domain_offset_low()==0);
    lvDbgAssert(m_aLabelSimCostGradFactLUT.domain_index_step()==1.0 && m_aLabelSimCostGradFactLUT.domain_index_scale()==1.0);
    ////////////////////////////////////////////// put in new func, dupe for resegm model
    const size_t nStereoLabels = vStereoLabels.size();
    const size_t nRealStereoLabels = vStereoLabels.size()-2;
    const size_t nResegmLabels = size_t(2);
    const size_t nRows = m_oGridSize(0);
    const size_t nCols = m_oGridSize(1);
    const size_t nNodes = m_oGridSize.total();
    const size_t nStereoUnaryFuncDataSize = nNodes*nStereoLabels;
    const size_t nStereoPairwFuncDataSize = nNodes*2*(nStereoLabels*nStereoLabels);
    const size_t nStereoFuncDataSize = nStereoUnaryFuncDataSize+nStereoPairwFuncDataSize/*+...@@@@hoet*/;
    const size_t nResegmUnaryFuncDataSize = nNodes*nResegmLabels;
    const size_t nResegmPairwFuncDataSize = nNodes*2*(nResegmLabels*nResegmLabels);
    const size_t nResegmFuncDataSize = nResegmUnaryFuncDataSize+nResegmPairwFuncDataSize/*+...@@@@hoet*/;
    const size_t nModelSize = (nStereoFuncDataSize+nResegmFuncDataSize)*sizeof(ValueType)/*+...@@@@externals*/;
    lvLog_(1,"Expecting model size = %zu mb",nModelSize/1024/1024);
    lvAssert_(nModelSize<(CACHE_MAX_SIZE_GB*1024*1024*1024),"too many nodes/labels; model is unlikely to fit in memory");
    lvLog(1,"Constructing graphical models...");
    lv::StopWatch oLocalTimer;
    const size_t nStereoFactorsPerNode = 3; // @@@@
    const size_t nResegmFactorsPerNode = 3; // @@@@
    const size_t nStereoFunctions = nNodes*nStereoFactorsPerNode;
    const size_t nResegmFunctions = nNodes*nResegmFactorsPerNode;
    m_pStereoModel = std::make_unique<StereoModelType>(StereoSpaceType(nNodes,(InternalLabelType)vStereoLabels.size()),nStereoFactorsPerNode);
    m_pStereoModel->reserveFunctions<ExplicitFunction>(nStereoFunctions);
    m_pResegmModel = std::make_unique<ResegmModelType>(ResegmSpaceType(nNodes),nResegmFactorsPerNode);
    m_pResegmModel->reserveFunctions<ExplicitFunction>(nResegmFunctions);
    const std::array<int,2> anAssocCountsDims{int(m_oGridSize[0]/m_nDispOffsetStep),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep)};
    m_oAssocCounts.create(2,anAssocCountsDims.data());
    const std::array<int,3> anAssocMapDims{int(m_oGridSize[0]/m_nDispOffsetStep),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep),int(nRealStereoLabels*m_nDispOffsetStep)};
    m_oAssocMap.create(3,anAssocMapDims.data());
    m_oAssocCosts.create(m_oGridSize);
    m_oStereoUnaryCosts.create(m_oGridSize);
    m_oResegmUnaryCosts.create(m_oGridSize);
    m_oStereoPairwCosts.create(m_oGridSize); // for dbg only
    m_oResegmPairwCosts.create(m_oGridSize); // for dbg only
    m_vNodeInfos.resize(nNodes);
    for(int nRowIdx = 0; nRowIdx<(int)nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<(int)nCols; ++nColIdx) {
            const size_t nNodeIdx = nRowIdx*nCols+nColIdx;
            m_vNodeInfos[nNodeIdx].nRowIdx = nRowIdx;
            m_vNodeInfos[nNodeIdx].nColIdx = nColIdx;
            // the LUT members below will be properly initialized in the following sections
            m_vNodeInfos[nNodeIdx].nStereoUnaryFactID = SIZE_MAX;
            m_vNodeInfos[nNodeIdx].nResegmUnaryFactID = SIZE_MAX;
            m_vNodeInfos[nNodeIdx].pStereoUnaryFunc = nullptr;
            m_vNodeInfos[nNodeIdx].pResegmUnaryFunc = nullptr;
            m_vNodeInfos[nNodeIdx].anPairwNodeIdxs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nNodeIdx].anStereoPairwFactIDs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nNodeIdx].anResegmPairwFactIDs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nNodeIdx].apStereoPairwFuncs = std::array<StereoFunc*,2>{nullptr,nullptr};
            m_vNodeInfos[nNodeIdx].apResegmPairwFuncs = std::array<ResegmFunc*,2>{nullptr,nullptr};
        }
    }
    m_vStereoUnaryFuncs.reserve(nNodes);
    m_avStereoPairwFuncs[0].reserve(nNodes);
    m_avStereoPairwFuncs[1].reserve(nNodes);
    m_vResegmUnaryFuncs.reserve(nNodes);
    m_avResegmPairwFuncs[0].reserve(nNodes);
    m_avResegmPairwFuncs[1].reserve(nNodes);
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
    m_pStereoUnaryFuncsDataBase = nullptr;
    m_pStereoPairwFuncsDataBase = nullptr;
    m_pResegmUnaryFuncsDataBase = nullptr;
    m_pResegmPairwFuncsDataBase = nullptr;
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
    m_aStereoFuncsData = std::make_unique<ValueType[]>(nStereoFuncDataSize);
    m_pStereoUnaryFuncsDataBase = m_aStereoFuncsData.get();
    m_pStereoPairwFuncsDataBase = m_pStereoUnaryFuncsDataBase+nStereoUnaryFuncDataSize;
    m_aResegmFuncsData = std::make_unique<ValueType[]>(nResegmFuncDataSize);
    m_pResegmUnaryFuncsDataBase = m_aResegmFuncsData.get();
    m_pResegmPairwFuncsDataBase = m_pResegmUnaryFuncsDataBase+nResegmUnaryFuncDataSize;
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
    {
        lvLog(1,"\tadding unary factors to each grid node...");
        const std::array<size_t,1> aUnaryStereoFuncDims = {nStereoLabels};
        const std::array<size_t,1> aUnaryResegmFuncDims = {nResegmLabels};
        for(size_t nNodeIdx = 0; nNodeIdx<nNodes; ++nNodeIdx) {
            m_vStereoUnaryFuncs.push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
            m_vResegmUnaryFuncs.push_back(m_pResegmModel->addFunctionWithRefReturn(ExplicitFunction()));
            StereoFunc& oStereoFunc = m_vStereoUnaryFuncs.back();
            ResegmFunc& oResegmFunc = m_vResegmUnaryFuncs.back();
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
            oStereoFunc.second.resize(aUnaryStereoFuncDims.begin(),aUnaryStereoFuncDims.end());
            oResegmFunc.second.resize(aUnaryResegmFuncDims.begin(),aUnaryResegmFuncDims.end());
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
            oStereoFunc.second.assign(aUnaryStereoFuncDims.begin(),aUnaryStereoFuncDims.end(),m_pStereoUnaryFuncsDataBase+(nNodeIdx*nStereoLabels));
            oResegmFunc.second.assign(aUnaryResegmFuncDims.begin(),aUnaryResegmFuncDims.end(),m_pResegmUnaryFuncsDataBase+(nNodeIdx*nResegmLabels));
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
            lvDbgAssert(oStereoFunc.second.strides(0)==1 && oResegmFunc.second.strides(0)==1); // expect no padding
            const std::array<size_t,1> aNodeIndices = {nNodeIdx};
            m_vNodeInfos[nNodeIdx].nStereoUnaryFactID = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aNodeIndices.begin(),aNodeIndices.end());
            m_vNodeInfos[nNodeIdx].nResegmUnaryFactID = m_pResegmModel->addFactorNonFinalized(oResegmFunc.first,aNodeIndices.begin(),aNodeIndices.end());
            m_vNodeInfos[nNodeIdx].pStereoUnaryFunc = &oStereoFunc;
            m_vNodeInfos[nNodeIdx].pResegmUnaryFunc = &oResegmFunc;
        }
    }
    {
        lvLog(1,"\tadding pairwise factors to each grid node pair...");
        // note: current def w/ explicit stereo function will require too much memory if using >>50 disparity labels
        const std::array<size_t,2> aPairwiseStereoFuncDims = {nStereoLabels,nStereoLabels};
        const std::array<size_t,2> aPairwiseResegmFuncDims = {nResegmLabels,nResegmLabels};
        std::array<size_t,2> aNodeIndices;
        for(size_t nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(size_t nColIdx=0; nColIdx<nCols; ++nColIdx) {
                aNodeIndices[0] = nRowIdx*nCols+nColIdx;
                if(nRowIdx+1<nRows) { // vertical pair
                    aNodeIndices[1] = (nRowIdx+1)*nCols+nColIdx;
                    m_avStereoPairwFuncs[0].push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
                    m_avResegmPairwFuncs[0].push_back(m_pResegmModel->addFunctionWithRefReturn(ExplicitFunction()));
                    StereoFunc& oStereoFunc = m_avStereoPairwFuncs[0].back();
                    ResegmFunc& oResegmFunc = m_avResegmPairwFuncs[0].back();
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.resize(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end());
                    oResegmFunc.second.resize(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end());
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.assign(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end(),m_pStereoPairwFuncsDataBase+((aNodeIndices[0]*2)*nStereoLabels*nStereoLabels));
                    oResegmFunc.second.assign(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end(),m_pResegmPairwFuncsDataBase+((aNodeIndices[0]*2)*nResegmLabels*nResegmLabels));
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==nStereoLabels && oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==nResegmLabels); // expect last-idx-major
                    m_vNodeInfos[aNodeIndices[0]].anStereoPairwFactIDs[0] = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].anResegmPairwFactIDs[0] = m_pResegmModel->addFactorNonFinalized(oResegmFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].apStereoPairwFuncs[0] = &oStereoFunc;
                    m_vNodeInfos[aNodeIndices[0]].apResegmPairwFuncs[0] = &oResegmFunc;
                    m_vNodeInfos[aNodeIndices[0]].anPairwNodeIdxs[0] = aNodeIndices[1];
                }
                if(nColIdx+1<nCols) { // horizontal pair
                    aNodeIndices[1] = nRowIdx*nCols+nColIdx+1;
                    m_avStereoPairwFuncs[1].push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
                    m_avResegmPairwFuncs[1].push_back(m_pResegmModel->addFunctionWithRefReturn(ExplicitFunction()));
                    StereoFunc& oStereoFunc = m_avStereoPairwFuncs[1].back();
                    ResegmFunc& oResegmFunc = m_avResegmPairwFuncs[1].back();
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.resize(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end());
                    oResegmFunc.second.resize(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end());
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.assign(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end(),m_pStereoPairwFuncsDataBase+((aNodeIndices[0]*2+1)*nStereoLabels*nStereoLabels));
                    oResegmFunc.second.assign(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end(),m_pResegmPairwFuncsDataBase+((aNodeIndices[0]*2+1)*nResegmLabels*nResegmLabels));
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==nStereoLabels && oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==nResegmLabels); // expect last-idx-major
                    m_vNodeInfos[aNodeIndices[0]].anStereoPairwFactIDs[1] = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].anResegmPairwFactIDs[1] = m_pResegmModel->addFactorNonFinalized(oResegmFunc.first,aNodeIndices.begin(),aNodeIndices.end());
                    m_vNodeInfos[aNodeIndices[0]].apStereoPairwFuncs[1] = &oStereoFunc;
                    m_vNodeInfos[aNodeIndices[0]].apResegmPairwFuncs[1] = &oResegmFunc;
                    m_vNodeInfos[aNodeIndices[0]].anPairwNodeIdxs[1] = aNodeIndices[1];
                }
            }
        }
    }/*{
     // add 3rd order function and factors to the model (test)
        const std::array<InternalLabelType,3> aHOEFuncDims = {nLabels,nLabels,nLabels};
        ExplicitFunction vHOEFunc(aHOEFuncDims.begin(),aHOEFuncDims.end(),0.5f);
        FunctionID nFID = m_pGM->addFunction(vHOEFunc);
        for(size_t nLabelIdx1=0; nLabelIdx1<nRealStereoLabels; ++nLabelIdx1) {
            for(size_t nLabelIdx2 = 0; nLabelIdx2<nRealStereoLabels; ++nLabelIdx2) {
                for(size_t nLabelIdx3 = 0; nLabelIdx3<nRealStereoLabels; ++nLabelIdx3) {
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
    m_pResegmModel->finalize();
    lvLog_(1,"Graph models constructed in %f second(s).",oLocalTimer.tock());
    lvCout << "Stereo:\n";
    lv::gm::printModelInfo(*m_pStereoModel);
    lvCout << "Resegm:\n";
    lv::gm::printModelInfo(*m_pResegmModel);
    m_pStereoInf = std::make_unique<StereoGraphInference>(*this);
    m_pResegmInf = std::make_unique<ResegmGraphInference>(*this);
}

inline void StereoSegmMatcher::GraphModelData::resetLabelings(const MatArrayIn& aInputs) {
    const StereoModelType& oStereoGM = *m_pStereoModel;
    const ResegmModelType& oResegmGM = *m_pResegmModel;
    const size_t nStereoLabels = m_vStereoLabels.size();
    const size_t nRealStereoLabels = m_vStereoLabels.size()-2;
    const size_t nNodes = m_oGridSize.total();
    lvDbgAssert(nStereoLabels>3 && nNodes==m_vNodeInfos.size());
    lvIgnore(oStereoGM);
    lvIgnore(oResegmGM);
    if(m_eStereoLabelInitType==LabelInit_Default) {
        m_oStereoLabeling.create(m_oGridSize);
        std::fill(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),InternalLabelType(0));
    }
    else if(m_eStereoLabelInitType==LabelInit_Random) {
        m_oStereoLabeling.create(m_oGridSize);
        std::mt19937 oGen(m_nStereoLabelingRandomSeed);
        std::generate(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),[&](){return InternalLabelType(oGen()%nRealStereoLabels);});
    }
    else if(m_eStereoLabelInitType==LabelInit_LocalOptim) {
        m_oStereoLabeling.create(m_oGridSize);
        std::fill(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),InternalLabelType(0));
        for(size_t nNodeIdx=0; nNodeIdx<nNodes; ++nNodeIdx) {
            const NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
            if(oNode.nStereoUnaryFactID!=SIZE_MAX) {
                lvDbgAssert(oNode.nStereoUnaryFactID<oStereoGM.numberOfFactors());
                lvDbgAssert(oStereoGM.numberOfLabels(oNode.nStereoUnaryFactID)==nStereoLabels);
                lvDbgAssert(StereoFuncID(oStereoGM[oNode.nStereoUnaryFactID].functionIndex(),oStereoGM[oNode.nStereoUnaryFactID].functionType())==oNode.pStereoUnaryFunc->first);
                InternalLabelType nEvalLabel = 0; // == value already assigned via std::fill
                ValueType fOptimalEnergy = oNode.pStereoUnaryFunc->second(&nEvalLabel);
                for(nEvalLabel = 1; nEvalLabel<nStereoLabels; ++nEvalLabel) {
                    const ValueType fCurrEnergy = oNode.pStereoUnaryFunc->second(&nEvalLabel);
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
    aInputs[InputPack_LeftMask].copyTo(m_oResegmLabeling); // @@@@ update for right image inference via param?
    m_oAssocCounts = (AssocCountType)0;
    m_oAssocMap = (AssocIdxType)-1;
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<(int)m_oGridSize[1]; ++nColIdx) {
            const InternalLabelType nLabel = m_oStereoLabeling(nRowIdx,nColIdx);
            if(nLabel<m_nStereoDontCareLabelIdx) // both special labels avoided here
                addAssoc(nRowIdx,nColIdx,nLabel);
        }
    }
    if(m_eStereoLabelOrderType==LabelOrder_Default) {
        m_vStereoLabelOrdering.resize(nStereoLabels);
        std::iota(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),0);
    }
    else if(m_eStereoLabelOrderType==LabelOrder_Random) {
        m_vStereoLabelOrdering.resize(nStereoLabels);
        std::iota(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),0);
        std::mt19937 oGen(m_nStereoLabelOrderRandomSeed);
        std::shuffle(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end(),oGen);
    }
    else /*if(m_eStereoLabelOrderType==LabelOrder_Explicit)*/ {
        lvAssert_(m_vStereoLabelOrdering.size()==nStereoLabels,"label order array did not contain all labels");
        lvAssert_(lv::unique(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end())==lv::make_range(InternalLabelType(0),InternalLabelType(nStereoLabels-1)),"label order array did not contain all labels");
    }
}

inline void StereoSegmMatcher::GraphModelData::updateModels(const MatArrayIn& aInputs) {
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx)
        lvDbgAssert__(aInputs[nInputIdx].dims==2 && m_oGridSize==aInputs[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
    lvDbgAssert_(aInputs[InputPack_LeftMask].type()==CV_8UC1,"unexpected left input mask type");
    lvDbgAssert_(aInputs[InputPack_RightMask].type()==CV_8UC1,"unexpected right input mask type");
    cv::imshow("left input (a)",aInputs[InputPack_LeftImg]);//@@@@
    cv::imshow("right input (b)",aInputs[InputPack_RightImg]);//@@@@
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    const size_t nStereoLabels = m_vStereoLabels.size();
    const size_t nRealStereoLabels = nStereoLabels-2;
    const size_t nResegmLabels = size_t(2);
    lvIgnore(nResegmLabels); // @@@@
    const int nMinDescColIdx = (int)m_nGridBorderSize;
    const int nMinDescRowIdx = (int)m_nGridBorderSize;
    const int nMaxDescColIdx = nCols-(int)m_nGridBorderSize-1;
    const int nMaxDescRowIdx = nRows-(int)m_nGridBorderSize-1;
    const auto lHasValidDesc = [&nMinDescRowIdx,&nMaxDescRowIdx,&nMinDescColIdx,&nMaxDescColIdx](int nRowIdx, int nColIdx) {
        return nRowIdx>=nMinDescRowIdx && nRowIdx<=nMaxDescRowIdx && nColIdx>=nMinDescColIdx && nColIdx<=nMaxDescColIdx;
    };
    lvAssert_(!m_bUsePrecalcFeatsNext || m_vNextFeats.size()==FeatPackSize,"unexpected precalculated features vec size");
    if(!m_bUsePrecalcFeatsNext)
        calcFeatures(aInputs);
    else
        m_bUsePrecalcFeatsNext = false;
    const cv::Mat_<float> oImgAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    const cv::Mat_<float> oShpAffinityMap = m_vNextFeats[FeatPack_ShpAffinity];
    lvDbgAssert(oImgAffinity.dims==3 && oImgAffinity.size[0]==nRows && oImgAffinity.size[1]==nCols && oImgAffinity.size[2]==(int)nRealStereoLabels);
    lvDbgAssert(oShpAffinityMap.dims==3 && oShpAffinityMap.size[0]==nRows && oShpAffinityMap.size[1]==nCols && oShpAffinityMap.size[2]==(int)nRealStereoLabels);
    const std::array<cv::Mat_<float>,2> aFGDistMaps = {m_vNextFeats[FeatPack_LeftFGDist],m_vNextFeats[FeatPack_RightFGDist]};
    const std::array<cv::Mat_<float>,2> aBGDistMaps = {m_vNextFeats[FeatPack_LeftBGDist],m_vNextFeats[FeatPack_RightBGDist]};
    lvDbgAssert(lv::MatInfo(aFGDistMaps[0])==lv::MatInfo(aFGDistMaps[1]) && m_oGridSize==aFGDistMaps[0].size);
    lvDbgAssert(lv::MatInfo(aBGDistMaps[0])==lv::MatInfo(aBGDistMaps[1]) && m_oGridSize==aBGDistMaps[0].size);
    const std::array<cv::Mat_<float>,2> aFGSimMaps = {m_vNextFeats[FeatPack_LeftFGSim],m_vNextFeats[FeatPack_RightFGSim]};
    const std::array<cv::Mat_<float>,2> aBGSimMaps = {m_vNextFeats[FeatPack_LeftBGSim],m_vNextFeats[FeatPack_RightBGSim]};
    lvDbgAssert(lv::MatInfo(aFGSimMaps[0])==lv::MatInfo(aFGSimMaps[1]) && m_oGridSize==aFGSimMaps[0].size);
    lvDbgAssert(lv::MatInfo(aBGSimMaps[0])==lv::MatInfo(aBGSimMaps[1]) && m_oGridSize==aBGSimMaps[0].size);
    const std::array<cv::Mat_<uchar>,2> aGradYMaps = {m_vNextFeats[FeatPack_LeftGradY],m_vNextFeats[FeatPack_RightGradY]};
    const std::array<cv::Mat_<uchar>,2> aGradXMaps = {m_vNextFeats[FeatPack_LeftGradX],m_vNextFeats[FeatPack_RightGradX]};
    const std::array<cv::Mat_<uchar>,2> aGradMagMaps = {m_vNextFeats[FeatPack_LeftGradMag],m_vNextFeats[FeatPack_RightGradMag]};
    lvDbgAssert(lv::MatInfo(aGradYMaps[0])==lv::MatInfo(aGradYMaps[1]) && m_oGridSize==aGradYMaps[0].size);
    lvDbgAssert(lv::MatInfo(aGradXMaps[0])==lv::MatInfo(aGradXMaps[1]) && m_oGridSize==aGradXMaps[0].size);
    lvDbgAssert(lv::MatInfo(aGradMagMaps[0])==lv::MatInfo(aGradMagMaps[1]) && m_oGridSize==aGradMagMaps[0].size);
    lvLog(1,"Updating graph models energy terms based on input data...");
    lv::StopWatch oLocalTimer;
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
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

            // update unary terms for each grid node
            lvDbgAssert(oNode.pStereoUnaryFunc && oNode.nStereoUnaryFactID!=SIZE_MAX); // @@@@ will no longer be the case w/ roi check
            lvDbgAssert((&m_pStereoModel->getFunction<ExplicitFunction>(oNode.pStereoUnaryFunc->first))==(&oNode.pStereoUnaryFunc->second));
            ExplicitFunction& vUnaryStereoFunc = oNode.pStereoUnaryFunc->second;
            lvDbgAssert(vUnaryStereoFunc.dimension()==1 && vUnaryStereoFunc.size()==nStereoLabels);
            lvDbgAssert(m_pImgDescExtractor->defaultNorm()==cv::NORM_L2);
            const float* pImgAffinityPtr = oImgAffinity.ptr<float>(nRowIdx,nColIdx);
            const float* pShpAffinityPtr = oShpAffinityMap.ptr<float>(nRowIdx,nColIdx);
            for(InternalLabelType nLabelIdx=0; nLabelIdx<nRealStereoLabels; ++nLabelIdx) {
                const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                const int nOffsetColIdx = nColIdx-nRealLabel;
                //const bool bValidDescOffsetNode = lHasValidDesc(nRowIdx,nOffsetColIdx); // @@@ update to use roi?
                vUnaryStereoFunc(nLabelIdx) = ValueType(0);
                const float& fImgAffinity = pImgAffinityPtr[nLabelIdx];
                const float& fShpAffinity = pShpAffinityPtr[nLabelIdx];
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    if(!(fImgAffinity>=0.0f && fShpAffinity>=0.0f))
                        lvPrint(fImgAffinity) << "nRowIdx="<<nRowIdx << ", nColIdx=" << nColIdx <<", nLabelIdx="<<(int)nLabelIdx<<"\n";
                    lvAssert(fImgAffinity>=0.0f && fShpAffinity>=0.0f);
                    vUnaryStereoFunc(nLabelIdx) += ValueType(fImgAffinity*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE);
                    vUnaryStereoFunc(nLabelIdx) += ValueType(fShpAffinity*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE);
                }
                else {
                    lvAssert(fImgAffinity<1.0f && fShpAffinity<0.0f);
                    vUnaryStereoFunc(nLabelIdx) += STEREOSEGMATCH_UNARY_COST_OOB_CST;
                }
                vUnaryStereoFunc(nLabelIdx) = std::min(STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST,vUnaryStereoFunc(nLabelIdx));
            }
            // @@@@ add discriminativeness factor --- vect-append all sim vals, sort, setup discrim costs
            vUnaryStereoFunc(m_nStereoDontCareLabelIdx) = ValueType(100000); // @@@@ check roi, if dc set to 0, otherwise set to inf
            vUnaryStereoFunc(m_nStereoOccludedLabelIdx) = ValueType(100000);//STEREOSEGMATCH_IMGSIM_COST_OCCLUDED_CST;

            // update stereo smoothness pairwise terms for each grid node
            for(size_t nOrientIdx=0; nOrientIdx<oNode.anStereoPairwFactIDs.size(); ++nOrientIdx) {
                if(oNode.anStereoPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                    lvDbgAssert(oNode.apStereoPairwFuncs[nOrientIdx]);
                    lvDbgAssert((&m_pStereoModel->getFunction<ExplicitFunction>(oNode.apStereoPairwFuncs[nOrientIdx]->first))==(&oNode.apStereoPairwFuncs[nOrientIdx]->second));
                    ExplicitFunction& vPairwiseStereoFunc = oNode.apStereoPairwFuncs[nOrientIdx]->second;
                    lvDbgAssert(vPairwiseStereoFunc.dimension()==2 && vPairwiseStereoFunc.size()==nStereoLabels*nStereoLabels);
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<nRealStereoLabels; ++nLabelIdx1) {
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<nRealStereoLabels; ++nLabelIdx2) {
                            const OutputLabelType nRealLabel1 = getRealLabel(nLabelIdx1);
                            const OutputLabelType nRealLabel2 = getRealLabel(nLabelIdx2);
                            const int nRealLabelDiff = std::min(std::abs((int)nRealLabel1-(int)nRealLabel2),STEREOSEGMATCH_LBLSIM_COST_MAXDIFF_CST);
                            const int nLocalGrad = (int)((nOrientIdx==0)?aGradYMaps[0]:(nOrientIdx==1)?aGradXMaps[0]:aGradMagMaps[0])(nRowIdx,nColIdx);
                            const float fGradScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                            lvDbgAssert(fGradScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                            vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) = ValueType((nRealLabelDiff*nRealLabelDiff)*fGradScaleFact);
                            const bool bValidDescOffsetNode1 = lHasValidDesc(nRowIdx,nColIdx-(int)nRealLabel1);
                            const bool bValidDescOffsetNode2 = lHasValidDesc(nRowIdx,nColIdx-(int)nRealLabel2);
                            if(!bValidDescOffsetNode1 || !bValidDescOffsetNode2)
                                vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) *= 2; // incr smoothness cost for node pairs in border regions to outweight bad unaries
                        }
                    }
                    for(size_t nLabelIdx=0; nLabelIdx<nRealStereoLabels; ++nLabelIdx) {
                        // @@@ change later for img-data-dependent or roi-dependent energies?
                        // @@@@ outside-roi-dc to inside-roi-something is ok (0 cost)
                        vPairwiseStereoFunc(m_nStereoDontCareLabelIdx,nLabelIdx) = ValueType(100000);
                        vPairwiseStereoFunc(m_nStereoOccludedLabelIdx,nLabelIdx) = ValueType(100000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                        vPairwiseStereoFunc(nLabelIdx,m_nStereoDontCareLabelIdx) = ValueType(100000);
                        vPairwiseStereoFunc(nLabelIdx,m_nStereoOccludedLabelIdx) = ValueType(100000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                    }
                    vPairwiseStereoFunc(m_nStereoDontCareLabelIdx,m_nStereoDontCareLabelIdx) = ValueType(0);
                    vPairwiseStereoFunc(m_nStereoOccludedLabelIdx,m_nStereoOccludedLabelIdx) = ValueType(0);
                }
            }
            const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0 && oLocalTimer.elapsed()>2.0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
                bProgressDisplayed = true;
            }
        }
    }
    if(bProgressDisplayed)
        lv::cleanConsoleRow();
    lvLog_(1,"Graph energy terms update completed in %f second(s).",oLocalTimer.tock());
    resetLabelings(aInputs);
    m_bModelUpToDate = true;
}

inline void StereoSegmMatcher::GraphModelData::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"lots of hardcoded indices below");
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputImg = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        lvAssert__(oInputImg.dims==2 && m_oGridSize==oInputImg.size(),"input image in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3,"unexpected input image type");
        const cv::Mat& oInputMask = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask];
        lvAssert__(oInputMask.dims==2 && m_oGridSize==oInputMask.size(),"input mask in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(oInputMask.type()==CV_8UC1,"unexpected input mask type");
    }
    m_vNextFeats.resize(FeatPackSize);
    calcImageFeatures(std::array<cv::Mat,InputPackSize/InputPackOffset>{aInputs[InputPack_LeftImg],aInputs[InputPack_RightImg]});
    calcShapeFeatures(std::array<cv::Mat,InputPackSize/InputPackOffset>{aInputs[InputPack_LeftMask],aInputs[InputPack_RightMask]});
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


inline void StereoSegmMatcher::GraphModelData::calcImageFeatures(const std::array<cv::Mat,InputPackSize/InputPackOffset>& aInputImages) {
    static_assert(getCameraCount()==size_t(2),"bad input image array size");
    lvLog(1,"Calculating image features maps...");
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputImg = aInputImages[nCamIdx];
        lvDbgAssert(m_pImgDescExtractor);
        cv::Mat& oImgDescs = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_ImgDescs];
        lvLog_(2,"\tcam[%d] image descriptors...",(int)nCamIdx);
        m_pImgDescExtractor->compute2(oInputImg,oImgDescs);
        lvLog_(2,"\tcam[%d] image gradient magnitudes...",(int)nCamIdx);
        cv::Mat oBlurredInput;
        cv::GaussianBlur(oInputImg,oBlurredInput,cv::Size(3,3),0);
        cv::Mat oBlurredGrayInput;
        if(oBlurredInput.channels()==3)
            cv::cvtColor(oBlurredInput,oBlurredGrayInput,cv::COLOR_BGR2GRAY);
        else
            oBlurredGrayInput = oBlurredInput;
        cv::Mat oGradInput_X,oGradInput_Y;
        cv::Sobel(oBlurredGrayInput,oGradInput_Y,CV_16S,0,1,STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::Sobel(oBlurredGrayInput,oGradInput_X,CV_16S,1,0,STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::Mat& oGradY = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradY];
        cv::convertScaleAbs(oGradInput_Y,oGradY);
        cv::Mat& oGradX = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradX];
        cv::convertScaleAbs(oGradInput_X,oGradX);
        cv::Mat& oGradMag = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradMag];
        cv::addWeighted(oGradY,0.5,oGradX,0.5,0,oGradMag);
    }
    lvLog_(1,"Image features maps computed in %f second(s).",oLocalTimer.tock());
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    const size_t nStereoLabels = m_vStereoLabels.size();
    const size_t nRealStereoLabels = nStereoLabels-2;
    const int nMinDescColIdx = (int)m_nGridBorderSize;
    const int nMinDescRowIdx = (int)m_nGridBorderSize;
    const int nMaxDescColIdx = nCols-(int)m_nGridBorderSize-1;
    const int nMaxDescRowIdx = nRows-(int)m_nGridBorderSize-1;
    const auto lHasValidDesc = [&nMinDescRowIdx,&nMaxDescRowIdx,&nMinDescColIdx,&nMaxDescColIdx](int nRowIdx, int nColIdx) {
        return nRowIdx>=nMinDescRowIdx && nRowIdx<=nMaxDescRowIdx && nColIdx>=nMinDescColIdx && nColIdx<=nMaxDescColIdx;
    };
    const std::array<cv::Mat_<float>,2> aImgDescs = {m_vNextFeats[FeatPack_LeftImgDescs],m_vNextFeats[FeatPack_RightImgDescs]};
    lvDbgAssert(lv::MatInfo(aImgDescs[0])==lv::MatInfo(aImgDescs[1]) && aImgDescs[0].dims==3);
    lvDbgAssert(aImgDescs[0].size[0]==nRows && aImgDescs[0].size[1]==nCols);
    lvLog(1,"Calculating image affinity maps...");
    const std::array<int,3> anDistMapDims = {nRows,nCols,(int)nRealStereoLabels};
    cv::Mat& oImgAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    oImgAffinity.create(3,anDistMapDims.data(),CV_32FC1);
    /*std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;*/
#if USING_OPENMP
#pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float* pImgAffinityPtr = oImgAffinity.ptr<float>(nRowIdx,nColIdx);
            const bool bValidDescNode = lHasValidDesc(nRowIdx,nColIdx);
            for(InternalLabelType nLabelIdx=0; nLabelIdx<nRealStereoLabels; ++nLabelIdx) {
                const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                const int nOffsetColIdx = nColIdx-nRealLabel;
                const bool bValidDescOffsetNode = lHasValidDesc(nRowIdx,nOffsetColIdx); // @@@ update to use roi?
                if(bValidDescNode && bValidDescOffsetNode) {
                    // test w/ root-sift transform here? @@@@@
                    pImgAffinityPtr[nLabelIdx] = (float)m_pImgDescExtractor->calcDistance(aImgDescs[0].ptr<float>(nRowIdx,nColIdx),aImgDescs[1].ptr<float>(nRowIdx,nOffsetColIdx));
                }
                else if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    lvDbgAssert(aInputImages[0].type()==CV_8UC1 && aInputImages[1].type()==CV_8UC1);
                    const int nCurrImgDiff = (int)aInputImages[0].at<uchar>(nRowIdx,nColIdx)-(int)aInputImages[1].at<uchar>(nRowIdx,nOffsetColIdx);
                    pImgAffinityPtr[nLabelIdx] = (std::abs(nCurrImgDiff)*STEREOSEGMATCH_IMGSIM_COST_RAW_SCALE);
                }
                else
                    pImgAffinityPtr[nLabelIdx] = -1.0f; // OOB
            }
            /*const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0 && oLocalTimer.elapsed()>2.0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
                bProgressDisplayed = true;
            }*/
        }
    }
    /*if(bProgressDisplayed)
        lv::cleanConsoleRow();*/
    lvLog_(1,"Image affinity maps computed in %f second(s).",oLocalTimer.tock());
    /*for(InternalLabelType nLabelIdx=0; nLabelIdx<nRealStereoLabels; ++nLabelIdx) {
        cv::Mat_<float> tmp = lv::squeeze(lv::getSubMat(oImgAffinity,2,(int)nLabelIdx));
        //cv::Mat_<float> tmpdescs = lv::squeeze(lv::getSubMat(oImgDescMap,2,(int)i));
        cv::Rect roi(0,128,256,1);
        cv::imshow("tmp",tmp);
        lvPrint(tmp(roi));
        //lvPrint(tmpdescs(roi));
        cv::Mat_<float> exp;
        cv::exp(-tmp,exp);
        cv::imshow("exp",exp);
        lvPrint(exp(roi));
        cv::waitKey(0);
    }*/
}

inline void StereoSegmMatcher::GraphModelData::calcShapeFeatures(const std::array<cv::Mat,InputPackSize/InputPackOffset>& aInputMasks) {
    static_assert(getCameraCount()==size_t(2),"bad input mask array size");
    lvLog(1,"Calculating shape features maps...");
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {

        // @@@@@ use swap mask to prevent recalc of identical descriptors? (swap + radius based on desc)

        const cv::Mat& oInputMask = aInputMasks[nCamIdx];
        lvDbgAssert(m_pShpDescExtractor);
        cv::Mat& oShpDescs = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_ShpDescs];
        lvLog_(2,"\tcam[%d] shape descriptors...",(int)nCamIdx);
        m_pShpDescExtractor->compute2(oInputMask,oShpDescs);
        lvLog_(2,"\tcam[%d] shape distance fields...",(int)nCamIdx);
        cv::Mat& oFGDist = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist];
        cv::distanceTransform(oInputMask==0,oFGDist,cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
        cv::Mat& oFGSim = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_FGSim];
        cv::exp(STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE*oFGDist,oFGSim);
        //lvPrint(cv::Mat_<float>(oFGSim(cv::Rect(0,128,256,1))));
        cv::divide(1.0,oFGSim+STEREOSEGMATCH_DEFAULT_DISTTRANSF_OFFSET,oFGDist);
        //lvPrint(cv::Mat_<float>(oFGDist(cv::Rect(0,128,256,1))));
        cv::Mat& oBGDist = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist];
        cv::distanceTransform(oInputMask>0,oBGDist,cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
        cv::Mat& oBGSim = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_BGSim];
        cv::exp(STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE*oBGDist,oBGSim);
        //lvPrint(cv::Mat_<float>(oBGSim(cv::Rect(0,128,256,1))));
        cv::divide(1.0,oBGSim+STEREOSEGMATCH_DEFAULT_DISTTRANSF_OFFSET,oBGDist);
        //lvPrint(cv::Mat_<float>(oBGDist(cv::Rect(0,128,256,1))));
    }
    lvLog_(1,"Shape features maps computed in %f second(s).",oLocalTimer.tock());
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    const size_t nStereoLabels = m_vStereoLabels.size();
    const size_t nRealStereoLabels = nStereoLabels-2;
    const std::array<cv::Mat_<float>,2> aShpDescs = {m_vNextFeats[FeatPack_LeftShpDescs],m_vNextFeats[FeatPack_RightShpDescs]};
    lvDbgAssert(lv::MatInfo(aShpDescs[0])==lv::MatInfo(aShpDescs[1]) && aShpDescs[0].dims==3);
    lvDbgAssert(aShpDescs[0].size[0]==nRows && aShpDescs[0].size[1]==nCols);
    lvLog(1,"Calculating shape affinity maps...");
    const std::array<int,3> anDistMapDims = {nRows,nCols,(int)nRealStereoLabels};
    cv::Mat& oShpAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    oShpAffinity.create(3,anDistMapDims.data(),CV_32FC1);
    /*std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;*/
#if USING_OPENMP
#pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {

            // @@@@@ use swap mask to prevent recalc of identical descriptors? (swap + radius based on desc)

            float* pShpAffinityPtr = oShpAffinity.ptr<float>(nRowIdx,nColIdx);
            for(InternalLabelType nLabelIdx=0; nLabelIdx<nRealStereoLabels; ++nLabelIdx) {
                const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                const int nOffsetColIdx = nColIdx-nRealLabel;
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    // test w/ root-sift transform here? @@@@@
#if STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_SIM
                    // @@@@ add scale factor?
                    pShpAffinityPtr[nLabelIdx] = (float)m_pShpDescExtractor->calcDistance(aShpDescs[0].ptr<float>(nRowIdx,nColIdx),aShpDescs[1].ptr<float>(nRowIdx,nOffsetColIdx));
#else //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_SIM
                    const cv::Mat_<float> oDesc1(1,aShpDescs[0].size[2],const_cast<float*>(aShpDescs[0].ptr<float>(nRowIdx,nColIdx)));
                    const cv::Mat_<float> oDesc2(1,aShpDescs[0].size[2],const_cast<float*>(aShpDescs[1].ptr<float>(nRowIdx,nOffsetColIdx)));
                    pShpAffinityPtr[nLabelIdx] = (float)cv::norm(oDesc1,oDesc2,cv::NORM_L2);
#endif //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_SIM
                }
                else
                    pShpAffinityPtr[nLabelIdx] = -1.0f; // OOB
            }
            /*const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0 && oLocalTimer.elapsed()>2.0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
                bProgressDisplayed = true;
            }*/
        }
    }
    /*if(bProgressDisplayed)
        lv::cleanConsoleRow();*/
    lvLog_(1,"Affinity maps computed in %f second(s).",oLocalTimer.tock());
    /*for(InternalLabelType nLabelIdx=0; nLabelIdx<nRealStereoLabels; ++nLabelIdx) {
        cv::Mat_<float> tmp = lv::squeeze(lv::getSubMat(oShpAffinity,2,(int)nLabelIdx));
        //cv::Mat_<float> tmpdescs = lv::squeeze(lv::getSubMat(oShpDescMap,2,(int)i));
        cv::Rect roi(0,128,256,1);
        cv::imshow("tmp",tmp);
        lvPrint(tmp(roi));
        //lvPrint(tmpdescs(roi));
        cv::Mat_<float> exp;
        cv::exp(-tmp,exp);
        cv::imshow("exp",exp);
        lvPrint(exp(roi));
        cv::waitKey(0);
    }*/
}

inline void StereoSegmMatcher::GraphModelData::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvAssert_(!oPackedFeats.empty(),"features packet must be non-empty");
    lvDbgExceptionWatch;
    if(m_vExpectedFeatPackInfo.empty()) {
        m_vExpectedFeatPackInfo.resize(FeatPackSize);
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_ImgDescs] = m_pImgDescExtractor->getOutputInfo(lv::MatInfo(m_oGridSize,CV_8UC3));
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_ShpDescs] = m_pShpDescExtractor->getOutputInfo(lv::MatInfo(m_oGridSize,CV_8UC1));
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_FGSim] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_BGSim] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradY] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradX] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradMag] = lv::MatInfo(m_oGridSize,CV_8UC1);
        }
        const int nRows = (int)m_oGridSize(0);
        const int nCols = (int)m_oGridSize(1);
        const int nRealStereoLabels = (int)m_vStereoLabels.size()-2;
        lvDbgAssert(nRealStereoLabels>1);
        m_vExpectedFeatPackInfo[FeatPack_ImgAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,nRealStereoLabels},CV_32FC1);
        m_vExpectedFeatPackInfo[FeatPack_ShpAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,nRealStereoLabels},CV_32FC1);
    }
    m_oNextPackedFeats = oPackedFeats; // get pointer to data instead of copy; provider should not overwrite until next 'apply' call!
    m_vNextFeats = lv::unpackData(m_oNextPackedFeats,m_vExpectedFeatPackInfo);
    m_bUsePrecalcFeatsNext = true;
    m_bModelUpToDate = false;
}

inline StereoSegmMatcher::OutputLabelType StereoSegmMatcher::GraphModelData::getRealLabel(InternalLabelType nLabel) const {
    lvDbgAssert(nLabel<m_vStereoLabels.size());
    return m_vStereoLabels[nLabel];
}

inline StereoSegmMatcher::InternalLabelType StereoSegmMatcher::GraphModelData::getInternalLabel(OutputLabelType nRealLabel) const {
    lvDbgAssert(nRealLabel==s_nStereoOccludedLabel || nRealLabel==s_nStereoDontCareLabel || (nRealLabel>=(OutputLabelType)m_nMinDispOffset && nRealLabel<=(OutputLabelType)m_nMaxDispOffset));
    return (InternalLabelType)std::distance(m_vStereoLabels.begin(),std::find(m_vStereoLabels.begin(),m_vStereoLabels.end(),nRealLabel));
}

inline StereoSegmMatcher::AssocCountType StereoSegmMatcher::GraphModelData::getAssocCount(int nRowIdx, int nColIdx) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(nColIdx>=-(int)m_nMaxDispOffset && nColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    return ((AssocCountType*)m_oAssocCounts.data)[(nRowIdx/m_nDispOffsetStep)*m_oAssocCounts.cols + (nColIdx+m_nMaxDispOffset)/m_nDispOffsetStep];
}

inline void StereoSegmMatcher::GraphModelData::addAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
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

inline void StereoSegmMatcher::GraphModelData::removeAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
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

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcAddAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
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

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcRemoveAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
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

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcTotalAssocCost() const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
    ValueType tEnergy = ValueType(0);
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx)
        for(int nColIdx=-(int)m_nMaxDispOffset; nColIdx<(int)(m_oGridSize[1]-m_nMinDispOffset); ++nColIdx)
            tEnergy += m_aAssocCostSumLUT[getAssocCount(nRowIdx,nColIdx)];
    // @@@@ really needed?
    const size_t nTotNodeCount = m_oGridSize.total();
    for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
        const InternalLabelType nCurrLabel = ((InternalLabelType*)m_oStereoLabeling.data)[nNodeIdx];
        if(nCurrLabel>=m_nStereoDontCareLabelIdx) // both special labels treated here
            tEnergy += ValueType(100000); // @@@@ dirty
    }
    lvDbgAssert(tEnergy>=ValueType(0));
    return tEnergy;
}

inline void StereoSegmMatcher::GraphModelData::calcStereoMoveCosts(InternalLabelType nNewLabel) const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1);
    lvDbgAssert(m_oGridSize==m_oAssocCosts.size && m_oGridSize==m_oStereoUnaryCosts.size && m_oGridSize==m_oStereoPairwCosts.size);
    // @@@@@ openmp here?
    const size_t nTotNodeCount = m_oGridSize.total();
    for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
        const GraphModelData::NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
        const InternalLabelType& nInitLabel = ((InternalLabelType*)m_oStereoLabeling.data)[nNodeIdx];
        lvDbgAssert(&nInitLabel==&m_oStereoLabeling(oNode.nRowIdx,oNode.nColIdx));
        ValueType& tAssocCost = ((ValueType*)m_oAssocCosts.data)[nNodeIdx];
        lvDbgAssert(&tAssocCost==&m_oAssocCosts(oNode.nRowIdx,oNode.nColIdx));
        ValueType& tUnaryCost = ((ValueType*)m_oStereoUnaryCosts.data)[nNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_oStereoUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
        if(nInitLabel!=nNewLabel) {
            const ValueType tAssocEnergyCost = calcRemoveAssocCost(oNode.nRowIdx,oNode.nColIdx,nInitLabel)+calcAddAssocCost(oNode.nRowIdx,oNode.nColIdx,nNewLabel);
            const ValueType tUnaryEnergyInit = oNode.pStereoUnaryFunc->second(&nInitLabel);
            const ValueType tUnaryEnergyModif = oNode.pStereoUnaryFunc->second(&nNewLabel);
            // @@@@@ merge assoc + vissim back into one (unary) at final cleanup
            tUnaryCost = tUnaryEnergyModif-tUnaryEnergyInit;
            tAssocCost = tAssocEnergyCost;
        }
        else
            tAssocCost = tUnaryCost = ValueType(0);
        // @@@@ CODE BELOW IS ONLY FOR DEBUG/DISPLAY
        ValueType& tPairwCost = ((ValueType*)m_oStereoPairwCosts.data)[nNodeIdx];
        lvDbgAssert(&tPairwCost==&m_oStereoPairwCosts(oNode.nRowIdx,oNode.nColIdx));
        tPairwCost = ValueType(0);
        for(size_t nOrientIdx=0; nOrientIdx<m_vNodeInfos[nNodeIdx].anStereoPairwFactIDs.size(); ++nOrientIdx) {
            if(oNode.anStereoPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                lvDbgAssert(oNode.apStereoPairwFuncs[nOrientIdx] && oNode.anPairwNodeIdxs[nOrientIdx]<m_oGridSize.total());
                std::array<InternalLabelType,2> aLabels = {nInitLabel,((InternalLabelType*)m_oStereoLabeling.data)[oNode.anPairwNodeIdxs[nOrientIdx]]};
                const ValueType tPairwEnergyInit = oNode.apStereoPairwFuncs[nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyInit>=ValueType(0));
                aLabels[0] = nNewLabel;
                const ValueType tPairwEnergyModif = oNode.apStereoPairwFuncs[nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyModif>=ValueType(0));
                tPairwCost += tPairwEnergyModif-tPairwEnergyInit;
            }
        }
    }
}

inline void StereoSegmMatcher::GraphModelData::calcResegmMoveCosts(InternalLabelType nNewLabel) const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1);
    lvDbgAssert(m_oGridSize==m_oAssocCosts.size && m_oGridSize==m_oResegmUnaryCosts.size && m_oGridSize==m_oResegmPairwCosts.size);
    // @@@@@ openmp here?
    const size_t nTotNodeCount = m_oGridSize.total();
    for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
        const GraphModelData::NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
        const InternalLabelType& nInitLabel = ((InternalLabelType*)m_oResegmLabeling.data)[nNodeIdx];
        lvDbgAssert(&nInitLabel==&m_oResegmLabeling(oNode.nRowIdx,oNode.nColIdx));
        ValueType& tUnaryCost = ((ValueType*)m_oResegmUnaryCosts.data)[nNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_oResegmUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
        if(nInitLabel!=nNewLabel) {
            const ValueType tEnergyInit = oNode.pResegmUnaryFunc->second(&nInitLabel);
            const ValueType tEnergyModif = oNode.pResegmUnaryFunc->second(&nNewLabel);
            tUnaryCost = tEnergyModif-tEnergyInit;
        }
        else
            tUnaryCost = ValueType(0);
        // @@@@ CODE BELOW IS ONLY FOR DEBUG/DISPLAY
        ValueType& tPairwCost = ((ValueType*)m_oResegmPairwCosts.data)[nNodeIdx];
        lvDbgAssert(&tPairwCost==&m_oResegmPairwCosts(oNode.nRowIdx,oNode.nColIdx));
        tPairwCost = ValueType(0);
        for(size_t nOrientIdx=0; nOrientIdx<m_vNodeInfos[nNodeIdx].anResegmPairwFactIDs.size(); ++nOrientIdx) {
            if(oNode.anResegmPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                lvDbgAssert(oNode.apResegmPairwFuncs[nOrientIdx] && oNode.anPairwNodeIdxs[nOrientIdx]<m_oGridSize.total());
                std::array<InternalLabelType,2> aLabels = {nInitLabel,((InternalLabelType*)m_oResegmLabeling.data)[oNode.anPairwNodeIdxs[nOrientIdx]]};
                const ValueType tPairwEnergyInit = oNode.apResegmPairwFuncs[nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyInit>=ValueType(0));
                aLabels[0] = nNewLabel;
                const ValueType tPairwEnergyModif = oNode.apResegmPairwFuncs[nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyModif>=ValueType(0));
                tPairwCost += tPairwEnergyModif-tPairwEnergyInit;
            }
        }
    }
}

inline opengm::InferenceTermination StereoSegmMatcher::GraphModelData::infer() const {
    lvDbgAssert(m_pStereoModel && m_pResegmModel);
    lvAssert_(m_bModelUpToDate,"model must be up-to-date with new data before inference can begin");
    lvDbgAssert_(m_pStereoModel->numberOfVariables()==m_oStereoLabeling.total(),"stereo graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_pResegmModel->numberOfVariables()==m_oResegmLabeling.total(),"resegm graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_oGridSize.dims()==2 && m_oGridSize==m_oStereoLabeling.size && m_oGridSize==m_oResegmLabeling.size,"output labelings must be 2d grid");
    // @@@@@@ use one gm labeling/output to infer the stereo result for another camera?
    const size_t nTotNodeCount = m_oGridSize.total();
    lvDbgAssert(nTotNodeCount==m_vNodeInfos.size());
    const size_t nStereoLabels = m_vStereoLabels.size();
    const size_t nResegmLabels = size_t(2);
    const cv::Point2i oDisplayOffset = {-(int)m_nGridBorderSize,-(int)m_nGridBorderSize};
    lvIgnore(oDisplayOffset); // @@@@@@
    const cv::Rect oAssocCountsROI((int)m_nMaxDispOffset,0,(int)m_oGridSize[1],(int)m_oGridSize[0]);
    lvIgnore(oAssocCountsROI); // @@@@@@
    const cv::Rect oFeatROI((int)m_nGridBorderSize,(int)m_nGridBorderSize,(int)(m_oGridSize[1]-m_nGridBorderSize*2),(int)(m_oGridSize[0]-m_nGridBorderSize*2));
    lvIgnore(oFeatROI); // @@@@@@
    // @@@@@@ use member-alloc QPBO here instead of stack
    kolmogorov::qpbo::QPBO<ValueType> oBinaryEnergyMinimizer((int)nTotNodeCount,0); // @@@@@@@ preset max edge count using max clique size times node count
    // @@@@@@ THIS IS A DYNAMIC MRF OPTIMIZATION PROBLEM; MUST TRY TO REUSE MRF STRUCTURE
    std::array<typename HigherOrderEnergy<ValueType,s_nMaxOrder>::VarId,s_nMaxOrder> aTermEnergyLUT;
    std::array<InternalLabelType,s_nMaxOrder> aCliqueLabels;
    std::array<ValueType,s_nMaxCliqueAssign> aCliqueCoeffs;
    size_t nMoveIter=0, nConsecUnchangedLabels=0, nOrderingIdx=0;
    lvDbgAssert(m_vStereoLabelOrdering.size()==m_vStereoLabels.size());
    InternalLabelType nStereoAlphaLabel = m_vStereoLabelOrdering[nOrderingIdx];
    lv::StopWatch oLocalTimer;
    ValueType tLastStereoEnergy = m_pStereoInf->value();
    ValueType tLastResegmEnergy = m_pResegmInf->value();
    lvIgnore(tLastStereoEnergy);
    lvIgnore(tLastResegmEnergy);
    // each iter below is an alpha-exp move based on A. Fix's primal-dual energy minimization method for higher-order MRFs
    // see "A Primal-Dual Algorithm for Higher-Order Multilabel Markov Random Fields" in CVPR2014 for more info (doi = 10.1109/CVPR.2014.149)
    while(++nMoveIter<=m_nMaxStereoIterCount && nConsecUnchangedLabels<nStereoLabels+nResegmLabels) {
        calcStereoMoveCosts(nStereoAlphaLabel);

        if(lv::getVerbosity()>=3) {

            /*lvCout << "\n\n\n\n\n\n@@@ next label idx = " << (int)nStereoAlphaLabel << ",  real = " << (int)getRealLabel(nStereoAlphaLabel) << "\n\n" << std::endl;

            lvCout << "disp = " << lv::to_string(m_oStereoLabeling,oDisplayOffset) << std::endl;
            lvCout << "assoc_counts = " << lv::to_string(m_oAssocCounts(oAssocCountsROI),oDisplayOffset) << std::endl;
            lvCout << "assoc_cost = " << lv::to_string(m_oAssocCosts,oDisplayOffset) << std::endl;
            lvCout << "unary = " << lv::to_string(m_oUnaryCosts,oDisplayOffset) << std::endl;
            lvCout << "pairw = " << lv::to_string(m_oPairwCosts,oDisplayOffset) << std::endl;

            lvCout << "disp = " << lv::to_string(m_oStereoLabeling(oFeatROI)) << std::endl;
            lvCout << "assoc_counts = " << lv::to_string(m_oAssocCounts(oFeatROI+cv::Point2i((int)m_nMaxDispOffset,0))) << std::endl;
            lvCout << "assoc_cost = " << lv::to_string(m_oAssocCosts(oFeatROI)) << std::endl;
            lvCout << "unary = " << lv::to_string(m_oUnaryCosts(oFeatROI)) << std::endl;
            lvCout << "pairw = " << lv::to_string(m_oPairwCosts(oFeatROI)) << std::endl;*/

            const cv::Rect oLineROI(20,128,200,1);
            lvCout << "-----\n\n\n";
            lvCout << "inputa = " << lv::to_string(m_aInputs[InputPack_LeftImg](oLineROI)) << '\n';
            lvCout << "inputb = " << lv::to_string(m_aInputs[InputPack_RightImg](oLineROI)) << '\n';
            lvCout << "disp = " << lv::to_string(m_oStereoLabeling(oLineROI)) << '\n';
            lvCout << "assoc_counts = " << lv::to_string(m_oAssocCounts(oLineROI+cv::Point2i((int)m_nMaxDispOffset,0))) << '\n';
            lvCout << "assoc_cost = " << lv::to_string(m_oAssocCosts(oLineROI)) << '\n';
            lvCout << "unary = " << lv::to_string(m_oStereoUnaryCosts(oLineROI)) << '\n';
            lvCout << "pairw = " << lv::to_string(m_oStereoPairwCosts(oLineROI)) << '\n';

            lvCout << "next label = " << (int)getRealLabel(nStereoAlphaLabel) << '\n';
            cv::Mat oCurrAssocCountsDisplay = StereoSegmMatcher::getAssocCountsMapDisplay(*this);
            cv::resize(oCurrAssocCountsDisplay,oCurrAssocCountsDisplay,cv::Size(),4,4,cv::INTER_NEAREST);
            cv::imshow("assoc_counts",oCurrAssocCountsDisplay);
            cv::Mat oCurrLabelingDisplay = StereoSegmMatcher::getStereoMapDisplay(*this);
            cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),4,4,cv::INTER_NEAREST);
            cv::imshow("disp",oCurrLabelingDisplay);
            cv::waitKey(0);
        }
        HigherOrderEnergy<ValueType,s_nMaxOrder> oHigherOrderEnergyReducer; // @@@@@@ check if HigherOrderEnergy::Clear() can be used instead of realloc
        oHigherOrderEnergyReducer.AddVars((int)nTotNodeCount);
        const auto lHigherOrderFactorAdder = [&](auto& oGraphFactor, size_t nFactOrder) {
            lvDbgAssert(oGraphFactor.numberOfVariables()==nFactOrder);
            const size_t nAssignCount = 1UL<<nFactOrder;
            std::fill_n(aCliqueCoeffs.begin(),nAssignCount,(ValueType)0);
            for(size_t nAssignIdx=0; nAssignIdx<nAssignCount; ++nAssignIdx) {
                for(size_t nVarIdx=0; nVarIdx<nFactOrder; ++nVarIdx)
                    aCliqueLabels[nVarIdx] = (nAssignIdx&(1<<nVarIdx))?nStereoAlphaLabel:m_oStereoLabeling((int)oGraphFactor.variableIndex(nVarIdx));
                for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                    if(!(nAssignIdx&~nAssignSubsetIdx)) {
                        int nParityBit = 0;
                        for(size_t nVarIdx=0; nVarIdx<nFactOrder; ++nVarIdx)
                            nParityBit ^= (((nAssignIdx^nAssignSubsetIdx)&(1<<nVarIdx))!=0);
                        const ValueType fCurrAssignEnergy = oGraphFactor(aCliqueLabels.begin());
                        aCliqueCoeffs[nAssignSubsetIdx] += nParityBit?-fCurrAssignEnergy:fCurrAssignEnergy;
                    }
                }
            }
            for(size_t nAssignSubsetIdx=1; nAssignSubsetIdx<nAssignCount; ++nAssignSubsetIdx) {
                int nCurrTermDegree = 0;
                for(size_t nVarIdx=0; nVarIdx<nFactOrder; ++nVarIdx)
                    if(nAssignSubsetIdx&(1<<nVarIdx))
                        aTermEnergyLUT[nCurrTermDegree++] = (typename HigherOrderEnergy<ValueType,s_nMaxOrder>::VarId)oGraphFactor.variableIndex(nVarIdx);
                std::sort(aTermEnergyLUT.begin(),aTermEnergyLUT.begin()+nCurrTermDegree);
                oHigherOrderEnergyReducer.AddTerm(aCliqueCoeffs[nAssignSubsetIdx],nCurrTermDegree,aTermEnergyLUT.data());
            }
        };
        for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
            const GraphModelData::NodeInfo& oNode = m_vNodeInfos[nNodeIdx];
            // manually add 1st order factors while evaluating new assoc energy
            if(oNode.nStereoUnaryFactID!=SIZE_MAX) {
                const ValueType& tAssocCost = ((ValueType*)m_oAssocCosts.data)[nNodeIdx];
                lvDbgAssert(&tAssocCost==&m_oAssocCosts(oNode.nRowIdx,oNode.nColIdx));
                const ValueType& tUnaryCost = ((ValueType*)m_oStereoUnaryCosts.data)[nNodeIdx];
                lvDbgAssert(&tUnaryCost==&m_oStereoUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
                oHigherOrderEnergyReducer.AddUnaryTerm((int)nNodeIdx,tAssocCost+tUnaryCost);
            }
            // now add 2nd order & higher order factors via lambda
            if(oNode.anStereoPairwFactIDs[0]!=SIZE_MAX)
                lHigherOrderFactorAdder(m_pStereoModel->operator[](oNode.anStereoPairwFactIDs[0]),2);
            if(oNode.anStereoPairwFactIDs[1]!=SIZE_MAX)
                lHigherOrderFactorAdder(m_pStereoModel->operator[](oNode.anStereoPairwFactIDs[1]),2);
            // @@@@@ add higher o facts here (3-conn on epi lines?)
            lvIgnore(lHigherOrderFactorAdder);
        }
        oBinaryEnergyMinimizer.Reset();
        oHigherOrderEnergyReducer.ToQuadratic(oBinaryEnergyMinimizer); // @@@@@ internally might call addNodes/edges to QPBO object; optim tbd
        //@@@@@@ oBinaryEnergyMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11); @@@ can still add more if needed
        oBinaryEnergyMinimizer.Solve();
        oBinaryEnergyMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
        size_t nChangedLabelings = 0;
        //cv::Mat_<uchar> SWAPS(m_oGridSize(),0);
        for(size_t nNodeIdx=0; nNodeIdx<nTotNodeCount; ++nNodeIdx) {
            const int nMoveLabel = oBinaryEnergyMinimizer.GetLabel((int)nNodeIdx);
            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
            if(nMoveLabel==1) { // node label changed to alpha
                const int nRowIdx = m_vNodeInfos[nNodeIdx].nRowIdx;
                const int nColIdx = m_vNodeInfos[nNodeIdx].nColIdx;
                //SWAPS(nRowIdx,nColIdx) = 255;
                const InternalLabelType nOldStereoLabel = m_oStereoLabeling(nRowIdx,nColIdx);
                if(nOldStereoLabel<m_nStereoDontCareLabelIdx)
                    removeAssoc(nRowIdx,nColIdx,nOldStereoLabel);
                m_oStereoLabeling(nRowIdx,nColIdx) = nStereoAlphaLabel;
                if(nStereoAlphaLabel<m_nStereoDontCareLabelIdx)
                    addAssoc(nRowIdx,nColIdx,nStereoAlphaLabel);
                ++nChangedLabelings;
            }
        }
        /*lvCout << "\n\n\n\n" << std::endl;
        lvCout << "swaps (tot=" << (size_t)cv::sum(SWAPS)[0]/255 << ") = " << lv::to_string(SWAPS,oDisplayOffset) << std::endl;
        lvCout << "swaps (tot=" << (size_t)cv::sum(SWAPS(oFeatROI))[0]/255 << ") = " << lv::to_string(SWAPS(oFeatROI)) << std::endl;
        lvCout << "\n\n\n\n" << std::endl;*/
        lvDbgAssert__(tLastStereoEnergy>=m_pStereoInf->value() && (tLastStereoEnergy=m_pStereoInf->value())>=ValueType(0),"not minimizing! curr=%f",(float)m_pStereoInf->value());
        nConsecUnchangedLabels = (nChangedLabelings>0)?0:nConsecUnchangedLabels+1;
        // @@@@ order of future moves can be influenced by labels that cause the most changes? (but only late, to avoid bad local minima?)
        nStereoAlphaLabel = m_vStereoLabelOrdering[(++nOrderingIdx%=nStereoLabels)];
    }
    lvLog_(1,"Inference completed in %f second(s).",oLocalTimer.tock());
    return opengm::InferenceTermination::NORMAL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::StereoGraphInference::StereoGraphInference(const GraphModelData& oData) :
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

inline std::string StereoSegmMatcher::StereoGraphInference::name() const {
    return std::string("litiv-stereo-matcher");
}

inline const StereoSegmMatcher::StereoModelType& StereoSegmMatcher::StereoGraphInference::graphicalModel() const {
    lvDbgAssert(m_oData.m_pStereoModel);
    return *m_oData.m_pStereoModel;
}

inline opengm::InferenceTermination StereoSegmMatcher::StereoGraphInference::infer() {
    return m_oData.infer();
}

inline void StereoSegmMatcher::StereoGraphInference::setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) {
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_pStereoModel->numberOfVariables()),lv::make_range(InternalLabelType(0),InternalLabelType(m_oData.m_vStereoLabels.size()-1))).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_pStereoModel->numberOfVariables(),m_oData.m_oStereoLabeling.begin());
}

inline void StereoSegmMatcher::StereoGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),m_oData.m_vStereoLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_oStereoLabeling.begin(),[&](const OutputLabelType& nRealLabel){return m_oData.getInternalLabel(nRealLabel);});
}

inline opengm::InferenceTermination StereoSegmMatcher::StereoGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    if(n==1) {
        lvDbgAssert_(m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_oStereoLabeling.total());
        std::copy(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones!
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void StereoSegmMatcher::StereoGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_pStereoModel->numberOfVariables(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_oStereoLabeling.size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return m_oData.getRealLabel(nLabel);});
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::StereoGraphInference::value() const {
    lvDbgAssert_(m_oData.m_pStereoModel->numberOfVariables()==m_oData.m_oStereoLabeling.total(),"graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oStereoLabeling.size,"output labeling must be a 2d grid");
    lvAssert_(m_oData.m_bModelUpToDate,"model must be up-to-date with new data before inference can begin");
    return m_oData.m_pStereoModel->evaluate((InternalLabelType*)m_oData.m_oStereoLabeling.data)+m_oData.calcTotalAssocCost();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::ResegmGraphInference::ResegmGraphInference(const GraphModelData& oData) :
        m_oData(oData) {
    lvDbgAssert_(m_oData.m_pResegmModel && m_oData.m_pResegmModel->numberOfFactors()>0,"invalid graph");
    const ResegmModelType& oGM = *m_oData.m_pResegmModel;
    for(size_t nFactIdx=0; nFactIdx<m_oData.m_pResegmModel->numberOfFactors(); ++nFactIdx)
        lvDbgAssert__(oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvDbgAssert_(m_oData.m_pResegmModel->numberOfVariables()>0 && m_oData.m_pResegmModel->numberOfVariables()==(IndexType)m_oData.m_oGridSize.total(),"graph node count must match grid size");
    for(size_t nNodeIdx=0; nNodeIdx<m_oData.m_pResegmModel->numberOfVariables(); ++nNodeIdx)
        lvDbgAssert_(m_oData.m_pResegmModel->numberOfLabels(nNodeIdx)==size_t(2),"graph nodes must all have the same number of labels");
    lvIgnore(oGM);
}

inline std::string StereoSegmMatcher::ResegmGraphInference::name() const {
    return std::string("litiv-segm-matcher");
}

inline const StereoSegmMatcher::ResegmModelType& StereoSegmMatcher::ResegmGraphInference::graphicalModel() const {
    lvDbgAssert(m_oData.m_pResegmModel);
    return *m_oData.m_pResegmModel;
}

inline opengm::InferenceTermination StereoSegmMatcher::ResegmGraphInference::infer() {
    return m_oData.infer();
}

inline void StereoSegmMatcher::ResegmGraphInference::setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) {
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_pResegmModel->numberOfVariables()),std::vector<InternalLabelType>{s_nBackgroundLabel,s_nForegroundLabel}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_oResegmLabeling.isContinuous() && m_oData.m_oResegmLabeling.total()==m_oData.m_pResegmModel->numberOfVariables(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_pResegmModel->numberOfVariables(),m_oData.m_oResegmLabeling.begin());
}

inline void StereoSegmMatcher::ResegmGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),std::vector<OutputLabelType>{s_nBackgroundLabel,s_nForegroundLabel}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_oResegmLabeling.begin(),[&](const OutputLabelType& nRealLabel){return InternalLabelType(nRealLabel?s_nForegroundLabel:s_nBackgroundLabel);});
}

inline opengm::InferenceTermination StereoSegmMatcher::ResegmGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    if(n==1) {
        lvDbgAssert_(m_oData.m_oResegmLabeling.total()==m_oData.m_pResegmModel->numberOfVariables(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_oResegmLabeling.total());
        std::copy(m_oData.m_oResegmLabeling.begin(),m_oData.m_oResegmLabeling.end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones (same values, different type)
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void StereoSegmMatcher::ResegmGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgAssert_(m_oData.m_oResegmLabeling.isContinuous() && m_oData.m_oResegmLabeling.total()==m_oData.m_pResegmModel->numberOfVariables(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_oResegmLabeling.size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_oResegmLabeling.begin(),m_oData.m_oResegmLabeling.end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return nLabel?s_nForegroundLabel:s_nBackgroundLabel;});
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::ResegmGraphInference::value() const {
    lvDbgAssert_(m_oData.m_pResegmModel->numberOfVariables()==m_oData.m_oResegmLabeling.total(),"graph node count and labeling mat size mismatch");
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oResegmLabeling.size,"output labeling must be a 2d grid");
    lvAssert_(m_oData.m_bModelUpToDate,"model must be up-to-date with new data before inference can begin");
    return m_oData.m_pResegmModel->evaluate((InternalLabelType*)m_oData.m_oResegmLabeling.data)+m_oData.calcTotalAssocCost();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline cv::Mat StereoSegmMatcher::getStereoMapDisplay(const GraphModelData& oData) {
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

inline cv::Mat StereoSegmMatcher::getAssocCountsMapDisplay(const GraphModelData& oData) {
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