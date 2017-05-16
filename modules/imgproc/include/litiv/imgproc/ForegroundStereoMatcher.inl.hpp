
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
#if (STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY/*+...*/\
    )!=1
#error "Must specify only one image affinity map computation approach to use."
#endif //(features config ...)!=1
#define STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY (STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY||STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY||STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY)

inline StereoSegmMatcher::StereoSegmMatcher(size_t nMinDispOffset, size_t nMaxDispOffset) {
    static_assert(getInputStreamCount()==4 && getOutputStreamCount()==4 && getCameraCount()==2,"i/o stream must be two image-mask pairs");
    static_assert(getInputStreamCount()==InputPackSize && getOutputStreamCount()==OutputPackSize,"bad i/o internal enum mapping");
    m_nDispStep = STEREOSEGMATCH_DEFAULT_DISPARITY_STEP;
    lvAssert_(m_nDispStep>0,"specified disparity offset step size must be strictly positive");
    if(nMaxDispOffset<nMinDispOffset)
        std::swap(nMaxDispOffset,nMinDispOffset);
    nMaxDispOffset -= (nMaxDispOffset-nMinDispOffset)%m_nDispStep;
    lvAssert_(nMaxDispOffset<size_t(s_nOccludedLabel),"using reserved disparity integer label value");
    lvAssert_((nMaxDispOffset-nMinDispOffset)/m_nDispStep>size_t(0),"disparity range must not be null");
    lvAssert_(((nMaxDispOffset-nMinDispOffset)%m_nDispStep)==0,"irregular disparity range label count with given step size");
    const size_t nMaxAllowedDispLabelCount = size_t(std::numeric_limits<InternalLabelType>::max()-2);
    const size_t nExpectedDispLabelCount = ((nMaxDispOffset-nMinDispOffset)/m_nDispStep)+1; // +1 since max label is included in the range
    lvAssert__(nMaxAllowedDispLabelCount>=nExpectedDispLabelCount,"internal stereo label type too small for given disparity range (max = %d)",(int)nMaxAllowedDispLabelCount);
    m_vStereoLabels = lv::make_range((OutputLabelType)nMinDispOffset,(OutputLabelType)nMaxDispOffset,(OutputLabelType)m_nDispStep);
    lvDbgAssert(nExpectedDispLabelCount==m_vStereoLabels.size());
    lvAssert_(m_vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
}

inline void StereoSegmMatcher::initialize(const std::array<cv::Mat,2>& aROIs) {
    lvAssert_(!aROIs[0].empty() && aROIs[0].total()>1 && aROIs[0].type()==CV_8UC1,"bad input ROI size/type");
    lvAssert_(lv::MatInfo(aROIs[0])==lv::MatInfo(aROIs[1]),"mismatched ROI size/type");
    lvAssert_(m_nDispStep>0,"specified disparity offset step size must be strictly positive");
    lvAssert_(m_vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
    m_pModelData = std::make_unique<GraphModelData>(aROIs,m_vStereoLabels,m_nDispStep);
}

inline void StereoSegmMatcher::apply(const MatArrayIn& aInputs, MatArrayOut& aOutputs) {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"lots of hardcoded indices below");
    lvAssert_(m_pModelData,"model must be initialized first");
    const cv::Size oExpectedSize = m_pModelData->m_oGridSize;
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx) {
        lvAssert__(aInputs[nInputIdx].dims==2 && oExpectedSize==aInputs[nInputIdx].size(),"input in array at index=%d had the wrong size",(int)nInputIdx);
        lvAssert_((nInputIdx%InputPackOffset)==InputPackOffset_Mask || (aInputs[nInputIdx].type()==CV_8UC1 || aInputs[nInputIdx].type()==CV_8UC3),"unexpected input image type");
        lvAssert_((nInputIdx%InputPackOffset)==InputPackOffset_Img || aInputs[nInputIdx].type()==CV_8UC1,"unexpected input mask type");
        aInputs[nInputIdx].copyTo(m_pModelData->m_aInputs[nInputIdx]);
    }
    if(lv::getVerbosity()>=2) {
        cv::imshow("left input (a)",aInputs[InputPack_LeftImg]);
        cv::imshow("right input (b)",aInputs[InputPack_RightImg]);
    }
    lvAssert_(!m_pModelData->m_bUsePrecalcFeatsNext || m_pModelData->m_vNextFeats.size()==GraphModelData::FeatPackSize,"unexpected precalculated features vec size");
    if(!m_pModelData->m_bUsePrecalcFeatsNext)
        m_pModelData->calcFeatures(aInputs);
    else
        m_pModelData->m_bUsePrecalcFeatsNext = false;
    m_pModelData->infer();
    m_pModelData->m_oStereoLabeling.copyTo(aOutputs[OutputPack_LeftDisp]);
    m_pModelData->m_oResegmLabeling.copyTo(aOutputs[OutputPack_LeftMask]);
    // @@@@@ fill missing outputs
    cv::Mat_<OutputLabelType>(oExpectedSize,OutputLabelType(0)).copyTo(aOutputs[OutputPack_RightDisp]);
    cv::Mat_<OutputLabelType>(oExpectedSize,OutputLabelType(0)).copyTo(aOutputs[OutputPack_RightMask]);
    for(size_t nOutputIdx=0; nOutputIdx<aOutputs.size(); ++nOutputIdx)
        aOutputs[nOutputIdx].copyTo(m_pModelData->m_aOutputs[nOutputIdx]); // @@@@ copy for temporal stuff later
}

inline void StereoSegmMatcher::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    lvAssert_(m_pModelData,"model must be initialized first");
    m_pModelData->calcFeatures(aInputs,pFeatsPacket);
}

inline void StereoSegmMatcher::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvAssert_(m_pModelData,"model must be initialized first");
    m_pModelData->setNextFeatures(oPackedFeats);
}

inline std::string StereoSegmMatcher::getFeatureExtractorName() const {
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY
    return "sc-dasc-gf";
#elif STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY
    return "sc-dasc-rf";
#elif STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY
    return "sc-lss";
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    return "sc-mi";
#elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    return "sc-ssqrdiff";
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
}

inline size_t StereoSegmMatcher::getMaxLabelCount() const {
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->m_vStereoLabels.size();
}

inline const std::vector<StereoSegmMatcher::OutputLabelType>& StereoSegmMatcher::getLabels() const {
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->m_vStereoLabels;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::GraphModelData::GraphModelData(const std::array<cv::Mat,2>& aROIs, const std::vector<OutputLabelType>& vRealStereoLabels, size_t nStereoLabelStep) :
        m_nMaxMoveIterCount(STEREOSEGMATCH_DEFAULT_MAX_MOVE_ITER),
        m_nStereoLabelOrderRandomSeed(size_t(0)),
        m_nStereoLabelingRandomSeed(size_t(0)),
        m_aROIs(std::array<cv::Mat_<uchar>,2>{aROIs[0]>0,aROIs[1]>0}),
        m_oGridSize(m_aROIs[0].size()),
        m_vStereoLabels(lv::concat<OutputLabelType>(vRealStereoLabels,std::vector<OutputLabelType>{s_nDontCareLabel,s_nOccludedLabel})),
        m_nRealStereoLabels(vRealStereoLabels.size()),
        m_nStereoLabels(vRealStereoLabels.size()+2),
        m_nDispOffsetStep(nStereoLabelStep),
        m_nMinDispOffset(size_t(m_vStereoLabels[0])),
        m_nMaxDispOffset(size_t(m_vStereoLabels.size()>3?m_vStereoLabels[m_vStereoLabels.size()-3]:m_vStereoLabels.back())),
        m_nDontCareLabelIdx(InternalLabelType(m_vStereoLabels.size()-2)),
        m_nOccludedLabelIdx(InternalLabelType(m_vStereoLabels.size()-1)),
        m_bUsePrecalcFeatsNext(false) {
    lvAssert(lv::MatInfo(m_aROIs[0])==lv::MatInfo(m_aROIs[1]));
    lvAssert_(m_nMaxMoveIterCount>0,"max iter counts must be strictly positive");
    lvAssert_(cv::countNonZero(m_aROIs[0]>0)>1 && cv::countNonZero(m_aROIs[1]>0)>1,"graph ROIs must have at least two nodes");
    lvAssert_(m_oGridSize.dims()==2 && m_oGridSize.total()>size_t(1),"graph grid must be 2D and have at least two nodes");
    lvAssert_(m_vStereoLabels.size()>3,"graph must have at least two possible output stereo labels, beyond reserved ones");
    lvAssert_(m_vStereoLabels.size()<=size_t(std::numeric_limits<InternalLabelType>::max()),"too many labels for internal type");
    lvDbgAssert(m_vStereoLabels[m_nDontCareLabelIdx]==s_nDontCareLabel && m_vStereoLabels[m_nOccludedLabelIdx]==s_nOccludedLabel);
    lvDbgAssert(std::min_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==m_vStereoLabels.begin() && m_vStereoLabels[0]>=OutputLabelType(0));
    lvDbgAssert(std::max_element(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2)==(m_vStereoLabels.begin()+m_vStereoLabels.size()-3));
    lvDbgAssert(std::equal(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_vStereoLabels.size()-2,lv::unique(m_vStereoLabels.begin(),m_vStereoLabels.end()).begin()+1));
    lvAssert_(m_nDispOffsetStep>0,"label step size must be positive");
    lvAssert_(m_oGridSize[1]>m_nMinDispOffset,"row length too small for smallest disp");
    lvAssert_(m_nMinDispOffset<m_nMaxDispOffset,"min/max disp offsets mismatch");
    lvDbgAssert_(std::numeric_limits<AssocCountType>::max()>m_oGridSize[1],"grid width is too large for association counter type");
#if STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY
    m_pImgDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS,DASC_DEFAULT_GF_SUBSPL,DASC_DEFAULT_PREPROCESS);
#elif STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY
    m_pImgDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR,DASC_DEFAULT_RF_ITERS,DASC_DEFAULT_PREPROCESS);
#elif STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY
    m_pImgDescExtractor = std::make_unique<LSS>(STEREOSEGMATCH_DEFAULT_LSSDESC_PATCH,STEREOSEGMATCH_DEFAULT_LSSDESC_RAD);
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    cv::Size oMIWinSize(int(STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD)*2+1,int(STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD)*2+1);
    m_pImgDescExtractor = std::make_unique<MutualInfo>(oMIWinSize,true);
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
    const size_t nShapeContextInnerRadius=2, nShapeContextOuterRadius=STEREOSEGMATCH_DEFAULT_SHAPEDESC_RAD;
    m_pShpDescExtractor = std::make_unique<ShapeContext>(nShapeContextInnerRadius,nShapeContextOuterRadius,SHAPECONTEXT_DEFAULT_ANG_BINS,SHAPECONTEXT_DEFAULT_RAD_BINS);
#if STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    constexpr int nSSqrDiffKernelSize = int(STEREOSEGMATCH_DEFAULT_SSQDIFF_PATCH);
    const cv::Size oMinWindowSize(nSSqrDiffKernelSize,nSSqrDiffKernelSize);
    m_nGridBorderSize = size_t(nSSqrDiffKernelSize/2);
#else //!STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    const cv::Size oMinWindowSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#endif //!STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    lvAssert__(oMinWindowSize.width<=(int)m_oGridSize[1] && oMinWindowSize.height<=(int)m_oGridSize[0],"image is too small to compute descriptors with current pattern size -- need at least (%d,%d) and got (%d,%d)",oMinWindowSize.width,oMinWindowSize.height,(int)m_oGridSize[1],(int)m_oGridSize[0]);
    lvDbgAssert(m_nGridBorderSize<m_oGridSize[0] && m_nGridBorderSize<m_oGridSize[1]);
    lvDbgAssert((size_t)std::max(m_pShpDescExtractor->borderSize(0),m_pShpDescExtractor->borderSize(1))<=m_nGridBorderSize);
    lvDbgAssert(m_aAssocCostRealAddLUT.size()==m_aAssocCostRealSumLUT.size() && m_aAssocCostRealRemLUT.size()==m_aAssocCostRealSumLUT.size());
    lvDbgAssert(m_aAssocCostApproxAddLUT.size()==m_aAssocCostRealAddLUT.size() && m_aAssocCostApproxRemLUT.size()==m_aAssocCostRealRemLUT.size());
    lvDbgAssert_(m_nMaxDispOffset+m_nDispOffsetStep<m_aAssocCostRealSumLUT.size(),"assoc cost lut size might not be large enough");
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT<m_aAssocCostRealSumLUT.size());
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(0)==0.0f);
    std::fill_n(m_aAssocCostRealAddLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    std::fill_n(m_aAssocCostRealRemLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    std::fill_n(m_aAssocCostRealSumLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,ValueType(0));
    for(size_t nIdx=STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT; nIdx<m_aAssocCostRealAddLUT.size(); ++nIdx) {
        m_aAssocCostRealAddLUT[nIdx] = ValueType(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(nIdx+1-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRealRemLUT[nIdx] = -ValueType(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(nIdx-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRealSumLUT[nIdx] = (nIdx==size_t(0)?ValueType(0):(m_aAssocCostRealSumLUT[nIdx-1]+m_aAssocCostRealAddLUT[nIdx-1]));
    }
    for(size_t nIdx=0; nIdx<m_aAssocCostRealAddLUT.size(); ++nIdx) {
        // 'average' cost of removing one assoc from target pixel (not 'true' energy, but easy-to-optimize worse case)
        m_aAssocCostApproxRemLUT[nIdx] = (nIdx==size_t(0)?ValueType(0):ValueType(-1.0f*m_aAssocCostRealSumLUT[nIdx]/nIdx+0.5f));
        if(m_nDispOffsetStep==size_t(1))
            // if m_nDispOffsetStep==1, then use real cost of adding one assoc to target pixel
            // (target can only have one new assoc per iter, due to single label move)
            m_aAssocCostApproxAddLUT[nIdx] = m_aAssocCostRealAddLUT[nIdx];
        else {
            // otherwise, use average cost of adding 'm_nDispOffsetStep' assocs to target block
            // (i.e. the average cost of adding the max possible number of new assocs to a block per iter)
            m_aAssocCostApproxAddLUT[nIdx] = ValueType(0);
            for(size_t nOffsetIdx=nIdx; nOffsetIdx<nIdx+m_nDispOffsetStep; ++nOffsetIdx)
                m_aAssocCostApproxAddLUT[nIdx] += m_aAssocCostRealAddLUT[std::min(nOffsetIdx,m_aAssocCostRealAddLUT.size()-1)];
            m_aAssocCostApproxAddLUT[nIdx] = ValueType(float(m_aAssocCostApproxAddLUT[nIdx])/m_nDispOffsetStep+0.5f);
        }
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
    lvDbgExec(lvPrint(m_vStereoLabels));
    const size_t nRows = m_oGridSize(0);
    const size_t nCols = m_oGridSize(1);
    const cv::Mat_<uchar>& oROI = m_aROIs[0];
    const size_t nValidNodes = (size_t)cv::countNonZero(oROI);
    lvAssert(nValidNodes<=m_oGridSize.total());
    cv::Mat_<uchar> oErodedROI = oROI.clone();
    cv::erode(oErodedROI,oErodedROI,cv::getStructuringElement(cv::MORPH_RECT,cv::Size((int)m_nGridBorderSize,(int)m_nGridBorderSize)),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar_<uchar>::all(0));
    const size_t nStereoUnaryFuncDataSize = nValidNodes*m_nStereoLabels;
    const size_t nStereoPairwFuncDataSize = nValidNodes*2*(m_nStereoLabels*m_nStereoLabels);
    const size_t nStereoFuncDataSize = nStereoUnaryFuncDataSize+nStereoPairwFuncDataSize/*+...@@@@hoet*/;
    const size_t nResegmUnaryFuncDataSize = nValidNodes*s_nResegmLabels;
    const size_t nResegmPairwFuncDataSize = nValidNodes*2*(s_nResegmLabels*s_nResegmLabels);
    const size_t nResegmFuncDataSize = nResegmUnaryFuncDataSize+nResegmPairwFuncDataSize/*+...@@@@hoet*/;
    const size_t nModelSize = (nStereoFuncDataSize+nResegmFuncDataSize)*sizeof(ValueType)/*+...@@@@externals*/;
    lvLog_(1,"Expecting model size = %zu mb",nModelSize/1024/1024);
    lvAssert__(nModelSize<(CACHE_MAX_SIZE_GB*1024*1024*1024),"too many nodes/labels; model is unlikely to fit in memory (estimated: %zu mb)",nModelSize/1024/1024);
    lvLog(1,"Constructing graphical models...");
    lv::StopWatch oLocalTimer;
    const size_t nStereoFactorsPerNode = 3; // @@@@
    const size_t nResegmFactorsPerNode = 3; // @@@@
    const size_t nStereoFunctions = nValidNodes*nStereoFactorsPerNode;
    const size_t nResegmFunctions = nValidNodes*nResegmFactorsPerNode;
    m_pStereoModel = std::make_unique<StereoModelType>(StereoSpaceType(nValidNodes,(InternalLabelType)m_nStereoLabels),nStereoFactorsPerNode);
    m_pStereoModel->reserveFunctions<ExplicitFunction>(nStereoFunctions);
    m_pResegmModel = std::make_unique<ResegmModelType>(ResegmSpaceType(nValidNodes),nResegmFactorsPerNode);
    m_pResegmModel->reserveFunctions<ExplicitFunction>(nResegmFunctions);
    const std::array<int,2> anAssocCountsDims{int(m_oGridSize[0]),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep)};
    m_oAssocCounts.create(2,anAssocCountsDims.data());
    const std::array<int,3> anAssocMapDims{int(m_oGridSize[0]),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep),int(m_nRealStereoLabels*m_nDispOffsetStep)};
    m_oAssocMap.create(3,anAssocMapDims.data());
    m_oAssocCosts.create(m_oGridSize);
    m_oStereoUnaryCosts.create(m_oGridSize);
    m_oResegmUnaryCosts.create(m_oGridSize);
    m_oStereoPairwCosts.create(m_oGridSize); // for dbg only
    m_oResegmPairwCosts.create(m_oGridSize); // for dbg only
    m_vNodeInfos.resize(m_oGridSize.total());
    m_vValidLUTNodeIdxs.reserve(nValidNodes);
    m_nValidGraphNodes = 0;
    for(int nRowIdx = 0; nRowIdx<(int)nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<(int)nCols; ++nColIdx) {
            const size_t nLUTNodeIdx = nRowIdx*nCols+nColIdx;
            m_vNodeInfos[nLUTNodeIdx].nRowIdx = nRowIdx;
            m_vNodeInfos[nLUTNodeIdx].nColIdx = nColIdx;
            m_vNodeInfos[nLUTNodeIdx].bValidGraphNode = oROI(nRowIdx,nColIdx)>0;
            m_vNodeInfos[nLUTNodeIdx].bNearGraphBorders = (oROI(nRowIdx,nColIdx)==0 || oErodedROI(nRowIdx,nColIdx)==0);
            if(m_vNodeInfos[nLUTNodeIdx].bValidGraphNode) {
                m_vNodeInfos[nLUTNodeIdx].nGraphNodeIdx = m_nValidGraphNodes++;
                m_vValidLUTNodeIdxs.push_back(nLUTNodeIdx);
            }
            // the LUT members below will be properly initialized in the following sections if node is valid
            m_vNodeInfos[nLUTNodeIdx].nStereoUnaryFactID = SIZE_MAX;
            m_vNodeInfos[nLUTNodeIdx].nResegmUnaryFactID = SIZE_MAX;
            m_vNodeInfos[nLUTNodeIdx].pStereoUnaryFunc = nullptr;
            m_vNodeInfos[nLUTNodeIdx].pResegmUnaryFunc = nullptr;
            m_vNodeInfos[nLUTNodeIdx].anPairwLUTNodeIdxs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nLUTNodeIdx].anPairwGraphNodeIdxs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nLUTNodeIdx].anStereoPairwFactIDs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nLUTNodeIdx].anResegmPairwFactIDs = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
            m_vNodeInfos[nLUTNodeIdx].apStereoPairwFuncs = std::array<StereoFunc*,2>{nullptr,nullptr};
            m_vNodeInfos[nLUTNodeIdx].apResegmPairwFuncs = std::array<ResegmFunc*,2>{nullptr,nullptr};
        }
    }
    lvAssert(m_vValidLUTNodeIdxs.size()==nValidNodes);
    lvAssert(m_nValidGraphNodes==nValidNodes);
    m_vStereoUnaryFuncs.reserve(m_nValidGraphNodes);
    m_avStereoPairwFuncs[0].reserve(m_nValidGraphNodes);
    m_avStereoPairwFuncs[1].reserve(m_nValidGraphNodes);
    m_vResegmUnaryFuncs.reserve(m_nValidGraphNodes);
    m_avResegmPairwFuncs[0].reserve(m_nValidGraphNodes);
    m_avResegmPairwFuncs[1].reserve(m_nValidGraphNodes);
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
        lvLog(1,"\tadding unary factors to each graph node...");
        const std::array<size_t,1> aUnaryStereoFuncDims = {m_nStereoLabels};
        const std::array<size_t,1> aUnaryResegmFuncDims = {s_nResegmLabels};
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
            NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            m_vStereoUnaryFuncs.push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
            m_vResegmUnaryFuncs.push_back(m_pResegmModel->addFunctionWithRefReturn(ExplicitFunction()));
            StereoFunc& oStereoFunc = m_vStereoUnaryFuncs.back();
            ResegmFunc& oResegmFunc = m_vResegmUnaryFuncs.back();
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
            oStereoFunc.second.resize(aUnaryStereoFuncDims.begin(),aUnaryStereoFuncDims.end());
            oResegmFunc.second.resize(aUnaryResegmFuncDims.begin(),aUnaryResegmFuncDims.end());
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
            oStereoFunc.second.assign(aUnaryStereoFuncDims.begin(),aUnaryStereoFuncDims.end(),m_pStereoUnaryFuncsDataBase+(nGraphNodeIdx*m_nStereoLabels));
            oResegmFunc.second.assign(aUnaryResegmFuncDims.begin(),aUnaryResegmFuncDims.end(),m_pResegmUnaryFuncsDataBase+(nGraphNodeIdx*s_nResegmLabels));
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
            lvDbgAssert(oStereoFunc.second.strides(0)==1 && oResegmFunc.second.strides(0)==1); // expect no padding
            const std::array<size_t,1> aGraphNodeIndices = {nGraphNodeIdx};
            oNode.nStereoUnaryFactID = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
            oNode.nResegmUnaryFactID = m_pResegmModel->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
            oNode.pStereoUnaryFunc = &oStereoFunc;
            oNode.pResegmUnaryFunc = &oResegmFunc;
        }
    }
    {
        lvLog(1,"\tadding pairwise factors to each graph node pair...");
        // note: current def w/ explicit stereo function will require too much memory if using >>50 disparity labels
        const std::array<size_t,2> aPairwiseStereoFuncDims = {m_nStereoLabels,m_nStereoLabels};
        const std::array<size_t,2> aPairwiseResegmFuncDims = {s_nResegmLabels,s_nResegmLabels};
        std::array<size_t,2> aGraphNodeIndices;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
            const size_t nBaseLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
            NodeInfo& oBaseNode = m_vNodeInfos[nBaseLUTNodeIdx];
            if(!oBaseNode.bValidGraphNode)
                continue;
            const size_t nRowIdx = (size_t)oBaseNode.nRowIdx;
            const size_t nColIdx = (size_t)oBaseNode.nColIdx;
            aGraphNodeIndices[0] = nGraphNodeIdx;
            if(nRowIdx+1<nRows) { // vertical pair
                const size_t nOffsetLUTNodeIdx = (nRowIdx+1)*nCols+nColIdx;
                const NodeInfo& oOffsetNode = m_vNodeInfos[nOffsetLUTNodeIdx];
                if(oOffsetNode.bValidGraphNode) {
                    aGraphNodeIndices[1] = oOffsetNode.nGraphNodeIdx;
                    m_avStereoPairwFuncs[0].push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
                    m_avResegmPairwFuncs[0].push_back(m_pResegmModel->addFunctionWithRefReturn(ExplicitFunction()));
                    StereoFunc& oStereoFunc = m_avStereoPairwFuncs[0].back();
                    ResegmFunc& oResegmFunc = m_avResegmPairwFuncs[0].back();
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.resize(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end());
                    oResegmFunc.second.resize(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end());
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.assign(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end(),m_pStereoPairwFuncsDataBase+((aGraphNodeIndices[0]*2)*m_nStereoLabels*m_nStereoLabels));
                    oResegmFunc.second.assign(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end(),m_pResegmPairwFuncsDataBase+((aGraphNodeIndices[0]*2)*s_nResegmLabels*s_nResegmLabels));
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==m_nStereoLabels && oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==s_nResegmLabels); // expect last-idx-major
                    oBaseNode.anStereoPairwFactIDs[0] = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.anResegmPairwFactIDs[0] = m_pResegmModel->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.apStereoPairwFuncs[0] = &oStereoFunc;
                    oBaseNode.apResegmPairwFuncs[0] = &oResegmFunc;
                    oBaseNode.anPairwLUTNodeIdxs[0] = nOffsetLUTNodeIdx;
                    oBaseNode.anPairwGraphNodeIdxs[0] = aGraphNodeIndices[1];
                }
            }
            if(nColIdx+1<nCols) { // horizontal pair
                const size_t nOffsetLUTNodeIdx = nRowIdx*nCols+nColIdx+1;
                const NodeInfo& oOffsetNode = m_vNodeInfos[nOffsetLUTNodeIdx];
                if(oOffsetNode.bValidGraphNode) {
                    aGraphNodeIndices[1] = oOffsetNode.nGraphNodeIdx;
                    m_avStereoPairwFuncs[1].push_back(m_pStereoModel->addFunctionWithRefReturn(ExplicitFunction()));
                    m_avResegmPairwFuncs[1].push_back(m_pResegmModel->addFunctionWithRefReturn(ExplicitFunction()));
                    StereoFunc& oStereoFunc = m_avStereoPairwFuncs[1].back();
                    ResegmFunc& oResegmFunc = m_avResegmPairwFuncs[1].back();
#if STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.resize(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end());
                    oResegmFunc.second.resize(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end());
#else //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    oStereoFunc.second.assign(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end(),m_pStereoPairwFuncsDataBase+((aGraphNodeIndices[0]*2+1)*m_nStereoLabels*m_nStereoLabels));
                    oResegmFunc.second.assign(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end(),m_pResegmPairwFuncsDataBase+((aGraphNodeIndices[0]*2+1)*s_nResegmLabels*s_nResegmLabels));
#endif //!STEREOSEGMATCH_CONFIG_ALLOC_IN_MODEL
                    lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==m_nStereoLabels && oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==s_nResegmLabels); // expect last-idx-major
                    oBaseNode.anStereoPairwFactIDs[1] = m_pStereoModel->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.anResegmPairwFactIDs[1] = m_pResegmModel->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.apStereoPairwFuncs[1] = &oStereoFunc;
                    oBaseNode.apResegmPairwFuncs[1] = &oResegmFunc;
                    oBaseNode.anPairwLUTNodeIdxs[1] = nOffsetLUTNodeIdx;
                    oBaseNode.anPairwGraphNodeIdxs[1] = aGraphNodeIndices[1];
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

inline void StereoSegmMatcher::GraphModelData::resetStereoLabelings() {
    /*if(@@@@ INIT ONLY)*/ {
        m_oStereoLabeling.create(m_oGridSize);
        std::fill(m_oStereoLabeling.begin(),m_oStereoLabeling.end(),m_nDontCareLabelIdx);
        lvDbgAssert(m_nValidGraphNodes==m_vValidLUTNodeIdxs.size());
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
            const NodeInfo& oNode = m_vNodeInfos[m_vValidLUTNodeIdxs[nGraphNodeIdx]];
            lvDbgAssert(oNode.bValidGraphNode);
            lvDbgAssert(oNode.nGraphNodeIdx==nGraphNodeIdx);
            lvDbgAssert(oNode.nStereoUnaryFactID<m_pStereoModel->numberOfFactors());
            lvDbgAssert(m_pStereoModel->numberOfLabels(oNode.nStereoUnaryFactID)==m_nStereoLabels);
            lvDbgAssert(StereoFuncID((*m_pStereoModel)[oNode.nStereoUnaryFactID].functionIndex(),(*m_pStereoModel)[oNode.nStereoUnaryFactID].functionType())==oNode.pStereoUnaryFunc->first);
            InternalLabelType nEvalLabel = m_oStereoLabeling(oNode.nRowIdx,oNode.nColIdx) = 0;
            ValueType fOptimalEnergy = oNode.pStereoUnaryFunc->second(&nEvalLabel);
            for(nEvalLabel = 1; nEvalLabel<m_nStereoLabels; ++nEvalLabel) {
                const ValueType fCurrEnergy = oNode.pStereoUnaryFunc->second(&nEvalLabel);
                if(fOptimalEnergy>fCurrEnergy) {
                    m_oStereoLabeling(oNode.nRowIdx,oNode.nColIdx) = nEvalLabel;
                    fOptimalEnergy = fCurrEnergy;
                }
            }
        }
    }
    m_oAssocCounts = (AssocCountType)0;
    m_oAssocMap = (AssocIdxType)-1;
    std::vector<int> vLabelCounts(m_nStereoLabels,0);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
        const InternalLabelType nLabel = ((InternalLabelType*)m_oStereoLabeling.data)[nLUTNodeIdx];
        if(nLabel<m_nDontCareLabelIdx) // both special labels avoided here
            addAssoc(oNode.nRowIdx,oNode.nColIdx,nLabel);
        ++vLabelCounts[nLabel];
    }
    m_vStereoLabelOrdering = lv::sort_indices<InternalLabelType>(vLabelCounts,[&vLabelCounts](int a, int b){return vLabelCounts[a]>vLabelCounts[b];});
    lvDbgAssert(lv::unique(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end())==lv::make_range(InternalLabelType(0),InternalLabelType(m_nStereoLabels-1)));
}

inline void StereoSegmMatcher::GraphModelData::updateStereoModel(bool bInit) {
    lvDbgAssert(m_pStereoModel && m_pStereoModel->numberOfVariables()==m_nValidGraphNodes);
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    /*const int nMinDescColIdx = (int)m_nGridBorderSize;
    const int nMinDescRowIdx = (int)m_nGridBorderSize;
    const int nMaxDescColIdx = nCols-(int)m_nGridBorderSize-1;
    const int nMaxDescRowIdx = nRows-(int)m_nGridBorderSize-1;
    const auto lHasValidDesc = [&nMinDescRowIdx,&nMaxDescRowIdx,&nMinDescColIdx,&nMaxDescColIdx](int nRowIdx, int nColIdx) {
        return nRowIdx>=nMinDescRowIdx && nRowIdx<=nMaxDescRowIdx && nColIdx>=nMinDescColIdx && nColIdx<=nMaxDescColIdx;
    };*/
    const cv::Mat_<float> oImgAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    const cv::Mat_<float> oShpAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    lvDbgAssert(oImgAffinity.dims==3 && oImgAffinity.size[0]==nRows && oImgAffinity.size[1]==nCols && oImgAffinity.size[2]==(int)m_nRealStereoLabels);
    lvDbgAssert(oShpAffinity.dims==3 && oShpAffinity.size[0]==nRows && oShpAffinity.size[1]==nCols && oShpAffinity.size[2]==(int)m_nRealStereoLabels);
    const std::array<cv::Mat_<float>,2> aInitFGDist = {m_vNextFeats[FeatPack_LeftInitFGDist],m_vNextFeats[FeatPack_RightInitFGDist]};
    const std::array<cv::Mat_<float>,2> aInitBGDist = {m_vNextFeats[FeatPack_LeftInitBGDist],m_vNextFeats[FeatPack_RightInitBGDist]};
    lvDbgAssert(lv::MatInfo(aInitFGDist[0])==lv::MatInfo(aInitFGDist[1]) && m_oGridSize==aInitFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aInitBGDist[0])==lv::MatInfo(aInitBGDist[1]) && m_oGridSize==aInitBGDist[0].size);
    const std::array<cv::Mat_<float>,2> aFGDist = {m_vNextFeats[FeatPack_LeftFGDist],m_vNextFeats[FeatPack_RightFGDist]};
    const std::array<cv::Mat_<float>,2> aBGDist = {m_vNextFeats[FeatPack_LeftBGDist],m_vNextFeats[FeatPack_RightBGDist]};
    lvDbgAssert(lv::MatInfo(aFGDist[0])==lv::MatInfo(aFGDist[1]) && m_oGridSize==aFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aBGDist[0])==lv::MatInfo(aBGDist[1]) && m_oGridSize==aBGDist[0].size);
    const std::array<cv::Mat_<uchar>,2> aGradY = {m_vNextFeats[FeatPack_LeftGradY],m_vNextFeats[FeatPack_RightGradY]};
    const std::array<cv::Mat_<uchar>,2> aGradX = {m_vNextFeats[FeatPack_LeftGradX],m_vNextFeats[FeatPack_RightGradX]};
    const std::array<cv::Mat_<uchar>,2> aGradMag = {m_vNextFeats[FeatPack_LeftGradMag],m_vNextFeats[FeatPack_RightGradMag]};
    lvDbgAssert(lv::MatInfo(aGradY[0])==lv::MatInfo(aGradY[1]) && m_oGridSize==aGradY[0].size);
    lvDbgAssert(lv::MatInfo(aGradX[0])==lv::MatInfo(aGradX[1]) && m_oGridSize==aGradX[0].size);
    lvDbgAssert(lv::MatInfo(aGradMag[0])==lv::MatInfo(aGradMag[1]) && m_oGridSize==aGradMag[0].size);
    /*cv::imshow("aGradY_0",aGradY[0]>32);
    cv::imshow("aGradX_0",aGradX[0]>32);
    cv::imshow("aGradY_1",aGradY[1]>32);
    cv::imshow("aGradX_1",aGradX[1]>32);
    cv::waitKey(0);*/
    lvLog(1,"Updating stereo graph model energy terms based on new features...");
    lv::StopWatch oLocalTimer;
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;
    lvDbgAssert(m_nValidGraphNodes==m_vValidLUTNodeIdxs.size());
#if USING_OPENMP
//#pragma omp parallel for
#endif //USING_OPENMP
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        // update unary terms for each grid node
        lvDbgAssert(oNode.pStereoUnaryFunc && oNode.nStereoUnaryFactID!=SIZE_MAX);
        lvDbgAssert((&m_pStereoModel->getFunction<ExplicitFunction>(oNode.pStereoUnaryFunc->first))==(&oNode.pStereoUnaryFunc->second));
        ExplicitFunction& vUnaryStereoFunc = oNode.pStereoUnaryFunc->second;
        lvDbgAssert(vUnaryStereoFunc.dimension()==1 && vUnaryStereoFunc.size()==m_nStereoLabels);
        const float* pImgAffinityPtr = oImgAffinity.ptr<float>(nRowIdx,nColIdx);
        //const float* pShpAffinityPtr = oShpAffinity.ptr<float>(nRowIdx,nColIdx);
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
            const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
            const int nOffsetColIdx = nColIdx-nRealLabel;
            //const bool bValidDescOffsetNode = lHasValidDesc(nRowIdx,nOffsetColIdx); // @@@ update to use roi?
            vUnaryStereoFunc(nLabelIdx) = ValueType(0);
            const float& fImgAffinity = pImgAffinityPtr[nLabelIdx];
            //const float& fShpAffinity = pShpAffinityPtr[nLabelIdx];
            if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[1](nRowIdx,nOffsetColIdx)) {
                lvDbgAssert(fImgAffinity>=0.0f /*&& fImgAffinity<=1.0f*/);
                //lvDbgAssert(fShpAffinity>=0.0f && fShpAffinity<=1.0f);
                vUnaryStereoFunc(nLabelIdx) += ValueType(fImgAffinity*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE);
                //vUnaryStereoFunc(nLabelIdx) += ValueType(fShpAffinity*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE);
                vUnaryStereoFunc(nLabelIdx) = std::min(vUnaryStereoFunc(nLabelIdx),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            }
            else {
                lvDbgAssert((nOffsetColIdx>=0 && nOffsetColIdx<nCols) || (fImgAffinity<0.0f/* && fShpAffinity<0.0f*/));
                vUnaryStereoFunc(nLabelIdx) = STEREOSEGMATCH_UNARY_COST_OOB_CST;
            }
        }
        // @@@@ add discriminativeness factor --- vect-append all sim vals, sort, setup discrim costs
        vUnaryStereoFunc(m_nDontCareLabelIdx) = ValueType(10000); // @@@@ check roi, if dc set to 0, otherwise set to inf
        vUnaryStereoFunc(m_nOccludedLabelIdx) = ValueType(10000);//STEREOSEGMATCH_IMGSIM_COST_OCCLUDED_CST;
        if(bInit) { // @@@@@ does not change w.r.t. segm or stereo updates
            // update pairwise terms for each graph node
            for(size_t nOrientIdx=0; nOrientIdx<oNode.anStereoPairwFactIDs.size(); ++nOrientIdx) {
                if(oNode.anStereoPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                    lvDbgAssert(oNode.apStereoPairwFuncs[nOrientIdx]);
                    lvDbgAssert(oNode.anPairwLUTNodeIdxs[nOrientIdx]!=SIZE_MAX);
                    lvDbgAssert(m_vNodeInfos[oNode.anPairwLUTNodeIdxs[nOrientIdx]].bValidGraphNode);
                    lvDbgAssert((&m_pStereoModel->getFunction<ExplicitFunction>(oNode.apStereoPairwFuncs[nOrientIdx]->first))==(&oNode.apStereoPairwFuncs[nOrientIdx]->second));
                    ExplicitFunction& vPairwiseStereoFunc = oNode.apStereoPairwFuncs[nOrientIdx]->second;
                    lvDbgAssert(vPairwiseStereoFunc.dimension()==2 && vPairwiseStereoFunc.size()==m_nStereoLabels*m_nStereoLabels);
                    const int nLocalGrad = (int)((nOrientIdx==0)?aGradY[0]:(nOrientIdx==1)?aGradX[0]:aGradMag[0])(nRowIdx,nColIdx);
                    const float fGradScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    lvDbgAssert(fGradScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nRealStereoLabels; ++nLabelIdx1) {
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nRealStereoLabels; ++nLabelIdx2) {
                            const OutputLabelType nRealLabel1 = getRealLabel(nLabelIdx1);
                            const OutputLabelType nRealLabel2 = getRealLabel(nLabelIdx2);
                            const int nRealLabelDiff = std::min(std::abs((int)nRealLabel1-(int)nRealLabel2),STEREOSEGMATCH_LBLSIM_COST_MAXDIFF_CST);
                            vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) = ValueType((nRealLabelDiff*nRealLabelDiff)*fGradScaleFact*STEREOSEGMATCH_LBLSIM_COST_SCALE_CST);
                            /*const bool bValidDescOffsetNode1 = lHasValidDesc(nRowIdx,nColIdx-(int)nRealLabel1);
                            const bool bValidDescOffsetNode2 = lHasValidDesc(nRowIdx,nColIdx-(int)nRealLabel2);
                            if(!bValidDescOffsetNode1 || !bValidDescOffsetNode2)
                                vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) *= 2; // incr smoothness cost for node pairs in border regions to outweight bad unaries
                                // @@@@@@@ AUTO-MODULATE USING DISCRIM FACTOR INSTEAD
                                */
                            vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) = std::min(vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2),STEREOSEGMATCH_LBLSIM_COST_MAXTRUNC_CST);
                        }
                    }
                    for(size_t nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                        // @@@ change later for img-data-dependent or roi-dependent energies?
                        // @@@@ outside-roi-dc to inside-roi-something is ok (0 cost)
                        vPairwiseStereoFunc(m_nDontCareLabelIdx,nLabelIdx) = ValueType(10000);
                        vPairwiseStereoFunc(m_nOccludedLabelIdx,nLabelIdx) = ValueType(10000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                        vPairwiseStereoFunc(nLabelIdx,m_nDontCareLabelIdx) = ValueType(10000);
                        vPairwiseStereoFunc(nLabelIdx,m_nOccludedLabelIdx) = ValueType(10000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                    }
                    vPairwiseStereoFunc(m_nDontCareLabelIdx,m_nDontCareLabelIdx) = ValueType(0);
                    vPairwiseStereoFunc(m_nOccludedLabelIdx,m_nOccludedLabelIdx) = ValueType(0);
                }
            }
            /*const size_t nCurrNodeIdx =*/ ++nProcessedNodeCount;
            /*if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0 && oLocalTimer.elapsed()>2.0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
                bProgressDisplayed = true;
            }*/
        }
    }
    if(bProgressDisplayed)
        lv::cleanConsoleRow();
    lvLog_(1,"Stereo graph energy terms update completed in %f second(s).",oLocalTimer.tock());
}

inline void StereoSegmMatcher::GraphModelData::updateResegmModel(bool bInit) {
    lvDbgAssert(m_pResegmModel && m_pResegmModel->numberOfVariables()==m_nValidGraphNodes);
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    const cv::Mat_<float> oShpAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    lvDbgAssert(oShpAffinity.dims==3 && oShpAffinity.size[0]==nRows && oShpAffinity.size[1]==nCols && oShpAffinity.size[2]==(int)m_nRealStereoLabels);
    const std::array<cv::Mat_<float>,2> aInitFGDist = {m_vNextFeats[FeatPack_LeftInitFGDist],m_vNextFeats[FeatPack_RightInitFGDist]};
    const std::array<cv::Mat_<float>,2> aInitBGDist = {m_vNextFeats[FeatPack_LeftInitBGDist],m_vNextFeats[FeatPack_RightInitBGDist]};
    lvDbgAssert(lv::MatInfo(aInitFGDist[0])==lv::MatInfo(aInitFGDist[1]) && m_oGridSize==aInitFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aInitBGDist[0])==lv::MatInfo(aInitBGDist[1]) && m_oGridSize==aInitBGDist[0].size);
    const std::array<cv::Mat_<float>,2> aFGDist = {m_vNextFeats[FeatPack_LeftFGDist],m_vNextFeats[FeatPack_RightFGDist]};
    const std::array<cv::Mat_<float>,2> aBGDist = {m_vNextFeats[FeatPack_LeftBGDist],m_vNextFeats[FeatPack_RightBGDist]};
    lvDbgAssert(lv::MatInfo(aFGDist[0])==lv::MatInfo(aFGDist[1]) && m_oGridSize==aFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aBGDist[0])==lv::MatInfo(aBGDist[1]) && m_oGridSize==aBGDist[0].size);
    const std::array<cv::Mat_<uchar>,2> aGradY = {m_vNextFeats[FeatPack_LeftGradY],m_vNextFeats[FeatPack_RightGradY]};
    const std::array<cv::Mat_<uchar>,2> aGradX = {m_vNextFeats[FeatPack_LeftGradX],m_vNextFeats[FeatPack_RightGradX]};
    const std::array<cv::Mat_<uchar>,2> aGradMag = {m_vNextFeats[FeatPack_LeftGradMag],m_vNextFeats[FeatPack_RightGradMag]};
    lvDbgAssert(lv::MatInfo(aGradY[0])==lv::MatInfo(aGradY[1]) && m_oGridSize==aGradY[0].size);
    lvDbgAssert(lv::MatInfo(aGradX[0])==lv::MatInfo(aGradX[1]) && m_oGridSize==aGradX[0].size);
    lvDbgAssert(lv::MatInfo(aGradMag[0])==lv::MatInfo(aGradMag[1]) && m_oGridSize==aGradMag[0].size);
    lvLog(1,"Updating resegm graph model energy terms based on new features...");
    lv::StopWatch oLocalTimer;
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;
    lvDbgAssert(m_nValidGraphNodes==m_vValidLUTNodeIdxs.size());
#if USING_OPENMP
#pragma omp parallel for
#endif //USING_OPENMP
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        // update unary terms for each grid node
        lvDbgAssert(oNode.pResegmUnaryFunc && oNode.nResegmUnaryFactID!=SIZE_MAX); // @@@@ will no longer be the case w/ roi check
        lvDbgAssert((&m_pResegmModel->getFunction<ExplicitFunction>(oNode.pResegmUnaryFunc->first))==(&oNode.pResegmUnaryFunc->second));
        ExplicitFunction& vUnaryResegmFunc = oNode.pResegmUnaryFunc->second;
        lvDbgAssert(vUnaryResegmFunc.dimension()==1 && vUnaryResegmFunc.size()==s_nResegmLabels);
        const float fInitFGDist = ((float*)aInitFGDist[0].data)[nLUTNodeIdx];
        const float fCurrFGDist = ((float*)aFGDist[0].data)[nLUTNodeIdx];
        vUnaryResegmFunc(s_nForegroundLabelIdx) = std::min(ValueType((fCurrFGDist+fInitFGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
        if(vUnaryResegmFunc(s_nForegroundLabelIdx)<0) {
            lvPrint(nRowIdx);
            lvPrint(nColIdx);
            lvPrint(fInitFGDist);
            lvPrint(fCurrFGDist);
            lvPrint(ValueType((fCurrFGDist+fInitFGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE));
            lvPrint(vUnaryResegmFunc(s_nForegroundLabelIdx));
            lvDbgAssert(vUnaryResegmFunc(s_nForegroundLabelIdx)>=ValueType(0));
        }
        const float fInitBGDist = ((float*)aInitBGDist[0].data)[nLUTNodeIdx];
        const float fCurrBGDist = ((float*)aBGDist[0].data)[nLUTNodeIdx];
        vUnaryResegmFunc(s_nBackgroundLabelIdx) = std::min(ValueType((fCurrBGDist+fInitBGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
        if(vUnaryResegmFunc(s_nBackgroundLabelIdx)<0) {
            lvPrint(nRowIdx);
            lvPrint(nColIdx);
            lvPrint(fInitBGDist);
            lvPrint(fCurrBGDist);
            lvPrint(ValueType((fCurrBGDist+fInitBGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE));
            lvDbgAssert(vUnaryResegmFunc(s_nBackgroundLabelIdx)>=ValueType(0));
        }
        const InternalLabelType nStereoLabelIdx = ((InternalLabelType*)m_oStereoLabeling.data)[nLUTNodeIdx];
        if(nStereoLabelIdx<m_nRealStereoLabels) {
            const OutputLabelType nRealStereoLabel = getRealLabel(nStereoLabelIdx);
            const int nOffsetColIdx = nColIdx-nRealStereoLabel;
            if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) {
                const float fInitOffsetFGDist = ((float*)aInitFGDist[1].data)[nLUTNodeIdx-nRealStereoLabel];
                const float fCurrOffsetFGDist = ((float*)aFGDist[1].data)[nLUTNodeIdx-nRealStereoLabel];
                //vUnaryResegmFunc(s_nForegroundLabelIdx) = std::min(ValueType((fCurrOffsetFGDist+fInitOffsetFGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE),vUnaryResegmFunc(s_nForegroundLabelIdx));
                vUnaryResegmFunc(s_nForegroundLabelIdx) += ValueType((fCurrOffsetFGDist+fInitOffsetFGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE/2);
                vUnaryResegmFunc(s_nForegroundLabelIdx) = std::min(vUnaryResegmFunc(s_nForegroundLabelIdx),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
                const float fInitOffsetBGDist = ((float*)aInitBGDist[1].data)[nLUTNodeIdx-nRealStereoLabel];
                const float fCurrOffsetBGDist = ((float*)aBGDist[1].data)[nLUTNodeIdx-nRealStereoLabel];
                //vUnaryResegmFunc(s_nBackgroundLabelIdx) = std::min(ValueType((fCurrOffsetBGDist+fInitOffsetBGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE),vUnaryResegmFunc(s_nBackgroundLabelIdx));
                vUnaryResegmFunc(s_nBackgroundLabelIdx) += ValueType((fCurrOffsetBGDist+fInitOffsetBGDist)*STEREOSEGMATCH_SHPDIST_COST_SCALE/2);
                vUnaryResegmFunc(s_nBackgroundLabelIdx) = std::min(vUnaryResegmFunc(s_nBackgroundLabelIdx),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            }
        }
        if(bInit) { // @@@@@ does not change w.r.t. segm or stereo updates
            // update pairwise terms for each grid node
            for(size_t nOrientIdx=0; nOrientIdx<oNode.anResegmPairwFactIDs.size(); ++nOrientIdx) {
                if(oNode.anResegmPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                    lvDbgAssert(oNode.apResegmPairwFuncs[nOrientIdx]);
                    lvDbgAssert(oNode.anPairwLUTNodeIdxs[nOrientIdx]!=SIZE_MAX);
                    lvDbgAssert(m_vNodeInfos[oNode.anPairwLUTNodeIdxs[nOrientIdx]].bValidGraphNode);
                    lvDbgAssert((&m_pResegmModel->getFunction<ExplicitFunction>(oNode.apResegmPairwFuncs[nOrientIdx]->first))==(&oNode.apResegmPairwFuncs[nOrientIdx]->second));
                    ExplicitFunction& vPairwiseResegmFunc = oNode.apResegmPairwFuncs[nOrientIdx]->second;
                    lvDbgAssert(vPairwiseResegmFunc.dimension()==2 && vPairwiseResegmFunc.size()==s_nResegmLabels*s_nResegmLabels);
                    const int nLocalGrad = (int)((nOrientIdx==0)?aGradY[0]:(nOrientIdx==1)?aGradX[0]:aGradMag[0])(nRowIdx,nColIdx);
                    const float fGradScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    lvDbgAssert(fGradScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<s_nResegmLabels; ++nLabelIdx1)
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<s_nResegmLabels; ++nLabelIdx2)
                            vPairwiseResegmFunc(nLabelIdx1,nLabelIdx2) = std::min(ValueType((nLabelIdx1^nLabelIdx2)*fGradScaleFact),STEREOSEGMATCH_LBLSIM_COST_MAXTRUNC_CST);
                    // @@@@@@@@ scale pairw cost here? (label dist too small)
                }
            }
        }
        /*const size_t nCurrNodeIdx =*/ ++nProcessedNodeCount;
        /*if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0 && oLocalTimer.elapsed()>2.0) {
            lv::mutex_lock_guard oLock(oPrintMutex);
            lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
            bProgressDisplayed = true;
        }*/
    }
    if(bProgressDisplayed)
        lv::cleanConsoleRow();
    lvLog_(1,"Resegm graph energy terms update completed in %f second(s).",oLocalTimer.tock());
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
    calcImageFeatures(std::array<cv::Mat,InputPackSize/InputPackOffset>{aInputs[InputPack_LeftImg],aInputs[InputPack_RightImg]},true);
    calcShapeFeatures(std::array<cv::Mat,InputPackSize/InputPackOffset>{aInputs[InputPack_LeftMask],aInputs[InputPack_RightMask]},true);
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
}

inline void StereoSegmMatcher::GraphModelData::calcImageFeatures(const std::array<cv::Mat,InputPackSize/InputPackOffset>& aInputImages, bool /*bInit*/) {
    static_assert(getCameraCount()==size_t(2),"bad input image array size");
    for(size_t nInputIdx=0; nInputIdx<aInputImages.size(); ++nInputIdx) {
        lvDbgAssert__(aInputImages[nInputIdx].dims==2 && m_oGridSize==aInputImages[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputImages[nInputIdx].type()==CV_8UC1 || aInputImages[nInputIdx].type()==CV_8UC3,"unexpected input image type");
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvLog(1,"Calculating image features maps...");
    const int nWinRadius = (int)m_nGridBorderSize;
    std::array<cv::Mat,InputPackSize/InputPackOffset> aEnlargedInput;
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    std::array<cv::Mat_<float>,InputPackSize/InputPackOffset> aInputDescs;
#endif //!STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::copyMakeBorder(aInputImages[nCamIdx],aEnlargedInput[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_DEFAULT);
    #if STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3)
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
    #elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3)
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
        aEnlargedInput[nCamIdx].convertTo(aEnlargedInput[nCamIdx],CV_64F,(1.0/UCHAR_MAX)/(nWinRadius*2+1));
        aEnlargedInput[nCamIdx] -= cv::mean(aEnlargedInput[nCamIdx])[0];
    #elif STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        lvLog_(2,"\tcam[%d] image descriptors...",(int)nCamIdx);
        m_pImgDescExtractor->compute2(aEnlargedInput[nCamIdx],aInputDescs[nCamIdx]);
        lvDbgAssert(aInputDescs[nCamIdx].dims==3 && aInputDescs[nCamIdx].size[0]==nRows+nWinRadius*2 && aInputDescs[nCamIdx].size[1]==nCols+nWinRadius*2);
        // test w/ root-sift transform here? @@@@@
    #endif //STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        lvLog_(2,"\tcam[%d] image gradient magnitudes...",(int)nCamIdx);
        cv::Mat oBlurredInput;
        cv::GaussianBlur(aInputImages[nCamIdx],oBlurredInput,cv::Size(3,3),0);
        cv::Mat oBlurredGrayInput;
        if(oBlurredInput.channels()==3)
            cv::cvtColor(oBlurredInput,oBlurredGrayInput,cv::COLOR_BGR2GRAY);
        else
            oBlurredGrayInput = oBlurredInput;
        cv::Mat oGradInput_X,oGradInput_Y;
        cv::Sobel(oBlurredGrayInput,oGradInput_Y,CV_16S,0,1,STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::Mat& oGradY = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradY];
        cv::normalize(cv::abs(oGradInput_Y),oGradY,255,0,cv::NORM_MINMAX,CV_8U);
        cv::Sobel(oBlurredGrayInput,oGradInput_X,CV_16S,1,0,STEREOSEGMATCH_DEFAULT_GRAD_KERNEL_SIZE);
        cv::Mat& oGradX = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradX];
        cv::normalize(cv::abs(oGradInput_X),oGradX,255,0,cv::NORM_MINMAX,CV_8U);
        cv::Mat& oGradMag = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_GradMag];
        cv::addWeighted(oGradY,0.5,oGradX,0.5,0,oGradMag);
        /*cv::imshow("gradm_full",oGradMag);
        cv::imshow("gradm_16",oGradMag>16);
        cv::imshow("gradm_32",oGradMag>32);
        cv::imshow("gradm_50",oGradMag>50);
        cv::imshow("gradm_100",oGradMag>100);
        cv::imshow("gradm_150",oGradMag>150);
        cv::waitKey(0);*/
    }
    lvLog_(1,"Image features maps computed in %f second(s).",oLocalTimer.tock());
    lvLog(1,"Calculating image affinity/discrim maps...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    cv::Mat& oAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    oAffinity.create(3,anAffinityMapDims.data(),CV_32FC1);
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;
#if USING_OPENMP
#pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nColIdx);
            for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                const int nOffsetColIdx = nColIdx-nRealLabel;
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    const int nRealRowIdx = nRowIdx+nWinRadius;
                    const int nRealColIdx = nColIdx+nWinRadius;
                    const int nRealOffsetColIdx = nOffsetColIdx+nWinRadius;
                #if STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
                    const cv::Rect oWindow(nRealColIdx-nWinRadius,nRealRowIdx-nWinRadius,nWinRadius*2+1,nWinRadius*2+1);
                    const cv::Rect oOffsetWindow(nRealOffsetColIdx-nWinRadius,nRealRowIdx-nWinRadius,nWinRadius*2+1,nWinRadius*2+1);
                    const double dMutualInfoScore = m_pImgDescExtractor->compute(aEnlargedInput[0](oWindow),aEnlargedInput[1](oOffsetWindow));
                    pAffinityPtr[nLabelIdx] = std::max(float(1.0-dMutualInfoScore),0.0f);
                #elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
                    const cv::Rect oWindow(nRealColIdx-nWinRadius,nRealRowIdx-nWinRadius,nWinRadius*2+1,nWinRadius*2+1);
                    const cv::Rect oOffsetWindow(nRealOffsetColIdx-nWinRadius,nRealRowIdx-nWinRadius,nWinRadius*2+1,nWinRadius*2+1);
                    pAffinityPtr[nLabelIdx] = (float)cv::norm(aEnlargedInput[0](oWindow),aEnlargedInput[1](oOffsetWindow),cv::NORM_L2);
                #elif STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
                    pAffinityPtr[nLabelIdx] = (float)m_pImgDescExtractor->calcDistance(aInputDescs[0].ptr<float>(nRealRowIdx,nRealColIdx),aInputDescs[1].ptr<float>(nRealRowIdx,nRealOffsetColIdx));
                #endif //!STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
                    if(!(pAffinityPtr[nLabelIdx]>=0.0f && pAffinityPtr[nLabelIdx]<=1.0f))
                        lvPrint(pAffinityPtr[nLabelIdx]);
                    lvDbgAssert(pAffinityPtr[nLabelIdx]>=0.0f /*&& pAffinityPtr[nLabelIdx]<=1.0f*/);
                }
                else
                    pAffinityPtr[nLabelIdx] = -1.0f; // OOB
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
    cv::Mat& oDiscrimPow = m_vNextFeats[FeatPack_ImgDiscrimPow];
    oDiscrimPow.create(3,anAffinityMapDims.data(),CV_32FC1);
#if USING_OPENMP
#pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float* pDiscrimPowPtr = oDiscrimPow.ptr<float>(nRowIdx,nColIdx);
            std::fill_n(pDiscrimPowPtr,m_nRealStereoLabels,0.0f);
            // @@@@@@@@@@@@@@@@@@@ TODO
            // GIVE LOW DISCRIM TO AFFINITY NEAR BORDERS
        }
    }
    lvLog_(1,"Image affinity/discrim maps computed in %f second(s).",oLocalTimer.tock());
    /*for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
        cv::Mat_<float> tmp = lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx));
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

inline void StereoSegmMatcher::GraphModelData::calcShapeFeatures(const std::array<cv::Mat,InputPackSize/InputPackOffset>& aInputMasks, bool bInit) {
    static_assert(getCameraCount()==size_t(2),"bad input mask array size");
    for(size_t nInputIdx=0; nInputIdx<aInputMasks.size(); ++nInputIdx) {
        lvDbgAssert__(aInputMasks[nInputIdx].dims==2 && m_oGridSize==aInputMasks[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputMasks[nInputIdx].type()==CV_8UC1,"unexpected input mask type");
    }
    lvLog(1,"Calculating shape features maps...");
    std::array<cv::Mat_<float>,InputPackSize/InputPackOffset> aInputDescs;
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {

        // @@@@@ use swap mask to prevent recalc of identical descriptors? (swap + radius based on desc)

        const cv::Mat& oInputMask = aInputMasks[nCamIdx];
        lvLog_(2,"\tcam[%d] shape descriptors...",(int)nCamIdx);
        m_pShpDescExtractor->compute2(oInputMask,aInputDescs[nCamIdx]);
        lvLog_(2,"\tcam[%d] shape distance fields...",(int)nCamIdx);
        cv::Mat& oFGDist = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist];
        cv::distanceTransform(oInputMask==0,oFGDist,cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
        cv::exp(STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE*oFGDist,oFGDist);
        //lvPrint(cv::Mat_<float>(oFGDist(cv::Rect(0,128,256,1))));
        cv::divide(1.0,oFGDist,oFGDist);
        oFGDist -= 1.0f;
        cv::min(oFGDist,1000.0f,oFGDist);
        //lvPrint(cv::Mat_<float>(oFGDist(cv::Rect(0,128,256,1))));
        cv::Mat& oBGDist = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist];
        cv::distanceTransform(oInputMask>0,oBGDist,cv::DIST_L2,cv::DIST_MASK_PRECISE,CV_32F);
        cv::exp(STEREOSEGMATCH_DEFAULT_DISTTRANSF_SCALE*oBGDist,oBGDist);
        //lvPrint(cv::Mat_<float>(oBGSim(cv::Rect(0,128,256,1))));
        cv::divide(1.0,oBGDist,oBGDist);
        oBGDist -= 1.0f;
        cv::min(oBGDist,1000.0f,oBGDist);
        //lvPrint(cv::Mat_<float>(oBGDist(cv::Rect(0,128,256,1))));
        if(bInit) {
            oFGDist.copyTo(m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_InitFGDist]);
            oBGDist.copyTo(m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_InitBGDist]);
        }
    }
    lvLog_(1,"Shape features maps computed in %f second(s).",oLocalTimer.tock());
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvDbgAssert(lv::MatInfo(aInputDescs[0])==lv::MatInfo(aInputDescs[1]) && aInputDescs[0].dims==3);
    lvDbgAssert(aInputDescs[0].size[0]==nRows && aInputDescs[0].size[1]==nCols);
    lvLog(1,"Calculating shape affinity/discrim maps...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    cv::Mat& oAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    oAffinity.create(3,anAffinityMapDims.data(),CV_32FC1);
    /*std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;*/
#if USING_OPENMP
#pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {

            // @@@@@ use swap mask to prevent recalc of identical descriptors? (swap + radius based on desc)

            float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nColIdx);
            for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                const OutputLabelType nRealLabel = getRealLabel(nLabelIdx);
                const int nOffsetColIdx = nColIdx-nRealLabel;
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) { // @@@ update to use roi?
                    // test w/ root-sift transform here? @@@@@
#if STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_SIM
                    // @@@@ add scale factor?
                    pAffinityPtr[nLabelIdx] = (float)m_pShpDescExtractor->calcDistance(aInputDescs[0].ptr<float>(nRowIdx,nColIdx),aInputDescs[1].ptr<float>(nRowIdx,nOffsetColIdx));
#else //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_SIM
                    const cv::Mat_<float> oDesc1(1,aInputDescs[0].size[2],aInputDescs[0].ptr<float>(nRowIdx,nColIdx));
                    const cv::Mat_<float> oDesc2(1,aInputDescs[0].size[2],aInputDescs[1].ptr<float>(nRowIdx,nOffsetColIdx));
                    pAffinityPtr[nLabelIdx] = (float)cv::norm(oDesc1,oDesc2,cv::NORM_L2);
#endif //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_SIM
                }
                else
                    pAffinityPtr[nLabelIdx] = -1.0f; // OOB
            }
            /*const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(size_t(nRows*nCols)/20))==0 && oLocalTimer.elapsed()>2.0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/size_t(nRows*nCols));
                bProgressDisplayed = true;
            }*/
        }
    }
    cv::Mat& oDiscrimPow = m_vNextFeats[FeatPack_ShpDiscrimPow];
    oDiscrimPow.create(3,anAffinityMapDims.data(),CV_32FC1);
#if USING_OPENMP
#pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float* pDiscrimPowPtr = oDiscrimPow.ptr<float>(nRowIdx,nColIdx);
            std::fill_n(pDiscrimPowPtr,m_nRealStereoLabels,0.0f);
            // @@@@@@@@@@@@@@@@@@@ TODO
            // GIVE LOW DISCRIM TO AFFINITY NEAR BORDERS
        }
    }
    lvLog_(1,"Shape affinity/discrim maps computed in %f second(s).",oLocalTimer.tock());
    /*for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
        cv::Mat_<float> tmp = lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx));
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
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_InitFGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_InitBGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradY] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradX] = lv::MatInfo(m_oGridSize,CV_8UC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_GradMag] = lv::MatInfo(m_oGridSize,CV_8UC1);
        }
        const int nRows = (int)m_oGridSize(0);
        const int nCols = (int)m_oGridSize(1);
        m_vExpectedFeatPackInfo[FeatPack_ImgAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
        m_vExpectedFeatPackInfo[FeatPack_ShpAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
        m_vExpectedFeatPackInfo[FeatPack_ImgDiscrimPow] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
        m_vExpectedFeatPackInfo[FeatPack_ShpDiscrimPow] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
    }
    m_oNextPackedFeats = oPackedFeats; // get pointer to data instead of copy; provider should not overwrite until next 'apply' call!
    m_vNextFeats = lv::unpackData(m_oNextPackedFeats,m_vExpectedFeatPackInfo);
    m_bUsePrecalcFeatsNext = true;
}

inline StereoSegmMatcher::OutputLabelType StereoSegmMatcher::GraphModelData::getRealLabel(InternalLabelType nLabel) const {
    lvDbgAssert(nLabel<m_vStereoLabels.size());
    return m_vStereoLabels[nLabel];
}

inline StereoSegmMatcher::InternalLabelType StereoSegmMatcher::GraphModelData::getInternalLabel(OutputLabelType nRealLabel) const {
    lvDbgAssert(nRealLabel==s_nOccludedLabel || nRealLabel==s_nDontCareLabel);
    lvDbgAssert(nRealLabel>=(OutputLabelType)m_nMinDispOffset && nRealLabel<=(OutputLabelType)m_nMaxDispOffset);
    lvDbgAssert(((nRealLabel-m_nMinDispOffset)%m_nDispOffsetStep)==0);
    return (InternalLabelType)std::distance(m_vStereoLabels.begin(),std::find(m_vStereoLabels.begin(),m_vStereoLabels.end(),nRealLabel));
}

inline StereoSegmMatcher::AssocCountType StereoSegmMatcher::GraphModelData::getAssocCount(int nRowIdx, int nColIdx) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(nColIdx>=-(int)m_nMaxDispOffset && nColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    return ((AssocCountType*)m_oAssocCounts.data)[nRowIdx*m_oAssocCounts.cols + (nColIdx+m_nMaxDispOffset)/m_nDispOffsetStep];
}

inline void StereoSegmMatcher::GraphModelData::addAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    lvDbgAssert(nLabel<m_nDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
    lvDbgAssert(nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset));
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_oAssocMap.data)+(nRowIdx*m_oAssocMap.size[1] + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)*m_oAssocMap.size[2];
    lvDbgAssert(pAssocList==m_oAssocMap.ptr<AssocIdxType>(nRowIdx,(nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_oAssocMap.size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(-1));
    pAssocList[nListOffset] = AssocIdxType(nColIdx);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[nRowIdx*m_oAssocCounts.cols + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(nRowIdx,int((nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount<std::numeric_limits<AssocCountType>::max());
    ++nAssocCount;
}

inline void StereoSegmMatcher::GraphModelData::removeAssoc(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    lvDbgAssert(nLabel<m_nDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
    lvDbgAssert(nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset));
    lvDbgAssert(m_oAssocMap.dims==3 && !m_oAssocMap.empty() && m_oAssocMap.isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_oAssocMap.data)+(nRowIdx*m_oAssocMap.size[1] + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)*m_oAssocMap.size[2];
    lvDbgAssert(pAssocList==m_oAssocMap.ptr<AssocIdxType>(nRowIdx,(nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_oAssocMap.size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(nColIdx));
    pAssocList[nListOffset] = AssocIdxType(-1);
    lvDbgAssert(m_oAssocCounts.dims==2 && !m_oAssocCounts.empty() && m_oAssocCounts.isContinuous());
    AssocCountType& nAssocCount = ((AssocCountType*)m_oAssocCounts.data)[nRowIdx*m_oAssocCounts.cols + (nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_oAssocCounts(nRowIdx,int((nAssocColIdx+m_nMaxDispOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount>AssocCountType(0));
    --nAssocCount;
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcAddAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nDontCareLabelIdx) {
        const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
        lvDbgAssert(nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset));
        const AssocCountType nAssocCount = getAssocCount(nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount<m_aAssocCostApproxAddLUT.size());
        return m_aAssocCostApproxAddLUT[nAssocCount];
    }
    return ValueType(100000); // @@@@ dirty
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcRemoveAssocCost(int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nDontCareLabelIdx) {
        const int nAssocColIdx = nColIdx-getRealLabel(nLabel);
        lvDbgAssert(nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset));
        const AssocCountType nAssocCount = getAssocCount(nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount>0); // cannot be zero, must have at least an association in order to remove it
        lvDbgAssert(nAssocCount<m_aAssocCostApproxAddLUT.size());
        return m_aAssocCostApproxRemLUT[nAssocCount];
    }
    return -ValueType(100000); // @@@@ dirty
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcTotalAssocCost() const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
    ValueType tEnergy = ValueType(0);
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx)
        for(int nColIdx=-(int)m_nMaxDispOffset; nColIdx<(int)(m_oGridSize[1]-m_nMinDispOffset); nColIdx+=m_nDispOffsetStep)
            tEnergy += m_aAssocCostRealSumLUT[getAssocCount(nRowIdx,nColIdx)];
    // @@@@ really needed?
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
        const InternalLabelType nCurrLabel = ((InternalLabelType*)m_oStereoLabeling.data)[m_vValidLUTNodeIdxs[nGraphNodeIdx]];
        if(nCurrLabel>=m_nDontCareLabelIdx) // both special labels treated here
            tEnergy += ValueType(100000); // @@@@ dirty
    }
    lvDbgAssert(tEnergy>=ValueType(0));
    return tEnergy;
}

inline void StereoSegmMatcher::GraphModelData::calcStereoMoveCosts(InternalLabelType nNewLabel) const {
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1);
    lvDbgAssert(m_oGridSize==m_oAssocCosts.size && m_oGridSize==m_oStereoUnaryCosts.size && m_oGridSize==m_oStereoPairwCosts.size);
    // @@@@@ openmp here?
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const InternalLabelType& nInitLabel = ((InternalLabelType*)m_oStereoLabeling.data)[nLUTNodeIdx];
        lvDbgAssert(&nInitLabel==&m_oStereoLabeling(oNode.nRowIdx,oNode.nColIdx));
        ValueType& tAssocCost = ((ValueType*)m_oAssocCosts.data)[nLUTNodeIdx];
        lvDbgAssert(&tAssocCost==&m_oAssocCosts(oNode.nRowIdx,oNode.nColIdx));
        ValueType& tUnaryCost = ((ValueType*)m_oStereoUnaryCosts.data)[nLUTNodeIdx];
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
        ValueType& tPairwCost = ((ValueType*)m_oStereoPairwCosts.data)[nLUTNodeIdx];
        lvDbgAssert(&tPairwCost==&m_oStereoPairwCosts(oNode.nRowIdx,oNode.nColIdx));
        tPairwCost = ValueType(0);
        for(size_t nOrientIdx=0; nOrientIdx<oNode.anStereoPairwFactIDs.size(); ++nOrientIdx) {
            if(oNode.anStereoPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                lvDbgAssert(oNode.apStereoPairwFuncs[nOrientIdx] && oNode.anPairwLUTNodeIdxs[nOrientIdx]<m_oGridSize.total());
                std::array<InternalLabelType,2> aLabels = {nInitLabel,((InternalLabelType*)m_oStereoLabeling.data)[oNode.anPairwLUTNodeIdxs[nOrientIdx]]};
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
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const InternalLabelType& nInitLabel = ((InternalLabelType*)m_oResegmLabeling.data)[nLUTNodeIdx];
        lvDbgAssert(&nInitLabel==&m_oResegmLabeling(oNode.nRowIdx,oNode.nColIdx));
        lvDbgAssert(nInitLabel==s_nForegroundLabelIdx || nInitLabel==s_nBackgroundLabelIdx);
        ValueType& tUnaryCost = ((ValueType*)m_oResegmUnaryCosts.data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_oResegmUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
        if(nInitLabel!=nNewLabel) {
            const ValueType tEnergyInit = oNode.pResegmUnaryFunc->second(&nInitLabel);
            const ValueType tEnergyModif = oNode.pResegmUnaryFunc->second(&nNewLabel);
            tUnaryCost = tEnergyModif-tEnergyInit;
        }
        else
            tUnaryCost = ValueType(0);
        // @@@@ CODE BELOW IS ONLY FOR DEBUG/DISPLAY
        ValueType& tPairwCost = ((ValueType*)m_oResegmPairwCosts.data)[nLUTNodeIdx];
        lvDbgAssert(&tPairwCost==&m_oResegmPairwCosts(oNode.nRowIdx,oNode.nColIdx));
        tPairwCost = ValueType(0);
        for(size_t nOrientIdx=0; nOrientIdx<oNode.anResegmPairwFactIDs.size(); ++nOrientIdx) {
            if(oNode.anResegmPairwFactIDs[nOrientIdx]!=SIZE_MAX) {
                lvDbgAssert(oNode.apResegmPairwFuncs[nOrientIdx] && oNode.anPairwLUTNodeIdxs[nOrientIdx]<m_oGridSize.total());
                std::array<InternalLabelType,2> aLabels = {nInitLabel,((InternalLabelType*)m_oResegmLabeling.data)[oNode.anPairwLUTNodeIdxs[nOrientIdx]]};
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

inline opengm::InferenceTermination StereoSegmMatcher::GraphModelData::infer() {
    lvDbgAssert(m_pStereoModel && m_pResegmModel);
    cv::Mat((m_aInputs[InputPack_LeftMask]>0)&s_nForegroundLabelIdx).copyTo(m_oResegmLabeling); // @@@@ update for right image inference via param? or do both at once?
    updateStereoModel(true);
    resetStereoLabelings();
    lvDbgAssert(m_oGridSize.dims()==2 && m_oGridSize==m_oStereoLabeling.size && m_oGridSize==m_oResegmLabeling.size);
    // @@@@@@ use one gm labeling/output to infer the stereo result for another camera?
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
    lvDbgAssert(m_nValidGraphNodes==m_vValidLUTNodeIdxs.size());
    const cv::Point2i oDisplayOffset = {-(int)m_nGridBorderSize,-(int)m_nGridBorderSize};
    lvIgnore(oDisplayOffset); // @@@@@@
    const cv::Rect oAssocCountsROI((int)m_nMaxDispOffset,0,(int)m_oGridSize[1],(int)m_oGridSize[0]);
    lvIgnore(oAssocCountsROI); // @@@@@@
    const cv::Rect oFeatROI((int)m_nGridBorderSize,(int)m_nGridBorderSize,(int)(m_oGridSize[1]-m_nGridBorderSize*2),(int)(m_oGridSize[0]-m_nGridBorderSize*2));
    lvIgnore(oFeatROI); // @@@@@@
    //const cv::Rect oDbgROI(0,0,int(m_oGridSize[1]),int(m_oGridSize[0]));
    const cv::Rect oDbgROI(0,160,int(m_oGridSize[1]),1);
    //const cv::Rect oDbgROI(0,128,int(m_oGridSize[1]),1);
    lvIgnore(oDbgROI); // @@@@@@
    if(lv::getVerbosity()>=2) {
        cv::Mat oLeftImg = m_aInputs[InputPack_LeftImg].clone();
        if(oLeftImg.channels()==1)
            cv::cvtColor(oLeftImg,oLeftImg,cv::COLOR_GRAY2BGR);
        cv::rectangle(oLeftImg,oDbgROI,cv::Scalar_<uchar>(0,255,0));
        cv::imshow("left input (a)",oLeftImg);
        cv::Mat oRightImg = m_aInputs[InputPack_RightImg].clone();
        cv::imshow("right input (b)",oRightImg);
    }
    // @@@@@@ use member-alloc QPBO here instead of stack
    kolmogorov::qpbo::QPBO<ValueType> oStereoMinimizer((int)m_nValidGraphNodes,0); // @@@@@@@ preset max edge count using max clique size times node count
    kolmogorov::qpbo::QPBO<ValueType> oResegmMinimizer((int)m_nValidGraphNodes,0); // @@@@@@@ preset max edge count using max clique size times node count
    using HOEReducer = HigherOrderEnergy<ValueType,s_nMaxOrder>;
    std::array<typename HOEReducer::VarId,s_nMaxOrder> aTermEnergyLUT;
    std::array<InternalLabelType,s_nMaxOrder> aCliqueLabels;
    std::array<ValueType,s_nMaxCliqueAssign> aCliqueCoeffs;
    size_t nMoveIter=0, nConsecUnchangedLabels=0, nOrderingIdx=0;
    lvDbgAssert(m_vStereoLabelOrdering.size()==m_vStereoLabels.size());
    InternalLabelType nStereoAlphaLabel = m_vStereoLabelOrdering[nOrderingIdx];
    const auto lFactorReducer = [&](auto& oGraphFactor, size_t nFactOrder, HOEReducer& oReducer, InternalLabelType nAlphaLabel, const InternalLabelType* aLabeling) {
        lvDbgAssert(oGraphFactor.numberOfVariables()==nFactOrder);
        const size_t nAssignCount = 1UL<<nFactOrder;
        std::fill_n(aCliqueCoeffs.begin(),nAssignCount,(ValueType)0);
        for(size_t nAssignIdx=0; nAssignIdx<nAssignCount; ++nAssignIdx) {
            for(size_t nVarIdx=0; nVarIdx<nFactOrder; ++nVarIdx)
                aCliqueLabels[nVarIdx] = (nAssignIdx&(1<<nVarIdx))?nAlphaLabel:aLabeling[m_vValidLUTNodeIdxs[oGraphFactor.variableIndex(nVarIdx)]];
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
            oReducer.AddTerm(aCliqueCoeffs[nAssignSubsetIdx],nCurrTermDegree,aTermEnergyLUT.data());
        }
    };
    lvIgnore(lFactorReducer);
    // @@@@@@ THIS IS A DYNAMIC MRF OPTIMIZATION PROBLEM; MUST TRY TO REUSE MRF STRUCTURE
    lv::StopWatch oLocalTimer;
    ValueType tLastStereoEnergy = m_pStereoInf->value();
    ValueType tLastResegmEnergy = std::numeric_limits<ValueType>::max();
    bool bJustUpdatedSegm = false;
    bool bResegmModelInitialized = false;
    // each iter below is an alpha-exp move based on A. Fix's primal-dual energy minimization method for higher-order MRFs
    // see "A Primal-Dual Algorithm for Higher-Order Multilabel Markov Random Fields" in CVPR2014 for more info (doi = 10.1109/CVPR.2014.149)
    ValueType tLastStereoAssocEnergy = calcTotalAssocCost(); // @@@@
    while(++nMoveIter<=m_nMaxMoveIterCount && nConsecUnchangedLabels<m_nStereoLabels) {
        if(lv::getVerbosity()>=2)
            lvCout << "\tdisp inf w/ lblidx=" << (int)nStereoAlphaLabel << "   [iter #" << nMoveIter << "]\n";
        const bool bNullifyStereoPairwCosts = (STEREOSEGMATCH_CONFIG_USE_UNARY_ONLY_FIRST)&&nMoveIter<=m_nStereoLabels;
        calcStereoMoveCosts(nStereoAlphaLabel);
        if(lv::getVerbosity()>=3) {
            lvCout << "-----\n\n\n";
            if(m_aInputs[InputPack_LeftImg].channels()==1 && m_aInputs[InputPack_RightImg].channels()==1) {
                lvCout << "inputa = " << lv::to_string(m_aInputs[InputPack_LeftImg](oDbgROI),oDbgROI.tl()) << '\n';
                lvCout << "inputb = " << lv::to_string(m_aInputs[InputPack_RightImg](oDbgROI),oDbgROI.tl()) << '\n';
            }
            lvCout << "disp = " << lv::to_string(m_oStereoLabeling(oDbgROI),oDbgROI.tl()) << '\n';
            cv::Mat_<InternalLabelType> oStereoLabelingReal = m_oStereoLabeling.clone();
            std::transform(oStereoLabelingReal.begin(),oStereoLabelingReal.end(),oStereoLabelingReal.begin(),[&](InternalLabelType n){return (InternalLabelType)getRealLabel(n);});
            lvCout << "disp (real) = " << lv::to_string(oStereoLabelingReal(oDbgROI),oDbgROI.tl()) << '\n';
            lvCout << "assoc_cost = " << lv::to_string(m_oAssocCosts(oDbgROI),oDbgROI.tl()) << '\n';
            lvCout << "unary = " << lv::to_string(m_oStereoUnaryCosts(oDbgROI),oDbgROI.tl()) << '\n';
            lvCout << "pairw = " << lv::to_string(m_oStereoPairwCosts(oDbgROI),oDbgROI.tl()) << '\n';
            if(nStereoAlphaLabel<m_nRealStereoLabels) {
                lvCout << "img affin next = " << lv::to_string(cv::Mat_<float>(lv::squeeze(lv::getSubMat(m_vNextFeats[FeatPack_ImgAffinity],2,nStereoAlphaLabel)))(oDbgROI),oDbgROI.tl()) << '\n';
                lvCout << "img discrim next = " << lv::to_string(cv::Mat_<float>(lv::squeeze(lv::getSubMat(m_vNextFeats[FeatPack_ImgDiscrimPow],2,nStereoAlphaLabel)))(oDbgROI),oDbgROI.tl()) << '\n';
                lvCout << "shp affin next = " << lv::to_string(cv::Mat_<float>(lv::squeeze(lv::getSubMat(m_vNextFeats[FeatPack_ShpAffinity],2,nStereoAlphaLabel)))(oDbgROI),oDbgROI.tl()) << '\n';
                lvCout << "shp discrim next = " << lv::to_string(cv::Mat_<float>(lv::squeeze(lv::getSubMat(m_vNextFeats[FeatPack_ShpDiscrimPow],2,nStereoAlphaLabel)))(oDbgROI),oDbgROI.tl()) << '\n';
            }
            lvCout << "upcoming inf w/ disp lblidx=" << (int)nStereoAlphaLabel << "   (real=" << (int)getRealLabel(nStereoAlphaLabel) << ")\n";
            cv::Mat oCurrAssocCountsDisplay = StereoSegmMatcher::getAssocCountsMapDisplay(*this);
            if(oCurrAssocCountsDisplay.size().area()<640*480)
                cv::resize(oCurrAssocCountsDisplay,oCurrAssocCountsDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow("assoc_counts",oCurrAssocCountsDisplay);
            cv::Mat oCurrLabelingDisplay = StereoSegmMatcher::getStereoDispMapDisplay(*this);
            cv::rectangle(oCurrLabelingDisplay,oDbgROI,cv::Scalar_<uchar>(0,255,0));
            if(oCurrLabelingDisplay.size().area()<640*480)
                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow("disp",oCurrLabelingDisplay);
            cv::waitKey(lv::getVerbosity()>=4?0:1);
        }
        HOEReducer oStereoReducer; // @@@@@@ check if HigherOrderEnergy::Clear() can be used instead of realloc
        oStereoReducer.AddVars((int)m_nValidGraphNodes);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            // manually add 1st order factors while evaluating new assoc energy
            if(oNode.nStereoUnaryFactID!=SIZE_MAX) {
                const ValueType& tAssocCost = ((ValueType*)m_oAssocCosts.data)[nLUTNodeIdx];
                lvDbgAssert(&tAssocCost==&m_oAssocCosts(oNode.nRowIdx,oNode.nColIdx));
                const ValueType& tUnaryCost = ((ValueType*)m_oStereoUnaryCosts.data)[nLUTNodeIdx];
                lvDbgAssert(&tUnaryCost==&m_oStereoUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
                oStereoReducer.AddUnaryTerm((int)nGraphNodeIdx,tAssocCost+tUnaryCost);
            }
            if(!bNullifyStereoPairwCosts) {
                // now add 2nd order & higher order factors via lambda
                if(oNode.anStereoPairwFactIDs[0]!=SIZE_MAX)
                    lFactorReducer(m_pStereoModel->operator[](oNode.anStereoPairwFactIDs[0]),2,oStereoReducer,nStereoAlphaLabel,(InternalLabelType*)m_oStereoLabeling.data);
                if(oNode.anStereoPairwFactIDs[1]!=SIZE_MAX)
                    lFactorReducer(m_pStereoModel->operator[](oNode.anStereoPairwFactIDs[1]),2,oStereoReducer,nStereoAlphaLabel,(InternalLabelType*)m_oStereoLabeling.data);
                // @@@@@ add higher o facts here (3-conn on epi lines?)
            }
        }
        oStereoMinimizer.Reset();
        oStereoReducer.ToQuadratic(oStereoMinimizer); // @@@@@ internally might call addNodes/edges to QPBO object; optim tbd
        //@@@@@@ oStereoMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11); @@@ can still add more if needed
        oStereoMinimizer.Solve();
        oStereoMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
        const cv::Mat_<AssocCountType> oLastAssocCounts = m_oAssocCounts.clone();
        const cv::Mat_<InternalLabelType> oLastStereoLabeling = m_oStereoLabeling.clone();
        size_t nChangedStereoLabels = 0;
        cv::Mat_<uchar> oDisparitySwaps(m_oGridSize(),0);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
            const int nMoveLabel = oStereoMinimizer.GetLabel((int)nGraphNodeIdx);
            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
            if(nMoveLabel==1) { // node label changed to alpha
                const int nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
                const int nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
                oDisparitySwaps(nRowIdx,nColIdx) = 255;
                const InternalLabelType nOldLabel = m_oStereoLabeling(nRowIdx,nColIdx);
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(nRowIdx,nColIdx,nOldLabel);
                m_oStereoLabeling(nRowIdx,nColIdx) = nStereoAlphaLabel;
                if(nStereoAlphaLabel<m_nDontCareLabelIdx)
                    addAssoc(nRowIdx,nColIdx,nStereoAlphaLabel);
                ++nChangedStereoLabels;
            }
        }
        if(lv::getVerbosity()>=2) {
            if(lv::getVerbosity()>=3) {
                lvCout << "post-inf for disp lblidx=" << (int)nStereoAlphaLabel << '\n';
                lvCout << "disp = " << lv::to_string(m_oStereoLabeling(oDbgROI),oDbgROI.tl()) << '\n';
                lvCout << "disp swaps = " << lv::to_string(oDisparitySwaps(oDbgROI),oDbgROI.tl()) << '\n';
            }
            cv::Mat oCurrLabelingDisplay = StereoSegmMatcher::getStereoDispMapDisplay(*this);
            cv::rectangle(oCurrLabelingDisplay,oDbgROI,cv::Scalar_<uchar>(0,255,0));
            if(oCurrLabelingDisplay.size().area()<640*480)
                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow("disp",oCurrLabelingDisplay);
            cv::Mat oDisparitySwapsDisplay = oDisparitySwaps.clone();
            if(oDisparitySwapsDisplay.size().area()<640*480)
                cv::resize(oDisparitySwapsDisplay,oDisparitySwapsDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow("disp-swaps",oDisparitySwapsDisplay);
        }

        // @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ CHECK IF MINIMIZATION STILL HOLDS
        const ValueType tCurrStereoEnergy = m_pStereoInf->value();
        lvDbgAssert(tCurrStereoEnergy>=ValueType(0));
        const ValueType tCurrStereoAssocEnergy = calcTotalAssocCost();
        if(lv::getVerbosity()>=2) {
            std::stringstream ssStereoEnergyDiff;
            if(tCurrStereoEnergy-tLastStereoEnergy==ValueType(0))
                ssStereoEnergyDiff << "null";
            else
                ssStereoEnergyDiff << std::showpos << tCurrStereoEnergy-tLastStereoEnergy;
            lvCout << "\t\tdisp e = " << tCurrStereoEnergy << "   (delta=" << ssStereoEnergyDiff.str() << ")\n";
            if(bNullifyStereoPairwCosts)
                lvCout << "\t\t\t(nullifying pairw costs)\n";
            else if(bJustUpdatedSegm)
                lvCout << "\t\t\t(just updated segmentation)\n";
            else if(tLastStereoEnergy<tCurrStereoEnergy) {
                lvCout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
                cv::Mat swappts;
                cv::findNonZero(oDisparitySwaps,swappts);
                lvPrint(swappts.total());
                cv::Point pt = swappts.at<cv::Point>(0);
                lvPrint(pt) << "x=" << pt.x << "   y=" << pt.y;
                InternalLabelType nLastLabel = oLastStereoLabeling(pt);
                lvPrint((int)nLastLabel);
                int nNeighbRadius = 1;
                if(pt.x>=nNeighbRadius && pt.x<oLastStereoLabeling.cols-nNeighbRadius && pt.y>=nNeighbRadius && pt.y<oLastStereoLabeling.rows-nNeighbRadius)
                    lvPrint(oLastStereoLabeling(cv::Rect(pt.x-nNeighbRadius,pt.y-nNeighbRadius,nNeighbRadius*2+1,nNeighbRadius*2+1)));
                InternalLabelType nNewLabel = m_oStereoLabeling(pt);
                lvPrint((int)nNewLabel);
                if(pt.x>=nNeighbRadius && pt.x<m_oStereoLabeling.cols-nNeighbRadius && pt.y>=nNeighbRadius && pt.y<m_oStereoLabeling.rows-nNeighbRadius)
                    lvPrint(m_oStereoLabeling(cv::Rect(pt.x-nNeighbRadius,pt.y-nNeighbRadius,nNeighbRadius*2+1,nNeighbRadius*2+1)));
                lvPrint(ValueType(m_vNextFeats[FeatPack_ImgAffinity].at<float>(pt.y,pt.x,nLastLabel)*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE));
                lvPrint(ValueType(m_vNextFeats[FeatPack_ImgAffinity].at<float>(pt.y,pt.x,nNewLabel)*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE));
                lvPrint(ValueType(m_vNextFeats[FeatPack_ShpAffinity].at<float>(pt.y,pt.x,nLastLabel)*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE));
                lvPrint(ValueType(m_vNextFeats[FeatPack_ShpAffinity].at<float>(pt.y,pt.x,nNewLabel)*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE));
                lvPrint(tLastStereoAssocEnergy);
                lvPrint(tCurrStereoAssocEnergy);
                lvPrint(tCurrStereoAssocEnergy-tLastStereoAssocEnergy);
                lvPrint(m_oAssocCosts.at<ValueType>(pt.y,pt.x));
                lvPrint(m_oStereoUnaryCosts.at<ValueType>(pt.y,pt.x));
                lvPrint(m_oStereoPairwCosts.at<ValueType>(pt.y,pt.x));
                cv::Mat tmp = m_oAssocCounts;
                m_oAssocCounts = oLastAssocCounts;
                lvPrint(getAssocCount(pt.y,pt.x-getRealLabel(nLastLabel))) << "^^^^^ OLD\n";
                lvPrint(getAssocCount(pt.y,pt.x-getRealLabel(nNewLabel))) << "^^^^^ OLD\n";
                ValueType old_assoc_cost = m_aAssocCostRealSumLUT[getAssocCount(pt.y,pt.x-getRealLabel(nLastLabel))]+m_aAssocCostRealSumLUT[getAssocCount(pt.y,pt.x-getRealLabel(nNewLabel))];
                lvPrint(old_assoc_cost);
                m_oAssocCounts = tmp;
                lvPrint(getAssocCount(pt.y,pt.x-getRealLabel(nLastLabel))) << "^^^^^ NEW\n";
                lvPrint(getAssocCount(pt.y,pt.x-getRealLabel(nNewLabel))) << "^^^^^ NEW\n";
                ValueType new_assoc_cost = m_aAssocCostRealSumLUT[getAssocCount(pt.y,pt.x-getRealLabel(nLastLabel))]+m_aAssocCostRealSumLUT[getAssocCount(pt.y,pt.x-getRealLabel(nNewLabel))];
                lvPrint(new_assoc_cost);
                ValueType assoc_cost_diff = new_assoc_cost - old_assoc_cost;
                lvPrint(assoc_cost_diff);
                lvPrint(m_oAssocCosts.at<ValueType>(pt.y,pt.x));

                //lvPrint(cv::countNonZero(m_oAssocCounts!=oLastAssocCounts));
                cv::Mat oNonZeroPts;
                cv::findNonZero(m_oAssocCounts!=oLastAssocCounts,oNonZeroPts);
                lvAssert(oNonZeroPts.total()>=2);
                lvPrint(oNonZeroPts.at<cv::Point>(0));
                lvPrint(oNonZeroPts.at<cv::Point>(1));
                cv::waitKey(0);
                lvCout << "\t\t\tstereo energy not minimizing!\n";
            }
        }
        tLastStereoEnergy = tCurrStereoEnergy;
        tLastStereoAssocEnergy = tCurrStereoAssocEnergy;
        bJustUpdatedSegm = false;

        nConsecUnchangedLabels = (nChangedStereoLabels>0)?0:nConsecUnchangedLabels+1;
        nStereoAlphaLabel = m_vStereoLabelOrdering[(++nOrderingIdx%=m_nStereoLabels)]; // @@@@ order of future moves can be influenced by labels that cause the most changes? (but only late, to avoid bad local minima?)

        if((nMoveIter%STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM)==0) {
            updateResegmModel(!bResegmModelInitialized);
            bResegmModelInitialized = true;
            for(InternalLabelType nResegmAlphaLabel : {s_nForegroundLabelIdx,s_nBackgroundLabelIdx}) {
                if(lv::getVerbosity()>=2)
                    lvCout << "\tsegm inf w/ lblidx=" << (int)nResegmAlphaLabel << "   [iter #" << nMoveIter << "]\n";
                calcResegmMoveCosts(nResegmAlphaLabel);
                if(lv::getVerbosity()>=3) {
                    lvCout << "-----\n\n\n";
                    lvCout << "segm = " << lv::to_string(m_oResegmLabeling(oDbgROI),oDbgROI.tl()) << '\n';
                    lvCout << "unary = " << lv::to_string(m_oResegmUnaryCosts(oDbgROI),oDbgROI.tl()) << '\n';
                    lvCout << "pairw = " << lv::to_string(m_oResegmPairwCosts(oDbgROI),oDbgROI.tl()) << '\n';
                    lvCout << "next label = " << (int)nResegmAlphaLabel << '\n';
                    cv::Mat oCurrLabelingDisplay = getResegmMapDisplay(*this);
                    if(oCurrLabelingDisplay.size().area()<640*480)
                        cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                    cv::imshow("segm",oCurrLabelingDisplay);
                    cv::waitKey(lv::getVerbosity()>=4?0:1);
                }
                HOEReducer oResegmReducer; // @@@@@@ check if HigherOrderEnergy::Clear() can be used instead of realloc
                oResegmReducer.AddVars((int)m_nValidGraphNodes);
                for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
                    const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
                    const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
                    // manually add 1st order factors while evaluating new assoc energy
                    if(oNode.nResegmUnaryFactID!=SIZE_MAX) {
                        const ValueType& tUnaryCost = ((ValueType*)m_oResegmUnaryCosts.data)[nLUTNodeIdx];
                        lvDbgAssert(&tUnaryCost==&m_oResegmUnaryCosts(oNode.nRowIdx,oNode.nColIdx));
                        oResegmReducer.AddUnaryTerm((int)nGraphNodeIdx,tUnaryCost);
                    }
                    // now add 2nd order & higher order factors via lambda
                    if(oNode.anResegmPairwFactIDs[0]!=SIZE_MAX)
                        lFactorReducer(m_pResegmModel->operator[](oNode.anResegmPairwFactIDs[0]),2,oResegmReducer,nResegmAlphaLabel,(InternalLabelType*)m_oResegmLabeling.data);
                    if(oNode.anResegmPairwFactIDs[1]!=SIZE_MAX)
                        lFactorReducer(m_pResegmModel->operator[](oNode.anResegmPairwFactIDs[1]),2,oResegmReducer,nResegmAlphaLabel,(InternalLabelType*)m_oResegmLabeling.data);
                    // @@@@@ add higher o facts here (3-conn on epi lines?)
                }
                oResegmMinimizer.Reset();
                oResegmReducer.ToQuadratic(oResegmMinimizer); // @@@@@ internally might call addNodes/edges to QPBO object; optim tbd
                //@@@@@@ oResegmMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11); @@@ can still add more if needed
                oResegmMinimizer.Solve();
                oResegmMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
                size_t nChangedResegmLabelings = 0;
                cv::Mat_<uchar> oSegmSwaps(m_oGridSize(),0);
                for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_nValidGraphNodes; ++nGraphNodeIdx) {
                    const size_t nLUTNodeIdx = m_vValidLUTNodeIdxs[nGraphNodeIdx];
                    const int nMoveLabel = oResegmMinimizer.GetLabel((int)nGraphNodeIdx);
                    lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
                    if(nMoveLabel==1) { // node label changed to alpha
                        const int nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
                        const int nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
                        oSegmSwaps(nRowIdx,nColIdx) = 255;
                        m_oResegmLabeling(nRowIdx,nColIdx) = nResegmAlphaLabel;
                        ++nChangedResegmLabelings;
                    }
                }
                if(lv::getVerbosity()>=2) {
                    if(lv::getVerbosity()>=3) {
                        lvCout << "post-inf for segm lblidx=" << (int)nResegmAlphaLabel << '\n';
                        lvCout << "segm = " << lv::to_string(m_oResegmLabeling(oDbgROI),oDbgROI.tl()) << '\n';
                        lvCout << "segm swaps = " << lv::to_string(oSegmSwaps(oDbgROI),oDbgROI.tl()) << '\n';
                    }
                    cv::Mat oCurrLabelingDisplay = getResegmMapDisplay(*this);
                    if(oCurrLabelingDisplay.size().area()<640*480)
                        cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                    cv::imshow("segm",oCurrLabelingDisplay);
                    cv::Mat oSegmSwapsDisplay = oSegmSwaps.clone();
                    if(oSegmSwapsDisplay.size().area()<640*480)
                        cv::resize(oSegmSwapsDisplay,oSegmSwapsDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                    cv::imshow("segm-swaps",oSegmSwapsDisplay);
                }
                const ValueType tCurrResegmEnergy = m_pResegmInf->value();
                lvDbgAssert(tCurrResegmEnergy>=ValueType(0));
                tLastResegmEnergy = tCurrResegmEnergy;
                if(lv::getVerbosity()>=2) {
                    std::stringstream ssResegmEnergyDiff;
                    if(tCurrResegmEnergy-tLastResegmEnergy==ValueType(0))
                        ssResegmEnergyDiff << "null";
                    else
                        ssResegmEnergyDiff << std::showpos << tCurrResegmEnergy-tLastResegmEnergy;
                    lvCout << "\t\tsegm [+" << (nResegmAlphaLabel==s_nForegroundLabelIdx?"fg":"bg") << "]  e = " << tCurrResegmEnergy << "   (delta=" << ssResegmEnergyDiff.str() << ")\n";
                    if(tLastResegmEnergy<tCurrResegmEnergy)
                        lvCout << "\t\t\tresegm energy not minimizing!\n";
                    cv::waitKey(1);
                }
                if(nChangedResegmLabelings) {
                    // @@@@@ add change mask to calc shape features? avoid recalc stuff that's too far (descriptors)
                    calcShapeFeatures(std::array<cv::Mat,2>{m_oResegmLabeling,m_aInputs[InputPack_RightMask]},false); // @@@@@ make resegm graph infer both?
                    updateResegmModel(false);
                    bJustUpdatedSegm = true;
                }
                if(lv::getVerbosity()>=2)
                    cv::waitKey(1);
            }
            updateStereoModel(false);
        }
        else if(lv::getVerbosity()>=2)
            cv::waitKey(1);
    }
    lvLog_(1,"Inference completed in %f second(s).",oLocalTimer.tock());
    if(lv::getVerbosity()>=2)
        cv::waitKey(0);
    return opengm::InferenceTermination::NORMAL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::StereoGraphInference::StereoGraphInference(GraphModelData& oData) :
        m_oData(oData) {
    lvAssert_(m_oData.m_pStereoModel,"invalid graph");
    const StereoModelType& oGM = *m_oData.m_pStereoModel;
    lvAssert_(oGM.numberOfFactors()>0,"invalid graph");
    for(size_t nFactIdx=0; nFactIdx<oGM.numberOfFactors(); ++nFactIdx)
        lvDbgAssert__(oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvDbgAssert_(oGM.numberOfVariables()>0 && oGM.numberOfVariables()<=(IndexType)m_oData.m_oGridSize.total(),"graph node count must match grid size");
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<oGM.numberOfVariables(); ++nGraphNodeIdx)
        lvDbgAssert_(oGM.numberOfLabels(nGraphNodeIdx)==m_oData.m_vStereoLabels.size(),"graph nodes must all have the same number of labels");
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
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_oGridSize.total()),lv::make_range(InternalLabelType(0),InternalLabelType(m_oData.m_vStereoLabels.size()-1))).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_oGridSize.total(),m_oData.m_oStereoLabeling.begin());
}

inline void StereoSegmMatcher::StereoGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),m_oData.m_vStereoLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_oStereoLabeling.begin(),[&](const OutputLabelType& nRealLabel){return m_oData.getInternalLabel(nRealLabel);});
}

inline opengm::InferenceTermination StereoSegmMatcher::StereoGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    if(n==1) {
        lvDbgAssert_(m_oData.m_oStereoLabeling.total()==m_oData.m_oGridSize.total(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_oStereoLabeling.total());
        std::copy(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones!
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void StereoSegmMatcher::StereoGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgAssert_(m_oData.m_oStereoLabeling.isContinuous() && m_oData.m_oStereoLabeling.total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_oStereoLabeling.size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_oStereoLabeling.begin(),m_oData.m_oStereoLabeling.end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return m_oData.getRealLabel(nLabel);});
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::StereoGraphInference::value() const {
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oStereoLabeling.size,"output labeling must be a 2d grid");
    const ValueType tTotAssocCost = m_oData.calcTotalAssocCost();
    struct GraphNodeLabelIter {
        GraphNodeLabelIter(const GraphModelData& oData) : m_oData(oData) {}
        InternalLabelType operator[](size_t nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_oData.m_vValidLUTNodeIdxs[nGraphNodeIdx];
            return ((InternalLabelType*)m_oData.m_oStereoLabeling.data)[nLUTNodeIdx];
        }
        const GraphModelData& m_oData;
    } oLabelIter(m_oData);
    const ValueType tTotStereoLabelCost = m_oData.m_pStereoModel->evaluate(oLabelIter);
    return tTotAssocCost+tTotStereoLabelCost;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::ResegmGraphInference::ResegmGraphInference(GraphModelData& oData) :
        m_oData(oData) {
    lvAssert_(m_oData.m_pResegmModel,"invalid graph");
    const ResegmModelType& oGM = *m_oData.m_pResegmModel;
    lvAssert_(oGM.numberOfFactors()>0,"invalid graph");
    for(size_t nFactIdx=0; nFactIdx<oGM.numberOfFactors(); ++nFactIdx)
        lvDbgAssert__(oGM[nFactIdx].numberOfVariables()<=s_nMaxOrder,"graph had some factors of order %d (max allowed is %d)",(int)oGM[nFactIdx].numberOfVariables(),(int)s_nMaxOrder);
    lvDbgAssert_(oGM.numberOfVariables()>0 && oGM.numberOfVariables()<=(IndexType)m_oData.m_oGridSize.total(),"graph node count must match grid size");
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<oGM.numberOfVariables(); ++nGraphNodeIdx)
        lvDbgAssert_(oGM.numberOfLabels(nGraphNodeIdx)==size_t(2),"graph nodes must all have the same number of labels");
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
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_oGridSize.total()),std::vector<InternalLabelType>{s_nBackgroundLabelIdx,s_nForegroundLabelIdx}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_oResegmLabeling.isContinuous() && m_oData.m_oResegmLabeling.total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_oGridSize.total(),m_oData.m_oResegmLabeling.begin());
}

inline void StereoSegmMatcher::ResegmGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),std::vector<OutputLabelType>{s_nBackgroundLabel,s_nForegroundLabel}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_oResegmLabeling.begin(),[&](const OutputLabelType& nRealLabel){return nRealLabel?s_nForegroundLabelIdx:s_nBackgroundLabelIdx;});
}

inline opengm::InferenceTermination StereoSegmMatcher::ResegmGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    if(n==1) {
        lvDbgAssert_(m_oData.m_oResegmLabeling.total()==m_oData.m_oGridSize.total(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_oResegmLabeling.total());
        std::copy(m_oData.m_oResegmLabeling.begin(),m_oData.m_oResegmLabeling.end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones (same values, different type)
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void StereoSegmMatcher::ResegmGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgAssert_(m_oData.m_oResegmLabeling.isContinuous() && m_oData.m_oResegmLabeling.total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_oResegmLabeling.size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_oResegmLabeling.begin(),m_oData.m_oResegmLabeling.end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return nLabel?s_nForegroundLabel:s_nBackgroundLabel;});
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::ResegmGraphInference::value() const {
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_oResegmLabeling.size,"output labeling must be a 2d grid");
    struct GraphNodeLabelIter {
        GraphNodeLabelIter(const GraphModelData& oData) : m_oData(oData) {}
        InternalLabelType operator[](size_t nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_oData.m_vValidLUTNodeIdxs[nGraphNodeIdx];
            return ((InternalLabelType*)m_oData.m_oResegmLabeling.data)[nLUTNodeIdx];
        }
        const GraphModelData& m_oData;
    } oLabelIter(m_oData);
    const ValueType tTotResegmLabelCost = m_oData.m_pResegmModel->evaluate(oLabelIter);
    return tTotResegmLabelCost;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline cv::Mat StereoSegmMatcher::getResegmMapDisplay(const GraphModelData& oData) {
    lvAssert(!oData.m_oResegmLabeling.empty() && oData.m_oGridSize==oData.m_oResegmLabeling.size);
    cv::Mat oOutput(oData.m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<oData.m_oResegmLabeling.rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<oData.m_oResegmLabeling.cols; ++nColIdx) {
            const InternalLabelType nCurrLabel = oData.m_oResegmLabeling(nRowIdx,nColIdx);
            const uchar nInitLabel = oData.m_aInputs[InputPack_LeftMask].at<uchar>(nRowIdx,nColIdx);
            if(nCurrLabel==s_nForegroundLabelIdx && nInitLabel>0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,127);
            else if(nCurrLabel==s_nBackgroundLabelIdx && nInitLabel>0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(127,0,63);
            else if(nCurrLabel==s_nForegroundLabelIdx && nInitLabel==0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,63,127);
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,0);
        }
    }
    cv::Mat oInputImageDisplay = oData.m_aInputs[InputPack_LeftImg];
    if(oInputImageDisplay.channels()==1)
        cv::cvtColor(oInputImageDisplay,oInputImageDisplay,cv::COLOR_GRAY2BGR);
    oOutput = (oOutput+oInputImageDisplay)/2;
    return oOutput;
}

inline cv::Mat StereoSegmMatcher::getStereoDispMapDisplay(const GraphModelData& oData) {
    lvAssert(!oData.m_oStereoLabeling.empty() && oData.m_oGridSize==oData.m_oStereoLabeling.size);
    const int nMinDescColIdx = int(oData.m_nGridBorderSize);
    const int nMinDescRowIdx = int(oData.m_nGridBorderSize);
    const int nMaxDescColIdx = int(oData.m_oGridSize[1]-oData.m_nGridBorderSize-1);
    const int nMaxDescRowIdx = int(oData.m_oGridSize[0]-oData.m_nGridBorderSize-1);
    const auto lHasValidDesc = [&nMinDescRowIdx,&nMaxDescRowIdx,&nMinDescColIdx,&nMaxDescColIdx](int nRowIdx, int nColIdx) {
        return nRowIdx>=nMinDescRowIdx && nRowIdx<=nMaxDescRowIdx && nColIdx>=nMinDescColIdx && nColIdx<=nMaxDescColIdx;
    };
    const float fRescaleFact = float(UCHAR_MAX)/(oData.m_nMaxDispOffset-oData.m_nMinDispOffset+1);
    cv::Mat oOutput(oData.m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<oData.m_oStereoLabeling.rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<oData.m_oStereoLabeling.cols; ++nColIdx) {
            const OutputLabelType nRealLabel = oData.getRealLabel(oData.m_oStereoLabeling(nRowIdx,nColIdx));
            if(nRealLabel==s_nDontCareLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,128,128);
            else if(nRealLabel==s_nOccludedLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(128,0,128);
            else {
                const uchar nIntensity = uchar((nRealLabel-oData.m_nMinDispOffset)*fRescaleFact);
                if(lHasValidDesc(nRowIdx,nColIdx/*-int(oData.m_nMaxDispOffset)*/))
                    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b::all(nIntensity);
                else if(lHasValidDesc(nRowIdx,nColIdx))
                    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(uchar(nIntensity/2),uchar(nIntensity/2),nIntensity);
                else
                    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,nIntensity);
            }
        }
    }
    return oOutput;
}

inline cv::Mat StereoSegmMatcher::getAssocCountsMapDisplay(const GraphModelData& oData) {
    lvAssert(!oData.m_oAssocCounts.empty() && oData.m_oAssocCounts.rows==int(oData.m_oGridSize[0]));
    lvAssert(oData.m_oAssocCounts.cols==int((oData.m_oGridSize[1]+oData.m_nMaxDispOffset)/oData.m_nDispOffsetStep));
    double dMax;
    cv::minMaxIdx(oData.m_oAssocCounts,nullptr,&dMax);
    const float fRescaleFact = float(UCHAR_MAX)/(int(dMax)+1);
    cv::Mat oOutput(int(oData.m_oGridSize[0]),int(oData.m_oGridSize[1]+oData.m_nMaxDispOffset),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<int(oData.m_oGridSize[0]); ++nRowIdx) {
        for(int nColIdx=-int(oData.m_nMaxDispOffset); nColIdx<int(oData.m_oGridSize[1]); ++nColIdx) {
            const AssocCountType nCount = oData.getAssocCount(nRowIdx,nColIdx);
            if(nColIdx<0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx+int(oData.m_nMaxDispOffset)) = cv::Vec3b(0,0,uchar(nCount*fRescaleFact));
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx+int(oData.m_nMaxDispOffset)) = cv::Vec3b::all(uchar(nCount*fRescaleFact));
        }
    }
    return oOutput;
}