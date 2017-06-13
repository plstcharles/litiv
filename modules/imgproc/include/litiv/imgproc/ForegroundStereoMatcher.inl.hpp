
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
    lvDbgExceptionWatch;
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
    static_assert(getCameraCount()==2,"bad static array size, mismatch with cam head count (hardcoded stuff below will break)");
    lvDbgExceptionWatch;
    lvAssert_(!aROIs[0].empty() && aROIs[0].total()>1 && aROIs[0].type()==CV_8UC1,"bad input ROI size/type");
    lvAssert_(lv::MatInfo(aROIs[0])==lv::MatInfo(aROIs[1]),"mismatched ROI size/type");
    lvAssert_(m_nDispStep>0,"specified disparity offset step size must be strictly positive");
    lvAssert_(m_vStereoLabels.size()>1,"graph must have at least two possible output labels, beyond reserved ones");
    m_pModelData = std::make_unique<GraphModelData>(aROIs,m_vStereoLabels,m_nDispStep);
    if(m_pDisplayHelper)
        m_pModelData->m_pDisplayHelper = m_pDisplayHelper;
}

inline void StereoSegmMatcher::apply(const MatArrayIn& aInputs, MatArrayOut& aOutputs) {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"lots of hardcoded indices below");
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx)
        lvAssert__(aInputs[nInputIdx].dims==2 && m_pModelData->m_oGridSize==aInputs[nInputIdx].size(),"input in array at index=%d had the wrong size",(int)nInputIdx);
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputImg = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        const cv::Mat& oInputMask = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask];
        lvAssert_(oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3,"unexpected input image type");
        lvAssert_(oInputMask.type()==CV_8UC1,"unexpected input mask type");
        lvAssert_((cv::countNonZero(oInputMask==0)+cv::countNonZero(oInputMask==255))==(int)oInputMask.total(),"input mask must be binary (0 or 255 only)");
    }
    lvAssert_(!m_pModelData->m_bUsePrecalcFeatsNext || m_pModelData->m_vNextFeats.size()==GraphModelData::FeatPackSize,"unexpected precalculated features vec size");
    for(size_t nInputIdx=0; nInputIdx<aInputs.size(); ++nInputIdx)
        aInputs[nInputIdx].copyTo(m_pModelData->m_aInputs[nInputIdx]);
    if(!m_pModelData->m_bUsePrecalcFeatsNext)
        m_pModelData->calcFeatures(aInputs);
    else
        m_pModelData->m_bUsePrecalcFeatsNext = false;
    m_pModelData->infer();
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_pModelData->m_aStereoLabelings[nCamIdx].copyTo(aOutputs[nCamIdx*OutputPackOffset+OutputPackOffset_Disp]);
        m_pModelData->m_aResegmLabelings[nCamIdx].copyTo(aOutputs[nCamIdx*OutputPackOffset+OutputPackOffset_Mask]);
    }
    for(size_t nOutputIdx=0; nOutputIdx<aOutputs.size(); ++nOutputIdx)
        aOutputs[nOutputIdx].copyTo(m_pModelData->m_aOutputs[nOutputIdx]); // copy for temporal analysis later
}

inline void StereoSegmMatcher::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    m_pModelData->calcFeatures(aInputs,pFeatsPacket);
}

inline void StereoSegmMatcher::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvDbgExceptionWatch;
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
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->m_vStereoLabels.size();
}

inline const std::vector<StereoSegmMatcher::OutputLabelType>& StereoSegmMatcher::getLabels() const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->m_vStereoLabels;
}


inline cv::Mat StereoSegmMatcher::getResegmMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->getResegmMapDisplay(nCamIdx);
}

inline cv::Mat StereoSegmMatcher::getStereoDispMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->getStereoDispMapDisplay(nCamIdx);
}

inline cv::Mat StereoSegmMatcher::getAssocCountsMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(m_pModelData,"model must be initialized first");
    return m_pModelData->getAssocCountsMapDisplay(nCamIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::GraphModelData::GraphModelData(const CamArray<cv::Mat>& aROIs, const std::vector<OutputLabelType>& vRealStereoLabels, size_t nStereoLabelStep) :
        m_nMaxMoveIterCount(STEREOSEGMATCH_DEFAULT_MAX_MOVE_ITER),
        m_nStereoLabelOrderRandomSeed(size_t(0)),
        m_nStereoLabelingRandomSeed(size_t(0)),
        m_aROIs(CamArray<cv::Mat_<uchar>>{aROIs[0]>0,aROIs[1]>0}),
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
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff in constr init list and below will break");
    lvDbgExceptionWatch;
    lvAssert_(m_nMaxMoveIterCount>0,"max iter counts must be strictly positive");
    lvAssert_(lv::MatInfo(m_aROIs[0])==lv::MatInfo(m_aROIs[1]),"ROIs info must match");
    lvAssert_(cv::countNonZero(m_aROIs[0]>0)>1 && cv::countNonZero(m_aROIs[1]>0)>1,"ROIs must have at least two nodes");
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
    const int nLSSInnerRadius = 0;
    const int nLSSOuterRadius = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_RAD;
    const int nLSSPatchSize = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_PATCH;
    const int nLSSAngBins = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_ANG_BINS;
    const int nLSSRadBins = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_RAD_BINS;
    m_pImgDescExtractor = std::make_unique<LSS>(nLSSInnerRadius,nLSSOuterRadius,nLSSPatchSize,nLSSAngBins,nLSSRadBins);
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    cv::Size oMIWinSize(int(STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD)*2+1,int(STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD)*2+1);
    m_pImgDescExtractor = std::make_unique<MutualInfo>(oMIWinSize,true);
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
    const size_t nShapeContextInnerRadius = 2;
    const size_t nShapeContextOuterRadius = STEREOSEGMATCH_DEFAULT_SCDESC_WIN_RAD;
    const size_t nShapeContextAngBins = STEREOSEGMATCH_DEFAULT_SCDESC_ANG_BINS;
    const size_t nShapeContextRadBins = STEREOSEGMATCH_DEFAULT_SCDESC_RAD_BINS;
    m_pShpDescExtractor = std::make_unique<ShapeContext>(nShapeContextInnerRadius,nShapeContextOuterRadius,nShapeContextAngBins,nShapeContextRadBins);
#if STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    constexpr int nSSqrDiffKernelSize = int(STEREOSEGMATCH_DEFAULT_SSQDIFF_PATCH);
    const cv::Size oDescWinSize(nSSqrDiffKernelSize,nSSqrDiffKernelSize);
    m_nGridBorderSize = size_t(nSSqrDiffKernelSize/2);
#else //!STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    const cv::Size oDescWinSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#endif //!STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    lvAssert__(oDescWinSize.width<=(int)m_oGridSize[1] && oDescWinSize.height<=(int)m_oGridSize[0],"image is too small to compute descriptors with current pattern size -- need at least (%d,%d) and got (%d,%d)",oDescWinSize.width,oDescWinSize.height,(int)m_oGridSize[1],(int)m_oGridSize[0]);
    lvDbgAssert(m_nGridBorderSize<m_oGridSize[0] && m_nGridBorderSize<m_oGridSize[1]);
    lvDbgAssert(m_nGridBorderSize<(size_t)oDescWinSize.width && m_nGridBorderSize<(size_t)oDescWinSize.height);
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
    lvLog_(2,"\toutput disp labels:\n%s\n",lv::to_string(std::vector<OutputLabelType>(m_vStereoLabels.begin(),m_vStereoLabels.begin()+m_nRealStereoLabels)).c_str());
    const size_t nRows = m_oGridSize(0);
    const size_t nCols = m_oGridSize(1);
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below will break");
    const CamArray<size_t> anValidNodes{(size_t)cv::countNonZero(m_aROIs[0]),(size_t)cv::countNonZero(m_aROIs[1])};
    const size_t nMaxValidNodes = std::max(anValidNodes[0],anValidNodes[1]);
    lvAssert(anValidNodes[0]<=m_oGridSize.total() && anValidNodes[1]<=m_oGridSize.total());
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx)
        cv::erode(m_aROIs[nCamIdx],m_aDescROIs[nCamIdx],cv::getStructuringElement(cv::MORPH_RECT,oDescWinSize),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar_<uchar>::all(0));
    const size_t nStereoUnaryFuncDataSize = nMaxValidNodes*m_nStereoLabels;
    const size_t nStereoPairwFuncDataSize = nMaxValidNodes*2*(m_nStereoLabels*m_nStereoLabels);
    const size_t nStereoFuncDataSize = nStereoUnaryFuncDataSize+nStereoPairwFuncDataSize/*+...@@@@hoet*/;
    const size_t nResegmUnaryFuncDataSize = nMaxValidNodes*s_nResegmLabels;
    const size_t nResegmPairwFuncDataSize = nMaxValidNodes*2*(s_nResegmLabels*s_nResegmLabels);
    const size_t nResegmFuncDataSize = nResegmUnaryFuncDataSize+nResegmPairwFuncDataSize/*+...@@@@hoet*/;
    const size_t nModelSize = ((nStereoFuncDataSize+nResegmFuncDataSize)*sizeof(ValueType)/*+...@@@@externals*/)*2;
    lvLog_(1,"Expecting dual graph model size = %zu mb",nModelSize/1024/1024);
    lvAssert__(nModelSize<(CACHE_MAX_SIZE_GB*1024*1024*1024),"too many nodes/labels; model is unlikely to fit in memory (estimated: %zu mb)",nModelSize/1024/1024);
    lvLog(2,"Constructing graphical models...");
    lv::StopWatch oLocalTimer;
    const size_t nStereoFactorsPerNode = 3; // @@@@
    const size_t nResegmFactorsPerNode = 3; // @@@@
    const CamArray<size_t> anStereoFunctions{anValidNodes[0]*nStereoFactorsPerNode,anValidNodes[1]*nStereoFactorsPerNode};
    const CamArray<size_t> anResegmFunctions{anValidNodes[0]*nResegmFactorsPerNode,anValidNodes[1]*nResegmFactorsPerNode};
    const std::array<int,3> anAssocMapDims{int(m_oGridSize[0]),int((m_oGridSize[1]+m_nMaxDispOffset/*for oob usage*/)/m_nDispOffsetStep),int(m_nRealStereoLabels*m_nDispOffsetStep)};
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_aStereoLabelings[nCamIdx].create(m_oGridSize);
        m_aResegmLabelings[nCamIdx].create(m_oGridSize);
        m_aGMMCompAssignMap[nCamIdx].create(m_oGridSize);
        m_apStereoModels[nCamIdx] = std::make_unique<StereoModelType>(StereoSpaceType(anValidNodes[nCamIdx],(InternalLabelType)m_nStereoLabels),nStereoFactorsPerNode);
        m_apStereoModels[nCamIdx]->reserveFunctions<ExplicitFunction>(anStereoFunctions[nCamIdx]);
        m_apResegmModels[nCamIdx] = std::make_unique<ResegmModelType>(ResegmSpaceType(anValidNodes[nCamIdx]),nResegmFactorsPerNode);
        m_apResegmModels[nCamIdx]->reserveFunctions<ExplicitFunction>(anResegmFunctions[nCamIdx]);
        m_aAssocCounts[nCamIdx].create(2,anAssocMapDims.data());
        m_aAssocMaps[nCamIdx].create(3,anAssocMapDims.data());
        m_aAssocCosts[nCamIdx].create(m_oGridSize);
        m_aStereoUnaryCosts[nCamIdx].create(m_oGridSize);
        m_aResegmUnaryCosts[nCamIdx].create(m_oGridSize);
        m_aStereoPairwCosts[nCamIdx].create(m_oGridSize); // for dbg only
        m_aResegmPairwCosts[nCamIdx].create(m_oGridSize); // for dbg only
        m_vNodeInfos.resize(m_oGridSize.total());
        m_avValidLUTNodeIdxs[nCamIdx].reserve(anValidNodes[nCamIdx]);
    }
    m_anValidGraphNodes = CamArray<size_t>{};
    m_nTotValidGraphNodes = 0;
    for(int nRowIdx = 0; nRowIdx<(int)nRows; ++nRowIdx) {
        for(int nColIdx = 0; nColIdx<(int)nCols; ++nColIdx) {
            const size_t nLUTNodeIdx = nRowIdx*nCols+nColIdx;
            m_vNodeInfos[nLUTNodeIdx].nRowIdx = nRowIdx;
            m_vNodeInfos[nLUTNodeIdx].nColIdx = nColIdx;
            for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
                m_vNodeInfos[nLUTNodeIdx].abValidGraphNode[nCamIdx] = m_aROIs[nCamIdx](nRowIdx,nColIdx)>0;
                m_vNodeInfos[nLUTNodeIdx].abNearGraphBorders[nCamIdx] = m_aDescROIs[nCamIdx](nRowIdx,nColIdx)==0;
                if(m_vNodeInfos[nLUTNodeIdx].abValidGraphNode[nCamIdx]) {
                    m_vNodeInfos[nLUTNodeIdx].anGraphNodeIdxs[nCamIdx] = m_anValidGraphNodes[nCamIdx]++;
                    m_avValidLUTNodeIdxs[nCamIdx].push_back(nLUTNodeIdx);
                }
                // the LUT members below will be properly initialized in the following sections if node is valid
                m_vNodeInfos[nLUTNodeIdx].anStereoUnaryFactIDs[nCamIdx] = SIZE_MAX;
                m_vNodeInfos[nLUTNodeIdx].anResegmUnaryFactIDs[nCamIdx] = SIZE_MAX;
                m_vNodeInfos[nLUTNodeIdx].apStereoUnaryFuncs[nCamIdx] = nullptr;
                m_vNodeInfos[nLUTNodeIdx].apResegmUnaryFuncs[nCamIdx] = nullptr;
                m_vNodeInfos[nLUTNodeIdx].aanPairwLUTNodeIdxs[nCamIdx] = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
                m_vNodeInfos[nLUTNodeIdx].aanPairwGraphNodeIdxs[nCamIdx] = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
                m_vNodeInfos[nLUTNodeIdx].aanStereoPairwFactIDs[nCamIdx] = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
                m_vNodeInfos[nLUTNodeIdx].aanResegmPairwFactIDs[nCamIdx] = std::array<size_t,2>{SIZE_MAX,SIZE_MAX};
                m_vNodeInfos[nLUTNodeIdx].aapStereoPairwFuncs[nCamIdx] = std::array<StereoFunc*,2>{nullptr,nullptr};
                m_vNodeInfos[nLUTNodeIdx].aapResegmPairwFuncs[nCamIdx] = std::array<ResegmFunc*,2>{nullptr,nullptr};
            }
        }
    }
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvAssert(m_avValidLUTNodeIdxs[nCamIdx].size()==anValidNodes[nCamIdx]);
        lvAssert(m_anValidGraphNodes[nCamIdx]==anValidNodes[nCamIdx]);
        m_nTotValidGraphNodes += m_anValidGraphNodes[nCamIdx];
        m_avStereoUnaryFuncs[nCamIdx].reserve(m_anValidGraphNodes[nCamIdx]);
        m_aavStereoPairwFuncs[nCamIdx][0].reserve(m_anValidGraphNodes[nCamIdx]);
        m_aavStereoPairwFuncs[nCamIdx][1].reserve(m_anValidGraphNodes[nCamIdx]);
        m_avResegmUnaryFuncs[nCamIdx].reserve(m_anValidGraphNodes[nCamIdx]);
        m_aavResegmPairwFuncs[nCamIdx][0].reserve(m_anValidGraphNodes[nCamIdx]);
        m_aavResegmPairwFuncs[nCamIdx][1].reserve(m_anValidGraphNodes[nCamIdx]);
        m_aaStereoFuncsData[nCamIdx] = std::make_unique<ValueType[]>(nStereoFuncDataSize);
        m_apStereoUnaryFuncsDataBase[nCamIdx] = m_aaStereoFuncsData[nCamIdx].get();
        m_apStereoPairwFuncsDataBase[nCamIdx] = m_apStereoUnaryFuncsDataBase[nCamIdx]+nStereoUnaryFuncDataSize;
        m_aaResegmFuncsData[nCamIdx] = std::make_unique<ValueType[]>(nResegmFuncDataSize);
        m_apResegmUnaryFuncsDataBase[nCamIdx] = m_aaResegmFuncsData[nCamIdx].get();
        m_apResegmPairwFuncsDataBase[nCamIdx] = m_apResegmUnaryFuncsDataBase[nCamIdx]+nResegmUnaryFuncDataSize;
    }
    lvLog(2,"\tadding unary factors to each graph node...");
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const std::array<size_t,1> aUnaryStereoFuncDims = {m_nStereoLabels};
        const std::array<size_t,1> aUnaryResegmFuncDims = {s_nResegmLabels};
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            m_avStereoUnaryFuncs[nCamIdx].push_back(m_apStereoModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
            m_avResegmUnaryFuncs[nCamIdx].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
            StereoFunc& oStereoFunc = m_avStereoUnaryFuncs[nCamIdx].back();
            ResegmFunc& oResegmFunc = m_avResegmUnaryFuncs[nCamIdx].back();
            oStereoFunc.second.assign(aUnaryStereoFuncDims.begin(),aUnaryStereoFuncDims.end(),m_apStereoUnaryFuncsDataBase[nCamIdx]+(nGraphNodeIdx*m_nStereoLabels));
            oResegmFunc.second.assign(aUnaryResegmFuncDims.begin(),aUnaryResegmFuncDims.end(),m_apResegmUnaryFuncsDataBase[nCamIdx]+(nGraphNodeIdx*s_nResegmLabels));
            lvDbgAssert(oStereoFunc.second.strides(0)==1 && oResegmFunc.second.strides(0)==1); // expect no padding
            const std::array<size_t,1> aGraphNodeIndices = {nGraphNodeIdx};
            oNode.anStereoUnaryFactIDs[nCamIdx] = m_apStereoModels[nCamIdx]->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
            oNode.anResegmUnaryFactIDs[nCamIdx] = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
            oNode.apStereoUnaryFuncs[nCamIdx] = &oStereoFunc;
            oNode.apResegmUnaryFuncs[nCamIdx] = &oResegmFunc;
        }
    }
    lvLog(2,"\tadding pairwise factors to each graph node pair...");
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        // note: current def w/ explicit stereo function will require too much memory if using >>50 disparity labels
        const std::array<size_t,2> aPairwiseStereoFuncDims = {m_nStereoLabels,m_nStereoLabels};
        const std::array<size_t,2> aPairwiseResegmFuncDims = {s_nResegmLabels,s_nResegmLabels};
        std::array<size_t,2> aGraphNodeIndices;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nBaseLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            NodeInfo& oBaseNode = m_vNodeInfos[nBaseLUTNodeIdx];
            if(!oBaseNode.abValidGraphNode[nCamIdx])
                continue;
            const size_t nRowIdx = (size_t)oBaseNode.nRowIdx;
            const size_t nColIdx = (size_t)oBaseNode.nColIdx;
            aGraphNodeIndices[0] = nGraphNodeIdx;
            if(nRowIdx+1<nRows) { // vertical pair
                const size_t nOffsetLUTNodeIdx = (nRowIdx+1)*nCols+nColIdx;
                const NodeInfo& oOffsetNode = m_vNodeInfos[nOffsetLUTNodeIdx];
                if(oOffsetNode.abValidGraphNode[nCamIdx]) {
                    aGraphNodeIndices[1] = oOffsetNode.anGraphNodeIdxs[nCamIdx];
                    m_aavStereoPairwFuncs[nCamIdx][0].push_back(m_apStereoModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                    m_aavResegmPairwFuncs[nCamIdx][0].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                    StereoFunc& oStereoFunc = m_aavStereoPairwFuncs[nCamIdx][0].back();
                    ResegmFunc& oResegmFunc = m_aavResegmPairwFuncs[nCamIdx][0].back();
                    oStereoFunc.second.assign(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end(),m_apStereoPairwFuncsDataBase[nCamIdx]+((aGraphNodeIndices[0]*2)*m_nStereoLabels*m_nStereoLabels));
                    oResegmFunc.second.assign(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end(),m_apResegmPairwFuncsDataBase[nCamIdx]+((aGraphNodeIndices[0]*2)*s_nResegmLabels*s_nResegmLabels));
                    lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==m_nStereoLabels && oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==s_nResegmLabels); // expect last-idx-major
                    oBaseNode.aanStereoPairwFactIDs[nCamIdx][0] = m_apStereoModels[nCamIdx]->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.aanResegmPairwFactIDs[nCamIdx][0] = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.aapStereoPairwFuncs[nCamIdx][0] = &oStereoFunc;
                    oBaseNode.aapResegmPairwFuncs[nCamIdx][0] = &oResegmFunc;
                    oBaseNode.aanPairwLUTNodeIdxs[nCamIdx][0] = nOffsetLUTNodeIdx;
                    oBaseNode.aanPairwGraphNodeIdxs[nCamIdx][0] = aGraphNodeIndices[1];
                }
            }
            if(nColIdx+1<nCols) { // horizontal pair
                const size_t nOffsetLUTNodeIdx = nRowIdx*nCols+nColIdx+1;
                const NodeInfo& oOffsetNode = m_vNodeInfos[nOffsetLUTNodeIdx];
                if(oOffsetNode.abValidGraphNode[nCamIdx]) {
                    aGraphNodeIndices[1] = oOffsetNode.anGraphNodeIdxs[nCamIdx];
                    m_aavStereoPairwFuncs[nCamIdx][1].push_back(m_apStereoModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                    m_aavResegmPairwFuncs[nCamIdx][1].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                    StereoFunc& oStereoFunc = m_aavStereoPairwFuncs[nCamIdx][1].back();
                    ResegmFunc& oResegmFunc = m_aavResegmPairwFuncs[nCamIdx][1].back();
                    oStereoFunc.second.assign(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end(),m_apStereoPairwFuncsDataBase[nCamIdx]+((aGraphNodeIndices[0]*2+1)*m_nStereoLabels*m_nStereoLabels));
                    oResegmFunc.second.assign(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end(),m_apResegmPairwFuncsDataBase[nCamIdx]+((aGraphNodeIndices[0]*2+1)*s_nResegmLabels*s_nResegmLabels));
                    lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==m_nStereoLabels && oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==s_nResegmLabels); // expect last-idx-major
                    oBaseNode.aanStereoPairwFactIDs[nCamIdx][1] = m_apStereoModels[nCamIdx]->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.aanResegmPairwFactIDs[nCamIdx][1] = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oBaseNode.aapStereoPairwFuncs[nCamIdx][1] = &oStereoFunc;
                    oBaseNode.aapResegmPairwFuncs[nCamIdx][1] = &oResegmFunc;
                    oBaseNode.aanPairwLUTNodeIdxs[nCamIdx][1] = nOffsetLUTNodeIdx;
                    oBaseNode.aanPairwGraphNodeIdxs[nCamIdx][1] = aGraphNodeIndices[1];
                }
            }
        }
    }
    /*
    // add 3rd order function and factors to the model (test)
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
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
    lvLog_(2,"Graph models constructed in %f second(s); finalizing...",oLocalTimer.tock());
    std::once_flag oPrintFlag;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_apStereoModels[nCamIdx]->finalize();
        m_apStereoInfs[nCamIdx] = std::make_unique<StereoGraphInference>(nCamIdx,*this);
        m_apResegmModels[nCamIdx]->finalize();
        m_apResegmInfs[nCamIdx] = std::make_unique<ResegmGraphInference>(nCamIdx,*this);
        if(lv::getVerbosity()>=2) {
            std::call_once(oPrintFlag,[&](){
                lvCout << "Stereo :\n";
                lv::gm::printModelInfo(*m_apStereoModels[nCamIdx]);
                lvCout << "Resegm :\n";
                lv::gm::printModelInfo(*m_apResegmModels[nCamIdx]);
                lvCout << "\n";
            });
        }
    }
}

inline void StereoSegmMatcher::GraphModelData::resetLabelings() {
    lvDbgExceptionWatch;
    std::vector<int> vLabelCounts(m_nStereoLabels,0);
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::Mat((m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask]>0)&s_nForegroundLabelIdx).copyTo(m_aResegmLabelings[nCamIdx]);
        std::fill(m_aStereoLabelings[nCamIdx].begin(),m_aStereoLabelings[nCamIdx].end(),m_nDontCareLabelIdx);
        lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const NodeInfo& oNode = m_vNodeInfos[m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx]];
            lvDbgAssert(oNode.abValidGraphNode[nCamIdx]);
            lvDbgAssert(oNode.anGraphNodeIdxs[nCamIdx]==nGraphNodeIdx);
            lvDbgAssert(oNode.anStereoUnaryFactIDs[nCamIdx]<m_apStereoModels[nCamIdx]->numberOfFactors());
            lvDbgAssert(m_apStereoModels[nCamIdx]->numberOfLabels(oNode.anStereoUnaryFactIDs[nCamIdx])==m_nStereoLabels);
            lvDbgAssert(StereoFuncID((*m_apStereoModels[nCamIdx])[oNode.anStereoUnaryFactIDs[nCamIdx]].functionIndex(),(*m_apStereoModels[nCamIdx])[oNode.anStereoUnaryFactIDs[nCamIdx]].functionType())==oNode.apStereoUnaryFuncs[nCamIdx]->first);
            InternalLabelType nEvalLabel = m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx) = 0;
            ValueType fOptimalEnergy = oNode.apStereoUnaryFuncs[nCamIdx]->second(&nEvalLabel);
            for(nEvalLabel = 1; nEvalLabel<m_nStereoLabels; ++nEvalLabel) {
                const ValueType fCurrEnergy = oNode.apStereoUnaryFuncs[nCamIdx]->second(&nEvalLabel);
                if(fOptimalEnergy>fCurrEnergy) {
                    m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx) = nEvalLabel;
                    fOptimalEnergy = fCurrEnergy;
                }
            }
        }
        m_aAssocCounts[nCamIdx] = (AssocCountType)0;
        m_aAssocMaps[nCamIdx] = (AssocIdxType)-1;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
            const InternalLabelType nLabel = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data)[nLUTNodeIdx];
            if(nLabel<m_nDontCareLabelIdx) // both special labels avoided here
                addAssoc(nCamIdx,oNode.nRowIdx,oNode.nColIdx,nLabel);
            ++vLabelCounts[nLabel];
        }
    }
    m_vStereoLabelOrdering = lv::sort_indices<InternalLabelType>(vLabelCounts,[&vLabelCounts](int a, int b){return vLabelCounts[a]>vLabelCounts[b];});
    lvDbgAssert(lv::unique(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end())==lv::make_range(InternalLabelType(0),InternalLabelType(m_nStereoLabels-1)));
}

inline void StereoSegmMatcher::GraphModelData::updateStereoModels(bool bInit) {
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below (incl xor) will break");
    lvDbgExceptionWatch;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvDbgAssert(m_apStereoModels[nCamIdx] && m_apStereoModels[nCamIdx]->numberOfVariables()==m_anValidGraphNodes[nCamIdx]);
        lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    const cv::Mat_<float> oImgAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    const cv::Mat_<float> oShpAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    lvDbgAssert(oImgAffinity.dims==3 && oImgAffinity.size[0]==nRows && oImgAffinity.size[1]==nCols && oImgAffinity.size[2]==(int)m_nRealStereoLabels);
    lvDbgAssert(oShpAffinity.dims==3 && oShpAffinity.size[0]==nRows && oShpAffinity.size[1]==nCols && oShpAffinity.size[2]==(int)m_nRealStereoLabels);
    //const CamArray<cv::Mat_<float>> aInitFGDist = {m_vNextFeats[FeatPack_LeftInitFGDist],m_vNextFeats[FeatPack_RightInitFGDist]};
    //const CamArray<cv::Mat_<float>> aInitBGDist = {m_vNextFeats[FeatPack_LeftInitBGDist],m_vNextFeats[FeatPack_RightInitBGDist]};
    //lvDbgAssert(lv::MatInfo(aInitFGDist[0])==lv::MatInfo(aInitFGDist[1]) && m_oGridSize==aInitFGDist[0].size);
    //lvDbgAssert(lv::MatInfo(aInitBGDist[0])==lv::MatInfo(aInitBGDist[1]) && m_oGridSize==aInitBGDist[0].size);
    //const CamArray<cv::Mat_<float>> aFGDist = {m_vNextFeats[FeatPack_LeftFGDist],m_vNextFeats[FeatPack_RightFGDist]};
    //const CamArray<cv::Mat_<float>> aBGDist = {m_vNextFeats[FeatPack_LeftBGDist],m_vNextFeats[FeatPack_RightBGDist]};
    //lvDbgAssert(lv::MatInfo(aFGDist[0])==lv::MatInfo(aFGDist[1]) && m_oGridSize==aFGDist[0].size);
    //lvDbgAssert(lv::MatInfo(aBGDist[0])==lv::MatInfo(aBGDist[1]) && m_oGridSize==aBGDist[0].size);
    const CamArray<cv::Mat_<uchar>> aGradY = {m_vNextFeats[FeatPack_LeftGradY],m_vNextFeats[FeatPack_RightGradY]};
    const CamArray<cv::Mat_<uchar>> aGradX = {m_vNextFeats[FeatPack_LeftGradX],m_vNextFeats[FeatPack_RightGradX]};
    const CamArray<cv::Mat_<uchar>> aGradMag = {m_vNextFeats[FeatPack_LeftGradMag],m_vNextFeats[FeatPack_RightGradMag]};
    lvDbgAssert(lv::MatInfo(aGradY[0])==lv::MatInfo(aGradY[1]) && m_oGridSize==aGradY[0].size);
    lvDbgAssert(lv::MatInfo(aGradX[0])==lv::MatInfo(aGradX[1]) && m_oGridSize==aGradX[0].size);
    lvDbgAssert(lv::MatInfo(aGradMag[0])==lv::MatInfo(aGradMag[1]) && m_oGridSize==aGradMag[0].size);
    const CamArray<cv::Mat_<float>> aImgSaliency = {m_vNextFeats[FeatPack_LeftImgSaliency],m_vNextFeats[FeatPack_RightImgSaliency]};
    const CamArray<cv::Mat_<float>> aShpSaliency = {m_vNextFeats[FeatPack_LeftShpSaliency],m_vNextFeats[FeatPack_RightShpSaliency]};
    lvDbgAssert(lv::MatInfo(aImgSaliency[0])==lv::MatInfo(aImgSaliency[1]) && m_oGridSize==aImgSaliency[0].size);
    lvDbgAssert(lv::MatInfo(aShpSaliency[0])==lv::MatInfo(aShpSaliency[1]) && m_oGridSize==aShpSaliency[0].size);
    /*const int nMinGradThrs = STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-5;
    const int nMaxGradThrs = STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST+5;
    cv::imshow("aGradY_0",(aGradY[0]>nMinGradThrs)&(aGradY[0]<nMaxGradThrs));
    cv::imshow("aGradX_0",(aGradX[0]>nMinGradThrs)&(aGradX[0]<nMaxGradThrs));
    //cv::imshow("aGradY_1",(aGradY[1]>nMinGradThrs)&(aGradY[1]<nMaxGradThrs));
    //cv::imshow("aGradX_1",(aGradX[1]>nMinGradThrs)&(aGradX[1]<nMaxGradThrs));
    cv::waitKey(0);*/
    lvLog(2,"Updating stereo graph models energy terms based on new features...");
    lv::StopWatch oLocalTimer;
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
    #if USING_OPENMP
        #pragma omp parallel for
    #endif //USING_OPENMP
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            const int nRowIdx = oNode.nRowIdx;
            const int nColIdx = oNode.nColIdx;
            // update unary terms for each grid node
            lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
            lvDbgAssert(oNode.apStereoUnaryFuncs[nCamIdx] && oNode.anStereoUnaryFactIDs[nCamIdx]!=SIZE_MAX);
            lvDbgAssert((&m_apStereoModels[nCamIdx]->getFunction<ExplicitFunction>(oNode.apStereoUnaryFuncs[nCamIdx]->first))==(&oNode.apStereoUnaryFuncs[nCamIdx]->second));
            ExplicitFunction& vUnaryStereoFunc = oNode.apStereoUnaryFuncs[nCamIdx]->second;
            lvDbgAssert(vUnaryStereoFunc.dimension()==1 && vUnaryStereoFunc.size()==m_nStereoLabels);
            lvDbgAssert__(aImgSaliency[nCamIdx](nRowIdx,nColIdx)>=-1e-6f && aImgSaliency[nCamIdx](nRowIdx,nColIdx)<=1.0f,"fImgSaliency = %1.10f @ [%d,%d]",aImgSaliency[nCamIdx](nRowIdx,nColIdx),nRowIdx,nColIdx);
            lvDbgAssert__(aShpSaliency[nCamIdx](nRowIdx,nColIdx)>=-1e-6f && aShpSaliency[nCamIdx](nRowIdx,nColIdx)<=1.0f,"fShpSaliency = %1.10f @ [%d,%d]",aShpSaliency[nCamIdx](nRowIdx,nColIdx),nRowIdx,nColIdx);
            const float fImgSaliency = std::max(aImgSaliency[nCamIdx](nRowIdx,nColIdx),0.0f);
            const float fShpSaliency = std::max(aShpSaliency[nCamIdx](nRowIdx,nColIdx),0.0f);
            for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                vUnaryStereoFunc(nLabelIdx) = ValueType(0);
                const int nOffsetColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabelIdx);
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[nCamIdx^1](nRowIdx,nOffsetColIdx)) {
                    const int nAffMapColIdx = (nCamIdx==size_t(0))?nColIdx:nOffsetColIdx;
                    const float fImgAffinity = oImgAffinity(nRowIdx,nAffMapColIdx,nLabelIdx);
                    const float fShpAffinity = oShpAffinity(nRowIdx,nAffMapColIdx,nLabelIdx);
                    lvDbgAssert__(fImgAffinity>=0.0f && fImgAffinity<=(float)M_SQRT2,"fImgAffinity = %1.10f @ [%d,%d]",fImgAffinity,nRowIdx,nColIdx);
                    lvDbgAssert__(fShpAffinity>=0.0f && fShpAffinity<=(float)M_SQRT2,"fShpAffinity = %1.10f @ [%d,%d]",fShpAffinity,nRowIdx,nColIdx);
                    vUnaryStereoFunc(nLabelIdx) += ValueType(fImgAffinity*fImgSaliency*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE);
                    vUnaryStereoFunc(nLabelIdx) += ValueType(fShpAffinity*fShpSaliency*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE);
                    vUnaryStereoFunc(nLabelIdx) = std::min(vUnaryStereoFunc(nLabelIdx),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
                }
                else {
                    vUnaryStereoFunc(nLabelIdx) = STEREOSEGMATCH_UNARY_COST_OOB_CST;
                }
            }
            vUnaryStereoFunc(m_nDontCareLabelIdx) = ValueType(10000); // @@@@ check roi, if dc set to 0, otherwise set to inf
            vUnaryStereoFunc(m_nOccludedLabelIdx) = ValueType(10000);//STEREOSEGMATCH_IMGSIM_COST_OCCLUDED_CST;
            if(bInit) { // inter-spectral pairwise term updates do not change w.r.t. segm or stereo updates
                for(size_t nOrientIdx=0; nOrientIdx<oNode.aanStereoPairwFactIDs[nCamIdx].size(); ++nOrientIdx) {
                    if(oNode.aanStereoPairwFactIDs[nCamIdx][nOrientIdx]!=SIZE_MAX) {
                        lvDbgAssert(oNode.aapStereoPairwFuncs[nCamIdx][nOrientIdx]);
                        lvDbgAssert(oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]!=SIZE_MAX);
                        lvDbgAssert(m_vNodeInfos[oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]].abValidGraphNode[nCamIdx]);
                        lvDbgAssert((&m_apStereoModels[nCamIdx]->getFunction<ExplicitFunction>(oNode.aapStereoPairwFuncs[nCamIdx][nOrientIdx]->first))==(&oNode.aapStereoPairwFuncs[nCamIdx][nOrientIdx]->second));
                        ExplicitFunction& vPairwiseStereoFunc = oNode.aapStereoPairwFuncs[nCamIdx][nOrientIdx]->second;
                        lvDbgAssert(vPairwiseStereoFunc.dimension()==2 && vPairwiseStereoFunc.size()==m_nStereoLabels*m_nStereoLabels);
                        const int nLocalGrad = (int)((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx);
                        const float fGradScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                        lvDbgAssert(fGradScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                        for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nRealStereoLabels; ++nLabelIdx1) {
                            for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nRealStereoLabels; ++nLabelIdx2) {
                                const OutputLabelType nRealLabel1 = getRealLabel(nLabelIdx1);
                                const OutputLabelType nRealLabel2 = getRealLabel(nLabelIdx2);
                                const int nRealLabelDiff = std::min(std::abs((int)nRealLabel1-(int)nRealLabel2),STEREOSEGMATCH_LBLSIM_STEREO_MAXDIFF_CST);
                                vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) = ValueType((nRealLabelDiff*nRealLabelDiff)*fGradScaleFact*STEREOSEGMATCH_LBLSIM_STEREO_SCALE_CST);
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
            }
            const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(m_nTotValidGraphNodes/40))==0 && oLocalTimer.elapsed()>2.0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/m_nTotValidGraphNodes);
                bProgressDisplayed = true;
            }
        }
    }
    if(bProgressDisplayed)
        lv::cleanConsoleRow();
    lvLog_(2,"Stereo graphs energy terms update completed in %f second(s).",oLocalTimer.tock());
}

inline void StereoSegmMatcher::GraphModelData::updateResegmModels(bool bInit) {
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below (incl xor) will break");
    lvDbgExceptionWatch;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvDbgAssert(m_apResegmModels[nCamIdx] && m_apResegmModels[nCamIdx]->numberOfVariables()==m_anValidGraphNodes[nCamIdx]);
        lvDbgAssert(m_oGridSize==m_aResegmLabelings[nCamIdx].size && m_oGridSize==m_aGMMCompAssignMap[nCamIdx].size);
        lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    const CamArray<cv::Mat_<float>> aInitFGDist = {m_vNextFeats[FeatPack_LeftInitFGDist],m_vNextFeats[FeatPack_RightInitFGDist]};
    const CamArray<cv::Mat_<float>> aInitBGDist = {m_vNextFeats[FeatPack_LeftInitBGDist],m_vNextFeats[FeatPack_RightInitBGDist]};
    lvDbgAssert(lv::MatInfo(aInitFGDist[0])==lv::MatInfo(aInitFGDist[1]) && m_oGridSize==aInitFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aInitBGDist[0])==lv::MatInfo(aInitBGDist[1]) && m_oGridSize==aInitBGDist[0].size);
    const CamArray<cv::Mat_<float>> aFGDist = {m_vNextFeats[FeatPack_LeftFGDist],m_vNextFeats[FeatPack_RightFGDist]};
    const CamArray<cv::Mat_<float>> aBGDist = {m_vNextFeats[FeatPack_LeftBGDist],m_vNextFeats[FeatPack_RightBGDist]};
    lvDbgAssert(lv::MatInfo(aFGDist[0])==lv::MatInfo(aFGDist[1]) && m_oGridSize==aFGDist[0].size);
    lvDbgAssert(lv::MatInfo(aBGDist[0])==lv::MatInfo(aBGDist[1]) && m_oGridSize==aBGDist[0].size);
    const CamArray<cv::Mat_<uchar>> aGradY = {m_vNextFeats[FeatPack_LeftGradY],m_vNextFeats[FeatPack_RightGradY]};
    const CamArray<cv::Mat_<uchar>> aGradX = {m_vNextFeats[FeatPack_LeftGradX],m_vNextFeats[FeatPack_RightGradX]};
    const CamArray<cv::Mat_<uchar>> aGradMag = {m_vNextFeats[FeatPack_LeftGradMag],m_vNextFeats[FeatPack_RightGradMag]};
    lvDbgAssert(lv::MatInfo(aGradY[0])==lv::MatInfo(aGradY[1]) && m_oGridSize==aGradY[0].size);
    lvDbgAssert(lv::MatInfo(aGradX[0])==lv::MatInfo(aGradX[1]) && m_oGridSize==aGradX[0].size);
    lvDbgAssert(lv::MatInfo(aGradMag[0])==lv::MatInfo(aGradMag[1]) && m_oGridSize==aGradMag[0].size);
    const float fInitDistScale = STEREOSEGMATCH_SHPDIST_INITDIST_SCALE;
    const float fMaxDist = STEREOSEGMATCH_SHPDIST_PX_MAX_CST;
    if(bInit) {
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
            if(oInputImg.channels()==1) {
                lv::initGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],m_aROIs[nCamIdx]);
                lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],m_aROIs[nCamIdx]);
            }
            else {// 3ch
                lv::initGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],m_aROIs[nCamIdx]);
                lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],m_aROIs[nCamIdx]);
            }
        }
    }
    else {
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
            if(oInputImg.channels()==1) {
                lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],m_aROIs[nCamIdx]);
                lv::learnGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],m_aROIs[nCamIdx]);
            }
            else { // 3ch
                lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],m_aROIs[nCamIdx]);
                lv::learnGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],m_aROIs[nCamIdx]);
            }
        }
    }
    if(lv::getVerbosity()>=4) {
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            cv::Mat_<int> oClusterLabels = m_aGMMCompAssignMap[nCamIdx].clone();
            for(size_t nNodeIdx=0; nNodeIdx<m_aResegmLabelings[nCamIdx].total(); ++nNodeIdx)
                if(((InternalLabelType*)m_aResegmLabelings[nCamIdx].data)[nNodeIdx])
                    ((int*)oClusterLabels.data)[nNodeIdx] += 1<<31;
            cv::Mat oClusterLabelsDisplay = lv::getUniqueColorMap(oClusterLabels);
            cv::imshow(std::string("gmm_clusters_")+std::to_string(nCamIdx),oClusterLabelsDisplay);
            cv::waitKey(1);
        }
    }
    lvLog(2,"Updating resegm graph models energy terms based on new features...");
    lv::StopWatch oLocalTimer;
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        const bool bUsing3ChannelInput = oInputImg.channels()==3;
        //cv::Mat_<double> oFGProbMap(m_oGridSize),oBGProbMap(m_oGridSize);
        //oFGProbMap = 0.0; oBGProbMap = 0.0;
    #if USING_OPENMP
        #pragma omp parallel for
    #endif //USING_OPENMP
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            const int nRowIdx = oNode.nRowIdx;
            const int nColIdx = oNode.nColIdx;
            // update unary terms for each grid node
            lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
            lvDbgAssert(oNode.apResegmUnaryFuncs[nCamIdx] && oNode.anResegmUnaryFactIDs[nCamIdx]!=SIZE_MAX);
            lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oNode.apResegmUnaryFuncs[nCamIdx]->first))==(&oNode.apResegmUnaryFuncs[nCamIdx]->second));
            ExplicitFunction& vUnaryResegmFunc = oNode.apResegmUnaryFuncs[nCamIdx]->second;
            lvDbgAssert(vUnaryResegmFunc.dimension()==1 && vUnaryResegmFunc.size()==s_nResegmLabels);
            const double dMinProbDensity = 1e-10;
            const double dMaxProbDensity = 1.0;
            const uchar* acInputColorSample = bUsing3ChannelInput?(oInputImg.data+nLUTNodeIdx*3):(oInputImg.data+nLUTNodeIdx);
            const float fInitFGDist = std::min(((float*)aInitFGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
            const float fCurrFGDist = std::min(((float*)aFGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
            const ValueType tFGDistUnaryCost = ValueType((fCurrFGDist+fInitFGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE);
            lvDbgAssert(tFGDistUnaryCost>=ValueType(0));
            const double dColorFGProb = std::min(std::max((bUsing3ChannelInput?m_aFGModels_3ch[nCamIdx](acInputColorSample):m_aFGModels_1ch[nCamIdx](acInputColorSample)),dMinProbDensity),dMaxProbDensity);
            //((double*)oFGProbMap.data)[nLUTNodeIdx] = dColorFGProb;
            const ValueType tFGColorUnaryCost = ValueType(-std::log(dColorFGProb)*STEREOSEGMATCH_IMGSIM_COST_COLOR_SCALE);
            lvDbgAssert(tFGColorUnaryCost>=ValueType(0));
            vUnaryResegmFunc(s_nForegroundLabelIdx) = std::min(tFGDistUnaryCost+tFGColorUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            lvDbgAssert(vUnaryResegmFunc(s_nForegroundLabelIdx)>=ValueType(0));
            const float fInitBGDist = std::min(((float*)aInitBGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
            const float fCurrBGDist = std::min(((float*)aBGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
            const ValueType tBGDistUnaryCost = ValueType((fCurrBGDist+fInitBGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE);
            lvDbgAssert(tBGDistUnaryCost>=ValueType(0));
            const double dColorBGProb = std::min(std::max((bUsing3ChannelInput?m_aBGModels_3ch[nCamIdx](acInputColorSample):m_aBGModels_1ch[nCamIdx](acInputColorSample)),dMinProbDensity),dMaxProbDensity);
            //((double*)oBGProbMap.data)[nLUTNodeIdx] = dColorBGProb;
            const ValueType tBGColorUnaryCost = ValueType(-std::log(dColorBGProb)*STEREOSEGMATCH_IMGSIM_COST_COLOR_SCALE);
            lvDbgAssert(tBGColorUnaryCost>=ValueType(0));
            vUnaryResegmFunc(s_nBackgroundLabelIdx) = std::min(tBGDistUnaryCost+tBGColorUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            lvDbgAssert(vUnaryResegmFunc(s_nBackgroundLabelIdx)>=ValueType(0));
            const InternalLabelType nStereoLabelIdx = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data)[nLUTNodeIdx];
            const int nOffsetColIdx = (nStereoLabelIdx<m_nRealStereoLabels)?getOffsetColIdx(nCamIdx,nColIdx,nStereoLabelIdx):INT_MAX;
            if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[nCamIdx^1](nRowIdx,nOffsetColIdx)) {
                const float fInitOffsetFGDist = std::min(aInitFGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
                const float fCurrOffsetFGDist = std::min(aFGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
                const ValueType tAddedFGDistUnaryCost = ValueType((fCurrOffsetFGDist+fInitOffsetFGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE*STEREOSEGMATCH_SHPDIST_INTERSPEC_SCALE);
                vUnaryResegmFunc(s_nForegroundLabelIdx) = std::min(vUnaryResegmFunc(s_nForegroundLabelIdx)+tAddedFGDistUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
                const float fInitOffsetBGDist = std::min(aInitBGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
                const float fCurrOffsetBGDist = std::min(aBGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
                const ValueType tAddedBGDistUnaryCost = ValueType((fCurrOffsetBGDist+fInitOffsetBGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE*STEREOSEGMATCH_SHPDIST_INTERSPEC_SCALE);
                vUnaryResegmFunc(s_nBackgroundLabelIdx) = std::min(vUnaryResegmFunc(s_nBackgroundLabelIdx)+tAddedBGDistUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
                for(size_t nOrientIdx=0; nOrientIdx<oNode.aanResegmPairwFactIDs[nCamIdx].size(); ++nOrientIdx) {
                    if(oNode.aanResegmPairwFactIDs[nCamIdx][nOrientIdx]!=SIZE_MAX) {
                        lvDbgAssert(oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]);
                        lvDbgAssert(oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]!=SIZE_MAX);
                        lvDbgAssert(m_vNodeInfos[oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]].abValidGraphNode[nCamIdx]);
                        lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->first))==(&oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->second));
                        ExplicitFunction& vPairwiseResegmFunc = oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->second;
                        lvDbgAssert(vPairwiseResegmFunc.dimension()==2 && vPairwiseResegmFunc.size()==s_nResegmLabels*s_nResegmLabels);
                        const int nLocalGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx));
                        const int nOffsetGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx^1]:(nOrientIdx==1)?aGradX[nCamIdx^1]:aGradMag[nCamIdx^1])(nRowIdx,nOffsetColIdx));
                        const float fLocalScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                        const float fOffsetScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nOffsetGrad);
                        lvDbgAssert(fLocalScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                        lvDbgAssert(fOffsetScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nOffsetGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                        //const float fScaleFact = std::min(fLocalScaleFact,fOffsetScaleFact);
                        const float fScaleFact = fLocalScaleFact*fOffsetScaleFact;
                        for(InternalLabelType nLabelIdx1=0; nLabelIdx1<s_nResegmLabels; ++nLabelIdx1) {
                            for(InternalLabelType nLabelIdx2=0; nLabelIdx2<s_nResegmLabels; ++nLabelIdx2) {
                                const ValueType tPairwiseCost = ValueType((nLabelIdx1^nLabelIdx2)*fScaleFact*STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST);
                                vPairwiseResegmFunc(nLabelIdx1,nLabelIdx2) = std::min(tPairwiseCost,STEREOSEGMATCH_LBLSIM_COST_MAXTRUNC_CST);
                            }
                        }
                    }
                }
            }
            else {
                vUnaryResegmFunc(s_nForegroundLabelIdx) = STEREOSEGMATCH_UNARY_COST_OOB_CST;
                vUnaryResegmFunc(s_nBackgroundLabelIdx) = ValueType(0);
                for(size_t nOrientIdx=0; nOrientIdx<oNode.aanResegmPairwFactIDs[nCamIdx].size(); ++nOrientIdx) {
                    if(oNode.aanResegmPairwFactIDs[nCamIdx][nOrientIdx]!=SIZE_MAX) {
                        lvDbgAssert(oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]);
                        lvDbgAssert(oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]!=SIZE_MAX);
                        lvDbgAssert(m_vNodeInfos[oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]].abValidGraphNode[nCamIdx]);
                        lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->first))==(&oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->second));
                        ExplicitFunction& vPairwiseResegmFunc = oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->second;
                        lvDbgAssert(vPairwiseResegmFunc.dimension()==2 && vPairwiseResegmFunc.size()==s_nResegmLabels*s_nResegmLabels);
                        const int nLocalGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx));
                        const float fLocalScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                        lvDbgAssert(fLocalScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                        for(InternalLabelType nLabelIdx1=0; nLabelIdx1<s_nResegmLabels; ++nLabelIdx1) {
                            for(InternalLabelType nLabelIdx2=0; nLabelIdx2<s_nResegmLabels; ++nLabelIdx2) {
                                const ValueType tPairwiseCost = ValueType((nLabelIdx1^nLabelIdx2)*fLocalScaleFact*STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST);
                                vPairwiseResegmFunc(nLabelIdx1,nLabelIdx2) = std::min(tPairwiseCost,STEREOSEGMATCH_LBLSIM_COST_MAXTRUNC_CST);
                            }
                        }
                    }
                }
            }
            const size_t nCurrNodeIdx = ++nProcessedNodeCount;
            if((nCurrNodeIdx%(m_nTotValidGraphNodes/40))==0 && oLocalTimer.elapsed()>2.0) {
                lv::mutex_lock_guard oLock(oPrintMutex);
                lv::updateConsoleProgressBar("\tprogress:",float(nCurrNodeIdx)/m_nTotValidGraphNodes);
                bProgressDisplayed = true;
            }
        }
        //cv::imshow(std::string("oFGProbMap_")+std::to_string(nCamIdx),oFGProbMap);
        //cv::imshow(std::string("oBGProbMap_")+std::to_string(nCamIdx),oBGProbMap);
    }
    if(bProgressDisplayed)
        lv::cleanConsoleRow();
    lvLog_(2,"Resegm graphs energy terms update completed in %f second(s).",oLocalTimer.tock());
}

inline void StereoSegmMatcher::GraphModelData::calcFeatures(const MatArrayIn& aInputs, cv::Mat* pFeatsPacket) {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"lots of hardcoded indices below");
    lvDbgExceptionWatch;
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputImg = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        lvAssert__(oInputImg.dims==2 && m_oGridSize==oInputImg.size(),"input image in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3,"unexpected input image type");
        const cv::Mat& oInputMask = aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask];
        lvAssert__(oInputMask.dims==2 && m_oGridSize==oInputMask.size(),"input mask in array at index=%d had the wrong size",(int)nCamIdx);
        lvAssert_(oInputMask.type()==CV_8UC1,"unexpected input mask type");
    }
    m_vNextFeats.resize(FeatPackSize);
    calcImageFeatures(CamArray<cv::Mat>{aInputs[InputPack_LeftImg],aInputs[InputPack_RightImg]},true);
    calcShapeFeatures(CamArray<cv::Mat_<InternalLabelType>>{aInputs[InputPack_LeftMask],aInputs[InputPack_RightMask]},true);
    for(size_t nMatIdx=0; nMatIdx<m_vNextFeats.size(); ++nMatIdx)
        lvAssert_(m_vNextFeats[nMatIdx].isContinuous(),"internal func used non-continuous data block for feature maps");
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

inline void StereoSegmMatcher::GraphModelData::calcImageFeatures(const CamArray<cv::Mat>& aInputImages, bool /*bInit*/) {
    static_assert(getCameraCount()==2,"bad input image array size");
    lvDbgExceptionWatch;
    for(size_t nInputIdx=0; nInputIdx<aInputImages.size(); ++nInputIdx) {
        lvDbgAssert__(aInputImages[nInputIdx].dims==2 && m_oGridSize==aInputImages[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputImages[nInputIdx].type()==CV_8UC1 || aInputImages[nInputIdx].type()==CV_8UC3,"unexpected input image type");
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvLog(2,"Calculating image features maps...");
    const int nWinRadius = (int)m_nGridBorderSize;
    const int nWinSize = nWinRadius*2+1;
    lvIgnore(nWinSize);
    CamArray<cv::Mat> aEnlargedInput,aDownScaledEnlargedInput;
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    // @@@@ remove downscaled once sure is bad?
    CamArray<cv::Mat_<float>> aFullScaledEnlargedDescs,aDownScaledEnlargedDescs;
    CamArray<cv::Mat_<float>> aFullScaledDescs,aDownScaledDescs;
    const cv::Size oPatchSize(STEREOSEGMATCH_DEFAULT_DESC_PATCH_WIDTH,STEREOSEGMATCH_DEFAULT_DESC_PATCH_HEIGHT);
    lvAssert_(oPatchSize.area()>=1,"patch area must be strictly positive");
    lvAssert_((oPatchSize.height%2)==1 && (oPatchSize.width%2)==1,"patch sizes must be odd (will use radius)");
#endif //STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::copyMakeBorder(aInputImages[nCamIdx],aEnlargedInput[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_DEFAULT);
        const int nDownScaledRows = std::max(1,aInputImages[nCamIdx].rows/2);
        const int nDownScaledCols = std::max(1,aInputImages[nCamIdx].cols/2);
        cv::Mat oTempInput(nDownScaledRows,nDownScaledCols,aInputImages[nCamIdx].type());
        cv::pyrDown(aInputImages[nCamIdx],oTempInput,cv::Size(nDownScaledCols,nDownScaledRows));
        cv::copyMakeBorder(oTempInput,aDownScaledEnlargedInput[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_DEFAULT);
    #if STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3) {
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
            cv::cvtColor(aDownScaledEnlargedInput[nCamIdx],aDownScaledEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
        }
    #elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3) {
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
            cv::cvtColor(aDownScaledEnlargedInput[nCamIdx],aDownScaledEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
        }
        aEnlargedInput[nCamIdx].convertTo(aEnlargedInput[nCamIdx],CV_64F,(1.0/UCHAR_MAX)/nWinSize);
        aEnlargedInput[nCamIdx] -= cv::mean(aEnlargedInput[nCamIdx])[0];
        aDownScaledEnlargedInput[nCamIdx].convertTo(aDownScaledEnlargedInput[nCamIdx],CV_64F,(1.0/UCHAR_MAX)/nWinSize);
        aDownScaledEnlargedInput[nCamIdx] -= cv::mean(aDownScaledEnlargedInput[nCamIdx])[0];
    #elif STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        lvLog_(2,"\tcam[%d] image descriptors...",(int)nCamIdx);
        m_pImgDescExtractor->compute2(aEnlargedInput[nCamIdx],aFullScaledEnlargedDescs[nCamIdx]);
        m_pImgDescExtractor->compute2(aDownScaledEnlargedInput[nCamIdx],aDownScaledEnlargedDescs[nCamIdx]);
        lvDbgAssert(aFullScaledEnlargedDescs[nCamIdx].dims==3 && aFullScaledEnlargedDescs[nCamIdx].size[0]==nRows+nWinRadius*2 && aFullScaledEnlargedDescs[nCamIdx].size[1]==nCols+nWinRadius*2);
        lvDbgAssert(aDownScaledEnlargedDescs[nCamIdx].dims==3 && aDownScaledEnlargedDescs[nCamIdx].size[0]==nDownScaledRows+nWinRadius*2 && aDownScaledEnlargedDescs[nCamIdx].size[1]==nDownScaledCols+nWinRadius*2);
        lvDbgAssert(aFullScaledEnlargedDescs[nCamIdx].size[2]==aDownScaledEnlargedDescs[nCamIdx].size[2]);
        std::vector<cv::Range> vRanges(size_t(3),cv::Range::all());
        vRanges[0] = cv::Range(nWinRadius,nRows+nWinRadius);
        vRanges[1] = cv::Range(nWinRadius,nCols+nWinRadius);
        aFullScaledDescs[nCamIdx] = aFullScaledEnlargedDescs[nCamIdx](vRanges.data());
        vRanges[0] = cv::Range(nWinRadius,nDownScaledRows+nWinRadius);
        vRanges[1] = cv::Range(nWinRadius,nDownScaledCols+nWinRadius);
        aDownScaledDescs[nCamIdx] = aDownScaledEnlargedDescs[nCamIdx](vRanges.data());
        lvDbgAssert(aFullScaledDescs[nCamIdx].dims==3 && aFullScaledDescs[nCamIdx].size[0]==nRows && aFullScaledDescs[nCamIdx].size[1]==nCols);
        lvDbgAssert(aDownScaledDescs[nCamIdx].dims==3 && aDownScaledDescs[nCamIdx].size[0]==nDownScaledRows && aDownScaledDescs[nCamIdx].size[1]==nDownScaledCols);
        lvDbgAssert(std::equal(aFullScaledDescs[nCamIdx].ptr<float>(0,0),aFullScaledDescs[nCamIdx].ptr<float>(0,0)+aFullScaledDescs[nCamIdx].size[2],aFullScaledEnlargedDescs[nCamIdx].ptr<float>(nWinRadius,nWinRadius)));
        lvDbgAssert(std::equal(aDownScaledDescs[nCamIdx].ptr<float>(0,0),aDownScaledDescs[nCamIdx].ptr<float>(0,0)+aDownScaledDescs[nCamIdx].size[2],aDownScaledEnlargedDescs[nCamIdx].ptr<float>(nWinRadius,nWinRadius)));
    #if STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        const size_t nDescSize = size_t(aFullScaledDescs[nCamIdx].size[2]);
        for(size_t nDescIdx=0; nDescIdx<aFullScaledDescs[nCamIdx].total(); nDescIdx+=nDescSize)
            lv::rootSIFT(((float*)aFullScaledDescs[nCamIdx].data)+nDescIdx,nDescSize);
        for(size_t nDescIdx=0; nDescIdx<aDownScaledDescs[nCamIdx].total(); nDescIdx+=nDescSize)
            lv::rootSIFT(((float*)aDownScaledDescs[nCamIdx].data)+nDescIdx,nDescSize);
    #endif //STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
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
        cv::imshow("gradm_0.5piv",oGradMag>STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST/2);
        cv::imshow("gradm_1.0piv",oGradMag>STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST);
        cv::imshow("gradm_2.0piv",oGradMag>STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST*2);
        cv::imshow("gradm_100",oGradMag>100);
        cv::imshow("gradm_150",oGradMag>150);
        cv::waitKey(0);*/
    }
    lvLog_(2,"Image features maps computed in %f second(s).",oLocalTimer.tock());
    lvLog(2,"Calculating image affinity map...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    cv::Mat& oAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    oAffinity.create(3,anAffinityMapDims.data(),CV_32FC1);
    oAffinity = -1.0f; // default value for OOB pixels
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    static thread_local lv::AutoBuffer<float> s_aRawAffinityData;
    s_aRawAffinityData.resize(oAffinity.total());
    cv::Mat_<float> oRawAffinity(3,anAffinityMapDims.data(),s_aRawAffinityData.data());
    oRawAffinity = -1.0f; // default value for OOB pixels
#if USING_OPENMP
    #pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            if(!m_aROIs[0](nRowIdx,nColIdx))
                continue;
            float* pRawAffinityPtr = oRawAffinity.ptr<float>(nRowIdx,nColIdx);
            for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                const int nOffsetColIdx = getOffsetColIdx(0,nColIdx,nLabelIdx);
                if(nOffsetColIdx<0 || nOffsetColIdx>=nCols || !m_aROIs[1](nRowIdx,nOffsetColIdx))
                    continue;
                const float* pFullScaledDesc = aFullScaledDescs[0].ptr<float>(nRowIdx,nColIdx);
                const float* pFullScaledOffsetDesc = aFullScaledDescs[1].ptr<float>(nRowIdx,nOffsetColIdx);
                const double dFullScaledAffinity = m_pImgDescExtractor->calcDistance(pFullScaledDesc,pFullScaledOffsetDesc);
                const float* pDownScaledDesc = aDownScaledDescs[0].ptr<float>(nRowIdx/2,nColIdx/2);
                const float* pDownScaledOffsetDesc = aDownScaledDescs[1].ptr<float>(nRowIdx/2,nOffsetColIdx/2);
                const double dDownScaledAffinity = m_pImgDescExtractor->calcDistance(pDownScaledDesc,pDownScaledOffsetDesc);
                lvIgnore(dDownScaledAffinity);
            #if STEREOSEGMATCH_CONFIG_USE_MULTILEVEL_AFFIN
                pRawAffinityPtr[nLabelIdx] = float((dFullScaledAffinity+dDownScaledAffinity)/2);
            #else //!STEREOSEGMATCH_CONFIG_USE_MULTILEVEL_AFFIN
                pRawAffinityPtr[nLabelIdx] = float(dFullScaledAffinity);
            #endif //!STEREOSEGMATCH_CONFIG_USE_MULTILEVEL_AFFIN
                lvDbgAssert(pRawAffinityPtr[nLabelIdx]>=0.0f && pRawAffinityPtr[nLabelIdx]<=(float)M_SQRT2);
            }
        }
    }
#endif //STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;
    CamArray<cv::Mat_<cv::Vec3b>> aBestLocalDispMaps;
    const bool bFillBestLocalDispMaps = lv::getVerbosity()>=4;
    if(bFillBestLocalDispMaps) {
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            aBestLocalDispMaps[nCamIdx].create(nRows,nCols);
            aBestLocalDispMaps[nCamIdx] = cv::Vec3b(0,0,255);
        }
    }
    // note: we only create the dense affinity map for 1st cam here; affinity for 2nd cam will be deduced from it
#if USING_OPENMP
    #pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            CamArray<float> afMinValidAffinityVals = {-1.0f,-1.0f};
            for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                CamArray<size_t> anValidCounts = {size_t(0),size_t(0)};
                CamArray<double> adAccumAffs = {0.0,0.0};
            #if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
                const int nPatchHeightRadius = oPatchSize.height/2;
                const int nPatchWidthRadius = oPatchSize.width/2;
                const int nColOffset = getRealLabel(nLabelIdx);
                for(int nPatchRowIdx=nRowIdx-nPatchHeightRadius; nPatchRowIdx<=nRowIdx+nPatchHeightRadius; ++nPatchRowIdx) {
                    if(nPatchRowIdx<0 || nPatchRowIdx>=nRows)
                        continue;
                    for(int nPatchColIdx=nColIdx-nPatchWidthRadius; nPatchColIdx<=nColIdx+nPatchWidthRadius; ++nPatchColIdx) {
                        if(nPatchColIdx<0 || nPatchColIdx>=nCols)
                            continue;
                        const float* pRawAffinityPtr = oRawAffinity.ptr<float>(nPatchRowIdx,nPatchColIdx);
                        if(pRawAffinityPtr[nLabelIdx]!=-1.0f) {
                            adAccumAffs[0] += pRawAffinityPtr[nLabelIdx];
                            ++anValidCounts[0];
                        }
                    }
                    for(int nOffsetPatchColIdx=nColIdx-nPatchWidthRadius+nColOffset; nOffsetPatchColIdx<=nColIdx+nPatchWidthRadius+nColOffset; ++nOffsetPatchColIdx) {
                        if(nOffsetPatchColIdx<0 || nOffsetPatchColIdx>=nCols)
                            continue;
                        const float* pRawAffinityPtr = oRawAffinity.ptr<float>(nPatchRowIdx,nOffsetPatchColIdx);
                        if(pRawAffinityPtr[nLabelIdx]!=-1.0f) {
                            adAccumAffs[1] += pRawAffinityPtr[nLabelIdx];
                            ++anValidCounts[1];
                        }
                    }
                }
            #else //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
                const int nOffsetColIdx = getOffsetColIdx(0,nColIdx,nLabelIdx);
                if(!m_aROIs[0](nRowIdx,nColIdx) || nOffsetColIdx<0 || nOffsetColIdx>=nCols || !m_aROIs[1](nRowIdx,nOffsetColIdx))
                    continue;
                anValidCounts[0] = anValidCounts[1] = size_t(1);
                const cv::Rect oWindow(nColIdx,nRowIdx,nWinSize,nWinSize);
                const cv::Rect oOffsetWindow(nOffsetColIdx,nRowIdx,nWinSize,nWinSize);
            #if STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
                const double dMutualInfoScore = m_pImgDescExtractor->compute(aEnlargedInput[0](oWindow),aEnlargedInput[1](oOffsetWindow));
                adAccumAffs[0] = adAccumAffs[1] = std::max(float(1.0-dMutualInfoScore),0.0f);
            #elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
                adAccumAffs[0] = adAccumAffs[1] = (float)cv::norm(aEnlargedInput[0](oWindow),aEnlargedInput[1](oOffsetWindow),cv::NORM_L2);
            #endif //STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
            #endif //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
                for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
                    if(anValidCounts[nCamIdx]==size_t(0))
                        continue;
                    const float fAffinity = float(adAccumAffs[nCamIdx]/anValidCounts[nCamIdx]);
                    lvDbgAssert(fAffinity>=0.0f && fAffinity<=(float)M_SQRT2);
                    if(nCamIdx==size_t(0))
                        oAffinity.at<float>(nRowIdx,nColIdx,nLabelIdx) = fAffinity;
                    if(bFillBestLocalDispMaps && (afMinValidAffinityVals[nCamIdx]==-1.0f || afMinValidAffinityVals[nCamIdx]>fAffinity)) {
                        aBestLocalDispMaps[nCamIdx](nRowIdx,nColIdx) = cv::Vec3b::all(uchar(float(nColOffset)/m_nMaxDispOffset*255));
                        afMinValidAffinityVals[nCamIdx] = fAffinity;
                    }
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
    lvLog_(2,"Image affinity map computed in %f second(s).",oLocalTimer.tock());
    lvLog(2,"Calculating image saliency maps...");
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::Mat& oSaliency = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_ImgSaliency];
        oSaliency.create(2,anAffinityMapDims.data(),CV_32FC1);
        oSaliency = 0.0f; // default value for OOB pixels
        std::vector<float> vValidAffinityVals;
        vValidAffinityVals.reserve(m_nRealStereoLabels);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            lvDbgAssert(oNode.abValidGraphNode[nCamIdx]);
            const int nRowIdx = oNode.nRowIdx;
            const int nColIdx = oNode.nColIdx;
            lvDbgAssert(m_aROIs[nCamIdx](nRowIdx,nColIdx)>0);
            vValidAffinityVals.resize(0);
            if(nCamIdx==0) {
                const float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nColIdx);
                std::copy_if(pAffinityPtr,pAffinityPtr+m_nRealStereoLabels,std::back_inserter(vValidAffinityVals),[](float v){return v>=0.0f;});
            }
            else /*nCamIdx==1*/ {
                for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                    const int nOffsetColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabelIdx);
                    if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) {
                        const float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nOffsetColIdx);
                        if(pAffinityPtr[nLabelIdx]>=0.0f)
                            vValidAffinityVals.push_back(pAffinityPtr[nLabelIdx]);
                    }
                }
            }
            const float fCurrDistSparseness = vValidAffinityVals.size()>1?(float)lv::sparseness(vValidAffinityVals.data(),vValidAffinityVals.size()):0.0f;
            const float fCurrDescSparseness = (float)lv::sparseness(aFullScaledDescs[nCamIdx].ptr<float>(nRowIdx,nColIdx),size_t(aFullScaledDescs[nCamIdx].size[2]));
            oSaliency.at<float>(nRowIdx,nColIdx) = std::max(fCurrDescSparseness,fCurrDistSparseness);
        }
        cv::normalize(oSaliency,oSaliency,1,0,cv::NORM_MINMAX,-1,m_aROIs[nCamIdx]);
        lvDbgExec( // cv::normalize leftover fp errors are sometimes awful; need to 0-max when using map
            for(int nRowIdx=0; nRowIdx<oSaliency.rows; ++nRowIdx)
                for(int nColIdx=0; nColIdx<oSaliency.cols; ++nColIdx)
                    lvDbgAssert(oSaliency.at<float>(nRowIdx,nColIdx)>=-1e-6f || m_aROIs[nCamIdx](nRowIdx,nColIdx)==0);
        );
    #if STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        cv::multiply(oSaliency,cv::Mat_<float>(oSaliency.size(),1.0f).setTo(0.5f,m_aDescROIs[nCamIdx]==0),oSaliency);
    #endif //STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        if(bFillBestLocalDispMaps) {
            cv::Mat_<cv::Vec3f> oBestLocalDispMap_good,oBestLocalDispMap_bad,oBestLocalDispMap_float;
            cv::cvtColor((1.0f-oSaliency),oBestLocalDispMap_bad,cv::COLOR_GRAY2BGR);
            cv::cvtColor(oSaliency,oBestLocalDispMap_good,cv::COLOR_GRAY2BGR);
            aBestLocalDispMaps[nCamIdx].convertTo(oBestLocalDispMap_float,CV_32F,1.0/255);
            cv::multiply(oBestLocalDispMap_bad,cv::Mat_<cv::Vec3f>(nRows,nCols,cv::Vec3f(0.f,0.f,1.f)),oBestLocalDispMap_bad);
            cv::multiply(oBestLocalDispMap_good,oBestLocalDispMap_float,oBestLocalDispMap_good);
            cv::imshow(std::string("oBestLocalDispMap_img_")+std::to_string(nCamIdx),oBestLocalDispMap_bad+oBestLocalDispMap_good);
            cv::imshow(std::string("oSaliency_img_")+std::to_string(nCamIdx),oSaliency);
            cv::waitKey(1);
        }
    }
    lvLog_(2,"Image saliency maps computed in %f second(s).",oLocalTimer.tock());
    /*if(m_pDisplayHelper) {
        std::vector<std::pair<cv::Mat,std::string>> vAffMaps;
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
            vAffMaps.push_back(std::pair<cv::Mat,std::string>(lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx)),std::string()));
        m_pDisplayHelper->displayAlbumAndWaitKey(vAffMaps);
    }*/
}

inline void StereoSegmMatcher::GraphModelData::calcShapeFeatures(const CamArray<cv::Mat_<InternalLabelType>>& aInputMasks, bool bInit) {
    static_assert(getCameraCount()==2,"bad input mask array size");
    lvDbgExceptionWatch;
    for(size_t nInputIdx=0; nInputIdx<aInputMasks.size(); ++nInputIdx) {
        lvDbgAssert__(aInputMasks[nInputIdx].dims==2 && m_oGridSize==aInputMasks[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputMasks[nInputIdx].type()==CV_8UC1,"unexpected input mask type");
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvLog(2,"Calculating shape features maps...");
    CamArray<cv::Mat_<float>> aDescs;
    const cv::Size oPatchSize(STEREOSEGMATCH_DEFAULT_DESC_PATCH_WIDTH,STEREOSEGMATCH_DEFAULT_DESC_PATCH_HEIGHT);
    lvAssert_(oPatchSize.area()>=1,"patch area must be strictly positive");
    lvAssert_((oPatchSize.height%2)==1 && (oPatchSize.width%2)==1,"patch sizes must be odd (will use radius)");
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {

        // @@@@@ use swap mask to prevent recalc of identical descriptors? (swap + radius based on desc)
        // LOW PRIORITY (segm will swap a lot every iter, might cover pretty much everything in segmented ROIs)
        // convert 8ui mask to keypoint list, then pass to compute2 (keeps desc struct/old values for untouched kps)
        // also base 8ui mask w.r.t fg dist? (2x radius = too far)
        // @@@@ use also for aff maps? or too fast to care?

        const cv::Mat& oInputMask = aInputMasks[nCamIdx];
        lvLog_(2,"\tcam[%d] shape descriptors...",(int)nCamIdx);
        m_pShpDescExtractor->compute2(oInputMask,aDescs[nCamIdx]);
        lvDbgAssert(aDescs[nCamIdx].dims==3 && aDescs[nCamIdx].size[0]==nRows && aDescs[nCamIdx].size[1]==nCols);
#if STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        const size_t nDescSize = size_t(aDescs[nCamIdx].size[2]);
        for(size_t nDescIdx=0; nDescIdx<aDescs[nCamIdx].total(); nDescIdx+=nDescSize)
            lv::rootSIFT(((float*)aDescs[nCamIdx].data)+nDescIdx,nDescSize);
#endif //STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
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
    lvLog_(2,"Shape features maps computed in %f second(s).",oLocalTimer.tock());
    lvLog(2,"Calculating shape affinity map...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    cv::Mat& oAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    oAffinity.create(3,anAffinityMapDims.data(),CV_32FC1);
    oAffinity = -1.0f; // default value for OOB pixels
    static thread_local lv::AutoBuffer<float> s_aRawAffinityData;
    s_aRawAffinityData.resize(oAffinity.total());
    cv::Mat_<float> oRawAffinity(3,anAffinityMapDims.data(),s_aRawAffinityData.data());
    oRawAffinity = -1.0f; // default value for OOB pixels
#if USING_OPENMP
    #pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            if(!m_aROIs[0](nRowIdx,nColIdx))
                continue;
            float* pRawAffinityPtr = oRawAffinity.ptr<float>(nRowIdx,nColIdx);
            for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                const int nOffsetColIdx = getOffsetColIdx(0,nColIdx,nLabelIdx);
                if(nOffsetColIdx<0 || nOffsetColIdx>=nCols || !m_aROIs[1](nRowIdx,nOffsetColIdx))
                    continue;
                const float* pDesc = aDescs[0].ptr<float>(nRowIdx,nColIdx);
                const float* pOffsetDesc = aDescs[1].ptr<float>(nRowIdx,nOffsetColIdx);
#if STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
                pRawAffinityPtr[nLabelIdx] = float(m_pShpDescExtractor->calcDistance_EMD(pDesc,pOffsetDesc));
                lvDbgAssert(pRawAffinityPtr[nLabelIdx]>=0.0f);
#else //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
                pRawAffinityPtr[nLabelIdx] = float(m_pShpDescExtractor->calcDistance_L2(pDesc,pOffsetDesc));
                lvDbgAssert(pRawAffinityPtr[nLabelIdx]>=0.0f && pRawAffinityPtr[nLabelIdx]<=(float)M_SQRT2);
#endif //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
            }
        }
    }
    std::atomic_size_t nProcessedNodeCount(size_t(0));
    std::atomic_bool bProgressDisplayed(false);
    std::mutex oPrintMutex;
    CamArray<cv::Mat_<cv::Vec3b>> aBestLocalDispMaps;
    const bool bFillBestLocalDispMaps = lv::getVerbosity()>=4;
    if(bFillBestLocalDispMaps) {
        for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
            aBestLocalDispMaps[nCamIdx].create(nRows,nCols);
            aBestLocalDispMaps[nCamIdx] = cv::Vec3b(0,0,255);
        }
    }
    // note: we only create the dense affinity map for 1st cam here; affinity for 2nd cam will be deduced from it
#if USING_OPENMP
    #pragma omp parallel for collapse(2)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            CamArray<float> afMinValidAffinityVals = {-1.0f,-1.0f};
            for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                CamArray<size_t> anValidCounts = {size_t(0),size_t(0)};
                CamArray<double> adAccumAffs = {0.0,0.0};
                const int nPatchHeightRadius = oPatchSize.height/2;
                const int nPatchWidthRadius = oPatchSize.width/2;
                const int nColOffset = getRealLabel(nLabelIdx);
                for(int nPatchRowIdx=nRowIdx-nPatchHeightRadius; nPatchRowIdx<=nRowIdx+nPatchHeightRadius; ++nPatchRowIdx) {
                    if(nPatchRowIdx<0 || nPatchRowIdx>=nRows)
                        continue;
                    for(int nPatchColIdx=nColIdx-nPatchWidthRadius; nPatchColIdx<=nColIdx+nPatchWidthRadius; ++nPatchColIdx) {
                        if(nPatchColIdx<0 || nPatchColIdx>=nCols)
                            continue;
                        const float* pRawAffinityPtr = oRawAffinity.ptr<float>(nPatchRowIdx,nPatchColIdx);
                        if(pRawAffinityPtr[nLabelIdx]!=-1.0f) {
                            adAccumAffs[0] += pRawAffinityPtr[nLabelIdx];
                            ++anValidCounts[0];
                        }
                    }
                    for(int nOffsetPatchColIdx=nColIdx-nPatchWidthRadius+nColOffset; nOffsetPatchColIdx<=nColIdx+nPatchWidthRadius+nColOffset; ++nOffsetPatchColIdx) {
                        if(nOffsetPatchColIdx<0 || nOffsetPatchColIdx>=nCols)
                            continue;
                        const float* pRawAffinityPtr = oRawAffinity.ptr<float>(nPatchRowIdx,nOffsetPatchColIdx);
                        if(pRawAffinityPtr[nLabelIdx]!=-1.0f) {
                            adAccumAffs[1] += pRawAffinityPtr[nLabelIdx];
                            ++anValidCounts[1];
                        }
                    }
                }
                for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
                    if(anValidCounts[nCamIdx]==size_t(0))
                        continue;
                    const float fAffinity = float(adAccumAffs[nCamIdx]/anValidCounts[nCamIdx]);
                    lvDbgAssert(fAffinity>=0.0f && fAffinity<=(float)M_SQRT2);
                    if(nCamIdx==size_t(0))
                        oAffinity.at<float>(nRowIdx,nColIdx,nLabelIdx) = fAffinity;
                    if(bFillBestLocalDispMaps && (afMinValidAffinityVals[nCamIdx]==-1.0f || afMinValidAffinityVals[nCamIdx]>fAffinity)) {
                        aBestLocalDispMaps[nCamIdx](nRowIdx,nColIdx) = cv::Vec3b::all(uchar(float(nColOffset)/m_nMaxDispOffset*255));
                        afMinValidAffinityVals[nCamIdx] = fAffinity;
                    }
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
    lvLog_(2,"Shape affinity map computed in %f second(s).",oLocalTimer.tock());
    lvLog(2,"Calculating shape saliency maps...");
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::Mat& oSaliency = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_ShpSaliency];
        oSaliency.create(2,anAffinityMapDims.data(),CV_32FC1);
        oSaliency = 0.0f; // default value for OOB pixels
        std::vector<float> vValidAffinityVals;
        vValidAffinityVals.reserve(m_nRealStereoLabels);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            lvDbgAssert(oNode.abValidGraphNode[nCamIdx]);
            const int nRowIdx = oNode.nRowIdx;
            const int nColIdx = oNode.nColIdx;
            vValidAffinityVals.resize(0);
            if(nCamIdx==0) {
                const float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nColIdx);
                std::copy_if(pAffinityPtr,pAffinityPtr+m_nRealStereoLabels,std::back_inserter(vValidAffinityVals),[](float v){return v>=0.0f;});
            }
            else /*nCamIdx==1*/ {
                for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                    const int nOffsetColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabelIdx);
                    if(nOffsetColIdx>=0 && nOffsetColIdx<nCols) {
                        const float* pAffinityPtr = oAffinity.ptr<float>(nRowIdx,nOffsetColIdx);
                        if(pAffinityPtr[nLabelIdx]>=0.0f)
                            vValidAffinityVals.push_back(pAffinityPtr[nLabelIdx]);
                    }
                }
            }
            const float fCurrDistSparseness = vValidAffinityVals.size()>1?(float)lv::sparseness(vValidAffinityVals.data(),vValidAffinityVals.size()):0.0f;
            const float fCurrDescSparseness = (float)lv::sparseness(aDescs[nCamIdx].ptr<float>(nRowIdx,nColIdx),size_t(aDescs[nCamIdx].size[2]));
            oSaliency.at<float>(nRowIdx,nColIdx) = std::max(fCurrDescSparseness,fCurrDistSparseness);
        #if STEREOSEGMATCH_DEFAULT_SALIENT_SHP_RAD>0
            const cv::Mat& oFGDist = m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist];
            const float fCurrFGDist = oFGDist.at<float>(nRowIdx,nColIdx);
            oSaliency.at<float>(nRowIdx,nColIdx) *= std::max(1-fCurrFGDist/STEREOSEGMATCH_DEFAULT_SALIENT_SHP_RAD,0.0f);
        #endif //STEREOSEGMATCH_DEFAULT_SALIENT_SHP_RAD>0
        }
        cv::normalize(oSaliency,oSaliency,1,0,cv::NORM_MINMAX,-1,m_aROIs[nCamIdx]);
        lvDbgExec( // cv::normalize leftover fp errors are sometimes awful; need to 0-max when using map
            for(int nRowIdx=0; nRowIdx<oSaliency.rows; ++nRowIdx)
                for(int nColIdx=0; nColIdx<oSaliency.cols; ++nColIdx)
                    lvDbgAssert(oSaliency.at<float>(nRowIdx,nColIdx)>=-1e-6f || m_aROIs[nCamIdx](nRowIdx,nColIdx)==0);
        );
    #if STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        cv::multiply(oSaliency,cv::Mat_<float>(oSaliency.size(),1.0f).setTo(0.5f,m_aDescROIs[nCamIdx]==0),oSaliency);
    #endif //STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        if(bFillBestLocalDispMaps) {
            cv::Mat_<cv::Vec3f> oBestLocalDispMap_good,oBestLocalDispMap_bad,oBestLocalDispMap_float;
            cv::cvtColor((1.0f-oSaliency),oBestLocalDispMap_bad,cv::COLOR_GRAY2BGR);
            cv::cvtColor(oSaliency,oBestLocalDispMap_good,cv::COLOR_GRAY2BGR);
            aBestLocalDispMaps[nCamIdx].convertTo(oBestLocalDispMap_float,CV_32F,1.0/255);
            cv::multiply(oBestLocalDispMap_bad,cv::Mat_<cv::Vec3f>(nRows,nCols,cv::Vec3f(0.f,0.f,1.f)),oBestLocalDispMap_bad);
            cv::multiply(oBestLocalDispMap_good,oBestLocalDispMap_float,oBestLocalDispMap_good);
            cv::imshow(std::string("oBestLocalDispMap_shp_")+std::to_string(nCamIdx),oBestLocalDispMap_bad+oBestLocalDispMap_good);
            cv::imshow(std::string("oSaliency_shp_")+std::to_string(nCamIdx),oSaliency);
            cv::waitKey(1);
        }
    }
    lvLog_(2,"Shape saliency maps computed in %f second(s).",oLocalTimer.tock());
    /*if(m_pDisplayHelper) {
        std::vector<std::pair<cv::Mat,std::string>> vAffMaps;
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
            vAffMaps.push_back(std::pair<cv::Mat,std::string>(lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx)),std::string()));
        m_pDisplayHelper->displayAlbumAndWaitKey(vAffMaps);
    }*/
}

inline void StereoSegmMatcher::GraphModelData::setNextFeatures(const cv::Mat& oPackedFeats) {
    lvDbgExceptionWatch;
    lvAssert_(!oPackedFeats.empty() && oPackedFeats.isContinuous(),"features packet must be non-empty and continuous");
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
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_ImgSaliency] = lv::MatInfo(m_oGridSize,CV_32FC1);
            m_vExpectedFeatPackInfo[nCamIdx*FeatPackOffset+FeatPackOffset_ShpSaliency] = lv::MatInfo(m_oGridSize,CV_32FC1);
        }
        const int nRows = (int)m_oGridSize(0);
        const int nCols = (int)m_oGridSize(1);
        m_vExpectedFeatPackInfo[FeatPack_ImgAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
        m_vExpectedFeatPackInfo[FeatPack_ShpAffinity] = lv::MatInfo(std::array<int,3>{nRows,nCols,(int)m_nRealStereoLabels},CV_32FC1);
    }
    m_oNextPackedFeats = oPackedFeats; // get pointer to data instead of copy; provider should not overwrite until next 'apply' call!
    m_vNextFeats = lv::unpackData(m_oNextPackedFeats,m_vExpectedFeatPackInfo);
    for(size_t nMatIdx=0; nMatIdx<m_vNextFeats.size(); ++nMatIdx)
        lvAssert_(m_vNextFeats[nMatIdx].isContinuous(),"internal func used non-continuous data block for feature maps");
    m_bUsePrecalcFeatsNext = true;
}

inline StereoSegmMatcher::OutputLabelType StereoSegmMatcher::GraphModelData::getRealLabel(InternalLabelType nLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nLabel<m_vStereoLabels.size());
    return m_vStereoLabels[nLabel];
}

inline StereoSegmMatcher::InternalLabelType StereoSegmMatcher::GraphModelData::getInternalLabel(OutputLabelType nRealLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nRealLabel==s_nOccludedLabel || nRealLabel==s_nDontCareLabel);
    lvDbgAssert(nRealLabel>=(OutputLabelType)m_nMinDispOffset && nRealLabel<=(OutputLabelType)m_nMaxDispOffset);
    lvDbgAssert(((nRealLabel-m_nMinDispOffset)%m_nDispOffsetStep)==0);
    return (InternalLabelType)std::distance(m_vStereoLabels.begin(),std::find(m_vStereoLabels.begin(),m_vStereoLabels.end(),nRealLabel));
}

inline int StereoSegmMatcher::GraphModelData::getOffsetColIdx(size_t nCamIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded offset sign");
    lvDbgExceptionWatch;
    lvDbgAssert(nCamIdx==size_t(0) || nCamIdx==size_t(1));
    lvDbgAssert(nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    lvDbgAssert(nLabel<m_nRealStereoLabels);
    const OutputLabelType nRealLabel = getRealLabel(nLabel);
    lvDbgAssert((int)nRealLabel>=(int)m_nMinDispOffset && (int)nRealLabel<=(int)m_nMaxDispOffset);
    return (nCamIdx==size_t(0))?(nColIdx-nRealLabel):(nColIdx+nRealLabel);
}

inline StereoSegmMatcher::AssocCountType StereoSegmMatcher::GraphModelData::getAssocCount(size_t nCamIdx, int nRowIdx, int nColIdx) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0]);
    lvDbgAssert(nCamIdx==1 || (nColIdx>=-(int)m_nMaxDispOffset && nColIdx<(int)m_oGridSize[1]));
    lvDbgAssert(nCamIdx==0 || (nColIdx>=0 && nColIdx<int(m_oGridSize[1]+m_nMaxDispOffset)));
    lvDbgAssert(m_aAssocCounts[nCamIdx].dims==2 && !m_aAssocCounts[nCamIdx].empty() && m_aAssocCounts[nCamIdx].isContinuous());
    const size_t nMapOffset = ((nCamIdx==size_t(1))?size_t(0):m_nMaxDispOffset);
    return ((AssocCountType*)m_aAssocCounts[nCamIdx].data)[nRowIdx*m_aAssocCounts[nCamIdx].cols + (nColIdx+nMapOffset)/m_nDispOffsetStep];
}

inline void StereoSegmMatcher::GraphModelData::addAssoc(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nLabel<m_nDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabel);
    lvDbgAssert(nCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
    lvDbgAssert(nCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
    const size_t nMapOffset = ((nCamIdx==size_t(1))?size_t(0):m_nMaxDispOffset);
    lvDbgAssert(m_aAssocMaps[nCamIdx].dims==3 && !m_aAssocMaps[nCamIdx].empty() && m_aAssocMaps[nCamIdx].isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_aAssocMaps[nCamIdx].data)+(nRowIdx*m_aAssocMaps[nCamIdx].size[1] + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep)*m_aAssocMaps[nCamIdx].size[2];
    lvDbgAssert(pAssocList==m_aAssocMaps[nCamIdx].ptr<AssocIdxType>(nRowIdx,(nAssocColIdx+nMapOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_aAssocMaps[nCamIdx].size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(-1));
    pAssocList[nListOffset] = AssocIdxType(nColIdx);
    lvDbgAssert(m_aAssocCounts[nCamIdx].dims==2 && !m_aAssocCounts[nCamIdx].empty() && m_aAssocCounts[nCamIdx].isContinuous());
    AssocCountType& nAssocCount = ((AssocCountType*)m_aAssocCounts[nCamIdx].data)[nRowIdx*m_aAssocCounts[nCamIdx].cols + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_aAssocCounts[nCamIdx](nRowIdx,int((nAssocColIdx+nMapOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount<std::numeric_limits<AssocCountType>::max());
    ++nAssocCount;
}

inline void StereoSegmMatcher::GraphModelData::removeAssoc(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nLabel<m_nDontCareLabelIdx);
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    const int nAssocColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabel);
    lvDbgAssert(nCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
    lvDbgAssert(nCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
    const size_t nMapOffset = ((nCamIdx==size_t(1))?size_t(0):m_nMaxDispOffset);
    lvDbgAssert(m_aAssocMaps[nCamIdx].dims==3 && !m_aAssocMaps[nCamIdx].empty() && m_aAssocMaps[nCamIdx].isContinuous());
    AssocIdxType* const pAssocList = ((AssocIdxType*)m_aAssocMaps[nCamIdx].data)+(nRowIdx*m_aAssocMaps[nCamIdx].size[1] + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep)*m_aAssocMaps[nCamIdx].size[2];
    lvDbgAssert(pAssocList==m_aAssocMaps[nCamIdx].ptr<AssocIdxType>(nRowIdx,(nAssocColIdx+nMapOffset)/m_nDispOffsetStep));
    const size_t nListOffset = size_t(nLabel)*m_nDispOffsetStep + nColIdx%m_nDispOffsetStep;
    lvDbgAssert((int)nListOffset<m_aAssocMaps[nCamIdx].size[2]);
    lvDbgAssert(pAssocList[nListOffset]==AssocIdxType(nColIdx));
    pAssocList[nListOffset] = AssocIdxType(-1);
    lvDbgAssert(m_aAssocCounts[nCamIdx].dims==2 && !m_aAssocCounts[nCamIdx].empty() && m_aAssocCounts[nCamIdx].isContinuous());
    AssocCountType& nAssocCount = ((AssocCountType*)m_aAssocCounts[nCamIdx].data)[nRowIdx*m_aAssocCounts[nCamIdx].cols + (nAssocColIdx+nMapOffset)/m_nDispOffsetStep];
    lvDbgAssert((&nAssocCount)==&m_aAssocCounts[nCamIdx](nRowIdx,int((nAssocColIdx+nMapOffset)/m_nDispOffsetStep)));
    lvDbgAssert(nAssocCount>AssocCountType(0));
    --nAssocCount;
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcAddAssocCost(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nDontCareLabelIdx) {
        const int nAssocColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabel);
        lvDbgAssert(nCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
        lvDbgAssert(nCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
        const AssocCountType nAssocCount = getAssocCount(nCamIdx,nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount<m_aAssocCostApproxAddLUT.size());
        return m_aAssocCostApproxAddLUT[nAssocCount];
    }
    return ValueType(100000); // @@@@ dirty
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcRemoveAssocCost(size_t nCamIdx, int nRowIdx, int nColIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded assoc range check");
    lvDbgExceptionWatch;
    lvDbgAssert(nRowIdx>=0 && nRowIdx<(int)m_oGridSize[0] && nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    if(nLabel<m_nDontCareLabelIdx) {
        const int nAssocColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabel);
        lvDbgAssert(nCamIdx==1 || (nAssocColIdx<=int(nColIdx-m_nMinDispOffset) && nAssocColIdx>=int(nColIdx-m_nMaxDispOffset)));
        lvDbgAssert(nCamIdx==0 || (nAssocColIdx>=int(nColIdx+m_nMinDispOffset) && nAssocColIdx<=int(nColIdx+m_nMaxDispOffset)));
        const AssocCountType nAssocCount = getAssocCount(nCamIdx,nRowIdx,nAssocColIdx);
        lvDbgAssert(nAssocCount>0); // cannot be zero, must have at least an association in order to remove it
        lvDbgAssert(nAssocCount<m_aAssocCostApproxAddLUT.size());
        return m_aAssocCostApproxRemLUT[nAssocCount];
    }
    return -ValueType(100000); // @@@@ dirty
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcTotalAssocCost(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
    ValueType tEnergy = ValueType(0);
    const int nColIdxStart = ((nCamIdx==size_t(1))?int(m_nMinDispOffset):-int(m_nMaxDispOffset));
    const int nColIdxEnd = ((nCamIdx==size_t(1))?int(m_oGridSize[1]+m_nMaxDispOffset):int(m_oGridSize[1]-m_nMinDispOffset));
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx)
        for(int nColIdx=nColIdxStart; nColIdx<nColIdxEnd; nColIdx+=m_nDispOffsetStep)
            tEnergy += m_aAssocCostRealSumLUT[getAssocCount(nCamIdx,nRowIdx,nColIdx)];
    lvDbgAssert(tEnergy>=ValueType(0));
    return tEnergy;
}

inline void StereoSegmMatcher::GraphModelData::calcStereoMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1);
    lvDbgAssert(m_oGridSize==m_aAssocCosts[nCamIdx].size && m_oGridSize==m_aStereoUnaryCosts[nCamIdx].size && m_oGridSize==m_aStereoPairwCosts[nCamIdx].size);
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data);
    // @@@@@ openmp here?
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const InternalLabelType& nInitLabel = pLabeling[nLUTNodeIdx];
        lvDbgAssert(&nInitLabel==&m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        ValueType& tAssocCost = ((ValueType*)m_aAssocCosts[nCamIdx].data)[nLUTNodeIdx];
        lvDbgAssert(&tAssocCost==&m_aAssocCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        ValueType& tUnaryCost = ((ValueType*)m_aStereoUnaryCosts[nCamIdx].data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_aStereoUnaryCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        if(nInitLabel!=nNewLabel) {
            const ValueType tAssocEnergyCost = calcRemoveAssocCost(nCamIdx,oNode.nRowIdx,oNode.nColIdx,nInitLabel)+calcAddAssocCost(nCamIdx,oNode.nRowIdx,oNode.nColIdx,nNewLabel);
            const ValueType tUnaryEnergyInit = oNode.apStereoUnaryFuncs[nCamIdx]->second(&nInitLabel);
            const ValueType tUnaryEnergyModif = oNode.apStereoUnaryFuncs[nCamIdx]->second(&nNewLabel);
            // @@@@@ merge assoc + vissim back into one (unary) at final cleanup
            tUnaryCost = tUnaryEnergyModif-tUnaryEnergyInit;
            tAssocCost = tAssocEnergyCost;
        }
        else
            tAssocCost = tUnaryCost = ValueType(0);
        // @@@@ CODE BELOW IS ONLY FOR DEBUG/DISPLAY
        ValueType& tPairwCost = ((ValueType*)m_aStereoPairwCosts[nCamIdx].data)[nLUTNodeIdx];
        lvDbgAssert(&tPairwCost==&m_aStereoPairwCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        tPairwCost = ValueType(0);
        for(size_t nOrientIdx=0; nOrientIdx<oNode.aanStereoPairwFactIDs[nCamIdx].size(); ++nOrientIdx) {
            if(oNode.aanStereoPairwFactIDs[nCamIdx][nOrientIdx]!=SIZE_MAX) {
                lvDbgAssert(oNode.aapStereoPairwFuncs[nCamIdx][nOrientIdx] && oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]<m_oGridSize.total());
                std::array<InternalLabelType,2> aLabels = {nInitLabel,pLabeling[oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]]};
                const ValueType tPairwEnergyInit = oNode.aapStereoPairwFuncs[nCamIdx][nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyInit>=ValueType(0));
                aLabels[0] = nNewLabel;
                const ValueType tPairwEnergyModif = oNode.aapStereoPairwFuncs[nCamIdx][nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyModif>=ValueType(0));
                tPairwCost += tPairwEnergyModif-tPairwEnergyInit;
            }
        }
    }
}

inline void StereoSegmMatcher::GraphModelData::calcResegmMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1 && m_oGridSize==m_aResegmLabelings[nCamIdx].size);
    lvDbgAssert(m_oGridSize==m_aResegmUnaryCosts[nCamIdx].size && m_oGridSize==m_aResegmPairwCosts[nCamIdx].size);
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aResegmLabelings[nCamIdx].data);
    // @@@@@ openmp here?
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const InternalLabelType& nInitLabel = pLabeling[nLUTNodeIdx];
        lvDbgAssert(&nInitLabel==&m_aResegmLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        lvDbgAssert(nInitLabel==s_nForegroundLabelIdx || nInitLabel==s_nBackgroundLabelIdx);
        ValueType& tUnaryCost = ((ValueType*)m_aResegmUnaryCosts[nCamIdx].data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_aResegmUnaryCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        if(nInitLabel!=nNewLabel) {
            const ValueType tEnergyInit = oNode.apResegmUnaryFuncs[nCamIdx]->second(&nInitLabel);
            const ValueType tEnergyModif = oNode.apResegmUnaryFuncs[nCamIdx]->second(&nNewLabel);
            tUnaryCost = tEnergyModif-tEnergyInit;
        }
        else
            tUnaryCost = ValueType(0);
        // @@@@ CODE BELOW IS ONLY FOR DEBUG/DISPLAY
        ValueType& tPairwCost = ((ValueType*)m_aResegmPairwCosts[nCamIdx].data)[nLUTNodeIdx];
        lvDbgAssert(&tPairwCost==&m_aResegmPairwCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        tPairwCost = ValueType(0);
        for(size_t nOrientIdx=0; nOrientIdx<oNode.aanResegmPairwFactIDs[nCamIdx].size(); ++nOrientIdx) {
            if(oNode.aanResegmPairwFactIDs[nCamIdx][nOrientIdx]!=SIZE_MAX) {
                lvDbgAssert(oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx] && oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]<m_oGridSize.total());
                std::array<InternalLabelType,2> aLabels = {nInitLabel,pLabeling[oNode.aanPairwLUTNodeIdxs[nCamIdx][nOrientIdx]]};
                const ValueType tPairwEnergyInit = oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyInit>=ValueType(0));
                aLabels[0] = nNewLabel;
                const ValueType tPairwEnergyModif = oNode.aapResegmPairwFuncs[nCamIdx][nOrientIdx]->second(aLabels.data());
                lvDbgAssert(tPairwEnergyModif>=ValueType(0));
                tPairwCost += tPairwEnergyModif-tPairwEnergyInit;
            }
        }
    }
}

inline opengm::InferenceTermination StereoSegmMatcher::GraphModelData::infer(size_t nPrimaryCamIdx) {
    lvDbgExceptionWatch;
    lvAssert(nPrimaryCamIdx==size_t(0) || nPrimaryCamIdx==size_t(1));
    updateStereoModels(true);
    resetLabelings();
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvDbgAssert(m_oGridSize.dims()==2 && m_oGridSize==m_aStereoLabelings[nCamIdx].size && m_oGridSize==m_aResegmLabelings[nCamIdx].size);
        lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
        lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
    }
    //const cv::Rect oDbgRect(0,0,int(m_oGridSize[1]),int(m_oGridSize[0]));
    const cv::Rect oDbgRect(0,160,int(m_oGridSize[1]),1);
    //const cv::Rect oDbgRect(0,128,int(m_oGridSize[1]),1);
    lvIgnore(oDbgRect);
    lvLog_(2,"Running inference for primary camera idx=%d...",(int)nPrimaryCamIdx);
    if(lv::getVerbosity()>=3) {
        cv::Mat oTargetImg = m_aInputs[nPrimaryCamIdx*InputPackOffset+InputPackOffset_Img].clone();
        if(oTargetImg.channels()==1)
            cv::cvtColor(oTargetImg,oTargetImg,cv::COLOR_GRAY2BGR);
        cv::rectangle(oTargetImg,oDbgRect,cv::Scalar_<uchar>(0,255,0));
        cv::Mat oTargetMask = m_aInputs[nPrimaryCamIdx*InputPackOffset+InputPackOffset_Mask].clone();
        cv::cvtColor(oTargetMask,oTargetMask,cv::COLOR_GRAY2BGR);
        oTargetMask &= cv::Vec3b(255,0,0);
        cv::imshow("target input",(oTargetImg+oTargetMask)/2);
        cv::Mat oOtherImg = m_aInputs[(nPrimaryCamIdx^1)*InputPackOffset+InputPackOffset_Img].clone();
        if(oOtherImg.channels()==1)
            cv::cvtColor(oOtherImg,oOtherImg,cv::COLOR_GRAY2BGR);
        cv::Mat oOtherMask = m_aInputs[(nPrimaryCamIdx^1)*InputPackOffset+InputPackOffset_Mask].clone();
        cv::cvtColor(oOtherMask,oOtherMask,cv::COLOR_GRAY2BGR);
        oOtherMask &= cv::Vec3b(255,0,0);
        cv::imshow("other input",(oOtherImg+oOtherMask)/2);
        cv::waitKey(1);
    }
    // @@@@@@ use member-alloc QPBO here instead of stack
    kolmogorov::qpbo::QPBO<ValueType> oStereoMinimizer((int)m_anValidGraphNodes[nPrimaryCamIdx],0); // @@@@@@@ preset max edge count using max clique size times node count
    kolmogorov::qpbo::QPBO<ValueType> oResegmMinimizer((int)m_anValidGraphNodes[nPrimaryCamIdx],0); // @@@@@@@ preset max edge count using max clique size times node count
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
                aCliqueLabels[nVarIdx] = (nAssignIdx&(1<<nVarIdx))?nAlphaLabel:aLabeling[m_avValidLUTNodeIdxs[nPrimaryCamIdx][oGraphFactor.variableIndex(nVarIdx)]];
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
    ValueType tLastStereoEnergy = m_apStereoInfs[nPrimaryCamIdx]->value();
    ValueType tLastResegmEnergy = std::numeric_limits<ValueType>::max();
    cv::Mat_<InternalLabelType>& oCurrStereoLabeling = m_aStereoLabelings[nPrimaryCamIdx];
    cv::Mat_<InternalLabelType>& oCurrResegmLabeling = m_aResegmLabelings[nPrimaryCamIdx];
    bool bJustUpdatedSegm = false;
    bool bResegmModelInitialized = false;
    // each iter below is an alpha-exp move based on A. Fix's primal-dual energy minimization method for higher-order MRFs
    // see "A Primal-Dual Algorithm for Higher-Order Multilabel Markov Random Fields" in CVPR2014 for more info (doi = 10.1109/CVPR.2014.149)
    ValueType tLastStereoAssocEnergy = calcTotalAssocCost(nPrimaryCamIdx); // @@@@ costly, and temporary, for dbg (to remove)
    while(++nMoveIter<=m_nMaxMoveIterCount && nConsecUnchangedLabels<m_nStereoLabels) {
        lvLog_(2,"\tdisp inf w/ lblidx=%d   [iter #%d]",(int)nStereoAlphaLabel,(int)nMoveIter);
        const bool bNullifyStereoPairwCosts = (STEREOSEGMATCH_CONFIG_USE_UNARY_ONLY_FIRST)&&nMoveIter<=m_nStereoLabels;
        calcStereoMoveCosts(nPrimaryCamIdx,nStereoAlphaLabel);
        if(lv::getVerbosity()>=5) {
            lvCout << "-----\n\n\n";
            if(m_aInputs[InputPack_LeftImg].channels()==1 && m_aInputs[InputPack_RightImg].channels()==1) {
                lvCout << "inputa = " << lv::to_string(m_aInputs[InputPack_LeftImg](oDbgRect),oDbgRect.tl()) << '\n';
                lvCout << "inputb = " << lv::to_string(m_aInputs[InputPack_RightImg](oDbgRect),oDbgRect.tl()) << '\n';
            }
            lvCout << "disp = " << lv::to_string(oCurrStereoLabeling(oDbgRect),oDbgRect.tl()) << '\n';
            cv::Mat_<InternalLabelType> oStereoLabelingReal = oCurrStereoLabeling.clone();
            std::transform(oStereoLabelingReal.begin(),oStereoLabelingReal.end(),oStereoLabelingReal.begin(),[&](InternalLabelType n){return (InternalLabelType)getRealLabel(n);});
            lvCout << "disp (real) = " << lv::to_string(oStereoLabelingReal(oDbgRect),oDbgRect.tl()) << '\n';
            lvCout << "assoc_cost = " << lv::to_string(m_aAssocCosts[nPrimaryCamIdx](oDbgRect),oDbgRect.tl()) << '\n';
            lvCout << "unary = " << lv::to_string(m_aStereoUnaryCosts[nPrimaryCamIdx](oDbgRect),oDbgRect.tl()) << '\n';
            lvCout << "pairw = " << lv::to_string(m_aStereoPairwCosts[nPrimaryCamIdx](oDbgRect),oDbgRect.tl()) << '\n';
            lvCout << "img-saliency = " << lv::to_string(m_vNextFeats[nPrimaryCamIdx*FeatPackOffset+FeatPackOffset_ImgSaliency](oDbgRect),oDbgRect.tl()) << '\n';
            lvCout << "shp-saliency = " << lv::to_string(m_vNextFeats[nPrimaryCamIdx*FeatPackOffset+FeatPackOffset_ShpSaliency](oDbgRect),oDbgRect.tl()) << '\n';
            if(nStereoAlphaLabel<m_nRealStereoLabels) {
                lvCout << "img affin next = " << lv::to_string(lv::squeeze(lv::getSubMat(m_vNextFeats[FeatPack_ImgAffinity],2,nStereoAlphaLabel))(oDbgRect),oDbgRect.tl()) << '\n';
                lvCout << "shp affin next = " << lv::to_string(lv::squeeze(lv::getSubMat(m_vNextFeats[FeatPack_ShpAffinity],2,nStereoAlphaLabel))(oDbgRect),oDbgRect.tl()) << '\n';
            }
            lvCout << "upcoming inf w/ disp lblidx=" << (int)nStereoAlphaLabel << "   (real=" << (int)getRealLabel(nStereoAlphaLabel) << ")\n";
            cv::Mat oCurrAssocCountsDisplay = getAssocCountsMapDisplay(nPrimaryCamIdx);
            if(oCurrAssocCountsDisplay.size().area()<640*480)
                cv::resize(oCurrAssocCountsDisplay,oCurrAssocCountsDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow("assoc_counts",oCurrAssocCountsDisplay);
            cv::Mat oCurrLabelingDisplay = getStereoDispMapDisplay(nPrimaryCamIdx);
            cv::rectangle(oCurrLabelingDisplay,oDbgRect,cv::Scalar_<uchar>(0,255,0));
            if(oCurrLabelingDisplay.size().area()<640*480)
                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow("disp",oCurrLabelingDisplay);
            cv::waitKey(0);
        }
        HOEReducer oStereoReducer; // @@@@@@ check if HigherOrderEnergy::Clear() can be used instead of realloc
        oStereoReducer.AddVars((int)m_anValidGraphNodes[nPrimaryCamIdx]);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            // manually add 1st order factors while evaluating new assoc energy
            if(oNode.anStereoUnaryFactIDs[nPrimaryCamIdx]!=SIZE_MAX) {
                const ValueType& tAssocCost = ((ValueType*)m_aAssocCosts[nPrimaryCamIdx].data)[nLUTNodeIdx];
                lvDbgAssert(&tAssocCost==&m_aAssocCosts[nPrimaryCamIdx](oNode.nRowIdx,oNode.nColIdx));
                const ValueType& tUnaryCost = ((ValueType*)m_aStereoUnaryCosts[nPrimaryCamIdx].data)[nLUTNodeIdx];
                lvDbgAssert(&tUnaryCost==&m_aStereoUnaryCosts[nPrimaryCamIdx](oNode.nRowIdx,oNode.nColIdx));
                oStereoReducer.AddUnaryTerm((int)nGraphNodeIdx,tAssocCost+tUnaryCost);
            }
            if(!bNullifyStereoPairwCosts) {
                // now add 2nd order & higher order factors via lambda
                if(oNode.aanStereoPairwFactIDs[nPrimaryCamIdx][0]!=SIZE_MAX)
                    lFactorReducer(m_apStereoModels[nPrimaryCamIdx]->operator[](oNode.aanStereoPairwFactIDs[nPrimaryCamIdx][0]),2,oStereoReducer,nStereoAlphaLabel,(InternalLabelType*)oCurrStereoLabeling.data);
                if(oNode.aanStereoPairwFactIDs[nPrimaryCamIdx][1]!=SIZE_MAX)
                    lFactorReducer(m_apStereoModels[nPrimaryCamIdx]->operator[](oNode.aanStereoPairwFactIDs[nPrimaryCamIdx][1]),2,oStereoReducer,nStereoAlphaLabel,(InternalLabelType*)oCurrStereoLabeling.data);
                // @@@@@ add higher o facts here (3-conn on epi lines?)
            }
        }
        oStereoMinimizer.Reset();
        oStereoReducer.ToQuadratic(oStereoMinimizer); // @@@@@ internally might call addNodes/edges to QPBO object; optim tbd
        //oStereoMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11); // can still add more if needed
        oStereoMinimizer.Solve();
        oStereoMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
        const cv::Mat_<AssocCountType> oLastAssocCounts = m_aAssocCounts[nPrimaryCamIdx].clone();
        const cv::Mat_<InternalLabelType> oLastStereoLabeling = oCurrStereoLabeling.clone();
        size_t nChangedStereoLabels = 0;
        cv::Mat_<uchar> oDisparitySwaps(m_oGridSize(),0);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const int nMoveLabel = oStereoMinimizer.GetLabel((int)nGraphNodeIdx);
            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
            if(nMoveLabel==1) { // node label changed to alpha
                const int nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
                const int nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
                oDisparitySwaps(nRowIdx,nColIdx) = 255;
                const InternalLabelType nOldLabel = oCurrStereoLabeling(nRowIdx,nColIdx);
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nOldLabel);
                oCurrStereoLabeling(nRowIdx,nColIdx) = nStereoAlphaLabel;
                if(nStereoAlphaLabel<m_nDontCareLabelIdx)
                    addAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nStereoAlphaLabel);
                ++nChangedStereoLabels;
            }
        }
        if(lv::getVerbosity()>=3) {
            cv::Mat oCurrLabelingDisplay = getStereoDispMapDisplay(nPrimaryCamIdx);
            cv::rectangle(oCurrLabelingDisplay,oDbgRect,cv::Scalar_<uchar>(0,255,0));
            if(oCurrLabelingDisplay.size().area()<640*480)
                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow("disp",oCurrLabelingDisplay);
            if(lv::getVerbosity()>=4) {
                cv::Mat oDisparitySwapsDisplay = oDisparitySwaps.clone();
                if(oDisparitySwapsDisplay.size().area()<640*480)
                    cv::resize(oDisparitySwapsDisplay,oDisparitySwapsDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                cv::imshow("disp-swaps",oDisparitySwapsDisplay);
            }
            if(lv::getVerbosity()>=5) {
                lvCout << "post-inf for disp lblidx=" << (int)nStereoAlphaLabel << '\n';
                lvCout << "disp = " << lv::to_string(oCurrStereoLabeling(oDbgRect),oDbgRect.tl()) << '\n';
                lvCout << "disp swaps = " << lv::to_string(oDisparitySwaps(oDbgRect),oDbgRect.tl()) << '\n';
            }
            cv::waitKey(1);
        }

        // @@@@@@@@@@@ CHECK IF MINIMIZATION STILL HOLDS
        const ValueType tCurrStereoEnergy = m_apStereoInfs[nPrimaryCamIdx]->value();
        lvDbgAssert(tCurrStereoEnergy>=ValueType(0));
        const ValueType tCurrStereoAssocEnergy = calcTotalAssocCost(nPrimaryCamIdx);
        std::stringstream ssStereoEnergyDiff;
        if(tCurrStereoEnergy-tLastStereoEnergy==ValueType(0))
            ssStereoEnergyDiff << "null";
        else
            ssStereoEnergyDiff << std::showpos << tCurrStereoEnergy-tLastStereoEnergy;
        lvLog_(2,"\t\tdisp   e = %d   (delta=%s)",(int)tCurrStereoEnergy,ssStereoEnergyDiff.str().c_str());
        if(bNullifyStereoPairwCosts)
            lvLog(2,"\t\t\t(nullifying pairw costs)");
        else if(bJustUpdatedSegm)
            lvLog(2,"\t\t\t(just updated segmentation)");
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
            InternalLabelType nNewLabel = oCurrStereoLabeling(pt);
            lvPrint((int)nNewLabel);
            if(pt.x>=nNeighbRadius && pt.x<oCurrStereoLabeling.cols-nNeighbRadius && pt.y>=nNeighbRadius && pt.y<oCurrStereoLabeling.rows-nNeighbRadius)
                lvPrint(oCurrStereoLabeling(cv::Rect(pt.x-nNeighbRadius,pt.y-nNeighbRadius,nNeighbRadius*2+1,nNeighbRadius*2+1)));
            const float fLastImgAffinity = m_vNextFeats[FeatPack_ImgAffinity].at<float>(pt.y,pt.x,nLastLabel);
            const float fNewImgAffinity = m_vNextFeats[FeatPack_ImgAffinity].at<float>(pt.y,pt.x,nNewLabel);
            const float fLastShpAffinity = m_vNextFeats[FeatPack_ShpAffinity].at<float>(pt.y,pt.x,nLastLabel);
            const float fNewShpAffinity = m_vNextFeats[FeatPack_ShpAffinity].at<float>(pt.y,pt.x,nNewLabel);
            const float fImgSaliency = std::max(m_vNextFeats[nPrimaryCamIdx*FeatPackOffset+FeatPackOffset_ImgSaliency].at<float>(pt.y,pt.x),0.0f);
            const float fShpSaliency = std::max(m_vNextFeats[nPrimaryCamIdx*FeatPackOffset+FeatPackOffset_ShpSaliency].at<float>(pt.y,pt.x),0.0f);
            lvPrint(ValueType(fLastImgAffinity*fImgSaliency*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE));
            lvPrint(ValueType(fNewImgAffinity*fImgSaliency*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE));
            lvPrint(ValueType(fLastShpAffinity*fShpSaliency*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE));
            lvPrint(ValueType(fNewShpAffinity*fShpSaliency*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE));
            lvPrint(tLastStereoAssocEnergy);
            lvPrint(tCurrStereoAssocEnergy);
            lvPrint(tCurrStereoAssocEnergy-tLastStereoAssocEnergy);
            lvPrint(m_aAssocCosts[nPrimaryCamIdx].at<ValueType>(pt.y,pt.x));
            lvPrint(m_aStereoUnaryCosts[nPrimaryCamIdx].at<ValueType>(pt.y,pt.x));
            lvPrint(m_aStereoPairwCosts[nPrimaryCamIdx].at<ValueType>(pt.y,pt.x));
            cv::Mat tmp = m_aAssocCounts[nPrimaryCamIdx];
            m_aAssocCounts[nPrimaryCamIdx] = oLastAssocCounts;
            lvPrint(getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nLastLabel))) << "^^^^^ OLD\n";
            lvPrint(getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nNewLabel))) << "^^^^^ OLD\n";
            ValueType old_assoc_cost = m_aAssocCostRealSumLUT[getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nLastLabel))]+m_aAssocCostRealSumLUT[getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nNewLabel))];
            lvPrint(old_assoc_cost);
            m_aAssocCounts[nPrimaryCamIdx] = tmp;
            lvPrint(getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nLastLabel))) << "^^^^^ NEW\n";
            lvPrint(getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nNewLabel))) << "^^^^^ NEW\n";
            ValueType new_assoc_cost = m_aAssocCostRealSumLUT[getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nLastLabel))]+m_aAssocCostRealSumLUT[getAssocCount(nPrimaryCamIdx,pt.y,pt.x-getRealLabel(nNewLabel))];
            lvPrint(new_assoc_cost);
            ValueType assoc_cost_diff = new_assoc_cost - old_assoc_cost;
            lvPrint(assoc_cost_diff);
            lvPrint(m_aAssocCosts[nPrimaryCamIdx].at<ValueType>(pt.y,pt.x));

            //lvPrint(cv::countNonZero(m_aAssocCounts[nPrimaryCamIdx]!=oLastAssocCounts));
            cv::Mat oNonZeroPts;
            cv::findNonZero(m_aAssocCounts[nPrimaryCamIdx]!=oLastAssocCounts,oNonZeroPts);
            lvAssert(oNonZeroPts.total()>=2);
            lvPrint(oNonZeroPts.at<cv::Point>(0));
            lvPrint(oNonZeroPts.at<cv::Point>(1));
            cv::waitKey(0);
            lvCout << "\t\t\tstereo energy not minimizing!\n";
        }
        tLastStereoEnergy = tCurrStereoEnergy;
        tLastStereoAssocEnergy = tCurrStereoAssocEnergy;
        bJustUpdatedSegm = false;

        nConsecUnchangedLabels = (nChangedStereoLabels>0)?0:nConsecUnchangedLabels+1;
        nStereoAlphaLabel = m_vStereoLabelOrdering[(++nOrderingIdx%=m_nStereoLabels)]; // @@@@ order of future moves can be influenced by labels that cause the most changes? (but only late, to avoid bad local minima?)

        if((nMoveIter%STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM)==0) {
            updateResegmModels(!bResegmModelInitialized);
            bResegmModelInitialized = true;
            for(InternalLabelType nResegmAlphaLabel : {s_nForegroundLabelIdx,s_nBackgroundLabelIdx}) {
                lvLog_(2,"\tsegm inf w/ lblidx=%d   [iter #%d]",(int)nResegmAlphaLabel,(int)nMoveIter);
                calcResegmMoveCosts(nPrimaryCamIdx,nResegmAlphaLabel);
                if(lv::getVerbosity()>=3) {
                    cv::Mat oCurrLabelingDisplay = getResegmMapDisplay(nPrimaryCamIdx);
                    if(oCurrLabelingDisplay.size().area()<640*480)
                        cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                    cv::imshow("segm",oCurrLabelingDisplay);
                    if(lv::getVerbosity()>=5) {
                        lvCout << "-----\n\n\n";
                        lvCout << "segm = " << lv::to_string(oCurrResegmLabeling(oDbgRect),oDbgRect.tl()) << '\n';
                        lvCout << "unary = " << lv::to_string(m_aResegmUnaryCosts[nPrimaryCamIdx](oDbgRect),oDbgRect.tl()) << '\n';
                        lvCout << "pairw = " << lv::to_string(m_aResegmPairwCosts[nPrimaryCamIdx](oDbgRect),oDbgRect.tl()) << '\n';
                        lvCout << "next label = " << (int)nResegmAlphaLabel << '\n';
                        cv::waitKey(0);
                    }
                    else
                        cv::waitKey(1);
                }
                HOEReducer oResegmReducer; // @@@@@@ check if HigherOrderEnergy::Clear() can be used instead of realloc
                oResegmReducer.AddVars((int)m_anValidGraphNodes[nPrimaryCamIdx]);
                for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
                    const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
                    const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
                    // manually add 1st order factors while evaluating new assoc energy
                    if(oNode.anResegmUnaryFactIDs[nPrimaryCamIdx]!=SIZE_MAX) {
                        const ValueType& tUnaryCost = ((ValueType*)m_aResegmUnaryCosts[nPrimaryCamIdx].data)[nLUTNodeIdx];
                        lvDbgAssert(&tUnaryCost==&m_aResegmUnaryCosts[nPrimaryCamIdx](oNode.nRowIdx,oNode.nColIdx));
                        oResegmReducer.AddUnaryTerm((int)nGraphNodeIdx,tUnaryCost);
                    }
                    // now add 2nd order & higher order factors via lambda
                    if(oNode.aanResegmPairwFactIDs[nPrimaryCamIdx][0]!=SIZE_MAX)
                        lFactorReducer(m_apResegmModels[nPrimaryCamIdx]->operator[](oNode.aanResegmPairwFactIDs[nPrimaryCamIdx][0]),2,oResegmReducer,nResegmAlphaLabel,(InternalLabelType*)oCurrResegmLabeling.data);
                    if(oNode.aanResegmPairwFactIDs[nPrimaryCamIdx][1]!=SIZE_MAX)
                        lFactorReducer(m_apResegmModels[nPrimaryCamIdx]->operator[](oNode.aanResegmPairwFactIDs[nPrimaryCamIdx][1]),2,oResegmReducer,nResegmAlphaLabel,(InternalLabelType*)oCurrResegmLabeling.data);
                    // @@@@@ add higher o facts here (3-conn on epi lines?)
                }
                oResegmMinimizer.Reset();
                oResegmReducer.ToQuadratic(oResegmMinimizer); // @@@@@ internally might call addNodes/edges to QPBO object; optim tbd
                //oResegmMinimizer.AddPairwiseTerm(nodei,nodej,val00,val01,val10,val11); // can still add more if needed
                oResegmMinimizer.Solve();
                oResegmMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
                size_t nChangedResegmLabelings = 0;
                cv::Mat_<uchar> oSegmSwaps(m_oGridSize(),0);
                for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
                    const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
                    const int nMoveLabel = oResegmMinimizer.GetLabel((int)nGraphNodeIdx);
                    lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
                    if(nMoveLabel==1) { // node label changed to alpha
                        const int nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
                        const int nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
                        oSegmSwaps(nRowIdx,nColIdx) = 255;
                        oCurrResegmLabeling(nRowIdx,nColIdx) = nResegmAlphaLabel;
                        ++nChangedResegmLabelings;
                    }
                }
                if(lv::getVerbosity()>=3) {
                    cv::Mat oCurrLabelingDisplay = getResegmMapDisplay(nPrimaryCamIdx);
                    if(oCurrLabelingDisplay.size().area()<640*480)
                        cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                    cv::imshow("segm",oCurrLabelingDisplay);
                    if(lv::getVerbosity()>=4) {
                        cv::Mat oSegmSwapsDisplay = oSegmSwaps.clone();
                        if(oSegmSwapsDisplay.size().area()<640*480)
                            cv::resize(oSegmSwapsDisplay,oSegmSwapsDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                        cv::imshow("segm-swaps",oSegmSwapsDisplay);
                    }
                    if(lv::getVerbosity()>=5) {
                        lvCout << "post-inf for segm lblidx=" << (int)nResegmAlphaLabel << '\n';
                        lvCout << "segm = " << lv::to_string(oCurrResegmLabeling(oDbgRect),oDbgRect.tl()) << '\n';
                        lvCout << "segm swaps = " << lv::to_string(oSegmSwaps(oDbgRect),oDbgRect.tl()) << '\n';
                    }
                    cv::waitKey(1);
                }
                const ValueType tCurrResegmEnergy = m_apResegmInfs[nPrimaryCamIdx]->value();
                lvDbgAssert(tCurrResegmEnergy>=ValueType(0));
                tLastResegmEnergy = tCurrResegmEnergy;
                std::stringstream ssResegmEnergyDiff;
                if(tCurrResegmEnergy-tLastResegmEnergy==ValueType(0))
                    ssResegmEnergyDiff << "null";
                else
                    ssResegmEnergyDiff << std::showpos << tCurrResegmEnergy-tLastResegmEnergy;
                lvLog_(2,"\t\tsegm [+%s]   e = %d   (delta=%s)",(nResegmAlphaLabel==s_nForegroundLabelIdx?"fg":"bg"),(int)tCurrResegmEnergy,ssResegmEnergyDiff.str().c_str());
                if(tLastResegmEnergy<tCurrResegmEnergy) {
                    lvCout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n";
                    // @@@@ add debug stuff here (same as disp)
                    lvCout << "\t\t\tresegm energy not minimizing!\n";
                }
                if(nChangedResegmLabelings) {
                    // @@@@@ add change mask to calc shape features? avoid recalc stuff that's too far (descriptors)
                    calcShapeFeatures(m_aResegmLabelings,false); // @@@@@ make resegm graph infer both?
                    updateResegmModels(false);
                    bJustUpdatedSegm = true;
                }
            }
            updateStereoModels(false);
        }
    }
    lvLog_(2,"Inference for primary camera idx=%d completed in %f second(s).",(int)nPrimaryCamIdx,oLocalTimer.tock());
    if(lv::getVerbosity()>=3)
        cv::waitKey(0);
    return opengm::InferenceTermination::NORMAL;
}

inline cv::Mat StereoSegmMatcher::GraphModelData::getResegmMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert(!m_aResegmLabelings[nCamIdx].empty() && m_oGridSize==m_aResegmLabelings[nCamIdx].size);
    cv::Mat oOutput(m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<m_aResegmLabelings[nCamIdx].rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<m_aResegmLabelings[nCamIdx].cols; ++nColIdx) {
            const InternalLabelType nCurrLabel = m_aResegmLabelings[nCamIdx](nRowIdx,nColIdx);
            const uchar nInitLabel = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask].at<uchar>(nRowIdx,nColIdx);
            if(nCurrLabel==s_nForegroundLabelIdx && nInitLabel>0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(127,127,127);
            else if(nCurrLabel==s_nBackgroundLabelIdx && nInitLabel>0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,127);
            else if(nCurrLabel==s_nForegroundLabelIdx && nInitLabel==0)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,127,0);
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,0);
        }
    }
    cv::Mat oInputDisplay = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
    if(oInputDisplay.channels()==1)
        cv::cvtColor(oInputDisplay,oInputDisplay,cv::COLOR_GRAY2BGR);
    oOutput = (oOutput+oInputDisplay)/2;
    return oOutput;
}

inline cv::Mat StereoSegmMatcher::GraphModelData::getStereoDispMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert(!m_aStereoLabelings[nCamIdx].empty() && m_oGridSize==m_aStereoLabelings[nCamIdx].size);
    const float fRescaleFact = float(UCHAR_MAX)/(m_nMaxDispOffset-m_nMinDispOffset+1);
    cv::Mat oOutput(m_oGridSize(),CV_8UC3);
    for(int nRowIdx=0; nRowIdx<m_aStereoLabelings[nCamIdx].rows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<m_aStereoLabelings[nCamIdx].cols; ++nColIdx) {
            const size_t nLUTNodeIdx = nRowIdx*m_oGridSize[1]+nColIdx;
            const GraphModelData::NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            const OutputLabelType nRealLabel = getRealLabel(m_aStereoLabelings[nCamIdx](nRowIdx,nColIdx));
            if(nRealLabel==s_nDontCareLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,128,128);
            else if(nRealLabel==s_nOccludedLabel)
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(128,0,128);
            else {
                const uchar nIntensity = uchar((nRealLabel-m_nMinDispOffset)*fRescaleFact);
                if(oNode.abNearGraphBorders[nCamIdx])
                    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(0,0,nIntensity);
                //else if( has invalid offset desc at max disp )
                //    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(uchar(nIntensity/2),uchar(nIntensity/2),nIntensity);
                else
                    oOutput.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b::all(nIntensity);
            }
        }
    }
    return oOutput;
}

inline cv::Mat StereoSegmMatcher::GraphModelData::getAssocCountsMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert(!m_aAssocCounts[nCamIdx].empty() && m_aAssocCounts[nCamIdx].rows==int(m_oGridSize(0)));
    lvAssert(m_aAssocCounts[nCamIdx].cols==int((m_oGridSize(1)+m_nMaxDispOffset)/m_nDispOffsetStep));
    double dMax;
    cv::minMaxIdx(m_aAssocCounts[nCamIdx],nullptr,&dMax);
    const float fRescaleFact = float(UCHAR_MAX)/(int(dMax)+1);
    cv::Mat oOutput(int(m_oGridSize(0)),int(m_oGridSize(1)+m_nMaxDispOffset),CV_8UC3);
    const int nColIdxStart = ((nCamIdx==size_t(1))?0:-int(m_nMaxDispOffset));
    const int nColIdxEnd = ((nCamIdx==size_t(1))?int(m_oGridSize(1)+m_nMaxDispOffset):int(m_oGridSize(1)));
    for(int nRowIdx=0; nRowIdx<int(m_oGridSize(0)); ++nRowIdx) {
        for(int nColIdx=nColIdxStart; nColIdx<nColIdxEnd; ++nColIdx) {
            const AssocCountType nCount = getAssocCount(nCamIdx,nRowIdx,nColIdx);
            if(nColIdx<0 || nColIdx>=int(m_oGridSize(1)))
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx-nColIdxStart) = cv::Vec3b(0,0,uchar(nCount*fRescaleFact));
            else
                oOutput.at<cv::Vec3b>(nRowIdx,nColIdx-nColIdxStart) = cv::Vec3b::all(uchar(nCount*fRescaleFact));
        }
    }
    return oOutput;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::StereoGraphInference::StereoGraphInference(size_t nCamIdx, GraphModelData& oData) :
        m_oData(oData),m_nCamIdx(nCamIdx) {
    lvDbgExceptionWatch;
    lvAssert_(m_nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert_(m_oData.m_apStereoModels[m_nCamIdx],"invalid graph");
    const StereoModelType& oGM = *m_oData.m_apStereoModels[m_nCamIdx];
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
    lvDbgExceptionWatch;
    lvDbgAssert(m_oData.m_apStereoModels[m_nCamIdx]);
    return *m_oData.m_apStereoModels[m_nCamIdx];
}

inline opengm::InferenceTermination StereoSegmMatcher::StereoGraphInference::infer() {
    lvDbgExceptionWatch;
    return m_oData.infer(m_nCamIdx);
}

inline void StereoSegmMatcher::StereoGraphInference::setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_oGridSize.total()),lv::make_range(InternalLabelType(0),InternalLabelType(m_oData.m_vStereoLabels.size()-1))).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_aStereoLabelings[m_nCamIdx].isContinuous() && m_oData.m_aStereoLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_oGridSize.total(),m_oData.m_aStereoLabelings[m_nCamIdx].begin());
}

inline void StereoSegmMatcher::StereoGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),m_oData.m_vStereoLabels).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_aStereoLabelings[m_nCamIdx].begin(),[&](const OutputLabelType& nRealLabel){return m_oData.getInternalLabel(nRealLabel);});
}

inline opengm::InferenceTermination StereoSegmMatcher::StereoGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    lvDbgExceptionWatch;
    if(n==1) {
        lvDbgAssert_(m_oData.m_aStereoLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_aStereoLabelings[m_nCamIdx].total());
        std::copy(m_oData.m_aStereoLabelings[m_nCamIdx].begin(),m_oData.m_aStereoLabelings[m_nCamIdx].end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones!
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void StereoSegmMatcher::StereoGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_aStereoLabelings[m_nCamIdx].isContinuous() && m_oData.m_aStereoLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_aStereoLabelings[m_nCamIdx].size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_aStereoLabelings[m_nCamIdx].begin(),m_oData.m_aStereoLabelings[m_nCamIdx].end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return m_oData.getRealLabel(nLabel);});
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::StereoGraphInference::value() const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_aStereoLabelings[m_nCamIdx].size,"output labeling must be a 2d grid");
    const ValueType tTotAssocCost = m_oData.calcTotalAssocCost(m_nCamIdx);
    struct GraphNodeLabelIter {
        GraphNodeLabelIter(size_t nCamIdx, const GraphModelData& oData) : m_oData(oData),m_nCamIdx(nCamIdx) {}
        InternalLabelType operator[](size_t nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_oData.m_avValidLUTNodeIdxs[m_nCamIdx][nGraphNodeIdx];
            return ((InternalLabelType*)m_oData.m_aStereoLabelings[m_nCamIdx].data)[nLUTNodeIdx];
        }
        const GraphModelData& m_oData;
        const size_t m_nCamIdx;
    } oLabelIter(m_nCamIdx,m_oData);
    const ValueType tTotStereoLabelCost = m_oData.m_apStereoModels[m_nCamIdx]->evaluate(oLabelIter);
    return tTotAssocCost+tTotStereoLabelCost;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////

inline StereoSegmMatcher::ResegmGraphInference::ResegmGraphInference(size_t nCamIdx, GraphModelData& oData) :
        m_oData(oData),m_nCamIdx(nCamIdx) {
    lvDbgExceptionWatch;
    lvAssert_(m_nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert_(m_oData.m_apResegmModels[m_nCamIdx],"invalid graph");
    const ResegmModelType& oGM = *m_oData.m_apResegmModels[m_nCamIdx];
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
    lvDbgExceptionWatch;
    lvDbgAssert(m_oData.m_apResegmModels[m_nCamIdx]);
    return *m_oData.m_apResegmModels[m_nCamIdx];
}

inline opengm::InferenceTermination StereoSegmMatcher::ResegmGraphInference::infer() {
    lvDbgExceptionWatch;
    return m_oData.infer(m_nCamIdx);
}

inline void StereoSegmMatcher::ResegmGraphInference::setStartingPoint(typename std::vector<InternalLabelType>::const_iterator begin) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(begin,begin+m_oData.m_oGridSize.total()),std::vector<InternalLabelType>{s_nBackgroundLabelIdx,s_nForegroundLabelIdx}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvDbgAssert_(m_oData.m_aResegmLabelings[m_nCamIdx].isContinuous() && m_oData.m_aResegmLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    std::copy_n(begin,m_oData.m_oGridSize.total(),m_oData.m_aResegmLabelings[m_nCamIdx].begin());
}

inline void StereoSegmMatcher::ResegmGraphInference::setStartingPoint(const cv::Mat_<OutputLabelType>& oLabeling) {
    lvDbgExceptionWatch;
    lvDbgAssert_(lv::filter_out(lv::unique(oLabeling.begin(),oLabeling.end()),std::vector<OutputLabelType>{s_nBackgroundLabel,s_nForegroundLabel}).empty(),"provided labeling possesses invalid/out-of-range labels");
    lvAssert_(m_oData.m_oGridSize==oLabeling.size && oLabeling.isContinuous(),"provided labeling must fit grid size & be continuous");
    std::transform(oLabeling.begin(),oLabeling.end(),m_oData.m_aResegmLabelings[m_nCamIdx].begin(),[&](const OutputLabelType& nRealLabel){return nRealLabel?s_nForegroundLabelIdx:s_nBackgroundLabelIdx;});
}

inline opengm::InferenceTermination StereoSegmMatcher::ResegmGraphInference::arg(std::vector<InternalLabelType>& oLabeling, const size_t n) const {
    lvDbgExceptionWatch;
    if(n==1) {
        lvDbgAssert_(m_oData.m_aResegmLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"mismatch between internal graph label count and labeling mat size");
        oLabeling.resize(m_oData.m_aResegmLabelings[m_nCamIdx].total());
        std::copy(m_oData.m_aResegmLabelings[m_nCamIdx].begin(),m_oData.m_aResegmLabelings[m_nCamIdx].end(),oLabeling.begin()); // the labels returned here are NOT the 'real' ones (same values, different type)
        return opengm::InferenceTermination::NORMAL;
    }
    return opengm::InferenceTermination::UNKNOWN;
}

inline void StereoSegmMatcher::ResegmGraphInference::getOutput(cv::Mat_<OutputLabelType>& oLabeling) const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_aResegmLabelings[m_nCamIdx].isContinuous() && m_oData.m_aResegmLabelings[m_nCamIdx].total()==m_oData.m_oGridSize.total(),"unexpected internal labeling size");
    oLabeling.create(m_oData.m_aResegmLabelings[m_nCamIdx].size());
    lvDbgAssert_(oLabeling.isContinuous(),"provided matrix must be continuous for in-place label transform");
    std::transform(m_oData.m_aResegmLabelings[m_nCamIdx].begin(),m_oData.m_aResegmLabelings[m_nCamIdx].end(),oLabeling.begin(),[&](const InternalLabelType& nLabel){return nLabel?s_nForegroundLabel:s_nBackgroundLabel;});
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::ResegmGraphInference::value() const {
    lvDbgExceptionWatch;
    lvDbgAssert_(m_oData.m_oGridSize.dims()==2 && m_oData.m_oGridSize==m_oData.m_aResegmLabelings[m_nCamIdx].size,"output labeling must be a 2d grid");
    struct GraphNodeLabelIter {
        GraphNodeLabelIter(size_t nCamIdx, const GraphModelData& oData) : m_oData(oData),m_nCamIdx(nCamIdx) {}
        InternalLabelType operator[](size_t nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_oData.m_avValidLUTNodeIdxs[m_nCamIdx][nGraphNodeIdx];
            return ((InternalLabelType*)m_oData.m_aResegmLabelings[m_nCamIdx].data)[nLUTNodeIdx];
        }
        const GraphModelData& m_oData;
        const size_t m_nCamIdx;
    } oLabelIter(m_nCamIdx,m_oData);
    const ValueType tTotResegmLabelCost = m_oData.m_apResegmModels[m_nCamIdx]->evaluate(oLabelIter);
    return tTotResegmLabelCost;
}