
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
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
#if !HAVE_OPENGM_EXTLIB
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ QPBO for inference."
#endif //!HAVE_OPENGM_EXTLIB
#if !HAVE_OPENGM_EXTLIB_QPBO
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ QPBO for inference."
#endif //!HAVE_OPENGM_EXTLIB_QPBO
#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
#if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
#if !HAVE_OPENGM_EXTLIB
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ FastPD for inference."
#endif //!HAVE_OPENGM_EXTLIB
#if !HAVE_OPENGM_EXTLIB_FASTPD
#error "ForegroundStereoMatcher config requires OpenGM external lib w/ FastPD for inference."
#endif //!HAVE_OPENGM_EXTLIB_FASTPD
#endif //STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
#if (STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)
#if !HAVE_BOOST
#error "ForegroundStereoMatcher config requires boost due to 3rdparty sospd module for inference."
#endif //!HAVE_BOOST
#define STEREOSEGMATCH_CONFIG_USE_SOSPD_ALPHA_HEIGHTS_LABEL_ORDERING 0
#endif //(STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)
#if (STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY+\
     STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY/*+...*/\
    )!=1
#error "Must specify only one image affinity map computation approach to use."
#endif //(features config ...)!=1
#define STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY (STEREOSEGMATCH_CONFIG_USE_DASCGF_AFFINITY||STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY||STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY)
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF+\
     STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF+\
     STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF/*+...*/\
    )!=1
#error "Must specify only one stereo inference approach to use."
#endif //(stereo inf config ...)!=1
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF+\
     STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF/*+...*/\
    )!=1
#error "Must specify only one resegm inference approach to use."
#endif //(resegm inf config ...)!=1

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

inline void StereoSegmMatcher::initialize(const std::array<cv::Mat,s_nCameraCount>& aROIs) {
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
        m_nResegmLabels(2),
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
    const cv::Size oDescWinSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#elif STEREOSEGMATCH_CONFIG_USE_DASCRF_AFFINITY
    m_pImgDescExtractor = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR,DASC_DEFAULT_RF_ITERS,DASC_DEFAULT_PREPROCESS);
    const cv::Size oDescWinSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#elif STEREOSEGMATCH_CONFIG_USE_LSS_AFFINITY
    const int nLSSInnerRadius = 0;
    const int nLSSOuterRadius = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_RAD;
    const int nLSSPatchSize = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_PATCH;
    const int nLSSAngBins = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_ANG_BINS;
    const int nLSSRadBins = (int)STEREOSEGMATCH_DEFAULT_LSSDESC_RAD_BINS;
    m_pImgDescExtractor = std::make_unique<LSS>(nLSSInnerRadius,nLSSOuterRadius,nLSSPatchSize,nLSSAngBins,nLSSRadBins);
    const cv::Size oDescWinSize = m_pImgDescExtractor->windowSize();
    m_nGridBorderSize = (size_t)std::max(m_pImgDescExtractor->borderSize(0),m_pImgDescExtractor->borderSize(1));
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    const int nWindowSize = int(STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD*2+1);
    const cv::Size oDescWinSize = cv::Size(nWindowSize,nWindowSize);
    m_nGridBorderSize = STEREOSEGMATCH_DEFAULT_MI_WINDOW_RAD;
#elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    constexpr int nSSqrDiffKernelSize = int(STEREOSEGMATCH_DEFAULT_SSQDIFF_PATCH);
    const cv::Size oDescWinSize(nSSqrDiffKernelSize,nSSqrDiffKernelSize);
    m_nGridBorderSize = size_t(nSSqrDiffKernelSize/2);
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
    const size_t nShapeContextInnerRadius = 2;
    const size_t nShapeContextOuterRadius = STEREOSEGMATCH_DEFAULT_SCDESC_WIN_RAD;
    const size_t nShapeContextAngBins = STEREOSEGMATCH_DEFAULT_SCDESC_ANG_BINS;
    const size_t nShapeContextRadBins = STEREOSEGMATCH_DEFAULT_SCDESC_RAD_BINS;
    m_pShpDescExtractor = std::make_unique<ShapeContext>(nShapeContextInnerRadius,nShapeContextOuterRadius,nShapeContextAngBins,nShapeContextRadBins);
    lvAssert__(oDescWinSize.width<=(int)m_oGridSize[1] && oDescWinSize.height<=(int)m_oGridSize[0],"image is too small to compute descriptors with current pattern size -- need at least (%d,%d) and got (%d,%d)",oDescWinSize.width,oDescWinSize.height,(int)m_oGridSize[1],(int)m_oGridSize[0]);
    lvDbgAssert(m_nGridBorderSize<m_oGridSize[0] && m_nGridBorderSize<m_oGridSize[1]);
    lvDbgAssert(m_nGridBorderSize<(size_t)oDescWinSize.width && m_nGridBorderSize<(size_t)oDescWinSize.height);
    lvDbgAssert((size_t)std::max(m_pShpDescExtractor->borderSize(0),m_pShpDescExtractor->borderSize(1))<=m_nGridBorderSize);
    lvDbgAssert(m_aAssocCostRealAddLUT.size()==m_aAssocCostRealSumLUT.size() && m_aAssocCostRealRemLUT.size()==m_aAssocCostRealSumLUT.size());
    lvDbgAssert(m_aAssocCostApproxAddLUT.size()==m_aAssocCostRealAddLUT.size() && m_aAssocCostApproxRemLUT.size()==m_aAssocCostRealRemLUT.size());
    lvDbgAssert_(m_nMaxDispOffset+m_nDispOffsetStep<m_aAssocCostRealSumLUT.size(),"assoc cost lut size might not be large enough");
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT<m_aAssocCostRealSumLUT.size());
    lvDbgAssert(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(0)==0.0f);
    std::fill_n(m_aAssocCostRealAddLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,cost_cast(0));
    std::fill_n(m_aAssocCostRealRemLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,cost_cast(0));
    std::fill_n(m_aAssocCostRealSumLUT.begin(),STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT,cost_cast(0));
    for(size_t nIdx=STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT; nIdx<m_aAssocCostRealAddLUT.size(); ++nIdx) {
        m_aAssocCostRealAddLUT[nIdx] = cost_cast(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(nIdx+1-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRealRemLUT[nIdx] = -cost_cast(STEREOSEGMATCH_UNIQUE_COST_INCR_REL(nIdx-STEREOSEGMATCH_UNIQUE_COST_ZERO_COUNT)*STEREOSEGMATCH_UNIQUE_COST_OVER_SCALE/m_nDispOffsetStep);
        m_aAssocCostRealSumLUT[nIdx] = (nIdx==size_t(0)?cost_cast(0):(m_aAssocCostRealSumLUT[nIdx-1]+m_aAssocCostRealAddLUT[nIdx-1]));
    }
    for(size_t nIdx=0; nIdx<m_aAssocCostRealAddLUT.size(); ++nIdx) {
        // 'average' cost of removing one assoc from target pixel (not 'true' energy, but easy-to-optimize worse case)
        m_aAssocCostApproxRemLUT[nIdx] = (nIdx==size_t(0)?cost_cast(0):cost_cast(-1.0f*m_aAssocCostRealSumLUT[nIdx]/nIdx+0.5f));
        if(m_nDispOffsetStep==size_t(1))
            // if m_nDispOffsetStep==1, then use real cost of adding one assoc to target pixel
            // (target can only have one new assoc per iter, due to single label move)
            m_aAssocCostApproxAddLUT[nIdx] = m_aAssocCostRealAddLUT[nIdx];
        else {
            // otherwise, use average cost of adding 'm_nDispOffsetStep' assocs to target block
            // (i.e. the average cost of adding the max possible number of new assocs to a block per iter)
            m_aAssocCostApproxAddLUT[nIdx] = cost_cast(0);
            for(size_t nOffsetIdx=nIdx; nOffsetIdx<nIdx+m_nDispOffsetStep; ++nOffsetIdx)
                m_aAssocCostApproxAddLUT[nIdx] += m_aAssocCostRealAddLUT[std::min(nOffsetIdx,m_aAssocCostRealAddLUT.size()-1)];
            m_aAssocCostApproxAddLUT[nIdx] = cost_cast(float(m_aAssocCostApproxAddLUT[nIdx])/m_nDispOffsetStep+0.5f);
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
    const size_t nResegmUnaryFuncDataSize = nMaxValidNodes*m_nResegmLabels;
    const size_t nResegmPairwFuncDataSize = nMaxValidNodes*2*(m_nResegmLabels*m_nResegmLabels);
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
    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
        m_aStereoUnaryCosts[nCamIdx].create(m_oGridSize);
    #elif STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
        m_aStereoUnaryCosts[nCamIdx].create(int(m_nStereoLabels),int(anValidNodes[nCamIdx])); // @@@ flip for optim?
    #endif //STEREOSEGMATCH_CONFIG_USE_..._STEREO_INF
    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF
        m_aResegmUnaryCosts[nCamIdx].create(m_oGridSize);
    #endif //STEREOSEGMATCH_CONFIG_USE_..._RESEGM_INF
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
                std::fill_n(m_vNodeInfos[nLUTNodeIdx].aafStereoPairwWeights[nCamIdx].begin(),s_nPairwOrients,0.0f);
                // aaStereoPairwCliques,aaResegmPairwCliques are fine when default-constructed
            }
        }
    }
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        lvAssert(m_avValidLUTNodeIdxs[nCamIdx].size()==anValidNodes[nCamIdx]);
        lvAssert(m_anValidGraphNodes[nCamIdx]==anValidNodes[nCamIdx]);
        m_nTotValidGraphNodes += m_anValidGraphNodes[nCamIdx];
        m_avStereoUnaryFuncs[nCamIdx].reserve(m_anValidGraphNodes[nCamIdx]);
        m_avResegmUnaryFuncs[nCamIdx].reserve(m_anValidGraphNodes[nCamIdx]);
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            m_aavStereoPairwFuncs[nCamIdx][nOrientIdx].reserve(m_anValidGraphNodes[nCamIdx]);
            m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].reserve(m_anValidGraphNodes[nCamIdx]);
        }
        m_aaStereoFuncsData[nCamIdx] = std::make_unique<ValueType[]>(nStereoFuncDataSize);
        m_apStereoUnaryFuncsDataBase[nCamIdx] = m_aaStereoFuncsData[nCamIdx].get();
        m_apStereoPairwFuncsDataBase[nCamIdx] = m_apStereoUnaryFuncsDataBase[nCamIdx]+nStereoUnaryFuncDataSize;
        m_aaResegmFuncsData[nCamIdx] = std::make_unique<ValueType[]>(nResegmFuncDataSize);
        m_apResegmUnaryFuncsDataBase[nCamIdx] = m_aaResegmFuncsData[nCamIdx].get();
        m_apResegmPairwFuncsDataBase[nCamIdx] = m_apResegmUnaryFuncsDataBase[nCamIdx]+nResegmUnaryFuncDataSize;
    }
    lvLog(2,"\tadding unary factors to each graph node...");
    m_anUnaryFactCounts = CamArray<size_t>{};
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const std::array<size_t,1> aUnaryStereoFuncDims = {m_nStereoLabels};
        const std::array<size_t,1> aUnaryResegmFuncDims = {m_nResegmLabels};
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            m_avStereoUnaryFuncs[nCamIdx].push_back(m_apStereoModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
            m_avResegmUnaryFuncs[nCamIdx].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
            FuncPairType& oStereoFunc = m_avStereoUnaryFuncs[nCamIdx].back();
            FuncPairType& oResegmFunc = m_avResegmUnaryFuncs[nCamIdx].back();
            lvDbgAssert((&m_apStereoModels[nCamIdx]->getFunction<ExplicitFunction>(oStereoFunc.first))==(&oStereoFunc.second));
            lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oResegmFunc.first))==(&oResegmFunc.second));
            oStereoFunc.second.assign(aUnaryStereoFuncDims.begin(),aUnaryStereoFuncDims.end(),m_apStereoUnaryFuncsDataBase[nCamIdx]+(nGraphNodeIdx*m_nStereoLabels));
            oResegmFunc.second.assign(aUnaryResegmFuncDims.begin(),aUnaryResegmFuncDims.end(),m_apResegmUnaryFuncsDataBase[nCamIdx]+(nGraphNodeIdx*m_nResegmLabels));
            lvDbgAssert(oStereoFunc.second.strides(0)==1 && oResegmFunc.second.strides(0)==1); // expect no padding
            const std::array<size_t,1> aGraphNodeIndices = {nGraphNodeIdx};
            oNode.anStereoUnaryFactIDs[nCamIdx] = m_apStereoModels[nCamIdx]->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
            oNode.anResegmUnaryFactIDs[nCamIdx] = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
            lvDbgAssert(FuncIdentifType((*m_apStereoModels[nCamIdx])[oNode.anStereoUnaryFactIDs[nCamIdx]].functionIndex(),(*m_apStereoModels[nCamIdx])[oNode.anStereoUnaryFactIDs[nCamIdx]].functionType())==oStereoFunc.first);
            lvDbgAssert(FuncIdentifType((*m_apResegmModels[nCamIdx])[oNode.anResegmUnaryFactIDs[nCamIdx]].functionIndex(),(*m_apResegmModels[nCamIdx])[oNode.anResegmUnaryFactIDs[nCamIdx]].functionType())==oResegmFunc.first);
            lvDbgAssert(m_apStereoModels[nCamIdx]->operator[](oNode.anStereoUnaryFactIDs[nCamIdx]).numberOfVariables()==size_t(1));
            lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oNode.anResegmUnaryFactIDs[nCamIdx]).numberOfVariables()==size_t(1));
            lvDbgAssert(oNode.anStereoUnaryFactIDs[nCamIdx]==m_anUnaryFactCounts[nCamIdx]);
            lvDbgAssert(oNode.anResegmUnaryFactIDs[nCamIdx]==m_anUnaryFactCounts[nCamIdx]);
            oNode.apStereoUnaryFuncs[nCamIdx] = &oStereoFunc.second;
            oNode.apResegmUnaryFuncs[nCamIdx] = &oResegmFunc.second;
            ++m_anUnaryFactCounts[nCamIdx];
        }
    }
    lvLog(2,"\tadding pairwise factors to each graph node pair...");
    m_anPairwFactCounts = CamArray<size_t>{};
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const std::vector<size_t> aPairwiseStereoFuncDims(s_nPairwOrients,m_nStereoLabels);
        const std::vector<size_t> aPairwiseResegmFuncDims(s_nPairwOrients,m_nResegmLabels);
        // note: current def w/ explicit stereo function will require too much memory if using >>50 disparity labels
        // note2: if inference is based on fastpd/bcd and stereo pairw terms are shared + weighted, *could* remove indiv pairw allocs
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const size_t nBaseLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
            NodeInfo& oBaseNode = m_vNodeInfos[nBaseLUTNodeIdx];
            if(!oBaseNode.abValidGraphNode[nCamIdx])
                continue;
            const auto lPairwCliqueCreator = [&](size_t nOrientIdx, size_t nOffsetLUTNodeIdx) {
                const NodeInfo& oOffsetNode = m_vNodeInfos[nOffsetLUTNodeIdx];
                if(oOffsetNode.abValidGraphNode[nCamIdx]) {
                    m_aavStereoPairwFuncs[nCamIdx][nOrientIdx].push_back(m_apStereoModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                    m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].push_back(m_apResegmModels[nCamIdx]->addFunctionWithRefReturn(ExplicitFunction()));
                    FuncPairType& oStereoFunc = m_aavStereoPairwFuncs[nCamIdx][nOrientIdx].back();
                    FuncPairType& oResegmFunc = m_aavResegmPairwFuncs[nCamIdx][nOrientIdx].back();
                    oStereoFunc.second.assign(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end(),m_apStereoPairwFuncsDataBase[nCamIdx]+((nGraphNodeIdx*s_nPairwOrients+nOrientIdx)*m_nStereoLabels*m_nStereoLabels));
                    oResegmFunc.second.assign(aPairwiseResegmFuncDims.begin(),aPairwiseResegmFuncDims.end(),m_apResegmPairwFuncsDataBase[nCamIdx]+((nGraphNodeIdx*s_nPairwOrients+nOrientIdx)*m_nResegmLabels*m_nResegmLabels));
                    lvDbgAssert(oStereoFunc.second.strides(0)==1 && oStereoFunc.second.strides(1)==m_nStereoLabels); // expect last-idx-major
                    lvDbgAssert(oResegmFunc.second.strides(0)==1 && oResegmFunc.second.strides(1)==m_nResegmLabels); // expect last-idx-major
                    lvDbgAssert((&m_apStereoModels[nCamIdx]->getFunction<ExplicitFunction>(oStereoFunc.first))==(&oStereoFunc.second));
                    lvDbgAssert((&m_apResegmModels[nCamIdx]->getFunction<ExplicitFunction>(oResegmFunc.first))==(&oResegmFunc.second));
                    PairwClique& oStereoClique = oBaseNode.aaStereoPairwCliques[nCamIdx][nOrientIdx];
                    PairwClique& oResegmClique = oBaseNode.aaResegmPairwCliques[nCamIdx][nOrientIdx];
                    oStereoClique.m_bValid = true;
                    oResegmClique.m_bValid = true;
                    const std::array<size_t,2> aLUTNodeIndices = {nBaseLUTNodeIdx,nOffsetLUTNodeIdx};
                    oStereoClique.m_anLUTNodeIdxs = aLUTNodeIndices;
                    oResegmClique.m_anLUTNodeIdxs = aLUTNodeIndices;
                    const std::array<size_t,2> aGraphNodeIndices = {nGraphNodeIdx,oOffsetNode.anGraphNodeIdxs[nCamIdx]};
                    oStereoClique.m_anGraphNodeIdxs = aGraphNodeIndices;
                    oResegmClique.m_anGraphNodeIdxs = aGraphNodeIndices;
                    oStereoClique.m_nGraphFactorId = m_apStereoModels[nCamIdx]->addFactorNonFinalized(oStereoFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    oResegmClique.m_nGraphFactorId = m_apResegmModels[nCamIdx]->addFactorNonFinalized(oResegmFunc.first,aGraphNodeIndices.begin(),aGraphNodeIndices.end());
                    lvDbgAssert(FuncIdentifType((*m_apStereoModels[nCamIdx])[oStereoClique.m_nGraphFactorId].functionIndex(),(*m_apStereoModels[nCamIdx])[oStereoClique.m_nGraphFactorId].functionType())==oStereoFunc.first);
                    lvDbgAssert(FuncIdentifType((*m_apResegmModels[nCamIdx])[oResegmClique.m_nGraphFactorId].functionIndex(),(*m_apResegmModels[nCamIdx])[oResegmClique.m_nGraphFactorId].functionType())==oResegmFunc.first);
                    lvDbgAssert(m_apStereoModels[nCamIdx]->operator[](oStereoClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oResegmClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(oStereoClique.m_nGraphFactorId==m_anUnaryFactCounts[nCamIdx]+m_anPairwFactCounts[nCamIdx]);
                    lvDbgAssert(oResegmClique.m_nGraphFactorId==m_anUnaryFactCounts[nCamIdx]+m_anPairwFactCounts[nCamIdx]);
                    oStereoClique.m_pGraphFunctionPtr = &oStereoFunc.second;
                    oResegmClique.m_pGraphFunctionPtr = &oResegmFunc.second;
                    ++m_anPairwFactCounts[nCamIdx];
                }
            };
            const size_t nRowIdx = (size_t)oBaseNode.nRowIdx;
            const size_t nColIdx = (size_t)oBaseNode.nColIdx;
            if(nRowIdx+1<nRows) { // vertical pair
                const size_t nOrientIdx = size_t(0);
                const size_t nOffsetLUTNodeIdx = (nRowIdx+1)*nCols+nColIdx;
                lPairwCliqueCreator(nOrientIdx,nOffsetLUTNodeIdx);
            }
            if(nColIdx+1<nCols) { // horizontal pair
                const size_t nOrientIdx = size_t(1);
                const size_t nOffsetLUTNodeIdx = nRowIdx*nCols+nColIdx+1;
                lPairwCliqueCreator(nOrientIdx,nOffsetLUTNodeIdx);
            }
            static_assert(s_nPairwOrients==2,"missing some pairw instantiations here");
        }
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            m_aaStereoPairwFuncIDs_base[nCamIdx][nOrientIdx] = m_apStereoModels[nCamIdx]->addFunction(ExplicitAllocFunction(aPairwiseStereoFuncDims.begin(),aPairwiseStereoFuncDims.end()));
            ExplicitAllocFunction& oStereoBaseFunc = m_apStereoModels[nCamIdx]->getFunction<ExplicitAllocFunction>(m_aaStereoPairwFuncIDs_base[nCamIdx][nOrientIdx]);
            lvDbgAssert(oStereoBaseFunc.size()==m_nStereoLabels*m_nStereoLabels);
            for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nRealStereoLabels; ++nLabelIdx1) {
                for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nRealStereoLabels; ++nLabelIdx2) {
                    const OutputLabelType nRealLabel1 = getRealLabel(nLabelIdx1);
                    const OutputLabelType nRealLabel2 = getRealLabel(nLabelIdx2);
                    const int nRealLabelDiff = std::min(std::abs((int)nRealLabel1-(int)nRealLabel2),STEREOSEGMATCH_LBLSIM_STEREO_MAXDIFF_CST);
                    oStereoBaseFunc(nLabelIdx1,nLabelIdx2) = cost_cast(nRealLabelDiff*nRealLabelDiff);
                }
            }
            for(size_t nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
                // @@@@ outside-roi-dc to inside-roi-something is ok (0 cost)
                oStereoBaseFunc(m_nDontCareLabelIdx,nLabelIdx) = cost_cast(10000);
                oStereoBaseFunc(m_nOccludedLabelIdx,nLabelIdx) = cost_cast(10000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
                oStereoBaseFunc(nLabelIdx,m_nDontCareLabelIdx) = cost_cast(10000);
                oStereoBaseFunc(nLabelIdx,m_nOccludedLabelIdx) = cost_cast(10000); // @@@@ STEREOSEGMATCH_LBLSIM_COST_MAXOCCL scaled down if other label is high disp
            }
            oStereoBaseFunc(m_nDontCareLabelIdx,m_nDontCareLabelIdx) = cost_cast(0);
            oStereoBaseFunc(m_nOccludedLabelIdx,m_nOccludedLabelIdx) = cost_cast(0);
            // @@@@ messes up factor count for higher order terms in stereo model
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
    lvLog_(2,"Graph models constructed in %f second(s); finalizing...\n",oLocalTimer.tock());
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_apStereoModels[nCamIdx]->finalize();
        m_apStereoInfs[nCamIdx] = std::make_unique<StereoGraphInference>(nCamIdx,*this);
        m_apResegmModels[nCamIdx]->finalize();
        m_apResegmInfs[nCamIdx] = std::make_unique<ResegmGraphInference>(nCamIdx,*this);
        if(lv::getVerbosity()>=2) {
            lvCout << "Stereo[" << nCamIdx << "] :\n";
            lv::gm::printModelInfo(*m_apStereoModels[nCamIdx]);
            lvCout << "Resegm[" << nCamIdx << "] :\n";
            lv::gm::printModelInfo(*m_apResegmModels[nCamIdx]);
            lvCout << "\n";
        }
    }
}

inline void StereoSegmMatcher::GraphModelData::resetStereoLabelings(size_t nCamIdx, bool bIsPrimaryCam) {
    lvDbgAssert_(nCamIdx<getCameraCount(),"bad input cam index");
    lvDbgExceptionWatch;
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    std::fill(m_aStereoLabelings[nCamIdx].begin(),m_aStereoLabelings[nCamIdx].end(),m_nDontCareLabelIdx);
    lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const NodeInfo& oNode = m_vNodeInfos[m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx]];
        lvDbgAssert(oNode.abValidGraphNode[nCamIdx]);
        lvDbgAssert(oNode.anGraphNodeIdxs[nCamIdx]==nGraphNodeIdx);
        if(bIsPrimaryCam) {
            lvDbgAssert(oNode.anStereoUnaryFactIDs[nCamIdx]<m_apStereoModels[nCamIdx]->numberOfFactors());
            lvDbgAssert(m_apStereoModels[nCamIdx]->numberOfLabels(oNode.anStereoUnaryFactIDs[nCamIdx])==m_nStereoLabels);
            InternalLabelType nEvalLabel = m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx) = 0;
            const ExplicitFunction& vUnaryStereoLUT = *oNode.apStereoUnaryFuncs[nCamIdx];
            ValueType fOptimalEnergy = vUnaryStereoLUT(nEvalLabel);
            for(nEvalLabel=1; nEvalLabel<m_nStereoLabels; ++nEvalLabel) {
                const ValueType fCurrEnergy = vUnaryStereoLUT(nEvalLabel);
                if(fOptimalEnergy>fCurrEnergy) {
                    m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx) = nEvalLabel;
                    fOptimalEnergy = fCurrEnergy;
                }
            }
        }
        else {
            // results will be pretty rough!
            const int nColIdx = oNode.nColIdx;
            const int nRowIdx = oNode.nRowIdx;
            std::map<InternalLabelType,size_t> mWTALookupCounts;
            for(InternalLabelType nLookupLabel=0; nLookupLabel<m_nRealStereoLabels; ++nLookupLabel) {
                const int nOffsetColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLookupLabel);
                if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[nCamIdx^1](nRowIdx,nOffsetColIdx))
                    if(m_aStereoLabelings[nCamIdx^1](nRowIdx,nOffsetColIdx)==nLookupLabel)
                        ++mWTALookupCounts[nLookupLabel];
            }
            auto pWTAPairIter = std::max_element(mWTALookupCounts.begin(),mWTALookupCounts.end(),[](const auto& p1, const auto& p2) {
                return p1.second<p2.second;
            });
            if(pWTAPairIter!=mWTALookupCounts.end() && pWTAPairIter->second>size_t(0))
                m_aStereoLabelings[nCamIdx](nRowIdx,nColIdx) = pWTAPairIter->first;
        }
    }
    if(!bIsPrimaryCam) {
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
            const NodeInfo& oNode = m_vNodeInfos[m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx]];
            const int nColIdx = oNode.nColIdx;
            const int nRowIdx = oNode.nRowIdx;
            InternalLabelType& nCurrLabel = m_aStereoLabelings[nCamIdx](nRowIdx,nColIdx);
            if(nCurrLabel==m_nDontCareLabelIdx && m_aROIs[nCamIdx](nRowIdx,nColIdx)) {
                for(int nOffset=0; nOffset<=(int)m_nMaxDispOffset; ++nOffset) {
                    const int nOffsetColIdx_pos = oNode.nColIdx+nOffset;
                    if(nOffsetColIdx_pos>=0 && nOffsetColIdx_pos<nCols && m_aROIs[nCamIdx](nRowIdx,nOffsetColIdx_pos)) {
                        const InternalLabelType& nNewLabel = m_aStereoLabelings[nCamIdx](nRowIdx,nOffsetColIdx_pos);
                        if(nNewLabel!=m_nDontCareLabelIdx) {
                            nCurrLabel = nNewLabel;
                            break;
                        }
                    }
                    const int nOffsetColIdx_neg = oNode.nColIdx-nOffset;
                    if(nOffsetColIdx_neg>=0 && nOffsetColIdx_neg<nCols && m_aROIs[nCamIdx](nRowIdx,nOffsetColIdx_neg)) {
                        const InternalLabelType& nNewLabel = m_aStereoLabelings[nCamIdx](nRowIdx,nOffsetColIdx_neg);
                        if(nNewLabel!=m_nDontCareLabelIdx) {
                            nCurrLabel = nNewLabel;
                            break;
                        }
                    }
                }
            }
        }
    }
    m_aAssocCounts[nCamIdx] = (AssocCountType)0;
    m_aAssocMaps[nCamIdx] = (AssocIdxType)-1;
    std::vector<int> vLabelCounts(m_nStereoLabels,0);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
        const InternalLabelType nLabel = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data)[nLUTNodeIdx];
        if(nLabel<m_nDontCareLabelIdx) // both special labels avoided here
            addAssoc(nCamIdx,oNode.nRowIdx,oNode.nColIdx,nLabel);
        ++vLabelCounts[nLabel];
    }
    if(bIsPrimaryCam) {
        m_vStereoLabelOrdering = lv::sort_indices<InternalLabelType>(vLabelCounts,[&vLabelCounts](int a, int b){return vLabelCounts[a]>vLabelCounts[b];});
        lvDbgAssert(lv::unique(m_vStereoLabelOrdering.begin(),m_vStereoLabelOrdering.end())==lv::make_range(InternalLabelType(0),InternalLabelType(m_nStereoLabels-1)));
        // note: sospd might not follow this label order if using alpha heights strategy
    }
}

inline void StereoSegmMatcher::GraphModelData::updateStereoModel(size_t nCamIdx, bool bInit) {
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below (incl xor) will break");
    lvDbgAssert_(nCamIdx<getCameraCount(),"bad input cam index");
    lvDbgAssert(m_apStereoModels[nCamIdx] && m_apStereoModels[nCamIdx]->numberOfVariables()==m_anValidGraphNodes[nCamIdx]);
    lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
    lvDbgExceptionWatch;
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvIgnore(nRows); lvIgnore(nCols);
    const cv::Mat_<float> oImgAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    const cv::Mat_<float> oShpAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    lvDbgAssert(oImgAffinity.dims==3 && oImgAffinity.size[0]==nRows && oImgAffinity.size[1]==nCols && oImgAffinity.size[2]==(int)m_nRealStereoLabels);
    lvDbgAssert(oShpAffinity.dims==3 && oShpAffinity.size[0]==nRows && oShpAffinity.size[1]==nCols && oShpAffinity.size[2]==(int)m_nRealStereoLabels);
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
    lvLog_(4,"Updating stereo graph model[%d] energy terms based on new features...",(int)nCamIdx);
    lv::StopWatch oLocalTimer;
#if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    lv::ProgressBarManager oProgressBarMgr("\tprogress:");
#endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
#if USING_OPENMP
    #pragma omp parallel for
#endif //USING_OPENMP
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        // update unary terms for each grid node
        lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
        lvDbgAssert(oNode.anResegmUnaryFactIDs[nCamIdx]<m_anUnaryFactCounts[nCamIdx]);
        lvDbgAssert(oNode.apStereoUnaryFuncs[nCamIdx] && oNode.anStereoUnaryFactIDs[nCamIdx]!=SIZE_MAX);
        lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oNode.anResegmUnaryFactIDs[nCamIdx]).numberOfVariables()==size_t(1));
        ExplicitFunction& vUnaryStereoLUT = *oNode.apStereoUnaryFuncs[nCamIdx];
        lvDbgAssert(vUnaryStereoLUT.dimension()==1 && vUnaryStereoLUT.size()==m_nStereoLabels);
        lvDbgAssert__(aImgSaliency[nCamIdx](nRowIdx,nColIdx)>=-1e-6f && aImgSaliency[nCamIdx](nRowIdx,nColIdx)<=1.0f+1e-6f,"fImgSaliency = %1.10f @ [%d,%d]",aImgSaliency[nCamIdx](nRowIdx,nColIdx),nRowIdx,nColIdx);
        lvDbgAssert__(aShpSaliency[nCamIdx](nRowIdx,nColIdx)>=-1e-6f && aShpSaliency[nCamIdx](nRowIdx,nColIdx)<=1.0f+1e-6f,"fShpSaliency = %1.10f @ [%d,%d]",aShpSaliency[nCamIdx](nRowIdx,nColIdx),nRowIdx,nColIdx);
        const float fImgSaliency = std::max(aImgSaliency[nCamIdx](nRowIdx,nColIdx),0.0f);
        const float fShpSaliency = std::max(aShpSaliency[nCamIdx](nRowIdx,nColIdx),0.0f);
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx) {
            vUnaryStereoLUT(nLabelIdx) = cost_cast(0);
            const int nOffsetColIdx = getOffsetColIdx(nCamIdx,nColIdx,nLabelIdx);
            if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[nCamIdx^1](nRowIdx,nOffsetColIdx)) {
                const int nAffMapColIdx = (nCamIdx==size_t(0))?nColIdx:nOffsetColIdx;
                const float fImgAffinity = oImgAffinity(nRowIdx,nAffMapColIdx,nLabelIdx);
                const float fShpAffinity = oShpAffinity(nRowIdx,nAffMapColIdx,nLabelIdx);
                lvDbgAssert__(fImgAffinity>=0.0f,"fImgAffinity = %1.10f @ [%d,%d]",fImgAffinity,nRowIdx,nColIdx);
                lvDbgAssert__(fShpAffinity>=0.0f,"fShpAffinity = %1.10f @ [%d,%d]",fShpAffinity,nRowIdx,nColIdx);
                vUnaryStereoLUT(nLabelIdx) += cost_cast(fImgAffinity*fImgSaliency*STEREOSEGMATCH_IMGSIM_COST_DESC_SCALE);
                vUnaryStereoLUT(nLabelIdx) += cost_cast(fShpAffinity*fShpSaliency*STEREOSEGMATCH_SHPSIM_COST_DESC_SCALE);
                vUnaryStereoLUT(nLabelIdx) = std::min(vUnaryStereoLUT(nLabelIdx),STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            }
            else {
                vUnaryStereoLUT(nLabelIdx) = STEREOSEGMATCH_UNARY_COST_OOB_CST;
            }
        }
        vUnaryStereoLUT(m_nDontCareLabelIdx) = cost_cast(10000); // @@@@ check roi, if dc set to 0, otherwise set to inf
        vUnaryStereoLUT(m_nOccludedLabelIdx) = cost_cast(10000);//STEREOSEGMATCH_IMGSIM_COST_OCCLUDED_CST;
        if(bInit) { // inter-spectral pairwise term updates do not change w.r.t. segm or stereo updates
            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                ExplicitAllocFunction& vPairwiseStereoBaseFunc = m_apStereoModels[nCamIdx]->getFunction<ExplicitAllocFunction>(m_aaStereoPairwFuncIDs_base[nCamIdx][nOrientIdx]);
                PairwClique& oClique = oNode.aaStereoPairwCliques[nCamIdx][nOrientIdx];
                if(oClique) {
                    lvDbgAssert(oClique.m_nGraphFactorId>=m_anUnaryFactCounts[nCamIdx]);
                    lvDbgAssert(oClique.m_nGraphFactorId<m_anUnaryFactCounts[nCamIdx]+m_anPairwFactCounts[nCamIdx]);
                    lvDbgAssert(m_apStereoModels[nCamIdx]->operator[](oClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(oClique.m_anGraphNodeIdxs[0]!=SIZE_MAX && oClique.m_anLUTNodeIdxs[0]!=SIZE_MAX);
                    lvDbgAssert(oClique.m_anGraphNodeIdxs[1]!=SIZE_MAX && oClique.m_anLUTNodeIdxs[1]!=SIZE_MAX);
                    lvDbgAssert(m_vNodeInfos[oClique.m_anLUTNodeIdxs[1]].abValidGraphNode[nCamIdx] && oClique.m_pGraphFunctionPtr);
                    ExplicitFunction& vPairwiseStereoFunc = *oClique.m_pGraphFunctionPtr;
                    lvDbgAssert(vPairwiseStereoFunc.dimension()==2 && vPairwiseStereoFunc.size()==m_nStereoLabels*m_nStereoLabels);
                    const int nLocalGrad = (int)((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx);
                    const float fGradScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    lvDbgAssert(fGradScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    const float fPairwWeight = (float)(fGradScaleFact*STEREOSEGMATCH_LBLSIM_STEREO_SCALE_CST); // should be constant & uncapped for use in fastpd/bcd
                    oNode.aafStereoPairwWeights[nCamIdx][nOrientIdx] = fPairwWeight;
                    // all stereo pairw functions are identical, but weighted differently (see base init in constructor)
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nStereoLabels; ++nLabelIdx1)
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nStereoLabels; ++nLabelIdx2)
                            vPairwiseStereoFunc(nLabelIdx1,nLabelIdx2) = cost_cast(vPairwiseStereoBaseFunc(nLabelIdx1,nLabelIdx2)*fPairwWeight);
                }
            }
        }
    #if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
        if(lv::getVerbosity()>=3)
            oProgressBarMgr.update(float(nGraphNodeIdx)/m_anValidGraphNodes[nCamIdx]);
    #endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    }
    lvLog_(4,"Stereo graph model[%d] energy terms update completed in %f second(s).",(int)nCamIdx,oLocalTimer.tock());
}

inline void StereoSegmMatcher::GraphModelData::updateResegmModel(size_t nCamIdx, bool bInit) {
    static_assert(getCameraCount()==2,"bad static array size, hardcoded stuff below (incl xor) will break");
    lvDbgAssert_(nCamIdx<getCameraCount(),"bad input cam index");
    lvDbgAssert(m_apResegmModels[nCamIdx] && m_apResegmModels[nCamIdx]->numberOfVariables()==m_anValidGraphNodes[nCamIdx]);
    lvDbgAssert(m_oGridSize==m_aResegmLabelings[nCamIdx].size && m_oGridSize==m_aGMMCompAssignMap[nCamIdx].size);
    lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
    lvDbgExceptionWatch;
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
#if STEREOSEGMATCH_CONFIG_USE_GMM_LOCAL_BACKGR
    cv::Mat_<uchar> oGMMROI = (m_aResegmLabelings[nCamIdx]>0);
    cv::dilate(oGMMROI,oGMMROI,cv::getStructuringElement(cv::MORPH_RECT,cv::Size(75,75)));
    cv::bitwise_and(oGMMROI,m_aROIs[nCamIdx],oGMMROI);
#else //!STEREOSEGMATCH_CONFIG_USE_GMM_LOCAL_BACKGR
    const cv::Mat_<uchar>& oGMMROI = m_aROIs[nCamIdx];
#endif //!STEREOSEGMATCH_CONFIG_USE_GMM_LOCAL_BACKGR
    if(bInit) {
        const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        if(oInputImg.channels()==1) {
            lv::initGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
        }
        else {// 3ch
            lv::initGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
        }
    }
    else {
        const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
        if(oInputImg.channels()==1) {
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
            lv::learnGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_1ch[nCamIdx],m_aFGModels_1ch[nCamIdx],oGMMROI);
        }
        else { // 3ch
            lv::assignGaussianMixtureComponents(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
            lv::learnGaussianMixtureParams(oInputImg,m_aResegmLabelings[nCamIdx],m_aGMMCompAssignMap[nCamIdx],m_aBGModels_3ch[nCamIdx],m_aFGModels_3ch[nCamIdx],oGMMROI);
        }
    }
    if(lv::getVerbosity()>=4) {
        cv::Mat_<int> oClusterLabels = m_aGMMCompAssignMap[nCamIdx].clone();
        for(size_t nNodeIdx=0; nNodeIdx<m_aResegmLabelings[nCamIdx].total(); ++nNodeIdx)
            if(((InternalLabelType*)m_aResegmLabelings[nCamIdx].data)[nNodeIdx])
                ((int*)oClusterLabels.data)[nNodeIdx] += 1<<31;
        cv::Mat oClusterLabelsDisplay = lv::getUniqueColorMap(oClusterLabels);
        cv::imshow(std::string("gmm_clusters_")+std::to_string(nCamIdx),oClusterLabelsDisplay);
        cv::waitKey(1);
    }
    const float fInterSpectrScale = STEREOSEGMATCH_SHPDIST_INTERSPEC_SCALE;
    const float fInterSpectrRatioTot = 1.0f+fInterSpectrScale;
    const float fInitDistScale = STEREOSEGMATCH_SHPDIST_INITDIST_SCALE;
    const float fMaxDist = STEREOSEGMATCH_SHPDIST_PX_MAX_CST;
    lvLog_(4,"Updating resegm graph model[%d] energy terms based on new features...",(int)nCamIdx);
    lv::StopWatch oLocalTimer;
#if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    lv::ProgressBarManager oProgressBarMgr("\tprogress:");
#endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    const cv::Mat oInputImg = m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img];
    const bool bUsing3ChannelInput = oInputImg.channels()==3;
    /*cv::Mat_<double> oFGProbMap(m_oGridSize),oBGProbMap(m_oGridSize);
    oFGProbMap = 0.0; oBGProbMap = 0.0;
    double dMinFGProb,dMinBGProb;
    dMinFGProb = dMinBGProb = 9999999;
    double dMaxFGProb,dMaxBGProb;
    dMaxFGProb = dMaxBGProb = 0;*/
#if USING_OPENMP
    #pragma omp parallel for
#endif //USING_OPENMP
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.anResegmUnaryFactIDs[nCamIdx]<m_anUnaryFactCounts[nCamIdx]);
        lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oNode.anResegmUnaryFactIDs[nCamIdx]).numberOfVariables()==size_t(1));
        const int nRowIdx = oNode.nRowIdx;
        const int nColIdx = oNode.nColIdx;
        // update unary terms for each grid node
        lvDbgAssert(nLUTNodeIdx==(oNode.nRowIdx*m_oGridSize[1]+oNode.nColIdx));
        lvDbgAssert(oNode.apResegmUnaryFuncs[nCamIdx] && oNode.anResegmUnaryFactIDs[nCamIdx]!=SIZE_MAX);
        ExplicitFunction& vUnaryResegmLUT = *oNode.apResegmUnaryFuncs[nCamIdx];
        lvDbgAssert(vUnaryResegmLUT.dimension()==1 && vUnaryResegmLUT.size()==m_nResegmLabels);
        const double dMinProbDensity = 1e-10;
        const double dMaxProbDensity = 1.0;
        const uchar* acInputColorSample = bUsing3ChannelInput?(oInputImg.data+nLUTNodeIdx*3):(oInputImg.data+nLUTNodeIdx);
        const float fInitFGDist = std::min(((float*)aInitFGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const float fCurrFGDist = std::min(((float*)aFGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const ValueType tFGDistUnaryCost = cost_cast((fCurrFGDist+fInitFGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE);
        lvDbgAssert(tFGDistUnaryCost>=cost_cast(0));
        const double dColorFGProb = std::min(std::max((bUsing3ChannelInput?m_aFGModels_3ch[nCamIdx](acInputColorSample):m_aFGModels_1ch[nCamIdx](acInputColorSample)),dMinProbDensity),dMaxProbDensity);
        /*((double*)oFGProbMap.data)[nLUTNodeIdx] = dColorFGProb;
        dMinFGProb = std::min(dMinFGProb,dColorFGProb);
        dMaxFGProb = std::max(dMaxFGProb,dColorFGProb);*/
        const ValueType tFGColorUnaryCost = cost_cast(-std::log2(dColorFGProb)*STEREOSEGMATCH_IMGSIM_COST_COLOR_SCALE);
        lvDbgAssert(tFGColorUnaryCost>=cost_cast(0));
        vUnaryResegmLUT(s_nForegroundLabelIdx) = std::min(tFGDistUnaryCost+tFGColorUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
        lvDbgAssert(vUnaryResegmLUT(s_nForegroundLabelIdx)>=cost_cast(0));
        const float fInitBGDist = std::min(((float*)aInitBGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const float fCurrBGDist = std::min(((float*)aBGDist[nCamIdx].data)[nLUTNodeIdx],fMaxDist);
        const ValueType tBGDistUnaryCost = cost_cast((fCurrBGDist+fInitBGDist*fInitDistScale)*STEREOSEGMATCH_SHPDIST_COST_SCALE);
        lvDbgAssert(tBGDistUnaryCost>=cost_cast(0));
        const double dColorBGProb = std::min(std::max((bUsing3ChannelInput?m_aBGModels_3ch[nCamIdx](acInputColorSample):m_aBGModels_1ch[nCamIdx](acInputColorSample)),dMinProbDensity),dMaxProbDensity);
        /*((double*)oBGProbMap.data)[nLUTNodeIdx] = dColorBGProb;
        dMinBGProb = std::min(dMinBGProb,dColorBGProb);
        dMaxBGProb = std::max(dMaxBGProb,dColorBGProb);*/
        const ValueType tBGColorUnaryCost = cost_cast(-std::log2(dColorBGProb)*STEREOSEGMATCH_IMGSIM_COST_COLOR_SCALE);
        lvDbgAssert(tBGColorUnaryCost>=cost_cast(0));
        vUnaryResegmLUT(s_nBackgroundLabelIdx) = std::min(tBGDistUnaryCost+tBGColorUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
        lvDbgAssert(vUnaryResegmLUT(s_nBackgroundLabelIdx)>=cost_cast(0));
        const InternalLabelType nStereoLabelIdx = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data)[nLUTNodeIdx];
        const int nOffsetColIdx = (nStereoLabelIdx<m_nRealStereoLabels)?getOffsetColIdx(nCamIdx,nColIdx,nStereoLabelIdx):INT_MAX;
        if(nOffsetColIdx>=0 && nOffsetColIdx<nCols && m_aROIs[nCamIdx^1](nRowIdx,nOffsetColIdx)) {
            const float fInitOffsetFGDist = std::min(aInitFGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const float fCurrOffsetFGDist = std::min(aFGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const ValueType tAddedFGDistUnaryCost = cost_cast((fCurrOffsetFGDist+fInitOffsetFGDist*fInitDistScale)*fInterSpectrScale*STEREOSEGMATCH_SHPDIST_COST_SCALE);
            vUnaryResegmLUT(s_nForegroundLabelIdx) = std::min(vUnaryResegmLUT(s_nForegroundLabelIdx)+tAddedFGDistUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            const float fInitOffsetBGDist = std::min(aInitBGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const float fCurrOffsetBGDist = std::min(aBGDist[nCamIdx^1](nRowIdx,nOffsetColIdx),fMaxDist);
            const ValueType tAddedBGDistUnaryCost = cost_cast((fCurrOffsetBGDist+fInitOffsetBGDist*fInitDistScale)*fInterSpectrScale*STEREOSEGMATCH_SHPDIST_COST_SCALE);
            vUnaryResegmLUT(s_nBackgroundLabelIdx) = std::min(vUnaryResegmLUT(s_nBackgroundLabelIdx)+tAddedBGDistUnaryCost,STEREOSEGMATCH_UNARY_COST_MAXTRUNC_CST);
            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                PairwClique& oClique = oNode.aaResegmPairwCliques[nCamIdx][nOrientIdx];
                if(oClique) {
                    lvDbgAssert(oClique.m_nGraphFactorId>=m_anUnaryFactCounts[nCamIdx]);
                    lvDbgAssert(oClique.m_nGraphFactorId<m_anUnaryFactCounts[nCamIdx]+m_anPairwFactCounts[nCamIdx]);
                    lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(oClique.m_anGraphNodeIdxs[0]!=SIZE_MAX && oClique.m_anLUTNodeIdxs[0]!=SIZE_MAX);
                    lvDbgAssert(oClique.m_anGraphNodeIdxs[1]!=SIZE_MAX && oClique.m_anLUTNodeIdxs[1]!=SIZE_MAX);
                    lvDbgAssert(m_vNodeInfos[oClique.m_anLUTNodeIdxs[1]].abValidGraphNode[nCamIdx] && oClique.m_pGraphFunctionPtr);
                    ExplicitFunction& vPairwResegmLUT = *oClique.m_pGraphFunctionPtr;
                    lvDbgAssert(vPairwResegmLUT.dimension()==2 && vPairwResegmLUT.size()==m_nResegmLabels*m_nResegmLabels);
                    const int nLocalGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx));
                    const int nOffsetGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx^1]:(nOrientIdx==1)?aGradX[nCamIdx^1]:aGradMag[nCamIdx^1])(nRowIdx,nOffsetColIdx));
                    const float fLocalScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    const float fOffsetScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nOffsetGrad);
                    lvDbgAssert(fLocalScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    lvDbgAssert(fOffsetScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nOffsetGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    //const float fScaleFact = std::min(fLocalScaleFact,fOffsetScaleFact);
                    //const float fScaleFact = fLocalScaleFact*fOffsetScaleFact;
                    const float fScaleFact = (fLocalScaleFact+fOffsetScaleFact*fInterSpectrScale)/fInterSpectrRatioTot;
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nResegmLabels; ++nLabelIdx1) {
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nResegmLabels; ++nLabelIdx2) {
                            vPairwResegmLUT(nLabelIdx1,nLabelIdx2) = cost_cast((nLabelIdx1^nLabelIdx2)*fScaleFact*STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST);
                        }
                    }
                }
            }
        }
        else {
            vUnaryResegmLUT(s_nForegroundLabelIdx) = STEREOSEGMATCH_UNARY_COST_OOB_CST;
            vUnaryResegmLUT(s_nBackgroundLabelIdx) = cost_cast(0);
            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                PairwClique& oClique = oNode.aaResegmPairwCliques[nCamIdx][nOrientIdx];
                if(oClique) {
                    lvDbgAssert(oClique.m_nGraphFactorId>=m_anUnaryFactCounts[nCamIdx]);
                    lvDbgAssert(oClique.m_nGraphFactorId<m_anUnaryFactCounts[nCamIdx]+m_anPairwFactCounts[nCamIdx]);
                    lvDbgAssert(m_apResegmModels[nCamIdx]->operator[](oClique.m_nGraphFactorId).numberOfVariables()==size_t(2));
                    lvDbgAssert(oClique.m_anGraphNodeIdxs[0]!=SIZE_MAX && oClique.m_anLUTNodeIdxs[0]!=SIZE_MAX);
                    lvDbgAssert(oClique.m_anGraphNodeIdxs[1]!=SIZE_MAX && oClique.m_anLUTNodeIdxs[1]!=SIZE_MAX);
                    lvDbgAssert(m_vNodeInfos[oClique.m_anLUTNodeIdxs[1]].abValidGraphNode[nCamIdx] && oClique.m_pGraphFunctionPtr);
                    ExplicitFunction& vPairwResegmLUT = *oClique.m_pGraphFunctionPtr;
                    lvDbgAssert(vPairwResegmLUT.dimension()==2 && vPairwResegmLUT.size()==m_nResegmLabels*m_nResegmLabels);
                    const int nLocalGrad = (int)(((nOrientIdx==0)?aGradY[nCamIdx]:(nOrientIdx==1)?aGradX[nCamIdx]:aGradMag[nCamIdx])(nRowIdx,nColIdx));
                    const float fLocalScaleFact = m_aLabelSimCostGradFactLUT.eval_raw(nLocalGrad);
                    lvDbgAssert(fLocalScaleFact==(float)std::exp(float(STEREOSEGMATCH_LBLSIM_COST_GRADPIVOT_CST-nLocalGrad)/STEREOSEGMATCH_LBLSIM_COST_GRADRAW_SCALE));
                    for(InternalLabelType nLabelIdx1=0; nLabelIdx1<m_nResegmLabels; ++nLabelIdx1) {
                        for(InternalLabelType nLabelIdx2=0; nLabelIdx2<m_nResegmLabels; ++nLabelIdx2) {
                            vPairwResegmLUT(nLabelIdx1,nLabelIdx2) = cost_cast((nLabelIdx1^nLabelIdx2)*fLocalScaleFact*STEREOSEGMATCH_LBLSIM_RESEGM_SCALE_CST);
                        }
                    }
                }
            }
        }
    #if STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
        if(lv::getVerbosity()>=3)
            oProgressBarMgr.update(float(nGraphNodeIdx)/m_anValidGraphNodes[nCamIdx]);
    #endif //STEREOSEGMATCH_CONFIG_USE_PROGRESS_BARS
    }
    /*cv::imshow(std::string("oFGProbMap_")+std::to_string(nCamIdx),oFGProbMap);
    lvCout << " fg : min=" << dMinFGProb << ", max=" << dMaxFGProb << std::endl;
    cv::imshow(std::string("oBGProbMap_")+std::to_string(nCamIdx),oBGProbMap);
    lvCout << " bg : min=" << dMinBGProb << ", max=" << dMaxBGProb << std::endl;
    cv::waitKey(0);*/
    lvLog_(4,"Resegm graph model[%d] energy terms update completed in %f second(s).",(int)nCamIdx,oLocalTimer.tock());
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
    calcImageFeatures(CamArray<cv::Mat>{aInputs[InputPack_LeftImg],aInputs[InputPack_RightImg]});
    calcShapeFeatures(CamArray<cv::Mat_<InternalLabelType>>{aInputs[InputPack_LeftMask],aInputs[InputPack_RightMask]});
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_FGDist].copyTo(m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_InitFGDist]);
        m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_BGDist].copyTo(m_vNextFeats[nCamIdx*FeatPackOffset+FeatPackOffset_InitBGDist]);
    }
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

inline void StereoSegmMatcher::GraphModelData::calcImageFeatures(const CamArray<cv::Mat>& aInputImages) {
    static_assert(getCameraCount()==2,"bad input image array size");
    lvDbgExceptionWatch;
    for(size_t nInputIdx=0; nInputIdx<aInputImages.size(); ++nInputIdx) {
        lvDbgAssert__(aInputImages[nInputIdx].dims==2 && m_oGridSize==aInputImages[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputImages[nInputIdx].type()==CV_8UC1 || aInputImages[nInputIdx].type()==CV_8UC3,"unexpected input image type");
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvLog(3,"Calculating image features maps...");
    const int nWinRadius = (int)m_nGridBorderSize;
    const int nWinSize = nWinRadius*2+1;
    CamArray<cv::Mat> aEnlargedInput;
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    lvIgnore(nWinSize);
    CamArray<cv::Mat_<float>> aEnlargedDescs,aDescs;
    const int nPatchSize = STEREOSEGMATCH_DEFAULT_DESC_PATCH_SIZE;
#else //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    CamArray<cv::Mat_<uchar>> aEnlargedROIs;
    const int nPatchSize = nWinSize;
#endif //STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    lvAssert_((nPatchSize%2)==1,"patch sizes must be odd");
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        cv::copyMakeBorder(aInputImages[nCamIdx],aEnlargedInput[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_DEFAULT);
    #if STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3)
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
        cv::copyMakeBorder(m_aROIs[nCamIdx],aEnlargedROIs[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_CONSTANT,cv::Scalar(0));
    #elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
        if(aEnlargedInput[nCamIdx].channels()==3)
            cv::cvtColor(aEnlargedInput[nCamIdx],aEnlargedInput[nCamIdx],cv::COLOR_BGR2GRAY);
        aEnlargedInput[nCamIdx].convertTo(aEnlargedInput[nCamIdx],CV_64F,(1.0/UCHAR_MAX)/nWinSize);
        aEnlargedInput[nCamIdx] -= cv::mean(aEnlargedInput[nCamIdx])[0];
        cv::copyMakeBorder(m_aROIs[nCamIdx],aEnlargedROIs[nCamIdx],nWinRadius,nWinRadius,nWinRadius,nWinRadius,cv::BORDER_CONSTANT,cv::Scalar(0));
    #elif STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        lvLog_(3,"\tcam[%d] image descriptors...",(int)nCamIdx);
        m_pImgDescExtractor->compute2(aEnlargedInput[nCamIdx],aEnlargedDescs[nCamIdx]);
        lvDbgAssert(aEnlargedDescs[nCamIdx].dims==3 && aEnlargedDescs[nCamIdx].size[0]==nRows+nWinRadius*2 && aEnlargedDescs[nCamIdx].size[1]==nCols+nWinRadius*2);
        std::vector<cv::Range> vRanges(size_t(3),cv::Range::all());
        vRanges[0] = cv::Range(nWinRadius,nRows+nWinRadius);
        vRanges[1] = cv::Range(nWinRadius,nCols+nWinRadius);
        aEnlargedDescs[nCamIdx](vRanges.data()).copyTo(aDescs[nCamIdx]); // copy to avoid bugs when reshaping non-continuous data
        lvDbgAssert(aDescs[nCamIdx].dims==3 && aDescs[nCamIdx].size[0]==nRows && aDescs[nCamIdx].size[1]==nCols);
        lvDbgAssert(std::equal(aDescs[nCamIdx].ptr<float>(0,0),aDescs[nCamIdx].ptr<float>(0,0)+aDescs[nCamIdx].size[2],aEnlargedDescs[nCamIdx].ptr<float>(nWinRadius,nWinRadius)));
    #if STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        const size_t nDescSize = size_t(aDescs[nCamIdx].size[2]);
        for(size_t nDescIdx=0; nDescIdx<aDescs[nCamIdx].total(); nDescIdx+=nDescSize)
            lv::rootSIFT(((float*)aDescs[nCamIdx].data)+nDescIdx,nDescSize);
    #endif //STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
    #endif //STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        lvLog_(3,"\tcam[%d] image gradient magnitudes...",(int)nCamIdx);
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
    lvLog_(3,"Image features maps computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating image affinity map...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    m_vNextFeats[FeatPack_ImgAffinity].create(3,anAffinityMapDims.data(),CV_32FC1);
    cv::Mat_<float> oAffinity = m_vNextFeats[FeatPack_ImgAffinity];
    std::vector<int> vDisparityOffsets;
    for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
        vDisparityOffsets.push_back(getOffsetValue(0,nLabelIdx));
    // note: we only create the dense affinity map for 1st cam here; affinity for 2nd cam will be deduced from it
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
    lv::computeDescriptorAffinity(aDescs[0],aDescs[1],nPatchSize,oAffinity,vDisparityOffsets,lv::AffinityDist_L2,m_aROIs[0],m_aROIs[1]);
#elif STEREOSEGMATCH_CONFIG_USE_MI_AFFINITY
    lv::computeImageAffinity(aEnlargedInput[0],aEnlargedInput[1],nWinSize,oAffinity,vDisparityOffsets,lv::AffinityDist_MI,aEnlargedROIs[0],aEnlargedROIs[1]);
#elif STEREOSEGMATCH_CONFIG_USE_SSQDIFF_AFFINITY
    lv::computeImageAffinity(aEnlargedInput[0],aEnlargedInput[1],nWinSize,oAffinity,vDisparityOffsets,lv::AffinityDist_SSD,aEnlargedROIs[0],aEnlargedROIs[1]);
#endif //STEREOSEGMATCH_CONFIG_USE_..._AFFINITY
    lvDbgAssert(lv::MatInfo(oAffinity)==lv::MatInfo(lv::MatSize(3,anAffinityMapDims.data()),CV_32FC1));
    lvDbgAssert(m_vNextFeats[FeatPack_ImgAffinity].data==oAffinity.data);
    lvLog_(3,"Image affinity map computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating image saliency maps...");
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
#if STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
            const float fCurrDescSparseness = (float)lv::sparseness(aDescs[nCamIdx].ptr<float>(nRowIdx,nColIdx),size_t(aDescs[nCamIdx].size[2]));
            oSaliency.at<float>(nRowIdx,nColIdx) = std::max(fCurrDescSparseness,fCurrDistSparseness);
#else //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
            oSaliency.at<float>(nRowIdx,nColIdx) = fCurrDistSparseness;
#endif //!STEREOSEGMATCH_CONFIG_USE_DESC_BASED_AFFINITY
        }
        cv::normalize(oSaliency,oSaliency,1,0,cv::NORM_MINMAX,-1,m_aROIs[nCamIdx]);
        lvDbgExec( // cv::normalize leftover fp errors are sometimes awful; need to 0-max when using map
            for(int nRowIdx=0; nRowIdx<oSaliency.rows; ++nRowIdx)
                for(int nColIdx=0; nColIdx<oSaliency.cols; ++nColIdx)
                    lvDbgAssert((oSaliency.at<float>(nRowIdx,nColIdx)>=-1e-6f && oSaliency.at<float>(nRowIdx,nColIdx)<=1.0f+1e-6f) || m_aROIs[nCamIdx](nRowIdx,nColIdx)==0);
        );
    #if STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        cv::multiply(oSaliency,cv::Mat_<float>(oSaliency.size(),1.0f).setTo(0.5f,m_aDescROIs[nCamIdx]==0),oSaliency);
    #endif //STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        if(lv::getVerbosity()>=4) {
            cv::imshow(std::string("oSaliency_img_")+std::to_string(nCamIdx),oSaliency);
            cv::waitKey(1);
        }
    }
    lvLog_(3,"Image saliency maps computed in %f second(s).",oLocalTimer.tock());
    /*if(m_pDisplayHelper) {
        std::vector<std::pair<cv::Mat,std::string>> vAffMaps;
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
            vAffMaps.push_back(std::pair<cv::Mat,std::string>(lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx)),std::string()));
        m_pDisplayHelper->displayAlbumAndWaitKey(vAffMaps);
    }*/
}

inline void StereoSegmMatcher::GraphModelData::calcShapeFeatures(const CamArray<cv::Mat_<InternalLabelType>>& aInputMasks) {
    static_assert(getCameraCount()==2,"bad input mask array size");
    lvDbgExceptionWatch;
    for(size_t nInputIdx=0; nInputIdx<aInputMasks.size(); ++nInputIdx) {
        lvDbgAssert__(aInputMasks[nInputIdx].dims==2 && m_oGridSize==aInputMasks[nInputIdx].size(),"input at index=%d had the wrong size",(int)nInputIdx);
        lvDbgAssert_(aInputMasks[nInputIdx].type()==CV_8UC1,"unexpected input mask type");
    }
    const int nRows = (int)m_oGridSize(0);
    const int nCols = (int)m_oGridSize(1);
    lvLog(3,"Calculating shape features maps...");
    CamArray<cv::Mat_<float>> aDescs;
    const int nPatchSize = STEREOSEGMATCH_DEFAULT_DESC_PATCH_SIZE;
    lvAssert_((nPatchSize%2)==1,"patch sizes must be odd");
    lv::StopWatch oLocalTimer;
#if USING_OPENMP
    //#pragma omp parallel for num_threads(getCameraCount())
#endif //USING_OPENMP
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        const cv::Mat& oInputMask = aInputMasks[nCamIdx];
        // @@@@@ use swap mask to prevent recalc of identical descriptors? (swap + radius based on desc)
        // LOW PRIORITY (segm will swap a lot every iter, might cover pretty much everything in segmented ROIs)
        // convert 8ui mask to keypoint list, then pass to compute2 (keeps desc struct/old values for untouched kps)
        // also base 8ui mask w.r.t fg dist? (2x radius = too far)
        // @@@@ use also for aff maps? or too fast to care?
        lvLog_(3,"\tcam[%d] shape descriptors...",(int)nCamIdx);
        m_pShpDescExtractor->compute2(oInputMask,aDescs[nCamIdx]);
        lvDbgAssert(aDescs[nCamIdx].dims==3 && aDescs[nCamIdx].size[0]==nRows && aDescs[nCamIdx].size[1]==nCols);
    #if STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        const size_t nDescSize = size_t(aDescs[nCamIdx].size[2]);
        for(size_t nDescIdx=0; nDescIdx<aDescs[nCamIdx].total(); nDescIdx+=nDescSize)
            lv::rootSIFT(((float*)aDescs[nCamIdx].data)+nDescIdx,nDescSize);
    #endif //STEREOSEGMATCH_CONFIG_USE_ROOT_SIFT_DESCS
        lvLog_(3,"\tcam[%d] shape distance fields...",(int)nCamIdx);
        calcShapeDistFeatures(aInputMasks[nCamIdx],nCamIdx);
    }
    lvLog_(3,"Shape features maps computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating shape affinity map...");
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,(int)m_nRealStereoLabels};
    m_vNextFeats[FeatPack_ShpAffinity].create(3,anAffinityMapDims.data(),CV_32FC1);
    cv::Mat_<float> oAffinity = m_vNextFeats[FeatPack_ShpAffinity];
    std::vector<int> vDisparityOffsets;
    for(InternalLabelType nLabelIdx = 0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
        vDisparityOffsets.push_back(getOffsetValue(0,nLabelIdx));
#if STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
    lv::computeDescriptorAffinity(aDescs[0],aDescs[1],nPatchSize,oAffinity,vDisparityOffsets,lv::AffinityDist_EMD,m_aROIs[0],m_aROIs[1],m_pShpDescExtractor->getEMDCostMap());
#else //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
    lv::computeDescriptorAffinity(aDescs[0],aDescs[1],nPatchSize,oAffinity,vDisparityOffsets,lv::AffinityDist_L2,m_aROIs[0],m_aROIs[1]);
#endif //!STEREOSEGMATCH_CONFIG_USE_SHAPE_EMD_AFFIN
    lvDbgAssert(lv::MatInfo(oAffinity)==lv::MatInfo(lv::MatSize(3,anAffinityMapDims.data()),CV_32FC1));
    lvDbgAssert(m_vNextFeats[FeatPack_ShpAffinity].data==oAffinity.data);
    lvLog_(3,"Shape affinity map computed in %f second(s).",oLocalTimer.tock());
    lvLog(3,"Calculating shape saliency maps...");
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
                    lvDbgAssert((oSaliency.at<float>(nRowIdx,nColIdx)>=-1e-6f && oSaliency.at<float>(nRowIdx,nColIdx)<=1.0f+1e-6f) || m_aROIs[nCamIdx](nRowIdx,nColIdx)==0);
        );
    #if STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        cv::multiply(oSaliency,cv::Mat_<float>(oSaliency.size(),1.0f).setTo(0.5f,m_aDescROIs[nCamIdx]==0),oSaliency);
    #endif //STEREOSEGMATCH_CONFIG_USE_SALIENT_MAP_BORDR
        if(lv::getVerbosity()>=4) {
            cv::imshow(std::string("oSaliency_shp_")+std::to_string(nCamIdx),oSaliency);
            cv::waitKey(1);
        }
    }
    lvLog_(3,"Shape saliency maps computed in %f second(s).",oLocalTimer.tock());
    /*if(m_pDisplayHelper) {
        std::vector<std::pair<cv::Mat,std::string>> vAffMaps;
        for(InternalLabelType nLabelIdx=0; nLabelIdx<m_nRealStereoLabels; ++nLabelIdx)
            vAffMaps.push_back(std::pair<cv::Mat,std::string>(lv::squeeze(lv::getSubMat(oAffinity,2,(int)nLabelIdx)),std::string()));
        m_pDisplayHelper->displayAlbumAndWaitKey(vAffMaps);
    }*/
}

inline void StereoSegmMatcher::GraphModelData::calcShapeDistFeatures(const cv::Mat_<InternalLabelType>& oInputMask, size_t nCamIdx) {
    lvDbgAssert_(nCamIdx<getCameraCount(),"bad input cam index");
    lvDbgAssert_(oInputMask.dims==2 && m_oGridSize==oInputMask.size(),"input had the wrong size");
    lvDbgAssert_(oInputMask.type()==CV_8UC1,"unexpected input mask type");
    lvDbgExceptionWatch;
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

inline int StereoSegmMatcher::GraphModelData::getOffsetValue(size_t nCamIdx, InternalLabelType nLabel) const {
    static_assert(getCameraCount()==2,"bad hardcoded offset sign");
    lvDbgExceptionWatch;
    lvDbgAssert(nCamIdx==size_t(0) || nCamIdx==size_t(1));
    lvDbgAssert(nLabel<m_nRealStereoLabels);
    const OutputLabelType nRealLabel = getRealLabel(nLabel);
    lvDbgAssert((int)nRealLabel>=(int)m_nMinDispOffset && (int)nRealLabel<=(int)m_nMaxDispOffset);
    return (nCamIdx==size_t(0))?(-nRealLabel):(nRealLabel);
}

inline int StereoSegmMatcher::GraphModelData::getOffsetColIdx(size_t nCamIdx, int nColIdx, InternalLabelType nLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nColIdx>=0 && nColIdx<(int)m_oGridSize[1]);
    return nColIdx+getOffsetValue(nCamIdx,nLabel);
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
    return cost_cast(100000); // @@@@ dirty
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
    return -cost_cast(100000); // @@@@ dirty
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcTotalAssocCost(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
    ValueType tEnergy = cost_cast(0);
    const int nColIdxStart = ((nCamIdx==size_t(1))?int(m_nMinDispOffset):-int(m_nMaxDispOffset));
    const int nColIdxEnd = ((nCamIdx==size_t(1))?int(m_oGridSize[1]+m_nMaxDispOffset):int(m_oGridSize[1]-m_nMinDispOffset));
    for(int nRowIdx=0; nRowIdx<(int)m_oGridSize[0]; ++nRowIdx)
        for(int nColIdx=nColIdxStart; nColIdx<nColIdxEnd; nColIdx+=m_nDispOffsetStep)
            tEnergy += m_aAssocCostRealSumLUT[getAssocCount(nCamIdx,nRowIdx,nColIdx)];
    lvDbgAssert(tEnergy>=cost_cast(0));
    return tEnergy;
}

inline StereoSegmMatcher::ValueType StereoSegmMatcher::GraphModelData::calcStereoUnaryMoveCost(size_t nCamIdx, size_t nGraphNodeIdx, InternalLabelType nOldLabel, InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]);
    const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
    const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
    lvDbgAssert(oNode.abValidGraphNode[nCamIdx]);
    if(nOldLabel!=nNewLabel) {
        lvDbgAssert(nOldLabel<m_nStereoLabels && nNewLabel<m_nStereoLabels);
        const ValueType tAssocEnergyCost = calcRemoveAssocCost(nCamIdx,oNode.nRowIdx,oNode.nColIdx,nOldLabel)+calcAddAssocCost(nCamIdx,oNode.nRowIdx,oNode.nColIdx,nNewLabel);
        const ExplicitFunction& vUnaryStereoLUT = *oNode.apStereoUnaryFuncs[nCamIdx];
        const ValueType tUnaryEnergyInit = vUnaryStereoLUT(nOldLabel);
        const ValueType tUnaryEnergyModif = vUnaryStereoLUT(nNewLabel);
        return tAssocEnergyCost+tUnaryEnergyModif-tUnaryEnergyInit;
    }
    else
        return cost_cast(0);
}

#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)

inline void StereoSegmMatcher::GraphModelData::calcStereoMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1 && m_oGridSize==m_aStereoUnaryCosts[nCamIdx].size);
    const InternalLabelType* pInitLabeling = ((InternalLabelType*)m_aStereoLabelings[nCamIdx].data);
    // @@@@@ openmp here?
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        const InternalLabelType& nInitLabel = pInitLabeling[nLUTNodeIdx];
        lvIgnore(oNode);
        lvDbgAssert(oNode.abValidGraphNode[nCamIdx]);
        lvDbgAssert(&nInitLabel==&m_aStereoLabelings[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
        ValueType& tUnaryCost = ((ValueType*)m_aStereoUnaryCosts[nCamIdx].data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_aStereoUnaryCosts[nCamIdx](m_vNodeInfos[nLUTNodeIdx].nRowIdx,m_vNodeInfos[nLUTNodeIdx].nColIdx));
    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
        tUnaryCost = calcStereoUnaryMoveCost(nCamIdx,nGraphNodeIdx,nInitLabel,nNewLabel);
    #elif STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
        tUnaryCost = -calcStereoUnaryMoveCost(nCamIdx,nGraphNodeIdx,nInitLabel,nNewLabel);
        for(const auto& p : m_node_clique_list[nGraphNodeIdx])
            tUnaryCost += dualVariable((int)p.first,(int)p.second,nInitLabel) - dualVariable((int)p.first,(int)p.second,nNewLabel);
    #endif //STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
    }
}

#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF)

#if STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF

inline void StereoSegmMatcher::GraphModelData::SetupAlphaEnergy(SubmodularIBFS<ValueType,VarId>& crf) {
    crf.ClearUnaries();
    crf.AddConstantTerm(-crf.GetConstantTerm());
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const InternalLabelType& nInitLabel = pLabeling[nLUTNodeIdx];
        lvDbgAssert(&nInitLabel==&m_aStereoLabelings[__cam](m_vNodeInfos[nLUTNodeIdx].nRowIdx,m_vNodeInfos[nLUTNodeIdx].nColIdx));
        const ValueType& tUnaryCost = ((ValueType*)m_aStereoUnaryCosts[__cam].data)[nLUTNodeIdx];
        lvDbgAssert(&tUnaryCost==&m_aStereoUnaryCosts[__cam](m_vNodeInfos[nLUTNodeIdx].nRowIdx,m_vNodeInfos[nLUTNodeIdx].nColIdx));
        if (tUnaryCost>cost_cast(0))
            crf.AddUnaryTerm((int)nGraphNodeIdx, tUnaryCost, 0);
        else
            crf.AddUnaryTerm((int)nGraphNodeIdx, 0, -tUnaryCost);
    }
}

inline bool StereoSegmMatcher::GraphModelData::InitialFusionLabeling() {
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        if(pLabeling[nLUTNodeIdx]!=__alpha)
            return true;
    }
    return false;
}

inline void StereoSegmMatcher::GraphModelData::PreEditDual(SubmodularIBFS<ValueType,VarId>& crf) {
    auto& fixedVars = crf.Params().fixedVars;
    fixedVars.resize(m_anValidGraphNodes[__cam]);
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        fixedVars[nGraphNodeIdx] = (pLabeling[nLUTNodeIdx]==__alpha);
    }
    // Allocate all the buffers we need in one place, resize as necessary
    Label label_buf[32];
    std::vector<Label> current_labels;
    std::vector<Label> fusion_labels;
    std::vector<REAL> psi;
    std::vector<REAL> current_lambda;
    std::vector<REAL> fusion_lambda;

    auto& crf_cliques = crf.Graph().GetCliques();
    lvDbgAssert(crf_cliques.size() == m_nStereoCliqueCount);
    int clique_index = 0;
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.abValidGraphNode[__cam]);
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[__cam][nOrientIdx];
            if(oStereoClique) {
                constexpr size_t nCliqueSize = oStereoClique.s_nCliqueSize;
                auto& lambda_a = lambdaAlpha(clique_index);
                auto& crf_c = crf_cliques[clique_index];
                lvDbgAssert(nCliqueSize == crf_c.Size());
                std::vector<REAL>& energy_table = crf_c.EnergyTable();
                sospd::Assgn max_assgn = sospd::Assgn(size_t(1)<<nCliqueSize);
                lvDbgAssert(energy_table.size() == max_assgn);
                psi.resize(nCliqueSize);
                current_labels.resize(nCliqueSize);
                fusion_labels.resize(nCliqueSize);
                current_lambda.resize(nCliqueSize);
                fusion_lambda.resize(nCliqueSize);
                for(size_t i = 0; i < nCliqueSize; ++i) {
                    current_labels[i] = pLabeling[m_avValidLUTNodeIdxs[__cam][crf_c.Nodes()[i]]];
                    fusion_labels[i] = __alpha;
                    current_lambda[i] = dualVariable(lambda_a, i, current_labels[i]);
                    fusion_lambda[i] = dualVariable(lambda_a, i, fusion_labels[i]);
                }
                // compute costs of all fusion assignments
                const ExplicitFunction& vPairwStereoLUT = *oStereoClique.m_pGraphFunctionPtr;
                sospd::Assgn last_gray = 0;
                for(size_t i_idx = 0; i_idx < nCliqueSize; ++i_idx)
                    label_buf[i_idx] = current_labels[i_idx];
                energy_table[0] = vPairwStereoLUT(label_buf);
                for(sospd::Assgn a = 1; a < max_assgn; ++a) {
                    sospd::Assgn gray = a ^ (a >> 1);
                    sospd::Assgn diff = gray ^ last_gray;
                    int changed_idx = __builtin_ctz(diff);
                    if (diff & gray)
                        label_buf[changed_idx] = fusion_labels[changed_idx];
                    else
                        label_buf[changed_idx] = current_labels[changed_idx];
                    last_gray = gray;
                    energy_table[gray] = vPairwStereoLUT(label_buf);
                }
                // compute the residual function: g(S) - lambda_fusion(S) - lambda_current(C\S)
                sospd::SubtractLinear(nCliqueSize,energy_table,fusion_lambda,current_lambda);
                lvDbgAssert(energy_table[0] == 0); // check tightness of current labeling
                ++clique_index;
            }
        }
    }
}

inline bool StereoSegmMatcher::GraphModelData::UpdatePrimalDual(SubmodularIBFS<ValueType,VarId>& crf) {
    bool ret = false;
    SetupAlphaEnergy(crf);
    crf.Solve();
    cv::Mat_<InternalLabelType>& oCurrStereoLabeling = m_aStereoLabelings[__cam];
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const int& nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
        const int& nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
        const int nMoveLabel = crf.GetLabel((int)nGraphNodeIdx);
        lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
        if(nMoveLabel==1) { // node label changed to alpha
            const InternalLabelType nOldLabel = oCurrStereoLabeling(nRowIdx,nColIdx);
            if(nOldLabel!=__alpha) {
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(__cam,nRowIdx,nColIdx,nOldLabel);
                oCurrStereoLabeling(nRowIdx,nColIdx) = __alpha;
                if(__alpha<m_nDontCareLabelIdx)
                    addAssoc(__cam,nRowIdx,nColIdx,__alpha);
                ret = true;
            }
        }
    }
    const auto& clique = crf.Graph().GetCliques();
    size_t nCliqueIdx = 0;
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.abValidGraphNode[__cam]);
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[__cam][nOrientIdx];
            if(oStereoClique) {
                auto& crf_c = clique[nCliqueIdx];
                const std::vector<REAL>& phiCi = crf_c.AlphaCi();
                for (size_t j = 0; j < phiCi.size(); ++j) {
                    dualVariable((int)nCliqueIdx, (int)j, __alpha) += phiCi[j];
                    Height(crf_c.Nodes()[j], __alpha) += phiCi[j];
                }
                ++nCliqueIdx;
            }
        }
    }
    return ret;
}

inline void StereoSegmMatcher::GraphModelData::PostEditDual(SubmodularIBFS<ValueType,VarId>& crf/*temp for clique nodes & dbg*/) {
    Label labelBuf[32];
    int clique_index = 0;
    const auto& clique = crf.Graph().GetCliques();
    const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[__cam].data);
    for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[__cam]; ++nGraphNodeIdx) {
        const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[__cam][nGraphNodeIdx];
        const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
        lvDbgAssert(oNode.abValidGraphNode[__cam]);
        for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
            const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[__cam][nOrientIdx];
            if(oStereoClique) {
                auto& crf_c = clique[clique_index];
                int k = (int)crf_c.Nodes().size();
                ASSERT(k < 32);
                REAL lambdaSum = 0;
                for (int i = 0; i < k; ++i) {
                    labelBuf[i] = pLabeling[m_avValidLUTNodeIdxs[__cam][crf_c.Nodes()[i]]];
                    lambdaSum += dualVariable(clique_index, i, labelBuf[i]);
                }
                const ExplicitFunction& vPairwStereoLUT = *oStereoClique.m_pGraphFunctionPtr;
                REAL energy = vPairwStereoLUT(labelBuf);
                REAL correction = energy - lambdaSum;
                if (correction > 0) {
                    std::cout << "Bad clique in PostEditDual!\t Id:" << clique_index << "\n";
                    std::cout << "Correction: " << correction << "\tenergy: " << energy << "\tlambdaSum " << lambdaSum << "\n";
                    const auto& c = crf.Graph().GetCliques()[clique_index];
                    std::cout << "EnergyTable: ";
                    for (const auto& e : c.EnergyTable())
                        std::cout << e << ", ";
                    std::cout << "\n";
                }
                ASSERT(correction <= 0);
                REAL avg = correction / k;
                int remainder = correction % k;
                if (remainder < 0) {
                    avg -= 1;
                    remainder += k;
                }
                for (int i = 0; i < k; ++i) {
                    auto& lambda_ail = dualVariable(clique_index,  i, labelBuf[i]);
                    Height(crf_c.Nodes()[i], labelBuf[i]) -= lambda_ail;
                    lambda_ail += avg;
                    if (i < remainder)
                        lambda_ail += 1;
                    Height(crf_c.Nodes()[i], labelBuf[i]) += lambda_ail;
                }
                ++clique_index;
            }
        }
    }
}

#endif //STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF

#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)

inline void StereoSegmMatcher::GraphModelData::calcResegmMoveCosts(size_t nCamIdx, InternalLabelType nNewLabel) const {
    lvDbgExceptionWatch;
    lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size() && m_oGridSize.total()>1 && m_oGridSize==m_aResegmLabelings[nCamIdx].size);
    lvDbgAssert(m_oGridSize==m_aResegmUnaryCosts[nCamIdx].size);
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
            const ExplicitFunction& vUnaryResegmLUT = *oNode.apResegmUnaryFuncs[nCamIdx];
            const ValueType tEnergyInit = vUnaryResegmLUT(nInitLabel);
            const ValueType tEnergyModif = vUnaryResegmLUT(nNewLabel);
            tUnaryCost = tEnergyModif-tEnergyInit;
        }
        else
            tUnaryCost = cost_cast(0);
    }
}

#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF || STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF)

inline opengm::InferenceTermination StereoSegmMatcher::GraphModelData::infer(size_t nPrimaryCamIdx) {
    static_assert(s_nInputArraySize==4 && getCameraCount()==2,"hardcoded indices below will break");
    lvAssert_(nPrimaryCamIdx==size_t(0) || nPrimaryCamIdx==size_t(1),"bad primary camera index");
    lvDbgExceptionWatch;
    const size_t nSecondaryCamIdx = nPrimaryCamIdx^1;
    if(lv::getVerbosity()>=3) {
        cv::Mat oTargetImg = m_aInputs[nPrimaryCamIdx*InputPackOffset+InputPackOffset_Img].clone();
        if(oTargetImg.channels()==1)
            cv::cvtColor(oTargetImg,oTargetImg,cv::COLOR_GRAY2BGR);
        cv::Mat oTargetMask = m_aInputs[nPrimaryCamIdx*InputPackOffset+InputPackOffset_Mask].clone();
        cv::cvtColor(oTargetMask,oTargetMask,cv::COLOR_GRAY2BGR);
        oTargetMask &= cv::Vec3b(255,0,0);
        cv::imshow("target input",(oTargetImg+oTargetMask)/2);
        cv::Mat oOtherImg = m_aInputs[(nSecondaryCamIdx)*InputPackOffset+InputPackOffset_Img].clone();
        if(oOtherImg.channels()==1)
            cv::cvtColor(oOtherImg,oOtherImg,cv::COLOR_GRAY2BGR);
        cv::Mat oOtherMask = m_aInputs[(nSecondaryCamIdx)*InputPackOffset+InputPackOffset_Mask].clone();
        cv::cvtColor(oOtherMask,oOtherMask,cv::COLOR_GRAY2BGR);
        oOtherMask &= cv::Vec3b(255,0,0);
        cv::imshow("other input",(oOtherImg+oOtherMask)/2);
        cv::waitKey(1);
    }
    for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
        if(nPrimaryCamIdx==nCamIdx) {
            updateStereoModel(nCamIdx,true);
            resetStereoLabelings(nCamIdx,true);
        }
        cv::Mat(((m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask]>0)&m_aROIs[nCamIdx])&s_nForegroundLabelIdx).copyTo(m_aResegmLabelings[nCamIdx]);
        updateResegmModel(nCamIdx,true);
        lvDbgAssert(m_oGridSize.dims()==2 && m_oGridSize==m_aStereoLabelings[nCamIdx].size && m_oGridSize==m_aResegmLabelings[nCamIdx].size);
        lvDbgAssert(m_oGridSize.total()==m_vNodeInfos.size());
        lvDbgAssert(m_anValidGraphNodes[nCamIdx]==m_avValidLUTNodeIdxs[nCamIdx].size());
    }
    lvLog_(2,"Running inference for primary camera idx=%d...",(int)nPrimaryCamIdx);
#if (STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
    using HOEReducer = HigherOrderEnergy<ValueType,s_nMaxOrder>;
    const auto lFactorReducer = [&](auto& oGraphFactor, size_t nFactOrder, HOEReducer& oReducer, InternalLabelType nAlphaLabel, const size_t* pValidLUTNodeIdxs, const InternalLabelType* aLabeling) {
        lvDbgAssert(oGraphFactor.numberOfVariables()==nFactOrder);
        std::array<typename HOEReducer::VarId,s_nMaxOrder> aTermEnergyLUT;
        std::array<InternalLabelType,s_nMaxOrder> aCliqueLabels;
        std::array<ValueType,s_nMaxCliqueAssign> aCliqueCoeffs;
        const size_t nAssignCount = 1UL<<nFactOrder;
        std::fill_n(aCliqueCoeffs.begin(),nAssignCount,cost_cast(0));
        for(size_t nAssignIdx=0; nAssignIdx<nAssignCount; ++nAssignIdx) {
            for(size_t nVarIdx=0; nVarIdx<nFactOrder; ++nVarIdx)
                aCliqueLabels[nVarIdx] = (nAssignIdx&(1<<nVarIdx))?nAlphaLabel:aLabeling[pValidLUTNodeIdxs[oGraphFactor.variableIndex(nVarIdx)]];
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
    lvIgnore(lFactorReducer); // @@@@ make inline func? move up like sospd helpers?
#endif //(STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF || STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF)
#if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
    //calcStereoCosts(nPrimaryCamIdx);
    //const size_t nStereoPairwCount = m_anValidGraphNodes[nPrimaryCamIdx]*s_nPairwOrients; (max, not actual) // numPairs_
    // pairs_ : lookup node, 1st pairw node idx + 2nd pairw node idx
    // distance_[2ndlabel*nlabels + 1stlabel] : m_aaStereoPairwFuncIDs_base[nCamIdx][nOrientIdx](1stlabel,2ndlabel)
    // weights_ : lookup node, each orient has its weight
    /*@@@ todo fastpd setup
    setLabelCosts();
    getNumPairs();
    setPairs();
    setDistance();
    setWeights();
    pdInference_ = new fastPDLib::CV_Fast_PD(
            gm_.numberOfVariables(),
            gm_.numberOfLabels(0),
            labelCosts_,
            numPairs_,
            pairs_,
            distance_,
            parameter_.numberOfIterations_,
            weights_
    );*/
    // @@@@ see if maxflow used in fastpd can be replaced by https://github.com/gerddie/maxflow?
#elif STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
    constexpr int nMaxStereoEdgesPerNode = (s_nPairwOrients/*+...@@@*/);
    kolmogorov::qpbo::QPBO<ValueType> oStereoMinimizer((int)m_anValidGraphNodes[nPrimaryCamIdx],(int)m_anValidGraphNodes[nPrimaryCamIdx]*nMaxStereoEdgesPerNode);
    HOEReducer oStereoReducer;
    size_t nOrderingIdx = 0;
#elif STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF
    static_assert(std::is_integral<StereoSegmMatcher::ValueType>::value,"sospd height weight redistr requires integer type"); // @@@ could rewrite for float type
    static_assert(std::is_same<REAL,ValueType>::value,"@@@"); // @@@@ must REALLY get rid of REAL
    //calcStereoCosts(nPrimaryCamIdx);
    constexpr bool bUseHeightAlphaExp = STEREOSEGMATCH_CONFIG_USE_SOSPD_ALPHA_HEIGHTS_LABEL_ORDERING;
    lvAssert_(!bUseHeightAlphaExp,"missing impl"); // @@@@
    size_t nOrderingIdx = 0;
    // setup graph/dual/cliquelist
    SubmodularIBFS<ValueType,VarId> crf; //oStereoMinimizer @@@@
    __cam = nPrimaryCamIdx;
    m_nStereoCliqueCount = size_t(0);
    {
        crf.AddNode((int)m_anValidGraphNodes[nPrimaryCamIdx]);
        m_heights.resize(m_anValidGraphNodes[nPrimaryCamIdx]*m_nStereoLabels);
        std::fill_n(m_heights.begin(),m_anValidGraphNodes[nPrimaryCamIdx]*m_nStereoLabels,cost_cast(0));
        m_dual.clear();
        m_dual.reserve(m_anValidGraphNodes[nPrimaryCamIdx]*(s_nPairwOrients/*+...@@@*/));
        m_node_clique_list.clear();
        m_node_clique_list.resize(m_anValidGraphNodes[nPrimaryCamIdx]);
        const InternalLabelType* pLabeling = ((InternalLabelType*)m_aStereoLabelings[nPrimaryCamIdx].data);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
            const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            lvDbgAssert(oNode.abValidGraphNode[nPrimaryCamIdx]);
            const InternalLabelType& nInitLabel = pLabeling[nLUTNodeIdx];
            const ExplicitFunction& vUnaryStereoLUT = *oNode.apStereoUnaryFuncs[nPrimaryCamIdx];
            for(size_t nStereoLabel=0; nStereoLabel<m_nStereoLabels; ++nStereoLabel)
                m_heights[nGraphNodeIdx*m_nStereoLabels+nStereoLabel] += vUnaryStereoLUT(nStereoLabel);
            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[__cam][nOrientIdx];
                if(oStereoClique) {
                    constexpr size_t nCliqueSize = oStereoClique.s_nCliqueSize;
                    const size_t& nOffsetLUTNodeIdx = oStereoClique.m_anLUTNodeIdxs[1];
                    lvDbgAssert(nOffsetLUTNodeIdx!=SIZE_MAX && m_vNodeInfos[nOffsetLUTNodeIdx].abValidGraphNode[nPrimaryCamIdx]);
                    const size_t& nOffsetGraphNodeIdx = oStereoClique.m_anGraphNodeIdxs[1];
                    lvDbgAssert(nOffsetGraphNodeIdx!=SIZE_MAX);
                    crf.AddClique( // get rid of NodeId, use IndexType instead
                        std::vector<SubmodularIBFS<ValueType,VarId>::NodeId>{(SubmodularIBFS<ValueType,VarId>::NodeId)nGraphNodeIdx,(SubmodularIBFS<ValueType,VarId>::NodeId)nOffsetGraphNodeIdx},
                        std::vector<ValueType>(size_t(1<<nCliqueSize),cost_cast(0))
                    );
                    m_dual.emplace_back(nCliqueSize*m_nStereoLabels,0); // @@@ 0-sized vec init
                    LambdaAlpha& lambda_a = m_dual.back();
                    lvDbgAssert(oStereoClique.m_pGraphFunctionPtr);
                    const ExplicitFunction& vPairwStereoLUT = *oStereoClique.m_pGraphFunctionPtr;
                    lvDbgAssert(vPairwStereoLUT.dimension()==2 && vPairwStereoLUT.size()==m_nStereoLabels*m_nStereoLabels);
                    const InternalLabelType& nOffsetInitLabel = pLabeling[nOffsetLUTNodeIdx];
                    const ValueType tCurrPairwCost = vPairwStereoLUT(nInitLabel,nOffsetInitLabel);
                    lvDbgAssert(tCurrPairwCost>=cost_cast(0));
                    m_heights[nGraphNodeIdx*m_nStereoLabels+nInitLabel] += (dualVariable(lambda_a,IndexType(0),nInitLabel) = tCurrPairwCost - tCurrPairwCost/2);
                    m_heights[nOffsetGraphNodeIdx*m_nStereoLabels+nOffsetInitLabel] += (dualVariable(lambda_a,IndexType(1),nOffsetInitLabel) = tCurrPairwCost/2);
                    m_node_clique_list[nGraphNodeIdx].push_back(std::make_pair(m_nStereoCliqueCount,IndexType(0)));
                    m_node_clique_list[nOffsetGraphNodeIdx].push_back(std::make_pair(m_nStereoCliqueCount,IndexType(1)));
                    ++m_nStereoCliqueCount;
                }
            }
        }
    }
#endif //STEREOSEGMATCH_CONFIG_USE_..._STEREO_INF
#if STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF
    constexpr int nMaxResegmEdgesPerNode = (s_nPairwOrients/*+...@@@*/);
    kolmogorov::qpbo::QPBO<ValueType> oResegmMinimizer_Left((int)m_anValidGraphNodes[0],(int)m_anValidGraphNodes[0]*nMaxResegmEdgesPerNode);
    kolmogorov::qpbo::QPBO<ValueType> oResegmMinimizer_Right((int)m_anValidGraphNodes[1],(int)m_anValidGraphNodes[1]*nMaxResegmEdgesPerNode);
    CamArray<kolmogorov::qpbo::QPBO<ValueType>*> apResegmMinimizers = {&oResegmMinimizer_Left,&oResegmMinimizer_Right};
    CamArray<HOEReducer> aResegmReducers;
#elif STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF
    @@@ todo sospd resegm setup
#endif //STEREOSEGMATCH_CONFIG_USE_..._RESEGM_INF
    size_t nMoveIter=0, nConsecUnchangedLabels=0;
    lvDbgAssert(m_vStereoLabelOrdering.size()==m_vStereoLabels.size());
    lv::StopWatch oLocalTimer;
    ValueType tLastStereoEnergy = m_apStereoInfs[nPrimaryCamIdx]->value();
    ValueType tLastResegmEnergyTotal = std::numeric_limits<ValueType>::max();
    CamArray<ValueType> atLastResegmEnergies = {std::numeric_limits<ValueType>::max(),std::numeric_limits<ValueType>::max()};
    cv::Mat_<InternalLabelType>& oCurrStereoLabeling = m_aStereoLabelings[nPrimaryCamIdx];
    bool bJustUpdatedSegm = false;

    while(++nMoveIter<=m_nMaxMoveIterCount && nConsecUnchangedLabels<m_nStereoLabels) {
        const bool bNullifyStereoPairwCosts = false;//(STEREOSEGMATCH_CONFIG_USE_UNARY_ONLY_FIRST)&&nMoveIter<=m_nStereoLabels;
    #if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF

        // fastpd only works with shared+scaled pairwise costs, and no higher order terms
        opengm::external::FastPD<StereoModelType> oStereoMinimizer2(*m_apStereoModels[nPrimaryCamIdx],opengm::external::FastPD<StereoModelType>::Parameter());
        oStereoMinimizer2.infer();
        std::vector<InternalLabelType> outputlabels;
        oStereoMinimizer2.arg(outputlabels);
        lvAssert(outputlabels.size()==m_anValidGraphNodes[nPrimaryCamIdx]);

        //pdInference_->run(); @@@ just one iter at a time --- split and use while above
        // @@@ nullify stereo pairw costs?

        size_t nChangedStereoLabels = 0;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
            const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const int& nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
            const int& nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
            const InternalLabelType nOldLabel = oCurrStereoLabeling(nRowIdx,nColIdx);
            //const InternalLabelType nNewLabel = pdInference_->_pinfo[nGraphNodeIdx].label;
            const InternalLabelType nNewLabel = outputlabels[nGraphNodeIdx];
            if(nOldLabel!=nNewLabel) {
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nOldLabel);
                oCurrStereoLabeling(nRowIdx,nColIdx) = nNewLabel;
                if(nNewLabel<m_nDontCareLabelIdx)
                    addAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nNewLabel);
                ++nChangedStereoLabels;
            }
        }
        nMoveIter += m_nRealStereoLabels;
        nConsecUnchangedLabels = (nChangedStereoLabels>0)?0:nConsecUnchangedLabels+m_nRealStereoLabels;
        const bool bResegmNext = true;
    #elif STEREOSEGMATCH_CONFIG_USE_FGBZ_STEREO_INF
        // each iter below is a fusion move based on A. Fix's energy minimization method for higher-order MRFs
        // see "A Graph Cut Algorithm for Higher-order Markov Random Fields" in ICCV2011 for more info (doi = 10.1109/ICCV.2011.6126347)
        // (note: this approach is very generic, and not very well adapted to a dynamic MRF problem!)
        const InternalLabelType nStereoAlphaLabel = m_vStereoLabelOrdering[nOrderingIdx];
        calcStereoMoveCosts(nPrimaryCamIdx,nStereoAlphaLabel);
        oStereoReducer.Clear();
        oStereoReducer.AddVars((int)m_anValidGraphNodes[nPrimaryCamIdx]);
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
            if(oNode.anStereoUnaryFactIDs[nPrimaryCamIdx]!=SIZE_MAX) {
                const ValueType& tUnaryCost = ((ValueType*)m_aStereoUnaryCosts[nPrimaryCamIdx].data)[nLUTNodeIdx];
                lvDbgAssert(&tUnaryCost==&m_aStereoUnaryCosts[nPrimaryCamIdx](oNode.nRowIdx,oNode.nColIdx));
                oStereoReducer.AddUnaryTerm((int)nGraphNodeIdx,tUnaryCost);
            }
            if(!bNullifyStereoPairwCosts) {
                for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                    const PairwClique& oStereoClique = oNode.aaStereoPairwCliques[nPrimaryCamIdx][nOrientIdx];
                    if(oStereoClique)
                        lFactorReducer(m_apStereoModels[nPrimaryCamIdx]->operator[](oStereoClique.m_nGraphFactorId),2,oStereoReducer,nStereoAlphaLabel,m_avValidLUTNodeIdxs[nPrimaryCamIdx].data(),(InternalLabelType*)oCurrStereoLabeling.data);
                }
                // @@@@@ add higher o facts here (3-conn on epi lines?)
            }
        }
        oStereoMinimizer.Reset();
        oStereoReducer.ToQuadratic(oStereoMinimizer);
        oStereoMinimizer.Solve();
        oStereoMinimizer.ComputeWeakPersistencies(); // @@@@ check if any good
        size_t nChangedStereoLabels = 0;
        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nPrimaryCamIdx]; ++nGraphNodeIdx) {
            const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[nPrimaryCamIdx][nGraphNodeIdx];
            const int& nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
            const int& nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
            const int nMoveLabel = oStereoMinimizer.GetLabel((int)nGraphNodeIdx);
            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
            if(nMoveLabel==1) { // node label changed to alpha
                const InternalLabelType nOldLabel = oCurrStereoLabeling(nRowIdx,nColIdx);
                if(nOldLabel<m_nDontCareLabelIdx)
                    removeAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nOldLabel);
                oCurrStereoLabeling(nRowIdx,nColIdx) = nStereoAlphaLabel;
                if(nStereoAlphaLabel<m_nDontCareLabelIdx)
                    addAssoc(nPrimaryCamIdx,nRowIdx,nColIdx,nStereoAlphaLabel);
                ++nChangedStereoLabels;
            }
        }
        ++nOrderingIdx %= m_nStereoLabels;
        nConsecUnchangedLabels = (nChangedStereoLabels>0)?0:nConsecUnchangedLabels+1;
        const bool bResegmNext = (nMoveIter%STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM)==0;
    #elif STEREOSEGMATCH_CONFIG_USE_SOSPD_STEREO_INF

        // @@@ nullify stereo pairw costs?
        const InternalLabelType nStereoAlphaLabel = m_vStereoLabelOrdering[nOrderingIdx];
        __alpha = nStereoAlphaLabel;
        calcStereoMoveCosts(nPrimaryCamIdx,nStereoAlphaLabel);
        bool bGotLabelChange = false;
        if(InitialFusionLabeling()) {
            PreEditDual(crf);
            bGotLabelChange = UpdatePrimalDual(crf);
            PostEditDual(crf);
        }
        lvIgnore(oCurrStereoLabeling);
        ++nOrderingIdx %= m_nStereoLabels;
        nConsecUnchangedLabels = bGotLabelChange?0:nConsecUnchangedLabels+1;
        const bool bResegmNext = (nMoveIter%STEREOSEGMATCH_DEFAULT_ITER_PER_RESEGM)==0;
    #endif //STEREOSEGMATCH_CONFIG_USE_..._STEREO_INF
        if(lv::getVerbosity()>=3) {
            cv::Mat oCurrLabelingDisplay = getStereoDispMapDisplay(nPrimaryCamIdx);
            if(oCurrLabelingDisplay.size().area()<640*480)
                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
            cv::imshow(std::string("disp-")+std::to_string(nPrimaryCamIdx),oCurrLabelingDisplay);
            cv::waitKey(1);
        }
        const ValueType tCurrStereoEnergy = m_apStereoInfs[nPrimaryCamIdx]->value();
        lvDbgAssert(tCurrStereoEnergy>=cost_cast(0));
        std::stringstream ssStereoEnergyDiff;
        if((tCurrStereoEnergy-tLastStereoEnergy)==cost_cast(0))
            ssStereoEnergyDiff << "null";
        else
            ssStereoEnergyDiff << std::showpos << tCurrStereoEnergy-tLastStereoEnergy;
    #if STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
        // no control on label w/ fastpd (could decompose algo later on...) @@@
        lvLog_(2,"\t\tdisp      e = %d      (delta=%s)      [iter=%d]",(int)tCurrStereoEnergy,ssStereoEnergyDiff.str().c_str(),(int)nMoveIter);
    #else //!STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
        lvLog_(2,"\t\tdisp [+label:%d]   e = %d   (delta=%s)      [iter=%d]",(int)nStereoAlphaLabel,(int)tCurrStereoEnergy,ssStereoEnergyDiff.str().c_str(),(int)nMoveIter);
    #endif //!STEREOSEGMATCH_CONFIG_USE_FASTPD_STEREO_INF
        if(bNullifyStereoPairwCosts)
            lvLog(2,"\t\t\t(nullifying pairw costs)");
        else if(bJustUpdatedSegm)
            lvLog(2,"\t\t\t(just updated segmentation)");
 //       else
//            lvAssert_(tLastStereoEnergy>=tCurrStereoEnergy,"stereo energy not minimizing!");
        tLastStereoEnergy = tCurrStereoEnergy;
        bJustUpdatedSegm = false;
        if(bResegmNext) {
            resetStereoLabelings(nSecondaryCamIdx,false);
            if(lv::getVerbosity()>=3) {
                cv::Mat oCurrLabelingDisplay = getStereoDispMapDisplay(nSecondaryCamIdx);
                if(oCurrLabelingDisplay.size().area()<640*480)
                    cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                cv::imshow(std::string("disp-")+std::to_string(nSecondaryCamIdx),oCurrLabelingDisplay);
                cv::waitKey(1);
            }
            size_t nTotChangedResegmLabelings = 0;
            for(size_t nResegmLoopIdx=0; nResegmLoopIdx<STEREOSEGMATCH_DEFAULT_RESEGM_PER_LOOP; ++nResegmLoopIdx) {
                for(size_t nCamIdx=0; nCamIdx<getCameraCount(); ++nCamIdx) {
                    cv::Mat_<InternalLabelType>& oCurrResegmLabeling = m_aResegmLabelings[nCamIdx];
                    for(InternalLabelType nResegmAlphaLabel : {s_nForegroundLabelIdx,s_nBackgroundLabelIdx}) {
                        updateResegmModel(nCamIdx,false);
                    #if STEREOSEGMATCH_CONFIG_USE_FGBZ_RESEGM_INF
                        calcResegmMoveCosts(nCamIdx,nResegmAlphaLabel);
                        aResegmReducers[nCamIdx].Clear();
                        aResegmReducers[nCamIdx].AddVars((int)m_anValidGraphNodes[nCamIdx]);
                        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
                            const size_t nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
                            const NodeInfo& oNode = m_vNodeInfos[nLUTNodeIdx];
                            if(oNode.anResegmUnaryFactIDs[nCamIdx]!=SIZE_MAX) {
                                const ValueType& tUnaryCost = ((ValueType*)m_aResegmUnaryCosts[nCamIdx].data)[nLUTNodeIdx];
                                lvDbgAssert(&tUnaryCost==&m_aResegmUnaryCosts[nCamIdx](oNode.nRowIdx,oNode.nColIdx));
                                aResegmReducers[nCamIdx].AddUnaryTerm((int)nGraphNodeIdx,tUnaryCost);
                            }
                            for(size_t nOrientIdx=0; nOrientIdx<s_nPairwOrients; ++nOrientIdx) {
                                const PairwClique& oResegmClique = oNode.aaResegmPairwCliques[nCamIdx][nOrientIdx];
                                if(oResegmClique)
                                    lFactorReducer(m_apResegmModels[nCamIdx]->operator[](oResegmClique.m_nGraphFactorId),2,aResegmReducers[nCamIdx],nResegmAlphaLabel,m_avValidLUTNodeIdxs[nCamIdx].data(),(InternalLabelType*)oCurrResegmLabeling.data);
                            }
                            // @@@@@ add higher o facts here (3/4-conn spatiotemporal?)
                        }
                        apResegmMinimizers[nCamIdx]->Reset();
                        aResegmReducers[nCamIdx].ToQuadratic(*apResegmMinimizers[nCamIdx]);
                        apResegmMinimizers[nCamIdx]->Solve();
                        apResegmMinimizers[nCamIdx]->ComputeWeakPersistencies(); // @@@@ check if any good
                        size_t nChangedResegmLabelings = 0;
                        for(size_t nGraphNodeIdx=0; nGraphNodeIdx<m_anValidGraphNodes[nCamIdx]; ++nGraphNodeIdx) {
                            const size_t& nLUTNodeIdx = m_avValidLUTNodeIdxs[nCamIdx][nGraphNodeIdx];
                            const int& nRowIdx = m_vNodeInfos[nLUTNodeIdx].nRowIdx;
                            const int& nColIdx = m_vNodeInfos[nLUTNodeIdx].nColIdx;
                            const int nMoveLabel = apResegmMinimizers[nCamIdx]->GetLabel((int)nGraphNodeIdx);
                            lvDbgAssert(nMoveLabel==0 || nMoveLabel==1 || nMoveLabel<0);
                            if(nMoveLabel==1) { // node label changed to alpha
                                oCurrResegmLabeling(nRowIdx,nColIdx) = nResegmAlphaLabel;
                                ++nChangedResegmLabelings;
                            }
                        }
                    #elif STEREOSEGMATCH_CONFIG_USE_SOSPD_RESEGM_INF
                        @@@ todo solve 1x iter w/ sospd
                    #endif //STEREOSEGMATCH_CONFIG_USE_..._RESEGM_INF
                        if(lv::getVerbosity()>=3) {
                            cv::Mat oCurrLabelingDisplay = getResegmMapDisplay(nCamIdx);
                            if(oCurrLabelingDisplay.size().area()<640*480)
                                cv::resize(oCurrLabelingDisplay,oCurrLabelingDisplay,cv::Size(),2,2,cv::INTER_NEAREST);
                            cv::imshow(std::string("segm-")+std::to_string(nCamIdx),oCurrLabelingDisplay);
                            cv::waitKey(1);
                        }
                        const ValueType tCurrResegmEnergy = m_apResegmInfs[nCamIdx]->value();
                        lvDbgAssert(tCurrResegmEnergy>=cost_cast(0));
                        std::stringstream ssResegmEnergyDiff;
                        if((tCurrResegmEnergy-atLastResegmEnergies[nCamIdx])==cost_cast(0))
                            ssResegmEnergyDiff << "null";
                        else
                            ssResegmEnergyDiff << std::showpos << tCurrResegmEnergy-atLastResegmEnergies[nCamIdx];
                        lvLog_(2,"\t\tsegm [%d][+%s]   e = %d   (delta=%s)      [iter=%d]",(int)nCamIdx,(nResegmAlphaLabel==s_nForegroundLabelIdx?"fg":"bg"),(int)tCurrResegmEnergy,ssResegmEnergyDiff.str().c_str(),(int)nMoveIter);
                        if(nChangedResegmLabelings) {
                            calcShapeDistFeatures(m_aResegmLabelings[nCamIdx],nCamIdx);
                            nTotChangedResegmLabelings += nChangedResegmLabelings;
                        }
                        atLastResegmEnergies[nCamIdx] = tCurrResegmEnergy;
                    }
                }
            }
            const ValueType tCurrResegmEnergyTotal = std::accumulate(atLastResegmEnergies.begin(),atLastResegmEnergies.end(),cost_cast(0));
            std::stringstream ssResegmEnergyDiff;
            if((tCurrResegmEnergyTotal-tLastResegmEnergyTotal)==cost_cast(0))
                ssResegmEnergyDiff << "null";
            else
                ssResegmEnergyDiff << std::showpos << tCurrResegmEnergyTotal-tLastResegmEnergyTotal;
            lvLog_(2,"\t\tsegm overall   e = %d   (delta=%s)      [iter=%d]",(int)tCurrResegmEnergyTotal,ssResegmEnergyDiff.str().c_str(),(int)nMoveIter);
            if(nTotChangedResegmLabelings) {
                calcShapeFeatures(m_aResegmLabelings);
                updateStereoModel(nPrimaryCamIdx,false);
                bJustUpdatedSegm = true;
                nConsecUnchangedLabels = 0;
            }
            tLastResegmEnergyTotal = tCurrResegmEnergyTotal;
        }
    }
    resetStereoLabelings(nSecondaryCamIdx,false);
    lvLog_(2,"Inference for primary camera idx=%d completed in %f second(s).",(int)nPrimaryCamIdx,oLocalTimer.tock());
    if(lv::getVerbosity()>=4)
        cv::waitKey(0);
    return opengm::InferenceTermination::NORMAL;
}

inline cv::Mat StereoSegmMatcher::GraphModelData::getResegmMapDisplay(size_t nCamIdx) const {
    lvDbgExceptionWatch;
    lvAssert_(nCamIdx<getCameraCount(),"camera index out of range");
    lvAssert(!m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img].empty());
    lvAssert(!m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask].empty());
    lvAssert(m_oGridSize==m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Img].size);
    lvAssert(m_oGridSize==m_aInputs[nCamIdx*InputPackOffset+InputPackOffset_Mask].size);
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
    lvAssert(m_nMaxDispOffset>m_nMinDispOffset);
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
    lvAssert(m_nMaxDispOffset>m_nMinDispOffset);
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
    lvIgnore(tTotAssocCost);
    return /*tTotAssocCost+*/tTotStereoLabelCost;
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