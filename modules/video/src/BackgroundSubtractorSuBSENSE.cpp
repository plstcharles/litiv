
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

#include "litiv/video/BackgroundSubtractorSuBSENSE.hpp"

//
// NOTE: this version of SuBSENSE is still pretty messy (debug). The cleaner (but older) implementation made available on
// bitbucket is a good alternative in case of serious eyebleed; see https://bitbucket.org/pierre_luc_st_charles/subsense
//

// defines the threshold value(s) used to detect long-term ghosting and trigger the fast edge-based absorption heuristic
#define GHOSTDET_D_MAX (0.010f) // defines 'negligible' change here
#define GHOSTDET_S_MIN (0.995f) // defines the required minimum local foreground saturation value
// parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR (0.01f)
// parameters used to adjust the variation step size of 'v(x)'
#define FEEDBACK_V_INCR  (1.000f)
#define FEEDBACK_V_DECR  (0.100f)
// parameters used to scale dynamic learning rate adjustments  ('T(x)')
#define FEEDBACK_T_DECR  (0.2500f)
#define FEEDBACK_T_INCR  (0.5000f)
#define FEEDBACK_T_LOWER (2.0000f)
#define FEEDBACK_T_UPPER (256.00f)
// parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN (0.100f)
#define UNSTABLE_REG_RDIST_MIN (3.000f)
// parameters used to scale the relative LBSP intensity threshold used for internal comparisons
#define LBSPDESC_NONZERO_RATIO_MIN (0.100f)
#define LBSPDESC_NONZERO_RATIO_MAX (0.500f)
// parameters used to define model reset/learning rate boosts in our frame-level component
#define FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD  (m_nMinColorDistThreshold/2)
#define FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO (8)

// local define used to display debug information
#define DISPLAY_SUBSENSE_DEBUG_INFO 0
// local define used to specify the default frame size (320x240 = QVGA)
#define DEFAULT_FRAME_SIZE cv::Size(320,240)
// local define used to specify the color dist threshold offset used for unstable regions
#define STAB_COLOR_DIST_OFFSET (m_nMinColorDistThreshold/5)
// local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET (m_nDescDistThresholdOffset)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE_BITS;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;

BackgroundSubtractorSuBSENSE::BackgroundSubtractorSuBSENSE_(size_t nDescDistThresholdOffset, size_t nMinColorDistThreshold, size_t nBGSamples,
                                                            size_t nRequiredBGSamples, size_t nSamplesForMovingAvgs, float fRelLBSPThreshold) :
        IBackgroundSubtractorLBSP(fRelLBSPThreshold),
        m_nMinColorDistThreshold(nMinColorDistThreshold),
        m_nDescDistThresholdOffset(nDescDistThresholdOffset),
        m_nBGSamples(nBGSamples),
        m_nRequiredBGSamples(nRequiredBGSamples),
        m_nSamplesForMovingAvgs(nSamplesForMovingAvgs),
        m_fLastNonZeroDescRatio(0.0f),
        m_bLearningRateScalingEnabled(true),
        m_fCurrLearningRateLowerCap(FEEDBACK_T_LOWER),
        m_fCurrLearningRateUpperCap(FEEDBACK_T_UPPER),
        m_nMedianBlurKernelSize(m_nDefaultMedianBlurKernelSize),
        m_bUse3x3Spread(true) {
    lvAssert_(m_nBGSamples>0 && m_nRequiredBGSamples<=m_nBGSamples,"algo cannot require more sample matches than sample count in model");
    lvAssert_(m_nMinColorDistThreshold>0 || m_nDescDistThresholdOffset>0,"distance thresholds must be positive values");
}

void BackgroundSubtractorSuBSENSE::refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate) {
    // == refresh
    lvAssert_(m_bInitialized,"algo must be initialized first");
    lvAssert_(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f,"model refresh must be given as a non-null fraction");
    lvDbgAssert(!m_voBGColorSamples.empty() && !m_voBGColorSamples[0].empty());
    const size_t nModelSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
    const size_t nRefreshSampleStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
    const size_t nChannels = m_voBGColorSamples[0].channels();
    for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
        const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
        if(bForceFGUpdate || !m_oLastFGMask.data[nPxIter]) {
            for(size_t nCurrModelSampleIdx=nRefreshSampleStartPos; nCurrModelSampleIdx<nRefreshSampleStartPos+nModelSamplesToRefresh; ++nCurrModelSampleIdx) {
                int nSampleImgCoord_Y, nSampleImgCoord_X;
                cv::getRandSamplePosition_7x7_std2(nSampleImgCoord_X,nSampleImgCoord_Y,m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
                if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
                    const size_t nCurrRealModelSampleIdx = nCurrModelSampleIdx%m_nBGSamples;
                    for(size_t c=0; c<nChannels; ++c) {
                        m_voBGColorSamples[nCurrRealModelSampleIdx].data[nPxIter*nChannels+c] = m_oLastColorFrame.data[nSamplePxIdx*nChannels+c];
                        *((ushort*)(m_voBGDescSamples[nCurrRealModelSampleIdx].data+(nPxIter*nChannels+c)*2)) = *((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*nChannels+c)*2));
                    }
                }
            }
        }
    }
}

void BackgroundSubtractorSuBSENSE::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    // == init
    IBackgroundSubtractorLBSP::initialize_common(oInitImg,oROI);
    m_fLastNonZeroDescRatio = 0.0f;
    const int nTotImgPixels = m_oImgSize.height*m_oImgSize.width;
    if(m_nOrigROIPxCount>=m_nTotPxCount/2 && (int)m_nTotPxCount>=DEFAULT_FRAME_SIZE.area()) {
        m_bLearningRateScalingEnabled = true;
        m_bAutoModelResetEnabled = true;
        m_bUse3x3Spread = !(nTotImgPixels>DEFAULT_FRAME_SIZE.area()*2);
        const int nRawMedianBlurKernelSize = std::min((int)floor((float)nTotImgPixels/DEFAULT_FRAME_SIZE.area()+0.5f)+m_nDefaultMedianBlurKernelSize,14);
        m_nMedianBlurKernelSize = (nRawMedianBlurKernelSize%2)?nRawMedianBlurKernelSize:nRawMedianBlurKernelSize-1;
        m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
        m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
    }
    else {
        m_bLearningRateScalingEnabled = false;
        m_bAutoModelResetEnabled = false;
        m_bUse3x3Spread = true;
        m_nMedianBlurKernelSize = m_nDefaultMedianBlurKernelSize;
        m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER*2;
        m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER*2;
    }
    m_oUpdateRateFrame.create(m_oImgSize,CV_32FC1);
    m_oUpdateRateFrame = cv::Scalar(m_fCurrLearningRateLowerCap);
    m_oDistThresholdFrame.create(m_oImgSize,CV_32FC1);
    m_oDistThresholdFrame = cv::Scalar(1.0f);
    m_oVariationModulatorFrame.create(m_oImgSize,CV_32FC1);
    m_oVariationModulatorFrame = cv::Scalar(10.0f); // should always be >= FEEDBACK_V_DECR
    m_oMeanLastDistFrame.create(m_oImgSize,CV_32FC1);
    m_oMeanLastDistFrame = cv::Scalar(0.0f);
    m_oMeanMinDistFrame_LT.create(m_oImgSize,CV_32FC1);
    m_oMeanMinDistFrame_LT = cv::Scalar(0.0f);
    m_oMeanMinDistFrame_ST.create(m_oImgSize,CV_32FC1);
    m_oMeanMinDistFrame_ST = cv::Scalar(0.0f);
    m_oDownSampledFrameSize = cv::Size(m_oImgSize.width/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO,m_oImgSize.height/FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO);
    m_oMeanDownSampledLastDistFrame_LT.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
    m_oMeanDownSampledLastDistFrame_LT = cv::Scalar(0.0f);
    m_oMeanDownSampledLastDistFrame_ST.create(m_oDownSampledFrameSize,CV_32FC((int)m_nImgChannels));
    m_oMeanDownSampledLastDistFrame_ST = cv::Scalar(0.0f);
    m_oMeanRawSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
    m_oMeanRawSegmResFrame_LT = cv::Scalar(0.0f);
    m_oMeanRawSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
    m_oMeanRawSegmResFrame_ST = cv::Scalar(0.0f);
    m_oMeanFinalSegmResFrame_LT.create(m_oImgSize,CV_32FC1);
    m_oMeanFinalSegmResFrame_LT = cv::Scalar(0.0f);
    m_oMeanFinalSegmResFrame_ST.create(m_oImgSize,CV_32FC1);
    m_oMeanFinalSegmResFrame_ST = cv::Scalar(0.0f);
    m_oUnstableRegionMask.create(m_oImgSize,CV_8UC1);
    m_oUnstableRegionMask = cv::Scalar_<uchar>(0);
    m_oBlinksFrame.create(m_oImgSize,CV_8UC1);
    m_oBlinksFrame = cv::Scalar_<uchar>(0);
    m_oDownSampledFrame_MotionAnalysis.create(m_oDownSampledFrameSize,CV_8UC((int)m_nImgChannels));
    m_oDownSampledFrame_MotionAnalysis = cv::Scalar_<uchar>::all(0);
    m_oLastRawFGMask.create(m_oImgSize,CV_8UC1);
    m_oLastRawFGMask = cv::Scalar_<uchar>(0);
    m_oLastFGMask_dilated.create(m_oImgSize,CV_8UC1);
    m_oLastFGMask_dilated = cv::Scalar_<uchar>(0);
    m_oLastFGMask_dilated_inverted.create(m_oImgSize,CV_8UC1);
    m_oLastFGMask_dilated_inverted = cv::Scalar_<uchar>(0);
    m_oFGMask_FloodedHoles.create(m_oImgSize,CV_8UC1);
    m_oFGMask_FloodedHoles = cv::Scalar_<uchar>(0);
    m_oFGMask_PreFlood.create(m_oImgSize,CV_8UC1);
    m_oFGMask_PreFlood = cv::Scalar_<uchar>(0);
    m_oCurrRawFGBlinkMask.create(m_oImgSize,CV_8UC1);
    m_oCurrRawFGBlinkMask = cv::Scalar_<uchar>(0);
    m_oLastRawFGBlinkMask.create(m_oImgSize,CV_8UC1);
    m_oLastRawFGBlinkMask = cv::Scalar_<uchar>(0);
    m_oMorphExStructElement = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
    m_voBGColorSamples.resize(m_nBGSamples);
    m_voBGDescSamples.resize(m_nBGSamples);
    for(size_t s=0; s<m_nBGSamples; ++s) {
        m_voBGColorSamples[s].create(m_oImgSize,CV_8UC((int)m_nImgChannels));
        m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);
        m_voBGDescSamples[s].create(m_oImgSize,CV_16UC((int)m_nImgChannels));
        m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
    }
    m_bInitialized = true;
    refreshModel(1.0f);
    m_bModelInitialized = true;
}

void BackgroundSubtractorSuBSENSE::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
    // == process
    lvAssert_(m_bInitialized && m_bModelInitialized,"algo & model must be initialized first");
    cv::Mat oInputImg = _image.getMat();
    lvAssert_(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize,"input image type/size mismatch with initialization type/size");
    lvAssert_(oInputImg.isContinuous(),"input image data must be continuous");
    _fgmask.create(m_oImgSize,CV_8UC1);
    cv::Mat oCurrFGMask = _fgmask.getMat();
    memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
    size_t nNonZeroDescCount = 0;
    const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIdx,m_nSamplesForMovingAvgs);
    const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIdx,m_nSamplesForMovingAvgs/4);
    if(m_nImgChannels==1) {
        for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
            const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
            const size_t nDescIter = nPxIter*2;
            const size_t nFloatIter = nPxIter*4;
            const int nCurrImgCoord_X = m_voPxInfoLUT[nPxIter].nImgCoord_X;
            const int nCurrImgCoord_Y = m_voPxInfoLUT[nPxIter].nImgCoord_Y;
            const uchar nCurrColor = oInputImg.data[nPxIter];
            size_t nMinDescDist = s_nDescMaxDataRange_1ch;
            size_t nMinSumDist = s_nColorMaxDataRange_1ch;
            float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+nFloatIter);
            float* pfCurrVariationFactor = (float*)(m_oVariationModulatorFrame.data+nFloatIter);
            float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+nFloatIter));
            float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+nFloatIter));
            float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+nFloatIter));
            float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+nFloatIter));
            ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+nDescIter));
            uchar& nLastColor = m_oLastColorFrame.data[nPxIter];
            const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[nPxIter])*STAB_COLOR_DIST_OFFSET))/2;
            const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThresholdOffset+(m_oUnstableRegionMask.data[nPxIter]*UNSTAB_DESC_DIST_OFFSET);
            alignas(16) std::array<uchar,LBSP::DESC_SIZE_BITS> anLBSPLookupVals;
            LBSP::computeDescriptor_lookup<1>(oInputImg,nCurrImgCoord_X,nCurrImgCoord_Y,0,anLBSPLookupVals);
            const ushort nCurrIntraDesc = LBSP::computeDescriptor_threshold(anLBSPLookupVals,nCurrColor,m_anLBSPThreshold_8bitLUT[nCurrColor]);
            m_oUnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
            size_t nGoodSamplesCount=0, nSampleIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
                const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[nPxIter];
                {
                    const size_t nColorDist = lv::L1dist(nCurrColor,nBGColor);
                    if(nColorDist>nCurrColorDistThreshold)
                        goto failedcheck1ch;
                    const ushort& nBGIntraDesc = *((ushort*)(m_voBGDescSamples[nSampleIdx].data+nDescIter));
                    const size_t nIntraDescDist = lv::hdist(nCurrIntraDesc,nBGIntraDesc);
                    const ushort nCurrInterDesc = LBSP::computeDescriptor_threshold(anLBSPLookupVals,nBGColor,m_anLBSPThreshold_8bitLUT[nBGColor]);
                    const size_t nInterDescDist = lv::hdist(nCurrInterDesc,nBGIntraDesc);
                    const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
                    if(nDescDist>nCurrDescDistThreshold)
                        goto failedcheck1ch;
                    const size_t nSumDist = std::min((nDescDist/4)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
                    if(nSumDist>nCurrColorDistThreshold)
                        goto failedcheck1ch;
                    if(nMinDescDist>nDescDist)
                        nMinDescDist = nDescDist;
                    if(nMinSumDist>nSumDist)
                        nMinSumDist = nSumDist;
                    nGoodSamplesCount++;
                }
                failedcheck1ch:
                nSampleIdx++;
            }
            const float fNormalizedLastDist = ((float)lv::L1dist(nLastColor,nCurrColor)/s_nColorMaxDataRange_1ch+(float)lv::hdist(nLastIntraDesc,nCurrIntraDesc)/s_nDescMaxDataRange_1ch)/2;
            *pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
            if(nGoodSamplesCount<m_nRequiredBGSamples) {
                // == foreground
                const float fNormalizedMinDist = std::min(1.0f,((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
                oCurrFGMask.data[nPxIter] = UCHAR_MAX;
                if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIter)) = nCurrIntraDesc;
                    m_voBGColorSamples[s_rand].data[nPxIter] = nCurrColor;
                }
            }
            else {
                // == background
                const float fNormalizedMinDist = ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2;
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
                const size_t nLearningRate = std::isinf(learningRateOverride)?SIZE_MAX:(learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate));
                if((rand()%nLearningRate)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIter)) = nCurrIntraDesc;
                    m_voBGColorSamples[s_rand].data[nPxIter] = nCurrColor;
                }
                int nSampleImgCoord_Y, nSampleImgCoord_X;
                const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[nPxIter];
                if(bCurrUsing3x3Spread)
                    cv::getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                else
                    cv::getRandNeighborPosition_5x5(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                const size_t n_rand = rand();
                const size_t idx_rand_uchar = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
                const size_t idx_rand_flt32 = idx_rand_uchar*4;
                const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
                const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
                if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
                    || (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
                    const size_t idx_rand_ushrt = idx_rand_uchar*2;
                    const size_t s_rand = rand()%m_nBGSamples;
                    *((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt)) = nCurrIntraDesc;
                    m_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
                }
            }
            if(m_oLastFGMask.data[nPxIter] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[nPxIter])) {
                if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
                    *pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
            }
            else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
            if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate = m_fCurrLearningRateLowerCap;
            else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
                *pfCurrLearningRate = m_fCurrLearningRateUpperCap;
            if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[nPxIter])
                (*pfCurrVariationFactor) += FEEDBACK_V_INCR;
            else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
                (*pfCurrVariationFactor) -= m_oLastFGMask.data[nPxIter]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[nPxIter]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
                if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
                    (*pfCurrVariationFactor) = FEEDBACK_V_DECR;
            }
            if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
                (*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
            else {
                (*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
                if((*pfCurrDistThresholdFactor)<1.0f)
                    (*pfCurrDistThresholdFactor) = 1.0f;
            }
            if(lv::popcount(nCurrIntraDesc)>=2)
                ++nNonZeroDescCount;
            nLastIntraDesc = nCurrIntraDesc;
            nLastColor = nCurrColor;
        }
    }
    else { //m_nImgChannels==3
        for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
            const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
            const int nCurrImgCoord_X = m_voPxInfoLUT[nPxIter].nImgCoord_X;
            const int nCurrImgCoord_Y = m_voPxInfoLUT[nPxIter].nImgCoord_Y;
            const size_t nPxIterRGB = nPxIter*3;
            const size_t nDescIterRGB = nPxIterRGB*2;
            const size_t nFloatIter = nPxIter*4;
            const uchar* const anCurrColor = oInputImg.data+nPxIterRGB;
            size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
            size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
            float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+nFloatIter);
            float* pfCurrVariationFactor = (float*)(m_oVariationModulatorFrame.data+nFloatIter);
            float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+nFloatIter));
            float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+nFloatIter));
            float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+nFloatIter));
            float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+nFloatIter));
            ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+nDescIterRGB));
            uchar* anLastColor = m_oLastColorFrame.data+nPxIterRGB;
            const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[nPxIter])*STAB_COLOR_DIST_OFFSET));
            const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThresholdOffset+(m_oUnstableRegionMask.data[nPxIter]*UNSTAB_DESC_DIST_OFFSET);
            const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
            const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
            const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
            alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,3> aanLBSPLookupVals;
            LBSP::computeDescriptor_lookup(oInputImg,nCurrImgCoord_X,nCurrImgCoord_Y,aanLBSPLookupVals);
            std::array<ushort,3> anCurrIntraDesc;
            for(size_t c=0; c<3; ++c)
                anCurrIntraDesc[c] = LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],anCurrColor[c],m_anLBSPThreshold_8bitLUT[anCurrColor[c]]);
            m_oUnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
            size_t nGoodSamplesCount=0, nSampleIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
                const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+nDescIterRGB);
                const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+nPxIterRGB;
                size_t nTotDescDist = 0;
                size_t nTotSumDist = 0;
                for(size_t c=0;c<3; ++c) {
                    const size_t nColorDist = lv::L1dist(anCurrColor[c],anBGColor[c]);
                    if(nColorDist>nCurrSCColorDistThreshold)
                        goto failedcheck3ch;
                    const size_t nIntraDescDist = lv::hdist(anCurrIntraDesc[c],anBGIntraDesc[c]);
                    const ushort nCurrInterDesc = LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],anBGColor[c],m_anLBSPThreshold_8bitLUT[anBGColor[c]]);
                    const size_t nInterDescDist = lv::hdist(nCurrInterDesc,anBGIntraDesc[c]);
                    const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
                    const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
                    if(nSumDist>nCurrSCColorDistThreshold)
                        goto failedcheck3ch;
                    nTotDescDist += nDescDist;
                    nTotSumDist += nSumDist;
                }
                if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
                    goto failedcheck3ch;
                if(nMinTotDescDist>nTotDescDist)
                    nMinTotDescDist = nTotDescDist;
                if(nMinTotSumDist>nTotSumDist)
                    nMinTotSumDist = nTotSumDist;
                nGoodSamplesCount++;
                failedcheck3ch:
                nSampleIdx++;
            }
            const float fNormalizedLastDist = ((float)lv::L1dist<3>(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)lv::hdist<3>(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
            *pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
            if(nGoodSamplesCount<m_nRequiredBGSamples) {
                // == foreground
                const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
                oCurrFGMask.data[nPxIter] = UCHAR_MAX;
                if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    for(size_t c=0; c<3; ++c) {
                        *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIterRGB+2*c)) = anCurrIntraDesc[c];
                        *(m_voBGColorSamples[s_rand].data+nPxIterRGB+c) = anCurrColor[c];
                    }
                }
            }
            else {
                // == background
                const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
                const size_t nLearningRate = std::isinf(learningRateOverride)?SIZE_MAX:(learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate));
                if((rand()%nLearningRate)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    for(size_t c=0; c<3; ++c) {
                        *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIterRGB+2*c)) = anCurrIntraDesc[c];
                        *(m_voBGColorSamples[s_rand].data+nPxIterRGB+c) = anCurrColor[c];
                    }
                }
                int nSampleImgCoord_Y, nSampleImgCoord_X;
                const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[nPxIter];
                if(bCurrUsing3x3Spread)
                    cv::getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                else
                    cv::getRandNeighborPosition_5x5(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                const size_t n_rand = rand();
                const size_t idx_rand_uchar = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
                const size_t idx_rand_flt32 = idx_rand_uchar*4;
                const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
                const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
                if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
                    || (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
                    const size_t idx_rand_uchar_rgb = idx_rand_uchar*3;
                    const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb*2;
                    const size_t s_rand = rand()%m_nBGSamples;
                    for(size_t c=0; c<3; ++c) {
                        *((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
                        *(m_voBGColorSamples[s_rand].data+idx_rand_uchar_rgb+c) = anCurrColor[c];
                    }
                }
            }
            if(m_oLastFGMask.data[nPxIter] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[nPxIter])) {
                if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
                    *pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
            }
            else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
            if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate = m_fCurrLearningRateLowerCap;
            else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
                *pfCurrLearningRate = m_fCurrLearningRateUpperCap;
            if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[nPxIter])
                (*pfCurrVariationFactor) += FEEDBACK_V_INCR;
            else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
                (*pfCurrVariationFactor) -= m_oLastFGMask.data[nPxIter]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[nPxIter]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
                if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
                    (*pfCurrVariationFactor) = FEEDBACK_V_DECR;
            }
            if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
                (*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
            else {
                (*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
                if((*pfCurrDistThresholdFactor)<1.0f)
                    (*pfCurrDistThresholdFactor) = 1.0f;
            }
            if(lv::popcount<3>(anCurrIntraDesc)>=4)
                ++nNonZeroDescCount;
            for(size_t c=0; c<3; ++c) {
                anLastIntraDesc[c] = anCurrIntraDesc[c];
                anLastColor[c] = anCurrColor[c];
            }
        }
    }
#if DISPLAY_SUBSENSE_DEBUG_INFO
    cv::Point2i oDbgPt(-1,-1);
    if(m_pDisplayHelper) {
        std::mutex_lock_guard oLock(m_pDisplayHelper->m_oEventMutex);
        const cv::Point2f& oDbgPt_rel = cv::Point2f(float(m_pDisplayHelper->m_oLatestMouseEvent.oPosition.x)/m_pDisplayHelper->m_oLatestMouseEvent.oDisplaySize.width,float(m_pDisplayHelper->m_oLatestMouseEvent.oPosition.y)/m_pDisplayHelper->m_oLatestMouseEvent.oDisplaySize.height);
        oDbgPt = cv::Point2i(int(oDbgPt_rel.x*m_oImgSize.width),int(oDbgPt_rel.y*m_oImgSize.height));
    }
    if(oDbgPt.x>=0 && oDbgPt.x<m_oImgSize.width && oDbgPt.y>=0 && oDbgPt.y<m_oImgSize.height) {
        std::cout << std::endl;
        cv::Mat oMeanMinDistFrameNormalized;
        m_oMeanMinDistFrame_ST.copyTo(oMeanMinDistFrameNormalized);
        cv::circle(oMeanMinDistFrameNormalized,oDbgPt,5,cv::Scalar(1.0f));
        cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,DEFAULT_FRAME_SIZE);
        cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
        std::cout << std::fixed << std::setprecision(5) << "  d_min(" << oDbgPt << ") = " << m_oMeanMinDistFrame_ST.at<float>(oDbgPt) << std::endl;
        cv::Mat oMeanLastDistFrameNormalized;
        m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
        cv::circle(oMeanLastDistFrameNormalized,oDbgPt,5,cv::Scalar(1.0f));
        cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,DEFAULT_FRAME_SIZE);
        cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
        std::cout << std::fixed << std::setprecision(5) << " d_last(" << oDbgPt << ") = " << m_oMeanLastDistFrame.at<float>(oDbgPt) << std::endl;
        cv::Mat oMeanRawSegmResFrameNormalized;
        m_oMeanRawSegmResFrame_ST.copyTo(oMeanRawSegmResFrameNormalized);
        cv::circle(oMeanRawSegmResFrameNormalized,oDbgPt,5,cv::Scalar(1.0f));
        cv::resize(oMeanRawSegmResFrameNormalized,oMeanRawSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
        cv::imshow("s_avg(x)",oMeanRawSegmResFrameNormalized);
        std::cout << std::fixed << std::setprecision(5) << "  s_avg(" << oDbgPt << ") = " << m_oMeanRawSegmResFrame_ST.at<float>(oDbgPt) << std::endl;
        cv::Mat oMeanFinalSegmResFrameNormalized;
        m_oMeanFinalSegmResFrame_ST.copyTo(oMeanFinalSegmResFrameNormalized);
        cv::circle(oMeanFinalSegmResFrameNormalized,oDbgPt,5,cv::Scalar(1.0f));
        cv::resize(oMeanFinalSegmResFrameNormalized,oMeanFinalSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
        cv::imshow("z_avg(x)",oMeanFinalSegmResFrameNormalized);
        std::cout << std::fixed << std::setprecision(5) << "  z_avg(" << oDbgPt << ") = " << m_oMeanFinalSegmResFrame_ST.at<float>(oDbgPt) << std::endl;
        cv::Mat oDistThresholdFrameNormalized;
        m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,0.25f,-0.25f);
        cv::circle(oDistThresholdFrameNormalized,oDbgPt,5,cv::Scalar(1.0f));
        cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEFAULT_FRAME_SIZE);
        cv::imshow("r(x)",oDistThresholdFrameNormalized);
        std::cout << std::fixed << std::setprecision(5) << "      r(" << oDbgPt << ") = " << m_oDistThresholdFrame.at<float>(oDbgPt) << std::endl;
        cv::Mat oVariationModulatorFrameNormalized;
        cv::normalize(m_oVariationModulatorFrame,oVariationModulatorFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
        cv::circle(oVariationModulatorFrameNormalized,oDbgPt,5,cv::Scalar(255));
        cv::resize(oVariationModulatorFrameNormalized,oVariationModulatorFrameNormalized,DEFAULT_FRAME_SIZE);
        cv::imshow("v(x)",oVariationModulatorFrameNormalized);
        std::cout << std::fixed << std::setprecision(5) << "      v(" << oDbgPt << ") = " << m_oVariationModulatorFrame.at<float>(oDbgPt) << std::endl;
        cv::Mat oUpdateRateFrameNormalized;
        m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/FEEDBACK_T_UPPER,-FEEDBACK_T_LOWER/FEEDBACK_T_UPPER);
        cv::circle(oUpdateRateFrameNormalized,oDbgPt,5,cv::Scalar(1.0f));
        cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,DEFAULT_FRAME_SIZE);
        cv::imshow("t(x)",oUpdateRateFrameNormalized);
        std::cout << std::fixed << std::setprecision(5) << "      t(" << oDbgPt << ") = " << m_oUpdateRateFrame.at<float>(oDbgPt) << std::endl;
    }
#endif //DISPLAY_SUBSENSE_DEBUG_INFO
    cv::bitwise_xor(oCurrFGMask,m_oLastRawFGMask,m_oCurrRawFGBlinkMask);
    cv::bitwise_or(m_oCurrRawFGBlinkMask,m_oLastRawFGBlinkMask,m_oBlinksFrame);
    m_oCurrRawFGBlinkMask.copyTo(m_oLastRawFGBlinkMask);
    oCurrFGMask.copyTo(m_oLastRawFGMask);
    cv::morphologyEx(oCurrFGMask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,m_oMorphExStructElement);
    m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
    cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
    cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
    cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
    cv::bitwise_or(oCurrFGMask,m_oFGMask_FloodedHoles,oCurrFGMask);
    cv::bitwise_or(oCurrFGMask,m_oFGMask_PreFlood,oCurrFGMask);
    cv::medianBlur(oCurrFGMask,m_oLastFGMask,m_nMedianBlurKernelSize);
    cv::dilate(m_oLastFGMask,m_oLastFGMask_dilated,cv::Mat(),cv::Point(-1,-1),3);
    cv::bitwise_and(m_oBlinksFrame,m_oLastFGMask_dilated_inverted,m_oBlinksFrame);
    cv::bitwise_not(m_oLastFGMask_dilated,m_oLastFGMask_dilated_inverted);
    cv::bitwise_and(m_oBlinksFrame,m_oLastFGMask_dilated_inverted,m_oBlinksFrame);
    m_oLastFGMask.copyTo(oCurrFGMask);
    cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oLastFGMask,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
    cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oLastFGMask,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);
    const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nTotRelevantPxCount;
    if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
                --m_anLBSPThreshold_8bitLUT[t];
    }
    else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
                ++m_anLBSPThreshold_8bitLUT[t];
    }
    m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
    if(m_bLearningRateScalingEnabled) {
        cv::resize(oInputImg,m_oDownSampledFrame_MotionAnalysis,m_oDownSampledFrameSize,0,0,cv::INTER_AREA);
        cv::accumulateWeighted(m_oDownSampledFrame_MotionAnalysis,m_oMeanDownSampledLastDistFrame_LT,fRollAvgFactor_LT);
        cv::accumulateWeighted(m_oDownSampledFrame_MotionAnalysis,m_oMeanDownSampledLastDistFrame_ST,fRollAvgFactor_ST);
        size_t nTotColorDiff = 0;
        for(int i=0; i<m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
            const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0]*i;
            for(int j=0; j<m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
                const size_t idx2 = idx1+m_oMeanDownSampledLastDistFrame_ST.step.p[1]*j;
                nTotColorDiff += (m_nImgChannels==1)?
                    (size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2)))/2
                            :  //(m_nImgChannels==3)
                        std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2))),
                            std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+4))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+4))),
                                        (size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+8))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+8)))));
            }
        }
        const float fCurrColorDiffRatio = (float)nTotColorDiff/(m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
        if(m_bAutoModelResetEnabled) {
            if(m_nFramesSinceLastReset>1000)
                m_bAutoModelResetEnabled = false;
            else if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD && m_nModelResetCooldown==0) {
                m_nFramesSinceLastReset = 0;
                refreshModel(0.1f); // reset 10% of the bg model
                m_nModelResetCooldown = m_nSamplesForMovingAvgs/4;
                m_oUpdateRateFrame = cv::Scalar(1.0f);
            }
            else
                ++m_nFramesSinceLastReset;
        }
        else if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD*2) {
            m_nFramesSinceLastReset = 0;
            m_bAutoModelResetEnabled = true;
        }
        if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD/2) {
            m_fCurrLearningRateLowerCap = (float)std::max((int)FEEDBACK_T_LOWER>>(int)(fCurrColorDiffRatio/2),1);
            m_fCurrLearningRateUpperCap = (float)std::max((int)FEEDBACK_T_UPPER>>(int)(fCurrColorDiffRatio/2),1);
        }
        else {
            m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
            m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
        }
        if(m_nModelResetCooldown>0)
            --m_nModelResetCooldown;
    }
}

void BackgroundSubtractorSuBSENSE::getBackgroundImage(cv::OutputArray backgroundImage) const {
    lvAssert_(m_bInitialized,"algo must be initialized first");
    cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
    for(size_t s=0; s<m_nBGSamples; ++s) {
        for(int y=0; y<m_oImgSize.height; ++y) {
            for(int x=0; x<m_oImgSize.width; ++x) {
                const size_t idx_nimg = m_voBGColorSamples[s].step.p[0]*y + m_voBGColorSamples[s].step.p[1]*x;
                const size_t nFloatIter = idx_nimg*4;
                float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+nFloatIter);
                const uchar* const oBGImgPtr = m_voBGColorSamples[s].data+idx_nimg;
                for(size_t c=0; c<m_nImgChannels; ++c)
                    oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
            }
        }
    }
    oAvgBGImg.convertTo(backgroundImage,CV_8U);
}

void BackgroundSubtractorSuBSENSE::getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const {
    static_assert(LBSP::DESC_SIZE==2,"bad assumptions in impl below");
    lvAssert_(m_bInitialized,"algo must be initialized first");
    cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
    for(size_t n=0; n<m_voBGDescSamples.size(); ++n) {
        for(int y=0; y<m_oImgSize.height; ++y) {
            for(int x=0; x<m_oImgSize.width; ++x) {
                const size_t idx_ndesc = m_voBGDescSamples[n].step.p[0]*y + m_voBGDescSamples[n].step.p[1]*x;
                const size_t nFloatIter = idx_ndesc*2;
                float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+nFloatIter);
                const ushort* const oBGDescPtr = (ushort*)(m_voBGDescSamples[n].data+idx_ndesc);
                for(size_t c=0; c<m_nImgChannels; ++c)
                    oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDescSamples.size();
            }
        }
    }
    oAvgBGDesc.convertTo(backgroundDescImage,CV_16U);
}
