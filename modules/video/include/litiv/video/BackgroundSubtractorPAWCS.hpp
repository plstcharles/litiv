
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

#include "litiv/video/BackgroundSubtractorLBSP.hpp"

/// defines the default value for BackgroundSubtractorPAWCS::m_nDescDistThresholdOffset
#define BGSPAWCS_DEFAULT_DESC_DIST_THRESHOLD_OFFSET (2)
/// defines the default value for BackgroundSubtractorPAWCS::m_nMinColorDistThreshold
#define BGSPAWCS_DEFAULT_MIN_COLOR_DIST_THRESHOLD (20)
/// defines the default value for BackgroundSubtractorPAWCS::m_nMaxLocalWords and m_nMaxGlobalWords
#define BGSPAWCS_DEFAULT_MAX_NB_WORDS (50)
/// defines the default value for BackgroundSubtractorPAWCS::m_nSamplesForMovingAvgs
#define BGSPAWCS_DEFAULT_N_SAMPLES_FOR_MV_AVGS (100)

/**
    Pixel-based Adaptive Word Consensus Segmenter (PAWCS) algorithm for FG/BG video segmentation via change detection.

    Note: both grayscale and RGB/BGR images may be used with this extractor (parameters are adjusted automatically).
    For optimal grayscale results, use CV_8UC1 frames instead of CV_8UC3.

    For now, only the algorithm's default CPU implementation is offered here.

    For more details on the different parameters or on the algorithm itself, see P.-L. St-Charles et al.,
    "A Self-Adjusting Approach to Change Detection Based on Background Word Consensus", in WACV 2015, or
    P.-L. St-Charles et al., "Universal Background Subtraction Using Word Consensus Models", in IEEE Trans.
    on Image Processing, vol. 25 (10), 2016.
*/
template<lv::ParallelAlgoType eImpl>
struct BackgroundSubtractorPAWCS_;

template<>
struct BackgroundSubtractorPAWCS_<lv::NonParallel> : public IBackgroundSubtractorLBSP {
public:
    /// full constructor
    BackgroundSubtractorPAWCS_(size_t nDescDistThresholdOffset=BGSPAWCS_DEFAULT_DESC_DIST_THRESHOLD_OFFSET,
                               size_t nMinColorDistThreshold=BGSPAWCS_DEFAULT_MIN_COLOR_DIST_THRESHOLD,
                               size_t nMaxNbWords=BGSPAWCS_DEFAULT_MAX_NB_WORDS,
                               size_t nSamplesForMovingAvgs=BGSPAWCS_DEFAULT_N_SAMPLES_FOR_MV_AVGS,
                               float fRelLBSPThreshold=BGSLBSP_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD);
    /// refreshes all local (+ global) dictionaries based on the last analyzed frame
    virtual void refreshModel(size_t nBaseOccCount, float fOccDecrFrac, bool bForceFGUpdate=false);
    /// (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) override;
    /// primary model update function; the learning param is used to override the internal learning speed (ignored when <= 0)
    virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0) override;
    /// returns a copy of the latest reconstructed background image
    virtual void getBackgroundImage(cv::OutputArray backgroundImage) const override;
    /// returns a copy of the latest reconstructed background descriptors image
    virtual void getBackgroundDescriptorsImage(cv::OutputArray backgroundDescImage) const override;
    /// returns the default learning rate value used in 'apply'
    virtual double getDefaultLearningRate() const override {return 0;}

protected:
    template<size_t nChannels>
    struct ColorLBSPFeature {
        std::array<uchar,nChannels> anColor;
        std::array<ushort,nChannels> anDesc;
    };
    struct LocalWordBase {
        size_t nFirstOcc;
        size_t nLastOcc;
        size_t nOccurrences;
    };
    template<typename T>
    struct LocalWord : LocalWordBase {
        T oFeature;
    };
    struct GlobalWordBase {
        float fLatestWeight;
        cv::Mat oSpatioOccMap;
        uchar nDescBITS;
    };
    template<typename T>
    struct GlobalWord : GlobalWordBase {
        T oFeature;
    };
    typedef LocalWord<ColorLBSPFeature<1>> LocalWord_1ch;
    typedef LocalWord<ColorLBSPFeature<3>> LocalWord_3ch;
    typedef GlobalWord<ColorLBSPFeature<1>> GlobalWord_1ch;
    typedef GlobalWord<ColorLBSPFeature<3>> GlobalWord_3ch;
    struct PxInfo_PAWCS : PxInfoBase {
        size_t nGlobalWordMapLookupIdx;
        std::vector<GlobalWordBase*> vpGlobalDictSortLUT;
    };
    /// absolute minimal color distance threshold ('R' or 'radius' in the original ViBe paper, used as the default/initial 'R(x)' value here)
    const size_t m_nMinColorDistThreshold;
    /// absolute descriptor distance threshold offset
    const size_t m_nDescDistThresholdOffset;
    /// max/curr number of local words used to build background submodels (for a single pixel, similar to 'N' in ViBe/PBAS, may vary based on img/channel size)
    size_t m_nMaxLocalWords, m_nCurrLocalWords;
    /// max/curr number of global words used to build the global background model (may vary based on img/channel size)
    size_t m_nMaxGlobalWords, m_nCurrGlobalWords;
    /// number of samples to use to compute the learning rate of moving averages
    const size_t m_nSamplesForMovingAvgs;
    /// last calculated non-flat region ratio
    float m_fLastNonFlatRegionRatio;
    /// current kernel size for median blur post-proc filtering
    int m_nMedianBlurKernelSize;
    /// specifies the downsampled frame size used for cam motion analysis & gword lookup maps
    cv::Size m_oDownSampledFrameSize_MotionAnalysis, m_oDownSampledFrameSize_GlobalWordLookup;
    /// downsampled version of the ROI used for cam motion analysis
    cv::Mat m_oDownSampledROI_MotionAnalysis;
    /// total pixel count for the downsampled ROIs
    size_t m_nDownSampledROIPxCount;
    /// current local word weight offset
    size_t m_nLocalWordWeightOffset;

    /// word lists & dictionaries
    std::vector<LocalWordBase*> m_vpLocalWordDict;
    std::vector<LocalWord_1ch> m_voLocalWordList_1ch;
    std::vector<LocalWord_3ch> m_voLocalWordList_3ch;
    std::vector<LocalWord_1ch>::iterator m_pLocalWordListIter_1ch;
    std::vector<LocalWord_3ch>::iterator m_pLocalWordListIter_3ch;
    std::vector<GlobalWordBase*> m_vpGlobalWordDict;
    std::vector<GlobalWord_1ch> m_voGlobalWordList_1ch;
    std::vector<GlobalWord_3ch> m_voGlobalWordList_3ch;
    std::vector<GlobalWord_1ch>::iterator m_pGlobalWordListIter_1ch;
    std::vector<GlobalWord_3ch>::iterator m_pGlobalWordListIter_3ch;
    std::vector<PxInfo_PAWCS> m_voPxInfoLUT_PAWCS;

    /// a lookup map used to keep track of regions where illumination recently changed
    cv::Mat m_oIllumUpdtRegionMask;
    /// per-pixel update rates ('T(x)' in PBAS, which contains pixel-level 'sigmas', as referred to in ViBe)
    cv::Mat m_oUpdateRateFrame;
    /// per-pixel distance thresholds (equivalent to 'R(x)' in PBAS, but used as a relative value to determine both intensity and descriptor variation thresholds)
    cv::Mat m_oDistThresholdFrame;
    /// per-pixel distance threshold variation modulators ('v(x)', relative value used to modulate 'R(x)' and 'T(x)' variations)
    cv::Mat m_oDistThresholdVariationFrame;
    /// per-pixel mean minimal distances from the model ('D_min(x)' in PBAS, used to control variation magnitude and direction of 'T(x)' and 'R(x)')
    cv::Mat m_oMeanMinDistFrame_LT, m_oMeanMinDistFrame_ST;
    /// per-pixel mean downsampled distances between consecutive frames (used to analyze camera movement and force global model resets automatically)
    cv::Mat m_oMeanDownSampledLastDistFrame_LT, m_oMeanDownSampledLastDistFrame_ST;
    /// per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
    cv::Mat m_oMeanRawSegmResFrame_LT, m_oMeanRawSegmResFrame_ST;
    /// per-pixel mean raw segmentation results (used to detect unstable segmentation regions)
    cv::Mat m_oMeanFinalSegmResFrame_LT, m_oMeanFinalSegmResFrame_ST;
    /// a lookup map used to keep track of unstable regions (based on segm. noise & local dist. thresholds)
    cv::Mat m_oUnstableRegionMask;
    /// per-pixel blink detection map ('Z(x)')
    cv::Mat m_oBlinksFrame;
    /// pre-allocated matrix used to downsample the input frame when needed
    cv::Mat m_oDownSampledFrame_MotionAnalysis;
    /// the foreground mask generated by the method at t-1 (without post-proc, used for blinking px detection)
    cv::Mat m_oLastRawFGMask;

    /// pre-allocated CV_8UC1 matrices used to speed up morph ops
    cv::Mat m_oFGMask_PreFlood;
    cv::Mat m_oFGMask_FloodedHoles;
    cv::Mat m_oLastFGMask_dilated;
    cv::Mat m_oLastFGMask_dilated_inverted;
    cv::Mat m_oCurrRawFGBlinkMask;
    cv::Mat m_oLastRawFGBlinkMask;
    cv::Mat m_oTempGlobalWordWeightDiffFactor;
    cv::Mat m_oMorphExStructElement;

    /// internal weight lookup function for local words
    static float GetLocalWordWeight(const LocalWordBase& w, size_t nCurrFrame, size_t nOffset);
    /// internal weight lookup function for global words
    static float GetGlobalWordWeight(const GlobalWordBase& w);
};

using BackgroundSubtractorPAWCS = BackgroundSubtractorPAWCS_<lv::NonParallel>;
