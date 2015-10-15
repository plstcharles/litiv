
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

#include <opencv2/video/background_segm.hpp>

//! defines the default value for BackgroundSubtractorViBe::m_nColorDistThreshold
#define BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD (20)
//! defines the default value for BackgroundSubtractorViBe::m_nBGSamples
#define BGSVIBE_DEFAULT_NB_BG_SAMPLES (20)
//! defines the default value for BackgroundSubtractorViBe::m_nRequiredBGSamples
#define BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES (2)
//! defines the default value for the learning rate passed to BackgroundSubtractorViBe::apply (the 'subsampling' factor in the original ViBe paper)
#define BGSVIBE_DEFAULT_LEARNING_RATE (16)
//! defines the internal threshold adjustment factor to use when determining if the variation of a single channel is enough to declare the pixel as foreground
#define BGSVIBE_SINGLECHANNEL_THRESHOLD_DIFF_FACTOR (1.60f)
//! defines whether we should use single channel variation checks for fg/bg segmentation validation or not
#define BGSVIBE_USE_SC_THRS_VALIDATION 0
//! defines whether we should use L1 distance or L2 distance for change detection
#define BGSVIBE_USE_L1_DISTANCE_CHECK 0

/*!
    ViBe foreground-background segmentation algorithm (abstract version).
 */
class BackgroundSubtractorViBe : public cv::BackgroundSubtractor {
public:
    //! full constructor
    BackgroundSubtractorViBe(size_t nColorDistThreshold=BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
                             size_t nBGSamples=BGSVIBE_DEFAULT_NB_BG_SAMPLES,
                             size_t nRequiredBGSamples=BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    //! default destructor
    virtual ~BackgroundSubtractorViBe();
    //! (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg) = 0;
    //! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBE_DEFAULT_LEARNING_RATE) = 0;
    //! returns a copy of the latest reconstructed background image
    void getBackgroundImage(cv::OutputArray backgroundImage) const;

protected:
    //! number of different samples per pixel/block to be taken from input frames to build the background model ('N' in the original ViBe paper)
    const size_t m_nBGSamples;
    //! number of similar samples needed to consider the current pixel/block as 'background' ('#_min' in the original ViBe paper)
    const size_t m_nRequiredBGSamples;
    //! background model pixel intensity samples
    std::vector<cv::Mat> m_voBGImg;
    //! input image size
    cv::Size m_oImgSize;
    //! absolute color distance threshold ('R' or 'radius' in the original ViBe paper)
    const size_t m_nColorDistThreshold;
    //! defines whether or not the subtractor is fully initialized
    bool m_bInitialized;
};

/*!
    ViBe foreground-background segmentation algorithm (1ch/grayscale version).
 */
class BackgroundSubtractorViBe_1ch : public BackgroundSubtractorViBe {
public:
    //! full constructor
    BackgroundSubtractorViBe_1ch(size_t nColorDistThreshold=BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
                                 size_t nBGSamples=BGSVIBE_DEFAULT_NB_BG_SAMPLES,
                                 size_t nRequiredBGSamples=BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    //! default destructor
    virtual ~BackgroundSubtractorViBe_1ch();
    //! (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg);
    //! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBE_DEFAULT_LEARNING_RATE);
};

/*!
    ViBe foreground-background segmentation algorithm (3ch/RGB version).
 */
class BackgroundSubtractorViBe_3ch : public BackgroundSubtractorViBe {
public:
    //! full constructor
    BackgroundSubtractorViBe_3ch(size_t nColorDistThreshold=BGSVIBE_DEFAULT_COLOR_DIST_THRESHOLD,
                                 size_t nBGSamples=BGSVIBE_DEFAULT_NB_BG_SAMPLES,
                                 size_t nRequiredBGSamples=BGSVIBE_DEFAULT_REQUIRED_NB_BG_SAMPLES);
    //! default destructor
    virtual ~BackgroundSubtractorViBe_3ch();
    //! (re)initiaization method; needs to be called before starting background subtraction
    virtual void initialize(const cv::Mat& oInitImg);
    //! primary model update function; the learning param is reinterpreted as an integer and should be > 0 (smaller values == faster adaptation)
    virtual void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate=BGSVIBE_DEFAULT_LEARNING_RATE);
};
