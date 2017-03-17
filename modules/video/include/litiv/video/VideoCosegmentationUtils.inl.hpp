
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

#ifndef __LITIV_VIDEOCOSEGM_HPP__
#error "Cannot include .inl.hpp headers directly!"
#endif //ndef(__LITIV_VIDEOCOSEGM_HPP__)
#pragma once

template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void IVideoCosegmentor_<eImpl,TLabel,nInputArraySize,nOutputArraySize>::initialize(const FrameArrayIn& aImages) {
    initialize(aImages,FrameArrayROI());
}

template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void IVideoCosegmentor_<eImpl,TLabel,nInputArraySize,nOutputArraySize>::setAutomaticModelReset(bool bVal) {
    m_bAutoModelResetEnabled = bVal;
}

template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void IVideoCosegmentor_<eImpl,TLabel,nInputArraySize,nOutputArraySize>::validateROIs(FrameArrayROI& aROIs) const {
    for(size_t nArrayIdx=0; nArrayIdx<aROIs.size(); ++nArrayIdx) {
        lvAssert_(!aROIs[nArrayIdx].empty(),"provided ROIs must be non-empty");
        lvAssert_(cv::countNonZero(aROIs[nArrayIdx])>0,"provided ROIs must have at least one non-zero pixel");
    }
    if(m_nROIBorderSize>0) {
        for(size_t nArrayIdx=0; nArrayIdx<aROIs.size(); ++nArrayIdx) {
            cv::Mat oROI_new(aROIs[nArrayIdx].size(),CV_8UC1,cv::Scalar_<uchar>(0));
            const cv::Rect oROI_inner((int)m_nROIBorderSize,(int)m_nROIBorderSize,aROIs[nArrayIdx].cols-int(m_nROIBorderSize*2),aROIs[nArrayIdx].rows-int(m_nROIBorderSize*2));
            cv::Mat(aROIs[nArrayIdx],oROI_inner).copyTo(cv::Mat(oROI_new,oROI_inner));
            aROIs[nArrayIdx] = oROI_new;
        }
    }
}

template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void IVideoCosegmentor_<eImpl,TLabel,nInputArraySize,nOutputArraySize>::setROIs(FrameArrayROI& aROIs) {
    validateROIs(aROIs);
    for(size_t nArrayIdx=0; nArrayIdx<aROIs.size(); ++nArrayIdx)
        m_aROIs[nArrayIdx] = aROIs[nArrayIdx].clone();
}

template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
std::array<cv::Mat_<uchar>,nInputArraySize> IVideoCosegmentor_<eImpl,TLabel,nInputArraySize,nOutputArraySize>::getROIsCopy() const {
    FrameArrayROI aROIs;
    for(size_t nArrayIdx=0; nArrayIdx<aROIs.size(); ++nArrayIdx)
        aROIs[nArrayIdx] = m_aROIs[nArrayIdx].clone();
    return aROIs;
}

template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
IVideoCosegmentor_<eImpl,TLabel,nInputArraySize,nOutputArraySize>::IVideoCosegmentor_() :
        m_bInitialized(false),
        m_bModelInitialized(false),
        m_bAutoModelResetEnabled(true),
        m_nROIBorderSize(0),
        m_nFrameIdx(SIZE_MAX),
        m_nFramesSinceLastReset(0),
        m_nModelResetCooldown(0),
        m_aROIs{},
        m_anTotPxCounts{},
        m_anOrigROIPxCounts{},
        m_anFinalROIPxCounts{},
        m_aLastMasks{},
        m_aLastInputs{} {}

template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void IVideoCosegmentor_<eImpl,TLabel,nInputArraySize,nOutputArraySize>::initialize_common(const FrameArrayIn& aImages, const FrameArrayROI& aROIs) {
    lvAssert_(nInputArraySize>0,"must have at least one mat in input array");
    m_bInitialized = false;
    m_bModelInitialized = false;
    bool bFoundChDiff = false;
    std::array<cv::Mat_<uchar>,nInputArraySize> aNewROIs;
    for(size_t nArrayIdx=0; nArrayIdx<aROIs.size(); ++nArrayIdx) {
        lvAssert_(!aImages[nArrayIdx].empty() && aImages[nArrayIdx].isContinuous(),"provided images for initialization must be non-empty and continuous");
        if(aImages[nArrayIdx].channels()>1 && !bFoundChDiff) {
            std::vector<cv::Mat> vChannels;
            cv::split(aImages[nArrayIdx],vChannels);
            for(size_t c = 1; c<vChannels.size(); ++c)
                if((bFoundChDiff = (cv::countNonZero(vChannels[0]!=vChannels[c])!=0)))
                    break;
            if(!bFoundChDiff)
                std::cerr << "\n\tIVideoCosegmentor_ : Warning, grayscale images should always be passed in CV_8UC1 format for optimal performance.\n" << std::endl;
        }
        if(aROIs[nArrayIdx].empty())
            aNewROIs[nArrayIdx] = cv::Mat_<uchar>(aImages[nArrayIdx].size(),uchar(UCHAR_MAX));
        else {
            lvAssert_(aROIs[nArrayIdx].size==aImages[nArrayIdx].size,"provided ROIs mat size must be equal to the init images size");
            aNewROIs[nArrayIdx] = aROIs[nArrayIdx].clone();
        }
        m_anOrigROIPxCounts[nArrayIdx] = (size_t)cv::countNonZero(aNewROIs[nArrayIdx]);
        lvAssert_(m_anOrigROIPxCounts[nArrayIdx]>0,"provided ROI mat contains no useful pixels");
    }
    validateROIs(aNewROIs);
    for(size_t nArrayIdx=0; nArrayIdx<aROIs.size(); ++nArrayIdx) {
        m_anFinalROIPxCounts[nArrayIdx] = (size_t)cv::countNonZero(aNewROIs[nArrayIdx]);
        lvAssert_(m_anFinalROIPxCounts[nArrayIdx]>0,"provided ROI mat contains no useful pixels away from borders (descriptors will hit image bounds)");
        m_aROIs[nArrayIdx] = aNewROIs[nArrayIdx];
        this->m_anInputInfos[nArrayIdx] = lv::MatInfo(aImages[nArrayIdx]);
        m_anTotPxCounts[nArrayIdx] = aNewROIs[nArrayIdx].total();
        m_nFrameIdx = 0;
        m_nFramesSinceLastReset = 0;
        m_nModelResetCooldown = 0;
        m_aLastMasks[nArrayIdx].create(this->m_anInputInfos[nArrayIdx].size.dims(),this->m_anInputInfos[nArrayIdx].size.sizes);
        m_aLastMasks[nArrayIdx] = cv::Scalar::all(0);
        m_aLastInputs[nArrayIdx].create(this->m_anInputInfos[nArrayIdx].size.dims(),this->m_anInputInfos[nArrayIdx].size.sizes,this->m_anInputInfos[nArrayIdx].type());
        m_aLastInputs[nArrayIdx] = cv::Scalar::all(0);
    }
}