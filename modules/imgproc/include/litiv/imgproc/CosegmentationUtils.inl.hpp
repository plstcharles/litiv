
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

#ifndef __LITIV_COSEGM_HPP__
#error "Cannot include .inl.hpp headers directly!"
#endif //ndef(__LITIV_COSEGM_HPP__)
#pragma once

template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void IICosegmentor<TLabel,nInputArraySize,nOutputArraySize>::apply(const std::vector<cv::Mat>& vImages, std::vector<cv::Mat_<LabelType>>& vMasks) {
    MatArrayIn aImages;
    lvAssert__(vImages.size()==aImages.size(),"number of images in the input array must match the predetermined one (got %d instead of %d)",(int)vImages.size(),(int)aImages.size());
    std::copy_n(vImages.begin(),aImages.size(),aImages.begin());
    MatArrayOut aMasks;
    if(!vMasks.empty()) {
        lvAssert__(vMasks.size()==aMasks.size(),"number of images in the output array must match the predetermined one (got %d instead of %d)",(int)vMasks.size(),(int)aMasks.size());
        for(size_t nArrayIdx=0; nArrayIdx<aMasks.size(); ++nArrayIdx)
            aMasks[nArrayIdx] = vMasks[nArrayIdx];
    }
    else
        vMasks.resize(aMasks.size());
    apply(aImages,aMasks);
    for(size_t nArrayIdx=0; nArrayIdx<aMasks.size(); ++nArrayIdx)
        vMasks[nArrayIdx] = aMasks[nArrayIdx];
}

template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void IICosegmentor<TLabel,nInputArraySize,nOutputArraySize>::apply(cv::InputArrayOfArrays _vImages, cv::OutputArrayOfArrays _vMasks) {
    lvAssert_(_vImages.isMatVector(),"first argument must be a mat vector (or mat array)");
    std::vector<cv::Mat> vImages;
    _vImages.getMatVector(vImages);
    MatArrayIn aImages;
    lvAssert__(vImages.size()==aImages.size(),"number of images in the input array must match the predetermined one (got %d instead of %d)",(int)vImages.size(),(int)aImages.size());
    std::copy_n(vImages.begin(),aImages.size(),aImages.begin());
    MatArrayOut aMasks;
    if(!_vMasks.empty()) {
        lvAssert_(_vMasks.isMatVector(),"second argument must be an empty mat or a mat vector (or mat array)");
        std::vector<cv::Mat> vMasks;
        _vMasks.getMatVector(vMasks);
        lvAssert__(vMasks.size()==aMasks.size(),"number of images in the output array must match the predetermined one (got %d instead of %d)",(int)vMasks.size(),(int)aMasks.size());
        for(size_t nArrayIdx=0; nArrayIdx<aMasks.size(); ++nArrayIdx)
            aMasks[nArrayIdx] = vMasks[nArrayIdx];
    }
    else
        _vMasks.create((int)aMasks.size(),1,0);
    apply(aImages,aMasks);
    for(size_t nArrayIdx=0; nArrayIdx<aMasks.size(); ++nArrayIdx)
        _vMasks.getMatRef((int)nArrayIdx) = aMasks[nArrayIdx];
}

template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
IICosegmentor<TLabel,nInputArraySize,nOutputArraySize>::IICosegmentor() :
        m_anInputInfos{} {}