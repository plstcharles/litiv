
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

#include "litiv/utils/opencv.hpp"

/// super-interface for cosegmentation algos which exposes common interface functions
template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize=nInputArraySize>
struct ICosegmentor : public cv::Algorithm {
    /// shortcut to label template typename parameter
    using LabelType = TLabel;
    /// shortcut to input matrix array type
    using MatArrayIn = std::array<cv::Mat,nInputArraySize>;
    /// shortcut to output matrix array type
    using MatArrayOut = std::array<cv::Mat_<LabelType>,nOutputArraySize>;
    /// image cosegmentation function; will isolate visible structures common to all input images and label them similarily in all output masks
    virtual void apply(const MatArrayIn& aImages, MatArrayOut& aMasks) = 0;
    /// image cosegmentation function; check that the input/output arrays are the right size+type, and redirect to the other 'apply' interface
    void apply(cv::InputArrayOfArrays aImages, cv::OutputArrayOfArrays aMasks);
    /// returns the maximum number of labels used in the output masks, or 0 if it cannot be predetermined
    virtual size_t getMaxLabelCount() const = 0;
    /// returns the list of labels used in the output masks, or an empty array if it cannot be predetermined
    virtual const std::vector<LabelType>& getLabels() const = 0;
    /// required for derived class destruction from this interface
    virtual ~ICosegmentor() = default;
};

template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
void ICosegmentor<TLabel,nInputArraySize,nOutputArraySize>::apply(cv::InputArrayOfArrays _aImages, cv::OutputArrayOfArrays _aMasks) {
    lvAssert_(_aImages.isMatVector(),"first argument must be a mat vector (or mat array)");
    std::vector<cv::Mat> vImages;
    _aImages.getMatVector(vImages);
    MatArrayIn aImages;
    lvAssert__(vImages.size()==aImages.size(),"number of images in the input array must match the predetermined one (%d)",(int)aImages.size());
    std::copy_n(vImages.begin(),aImages.size(),aImages.begin());
    MatArrayOut aMasks;
    if(!_aMasks.empty()) {
        lvAssert_(_aMasks.isMatVector(),"second argument must be an empty mat or a mat vector (or mat array)");
        std::vector<cv::Mat> vMasks;
        _aMasks.getMatVector(vMasks);
        lvAssert__(vMasks.size()==aMasks.size(),"number of images in the output array must match the predetermined one (%d)",(int)aMasks.size());
        for(size_t nArrayIdx=0; nArrayIdx<aMasks.size(); ++nArrayIdx) {
            lvAssert_(vMasks[nArrayIdx].elemSize()==sizeof(LabelType),"depth of images in the output array must match sizeof(LabelType)");
            aMasks[nArrayIdx] = vMasks[nArrayIdx];
        }
    }
    apply(aImages,aMasks);
}
