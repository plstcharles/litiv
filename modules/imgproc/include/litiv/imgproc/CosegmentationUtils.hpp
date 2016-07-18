
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

template<typename TLabel, size_t nArraySize>
struct ICosegmentor : public cv::Algorithm {
    /// referenceable variable holding image array size template parameter
    static constexpr size_t s_nArraySize = nArraySize;
    /// shortcut to matrix array type for input/output operations
    using MatArray = std::array<cv::Mat,s_nArraySize>;
    /// shortcut to label template typename parameter
    using LabelType = TLabel;

    /// image cosegmentation function; will isolate visible structures common to all input images and label them similarily in all output masks
    virtual void apply(const MatArray& aImages, MatArray& aMasks) = 0;
    /// image cosegmentation function; check that the input/output arrays are the right size, and redirect to the other 'apply' interface
    void apply(cv::InputArrayOfArrays _aImages, cv::OutputArrayOfArrays _aMasks);
    /// returns the (maximum) number of labels used in the output masks, or -1 if it cannot be predetermined
    virtual int getMaxLabelCount() const = 0;
    /// returns the list of labels used in the output masks, or an empty array if it cannot be predetermined
    virtual std::vector<LabelType> getLabels() const = 0;
    /// required for derived class destruction from this interface
    virtual ~ICosegmentor() {}
    /// default (empty) constructor for base classes
    inline ICosegmentor() {}

private:
    ICosegmentor& operator=(const ICosegmentor&) = delete;
    ICosegmentor(const ICosegmentor&) = delete;
};

template<typename TLabel, size_t nArraySize>
void ICosegmentor<TLabel,nArraySize>::apply(cv::InputArrayOfArrays _aImages, cv::OutputArrayOfArrays _aMasks) {
    lvAssert_(_aImages.isMatVector(),"first argument must be a mat vector (or mat array)");
    std::vector<cv::Mat> vImages;
    _aImages.getMatVector(vImages);
    lvAssert__(vImages.size()==s_nArraySize,"number of images in the input array must match the predetermined one (%d)",(int)s_nArraySize);
    MatArray aImages,aMasks;
    std::copy_n(vImages.begin(),s_nArraySize,aImages.begin());
    if(aMasks.isMatVector()) {
        std::vector<cv::Mat> vMasks;
        aMasks.getMatVector(vMasks);
        lvAssert__(vMasks.size()==s_nArraySize,"number of images in the output array must match the predetermined one (%d)",(int)s_nArraySize);
        std::copy_n(vMasks.begin(),s_nArraySize,aMasks.begin());
    }
    apply(aImages,aMasks);
}
