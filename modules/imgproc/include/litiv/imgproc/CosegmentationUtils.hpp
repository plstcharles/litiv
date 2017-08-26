
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
struct IICosegmentor : public cv::Algorithm {
    static_assert(lv::isDataTypeCompat<TLabel>(),"specified label type cannot be used in cv::Mat_'s");
    /// templated size of the input image array
    static constexpr size_t s_nInputArraySize = nInputArraySize;
    /// templated size of the output image array
    static constexpr size_t s_nOutputArraySize = nOutputArraySize;
    /// shortcut to label template typename parameter
    using LabelType = TLabel;
    /// shortcut to input matrix array type
    using MatArrayIn = std::array<cv::Mat,nInputArraySize>;
    /// shortcut to output matrix array type
    using MatArrayOut = std::array<cv::Mat_<LabelType>,nOutputArraySize>;
    /// image cosegmentation function; will isolate visible structures common to all input images and label them similarily in all output masks
    virtual void apply(const MatArrayIn& aImages, MatArrayOut& aMasks) = 0;
    /// image cosegmentation function; check that the input/output vectors are the right size+type, and redirect to the 'apply' interface using static arrays
    void apply(const std::vector<cv::Mat>& vImages, std::vector<cv::Mat_<LabelType>>& vMasks);
    /// image cosegmentation function; check that the input/output arrays are the right size+type, and redirect to the 'apply' interface using static arrays
    void apply(cv::InputArrayOfArrays vImages, cv::OutputArrayOfArrays vMasks);
    /// returns the maximum number of labels used in the output masks, or 0 if it cannot be predetermined
    virtual size_t getMaxLabelCount() const = 0;
    /// returns the list of labels used in the output masks, or an empty array if it cannot be predetermined
    virtual const std::vector<LabelType>& getLabels() const = 0;
    /// returns the expected input stream array size
    static constexpr size_t getInputStreamCount() {return nInputArraySize;}
    /// returns the output stream array size
    static constexpr size_t getOutputStreamCount() {return nOutputArraySize;}
    /// required for derived class destruction from this interface
    virtual ~IICosegmentor() = default;
protected:
    /// default impl constructor (for common parameters only -- none must be const to avoid constructor hell when deriving)
    IICosegmentor();
    /// input image infos
    std::array<lv::MatInfo,nInputArraySize> m_anInputInfos;
private:
    IICosegmentor& operator=(const IICosegmentor&) = delete;
    IICosegmentor(const IICosegmentor&) = delete;
};

/// interface for specialized cosegm algo impl types
template<lv::ParallelAlgoType eImpl, typename TLabel, size_t nInputArraySize, size_t nOutputArraySize=nInputArraySize>
struct ICosegmentor_;

#if HAVE_CUDA

/// interface for cuda cosegm algo impls
template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
struct ICosegmentor_<lv::CUDA,TLabel,nInputArraySize,nOutputArraySize> :
        public lv::IParallelAlgo_CUDA,
        public IICosegmentor<TLabel,nInputArraySize,nOutputArraySize> {
    /// required for derived class destruction from this interface
    virtual ~ICosegmentor_() = default;
    using IICosegmentor<TLabel,nInputArraySize,nOutputArraySize>::apply;
};

/// typename shortcut for cuda cosegm algo impls
template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize=nInputArraySize>
using ICosegmentor_CUDA = ICosegmentor_<lv::CUDA,TLabel,nInputArraySize,nOutputArraySize>;

#endif

/// interface for non-parallel (default) cosegm algo impls
template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize>
struct ICosegmentor_<lv::NonParallel,TLabel,nInputArraySize,nOutputArraySize> :
        public lv::NonParallelAlgo,
        public IICosegmentor<TLabel,nInputArraySize,nOutputArraySize> {
    /// required for derived class destruction from this interface
    virtual ~ICosegmentor_() = default;
    using IICosegmentor<TLabel,nInputArraySize,nOutputArraySize>::apply;
};

/// typename shortcut for non-parallel (default) cosegm algo impls
template<typename TLabel, size_t nInputArraySize, size_t nOutputArraySize=nInputArraySize>
using ICosegmentor = ICosegmentor_<lv::NonParallel,TLabel,nInputArraySize,nOutputArraySize>;

#define __LITIV_COSEGM_HPP__
#include "litiv/imgproc/CosegmentationUtils.inl.hpp"
#undef __LITIV_COSEGM_HPP__