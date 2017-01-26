
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

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "litiv/utils/math.hpp"

/**
    Local Self-Similarirty (LSS) feature extractor

    For more details on the different parameters, see E. Shechtman and M. Irani, "Matching Local
    Self-Similarities across Images and Videos", in CVPR2007.

*/
class LSS : public cv::DescriptorExtractor {
public:
    /// default constructor
    LSS(int nDescPatchSize=5, int nDescRadius=40, int nRadialBins=3, int nAngularBins=12, float fStaticNoiseVar=300000.f, bool bPreProcess=false);
    /// loads extractor params from the specified file node @@@@ not impl
    virtual void read(const cv::FileNode&) override;
    /// writes extractor params to the specified file storage @@@@ not impl
    virtual void write(cv::FileStorage&) const override;
    /// returns the window size that will be used around each keypoint (also gives the minimum image size required for description)
    virtual cv::Size windowSize() const;
    /// returns the border size required around each keypoint in x or y direction (also gives the invalid descriptor border size for output maps to ignore)
    virtual int borderSize(int nDim=0) const; // typically equal to windowSize().width/2
    /// returns the current descriptor size, in bytes (overrides cv::DescriptorExtractor's)
    virtual int descriptorSize() const override;
    /// returns the current descriptor data type (overrides cv::DescriptorExtractor's)
    virtual int descriptorType() const override;
    /// returns the default norm type to use with this descriptor (overrides cv::DescriptorExtractor's)
    virtual int defaultNorm() const override;
    /// return true if detector object is empty (overrides cv::DescriptorExtractor's)
    virtual bool empty() const override;

    /// returns whether the noise variation levels will be dynamically determined or not for internal normalization
    bool isUsingDynamicNoiseVarNorm() const;
    /// returns whether the algorithm will use iterative SSD processing for fast dense description (with approx result)
    bool isUsingIterativeSSD() const;
    /// returns whether descriptor bin arrays will be 0-1 normalized before returning or not
    bool isNormalizingBins() const;
    /// returns whether input images will be preprocessed using a gaussian filter or not
    bool isPreProcessing() const;

    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix, and all image points are described
    void compute2(const cv::Mat& oImage, cv::Mat_<float>& oDescriptors);
    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix
    void compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescriptors);
    /// batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<cv::Mat_<float>>& voDescCollection);
    /// batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat_<float>>& voDescCollection);

    /// utility function, used to reshape a descriptors matrix to its input image size (assumes fully-dense keypoints over input)
    void reshapeDesc(cv::Size oSize, cv::Mat& oDescriptors) const;
    /// utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
    void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) const;
    /// utility function, used to filter out bad pixels in a ROI that would trigger out of bounds error because they're too close to the image border
    void validateROI(cv::Mat& oROI) const;
    /// utility function, used to calculate per-desc distance between two descriptor sets/maps using L2 distance
    void calcDistance(const cv::Mat_<float>& oDescriptors1, const cv::Mat_<float>& oDescriptors2, cv::Mat_<float>& oDistances) const;

protected:
    /// hides default keypoint detection impl (this class is a descriptor extractor only)
    using cv::DescriptorExtractor::detect;
    /// classic 'compute' implementation, based on DescriptorExtractor's arguments & expected output
    virtual void detectAndCompute(cv::InputArray oImage, cv::InputArray oMask, std::vector<cv::KeyPoint>& voKeypoints, cv::OutputArray oDescriptors, bool bUseProvidedKeypoints=false) override;
    /// defines whether input images should be preprocessed using a gaussian filter or not
    const bool m_bPreProcess; // default = false
    /// size of internal ssd patches (must be odd)
    const int m_nDescPatchSize; // default = 5
    /// radius of the LSS descriptor (center to patch ring distance)
    const int m_nDescRadius; // default = 40
    /// size of the LSS descriptor correlation window
    const int m_nCorrWinSize; // deduced
    /// size of the SSD correlation output patch
    const int m_nCorrPatchSize; // deduced
    /// radial bins count
    const int m_nRadialBins; // default = 3
    /// angular bins count
    const int m_nAngularBins; // default = 12
    /// static noise suppression level to use while binning
    const float m_fStaticNoiseVar; // default = 300000

private:
    /// keypoint-based description approach impl
    void ssdescs_impl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescriptors);
    /// dense description approach impl
    void ssdescs_impl(const cv::Mat& oImage, cv::Mat_<float>& oDescriptors);
    /// descriptor normalisation approach impl
    void ssdescs_norm(cv::Mat_<float>& oDescriptors) const;

    // helper variables for internal impl (helps avoid continuous mem realloc)
    cv::Mat_<float> m_oCorrMap,m_oCorrDiffMap,m_oFullColCorrMap;
    cv::Mat_<int> m_oDescLUMap;
    int m_nFirstMaskIdx,m_nLastMaskIdx;
};