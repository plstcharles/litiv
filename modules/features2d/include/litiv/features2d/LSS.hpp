
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

#include "litiv/utils/opencv.hpp"
#include "litiv/utils/math.hpp"
#include <opencv2/features2d.hpp>

#define LSS_DEFAULT_INNER_RADIUS   (0)
#define LSS_DEFAULT_OUTER_RADIUS   (20)
#define LSS_DEFAULT_PATCH_SIZE     (5)
#define LSS_DEFAULT_RADIAL_BINS    (3)
#define LSS_DEFAULT_ANGULAR_BINS   (12)
#define LSS_DEFAULT_STATNOISE_VAR  (300000.f)
#define LSS_DEFAULT_NORM_BINS      (true)
#define LSS_DEFAULT_PREPROCESS     (true)
#define LSS_DEFAULT_USE_LIENH_MASK (true)

/**
    Local Self-Similarirty (LSS) feature extractor

    For more details on the different parameters, see E. Shechtman and M. Irani, "Matching Local
    Self-Similarities across Images and Videos", in CVPR2007.

*/
class LSS : public cv::DescriptorExtractor {
public:
    /// default constructor
    LSS(int nInnerRadius=LSS_DEFAULT_INNER_RADIUS,
        int nOuterRadius=LSS_DEFAULT_OUTER_RADIUS,
        int nPatchSize=LSS_DEFAULT_PATCH_SIZE,
        int nAngularBins=LSS_DEFAULT_ANGULAR_BINS,
        int nRadialBins=LSS_DEFAULT_RADIAL_BINS,
        float fStaticNoiseVar=LSS_DEFAULT_STATNOISE_VAR,
        bool bNormalizeBins=LSS_DEFAULT_NORM_BINS,
        bool bPreProcess=LSS_DEFAULT_PREPROCESS,
        bool bUseLienhartMask=LSS_DEFAULT_USE_LIENH_MASK);
    /// loads extractor params from the specified file node @@@@ not impl
    virtual void read(const cv::FileNode&) override;
    /// writes extractor params to the specified file storage @@@@ not impl
    virtual void write(cv::FileStorage&) const override;
    /// returns the window size that will be used around each keypoint (also gives the minimum image size required for description)
    virtual cv::Size windowSize() const;
    /// returns the border size required around each keypoint in x or y direction (also gives the invalid descriptor border size for output maps to ignore)
    virtual int borderSize(int nDim=0) const; // typically equal to windowSize().width/2
    /// returns the expected dense descriptor matrix output info, for a given input matrix size/type
    virtual lv::MatInfo getOutputInfo(const lv::MatInfo& oInputInfo) const;
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
    /// returns whether descriptor bin arrays will be 0-1 normalized before returning or not
    bool isNormalizingBins() const;
    /// returns whether input images will be preprocessed using a gaussian filter or not
    bool isPreProcessing() const;
    /// returns whether using Lienhart's lookup mask implementation instead of Chatfield's
    bool isUsingLienhartMask() const;

    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix, and all image points are described (note: descriptors close to borders will be invalid)
    void compute2(const cv::Mat& oImage, cv::Mat& oDescMap);
    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix, and all image points are described (note: descriptors close to borders will be invalid)
    void compute2(const cv::Mat& oImage, cv::Mat_<float>& oDescMap);
    /// similar to DescriptorExtractor::compute(const cv::Mat& image, ...), but in this case, the descriptors matrix has the same shape as the input matrix
    void compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescMap);
    /// batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<cv::Mat_<float>>& voDescMapCollection);
    /// batch version of LBSP::compute2(const cv::Mat& image, ...)
    void compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat_<float>>& voDescMapCollection);

    /// utility function, used to reshape a descriptors matrix to its input image size (assumes fully-dense keypoints over input)
    void reshapeDesc(cv::Size oSize, cv::Mat& oDescriptors) const;
    /// utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
    void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) const;
    /// utility function, used to filter out bad pixels in a ROI that would trigger out of bounds error because they're too close to the image border
    void validateROI(cv::Mat& oROI) const;
    /// utility function, used to calculate the L2 distance between two individual descriptors
    inline double calcDistance(const float* aDescriptor1, const float* aDescriptor2) const {
        const cv::Mat_<float> oDesc1(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(aDescriptor1));
        const cv::Mat_<float> oDesc2(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(aDescriptor2));
        return cv::norm(oDesc1,oDesc2,cv::NORM_L2);
    }
    /// utility function, used to calculate the L2 distance between two individual descriptors
    inline double calcDistance(const cv::Mat_<float>& oDescriptor1, const cv::Mat_<float>& oDescriptor2) const {
        lvAssert_(oDescriptor1.dims==oDescriptor2.dims && oDescriptor1.size==oDescriptor2.size,"descriptor mat sizes mismatch");
        lvAssert_(oDescriptor1.dims==2 || oDescriptor1.dims==3,"unexpected descriptor matrix dim count");
        lvAssert_(oDescriptor1.dims!=2 || oDescriptor1.total()==size_t(m_nRadialBins*m_nAngularBins),"unexpected descriptor size");
        lvAssert_(oDescriptor1.dims!=3 || (oDescriptor1.size[0]==1 && oDescriptor1.size[1]==1 && oDescriptor1.size[2]==m_nRadialBins*m_nAngularBins),"unexpected descriptor size");
        return calcDistance(oDescriptor1.ptr<float>(0),oDescriptor2.ptr<float>(0));
    }
    /// utility function, used to calculate per-desc L2 distance between two descriptor sets/maps
    void calcDistances(const cv::Mat_<float>& oDescriptors1, const cv::Mat_<float>& oDescriptors2, cv::Mat_<float>& oDistances) const;

protected:
    /// hides default keypoint detection impl (this class is a descriptor extractor only)
    using cv::DescriptorExtractor::detect;
    /// classic 'compute' implementation, based on DescriptorExtractor's arguments & expected output
    virtual void detectAndCompute(cv::InputArray oImage, cv::InputArray oMask, std::vector<cv::KeyPoint>& voKeypoints, cv::OutputArray oDescriptors, bool bUseProvidedKeypoints=false) override;
    /// defines whether input images should be preprocessed using a gaussian filter or not
    const bool m_bPreProcess;
    /// defines whether output descriptor bins should be normalized or not (better for illum variations)
    const bool m_bNormalizeBins;
    /// defines whether using Lienhart's lookup mask implementation instead of Chatfield's
    const bool m_bUsingLienhartMask;
    /// size of internal ssd patches (must be odd)
    const int m_nPatchSize;
    /// inner radius of the LSS descriptor (added to inner patch rings distance)
    const int m_nInnerRadius;
    /// outer radius of the LSS descriptor (maximum reach of outer patch ring)
    const int m_nOuterRadius;
    /// size of the LSS descriptor correlation window (deduced from outer radius and patch size)
    const int m_nCorrWinSize; // deduced
    /// size of the SSD correlation output patch
    const int m_nCorrPatchSize; // deduced
    /// radial bins count
    const int m_nRadialBins;
    /// angular bins count
    const int m_nAngularBins;
    /// static noise suppression level to use while binning
    const float m_fStaticNoiseVar;

private:
    /// keypoint-based description approach impl
    void ssdescs_impl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescriptors, bool bGenDescMap);
    /// dense description approach impl
    void ssdescs_impl(const cv::Mat& oImage, cv::Mat_<float>& oDescriptors);
    /// descriptor normalisation approach impl
    void ssdescs_norm(cv::Mat_<float>& oDescriptors) const;
    /// descriptor bin lookup map
    cv::Mat_<int> m_oDescLUMap;
    /// indices of first/last non-null map lookups
    int m_nFirstMaskIdx,m_nLastMaskIdx;
};