
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2017 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/features2d.hpp"

/// Mutual Information (MI) calculation helper (interface is similar to feature extractors, works only for 8U images)
class MutualInfo : public cv::Algorithm {
public:
    /// default constructor
    MutualInfo(const cv::Size& oWinSize=cv::Size(40,40), bool bUseDenseHist=false, bool bUse24BitPair=true);
    /// loads params from the specified file node @@@@ not impl
    virtual void read(const cv::FileNode&) override;
    /// writes params to the specified file storage @@@@ not impl
    virtual void write(cv::FileStorage&) const override;
    /// returns the minimum expected border size around a pixel (i.e. max dimension of win size)
    virtual int borderSize() const;
    /// returns the window size used with keypoint-based calls
    virtual const cv::Size& windowSize() const;
    /// returns the mutual information score for the given image pair (will use full matrices instead of subwindow)
    double compute(const cv::Mat& oImage1, const cv::Mat& oImage2);
    /// returns the mutual information scores for the given keypoints located in the image pair using subwindows of the size passed in constructor
    void compute(const cv::Mat& oImage1, const cv::Mat& oImage2, const std::vector<cv::KeyPoint>& voKeypoints, std::vector<double>& vdScores);
    /// returns the mutual information scores for the given keypoints located in the image pair using subwindows of the size passed in constructor (inline version)
    std::vector<double> compute(const cv::Mat& oImage1, const cv::Mat& oImage2, const std::vector<cv::KeyPoint>& voKeypoints);
    /// utility function, used to filter out bad keypoints that would trigger out of bounds error because they're too close to the image border
    void validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) const;
    /// utility function, used to filter out bad pixels in a ROI that would trigger out of bounds error because they're too close to the image border
    void validateROI(cv::Mat& oROI) const;

protected:
    /// defines the window size to use with keypoint-based calls (window will always be center on keypoint, and size should be odd)
    const cv::Size m_oWinSize;
    /// defines whether dense histograms will be used for joint probability extimation or not
    const bool m_bUseDenseHist;
    /// defines whether color images in color-grayscale pairs will be quantized to 16-bit YCbCr
    const bool m_bUse24BitPair;

private:
    // helper variables for internal impl (helps avoid continuous mem realloc)
    lv::JointSparseHistData<uchar,uchar> oSparseHistData;
    lv::JointSparseHistData<ushort,uchar> oSparse24BitHistData;
    lv::JointDenseHistData<uchar,uchar> oDenseHistData;
    lv::JointDenseHistData<ushort,uchar> oDense24BitHistData;
};