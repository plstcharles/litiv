
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2018 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/utils/algo.hpp"

/// image warping algorithm based on Mean Least Squares, inspired by imgwarp-opencv
struct ImageWarper : public cv::Algorithm {

    /// warp modes for the mapping
    enum WarpModes {
        RIGID,
        SIMILARITY,
        CUSTOM
    };

    /// default algo constructor; initialize will need to be called before warping can be done
    ImageWarper();
    /// full algo constructor which also initializes the internal transformation model
    ImageWarper(const std::vector<cv::Point2d>& vSourcePts, const cv::Size& oSourceSize,
                const std::vector<cv::Point2d>& vDestPts, const cv::Size& oDestSize,
                int nGridSize=5, WarpModes eMode=RIGID);
    /// set up the transformation model parameters to allow warping --- the ratio dictates the warp strength (1=full warp, 0=none)
    void initialize(const std::vector<cv::Point2d>& vSourcePts, const cv::Size& oSourceSize,
                    const std::vector<cv::Point2d>& vDestPts, const cv::Size& oDestSize,
                    int nGridSize=5, WarpModes eMode=RIGID);
    /// computes the warp result for an input image, given the current model paramters, and the warp strength ratio (1=full warp, 0=none)
    void warp(const cv::Mat& oInput, cv::Mat& oOutput, double dRatio=1.0);
    /// required for derived class destruction from this interface
    virtual ~ImageWarper() = default;
    /// returns whether the algo is initialized or not
    bool isInitialized() const {return m_bInitialized;}
protected:
    /// computes the internal transformation used in the warping step
    virtual bool computeTransform();
    bool m_bInitialized;
    int m_nGridSize;
    WarpModes m_eWarpMode;
    cv::Size m_oSourceSize,m_oDestSize;
    std::vector<cv::Point2d> m_vSourcePts,m_vDestPts;
    cv::Mat_<double> m_oDeltaX,m_oDeltaY;
};



