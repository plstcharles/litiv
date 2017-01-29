
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
//
// //////////////////////////////////////////////////////////////////////////
//
//               SLIC Superpixel Oversegmentation Algorithm
//       CUDA implementation of Achanta et al.'s method (TPAMI 2012)
//
// Note: requires CUDA compute architecture >= 3.0
// Author: Francois-Xavier Derue
// Contact: francois.xavier.derue@gmail.com
// Source: https://github.com/fderue/SLIC_CUDA
//
// Copyright (c) 2016 fderue
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "litiv/utils/opencv.hpp"
#include "litiv/utils/cuda.hpp"

/// SLIC superpixel segmentation algorithm
struct SLIC {

    /// algorithm initialization method type list
    enum InitType {
        SLIC_SIZE, ///< initialize with spx size
        SLIC_NSPX ///< initialize with spx count
    };

    SLIC();
    ~SLIC();

    /// set up the parameters and initalize all gpu buffer for faster video segmentation
    void initialize(const cv::Size& size, const int diamSpxOrNbSpx = 15, const InitType initType = SLIC_SIZE, const float wc = 35, const int nbIteration = 5);
    /// segment a frame in superpixel
    void segment(const cv::Mat& frame);
    /// returns computed superpixel labels for the previous frame
    inline const cv::Mat& getLabels() const {
        return m_oLabels;
    }
    /// discard orphan clusters (optional)
    int enforceConnectivity();
    /// returns a displayable version of the given input with overlying superpixels (cpu-side drawing)
    static cv::Mat displayBound(const cv::Mat& image, const cv::Mat& labels, const cv::Scalar& colour=cv::Scalar(255,0,0), const int& boundWidth = 1);

	/// returns a displayable version of the given input represented with RGB mean of superpixels (cpu-side drawing)
	static cv::Mat displayMean(const cv::Mat& image, const cv::Mat& labels);

protected:
    const int m_deviceId = 0;
    cudaDeviceProp m_deviceProp;
    int m_nbPx;
    int m_nbSpx;
    int m_SpxDiam;
    int m_SpxWidth, m_SpxHeight, m_SpxArea;
    int m_FrameWidth, m_FrameHeight;
    float m_wc;
    int m_nbIteration;
    InitType m_InitType;

    // cpu buffer
    cv::Mat_<float> m_oLabels;

    // gpu variable
    float* d_fClusters;
    float* d_fLabels;
    float* d_fAccAtt;

    // cudaArray
    cudaArray* cuArrayFrameBGRA;
    cudaArray* cuArrayFrameLab;
    cudaArray* cuArrayLabels;

    // texture and surface Object
    cudaTextureObject_t oTexFrameBGRA;
    cudaSurfaceObject_t oSurfFrameLab;
    cudaSurfaceObject_t oSurfLabels;

    /// assign the closest centroid to each pixel
    void assignment();
    /// update the clusters' centroids with the belonging pixels
    void update();
};