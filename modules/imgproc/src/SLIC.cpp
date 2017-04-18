
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

#include "litiv/imgproc/SLIC.hpp"
#include "SLIC.cuh"

inline int iDivUp(int a, int b) {
    return (a%b == 0) ? a / b : a / b + 1;
}

SLIC::SLIC() {
    int nbGpu = 0;
    cudaErrorCheck(cudaGetDeviceCount(&nbGpu));
    lvCout << "Detected " << nbGpu << " cuda-capable gpu(s)\n";
    cudaErrorCheck(cudaSetDevice(m_deviceId));
    cudaErrorCheck(cudaGetDeviceProperties(&m_deviceProp, m_deviceId));
    if (m_deviceProp.major < 3) {
        lvCerr_(-1) << "compute capability found = " << m_deviceProp.major << ", compute capability >= 3 required!\n";
        std::exit(EXIT_FAILURE);
    }
    lv::cuda::test(); // @@@ should succeed
}

SLIC::~SLIC() {
    cudaErrorCheck(cudaFree(d_fClusters));
    cudaErrorCheck(cudaFree(d_fAccAtt));
    cudaErrorCheck(cudaFreeArray(cuArrayFrameBGRA));
    cudaErrorCheck(cudaFreeArray(cuArrayFrameLab));
    cudaErrorCheck(cudaFreeArray(cuArrayLabels));
}

void SLIC::initialize(const cv::Size& size, const int diamSpxOrNbSpx , const InitType initType, const float wc , const int nbIteration ) {
    m_nbIteration = nbIteration;
    m_FrameWidth = size.width;
    m_FrameHeight = size.height;
    m_nbPx = m_FrameWidth*m_FrameHeight;
    m_InitType = initType;
    m_wc = wc;
    if (m_InitType == SLIC_NSPX){
        m_SpxDiam = diamSpxOrNbSpx;
        m_SpxDiam = (int)sqrt(m_nbPx / (float)diamSpxOrNbSpx);
    }
    else m_SpxDiam = diamSpxOrNbSpx;

    m_nbSpxPerRow = iDivUp(m_FrameWidth, m_SpxDiam);
    m_nbSpxPerCol = iDivUp(m_FrameHeight, m_SpxDiam);
    m_nbSpx = m_nbSpxPerRow*m_nbSpxPerCol;

    m_oLabels.create(m_FrameHeight,m_FrameWidth);

    cudaErrorCheck(cudaGetLastError());

    //allocate buffers on gpu
    const cudaChannelFormatDesc channelDescrBGRA = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    cudaErrorCheck(cudaMallocArray(&cuArrayFrameBGRA, &channelDescrBGRA, m_FrameWidth, m_FrameHeight));

    const cudaChannelFormatDesc channelDescrLab = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaErrorCheck(cudaMallocArray(&cuArrayFrameLab, &channelDescrLab, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));

    const cudaChannelFormatDesc channelDescrLabels = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaErrorCheck(cudaMallocArray(&cuArrayLabels, &channelDescrLabels, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));

    // Specify texture frameBGRA object parameters
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArrayFrameBGRA;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    cudaErrorCheck(cudaCreateTextureObject(&oTexFrameBGRA, &resDesc, &texDesc, NULL));

    // surface frameLab
    cudaResourceDesc rescDescFrameLab;
    memset(&rescDescFrameLab, 0, sizeof(rescDescFrameLab));
    rescDescFrameLab.resType = cudaResourceTypeArray;
    rescDescFrameLab.res.array.array = cuArrayFrameLab;
    cudaErrorCheck(cudaCreateSurfaceObject(&oSurfFrameLab, &rescDescFrameLab));

    // surface labels
    cudaResourceDesc resDescLabels;
    memset(&resDescLabels, 0, sizeof(resDescLabels));
    resDescLabels.resType = cudaResourceTypeArray;
    resDescLabels.res.array.array = cuArrayLabels;
    cudaErrorCheck(cudaCreateSurfaceObject(&oSurfLabels, &resDescLabels));

    // buffers clusters , accAtt
    cudaErrorCheck(cudaMalloc((void**)&d_fClusters, m_nbSpx*sizeof(float) * 5)); // 5-D centroid
    cudaErrorCheck(cudaMemset(d_fClusters, 0, m_nbSpx*sizeof(float) * 5));
    cudaErrorCheck(cudaMalloc((void**)&d_fAccAtt, m_nbSpx*sizeof(float) * 6)); // 5-D centroid acc + 1 counter
    cudaErrorCheck(cudaMemset(d_fAccAtt, 0, m_nbSpx*sizeof(float) * 6));
}

void SLIC::segment(const cv::Mat& frameBGR) {
    cv::Mat frameBGRA;
    cv::cvtColor(frameBGR, frameBGRA, CV_BGR2BGRA);
    CV_Assert(frameBGRA.type() == CV_8UC4);
    CV_Assert(frameBGRA.isContinuous());
    cudaErrorCheck(cudaMemcpyToArray(cuArrayFrameBGRA, 0, 0, (uchar*)frameBGRA.data, m_nbPx* sizeof(uchar4), cudaMemcpyHostToDevice));
    {
        const int blockW = 16;
        const int blockH = blockW;
        CV_Assert(blockW*blockH <= m_deviceProp.maxThreadsPerBlock);
        const lv::cuda::KernelParams oParams(dim3(iDivUp(m_FrameWidth,blockW),iDivUp(m_FrameHeight,blockH)),dim3(blockW,blockH));
        host::kRgb2CIELab(oParams,oTexFrameBGRA,oSurfFrameLab,m_FrameWidth,m_FrameHeight);
    }
    {
        const int blockW = 16;
        const lv::cuda::KernelParams oParams(dim3(iDivUp(m_nbSpx, blockW)),dim3(blockW));
        host::kInitClusters(oParams,oSurfFrameLab,d_fClusters,m_FrameWidth,m_FrameHeight,m_nbSpxPerRow,m_nbSpxPerCol,m_SpxDiam/2.f);
    }
    for (int i = 0; i<m_nbIteration; i++) {
        assignment();
        cudaDeviceSynchronize();
        update();
        cudaDeviceSynchronize();
    }
    cudaErrorCheck(cudaMemcpyFromArray(m_oLabels.data,cuArrayLabels,0,0,m_oLabels.total()*m_oLabels.elemSize(),cudaMemcpyDeviceToHost));
}

void SLIC::assignment() {
    const int nbBlockPerClust = iDivUp(m_SpxDiam*m_SpxDiam, m_deviceProp.maxThreadsPerBlock);
    const dim3 gridSize(m_nbSpxPerRow, m_nbSpxPerCol,nbBlockPerClust);

    const int hMax = m_deviceProp.maxThreadsPerBlock / m_SpxDiam;
    const dim3 blockSize(m_SpxDiam, std::min(m_SpxDiam, hMax));

    CV_Assert(blockSize.x >= 3 && blockSize.y >= 3);
    const float wc2 = m_wc * m_wc;
    host::kAssignment(lv::cuda::KernelParams(gridSize, blockSize),
        oSurfFrameLab,
        d_fClusters,
        m_FrameWidth,
        m_FrameHeight,
        m_nbSpxPerRow,
        m_nbSpx,
        m_SpxDiam,
        wc2,
        oSurfLabels,
        d_fAccAtt);
}

void SLIC::update() {
    host::kUpdate(lv::cuda::KernelParams(dim3(iDivUp(m_nbSpx, m_deviceProp.maxThreadsPerBlock)),dim3(m_deviceProp.maxThreadsPerBlock)),m_nbSpx,d_fClusters,d_fAccAtt);
}

int SLIC::enforceConnectivity() {
    int label = 0, adjlabel = 0;
    int lims = (m_FrameWidth * m_FrameHeight) / (m_nbSpx);
    lims = lims >> 2;
    const int dx4[4] = { -1, 0, 1, 0 };
    const int dy4[4] = { 0, -1, 0, 1 };
    std::vector<std::vector<int>>newLabels;
    for (int i = 0; i < m_FrameHeight; i++) {
        std::vector<int> nv(m_FrameWidth,-1);
        newLabels.push_back(nv);
    }
    for (int i = 0; i < m_FrameHeight; i++) {
        for (int j = 0; j < m_FrameWidth; j++) {
            if (newLabels[i][j] == -1){
                std::vector<cv::Point> elements;
                elements.push_back(cv::Point(j,i));
                for (int k = 0; k < 4; k++){
                    int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
                    if (x >= 0 && x < m_FrameWidth && y >= 0 && y < m_FrameHeight){
                        if (newLabels[y][x] >= 0){
                            adjlabel = newLabels[y][x];
                        }
                    }
                }
                int count = 1;
                for (int c = 0; c < count; c++) {
                    for (int k = 0; k < 4; k++) {
                        int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
                        if (x >= 0 && x < m_FrameWidth && y >= 0 && y < m_FrameHeight){
                            if (newLabels[y][x] == -1 && m_oLabels(i,j) == m_oLabels(y,x)) {
                                elements.push_back(cv::Point(x, y));
                                newLabels[y][x] = label;//m_labels[i][j];
                                count += 1;
                            }
                        }
                    }
                }
                if (count <= lims) {
                    for (int c = 0; c < count; c++) {
                        newLabels[elements[c].y][elements[c].x] = adjlabel;
                    }
                    label -= 1;
                }
                label += 1;
            }
        }
    }
    int nbSpxNoOrphan = label; // new number of spx
    for (int i = 0; i < (int)newLabels.size(); i++)
        for (int j = 0; j < (int)newLabels[i].size(); j++)
            m_oLabels(i,j) = (float)newLabels[i][j];

    return nbSpxNoOrphan;
}

cv::Mat SLIC::displayMean(const cv::Mat& image, const cv::Mat& labels) {
    CV_Assert(labels.type() == CV_32FC1);
    cv::Mat outImage(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    double minVal, maxVal;
    cv::minMaxLoc(labels, &minVal, &maxVal);

    std::vector<std::vector<cv::Point>> vLoc((int)maxVal + 1);
    std::vector<std::vector<cv::Vec3b>> vColor((int)maxVal + 1);
    for (int i = 0; i < image.rows; i++) {
        const cv::Vec3b* pImage = image.ptr<cv::Vec3b>(i);
        const float* pLabels = labels.ptr<float>(i);

        for (int j = 0; j < image.cols; j++) {
            int label = (int)pLabels[j];
            if (label >= 0) {
                vLoc[label].push_back(cv::Point(j, i));
                vColor[label].push_back(pImage[j]);
            }

        }
    }

    for (int i = 0; i < (int)vLoc.size(); i++) {
        if (!vColor[i].empty()) {
            cv::Vec3f meanColor(0, 0, 0);
            for (int j = 0; j < (int)vColor[i].size(); j++) {
                meanColor += vColor[i][j];
            }
            for (int j = 0; j<3; j++) meanColor[j] /= vColor[i].size();
            for (int j = 0; j < (int)vLoc[i].size(); j++) {
                outImage.at<cv::Vec3b>(vLoc[i][j]) = cv::Vec3b(meanColor);
            }
        }
    }
    return outImage;
}

cv::Mat SLIC::displayBound(const cv::Mat& image, const cv::Mat& labels, const cv::Scalar& colour, const int& boundWidth) {
    CV_Assert(image.size() == labels.size());
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_8UC3);
    CV_Assert(labels.type() == CV_32FC1);
    cv::Mat_<uchar> mask(image.size(), uchar(0));
    cv::Mat output = image.clone();
    cv::Mat segMask(image.size(), CV_8UC1, cv::Scalar(0));
    if (output.channels() == 1)
        cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);
    const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
    for (int i = 0; i<image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            for (int k = 0; k < 8; k++) {
                const int x = j + dx8[k], y = i + dy8[k];
                if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
                    if (labels.at<float>(i, j) != labels.at<float>(y, x) && mask(y, x)<2) {
                        ++mask(i, j);
                    }
                }
            }
            if (mask(i, j) >= 2) {
                segMask.at<uchar>(i, j) = 255;
            }

        }
    }
    if (boundWidth > 1) {
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(boundWidth, boundWidth), cv::Point(1, 1));
        cv::morphologyEx(segMask, segMask, cv::MORPH_DILATE, element);
    }
    cv::Mat colorMask(image.size(), CV_8UC3, colour);
    colorMask.copyTo(output, segMask);
    return output;
}