
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

#include "litiv/imgproc/imwarp.hpp"

#define USE_RIGID_PRESCALE 0
#define DIST_EXP_ALPHA 1.0

template<typename TValue>
inline double calcArea(const std::vector<cv::Point_<TValue>>& vPts) {
    if(vPts.empty())
        return 0;
    cv::Point_<TValue> vTopLeft=vPts[0], vBottomRight=vPts[0];
    for(const auto& oPt : vPts) {
        vTopLeft.x = std::min(vTopLeft.x,oPt.x);
        vTopLeft.y = std::min(vTopLeft.y,oPt.y);
        vBottomRight.x = std::max(vBottomRight.x,oPt.x);
        vBottomRight.y = std::max(vBottomRight.y,oPt.y);
    }
    return (double)cv::Rect_<TValue>(vTopLeft,vBottomRight).area();
}

template<typename TValue>
inline TValue interp(double x, double y, TValue v11, TValue v12, TValue v21, TValue v22) {
    return TValue((v11*(1.0-y)+v12*y)*(1.0-x) + (v21*(1.0-y)+v22*y)*x);
}

ImageWarper::ImageWarper() : m_bInitialized(false) {}

ImageWarper::ImageWarper(const std::vector<cv::Point2d>& vSourcePts, const cv::Size& oSourceSize,
                         const std::vector<cv::Point2d>& vDestPts, const cv::Size& oDestSize,
                         int nGridSize, WarpModes eMode) :
        m_bInitialized(false) {
    lvDbgExceptionWatch;
    initialize(vSourcePts,oSourceSize,vDestPts,oDestSize,nGridSize,eMode);
}

void ImageWarper::initialize(const std::vector<cv::Point2d>& vSourcePts, const cv::Size& oSourceSize,
                             const std::vector<cv::Point2d>& vDestPts, const cv::Size& oDestSize,
                             int nGridSize, WarpModes eMode) {
    lvDbgExceptionWatch;
    lvAssert_(nGridSize>0,"grid size must be strictly positive");
    lvAssert_(!vSourcePts.empty() && !vDestPts.empty(),"provided point vectors must not be empty");
    lvAssert_(vSourcePts.size()==vDestPts.size(),"source/dest point count mismatch");
    lvAssert_(oSourceSize.area()>0 && oDestSize.area()>0,"image sizes must be strictly positive");
    m_bInitialized = false;
    m_nGridSize = nGridSize;
    m_eWarpMode = eMode;
    m_oSourceSize = oSourceSize;
    m_oDestSize = oDestSize;
    m_vSourcePts = vSourcePts;
    m_vDestPts = vDestPts;
    m_bInitialized = computeTransform();
}

void ImageWarper::warp(const cv::Mat& oInput, cv::Mat& oOutput, double dRatio) {
    lvDbgExceptionWatch;
    lvAssert_(m_bInitialized,"transformation model must be initialized first!");
    lvAssert_(!oInput.empty() && oInput.size()==m_oSourceSize,"bad input image size");
    lvAssert_(oInput.depth()==CV_8U,"implementation only supports 8u mats for now");
    lvAssert_(oInput.isContinuous(),"input matrix data must be a continuous block");
    lvAssert_(!m_oDeltaX.empty() && !m_oDeltaY.empty(),"initialize failed");
    oOutput.create(m_oDestSize,oInput.type());
    const int nChannels = oInput.channels();
    for(int nRowIdx=0; nRowIdx<m_oDestSize.height; nRowIdx+=m_nGridSize) {
        for(int nColIdx=0; nColIdx<m_oDestSize.width; nColIdx+=m_nGridSize) {
            int nNextRowIdx = nRowIdx+m_nGridSize;
            int nNextColIdx = nColIdx+m_nGridSize;
            int nCellHeight = m_nGridSize;
            int nCellWidth = m_nGridSize;
            if(nNextRowIdx>=m_oDestSize.height) {
                nNextRowIdx = m_oDestSize.height - 1;
                nCellHeight = nNextRowIdx - nRowIdx + 1;
            }
            if(nNextColIdx>=m_oDestSize.width) {
                nNextColIdx = m_oDestSize.width - 1;
                nCellWidth = nNextColIdx - nColIdx + 1;
            }
            for(int nCellRowIdx=0; nCellRowIdx<nCellHeight; ++nCellRowIdx) {
                for(int nCellColIdx=0; nCellColIdx<nCellWidth; ++nCellColIdx) {
                    const double dCellX = double(nCellRowIdx)/nCellHeight;
                    const double dCellY = double(nCellColIdx)/nCellWidth;
                    const double dDeltaX = interp(dCellX,dCellY,m_oDeltaX(nRowIdx,nColIdx),m_oDeltaX(nRowIdx,nNextColIdx),m_oDeltaX(nNextRowIdx,nColIdx),m_oDeltaX(nNextRowIdx,nNextColIdx));
                    const double dDeltaY = interp(dCellX,dCellY,m_oDeltaY(nRowIdx,nColIdx),m_oDeltaY(nRowIdx,nNextColIdx),m_oDeltaY(nNextRowIdx,nColIdx),m_oDeltaY(nNextRowIdx,nNextColIdx));
                    const double dOffsetColIdx = std::max(std::min(nColIdx+nCellColIdx+dDeltaX*dRatio,m_oSourceSize.width-1.0),0.0);
                    const double dOffsetRowIdx = std::max(std::min(nRowIdx+nCellRowIdx+dDeltaY*dRatio,m_oSourceSize.height-1.0),0.0);
                    const int nInputRowIdxLow = (int)dOffsetRowIdx;
                    const int nInputColIdxLow = (int)dOffsetColIdx;
                    const int nInputRowIdxHigh = (int)std::ceil(dOffsetRowIdx);
                    const int nInputColIdxHigh = (int)std::ceil(dOffsetColIdx);
                    for(int nChIdx=0; nChIdx<nChannels; ++nChIdx)
                        oOutput.ptr<uchar>(nRowIdx+nCellRowIdx,nColIdx+nCellColIdx)[nChIdx] =
                            interp(
                                dOffsetRowIdx-nInputRowIdxLow,
                                dOffsetColIdx-nInputColIdxLow,
                                oInput.ptr<uchar>(nInputRowIdxLow,nInputColIdxLow)[nChIdx],
                                oInput.ptr<uchar>(nInputRowIdxLow,nInputColIdxHigh)[nChIdx],
                                oInput.ptr<uchar>(nInputRowIdxHigh,nInputColIdxLow)[nChIdx],
                                oInput.ptr<uchar>(nInputRowIdxHigh,nInputColIdxHigh)[nChIdx]
                            );
                }
            }

        }
    }
}

bool ImageWarper::computeTransform() {
    lvAssert_(m_eWarpMode==RIGID || m_eWarpMode==SIMILARITY,"unknown warp mode (override failed?)");
    lvDbgAssert_(!m_vSourcePts.empty() && !m_vDestPts.empty(),"provided point vectors must not be empty");
    lvDbgAssert_(m_vSourcePts.size()==m_vDestPts.size(),"source/dest point count mismatch");
    lvDbgAssert_(m_oSourceSize.area()>0 && m_oDestSize.area()>0,"image sizes must be strictly positive");
    const size_t nPtCount = m_vDestPts.size();
    std::vector<double> vL2SqrDists(nPtCount);
    if(m_eWarpMode==RIGID) {
        const double dAlpha = DIST_EXP_ALPHA;
    #if USE_RIGID_PRESCALE
        const double dAreaRatio = sqrt(calcArea(m_vSourcePts)/calcArea(m_vDestPts));
        for(auto& vPt : m_vSourcePts)
            vPt *= 1.0/dAreaRatio;
    #endif //USE_RIGID_PRESCALE
        m_oDeltaX.create(m_oDestSize);
        m_oDeltaY.create(m_oDestSize);
        if(nPtCount<2u) {
            m_oDeltaX.setTo(0);
            m_oDeltaY.setTo(0);
            return true;
        }
        for(int nColIdx=0;; nColIdx+=m_nGridSize) {
            if(nColIdx>=m_oDestSize.width && nColIdx<m_oDestSize.width+m_nGridSize-1)
                nColIdx = m_oDestSize.width-1;
            else if(nColIdx>=m_oDestSize.width)
                break;
            for(int nRowIdx=0;; nRowIdx+=m_nGridSize) {
                if(nRowIdx>=m_oDestSize.height && nRowIdx<m_oDestSize.height+m_nGridSize-1)
                    nRowIdx = m_oDestSize.height-1;
                else if(nRowIdx>=m_oDestSize.height)
                    break;
                double dInvDistSum = 0;
                cv::Point2d vDestPtDistSum(0,0),vSourcePtDistSum(0,0);
                cv::Point2d vCurrPt(nColIdx,nRowIdx);
                size_t nPtIdx = 0u;
                while(nPtIdx<nPtCount) {
                    if((nColIdx==m_vDestPts[nPtIdx].x) && nRowIdx==m_vDestPts[nPtIdx].y)
                        break;
                    const double dL2SqrDist_raw = (nColIdx-m_vDestPts[nPtIdx].x)*(nColIdx-m_vDestPts[nPtIdx].x) + (nRowIdx-m_vDestPts[nPtIdx].y)*(nRowIdx-m_vDestPts[nPtIdx].y);
                    if(dAlpha==1.0)
                        vL2SqrDists[nPtIdx] = 1.0/dL2SqrDist_raw;
                    else
                        vL2SqrDists[nPtIdx] = std::pow(dL2SqrDist_raw,-dAlpha);
                    dInvDistSum += vL2SqrDists[nPtIdx];
                    vDestPtDistSum +=  vL2SqrDists[nPtIdx]*m_vDestPts[nPtIdx];
                    vSourcePtDistSum +=  vL2SqrDists[nPtIdx]*m_vSourcePts[nPtIdx];
                    ++nPtIdx;
                }
                cv::Point2d oNewPoint(0,0);
                if(nPtIdx!=nPtCount)
                    oNewPoint = m_vSourcePts[nPtIdx];
                else {
                    const double dDistSum = 1.0/dInvDistSum;
                    const cv::Point2d vWgDestPt = dDistSum*vDestPtDistSum;
                    const cv::Point2d vWgSourcePt = dDistSum*vSourcePtDistSum;
                    double s1 = 0, s2 = 0;
                    for(nPtIdx=0u; nPtIdx<nPtCount; ++nPtIdx) {
                        if(nColIdx==m_vDestPts[nPtIdx].x && nRowIdx==m_vDestPts[nPtIdx].y)
                            continue;
                        const cv::Point2d vPtI = m_vDestPts[nPtIdx]-vWgDestPt;
                        const cv::Point2d vPtI_R(-vPtI.y,vPtI.x);
                        const cv::Point2d vPtJ = m_vSourcePts[nPtIdx]-vWgSourcePt;
                        s1 += vL2SqrDists[nPtIdx]*vPtJ.dot(vPtI);
                        s2 += vL2SqrDists[nPtIdx]*vPtJ.dot(vPtI_R);
                    }
                    const double dMIU = sqrt(s1*s1+s2*s2);
                    vCurrPt -= vWgDestPt;
                    const cv::Point2d vCurrPt_R(-vCurrPt.y,vCurrPt.x);
                    for(nPtIdx=0u; nPtIdx<nPtCount; ++nPtIdx) {
                        if(nColIdx==m_vDestPts[nPtIdx].x && nRowIdx==m_vDestPts[nPtIdx].y)
                            continue;
                        const cv::Point2d vPtI = m_vDestPts[nPtIdx]-vWgDestPt;
                        const cv::Point2d vPtI_R(-vPtI.y,vPtI.x);
                        cv::Point2d vTmpPt(
                            vPtI.dot(vCurrPt)*m_vSourcePts[nPtIdx].x - vPtI_R.dot(vCurrPt)*m_vSourcePts[nPtIdx].y,
                            -vPtI.dot(vCurrPt_R)*m_vSourcePts[nPtIdx].x + vPtI_R.dot(vCurrPt_R)*m_vSourcePts[nPtIdx].y);
                        vTmpPt *= vL2SqrDists[nPtIdx]/dMIU;
                        oNewPoint += vTmpPt;
                    }
                    oNewPoint += vWgSourcePt;
                }
            #if USE_RIGID_PRESCALE
                m_oDeltaX(nRowIdx,nColIdx) = oNewPoint.x*dAreaRatio-nColIdx;
                m_oDeltaY(nRowIdx,nColIdx) = oNewPoint.y*dAreaRatio-nRowIdx;
            #else //!USE_RIGID_PRESCALE
                m_oDeltaX(nRowIdx,nColIdx) = oNewPoint.x-nColIdx;
                m_oDeltaY(nRowIdx,nColIdx) = oNewPoint.y-nRowIdx;
            #endif //!USE_RIGID_PRESCALE
            }
        }
    #if USE_RIGID_PRESCALE
        for(auto& vPt : m_vSourcePts)
            vPt *= dAreaRatio;
    #endif //USE_RIGID_PRESCALE
    }
    else if(m_eWarpMode==SIMILARITY) {
        m_oDeltaX.create(m_oDestSize);
        m_oDeltaY.create(m_oDestSize);
        if(nPtCount<2u) {
            m_oDeltaX.setTo(0);
            m_oDeltaY.setTo(0);
            return true;
        }
        for(int nColIdx=0;; nColIdx+=m_nGridSize) {
            if(nColIdx>=m_oDestSize.width && nColIdx<m_oDestSize.width+m_nGridSize-1)
                nColIdx = m_oDestSize.width-1;
            else if(nColIdx>=m_oDestSize.width)
                break;
            for(int nRowIdx=0;; nRowIdx+=m_nGridSize) {
                if(nRowIdx>=m_oDestSize.height && nRowIdx<m_oDestSize.height+m_nGridSize-1)
                    nRowIdx = m_oDestSize.height-1;
                else if(nRowIdx>=m_oDestSize.height)
                    break;
                double dInvDistSum = 0;
                cv::Point2d vDestPtDistSum(0,0),vSourcePtDistSum(0,0);
                cv::Point2d vCurrPt(nColIdx,nRowIdx);
                size_t nPtIdx = 0u;
                while(nPtIdx<nPtCount) {
                    if((nColIdx==m_vDestPts[nPtIdx].x) && nRowIdx==m_vDestPts[nPtIdx].y)
                        break;
                    const double dL2SqrDist_raw = (nColIdx-m_vDestPts[nPtIdx].x)*(nColIdx-m_vDestPts[nPtIdx].x) + (nRowIdx-m_vDestPts[nPtIdx].y)*(nRowIdx-m_vDestPts[nPtIdx].y);
                    vL2SqrDists[nPtIdx] = 1.0/dL2SqrDist_raw;
                    dInvDistSum += vL2SqrDists[nPtIdx];
                    vDestPtDistSum += vL2SqrDists[nPtIdx]*m_vDestPts[nPtIdx];
                    vSourcePtDistSum += vL2SqrDists[nPtIdx]*m_vSourcePts[nPtIdx];
                    ++nPtIdx;
                }
                cv::Point2d oNewPoint(0,0);
                if(nPtIdx!=nPtCount)
                    oNewPoint = m_vSourcePts[nPtIdx];
                else {
                    const double dDistSum = 1.0/dInvDistSum;
                    const cv::Point2d vWgDestPt = dDistSum*vDestPtDistSum;
                    const cv::Point2d vWgSourcePt = dDistSum*vSourcePtDistSum;
                    double dMIU = 0;
                    for(nPtIdx=0u; nPtIdx<nPtCount; ++nPtIdx) {
                        if(nColIdx==m_vDestPts[nPtIdx].x && nRowIdx==m_vDestPts[nPtIdx].y)
                            continue;
                        const cv::Point2d vPtI = m_vDestPts[nPtIdx]-vWgDestPt;
                        dMIU += vL2SqrDists[nPtIdx] * vPtI.dot(vPtI);
                    }
                    vCurrPt -= vWgDestPt;
                    const cv::Point2d vCurrPt_R(-vCurrPt.y,vCurrPt.x);
                    for(nPtIdx=0u; nPtIdx<nPtCount; ++nPtIdx) {
                        if(nColIdx==m_vDestPts[nPtIdx].x && nRowIdx==m_vDestPts[nPtIdx].y)
                            continue;
                        const cv::Point2d vPtI = m_vDestPts[nPtIdx]-vWgDestPt;
                        const cv::Point2d vPtI_R(-vPtI.y,vPtI.x);
                        cv::Point2d vTmpPt(
                            vPtI.dot(vCurrPt)*m_vSourcePts[nPtIdx].x - vPtI_R.dot(vCurrPt)*m_vSourcePts[nPtIdx].y,
                            -vPtI.dot(vCurrPt_R)*m_vSourcePts[nPtIdx].x + vPtI_R.dot(vCurrPt_R)*m_vSourcePts[nPtIdx].y);
                        vTmpPt *= vL2SqrDists[nPtIdx]/dMIU;
                        oNewPoint += vTmpPt;
                    }
                    oNewPoint += vWgSourcePt;
                }
                m_oDeltaX(nRowIdx,nColIdx) = oNewPoint.x-nColIdx;
                m_oDeltaY(nRowIdx,nColIdx) = oNewPoint.y-nRowIdx;
            }
        }
    }
    return true;
}