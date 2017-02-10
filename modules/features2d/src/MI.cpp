
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

#if __cplusplus>=201402L

#define HIST_QUANTIF_FACTOR 1
#define USE_FAST_NUM_APPROX false
#define SKIP_MINMAX_HIST    false

#include "litiv/features2d/MI.hpp"

MutualInfo::MutualInfo(const cv::Size& oWinSize, bool bUseDenseHist, bool bUse24BitPair) :
        m_oWinSize(oWinSize),
        m_bUseDenseHist(bUseDenseHist),
        m_bUse24BitPair(bUse24BitPair) {
    lvAssert_(m_oWinSize.area()>0 && (m_oWinSize.height%2)==1 && (m_oWinSize.width%2)==1,"invalid parameter(s)");
}

void MutualInfo::read(const cv::FileNode& /*fn*/) {
    // ... = fn["..."];
}

void MutualInfo::write(cv::FileStorage& /*fs*/) const {
    //fs << "..." << ...;
}

cv::Size MutualInfo::windowSize() const {
    return m_oWinSize;
}

int MutualInfo::borderSize(int nDim) const {
    lvAssert(nDim==0 || nDim==1);
    return (nDim==0?m_oWinSize.width:m_oWinSize.height)/2;
}

double MutualInfo::compute(const cv::Mat& _oImage1, const cv::Mat& _oImage2) {
    lvAssert_(_oImage1.rows>=m_oWinSize.height && _oImage1.cols>=m_oWinSize.width && _oImage1.size()==_oImage2.size(),"invalid input image(s) size");
    if(m_bUse24BitPair && ((_oImage1.type()==CV_8UC3 && _oImage2.type()==CV_8UC1) || (_oImage2.type()==CV_8UC3 && _oImage1.type()==CV_8UC1))) {
        const cv::Mat_<ushort> oImage1 = cv::cvtBGRToPackedYCbCr(_oImage1.type()==CV_8UC3?_oImage1:_oImage2);
        const cv::Mat_<uchar> oImage2 = _oImage1.type()==CV_8UC3?_oImage2:_oImage1;
        if(m_bUseDenseHist)
            return lv::calcMutualInfo<HIST_QUANTIF_FACTOR,false,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1,oImage2,&oDense24BitHistData);
        else
            return lv::calcMutualInfo<HIST_QUANTIF_FACTOR,true,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1,oImage2,&oSparse24BitHistData);
    }
    else if(_oImage1.type()==CV_8UC1 && _oImage2.type()==CV_8UC1) {
        const cv::Mat_<uchar> oImage1 = _oImage1;
        const cv::Mat_<uchar> oImage2 = _oImage2;
        if(m_bUseDenseHist)
            return lv::calcMutualInfo<HIST_QUANTIF_FACTOR,false,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1,oImage2,&oDenseHistData);
        else
            return lv::calcMutualInfo<HIST_QUANTIF_FACTOR,true,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1,oImage2,&oSparseHistData);
    }
    else
        lvError("unsupported input matrices types (need 8uc1 on both, or 8uc1+8uc3 if using 24bit pair)");
}

void MutualInfo::compute(const cv::Mat& _oImage1, const cv::Mat& _oImage2, const std::vector<cv::KeyPoint>& voKeypoints, std::vector<double>& vdScores) {
    lvAssert_(_oImage1.rows>=m_oWinSize.height && _oImage1.cols>=m_oWinSize.width && _oImage1.size()==_oImage2.size(),"invalid input image(s) size");
    vdScores.resize(voKeypoints.size());
    if(m_bUse24BitPair && ((_oImage1.type()==CV_8UC3 && _oImage2.type()==CV_8UC1) || (_oImage2.type()==CV_8UC3 && _oImage1.type()==CV_8UC1))) {
        const cv::Mat_<ushort> oImage1 = cv::cvtBGRToPackedYCbCr(_oImage1.type()==CV_8UC3?_oImage1:_oImage2);
        const cv::Mat_<uchar> oImage2 = _oImage1.type()==CV_8UC3?_oImage2:_oImage1;
        const cv::Rect oSourceRect(0,0,oImage1.cols,oImage1.rows);
        for(size_t nKPIdx=0; nKPIdx<voKeypoints.size(); ++nKPIdx) {
            const cv::Point oTargetPt(int(std::round(voKeypoints[nKPIdx].pt.x)),int(std::round(voKeypoints[nKPIdx].pt.y)));
            const cv::Rect oTargetRect(oTargetPt.x-m_oWinSize.width/2,oTargetPt.y-m_oWinSize.height/2,m_oWinSize.width,m_oWinSize.height);
            lvAssert_(oSourceRect.contains(oTargetRect.tl()) && oSourceRect.contains(oTargetRect.br()-cv::Point2i(1,1)),"got invalid input keypoint (oob)");
            if(m_bUseDenseHist)
                vdScores[nKPIdx] = lv::calcMutualInfo<HIST_QUANTIF_FACTOR,false,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1(oTargetRect),oImage2(oTargetRect),&oDense24BitHistData);
            else
                vdScores[nKPIdx] = lv::calcMutualInfo<HIST_QUANTIF_FACTOR,true,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1(oTargetRect),oImage2(oTargetRect),&oSparse24BitHistData);
        }
    }
    else if(_oImage1.type()==CV_8UC1 && _oImage2.type()==CV_8UC1) {
        const cv::Mat_<uchar> oImage1 = _oImage1;
        const cv::Mat_<uchar> oImage2 = _oImage2;
        const cv::Rect oSourceRect(0,0,oImage1.cols,oImage1.rows);
        for(size_t nKPIdx=0; nKPIdx<voKeypoints.size(); ++nKPIdx) {
            const cv::Point oTargetPt(int(std::round(voKeypoints[nKPIdx].pt.x)),int(std::round(voKeypoints[nKPIdx].pt.y)));
            const cv::Rect oTargetRect(oTargetPt.x-m_oWinSize.width/2,oTargetPt.y-m_oWinSize.height/2,m_oWinSize.width,m_oWinSize.height);
            lvAssert_(oSourceRect.contains(oTargetRect.tl()) && oSourceRect.contains(oTargetRect.br()-cv::Point2i(1,1)),"got invalid input keypoint (oob)");
            if(m_bUseDenseHist)
                vdScores[nKPIdx] = lv::calcMutualInfo<HIST_QUANTIF_FACTOR,false,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1(oTargetRect),oImage2(oTargetRect),&oDenseHistData);
            else
                vdScores[nKPIdx] = lv::calcMutualInfo<HIST_QUANTIF_FACTOR,true,USE_FAST_NUM_APPROX,SKIP_MINMAX_HIST>(oImage1(oTargetRect),oImage2(oTargetRect),&oSparseHistData);
        }
    }
    else
        lvError("unsupported input matrices types (need 8uc1 on both, or 8uc1+8uc3 if using 24bit pair)");
}

std::vector<double> MutualInfo::compute(const cv::Mat& oImage1, const cv::Mat& oImage2, const std::vector<cv::KeyPoint>& voKeypoints) {
    std::vector<double> vdScores;
    compute(oImage1,oImage2,voKeypoints,vdScores);
    return vdScores;
}

void MutualInfo::validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) const {
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImgSize,std::max(m_oWinSize.width,m_oWinSize.height));
}

void MutualInfo::validateROI(cv::Mat& oROI) const {
    lvAssert_(!oROI.empty() && oROI.type()==CV_8UC1,"input ROI must be non-empty and of type 8UC1");
    cv::Mat oROI_new(oROI.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    const cv::Rect nROI_inner(m_oWinSize.width/2,m_oWinSize.height/2,oROI.cols-m_oWinSize.width,oROI.rows-m_oWinSize.height);
    cv::Mat(oROI,nROI_inner).copyTo(cv::Mat(oROI_new,nROI_inner));
    oROI = oROI_new;
}

#endif //__cplusplus>=201402L