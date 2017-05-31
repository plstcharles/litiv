
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

// note: this implementation is inspired by OpenCV's SCD class (see shape module)

#include "litiv/features2d.hpp"

#define USE_LIENHART_LOOKUP_MASK 1

ShapeContext::ShapeContext(size_t nInnerRadius, size_t nOuterRadius, size_t nAngularBins,
                           size_t nRadialBins, bool bRotationInvariant, bool bNormalizeBins) :
        m_nAngularBins((int)nAngularBins),
        m_nRadialBins((int)nRadialBins),
        m_nDescSize(m_nRadialBins*m_nAngularBins),
        m_nInnerRadius((int)nInnerRadius),
        m_nOuterRadius((int)nOuterRadius),
        m_dInnerRadius(0.0),
        m_dOuterRadius(0.0),
        m_bUseRelativeSpace(false),
        m_bRotationInvariant(bRotationInvariant),
        m_bNormalizeBins(bNormalizeBins) {
    lvAssert_(m_nAngularBins>0,"invalid parameter");
    lvAssert_(m_nRadialBins>0,"invalid parameter");
    lvAssert_(m_nInnerRadius>0,"invalid parameter");
    lvAssert_(m_nOuterRadius>0 && m_nInnerRadius<m_nOuterRadius,"invalid parameter");
    scdesc_generate_angmask();
    scdesc_generate_radmask();
    scdesc_generate_emdmask();
    lvAssert((int)m_vAngularLimits.size()==m_nAngularBins && (int)m_vRadialLimits.size()==m_nRadialBins);
    lvAssert(m_oEMDCostMap.dims==2 && m_oEMDCostMap.rows==m_nRadialBins*m_nAngularBins && m_oEMDCostMap.cols==m_nRadialBins*m_nAngularBins);
    if(!m_bRotationInvariant) {
        const int nKernelDiameter = int(nOuterRadius)*2+1;
        m_oDilateKernel.create(nKernelDiameter,nKernelDiameter);
        cv::circle(m_oDilateKernel,cv::Point(int(nOuterRadius),int(nOuterRadius)),int(nOuterRadius),cv::Scalar_<uchar>(255),-1);
        m_oDilateKernel = m_oDilateKernel>0;
#if USE_LIENHART_LOOKUP_MASK
        const int nMaskSize = m_nOuterRadius*2+1;
        lv::getLogPolarMask(nMaskSize,m_nRadialBins,m_nAngularBins,m_oAbsDescLUMap,true,(float)m_nInnerRadius);
        lvDbgAssert(m_oAbsDescLUMap.cols==nMaskSize && m_oAbsDescLUMap.rows==nMaskSize);
#endif //USE_LIENHART_LOOKUP_MASK
    }
}

ShapeContext::ShapeContext(double dRelativeInnerRadius, double dRelativeOuterRadius, size_t nAngularBins,
                           size_t nRadialBins, bool bRotationInvariant, bool bNormalizeBins) :
        m_nAngularBins((int)nAngularBins),
        m_nRadialBins((int)nRadialBins),
        m_nDescSize(m_nRadialBins*m_nAngularBins),
        m_nInnerRadius(0),
        m_nOuterRadius(0),
        m_dInnerRadius(dRelativeInnerRadius),
        m_dOuterRadius(dRelativeOuterRadius),
        m_bUseRelativeSpace(true),
        m_bRotationInvariant(bRotationInvariant),
        m_bNormalizeBins(bNormalizeBins) {
    lvAssert_(m_nAngularBins>0,"invalid parameter");
    lvAssert_(m_nRadialBins>0,"invalid parameter");
    lvAssert_(m_dInnerRadius>0.0,"invalid parameter");
    lvAssert_(m_dOuterRadius>0.0 && m_dInnerRadius<m_dOuterRadius,"invalid parameter");
    scdesc_generate_angmask();
    scdesc_generate_radmask();
    scdesc_generate_emdmask();
    lvAssert((int)m_vAngularLimits.size()==m_nAngularBins && (int)m_vRadialLimits.size()==m_nRadialBins);
    lvAssert(m_oEMDCostMap.dims==2 && m_oEMDCostMap.rows==m_nRadialBins*m_nAngularBins && m_oEMDCostMap.cols==m_nRadialBins*m_nAngularBins);
}

void ShapeContext::read(const cv::FileNode& /*fn*/) {
    // ... = fn["..."];
}

void ShapeContext::write(cv::FileStorage& /*fs*/) const {
    //fs << "..." << ...;
}

cv::Size ShapeContext::windowSize() const {
    return cv::Size(0,0);
}

int ShapeContext::borderSize(int nDim) const {
    lvAssert(nDim==0 || nDim==1);
    return 0;
}

lv::MatInfo ShapeContext::getOutputInfo(const lv::MatInfo& oInputInfo) const {
    lvAssert_(oInputInfo.type()==CV_8UC1,"invalid input image type");
    lvAssert_(oInputInfo.size.dims()==size_t(2) && oInputInfo.size.total()>0,"invalid input image size");
    const int nRows = (int)oInputInfo.size(0);
    const int nCols = (int)oInputInfo.size(1);
    const std::array<int,3> anDescDims = {nRows,nCols,m_nDescSize};
    return lv::MatInfo(lv::MatSize(anDescDims),CV_32FC1);
}

int ShapeContext::descriptorSize() const {
    return m_nRadialBins*m_nAngularBins*int(sizeof(float));
}

int ShapeContext::descriptorType() const {
    return CV_32F;
}

int ShapeContext::defaultNorm() const {
    return cv::NORM_L1;
}

bool ShapeContext::empty() const {
    return true;
}

bool ShapeContext::isNormalizingBins() const {
    return m_bNormalizeBins;
}

int ShapeContext::chainDetectMethod() const {
    return cv::CHAIN_APPROX_NONE;
}

void ShapeContext::compute2(const cv::Mat& oImage, cv::Mat& oDescMap_) {
    lvAssert_(oDescMap_.empty() || oDescMap_.type()==CV_32FC1,"wrong output desc map type");
    cv::Mat_<float> oDescMap = oDescMap_;
    const bool bEmptyInit = oDescMap.empty();
    compute2(oImage,oDescMap);
    if(bEmptyInit)
        oDescMap_ = oDescMap;
}

void ShapeContext::compute2(const cv::Mat& oImage, cv::Mat_<float>& oDescMap) {
    scdesc_fill_contours(oImage);
    m_oKeyPts.create((int)oImage.total(),1);
    int nKeyPtIdx = 0;
    for(int nRowIdx=0; nRowIdx<oImage.rows; ++nRowIdx)
        for(int nColIdx=0; nColIdx<oImage.cols; ++nColIdx)
            m_oKeyPts(nKeyPtIdx++) = cv::Point2f((float)nColIdx,(float)nRowIdx);
    if(!m_bUseRelativeSpace && !m_bRotationInvariant)
        scdesc_fill_desc_direct(oDescMap,true);
    else
        scdesc_fill_desc(oDescMap,true);
}

void ShapeContext::compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescMap) {
    scdesc_fill_contours(oImage);
    if(voKeypoints.empty()) {
        voKeypoints.resize(m_oContourPts.total());
        for(size_t nContourPtIdx=0; nContourPtIdx<voKeypoints.size(); ++nContourPtIdx)
            voKeypoints[nContourPtIdx] = cv::KeyPoint(m_oContourPts((int)nContourPtIdx),1.0f);
        m_oContourPts.copyTo(m_oKeyPts);
    }
    else {
        m_oKeyPts.create((int)voKeypoints.size(),1);
        for(size_t nKeyPtIdx=0; nKeyPtIdx<voKeypoints.size(); ++nKeyPtIdx)
            m_oKeyPts((int)nKeyPtIdx) = voKeypoints[nKeyPtIdx].pt;
    }
    if(!m_bUseRelativeSpace && !m_bRotationInvariant)
        scdesc_fill_desc_direct(oDescMap,true);
    else
        scdesc_fill_desc(oDescMap,true);
}

void ShapeContext::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<cv::Mat_<float>>& voDescMapCollection) {
    voDescMapCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],voDescMapCollection[i]);
}

void ShapeContext::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat_<float>>& voDescMapCollection) {
    lvAssert_(voImageCollection.size()==vvoPointCollection.size(),"number of images must match number of keypoint lists");
    voDescMapCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],vvoPointCollection[i],voDescMapCollection[i]);
}

void ShapeContext::detectAndCompute(cv::InputArray _oImage, cv::InputArray _oMask, std::vector<cv::KeyPoint>& voKeypoints, cv::OutputArray _oDescriptors, bool bUseProvidedKeypoints) {
    cv::Mat oImage = _oImage.getMat();
    cv::Mat oMask = _oMask.getMat();
    lvAssert_(oMask.empty() || (!oMask.empty() && oMask.size()==oImage.size()),"mask must be empty or of equal size to the input image");
    scdesc_fill_contours(oImage);
    if(!bUseProvidedKeypoints) {
        voKeypoints.resize(m_oContourPts.total());
        for(size_t nContourPtIdx=0; nContourPtIdx<voKeypoints.size(); ++nContourPtIdx)
            voKeypoints[nContourPtIdx] = cv::KeyPoint(m_oContourPts((int)nContourPtIdx),1.0f);
    }
    if(!oMask.empty())
        cv::KeyPointsFilter::runByPixelsMask(voKeypoints,oMask);
    if(voKeypoints.empty()) {
        _oDescriptors.release();
        return;
    }
    m_oKeyPts.create((int)voKeypoints.size(),1);
    for(size_t nKeyPtIdx=0; nKeyPtIdx<voKeypoints.size(); ++nKeyPtIdx)
        m_oKeyPts((int)nKeyPtIdx) = voKeypoints[nKeyPtIdx].pt;
    _oDescriptors.create((int)voKeypoints.size(),m_nRadialBins*m_nAngularBins,CV_32FC1);
    cv::Mat_<float> oDescriptors = cv::Mat_<float>(_oDescriptors.getMat());
    if(!m_bUseRelativeSpace && !m_bRotationInvariant)
        scdesc_fill_desc_direct(oDescriptors,false);
    else
        scdesc_fill_desc(oDescriptors,false);
}

void ShapeContext::reshapeDesc(cv::Size oSize, cv::Mat& oDescriptors) const {
    lvAssert_(!oDescriptors.empty() && oDescriptors.isContinuous(),"descriptor mat must be non-empty, and continuous");
    lvAssert_(oSize.area()>0 && oDescriptors.total()==size_t(oSize.area()*m_nRadialBins*m_nAngularBins),"bad expected output desc image size");
    lvAssert_(oDescriptors.dims==2 && oDescriptors.rows==oSize.area() && oDescriptors.type()==CV_32FC1,"descriptor mat type must be 2D, and 32FC1");
    const int anDescDims[3] = {oSize.height,oSize.width,m_nRadialBins*m_nAngularBins};
    oDescriptors = oDescriptors.reshape(0,3,anDescDims);
}

void ShapeContext::validateKeyPoints(std::vector<cv::KeyPoint>& /*voKeypoints*/, cv::Size /*oImgSize*/) const {}

void ShapeContext::validateROI(cv::Mat& oROI) const {
    lvAssert_(!oROI.empty() && oROI.type()==CV_8UC1,"input ROI must be non-empty and of type 8UC1");
}

void ShapeContext::scdesc_generate_radmask() {
    m_vRadialLimits.resize((size_t)m_nRadialBins);
    const double dMin = m_bUseRelativeSpace?m_dInnerRadius:(double)m_nInnerRadius;
    const double dMax = m_bUseRelativeSpace?m_dOuterRadius:(double)m_nOuterRadius;
    if(m_nRadialBins==1) {
        m_vRadialLimits[0] = dMax;
        return;
    }
    const double dLogMin = std::log2(dMin);
    const double dLogMax = std::log2(dMax);
    const double dDelta = (dLogMax-dLogMin)/(m_nRadialBins-1);
    m_vRadialLimits[0] = dMin;
    double dAccDelta=dDelta;
    for(int nBinIdx=1; nBinIdx<m_nRadialBins; ++nBinIdx, dAccDelta+=dDelta)
        m_vRadialLimits[nBinIdx] = std::exp2(dLogMin+dAccDelta);
}

void ShapeContext::scdesc_generate_angmask() {
    const double dDelta = 2*CV_PI/m_nAngularBins;
    double dAccDelta = 0.0;
    m_vAngularLimits.resize((size_t)m_nAngularBins);
    for(int nBinIdx=0; nBinIdx<m_nAngularBins; ++nBinIdx)
        m_vAngularLimits[nBinIdx] = (dAccDelta+=dDelta);
}

void ShapeContext::scdesc_generate_emdmask() {
    lvDbgAssert((int)m_vAngularLimits.size()==m_nAngularBins && (int)m_vRadialLimits.size()==m_nRadialBins);
    m_oEMDCostMap.create(m_nDescSize,m_nDescSize);
    for(int nBaseRadIdx=0; nBaseRadIdx<m_nRadialBins; ++nBaseRadIdx) {
        const double dBaseMeanRadius = (m_vRadialLimits[nBaseRadIdx]+(nBaseRadIdx==0?0.0:m_vRadialLimits[nBaseRadIdx-1]))/2;
        for(int nBaseAngIdx=0; nBaseAngIdx<m_nAngularBins; ++nBaseAngIdx) {
            const double dBaseMeanAngle = (m_vAngularLimits[nBaseAngIdx]+(nBaseAngIdx==0?0.0:m_vAngularLimits[nBaseAngIdx-1]))/2;
            const cv::Point2d vBaseCoord(dBaseMeanRadius*std::cos(dBaseMeanAngle),dBaseMeanRadius*std::sin(dBaseMeanAngle));
            const int nBaseDescIdx = nBaseAngIdx+nBaseRadIdx*m_nAngularBins;
            for(int nOffsetRadIdx=0; nOffsetRadIdx<m_nRadialBins; ++nOffsetRadIdx) {
                const double dOffsetMeanRadius = (m_vRadialLimits[nOffsetRadIdx]+(nOffsetRadIdx==0?0.0:m_vRadialLimits[nOffsetRadIdx-1]))/2;
                for(int nOffsetAngIdx=0; nOffsetAngIdx<m_nAngularBins; ++nOffsetAngIdx) {
                    const double dOffsetMeanAngle = (m_vAngularLimits[nOffsetAngIdx]+(nOffsetAngIdx==0?0.0:m_vAngularLimits[nOffsetAngIdx-1]))/2;
                    const cv::Point2d vOffsetCoord(dOffsetMeanRadius*std::cos(dOffsetMeanAngle),dOffsetMeanRadius*std::sin(dOffsetMeanAngle));
                    const int nOffsetDescIdx = nOffsetAngIdx+nOffsetRadIdx*m_nAngularBins;
                    m_oEMDCostMap(nBaseDescIdx,nOffsetDescIdx) = (float)cv::norm(cv::Mat(vOffsetCoord-vBaseCoord),cv::NORM_L2);
                }
            }
        }
    }
    m_oEMDCostMap /= m_vRadialLimits.back()/2; // @@@@@ normalize costs based on half desc radius
}

void ShapeContext::scdesc_fill_contours(const cv::Mat& oImage) {
    lvAssert_(!oImage.empty(),"input image must be non-empty");
    lvAssert_(oImage.type()==CV_8UC1,"input image type must be 8UC1");
    m_oCurrImageSize = oImage.size();
    std::vector<std::vector<cv::Point>> vvContours;
    cv::compare(oImage,0,m_oBinMask,cv::CMP_GT);
    if(!m_bUseRelativeSpace && !m_bRotationInvariant)
        cv::dilate(m_oBinMask,m_oDistMask,m_oDilateKernel);
    cv::findContours(m_oBinMask,vvContours,cv::RETR_LIST,cv::CHAIN_APPROX_NONE);
    size_t nContourPtCount = size_t(0);
    for(size_t nContourIdx=0; nContourIdx<vvContours.size(); ++nContourIdx)
        nContourPtCount += vvContours[nContourIdx].size();
    if(nContourPtCount>0) {
        m_oContourPts.create((int)nContourPtCount,1);
        int nContourPtIdx = 0;
        for(size_t nContourIdx = 0; nContourIdx<vvContours.size(); ++nContourIdx)
            for(size_t nPointIdx = 0; nPointIdx<vvContours[nContourIdx].size(); ++nPointIdx)
                m_oContourPts(nContourPtIdx++) = cv::Point2f(float(vvContours[nContourIdx][nPointIdx].x),float(vvContours[nContourIdx][nPointIdx].y));
    }
    else
        m_oContourPts.release();
}

void ShapeContext::scdesc_fill_maps(double dMeanDist) {
    lvDbgAssert(m_oContourPts.type()==CV_32FC2 && (m_oContourPts.total()==(size_t)m_oContourPts.rows || m_oContourPts.total()==(size_t)m_oContourPts.cols));
    lvDbgAssert(m_oKeyPts.type()==CV_32FC2 && (m_oKeyPts.total()==(size_t)m_oKeyPts.rows || m_oKeyPts.total()==(size_t)m_oKeyPts.cols));
    lvDbgAssert(m_vKeyInliers.empty() || m_oKeyPts.total()==m_vKeyInliers.size());
    lvDbgAssert(m_vContourInliers.empty() || m_oKeyPts.total()==m_vContourInliers.size());
    lvDbgAssert(m_oKeyPts.total()>size_t(0));
    if(m_oContourPts.empty())
        return;
    const bool bUseDistMask = m_bUseRelativeSpace && dMeanDist<0 && (!m_vKeyInliers.empty() || !m_vContourInliers.empty());
    if(bUseDistMask) {
        m_oDistMask.create((int)m_oKeyPts.total(),(int)m_oContourPts.total());
        for(int nKeyPtIdx=0; nKeyPtIdx<(int)m_oKeyPts.total(); ++nKeyPtIdx)
            for(int nContourPtIdx=0; nContourPtIdx<(int)m_oContourPts.total(); ++nContourPtIdx)
                m_oDistMask(nKeyPtIdx,nContourPtIdx) = uchar((m_vKeyInliers.empty() || m_vKeyInliers[nKeyPtIdx]!=0) && (m_vContourInliers.empty() || m_vContourInliers[nContourPtIdx]!=0));
    }
    m_oDistMap.create((int)m_oKeyPts.total(),(int)m_oContourPts.total());
    m_oAngMap.create((int)m_oKeyPts.total(),(int)m_oContourPts.total());
    cv::Point2f vMassCenter(0,0);
    if(m_bRotationInvariant) {
        for(int nContourPtIdx=0; nContourPtIdx<(int)m_oContourPts.total(); ++nContourPtIdx)
            vMassCenter += ((cv::Point2f*)m_oContourPts.data)[nContourPtIdx];
        vMassCenter.x = vMassCenter.x/m_oContourPts.total();
        vMassCenter.y = vMassCenter.y/m_oContourPts.total();
    }
    cv::Matx12f vPtDiff;
    for(int nKeyPtIdx=0; nKeyPtIdx<(int)m_oKeyPts.total(); ++nKeyPtIdx) {
        for(int nContourPtIdx=0; nContourPtIdx<(int)m_oContourPts.total(); ++nContourPtIdx) {
            const cv::Point2f& vKeyPt = ((cv::Point2f*)m_oKeyPts.data)[nKeyPtIdx];
            const cv::Point2f& vContourPt = ((cv::Point2f*)m_oContourPts.data)[nContourPtIdx];
            vPtDiff(0) = vKeyPt.y-vContourPt.y;
            vPtDiff(1) = vKeyPt.x-vContourPt.x;
            m_oDistMap(nKeyPtIdx,nContourPtIdx) = cv::norm(vPtDiff,cv::NORM_L2);
            if(m_oDistMap(nKeyPtIdx,nContourPtIdx)<0.01)
                m_oAngMap(nKeyPtIdx,nContourPtIdx) = 0.0;
            else {
                m_oAngMap(nKeyPtIdx,nContourPtIdx) = std::atan2(vPtDiff(0),-vPtDiff(1));
                if(m_bRotationInvariant) {
                    const cv::Point2d vRefPt = vContourPt-vMassCenter;
                    m_oAngMap(nKeyPtIdx,nContourPtIdx) -= std::atan2(-vRefPt.y,vRefPt.x);
                }
                m_oAngMap(nKeyPtIdx,nContourPtIdx) = std::fmod(m_oAngMap(nKeyPtIdx,nContourPtIdx)+2*CV_PI+FLT_EPSILON,2*CV_PI);
            }
        }
    }
    if(m_bUseRelativeSpace) {
        if(dMeanDist<0)
            dMeanDist = cv::mean(m_oDistMap,bUseDistMask?m_oDistMask:cv::noArray())[0];
        m_oDistMap /= (dMeanDist+FLT_EPSILON);
    }
}

void ShapeContext::scdesc_fill_desc(cv::Mat_<float>& oDescriptors, bool bGenDescMap) {
    if(m_oKeyPts.empty()) {
        oDescriptors.release();
        return;
    }
    if(bGenDescMap)
        oDescriptors.create(3,std::array<int,3>{m_oCurrImageSize.height,m_oCurrImageSize.width,m_nDescSize}.data());
    else
        oDescriptors.create((int)m_oKeyPts.total(),m_nDescSize);
    oDescriptors = m_bNormalizeBins?1.0f/m_nDescSize:0.0f;
    scdesc_fill_maps();
    for(int nKeyPtIdx=0; nKeyPtIdx<(int)m_oKeyPts.total(); ++nKeyPtIdx) {
        if(!m_vKeyInliers.empty() && !m_vKeyInliers[nKeyPtIdx])
            continue;
        size_t nValidPts = 1;
        const cv::Point2f& vKeyPt = ((cv::Point2f*)m_oKeyPts.data)[nKeyPtIdx];
        const int nKeyPtRowIdx = (int)std::round(vKeyPt.y);
        const int nKeyPtColIdx = (int)std::round(vKeyPt.x);
        float* aDesc = bGenDescMap?oDescriptors.ptr<float>(nKeyPtRowIdx,nKeyPtColIdx):oDescriptors.ptr<float>(nKeyPtIdx);
        for(int nContourPtIdx=0; nContourPtIdx<(int)m_oContourPts.total(); ++nContourPtIdx) {
            if(!m_vContourInliers.empty() && !m_vContourInliers[nContourPtIdx])
                continue;
            const cv::Point2f& vContourPt = ((cv::Point2f*)m_oContourPts.data)[nContourPtIdx];
            if(std::abs(vKeyPt.x-vContourPt.x)<0.01f && std::abs(vKeyPt.y-vContourPt.y)<0.01f)
                continue;
            int nAngularBinMatch=-1,nRadialBinMatch=-1;
            for(int nRadialBinIdx=0; nRadialBinIdx<m_nRadialBins; ++nRadialBinIdx) {
                if(m_oDistMap(nKeyPtIdx,nContourPtIdx)<m_vRadialLimits[nRadialBinIdx]) {
                    nRadialBinMatch = nRadialBinIdx;
                    break;
                }
            }
            for(int nAngularBinIdx=0; nAngularBinIdx<m_nAngularBins; ++nAngularBinIdx) {
                if(m_oAngMap(nKeyPtIdx,nContourPtIdx)<m_vAngularLimits[nAngularBinIdx]) {
                    nAngularBinMatch = nAngularBinIdx;
                    break;
                }
            }
            if(nAngularBinMatch!=-1 && nRadialBinMatch!=-1) {
                ++(aDesc[nAngularBinMatch+nRadialBinMatch*m_nAngularBins]);
                ++nValidPts;
            }
        }
        if(m_bNormalizeBins && nValidPts>1)
            cv::Mat_<float>(1,m_nDescSize,aDesc) *= 1.0f/nValidPts;
    }
}

void ShapeContext::scdesc_fill_desc_direct(cv::Mat_<float>& oDescriptors, bool bGenDescMap) {
    lvAssert_(!m_bUseRelativeSpace && !m_bRotationInvariant,"mapless impl cannot handle relative dist space/rot inv");
    // this impl specialization will likely be faster for very large shapes, but only handles the conditions above
    if(m_oKeyPts.empty()) {
        oDescriptors.release();
        return;
    }
    if(bGenDescMap)
        oDescriptors.create(3,std::array<int,3>{m_oCurrImageSize.height,m_oCurrImageSize.width,m_nDescSize}.data());
    else
        oDescriptors.create((int)m_oKeyPts.total(),m_nDescSize);
    oDescriptors = m_bNormalizeBins?1.0f/m_nDescSize:0.0f;
    lvDbgAssert(m_oContourPts.type()==CV_32FC2 && (m_oContourPts.total()==(size_t)m_oContourPts.rows || m_oContourPts.total()==(size_t)m_oContourPts.cols));
    lvDbgAssert(m_oKeyPts.type()==CV_32FC2 && (m_oKeyPts.total()==(size_t)m_oKeyPts.rows || m_oKeyPts.total()==(size_t)m_oKeyPts.cols));
    lvDbgAssert(m_vKeyInliers.empty() || m_oKeyPts.total()==m_vKeyInliers.size());
    lvDbgAssert(m_vContourInliers.empty() || m_oKeyPts.total()==m_vContourInliers.size());
    lvDbgAssert(m_oKeyPts.total()>size_t(0));
    lvDbgAssert(m_oDistMask.size()==m_oCurrImageSize);
#if USE_LIENHART_LOOKUP_MASK
    lvDbgAssert(!m_oAbsDescLUMap.empty() && m_oAbsDescLUMap.rows==m_nOuterRadius*2+1 && m_oAbsDescLUMap.rows==m_oAbsDescLUMap.cols);
#endif //USE_LIENHART_LOOKUP_MASK
    for(int nKeyPtIdx=0; nKeyPtIdx<(int)m_oKeyPts.total(); ++nKeyPtIdx) {
        if(!m_vKeyInliers.empty() && !m_vKeyInliers[nKeyPtIdx])
            continue;
        size_t nValidPts = 1;
        const cv::Point2f& vKeyPt = ((cv::Point2f*)m_oKeyPts.data)[nKeyPtIdx];
        const int nKeyPtRowIdx = (int)std::round(vKeyPt.y);
        const int nKeyPtColIdx = (int)std::round(vKeyPt.x);
        float* aDesc = bGenDescMap?oDescriptors.ptr<float>(nKeyPtRowIdx,nKeyPtColIdx):oDescriptors.ptr<float>(nKeyPtIdx);
        if(m_oDistMask(nKeyPtRowIdx,nKeyPtColIdx)) {
            for(int nContourPtIdx=0; nContourPtIdx<(int)m_oContourPts.total(); ++nContourPtIdx) {
                if(!m_vContourInliers.empty() && !m_vContourInliers[nContourPtIdx])
                    continue;
                const cv::Point2f& vContourPt = ((cv::Point2f*)m_oContourPts.data)[nContourPtIdx];
#if USE_LIENHART_LOOKUP_MASK
                const int nLookupRow = (int)std::round(vContourPt.y-vKeyPt.y)+m_nOuterRadius;
                const int nLookupCol = (int)std::round(vContourPt.x-vKeyPt.x)+m_nOuterRadius;
                if(nLookupRow<0 || nLookupRow>=m_oAbsDescLUMap.rows || nLookupCol<0 || nLookupCol>=m_oAbsDescLUMap.cols)
                    continue;
                const int nDescIdx = m_oAbsDescLUMap(nLookupRow,nLookupCol);
                if(nDescIdx==-1)
                    continue;
                ++(aDesc[nDescIdx]);
#else //!USE_LIENHART_LOOKUP_MASK
                cv::Matx12f vPtDiff;
                vPtDiff(0) = vKeyPt.y-vContourPt.y;
                vPtDiff(1) = vKeyPt.x-vContourPt.x;
                const double dCurrDist = cv::norm(vPtDiff,cv::NORM_L2);
                if(dCurrDist<0.01 || dCurrDist>m_vRadialLimits.back())
                    continue;
                int nRadialBinMatch=-1;
                for(int nRadialBinIdx=0; nRadialBinIdx<m_nRadialBins; ++nRadialBinIdx) {
                    if(dCurrDist<m_vRadialLimits[nRadialBinIdx]) {
                        nRadialBinMatch = nRadialBinIdx;
                        break;
                    }
                }
                if(nRadialBinMatch<0)
                    continue;
                int nAngularBinMatch=-1;
                const double dCurrAng = (dCurrDist<0.01)?0.0:std::fmod(std::atan2(vPtDiff(0),-vPtDiff(1))+2*CV_PI+FLT_EPSILON,2*CV_PI);
                for(int nAngularBinIdx=0; nAngularBinIdx<m_nAngularBins; ++nAngularBinIdx) {
                    if(dCurrAng<m_vAngularLimits[nAngularBinIdx]) {
                        nAngularBinMatch = nAngularBinIdx;
                        break;
                    }
                }
                if(nAngularBinMatch<0)
                    continue;
                ++(aDesc[nAngularBinMatch+nRadialBinMatch*m_nAngularBins]);
#endif //!USE_LIENHART_LOOKUP_MASK
                ++nValidPts;
            }
        }
        if(m_bNormalizeBins && nValidPts>1)
            cv::Mat_<float>(1,m_nDescSize,aDesc) *= 1.0f/nValidPts;
    }
}