
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

// note: this implementation is inspired by Ken Chatfield's and Rainer Lienhart's
// original implementations; see the originals at:
//    http://www.robots.ox.ac.uk/~vgg/software/SelfSimilarity/
//    https://github.com/opencv/opencv/blob/2.4/modules/contrib/src/selfsimilarity.cpp

// @@@@ test with nan in oob lookup

#define USE_STATIC_VAR_NOISE    1 // 0 == dynamically determine variation based on original paper suggestion

#define _USE_MATH_DEFINES
#include "litiv/features2d.hpp"

LSS::LSS(int nInnerRadius, int nOuterRadius, int nDescPatchSize, int nAngularBins, int nRadialBins, float fStaticNoiseVar, bool bNormalizeBins, bool bPreProcess, bool bUseLienhartMask) :
        m_bPreProcess(bPreProcess),
        m_bNormalizeBins(bNormalizeBins),
        m_bUsingLienhartMask(bUseLienhartMask),
        m_nPatchSize(nDescPatchSize),
        m_nInnerRadius(nInnerRadius),
        m_nOuterRadius(nOuterRadius),
        m_nCorrWinSize(m_nOuterRadius*2+m_nPatchSize),
        m_nCorrPatchSize(m_nCorrWinSize-m_nPatchSize+1),
        m_nRadialBins(nRadialBins),
        m_nAngularBins(nAngularBins),
        m_fStaticNoiseVar(fStaticNoiseVar) {
    lvAssert_(m_nPatchSize>0 && (m_nPatchSize%2)==1,"invalid parameter");
    lvAssert_(m_nOuterRadius>0 && m_nOuterRadius>=m_nPatchSize,"invalid parameter");
    lvAssert_(m_nInnerRadius>=0 && m_nOuterRadius>m_nInnerRadius,"invalid parameter");
    lvAssert_(m_nRadialBins>0,"invalid parameter");
    lvAssert_(m_nAngularBins>0,"invalid parameter");
    lvAssert_(m_fStaticNoiseVar>0.0f,"invalid parameter");
    lv::getLogPolarMask(m_nCorrPatchSize,m_nRadialBins,m_nAngularBins,m_oDescLUMap,m_bUsingLienhartMask,(float)m_nInnerRadius,&m_nFirstMaskIdx,&m_nLastMaskIdx);
    lvDbgAssert(m_oDescLUMap.cols==m_nCorrPatchSize && m_oDescLUMap.rows==m_nCorrPatchSize);
    lvDbgAssert(m_nFirstMaskIdx>=0 && m_nLastMaskIdx>=m_nFirstMaskIdx);
}

void LSS::read(const cv::FileNode& /*fn*/) {
    // ... = fn["..."];
}

void LSS::write(cv::FileStorage& /*fs*/) const {
    //fs << "..." << ...;
}

cv::Size LSS::windowSize() const {
    return cv::Size(m_nCorrWinSize,m_nCorrWinSize);
}

int LSS::borderSize(int nDim) const {
    lvAssert(nDim==0 || nDim==1);
    return m_nCorrWinSize/2;
}

lv::MatInfo LSS::getOutputInfo(const lv::MatInfo& oInputInfo) const {
    lvAssert_(oInputInfo.type()==CV_8UC1 || oInputInfo.type()==CV_8UC3,"invalid input image type");
    lvAssert_(oInputInfo.size.dims()==size_t(2) && oInputInfo.size.total()>0,"invalid input image size");
    const int nRows = (int)oInputInfo.size(0);
    const int nCols = (int)oInputInfo.size(1);
    lvAssert__(m_nCorrWinSize<=nCols && m_nCorrWinSize<=nRows,"input image size is too small to compute descriptors with current correlation area size -- need at least (%d,%d) and got (%d,%d)",m_nCorrWinSize,m_nCorrWinSize,nCols,nRows);
    const int nDescSize = m_nRadialBins*m_nAngularBins;
    const std::array<int,3> anDescDims = {nRows,nCols,nDescSize};
    return lv::MatInfo(lv::MatSize(anDescDims),CV_32FC1);
}

int LSS::descriptorSize() const {
    return m_nRadialBins*m_nAngularBins*int(sizeof(float));
}

int LSS::descriptorType() const {
    return CV_32F;
}

int LSS::defaultNorm() const {
    return cv::NORM_L2;
}

bool LSS::empty() const {
    return true;
}

bool LSS::isUsingDynamicNoiseVarNorm() const {
    return USE_STATIC_VAR_NOISE;
}

bool LSS::isNormalizingBins() const {
    return m_bNormalizeBins;
}

bool LSS::isPreProcessing() const {
    return m_bPreProcess;
}

bool LSS::isUsingLienhartMask() const {
    return m_bUsingLienhartMask;
}

void LSS::compute2(const cv::Mat& oImage, cv::Mat& oDescMap_) {
    lvAssert_(oDescMap_.empty() || oDescMap_.type()==CV_32FC1,"wrong output desc map type");
    cv::Mat_<float> oDescMap = oDescMap_;
    const bool bEmptyInit = oDescMap.empty();
    compute2(oImage,oDescMap);
    if(bEmptyInit)
        oDescMap_ = oDescMap;
}

void LSS::compute2(const cv::Mat& oImage, cv::Mat_<float>& oDescMap) {
    ssdescs_impl(oImage,oDescMap);
}

void LSS::compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescMap) {
    ssdescs_impl(oImage,voKeypoints,oDescMap,true);
}

void LSS::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<cv::Mat_<float>>& voDescMapCollection) {
    voDescMapCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],voDescMapCollection[i]);
}

void LSS::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat_<float>>& voDescMapCollection) {
    lvAssert_(voImageCollection.size()==vvoPointCollection.size(),"number of images must match number of keypoint lists");
    voDescMapCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],vvoPointCollection[i],voDescMapCollection[i]);
}

void LSS::detectAndCompute(cv::InputArray _oImage, cv::InputArray _oMask, std::vector<cv::KeyPoint>& voKeypoints, cv::OutputArray _oDescriptors, bool bUseProvidedKeypoints) {
    cv::Mat oImage = _oImage.getMat();
    lvAssert_(!oImage.empty(),"input image must be non-empty");
    cv::Mat oMask = _oMask.getMat();
    lvAssert_(oMask.empty() || (!oMask.empty() && oMask.size()==oImage.size()),"mask must be empty or of equal size to the input image");
    if(!bUseProvidedKeypoints) {
        voKeypoints.clear();
        voKeypoints.reserve(size_t(oImage.rows*oImage.cols));
        for(int nRowIdx=0; nRowIdx<oImage.rows; ++nRowIdx)
            for(int nColIdx=0; nColIdx<oImage.cols; ++nColIdx)
                voKeypoints.emplace_back(cv::Point2f((float)nColIdx,(float)nRowIdx),(float)m_nCorrWinSize);
    }
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImage.size(),m_nCorrWinSize/2);
    if(!oMask.empty())
        cv::KeyPointsFilter::runByPixelsMask(voKeypoints,oMask);
    if(voKeypoints.empty()) {
        _oDescriptors.release();
        return;
    }
    _oDescriptors.create((int)voKeypoints.size(),m_nRadialBins*m_nAngularBins,CV_32FC1);
    cv::Mat_<float> oDescriptors = cv::Mat_<float>(_oDescriptors.getMat());
    ssdescs_impl(oImage,voKeypoints,oDescriptors,false);
}

void LSS::reshapeDesc(cv::Size oSize, cv::Mat& oDescriptors) const {
    lvAssert_(!oDescriptors.empty() && oDescriptors.isContinuous(),"descriptor mat must be non-empty, and continuous");
    lvAssert_(oSize.area()>0 && oDescriptors.total()==size_t(oSize.area()*m_nRadialBins*m_nAngularBins),"bad expected output desc image size");
    lvAssert_(oDescriptors.dims==2 && oDescriptors.rows==oSize.area() && oDescriptors.type()==CV_32FC1,"descriptor mat type must be 2D, and 32FC1");
    const int anDescDims[3] = {oSize.height,oSize.width,m_nRadialBins*m_nAngularBins};
    oDescriptors = oDescriptors.reshape(0,3,anDescDims);
}

void LSS::validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) const {
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImgSize,m_nCorrWinSize/2);
}

void LSS::validateROI(cv::Mat& oROI) const {
    lvAssert_(!oROI.empty() && oROI.type()==CV_8UC1,"input ROI must be non-empty and of type 8UC1");
    cv::Mat oROI_new(oROI.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    const int nBorderSize = m_nCorrWinSize/2;
    const cv::Rect nROI_inner(nBorderSize,nBorderSize,oROI.cols-nBorderSize*2,oROI.rows-nBorderSize*2);
    cv::Mat(oROI,nROI_inner).copyTo(cv::Mat(oROI_new,nROI_inner));
    oROI = oROI_new;
}

void LSS::calcDistances(const cv::Mat_<float>& oDescriptors1, const cv::Mat_<float>& oDescriptors2, cv::Mat_<float>& oDistances) const {
    lvAssert_(oDescriptors1.dims==oDescriptors2.dims && oDescriptors1.size==oDescriptors2.size,"descriptor mat sizes mismatch");
    lvAssert_(oDescriptors1.dims==2 || oDescriptors1.dims==3,"unexpected descriptor matrix dim count");
    if(oDescriptors1.dims==2) {
        lvAssert_(oDescriptors1.cols==m_nRadialBins*m_nAngularBins,"unexpected descriptor size");
        oDistances.create(oDescriptors1.rows,1);
        for(int nDescIdx=0; nDescIdx<oDescriptors1.rows; ++nDescIdx) {
            const cv::Mat_<float> oDesc1(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(oDescriptors1.ptr<float>(nDescIdx)));
            const cv::Mat_<float> oDesc2(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(oDescriptors2.ptr<float>(nDescIdx)));
            oDistances(nDescIdx) = (float)cv::norm(oDesc1,oDesc2,cv::NORM_L2);
        }
    }
    else { //oDescriptors1.dims==3
        lvAssert_(oDescriptors1.size[2]==m_nRadialBins*m_nAngularBins,"unexpected descriptor size");
        oDistances.create(oDescriptors1.size[0],oDescriptors1.size[1]);
        for(int nDescRowIdx=0; nDescRowIdx<oDescriptors1.size[0]; ++nDescRowIdx) {
            for(int nDescColIdx=0; nDescColIdx<oDescriptors1.size[1]; ++nDescColIdx) {
                const cv::Mat_<float> oDesc1(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(oDescriptors1.ptr<float>(nDescRowIdx,nDescColIdx)));
                const cv::Mat_<float> oDesc2(1,m_nRadialBins*m_nAngularBins,const_cast<float*>(oDescriptors2.ptr<float>(nDescRowIdx,nDescColIdx)));
                oDistances(nDescRowIdx,nDescColIdx) = (float)cv::norm(oDesc1,oDesc2,cv::NORM_L2);
            }
        }
    }
}

void LSS::ssdescs_impl(const cv::Mat& _oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescriptors, bool bGenDescMap) {
    lvAssert_(!_oImage.empty() && ((_oImage.type()==CV_8UC1) || (_oImage.type()==CV_8UC3)),"invalid input image");
    lvAssert__(m_nCorrWinSize<=_oImage.cols && m_nCorrWinSize<=_oImage.rows,"image is too small to compute descriptors with current correlation area size -- need at least (%d,%d) and got (%d,%d)",m_nCorrWinSize,m_nCorrWinSize,_oImage.cols,_oImage.rows);
    lvDbgAssert(m_oDescLUMap.rows==m_nCorrPatchSize && m_oDescLUMap.cols==m_nCorrPatchSize);
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,_oImage.size(),m_nCorrWinSize/2);
    if(voKeypoints.empty()) {
        oDescriptors.release();
        return;
    }
    cv::Mat oImage;
    if(m_bPreProcess)
        cv::GaussianBlur(_oImage,oImage,cv::Size(7,7),1.0);
    else
        oImage = _oImage;
    const int nCorrWinRadius = m_nCorrWinSize/2;
    const int nPatchRadius = m_nPatchSize/2;
    const int nDescSize = m_nRadialBins*m_nAngularBins;
    static thread_local lv::AutoBuffer<float> s_aCorrData;
    s_aCorrData.resize(size_t(m_nCorrPatchSize*m_nCorrPatchSize));
    cv::Mat_<float> oCorrMap(m_nCorrPatchSize,m_nCorrPatchSize,s_aCorrData.data());
    static thread_local lv::AutoBuffer<float> s_aTempDesc;
    s_aTempDesc.resize(((size_t)nDescSize));
    cv::Mat_<float> oTempDesc(1,nDescSize,s_aTempDesc.data());
    if(bGenDescMap)
        oDescriptors.create(3,std::array<int,3>{oImage.rows,oImage.cols,nDescSize}.data());
    else
        oDescriptors.create(int(voKeypoints.size()),nDescSize);
    for(int nKeyPtIdx=0; nKeyPtIdx<int(voKeypoints.size()); ++nKeyPtIdx) {
        const cv::KeyPoint& oCurrKeyPt = voKeypoints[nKeyPtIdx];
        const int nRowIdx = int(oCurrKeyPt.pt.y);
        const int nColIdx = int(oCurrKeyPt.pt.x);
        lvDbgAssert(nRowIdx>=0 && nColIdx>=0);
        const cv::Mat oWindow = oImage(cv::Rect(nColIdx-nCorrWinRadius,nRowIdx-nCorrWinRadius,m_nCorrWinSize,m_nCorrWinSize));
        const cv::Mat oTempl = oImage(cv::Rect(nColIdx-nPatchRadius,nRowIdx-nPatchRadius,m_nPatchSize,m_nPatchSize));
        cv::matchTemplate(oWindow,oTempl,oCorrMap,cv::TM_SQDIFF);
#if USE_STATIC_VAR_NOISE
        const float fVarNormFact = -1.0f/m_fStaticNoiseVar;
#else //!USE_STATIC_VAR_NOISE
        float fMaxLocalVarNoise = 1000.0f;
        for(int nRowOffset=-1; nRowOffset<=1 ; ++nRowOffset)
            for(int nColOffset=-1; nColOffset<=1; ++nColOffset)
                fMaxLocalVarNoise = std::max(fMaxLocalVarNoise,oCorrMap(m_nCorrPatchSize/2+nRowOffset,m_nCorrPatchSize/2+nColOffset));
        const float fVarNormFact = -1.0f/fMaxLocalVarNoise;
#endif //!USE_STATIC_VAR_NOISE
        oTempDesc = std::numeric_limits<float>::max();
        for(int nDescBinIdx=m_nFirstMaskIdx; nDescBinIdx<=m_nLastMaskIdx; ++nDescBinIdx)
            if(m_oDescLUMap(nDescBinIdx)!=-1)
                s_aTempDesc[m_oDescLUMap(nDescBinIdx)] = std::min(s_aTempDesc[m_oDescLUMap(nDescBinIdx)],((float*)oCorrMap.data)[nDescBinIdx]);
        oTempDesc *= fVarNormFact;
        cv::exp(oTempDesc,cv::Mat_<float>(1,nDescSize,bGenDescMap?oDescriptors.ptr<float>(nRowIdx,nColIdx):oDescriptors.ptr<float>(nKeyPtIdx)));
    }
    if(m_bNormalizeBins)
        ssdescs_norm(oDescriptors);
}

void LSS::ssdescs_impl(const cv::Mat& _oImage, cv::Mat_<float>& oDescriptors) {
    lvAssert_(!_oImage.empty() && ((_oImage.type()==CV_8UC1) || (_oImage.type()==CV_8UC3)),"invalid input image");
    lvAssert__(m_nCorrWinSize<=_oImage.cols && m_nCorrWinSize<=_oImage.rows,"image is too small to compute descriptors with current correlation area size -- need at least (%d,%d) and got (%d,%d)",m_nCorrWinSize,m_nCorrWinSize,_oImage.cols,_oImage.rows);
    lvDbgAssert(m_oDescLUMap.rows==m_nCorrPatchSize && m_oDescLUMap.cols==m_nCorrPatchSize);
    cv::Mat oImage;
    if(m_bPreProcess)
        cv::GaussianBlur(_oImage,oImage,cv::Size(7,7),1.0);
    else
        oImage = _oImage;
    const int nRows = oImage.rows;
    const int nCols = oImage.cols;
    const int nCorrWinRadius = m_nCorrWinSize/2;
    const int nPatchRadius = m_nPatchSize/2;
    const int nDescSize = m_nRadialBins*m_nAngularBins;
    const int anDescDims[3] = {nRows,nCols,nDescSize};
    oDescriptors.create(3,anDescDims);
    std::fill_n(oDescriptors.ptr<float>(0,0),nDescSize*nCorrWinRadius*nCols,0.0f);
    std::fill_n(oDescriptors.ptr<float>(nRows-nCorrWinRadius,0),nDescSize*nCorrWinRadius*nCols,0.0f);
#if USING_OPENMP
    #pragma omp parallel for
#endif //USING_OPENMP
    for(int nRowIdx=nCorrWinRadius; nRowIdx<nRows-nCorrWinRadius; ++nRowIdx) {
        static thread_local lv::AutoBuffer<float> s_aCorrData;
        s_aCorrData.resize(size_t(m_nCorrPatchSize*m_nCorrPatchSize));
        cv::Mat_<float> oCorrMap(m_nCorrPatchSize,m_nCorrPatchSize,s_aCorrData.data());
        static thread_local lv::AutoBuffer<float> s_aTempDesc;
        s_aTempDesc.resize(size_t(nDescSize));
        cv::Mat_<float> oTempDesc(1,nDescSize,s_aTempDesc.data());
        std::fill_n(oDescriptors.ptr<float>(nRowIdx,0),nDescSize*nCorrWinRadius,0.0f);
        std::fill_n(oDescriptors.ptr<float>(nRowIdx,nCols-nCorrWinRadius),nDescSize*nCorrWinRadius,0.0f);
        for(int nColIdx=nCorrWinRadius; nColIdx<nCols-nCorrWinRadius; ++nColIdx) {
            const cv::Mat oWindow = oImage(cv::Rect(nColIdx-nCorrWinRadius,nRowIdx-nCorrWinRadius,m_nCorrWinSize,m_nCorrWinSize));
            const cv::Mat oTempl = oImage(cv::Rect(nColIdx-nPatchRadius,nRowIdx-nPatchRadius,m_nPatchSize,m_nPatchSize));
            cv::matchTemplate(oWindow,oTempl,oCorrMap,cv::TM_SQDIFF);
#if USE_STATIC_VAR_NOISE
            const float fVarNormFact = -1.0f/m_fStaticNoiseVar;
#else //!USE_STATIC_VAR_NOISE
            float fMaxLocalVarNoise = 1000.0f;
            for(int nRowOffset=-1; nRowOffset<=1 ; ++nRowOffset)
                for(int nColOffset=-1; nColOffset<=1; ++nColOffset)
                    fMaxLocalVarNoise = std::max(fMaxLocalVarNoise,oCorrMap(m_nCorrPatchSize/2+nRowOffset,m_nCorrPatchSize/2+nColOffset));
            const float fVarNormFact = -1.0f/fMaxLocalVarNoise;
#endif //!USE_STATIC_VAR_NOISE
            oTempDesc = std::numeric_limits<float>::max();
            for(int nDescBinIdx=m_nFirstMaskIdx; nDescBinIdx<=m_nLastMaskIdx; ++nDescBinIdx)
                if(m_oDescLUMap(nDescBinIdx)!=-1)
                    s_aTempDesc[m_oDescLUMap(nDescBinIdx)] = std::min(s_aTempDesc[m_oDescLUMap(nDescBinIdx)],((float*)oCorrMap.data)[nDescBinIdx]);
            oTempDesc *= fVarNormFact;
            cv::exp(oTempDesc,cv::Mat_<float>(1,nDescSize,oDescriptors.ptr<float>(nRowIdx,nColIdx)));
        }
    }
    if(m_bNormalizeBins)
        ssdescs_norm(oDescriptors);
}

void LSS::ssdescs_norm(cv::Mat_<float>& oDescriptors) const {
    if(oDescriptors.empty())
        return;
    const int nDescSize = m_nRadialBins*m_nAngularBins;
    lvDbgAssert((oDescriptors.total()%size_t(nDescSize))==0);
    lvDbgAssert(oDescriptors.size[oDescriptors.dims-1]==nDescSize);
    lvDbgAssert(oDescriptors.isContinuous());
    for(size_t nDescIdx=0; nDescIdx<oDescriptors.total(); nDescIdx+=size_t(nDescSize)) {
        cv::Mat_<float> oCurrDesc(1,nDescSize,((float*)oDescriptors.data)+nDescIdx);
        const double dNorm = cv::norm(oCurrDesc,cv::NORM_L2);
        if(dNorm>1e-6)
            oCurrDesc /= dNorm;
        else
            oCurrDesc = std::sqrt(1.0f/nDescSize);
    }
}