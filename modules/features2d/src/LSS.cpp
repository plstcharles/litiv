
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

#define USE_CHATFIELD_MASK      0 // 0 == use Lienhart's mask
#define USE_STATIC_VAR_NOISE    1 // 0 == dynamically determine variation based on original paper suggestion
#define USE_ITERATIVE_SSD       0 // 0 == use per-pixel 'matchTemplate' call (less prone to cumulative float error)
#define USE_POST_NORMALISATION  0 // 0 == leave descriptors as-is (may not match well during illum variations)

#define _USE_MATH_DEFINES
#include "litiv/features2d.hpp"
#include "litiv/utils/opencv.hpp"

inline void ssdesc_genmask(int nMaskSize, int nRadialBins, int nAngularBins, cv::Mat_<int>& oMask, int& nFirstMaskIdx, int& nLastMaskIdx) {
    lvDbgAssert_(nMaskSize>0 && (nMaskSize%2)==1,"mask size must be non-null, positive, odd value");
    lvDbgAssert_(nRadialBins>0,"radial bin count must be non-null positive value");
    lvDbgAssert_(nAngularBins>0,"angular bin count must be non-null positive value");
    oMask.create(nMaskSize,nMaskSize);
    oMask = -1;
    nFirstMaskIdx = nMaskSize*nMaskSize-1;
    nLastMaskIdx = 0;
    int nCurrMaskIdx = 0;
    const int nInitDistBin = 0; // previously passed as param; fix? @@@
#if USE_CHATFIELD_MASK
    const int nCenterIdx = (nMaskSize-1)/2;
    std::vector<float> vRadBinDists(nRadialBins);
    const float fRadBinPowBase = (float)std::pow(nRadialBins,1/(float)nRadialBins);
    for(int nRadBinIdx=0; nRadBinIdx<nRadialBins; ++nRadBinIdx)
        vRadBinDists[nRadBinIdx] = (std::pow(fRadBinPowBase,nRadBinIdx+1)-1)/(nRadialBins-1)*nCenterIdx;
    for(int nRowIdx=0; nRowIdx<nMaskSize; ++nRowIdx) {
        int* const pnMaskRow = oMask.ptr<int>(nRowIdx);
        for(int nColIdx=0; nColIdx<nMaskSize; ++nColIdx,++nCurrMaskIdx) {
            if((nRowIdx==nCenterIdx) && (nColIdx==nCenterIdx))
                continue;
            const float fDist = std::sqrt((float)((nCenterIdx-nRowIdx)*(nCenterIdx-nRowIdx)+(nCenterIdx-nColIdx)*(nCenterIdx-nColIdx)));
            int nRadBinIdx;
            for(nRadBinIdx=0; nRadBinIdx<nRadialBins; ++nRadBinIdx)
                if(fDist<=vRadBinDists[nRadBinIdx])
                    break;
            if(nRadBinIdx>=nInitDistBin && nRadBinIdx<nRadialBins) {
                const float fAng = std::atan2((float)(nCenterIdx-nColIdx),(float)(nCenterIdx-nRowIdx))+float(M_PI);
                const int nAngleBinIdx = int((fAng*nAngularBins)/float(2*M_PI))%nAngularBins;
                pnMaskRow[nColIdx] = nAngleBinIdx*(nRadialBins-nInitDistBin)+((nRadialBins-nInitDistBin-1)-(nRadBinIdx-nInitDistBin));
                nFirstMaskIdx = std::min(nFirstMaskIdx,nCurrMaskIdx);
                nLastMaskIdx = std::max(nLastMaskIdx,nCurrMaskIdx);
            }
        }
    }
#else //!USE_CHATFIELD_MASK
    const int nPatchRadius = nMaskSize/2, nAngBinSize = 360/nAngularBins;
    const float fRadBinLogBase = nRadialBins/(float)std::log10(nPatchRadius);
    for(int nRowIdx=-nPatchRadius; nRowIdx<=nPatchRadius; ++nRowIdx) {
        int* const pnMaskRow = oMask.ptr<int>(nRowIdx+nPatchRadius);
        for(int nColIdx=-nPatchRadius; nColIdx<=nPatchRadius; ++nColIdx,++nCurrMaskIdx) {
            if(nRowIdx==0 && nColIdx==0)
                continue;
            const float fDist = std::sqrt((float)(nColIdx*nColIdx) + (float)(nRowIdx*nRowIdx));
            const int nRadBinIdx = int(fDist>0.0f?std::log10(fDist)*fRadBinLogBase:0.0f);
            if(nRadBinIdx>=nInitDistBin && nRadBinIdx<nRadialBins) {
                const float fAng = std::atan2((float)nRowIdx,(float)nColIdx)/float(M_PI)*180.0f;
                const int nAngleBinIdx = (((int)std::round(fAng<0?fAng+360.0f:fAng)+nAngBinSize/2)%360)/nAngBinSize;
                pnMaskRow[nColIdx+nPatchRadius] = nAngleBinIdx*(nRadialBins-nInitDistBin)+((nRadialBins-nInitDistBin-1)-(nRadBinIdx-nInitDistBin));
                nFirstMaskIdx = std::min(nFirstMaskIdx,nCurrMaskIdx);
                nLastMaskIdx = std::max(nLastMaskIdx,nCurrMaskIdx);
            }
        }
    }
#endif //!USE_CHATFIELD_MASK
    //std::cout << "nMaskSize = " << nMaskSize << std::endl;
    //std::cout << "nAngularBins = " << nAngularBins << std::endl;
    //std::cout << "nRadialBins = " << nRadialBins << std::endl;
    //std::cout << "oMask = \n";
    //cv::printMatrix(oMask);
    //cv::Mat oMask_out = cv::getUniqueColorMap(oMask);
    //cv::resize(oMask_out,oMask_out,cv::Size(400,400),0,0,cv::INTER_NEAREST);
    //cv::imshow("oMask",oMask_out);
    //cv::waitKey(0);
}

LSS::LSS(int nDescPatchSize, int nDescRadius, int nRadialBins, int nAngularBins, float fStaticNoiseVar, bool bPreProcess) :
        m_bPreProcess(bPreProcess),
        m_nDescPatchSize(nDescPatchSize),
        m_nDescRadius(nDescRadius),
        m_nCorrWinSize(m_nDescRadius*2+m_nDescPatchSize),
        m_nCorrPatchSize(m_nCorrWinSize-m_nDescPatchSize+1),
        m_nRadialBins(nRadialBins),
        m_nAngularBins(nAngularBins),
        m_fStaticNoiseVar(fStaticNoiseVar) {
    lvAssert_(m_nDescPatchSize>0 && (m_nDescPatchSize%2)==1,"invalid parameter");
    lvAssert_(m_nDescRadius>0 && m_nDescRadius>=m_nDescPatchSize,"invalid parameter");
    lvAssert_(m_nRadialBins>0,"invalid parameter");
    lvAssert_(m_nAngularBins>0,"invalid parameter");
    lvAssert_(m_fStaticNoiseVar>0.0f,"invalid parameter");
    ssdesc_genmask(m_nCorrPatchSize,m_nRadialBins,m_nAngularBins,m_oDescLUMap,m_nFirstMaskIdx,m_nLastMaskIdx);
    lvDbgAssert(m_oDescLUMap.cols==m_nCorrPatchSize && m_oDescLUMap.rows==m_nCorrPatchSize);
    lvDbgAssert(m_nFirstMaskIdx>=0 && m_nLastMaskIdx>=m_nFirstMaskIdx);
    m_oCorrMap.create(m_oDescLUMap.size());
    m_oCorrDiffMap.create(m_nCorrPatchSize-1,m_nCorrPatchSize);
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

bool LSS::isUsingIterativeSSD() const {
    return USE_ITERATIVE_SSD;
}

bool LSS::isNormalizingBins() const {
    return USE_POST_NORMALISATION;
}

bool LSS::isPreProcessing() const {
    return m_bPreProcess;
}

void LSS::compute2(const cv::Mat& oImage, cv::Mat_<float>& oDescriptors) {
    ssdescs_impl(oImage,oDescriptors);
}

void LSS::compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescriptors) {
    ssdescs_impl(oImage,voKeypoints,oDescriptors);
}

void LSS::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<cv::Mat_<float>>& voDescCollection) {
    voDescCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],voDescCollection[i]);
}

void LSS::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat_<float>>& voDescCollection) {
    lvAssert_(voImageCollection.size()==vvoPointCollection.size(),"number of images must match number of keypoint lists");
    voDescCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],vvoPointCollection[i],voDescCollection[i]);
}

void LSS::detectAndCompute(cv::InputArray _oImage, cv::InputArray _oMask, std::vector<cv::KeyPoint>& voKeypoints, cv::OutputArray _oDescriptors, bool bUseProvidedKeypoints) {
    cv::Mat oImage = _oImage.getMat();
    lvAssert_(!oImage.empty(),"input image must be non-empty");
    cv::Mat oMask = _oMask.getMat();
    lvAssert_(oMask.empty() || (!oMask.empty() && oMask.size()==oImage.size()),"mask must be empty or of equal size to the input image");
    cv::Mat oDescriptors = _oDescriptors.getMat();
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
        oDescriptors.release();
        return;
    }
    cv::Mat_<float> oDenseDecriptors;
    ssdescs_impl(oImage,oDenseDecriptors);
    lvDbgAssert(oDenseDecriptors.isContinuous() && oDenseDecriptors.type()==CV_32FC1);
    lvDbgAssert(oDenseDecriptors.dims==3 && oDenseDecriptors.size[0]==(oImage.rows-m_nCorrWinSize+1) && oDenseDecriptors.size[1]==(oImage.cols-m_nCorrWinSize+1) && oDenseDecriptors.size[2]==m_nRadialBins*m_nAngularBins);
    oDescriptors.create((int)voKeypoints.size(),m_nRadialBins*m_nAngularBins,CV_32FC1);
    for(size_t nKeyPtIdx=0; nKeyPtIdx<voKeypoints.size(); ++nKeyPtIdx) {
        const int nRowIdx = int(voKeypoints[nKeyPtIdx].pt.y)-m_nCorrWinSize/2;
        const int nColIdx = int(voKeypoints[nKeyPtIdx].pt.x)-m_nCorrWinSize/2;
        const float* pData = (float*)(oDenseDecriptors.data+oDenseDecriptors.step[0]*nRowIdx+oDenseDecriptors.step[1]*nColIdx);
        std::copy_n(pData,m_nRadialBins*m_nAngularBins,oDescriptors.ptr<float>((int)nKeyPtIdx));
    }
}

void LSS::reshapeDesc(cv::Size oSize, cv::Mat& oDescriptors) const {
    lvAssert_(!oDescriptors.empty() && oDescriptors.isContinuous(),"descriptor mat must be non-empty, and continuous");
    lvAssert_(oSize.area()>0 && oDescriptors.total()==size_t(oSize.area()*m_nRadialBins*m_nAngularBins),"bad expected output desc image size");
    lvAssert_(oDescriptors.dims==2 && oDescriptors.rows==oSize.area() && oDescriptors.type()==CV_32FC1,"descriptor mat type must be 2D, and 32FC1");
    const int nDescRows = oSize.height-m_nCorrWinSize+1;
    const int nDescCols = oSize.width-m_nCorrWinSize+1;
    const int anDescDims[3] = {nDescRows,nDescCols,m_nRadialBins*m_nAngularBins};
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

void LSS::calcDistance(const cv::Mat_<float>& oDescriptors1, const cv::Mat_<float>& oDescriptors2, cv::Mat_<float>& oDistances) const {
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

void LSS::ssdescs_impl(const cv::Mat& _oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescriptors) {
    // this method specialization for targeted keypoint description is slower (per keypoint) than the dense description method
    lvAssert_(!_oImage.empty() && ((_oImage.type()==CV_8UC1) || (_oImage.type()==CV_8UC3)),"invalid input image");
    lvAssert_(m_nCorrWinSize<_oImage.cols && m_nCorrWinSize<_oImage.rows,"image is too small to compute descriptors with current correlation area size");
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
    std::vector<float> aTempDesc(size_t(m_nRadialBins*m_nAngularBins));
    cv::Mat_<float> oTempDesc(1,m_nRadialBins*m_nAngularBins,aTempDesc.data());
    oDescriptors.create(int(voKeypoints.size()),m_nRadialBins*m_nAngularBins);
    for(int nKeyPtIdx=0; nKeyPtIdx<int(voKeypoints.size()); ++nKeyPtIdx) {
        const cv::KeyPoint& oCurrKeyPt = voKeypoints[nKeyPtIdx];
        const int nRowIdx = int(oCurrKeyPt.pt.y)-m_nCorrWinSize/2;
        const int nColIdx = int(oCurrKeyPt.pt.x)-m_nCorrWinSize/2;
        lvDbgAssert(nRowIdx>=0 && nColIdx>=0);
        const cv::Mat oWindow = oImage(cv::Rect(nColIdx,nRowIdx,m_nCorrWinSize,m_nCorrWinSize));
        const cv::Mat oTempl = oImage(cv::Rect(nColIdx+m_nCorrWinSize/2-m_nDescPatchSize/2,nRowIdx+m_nCorrWinSize/2-m_nDescPatchSize/2,m_nDescPatchSize,m_nDescPatchSize));
        cv::matchTemplate(oWindow,oTempl,m_oCorrMap,cv::TM_SQDIFF);
#if USE_STATIC_VAR_NOISE
        const float fVarNormFact = -1.0f/m_fStaticNoiseVar;
#else //!USE_STATIC_VAR_NOISE
        float fMaxLocalVarNoise = 1000.0f;
        for(int nRowOffset=-1; nRowOffset<=1 ; ++nRowOffset)
            for(int nColOffset=-1; nColOffset<=1; ++nColOffset)
                fMaxLocalVarNoise = std::max(fMaxLocalVarNoise,m_oCorrMap(m_nCorrPatchSize/2+nRowOffset,m_nCorrPatchSize/2+nColOffset));
        const float fVarNormFact = -1.0f/fMaxLocalVarNoise;
#endif //!USE_STATIC_VAR_NOISE
        oTempDesc = std::numeric_limits<float>::max();
        for(int nDescBinIdx=m_nFirstMaskIdx; nDescBinIdx<=m_nLastMaskIdx; ++nDescBinIdx)
            if(m_oDescLUMap(nDescBinIdx)!=-1)
                aTempDesc[m_oDescLUMap(nDescBinIdx)] = std::min(aTempDesc[m_oDescLUMap(nDescBinIdx)],((float*)m_oCorrMap.data)[nDescBinIdx]);
        oTempDesc *= fVarNormFact;
        cv::exp(oTempDesc,cv::Mat_<float>(1,m_nRadialBins*m_nAngularBins,oDescriptors.ptr<float>(nKeyPtIdx)));
    }
#if USE_POST_NORMALISATION
    ssdescs_norm(oDescriptors);
#endif //USE_POST_NORMALISATION
}

void LSS::ssdescs_impl(const cv::Mat& _oImage, cv::Mat_<float>& oDescriptors) {
    lvAssert_(!_oImage.empty() && ((_oImage.type()==CV_8UC1) || (_oImage.type()==CV_8UC3)),"invalid input image");
    lvAssert_(m_nCorrWinSize<_oImage.cols && m_nCorrWinSize<_oImage.rows,"image is too small to compute descriptors with current correlation area size");
    cv::Mat oImage;
    if(m_bPreProcess)
        cv::GaussianBlur(_oImage,oImage,cv::Size(7,7),1.0);
    else
        oImage = _oImage;
    const int nDescRows = oImage.rows-m_nCorrWinSize+1;
    const int nDescCols = oImage.cols-m_nCorrWinSize+1;
    std::vector<float> aTempDesc(size_t(m_nRadialBins*m_nAngularBins));
    cv::Mat_<float> oTempDesc(1,m_nRadialBins*m_nAngularBins,aTempDesc.data());
    const int anDescDims[3] = {nDescRows,nDescCols,m_nRadialBins*m_nAngularBins};
    oDescriptors.create(3,anDescDims);
    m_oFullColCorrMap.create(m_nCorrPatchSize+nDescRows,m_nCorrPatchSize);
    const auto lCompDescr = [&](int nColIdx, int nRowIdx, const cv::Mat_<float>& oCorrMap) {
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
                aTempDesc[m_oDescLUMap(nDescBinIdx)] = std::min(aTempDesc[m_oDescLUMap(nDescBinIdx)],((float*)oCorrMap.data)[nDescBinIdx]);
        oTempDesc *= fVarNormFact;
        cv::exp(oTempDesc,cv::Mat_<float>(1,m_nRadialBins*m_nAngularBins,oDescriptors.ptr<float>(nRowIdx,nColIdx,0)));
    };
    for(int nColIdx=0; nColIdx<nDescCols; ++nColIdx) {
        const cv::Mat oInitWindow = oImage(cv::Rect(nColIdx,0,m_nCorrWinSize,m_nCorrWinSize));
        const cv::Mat oInitTempl = oImage(cv::Rect(nColIdx+m_nCorrWinSize/2-m_nDescPatchSize/2,m_nCorrWinSize/2-m_nDescPatchSize/2,m_nDescPatchSize,m_nDescPatchSize));
        cv::Mat_<float> oCorrMap(m_nCorrPatchSize,m_nCorrPatchSize,(float*)m_oFullColCorrMap.data);
        cv::matchTemplate(oInitWindow,oInitTempl,oCorrMap,cv::TM_SQDIFF);
        lCompDescr(nColIdx,0,oCorrMap);
        for(int nRowIdx=1; nRowIdx<nDescRows; ++nRowIdx) {
            const cv::Mat oWindow = oImage(cv::Rect(nColIdx,nRowIdx,m_nCorrWinSize,m_nCorrWinSize));
            const cv::Mat oTempl = oImage(cv::Rect(nColIdx+m_nCorrWinSize/2-m_nDescPatchSize/2,nRowIdx+m_nCorrWinSize/2-m_nDescPatchSize/2,m_nDescPatchSize,m_nDescPatchSize));
#if USE_ITERATIVE_SSD
            oCorrMap = cv::Mat_<float>(m_nCorrPatchSize,m_nCorrPatchSize,((float*)m_oFullColCorrMap.data)+m_nCorrPatchSize*nRowIdx);
            const cv::Mat oTemplMinus = oImage(cv::Rect(nColIdx+m_nCorrWinSize/2-m_nDescPatchSize/2,(nRowIdx-1)+m_nCorrWinSize/2-m_nDescPatchSize/2,m_nDescPatchSize,1));
            cv::matchTemplate(oWindow.rowRange(m_nDescPatchSize/2,m_nCorrWinSize-m_nDescPatchSize/2-1),oTemplMinus,m_oCorrDiffMap,cv::TM_SQDIFF);
            oCorrMap.rowRange(0,m_nCorrPatchSize-1) -= m_oCorrDiffMap;
            const cv::Mat oTemplPlus = oImage(cv::Rect(nColIdx+m_nCorrWinSize/2-m_nDescPatchSize/2,(nRowIdx-1+m_nDescPatchSize)+m_nCorrWinSize/2-m_nDescPatchSize/2,m_nDescPatchSize,1));
            cv::matchTemplate(oWindow.rowRange(m_nDescPatchSize/2,m_nCorrWinSize-m_nDescPatchSize/2-1),oTemplPlus,m_oCorrDiffMap,cv::TM_SQDIFF);
            oCorrMap.rowRange(0,m_nCorrPatchSize-1) += m_oCorrDiffMap;
            cv::matchTemplate(oWindow.rowRange(m_nCorrWinSize-m_nDescPatchSize,m_nCorrWinSize),oTempl,oCorrMap.row(m_nCorrPatchSize-1),cv::TM_SQDIFF);
#else //!USE_ITERATIVE_SSD
            cv::matchTemplate(oWindow,oTempl,oCorrMap,cv::TM_SQDIFF);
#endif //!USE_ITERATIVE_SSD
            lCompDescr(nColIdx,nRowIdx,oCorrMap);
        }
    }
#if USE_POST_NORMALISATION
    ssdescs_norm(oDescriptors);
#endif //USE_POST_NORMALISATION
}

void LSS::ssdescs_norm(cv::Mat_<float>& oDescriptors) const {
    if(oDescriptors.empty())
        return;
    lvDbgAssert((oDescriptors.total()%size_t(m_nRadialBins*m_nAngularBins))==0);
    lvDbgAssert(oDescriptors.size[oDescriptors.dims-1]==m_nRadialBins*m_nAngularBins);
    lvDbgAssert(oDescriptors.isContinuous());
    for(size_t nDescIdx=0; nDescIdx<oDescriptors.total(); nDescIdx+=size_t(m_nRadialBins*m_nAngularBins)) {
        float fMin = std::numeric_limits<float>::max(), fMax = std::numeric_limits<float>::min();
        float* pfCurrDesc = ((float*)oDescriptors.data)+nDescIdx;
        for(int nDescBinIdx=0; nDescBinIdx<m_nRadialBins*m_nAngularBins; ++nDescBinIdx) {
            fMin = std::min(fMin,pfCurrDesc[nDescBinIdx]);
            fMax = std::max(fMax,pfCurrDesc[nDescBinIdx]);
        }
        for(int nDescBinIdx=0; nDescBinIdx<m_nRadialBins*m_nAngularBins; ++nDescBinIdx)
            pfCurrDesc[nDescBinIdx] = (pfCurrDesc[nDescBinIdx]-fMin)/(fMax-fMin);
    }
}