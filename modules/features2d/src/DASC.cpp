
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

#include "litiv/features2d.hpp"

namespace pretrained { // obtained via middlebury dataset (imported here from original mat archives)

    constexpr size_t nLUTSize = 128;

    constexpr std::array<int,nLUTSize*2> anRP1 = {
        5,-9,-14,-5,10,0,-12,4,-4,-6,-13,-7,-6,-11,5,14,3,9,10,11,10,-2,15,3,6,5,14,5,14,-5,-9,-5,
        8,10,-8,-10,-3,-15,-11,6,12,4,9,3,-7,13,3,-15,-15,3,13,0,4,-12,-6,-11,12,-2,-4,-12,-5,14,-3,-15,
        10,-11,11,10,-10,0,-11,-10,-9,5,8,6,-12,-4,13,7,15,0,-15,3,5,14,-11,10,5,-6,-13,-7,3,-15,-9,-5,
        -6,-8,0,13,-7,13,-9,3,0,15,11,-6,10,11,5,14,-11,10,14,-5,8,10,0,15,-5,14,11,6,8,-6,-7,13,
        -13,0,-11,-6,-8,-13,2,12,-7,13,4,6,-10,-11,-6,-11,-10,-11,5,-9,-14,-5,-3,-15,-4,12,10,-2,13,7,-15,0,
        10,2,5,14,-5,-14,-13,0,5,-14,0,5,10,8,-12,4,-12,4,-10,8,-15,-3,14,5,-10,-8,15,0,-2,-10,-7,1,
        0,15,14,-5,3,-9,-13,0,0,15,2,12,-10,8,13,7,-5,-6,5,14,15,3,-8,-13,-9,3,-5,9,5,-9,-0,-13,
        6,-11,15,3,-0,-15,13,7,6,11,-2,10,-15,-3,6,-8,11,-6,-14,5,-5,-6,3,-7,-15,0,6,-8,14,5,0,13,
    };

    constexpr std::array<int,nLUTSize*2> anRP2 = {
        -0,-13,1,7,12,2,-10,-11,-3,-9,0,15,-11,10,-15,-3,-2,-12,-15,3,5,14,-12,2,15,0,-11,-6,-8,-6,-2,10,
        -0,-8,-8,6,-4,-12,-9,-5,10,-8,5,-6,11,10,-13,0,12,2,-8,-13,-11,10,10,8,-10,11,-0,-8,4,12,11,10,
        11,-10,6,-11,-3,9,-11,6,-14,5,8,13,-0,-15,4,12,14,5,5,14,-10,-2,13,0,-15,-3,-3,15,7,-3,-14,5,
        13,7,4,-6,13,-8,-15,3,0,15,-14,-5,-14,-5,14,-5,5,-9,11,-6,-3,15,-10,-8,-3,15,14,5,10,-2,5,14,
        -3,-15,6,4,0,15,-5,14,11,6,-3,-7,-5,14,12,-4,8,-13,9,-3,7,1,-14,-5,-8,-10,8,10,0,15,-5,-14,
        -0,-15,-8,6,-7,-3,8,6,-0,-10,-7,-3,-11,-10,8,10,10,11,-15,0,-11,6,-2,10,-12,2,15,0,-10,8,10,-11,
        3,-7,-6,-8,9,3,5,-2,8,10,10,-2,14,-5,-10,-11,-13,-7,0,15,-6,11,-10,8,5,9,-3,15,-8,-6,-5,-14,
        3,15,12,-2,-4,-12,3,-4,8,-13,-0,-13,-0,-10,10,11,-5,-6,-13,7,-8,-10,-11,-10,5,14,8,-13,8,10,-15,-3,
    };

    constexpr int static_diff(int a, int b) {return b-a;}
    constexpr int static_abs(int a) {return (a<0)?-a:a;}
    constexpr int static_absmax(int a, int b) {return std::max(static_abs(a),static_abs(b));}
    constexpr std::array<int,nLUTSize*2> anRPDiff = lv::static_transform(anRP1,anRP2,static_diff);
    constexpr int nRPAbsMax = lv::static_reduce(lv::static_transform(anRP1,anRP2,static_absmax),static_absmax);

} // namespace pretrained

DASC::DASC(float fSigma_s, float fSigma_r, size_t nIters, bool bPreProcess) :
        m_bUsingRF(true),
        m_bPreProcess(bPreProcess),
        m_fSigma_s(fSigma_s),
        m_fSigma_r(fSigma_r),
        m_nIters(nIters),
        m_nRadius(),
        m_fEpsilon(),
        m_nSubSamplFrac() {
    lvAssert_(fSigma_s>0.0f && fSigma_r>0.0f && nIters>0,"invalid parameter(s)");
}

DASC::DASC(size_t nRadius, float fEpsilon, size_t nSubSamplFrac, bool bPreProcess) :
        m_bUsingRF(false),
        m_bPreProcess(bPreProcess),
        m_fSigma_s(),
        m_fSigma_r(),
        m_nIters(),
        m_nRadius(nRadius),
        m_fEpsilon(fEpsilon),
        m_nSubSamplFrac(nSubSamplFrac) {
    lvAssert_(nRadius>0 && fEpsilon>0.0f && nSubSamplFrac>0 && nRadius>=nSubSamplFrac,"invalid parameter(s)");
}

void DASC::read(const cv::FileNode& /*fn*/) {
    // ... = fn["..."];
}

void DASC::write(cv::FileStorage& /*fs*/) const {
    //fs << "..." << ...;
}

int DASC::descriptorSize() const {
    return pretrained::nLUTSize*sizeof(float);
}

int DASC::descriptorType() const {
    return CV_32F;
}

int DASC::defaultNorm() const {
    return cv::NORM_L2;
}

bool DASC::empty() const {
    return true;
}

bool DASC::isUsingRF() const {
    return m_bUsingRF;
}

bool DASC::isPreProcessing() const {
    return m_bPreProcess;
}

void DASC::compute2(const cv::Mat& oImage, cv::Mat_<float>& oDescriptors) {
    lvAssert_(!oImage.empty(),"input image must be non-empty");
    if(m_bUsingRF)
        dasc_rf_impl(oImage,oDescriptors);
    else
        dasc_gf_impl(oImage,oDescriptors);
}

void DASC::compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat_<float>& oDescriptors) {
    lvAssert_(!oImage.empty(),"input image must be non-empty");
    voKeypoints.clear();
    voKeypoints.reserve(size_t(oImage.rows*oImage.cols));
    for(int nRowIdx=0; nRowIdx<oImage.rows; ++nRowIdx)
        for(int nColIdx=0; nColIdx<oImage.cols; ++nColIdx)
            voKeypoints.emplace_back(cv::Point2f((float)nColIdx,(float)nRowIdx),(float)pretrained::nRPAbsMax*2);
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImage.size(),pretrained::nRPAbsMax);
    if(m_bUsingRF)
        dasc_rf_impl(oImage,oDescriptors);
    else
        dasc_gf_impl(oImage,oDescriptors);
}

void DASC::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<cv::Mat_<float>>& voDescCollection) {
    voDescCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],voDescCollection[i]);
}

void DASC::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat_<float>>& voDescCollection) {
    lvAssert_(voImageCollection.size()==vvoPointCollection.size(),"number of images must match number of keypoint lists");
    voDescCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i],vvoPointCollection[i],voDescCollection[i]);
}

void DASC::detectAndCompute(cv::InputArray _oImage, cv::InputArray _oMask, std::vector<cv::KeyPoint>& voKeypoints, cv::OutputArray _oDescriptors, bool bUseProvidedKeypoints) {
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
                voKeypoints.emplace_back(cv::Point2f((float)nColIdx,(float)nRowIdx),(float)pretrained::nRPAbsMax*2);
    }
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImage.size(),pretrained::nRPAbsMax);
    if(!oMask.empty())
        cv::KeyPointsFilter::runByPixelsMask(voKeypoints,oMask);
    if(voKeypoints.empty()) {
        oDescriptors.release();
        return;
    }
    cv::Mat_<float> oDenseDecriptors;
    if(m_bUsingRF)
        dasc_rf_impl(oImage,oDenseDecriptors);
    else
        dasc_gf_impl(oImage,oDenseDecriptors);
    lvDbgAssert(oDenseDecriptors.isContinuous() && oDenseDecriptors.type()==CV_32FC1);
    lvDbgAssert(oDenseDecriptors.dims==3 && oDenseDecriptors.size[0]==oImage.rows && oDenseDecriptors.size[1]==oImage.cols && oDenseDecriptors.size[2]==pretrained::nLUTSize);
    oDescriptors.create((int)voKeypoints.size(),pretrained::nLUTSize,CV_32FC1);
    for(size_t nKeyPtIdx=0; nKeyPtIdx<voKeypoints.size(); ++nKeyPtIdx) {
        const int nRowIdx = (int)voKeypoints[nKeyPtIdx].pt.y;
        const int nColIdx = (int)voKeypoints[nKeyPtIdx].pt.x;
        const float* pData = (float*)(oDenseDecriptors.data+oDenseDecriptors.step[0]*nRowIdx+oDenseDecriptors.step[1]*nColIdx);
        std::copy_n(pData,pretrained::nLUTSize,oDescriptors.ptr<float>((int)nKeyPtIdx));
    }
}

void DASC::reshapeDesc(cv::Size oSize, cv::Mat& oDescriptors) {
    lvAssert_(!oDescriptors.empty() && oDescriptors.isContinuous(),"descriptor mat must be non-empty, and continuous");
    lvAssert_(oSize.area()>0 && oDescriptors.total()==(size_t)oSize.area()*pretrained::nLUTSize,"bad expected output desc image size");
    lvAssert_(oDescriptors.dims==2 && oDescriptors.rows==oSize.area() && oDescriptors.type()==CV_32FC1,"descriptor mat type must be 2D, and 32FC1");
    std::array<int,3> anDims = {oSize.height,oSize.width,pretrained::nLUTSize};
    oDescriptors = oDescriptors.reshape(0,(int)anDims.size(),anDims.data());
}

void DASC::validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) {
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImgSize,pretrained::nRPAbsMax);
}

void DASC::validateROI(cv::Mat& oROI) {
    lvAssert_(!oROI.empty() && oROI.type()==CV_8UC1,"input ROI must be non-empty and of type 8UC1");
    cv::Mat oROI_new(oROI.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    const int nBorderSize = pretrained::nRPAbsMax;
    const cv::Rect nROI_inner(nBorderSize,nBorderSize,oROI.cols-nBorderSize*2,oROI.rows-nBorderSize*2);
    cv::Mat(oROI,nROI_inner).copyTo(cv::Mat(oROI_new,nROI_inner));
    oROI = oROI_new;
}

void DASC::recursFilter(const cv::Mat_<float>& oImage, const cv::Mat_<float>& oRef_V_dHdx, const cv::Mat_<float>& oRef_V_dVdy_t, cv::Mat_<float>& oOutput) {
    lvDbgAssert(!oImage.empty() && !oRef_V_dHdx.empty() && !oRef_V_dHdx.empty() && m_nIters>0 && oImage.dims==2 && oRef_V_dHdx.dims==3 && oRef_V_dVdy_t.dims==3);
    lvDbgAssert(oImage.rows==oRef_V_dHdx.size[1] && oImage.rows==oRef_V_dVdy_t.size[2] && oImage.cols==oRef_V_dHdx.size[2] && oImage.cols==oRef_V_dVdy_t.size[1]);
    lvDbgAssert(oRef_V_dHdx.size[0]==(int)m_nIters && oRef_V_dVdy_t.size[0]==(int)m_nIters);
    oImage.copyTo(oOutput);
    const auto lTransfDomRecursFilter_H = [](cv::Mat_<float>& _oImage, const cv::Mat_<float>& oRef, int nIterIdx) {
        lvDbgAssert(!_oImage.empty() && _oImage.dims==2 && !oRef.empty() && oRef.dims==3);
        lvDbgAssert(_oImage.rows==oRef.size[1] && _oImage.cols==oRef.size[2]);
        for(int nRowIdx=0; nRowIdx<_oImage.rows; ++nRowIdx) {
            for(int nColIdx=1; nColIdx<_oImage.cols; ++nColIdx)
                _oImage(nRowIdx,nColIdx) += oRef(nIterIdx,nRowIdx,nColIdx)*(_oImage(nRowIdx,nColIdx-1)-_oImage(nRowIdx,nColIdx));
            for(int nColIdx=_oImage.cols-2; nColIdx>=0; --nColIdx)
                _oImage(nRowIdx,nColIdx) += oRef(nIterIdx,nRowIdx,nColIdx+1)*(_oImage(nRowIdx,nColIdx+1)-_oImage(nRowIdx,nColIdx));
        }
    };
    for(int nIterIdx=0; nIterIdx<(int)m_nIters; ++nIterIdx) {
        lTransfDomRecursFilter_H(oOutput,oRef_V_dHdx,nIterIdx);
        cv::transpose(oOutput,m_oTempTransp);
        lTransfDomRecursFilter_H(m_oTempTransp,oRef_V_dVdy_t,nIterIdx);
        cv::transpose(m_oTempTransp,oOutput);
    }
}

void DASC::dasc_rf_impl(const cv::Mat& _oImage, cv::Mat_<float>& oDescriptors) {
    lvAssert(!_oImage.empty() && (_oImage.channels()==1 || _oImage.channels()==3) && (_oImage.depth()==CV_32F || _oImage.depth()==CV_8U));
    cv::Mat oImageTemp;
    if(_oImage.depth()==CV_8U)
        _oImage.convertTo(oImageTemp,CV_32F,1.0/UCHAR_MAX);
    else
        oImageTemp = _oImage;
    if(oImageTemp.channels()==3)
        cv::cvtColor(oImageTemp,oImageTemp,cv::COLOR_BGR2GRAY);
    lvDbgAssert(cv::countNonZero((oImageTemp>1.0f)|(oImageTemp<0.0f))==0);
    cv::Mat_<float> oImage = oImageTemp;
    if(m_bPreProcess)
        cv::GaussianBlur(oImage,oImage,cv::Size(7,7),1.0);
    m_oImageSize = oImage.size();
    const int nRows = m_oImageSize.height;
    const int nCols = m_oImageSize.width;
    lv::localDiff<1,0>(oImage,m_oImageLocalDiff_Y);
    lv::localDiff<0,1>(oImage,m_oImageLocalDiff_X);
    m_oRef_dVdy = 1.0f + m_fSigma_s/m_fSigma_r*cv::abs(m_oImageLocalDiff_Y);
    m_oRef_dHdx = 1.0f + m_fSigma_s/m_fSigma_r*cv::abs(m_oImageLocalDiff_X);
    const std::array<int,3> anRefDims = {(int)m_nIters,nRows,nCols};
    m_oRef_V_dHdx.create(3,anRefDims.data());
    const std::array<int,3> anRefDims_t = {(int)m_nIters,nCols,nRows};
    m_oRef_V_dVdy_t.create(3,anRefDims_t.data());
    for(int nIterIdx=0; nIterIdx<(int)m_nIters; ++nIterIdx) {
        const float fBase = exp(-sqrt(2.0f)/(m_fSigma_s*sqrt(3.0f)*(float)pow(2.0f,(int)m_nIters-(nIterIdx+1))/sqrt((float)pow(4.0f,(int)m_nIters)-1)));
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                m_oRef_V_dHdx(nIterIdx,nRowIdx,nColIdx) = pow(fBase,m_oRef_dHdx(nRowIdx,nColIdx));
                m_oRef_V_dVdy_t(nIterIdx,nColIdx,nRowIdx) = pow(fBase,m_oRef_dVdy(nRowIdx,nColIdx));
            }
        }
    }
    recursFilter(oImage,m_oRef_V_dHdx,m_oRef_V_dVdy_t,m_oImage_AdaptiveMean);
    recursFilter(oImage.mul(oImage),m_oRef_V_dHdx,m_oRef_V_dVdy_t,m_oImage_AdaptiveMeanSqr);
    m_oLookupImage.create(m_oImageSize);
    m_oLookupImage_Sqr.create(m_oImageSize);
    m_oLookupImage_Mix.create(m_oImageSize);
    const std::array<int,3> anDescDims = {nRows,nCols,(int)pretrained::nLUTSize};
    oDescriptors.create(3,anDescDims.data());
    for(int nLUTIdx=0; nLUTIdx<(int)pretrained::nLUTSize; nLUTIdx++) {
        const int nRowOffset = pretrained::anRPDiff[nLUTIdx*2];
        const int nColOffset = pretrained::anRPDiff[nLUTIdx*2+1];
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                if(nRowIdx+nRowOffset>=0 && nRowIdx+nRowOffset<nRows && nColIdx+nColOffset>=0 && nColIdx+nColOffset<nCols) {
                    m_oLookupImage(nRowIdx,nColIdx) = oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                    m_oLookupImage_Sqr(nRowIdx,nColIdx) = oImage(nRowIdx+nRowOffset,nColIdx+nColOffset)*oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                    m_oLookupImage_Mix(nRowIdx,nColIdx) = oImage(nRowIdx,nColIdx)*oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                }
                else
                    m_oLookupImage(nRowIdx,nColIdx) = m_oLookupImage_Sqr(nRowIdx,nColIdx) = m_oLookupImage_Mix(nRowIdx,nColIdx) = 0.0f;
            }
        }
        recursFilter(m_oLookupImage,m_oRef_V_dHdx,m_oRef_V_dVdy_t,m_oLookupImage_AdaptiveMean);
        recursFilter(m_oLookupImage_Sqr,m_oRef_V_dHdx,m_oRef_V_dVdy_t,m_oLookupImage_AdaptiveMeanSqr);
        recursFilter(m_oLookupImage_Mix,m_oRef_V_dHdx,m_oRef_V_dVdy_t,m_oLookupImage_AdaptiveMeanMix);
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx = 0; nColIdx<nCols; ++nColIdx) {
                const int nOffsetRowIdx = nRowIdx+pretrained::anRP1[nLUTIdx*2];
                const int nOffsetColIdx = nColIdx+pretrained::anRP1[nLUTIdx*2+1];
                if(nOffsetRowIdx>0 && nOffsetRowIdx<nRows && nOffsetColIdx>0 && nOffsetColIdx<nCols) {
                    const float fCorrSurfDenom = sqrt((m_oImage_AdaptiveMeanSqr(nOffsetRowIdx,nOffsetColIdx)-m_oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*m_oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)) * (m_oLookupImage_AdaptiveMeanSqr(nOffsetRowIdx,nOffsetColIdx)-m_oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*m_oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)));
                    if(fCorrSurfDenom>1e-10)
                        oDescriptors(nRowIdx,nColIdx,nLUTIdx) = exp(-(1-(m_oLookupImage_AdaptiveMeanMix(nOffsetRowIdx,nOffsetColIdx)-m_oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*m_oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx))/fCorrSurfDenom)/0.5f);
                    else
                        oDescriptors(nRowIdx,nColIdx,nLUTIdx) = 1.0f;
                }
                else
                    oDescriptors(nRowIdx,nColIdx,nLUTIdx) = 0.0f;
            }
        }
    }
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float fNorm = 0;
            for(int nLUTIdx=0; nLUTIdx<(int)pretrained::nLUTSize; ++nLUTIdx)
                fNorm += oDescriptors(nRowIdx,nColIdx,nLUTIdx)*oDescriptors(nRowIdx,nColIdx,nLUTIdx);
            const float fNormSqrt = sqrt(fNorm);
            for(int nLUTIdx=0; nLUTIdx<(int)pretrained::nLUTSize; ++nLUTIdx)
                oDescriptors(nRowIdx,nColIdx,nLUTIdx) = (fNormSqrt>1e-10)?oDescriptors(nRowIdx,nColIdx,nLUTIdx)/fNormSqrt:0.0f;
        }
    }
}

void DASC::guidedFilter(const cv::Mat_<float>& oImage, const cv::Mat_<float>& oRef, cv::Mat_<float>& oOutput) {
    lvDbgAssert(!oImage.empty() && !oRef.empty());
    cv::resize(oRef,m_oRef_SubSampl,m_oSubSamplSize,0.0,0.0,cv::INTER_NEAREST);
	m_oRef_SubSamplCross = m_oImage_SubSampl.mul(m_oRef_SubSampl);
    cv::blur(m_oRef_SubSampl,m_oRef_SubSamplBlur,m_oBlurKernelSize);
    cv::blur(m_oRef_SubSamplCross,m_oRef_SubSamplCrossBlur,m_oBlurKernelSize);
	m_oNormVar_SubSampl = (m_oRef_SubSamplCrossBlur-m_oImage_SubSamplBlur.mul(m_oRef_SubSamplBlur))/m_oImage_SubSamplVar;
	m_oNormVarDiff_SubSampl = m_oRef_SubSamplBlur - m_oNormVar_SubSampl.mul(m_oImage_SubSamplBlur);
    cv::blur(m_oNormVar_SubSampl,m_oNormVar_SubSamplBlur,m_oBlurKernelSize);
    cv::blur(m_oNormVarDiff_SubSampl,m_oNormVarDiff_SubSamplBlur,m_oBlurKernelSize);
    cv::resize(m_oNormVar_SubSamplBlur,m_oNormVar,m_oImageSize,0,0,cv::INTER_LINEAR);
    cv::resize(m_oNormVarDiff_SubSamplBlur,m_oNormVarDiff,m_oImageSize,0,0,cv::INTER_LINEAR);
	oOutput = m_oNormVar.mul(oImage)+m_oNormVarDiff;
}

void DASC::dasc_gf_impl(const cv::Mat& _oImage, cv::Mat_<float>& oDescriptors) {
    lvAssert(!_oImage.empty() && (_oImage.channels()==1 || _oImage.channels()==3) && (_oImage.depth()==CV_32F || _oImage.depth()==CV_8U));
    cv::Mat oImageTemp;
    if(_oImage.depth()==CV_8U)
        _oImage.convertTo(oImageTemp,CV_32F,1.0/UCHAR_MAX);
    else
        oImageTemp = _oImage;
    if(oImageTemp.channels()==3)
        cv::cvtColor(oImageTemp,oImageTemp,cv::COLOR_BGR2GRAY);
    lvDbgAssert(cv::countNonZero((oImageTemp>1.0f)|(oImageTemp<0.0f))==0);
    cv::Mat_<float> oImage = oImageTemp;
    if(m_bPreProcess)
        cv::GaussianBlur(oImage,oImage,cv::Size(7,7),1.0);
    m_oImageSize = oImage.size();
    lvAssert(m_oImageSize.area()>0);
    m_oSubSamplSize = cv::Size(int(m_oImageSize.width/m_nSubSamplFrac),int(m_oImageSize.height/m_nSubSamplFrac));
    lvAssert(m_oSubSamplSize.area()>0);
    const int nKernelRadius = (int)(m_nRadius/m_nSubSamplFrac);
    lvAssert(nKernelRadius>0);
    m_oBlurKernelSize = cv::Size(2*nKernelRadius+1,2*nKernelRadius+1);
    const int nRows = m_oImageSize.height;
    const int nCols = m_oImageSize.width;
    cv::resize(oImage,m_oImage_SubSampl,m_oSubSamplSize,0.0,0.0,cv::INTER_NEAREST);
    cv::blur(m_oImage_SubSampl,m_oImage_SubSamplBlur,m_oBlurKernelSize);
    cv::blur(m_oImage_SubSampl.mul(m_oImage_SubSampl),m_oImage_SubSamplBlurSqr,m_oBlurKernelSize);
    m_oImage_SubSamplVar = m_oImage_SubSamplBlurSqr-m_oImage_SubSamplBlur.mul(m_oImage_SubSamplBlur)+m_fEpsilon;
	guidedFilter(oImage,oImage,m_oImage_AdaptiveMean);
	guidedFilter(oImage,oImage.mul(oImage),m_oImage_AdaptiveMeanSqr);
    m_oLookupImage.create(m_oImageSize);
    m_oLookupImage_Sqr.create(m_oImageSize);
    m_oLookupImage_Mix.create(m_oImageSize);
    const std::array<int,3> anDescDims = {nRows,nCols,(int)pretrained::nLUTSize};
    oDescriptors.create(3,anDescDims.data());
    for(int nLUTIdx=0; nLUTIdx<(int)pretrained::nLUTSize; nLUTIdx++) {
        const int nRowOffset = pretrained::anRPDiff[nLUTIdx*2];
        const int nColOffset = pretrained::anRPDiff[nLUTIdx*2+1];
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                if(nRowIdx+nRowOffset>=0 && nRowIdx+nRowOffset<nRows && nColIdx+nColOffset>=0 && nColIdx+nColOffset<nCols) {
                    m_oLookupImage(nRowIdx,nColIdx) = oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                    m_oLookupImage_Sqr(nRowIdx,nColIdx) = oImage(nRowIdx+nRowOffset,nColIdx+nColOffset)*oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                    m_oLookupImage_Mix(nRowIdx,nColIdx) = oImage(nRowIdx,nColIdx)*oImage(nRowIdx+nRowOffset,nColIdx+nColOffset);
                }
                else
                    m_oLookupImage(nRowIdx,nColIdx) = m_oLookupImage_Sqr(nRowIdx,nColIdx) = m_oLookupImage_Mix(nRowIdx,nColIdx) = 0.0f;
            }
        }
		guidedFilter(oImage,m_oLookupImage,m_oLookupImage_AdaptiveMean);
		guidedFilter(oImage,m_oLookupImage_Sqr,m_oLookupImage_AdaptiveMeanSqr);
		guidedFilter(oImage,m_oLookupImage_Mix,m_oLookupImage_AdaptiveMeanMix);
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx = 0; nColIdx<nCols; ++nColIdx) {
                const int nOffsetRowIdx = nRowIdx+pretrained::anRP1[nLUTIdx*2];
                const int nOffsetColIdx = nColIdx+pretrained::anRP1[nLUTIdx*2+1];
                if(nOffsetRowIdx>0 && nOffsetRowIdx<nRows && nOffsetColIdx>0 && nOffsetColIdx<nCols) {
                    const float fCorrSurfDenom = sqrt((m_oImage_AdaptiveMeanSqr(nOffsetRowIdx,nOffsetColIdx)-m_oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*m_oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)) * (m_oLookupImage_AdaptiveMeanSqr(nOffsetRowIdx,nOffsetColIdx)-m_oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*m_oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)));
                    if(fCorrSurfDenom>1e-10)
                        oDescriptors(nRowIdx,nColIdx,nLUTIdx) = exp(-(1-(m_oLookupImage_AdaptiveMeanMix(nOffsetRowIdx,nOffsetColIdx)-m_oImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx)*m_oLookupImage_AdaptiveMean(nOffsetRowIdx,nOffsetColIdx))/fCorrSurfDenom)/0.5f);
                    else
                        oDescriptors(nRowIdx,nColIdx,nLUTIdx) = 1.0f;
                }
                else
                    oDescriptors(nRowIdx,nColIdx,nLUTIdx) = 0.0f;
            }
        }
	}
	for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            float fNorm = 0;
            for(int nLUTIdx=0; nLUTIdx<(int)pretrained::nLUTSize; ++nLUTIdx)
                fNorm += oDescriptors(nRowIdx,nColIdx,nLUTIdx)*oDescriptors(nRowIdx,nColIdx,nLUTIdx);
            const float fNormSqrt = sqrt(fNorm);
            for(int nLUTIdx=0; nLUTIdx<(int)pretrained::nLUTSize; ++nLUTIdx)
                oDescriptors(nRowIdx,nColIdx,nLUTIdx) = (fNormSqrt>1e-10)?oDescriptors(nRowIdx,nColIdx,nLUTIdx)/fNormSqrt:0.0f;
        }
    }
}