
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/imgproc/EdgeDetectorCanny.hpp"
#include "litiv/imgproc/EdgeDetectorLBSP.hpp"
#include "litiv/imgproc/CosegmentationUtils.hpp"
#if HAVE_OPENGM
//#include "litiv/imgproc/ForegroundStereoMatcher.hpp"
#endif //HAVE_OPENGM
#if HAVE_CUDA
#include "litiv/imgproc/SLIC.hpp"
#endif //HAVE_CUDA

namespace lv {

    /// possible implementation modes for lv::thinning
    enum ThinningMode {
        ThinningMode_ZhangSuen=0,
        ThinningMode_LamLeeSuen
    };

    enum AffinityDistType {
        AffinityDist_L2=0,
        AffinityDist_EMD,
        AffinityDist_MI,
        AffinityDist_SSD
    };

    /// 'thins' the provided image (currently only works on 1ch 8UC1 images, treated as binary)
    void thinning(const cv::Mat& oInput, cv::Mat& oOutput, ThinningMode eMode=ThinningMode_LamLeeSuen);

    /// performs non-maximum suppression on the input image, with a (nWinSize)x(nWinSize) window
    template<int nWinSize>
    void nonMaxSuppression(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oMask=cv::Mat());

    /// computes a 3d affinity map from two images by matching them in patches across a given stereo disparity range
    void computeImageAffinity(const cv::Mat& oImage1, const cv::Mat& oImage2, int nPatchSize,
                              cv::Mat_<float>& oAffinityMap, const std::vector<int>& vDispRange, AffinityDistType eDist,
                              const cv::Mat_<uchar>& oROI1=cv::Mat(), const cv::Mat_<uchar>& oROI2=cv::Mat());

    /// computes a 3d affinity map from two 2d descriptor maps by matching them in patches across a given stereo disparity range
    void computeDescriptorAffinity(const cv::Mat_<float>& oDescMap1, const cv::Mat_<float>& oDescMap2, int nPatchSize,
                                   cv::Mat_<float>& oAffinityMap, const std::vector<int>& vDispRange, AffinityDistType eDist,
                                   const cv::Mat_<uchar>& oROI1=cv::Mat(), const cv::Mat_<uchar>& oROI2=cv::Mat(),
                                   const cv::Mat_<float>& oEMDCostMap=cv::Mat());

    /// determines if '*anMap' is a local maximum on the horizontal axis, given 'nMapColStep' spacing between horizontal elements in 'anMap'
    template<size_t nHalfWinSize, typename Tr>
    bool isLocalMaximum_Horizontal(const Tr* const anMap, const size_t nMapColStep, const size_t /*nMapRowStep*/);
    /// determines if '*anMap' is a local maximum on the vertical axis, given 'nMapRowStep' spacing between vertical elements in 'anMap'
    template<size_t nHalfWinSize, typename Tr>
    bool isLocalMaximum_Vertical(const Tr* const anMap, const size_t /*nMapColStep*/, const size_t nMapRowStep);
    /// determines if '*anMap' is a local maximum on the diagonal, given 'nMapColStep'/'nMapColStep' spacing between horizontal/vertical elements in 'anMap'
    template<size_t nHalfWinSize, bool bInvDiag, typename Tr>
    bool isLocalMaximum_Diagonal(const Tr* const anMap, const size_t nMapColStep, const size_t nMapRowStep);
    /// determines if '*anMap' is a local maximum on the diagonal, given 'nMapColStep'/'nMapColStep' spacing between horizontal/vertical elements in 'anMap'
    template<size_t nHalfWinSize, typename Tr>
    bool isLocalMaximum_Diagonal(const Tr* const anMap, const size_t nMapColStep, const size_t nMapRowStep, bool bInvDiag);

    /// initializes foreground and background GMM parameters via KNN using the given image and mask (where all values >0 are considered foreground)
    template<size_t nKMeansIters=10, size_t nC1, size_t nC2, size_t nD>
    void initGaussianMixtureParams(const cv::Mat& oInput, const cv::Mat& oMask, lv::GMM<nC1,nD>& oBGModel, lv::GMM<nC2,nD>& oFGModel, const cv::Mat& oROI=cv::Mat());
    /// assigns each input image pixel its most likely GMM component in the output map, using the BG or FG model as dictated by the input mask
    template<size_t nC1, size_t nC2, size_t nD>
    void assignGaussianMixtureComponents(const cv::Mat& oInput, const cv::Mat& oMask, cv::Mat& oAssignMap, const lv::GMM<nC1,nD>& oBGModel, const lv::GMM<nC2,nD>& oFGModel, const cv::Mat& oROI=cv::Mat());
    /// learns the ideal foreground and background GMM parameters to fit the components assigned to the pixels of the input image
    template<size_t nC1, size_t nC2, size_t nD>
    void learnGaussianMixtureParams(const cv::Mat& oInput, const cv::Mat& oMask, const cv::Mat& oAssignMap, lv::GMM<nC1,nD>& oBGModel, lv::GMM<nC2,nD>& oFGModel, const cv::Mat& oROI=cv::Mat());

} // namespace lv

template<int nWinSize>
void lv::nonMaxSuppression(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oMask) {
    // reimplemented from http://code.opencv.org/attachments/994/nms.cpp
    lvAssert(oInput.dims==2 && oInput.channels()==1);
    // initialise the block oMask and destination
    const int M = oInput.rows;
    const int N = oInput.cols;
    const bool masked = !oMask.empty();
    cv::Mat block = 255*cv::Mat_<uint8_t>::ones(cv::Size(2*nWinSize+1,2*nWinSize+1));
    oOutput.create(oInput.size(),CV_8UC1);
    oOutput = cv::Scalar_<uchar>(0);
    // iterate over image blocks
    for(int m = 0; m < M; m+=nWinSize+1) {
        for(int n = 0; n < N; n+=nWinSize+1) {
            cv::Point  ijmax;
            double vcmax, vnmax;
            // get the maximal candidate within the block
            cv::Range ic(m,std::min(m+nWinSize+1,M));
            cv::Range jc(n,std::min(n+nWinSize+1,N));
            cv::minMaxLoc(oInput(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? oMask(ic,jc) : cv::noArray());
            cv::Point cc = ijmax + cv::Point(jc.start,ic.start);
            // search the neighbours centered around the candidate for the true maxima
            cv::Range in(std::max(cc.y-nWinSize,0),std::min(cc.y+nWinSize+1,M));
            cv::Range jn(std::max(cc.x-nWinSize,0),std::min(cc.x+nWinSize+1,N));
            // mask out the block whose maxima we already know
            cv::Mat_<uint8_t> blockmask;
            block(cv::Range(0,in.size()),cv::Range(0,jn.size())).copyTo(blockmask);
            cv::Range iis(ic.start-in.start,std::min(ic.start-in.start+nWinSize+1, in.size()));
            cv::Range jis(jc.start-jn.start,std::min(jc.start-jn.start+nWinSize+1, jn.size()));
            blockmask(iis, jis) = cv::Mat_<uint8_t>::zeros(cv::Size(jis.size(),iis.size()));
            cv::minMaxLoc(oInput(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? oMask(in,jn).mul(blockmask) : blockmask);
            //cv::Point cn = ijmax + cv::Point(jn.start, in.start);
            // if the block centre is also the neighbour centre, then it's a local maxima
            if(vcmax > vnmax)
                oOutput.at<uint8_t>(cc.y, cc.x) = 255;
        }
    }
}

template<size_t nHalfWinSize, typename Tr>
bool lv::isLocalMaximum_Horizontal(const Tr* const anMap, const size_t nMapColStep, const size_t /*nMapRowStep*/) {
    static_assert(nHalfWinSize>=1,"Window size needs to be at least 3x3");
    const Tr nVal = *anMap;
    bool bRes = true;
    lv::unroll<nHalfWinSize>([&](int n){
        bRes &= nVal>anMap[(-n-1)*nMapColStep];
    });
    lv::unroll<nHalfWinSize>([&](int n){
        bRes &= nVal>=anMap[(n+1)*nMapColStep];
    });
    return bRes;
}

template<size_t nHalfWinSize, typename Tr>
bool lv::isLocalMaximum_Vertical(const Tr* const anMap, const size_t /*nMapColStep*/, const size_t nMapRowStep) {
    static_assert(nHalfWinSize>=1,"Window size needs to be at least 3x3");
    const Tr nVal = *anMap;
    bool bRes = true;
    lv::unroll<nHalfWinSize>([&](int n){
        bRes &= nVal>anMap[(-n-1)*nMapRowStep];
    });
    lv::unroll<nHalfWinSize>([&](int n){
        bRes &= nVal>=anMap[(n+1)*nMapRowStep];
    });
    return bRes;
}

template<size_t nHalfWinSize, bool bInvDiag, typename Tr>
bool lv::isLocalMaximum_Diagonal(const Tr* const anMap, const size_t nMapColStep, const size_t nMapRowStep) {
    static_assert(nHalfWinSize>=1,"Window size needs to be at least 3x3");
    const Tr nVal = *anMap;
    bool bRes = true;
    lv::unroll<nHalfWinSize>([&](int n){
        bRes &= nVal>anMap[(bInvDiag?-1:1)*(-n-1)*nMapColStep+(-n-1)*nMapRowStep];
    });
    lv::unroll<nHalfWinSize>([&](int n){
        bRes &= nVal>=anMap[(bInvDiag?-1:1)*(n+1)*nMapColStep+(n+1)*nMapRowStep];
    });
    return bRes;
}

template<size_t nHalfWinSize, typename Tr>
bool lv::isLocalMaximum_Diagonal(const Tr* const anMap, const size_t nMapColStep, const size_t nMapRowStep, bool bInvDiag) {
    if(bInvDiag)
        return isLocalMaximum_Diagonal<nHalfWinSize,true>(anMap,nMapColStep,nMapRowStep);
    else
        return isLocalMaximum_Diagonal<nHalfWinSize,false>(anMap,nMapColStep,nMapRowStep);
}

template<size_t nKMeansIters, size_t nC1, size_t nC2, size_t nD>
void lv::initGaussianMixtureParams(const cv::Mat& oInput, const cv::Mat& oMask, lv::GMM<nC1,nD>& oBGModel, lv::GMM<nC2,nD>& oFGModel, const cv::Mat& oROI) {
    static_assert(nKMeansIters>0,"bad iter count for kmeans");
    lvAssert_(!oInput.empty() && !oMask.empty() && oInput.size==oMask.size,"bad input image/mask size");
    lvAssert_(oInput.isContinuous() && oMask.isContinuous(),"need continuous mats (raw indexing in impl)");
    lvAssert_(oInput.depth()==CV_8U,"input image type should be 8U (only supported for now)");
    lvAssert_(oInput.channels()==int(nD),"input image channel count must match gmm sample dims");
    lvAssert_(oMask.type()==CV_8UC1,"input mask type must be 8UC1 (where all values >0 are considered foreground)");
    lvAssert_(oROI.empty() || (oROI.size==oInput.size && oROI.isContinuous() && oROI.type()==CV_8UC1),"bad ROI size/type");
    static thread_local lv::AutoBuffer<float> s_aBGSamples,s_aFGSamples;
    size_t nBGSamples=0,nFGSamples=0;
    const size_t nTotSamples = oInput.total();
    s_aBGSamples.resize(nTotSamples*nD);
    s_aFGSamples.resize(nTotSamples*nD);
    const uchar* pROI = oROI.empty()?nullptr:oROI.data;
    for(size_t nSampleIdx=0; nSampleIdx<nTotSamples; ++nSampleIdx) {
        if(!pROI || pROI[nSampleIdx]) {
            const uchar* pPixelData = oInput.data+nSampleIdx*nD;
            if(oMask.data[nSampleIdx])
                std::transform(pPixelData,pPixelData+nD,&s_aFGSamples[(nFGSamples++)*nD],[](uchar n){return float(n);});
            else
                std::transform(pPixelData,pPixelData+nD,&s_aBGSamples[(nBGSamples++)*nD],[](uchar n){return float(n);});
        }
    }
    cv::Mat oClusterLabels;
    std::array<double,nD> aSample;
    oBGModel.initLearning();
    if(nBGSamples>0) {
        cv::Mat oBGSamples(int(nBGSamples),(int)nD,CV_32FC1,s_aBGSamples.data());
        cv::kmeans(oBGSamples,int(nC1),oClusterLabels,cv::TermCriteria(CV_TERMCRIT_ITER,(int)nKMeansIters,0.0),0,cv::KMEANS_PP_CENTERS);
        for(size_t nSampleIdx=0; nSampleIdx<nBGSamples; ++nSampleIdx) {
            lv::unroll<nD>([&](size_t nDimIdx){aSample[nDimIdx] = double(s_aBGSamples[nSampleIdx*nD+nDimIdx]);});
            oBGModel.addSample(size_t(oClusterLabels.at<int>(int(nSampleIdx),0)),aSample);
        }
    }
    oBGModel.endLearning();
    oFGModel.initLearning();
    if(nFGSamples>0) {
        cv::Mat oFGSamples(int(nFGSamples),(int)nD,CV_32FC1,s_aFGSamples.data());
        cv::kmeans(oFGSamples,int(nC2),oClusterLabels,cv::TermCriteria(CV_TERMCRIT_ITER,(int)nKMeansIters,0.0),0,cv::KMEANS_PP_CENTERS);
        for(size_t nSampleIdx=0; nSampleIdx<nFGSamples; ++nSampleIdx) {
            lv::unroll<nD>([&](size_t nDimIdx){aSample[nDimIdx] = double(s_aFGSamples[nSampleIdx*nD+nDimIdx]);});
            oFGModel.addSample(size_t(oClusterLabels.at<int>(int(nSampleIdx),0)),aSample);
        }
    }
    oFGModel.endLearning();
}

template<size_t nC1, size_t nC2, size_t nD>
void lv::assignGaussianMixtureComponents(const cv::Mat& oInput, const cv::Mat& oMask, cv::Mat& oAssignMap, const lv::GMM<nC1,nD>& oBGModel, const lv::GMM<nC2,nD>& oFGModel, const cv::Mat& oROI) {
    lvAssert_(!oInput.empty() && !oMask.empty() && oInput.size==oMask.size,"bad input image/mask size");
    lvAssert_(oInput.isContinuous() && oMask.isContinuous(),"need continuous mats (raw indexing in impl)");
    lvAssert_(oInput.depth()==CV_8U,"input image type should be 8U (only supported for now)");
    lvAssert_(oInput.channels()==int(nD),"input image channel count must match gmm sample dims");
    lvAssert_(oMask.type()==CV_8UC1,"input mask type must be 8UC1 (where all values >0 are considered foreground)");
    oAssignMap.create(oInput.dims,oInput.size,CV_32SC1);
    lvAssert_(oAssignMap.isContinuous(),"need continuous mats (raw indexing in impl)");
    lvAssert_(oROI.empty() || (oROI.size==oInput.size && oROI.isContinuous() && oROI.type()==CV_8UC1),"bad ROI size/type");
    std::array<double,nD> aSample;
    const size_t nTotSamples = oInput.total();
    const uchar* pROI = oROI.empty()?nullptr:oROI.data;
    for(size_t nSampleIdx=0; nSampleIdx<nTotSamples; ++nSampleIdx) {
        if(!pROI || pROI[nSampleIdx]) {
            const uchar* pPixelData = oInput.data+nSampleIdx*nD;
            lv::unroll<nD>([&](size_t nDimIdx){aSample[nDimIdx] = double(pPixelData[nDimIdx]);});
            ((int*)oAssignMap.data)[nSampleIdx] = int(oMask.data[nSampleIdx]?oFGModel.getBestComponent(aSample):oBGModel.getBestComponent(aSample));
        }
    }
}

template<size_t nC1, size_t nC2, size_t nD>
void lv::learnGaussianMixtureParams(const cv::Mat& oInput, const cv::Mat& oMask, const cv::Mat& oAssignMap, lv::GMM<nC1,nD>& oBGModel, lv::GMM<nC2,nD>& oFGModel, const cv::Mat& oROI) {
    lvAssert_(!oInput.empty() && !oMask.empty() && !oAssignMap.empty() && oInput.size==oMask.size && oInput.size==oAssignMap.size,"bad input image/mask/assignmap size");
    lvAssert_(oInput.isContinuous() && oMask.isContinuous() && oAssignMap.isContinuous(),"need continuous mats (raw indexing in impl)");
    lvAssert_(oInput.depth()==CV_8U,"input image type should be 8U (only supported for now)");
    lvAssert_(oInput.channels()==int(nD),"input image channel count must match gmm sample dims");
    lvAssert_(oMask.type()==CV_8UC1,"input mask type must be 8UC1 (where all values >0 are considered foreground)");
    lvAssert_(oAssignMap.type()==CV_32SC1,"input component assignment map must be 32SC1 (see 'assignGaussianMixtureComponents')");
    lvAssert_(oROI.empty() || (oROI.size==oInput.size && oROI.isContinuous() && oROI.type()==CV_8UC1),"bad ROI size/type");
    oBGModel.initLearning();
    oFGModel.initLearning();
    std::array<double,nD> aSample;
    const size_t nTotSamples = oInput.total();
    const uchar* pROI = oROI.empty()?nullptr:oROI.data;
    for(size_t nSampleIdx=0; nSampleIdx<nTotSamples; ++nSampleIdx) {
        if(!pROI || pROI[nSampleIdx]) {
            const int nCompLabel = ((const int*)oAssignMap.data)[nSampleIdx];
            const bool bForeground = (oMask.data[nSampleIdx])!=0;
            if(nCompLabel>=0 && nCompLabel<int(bForeground?nC2:nC1)) {
                const uchar* pPixelData = oInput.data+nSampleIdx*nD;
                lv::unroll<nD>([&](size_t nDimIdx){aSample[nDimIdx] = double(pPixelData[nDimIdx]);});
                if(bForeground)
                    oFGModel.addSample(size_t(nCompLabel),aSample);
                else
                    oBGModel.addSample(size_t(nCompLabel),aSample);
            }
        }
    }
    oBGModel.endLearning();
    oFGModel.endLearning();
}