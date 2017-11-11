
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

#include "litiv/imgproc.hpp"
#include "litiv/features2d/MI.hpp"
#if HAVE_CUDA
#include "affinity.cuh"
#endif //HAVE_CUDA

void thinning_internal_ZhangSuen(cv::Mat& oInput, cv::Mat& oTempMarker, bool bIter) {
    oTempMarker.create(oInput.size(),CV_8UC1);
    oTempMarker = cv::Scalar_<uchar>(0);

    const uchar* pAbove = nullptr;
    const uchar* pCurr = oInput.ptr<uchar>(0);
    const uchar* pBelow = oInput.ptr<uchar>(1);
    const uchar* nw, *no, *ne;
    const uchar* we, *me, *ea;
    const uchar* sw, *so, *se;

    for(int y=1; y<oInput.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = oInput.ptr<uchar>(y+1);
        uchar* pDst = oTempMarker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for(int x=1; x<oInput.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            int A  = (!*no && *ne>0) + (!*ne && *ea>0) +
                     (!*ea && *se>0) + (!*se && *so>0) +
                     (!*so && *sw>0) + (!*sw && *we>0) +
                     (!*we && *nw>0) + (!*nw && *no>0);
            int B  = (*no>0)+(*ne>0)+(*ea>0)+(*se>0)+(*so>0)+(*sw>0)+(*we>0)+(*nw>0);
            int m1 = !bIter?((*no>0)*(*ea>0)*(*so>0)):((*no>0)*(*ea>0)*(*we>0));
            int m2 = !bIter?((*ea>0)*(*so>0)*(*we>0)):((*no>0)*(*so>0)*(*we>0));
            if(A==1 && B>=2 && B<=6 && !m1 && !m2)
                pDst[x] = UCHAR_MAX;
        }
    }
    oInput &= ~oTempMarker;
}

void thinning_internal_LamLeeSuen(cv::Mat& oInput, bool bIter) {
    for(int i=1; i<oInput.rows-1; ++i) {
        for(int j=1; j<oInput.cols-1; ++j) {
            if(!oInput.at<uchar>(i,j))
                continue;
            const std::array<uchar,8> anLUT{
                oInput.at<uchar>(i+1,j  ),
                oInput.at<uchar>(i+1,j-1),
                oInput.at<uchar>(i  ,j-1),
                oInput.at<uchar>(i-1,j-1),
                oInput.at<uchar>(i-1,j  ),
                oInput.at<uchar>(i-1,j+1),
                oInput.at<uchar>(i  ,j+1),
                oInput.at<uchar>(i+1,j+1)
            };
            size_t x_h = 0, n1 = 0, n2 = 0;
            for(size_t k=0; k<4; ++k) {
#if defined(_MSC_VER) || USE_IMGPROC_THINNING_MATLAB_IMPL_FIX
                // G1:
                x_h += bool(!anLUT[2*(k+1)-2] && (anLUT[2*(k+1)-1] || anLUT[(2*(k+1))%8]));
                // G2:
                n1 += bool(anLUT[2*(k+1)-2] || anLUT[2*(k+1)-1]);
                n2 += bool(anLUT[2*(k+1)-1] || anLUT[(2*(k+1))%8]);
#elif defined(__clang__)
#pragma clang diagnostic push
// but no warnings to push...?
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif //(defined(__GNUC__) || defined(__GNUG__))
                // G1:
                x_h += bool(!anLUT[2*(k+1)-2] && (anLUT[2*(k+1)-1] || anLUT[2*(k+1)]));
                // G2:
                n1 += bool(anLUT[2*(k+1)-2] || anLUT[2*(k+1)-1]);
                n2 += bool(anLUT[2*(k+1)-1] || anLUT[2*(k+1)]);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#endif //(defined(__GNUC__) || defined(__GNUG__))
            }
            size_t n_min = std::min(n1,n2);
            if(x_h==1 && n_min>=2 && n_min<=3) {
                // G3 || G3' :
                if( (!bIter && !((anLUT[1] || anLUT[2] || !anLUT[7]) && anLUT[0])) ||
                    (bIter && !((anLUT[5] || anLUT[6] || !anLUT[3]) && anLUT[4]))) {
                    oInput.at<uchar>(i,j) = 0;
                }
            }
        }
    }
}

void lv::thinning(const cv::Mat& oInput, cv::Mat& oOutput, ThinningMode eMode) {
    lvAssert_(!oInput.empty() && oInput.isContinuous(),"input image must be non-empty and continuous");
    lvAssert_(oInput.type()==CV_8UC1,"input image type must be 8UC1");
    lvAssert_(oInput.rows>3 && oInput.cols>3,"input image size must be greater than 3x3");
    oOutput.create(oInput.size(),CV_8UC1);
    oInput.copyTo(oOutput);
    cv::Mat oPrevious(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    cv::Mat oTempMarker;
    bool bEq;
    do {
        if(eMode==ThinningMode_ZhangSuen) {
            thinning_internal_ZhangSuen(oOutput,oTempMarker,false);
            thinning_internal_ZhangSuen(oOutput,oTempMarker,true);
        }
        else { //eMode==ThinningMode_LamLeeSuen
            thinning_internal_LamLeeSuen(oOutput,false);
            thinning_internal_LamLeeSuen(oOutput,true);
        }
        bEq = std::equal(oOutput.begin<uchar>(),oOutput.end<uchar>(),oPrevious.begin<uchar>());
        oOutput.copyTo(oPrevious);
    }
    while(!bEq);
}

void lv::remap_offset(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oOffsetMap, int nInterpType, int nBorderMode, const cv::Scalar& vBorderValue) {
    lvAssert_(!oInput.empty() && !oOffsetMap.empty() && oOffsetMap.type()==CV_32FC2,"invalid params");
    lvAssert_(oInput.dims==2 && lv::MatSize(oInput)==lv::MatSize(oOffsetMap),"invalid map size");
    oOutput.create(oInput.size(),oInput.type());
    cv::Mat_<cv::Vec2f> oLocalIdxMap(oInput.size());
    for(int nRowIdx=0; nRowIdx<oInput.rows; ++nRowIdx)
        for(int nColIdx=0; nColIdx<oInput.cols; ++nColIdx)
            oLocalIdxMap(nRowIdx,nColIdx) = cv::Vec2f(nColIdx-oOffsetMap.at<cv::Vec2f>(nRowIdx,nColIdx)[0],nRowIdx-oOffsetMap.at<cv::Vec2f>(nRowIdx,nColIdx)[1]);
    cv::remap(oInput,oOutput,oLocalIdxMap,cv::Mat(),nInterpType,nBorderMode,vBorderValue);
}

void lv::computeImageAffinity(const cv::Mat& oImage1, const cv::Mat& oImage2, int nPatchSize,
                              cv::Mat_<float>& oAffinityMap, const std::vector<int>& vDispRange, AffinityDistType eDist,
                              const cv::Mat_<uchar>& oROI1, const cv::Mat_<uchar>& oROI2) {
    lvAssert_(!oImage1.empty() && oImage1.size==oImage2.size && oImage1.dims==2,"bad input image sizes");
    lvAssert_(oROI1.empty() || (oROI1.dims==2 && oROI1.rows==oImage1.size[0] && oROI1.cols==oImage1.size[1]),"bad ROI1 map size");
    lvAssert_(oROI2.empty() || (oROI2.dims==2 && oROI2.rows==oImage2.size[0] && oROI2.cols==oImage2.size[1]),"bad ROI2 map size");
    lvAssert_(eDist==lv::AffinityDist_MI || eDist==lv::AffinityDist_SSD,"unsupported distance type");
    lvAssert_(nPatchSize>=1 && (nPatchSize%2)==1,"bad patch size");
    lvAssert_(nPatchSize<=oImage1.rows && nPatchSize<=oImage1.cols,"patch too large for input images");
    lvAssert_(vDispRange.size()>=1,"bad disparity range");
    cv::Mat_<uchar> oImage1_uchar,oImage2_uchar;
    if(eDist==lv::AffinityDist_MI) {
        lvAssert_(oImage1.type()==oImage2.type() && oImage1.type()==CV_8UC1,"bad input image types/depth");
        oImage1_uchar = oImage1;
        oImage2_uchar = oImage2;
    }
    else /*if(eDist==lv::AffinityDist_SSD)*/
        lvAssert_(oImage1.type()==oImage2.type() && oImage1.channels()==1,"bad input image types/depth");
    const bool bValidROI1 = !oROI1.empty();
    const bool bValidROI2 = !oROI2.empty();
    const int nRows = oImage1.rows;
    const int nCols = oImage1.cols;
    const int nPatchRadius = nPatchSize/2;
    const int nOffsets = int(vDispRange.size());
    const std::array<int,3> anAffinityMapDims = {nRows-nPatchRadius*2,nCols-nPatchRadius*2,nOffsets};
    oAffinityMap.create(3,anAffinityMapDims.data());
    oAffinityMap = -1.0f; // default value for OOB pixels
    thread_local lv::JointSparseHistData<uchar,uchar> oSparseHistData;
#if USING_OPENMP
#ifdef _MSC_VER
    #pragma omp parallel for // msvc only supports openmp 2.0
#else //ndef(_MSC_VER)
    #pragma omp parallel for collapse(3)
#endif //ndef(_MSC_VER)
#endif //USING_OPENMP
    for(int nRowIdx=nPatchRadius; nRowIdx<nRows-nPatchRadius; ++nRowIdx) {
        for(int nColIdx=nPatchRadius; nColIdx<nCols-nPatchRadius; ++nColIdx) {
            for(int nOffsetIdx=0; nOffsetIdx<nOffsets; ++nOffsetIdx) {
                const int nColOffset = vDispRange[nOffsetIdx];
                const int nOffsetColIdx = nColIdx+nColOffset;
                if((bValidROI1 && !oROI1(nRowIdx,nColIdx)) || nOffsetColIdx<nPatchRadius || nOffsetColIdx>=nCols-nPatchRadius || (bValidROI2 && !oROI2(nRowIdx,nOffsetColIdx)))
                    continue;
                const cv::Rect oWindow(nColIdx-nPatchRadius,nRowIdx-nPatchRadius,nPatchSize,nPatchSize);
                const cv::Rect oOffsetWindow(nOffsetColIdx-nPatchRadius,nRowIdx-nPatchRadius,nPatchSize,nPatchSize);
                if(eDist==lv::AffinityDist_MI) {
                    const double dMutualInfoScore = lv::calcMutualInfo<1,true,true,false,false>(oImage1_uchar(oWindow),oImage2_uchar(oOffsetWindow),&oSparseHistData);
                    oAffinityMap.at<float>(nRowIdx-nPatchRadius,nColIdx-nPatchRadius,nOffsetIdx) = std::max(float(1.0-dMutualInfoScore),0.0f);
                }
                else /*if(eDist==lv::AffinityDist_SSD)*/ {
                    oAffinityMap.at<float>(nRowIdx-nPatchRadius,nColIdx-nPatchRadius,nOffsetIdx) = (float)cv::norm(oImage1(oWindow),oImage2(oOffsetWindow),cv::NORM_L2);
                }
            }
        }
    }
}

void lv::computeDescriptorAffinity(const cv::Mat_<float>& oDescMap1, const cv::Mat_<float>& oDescMap2,
                                   int nPatchSize, cv::Mat_<float>& oAffinityMap, const std::vector<int>& vDispRange,
                                   AffinityDistType eDist, const cv::Mat_<uchar>& oROI1, const cv::Mat_<uchar>& oROI2,
                                   const cv::Mat_<float>& oEMDCostMap, bool bAllowCUDA) {
    lvDbgExceptionWatch;
    lvAssert_(!oDescMap1.empty() && oDescMap1.size==oDescMap2.size && oDescMap1.dims==3 && oDescMap1.size[2]>1,"bad input desc map sizes");
    lvAssert_(oROI1.empty() || (oROI1.dims==2 && oROI1.rows==oDescMap1.size[0] && oROI1.cols==oDescMap1.size[1]),"bad ROI1 map size");
    lvAssert_(oROI2.empty() || (oROI2.dims==2 && oROI2.rows==oDescMap2.size[0] && oROI2.cols==oDescMap2.size[1]),"bad ROI2 map size");
    lvAssert_(eDist==lv::AffinityDist_L2 || eDist==lv::AffinityDist_EMD,"unsupported distance type");
    lvAssert_(nPatchSize>=1 && (nPatchSize%2)==1,"bad patch size");
    lvAssert_(!vDispRange.empty(),"bad disparity range");
    if(eDist==lv::AffinityDist_EMD) {
        lvAssert_(!oEMDCostMap.empty() && oEMDCostMap.dims==2 && oEMDCostMap.rows==oEMDCostMap.cols,"bad emd cost map size");
        lvAssert_(oEMDCostMap.rows==oDescMap1.size[2],"bad emd cost map size for given desc size");
    }
    const int nRows = oDescMap1.size[0];
    const int nCols = oDescMap1.size[1];
    const int nDescSize = oDescMap1.size[2];
    const int nOffsets = int(vDispRange.size());
    const std::array<int,3> anAffinityMapDims = {nRows,nCols,nOffsets};
    lvIgnore(bAllowCUDA);
#if HAVE_CUDA
    static thread_local cv::cuda::GpuMat s_oDescMap1_dev,s_oDescMap2_dev,s_oAffinityMap_dev,s_oROI1_dev,s_oROI2_dev;
    if(bAllowCUDA && eDist==lv::AffinityDist_L2 && oROI1.empty()==oROI2.empty() && (nRows*nCols>64 || nOffsets>16 || nDescSize>32)) {
        lvAssert_(cv::cuda::deviceSupports(LITIV_CUDA_MIN_COMPUTE_CAP),"device compute capabilities too low");
        lvAssert_(oDescMap1.isContinuous() && oDescMap2.isContinuous(),"non-continuous n-dim matrices cannot be reshaped by opencv (check inputs)");
        // all uploads/downloads below are blocking calls @@@
        const std::array<int,2> anDescMapDims_dev = {nRows*nCols,nDescSize};
        s_oDescMap1_dev.upload(oDescMap1.reshape(0,2,anDescMapDims_dev.data()));
        s_oDescMap2_dev.upload(oDescMap2.reshape(0,2,anDescMapDims_dev.data()));
        if(!oROI1.empty()) {
            s_oROI1_dev.upload(oROI1);
            s_oROI2_dev.upload(oROI2);
        }
        else {
            s_oROI1_dev.release();
            s_oROI2_dev.release();
        }
        lv::computeDescriptorAffinity(s_oDescMap1_dev,s_oDescMap2_dev,cv::Size(nCols,nRows),nPatchSize,s_oAffinityMap_dev,vDispRange,lv::AffinityDist_L2,s_oROI1_dev,s_oROI2_dev);
        oAffinityMap.create(3,anAffinityMapDims.data());
        const std::array<int,2> anAffinityMapDims_dev = {nRows*nCols,nOffsets};
        oAffinityMap = oAffinityMap.reshape(0,2,anAffinityMapDims_dev.data());
        s_oAffinityMap_dev.download(oAffinityMap);
        oAffinityMap = oAffinityMap.reshape(0,3,anAffinityMapDims.data());
        return;
    }
#endif //HAVE_CUDA
    const bool bValidROI1 = !oROI1.empty();
    const bool bValidROI2 = !oROI2.empty();
    const int nPatchRadius = nPatchSize/2;
    oAffinityMap.create(3,anAffinityMapDims.data());
    oAffinityMap = -1.0f; // default value for OOB pixels
    cv::Mat_<float> oRawAffinity; // used to cache pixel-wise descriptor distances
    static thread_local lv::AutoBuffer<float> s_aRawAffinityData;
    if(nPatchSize>1) {
        s_aRawAffinityData.resize(oAffinityMap.total());
        oRawAffinity = cv::Mat_<float>(3,anAffinityMapDims.data(),s_aRawAffinityData.data());
        oRawAffinity = -1.0f; // default value for OOB pixels
    }
    else
        oRawAffinity = oAffinityMap;
    lvDbgExceptionWatch;
#if USING_OPENMP
#ifdef _MSC_VER
    #pragma omp parallel for // msvc only supports openmp 2.0
#else //ndef(_MSC_VER)
    #pragma omp parallel for collapse(2)
#endif //ndef(_MSC_VER)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            if(bValidROI1 && !oROI1(nRowIdx,nColIdx))
                continue;
            float* pRawAffinityPtr = oRawAffinity.ptr<float>(nRowIdx,nColIdx);
            for(int nOffsetIdx=0; nOffsetIdx<nOffsets; ++nOffsetIdx) {
                const int nOffsetColIdx = nColIdx+vDispRange[nOffsetIdx];
                if(nOffsetColIdx<0 || nOffsetColIdx>=nCols || (bValidROI2 && !oROI2(nRowIdx,nOffsetColIdx)))
                    continue;
                if(eDist==lv::AffinityDist_L2) {
                    const cv::Mat_<float> oDesc(1,nDescSize,const_cast<float*>(oDescMap1.ptr<float>(nRowIdx,nColIdx)));
                    const cv::Mat_<float> oOffsetDesc(1,nDescSize,const_cast<float*>(oDescMap2.ptr<float>(nRowIdx,nOffsetColIdx)));
                    pRawAffinityPtr[nOffsetIdx] = float(cv::norm(oDesc,oOffsetDesc,cv::NORM_L2));
                    lvDbgAssert(pRawAffinityPtr[nOffsetIdx]>=0.0f && pRawAffinityPtr[nOffsetIdx]<=(float)M_SQRT2);
                }
                else /*if(eDist==lv::AffinityDist_EMD)*/ {
                    const float* pDesc = oDescMap1.ptr<float>(nRowIdx,nColIdx);
                    const float* pOffsetDesc = oDescMap2.ptr<float>(nRowIdx,nOffsetColIdx);
                    const cv::Mat_<float> oDesc(nDescSize,1,const_cast<float*>(pDesc));
                    const cv::Mat_<float> oOffsetDesc(nDescSize,1,const_cast<float*>(pOffsetDesc));
                    lvDbgAssert_(!std::all_of(pDesc,pDesc+nDescSize,[](float v){
                        lvDbgAssert(v>=0.0f);
                        return v==0.0f;
                    }),"opencv emd cannot handle null descriptors");
                    lvDbgAssert_(!std::all_of(pOffsetDesc,pOffsetDesc+nDescSize,[](float v){
                        lvDbgAssert(v>=0.0f);
                        return v==0.0f;
                    }),"opencv emd cannot handle null descriptors");
                    pRawAffinityPtr[nOffsetIdx] = cv::EMD(oDesc,oOffsetDesc,-1,oEMDCostMap);
                    lvDbgAssert(pRawAffinityPtr[nOffsetIdx]>=0.0f);
                }
            }
        }
    }
    if(nPatchSize==1)
        return;
    lvDbgExceptionWatch;
#if USING_OPENMP
#ifdef _MSC_VER
    #pragma omp parallel for // msvc only supports openmp 2.0
#else //ndef(_MSC_VER)
    #pragma omp parallel for collapse(3)
#endif //ndef(_MSC_VER)
#endif //USING_OPENMP
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            for(int nOffsetIdx=0; nOffsetIdx<nOffsets; ++nOffsetIdx) {
                size_t nValidCount = size_t(0);
                double afAccumAff = 0.0f;
                for(int nPatchRowIdx=std::max(nRowIdx-nPatchRadius,0); nPatchRowIdx<=std::min(nRowIdx+nPatchRadius,nRows-1); ++nPatchRowIdx) {
                    for(int nPatchColIdx=std::max(nColIdx-nPatchRadius,0); nPatchColIdx<=std::min(nColIdx+nPatchRadius,nCols-1); ++nPatchColIdx) {
                        const float* pRawAffinityPtr = oRawAffinity.ptr<float>(nPatchRowIdx,nPatchColIdx);
                        if(pRawAffinityPtr[nOffsetIdx]!=-1.0f) {
                            afAccumAff += pRawAffinityPtr[nOffsetIdx];
                            ++nValidCount;
                        }
                    }
                }
                if(nValidCount)
                    oAffinityMap.at<float>(nRowIdx,nColIdx,nOffsetIdx) = float(afAccumAff/nValidCount);
            }
        }
    }
}

#if HAVE_CUDA

void lv::computeDescriptorAffinity(const cv::cuda::GpuMat& oDescMap1, const cv::cuda::GpuMat& oDescMap2, const cv::Size& oMapSize, int nPatchSize,
                                   cv::cuda::GpuMat& oAffinityMap, const std::vector<int>& vDispRange, AffinityDistType eDist,
                                   const cv::cuda::GpuMat& oROI1, const cv::cuda::GpuMat& oROI2) {
    lvDbgExceptionWatch;
    lvAssert_(!oDescMap1.empty() && oDescMap1.size()==oDescMap2.size(),"bad input desc map sizes");
    lvAssert_(oMapSize.area()>0 && oDescMap1.rows==oMapSize.area(),"bad 2d map size");
    lvAssert_(oROI1.empty() || (oROI1.size()==oMapSize && oROI1.type()==CV_8UC1),"bad ROI1 map size");
    lvAssert_(oROI2.empty() || (oROI2.size()==oMapSize && oROI2.type()==CV_8UC1),"bad ROI2 map size");
    lvAssert_((oROI1.empty() && oROI2.empty()) || (!oROI1.empty() && !oROI1.empty()),"both ROIs must be empty or non-empty");
    lvAssert_(eDist==lv::AffinityDist_L2,"unsupported distance type"); // @@@@ todo add emd distnace
    lvAssert_(nPatchSize>=1 && (nPatchSize%2)==1,"bad patch size");
    lvAssert_(!vDispRange.empty(),"bad disparity range");
    static constexpr size_t nMaxDispOffsets = AFF_MAP_DISP_RANGE_MAX;
    lvAssert_(vDispRange.size()<=nMaxDispOffsets,"disp offset count over impl limit");
    const int nOffsets = int(vDispRange.size());
    const int nDescSize = oDescMap1.cols;
    oAffinityMap.create(oMapSize.height*oMapSize.width,nOffsets,CV_32FC1);
    static thread_local cv::cuda::GpuMat oRawAffinityMap;
    if(nPatchSize>1)
        oRawAffinityMap.create(oMapSize.height*oMapSize.width,nOffsets,CV_32FC1);
    static std::mutex s_oConstMemKernelCallMutex; {
        std::lock_guard<std::mutex> oConstMemKernelCallLock(s_oConstMemKernelCallMutex);
        static thread_local std::array<int,nMaxDispOffsets> s_vDispRange_dev{};
        if(!std::equal(vDispRange.begin(),vDispRange.end(),s_vDispRange_dev.begin())) {
            std::copy(vDispRange.begin(),vDispRange.end(),s_vDispRange_dev.begin());
            device::setDisparityRange(s_vDispRange_dev);
        }
        lv::cuda::KernelParams oParams;
        const int nMinThreads = cv::cuda::DeviceInfo().warpSize()*2;
        const int nThreads = (nOffsets>nMinThreads)?(int)std::ceil(float(nOffsets)/nMinThreads)*nMinThreads:nMinThreads;
        lvDbgAssert((nThreads%nMinThreads)==0);
        const size_t nOffsets_LUT = size_t(std::ceil(float(nOffsets)/nThreads)*nThreads);
        const size_t nDescSize_LUT = size_t(std::ceil(float(nDescSize)/nThreads)*nThreads);
        oParams.vBlockSize.x = (uint)nThreads;
        oParams.vGridSize = dim3((uint)oMapSize.width,(uint)oMapSize.height);
        oParams.nSharedMemSize = (sizeof(device::DistCalcLUT)*nOffsets_LUT + sizeof(float)*nDescSize_LUT);
        if(oROI1.empty())
            device::compute_desc_affinity_l2(oParams,oDescMap1,oDescMap2,nPatchSize>1?oRawAffinityMap:oAffinityMap,nOffsets,nDescSize);
        else
            device::compute_desc_affinity_l2_roi(oParams,oDescMap1,oROI1,oDescMap2,oROI2,nPatchSize>1?oRawAffinityMap:oAffinityMap,nOffsets,nDescSize);
    }
    lvDbgExceptionWatch;
    if(nPatchSize>1) {
        lv::cuda::KernelParams oParams;
        const int nMinThreads = cv::cuda::DeviceInfo().warpSize();
        const int nPatchBins = nPatchSize*nPatchSize;
        const int nThreads = int(std::ceil(float(nPatchBins)/nMinThreads)*nMinThreads);
        lvDbgAssert((nThreads%nMinThreads)==0 && nThreads/nPatchSize>1);
        oParams.vBlockSize.x = (uint)nThreads;
        oParams.vGridSize = dim3((uint)oMapSize.width,(uint)oMapSize.height,(uint)nOffsets);
        oParams.nSharedMemSize = ((sizeof(int)+sizeof(float))*nThreads);
        device::compute_desc_affinity_patch(oParams,oRawAffinityMap,oAffinityMap,nPatchSize);
    }
}

#endif //HAVE_CUDA

void lv::computeTemporalAbsDiff(const cv::Mat& oImage1, const cv::Mat& oImage2, const cv::Mat& oFlow, cv::Mat& oOutput, int nSmoothKernelSize) {
    lvAssert_(!oImage1.empty() && !oImage2.empty() && !oFlow.empty() && nSmoothKernelSize>=0,"invalid parameter(s)");
    lvAssert_(oImage1.dims==2 && oImage1.depth()==CV_8U && lv::MatInfo(oImage1)==lv::MatInfo(oImage2),"invalid image type/size");
    lvAssert_(oImage1.isContinuous() && oImage2.isContinuous() && oFlow.isContinuous(),"all inputs must be continuous mats");
    lvAssert_(oFlow.type()==CV_32FC2 && lv::MatSize(oImage1)==lv::MatSize(oFlow),"invalid flow type/size");
    lvAssert_(nSmoothKernelSize==0 || (nSmoothKernelSize%2)==1,"smoothing kernel size must be odd or null");
    lvAssert_(oImage1.channels()==1 || oImage1.channels()==3,"current impl only supports 1 or 3 channels due to bilin subsampl");
    const int nRows = oImage1.rows, nCols = oImage1.cols, nChannels = oImage1.channels();
    oOutput.create(nRows,nCols,CV_32FC(nChannels));
    lvDbgAssert(oOutput.isContinuous());
    for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
        for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
            const float fOffsetX = ((float*)oFlow.data)[(nRowIdx*nCols+nColIdx)*2];
            const float fOffsetY = ((float*)oFlow.data)[(nRowIdx*nCols+nColIdx)*2+1];
            if(nChannels==1) {
                const int nOld = ((uint8_t*)oImage1.data)[nRowIdx*nCols+nColIdx];
                const float fNew = lv::getSubPix<uint8_t,float>(cv::Mat_<uint8_t>(oImage2),nColIdx+fOffsetX,nRowIdx+fOffsetY);
                ((float*)oOutput.data)[nRowIdx*nCols+nColIdx] = std::abs(nOld-fNew);
            }
            else { // nChannels==3
                const cv::Vec3b& vOld = oImage1.at<cv::Vec3b>(nRowIdx,nColIdx);
                const cv::Vec3f vNew = lv::getSubPix<uint8_t,3,float>(cv::Mat_<cv::Vec3b>(oImage2),nColIdx+fOffsetX,nRowIdx+fOffsetY);
                for(int nChIdx=0; nChIdx<nChannels; ++nChIdx)
                    ((float*)oOutput.data)[(nRowIdx*nCols+nColIdx)*nChannels+nChIdx] = std::abs(vOld[nChIdx]-vNew[nChIdx]);
            }
        }
    }
    if(nSmoothKernelSize>1)
        cv::GaussianBlur(oOutput,oOutput,cv::Size(nSmoothKernelSize,nSmoothKernelSize),0);
}