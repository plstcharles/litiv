
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

#include "litiv/video/BackgroundSubtractorLBSP.hpp"

template<lv::ParallelAlgoType eImpl>
void IBackgroundSubtractorLBSP_<eImpl>::initialize_common(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    lvDbgExceptionWatch;
    IIBackgroundSubtractor::initialize_common(oInitImg,oROI);
    m_oLastDescFrame.create(this->m_oImgSize,CV_16UC((int)this->m_nImgChannels));
    m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
    const int nLBSPBorderSize = (int)LBSP::PATCH_SIZE/2;
    if(this->m_nImgChannels==1) {
        lvAssert(m_oLastDescFrame.step.p[0]==this->m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==this->m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset)/3);
        for(size_t nPxIter=0; nPxIter<this->m_nTotPxCount; ++nPxIter) {
            const int nImgCoord_X = this->m_voPxInfoLUT[nPxIter].nImgCoord_X;
            const int nImgCoord_Y = this->m_voPxInfoLUT[nPxIter].nImgCoord_Y;
            if(this->m_oROI.data[nPxIter] && nImgCoord_X>nLBSPBorderSize && nImgCoord_Y>nLBSPBorderSize && nImgCoord_X<oInitImg.cols-nLBSPBorderSize && nImgCoord_Y<oInitImg.rows-nLBSPBorderSize) {
                const size_t nDescIter = nPxIter*2;
                LBSP::computeDescriptor<1>(oInitImg,oInitImg.data[nPxIter],nImgCoord_X,nImgCoord_Y,0,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxIter]],*((ushort*)(m_oLastDescFrame.data+nDescIter)));
            }
        }
    }
    else { //(m_nImgChannels==3 || m_nImgChannels==4)
        lvAssert(m_oLastDescFrame.step.p[0]==this->m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==this->m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset);
        for(size_t nPxIter=0; nPxIter<this->m_nTotPxCount; ++nPxIter) {
            const int nImgCoord_X = this->m_voPxInfoLUT[nPxIter].nImgCoord_X;
            const int nImgCoord_Y = this->m_voPxInfoLUT[nPxIter].nImgCoord_Y;
            if(this->m_oROI.data[nPxIter] && nImgCoord_X>nLBSPBorderSize && nImgCoord_Y>nLBSPBorderSize && nImgCoord_X<oInitImg.cols-nLBSPBorderSize && nImgCoord_Y<oInitImg.rows-nLBSPBorderSize) {
                const size_t nPxRGBIter = nPxIter*this->m_nImgChannels;
                const size_t nDescRGBIter = nPxRGBIter*2;
                if(this->m_nImgChannels==3) {
                    alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,3> aanLBSPLookupVals;
                    LBSP::computeDescriptor_lookup(oInitImg,nImgCoord_X,nImgCoord_Y,aanLBSPLookupVals);
                    for(size_t c=0; c<3; ++c)
                        ((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c] = LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],oInitImg.data[nPxRGBIter+c],m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]]);
                }
                else { //m_nImgChannels==4
                    alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,4> aanLBSPLookupVals;
                    LBSP::computeDescriptor_lookup(oInitImg,nImgCoord_X,nImgCoord_Y,aanLBSPLookupVals);
                    for(size_t c=0; c<4; ++c)
                        ((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c] = LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],oInitImg.data[nPxRGBIter+c],m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]]);
                }
            }
        }
    }
}

#if HAVE_GLSL

template<>
template<>
std::string IBackgroundSubtractorLBSP_GLSL::getLBSPThresholdLUTShaderSource<lv::GLSL>() const {
    lvDbgExceptionWatch;
    lvAssert_(m_bInitialized,"algo must be initialized first");
    std::stringstream ssSrc;
    ssSrc << "const uint anLBSPThresLUT[256] = uint[256](\n    ";
    for(size_t t=0; t<=UCHAR_MAX; ++t) {
        if(t>0 && (t%((UCHAR_MAX+1)/8))==(((UCHAR_MAX+1)/8)-1) && t<UCHAR_MAX)
            ssSrc << (int)m_anLBSPThreshold_8bitLUT[t] << ",\n    ";
        else if(t<UCHAR_MAX)
            ssSrc << (int)m_anLBSPThreshold_8bitLUT[t] << ",";
        else
            ssSrc << (int)m_anLBSPThreshold_8bitLUT[t] << "\n";
    }
    ssSrc << ");\n";
    return ssSrc.str();
}

template struct IBackgroundSubtractorLBSP_<lv::GLSL>;

#endif //HAVE_GLSL

#if HAVE_CUDA
// ... @@@ add impl later
//template struct IBackgroundSubtractorLBSP_<lv::CUDA>;
#endif //HAVE_CUDA

#if HAVE_OPENCL
// ... @@@ add impl later
//template struct IBackgroundSubtractorLBSP_<lv::OpenCL>;
#endif //HAVE_OPENCL

template struct IBackgroundSubtractorLBSP_<lv::NonParallel>;
