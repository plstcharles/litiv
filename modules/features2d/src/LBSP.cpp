
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

#include "litiv/features2d/LBSP.hpp"

// make sure static constexpr array addresses exist
constexpr int LBSP::s_anIdxLUT_16bitdbcross[16][2];
constexpr int LBSP::s_anIdxLUT_16bitdbcross_GradX[16];
constexpr int LBSP::s_anIdxLUT_16bitdbcross_GradY[16];
constexpr LBSP::IdxLUTOffsetArray LBSP::s_oIdxLUT_16bitdbcross_x;
constexpr LBSP::IdxLUTOffsetArray LBSP::s_oIdxLUT_16bitdbcross_y;

LBSP::LBSP(size_t nThreshold) :
        m_bOnlyUsingAbsThreshold(true),
        m_fRelThreshold(0), // unused
        m_nThreshold(nThreshold),
        m_oRefImage() {}

LBSP::LBSP(float fRelThreshold, size_t nThresholdOffset) :
        m_bOnlyUsingAbsThreshold(false),
        m_fRelThreshold(fRelThreshold),
        m_nThreshold(nThresholdOffset),
        m_oRefImage() {
    lvAssert_(m_fRelThreshold>=0,"relative LBSP threshold must be non-negative");
}

LBSP::~LBSP() {}

void LBSP::read(const cv::FileNode& /*fn*/) {
    // ... = fn["..."];
}

void LBSP::write(cv::FileStorage& /*fs*/) const {
    //fs << "..." << ...;
}

void LBSP::setReference(const cv::Mat& img) {
    lvAssert_(img.empty() || ((img.type()==CV_8UC1 || img.type()==CV_8UC3) && img.isContinuous()),"reference image must be non-empty, or of type 8UC1/8UC3 and continuous");
    m_oRefImage = img;
}

int LBSP::descriptorSize() const {
    return LBSP::DESC_SIZE;
}

int LBSP::descriptorType() const {
    return CV_16U;
}

bool LBSP::isUsingRelThreshold() const {
    return !m_bOnlyUsingAbsThreshold;
}

float LBSP::getRelThreshold() const {
    return m_fRelThreshold;
}

size_t LBSP::getAbsThreshold() const {
    return m_nThreshold;
}

namespace {

void lbsp_computeImpl(const cv::Mat& oInputImg, const cv::Mat& oRefImg, const std::vector<cv::KeyPoint>& voKeyPoints, cv::Mat& oDesc, bool bSingleColumnDesc, size_t nThreshold) {
    static_assert(LBSP::DESC_SIZE==2,"bad assumptions in impl below");
    lvAssert_(!oInputImg.empty() && oInputImg.isContinuous() && (oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3),"input image must be non-empty, continuous, and of type 8UC1/8UC3");
    lvAssert_(oRefImg.empty() || (oRefImg.size==oInputImg.size && oRefImg.type()==oInputImg.type()),"ref image must be empty, or of the same size/type as the input image");
    const size_t nChannels = (size_t)oInputImg.channels();
    const cv::Mat& oRefMat = oRefImg.empty()?oInputImg:oRefImg;
    const size_t nKeyPoints = voKeyPoints.size();
    const uchar t = cv::saturate_cast<uchar>(nThreshold);
    if(nChannels==1) {
        if(bSingleColumnDesc)
            oDesc.create((int)nKeyPoints,1,CV_16UC1);
        else
            oDesc.create(oInputImg.size(),CV_16UC1);
        for(size_t k=0; k<nKeyPoints; ++k) {
            const int x = (int)voKeyPoints[k].pt.x;
            const int y = (int)voKeyPoints[k].pt.y;
            LBSP::computeDescriptor<1>(oInputImg,oRefMat.at<uchar>(y,x),x,y,0,t,oDesc.at<ushort>((int)k));
        }
        return;
    }
    else { //nChannels==3
        if(bSingleColumnDesc)
            oDesc.create((int)nKeyPoints,1,CV_16UC3);
        else
            oDesc.create(oInputImg.size(),CV_16UC3);
        alignas(16) const std::array<uchar,3> anThreshold = {t,t,t};
        for(size_t k=0; k<nKeyPoints; ++k) {
            const int x = (int)voKeyPoints[k].pt.x;
            const int y = (int)voKeyPoints[k].pt.y;
            const uchar* acRef = oRefMat.data+oInputImg.step.p[0]*y+oInputImg.step.p[1]*x;
            LBSP::computeDescriptor(oInputImg,acRef,x,y,anThreshold,((ushort*)(oDesc.data+oDesc.step.p[0]*k)));
        }
    }
}

void lbsp_computeImpl(const cv::Mat& oInputImg, const cv::Mat& oRefImg, const std::vector<cv::KeyPoint>& voKeyPoints, cv::Mat& oDesc, bool bSingleColumnDesc, float fThreshold, size_t nThresholdOffset) {
    static_assert(LBSP::DESC_SIZE==2,"bad assumptions in impl below");
    lvAssert_(!oInputImg.empty() && oInputImg.isContinuous() && (oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3),"input image must be non-empty, continuous, and of type 8UC1/8UC3");
    lvAssert_(oRefImg.empty() || (oRefImg.size==oInputImg.size && oRefImg.type()==oInputImg.type()),"ref image must be empty, or of the same size/type as the input image");
    lvAssert_(fThreshold>=0,"lbsp internal relative threshold must be non-negative");
    const size_t nChannels = (size_t)oInputImg.channels();
    const cv::Mat& oRefMat = oRefImg.empty()?oInputImg:oRefImg;
    const size_t nKeyPoints = voKeyPoints.size();
    if(nChannels==1) {
        if(bSingleColumnDesc)
            oDesc.create((int)nKeyPoints,1,CV_16UC1);
        else
            oDesc.create(oInputImg.size(),CV_16UC1);
        for(size_t k=0; k<nKeyPoints; ++k) {
            const int x = (int)voKeyPoints[k].pt.x;
            const int y = (int)voKeyPoints[k].pt.y;
            const uchar nThreshold = cv::saturate_cast<uchar>(oRefMat.at<uchar>(y,x)*fThreshold+nThresholdOffset);
            ushort& nResult = bSingleColumnDesc?oDesc.at<ushort>((int)k):oDesc.at<ushort>(y,x);
            LBSP::computeDescriptor<1>(oInputImg,oRefMat.at<uchar>(y,x),x,y,0,nThreshold,nResult);
        }
    }
    else { //nChannels==3
        if(bSingleColumnDesc)
            oDesc.create((int)nKeyPoints,1,CV_16UC3);
        else
            oDesc.create(oInputImg.size(),CV_16UC3);
        for(size_t k=0; k<nKeyPoints; ++k) {
            const int x = (int)voKeyPoints[k].pt.x;
            const int y = (int)voKeyPoints[k].pt.y;
            const uchar* acRef = oRefMat.data+oInputImg.step.p[0]*y+oInputImg.step.p[1]*x;
            alignas(16) const std::array<uchar,3> anThreshold = {cv::saturate_cast<uchar>(acRef[0]*fThreshold+nThresholdOffset),
                                                                 cv::saturate_cast<uchar>(acRef[1]*fThreshold+nThresholdOffset),
                                                                 cv::saturate_cast<uchar>(acRef[2]*fThreshold+nThresholdOffset)};
            ushort* anResult = (ushort*)(bSingleColumnDesc?(oDesc.data+oDesc.step.p[0]*k):(oDesc.data+oDesc.step.p[0]*y+oDesc.step.p[1]*x));
            LBSP::computeDescriptor(oInputImg,acRef,x,y,anThreshold,anResult);
        }
    }
}

} // namespace

void LBSP::compute2(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const {
    lvAssert_(!oImage.empty(),"input image must be non-empty");
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImage.size(),PATCH_SIZE/2);
    cv::KeyPointsFilter::runByKeypointSize(voKeypoints,std::numeric_limits<float>::epsilon());
    if(voKeypoints.empty()) {
        oDescriptors.release();
        return;
    }
    if(m_bOnlyUsingAbsThreshold)
        lbsp_computeImpl(oImage,m_oRefImage,voKeypoints,oDescriptors,false,m_nThreshold);
    else
        lbsp_computeImpl(oImage,m_oRefImage,voKeypoints,oDescriptors,false,m_fRelThreshold,m_nThreshold);
}

void LBSP::compute2(const std::vector<cv::Mat>& voImageCollection, std::vector<std::vector<cv::KeyPoint> >& vvoPointCollection, std::vector<cv::Mat>& voDescCollection) const {
    lvAssert_(voImageCollection.size()==vvoPointCollection.size(),"number of images must match number of keypoint lists");
    voDescCollection.resize(voImageCollection.size());
    for(size_t i=0; i<voImageCollection.size(); i++)
        compute2(voImageCollection[i], vvoPointCollection[i], voDescCollection[i]);
}

void LBSP::computeImpl(const cv::Mat& oImage, std::vector<cv::KeyPoint>& voKeypoints, cv::Mat& oDescriptors) const {
    lvAssert_(!oImage.empty(),"input image must be non-empty");
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImage.size(),PATCH_SIZE/2);
    cv::KeyPointsFilter::runByKeypointSize(voKeypoints,std::numeric_limits<float>::epsilon());
    if(voKeypoints.empty()) {
        oDescriptors.release();
        return;
    }
    if(m_bOnlyUsingAbsThreshold)
        lbsp_computeImpl(oImage,m_oRefImage,voKeypoints,oDescriptors,true,m_nThreshold);
    else
        lbsp_computeImpl(oImage,m_oRefImage,voKeypoints,oDescriptors,true,m_fRelThreshold,m_nThreshold);
}

void LBSP::reshapeDesc(cv::Size oSize, const std::vector<cv::KeyPoint>& voKeypoints, const cv::Mat& oDescriptors, cv::Mat& oOutput) {
    static_assert(LBSP::DESC_SIZE==2,"bad assumptions in impl below");
    lvAssert_(!voKeypoints.empty(),"keypoint array must be non-empty");
    lvAssert_(!oDescriptors.empty() && oDescriptors.isContinuous() && oDescriptors.cols==1,"descriptor mat must be non-empty, continuous, and have only one column");
    lvAssert_(oSize.width>0 && oSize.height>0,"expected output desc image size must not contain null dimensions");
    lvAssert_(oDescriptors.type()==CV_16UC1 || oDescriptors.type()==CV_16UC3,"descriptor mat type must be 16UC1/16UC3");
    const size_t nChannels = (size_t)oDescriptors.channels();
    const size_t nKeyPoints = voKeypoints.size();
    if(nChannels==1) {
        oOutput.create(oSize,CV_16UC1);
        oOutput = cv::Scalar_<ushort>(0);
        for(size_t k=0; k<nKeyPoints; ++k)
            oOutput.at<ushort>(voKeypoints[k].pt) = oDescriptors.at<ushort>((int)k);
    }
    else { //nChannels==3
        oOutput.create(oSize,CV_16UC3);
        oOutput = cv::Scalar_<ushort>(0,0,0);
        for(size_t k=0; k<nKeyPoints; ++k) {
            ushort* output_ptr = (ushort*)(oOutput.data + oOutput.step.p[0]*(int)voKeypoints[k].pt.y);
            const ushort* const desc_ptr = (ushort*)(oDescriptors.data + oDescriptors.step.p[0]*k);
            const size_t idx = 3*(int)voKeypoints[k].pt.x;
            for(size_t n=0; n<3; ++n)
                output_ptr[idx+n] = desc_ptr[n];
        }
    }
}

void LBSP::calcDescImgDiff(const cv::Mat& oDesc1, const cv::Mat& oDesc2, cv::Mat& oOutput, bool bForceMergeChannels) {
    static_assert(LBSP::DESC_SIZE_BITS<=UCHAR_MAX,"bad assumptions in impl below");
    static_assert(LBSP::DESC_SIZE==2,"bad assumptions in impl below");
    lvAssert_(oDesc1.isContinuous() && (oDesc1.type()==CV_16UC1 || oDesc1.type()==CV_16UC3),"desc1 mat must be continuous, and of type 16UC1/16UC3");
    lvAssert_(oDesc2.isContinuous() && (oDesc2.type()==CV_16UC1 || oDesc2.type()==CV_16UC3),"desc2 mat must be continuous, and of type 16UC1/16UC3");
    lvAssert_(oDesc1.size()==oDesc2.size() && oDesc1.type()==oDesc2.type(),"size/type of descriptor mats must match");
    lvDbgAssert(oDesc1.step.p[0]==oDesc2.step.p[0] && oDesc1.step.p[1]==oDesc2.step.p[1]);
    const float fScaleFactor = (float)UCHAR_MAX/(LBSP::DESC_SIZE_BITS);
    const size_t nChannels = CV_MAT_CN(oDesc1.type());
    const size_t _step_row = oDesc1.step.p[0];
    if(nChannels==1) {
        oOutput.create(oDesc1.size(),CV_8UC1);
        oOutput = cv::Scalar(0);
        for(int i=0; i<oDesc1.rows; ++i) {
            const size_t idx = _step_row*i;
            const ushort* const desc1_ptr = (ushort*)(oDesc1.data+idx);
            const ushort* const desc2_ptr = (ushort*)(oDesc2.data+idx);
            for(int j=0; j<oDesc1.cols; ++j)
                oOutput.at<uchar>(i,j) = (uchar)(fScaleFactor*lv::hdist(desc1_ptr[j],desc2_ptr[j]));
        }
    }
    else { //nChannels==3
        if(bForceMergeChannels)
            oOutput.create(oDesc1.size(),CV_8UC1);
        else
            oOutput.create(oDesc1.size(),CV_8UC3);
        oOutput = cv::Scalar::all(0);
        for(int i=0; i<oDesc1.rows; ++i) {
            const size_t idx =  _step_row*i;
            const ushort* const desc1_ptr = (ushort*)(oDesc1.data+idx);
            const ushort* const desc2_ptr = (ushort*)(oDesc2.data+idx);
            uchar* output_ptr = oOutput.data + oOutput.step.p[0]*i;
            for(int j=0; j<oDesc1.cols; ++j) {
                for(size_t n=0;n<3; ++n) {
                    const size_t idx2 = 3*j+n;
                    if(bForceMergeChannels)
                        output_ptr[j] += (uchar)((fScaleFactor*lv::hdist(desc1_ptr[idx2],desc2_ptr[idx2]))/3);
                    else
                        output_ptr[idx2] = (uchar)(fScaleFactor*lv::hdist(desc1_ptr[idx2],desc2_ptr[idx2]));
                }
            }
        }
    }
}

void LBSP::validateKeyPoints(std::vector<cv::KeyPoint>& voKeypoints, cv::Size oImgSize) {
    cv::KeyPointsFilter::runByImageBorder(voKeypoints,oImgSize,PATCH_SIZE/2);
}

void LBSP::validateROI(cv::Mat& oROI) {
    lvAssert_(!oROI.empty() && oROI.type()==CV_8UC1,"input ROI must be non-empty and of type 8UC1");
    cv::Mat oROI_new(oROI.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    const size_t nBorderSize = PATCH_SIZE/2;
    const cv::Rect nROI_inner(nBorderSize,nBorderSize,oROI.cols-nBorderSize*2,oROI.rows-nBorderSize*2);
    cv::Mat(oROI,nROI_inner).copyTo(cv::Mat(oROI_new,nROI_inner));
    oROI = oROI_new;
}

#if HAVE_GLSL

std::string LBSP::getShaderFunctionSource(size_t nChannels, bool bUseSharedDataPreload, const glm::uvec2& vWorkGroupSize) {
    lvAssert_(nChannels==4 || nChannels==1,"code can only be generated for 1ch/4ch textures");
    std::stringstream ssSrc;
    // @@@@@ split lookup/threshold?
    if(!bUseSharedDataPreload) ssSrc <<
             "uvec3 lbsp(in uvec3 t, in uvec3 ref, in layout(" << (nChannels==4?"rgba8ui":"r8ui") << ") readonly uimage2D mData, in ivec2 vCoords) {\n"
             "    return (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2(-1, 1)).rgb)-ivec3(ref))),t)) << 15)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 1,-1)).rgb)-ivec3(ref))),t)) << 14)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 1, 1)).rgb)-ivec3(ref))),t)) << 13)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2(-1,-1)).rgb)-ivec3(ref))),t)) << 12)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 1, 0)).rgb)-ivec3(ref))),t)) << 11)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 0,-1)).rgb)-ivec3(ref))),t)) << 10)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2(-1, 0)).rgb)-ivec3(ref))),t)) << 9)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 0, 1)).rgb)-ivec3(ref))),t)) << 8)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2(-2,-2)).rgb)-ivec3(ref))),t)) << 7)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 2, 2)).rgb)-ivec3(ref))),t)) << 6)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 2,-2)).rgb)-ivec3(ref))),t)) << 5)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2(-2, 2)).rgb)-ivec3(ref))),t)) << 4)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 0, 2)).rgb)-ivec3(ref))),t)) << 3)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 0,-2)).rgb)-ivec3(ref))),t)) << 2)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2( 2, 0)).rgb)-ivec3(ref))),t)) << 1)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(imageLoad(mData,vCoords+ivec2(-2, 0)).rgb)-ivec3(ref))),t)));\n"
             "}\n";
    else { ssSrc <<
             GLShader::getComputeShaderFunctionSource_SharedDataPreLoad(nChannels,vWorkGroupSize,LBSP::PATCH_SIZE/2) <<
             "uvec3 lbsp(in uvec3 t, in uvec3 ref, in ivec2 vCoords) {\n"
             "    ivec2 vLocalCoords = vCoords-ivec2(gl_GlobalInvocationID.xy)+ivec2(gl_LocalInvocationID.xy)+ivec2(" << LBSP::PATCH_SIZE/2 << ");\n"
             "    return (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y+1][vLocalCoords.x-1].rgb)-ivec3(ref))),t)) << 15)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y-1][vLocalCoords.x+1].rgb)-ivec3(ref))),t)) << 14)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y+1][vLocalCoords.x+1].rgb)-ivec3(ref))),t)) << 13)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y-1][vLocalCoords.x-1].rgb)-ivec3(ref))),t)) << 12)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y  ][vLocalCoords.x+1].rgb)-ivec3(ref))),t)) << 11)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y-1][vLocalCoords.x  ].rgb)-ivec3(ref))),t)) << 10)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y  ][vLocalCoords.x-1].rgb)-ivec3(ref))),t)) << 9)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y+1][vLocalCoords.x  ].rgb)-ivec3(ref))),t)) << 8)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y-2][vLocalCoords.x-2].rgb)-ivec3(ref))),t)) << 7)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y+2][vLocalCoords.x+2].rgb)-ivec3(ref))),t)) << 6)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y-2][vLocalCoords.x+2].rgb)-ivec3(ref))),t)) << 5)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y+2][vLocalCoords.x-2].rgb)-ivec3(ref))),t)) << 4)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y+2][vLocalCoords.x  ].rgb)-ivec3(ref))),t)) << 3)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y-2][vLocalCoords.x  ].rgb)-ivec3(ref))),t)) << 2)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y  ][vLocalCoords.x+2].rgb)-ivec3(ref))),t)) << 1)\n"
             "         + (uvec3(greaterThan(uvec3(abs(ivec3(avPreloadData[vLocalCoords.y  ][vLocalCoords.x-2].rgb)-ivec3(ref))),t)));\n"
             "}\n";
    }
    return ssSrc.str();
}

#endif //HAVE_GLSL
