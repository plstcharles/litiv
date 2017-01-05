
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

#include "litiv/utils/platform.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define MAT_COND_DEPTH_TYPE(depth_flag,depth_type,depth_alt) \
    std::conditional_t<CV_MAT_DEPTH(nTypeFlag)==depth_flag,depth_type,depth_alt>

namespace cv { // extending cv

    struct DisplayHelper;
    using DisplayHelperPtr = std::shared_ptr<DisplayHelper>;

    /// type traits helper which provides basic static info on ocv matrix element types
    template<int nTypeFlag>
    struct MatTypeInfo {
        typedef std::enable_if_t<(CV_MAT_DEPTH(nTypeFlag)>=0 && CV_MAT_DEPTH(nTypeFlag)<=6),
            MAT_COND_DEPTH_TYPE(0,uchar,
                MAT_COND_DEPTH_TYPE(1,char,
                    MAT_COND_DEPTH_TYPE(2,ushort,
                        MAT_COND_DEPTH_TYPE(3,short,
                            MAT_COND_DEPTH_TYPE(4,int,
                                MAT_COND_DEPTH_TYPE(5,float,
                                    MAT_COND_DEPTH_TYPE(6,double,void)))))))> base_type;
        static constexpr int nChannels = CV_MAT_CN(nTypeFlag);
    };

    /// helper function to zero-init sparse and non-sparse matrices (sparse mat overload)
    template<typename T>
    inline void zeroMat(cv::SparseMat_<T>& oMat) {
        oMat.clear();
    }

    /// helper function to zero-init sparse and non-sparse matrices (regular mat specialization)
    template<typename T>
    inline void zeroMat(cv::Mat_<T>& oMat) {
        oMat = T();
    }

    /// helper function to fetch references from sparse and non-sparse matrices (sparse mat overload)
    template<typename T>
    inline T& getElem(cv::SparseMat_<T>& oMat, const int* idx) {
        return oMat.ref(idx);
    }

    /// helper function to fetch references from sparse and non-sparse matrices (regular mat overload)
    template<typename T>
    inline T& getElem(cv::Mat_<T>& oMat, const int* idx) {
        return oMat(idx);
    }

    /// helper function to count valid/allocated elements in sparse and non-sparse matrices (sparse mat overload)
    template<typename T>
    inline size_t getElemCount(cv::SparseMat_<T>& oMat) {
        return oMat.nzcount();
    }

    /// helper function to count valid/allocated elements in sparse and non-sparse matrices (regular mat overload)
    template<typename T>
    inline size_t getElemCount(cv::Mat_<T>& oMat) {
        return oMat.total();
    }

    /// returns pixel coordinates clamped to the given image & border size
    inline void clampImageCoords(int& nSampleCoord_X, int& nSampleCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        lvDbgAssert_(nBorderSize>=0,"border size cannot be negative");
        lvDbgAssert_(oImageSize.area()>=0,"image size cannot be negative");
        if(nSampleCoord_X<nBorderSize)
            nSampleCoord_X = nBorderSize;
        else if(nSampleCoord_X>=oImageSize.width-nBorderSize)
            nSampleCoord_X = oImageSize.width-nBorderSize-1;
        if(nSampleCoord_Y<nBorderSize)
            nSampleCoord_Y = nBorderSize;
        else if(nSampleCoord_Y>=oImageSize.height-nBorderSize)
            nSampleCoord_Y = oImageSize.height-nBorderSize-1;
    }

    /// returns a random init/sampling position for the specified pixel position, given a predefined kernel; also guards against out-of-bounds values via image/border size check
    template<int nKernelHeight, int nKernelWidth>
    inline void getRandSamplePosition(const std::array<std::array<int,nKernelWidth>,nKernelHeight>& anSamplesInitPattern,
                                      const int nSamplesInitPatternTot, int& nSampleCoord_X, int& nSampleCoord_Y,
                                      const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        static_assert(nKernelWidth>0 && nKernelHeight>0,"invalid init pattern array size");
        lvDbgAssert_(nSamplesInitPatternTot>0,"pattern max count must be positive");
        int r = 1+rand()%nSamplesInitPatternTot;
        for(nSampleCoord_Y=0; nSampleCoord_Y<nKernelHeight; ++nSampleCoord_Y) {
            for(nSampleCoord_X=0; nSampleCoord_X<nKernelWidth; ++nSampleCoord_X) {
                r -= anSamplesInitPattern[nSampleCoord_Y][nSampleCoord_X];
                if(r<=0)
                    goto stop;
            }
        }
        stop:
        nSampleCoord_X += nOrigCoord_X-nKernelWidth/2;
        nSampleCoord_Y += nOrigCoord_Y-nKernelHeight/2;
        clampImageCoords(nSampleCoord_X,nSampleCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandSamplePosition_3x3_std1(int& nSampleCoord_X, int& nSampleCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        // based on 'floor(fspecial('gaussian',3,1)*256)'
        static_assert(sizeof(std::array<int,3>)==sizeof(int)*3,"bad std::array stl impl");
        static const int s_nSamplesInitPatternTot = 256;
        static const std::array<std::array<int,3>,3> s_anSamplesInitPattern ={
                std::array<int,3>{19,32,19,},
                std::array<int,3>{32,52,32,},
                std::array<int,3>{19,32,19,},
        };
        getRandSamplePosition<3,3>(s_anSamplesInitPattern,s_nSamplesInitPatternTot,nSampleCoord_X,nSampleCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandSamplePosition_7x7_std2(int& nSampleCoord_X, int& nSampleCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        // based on 'floor(fspecial('gaussian',7,2)*512)'
        static_assert(sizeof(std::array<int,7>)==sizeof(int)*7,"bad std::array stl impl");
        static const int s_nSamplesInitPatternTot = 512;
        static const std::array<std::array<int,7>,7> s_anSamplesInitPattern ={
                std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
                std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
                std::array<int,7>{ 6,12,21,25,21,12, 6,},
                std::array<int,7>{ 7,14,25,28,25,14, 7,},
                std::array<int,7>{ 6,12,21,25,21,12, 6,},
                std::array<int,7>{ 4, 8,12,14,12, 8, 4,},
                std::array<int,7>{ 2, 4, 6, 7, 6, 4, 2,},
        };
        getRandSamplePosition<7,7>(s_anSamplesInitPattern,s_nSamplesInitPatternTot,nSampleCoord_X,nSampleCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random neighbor position for the specified pixel position, given a predefined neighborhood; also guards against out-of-bounds values via image/border size check.
    template<int nNeighborCount>
    inline void getRandNeighborPosition(const std::array<std::array<int,2>,nNeighborCount>& anNeighborPattern,
                                        int& nNeighborCoord_X, int& nNeighborCoord_Y,
                                        const int nOrigCoord_X, const int nOrigCoord_Y,
                                        const int nBorderSize, const cv::Size& oImageSize) {
        static_assert(nNeighborCount>0,"invalid input neighbor pattern array size");
        int r = rand()%nNeighborCount;
        nNeighborCoord_X = nOrigCoord_X+anNeighborPattern[r][0];
        nNeighborCoord_Y = nOrigCoord_Y+anNeighborPattern[r][1];
        clampImageCoords(nNeighborCoord_X,nNeighborCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandNeighborPosition_3x3(int& nNeighborCoord_X, int& nNeighborCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        typedef std::array<int,2> Nb;
        static const std::array<std::array<int,2>,8> s_anNeighborPattern ={
                Nb{-1, 1},Nb{0, 1},Nb{1, 1},
                Nb{-1, 0},         Nb{1, 0},
                Nb{-1,-1},Nb{0,-1},Nb{1,-1},
        };
        getRandNeighborPosition<8>(s_anNeighborPattern,nNeighborCoord_X,nNeighborCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    inline void getRandNeighborPosition_5x5(int& nNeighborCoord_X, int& nNeighborCoord_Y, const int nOrigCoord_X, const int nOrigCoord_Y, const int nBorderSize, const cv::Size& oImageSize) {
        typedef std::array<int,2> Nb;
        static const std::array<std::array<int,2>,24> s_anNeighborPattern ={
                Nb{-2, 2},Nb{-1, 2},Nb{0, 2},Nb{1, 2},Nb{2, 2},
                Nb{-2, 1},Nb{-1, 1},Nb{0, 1},Nb{1, 1},Nb{2, 1},
                Nb{-2, 0},Nb{-1, 0},         Nb{1, 0},Nb{2, 0},
                Nb{-2,-1},Nb{-1,-1},Nb{0,-1},Nb{1,-1},Nb{2,-1},
                Nb{-2,-2},Nb{-1,-2},Nb{0,-2},Nb{1,-2},Nb{2,-2},
        };
        getRandNeighborPosition<24>(s_anNeighborPattern,nNeighborCoord_X,nNeighborCoord_Y,nOrigCoord_X,nOrigCoord_Y,nBorderSize,oImageSize);
    }

    /// writes a given text string on an image using the original cv::putText (this function only acts as a simplification wrapper)
    inline void putText(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom=false, const cv::Point2i& oOffset=cv::Point2i(4,15), int nThickness=2, double dScale=1.2) {
        cv::putText(oImg,sText,cv::Point(oOffset.x,bBottom?(oImg.rows-oOffset.y):oOffset.y),cv::FONT_HERSHEY_PLAIN,dScale,vColor,nThickness,cv::LINE_AA);
    }

    /// prints the content of a matrix to the given stream with constant output element size
    template<typename T>
    inline void printMatrix(const cv::Mat_<T>& oMat, std::ostream& os=std::cout) {
        lvAssert_(oMat.dims==2,"function currently only defined for 2d mats; split dims and call for 2d slices");
        if(oMat.empty() || oMat.size().area()==0) {
            os << "   <empty>" << std::endl;
            return;
        }
        const size_t nMaxMetaColWidth = (size_t)std::max(lv::digit_count(oMat.cols),lv::digit_count(oMat.rows));
        double dMin,dMax;
        cv::minMaxIdx(oMat,&dMin,&dMax);
        const T tMin = (T)dMin;
        const T tMax = (T)dMax;
        constexpr bool bIsFloat = !std::is_integral<T>::value;
        using PrintType = std::conditional_t<bIsFloat,float,int64_t>;
        const bool bIsNormalized = tMax<=T(1) && tMin>=T(0); // useful for floats only
        const bool bHasNegative = int64_t(tMin)<int64_t(0);
        const size_t nMaxColWidth = size_t(bIsFloat?(bIsNormalized?6:(std::max(lv::digit_count((int64_t)tMin),lv::digit_count((int64_t)tMax))+5+int(bHasNegative!=0))):(std::max(lv::digit_count(tMin),lv::digit_count(tMax))+int(bHasNegative!=0)));
        const std::string sFormat = bIsFloat?(bIsNormalized?std::string("%6.4f"):((bHasNegative?std::string("%+"):std::string("%"))+std::to_string(nMaxColWidth)+std::string(".4f"))):((bHasNegative?std::string("%+"):std::string("%"))+std::to_string(nMaxColWidth)+std::string(PRId64));
        const std::string sMetaFormat = std::string("%")+std::to_string(nMaxMetaColWidth)+"i";
        const std::string sSpacer = "  ";
        const auto lPrinter = [&](const T& v) {os << sSpacer << lv::putf(sFormat.c_str(),(PrintType)v);};
        os << std::endl << std::string("   ")+std::string(nMaxMetaColWidth,' ')+std::string("x=");
        for(int nColIdx=0; nColIdx<oMat.cols; ++nColIdx)
            os << sSpacer << lv::clampString(lv::putf(sMetaFormat.c_str(),nColIdx),nMaxColWidth);
        os << std::endl << std::endl;
        for(int nRowIdx=0; nRowIdx<oMat.rows; ++nRowIdx) {
            os << " y=" << lv::putf(sMetaFormat.c_str(),nRowIdx) << sSpacer;
            for(int nColIdx=0; nColIdx<oMat.cols; ++nColIdx)
                lPrinter(oMat.template at<T>(nRowIdx,nColIdx));
            os << std::endl;
        }
        os << std::endl;
    }

    /// removes all keypoints from voKPs which fall on null values (or outside the bounds) of oROI
    inline void validateKeyPoints(const cv::Mat& oROI, std::vector<cv::KeyPoint>& voKPs) {
        if(oROI.empty())
            return;
        lvAssert_(oROI.type()==CV_8UC1,"input ROI must be of type 8UC1");
        std::vector<cv::KeyPoint> voNewKPs;
        voNewKPs.reserve(voKPs.size());
        for(size_t k=0; k<voKPs.size(); ++k) {
            if(voKPs[k].pt.x>=0 && voKPs[k].pt.x<oROI.cols && voKPs[k].pt.y>=0 && voKPs[k].pt.y<oROI.rows && oROI.at<uchar>(voKPs[k].pt)>0)
                voNewKPs.push_back(voKPs[k]);
        }
        voKPs = voNewKPs;
    }

    /// returns the vector of all sorted unique values contained in a templated matrix
    template<typename T>
    inline std::vector<T> unique(const cv::Mat_<T>& oMat) {
        if(oMat.empty())
            return std::vector<T>();
        const std::set<T> mMap(oMat.begin(),oMat.end());
        return std::vector<T>(mMap.begin(),mMap.end());
    }

    /// returns whether the two matrices are equal or not
    template<typename T>
    inline bool isEqual(const cv::Mat& a, const cv::Mat& b) {
        if(a.empty() && b.empty())
            return true;
        if(a.dims!=b.dims || a.size!=b.size || a.type()!=b.type())
            return false;
        lvDbgAssert(a.total()*a.elemSize()==b.total()*b.elemSize());
        if(a.isContinuous() && b.isContinuous())
            return std::equal((T*)a.data,(T*)(a.data+a.total()*a.elemSize()),(T*)b.data);
        else {
            for(size_t nElemIdx=0; nElemIdx<a.total(); ++nElemIdx)
                if(a.at<T>(int(nElemIdx))!=b.at<T>(int(nElemIdx)))
                    return false;
            return true;
        }
    }

    /// returns whether the two matrices are nearly equal or not, given a maximum allowed error
    template<typename T>
    inline bool isNearlyEqual(const cv::Mat& a, const cv::Mat& b, T eps) {
        if(a.empty() && b.empty())
            return true;
        if(a.dims!=b.dims || a.size!=b.size || a.type()!=b.type())
            return false;
        lvDbgAssert(a.total()*a.elemSize()==b.total()*b.elemSize());
        if(a.isContinuous() && b.isContinuous())
            return std::equal((T*)a.data,(T*)(a.data+a.total()*a.elemSize()),(T*)b.data,[&eps](const T& _a, const T& _b){
                return std::abs(double(_a)-double(_b))<=double(eps);
            });
        else {
            for(size_t nElemIdx=0; nElemIdx<a.total(); ++nElemIdx)
                if(std::abs(double(a.at<T>(int(nElemIdx)))-double(b.at<T>(int(nElemIdx))))>double(eps))
                    return false;
            return true;
        }
    }

    /// converts a single HSL triplet (0-360 hue, 0-1 sat & lightness) into an 8-bit RGB triplet
    inline cv::Vec3b getBGRFromHSL(float fHue, float fSaturation, float fLightness) {
        // this function is not intended for fast conversions; use OpenCV's cvtColor for large-scale stuff
        lvDbgAssert__(fHue>=0.0f && fHue<360.0f,"bad input hue range (fHue=%f)",fHue);
        lvDbgAssert__(fSaturation>=0.0f && fSaturation<=1.0f,"bad input saturation range (fSaturation=%f)",fSaturation);
        lvDbgAssert__(fLightness>=0.0f && fLightness<=1.0f,"bad input lightness range (fLightness=%f)",fLightness);
        if(fSaturation==0.0f)
            return cv::Vec3b::all(cv::saturate_cast<uchar>(std::round(fLightness*255)));
        if(fLightness==0.0f)
            return cv::Vec3b::all(0);
        if(fLightness==1.0f)
            return cv::Vec3b::all(255);
        const auto lH2RGB = [&](float p, float q, float t) {
            if(t<0.0f)
                t += 1;
            if(t>1.0f)
                t -= 1;
            if(t<1.0f/6)
                return p + (q - p) * 6.0f * t;
            if(t<1.0f/2)
                return q;
            if(t<2.0f/3)
                return p + (q - p) * (2.0f/3 - t) * 6.0f;
            return p;
        };
        const float q = (fLightness<0.5f)?fLightness*(1+fSaturation):fLightness+fSaturation-fLightness*fSaturation;
        const float p = 2.0f*fLightness-q;
        const float h = fHue/360.0f;
        return cv::Vec3b(cv::saturate_cast<uchar>(std::round(lH2RGB(p,q,h-1.0f/3)*255)),cv::saturate_cast<uchar>(std::round(lH2RGB(p,q,h)*255)),cv::saturate_cast<uchar>(std::round(lH2RGB(p,q,h+1.0f/3)*255)));
    }

    /// converts a single HSL triplet (0-360 hue, 0-1 sat & lightness) into an 8-bit RGB triplet
    inline cv::Vec3b getBGRFromHSL(const cv::Vec3f& vHSL) {
        return getBGRFromHSL(vHSL[0],vHSL[1],vHSL[2]);
    }

    /// converts a single 8-bit RGB triplet into an HSL triplet (0-360 hue, 0-1 sat & lightness)
    inline cv::Vec3f getHSLFromBGR(const cv::Vec3b& vBGR) {
        // this function is not intended for fast conversions; use OpenCV's cvtColor for large-scale stuff
        const float r = vBGR[2]/255.0f, g=vBGR[1]/255.0f, b=vBGR[0]/255.0f;
        const float fMaxChroma = std::max(r,std::max(g,b));
        const float fMinChroma = std::min(r,std::min(g,b));
        const float fLightness = (fMaxChroma+fMinChroma)/2.0f;
        if(fMaxChroma==fMinChroma)
            return cv::Vec3f(0.0f,0.0f,fLightness);
        const float fDiffChroma = fMaxChroma-fMinChroma;
        const float fSaturation = std::max(0.0f,std::min(fDiffChroma/(1.0f-std::abs(2.0f*fLightness-1.0f)),1.0f));
        const float fHue = (fMaxChroma==r?(((g-b)/fDiffChroma)+(g<b?6.0f:0.0f)):(fMaxChroma==g?((b-r)/fDiffChroma+2.0f):(r-g)/fDiffChroma+4.0f))*60.0f;
        return cv::Vec3f(fHue,fSaturation,fLightness);
    }

    /// returns a 8uc3 color map such that all equal values in the given matrix are assigned the same unique color in the map
    template<typename T>
    inline cv::Mat getUniqueColorMap(const cv::Mat_<T>& m, std::map<T,cv::Vec3b>* pmColorMap=nullptr) {
        static_assert(std::is_integral<T>::value,"function only defined for integer maps");
        lvAssert_(m.dims==2,"function currently only defined for 2d mats; split dims and call for 2d slices");
        if(m.empty())
            return cv::Mat();
        const std::vector<T> vUniques = cv::unique(m);
        const size_t nColors = vUniques.size();
        if(nColors<=1)
            return cv::Mat(m.size(),CV_8UC3,cv::Scalar::all(255));
        lvAssert_(nColors<720,"too many uniques for internal multi-slice HSL model");
        const size_t nMaxAng = 45;
        //const float fMinSat = 0.33f, fMaxSat = 1.0f;
        const float fAvgLight = 0.50f, fVarLight = 0.25f;
        const size_t nDivs = size_t(std::ceil(std::log2(nColors)));
        std::vector<cv::Vec3b> vColors(nColors);
        size_t nColorIdx = 0;
        for(size_t nDivIdx=0; nDivIdx<nDivs && nColorIdx<nColors; ++nDivIdx) {
            const size_t nSampleCount = std::min(std::max(nColors/(size_t(1)<<(nDivIdx+1)),(360/nMaxAng)-1),nColors-nColorIdx);
            const float fCurrSat = 1.0f; //const float fCurrSat = fMaxSat-((fMaxSat-fMinSat)/nDivs)*nDivIdx;
            const float fCurrLight = fAvgLight + int(nDivIdx>0)*(((nDivIdx%2)?-fVarLight:fVarLight)/((std::max(nDivIdx,size_t(1))+1)/2));
            std::unordered_set<ushort> mDivAngSet;
            ushort nCurrAng = ushort(rand())%nMaxAng;
            for(size_t nSampleIdx=0; nSampleIdx<nSampleCount; ++nSampleIdx) {
                lvDbgAssert(mDivAngSet.size()<360);
                while(mDivAngSet.count(nCurrAng))
                    ++nCurrAng %= 360;
                mDivAngSet.insert(nCurrAng);
                vColors[nColorIdx++] = cv::getBGRFromHSL(float(nCurrAng),fCurrSat,fCurrLight);
                nCurrAng = (nCurrAng+360/nSampleCount)%360;
            }
        }
        std::random_device oRandDev;
        std::default_random_engine oRandEng(oRandDev());
        std::shuffle(vColors.begin(),vColors.end(),oRandEng);
        std::map<T,cv::Vec3b> mColorMap;
        for(nColorIdx=0; nColorIdx<nColors; ++nColorIdx)
            mColorMap[vUniques[nColorIdx]] = vColors[nColorIdx];
        cv::Mat oOutputMap(m.size(),CV_8UC3);
        for(size_t nElemIdx=0; nElemIdx<m.total(); ++nElemIdx)
            oOutputMap.at<cv::Vec3b>((int)nElemIdx) = mColorMap[m((int)nElemIdx)];
        if(pmColorMap)
            std::swap(*pmColorMap,mColorMap);
        return oOutputMap;
    }

    /// helper struct for image display & callback management (must be created via DisplayHelper::create due to enable_shared_from_this interface)
    struct DisplayHelper : lv::enable_shared_from_this<DisplayHelper> {
        /// displayed window title (specified on creation)
        const std::string m_sDisplayName;
        /// displayed window maximum size (specified on creation)
        const cv::Size m_oMaxDisplaySize;
        /// general-use file storage tied to the display helper (will be closed & printed on helper destruction)
        cv::FileStorage m_oFS;
        /// public mutex that should be always used if m_oLatestMouseEvent is accessed externally
        std::mutex m_oEventMutex;
        /// raw-interpreted callback data structure
        struct CallbackData {
            cv::Point2i oPosition,oInternalPosition;
            cv::Size oTileSize,oDisplaySize;
            int nEvent,nFlags;
        } m_oLatestMouseEvent;
        /// by default, comes with a filestorage algorithms can use for debug
        static DisplayHelperPtr create(const std::string& sDisplayName,
                                       const std::string& sDebugFSDirPath="./",
                                       const cv::Size& oMaxSize=cv::Size(1920,1080),
                                       int nWindowFlags=cv::WINDOW_AUTOSIZE);
        /// will reformat the given image, print the index and mouse cursor point on it, and show it
        void display(const cv::Mat& oImage, size_t nIdx);
        /// will reformat the given images, print the index and mouse cursor point on them, and show them horizontally concatenated
        void display(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oOutputImg, size_t nIdx);
        /// will reformat the given images, print their names and mouse cursor point on them, and show them based on row-col ordering
        void display(const std::vector<std::vector<std::pair<cv::Mat,std::string>>>& vvImageNamePairs, const cv::Size& oSuggestedTileSize);
        /// sets the provided external function to be called when mouse events are captured for the displayed window
        void setMouseCallback(std::function<void(const CallbackData&)> lCallback);
        /// sets whether the waitKey call should block and wait for a key press or allow timeouts and return without one
        void setContinuousUpdates(bool b);
        /// calls cv::waitKey (blocking for a key press if m_bContinuousUpdates is false) and returns the cv::waitKey result
        int waitKey(int nDefaultSleepDelay=1);
        /// desctructor automatically closes its window
        ~DisplayHelper();
    protected:
        /// should always be constructor via static 'create' member due to enable_shared_from_this interface
        DisplayHelper(const std::string& sDisplayName, const std::string& sDebugFSDirPath, const cv::Size& oMaxSize, int nWindowFlags);
        /// local entrypoint for opencv mouse callbacks
        void onMouseEventCallback(int nEvent, int x, int y, int nFlags);
        /// global entrypoint for opencv mouse callbacks
        static void onMouseEvent(int nEvent, int x, int y, int nFlags, void* pData);
        cv::Size m_oLastDisplaySize,m_oLastTileSize;
        bool m_bContinuousUpdates,m_bFirstDisplay;
        cv::Mat m_oLastDisplay;
        std::function<void(int,int,int,int)> m_lInternalCallback;
        std::function<void(const CallbackData&)> m_lExternalCallback;
    private:
        DisplayHelper(const DisplayHelper&) = delete;
        DisplayHelper& operator=(const DisplayHelper&) = delete;
    };

    /// list of archive types supported by cv::write and cv::read
    enum MatArchiveList {
        MatArchive_FILESTORAGE,
        MatArchive_PLAINTEXT,
        MatArchive_BINARY
    };

    /// writes matrix data locally using a binary/yml/text file format
    void write(const std::string& sFilePath, const cv::Mat& _oData, MatArchiveList eArchiveType=MatArchive_BINARY);
    /// reads matrix data locally using a binary/yml/text file format
    void read(const std::string& sFilePath, cv::Mat& oData, MatArchiveList eArchiveType=MatArchive_BINARY);
    /// reads matrix data locally using a binary/yml/text file format (inline version)
    inline cv::Mat read(const std::string& sFilePath, MatArchiveList eArchiveType=MatArchive_BINARY) {
        cv::Mat oData;
        cv::read(sFilePath,oData,eArchiveType);
        return oData;
    }

    /// shifts the values in a matrix by an (x,y) offset (see definition for full info)
    void shift(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Point2f& vDelta, int nFillType=cv::BORDER_CONSTANT, const cv::Scalar& vConstantFillValue=cv::Scalar(0,0,0,0));

    /// returns an always-empty-mat by reference
    inline const cv::Mat& emptyMat() {
        static const cv::Mat s_oEmptyMat = cv::Mat();
        return s_oEmptyMat;
    }
    /// returns an always-empty-size by reference
    inline const cv::Size& emptySize() {
        static const cv::Size s_oEmptySize = cv::Size();
        return s_oEmptySize;
    }
    /// returns an always-empty-mat by reference
    inline const std::vector<cv::Mat>& emptyMatArray() {
        static const std::vector<cv::Mat> s_vEmptyMatArray = std::vector<cv::Mat>();
        return s_vEmptyMatArray;
    }
    /// returns an always-empty-size by reference
    inline const std::vector<cv::Size>& emptySizeArray() {
        static const std::vector<cv::Size> s_vEmptySizeArray = std::vector<cv::Size>();
        return s_vEmptySizeArray;
    }

    /// defines an aligned memory allocator to be used in matrices
    template<size_t nByteAlign, bool bAlignSingleElem=false>
    class AlignedMatAllocator : public cv::MatAllocator {
        static_assert(nByteAlign>0,"byte alignment must be a non-null value");
    public:
        typedef AlignedMatAllocator<nByteAlign,bAlignSingleElem> this_type;
        inline AlignedMatAllocator() noexcept {}
        inline AlignedMatAllocator(const AlignedMatAllocator<nByteAlign,bAlignSingleElem>&) noexcept {}
        template<typename T2>
        inline this_type& operator=(const AlignedMatAllocator<nByteAlign,bAlignSingleElem>&) noexcept {}
        virtual ~AlignedMatAllocator() noexcept {}
        virtual cv::UMatData* allocate(int dims, const int* sizes, int type, void* data, size_t* step, int /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const override {
            step[dims-1] = bAlignSingleElem?cv::alignSize(CV_ELEM_SIZE(type),nByteAlign):CV_ELEM_SIZE(type);
            for(int d=dims-2; d>=0; --d)
                step[d] = cv::alignSize(step[d+1]*sizes[d+1],nByteAlign);
            const size_t nTotBytes = (size_t)cv::alignSize(step[0]*size_t(sizes[0]),(int)nByteAlign);
            cv::UMatData* u = new cv::UMatData(this);
            u->size = nTotBytes;
            if(data) {
                u->data = u->origdata = static_cast<uchar*>(data);
                u->flags |= cv::UMatData::USER_ALLOCATED;
            }
            else
                u->data = u->origdata = lv::AlignedMemAllocator<uchar,nByteAlign>::allocate(nTotBytes);
            return u;
        }
        virtual bool allocate(cv::UMatData* data, int /*accessFlags*/, cv::UMatUsageFlags /*usageFlags*/) const override {
            return (data!=nullptr);
        }
        virtual void deallocate(cv::UMatData* data) const override {
            if(data==nullptr)
                return;
            lvDbgAssert(data->urefcount>=0 && data->refcount>=0);
            if(data->refcount==0) {
                if(!(data->flags & cv::UMatData::USER_ALLOCATED)) {
                    lv::AlignedMemAllocator<uchar,nByteAlign>::deallocate(data->origdata,data->size);
                    data->origdata = nullptr;
                }
                delete data;
            }
        }
    };

    /// temp function; msvc seems to disable cuda output unless it is passed as argument to an external-lib function call...?
    void doNotOptimize(const cv::Mat& m);

    /// returns a 16-byte aligned matrix allocator for SSE(1/2/3/4.1/4.2) support (should never be modified, despite non-const!)
    cv::MatAllocator* getMatAllocator16a();
    /// returns a 16-byte aligned matrix allocator for AVX(1/2) support (should never be modified, despite non-const!)
    cv::MatAllocator* getMatAllocator32a();

} // namespace cv
