
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

namespace cv { // extending cv

    struct DisplayHelper;
    using DisplayHelperPtr = std::shared_ptr<DisplayHelper>;

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
