
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

#include "litiv/datasets.hpp"
#include "litiv/video.hpp"
#include "litiv/3rdparty/dshowbase/ocvcompat.h"
#include <tuple>
#include <Kinect.h>

/////////////////////////////////
#define USE_FLIR_SENSOR         1
/////////////////////////////////
#define DISPLAY_OUTPUT          2
#define WRITE_OUTPUT            0
/////////////////////////////////
#define DEFAULT_QUEUE_BUFFER_SIZE   1024*1024*50  // max = 50MB per queue (default)
#define HIGHDEF_QUEUE_BUFFER_SIZE   1024*1024*1024 // max = 1GB per queue (high-defition stream)
#define VIDEO_FILE_PREALLOC_SIZE    1024*1024*1024 // tot = 1GB per video file
#define STRUCT_FILE_PREALLOC_SIZE   1024*1024*20 // tot = 200MB per struct file
/////////////////////////////////
//// cv::VideoWriter params /////          filename           fourcc  framerate       frame size      is color
#define FLIR_OUTPUT_VIDEO_PARAMS    "c:/temp/test_flir.avi",    -1,     30.0,     cv::Size(320,240),   false
#define BODYIDX_OUTPUT_VIDEO_PARAMS "c:/temp/test_bodyidx.avi", -1,     30.0,     cv::Size(512,424),   false
#define NIR_OUTPUT_VIDEO_PARAMS     "c:/temp/test_nir.avi",     -1,     30.0,     cv::Size(512,424),   false
#define DEPTH_OUTPUT_VIDEO_PARAMS   "c:/temp/test_depth.avi",   -1,     30.0,     cv::Size(512,424),   false
#define COLOR_OUTPUT_VIDEO_PARAMS   "e:/temp/test_color.avi",   -1,     30.0,     cv::Size(1920,1080), true
/////////////////////////////////
#define BODY_DATA_OUTPUT_PARAMS     "c:/temp/test_body.bin",     std::ios::out|std::ios::binary
#define META_DATA_OUTPUT_PARAMS     "c:/temp/test_metadata.yml", cv::FileStorage::WRITE
/////////////////////////////////

std::atomic_bool g_bIsActive = true;

int main() {
    try {
        lv::RegisterAllConsoleSignals([](int){g_bIsActive = false;});
        const auto lEncodeAndSaveFrame = [](const cv::Mat& oImage, size_t nIndex, cv::VideoWriter& oWriter, size_t& nLastSavedIndex) {
            lvAssert(!oImage.empty() && oWriter.isOpened());
            lvAssert(nLastSavedIndex==SIZE_MAX || nLastSavedIndex<nIndex);
            oWriter.write(oImage);
            nLastSavedIndex = nIndex;
            return (size_t)0;
        };
        const auto lEncodeAndSaveBodyFrame = [](const cv::Mat& oBodyFrame, size_t nIndex, std::ofstream* pWriter, size_t& nLastSavedIndex){
            lvAssert(!oBodyFrame.empty() && oBodyFrame.total()*oBodyFrame.elemSize()==sizeof(lv::KinectBodyFrame));
            lvAssert(pWriter && pWriter->is_open());
            lvAssert(nLastSavedIndex==SIZE_MAX || nLastSavedIndex<nIndex);
            ((lv::KinectBodyFrame*)oBodyFrame.data)->nFrameIdx = nIndex;
            pWriter->write((char*)oBodyFrame.data,sizeof(lv::KinectBodyFrame));
            nLastSavedIndex = nIndex;
            return (size_t)0;
        };
#if USE_FLIR_SENSOR
        std::cout << "Setting up FLIR device..." << std::endl;
        lvAssertHR(CoInitializeEx(0,COINIT_MULTITHREADED|COINIT_DISABLE_OLE1DDE));
        std::unique_ptr<lv::DShowCameraGrabber> pFLIRSensor = std::make_unique<lv::DShowCameraGrabber>("FLIR ThermaCAM");
        lvAssertHR(pFLIRSensor->Connect());
        cv::Mat oFLIRFrame;
        const cv::Size oFLIRFrameSize = std::get<3>(std::make_tuple(FLIR_OUTPUT_VIDEO_PARAMS));
#if WRITE_OUTPUT
        std::cout << "Setting up FLIR video writer..." << std::endl;
        size_t nLastSavedFLIRFrameIdx = SIZE_MAX;
        cv::VideoWriter oFLIRVideoWriter(FLIR_OUTPUT_VIDEO_PARAMS);
        lvAssert(oFLIRVideoWriter.isOpened());
        lv::DataWriter oFLIRVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oFLIRVideoWriter,nLastSavedFLIRFrameIdx));
        lvAssert(oFLIRVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
#endif //WRITE_OUTPUT
#endif //USE_FLIR_SENSOR
        std::cout << "Setting up Kinect device..." << std::endl;
        CComPtr<IKinectSensor> pKinectSensor;
        lvAssertHR(GetDefaultKinectSensor(&pKinectSensor));
        lvAssert(pKinectSensor);
        lvAssertHR(pKinectSensor->Open());
        CComPtr<IMultiSourceFrameReader> pMultiFrameReader;
        lvAssertHR(pKinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color|FrameSourceTypes_Depth|FrameSourceTypes_Infrared|FrameSourceTypes_Body|FrameSourceTypes_BodyIndex,&pMultiFrameReader));
        constexpr size_t nStreamCount = 5;
        lv::KinectBodyFrame oBodyFrame;
        const cv::Mat oBodyFrameWrapper(1,(int)sizeof(lv::KinectBodyFrame),CV_8UC1,&oBodyFrame);
        cv::Mat oBodyIdxFrame,oNIRFrame,oDepthFrame,oColorFrame;
        const cv::Size oBodyIdxFrameSize = std::get<3>(std::make_tuple(BODYIDX_OUTPUT_VIDEO_PARAMS));
        const cv::Size oNIRFrameSize = std::get<3>(std::make_tuple(NIR_OUTPUT_VIDEO_PARAMS));
        const cv::Size oDepthFrameSize = std::get<3>(std::make_tuple(DEPTH_OUTPUT_VIDEO_PARAMS));
        const cv::Size oColorFrameSize = std::get<3>(std::make_tuple(COLOR_OUTPUT_VIDEO_PARAMS));
#if WRITE_OUTPUT
        std::cout << "Setting up Kinect body data writer..." << std::endl;
        size_t nLastSavedBodyFrameIdx = SIZE_MAX;
        std::ofstream oBodyStructWriter(BODY_DATA_OUTPUT_PARAMS);
        lvAssert(oBodyStructWriter && oBodyStructWriter.is_open());
        lv::DataWriter oBodyStructAsyncWriter(std::bind(lEncodeAndSaveBodyFrame,std::placeholders::_1,std::placeholders::_2,&oBodyStructWriter,nLastSavedBodyFrameIdx));
        lvAssert(oBodyStructAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up BODYINDEX video writer..." << std::endl;
        size_t nLastSavedBodyIdxFrameIdx = SIZE_MAX;
        cv::VideoWriter oBodyIdxVideoWriter(BODYIDX_OUTPUT_VIDEO_PARAMS);
        lvAssert(oBodyIdxVideoWriter.isOpened());
        lv::DataWriter oBodyIdxVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oBodyIdxVideoWriter,nLastSavedBodyIdxFrameIdx));
        lvAssert(oBodyIdxVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up NIR video writer..." << std::endl;
        size_t nLastSavedNIRFrameIdx = SIZE_MAX;
        cv::VideoWriter oNIRVideoWriter(NIR_OUTPUT_VIDEO_PARAMS);
        lvAssert(oNIRVideoWriter.isOpened());
        lv::DataWriter oNIRVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oNIRVideoWriter,nLastSavedNIRFrameIdx));
        lvAssert(oNIRVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up DEPTH video writer..." << std::endl;
        size_t nLastSavedDepthFrameIdx = SIZE_MAX;
        cv::VideoWriter oDepthVideoWriter(DEPTH_OUTPUT_VIDEO_PARAMS);
        lvAssert(oDepthVideoWriter.isOpened());
        lv::DataWriter oDepthVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oDepthVideoWriter,nLastSavedDepthFrameIdx));
        lvAssert(oDepthVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up COLOR video writer..." << std::endl;
        size_t nLastSavedColorFrameIdx = SIZE_MAX;
        lvAssert(lv::CreateBinFileWithPrealloc(std::get<0>(std::make_tuple(COLOR_OUTPUT_VIDEO_PARAMS)),VIDEO_FILE_PREALLOC_SIZE));
        cv::VideoWriter oColorVideoWriter(COLOR_OUTPUT_VIDEO_PARAMS);
        lvAssert(oColorVideoWriter.isOpened());
        lv::DataWriter oColorVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oColorVideoWriter,nLastSavedColorFrameIdx));
        lvAssert(oColorVideoAsyncWriter.startAsyncWriting(HIGHDEF_QUEUE_BUFFER_SIZE,true));
#if WRITE_OUTPUT>1
        std::mutex oMetadataStorageMutex;
        cv::FileStorage oMetadataStorage(META_DATA_OUTPUT_PARAMS);
        oMetadataStorage << "htag" << lv::getVersionStamp();
        oMetadataStorage << "date" << lv::getTimeStamp();
#endif //WRITE_OUTPUT>1
#endif //WRITE_OUTPUT
#if DISPLAY_OUTPUT
#if DISPLAY_OUTPUT>1
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create("DISPLAY","c:/temp/",cv::Size(1920,1200),cv::WINDOW_NORMAL);
        CComPtr<ICoordinateMapper> pCoordMapper;
        lvAssertHR(pKinectSensor->get_CoordinateMapper(&pCoordMapper));
        std::map<UINT64,cv::Scalar_<uchar>> mBodyColors;
#else //!(DISPLAY_OUTPUT>1)
#if USE_FLIR_SENSOR
        cv::namedWindow("oFLIRFrame",cv::WINDOW_NORMAL);
#endif //USE_FLIR_SENSOR
        cv::namedWindow("oBodyIdxFrame",cv::WINDOW_NORMAL);
        cv::namedWindow("oNIRFrame",cv::WINDOW_NORMAL);
        cv::namedWindow("oDepthFrame",cv::WINDOW_NORMAL);
#endif //!(DISPLAY_OUTPUT>1)
        cv::namedWindow("oColorFrame",cv::WINDOW_NORMAL);
#endif //DISPLAY_OUTPUT
        CComPtr<IMultiSourceFrame> pMultiFrame;
        lv::WorkerPool<nStreamCount> oPool;
        std::array<std::future<bool>,nStreamCount> abGrabResults;
        const std::array<std::function<bool()>,nStreamCount> alGrabTasks = {
            [&]{
                CComPtr<IBodyFrameReference> pFrameRef;
                lvAssertHR(pMultiFrame->get_BodyFrameReference(&pFrameRef));
                CComPtr<IBodyFrame> pFrame;
                IBody* apBodies[BODY_COUNT] = {};
                const bool bGotFrame = SUCCEEDED(pFrameRef->AcquireFrame(&pFrame)) && SUCCEEDED(pFrame->GetAndRefreshBodyData(BODY_COUNT,apBodies));
                oBodyFrame.bIsValid = bGotFrame;
                if(bGotFrame) {
                    pFrameRef.Release();
                    for(size_t nBodyIdx=0; nBodyIdx<BODY_COUNT; ++nBodyIdx) {
                        lvAssertHR(apBodies[nBodyIdx]->get_IsRestricted(&oBodyFrame.aBodyData[nBodyIdx].bIsRestricted));
                        lvAssertHR(apBodies[nBodyIdx]->get_IsTracked(&oBodyFrame.aBodyData[nBodyIdx].bIsTracked));
                        if(oBodyFrame.aBodyData[nBodyIdx].bIsTracked) {
                            lvAssertHR(apBodies[nBodyIdx]->GetJoints((UINT)oBodyFrame.aBodyData[nBodyIdx].aJointData.size(),oBodyFrame.aBodyData[nBodyIdx].aJointData.data()));
                            lvAssertHR(apBodies[nBodyIdx]->GetJointOrientations((UINT)oBodyFrame.aBodyData[nBodyIdx].aJointOrientationData.size(),oBodyFrame.aBodyData[nBodyIdx].aJointOrientationData.data()));
                            lvAssertHR(apBodies[nBodyIdx]->get_HandLeftConfidence(&oBodyFrame.aBodyData[nBodyIdx].eLeftHandStateConfidence));
                            lvAssertHR(apBodies[nBodyIdx]->get_HandRightConfidence(&oBodyFrame.aBodyData[nBodyIdx].eRightHandStateConfidence));
                            lvAssertHR(apBodies[nBodyIdx]->get_HandLeftState(&oBodyFrame.aBodyData[nBodyIdx].eLeftHandState));
                            lvAssertHR(apBodies[nBodyIdx]->get_HandRightState(&oBodyFrame.aBodyData[nBodyIdx].eRightHandState));
                            lvAssertHR(apBodies[nBodyIdx]->get_ClippedEdges(&oBodyFrame.aBodyData[nBodyIdx].nClippedEdges));
                            lvAssertHR(apBodies[nBodyIdx]->get_TrackingId(&oBodyFrame.aBodyData[nBodyIdx].nTrackingID));
                            lvAssertHR(apBodies[nBodyIdx]->get_LeanTrackingState(&oBodyFrame.aBodyData[nBodyIdx].eLeanTrackState));
                            lvAssertHR(apBodies[nBodyIdx]->get_Lean(&oBodyFrame.aBodyData[nBodyIdx].vLean));
                        }
                        SafeRelease(&apBodies[nBodyIdx]);
                    }
                    lvAssertHR(pFrame->get_FloorClipPlane(&oBodyFrame.vFloorClipPlane));
                    lvAssertHR(pFrame->get_RelativeTime(&oBodyFrame.nTimeStamp));
                }
                return bGotFrame;
            },
            [&]{
                CComPtr<IBodyIndexFrameReference> pFrameRef;
                lvAssertHR(pMultiFrame->get_BodyIndexFrameReference(&pFrameRef));
                CComPtr<IBodyIndexFrame> pFrame;
                const bool bGotFrame = SUCCEEDED(pFrameRef->AcquireFrame(&pFrame));
                if(bGotFrame) {
                    pFrameRef.Release();
                    if(oBodyIdxFrame.empty()) { // must enter only once
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        lvAssert(cv::Size(nFrameWidth,nFrameHeight)==oBodyIdxFrameSize);
                        uint nFrameBytesPerPixel;
                        lvAssertHR(pFrameDesc->get_BytesPerPixel(&nFrameBytesPerPixel));
                        lvAssert_(nFrameBytesPerPixel==1,"bodyidx frame not one byte per pixel");
#if WRITE_OUTPUT>1
                        float fHorizFOV,fVertiFOV;
                        lvAssertHR(pFrameDesc->get_HorizontalFieldOfView(&fHorizFOV));
                        lvAssertHR(pFrameDesc->get_VerticalFieldOfView(&fVertiFOV));
                        {
                            std::lock_guard<std::mutex> oMetadataStorageLock(oMetadataStorageMutex);
                            oMetadataStorage << "bodyidx_metadata" << "{";
                            oMetadataStorage << "orig_size" << oBodyIdxFrameSize;
                            oMetadataStorage << "cv_type" << (int)CV_8UC1;
                            oMetadataStorage << "hfov" << fHorizFOV;
                            oMetadataStorage << "vfov" << fVertiFOV;
                            oMetadataStorage << "}";
                        }
#endif //WRITE_OUTPUT>1
                        oBodyIdxFrame.create(oBodyIdxFrameSize,CV_8UC1);
                    }
                    lvAssertHR(pFrame->CopyFrameDataToArray((UINT)oBodyIdxFrame.total(),oBodyIdxFrame.data));
                }
                return bGotFrame;
            },
            [&]{
                CComPtr<IInfraredFrameReference> pFrameRef;
                lvAssertHR(pMultiFrame->get_InfraredFrameReference(&pFrameRef));
                CComPtr<IInfraredFrame> pFrame;
                const bool bGotFrame = SUCCEEDED(pFrameRef->AcquireFrame(&pFrame));
                if(bGotFrame) {
                    pFrameRef.Release();
                    if(oNIRFrame.empty()) { // must enter only once
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        lvAssert(cv::Size(nFrameWidth,nFrameHeight)==oNIRFrameSize);
                        uint nFrameBytesPerPixel;
                        lvAssertHR(pFrameDesc->get_BytesPerPixel(&nFrameBytesPerPixel));
                        lvAssert_(nFrameBytesPerPixel==2,"nir frame not two bytes per pixel");
#if WRITE_OUTPUT>1
                        float fHorizFOV,fVertiFOV;
                        lvAssertHR(pFrameDesc->get_HorizontalFieldOfView(&fHorizFOV));
                        lvAssertHR(pFrameDesc->get_VerticalFieldOfView(&fVertiFOV));
                        {
                            std::lock_guard<std::mutex> oMetadataStorageLock(oMetadataStorageMutex);
                            oMetadataStorage << "nir_metadata" << "{";
                            oMetadataStorage << "orig_size" << oNIRFrameSize;
                            oMetadataStorage << "cv_type" << (int)CV_16UC1;
                            oMetadataStorage << "hfov" << fHorizFOV;
                            oMetadataStorage << "vfov" << fVertiFOV;
                            oMetadataStorage << "}";
                        }
#endif //WRITE_OUTPUT>1
                        oNIRFrame.create(oNIRFrameSize,CV_16UC1);
                    }
                    lvAssertHR(pFrame->CopyFrameDataToArray((UINT)oNIRFrame.total(),(uint16_t*)oNIRFrame.data));
                }
                return bGotFrame;
            },
            [&]{
                CComPtr<IDepthFrameReference> pFrameRef;
                lvAssertHR(pMultiFrame->get_DepthFrameReference(&pFrameRef));
                CComPtr<IDepthFrame> pFrame;
                const bool bGotFrame = SUCCEEDED(pFrameRef->AcquireFrame(&pFrame));
                if(bGotFrame) {
                    pFrameRef.Release();
                    if(oDepthFrame.empty()) { // must enter only once
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        lvAssert(cv::Size(nFrameWidth,nFrameHeight)==oDepthFrameSize);
                        uint nFrameBytesPerPixel;
                        lvAssertHR(pFrameDesc->get_BytesPerPixel(&nFrameBytesPerPixel));
                        lvAssert_(nFrameBytesPerPixel==2,"depth frame not two bytes per pixel");
#if WRITE_OUTPUT>1
                        float fHorizFOV,fVertiFOV;
                        lvAssertHR(pFrameDesc->get_HorizontalFieldOfView(&fHorizFOV));
                        lvAssertHR(pFrameDesc->get_VerticalFieldOfView(&fVertiFOV));
                        USHORT nMaxReliableDist,nMinReliableDist;
                        lvAssertHR(pFrame->get_DepthMaxReliableDistance(&nMaxReliableDist));
                        lvAssertHR(pFrame->get_DepthMinReliableDistance(&nMinReliableDist));
                        {
                            std::lock_guard<std::mutex> oMetadataStorageLock(oMetadataStorageMutex);
                            oMetadataStorage << "depth_metadata" << "{";
                            oMetadataStorage << "orig_size" << oDepthFrameSize;
                            oMetadataStorage << "cv_type" << (int)CV_16UC1;
                            oMetadataStorage << "hfov" << fHorizFOV;
                            oMetadataStorage << "vfov" << fVertiFOV;
                            oMetadataStorage << "max_reliable_dist" << nMaxReliableDist;
                            oMetadataStorage << "min_reliable_dist" << nMinReliableDist;
                            oMetadataStorage << "}";
                        }
#endif //WRITE_OUTPUT>1
                        oDepthFrame.create(oDepthFrameSize,CV_16UC1);
                    }
                    lvAssertHR(pFrame->CopyFrameDataToArray((UINT)oDepthFrame.total(),(uint16_t*)oDepthFrame.data));
                }
                return bGotFrame;
            },
            [&]{
                CComPtr<IColorFrameReference> pFrameRef;
                lvAssertHR(pMultiFrame->get_ColorFrameReference(&pFrameRef));
                CComPtr<IColorFrame> pFrame;
                const bool bGotFrame = SUCCEEDED(pFrameRef->AcquireFrame(&pFrame));
                if(bGotFrame) {
                    pFrameRef.Release();
                    if(oColorFrame.empty()) { // must enter only once
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        lvAssert(cv::Size(nFrameWidth,nFrameHeight)==oColorFrameSize);
                        uint nFrameBytesPerPixel;
                        lvAssertHR(pFrameDesc->get_BytesPerPixel(&nFrameBytesPerPixel));
                        lvAssert_(nFrameBytesPerPixel==2,"raw color frame not two bytes per pixel");
#if WRITE_OUTPUT>1
                        float fHorizFOV,fVertiFOV;
                        lvAssertHR(pFrameDesc->get_HorizontalFieldOfView(&fHorizFOV));
                        lvAssertHR(pFrameDesc->get_VerticalFieldOfView(&fVertiFOV));
                        ColorImageFormat eColorFormat;
                        lvAssertHR(pFrame->get_RawColorImageFormat(&eColorFormat));
                        lvAssert(eColorFormat==ColorImageFormat_Yuy2);
                        CComPtr<IColorCameraSettings> pColorCameraSettings;
                        lvAssertHR(pFrame->get_ColorCameraSettings(&pColorCameraSettings));
                        TIMESPAN nExposureTime,nFrameInterval;
                        lvAssertHR(pColorCameraSettings->get_ExposureTime(&nExposureTime));
                        lvAssertHR(pColorCameraSettings->get_FrameInterval(&nFrameInterval));
                        float fGain,fGamma;
                        lvAssertHR(pColorCameraSettings->get_Gain(&fGain));
                        lvAssertHR(pColorCameraSettings->get_Gamma(&fGamma));
                        {
                            std::lock_guard<std::mutex> oMetadataStorageLock(oMetadataStorageMutex);
                            oMetadataStorage << "color_metadata" << "{";
                            oMetadataStorage << "orig_size" << oColorFrameSize;
                            oMetadataStorage << "cv_type" << (int)CV_8UC3;
                            oMetadataStorage << "hfov" << fHorizFOV;
                            oMetadataStorage << "vfov" << fVertiFOV;
                            oMetadataStorage << "init_exposure" << (double)nExposureTime;
                            oMetadataStorage << "init_frameinterv" << (double)nFrameInterval;
                            oMetadataStorage << "init_gain" << (double)fGain;
                            oMetadataStorage << "init_gamma" << (double)fGamma;
                            oMetadataStorage << "}";
                        }
#endif //WRITE_OUTPUT>1
                        oColorFrame.create(oColorFrameSize,CV_8UC3);
                    }
                    UINT nBufferSize;
                    BYTE* pBuffer;
                    lvAssertHR(pFrame->AccessRawUnderlyingBuffer(&nBufferSize,&pBuffer));
                    const cv::Mat oRawColorFrame(oColorFrame.size(),CV_8UC2,pBuffer);
                    lvAssert(nBufferSize<=oRawColorFrame.total()*oRawColorFrame.elemSize());
                    cv::cvtColor(oRawColorFrame,oColorFrame,cv::COLOR_YUV2BGR_YUY2);
                }
                return bGotFrame;
            },
        };
        size_t nRealFrameIdx = 0;
        size_t nTempFrameIdx = 0;
        size_t nGoodFrameIdx = 0;
        int64_t nLastTick = 0;
        while(g_bIsActive) {
            pMultiFrame.Release();
            while(!SUCCEEDED(pMultiFrameReader->AcquireLatestFrame(&pMultiFrame)));
            for(size_t n=0; n<alGrabTasks.size(); ++n)
                abGrabResults[n] = oPool.queueTask(alGrabTasks[n]);
            bool bFinalGrabResult = true;
#if USE_FLIR_SENSOR
            bFinalGrabResult &= (pFLIRSensor->GetLatestFrame(oFLIRFrame,true)>=0);
#endif //USE_FLIR_SENSOR
            for(size_t n=0; n<abGrabResults.size(); ++n)
                bFinalGrabResult &= abGrabResults[n].get();
            if(bFinalGrabResult) {
#if USE_FLIR_SENSOR
                static std::once_flag s_oFLIRMetadataWriteFlag;
                std::call_once(s_oFLIRMetadataWriteFlag,[&]() {
                    lvAssert(oFLIRFrame.size()==oFLIRFrameSize);
#if WRITE_OUTPUT>1
                    {
                        std::lock_guard<std::mutex> oMetadataStorageLock(oMetadataStorageMutex);
                        oMetadataStorage << "flir_metadata" << "{";
                        oMetadataStorage << "orig_size" << oFLIRFrameSize;
                        oMetadataStorage << "cv_type" << oFLIRFrame.type();
                        oMetadataStorage << "}";
                    }
#endif //WRITE_OUTPUT>1
                });
#endif //USE_FLIR_SENSOR
#if DISPLAY_OUTPUT
#if DISPLAY_OUTPUT>1
                cv::Mat oBodyIdxFrameTemp = oBodyIdxFrame<UCHAR_MAX;
                std::vector<std::vector<cv::Point>> vvBodyIdxContours;
                cv::findContours(oBodyIdxFrameTemp,vvBodyIdxContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvImageNamePairs(2);
                const cv::Size oSuggestedTileSize = oDepthFrame.size();
                cv::Mat oNIRFrameRaw,oNIRFrameDisplay;
                oNIRFrame.convertTo(oNIRFrameRaw,CV_32F,1.0/USHRT_MAX);
                cv::Scalar vNIRFrameMean,vNIRFrameStdDev;
                cv::meanStdDev(oNIRFrameRaw,vNIRFrameMean,vNIRFrameStdDev);
                oNIRFrameRaw = cv::max(cv::min(oNIRFrameRaw/(vNIRFrameMean[0]*5),0.9f),0.01f);
                cv::applyColorMap(oNIRFrameRaw,oNIRFrameDisplay,cv::COLORMAP_COOL);
                cv::drawContours(oNIRFrameDisplay,vvBodyIdxContours,-1,cv::Scalar(USHRT_MAX),3);
                cv::flip(oNIRFrameDisplay,oNIRFrameDisplay,1);
                vvImageNamePairs[0].emplace_back(oNIRFrameDisplay,"NIR");
                cv::Mat oDepthFrameRaw,oDepthFrameDisplay;
                cv::normalize(oDepthFrame,oDepthFrameRaw,1.0,0.0,cv::NORM_MINMAX,CV_32FC1);
                cv::applyColorMap(oDepthFrameRaw,oDepthFrameDisplay,cv::COLORMAP_BONE);
                cv::drawContours(oDepthFrameDisplay,vvBodyIdxContours,-1,cv::Scalar(USHRT_MAX),3);
                cv::flip(oDepthFrameDisplay,oDepthFrameDisplay,1);
                vvImageNamePairs[0].emplace_back(oDepthFrameDisplay,"Depth");
#if USE_FLIR_SENSOR
                cv::Mat oFLIRFrameDisplay;
                cv::applyColorMap(oFLIRFrame,oFLIRFrameDisplay,cv::COLORMAP_HOT);
                vvImageNamePairs[1].emplace_back(oFLIRFrameDisplay,"Thermal");
#else //!(USE_FLIR_SENSOR)
                vvImageNamePairs[1].emplace_back(cv::Mat(oDepthFrameDisplay.size(),CV_8UC3,cv::Scalar_<uchar>::all(128)),"Thermal");
#endif //!(USE_FLIR_SENSOR)
                cv::Mat oBodyFrameDisplay(oDepthFrameDisplay.size(),CV_8UC3,cv::Scalar_<uchar>::all(88));
                const auto lBodyToScreen = [&](const CameraSpacePoint& oInputPt) {
                    DepthSpacePoint depthPoint = {0};
                    pCoordMapper->MapCameraPointToDepthSpace(oInputPt, &depthPoint);
                    return cv::Point2i((int)depthPoint.X,(int)depthPoint.Y);
                };
                const auto lDrawBone = [&](const Joint& i, const Joint& j, const cv::Scalar_<uchar>& vBodyColor) {
                    if(i.TrackingState<=TrackingState_Inferred && j.TrackingState<=TrackingState_Inferred)
                        return;
                    const int nThickness = (i.TrackingState==TrackingState_Inferred || j.TrackingState==TrackingState_Inferred)?3:6;
                    cv::line(oBodyFrameDisplay,lBodyToScreen(i.Position),lBodyToScreen(j.Position),vBodyColor,nThickness);
                };
                for(size_t nBodyIdx=0; nBodyIdx<BODY_COUNT; ++nBodyIdx) {
                    if(oBodyFrame.aBodyData[nBodyIdx].bIsTracked) {
                        if(mBodyColors.count(oBodyFrame.aBodyData[nBodyIdx].nTrackingID)==0)
                            mBodyColors.insert(std::make_pair(oBodyFrame.aBodyData[nBodyIdx].nTrackingID,cv::Scalar_<uchar>(55+rand()%200,55+rand()%200,55+rand()%200)));
                        const cv::Scalar_<uchar>& vBodyColor = mBodyColors[oBodyFrame.aBodyData[nBodyIdx].nTrackingID];
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_Head],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_Neck],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_Neck],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_SpineShoulder],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_SpineShoulder],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_SpineMid],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_SpineMid],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_SpineBase],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_SpineShoulder],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ShoulderRight],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_SpineShoulder],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ShoulderLeft],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ShoulderRight],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ElbowRight],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ElbowRight],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_WristRight],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_WristRight],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_HandRight],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_HandRight],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_HandTipRight],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_WristRight],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ThumbRight],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ShoulderLeft],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ElbowLeft],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ElbowLeft],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_WristLeft],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_WristLeft],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_HandLeft],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_HandLeft],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_HandTipLeft],vBodyColor);
                        lDrawBone(oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_WristLeft],oBodyFrame.aBodyData[nBodyIdx].aJointData[JointType_ThumbLeft],vBodyColor);
                        for(size_t nJointIdx=0; nJointIdx<JointType::JointType_Count; ++nJointIdx) {
                            if(oBodyFrame.aBodyData[nBodyIdx].aJointData[nJointIdx].TrackingState>TrackingState_NotTracked && (nJointIdx<(size_t)JointType_HipLeft || nJointIdx>(size_t)JointType_FootRight)) {
                                const int nThickness = (oBodyFrame.aBodyData[nBodyIdx].aJointData[nJointIdx].TrackingState==TrackingState_Inferred)?2:4;
                                cv::circle(oBodyFrameDisplay,lBodyToScreen(oBodyFrame.aBodyData[nBodyIdx].aJointData[nJointIdx].Position),nThickness,cv::Scalar_<uchar>::all(176),-1);
                            }
                        }
                    }
                }
                cv::flip(oBodyFrameDisplay,oBodyFrameDisplay,1);
                vvImageNamePairs[1].emplace_back(oBodyFrameDisplay,"Skeleton");
                pDisplayHelper->display(vvImageNamePairs,oSuggestedTileSize);
                cv::imshow("oColorFrame",oColorFrame);
#else //DISPLAY_OUTPUT<=1
#if USE_FLIR_SENSOR
                cv::imshow("oFLIRFrame",oFLIRFrame);
#endif //USE_FLIR_SENSOR
                cv::imshow("oBodyIdxFrame",oBodyIdxFrame);
                cv::imshow("oColorFrame",oColorFrame);
                cv::imshow("oNIRFrame",oNIRFrame);
                cv::imshow("oDepthFrame",oDepthFrame);
#endif //DISPLAY_OUTPUT<=1
                const char c = (char)cv::waitKey(1);
                if(c=='q' || c==27)
                    break;
#endif //DISPLAY_OUTPUT
#if WRITE_OUTPUT
                if(oColorVideoAsyncWriter.queue(oColorFrame,nRealFrameIdx)!=SIZE_MAX) {
                    // color queue is the only one might really overflow, so other queues depend only on that one
#if USE_FLIR_SENSOR
                    lvAssert(oFLIRVideoAsyncWriter.queue(oFLIRFrame,nRealFrameIdx)!=SIZE_MAX);
#endif //USE_FLIR_SENSOR
                    lvAssert(oBodyStructAsyncWriter.queue(oBodyFrameWrapper,nRealFrameIdx)!=SIZE_MAX);
                    lvAssert(oBodyIdxVideoAsyncWriter.queue(oBodyIdxFrame,nRealFrameIdx)!=SIZE_MAX);
                    lvAssert(oNIRVideoAsyncWriter.queue(oNIRFrame,nRealFrameIdx)!=SIZE_MAX);
                    lvAssert(oDepthVideoAsyncWriter.queue(oDepthFrame,nRealFrameIdx)!=SIZE_MAX);
                    ++nGoodFrameIdx;
                }
                else
                    std::cout << "The color stream is dropping frames!" << std::endl;
#endif //WRITE_OUTPUT
            }
            ++nTempFrameIdx;
            ++nRealFrameIdx;
            if(nLastTick==0)
                nLastTick = cv::getTickCount();
            else {
                const int64_t nCurrTick = cv::getTickCount();
                const int64_t nCurrTickDiff = nCurrTick-nLastTick;
                if(nCurrTickDiff>cv::getTickFrequency()*3) {
                    std::cout << "Main @ " << double(nTempFrameIdx)/(nCurrTickDiff/cv::getTickFrequency()) << " FPS   (saved = " << double(nGoodFrameIdx)/(nCurrTickDiff/cv::getTickFrequency()) << " FPS)" << std::endl;
                    nLastTick = nCurrTick;
                    nTempFrameIdx = 0;
                    nGoodFrameIdx = 0;
                }
            }
        }
        std::cout << "\n=================================\nGot interrupt, emptying queues...\n=================================\n" << std::endl;
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    CoUninitialize();
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}
