
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
#include "litiv/utils/kinect.hpp"
#include <fstream>

/////////////////////////////////
#define USE_LWIR_SENSOR         1
#define USE_NIR_SENSOR          0
#define USE_C2D_MAPPING         0
/////////////////////////////////
#define DISPLAY_OUTPUT          2
#define WRITE_OUTPUT            0
#define WRITE_OUTPUT_PREFIX     "test"
/////////////////////////////////
// note: values below are useless if larger than cmake's DATASETS_CACHE_SIZE
#define DEFAULT_QUEUE_BUFFER_SIZE   1024*1024*50  // max = 50MB per queue (default)
#define HIGHDEF_QUEUE_BUFFER_SIZE   1024*1024*512 // max = 512MB per queue (high-defition stream)
//#define VIDEO_FILE_PREALLOC_SIZE    1024*1024*1024 // tot = 1GB per video file
//#define STRUCT_FILE_PREALLOC_SIZE   1024*1024*20 // tot = 200MB per struct file
/////////////////////////////////
//// cv::VideoWriter params /////                      filename                  fourcc  framerate       frame size       is color
#define LWIR_OUTPUT_VIDEO_PARAMS    "c:/temp/" WRITE_OUTPUT_PREFIX "/lwir.avi",    -1,     30.0,     cv::Size(320,240),    false    // ffdshow mpeg4/xvid w/ 'grayscale' preset
#define NIR_OUTPUT_VIDEO_PARAMS     "c:/temp/" WRITE_OUTPUT_PREFIX "/nir.avi",     -1,     30.0,     cv::Size(512,424),    false    // ffdshow mpeg4/xvid w/ 'grayscale' preset
#define RGB_OUTPUT_VIDEO_PARAMS     "e:/temp/" WRITE_OUTPUT_PREFIX "/rgb.avi",     -1,     30.0,     cv::Size(1920,1080),  true     // x264/mpeg4 w/ 'superfast' preset
/////////////////////////////////
//////// custom params //////////                     folder name                         filename  fileext       frame size         compr params
#define BODYIDX_OUTPUT_IMGSEQ_PARAMS  "e:/temp/" WRITE_OUTPUT_PREFIX "/bodyidx/",          "%05d",   ".bin",   cv::Size(512,424),  std::vector<int>{}
#define DEPTH_OUTPUT_IMGSEQ_PARAMS    "e:/temp/" WRITE_OUTPUT_PREFIX "/depth/",            "%05d",   ".bin",   cv::Size(512,424),  std::vector<int>{}
#define COORDMAP_C2D_OUTPUT_IMGSEQ_PARAMS "e:/temp/" WRITE_OUTPUT_PREFIX "/coordmap_c2d/", "%05d",   ".bin"
#define COORDMAP_D2C_OUTPUT_IMGSEQ_PARAMS "e:/temp/" WRITE_OUTPUT_PREFIX "/coordmap_d2c/", "%05d",   ".bin"
/////////////////////////////////
#define BODY_DATA_OUTPUT_PARAMS     "c:/temp/" WRITE_OUTPUT_PREFIX "/body.bin",     std::ios::out|std::ios::binary
#define META_DATA_OUTPUT_PARAMS     "c:/temp/" WRITE_OUTPUT_PREFIX "/metadata.yml", cv::FileStorage::WRITE
/////////////////////////////////

std::atomic_bool g_bIsActive = true;

int main() {
    try {
    #if WRITE_OUTPUT
        lv::createDirIfNotExist("c:/temp/" WRITE_OUTPUT_PREFIX);
        lv::createDirIfNotExist("e:/temp/" WRITE_OUTPUT_PREFIX);
        std::vector<std::reference_wrapper<lv::DataWriter>> vWriters;
        std::vector<std::reference_wrapper<cv::Mat>> vPackets;
    #endif //WRITE_OUTPUT
        lv::registerAllConsoleSignals([](int){g_bIsActive = false;});
        const auto lEncodeAndSaveImage = [](const cv::Mat& oImage, size_t nRealIndex, const std::string& sPath, const std::vector<int>& vComprParams, size_t& nLastSavedIndex) {
            lvAssert(!oImage.empty() && !sPath.empty());
            lvAssert(nLastSavedIndex==SIZE_MAX || nLastSavedIndex<nRealIndex);
            std::string sRealPath = sPath;
            if(sRealPath.find("%")!=std::string::npos)
                sRealPath = lv::putf(sRealPath.c_str(),(int)++nLastSavedIndex);
            if(sRealPath.size()>=4 && sRealPath.compare(sRealPath.size()-4,sRealPath.size(),".bin")==0)
            #if USING_LZ4
                lv::write(sRealPath,oImage,lv::MatArchive_BINARY_LZ4);
            #else //!USING_LZ4
                lv::write(sRealPath,oImage);
            #endif //!USING_LZ4
            else
                cv::imwrite(sRealPath,oImage,vComprParams);
            return (size_t)0;
        };
        const auto lEncodeAndSaveFrame = [](const cv::Mat& oImage, size_t nRealIndex, cv::VideoWriter& oWriter, size_t& nLastSavedIndex) {
            lvAssert(!oImage.empty() && oWriter.isOpened());
            lvAssert(nLastSavedIndex==SIZE_MAX || nLastSavedIndex<nRealIndex);
            oWriter.write(oImage);
            ++nLastSavedIndex;
            return (size_t)0;
        };
        const auto lEncodeAndSaveBodyFrame = [](const cv::Mat& oBodyFrame, size_t nRealIndex, std::ofstream* pWriter, size_t& nLastSavedIndex){
            lvAssert(!oBodyFrame.empty() && oBodyFrame.total()*oBodyFrame.elemSize()==sizeof(lv::KinectBodyFrame));
            lvAssert(pWriter && pWriter->is_open());
            lvAssert(nLastSavedIndex==SIZE_MAX || nLastSavedIndex<nRealIndex);
            ((lv::KinectBodyFrame*)oBodyFrame.data)->nFrameIdx = ++nLastSavedIndex;
            pWriter->write((char*)oBodyFrame.data,sizeof(lv::KinectBodyFrame));
            return (size_t)0;
        };
    #if USE_LWIR_SENSOR
        std::cout << "Setting up LWIR device..." << std::endl;
        lvAssertHR(CoInitializeEx(0,COINIT_MULTITHREADED|COINIT_DISABLE_OLE1DDE));
        std::unique_ptr<lv::DShowCameraGrabber> pLWIRSensor = std::make_unique<lv::DShowCameraGrabber>("FLIR ThermaCAM");
        lvAssertHR(pLWIRSensor->Connect());
        cv::Mat oLWIRFrame;
        const cv::Size oLWIRFrameSize = std::get<3>(std::make_tuple(LWIR_OUTPUT_VIDEO_PARAMS));
    #if WRITE_OUTPUT
        std::cout << "Setting up LWIR video writer..." << std::endl;
        size_t nLastSavedLWIRFrameIdx = SIZE_MAX;
        cv::VideoWriter oLWIRVideoWriter(LWIR_OUTPUT_VIDEO_PARAMS);
        lvAssert(oLWIRVideoWriter.isOpened());
        lv::DataWriter oLWIRVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oLWIRVideoWriter,nLastSavedLWIRFrameIdx));
        lvAssert(oLWIRVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        vWriters.push_back(oLWIRVideoAsyncWriter);
        vPackets.push_back(oLWIRFrame);
    #endif //WRITE_OUTPUT
    #endif //USE_LWIR_SENSOR
        std::cout << "Setting up Kinect device..." << std::endl;
        CComPtr<IKinectSensor> pKinectSensor;
        lvAssertHR(GetDefaultKinectSensor(&pKinectSensor));
        lvAssert(pKinectSensor);
        lvAssertHR(pKinectSensor->Open());
        CComPtr<IMultiSourceFrameReader> pMultiFrameReader;
    #if USE_NIR_SENSOR
        lvAssertHR(pKinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color|FrameSourceTypes_Depth|FrameSourceTypes_Infrared|FrameSourceTypes_Body|FrameSourceTypes_BodyIndex,&pMultiFrameReader));
    #else //!USE_NIR_SENSOR
        lvAssertHR(pKinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color|FrameSourceTypes_Depth|FrameSourceTypes_Body|FrameSourceTypes_BodyIndex,&pMultiFrameReader));
    #endif //!USE_NIR_SENSOR
        constexpr size_t nStreamCount = size_t(USE_NIR_SENSOR?5:4);
        lv::KinectBodyFrame oBodyFrame;
        cv::Mat oBodyFrameWrapper(1,(int)sizeof(lv::KinectBodyFrame),CV_8UC1,&oBodyFrame);
        cv::Mat oBodyIdxFrame,oNIRFrame,oDepthFrame,oCoordMapFrame_c2d,oCoordMapFrame_d2c,oRGBFrame;
        lvIgnore(oCoordMapFrame_c2d);
        const cv::Size oBodyIdxFrameSize = std::get<3>(std::make_tuple(BODYIDX_OUTPUT_IMGSEQ_PARAMS));
        const cv::Size oNIRFrameSize = std::get<3>(std::make_tuple(NIR_OUTPUT_VIDEO_PARAMS));
        const cv::Size oDepthFrameSize = std::get<3>(std::make_tuple(DEPTH_OUTPUT_IMGSEQ_PARAMS));
        const cv::Size oRGBFrameSize = std::get<3>(std::make_tuple(RGB_OUTPUT_VIDEO_PARAMS));
        CComPtr<ICoordinateMapper> pCoordMapper;
        lvAssertHR(pKinectSensor->get_CoordinateMapper(&pCoordMapper));
    #if WRITE_OUTPUT
        std::cout << "Setting up Kinect body data writer..." << std::endl;
        size_t nLastSavedBodyFrameIdx = SIZE_MAX;
        std::ofstream oBodyStructWriter(BODY_DATA_OUTPUT_PARAMS);
        lvAssert(oBodyStructWriter && oBodyStructWriter.is_open());
        lv::DataWriter oBodyStructAsyncWriter(std::bind(lEncodeAndSaveBodyFrame,std::placeholders::_1,std::placeholders::_2,&oBodyStructWriter,nLastSavedBodyFrameIdx));
        lvAssert(oBodyStructAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        vWriters.push_back(oBodyStructAsyncWriter);
        vPackets.push_back(oBodyFrameWrapper);
        std::cout << "Setting up BODYINDEX video writer..." << std::endl;
        size_t nLastSavedBodyIdxFrameIdx = SIZE_MAX;
        lv::createDirIfNotExist(std::get<0>(std::make_tuple(BODYIDX_OUTPUT_IMGSEQ_PARAMS)));
        const std::string sBodyIdxFramePath = lv::addDirSlashIfMissing(std::get<0>(std::make_tuple(BODYIDX_OUTPUT_IMGSEQ_PARAMS)))+std::get<1>(std::make_tuple(BODYIDX_OUTPUT_IMGSEQ_PARAMS))+std::get<2>(std::make_tuple(BODYIDX_OUTPUT_IMGSEQ_PARAMS));
        const std::vector<int> vnBodyIdxComprParams = std::get<4>(std::make_tuple(BODYIDX_OUTPUT_IMGSEQ_PARAMS));
        lv::DataWriter oBodyIdxVideoAsyncWriter(std::bind(lEncodeAndSaveImage,std::placeholders::_1,std::placeholders::_2,sBodyIdxFramePath,vnBodyIdxComprParams,nLastSavedBodyIdxFrameIdx));
        lvAssert(oBodyIdxVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        vWriters.push_back(oBodyIdxVideoAsyncWriter);
        vPackets.push_back(oBodyIdxFrame);
    #if USE_NIR_SENSOR
        std::cout << "Setting up NIR video writer..." << std::endl;
        size_t nLastSavedNIRFrameIdx = SIZE_MAX;
        cv::VideoWriter oNIRVideoWriter(NIR_OUTPUT_VIDEO_PARAMS);
        lvAssert(oNIRVideoWriter.isOpened());
        lv::DataWriter oNIRVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oNIRVideoWriter,nLastSavedNIRFrameIdx));
        lvAssert(oNIRVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        vWriters.push_back(oNIRVideoAsyncWriter);
        vPackets.push_back(oNIRFrame);
    #else //!USE_NIR_SENSOR
        lvIgnore(oNIRFrameSize);
        lvIgnore(oNIRFrame);
    #endif //!USE_NIR_SENSOR
        std::cout << "Setting up DEPTH video writer..." << std::endl;
        size_t nLastSavedDepthFrameIdx = SIZE_MAX;
        lv::createDirIfNotExist(std::get<0>(std::make_tuple(DEPTH_OUTPUT_IMGSEQ_PARAMS)));
        const std::string sDepthFramePath = lv::addDirSlashIfMissing(std::get<0>(std::make_tuple(DEPTH_OUTPUT_IMGSEQ_PARAMS)))+std::get<1>(std::make_tuple(DEPTH_OUTPUT_IMGSEQ_PARAMS))+std::get<2>(std::make_tuple(DEPTH_OUTPUT_IMGSEQ_PARAMS));
        const std::vector<int> vnDepthComprParams = std::get<4>(std::make_tuple(DEPTH_OUTPUT_IMGSEQ_PARAMS));
        lv::DataWriter oDepthVideoAsyncWriter(std::bind(lEncodeAndSaveImage,std::placeholders::_1,std::placeholders::_2,sDepthFramePath,vnDepthComprParams,nLastSavedDepthFrameIdx));
        lvAssert(oDepthVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        vWriters.push_back(oDepthVideoAsyncWriter);
        vPackets.push_back(oDepthFrame);
        std::cout << "Setting up COORDMAP video writer..." << std::endl;
    #if USE_C2D_MAPPING
        size_t nLastSavedCoordMapFrameIdx_c2d = SIZE_MAX;
        lv::createDirIfNotExist(std::get<0>(std::make_tuple(COORDMAP_C2D_OUTPUT_IMGSEQ_PARAMS)));
        const std::string sCoordMapFramePath_c2d = lv::addDirSlashIfMissing(std::get<0>(std::make_tuple(COORDMAP_C2D_OUTPUT_IMGSEQ_PARAMS)))+std::get<1>(std::make_tuple(COORDMAP_C2D_OUTPUT_IMGSEQ_PARAMS))+std::get<2>(std::make_tuple(COORDMAP_C2D_OUTPUT_IMGSEQ_PARAMS));
        lv::DataWriter oCoordMapVideoAsyncWriter_c2d(std::bind(lEncodeAndSaveImage,std::placeholders::_1,std::placeholders::_2,sCoordMapFramePath_c2d,std::vector<int>(),nLastSavedCoordMapFrameIdx_c2d));
        lvAssert(oCoordMapVideoAsyncWriter_c2d.startAsyncWriting(HIGHDEF_QUEUE_BUFFER_SIZE,true));
        vWriters.push_back(oCoordMapVideoAsyncWriter_c2d);
        vPackets.push_back(oCoordMapFrame_c2d);
    #endif USE_C2D_MAPPING
        size_t nLastSavedCoordMapFrameIdx_d2c = SIZE_MAX;
        lv::createDirIfNotExist(std::get<0>(std::make_tuple(COORDMAP_D2C_OUTPUT_IMGSEQ_PARAMS)));
        const std::string sCoordMapFramePath_d2c = lv::addDirSlashIfMissing(std::get<0>(std::make_tuple(COORDMAP_D2C_OUTPUT_IMGSEQ_PARAMS)))+std::get<1>(std::make_tuple(COORDMAP_D2C_OUTPUT_IMGSEQ_PARAMS))+std::get<2>(std::make_tuple(COORDMAP_D2C_OUTPUT_IMGSEQ_PARAMS));
        lv::DataWriter oCoordMapVideoAsyncWriter_d2c(std::bind(lEncodeAndSaveImage,std::placeholders::_1,std::placeholders::_2,sCoordMapFramePath_d2c,std::vector<int>(),nLastSavedCoordMapFrameIdx_d2c));
        lvAssert(oCoordMapVideoAsyncWriter_d2c.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE*2,true));
        vWriters.push_back(oCoordMapVideoAsyncWriter_d2c);
        vPackets.push_back(oCoordMapFrame_d2c);
        std::cout << "Setting up RGB video writer..." << std::endl;
        size_t nLastSavedRGBFrameIdx = SIZE_MAX;
        //lvAssert(lv::createBinFileWithPrealloc(std::get<0>(std::make_tuple(RGB_OUTPUT_VIDEO_PARAMS)),VIDEO_FILE_PREALLOC_SIZE));
        cv::VideoWriter oRGBVideoWriter(RGB_OUTPUT_VIDEO_PARAMS);
        lvAssert(oRGBVideoWriter.isOpened());
        lv::DataWriter oRGBVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oRGBVideoWriter,nLastSavedRGBFrameIdx));
        lvAssert(oRGBVideoAsyncWriter.startAsyncWriting(HIGHDEF_QUEUE_BUFFER_SIZE,true));
        vWriters.push_back(oRGBVideoAsyncWriter);
        vPackets.push_back(oRGBFrame);
        std::mutex oMetadataStorageMutex;
        cv::FileStorage oMetadataStorage(META_DATA_OUTPUT_PARAMS);
        oMetadataStorage << "htag" << lv::getVersionStamp();
        oMetadataStorage << "date" << lv::getTimeStamp();
    #endif //WRITE_OUTPUT
    #if DISPLAY_OUTPUT
    #if DISPLAY_OUTPUT>1
        lv::DisplayHelperPtr pDisplayHelper = lv::DisplayHelper::create("DISPLAY","c:/temp/",cv::Size(1920,1200),cv::WINDOW_NORMAL);
        std::map<UINT64,cv::Scalar_<uchar>> mBodyColors;
    #else //!(DISPLAY_OUTPUT>1)
    #if USE_LWIR_SENSOR
        cv::namedWindow("oLWIRFrame",cv::WINDOW_NORMAL);
    #endif //USE_LWIR_SENSOR
    #endif //!(DISPLAY_OUTPUT>1)
        cv::namedWindow("oRGBFrame",cv::WINDOW_NORMAL);
    #else //!DISPLAY_OUTPUT
        // still need to display at least one window for visual feedback
    #if USE_LWIR_SENSOR
        cv::namedWindow("oLWIRFrame",cv::WINDOW_NORMAL);
    #else //!USE_LWIR_SENSOR
        cv::namedWindow("oRGBFrame",cv::WINDOW_NORMAL);
    #endif //!USE_LWIR_SENSOR
    #endif //!DISPLAY_OUTPUT
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
                    #if WRITE_OUTPUT
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
                    #endif //WRITE_OUTPUT
                        oBodyIdxFrame.create(oBodyIdxFrameSize,CV_8UC1);
                    }
                    lvAssertHR(pFrame->CopyFrameDataToArray((UINT)oBodyIdxFrame.total(),oBodyIdxFrame.data));
                }
                return bGotFrame;
            },
        #if USE_NIR_SENSOR
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
                    #if WRITE_OUTPUT
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
                    #endif //WRITE_OUTPUT
                        oNIRFrame.create(oNIRFrameSize,CV_16UC1);
                    }
                    lvAssertHR(pFrame->CopyFrameDataToArray((UINT)oNIRFrame.total(),(uint16_t*)oNIRFrame.data));
                }
                return bGotFrame;
            },
        #endif //USE_NIR_SENSOR
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
                    #if WRITE_OUTPUT
                        float fHorizFOV,fVertiFOV;
                        lvAssertHR(pFrameDesc->get_HorizontalFieldOfView(&fHorizFOV));
                        lvAssertHR(pFrameDesc->get_VerticalFieldOfView(&fVertiFOV));
                        USHORT nMaxReliableDist,nMinReliableDist;
                        lvAssertHR(pFrame->get_DepthMaxReliableDistance(&nMaxReliableDist));
                        lvAssertHR(pFrame->get_DepthMinReliableDistance(&nMinReliableDist));
                        CameraIntrinsics oIntrinsics;
                        lvAssertHR(pCoordMapper->GetDepthCameraIntrinsics(&oIntrinsics));
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
                            oMetadataStorage << "coordmap_metadata" << "{";
                            oMetadataStorage << "fx" << oIntrinsics.FocalLengthX;
                            oMetadataStorage << "fy" << oIntrinsics.FocalLengthY;
                            oMetadataStorage << "cx" << oIntrinsics.PrincipalPointX;
                            oMetadataStorage << "cy" << oIntrinsics.PrincipalPointY;
                            oMetadataStorage << "k1" << oIntrinsics.RadialDistortionSecondOrder;
                            oMetadataStorage << "k2" << oIntrinsics.RadialDistortionFourthOrder;
                            oMetadataStorage << "k3" << oIntrinsics.RadialDistortionSixthOrder;
                            oMetadataStorage << "}";
                        }
                    #endif //WRITE_OUTPUT
                        oDepthFrame.create(oDepthFrameSize,CV_16UC1);
                    #if USE_C2D_MAPPING
                        oCoordMapFrame_c2d.create(oRGBFrameSize,CV_32FC2);
                    #endif //USE_C2D_MAPPING
                        oCoordMapFrame_d2c.create(oDepthFrameSize,CV_32FC2);
                    }
                    const UINT nDepthPts = (UINT)oCoordMapFrame_d2c.total();
                    lvAssertHR(pFrame->CopyFrameDataToArray(nDepthPts,(uint16_t*)oDepthFrame.data));
                    static_assert(sizeof(ColorSpacePoint[3])==sizeof(float)*6,"cannot properly fit color space point data into ocv mat");
                #if USE_C2D_MAPPING
                    const UINT nColorPts = (UINT)oCoordMapFrame_c2d.total();
                    lvAssertHR(pCoordMapper->MapColorFrameToDepthSpace(nDepthPts,(uint16_t*)oDepthFrame.data,nColorPts,(DepthSpacePoint*)oCoordMapFrame_c2d.data));
                #endif //USE_C2D_MAPPING
                    lvAssertHR(pCoordMapper->MapDepthFrameToColorSpace(nDepthPts,(uint16_t*)oDepthFrame.data,nDepthPts,(ColorSpacePoint*)oCoordMapFrame_d2c.data));
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
                    if(oRGBFrame.empty()) { // must enter only once
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        lvAssert(cv::Size(nFrameWidth,nFrameHeight)==oRGBFrameSize);
                        uint nFrameBytesPerPixel;
                        lvAssertHR(pFrameDesc->get_BytesPerPixel(&nFrameBytesPerPixel));
                        lvAssert_(nFrameBytesPerPixel==2,"raw color frame not two bytes per pixel");
                    #if WRITE_OUTPUT
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
                            oMetadataStorage << "rgb_metadata" << "{";
                            oMetadataStorage << "orig_size" << oRGBFrameSize;
                            oMetadataStorage << "cv_type" << (int)CV_8UC3;
                            oMetadataStorage << "hfov" << fHorizFOV;
                            oMetadataStorage << "vfov" << fVertiFOV;
                            oMetadataStorage << "init_exposure" << (double)nExposureTime;
                            oMetadataStorage << "init_frameinterv" << (double)nFrameInterval;
                            oMetadataStorage << "init_gain" << (double)fGain;
                            oMetadataStorage << "init_gamma" << (double)fGamma;
                            oMetadataStorage << "}";
                        }
                    #endif //WRITE_OUTPUT
                        oRGBFrame.create(oRGBFrameSize,CV_8UC3);
                    }
                    UINT nBufferSize;
                    BYTE* pBuffer;
                    lvAssertHR(pFrame->AccessRawUnderlyingBuffer(&nBufferSize,&pBuffer));
                    const cv::Mat oRawRGBFrame(oRGBFrame.size(),CV_8UC2,pBuffer);
                    lvAssert(nBufferSize<=oRawRGBFrame.total()*oRawRGBFrame.elemSize());
                    cv::cvtColor(oRawRGBFrame,oRGBFrame,cv::COLOR_YUV2BGR_YUY2);
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
        #if USE_LWIR_SENSOR
            bFinalGrabResult &= (pLWIRSensor->GetLatestFrame(oLWIRFrame,true)>=0);
        #endif //USE_LWIR_SENSOR
            for(size_t n=0; n<abGrabResults.size(); ++n)
                bFinalGrabResult &= abGrabResults[n].get();
            if(bFinalGrabResult) {
            #if USE_LWIR_SENSOR
                static std::once_flag s_oLWIRMetadataWriteFlag;
                std::call_once(s_oLWIRMetadataWriteFlag,[&]() {
                    lvAssert(oLWIRFrame.size()==oLWIRFrameSize);
                #if WRITE_OUTPUT
                    {
                        std::lock_guard<std::mutex> oMetadataStorageLock(oMetadataStorageMutex);
                        oMetadataStorage << "lwir_metadata" << "{";
                        oMetadataStorage << "orig_size" << oLWIRFrameSize;
                        oMetadataStorage << "cv_type" << oLWIRFrame.type();
                        oMetadataStorage << "}";
                    }
                #endif //WRITE_OUTPUT
                });
            #endif //USE_LWIR_SENSOR
            #if DISPLAY_OUTPUT
            #if DISPLAY_OUTPUT>1
                cv::Mat oBodyIdxFrameTemp = oBodyIdxFrame<UCHAR_MAX;
                std::vector<std::vector<cv::Point>> vvBodyIdxContours,vvBodyIdxContours_ColorSpace;
                cv::findContours(oBodyIdxFrameTemp,vvBodyIdxContours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
                vvBodyIdxContours_ColorSpace.resize(vvBodyIdxContours.size());
                for(size_t nBodyIdx=0u; nBodyIdx<vvBodyIdxContours.size(); ++nBodyIdx) {
                    for (size_t nBodyContourPtIdx = 0u; nBodyContourPtIdx < vvBodyIdxContours[nBodyIdx].size(); ++nBodyContourPtIdx) {
                        const cv::Point& oContourPt = vvBodyIdxContours[nBodyIdx][nBodyContourPtIdx];
                        const UINT nDepthPtIdx = UINT(oContourPt.y*oDepthFrameSize.width+oContourPt.x);
                        const ColorSpacePoint& oContourPt_ColorSpace = ((ColorSpacePoint*)oCoordMapFrame_d2c.data)[nDepthPtIdx];
                        vvBodyIdxContours_ColorSpace[nBodyIdx].emplace_back((int)std::round(oContourPt_ColorSpace.X),(int)std::round(oContourPt_ColorSpace.Y));
                    }
                }
                std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvImageNamePairs(2);
                const cv::Size oSuggestedTileSize = oDepthFrame.size();
            #if USE_NIR_SENSOR
                cv::Mat oNIRFrameRaw,oNIRFrameDisplay;
                oNIRFrame.convertTo(oNIRFrameRaw,CV_32F,1.0/USHRT_MAX);
                cv::Scalar vNIRFrameMean,vNIRFrameStdDev;
                cv::meanStdDev(oNIRFrameRaw,vNIRFrameMean,vNIRFrameStdDev);
                oNIRFrameRaw = cv::max(cv::min(oNIRFrameRaw/(vNIRFrameMean[0]*5),0.9f),0.01f);
                cv::applyColorMap(oNIRFrameRaw,oNIRFrameDisplay,cv::COLORMAP_COOL);
                cv::drawContours(oNIRFrameDisplay,vvBodyIdxContours,-1,cv::Scalar(USHRT_MAX),3);
                cv::flip(oNIRFrameDisplay,oNIRFrameDisplay,1);
                vvImageNamePairs[0].emplace_back(oNIRFrameDisplay,"NIR");
            #else //!USE_NIR_SENSOR
                vvImageNamePairs[0].emplace_back(cv::Mat(oNIRFrameSize,CV_16UC1,cv::Scalar_<ushort>(0)),"NIR");
            #endif //!USE_NIR_SENSOR
                cv::Mat oDepthFrameRaw,oDepthFrameDisplay;
                cv::normalize(oDepthFrame,oDepthFrameRaw,1.0,0.0,cv::NORM_MINMAX,CV_32FC1);
                cv::applyColorMap(oDepthFrameRaw,oDepthFrameDisplay,cv::COLORMAP_BONE);
                cv::drawContours(oDepthFrameDisplay,vvBodyIdxContours,-1,cv::Scalar(USHRT_MAX),3);
                cv::flip(oDepthFrameDisplay,oDepthFrameDisplay,1);
                vvImageNamePairs[0].emplace_back(oDepthFrameDisplay,"Depth");
            #if USE_LWIR_SENSOR
                cv::Mat oLWIRFrameDisplay;
                cv::applyColorMap(oLWIRFrame,oLWIRFrameDisplay,cv::COLORMAP_HOT);
                vvImageNamePairs[1].emplace_back(oLWIRFrameDisplay,"LWIR");
            #else //!(USE_LWIR_SENSOR)
                vvImageNamePairs[1].emplace_back(cv::Mat(oDepthFrameDisplay.size(),CV_8UC3,cv::Scalar_<uchar>::all(128)),"LWIR");
            #endif //!(USE_LWIR_SENSOR)
                cv::Mat oBodyFrameDisplay(oBodyIdxFrameSize,CV_8UC3,cv::Scalar_<uchar>::all(88));
                const auto lBodyToScreen = [&](const CameraSpacePoint& oInputPt) {
                    DepthSpacePoint depthPoint = {0};
                    lvAssertHR(pCoordMapper->MapCameraPointToDepthSpace(oInputPt, &depthPoint));
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
                cv::Mat oRGBFrameDisplay = oRGBFrame.clone();
                cv::drawContours(oRGBFrameDisplay,vvBodyIdxContours_ColorSpace,-1,cv::Scalar(0,0,255),3);
                cv::imshow("oRGBFrame",oRGBFrameDisplay);
            #else //DISPLAY_OUTPUT<=1
            #if USE_LWIR_SENSOR
                cv::imshow("oLWIRFrame",oLWIRFrame);
            #endif //USE_LWIR_SENSOR
                cv::imshow("oRGBFrame",oRGBFrame);
            #endif //DISPLAY_OUTPUT<=1
            #else //!DISPLAY_OUTPUT
            #if USE_LWIR_SENSOR
                cv::imshow("oLWIRFrame",oLWIRFrame);
            #else //!USE_LWIR_SENSOR
                cv::imshow("oRGBFrame",oRGBFrame);
            #endif //!USE_LWIR_SENSOR
            #endif //DISPLAY_OUTPUT
                const char c = (char)cv::waitKey(1);
                if(c=='q' || c==27)
                    break;
            #if WRITE_OUTPUT
                lvDbgAssert(!vWriters.empty() && vWriters.size()==vPackets.size());
                std::vector<bool> vQueueAvailabilities(vWriters.size());
                for(size_t nWriterIdx=0; nWriterIdx<vWriters.size(); ++nWriterIdx)
                    vQueueAvailabilities[nWriterIdx] = vWriters[nWriterIdx].get().queue_check(vPackets[nWriterIdx].get(),nRealFrameIdx);
                const bool bCanProcess = std::all_of(vQueueAvailabilities.begin(),vQueueAvailabilities.end(),[](bool b){return b;});
                if(bCanProcess) {
                    lvAssert(oRGBVideoAsyncWriter.queue(oRGBFrame,nRealFrameIdx)!=SIZE_MAX);
                #if USE_LWIR_SENSOR
                    lvAssert(oLWIRVideoAsyncWriter.queue(oLWIRFrame,nRealFrameIdx)!=SIZE_MAX);
                #endif //USE_LWIR_SENSOR
                    lvAssert(oBodyStructAsyncWriter.queue(oBodyFrameWrapper,nRealFrameIdx)!=SIZE_MAX);
                    lvAssert(oBodyIdxVideoAsyncWriter.queue(oBodyIdxFrame,nRealFrameIdx)!=SIZE_MAX);
                #if USE_NIR_SENSOR
                    lvAssert(oNIRVideoAsyncWriter.queue(oNIRFrame,nRealFrameIdx)!=SIZE_MAX);
                #endif //USE_NIR_SENSOR
                    lvAssert(oDepthVideoAsyncWriter.queue(oDepthFrame,nRealFrameIdx)!=SIZE_MAX);
                #if USE_C2D_MAPPING
                    lvAssert(oCoordMapVideoAsyncWriter_c2d.queue(oCoordMapFrame_c2d,nRealFrameIdx)!=SIZE_MAX);
                #endif //USE_C2D_MAPPING
                    lvAssert(oCoordMapVideoAsyncWriter_d2c.queue(oCoordMapFrame_d2c,nRealFrameIdx)!=SIZE_MAX);
                    ++nGoodFrameIdx;
                }
                else {
                    // could add notification of which stream dropped frame here using vQueueAvailablities content
                    std::cout << "dropped a frame!" << std::endl;
                }
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
    catch(const lv::Exception&) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught lv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const cv::Exception&) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    CoUninitialize();
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}
