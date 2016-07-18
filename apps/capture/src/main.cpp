
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
#include "litiv/3rdparty/DShowBase/ocvcompat.h"
#include <Kinect.h>

/////////////////////////////////
#define USE_FLIR_SENSOR         0
/////////////////////////////////
#define DISPLAY_OUTPUT          0
#define WRITE_OUTPUT            1
/////////////////////////////////
#define DEFAULT_QUEUE_BUFFER_SIZE   1024*1024*50  // max = 50MB per queue (default)
#define HIGHDEF_QUEUE_BUFFER_SIZE   1024*1024*1024 // max = 1GB per queue (high-defition stream)
#define VIDEO_FILE_PREALLOC_SIZE    1024*1024*1024 // tot = 1GB per video file
#define STRUCT_FILE_PREALLOC_SIZE   1024*1024*20 // tot = 200MB per struct file
/////////////////////////////////

std::atomic_bool g_bIsActive = true;

int main() {
    try {
        PlatformUtils::RegisterAllConsoleSignals([](int){g_bIsActive = false;});
        const auto lEncodeAndSaveFrame = [](const cv::Mat& oImage, size_t nIndex, cv::VideoWriter& oWriter, size_t& nLastSavedIndex) {
            lvAssert(!oImage.empty() && oWriter.isOpened());
            lvAssert(nLastSavedIndex==SIZE_MAX || nLastSavedIndex<nIndex);
            oWriter.write(oImage);
            nLastSavedIndex = nIndex;
            return (size_t)0;
        };
        const auto lEncodeAndSaveBodyFrame = [](const cv::Mat& oBodyFrame, size_t nIndex, std::ofstream* pWriter, size_t& nLastSavedIndex){
            lvAssert(!oBodyFrame.empty() && oBodyFrame.total()*oBodyFrame.elemSize()==sizeof(PlatformUtils::KinectBodyFrame));
            lvAssert(pWriter && pWriter->is_open());
            lvAssert(nLastSavedIndex==SIZE_MAX || nLastSavedIndex<nIndex);
            ((PlatformUtils::KinectBodyFrame*)oBodyFrame.data)->nFrameIdx = nIndex;
            pWriter->write((char*)oBodyFrame.data,sizeof(PlatformUtils::KinectBodyFrame));
            nLastSavedIndex = nIndex;
            return (size_t)0;
        };
#if USE_FLIR_SENSOR
        std::cout << "Setting up FLIR device..." << std::endl;
        lvAssertHR(CoInitializeEx(0,COINIT_MULTITHREADED|COINIT_DISABLE_OLE1DDE));
        std::unique_ptr<litiv::DShowCameraGrabber> pFLIRSensor = std::make_unique<litiv::DShowCameraGrabber>("FLIR ThermaCAM",(bool)DISPLAY_OUTPUT);
        lvAssertHR(pFLIRSensor->Connect());
        cv::Mat oFLIRFrame;
#if WRITE_OUTPUT
        std::cout << "Setting up FLIR video writer..." << std::endl;
        size_t nLastSavedFLIRFrameIdx = SIZE_MAX;
        cv::VideoWriter oFLIRVideoWriter("c:/temp/test_flir.avi",-1,30.0,cv::Size(320,240),false);
        lvAssert(oFLIRVideoWriter.isOpened());
        litiv::DataWriter oFLIRVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oFLIRVideoWriter,nLastSavedFLIRFrameIdx));
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
        PlatformUtils::KinectBodyFrame oBodyFrame;
        const cv::Mat oBodyFrameWrapper(1,(int)sizeof(PlatformUtils::KinectBodyFrame),CV_8UC1,&oBodyFrame);
        cv::Mat oBodyIdxFrame,oNIRFrame,oDepthFrame,oColorFrame;
#if WRITE_OUTPUT
        std::cout << "Setting up Kinect body data writer..." << std::endl;
        size_t nLastSavedBodyFrameIdx = SIZE_MAX;
        std::ofstream oBodyStructWriter("c:/temp/test_body.bin",std::ios::out|std::ios::binary);
        lvAssert(oBodyStructWriter && oBodyStructWriter.is_open());
        litiv::DataWriter oBodyStructAsyncWriter(std::bind(lEncodeAndSaveBodyFrame,std::placeholders::_1,std::placeholders::_2,&oBodyStructWriter,nLastSavedBodyFrameIdx));
        lvAssert(oBodyStructAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up BODYINDEX video writer..." << std::endl;
        size_t nLastSavedBodyIdxFrameIdx = SIZE_MAX;
        cv::VideoWriter oBodyIdxVideoWriter("c:/temp/test_bodyidx.avi",-1,30.0,cv::Size(512,424),false);
        lvAssert(oBodyIdxVideoWriter.isOpened());
        litiv::DataWriter oBodyIdxVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oBodyIdxVideoWriter,nLastSavedBodyIdxFrameIdx));
        lvAssert(oBodyIdxVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up NIR video writer..." << std::endl;
        size_t nLastSavedNIRFrameIdx = SIZE_MAX;
        cv::VideoWriter oNIRVideoWriter("c:/temp/test_nir.avi",-1,30.0,cv::Size(512,424),false);
        lvAssert(oNIRVideoWriter.isOpened());
        litiv::DataWriter oNIRVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oNIRVideoWriter,nLastSavedNIRFrameIdx));
        lvAssert(oNIRVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up DEPTH video writer..." << std::endl;
        size_t nLastSavedDepthFrameIdx = SIZE_MAX;
        cv::VideoWriter oDepthVideoWriter("c:/temp/test_depth.avi",-1,30.0,cv::Size(512,424),false);
        lvAssert(oDepthVideoWriter.isOpened());
        litiv::DataWriter oDepthVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oDepthVideoWriter,nLastSavedDepthFrameIdx));
        lvAssert(oDepthVideoAsyncWriter.startAsyncWriting(DEFAULT_QUEUE_BUFFER_SIZE,true));
        std::cout << "Setting up COLOR video writer..." << std::endl;
        size_t nLastSavedColorFrameIdx = SIZE_MAX;
        const std::string sColorVideoFilePath = "e:/temp/test_color.avi";
        lvAssert(PlatformUtils::CreateBinFileWithPrealloc(sColorVideoFilePath,VIDEO_FILE_PREALLOC_SIZE));
        cv::VideoWriter oColorVideoWriter(sColorVideoFilePath,-1,30.0,cv::Size(1920,1080),true);
        lvAssert(oColorVideoWriter.isOpened());
        litiv::DataWriter oColorVideoAsyncWriter(std::bind(lEncodeAndSaveFrame,std::placeholders::_1,std::placeholders::_2,oColorVideoWriter,nLastSavedColorFrameIdx));
        lvAssert(oColorVideoAsyncWriter.startAsyncWriting(HIGHDEF_QUEUE_BUFFER_SIZE,true));
#endif //WRITE_OUTPUT

        CComPtr<IMultiSourceFrame> pMultiFrame;
        PlatformUtils::WorkerPool<nStreamCount> oPool;
        std::array<std::future<bool>,nStreamCount> abGrabResults;
        const std::array<std::function<bool()>,nStreamCount> alGrabTasks = {
            [&]{
                CComPtr<IBodyFrameReference> pFrameRef;
                lvAssertHR(pMultiFrame->get_BodyFrameReference(&pFrameRef));
                CComPtr<IBodyFrame> pFrame;
                IBody* apBodies[BODY_COUNT] = {};
                const bool bGotFrame = SUCCEEDED(pFrameRef->AcquireFrame(&pFrame)) && SUCCEEDED(pFrame->GetAndRefreshBodyData(BODY_COUNT,apBodies));
                if(bGotFrame) {
                    oBodyFrame.bIsValid = true;
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
                    if(oBodyIdxFrame.empty()) {
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        oBodyIdxFrame.create(nFrameHeight,nFrameWidth,CV_8UC1);
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
                    if(oNIRFrame.empty()) {
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        oNIRFrame.create(nFrameHeight,nFrameWidth,CV_16UC1);
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
                    if(oDepthFrame.empty()) {
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        oDepthFrame.create(nFrameHeight,nFrameWidth,CV_16UC1);
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
                    if(oColorFrame.empty()) {
                        //ColorImageFormat eColorFormat;
                        //lvAssertHR(pFrame->get_RawColorImageFormat(&eColorFormat));
                        //CComPtr<IColorCameraSettings> pColorCameraSettings;
                        //lvAssertHR(pFrame->get_ColorCameraSettings(&pColorCameraSettings));
                        CComPtr<IFrameDescription> pFrameDesc;
                        lvAssertHR(pFrame->get_FrameDescription(&pFrameDesc));
                        int nFrameHeight,nFrameWidth;
                        lvAssertHR(pFrameDesc->get_Height(&nFrameHeight));
                        lvAssertHR(pFrameDesc->get_Width(&nFrameWidth));
                        oColorFrame.create(nFrameHeight,nFrameWidth,CV_8UC3);
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
#if USE_FLIR_SENSOR
            pFLIRSensor->GetLatestFrame(oFLIRFrame,true);
#endif //USE_FLIR_SENSOR
            bool bFinalGrabResult = true;
            for(size_t n=0; n<abGrabResults.size(); ++n)
                bFinalGrabResult &= abGrabResults[n].get();
            if(bFinalGrabResult) {
#if DISPLAY_OUTPUT
#if USE_FLIR_SENSOR
                if(!oFLIRFrame.empty())
                    cv::imshow("oFIRFrame",oFIRFrame);
#endif //USE_FLIR_SENSOR
                if(!oBodyIdxFrame.empty())
                    cv::imshow("oBodyIdxFrame",oBodyIdxFrame);
                if(!oNIRFrame.empty())
                    cv::imshow("oNIRFrame",oNIRFrame);
                if(!oDepthFrame.empty())
                    cv::imshow("oDepthFrame",oDepthFrame);
                if(!oColorFrame.empty())
                    cv::imshow("oColorFrame",oColorFrame);
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
                    std::cout << "The color steam is dropping frames!" << std::endl;
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
    std::cout << "\n[" << CxxUtils::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}
