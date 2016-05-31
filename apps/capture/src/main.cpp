
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

#include "litiv/3rdparty/DShowBase/streams.h"
#include "litiv/3rdparty/DShowBase/ocvcompat.h"
#include <Kinect.h>

////////////////////////////////
#define WRITE_OUTPUT            0
#define DISPLAY_OUTPUT          0
////////////////////////////////

int main() {
    try {
        lvAssertHR(CoInitializeEx(0,COINIT_MULTITHREADED|COINIT_DISABLE_OLE1DDE));
        std::unique_ptr<litiv::DShowCameraGrabber> pFLIRSensor = std::make_unique<litiv::DShowCameraGrabber>("FLIR ThermaCAM",(bool)DISPLAY_OUTPUT);
        lvAssertHR(pFLIRSensor->Connect());
        CComPtr<IKinectSensor> pKinectSensor;
        lvAssertHR(GetDefaultKinectSensor(&pKinectSensor));
        lvAssert(pKinectSensor);
        lvAssertHR(pKinectSensor->Open());
        CComPtr<IMultiSourceFrameReader> pMultiFrameReader;
        lvAssertHR(pKinectSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color|FrameSourceTypes_Depth|FrameSourceTypes_Infrared|FrameSourceTypes_Body|FrameSourceTypes_BodyIndex,&pMultiFrameReader));
        cv::Mat oFIRFrame,oNIRFrame,oDepthFrame,oColorFrame;
        size_t nFrameIdx = 0;
        while(true) {
            CComPtr<IMultiSourceFrame> pMultiFrame;
            while(!SUCCEEDED(pMultiFrameReader->AcquireLatestFrame(&pMultiFrame)));
            pFLIRSensor->GetLatestFrame(oFIRFrame,true);
            bool bGotNIR=false, bGotDepth=false, bGotColor=false;
            {
                CComPtr<IInfraredFrameReference> pNIRFrameRef;
                lvAssertHR(pMultiFrame->get_InfraredFrameReference(&pNIRFrameRef));
                CComPtr<IInfraredFrame> pNIRFrame;
                if((bGotNIR=SUCCEEDED(pNIRFrameRef->AcquireFrame(&pNIRFrame)))!=0) {
                    pNIRFrameRef.Release();
                    if(oNIRFrame.empty()) {
                        CComPtr<IFrameDescription> pNIRFrameDesc;
                        lvAssertHR(pNIRFrame->get_FrameDescription(&pNIRFrameDesc));
                        int nNIRFrameHeight,nNIRFrameWidth;
                        lvAssertHR(pNIRFrameDesc->get_Height(&nNIRFrameHeight));
                        lvAssertHR(pNIRFrameDesc->get_Width(&nNIRFrameWidth));
                        oNIRFrame.create(nNIRFrameHeight,nNIRFrameWidth,CV_16UC1);
                    }
                    lvAssertHR(pNIRFrame->CopyFrameDataToArray(oNIRFrame.total(),(uint16_t*)oNIRFrame.data));
                }
            }
            {
                CComPtr<IDepthFrameReference> pDepthFrameRef;
                lvAssertHR(pMultiFrame->get_DepthFrameReference(&pDepthFrameRef));
                CComPtr<IDepthFrame> pDepthFrame;
                if((bGotDepth=SUCCEEDED(pDepthFrameRef->AcquireFrame(&pDepthFrame)))!=0) {
                    pDepthFrameRef.Release();
                    if(oDepthFrame.empty()) {
                        CComPtr<IFrameDescription> pDepthFrameDesc;
                        lvAssertHR(pDepthFrame->get_FrameDescription(&pDepthFrameDesc));
                        int nDepthFrameHeight,nDepthFrameWidth;
                        lvAssertHR(pDepthFrameDesc->get_Height(&nDepthFrameHeight));
                        lvAssertHR(pDepthFrameDesc->get_Width(&nDepthFrameWidth));
                        oDepthFrame.create(nDepthFrameHeight,nDepthFrameWidth,CV_16UC1);
                    }
                    lvAssertHR(pDepthFrame->CopyFrameDataToArray(oDepthFrame.total(),(uint16_t*)oDepthFrame.data));
                }
            }
            {
                CComPtr<IColorFrameReference> pColorFrameRef;
                lvAssertHR(pMultiFrame->get_ColorFrameReference(&pColorFrameRef));
                CComPtr<IColorFrame> pColorFrame;
                if((bGotColor=SUCCEEDED(pColorFrameRef->AcquireFrame(&pColorFrame)))!=0) {
                    pColorFrameRef.Release();
                    if(oColorFrame.empty()) {
                        //ColorImageFormat eColorFormat;
                        //lvAssertHR(pColorFrame->get_RawColorImageFormat(&eColorFormat));
                        //CComPtr<IColorCameraSettings> pColorCameraSettings;
                        //lvAssertHR(pColorFrame->get_ColorCameraSettings(&pColorCameraSettings));
                        CComPtr<IFrameDescription> pColorFrameDesc;
                        lvAssertHR(pColorFrame->get_FrameDescription(&pColorFrameDesc));
                        int nColorFrameHeight,nColorFrameWidth;
                        lvAssertHR(pColorFrameDesc->get_Height(&nColorFrameHeight));
                        lvAssertHR(pColorFrameDesc->get_Width(&nColorFrameWidth));
                        oColorFrame.create(nColorFrameHeight,nColorFrameWidth,CV_8UC4);
                    }
                    lvAssertHR(pColorFrame->CopyConvertedFrameDataToArray(oColorFrame.total()*oColorFrame.elemSize(),oColorFrame.data,ColorImageFormat_Bgra));
                }
            }
            cv::imshow("oFIRFrame",oFIRFrame);
            if(bGotNIR)
                cv::imshow("oNIRFrame",oNIRFrame);
            else
                std::cout << "#" << nFrameIdx << " NIR failed" << std::endl;
            if(bGotDepth)
                cv::imshow("oDepthFrame",oDepthFrame);
            else
                std::cout << "#" << nFrameIdx << " depth failed" << std::endl;
            if(bGotColor)
                cv::imshow("oColorFrame",oColorFrame);
            else
                std::cout << "#" << nFrameIdx << " color failed" << std::endl;
            cv::waitKey(1);
            ++nFrameIdx;
        }
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    CoUninitialize();
    std::cout << "\n[" << CxxUtils::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}
