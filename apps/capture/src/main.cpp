
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

#include "litiv/utils/PlatformUtils.hpp"
#include "litiv/3rdparty/DShowBase/streams.h"
#include "litiv/3rdparty/DShowBase/dshowutil.h"
#include <atlbase.h>
#include <Kinect.h>

////////////////////////////////
#define WRITE_OUTPUT            0
#define DISPLAY_OUTPUT          1
////////////////////////////////

const UINT WM_GRAPH_EVENT = WM_APP + 1;
typedef void (CALLBACK *GraphEventFN)(HWND hwnd, long eventCode, LONG_PTR param1, LONG_PTR param2);

struct DShowCaptureManager {

    DShowCaptureManager(HWND hwnd) {
        m_hwnd = hwnd;
        m_bIsRunning = false;
        m_pBuilder = nullptr;
        m_pControl = nullptr;
        m_pCam = nullptr;
        m_pVMR9 = nullptr;
        m_pEvent = nullptr;
        m_pFilterGraph = nullptr;
        m_pCapBuilder = nullptr;
        BuildGraph();
    }

    ~DShowCaptureManager() {
        TearDownGraph();
    }

    HRESULT Play() {
        if(m_bIsRunning)
            return VFW_E_WRONG_STATE;
        HRESULT hr = m_pControl->Run();
        m_bIsRunning = SUCCEEDED(hr);
        return hr;
    }

    HRESULT Stop() {
        if(!m_bIsRunning)
            return VFW_E_WRONG_STATE;
        HRESULT hr = m_pControl->Stop();
        m_bIsRunning = !SUCCEEDED(hr);
        return hr;
    }

    HRESULT HandleGraphEvent(GraphEventFN pfnOnGraphEvent) {
        if(!m_pEvent)
            return E_UNEXPECTED;
        long evCode = 0;
        LONG_PTR param1 = 0, param2 = 0;
        HRESULT hr = S_OK;
        while(SUCCEEDED(m_pEvent->GetEvent(&evCode,&param1,&param2,0))) {
            pfnOnGraphEvent(m_hwnd,evCode,param1,param2);
            hr = m_pEvent->FreeEventParams(evCode,param1,param2);
            if(FAILED(hr))
                break;
        }
        return hr;
    }

private:

    void BuildGraph() {
        TearDownGraph();
        lvAssertHR(CoCreateInstance(CLSID_FilterGraph,NULL,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&m_pBuilder)));
        lvAssertHR(CoCreateInstance(CLSID_CaptureGraphBuilder2,NULL,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&m_pCapBuilder)));
        lvAssertHR(m_pCapBuilder->SetFiltergraph(m_pBuilder));
        lvAssertHR(m_pBuilder->QueryInterface(IID_PPV_ARGS(&m_pControl)));
        lvAssertHR(m_pBuilder->QueryInterface(IID_PPV_ARGS(&m_pEvent)));
        lvAssertHR(m_pEvent->SetNotifyWindow((OAHWND)m_hwnd,WM_GRAPH_EVENT,NULL));
        lvAssertHR(m_pBuilder->QueryInterface(IID_PPV_ARGS(&m_pFilterGraph)));
        lvAssertHR(AddFilterByName(m_pBuilder,"FLIR ThermaCAM",&m_pCam,CLSID_VideoInputDeviceCategory,L"FLIR ThermaCAM"));
        lvAssertHR(AddFilterByCLSID(m_pBuilder,CLSID_VideoMixingRenderer9,&m_pVMR9,L"VMR9"));
        lvAssertHR(m_pCapBuilder->RenderStream(NULL,NULL,m_pCam,NULL,m_pVMR9));
        Play();
    }

    void TearDownGraph() {
        Stop();
        if(m_pEvent)
            m_pEvent->SetNotifyWindow((OAHWND)NULL,NULL,NULL);
        CComPtr<IEnumFilters> pEnum;
        CComPtr<IBaseFilter> pFilter;
        if(m_pBuilder) {
            if(SUCCEEDED(m_pBuilder->EnumFilters(&pEnum))) {
                while(pEnum->Next(1,&pFilter,NULL)==S_OK) {
                    m_pBuilder->RemoveFilter(pFilter);
                    pEnum->Reset();
                    pFilter = nullptr;
                }
                pEnum = nullptr;
            }
        }
        m_pVMR9 = nullptr;
        m_pCam = nullptr;
        m_pEvent = nullptr;
        m_pControl = nullptr;
        m_pCapBuilder = nullptr;
        m_pBuilder = nullptr;
    }

    HWND m_hwnd;
    bool m_bIsRunning;
    CComPtr<IGraphBuilder> m_pBuilder;
    CComPtr<IMediaControl> m_pControl;
    CComPtr<IMediaEventEx> m_pEvent;
    CComPtr<IBaseFilter> m_pCam;
    CComPtr<IBaseFilter> m_pVMR9;
    CComPtr<IFilterGraph2> m_pFilterGraph;
    CComPtr<ICaptureGraphBuilder2> m_pCapBuilder;
};

DShowCaptureManager* g_pPlayer = nullptr;

void CALLBACK OnGraphEvent(HWND hwnd, long evCode, LONG_PTR param1, LONG_PTR param2) {
    switch (evCode) {
        case EC_COMPLETE:
        case EC_USERABORT:
            g_pPlayer->Stop();
            break;
        case EC_ERRORABORT:
            MessageBox(hwnd, L"Playback error.", TEXT("Error"), MB_OK | MB_ICONERROR);
            g_pPlayer->Stop();
            break;
    }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CREATE:
            g_pPlayer = new (std::nothrow) DShowCaptureManager(hwnd);
            if(g_pPlayer==NULL)
                return -1;
            return 0;
        case WM_DESTROY:
            if(g_pPlayer)
                delete g_pPlayer;
            PostQuitMessage(0);
            return 0;
        case WM_GRAPH_EVENT:
            g_pPlayer->HandleGraphEvent(OnGraphEvent);
            return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

BOOL CtrlHandler(DWORD fdwCtrlType) {
    switch( fdwCtrlType ) {
        case CTRL_C_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
        default:
            if(g_pPlayer)
                delete g_pPlayer;
            return FALSE;
    }
}

int main() {
    try {
        lvAssert(SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler,true));
        HINSTANCE hInstance = GetModuleHandle(NULL);
        lvAssertHR(CoInitializeEx(0,COINIT_MULTITHREADED|COINIT_DISABLE_OLE1DDE));
        const wchar_t CLASS_NAME[]  = L"Sample Window Class";
        WNDCLASS wc = { };
        wc.lpfnWndProc   = WindowProc;
        wc.hInstance     = hInstance;
        wc.lpszClassName = CLASS_NAME;
        RegisterClass(&wc);
        HWND hwnd = CreateWindowEx(0, CLASS_NAME, L"DirectShow Playback",
            WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,
            CW_USEDEFAULT, NULL, NULL, hInstance, NULL);
        if (hwnd == NULL) {
            MessageBox(hwnd, L"CreateWindowEx failed.", TEXT("Error"), MB_OK | MB_ICONERROR);
            return 0;
        }
        MSG msg = { };
        while(GetMessage(&msg, NULL, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        /*CComPtr<IKinectSensor> pSensor;
        if(FAILED(GetDefaultKinectSensor(&pSensor))) {
            std::cout << "\nCould not connect to a Kinect sensor." << std::endl;
            return 1;
        }
        CV_Assert(pSensor);
        lvAssertHR(pSensor->Open());
        CComPtr<IMultiSourceFrameReader> pMultiFrameReader;
        lvAssertHR(pSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color|
            FrameSourceTypes_Depth|
            FrameSourceTypes_Infrared|
            FrameSourceTypes_Body|
            FrameSourceTypes_BodyIndex|
            0,&pMultiFrameReader));
        cv::Mat oNIRFrame,oDepthFrame,oColorFrame;
        size_t nFrameIdx = 0;
        while(true) {
            CComPtr<IMultiSourceFrame> pMultiFrame;
            while(!SUCCEEDED(pMultiFrameReader->AcquireLatestFrame(&pMultiFrame)));
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

            cv::waitKey(0);
            ++nFrameIdx;
        }*/
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    CoUninitialize();
    std::cout << "\n[" << CxxUtils::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}
