
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

#include "litiv/3rdparty/dshowbase/streams.h"
#include <string>
#include <vector>
#include <iostream>
#include <windows.h>
#include <winerror.h>
#include <comdef.h>
#include <stdint.h>
#include <direct.h>
#include <psapi.h>

template<class T>
void SafeRelease(T **ppT) {if(*ppT) {(*ppT)->Release();*ppT = nullptr;}}

#include "litiv/3rdparty/dshowbase/ocvcompat.h"

STDMETHODIMP lv::initMatFromMediaType(const AM_MEDIA_TYPE* pmt, cv::Mat& oOutput, bool* pbVFlip) {
    if(!pmt)
        return E_POINTER;
    if(!pmt->bFixedSizeSamples)
        return E_INVALIDARG;
    if(pmt->bTemporalCompression)
        return E_INVALIDARG;
    if(pmt->majortype!=MEDIATYPE_Video || pmt->formattype!=FORMAT_VideoInfo || pmt->cbFormat<sizeof(VIDEOINFOHEADER))
        return E_INVALIDARG; // for now, only supports simple video types (may add audio buffer support later)
    const VIDEOINFOHEADER* pvih = (VIDEOINFOHEADER*)pmt->pbFormat;
    const BITMAPINFOHEADER& bih = pvih->bmiHeader;
    if(bih.biCompression!=BI_RGB) // TODO @@@
        return E_INVALIDARG;
    if(pmt->subtype==MEDIASUBTYPE_RGB8) {
        oOutput.create(abs(bih.biHeight),bih.biWidth,CV_8UC1);
        if(pbVFlip) *pbVFlip = bih.biHeight>0;
    }
    else if(pmt->subtype==MEDIASUBTYPE_RGB24) {
        oOutput.create(abs(bih.biHeight),bih.biWidth,CV_8UC3);
        if(pbVFlip) *pbVFlip = bih.biHeight>0;
    }
    else if(pmt->subtype==MEDIASUBTYPE_ARGB32) {
        oOutput.create(abs(bih.biHeight),bih.biWidth,CV_8UC4);
        if(pbVFlip) *pbVFlip = bih.biHeight>0;
    }
    else if(pmt->subtype==MEDIASUBTYPE_RGB555) {
        oOutput.create(abs(bih.biHeight),bih.biWidth,CV_16UC1);
        if(pbVFlip) *pbVFlip = bih.biHeight>0;
    }
    else if(pmt->subtype==MEDIASUBTYPE_ARGB4444) {
        oOutput.create(abs(bih.biHeight),bih.biWidth,CV_16UC1);
        if(pbVFlip) *pbVFlip = bih.biHeight>0;
    }
    else if(pmt->subtype==MEDIASUBTYPE_YUY2) // TODO @@@
        return E_INVALIDARG;
    else if(pmt->subtype==MEDIASUBTYPE_YVYU) // TODO @@@
        return E_INVALIDARG;
    else
        return E_INVALIDARG;
    return S_OK;
}

lv::DShowFrameGrabber::DShowFrameGrabber(const CComPtr<ISampleGrabber>& pSG) {
    if(pSG==nullptr)
        throw std::invalid_argument("bad sample grabber interface");
    AM_MEDIA_TYPE mt;
    ZeroMemory(&mt,sizeof(mt));
    if(pSG->GetConnectedMediaType(&mt)!=S_OK)
        throw std::runtime_error("cannot get sample grabber media type");
    if(initMatFromMediaType(&mt,m_oInternalSampleCopy)!=S_OK) {
        FreeMediaType(mt);
        throw std::runtime_error("unhandled media type");
    }
    FreeMediaType(mt);
    if(m_oInternalSampleCopy.empty())
        throw std::runtime_error("bad internal sample copy type");
    m_bKeepInternalCopy = true;
    m_pParent = pSG;
    m_pParent->SetCallback(this,0);
    m_nTotFramesProcessed = 0;
}

lv::DShowFrameGrabber::~DShowFrameGrabber() {
    m_pParent->SetCallback(NULL,0);
}

int64_t lv::DShowFrameGrabber::GetLatestFrame(cv::Mat& oOutput, bool bVFlip) const {
    std::lock_guard<std::mutex> oLock(m_oMutex);
    if(!m_bKeepInternalCopy || m_oInternalSampleCopy.empty()) {
        oOutput = cv::Mat();
        return -1;
    }
    else {
        oOutput.create(m_oInternalSampleCopy.size(),m_oInternalSampleCopy.type());
        if(bVFlip) {
            for(int i=0; i<m_oInternalSampleCopy.rows; ++i) {
                const uchar* pInRow = m_oInternalSampleCopy.data+i*m_oInternalSampleCopy.step[0];
                uchar* pOutRow = oOutput.data+(oOutput.rows-1-i)*oOutput.step[0];
                std::copy(pInRow,pInRow+m_oInternalSampleCopy.step[0],pOutRow);
            }
        }
        else
            m_oInternalSampleCopy.copyTo(oOutput);
    }
    return m_nTotFramesProcessed;
}

void lv::DShowFrameGrabber::SetFrameCallback(std::function<void(const cv::Mat&, int64_t)> lCallback, bool bKeepInternalCopy) {
    std::lock_guard<std::mutex> oLock(m_oMutex);
    m_bKeepInternalCopy = bKeepInternalCopy;
    m_lCallback = lCallback;
}

STDMETHODIMP lv::DShowFrameGrabber::QueryInterface(REFIID riid, void** ppObj) {
    if(ppObj==nullptr)
        return E_POINTER;
    else if(riid==IID_ISampleGrabberCB) {
        *ppObj = (ISampleGrabberCB*)this;
        return S_OK;
    }
    return E_NOINTERFACE;
}

STDMETHODIMP lv::DShowFrameGrabber::SampleCB(double dSampleTime, IMediaSample* pSample) {
    uchar* pBuffer;
    if(pSample->GetPointer(&pBuffer)==S_OK) {
        AM_MEDIA_TYPE* pmt = nullptr;
        HRESULT hr = pSample->GetMediaType(&pmt);
        std::lock_guard<std::mutex> oLock(m_oMutex);
        if(SUCCEEDED(hr) && pmt!=nullptr) {
            hr = initMatFromMediaType(pmt,m_oInternalSampleCopy);
            DeleteMediaType(pmt);
            if(hr!=S_OK)
                return hr;
        }
        const size_t nPacketSize = pSample->GetActualDataLength();
        if(nPacketSize>m_oInternalSampleCopy.total()*m_oInternalSampleCopy.elemSize())
            return E_OUTOFMEMORY;
        if(m_lCallback)
            m_lCallback(cv::Mat(m_oInternalSampleCopy.size(),m_oInternalSampleCopy.type(),pBuffer),m_nTotFramesProcessed);
        if(m_bKeepInternalCopy)
            std::copy(pBuffer,pBuffer+nPacketSize,m_oInternalSampleCopy.data);
        ++m_nTotFramesProcessed;
    }
    return S_OK;
}

lv::DShowCameraGrabber::DShowCameraGrabber(const std::string& sVideoDeviceName, bool bDisplayOutput, HWND hwnd, UINT nMsgID) :
        m_hwnd(hwnd), m_nMsgID(nMsgID), m_bDisplayOutput(bDisplayOutput), m_sVideoDeviceName(sVideoDeviceName) {
    Disconnect(); // init all ptrs to zero
}

lv::DShowCameraGrabber::~DShowCameraGrabber() {
    Disconnect();
}

STDMETHODIMP lv::DShowCameraGrabber::HandleGraphEvent(GraphEventFN pfnOnGraphEvent) {
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

STDMETHODIMP lv::DShowCameraGrabber::Connect() {
    Disconnect();
    HRESULT hr;
    if(FAILED(hr=CoCreateInstance(CLSID_FilterGraph,NULL,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&m_pBuilder))))
        return hr;
    if(FAILED(hr=CoCreateInstance(CLSID_CaptureGraphBuilder2,NULL,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(&m_pCapBuilder))))
        return hr;
    if(FAILED(hr=m_pCapBuilder->SetFiltergraph(m_pBuilder)))
        return hr;
    if(FAILED(hr=m_pBuilder->QueryInterface(IID_PPV_ARGS(&m_pControl))))
        return hr;
    if(FAILED(hr=m_pBuilder->QueryInterface(IID_PPV_ARGS(&m_pEvent))))
        return hr;
    if(FAILED(hr=m_pEvent->SetNotifyWindow((OAHWND)m_hwnd,m_hwnd==NULL?NULL:m_nMsgID,NULL)))
        return hr;
    if(FAILED(hr=AddFilterByName(m_pBuilder,m_sVideoDeviceName,&m_pCam,CLSID_VideoInputDeviceCategory,L"DShow Camera Grabber Input Device")))
        return hr;
    if(m_bDisplayOutput && FAILED(hr=AddFilterByCLSID(m_pBuilder,CLSID_VideoMixingRenderer9,&m_pRenderer,L"VMR9")))
        return hr;
    else if(!m_bDisplayOutput && FAILED(hr=AddFilterByCLSID(m_pBuilder,CLSID_NullRenderer,&m_pRenderer,L"Null Renderer")))
        return hr;
    if(FAILED(hr=AddFilterByCLSID(m_pBuilder,CLSID_SampleGrabber,&m_pGrabber,L"Frame Grabber")))
        return hr;
    CComPtr<ISampleGrabber> pGrabberInterf;
    if(FAILED(hr=m_pGrabber->QueryInterface(IID_ISampleGrabber,(void**)&pGrabberInterf)))
        return hr;
    AM_MEDIA_TYPE mt;
    ZeroMemory(&mt,sizeof(mt));
    mt.majortype = MEDIATYPE_Video;
    mt.subtype = MEDIASUBTYPE_RGB24;
    mt.formattype = FORMAT_VideoInfo;
    if(FAILED(hr=pGrabberInterf->SetMediaType(&mt)))
        return hr;
    if(FAILED(hr=pGrabberInterf->SetOneShot(false)))
        return hr;
    if(FAILED(hr=pGrabberInterf->SetBufferSamples(false)))
        return hr;
    if(FAILED(hr=m_pCapBuilder->RenderStream(/*&PIN_CATEGORY_PREVIEW*/NULL,&MEDIATYPE_Video,m_pCam,m_pGrabber,m_pRenderer)))
        return hr;
    m_pFrameGrabber = std::make_unique<lv::DShowFrameGrabber>(pGrabberInterf);
    CComQIPtr<IMediaFilter> pMedia = m_pBuilder;
    if(pMedia==NULL)
        return E_NOINTERFACE;
    if(FAILED(hr=pMedia->SetSyncSource(NULL)))
        return hr;
    if(FAILED(hr=m_pControl->Run()))
        return hr;
    m_bIsConnected = true;
    return S_OK;
}

void lv::DShowCameraGrabber::Disconnect() {
    m_bIsConnected = false;
    if(m_pControl)
        m_pControl->Stop();
    if(m_pGrabber) {
        CComPtr<ISampleGrabber> pGrabberInterf;
        if(SUCCEEDED(m_pGrabber->QueryInterface(IID_ISampleGrabber,(void**)&pGrabberInterf)))
            pGrabberInterf->SetCallback(nullptr,0);
    }
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
    m_pCam = nullptr;
    m_pFrameGrabber = nullptr;
    m_pGrabber = nullptr;
    m_pRenderer = nullptr;
    m_pEvent = nullptr;
    m_pControl = nullptr;
    m_pCapBuilder = nullptr;
    m_pBuilder = nullptr;
}

bool lv::DShowCameraGrabber::SetFrameCallback(std::function<void(const cv::Mat&, int64_t)> lCallback, bool bKeepInternalCopy) {
    if(m_bIsConnected)
        m_pFrameGrabber->SetFrameCallback(lCallback,bKeepInternalCopy);
    return m_bIsConnected;
}
