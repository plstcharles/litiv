
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

#pragma once

#include "litiv/3rdparty/DShowBase/dshowutil.h"
#include "litiv/3rdparty/DShowBase/qedit.h"
#include <opencv2/core.hpp>
#include <atlbase.h>
#include <mutex>

namespace litiv {

    using GraphEventFN = void(CALLBACK*)(HWND hwnd, long eventCode, LONG_PTR param1, LONG_PTR param2);
    static STDMETHODIMP initMatFromMediaType(const AM_MEDIA_TYPE* pmt, cv::Mat& oOutput, bool* pbVFlip=nullptr);

    struct DShowFrameGrabber : ISampleGrabberCB {
        DShowFrameGrabber(const CComPtr<ISampleGrabber>& pSG); // never create via COM (cheat); must provide already connected sample grabber filter
        ~DShowFrameGrabber();
        inline int64_t GetLatestFrameIdx() const {return m_nTotFramesProcessed;}
        int64_t GetLatestFrame(cv::Mat& oOutput, bool bVFlip=false) const;
        void SetFrameCallback(std::function<void(const cv::Mat&)> lCallback, bool bKeepInternalCopy=false);
    private:
        virtual STDMETHODIMP_(ULONG) AddRef() override {return 1;} // cheat, create/release this object statically only
        virtual STDMETHODIMP_(ULONG) Release() override {return 2;} // cheat, create/release this object statically only
        virtual STDMETHODIMP QueryInterface(REFIID riid, void** ppObj) override;
        virtual STDMETHODIMP SampleCB(double dSampleTime, IMediaSample* pSample) override;
        virtual STDMETHODIMP BufferCB(double /*dSampleTime*/, BYTE* /*pBuffer*/, long /*nBufferLen*/) override {return E_NOTIMPL;}
        CComPtr<ISampleGrabber> m_pParent;
        cv::Mat m_oInternalSampleCopy;
        bool m_bKeepInternalCopy;
        int64_t m_nLastTick;
        int64_t m_nTotFramesProcessed;
        int64_t m_nCurrFramesProcessed;
        std::function<void(const cv::Mat&)> m_lCallback;
        mutable std::mutex m_oMutex;
    };

    struct DShowCameraGrabber {
        DShowCameraGrabber(const std::string& sVideoDeviceName, bool bDisplayOutput=false, HWND hwnd=NULL, UINT nMsgID=WM_APP+1);
        ~DShowCameraGrabber();
        STDMETHODIMP HandleGraphEvent(GraphEventFN pfnOnGraphEvent);
        STDMETHODIMP Connect();
        void Disconnect();
        inline bool IsConnected() const {return m_bIsConnected;}
        inline int64_t GetLatestFrameIdx() const {return m_bIsConnected?m_pFrameGrabber->GetLatestFrameIdx():-1;}
        inline int64_t GetLatestFrame(cv::Mat& oOutput, bool bVFlip=false) const {return m_bIsConnected?m_pFrameGrabber->GetLatestFrame(oOutput,bVFlip):-1;}
        bool SetFrameCallback(std::function<void(const cv::Mat&)> lCallback, bool bKeepInternalCopy=false);
    private:
        const HWND m_hwnd;
        const UINT m_nMsgID;
        const bool m_bDisplayOutput;
        const std::string m_sVideoDeviceName;
        std::unique_ptr<litiv::DShowFrameGrabber> m_pFrameGrabber;
        CComPtr<IMediaControl> m_pControl;
        CComPtr<IMediaEventEx> m_pEvent;
        CComPtr<IBaseFilter> m_pCam;
        CComPtr<IBaseFilter> m_pGrabber;
        CComPtr<IBaseFilter> m_pRenderer;
        CComPtr<IGraphBuilder> m_pBuilder;
        CComPtr<IFilterGraph2> m_pFilterGraph;
        CComPtr<ICaptureGraphBuilder2> m_pCapBuilder;
        bool m_bIsConnected;
    };


} //namespace litiv
