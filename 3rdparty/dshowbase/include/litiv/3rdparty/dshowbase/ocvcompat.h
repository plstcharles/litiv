
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

#include "litiv/3rdparty/dshowbase/dshowutil.h"
#include "litiv/3rdparty/dshowbase/qedit.h"
#include <opencv2/core.hpp>
#include <atlbase.h>
#include <mutex>

namespace lv {

    /// callback event function signature (used to grab stream events)
    using GraphEventFN = void(CALLBACK*)(HWND hwnd, long eventCode, LONG_PTR param1, LONG_PTR param2);
    /// initializes a cv::Mat object based on information contained in the stream's media packet type object
    static STDMETHODIMP initMatFromMediaType(const AM_MEDIA_TYPE* pmt, cv::Mat& oOutput, bool* pbVFlip=nullptr);

    /// directshow filter video frame (sample) grabber; inspired by the MSVC version, but exposes cv compat functions (non-COM!)
    struct DShowFrameGrabber : ISampleGrabberCB {
        /// default constructor, requires a pointer to a pre-inserted+connected SampleGrabber graph filter interface
        DShowFrameGrabber(const CComPtr<ISampleGrabber>& pSG);
        /// will ensure that the SampleGrabber interface callback is disabled
        ~DShowFrameGrabber();
        /// returns the latest (internal) caught sample frame index (or -1 if not ready)
        inline int64_t GetLatestFrameIdx() const {return m_nTotFramesProcessed-1;}
        /// fetches the latest caught sample frame, with optional vertical flipping, returning its index (or -1 if not ready/no internal copy)
        int64_t GetLatestFrame(cv::Mat& oOutput, bool bVFlip=false) const;
        /// sets a callback to use for asynchronuous frame grabbing (high-priority redirect)
        void SetFrameCallback(std::function<void(const cv::Mat&, int64_t)> lCallback, bool bKeepInternalCopy=false);
    protected:
        /// cheat, must create this object manually via constructor (non-COM!)
        virtual STDMETHODIMP_(ULONG) AddRef() override {return 1;}
        /// cheat, must release this object manually via destructor (non-COM!)
        virtual STDMETHODIMP_(ULONG) Release() override {return 2;}
        /// only exposes the 'IID_ISampleGrabberCB' interface
        virtual STDMETHODIMP QueryInterface(REFIID riid, void** ppObj) override;
        /// overrides 'ISampleGrabber' interface method (receives sample data, callsback, stores internally if needed)
        virtual STDMETHODIMP SampleCB(double dSampleTime, IMediaSample* pSample) override;
        /// overrides 'ISampleGrabber' interface method (unused here)
        virtual STDMETHODIMP BufferCB(double /*dSampleTime*/, BYTE* /*pBuffer*/, long /*nBufferLen*/) override {return E_NOTIMPL;}
        /// pointer to graph filter's ISampleGrabber interface
        CComPtr<ISampleGrabber> m_pParent;
        /// internal copy of callback sample data
        cv::Mat m_oInternalSampleCopy;
        /// defines whether a callback sample data copy should be kept internally or not
        bool m_bKeepInternalCopy;
        /// counts total number of packets received -- never resets on stop/start
        int64_t m_nTotFramesProcessed;
        /// internal frame callback copy
        std::function<void(const cv::Mat&, int64_t)> m_lCallback;
        /// internal callback set/get & packet fetching mutex (mutable for use in obvious const functions)
        mutable std::mutex m_oMutex;
    };

    /// directshow camera frame grabber; internally builds a complete/standalone filter graph to get live camera data
    struct DShowCameraGrabber {
        /// default constructor; requires the 'common' name of the targeted capture device, whether to display output or not, and which window handle/msg id to use for external event catching (optional)
        DShowCameraGrabber(const std::string& sVideoDeviceName, bool bDisplayOutput=false, HWND hwnd=NULL, UINT nMsgID=WM_APP+1);
        /// will ensure the graph is properly destroyed on destruction
        ~DShowCameraGrabber();
        /// event loop re-entrypoint; external window should pass all graph-related events to this function
        STDMETHODIMP HandleGraphEvent(GraphEventFN pfnOnGraphEvent);
        /// builds graph & connects filters; if displaying output, video stream will be rendered through the callback filter to a vmr9 window -- it will use a null renderer if display is not required
        STDMETHODIMP Connect();
        /// main graph 'cleanup' function; will stop processing, disconnect all components, and destroy them
        void Disconnect();
        /// returns whether we are connected to the capture device or not
        inline bool IsConnected() const {return m_bIsConnected;}
        /// returns the capture device's latest frame index (or -1 if not ready)
        inline int64_t GetLatestFrameIdx() const {return m_bIsConnected?m_pFrameGrabber->GetLatestFrameIdx():-1;}
        /// fetches the capture device's latest frame, with optional vertical flipping, returning its index (or -1 if not ready)
        inline int64_t GetLatestFrame(cv::Mat& oOutput, bool bVFlip=false) const {return m_bIsConnected?m_pFrameGrabber->GetLatestFrame(oOutput,bVFlip):-1;}
        /// sets a callback to use for asynchronuous frame grabbing (high-priority redirect)
        bool SetFrameCallback(std::function<void(const cv::Mat&, int64_t)> lCallback, bool bKeepInternalCopy=false);
    private:
        /// handle to external app window which should receive graph events (optional)
        const HWND m_hwnd;
        /// external application graph event message id to use when throwing events (optional)
        const UINT m_nMsgID;
        /// defines whether capture device stream should be displayed by the graph or not
        const bool m_bDisplayOutput;
        /// defines the 'common'/moniker name of the video capture device to connect to
        const std::string m_sVideoDeviceName;
        /// internal frame grabber object, which will manage the graph's ISampleGrabber callbacks
        std::unique_ptr<lv::DShowFrameGrabber> m_pFrameGrabber;
        /// graph media control interface to start/stop capture
        CComPtr<IMediaControl> m_pControl;
        /// gra[h media event interface to toggle event behavior
        CComPtr<IMediaEventEx> m_pEvent;
        /// graph video capture device filter
        CComPtr<IBaseFilter> m_pCam;
        /// graph media sample grabber filter
        CComPtr<IBaseFilter> m_pGrabber;
        /// graph media renderer filter
        CComPtr<IBaseFilter> m_pRenderer;
        /// graph builder helper interface
        CComPtr<IGraphBuilder> m_pBuilder;
        /// graph capture builder helper interface
        CComPtr<ICaptureGraphBuilder2> m_pCapBuilder;
        /// defines whether we are connected to the capture device or not
        bool m_bIsConnected;
    };

} // namespace lv
