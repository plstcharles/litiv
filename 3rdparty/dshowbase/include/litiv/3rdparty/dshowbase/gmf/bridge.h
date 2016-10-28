//
// GDCL Multigraph Framework
//
// Bridge.h: Declaration of BridgeController and BridgeStream classes
// that provide the connection between graphs
//
// Copyright (c) GDCL 2004. All Rights Reserved.
// You are free to re-use this as the basis for your own filter development,
// provided you retain this copyright notice in the source.
// http://www.gdcl.co.uk

#pragma once

#include "litiv/3rdparty/dshowbase/gmf/dispatch.h"	// cut-down ATL IDispatch handler

#include <vector>
using namespace std;

// interface definitions
#include "litiv/3rdparty/dshowbase/gmf/GMFBridge_h.h"

// classes declared below
class BridgeStream;			// one pin-to-pin connection between graphs
class BridgeController;		// one filter-to-filter connection

// bridge filter definitions
#include "litiv/3rdparty/dshowbase/gmf/sink.h"
#include "litiv/3rdparty/dshowbase/gmf/source.h"

// Connection Point support
#include "litiv/3rdparty/dshowbase/gmf/cp.h"


// writes log output to c:\GMFBridge.txt if present.
class Logger
{
public:
	Logger(const TCHAR* pFile);
	~Logger();
	void Log(const TCHAR* pFormat, ...);
private:
	CCritSec m_csLog;
	DWORD m_msBase;
	HANDLE m_hFile;
};
extern Logger theLogger;
#define LOG(x)	theLogger.Log x


// utility class implementing semaphore lock/release
class SemLock	// like hemlock - fatal if consumed accidentally!
{
public:
	SemLock()
	: m_hSem(NULL)
	{}
	SemLock(HANDLE hSem)
	: m_hSem(hSem)
	{
		WaitForSingleObject(hSem, INFINITE);
	}
	~SemLock()
	{
		if (m_hSem != NULL)
		{
			ReleaseSemaphore(m_hSem, 1, NULL);
		}
	}
private:
	HANDLE m_hSem;
};




// Represents a single stream (video or audio) that is part of
// a connection between two graphs.
class BridgeStream
{
public:
	BridgeStream(BridgeController* pController, BOOL bVideo, eFormatType AllowedTypes, BOOL bDiscard);
	~BridgeStream();

    // allow copies for vector/list containment
    BridgeStream(const BridgeStream& r);
    const BridgeStream& operator= (const BridgeStream& r);

	HRESULT Bridge(BridgeSinkInput* pInputPin, BridgeSourceOutput* pOutputPin);
	HRESULT DisconnectBridge();

	HRESULT CanReceiveType(const CMediaType* pmt);
    bool IsReceiveConnectionAware()
    {
        return m_bReceiveConnectAware;
    }
    void CanReceiveConnect(bool bRC)
    {
        m_bReceiveConnectAware = bRC;
    }
    bool CanSwitchTo(const CMediaType* pmt);
    HRESULT SwitchTo(const CMediaType* pmt);
	HRESULT GetSelectedType(CMediaType* pmt);
	HRESULT SetSelectedType(const CMediaType* pmt);
	HRESULT BeginFlush();
	HRESULT EndFlush();
	HRESULT Deliver(IMediaSample* pSample);

	HRESULT GetBufferProps(long* pcBuffers, long* pcBytes);
    bool GetDownstreamBufferProps(long* pcBuffers, long* pcBytes);
	HRESULT CanDeliverType(const CMediaType* pmt);
	HRESULT EnumOutputType(int iPosition, CMediaType* pmt);
	HRESULT NotifyQuality(IBaseFilter* pSender, Quality q);

    // source graph can change type until we start building
    // output stages. After that it must be by
    // dynamic type change only.
    void TypeFixed()
    {
        // get the input pin's current type
        m_pInputPin->CurrentType(&m_mt);
        m_bTypeFixed = true;
    }

	BridgeController* GetController()
	{
		return m_pController;
	}

    BOOL IsVideo()
    {
        return m_bVideo;
    }
    BOOL DiscardMode()
    {
        return m_bDiscard;
    }
    eFormatType AllowedTypes()
    {
        return m_AllowedTypes;
    }

    static HRESULT CheckMismatchedVideo(const CMediaType* pmt1, const CMediaType* pmt2);
    static HRESULT GetVideoDimensions(const CMediaType* pmt, long *pcx, long* pcy);

	void ResetOnStop();

private:
	BridgeController* m_pController;

    BOOL m_bVideo;
    eFormatType m_AllowedTypes;
    BOOL m_bDiscard;

    CCritSec m_csType;
	CMediaType m_mt;
	bool m_bTypeFixed;
    bool m_bReceiveConnectAware;

    // the first sink graph to be built will effectively lock the
    // buffer size in the non-video case.
    bool m_bBufferSizeLocked;
    long m_cBuffers;
    long m_cBytes;

	// refcount is held on filter by BridgeController
	// -- access is protected by semaphore m_hsemDelivery
	BridgeSinkInput* m_pInputPin;
	BridgeSourceOutput* m_pOutputPin;
};

// connection point container managing outgoing events from controller to (eg VB) clients
class BridgeEvents : public CPContainer
{
public:
	BridgeEvents(IUnknown* pUnk, HRESULT* phr);
	void Fire_OnSegmentEnd();
	bool hasClients();
};

//
// Manage a connection between two graphs, with a sink filter in
// one graph and a source filter in another graph. One source and one sink
// can be connected at once, but you can switch the source seamlessly and
// also change the sink (after stopping sink graph).
//
class BridgeController
: public CUnknown,
  public BaseIDispatch<IGMFBridgeController3, &__uuidof(IGMFBridgeController2)>
{
public:
	// constructor method used by class factory
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN pUnk, HRESULT* phr);

	DECLARE_IUNKNOWN
	STDMETHOD(NonDelegatingQueryInterface)(REFIID iid, void**ppv);

    int StreamCount()
    {
        return (int)m_Streams.size();
    }
    BridgeStream* GetStream(int n)
    {
        if ((n < 0) || (n > StreamCount()))
        {
            return NULL;
        }
        return &m_Streams[n];
    }

	// notify app that current segment is at end
	void OnEndOfSegment()
	{
		if (IsWindow(m_hwnd))
		{
			PostMessage(m_hwnd, m_NotifyMsg, 0, 0);
		}
		if (IsWindow(m_hwndNotify) && m_pContainer->hasClients())
		{
			PostMessage(m_hwndNotify, WM_USER, 0, 0);
		}
	}

	bool CanSupplyType(IPin* pPin, const GUID* pMajorType);
	HRESULT FindUnconnectedPin(IBaseFilter* pFilter, IPin** ppPin, PIN_DIRECTION dir, const GUID* pMajorType);
	HRESULT FindStreamSource(IBaseFilter* pFilter, const GUID* pMajorType, IPin** ppPin);
	long BufferMinimum()
	{
		return m_nMillisecs;
	}
    bool LiveTiming()   { return m_bLiveTiming; }


	// IGMFBridgeController interface
public:
    STDMETHOD(AddStream)(BOOL bVideo, eFormatType AllowedTypes, BOOL bDiscardUnconnected);
	STDMETHOD(InsertSinkFilter)(IUnknown* pGraph, IUnknown** ppFilter);
	STDMETHOD(InsertSourceFilter)(IUnknown* pUnkSourceGraphSinkFilter, IUnknown* pRenderGraph, IUnknown** ppFilter);
	STDMETHOD(CreateSourceGraph)(BSTR strFile, IUnknown* pGraph, IUnknown **pSinkFilter);
	STDMETHOD(CreateRenderGraph)(IUnknown* pSourceGraphSinkFilter, IUnknown* pRenderGraph, IUnknown** pRenderGraphSourceFilter);
	STDMETHOD(BridgeGraphs)(IUnknown* pSourceGraph, IUnknown* pRenderGraph);
	STDMETHOD(SetNotify)(LONG_PTR hwnd, long msg);
	STDMETHOD(SetBufferMinimum)(long nMillisecs);
	STDMETHOD(GetSegmentTime)(double* pdSeconds);
	STDMETHOD(NoMoreSegments)();
    STDMETHOD(GetSegmentOffset)(double* pdOffset);

    // IGMFBridgeController2 interface
public:
    STDMETHOD(BridgeAtDiscont)(IUnknown* pSourceGraphSinkFilter, IUnknown* pRenderGraphSourceFilter, BOOL bIsDiscontinuity);

	// IGMFBridgeController3 interface
public:
	STDMETHOD(SetLiveTiming)(BOOL bIsLiveTiming)
	{
		m_bLiveTiming = bIsLiveTiming ? true : false;
		return S_OK;
	}

//private: // should only access via COM... (but we can cheat)
	BridgeController(LPUNKNOWN pUnk);
	~BridgeController(void);

private:
	// helper method
	HRESULT GraphFromFilter(IUnknown *pFilter, IGraphBuilder** ppGraph);

	// for marshalling to UI thread
    static LRESULT CALLBACK DispatchWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
    LRESULT OnMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);


private:
	CCritSec m_csBridge;
    vector<BridgeStream> m_Streams;

	// currently bridged filters
	IBridgeSinkPtr m_pCurrentSourceGraphSinkFilter;
	IBridgeSourcePtr m_pCurrentRenderGraphSourceFilter;

	// remaining EOS calls expected
	long m_nEOS;

	HWND m_hwnd;
	long m_NotifyMsg;

	// buffering minimum for subsequent render graphs
	long m_nMillisecs;

	// for outgoing events (eg VB)
	smart_ptr<BridgeEvents> m_pContainer;
	HWND m_hwndNotify;

    bool m_bLiveTiming;
};


