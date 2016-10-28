
//
// GDCL Multigraph Framework
//
// Bridge.cpp: Implementation of BridgeController and BridgeStream classes
// that provide the connection between graphs
//
// Copyright (c) GDCL 2004. All Rights Reserved.
// You are free to re-use this as the basis for your own development,
// provided you retain this copyright notice in the source.
// http://www.gdcl.co.uk

#pragma warning(push)
#pragma warning(disable:4312)
#pragma warning(disable:4995)
#include "litiv/3rdparty/dshowbase/streams.h"
#include <comdef.h>
_COM_SMARTPTR_TYPEDEF(IMemAllocator, IID_IMemAllocator);
_COM_SMARTPTR_TYPEDEF(IGraphBuilder, IID_IGraphBuilder);
_COM_SMARTPTR_TYPEDEF(IBaseFilter, IID_IBaseFilter);
_COM_SMARTPTR_TYPEDEF(IEnumMediaTypes, IID_IEnumMediaTypes);
_COM_SMARTPTR_TYPEDEF(IMediaSample, IID_IMediaSample);
_COM_SMARTPTR_TYPEDEF(IMediaSample2, IID_IMediaSample2);
_COM_SMARTPTR_TYPEDEF(IEnumFilters, IID_IEnumFilters);
_COM_SMARTPTR_TYPEDEF(IQualityControl, IID_IQualityControl);
_COM_SMARTPTR_TYPEDEF(IEnumPins, IID_IEnumPins);
_COM_SMARTPTR_TYPEDEF(IPin, IID_IPin);
_COM_SMARTPTR_TYPEDEF(IEnumMediaTypes, IID_IEnumMediaTypes);
_COM_SMARTPTR_TYPEDEF(IMediaControl, IID_IMediaControl);
_COM_SMARTPTR_TYPEDEF(IMediaSeeking, IID_IMediaSeeking);
_COM_SMARTPTR_TYPEDEF(IVideoWindow, IID_IVideoWindow);
_COM_SMARTPTR_TYPEDEF(IBasicVideo, IID_IBasicVideo);
_COM_SMARTPTR_TYPEDEF(IPinConnection, IID_IPinConnection);
#pragma warning(pop)
#include "litiv/3rdparty/dshowbase/gmf/smartPtr.h"
#include "litiv/3rdparty/dshowbase/gmf/bridge.h"
#include <dvdmedia.h>

#ifndef INVALID_FILE_ATTRIBUTES
    #define INVALID_FILE_ATTRIBUTES ((DWORD)-1)
#endif

// logging support
#include <ShlObj.h>
Logger theLogger(TEXT("GMFBridge.txt"));
Logger::Logger(const TCHAR* pFile)
: m_hFile(NULL)
{
    // to turn this on, create the file c:\GMFBridge.txt

	TCHAR szPath[MAX_PATH];
	if (SUCCEEDED(SHGetFolderPath(NULL, CSIDL_PERSONAL, NULL, 0, szPath)))
	{
		StringCbCat(szPath, sizeof(szPath), TEXT("\\"));
		StringCbCat(szPath, sizeof(szPath), pFile);
		if (GetFileAttributes(szPath) != INVALID_FILE_ATTRIBUTES)
		{
			m_hFile = CreateFile(szPath, GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, 0, NULL);
			if (m_hFile == INVALID_HANDLE_VALUE)
			{
				m_hFile = NULL;
			}
		}
	}

	if (m_hFile == NULL)
	{
		StringCbCopy(szPath, sizeof(szPath), TEXT("c:\\"));
		StringCbCat(szPath, sizeof(szPath), pFile);
		if (GetFileAttributes(szPath) != INVALID_FILE_ATTRIBUTES)
		{
			m_hFile = CreateFile(szPath, GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, 0, NULL);
		}
	}
	if (m_hFile == INVALID_HANDLE_VALUE)
	{
		m_hFile = NULL;
	}
	if (m_hFile != NULL)
	{
		SetFilePointer(m_hFile, 0, NULL, FILE_END);
        m_msBase = timeGetTime();

        SYSTEMTIME st;
        GetLocalTime(&st);
        Log(TEXT("Started %04d-%02d-%02d %02d:%02d:%02d"),
            st.wYear,
            st.wMonth,
            st.wDay,
            st.wHour,
            st.wMinute,
            st.wSecond);
    }
}

Logger::~Logger()
{
    if (m_hFile != NULL)
    {
        CloseHandle(m_hFile);
    }
}

void
Logger::Log(const TCHAR* pFormat, ...)
{
    if (m_hFile != NULL)
    {
        va_list va;
        va_start(va, pFormat);
        TCHAR  ach[4096];
        int cchTime = wsprintf(ach, TEXT("%d:\t"), timeGetTime() - m_msBase);
        int cch = cchTime + wvsprintf(ach+cchTime, pFormat, va);
        va_end(va);

        // debug output without newline and without time (added by existing debug code)
		_bstr_t str = ach+cchTime;
        DbgLog((LOG_TRACE, 0, "%s", (char*)str));

        // add time at start and newline at end for file output
        ach[cch++] = TEXT('\r');
        ach[cch++] = TEXT('\n');

        CAutoLock lock(&m_csLog);
        DWORD cActual;
        WriteFile(m_hFile, ach, cch * sizeof(TCHAR), &cActual, NULL);
    }
}



BridgeController::BridgeController(LPUNKNOWN pUnk)
: CUnknown(NAME("BridgeController"), pUnk),
  m_hwnd(NULL),
  m_NotifyMsg(0),
  m_nMillisecs(0),
  m_bLiveTiming(false)
{
	HRESULT hr = S_OK;
	m_pContainer = new BridgeEvents(GetOwner(), &hr);

    // create hidden window for marshalling
    WNDCLASS cls;
    ZeroMemory(&cls,  sizeof(cls));
    cls.cbWndExtra = sizeof(DWORD_PTR);
    cls.hInstance = g_hInst;
    cls.lpfnWndProc = DispatchWndProc;
    cls.lpszClassName = TEXT("GMFBridgeNotify");
    RegisterClass(&cls);

    m_hwndNotify = CreateWindow(TEXT("GMFBridgeNotify"), NULL,  0, 0, 0, 0, 0, NULL, NULL, g_hInst, (LPVOID)this);
}

BridgeController::~BridgeController()
{
    // make sure no pins are held when we release the filters/graphs
    BridgeGraphs(NULL, NULL);

    // destroy the window
    if (IsWindow(m_hwndNotify))
    {
        DestroyWindow(m_hwndNotify);
        UnregisterClass(TEXT("GMFBridgeNotify"), g_hInst);
    }
}

//static
CUnknown* WINAPI
BridgeController::CreateInstance(LPUNKNOWN pUnk, HRESULT* phr)
{
	UNREFERENCED_PARAMETER(phr);
    return new BridgeController(pUnk);
}

STDMETHODIMP
BridgeController::NonDelegatingQueryInterface(REFIID iid, void**ppv)
{
    if ((iid == __uuidof(IGMFBridgeController)) ||
         (iid == __uuidof(IGMFBridgeController2)) ||
         (iid == __uuidof(IGMFBridgeController3)))
    {
        return GetInterface((IGMFBridgeController3*)this, ppv);
	} else if (iid == IID_IConnectionPointContainer)
	{
		return m_pContainer->NonDelegatingQueryInterface(iid, ppv);
    } else
    {
        return CUnknown::NonDelegatingQueryInterface(iid, ppv);
    }
}

//static
LRESULT CALLBACK
BridgeController::DispatchWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    BridgeController* pThis = NULL;
    if (msg == WM_CREATE)
    {
        CREATESTRUCT* pCS = reinterpret_cast<CREATESTRUCT*>(lParam);
        pThis = reinterpret_cast<BridgeController*>(pCS->lpCreateParams);
        SetWindowLongPtr(hwnd, 0, (LONG_PTR)pThis);
    }
    else
    {
        pThis = reinterpret_cast<BridgeController*>(GetWindowLongPtr(hwnd, 0));
    }
    LRESULT r = 0;
    if (pThis != NULL)
    {
        r = pThis->OnMessage(hwnd, msg, wParam, lParam);
    }
    else
    {
        r = DefWindowProc(hwnd, msg, wParam, lParam);
    }

    return r;
}

LRESULT
BridgeController::OnMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_USER)
    {
		// now on app creation thread, fire event to VB or other client
		m_pContainer->Fire_OnSegmentEnd();
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

STDMETHODIMP
BridgeController::SetNotify(LONG_PTR hwnd, long msg)
{
    CAutoLock lock(&m_csBridge);

    m_hwnd = (HWND)hwnd;
    m_NotifyMsg = (UINT)msg;

    LOG((TEXT("Notify window 0x%x"), hwnd));
    return S_OK;
}

STDMETHODIMP
BridgeController::SetBufferMinimum(long nMillisecs)
{
    CAutoLock lock(&m_csBridge);
    m_nMillisecs = nMillisecs;
    LOG((TEXT("Buffer minimum %d"), m_nMillisecs));
    return S_OK;
}


STDMETHODIMP
BridgeController::AddStream(BOOL bVideo, eFormatType AllowedTypes, BOOL bDiscardUnconnected)
{
    CAutoLock lock(&m_csBridge);

    m_Streams.push_back(BridgeStream(this, bVideo, AllowedTypes, bDiscardUnconnected));

    LOG((TEXT("Added stream %d: %s, %s, %s"),
         StreamCount(),
         bVideo?TEXT("video"):TEXT("audio"),
         (AllowedTypes == eMuxInputs)?TEXT("Mux Inputs"):(AllowedTypes == eUncompressed) ? TEXT("Decompressed Only") : TEXT("Any types"),
         bDiscardUnconnected?TEXT("Discard mode"):TEXT("Suspend mode")));

    return S_OK;
}

STDMETHODIMP
BridgeController::InsertSinkFilter(IUnknown* pUnkGraph, IUnknown** ppFilter)
{
    CAutoLock lock(&m_csBridge);

    if (StreamCount() == 0)
    {
        return E_FAIL;
    }

    IGraphBuilderPtr pGraph = pUnkGraph;
    if (pGraph == NULL)
    {
        return E_INVALIDARG;
    }

    // construct the filter and add to graph
    IBridgeSinkPtr pSink = new BridgeSink(this);
    IBaseFilterPtr pfSink = pSink;
    HRESULT hr = pGraph->AddFilter(pfSink, L"Bridge Sink filter");
	if (FAILED(hr))
	{
		return hr;
	}

	LOG((TEXT("Sink filter 0x%x in graph 0x%x"), pSink, pGraph));

    // return the filter as an IBaseFilter interface. For a more serious
    // VB developer, we might want to return an IFilterInfo -- this would mean
    // enumerating the graph's FilterCollection to find the one that
    // corresponds to this IBaseFilter*.
	// Since this is an automatable interface we have to return the corresponding IUnknown
    *ppFilter = pSink.Detach();
	return hr;
}

STDMETHODIMP
BridgeController::InsertSourceFilter(IUnknown* pUnkSourceGraphSinkFilter, IUnknown* pUnkRenderGraph, IUnknown** ppFilter)
{
    CAutoLock lock(&m_csBridge);

    if (StreamCount() == 0)
    {
        return E_FAIL;
    }

    HRESULT hr = S_OK;
    // make sure that we have a sink graph to pass negotiations and type queries up to
//    IGraphBuilderPtr pSink = pUnkSourceGraph;
    if (m_pCurrentSourceGraphSinkFilter == NULL)
    {
        m_pCurrentSourceGraphSinkFilter = pUnkSourceGraphSinkFilter;
		if (m_pCurrentSourceGraphSinkFilter == NULL)
		{
			// this is not an IBridgeSink filter
			return E_INVALIDARG;
		}

        for (int i = 0; i < StreamCount(); i++)
        {
			BridgeSinkInput* pPin;
			if (SUCCEEDED(m_pCurrentSourceGraphSinkFilter->GetBridgePin(i, &pPin)))
			{
	            m_Streams[i].Bridge(pPin, NULL);
			}
        }
    }

    IGraphBuilderPtr pGraph = pUnkRenderGraph;
    if (pGraph == NULL)
    {
        return E_INVALIDARG;
    }

    // type of input is now fixed except for dynamic type changes
    for (int i = 0; i < StreamCount(); i++)
    {
        m_Streams[i].TypeFixed();
    }

    // construct the filter and add to graph
    IBridgeSourcePtr pSource = new BridgeSource(this);
    IBaseFilterPtr pfSource = pSource;
    hr = pGraph->AddFilter(pfSource, L"Bridge Source filter");
	if (FAILED(hr))
	{
		return hr;
	}
    LOG((TEXT("Source filter 0x%x in graph 0x%x"), pSource, pGraph));

    // return the filter as an IBaseFilter interface. For a more serious
    // VB developer, we might want to return an IFilterInfo -- this would mean
    // enumerating the graph's FilterCollection to find the one that
    // corresponds to this IBaseFilter*.
	// Since this is an automatable interface we have to return the corresponding IUnknown

    *ppFilter = pSource.Detach();
    return hr;
}

HRESULT
BridgeController::FindUnconnectedPin(IBaseFilter* pFilter, IPin** ppPin, PIN_DIRECTION dir, const GUID* pMajorType)
{
    IEnumPinsPtr pEnum;
    pFilter->EnumPins(&pEnum);
    IPinPtr pPin;
    while (pEnum->Next(1, &pPin, NULL) == S_OK)
    {
        IPinPtr pPeer;
        if (pPin->ConnectedTo(&pPeer) != S_OK)
        {
            // not connected
            PIN_DIRECTION dirThis;
            pPin->QueryDirection(&dirThis);
            if (dir == dirThis)
            {
                if ((pMajorType == NULL) ||
                    CanSupplyType(pPin, pMajorType))
                {
                    *ppPin = pPin.Detach();
                    return S_OK;
                }
            }
        }
    }

    return E_INVALIDARG;
}

bool
BridgeController::CanSupplyType(IPin* pPin, const GUID* pMajorType)
{
    IEnumMediaTypesPtr pEnum;
    pPin->EnumMediaTypes(&pEnum);
    AM_MEDIA_TYPE* pmt;
    while (pEnum->Next(1, &pmt, NULL) == S_OK)
    {
        CMediaType mt(*pmt);
        DeleteMediaType(pmt);
        if (*mt.Type() == *pMajorType)
        {
            return true;
        }
    }
    return false;
}

// find the unconnected output pin for a given type, which may be downstream of here
HRESULT
BridgeController::FindStreamSource(IBaseFilter* pFilter, const GUID* pMajorType, IPin** ppPin)
{
    // is it on this filter?
    HRESULT hr = FindUnconnectedPin(pFilter, ppPin, PINDIR_OUTPUT, pMajorType);
    if (hr == S_OK)
    {
        return hr;
    }

    // recurse down all connected pins
    IEnumPinsPtr pEnum;
    pFilter->EnumPins(&pEnum);
    IPinPtr pPin;
    while (pEnum->Next(1, &pPin, NULL) == S_OK)
    {
        // is this output and connected?
        PIN_DIRECTION dirThis;
        pPin->QueryDirection(&dirThis);

        IPinPtr pPeer;
        if ((dirThis == PINDIR_OUTPUT) && (pPin->ConnectedTo(&pPeer) == S_OK))
        {
            // traverse to downstream filter
            PIN_INFO pi;
            pPeer->QueryPinInfo(&pi);
            IBaseFilterPtr pfPeer(pi.pFilter, 0);

            // recurse to find source on this filter
            hr = FindStreamSource(pfPeer, pMajorType, ppPin);
            if (hr != S_FALSE)
            {
                // fail if error, or success - pin found
                return hr;
            }
        }
    }
    // not on this branch
    return S_FALSE;
}

STDMETHODIMP
BridgeController::CreateSourceGraph(BSTR strFile, IUnknown* pUnkGraph, IUnknown **pSinkFilter)
{
    CAutoLock lock(&m_csBridge);

    // add the sink filter first
    IUnknownPtr pUnkSink;
    HRESULT hr = InsertSinkFilter(pUnkGraph, &pUnkSink);
    IBridgeSinkPtr pSink = pUnkSink;
    if (FAILED(hr) || (pSink == NULL))
    {
        return hr;
    }

    IGraphBuilderPtr pGraph = pUnkGraph;
    if (pGraph == NULL)
    {
        return E_INVALIDARG;
    }

    // render using AddSourceFilter and Connect, not Render so that
    // we don't get unwanted renderers for streams that we are not using

    IBaseFilterPtr pFile;
    hr = pGraph->AddSourceFilter(strFile, strFile, &pFile);
    if (FAILED(hr))
    {
        return hr;
    }

    for (int n = 0; n < StreamCount(); n++)
    {
        const GUID* pElemType;
        if (m_Streams[n].IsVideo())
        {
            pElemType = &MEDIATYPE_Video;
        } else
        {
            pElemType = &MEDIATYPE_Audio;
        }

        IPinPtr pOut;
        bool bPinIsStream = true;
        if (n == 0)
        {
            // start with source filter for first pin
            // -- expect the source to expose a muxed type
            hr = FindUnconnectedPin(pFile, &pOut, PINDIR_OUTPUT, &MEDIATYPE_Stream);
            if (FAILED(hr))
            {
                // try unmuxed type
                bPinIsStream = false;
                hr = FindUnconnectedPin(pFile, &pOut, PINDIR_OUTPUT, pElemType);
                if (FAILED(hr))
                {
                    return hr;
                }
            }
        } else
        {
            // for subsequent pins, track downstream to find the unconnected
            // output (probably on splitter)
            bPinIsStream = false;
            hr = FindStreamSource(pFile, pElemType, &pOut);
            if (hr != S_OK)
            {
                return VFW_E_UNSUPPORTED_AUDIO;
            }
        }
        BridgeSinkInput* pPin;
        hr = pSink->GetBridgePin(n, &pPin);
        if (SUCCEEDED(hr))
        {
            hr = pGraph->Connect(pOut, pPin);
            if (FAILED(hr))
            {
                hr = pGraph->Render(pOut);
                // if we've used render on the stream pin, we've done all the elementary streams
                // at the same time
                if (SUCCEEDED(hr) && bPinIsStream)
                {
                    break;
                }
            }
        }
        if (FAILED(hr))
        {
            return hr;
        }
    }


    // check all pins were connected
    for (int n = 0; n < StreamCount(); n++)
    {
        BridgeSinkInput* pPin;
        hr = pSink->GetBridgePin(n, &pPin);
        if (SUCCEEDED(hr))
        {
            if (!pPin->IsConnected())
            {
                return E_INVALIDARG;
            }
        }
    }
	*pSinkFilter = pUnkSink.Detach();
    return S_OK;
}

STDMETHODIMP
BridgeController::CreateRenderGraph(IUnknown* pSourceGraphSinkFilter, IUnknown* pRenderGraph, IUnknown** pRenderGraphSourceFilter)
{
    CAutoLock lock(&m_csBridge);

    // add the source filter first
    IUnknownPtr pUnkSource;
    HRESULT hr = InsertSourceFilter(pSourceGraphSinkFilter, pRenderGraph, &pUnkSource);
    IBridgeSourcePtr pSource = pUnkSource;
    if (FAILED(hr) || (pSource == NULL))
    {
        return hr;
    }

    IGraphBuilderPtr pGraph = pRenderGraph;
    if (pGraph == NULL)
    {
        return E_INVALIDARG;
    }

    // render the enabled pins
    for (int n = 0; n < StreamCount(); n++)
    {
        BridgeSourceOutput* pPin;
        pSource->GetBridgePin(n, &pPin);
        hr = pGraph->Render(pPin);
        if (FAILED(hr))
        {
            break;
        }
    }
	if (SUCCEEDED(hr))
	{
		*pRenderGraphSourceFilter = pUnkSource.Detach();
	}
    return hr;
}


STDMETHODIMP
BridgeController::BridgeGraphs(IUnknown* pSourceGraphSinkFilter, IUnknown* pRenderGraphSourceFilter)
{
    return BridgeAtDiscont(pSourceGraphSinkFilter, pRenderGraphSourceFilter, false);
}

STDMETHODIMP
BridgeController::BridgeAtDiscont(IUnknown* pSourceGraphSinkFilter, IUnknown* pRenderGraphSourceFilter, BOOL bIsDiscontinuity)
{
    CAutoLock lock(&m_csBridge);

    HRESULT hr = S_OK;

    LOG((TEXT("Bridging 0x%x to 0x%x"), pSourceGraphSinkFilter, pRenderGraphSourceFilter));

    // neither changed -- nothing to do
    if ((pSourceGraphSinkFilter == m_pCurrentSourceGraphSinkFilter) &&
        (pRenderGraphSourceFilter == m_pCurrentRenderGraphSourceFilter))
    {
        return S_OK;
    }

    if (m_pCurrentSourceGraphSinkFilter)
    {
        if (m_pCurrentRenderGraphSourceFilter == NULL)
        {
            // source filter reference is held for
            // render graph building -- clear this
            for (int n = 0; n < StreamCount(); n++)
            {
                m_Streams[n].DisconnectBridge();
            }
        } else {
            // actual connection current -- disconnect
            // we can't disconnect until we know that the
            // delivery threads are out of the downstream graph.
            // If the graph is paused, the threads could be blocked downstream
            // and we need to flush to unblock. Unnecessary flushing when
            // running however would cause glitches.
            // We flush the sink filter, which will pass it downstream.
            IGraphBuilderPtr pRenderGraph;
            HRESULT hr = GraphFromFilter(m_pCurrentRenderGraphSourceFilter, &pRenderGraph);
            if (FAILED(hr))
            {
                return hr;
            }
            IMediaControlPtr pMC = pRenderGraph;
            FILTER_STATE fs;
            pMC->GetState(0, (OAFilterState*)&fs);
            if (fs == State_Paused)
            {
				// we can safely avoid flushing if the sink is at EOS and thus guaranteed to be
				// not in the downstream graph
				if (m_pCurrentSourceGraphSinkFilter->IsAtEOS())
				{
					// prevent endflush
					fs = State_Running;
				}
				else
				{
					for (int n = 0; n < m_pCurrentSourceGraphSinkFilter->GetBridgePinCount(); n++)
					{
						BridgeSinkInput* pPin;
						m_pCurrentSourceGraphSinkFilter->GetBridgePin(n, &pPin);
						pPin->BeginFlush();
					}
				}
            }

            for (int n = 0; n < StreamCount(); n++)
            {
                m_Streams[n].DisconnectBridge();
            }

            if ((fs == State_Paused) && (m_pCurrentSourceGraphSinkFilter != NULL))
            {
                // need to EndFlush -- since we are disconnected, this will
                // only apply to the sink filter in the source graph, but
                // the sink pin will signal EndFlush downstream automatically if in BeginFlush
                // at disconnect
                for (int n = 0; n < m_pCurrentSourceGraphSinkFilter->GetBridgePinCount(); n++)
                {
                    BridgeSinkInput* pPin;
                    m_pCurrentSourceGraphSinkFilter->GetBridgePin(n, &pPin);
                    pPin->EndFlush();
                }
            }
        }
        m_pCurrentSourceGraphSinkFilter = NULL;
        m_pCurrentRenderGraphSourceFilter = NULL;
    }

    // if we are not given both filters, then
    // we need do nothing
    IBridgeSinkPtr pSink = pSourceGraphSinkFilter;
    IBridgeSourcePtr pSource = pRenderGraphSourceFilter;
    if ((pSink != NULL) && (pSource != NULL))
    {
        // make a connection between these two:
        // inform source graph of the target output

        // signal new connection, for time adjustments
        pSource->OnNewConnection(bIsDiscontinuity);

        // check that pin/stream counts match
        if ( (StreamCount() != pSource->GetBridgePinCount()) ||
             (StreamCount() != pSink->GetBridgePinCount()))
        {
            return E_INVALIDARG;
        }
		int n;
        for (n = 0; n < StreamCount(); n++)
        {
            BridgeSinkInput* pIn;
            pSink->GetBridgePin(n, &pIn);
            BridgeSourceOutput* pOut;
            pSource->GetBridgePin(n, &pOut);
            hr = m_Streams[n].Bridge(pIn, pOut);
            if (FAILED(hr))
            {
                break;
            }
        }
        if (FAILED(hr))
        {
            // undo any connections made prior to the error
            for (int m = 0; m < n; m++)
            {
                m_Streams[m].DisconnectBridge();
            }
        }
        m_pCurrentSourceGraphSinkFilter = pSourceGraphSinkFilter;
        m_pCurrentRenderGraphSourceFilter = pRenderGraphSourceFilter;
    }
    return hr;
}

STDMETHODIMP
BridgeController::GetSegmentTime(double* pdSeconds)
{
    // find stream time on render graph and subtract the
    // baseline line to get the time within current segment
    REFERENCE_TIME tNow = 0;
	if (m_pCurrentRenderGraphSourceFilter != NULL)
	{
        m_pCurrentRenderGraphSourceFilter->GetSegmentTime(&tNow);

	}
    // return as double seconds for VB compatibility
    *pdSeconds = double(tNow) / UNITS;
    return S_OK;
}


STDMETHODIMP
BridgeController::GetSegmentOffset(double* pdOffset)
{
    // find the offset in render-graph stream time to the
    // start of the current segment
    REFERENCE_TIME tOffset = 0;
	if (m_pCurrentRenderGraphSourceFilter != NULL)
	{
        m_pCurrentRenderGraphSourceFilter->GetSegmentOffset(&tOffset);

	}
    // return as double seconds for VB compatibility
    *pdOffset = double(tOffset) / UNITS;
    return S_OK;
}

STDMETHODIMP
BridgeController::NoMoreSegments()
{
    // app has been notified of EOS and there
    // are no more segments to connect, so
    // pass EOS downstream into render graph
	IGraphBuilderPtr pGraph;
    if (FAILED(GraphFromFilter(m_pCurrentRenderGraphSourceFilter, &pGraph)))
    {
        return E_NOINTERFACE;
    }

    IMediaControlPtr pMC = pGraph;
    FILTER_STATE fs;
    pMC->GetState(0, (OAFilterState*)&fs);
    if (fs != State_Stopped)
    {
        for (int n = 0; n < m_pCurrentRenderGraphSourceFilter->GetBridgePinCount(); n++)
        {
            BridgeSourceOutput* pPin;
            m_pCurrentRenderGraphSourceFilter->GetBridgePin(n, &pPin);
            pPin->DeliverEndOfStream();
        }
    }
    return S_OK;
}

HRESULT
BridgeController::GraphFromFilter(IUnknown *pFilter, IGraphBuilder** ppGraph)
{
	HRESULT hr = E_FAIL;
	IBaseFilterPtr pf = pFilter;
	if (pf == NULL)
	{
		return E_NOINTERFACE;
	}
	FILTER_INFO info;
	hr = pf->QueryFilterInfo(&info);
	if (FAILED(hr) || (info.pGraph == NULL))
	{
		return E_FAIL;
	}

    IGraphBuilderPtr pGraph = info.pGraph;
    info.pGraph->Release();

	*ppGraph = pGraph.Detach();
	return S_OK;
}


// ---- BridgeStream implementation -------------------------

BridgeStream::BridgeStream(BridgeController* pController, BOOL bVideo, eFormatType AllowedTypes, BOOL bDiscard)
: m_pController(pController),
  m_pInputPin(NULL),
  m_pOutputPin(NULL),
  m_bTypeFixed(false),
  m_bVideo(bVideo),
  m_AllowedTypes(AllowedTypes),
  m_bDiscard(bDiscard),
  m_bReceiveConnectAware(false),
  m_bBufferSizeLocked(false)
{
}

BridgeStream::~BridgeStream()
{
    DisconnectBridge();
}

BridgeStream::BridgeStream(const BridgeStream& r)
{
    m_pController = r.m_pController;
    m_bVideo = r.m_bVideo;
    m_AllowedTypes = r.m_AllowedTypes;
    m_bDiscard = r.m_bDiscard;
    m_bTypeFixed = r.m_bTypeFixed;
    m_mt = r.m_mt;
    m_bBufferSizeLocked = r.m_bBufferSizeLocked;
    m_cBuffers = r.m_cBuffers;
    m_cBytes = r.m_cBytes;
    m_pInputPin = r.m_pInputPin;
    m_pOutputPin = r.m_pOutputPin;
}

const BridgeStream&
BridgeStream::operator= (const BridgeStream& r)
{
    ASSERT(m_pInputPin == NULL);
    m_pController = r.m_pController;
    m_bVideo = r.m_bVideo;
    m_AllowedTypes = r.m_AllowedTypes;
    m_bDiscard = r.m_bDiscard;
    m_bTypeFixed = r.m_bTypeFixed;
    m_mt = r.m_mt;
    m_bBufferSizeLocked = r.m_bBufferSizeLocked;
    m_cBuffers = r.m_cBuffers;
    m_cBytes = r.m_cBytes;
    m_pInputPin = r.m_pInputPin;
    m_pOutputPin = r.m_pOutputPin;

    return *this;
}

HRESULT
BridgeStream::Bridge(BridgeSinkInput* pInputPin, BridgeSourceOutput* pOutputPin)
{
    DisconnectBridge();

    SemLock sem(pInputPin->DeliveryLock());

    if (pOutputPin == NULL)
    {
        // during graph building-- just remember the input pin so
        // we can pass calls appropriately
        m_pInputPin = pInputPin;
        return S_OK;
    }

    HRESULT hr = S_OK;
    IMemAllocatorPtr pAlloc;
    hr = pOutputPin->GetConnectionAllocator(&pAlloc);
    if (FAILED(hr))
    {
        return hr;
    }

    // need to set these before we call the input pin, so it
    // can refer downstream as necessary during the connection
    m_pOutputPin = pOutputPin;
    m_pInputPin = pInputPin;

    m_pOutputPin->SetStream(this);
    m_pInputPin->SetStream(this);

    hr = pInputPin->MakeBridge(pAlloc);

    if (FAILED(hr))
    {
        m_pOutputPin = NULL;
        m_pInputPin = NULL;
    }
    return hr;
}

HRESULT
BridgeStream::DisconnectBridge()
{

    if (m_pInputPin)
    {
        // must get semaphore before attempting to disconnect
        SemLock sem(m_pInputPin->DeliveryLock());
        m_pInputPin->DisconnectBridge();
    }
    m_pInputPin = NULL;
    m_pOutputPin = NULL;
    return S_OK;
}

HRESULT
BridgeStream::CanReceiveType(const CMediaType* pmt)
{
    CAutoLock lock(&m_csType);

    HRESULT hr = S_OK;
    if (m_bTypeFixed)
    {
        // we must ensure that the output pin can deliver the type we are
        // currently using. However, some filters will accept dynamic type changes to
        // formats that they will not initially connect with, so instead
        // of insisting here that *pmt == m_mt, we check that
        // the output pin can accept m_mt, and then arrange a dynamic type change
        // when this source becomes active.
        // In Discard mode, dynamic type changes are not possible, so both problem and
        // solution do not apply.

        if (*pmt != m_mt)
        {
            if (DiscardMode())
            {
                hr = VFW_E_TYPE_NOT_ACCEPTED;
            } else
            {
                // We are using a different type -- if the source can deliver the
                // type we are using now, we will switch the source at his first GetBuffer
                hr = CanDeliverType(&m_mt);
                if (hr == S_OK)
                {
                    // sadly, many codecs do not check the size
                    // they are offered, and will claim to output anything
                    hr = CheckMismatchedVideo(pmt, &m_mt);
                }

                // it seems that the colour space converter returns S_OK to a
                // QueryAccept but does not correctly switch sometimes. We should never accept a
                // connection that is a different subtype type to the main
                if ((*pmt->Type() == MEDIATYPE_Video) && (*pmt->Subtype() != *m_mt.Subtype()))
                {
                   hr = VFW_E_TYPE_NOT_ACCEPTED;
                }

                if (hr != S_OK)
                {
                    // new clip is not same format.
                    // we accept video dimension changes, and
                    // for audio, we accept any PCM format

                    if ((*pmt->Type() == MEDIATYPE_Audio) &&
                        (*pmt->FormatType() == FORMAT_WaveFormatEx))
                    {
                        WAVEFORMATEX* pwfx = (WAVEFORMATEX*) pmt->Format();
                        if (pwfx->wFormatTag == WAVE_FORMAT_PCM)
                        {
                            hr = S_OK;
                        }
                    } else if ((*pmt->Type() == MEDIATYPE_Video) && m_bReceiveConnectAware)
                    {
                        // for video, you can't dyn-switch by attaching a mt to the buffer,
                        // because you must reconfigure the buffer sizes, and the allocator is owned
                        // by the renderer. You can get round this by calling ReceiveConnection on the
                        // renderer's input pin (without disconnecting first).
                        //
                        // We believe that the render graph supports this for this stream. However
                        // we don't have access to the renderer right now, and we cannot be sure that
                        // there is a common type between source and renderer (eg videoinfo vs videoinfo2,
                        // acceptable pixel formats etc.
                        //
                        // If the source can produce the subtype we are using now, then it's probably ok.
                        if (*pmt->Subtype() == *m_mt.Subtype())
                        {
                            hr = S_OK;
                        }
                    }
                }
            }
        }
    }
    return hr;
}

// check basic video dimensions from media type
//static
HRESULT
BridgeStream::GetVideoDimensions(const CMediaType* pmt, long *pcx, long* pcy)
{
    RECT* prc;
    BITMAPINFOHEADER* pbmi;

    if (*pmt->FormatType() == FORMAT_VideoInfo)
    {
        VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)pmt->Format();
        prc = &pvi->rcSource;
        pbmi = &pvi->bmiHeader;
    } else if (*pmt->FormatType() == FORMAT_VideoInfo2)
    {
        VIDEOINFOHEADER2* pvi2 = (VIDEOINFOHEADER2*)pmt->Format();
        prc = &pvi2->rcSource;
        pbmi = &pvi2->bmiHeader;
    } else
    {
        return E_FAIL;
    }
    if (IsRectEmpty(prc))
    {
        *pcx = pbmi->biWidth;
        *pcy = abs(pbmi->biHeight);
    } else
    {
        *pcx = prc->right - prc->left;
        *pcy = prc->bottom - prc->top;
    }
    return S_OK;
}

// check if basic dimensions are correct
//static
HRESULT
BridgeStream::CheckMismatchedVideo(const CMediaType* pmt1, const CMediaType* pmt2)
{
    long cx1, cy1;
    GetVideoDimensions(pmt1, &cx1, &cy1);
    long cx2, cy2;
    GetVideoDimensions(pmt2, &cx2,  &cy2);
    if ((cx1 != cx2) || (cy1 != cy2))
    {
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    return S_OK;
}

bool
BridgeStream::CanSwitchTo(const CMediaType* pmt)
{
    if ((*pmt->Type() == *m_mt.Type()) &&
        (*pmt->Type() == MEDIATYPE_Video))
    {
        // if we are connected directly to one of the video renderers,
        // we can change dimensions with ReceiveConnection
        if ((m_pOutputPin != NULL) &&
            m_bReceiveConnectAware &&
            m_pOutputPin->CanDeliver(pmt))
        {
            return true;
        }
    }
    return false;
}

HRESULT
BridgeStream::SwitchTo(const CMediaType* pmt)
{
    HRESULT hr = m_pOutputPin->SwitchTo(pmt);
    return hr;
}

HRESULT
BridgeStream::GetSelectedType(CMediaType* pmt)
{
    CAutoLock lock(&m_csType);

    *pmt = m_mt;
    return S_OK;
}

HRESULT
BridgeStream::SetSelectedType(const CMediaType* pmt)
{
    CAutoLock lock(&m_csType);

    m_mt = *pmt;
    return S_OK;
}


HRESULT
BridgeStream::Deliver(IMediaSample* pSample)
{
    HRESULT hr = S_OK;
    if (m_pOutputPin)
    {
        hr = m_pOutputPin->Send(pSample);
    }
    return hr;
}

HRESULT
BridgeStream::BeginFlush()
{
    // protected by sink pin critsec
    HRESULT hr = S_OK;
    if (m_pOutputPin)
    {
        hr = m_pOutputPin->DeliverBeginFlush();
    }
    return hr;
}

HRESULT
BridgeStream::EndFlush()
{
    // protected by sink pin critsec
    HRESULT hr = S_OK;
    if (m_pOutputPin)
    {
        hr = m_pOutputPin->DeliverEndFlush();
    }
    return hr;
}

void
BridgeStream::ResetOnStop()
{
	if (m_pOutputPin)
	{
		m_pOutputPin->ResetOnStop();
	}
}

bool
BridgeStream::GetDownstreamBufferProps(long* pcBuffers, long* pcBytes)
{
    if (m_bBufferSizeLocked)
    {
        *pcBuffers = m_cBuffers;
        *pcBytes = m_cBytes;
    }
    return m_bBufferSizeLocked;
}

HRESULT
BridgeStream::GetBufferProps(long* pcBuffers, long* pcBytes)
{
    // the downstream connection wants to use the same
    // buffer configuration. We can get this from the upstream connection's
    // agreed allocator (whether the bridge allocator or a standard one).
    if (!m_pInputPin)
    {
        return E_FAIL;
    }
    HRESULT hr = S_OK;
    if (!m_bBufferSizeLocked)
    {
        hr =  m_pInputPin->GetBufferProps(&m_cBuffers, &m_cBytes);
        m_bBufferSizeLocked = true;
    }
    if (SUCCEEDED(hr))
    {
        *pcBuffers = m_cBuffers;
        *pcBytes = m_cBytes;
    }
    return hr;
}

HRESULT
BridgeStream::CanDeliverType(const CMediaType* pmt)
{
    // in "discard" mode, we cannot generate type changes, so
    // we therefore can only accept the type already
    // agreed upstream
    HRESULT hr = S_OK;
    if (!m_pInputPin)
    {
        hr = E_FAIL;
    } else if (DiscardMode())
    {
        if (*pmt != m_mt)
        {
            hr = VFW_E_TYPE_NOT_ACCEPTED;
        }
    } else
    {
        hr = m_pInputPin->CanDeliverType(pmt);
    }
    return hr;
}

HRESULT
BridgeStream::EnumOutputType(int iPosition, CMediaType* pmt)
{
    if (!m_pInputPin || !m_pInputPin->IsConnected())
    {
        return VFW_S_NO_MORE_ITEMS;
    }
    // first offer the type we are fixed with
    if (iPosition == 0)
    {
        *pmt = m_mt;
        return S_OK;
    }
    return m_pInputPin->EnumOutputType(iPosition-1, pmt);
}

HRESULT
BridgeStream::NotifyQuality(IBaseFilter* pSender, Quality q)
{
    HRESULT hr = E_NOTIMPL;
    if (m_pInputPin && m_pInputPin->IsConnected())
    {
        IQualityControlPtr pQSink = m_pInputPin->GetConnected();
        if (pQSink != NULL)
        {
            hr = pQSink->Notify(pSender, q);

            LOG((TEXT("Quality passed upstream: late %d ms at %d ms"),
                 LONG(q.Late / 10000),
                 LONG(q.TimeStamp / 10000)));
        }
    }
    return hr;
}

// --- outgoing event support ---------------------

BridgeEvents::BridgeEvents(IUnknown* pUnk, HRESULT* phr)
: CPContainer(pUnk, phr)
{
	AddCP(__uuidof(_IGMFBridgeEvents));
}

void
BridgeEvents::Fire_OnSegmentEnd()
{
	iterator pt = find(__uuidof(_IGMFBridgeEvents));
    if (pt != end())
	{
        ConnectionPoint::iterator conn = pt->second->begin();
        while(conn != pt->second->end())
		{
            IDispatchPtr pDisp = conn->second;
            if (pDisp != NULL)
			{
                _variant_t vResult;
                DISPPARAMS dp = {NULL, NULL, 0, 0};
                pDisp->Invoke(1, IID_NULL, LOCALE_USER_DEFAULT, DISPATCH_METHOD, &dp, &vResult, NULL, NULL);
            }

            conn++;
        }
    }
}

bool
BridgeEvents::hasClients()
{
	iterator pt = find(__uuidof(_IGMFBridgeEvents));
    if (pt != end())
	{
        ConnectionPoint::iterator conn = pt->second->begin();
        if (conn != pt->second->end())
		{
			return true;
		}
	}
	return false;
}
