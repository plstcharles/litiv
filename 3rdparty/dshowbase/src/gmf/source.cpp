//
// GDCL Multigraph Framework
//
// Source.cpp: implementation of source filter and output pin
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
#include "litiv/3rdparty/dshowbase/gmf/source.h"

// for VIDEOINFO2
#include <dvdmedia.h>
#include <sstream>

BridgeSource::BridgeSource(BridgeController* pController)
: m_tBase(0),
  m_dwROT(0),
  CBaseFilter(NAME("BridgeSource filter"), NULL, &m_csFilter, GUID_NULL)
{
	HRESULT hr = S_OK;
    m_nPins = pController->StreamCount();
    m_pPins = new smart_ptr<BridgeSourceOutput>[m_nPins];
    for (int n = 0; n < m_nPins; n++)
    {
		ostringstream strm;
		strm << "Output " << (n+1);
        _bstr_t strName = strm.str().c_str();

        m_pPins[n] = new BridgeSourceOutput(this, pController->GetStream(n), m_pLock, &hr, strName);
	}
	LOG((TEXT("Source 0x%x has %d pins"), this, m_nPins));
}

STDMETHODIMP
BridgeSource::NonDelegatingQueryInterface(REFIID iid, void** ppv)
{
	if (iid == __uuidof(IBridgeSource))
	{
		return GetInterface((IBridgeSource*)this, ppv);
	}
	return CBaseFilter::NonDelegatingQueryInterface(iid, ppv);
}

int
BridgeSource::GetPinCount()
{
    return m_nPins;
}

CBasePin*
BridgeSource::GetPin(int n)
{
    if ((n < 0) || (n >= m_nPins))
    {
        return NULL;
    }
    return m_pPins[n];
}


// called from controller via custom interface
HRESULT
BridgeSource::GetBridgePin(int n, BridgeSourceOutput** ppPin)
{
	if ((n < 0) || (n >= m_nPins))
	{
		return E_INVALIDARG;
	}
	*ppPin = m_pPins[n];
	return S_OK;
}


STDMETHODIMP
BridgeSource::OnNewConnection(BOOL bIsDiscont)
{
	// adjust timebase to latest of the stop times
	REFERENCE_TIME tLatest = 0;
    for (int n = 0; n < m_nPins; n++)
    {
        if (m_pPins[n]->StopTime() > tLatest)
        {
            tLatest = m_pPins[n]->StopTime();
        }
    }
    bool bNewBase = false;
	if (tLatest > m_tBase)
	{
        m_tBase = tLatest;
        bNewBase = true;
    }

    bool bDiscont = false;
    if (bIsDiscont)
    {
        // if this new clip is not intended to be contiguous with the old one,
        // and this bridge has been disconnected while the
        // graph is running, we need to jump the baseline forwards
        REFERENCE_TIME tNow = RealStreamTime();
        if ((tNow > 0) && (m_tBase < tNow))
        {
            bNewBase = true;
            bDiscont = true;
            // allow 300ms latency
            m_tBase = tNow + 300*10000;
        }
    }
    LOG((TEXT("Source 0x%x new connection, base %d"), this, long(m_tBase/10000)));
    if (bNewBase)
    {
		LOG((TEXT("Source 0x%x new baseline %d msecs"), this, long(m_tBase/10000)));
        for (int n = 0; n < m_nPins; n++)
        {
            m_pPins[n]->SetBaseline(m_tBase, bDiscont);
		}
	}
    return S_OK;
}

REFERENCE_TIME
BridgeSource::RealStreamTime()
{
	// base class implementation is not
	// adjusted correctly for stop and pause
	REFERENCE_TIME tNow = 0;
	if (m_State == State_Paused)
	{
		tNow = m_tPausedAt;
	} else if ((m_State == State_Running) && m_pClock)
	{
		m_pClock->GetTime(&tNow);
		tNow -= m_tStart;
	}
	return tNow;
}


REFERENCE_TIME
BridgeSource::SegmentStreamTime()
{
	REFERENCE_TIME tNow = RealStreamTime();
	tNow -= m_tBase;
	return tNow;
}

void
BridgeSource::Flush()
{
	LOG((TEXT("Source 0x%x flush from upstream, paused-at %d ms, base %d ms"),
				this,
				long(m_tPausedAt/10000),
				long(m_tBase/10000)));

	// the upstream graph was flushed, which will
	// reset the stream time. We cannot reset the stream time
	// of this graph, so we must offset all samples
	// by the current paused-at position
	if (m_State == State_Paused)
	{
        m_tBase = m_tPausedAt;

        for (int n = 0; n < m_nPins; n++)
        {
            m_pPins[n]->SetBaseline(m_tBase, false);
        }
	}
    else if (m_State == State_Running)
    {
        // for Geraint's mixer app where the render graph does not pause,
        // we must make the following 0-based samples contiguous with the
        // previous samples.
        // Use the current stream time to see where we should be: other samples
        // will have been flushed.
	    REFERENCE_TIME tNewBase = RealStreamTime();
        // allow some latency
        tNewBase += 200 * 10000L;
	    if (tNewBase > m_tBase)
	    {
            m_tBase = tNewBase;
            LOG((TEXT("Source 0x%x flush: new baseline %d msecs"), this, long(m_tBase/10000)));
            for (int n = 0; n < m_nPins; n++)
            {
                m_pPins[n]->SetBaseline(m_tBase, false);
		    }
    	}
    }
}

STDMETHODIMP
BridgeSource::Pause()
{
	LOG((TEXT("Source 0x%x pause from %d"), this, m_State));

	if ((m_State == State_Running) && (m_pClock))
	{
		m_pClock->GetTime(&m_tPausedAt);
		m_tPausedAt -= m_tStart;
	} else {
		m_tPausedAt = 0;
	}

	return CBaseFilter::Pause();
}

STDMETHODIMP
BridgeSource::Stop()
{
    LOG((TEXT("Source 0x%x stop from %d"), this, m_State));

    // stream time will restart from 0, so no need to
    // offset the times
    m_tBase = 0;

    return CBaseFilter::Stop();
}

STDMETHODIMP
BridgeSource::Run(REFERENCE_TIME tStart)
{
	REFERENCE_TIME tLatency = 0;
	if (m_pClock)
	{
		REFERENCE_TIME tNow;
		m_pClock->GetTime(&tNow);
		tLatency = tStart - tNow;
	}
	LOG((TEXT("Source 0x%x run from %d, latency %d ms"), this, m_State, long(tLatency/10000)));

	return CBaseFilter::Run(tStart);
}


// register in the running object table for graph debugging
STDMETHODIMP
BridgeSource::JoinFilterGraph(IFilterGraph * pGraph, LPCWSTR pName)
{
    HRESULT hr = CBaseFilter::JoinFilterGraph(pGraph, pName);

    // for debugging, we register in the ROT so that you can use
    // Graphedt's Connect command to view the graphs
    // disabled by default owing to refcount leak issue
    if (false) //SUCCEEDED(hr))
    {
        if (pGraph == NULL)
        {
            if (m_dwROT) {
                IRunningObjectTablePtr pROT;
                if (SUCCEEDED(GetRunningObjectTable(0, &pROT))) {
                    pROT->Revoke(m_dwROT);
                }
            }
        } else {
            IMonikerPtr pMoniker;
            IRunningObjectTablePtr pROT;
            if (SUCCEEDED(GetRunningObjectTable(0, &pROT))) {
				ostringstream strm;
				DWORD graphaddress = (DWORD)((DWORD_PTR)(IUnknown*)pGraph) & 0xFFFFFFFF;
				strm << "FilterGraph " << hex << graphaddress << " pid " << hex << GetCurrentProcessId();
				_bstr_t strName = strm.str().c_str();
                HRESULT hr = CreateItemMoniker(L"!", strName, &pMoniker);
                if (SUCCEEDED(hr)) {
                    hr = pROT->Register(0, pGraph, pMoniker, &m_dwROT);
                }
            }
        }
    }

    return hr;
}
// ----------- output pin implementation ---------------------------

BridgeSourceOutput::BridgeSourceOutput(
	BridgeSource* pFilter,
	BridgeStream* pStream,
	CCritSec* pLock,
	HRESULT* phr,
	LPCWSTR pName)
: CBaseOutputPin(NAME("BridgeSourceOutput"), pFilter, pLock, phr, pName),
  m_pStream(pStream),
  m_bUpstreamTypeChanged(false),
  m_tLastStop(0),
  m_bIncreasedBuffering(false),
  m_bActive(false),
  m_tBase(0),
  m_bDiscont(false),
  m_bNewBaseline(true),
  m_mtLast(0),
  m_mtBase(0)
{
    m_bAudio = !m_pStream->IsVideo();
	m_hsemActive = CreateSemaphore(NULL, 0, 1, NULL);

}

BridgeSourceOutput::~BridgeSourceOutput()
{
	CloseHandle(m_hsemActive);
}

HRESULT
BridgeSourceOutput::CompleteConnect(IPin *pReceivePin)
{
	// is our agreed type the same as the upstream sink?
	m_bUpstreamTypeChanged = false;
	CMediaType mt;
	GetStream()->GetSelectedType(&mt);
	if (mt != m_mt)
	{
		// we've changed the media type, so we cannot use the upstream
		// buffer size for our allocator negotiation
		m_bUpstreamTypeChanged = true;
	}

	// m_mt.GetSampleSize if type changed!!
	HRESULT hr = CBaseOutputPin::CompleteConnect(pReceivePin);
	if (SUCCEEDED(hr))
	{
		GetStream()->SetSelectedType(&m_mt);
	}

    if (SUCCEEDED(hr))
    {
        // are we connected to a ReceiveConnection-aware VR?
        // if so, we can accept changing video dimensions
        bool bRC = false;

        // check if it supports IPinConnection with current media type
        IPinConnectionPtr pPC = GetConnected();
        if (pPC != NULL)
        {
            HRESULT hrD = pPC->DynamicQueryAccept(&m_mt);
            if (SUCCEEDED(hrD))
            {
                bRC = true;
            }
        }
        LOG((TEXT("ReceiveConnection Aware: %s"), bRC?TEXT("true") : TEXT("false")));
        GetStream()->CanReceiveConnect(bRC);
    }
	return hr;
}

HRESULT
BridgeSourceOutput::DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* pprop)
{
	// set the allocator to use the same size/count preferred upstream
	GetStream()->GetBufferProps(&pprop->cBuffers, &pprop->cbBuffer);

	// -- unless we have change the type, in the video case only
	if (!m_bAudio && m_bUpstreamTypeChanged)
	{
		pprop->cbBuffer = m_mt.GetSampleSize();
	}

	// check for minimum buffering
	long nMinimum = GetStream()->GetController()->BufferMinimum();
	if (nMinimum > 0)
	{
		// from milliseconds to reference time
		REFERENCE_TIME tMin = nMinimum * 10000;

		REFERENCE_TIME tBuffer = 0;
		if (*m_mt.FormatType() == FORMAT_VideoInfo)
		{
			VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)m_mt.Format();
			tBuffer = pvi->AvgTimePerFrame;
		} else if (*m_mt.FormatType() == FORMAT_VideoInfo2)
		{
			VIDEOINFOHEADER2* pvi = (VIDEOINFOHEADER2*)m_mt.Format();
			tBuffer = pvi->AvgTimePerFrame;
		} else if (*m_mt.FormatType() == FORMAT_WaveFormatEx)
		{
			WAVEFORMATEX* pwfx = (WAVEFORMATEX*)m_mt.Format();
            if (pwfx->nAvgBytesPerSec > 0)
            {
                tBuffer = UNITS * pprop->cbBuffer / pwfx->nAvgBytesPerSec;
            }
		}
		// for other formats, we will just ignore this request.
		if ((tBuffer > 0) && ((pprop->cBuffers * tBuffer) < tMin))
		{
			m_bIncreasedBuffering = true;
			pprop->cBuffers = long(tMin / tBuffer);
		}
        if (!GetStream()->IsVideo())
        {
            if (pprop->cBuffers < 5)
            {
                pprop->cBuffers = 5;
            }
            if (pprop->cbBuffer < 128 * 1024)
            {
                pprop->cbBuffer = 128*1024;
            }
        }
	}

	ALLOCATOR_PROPERTIES propActual;
	return pAlloc->SetProperties(pprop, &propActual);
}

HRESULT
BridgeSourceOutput::CheckMediaType(const CMediaType* pmt)
{
	return GetStream()->CanDeliverType(pmt);
}

HRESULT
BridgeSourceOutput::GetMediaType(int iPosition, CMediaType* pmt)
{
	return GetStream()->EnumOutputType(iPosition, pmt);
}

STDMETHODIMP
BridgeSourceOutput::Notify(IBaseFilter * pSender, Quality q)
{
	return GetStream()->NotifyQuality(pSender, q);
}

HRESULT
BridgeSourceOutput::Active()
{
	LOG((TEXT("Source pin 0x%x active"), this));
	{
		CAutoLock lock(&m_csActive);
		m_bActive = true;
	}

	HRESULT hr = CBaseOutputPin::Active();
	if (IsConnected())
	{
        // always add the queue -- the slight inefficiency is outweighed
        // by the increased robustness against source pause/stop while render is paused.
		if (true)//m_bIncreasedBuffering)
		{
			m_pQueue = new COutputQueue(GetConnected(), &hr, true, false, 1, false);
            m_pQueue->SetPopEvent(m_evQueue);
		}
	}

	// safe to start delivery
	ReleaseSemaphore(m_hsemActive, 1, NULL);
	return hr;
}

HRESULT
BridgeSourceOutput::Inactive()
{
	LOG((TEXT("Source pin 0x%x inactive"), this));
	// we must not complete this method until
	// any thread from upstream has exited
	// our Receive function. We do this with a combination
	// of semaphore and protected boolean.

	{
		// signal blocked thread not to deliver
		CAutoLock lock(&m_csActive);
		m_bActive = false;
	}

	// wait to ensure that no-one is in Receive
	WaitForSingleObject(m_hsemActive, INFINITE);

	HRESULT hr = CBaseOutputPin::Inactive();

	m_pQueue = NULL;

	CAutoLock lock(&m_csTime);
	m_tBase = 0;
	m_tLastStop = 0;
	m_mtLast = 0;
	m_mtBase = 0;
	LOG((TEXT("Source pin 0x%x inactive completed"), this));
	return hr;
}

void
BridgeSourceOutput::SetBaseline(REFERENCE_TIME tBase, bool bDiscont)
{
	// specifies offset to this segment's timestamps
	CAutoLock lock(&m_csTime);
	m_tBase = tBase;
    m_bDiscont = bDiscont;
	m_bNewBaseline = true;

	m_mtBase = m_mtLast;
}

HRESULT
BridgeSourceOutput::Send(IMediaSample* pSample)
{
	// check for inactive before attempting to acquire
	// the mutex
	{
		CAutoLock lock(&m_csActive);
		if (!m_bActive || !IsConnected())
		{
			return S_FALSE;
		}
	}

	// ensure we are active
	WaitForSingleObject(m_hsemActive, INFINITE);

	{
		// check that we are not shutting down
		CAutoLock lock(&m_csActive);
		if (!m_bActive || !IsConnected())
		{
			ReleaseSemaphore(m_hsemActive, 1, NULL);
			return S_FALSE;
		}
	}

	//LOG((TEXT("Source pin 0x%x Send 0x%x", this, pSample));

	bool bDiscard = false;

	// adjust times so that segments abut correctly
	REFERENCE_TIME tStart, tEnd;
	if (SUCCEEDED(pSample->GetTime(&tStart, &tEnd)))
	{
		CAutoLock lock(&m_csTime);

        if (m_pStream->GetController()->LiveTiming())
        {
            REFERENCE_TIME tStart, tStop;
            if (pSample->GetTime(&tStart, &tStop) == S_OK)
            {
                REFERENCE_TIME tSTO = Filter()->STO();
                tStart -= tSTO;
                tStop -= tSTO;
                pSample->SetTime(&tStart, &tStop);
                if (tStart < m_tLastStop)
                {
					LOG((TEXT("Discarding: %d after %d"), long(tStart/10000), long(m_tLastStop/10000)));
                    bDiscard = true;
                }
                else
                {
                    m_tLastStop = tStop;
                }
            }
        }
        else
        {
            if (m_bNewBaseline && (m_tBase > 0) && (m_pStream->AllowedTypes() == eUncompressed))
            {
                if ((pSample->IsPreroll() == S_OK) ||
                    (tStart < 0))
                {

                    LOG((TEXT("Dropping preroll %d..%dms"), long(tStart/10000), long(tEnd/10000)));
                    bDiscard = true;
                }
            }
    		if (!bDiscard)
    		{
    			REFERENCE_TIME tStream = Filter()->RealStreamTime();
    			LOG((TEXT("Sample 0x%x adjust from %d ms to %d ms, latency %d ms %c"),
    						pSample,
    						long(tStart/10000),
    						long((tStart+m_tBase)/10000),
    						long((tStart+m_tBase-tStream)/10000),
    						m_bDiscont ? TEXT('D') : TEXT(' ')));

    			tStart += m_tBase;
    			tEnd += m_tBase;
    			m_tLastStop = tEnd;
    			pSample->SetTime(&tStart, &tEnd);

    			REFERENCE_TIME tMT = 0, tMTEnd = 0;
    			if (SUCCEEDED(pSample->GetMediaTime(&tMT, &tMTEnd)))
    			{
    				if (m_bNewBaseline)
    				{
    					m_mtBase = m_mtLast - tMT;
    				}
    				if (m_mtBase != 0)
    				{
    					tMT += m_mtBase;
    					tMTEnd += m_mtBase;
    					pSample->SetMediaTime(&tMT, &tMTEnd);
    				}
    				m_mtLast = tMTEnd;
    			}

    			LOG((TEXT("Duration: %d, MT %d..%d"),
    				long((tEnd - tStart) / 10000),
    				long(tMT), long(tMTEnd)));

    			// ensure disconts are only set on a timebase jump -- this
    			// allows filters later in the graph to detect the jump
    			pSample->SetDiscontinuity(m_bDiscont);
    			m_bDiscont = false;
    		}
        }
	}

	m_bNewBaseline = false;

	// the sample is from the correct allocator already
	HRESULT hr = S_OK;
	if (!bDiscard)
	{
		if (m_pQueue != NULL)
		{
			pSample->AddRef();
			hr = m_pQueue->Receive(pSample);
		} else {
			hr = Deliver(pSample);
		}
	}
	ReleaseSemaphore(m_hsemActive, 1, NULL);

	if (hr != S_OK)
	{
		LOG((TEXT("Source pin 0x%x, sample 0x%x, HRESULT 0x%x"), this, pSample, hr));
	}

	return hr;
}

HRESULT
BridgeSourceOutput::DeliverBeginFlush()
{
	LOG((TEXT("Source pin 0x%x begin flush"), this));
	if (m_pQueue)
	{
		m_pQueue->BeginFlush();
	} else {
		CBaseOutputPin::DeliverBeginFlush();
	}
	Filter()->Flush();
	m_tLastStop = 0;
	m_mtLast = 0;
	return S_OK;
}

HRESULT
BridgeSourceOutput::DeliverEndFlush()
{
	LOG((TEXT("Source pin 0x%x end flush"), this));
	if (m_pQueue)
	{
		m_pQueue->EndFlush();
	} else {
		CBaseOutputPin::DeliverEndFlush();
	}
	return S_OK;
}

HRESULT
BridgeSourceOutput::DeliverEndOfStream()
{
	LOG((TEXT("Source pin 0x%x EOS"), this));
	if (m_pQueue)
	{
		m_pQueue->EOS();
	} else {
		CBaseOutputPin::DeliverEndOfStream();
	}
	return S_OK;
}


bool
BridgeSourceOutput::IsVideoRenderer()
{
    // get the downstream filter
    IPinPtr pPeer = GetConnected();
    PIN_INFO info;
    pPeer->QueryPinInfo(&info);
    IBaseFilterPtr pf(info.pFilter, 0);

    // the video renderer should support one of these
    IVideoWindowPtr pVW = pf;
    if (pVW != NULL)
    {
        return true;
    }
    IBasicVideoPtr pBV = pf;
    if (pBV != NULL)
    {
        return true;
    }
    return false;
}

// is this type acceptable to our downstream peer?
bool
BridgeSourceOutput::CanDeliver(const CMediaType*pmt)
{
    bool bAccept = true;        // if not connected, anything is acceptable
    if (IsConnected())
    {
        HRESULT hr = GetConnected()->QueryAccept(pmt);
        if (hr != S_OK)
        {
            IPinConnectionPtr pPC = GetConnected();
            if (pPC != NULL)
            {
                hr = pPC->DynamicQueryAccept(pmt);
            }
            if (hr != S_OK)
            {
                bAccept = false;
            }
        }
    }
    return bAccept;
}

// dynamic switch with Video Renderer (to change allocated buffer size as well
// as format type
HRESULT
BridgeSourceOutput::SwitchTo(const CMediaType* pmt)
{
    // must wait until queue is empty
    if (m_pQueue != NULL)
    {
        while (!m_pQueue->IsIdle())
        {
            m_evQueue.Wait();
        }
    }


    // now perform request
    HRESULT hr = GetConnected()->ReceiveConnection(this, pmt);
    LOG((TEXT("ReceiveConnection 0x%x"), hr));

    if (SUCCEEDED(hr))
    {
        SetMediaType(pmt);

        // for VMR, that's enough, but for old VR we need to re-commit the allocator
        m_pAllocator->Decommit();

        m_bUpstreamTypeChanged = true;
        ALLOCATOR_PROPERTIES prop;
        hr = m_pAllocator->GetProperties(&prop);
        if (SUCCEEDED(hr))
        {
            hr = DecideBufferSize(m_pAllocator, &prop);
            if (FAILED(hr))
            {
                LOG((TEXT("Allocator failure on ReceiveConnection 0x%x"), hr));
            }
        }

        if (SUCCEEDED(hr))
        {
            m_pInputPin->NotifyAllocator(m_pAllocator, false);
        }
        m_pAllocator->Commit();
    }
    return hr;
}


void
BridgeSourceOutput::ResetOnStop()
{
	// the sink filter reports that the graph was stopped
	// while connected and is now sending data again (timestamped at 0).
	// Treat this as a discont connection.
	Filter()->OnNewConnection(true);
}
