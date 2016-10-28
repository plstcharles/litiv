//
// GDCL Multigraph Framework
//
// Sink.cpp: implementation of sink filter and input pin
//
// Copyright (c) GDCL 2004. All Rights Reserved.
// You are free to re-use this as the basis for your own filter development,
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
#include "litiv/3rdparty/dshowbase/gmf/sink.h"
#pragma warning(disable: 4786)	// debug info truncated
#include <list>
using namespace std;
#include <sstream>

BridgeSink::BridgeSink(BridgeController* pController)
: m_tFirst(0),
  m_bLastDiscarded(false),
  m_nEOS(0),
  m_dwROT(0),
  CBaseFilter(NAME("BridgeSink filter"), NULL, &m_csFilter, GUID_NULL)
{
	HRESULT hr = S_OK;
    m_nPins = pController->StreamCount();
    m_pPins = new smart_ptr<BridgeSinkInput>[m_nPins];

    for(int n = 0; n < m_nPins; n++)
    {
		ostringstream strm;
		strm << "Input " << (n+1);
        _bstr_t strName = strm.str().c_str();
        m_pPins[n] = new BridgeSinkInput(this, pController->GetStream(n), m_pLock, &hr, strName);
    }

	LOG((TEXT("Sink 0x%x has %d pins"), this, m_nPins));
}

STDMETHODIMP
BridgeSink::NonDelegatingQueryInterface(REFIID iid, void** ppv)
{
	if (iid == IID_IMediaSeeking)
	{
        // implement IMediaSeeking directly ourselves, not via
        // a pass-through proxy, so that we can aggregate multiple input pins
        // (needed for WMV playback)
        return GetInterface((IMediaSeeking*)this, ppv);
	} else if (iid == __uuidof(IBridgeSink))
	{
		return GetInterface((IBridgeSink*)this, ppv);
	} else {
		return CBaseFilter::NonDelegatingQueryInterface(iid, ppv);
	}
}

int
BridgeSink::GetPinCount()
{
    return m_nPins;
}

CBasePin*
BridgeSink::GetPin(int n)
{
    if ((n < 0) || (n >= m_nPins))
    {
        return NULL;
    }
    return m_pPins[n];
}

STDMETHODIMP
BridgeSink::GetBridgePin(int nPin, BridgeSinkInput** ppPin)
{
    if ((nPin < 0) || (nPin >= m_nPins))
    {
        return E_INVALIDARG;
    }
    *ppPin = m_pPins[nPin];
	return S_OK;
}

void
BridgeSink::OnEOS(bool bConnected)
{
	{
		CAutoLock lock(&m_csEOS);
		m_nEOS += 1;
		if (!IsAtEOS())
		{
			LOG((TEXT("Sink 0x%x EOS discarded"), this));
			return;
		}
	}
    if (bConnected)
    {
        BridgeController* pC = m_pPins[0]->GetStream()->GetController();
	    LOG((TEXT("Sink 0x%x EOS"), this));
	    pC->OnEndOfSegment();
    }
}

STDMETHODIMP_(BOOL) BridgeSink::IsAtEOS()
{
	CAutoLock lock(&m_csEOS);
	long nPins = 0;
    for (int n = 0; n < m_nPins; n++)
    {
        if (m_pPins[n]->IsConnected())
        {
            nPins++;
        }
    }
	return (m_nEOS == nPins);
}


void
BridgeSink::ResetEOSCount()
{
	CAutoLock lock(&m_csEOS);
	m_nEOS = 0;
	m_tFirst = 0;
	m_bLastDiscarded = false;

	LOG((TEXT("Sink 0x%x ResetEOSCount"), this));
}

STDMETHODIMP
BridgeSink::Stop()
{
	LOG((TEXT("Sink 0x%x Stop"), this));
	HRESULT hr = CBaseFilter::Stop();
	ResetEOSCount();
	return hr;
}

void
BridgeSink::Discard()
{
	CAutoLock lock(&m_csEOS);
	m_bLastDiscarded = true;
}

void
BridgeSink::AdjustTime(IMediaSample* pIn)
{
	CAutoLock lock(&m_csEOS);
	REFERENCE_TIME tStart, tEnd;
	if (SUCCEEDED(pIn->GetTime(&tStart, &tEnd)))
	{
		if (m_bLastDiscarded)
		{
			LOG((TEXT("Sink 0x%x setting offset %d msecs"), this, long(tStart / 10000)));

			m_tFirst = tStart;
			m_bLastDiscarded = false;
		}

		if (m_tFirst != 0)
		{
			LOG((TEXT("Sink adjusting %d to %d msecs"), long(tStart/10000), long((tStart-m_tFirst)/10000)));
		}
		tStart -= m_tFirst;
		tEnd -= m_tFirst;
		pIn->SetTime(&tStart, &tEnd);
	}
}

// register in the running object table for graph debugging
STDMETHODIMP
BridgeSink::JoinFilterGraph(IFilterGraph * pGraph, LPCWSTR pName)
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

// BridgeSink seeking implementation

// holds all the input pins that support seeking
class SeekingCollection
{
public:
    typedef list<IMediaSeeking*>::iterator iterator;

    // if bSet, only accept settable pins
    SeekingCollection(CBaseFilter* pFilter)
    {
        for (int i = 0; i < pFilter->GetPinCount(); i++)
        {
            CBasePin* pPin = pFilter->GetPin(i);
            PIN_DIRECTION pindir;
            pPin->QueryDirection(&pindir);
            if (pindir == PINDIR_INPUT)
            {
                IMediaSeekingPtr pSeek = pPin->GetConnected();
                if (pSeek != NULL)
                {
                    m_Pins.push_back(pSeek.Detach());
                }
            }
        }
    }
    ~SeekingCollection()
    {
        while (!m_Pins.empty())
        {
            IMediaSeekingPtr pSeek(m_Pins.front(), 0);
            m_Pins.pop_front();
            pSeek = NULL;
        }
    }

    iterator Begin()
    {
        return m_Pins.begin();
    }
    iterator End()
    {
        return m_Pins.end();
    }
private:
    list<IMediaSeeking*> m_Pins;
};

// --- not implemented -------------
STDMETHODIMP
BridgeSink::GetCurrentPosition(LONGLONG *pCurrent)
{
	UNREFERENCED_PARAMETER(pCurrent);
	// implemented in graph manager using stream time
    return E_NOTIMPL;
}


//  ---- by aggregation of input pin responses --------------
STDMETHODIMP
BridgeSink::GetCapabilities(DWORD * pCapabilities)
{
    SeekingCollection pins(this);
    DWORD caps = 0;
    for (SeekingCollection::iterator it = pins.Begin(); it != pins.End(); it++)
    {
        IMediaSeekingPtr pSeek = *it;

        DWORD capsThis;
        HRESULT hr = pSeek->GetCapabilities(&capsThis);
        if (SUCCEEDED(hr))
        {
            caps |= capsThis;
        }
    }
    *pCapabilities = caps;
    return S_OK;
}


STDMETHODIMP
BridgeSink::SetPositions(LONGLONG * pCurrent, DWORD dwCurrentFlags
		, LONGLONG * pStop, DWORD dwStopFlags)
{
    // pass call to all pins. Fails if any fail.
    SeekingCollection pins(this);
    HRESULT hr = S_OK;
    for (SeekingCollection::iterator it = pins.Begin(); it != pins.End(); it++)
    {
        IMediaSeekingPtr pSeek = *it;

        if (pSeek->IsUsingTimeFormat(&TIME_FORMAT_MEDIA_TIME) == S_OK)
        {
            HRESULT hrThis = pSeek->SetPositions(pCurrent, dwCurrentFlags,  pStop, dwStopFlags);
            if (FAILED(hrThis) && (hrThis != E_NOTIMPL) && SUCCEEDED(hr))
            {
                hr = hrThis;
            }
        }
    }
    return hr;
}

STDMETHODIMP
BridgeSink::GetPositions(LONGLONG * pCurrent,
                            LONGLONG * pStop)
{
    // cannot really aggregate this -- just return the
    // first one and assume they are all the same (they will
    // be if we set the params)
    SeekingCollection pins(this);
    HRESULT hr;
    if (pins.Begin() == pins.End())
    {
        hr = E_NOINTERFACE;
    } else {
        IMediaSeekingPtr pSeek = *pins.Begin();
        hr = pSeek->GetPositions(pCurrent, pStop);
    }
    return hr;
}

STDMETHODIMP
BridgeSink::GetAvailable(LONGLONG * pEarliest, LONGLONG * pLatest)
{
    // the available section reported is what is common to all inputs
    LONGLONG tEarly = 0;
    LONGLONG tLate = 0x7fffffffffffffff;

    SeekingCollection pins(this);
    for (SeekingCollection::iterator it = pins.Begin(); it != pins.End(); it++)
    {
        IMediaSeekingPtr pSeek = *it;

        LONGLONG tThisEarly, tThisLate;
        HRESULT hr = pSeek->GetAvailable(&tThisEarly, &tThisLate);
        if (SUCCEEDED(hr))
        {
            if (tThisEarly > tEarly)
            {
                tEarly = tThisEarly;
            }
            if (tThisLate < tLate)
            {
                tLate = tThisLate;
            }
        }
    }
    *pEarliest = tEarly;
    *pLatest = tLate;
    return S_OK;
}

STDMETHODIMP
BridgeSink::SetRate(double dRate)
{
    // pass call to all pins. Fails if any fail.
    SeekingCollection pins(this);
    HRESULT hr = S_OK;
    for (SeekingCollection::iterator it = pins.Begin(); it != pins.End(); it++)
    {
        IMediaSeekingPtr pSeek = *it;

        HRESULT hrThis = pSeek->SetRate(dRate);
        if (FAILED(hrThis) && SUCCEEDED(hr))
        {
            hr = hrThis;
        }
    }
    return hr;
}

STDMETHODIMP
BridgeSink::GetRate(double * pdRate)
{
    // cannot really aggregate this -- just return the
    // first one and assume they are all the same (they will
    // be if we set the params)
    SeekingCollection pins(this);
    HRESULT hr;
    if (pins.Begin() == pins.End())
    {
        hr = E_NOINTERFACE;
    } else {
        IMediaSeekingPtr pSeek = *pins.Begin();
        hr = pSeek->GetRate(pdRate);
    }
    return hr;
}

STDMETHODIMP
BridgeSink::GetPreroll(LONGLONG * pllPreroll)
{
    // the preroll requirement is the longest of any input
    SeekingCollection pins(this);
    LONGLONG tPreroll = 0;
    for (SeekingCollection::iterator it = pins.Begin(); it != pins.End(); it++)
    {
        IMediaSeekingPtr pSeek = *it;

        LONGLONG tThis;
        HRESULT hr = pSeek->GetPreroll(&tThis);
        if (SUCCEEDED(hr))
        {
            if (tThis > tPreroll)
            {
                tPreroll = tThis;
            }
        }
    }
    *pllPreroll = tPreroll;
    return S_OK;
}

STDMETHODIMP
BridgeSink::GetDuration(LONGLONG *pDuration)
{
    // the duration we report is the longest of any input duration

    SeekingCollection pins(this);
    LONGLONG tDur = 0;
    for (SeekingCollection::iterator it = pins.Begin(); it != pins.End(); it++)
    {
        IMediaSeekingPtr pSeek = *it;

        LONGLONG tThis;
        HRESULT hr = pSeek->GetDuration(&tThis);
        if (SUCCEEDED(hr))
        {
            if (tThis > tDur)
            {
                tDur = tThis;
            }
        }
    }
    *pDuration = tDur;
    return S_OK;
}

STDMETHODIMP
BridgeSink::GetStopPosition(LONGLONG *pStop)
{
    // cannot really aggregate this -- just return the
    // first one and assume they are all the same (they will
    // be if we set the params)
    SeekingCollection pins(this);
    HRESULT hr;
    if (pins.Begin() == pins.End())
    {
        hr = E_NOINTERFACE;
    } else {
        IMediaSeekingPtr pSeek = *pins.Begin();
        hr = pSeek->GetStopPosition(pStop);
    }
    return hr;
}

// -- implemented directly here --------------
STDMETHODIMP
BridgeSink::CheckCapabilities(DWORD * pCapabilities )
{
    DWORD dwActual;
    GetCapabilities(&dwActual);
    if (*pCapabilities & (~dwActual)) {
        return S_FALSE;
    }
    return S_OK;
}

STDMETHODIMP
BridgeSink::IsFormatSupported(const GUID * pFormat)
{
    if (*pFormat == TIME_FORMAT_MEDIA_TIME) {
        return S_OK;
    }
    return S_FALSE;
}

STDMETHODIMP
BridgeSink::QueryPreferredFormat(GUID * pFormat)
{
   *pFormat = TIME_FORMAT_MEDIA_TIME;
    return S_OK;
}

STDMETHODIMP
BridgeSink::GetTimeFormat(GUID *pFormat)
{
    return QueryPreferredFormat(pFormat);
}

STDMETHODIMP
BridgeSink::IsUsingTimeFormat(const GUID * pFormat)
{
    GUID guidActual;
    HRESULT hr = GetTimeFormat(&guidActual);

    if (SUCCEEDED(hr) && (guidActual == *pFormat)) {
        return S_OK;
    } else {
        return S_FALSE;
    }
}

STDMETHODIMP
BridgeSink::SetTimeFormat(const GUID * pFormat)
{
    if ((*pFormat == TIME_FORMAT_MEDIA_TIME) ||
        (*pFormat == TIME_FORMAT_NONE))
    {
        return S_OK;
    } else {
        return E_NOTIMPL;
    }
}

STDMETHODIMP
BridgeSink::ConvertTimeFormat(LONGLONG * pTarget, const GUID * pTargetFormat,
                            LONGLONG    Source, const GUID * pSourceFormat)
{
    // since we only support TIME_FORMAT_MEDIA_TIME, we don't really
    // offer any conversions.
    if(pTargetFormat == 0 || *pTargetFormat == TIME_FORMAT_MEDIA_TIME) {
        if(pSourceFormat == 0 || *pSourceFormat == TIME_FORMAT_MEDIA_TIME) {
            *pTarget = Source;
            return S_OK;
        }
    }

    return E_INVALIDARG;
}



// --- input pin implementation --------------------------------

BridgeSinkInput::BridgeSinkInput(
	BridgeSink* pFilter,
	BridgeStream* pStream,
	CCritSec* pLock,
	HRESULT* phr,
	LPCWSTR pName)
: CBaseInputPin(NAME("BridgeSinkInput"), pFilter, pLock, phr, pName),
  m_pStream(pStream),
  m_pRedirectedAlloc(NULL),
  m_bConnected(false),
  m_nDeliveryLocks(0),
  m_bSendDTC(false),
  m_bStoppedWhileConnected(false)
{
    m_bAudio = !m_pStream->IsVideo();

    m_hsemDelivery = CreateSemaphore(NULL, 1, 1, NULL);
	if (!m_pStream->DiscardMode())
	{
		m_pRedirectedAlloc = new BridgeAllocator(this);
		m_pRedirectedAlloc->AddRef();
	}
}

BridgeSinkInput::~BridgeSinkInput()
{
	if (m_pRedirectedAlloc)
	{
		m_pRedirectedAlloc->Release();
	}
	CloseHandle(m_hsemDelivery);
}

void
BridgeSinkInput::LockIncremental()
{
    // if there are multiple calls to this on separate
    // threads, there's a chance that the second one
    // will be left blocked on the semaphore when he should be seeing the
    // incremental indicator. To avoid this, timeout the semaphore and loop
    for (;;)
    {
        {
            CAutoLock lock(&m_csDelivery);
            if (m_nDeliveryLocks > 0)
            {
                // lock is incremental -- can add to it
                m_nDeliveryLocks++;
                return;
            }
        }

        // acquire exclusive lock
        DWORD dw = WaitForSingleObject(DeliveryLock(), 100);
        if (dw == WAIT_OBJECT_0)
        {
            break;
        }
    }

    // now mark it as incremental
    CAutoLock lock(&m_csDelivery);
    m_nDeliveryLocks++;
}

// if incremental, decrease count and release semaphore if 0.
// if exclusive, release semaphore (count is 0)
void
BridgeSinkInput::Unlock()
{
    CAutoLock lock(&m_csDelivery);
    if (m_nDeliveryLocks > 0)
    {
        // incremental lock
        m_nDeliveryLocks--;
        if (m_nDeliveryLocks > 0)
        {
            // not idle yet
            return;
        }
    }
    ReleaseSemaphore(m_hsemDelivery, 1, NULL);
}

// CBaseInputPin overrides
STDMETHODIMP
BridgeSinkInput::Receive(IMediaSample *pSampleIn)
{
	LOG((TEXT("Pin 0x%x receive 0x%x"), this, pSampleIn));

	// check state
	HRESULT hr = CBaseInputPin::Receive(pSampleIn);
	if (hr == S_OK)
	{
		// if not connected, do nothing.
		// For the "discard on not connected" option, this
		// is the correct behaviour. For the suspend on not connected option,
		// we should not get here since GetBuffer will suspend.

		{	// restrict critsec -- don't hold over blocking call
			CAutoLock lock(&m_csConnect);
			if (!m_bConnected)
			{
				LOG((TEXT("Sink pin 0x%x disconnected: discarding 0x%x"), this, pSampleIn));

				// remember that the segment is broken
				Filter()->Discard();

				// just ignore this sample
				return S_OK;
			}
		}


        // we must hold a lock during receive. If upstream is using the proxy alloc,
        // then the lock is held by the proxy sample. If not, we should get a proxy sample
        // just for the duration of Receive so it will hold the lock
        IProxySamplePtr pProxy = pSampleIn;
        IMediaSamplePtr pInner;
        if (pProxy != NULL)
        {
            // already has proxy and lock -- extract inner
            pProxy->GetInner(&pInner);
        }
        else
        {
            // make new proxy to hold lock
            hr = S_OK;
            pProxy = new ProxySample(this, &hr);

            // we already have the inner
            pInner = pSampleIn;
        }
        LOG((TEXT("Pin 0x%x outer 0x%x, inner 0x%x"), this, pSampleIn, pInner));

		// if we stopped while connected, our times will be reset to 0
		// so we must notify the source
		if (m_bStoppedWhileConnected)
		{
			GetStream()->ResetOnStop();
			m_bStoppedWhileConnected = false;
		}

        // before changing it, we must make a copy
        IMediaSamplePtr pLocal = pInner;
        hr = S_OK;
        if (GetStream()->GetController()->LiveTiming())
        {
            // map to absolute clock time here
            // (and back in source graph)
            // depends on using a common clock.
            REFERENCE_TIME tStart, tStop;
            if (pLocal->GetTime(&tStart, &tStop) == S_OK)
            {
                REFERENCE_TIME tSTO = Filter()->STO();
                tStart += tSTO;
                tStop += tSTO;
                pLocal->SetTime(&tStart, &tStop);
				pLocal->SetMediaTime(NULL, NULL);
            }
        }
        else
        {
            // if we are starting delivery after being disconnected,
            // the timestamps should begin at 0
            // -- handled in filter to ensure common adjustment for all streams
            Filter()->AdjustTime(pLocal);
        }

		// check for media type change
        AM_MEDIA_TYPE* pmt;
        CMediaType mtFromUpstream;
        bool bTypeChange = false;
        if (pLocal->GetMediaType(&pmt) == S_OK)
        {
            CMediaType mt(*pmt);
            DeleteMediaType(pmt);

			// is this a new type?
			CMediaType mtDownstream;
			GetStream()->GetSelectedType(&mtDownstream);
			if (mt != mtDownstream)
			{
				// must be upstream-originated
				GetStream()->SetSelectedType(&mt);

                mtFromUpstream = mt;
                bTypeChange = true;
            }
		}
        else if (pProxy->GetType(&mtFromUpstream) == S_OK)
        {
            LOG((TEXT("Using DTC from proxy")));

            // type change was attached to sample in GetBuffer
            // but then erased in upstream filter
            //re-attach to inner sample
            pLocal->SetMediaType(&mtFromUpstream);

            CMediaType mtDownstream;
            GetStream()->GetSelectedType(&mtDownstream);
            if (mtFromUpstream != mtDownstream)
            {
                // must be upstream-originated
                GetStream()->SetSelectedType(&mtFromUpstream);
                bTypeChange = true;
            }
        }
        else if (m_bSendDTC)
        {
            // the type of the input needs to be passed downstream.
            LOG((TEXT("Type change sample lost -- setting on next sample")));

            m_bSendDTC = false;
            mtFromUpstream = m_mt;
            bTypeChange = true;
            pLocal->SetMediaType(&m_mt);
        }



        // if we are in "discard" mode, we are not using the
        // same allocator, so we must copy here
        if (m_bUsingProxyAllocator)
        {
            hr = GetStream()->Deliver(pLocal);
        }
        else
        {
            // need to copy. For audio this might mean a repeated copy
            int cIn = pLocal->GetActualDataLength();
            int cOffset = 0;
            while (cOffset < cIn)
            {
                IMediaSamplePtr pOut;
                if (m_pCopyAllocator == NULL)
                {
                    return VFW_E_NO_ALLOCATOR;
                }
                hr = m_pCopyAllocator->GetBuffer(&pOut, NULL, NULL, 0);
                if (SUCCEEDED(hr))
                {
                    hr = CopySample(pLocal, pOut, cOffset, cIn - cOffset);
                    cOffset += pOut->GetActualDataLength();
                }
                if (SUCCEEDED(hr))
                {
                    if (bTypeChange)
                    {
                        pOut->SetMediaType(&mtFromUpstream);
                        bTypeChange = false;
                    }
                    hr = GetStream()->Deliver(pOut == NULL? pLocal : pOut);
                }
                if (FAILED(hr))
                {
                    return hr;
                }
            }
        }
	}

	return hr;
}

// copy a portion of the input buffer to the output. Copy times and flags on first portion
// only.
HRESULT
BridgeSinkInput::CopySample(IMediaSample* pIn, IMediaSample* pOut, int cOffset, int cLength)
{
    BYTE* pDest;
    pOut->GetPointer(&pDest);
    BYTE* pSrc;
    pIn->GetPointer(&pSrc);
    pSrc += cOffset;

    long cOut = pOut->GetSize();
    long cIn = pIn->GetActualDataLength();
    cLength = min(cLength, cOut);

	// ensure we copy whole samples if audio
	if ((*m_mt.Type() == MEDIATYPE_Audio) &&
		(*m_mt.FormatType() == FORMAT_WaveFormatEx))
	{
		WAVEFORMATEX* pwfx = (WAVEFORMATEX*)m_mt.Format();
		cLength -= cLength % pwfx->nBlockAlign;
	}

    if ((cOffset + cLength) > cIn)
    {
        return VFW_E_BUFFER_OVERFLOW;
    }


    CopyMemory(pDest, pSrc, cLength);
    pOut->SetActualDataLength(cLength);

    // properties are set on first buffer only
    if (cOffset == 0)
    {
        REFERENCE_TIME tStart, tEnd;
        if (SUCCEEDED(pIn->GetTime(&tStart, &tEnd)))
        {
            pOut->SetTime(&tStart, &tEnd);
        }

		if (SUCCEEDED(pIn->GetMediaTime(&tStart, &tEnd)))
		{
			pOut->SetMediaTime(&tStart, &tEnd);
		}

        if (pIn->IsSyncPoint() == S_OK)
        {
            pOut->SetSyncPoint(true);
        }
        if (pIn->IsDiscontinuity() == S_OK)
        {
            pOut->SetDiscontinuity(true);
        }
        if (pIn->IsPreroll() == S_OK)
        {
            pOut->SetPreroll(true);
        }
    }
	return S_OK;
}

STDMETHODIMP
BridgeSinkInput::EndOfStream(void)
{
	HRESULT hr = Filter()->NotifyEvent(EC_COMPLETE, S_OK, NULL);
    Filter()->OnEOS(m_bConnected);
	return CBaseInputPin::EndOfStream();
}

HRESULT
BridgeSinkInput::CheckMediaType(const CMediaType* pmt)
{
	// do we insist on the decoder being in the upstream segment?
	if (GetStream()->AllowedTypes() == eUncompressed)
	{
		if (!IsUncompressed(pmt))
		{
			return VFW_E_TYPE_NOT_ACCEPTED;
		}
	} else if (GetStream()->AllowedTypes() == eMuxInputs)
	{
		if (!IsAllowedMuxInput(pmt))
		{
			return VFW_E_TYPE_NOT_ACCEPTED;
		}
	}
	// check with bridge stream -- type is fixed once output
	// stage has been built
	return GetStream()->CanReceiveType(pmt);
}

HRESULT
BridgeSinkInput::GetMediaType(int iPosition, CMediaType* pmt)
{
	UNREFERENCED_PARAMETER(iPosition);
	UNREFERENCED_PARAMETER(pmt);
	return VFW_S_NO_MORE_ITEMS;
}

HRESULT
BridgeSinkInput::SetMediaType(const CMediaType* pmt)
{
	HRESULT hr = CBaseInputPin::SetMediaType(pmt);
	if (SUCCEEDED(hr))
	{
        CAutoLock lock(&m_csConnect);
        if (m_bConnected)
        {
            GetStream()->SetSelectedType(pmt);
        }
	}
	return hr;
}

bool
BridgeSinkInput::IsUncompressed(const CMediaType* pmt)
{
	if (m_bAudio)
	{
		if (*pmt->Type() != MEDIATYPE_Audio)
		{
			return false;
		}
		if (*pmt->FormatType() != FORMAT_WaveFormatEx)
		{
			return false;
		}
		WAVEFORMATEX* pwfx = (WAVEFORMATEX*)pmt->Format();
		if (pwfx->wFormatTag != WAVE_FORMAT_PCM)
		{
			return false;
		}
		return true;
	} else {
		if (*pmt->Type() != MEDIATYPE_Video)
		{
			return false;
		}
		if (
			(*pmt->Subtype() == MEDIASUBTYPE_ARGB32)||
			(*pmt->Subtype() == MEDIASUBTYPE_RGB32) ||
			(*pmt->Subtype() == MEDIASUBTYPE_RGB24) ||
			(*pmt->Subtype() == MEDIASUBTYPE_YUY2) ||
			(*pmt->Subtype() == MEDIASUBTYPE_UYVY) ||
			(*pmt->Subtype() == MEDIASUBTYPE_Y41P) ||
			(*pmt->Subtype() == MEDIASUBTYPE_RGB555) ||
			(*pmt->Subtype() == MEDIASUBTYPE_RGB565) ||
			(*pmt->Subtype() == MEDIASUBTYPE_RGB8)
			)
		{
			return true;
		}
	}
	return false;
}

bool
BridgeSinkInput::IsAllowedMuxInput(const CMediaType* pmt)
{
	// the AVI Mux only accepts certain formats -- we
	// must check for them at the sink.
    if (*pmt->Type() == MEDIATYPE_Video)
    {
		//must be either VideoInfo or DvInfo
        if (*pmt->FormatType() == FORMAT_VideoInfo)
        {
			// for VideoInfo, must have no target rect and no negative height

			VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)pmt->Format();
            if ((pvi->bmiHeader.biHeight < 0) ||
                (pvi->rcTarget.left != 0) ||
                ((pvi->rcTarget.right != 0) && (pvi->rcTarget.right != pvi->bmiHeader.biWidth)))
            {
                return false;
            }
        } else if (*pmt->FormatType() != FORMAT_DvInfo)
        {
            return false;
        }
    } else if (*pmt->Type() == MEDIATYPE_Audio)
    {
		// audio must be WaveFormatEx with a valid nBlockAlign
        if (*pmt->FormatType() != FORMAT_WaveFormatEx)
        {
            return false;
        }
        WAVEFORMATEX* pwfx = (WAVEFORMATEX*)pmt->Format();
        if (pwfx->nBlockAlign == 0)
        {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

STDMETHODIMP
BridgeSinkInput::GetAllocator(IMemAllocator **ppAllocator)
{
	/// if not connected, should we discard?
	HRESULT hr;
	if (GetStream()->DiscardMode())
	{
		// yes -- so we must use a standard allocator
		// (which will still work when not connected
		hr = CBaseInputPin::GetAllocator(ppAllocator);
	} else {
        // prefer our allocator since this handles dynamic type changes
		// as well as preventing copies
		hr = m_pRedirectedAlloc->QueryInterface(IID_IMemAllocator, (void**)ppAllocator);
	}
	return hr;
}

STDMETHODIMP
BridgeSinkInput::NotifyAllocator(IMemAllocator * pAllocator, BOOL bReadOnly)
{
	if (!GetStream()->DiscardMode())
	{
		// insist on our allocator
		IUnknownPtr pOurs = m_pRedirectedAlloc;
		IUnknownPtr pNotified = pAllocator;
		if (pOurs == pNotified)
		{
            m_bUsingProxyAllocator = true;
        }
        else
        {
            m_bUsingProxyAllocator = false;

            // for video, we must use the proxy alloc or we can't handle
            // type switching.
            // for audio, we could allow this, but we would need to add
            // code in Receive, since currently we rely on the allocator blocking
            // until connected -- with a foreign allocator, we would need to block in receive instead.
            // This would be a benefit in some DV cases where the audio buffer size is an issue and
            // hard to renegotiate between clips.

            // error code is not quite ideal, but at least points at the offending object
            return VFW_E_NO_ALLOCATOR;
		}
	}
    else
    {
        m_bUsingProxyAllocator = false;
    }

	return CBaseInputPin::NotifyAllocator(pAllocator, bReadOnly);
}

STDMETHODIMP
BridgeSinkInput::BeginFlush()
{
	LOG((TEXT("Pin 0x%x BeginFlush"), this));
	HRESULT hr = CBaseInputPin::BeginFlush();

	// allocator must fail without blocking while flushing - this is
	// the same as decommit state
	if (m_pRedirectedAlloc && m_bUsingProxyAllocator)
	{
		m_pRedirectedAlloc->Decommit();
	}

	// pass on to the downstream graph if connected
	CAutoLock lock(&m_csConnect);
	if (m_bConnected)
	{
		hr = GetStream()->BeginFlush();
	}

	return hr;
}

STDMETHODIMP
BridgeSinkInput::EndFlush()
{
	LOG((TEXT("Pin 0x%x EndFlush"), this));

	HRESULT hr = CBaseInputPin::EndFlush();

	// reset end-of-stream if delivered
	Filter()->ResetEOSCount();

	// undo the Decommit done during BeginFlush
	if (m_pRedirectedAlloc && m_bUsingProxyAllocator)
	{
		m_pRedirectedAlloc->Commit();
	}

	// pass on to the downstream graph if connected
	CAutoLock lock(&m_csConnect);
	if (m_bConnected)
	{
		hr = GetStream()->EndFlush();
	}
	return hr;
}

// we are now the selected source for the downstream
// graph.
HRESULT
BridgeSinkInput::MakeBridge(IMemAllocator* pAlloc)
{
	LOG((TEXT("Pin 0x%x Bridge to alloc 0x%x"), this, pAlloc));

	CAutoLock lock(&m_csConnect);

	m_bStoppedWhileConnected = false;
	m_bConnected = true;

    HRESULT hr = S_OK;
    if (m_bUsingProxyAllocator)
    {
    	// the previous source graph or the renderer may have
    	// made a type change -- ensure this is on the first sample
    	// -- so we set it before enabling the GetBuffer
    	CMediaType mt;
    	GetStream()->GetSelectedType(&mt);

        // is it compatible?
        if (mt != m_mt)
        {
            // dynamic changes sent upstream with GetBuffer only work reliably with video
            if (*m_mt.Type() == MEDIATYPE_Video)
            {
                hr = CanDeliverType(&mt);
                if (hr == S_OK)
                {
                    // sadly, many codecs do not check the size
                    // they are offered, and will claim to output anything
                    hr = BridgeStream::CheckMismatchedVideo(&mt, &m_mt);
                }
            } else {
                hr = E_FAIL;
            }

            if (hr == S_OK)
            {
                // switch our source to this type by attaching at
                // next GetBuffer
                LOG((TEXT("Switching source to stream type")));
                m_pRedirectedAlloc->ForceDTC(&mt);
            } else if ((*m_mt.Type() == MEDIATYPE_Video))
            {
                // don't switch back to the previously connected type, or the VR might
                // see an extended stride as part of the video. Instead, enumerate
                // the preferred formats from the source and try those. However, stick to
                // the same subtype.
                hr = VFW_E_TYPE_NOT_ACCEPTED;
                IEnumMediaTypesPtr pEnum;
                GetConnected()->EnumMediaTypes(&pEnum);
                AM_MEDIA_TYPE* pmt;
                while(pEnum->Next(1, &pmt, NULL) == S_OK)
                {
                    CMediaType mtEnum(*pmt);
                    DeleteMediaType(pmt);

                    if ((CanDeliverType(&mtEnum) == S_OK) &&
                        (*mtEnum.Subtype() == *m_mt.Subtype()) &&
                        GetStream()->CanSwitchTo(&mtEnum))
                    {
                        LOG((TEXT("ReceiveConnect to enumerated source type")));
                        m_pRedirectedAlloc->SwitchFormatTo(&mtEnum);
                        hr = S_OK;
                        break;
                    }
                }
                if (hr != S_OK)
                {
                    LOG((TEXT("No suitable video type - failing bridge")));
                    return hr;
                }
            } else
            {
                // attempt a dynamic switch downstream to the format
                // that our source is using
                LOG((TEXT("Switching downstream to source type")));
                m_pRedirectedAlloc->ForceDTC(&m_mt);

                // the type change is attached to a buffer from our GetBuffer,
                // and we need to pass it downstream, even if that sample is discarded
                // or the mt is cleared. So we set a flag here indicating that
                // we need to set the type onto the next incoming sample.
                m_bSendDTC = true;
            }
        }


        // redirect allocator
        hr =  m_pRedirectedAlloc->SetDownstreamAlloc(pAlloc);
	} else {
		m_pCopyAllocator = pAlloc;
	}

    return hr;
}

// we are now disconnected from downstream
HRESULT
BridgeSinkInput::DisconnectBridge()
{
	LOG((TEXT("Pin 0x%x disconnect"), this));
    if (m_pRedirectedAlloc && m_bUsingProxyAllocator)
    {
        m_pRedirectedAlloc->SetDownstreamAlloc(NULL);
    }

	CAutoLock lock(&m_csConnect);
	// if we are in mid-flush, remember that the endflush will
	// not be passed on
	if (m_bFlushing && m_bConnected)
	{
		LOG((TEXT("Pin 0x%x disconnect when mid-flush"), this));
		GetStream()->EndFlush();
	}
	m_bConnected = false;
	m_bStoppedWhileConnected = false;

	return S_OK;
}

HRESULT
BridgeSinkInput::GetBufferProps(long* pcBuffers, long* pcBytes)
{
	// return whatever we have agreed, whether the bridge allocator or not
	HRESULT hr = VFW_E_NO_ALLOCATOR;
	if (m_pAllocator)
	{
		ALLOCATOR_PROPERTIES prop;
		hr = m_pAllocator->GetProperties(&prop);
		*pcBuffers = prop.cBuffers;
		*pcBytes = prop.cbBuffer;
	}
	return hr;
}

HRESULT
BridgeSinkInput::CanDeliverType(const CMediaType* pmt)
{
	// do we insist on the decoder being in the upstream segment?
	if (GetStream()->AllowedTypes() == eUncompressed)
	{
		if (!IsUncompressed(pmt))
		{
			return VFW_E_TYPE_NOT_ACCEPTED;
		}
	} else if (GetStream()->AllowedTypes() == eMuxInputs)
	{
		if (!IsAllowedMuxInput(pmt))
		{
			return VFW_E_TYPE_NOT_ACCEPTED;
		}
	}

	if (!IsConnected())
	{
		return VFW_E_NOT_CONNECTED;
	}

	// query accept on upstream output pin
	return GetConnected()->QueryAccept(pmt);
}

HRESULT
BridgeSinkInput::EnumOutputType(int iPosition, CMediaType* pmt)
{
	// enumerate output types on upstream pin
	if (!IsConnected())
	{
		return VFW_E_NOT_CONNECTED;
	}
	IEnumMediaTypesPtr pEnum;
	HRESULT hr = GetConnected()->EnumMediaTypes(&pEnum);
	if (SUCCEEDED(hr))
	{
		pEnum->Skip(iPosition);
		AM_MEDIA_TYPE* amt;
		hr = pEnum->Next(1, &amt, NULL);
		if (hr == S_OK)
		{
			*pmt = *amt;
            DeleteMediaType(amt);
		}
	}
	return hr;
}

HRESULT
BridgeSinkInput::Inactive()
{
	if (Filter()->IsActive() && m_bConnected)
	{
		m_bStoppedWhileConnected = true;
	}
	return __super::Inactive();
}


// --------- Redirecting allocator implementation --------------------------

BridgeAllocator::BridgeAllocator(BridgeSinkInput* pPin)
: CUnknown(NAME("BridgeAllocator"), NULL),
  m_pPin(pPin),
  m_bForceDTC(false),
  m_bSwitchConnection(false),
  m_bCommitted(false),
  m_evNonBlocking(true)		// manual reset
{
	ZeroMemory(&m_props, sizeof(m_props));

	// while not committed, we just reject calls, not block
	// (blocking is only when we are active, but not connected to
	// the downstream graph)
	m_evNonBlocking.Set();
}

STDMETHODIMP
BridgeAllocator::NonDelegatingQueryInterface(REFIID iid, void** ppv)
{
	if (iid == IID_IMemAllocator)
	{
		return GetInterface((IMemAllocator*)this, ppv);
	} else {
		return CUnknown::NonDelegatingQueryInterface(iid, ppv);
	}
}

STDMETHODIMP
BridgeAllocator::SetProperties(
	ALLOCATOR_PROPERTIES* pRequest,
	ALLOCATOR_PROPERTIES* pActual)
{
	// requests are passed to the downstream allocator by
	// the BridgeSourceOutput pin when the downstream graph
	// is being built.
	CAutoLock lock(&m_csAlloc);
	m_props = *pRequest;

    // for uncompressed video, the buffer size will change as the output type changes
    // so we accept any properties
    HRESULT hr = S_OK;
    if (!m_pPin->GetStream()->IsVideo())
    {
        // for audio, we should tell the caller what buffers are actually in use and
        // allow them to reject it.
        long cBuffers, cBytes;
        if (m_pPin->GetStream()->GetDownstreamBufferProps(&cBuffers, &cBytes))
        {
            m_props.cbBuffer = cBytes;
            m_props.cBuffers = cBuffers;
            if (cBytes < pRequest->cbBuffer)
            {
                hr = VFW_E_BUFFER_UNDERFLOW;
            }
        }
    }
	*pActual = m_props;

	return hr;
}

STDMETHODIMP
BridgeAllocator::GetProperties(ALLOCATOR_PROPERTIES *pProps)
{
	// it's probably best to pass on the actual props if possible
	CAutoLock lock(&m_csAlloc);
	HRESULT hr = S_OK;
	if (m_pTarget == NULL)
	{
		*pProps = m_props;
	} else {
		hr = m_pTarget->GetProperties(pProps);
	}
	return hr;
}

STDMETHODIMP
BridgeAllocator::ReleaseBuffer(IMediaSample *pSample)
{
	UNREFERENCED_PARAMETER(pSample);
	// called via sample's pointer to originating allocator -- so
	// our implementation will never be called.
	return S_OK;
}

STDMETHODIMP
BridgeAllocator::GetBuffer(
	IMediaSample **ppBuffer,
	REFERENCE_TIME *pStart,
	REFERENCE_TIME *pEnd,
	DWORD dwFlags)
{
	LOG((TEXT("GetBuffer on sink pin 0x%x"), m_pPin));

    IProxySamplePtr pProxy;

	// block until connected and committed
	for(;;)
	{
		// wait on the event, then grab the locks in
		// the right order. Then check that we are
		// connected/committed, and if not, release the locks and try again
		m_evNonBlocking.Wait();

        // create a proxy -- locks the pin
        HRESULT hr = S_OK;
        pProxy = new ProxySample(m_pPin, &hr);

		// target-alloc critsec
		CAutoLock lock(&m_csAlloc);

		// committed?
		if (!m_bCommitted)
		{
			return VFW_E_NOT_COMMITTED;
		}

		// connected?
		if (m_pTarget != NULL)
		{
			break;
		}
        pProxy = NULL;
	}

    HRESULT hr = S_OK;
    if (m_bSwitchConnection)
    {
        hr = m_pPin->GetStream()->SwitchTo(&m_mtDTC);
        if (SUCCEEDED(hr))
        {
            m_bForceDTC = true;
            m_bSwitchConnection = false;
        }
    }

	// call target allocator
	// target cannot change while we hold the semaphore
    IMediaSamplePtr pSample;
    if (SUCCEEDED(hr))
    {
        hr = m_pTarget->GetBuffer(&pSample, pStart, pEnd, dwFlags);
    }

	// dynamic type changes?
	if (SUCCEEDED(hr))
	{
		CAutoLock lock(&m_csAlloc);

		// check for dynamic type change from downstream
        AM_MEDIA_TYPE* pmt;
        if (pSample->GetMediaType(&pmt) == S_OK)
        {
			CMediaType mt(*pmt);
			DeleteMediaType(pmt);

			// notify controller that this sample has a type change
			// (the source will normally clear it before calling our Receive method).
			// If we were actually processing the data within the sink, we would
			// need to switch type when this exact sample reappears at the input
			// (although by then the media type will have been removed from the sample).
			// However, for our purposes, we can switch now.
			m_pPin->SetMediaType(&mt);
		} else if (m_bForceDTC)
		{
			// initiate a dynamic type change ourselves
			pSample->SetMediaType(&m_mtDTC);

            // attach to our proxy so that we can detect this type change
            // on the way downstream even if the upstream filter erases it
            pProxy->SetType(&m_mtDTC);

            // if this is a switch to a new format, tell the pin & stream
			m_pPin->SetMediaType(&m_mtDTC);
		}
		m_bForceDTC = false;
	}

	if (SUCCEEDED(hr))
	{
        pProxy->SetInner(pSample);
        IMediaSamplePtr pOuter = pProxy;
        *ppBuffer = pOuter.Detach();
	}

	if (hr != S_OK)
	{
		LOG((TEXT("GetBuffer on pin 0x%x returns 0x%x"), m_pPin, hr));
	}

	return hr;
}

STDMETHODIMP
BridgeAllocator::Commit()
{
	// ensure that we block when active but disconnected
	CAutoLock lock(&m_csAlloc);
	m_bCommitted = true;
	if (m_pTarget == NULL)
	{
		m_evNonBlocking.Reset();
	}
	LOG((TEXT("BridgeAlloc 0x%x commit, %s"), this, m_pTarget==NULL?TEXT("Disconnected"):TEXT("Connected")));
	return S_OK;
}

STDMETHODIMP
BridgeAllocator::Decommit()
{
	// ensure that we block when active but disconnected
	// -- we are now inactive
	CAutoLock lock(&m_csAlloc);
	m_bCommitted = false;
	m_evNonBlocking.Set();
	LOG((TEXT("BridgeAlloc 0x%x decommit, %s"), this, m_pTarget==NULL?TEXT("Disconnected"):TEXT("Connected")));
	return S_OK;
}


HRESULT
BridgeAllocator::SetDownstreamAlloc(IMemAllocator* pAlloc)
{
	// lock cs *after* semaphore
	CAutoLock lock(&m_csAlloc);

	// target allocator -- could be null if disconnecting
	m_pTarget = pAlloc;

	// ensure non-blocking when connected or not active
	if ((m_pTarget != NULL) || !m_bCommitted)
	{
		m_evNonBlocking.Set();
	} else {
		m_evNonBlocking.Reset();
	}
	return S_OK;
}


// -- sample proxy implementation ------------------

ProxySample::ProxySample(BridgeSinkInput* pPin, HRESULT* phr)
: CUnknown(NAME("ProxySample"), NULL, phr),
  m_pPin(pPin),
  m_bDTC(false)
{
    // increment lock for each outstanding buffer
    m_pPin->LockIncremental();
}

ProxySample::~ProxySample()
{
    // release lock on deletion
    m_pPin->Unlock();
}

STDMETHODIMP
ProxySample::NonDelegatingQueryInterface(REFIID iid, void** ppv)
{
    if ((iid == IID_IMediaSample) || (iid == IID_IMediaSample2))
    {
        return GetInterface((IMediaSample2*)this, ppv);
    }
    else if (iid == __uuidof(IProxySample))
    {
        return GetInterface((IProxySample*)this, ppv);
    }
    else
    {
        return CUnknown::NonDelegatingQueryInterface(iid, ppv);
    }
}

STDMETHODIMP
ProxySample::SetInner(IMediaSample* pSample)
{
    m_pInner = pSample;
    return S_OK;
}

STDMETHODIMP
ProxySample::GetInner(IMediaSample** ppSample)
{
    *ppSample = m_pInner;
    if (m_pInner != NULL)
    {
        m_pInner->AddRef();
        return S_OK;
    }
    return S_FALSE;
}

STDMETHODIMP
ProxySample::ReleaseInner()
{
    m_pInner = NULL;
    return S_OK;
}

STDMETHODIMP
ProxySample::SetType(const CMediaType* pType)
{
    m_mtDTC = *pType;
    m_bDTC = true;
    return S_OK;
}

STDMETHODIMP
ProxySample::GetType(CMediaType* pType)
{
    if (m_bDTC)
    {
        *pType = m_mtDTC;
        return S_OK;
    }
    return S_FALSE;
}

STDMETHODIMP
ProxySample::GetPointer(BYTE ** ppBuffer)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->GetPointer(ppBuffer);
}

STDMETHODIMP_(LONG)
ProxySample::GetSize(void)
{
    if (m_pInner == NULL)
    {
        return 0;
    }
    return m_pInner->GetSize();
}

STDMETHODIMP
ProxySample::GetTime(
    REFERENCE_TIME * pTimeStart,     // put time here
    REFERENCE_TIME * pTimeEnd
)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->GetTime(pTimeStart, pTimeEnd);
}

STDMETHODIMP
ProxySample::SetTime(
    REFERENCE_TIME * pTimeStart,     // put time here
    REFERENCE_TIME * pTimeEnd
)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->SetTime(pTimeStart, pTimeEnd);
}

STDMETHODIMP
ProxySample::IsSyncPoint(void)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->IsSyncPoint();
}

STDMETHODIMP
ProxySample::SetSyncPoint(BOOL bIsSyncPoint)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->SetSyncPoint(bIsSyncPoint);
}

STDMETHODIMP
ProxySample::IsPreroll(void)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->IsPreroll();
}

STDMETHODIMP
ProxySample::SetPreroll(BOOL bIsPreroll)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->SetPreroll(bIsPreroll);
}

STDMETHODIMP_(LONG)
ProxySample::GetActualDataLength(void)
{
    if (m_pInner == NULL)
    {
        return 0;
    }
    return m_pInner->GetActualDataLength();
}

STDMETHODIMP
ProxySample::SetActualDataLength(LONG lActual)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->SetActualDataLength(lActual);
}

STDMETHODIMP
ProxySample::GetMediaType(AM_MEDIA_TYPE **ppMediaType)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->GetMediaType(ppMediaType);
}

STDMETHODIMP
ProxySample::SetMediaType(AM_MEDIA_TYPE *pMediaType)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->SetMediaType(pMediaType);
}

STDMETHODIMP
ProxySample::IsDiscontinuity(void)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->IsDiscontinuity();
}

STDMETHODIMP
ProxySample::SetDiscontinuity(BOOL bDiscontinuity)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->SetDiscontinuity(bDiscontinuity);
}

STDMETHODIMP
ProxySample::GetMediaTime(
    LONGLONG * pTimeStart,
    LONGLONG * pTimeEnd
)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->GetMediaTime(pTimeStart, pTimeEnd);
}

STDMETHODIMP
ProxySample::SetMediaTime(
    LONGLONG * pTimeStart,
    LONGLONG * pTimeEnd
)
{
    if (m_pInner == NULL)
    {
        return E_NOINTERFACE;
    }
    return m_pInner->SetMediaTime(pTimeStart, pTimeEnd);
}

STDMETHODIMP
ProxySample::GetProperties(
    DWORD cbProperties,
    BYTE * pbProperties
)
{
    IMediaSample2Ptr p2 = m_pInner;
    if (p2 == NULL)
    {
        return E_NOINTERFACE;
    }
    return p2->GetProperties(cbProperties, pbProperties);
}

STDMETHODIMP
ProxySample::SetProperties(
    DWORD cbProperties,
    const BYTE * pbProperties
)
{
    IMediaSample2Ptr p2 = m_pInner;
    if (p2 == NULL)
    {
        return E_NOINTERFACE;
    }
    return p2->SetProperties(cbProperties, pbProperties);
}

