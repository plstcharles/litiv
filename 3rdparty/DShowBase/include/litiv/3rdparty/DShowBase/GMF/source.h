//
// GDCL Multigraph Framework
//
// class declaration for Bridge source filter in rendering/file writing graphs
//
// Copyright (c) GDCL 2004. All Rights Reserved. 
// You are free to re-use this as the basis for your own filter development,
// provided you retain this copyright notice in the source.
// http://www.gdcl.co.uk
//

#pragma once

class BridgeSource;			// pseudo-source filter in render graph
class BridgeSourceOutput;	// output pin on source filter

// source filter's output pin delivers data
// downstream for rendering
class BridgeSourceOutput : public CBaseOutputPin
{
public:
	BridgeSourceOutput(
		BridgeSource* pFilter, 
		BridgeStream* pStream, 
		CCritSec* pLock, 
		HRESULT* phr, 
		LPCWSTR pName);

	~BridgeSourceOutput();

    // CBaseOutputPin overrides
    HRESULT DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* pprop);
    HRESULT CheckMediaType(const CMediaType* pmt);
    HRESULT GetMediaType(int iPosition, CMediaType* pmt);
    STDMETHODIMP Notify(IBaseFilter * pSender, Quality q);
    HRESULT CompleteConnect(IPin *pReceivePin);
	HRESULT DeliverBeginFlush();
	HRESULT DeliverEndFlush();
	HRESULT DeliverEndOfStream();

	HRESULT GetConnectionAllocator(IMemAllocator** ppAlloc)
	{
		if (!m_pAllocator)
		{
			return VFW_E_NO_ALLOCATOR;
		}
		*ppAlloc = m_pAllocator;
		m_pAllocator->AddRef();
		return S_OK;
	}

	HRESULT Send(IMediaSample* pSample);
	void SetBaseline(REFERENCE_TIME tBase, bool bDiscont);
	REFERENCE_TIME StopTime()
	{
		return m_tLastStop;
	}
	HRESULT Active();
	HRESULT Inactive();

	BridgeSource* Filter()
	{
		return (BridgeSource*)m_pFilter;
	}

    // format change via ReceiveConnection is only
    // available if we are connected to the video renderer
    bool IsVideoRenderer();
    bool CanDeliver(const CMediaType*pmt);
    HRESULT SwitchTo(const CMediaType*pmt);

    BridgeStream* GetStream()
    {
        CAutoLock lock(&m_csStream);
        return m_pStream;
    }
    void SetStream(BridgeStream* pStream)
    {
        CAutoLock lock(&m_csStream);
        m_pStream = pStream;
    }

	void ResetOnStop();

private:
    // ensure atomic changes to stream
    CCritSec m_csStream;
	BridgeStream* m_pStream;
	bool m_bAudio;
	bool m_bUpstreamTypeChanged;

	CCritSec m_csTime;
	REFERENCE_TIME m_tBase;
	REFERENCE_TIME m_tLastStop;

	// for media-time remapping: to cover a pause in AVI creation
	REFERENCE_TIME m_mtLast;
	REFERENCE_TIME m_mtBase;

	// increased buffering means a separate thread
	// for delivery (at least for the renderer)
	smart_ptr<COutputQueue> m_pQueue;
	bool m_bIncreasedBuffering;

	// protection against state changes
	CCritSec m_csActive;
	HANDLE m_hsemActive;
	bool m_bActive;

    // used to wait for output queue to deliver all data
    CAMEvent m_evQueue;

    // flag discont on next sample
    bool m_bDiscont;

	// for preroll detection -- true if nothing delivered
	// since last baseline change
	bool m_bNewBaseline;
};


// custom interface on BridgeSource filter, used by 
// controller
MIDL_INTERFACE("DDB9383D-5DCB-43e6-B565-262EEC9D2445") IBridgeSource;
DECLARE_INTERFACE_(IBridgeSource, IUnknown)
{
    STDMETHOD_(int, GetBridgePinCount)(THIS_) PURE;
    STDMETHOD(GetBridgePin)(THIS_ 
		int nPin,
		/* [out] */ BridgeSourceOutput** ppPin) PURE;
    STDMETHOD(GetSegmentTime)(THIS_ REFERENCE_TIME* pNow) PURE;
    STDMETHOD(OnNewConnection)(THIS_ BOOL bIsDiscont) PURE;
    STDMETHOD(GetSegmentOffset)(THIS_ REFERENCE_TIME* pOffset) PURE;
};
_COM_SMARTPTR_TYPEDEF(IBridgeSource, __uuidof(IBridgeSource));

// source filter, delivers data from selected sink filter
// in another graph.
class BridgeSource : public CBaseFilter, public IBridgeSource
{
public:
	BridgeSource(BridgeController* pController);
	
    DECLARE_IUNKNOWN
	STDMETHODIMP NonDelegatingQueryInterface(REFIID iid, void** ppv);

	int GetPinCount();
	CBasePin* GetPin(int n);

	REFERENCE_TIME RealStreamTime();
	REFERENCE_TIME SegmentStreamTime();
    REFERENCE_TIME STO()    { return m_tStart; }

	// override to debug graph via GraphEdt 
    STDMETHODIMP JoinFilterGraph(IFilterGraph * pGraph, LPCWSTR pName);

	STDMETHODIMP Run(REFERENCE_TIME tStart);
	STDMETHODIMP Pause();
    STDMETHODIMP Stop();
	void Flush();

// IBridgeSource
public:
	STDMETHODIMP GetBridgePin(int n, BridgeSourceOutput** ppPin);
    STDMETHODIMP_(int) GetBridgePinCount()
    {
        return GetPinCount();
    }
    STDMETHOD(GetSegmentTime)(REFERENCE_TIME* pNow)
    {
        *pNow = SegmentStreamTime();
        return S_OK;
    }
    STDMETHOD(OnNewConnection)(BOOL bIsDiscont);
    STDMETHOD(GetSegmentOffset)(REFERENCE_TIME* pOffset)
    {
        *pOffset = m_tBase;
        return S_OK;
    }
private:
	CCritSec m_csFilter;
    int m_nPins;
	typedef smart_ptr<BridgeSourceOutput> BridgeSourceOutputPtr;
    smart_array<BridgeSourceOutputPtr> m_pPins;
	REFERENCE_TIME m_tBase;
	REFERENCE_TIME m_tPausedAt;
	DWORD m_dwROT;
};
