//
// GDCL Multigraph Framework
//
// Sink.h: class declaration for Bridge sink filter and input pin in source graphs
//
// Copyright (c) GDCL 2004. All Rights Reserved. 
// You are free to re-use this as the basis for your own filter development,
// provided you retain this copyright notice in the source.
// http://www.gdcl.co.uk

#pragma once

class BridgeAllocator;
class BridgeSinkInput;
class BridgeSink;

// custom interface on BridgeSink filter, used by 
// controller
MIDL_INTERFACE("C6E8B9B1-F146-47aa-83FF-ACCFAE7B38B3") IProxySample;
DECLARE_INTERFACE_(IProxySample, IUnknown)
{
    STDMETHOD(SetInner)(THIS_ IMediaSample* pSample) PURE;
    STDMETHOD(GetInner)(THIS_ IMediaSample** ppSample) PURE;
    STDMETHOD(ReleaseInner)(THIS) PURE;
    STDMETHOD(SetType)(THIS_ const CMediaType* pType) PURE;
    STDMETHOD(GetType)(THIS_ CMediaType* pType) PURE;
};
_COM_SMARTPTR_TYPEDEF(IProxySample, __uuidof(IProxySample));


// We need to lock the bridge when any downstream buffers are
// held by the upstream graph. This means we must proxy the IMediaSample
// objects so that we can detect when they are released instead of being
// given to Receive (even though we proxy the allocator interface, the samples
// always return to the original allocator when released.
class ProxySample 
: public CUnknown,
  public IMediaSample2,
  public IProxySample
{
public:
    ProxySample(BridgeSinkInput* pPin, HRESULT* phr);
    ~ProxySample();

    // IProxySample
    STDMETHOD(SetInner)(THIS_ IMediaSample* pSample);
    STDMETHOD(GetInner)(THIS_ IMediaSample** ppSample);
    STDMETHOD(ReleaseInner)();
    STDMETHOD(SetType)(THIS_ const CMediaType* pType);
    STDMETHOD(GetType)(THIS_ CMediaType* pType);

    DECLARE_IUNKNOWN
    STDMETHOD(NonDelegatingQueryInterface)(REFIID iid, void** ppv);

// IMediaSample & IMediaSample2 methods
public:
    STDMETHODIMP GetPointer(BYTE ** ppBuffer);
    STDMETHODIMP_(LONG) GetSize(void);
    STDMETHODIMP GetTime(
        REFERENCE_TIME * pTimeStart,     // put time here
        REFERENCE_TIME * pTimeEnd
    );
    STDMETHODIMP SetTime(
        REFERENCE_TIME * pTimeStart,     // put time here
        REFERENCE_TIME * pTimeEnd
    );
    STDMETHODIMP IsSyncPoint(void);
    STDMETHODIMP SetSyncPoint(BOOL bIsSyncPoint);
    STDMETHODIMP IsPreroll(void);
    STDMETHODIMP SetPreroll(BOOL bIsPreroll);
    STDMETHODIMP_(LONG) GetActualDataLength(void);
    STDMETHODIMP SetActualDataLength(LONG lActual);
    STDMETHODIMP GetMediaType(AM_MEDIA_TYPE **ppMediaType);
    STDMETHODIMP SetMediaType(AM_MEDIA_TYPE *pMediaType);
    STDMETHODIMP IsDiscontinuity(void);
    STDMETHODIMP SetDiscontinuity(BOOL bDiscontinuity);
    STDMETHODIMP GetMediaTime(
    	LONGLONG * pTimeStart,
        LONGLONG * pTimeEnd
    );
    STDMETHODIMP SetMediaTime(
    	LONGLONG * pTimeStart,
        LONGLONG * pTimeEnd
    );
    STDMETHODIMP GetProperties(
        DWORD cbProperties,
        BYTE * pbProperties
    );
    STDMETHODIMP SetProperties(
        DWORD cbProperties,
        const BYTE * pbProperties
    );


private:
    BridgeSinkInput* m_pPin;
    IMediaSamplePtr m_pInner;

    // track DTC even if the type is erased by upstream
    CMediaType m_mtDTC;
    bool m_bDTC;
};


// allocator used on sink inputs. This allocator
// redirects all calls to the allocator used by the 
// BridgeSource output pin.
class BridgeAllocator 
: public CUnknown,
  public IMemAllocator
{
public:
	BridgeAllocator(BridgeSinkInput* pPin);

	DECLARE_IUNKNOWN
	STDMETHOD(NonDelegatingQueryInterface)(REFIID iid, void** ppv);

// IMemAllocator interface
    STDMETHODIMP SetProperties(
		    ALLOCATOR_PROPERTIES* pRequest,
		    ALLOCATOR_PROPERTIES* pActual);

    STDMETHODIMP ReleaseBuffer(IMediaSample *pSample);
    STDMETHODIMP GetBuffer(IMediaSample **ppBuffer,
                           REFERENCE_TIME *pStart,
                           REFERENCE_TIME *pEnd,
                           DWORD dwFlags);
    STDMETHODIMP GetProperties(ALLOCATOR_PROPERTIES *pProps);
    STDMETHODIMP Commit(void);
    STDMETHODIMP Decommit(void);

	HRESULT SetDownstreamAlloc(IMemAllocator* pAlloc);

	void ForceDTC(CMediaType* pmt)
	{
		m_mtDTC = *pmt;
		m_bForceDTC = true;
	}

    void SwitchFormatTo(const CMediaType* pmt)
    {
        m_mtDTC = *pmt;
        m_bSwitchConnection = true;
    }

private:
	// sink input pin that owns this allocator
	BridgeSinkInput* m_pPin;

	// protects changes to target allocator
	// Note: MUST get semaphore *before* m_csAlloc
	CCritSec m_csAlloc;
	IMemAllocatorPtr m_pTarget;
	ALLOCATOR_PROPERTIES m_props;

	// true if allocator is committed
	bool m_bCommitted;

	// set when we should not block (when connected, or when decommitted)
	CAMEvent m_evNonBlocking;

	// force dynamic type change on next buffer
	bool m_bForceDTC;
	CMediaType m_mtDTC;

    // force switch by ReceiveConnection on next buffer (to m_mtDTC)
    bool m_bSwitchConnection;
};


// input pin on sink filter.
class BridgeSinkInput : public CBaseInputPin
{
public:
	BridgeSinkInput(
		BridgeSink* pFilter, 
		BridgeStream* pStream, 
		CCritSec* pLock, 
		HRESULT* phr, 
		LPCWSTR pName);
	~BridgeSinkInput();

    // CBaseInputPin overrides
    STDMETHODIMP Receive(IMediaSample *pSample);
    STDMETHODIMP EndOfStream(void);
    HRESULT CheckMediaType(const CMediaType* pmt);
	HRESULT SetMediaType(const CMediaType* pmt);
	HRESULT GetMediaType(int iPosition, CMediaType* pmt);
    STDMETHODIMP GetAllocator(IMemAllocator **ppAllocator);
    STDMETHODIMP NotifyAllocator(IMemAllocator * pAllocator, BOOL bReadOnly);
	STDMETHODIMP BeginFlush();
	STDMETHODIMP EndFlush();

	HRESULT Inactive();

	HRESULT MakeBridge(IMemAllocator* pAlloc);
	HRESULT DisconnectBridge();
	bool IsUncompressed(const CMediaType* pmt);
	bool IsAllowedMuxInput(const CMediaType* pmt);

	// called from downstream graph
	HRESULT GetBufferProps(long* pcBuffers, long* pcBytes);
	HRESULT CanDeliverType(const CMediaType* pmt);
	HRESULT EnumOutputType(int iPosition, CMediaType* pmt);

    // locking of connection while upstream filter is active
    // Exclusive -- make sure no other locks exist
    void LockExclusive()
    {
        WaitForSingleObject(m_hsemDelivery, INFINITE);

        // for exclusive locks, this incremental count should be 0
        ASSERT(m_nDeliveryLocks == 0);
    }

    // incremental lock -- add another buffer to the lock 
    // -- if current holder is exclusive, will block on semaphore
    void LockIncremental();

    // if incremental, decrease count and release semaphore if 0.
    // if exclusive, release semaphore (count is 0)
    void Unlock();

    // Waiting on this handle is the same as an exclusive lock
    HANDLE DeliveryLock()
	{
		return m_hsemDelivery;
	}

	BridgeSink* Filter()
	{
		return (BridgeSink*)m_pFilter;
	}
	BridgeStream* GetStream()
	{
        // ensure atomic change from one stream controller to the other
        // -- this is for a multiple input configuration, where the source graphs
        // can be connected to either input on the render graphs. All building and type checking
        // are done with the primary controller, but the source graphs can be bridged with either 
        // controller
        CAutoLock lock(&m_csStream);
		return m_pStream;
	}
    void SetStream(BridgeStream* pStream)
    {
        // called from bridgestream when a bridge is made to ensure that we refer to the correct controller set.
        CAutoLock lock(&m_csStream);
        m_pStream = pStream;
    }
    void CurrentType(CMediaType* pmt)
    {
        *pmt = m_mt;
    }
private:
	HRESULT CopySample(IMediaSample* pIn, IMediaSample* pOut, int cOffset, int cLength);

private:
    // csStream is used to ensure that changes to the stream pointer are atomic
    CCritSec m_csStream;
	BridgeStream* m_pStream;

	BridgeAllocator* m_pRedirectedAlloc;
	bool m_bAudio;
	CCritSec m_csConnect;	// protect connection state
	bool m_bConnected;
	IMemAllocatorPtr m_pCopyAllocator;
    bool m_bUsingProxyAllocator;

	// held during GetBuffer -> Receive delivery cycle
	HANDLE m_hsemDelivery;
    // to allow repeated buffer locks while holding semaphore
    CCritSec m_csDelivery;
    long m_nDeliveryLocks;

    // true if we have a type change from upstream that should be passed on downstream
    bool m_bSendDTC;

	// true if we stopped while connected - this requires timestamp
	// management in the source output similar to a new connection
	bool m_bStoppedWhileConnected;
};

// custom interface on BridgeSink filter, used by 
// controller
MIDL_INTERFACE("8F676D73-0824-41d0-96A7-BF6B0D8DE86B") IBridgeSink;
DECLARE_INTERFACE_(IBridgeSink, IUnknown)
{
    STDMETHOD_(int, GetBridgePinCount)(THIS_) PURE;
    STDMETHOD(GetBridgePin)(THIS_ 
		int nPin,
		/* [out] */ BridgeSinkInput** ppPin) PURE;
	STDMETHOD_(BOOL, IsAtEOS)(THIS_) PURE;
};
_COM_SMARTPTR_TYPEDEF(IBridgeSink, __uuidof(IBridgeSink));

// 
// sink filter, receives data for delivery
// in another graph.
class BridgeSink 
: public CBaseFilter, 
  public IBridgeSink,
  public IMediaSeeking
{
public:
	BridgeSink(BridgeController* pController);

	// support seeking on the source graph segment -- redirected upstream
	DECLARE_IUNKNOWN
	STDMETHODIMP NonDelegatingQueryInterface(REFIID iid, void** ppv);

	int GetPinCount();
	CBasePin* GetPin(int n);

	void OnEOS(bool bConnected);
	void ResetEOSCount();
	STDMETHODIMP Stop();

	// in discard mode, first sample after
	// discard needs to be zero -- this is handled in filter
	// to ensure same offset is applied to multiple streams
	void Discard();
	void AdjustTime(IMediaSample* pIn);
    REFERENCE_TIME STO()
    {
        return m_tStart;
    }

	// override to debug graph via GraphEdt 
    STDMETHODIMP JoinFilterGraph(IFilterGraph * pGraph, LPCWSTR pName);
// IBridgeSink
public:
    STDMETHODIMP GetBridgePin(int nPin, BridgeSinkInput** ppPin);
    STDMETHODIMP_(int) GetBridgePinCount()
    {
        return GetPinCount();
    }
	STDMETHOD_(BOOL, IsAtEOS)();

// IMediaSeeking
public:
    STDMETHODIMP GetCapabilities(DWORD * pCapabilities );
    STDMETHODIMP CheckCapabilities(DWORD * pCapabilities );
    STDMETHODIMP IsFormatSupported(const GUID * pFormat);
    STDMETHODIMP QueryPreferredFormat(GUID * pFormat);
    STDMETHODIMP GetTimeFormat(GUID *pFormat);
    STDMETHODIMP IsUsingTimeFormat(const GUID * pFormat);
    STDMETHODIMP SetTimeFormat(const GUID * pFormat);
    STDMETHODIMP GetDuration(LONGLONG *pDuration);
    STDMETHODIMP GetStopPosition(LONGLONG *pStop);
    STDMETHODIMP GetCurrentPosition(LONGLONG *pCurrent);
    STDMETHODIMP ConvertTimeFormat(LONGLONG * pTarget, const GUID * pTargetFormat,
                              LONGLONG    Source, const GUID * pSourceFormat );
    STDMETHODIMP SetPositions(LONGLONG * pCurrent, DWORD dwCurrentFlags
			, LONGLONG * pStop, DWORD dwStopFlags );
    STDMETHODIMP GetPositions(LONGLONG * pCurrent,
                              LONGLONG * pStop );
    STDMETHODIMP GetAvailable(LONGLONG * pEarliest, LONGLONG * pLatest );
    STDMETHODIMP SetRate(double dRate);
    STDMETHODIMP GetRate(double * pdRate);
    STDMETHODIMP GetPreroll(LONGLONG * pllPreroll);


private:
	CCritSec m_csFilter;
    int m_nPins;
	typedef smart_ptr<BridgeSinkInput> BridgeSinkInputPtr;
    smart_array<BridgeSinkInputPtr> m_pPins;

	CCritSec m_csEOS;
	long m_nEOS;
	REFERENCE_TIME m_tFirst;
	bool m_bLastDiscarded;
	DWORD m_dwROT;
};

