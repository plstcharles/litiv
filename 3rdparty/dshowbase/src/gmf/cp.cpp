//
// Implementation of Connection Point classes for DirectShow filters,
// required to support outgoing ActiveX events
//

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
#include "litiv/3rdparty/dshowbase/gmf/cp.h"
#include <utility>

ConnectionPoint::ConnectionPoint(CPContainer* pContainer, IUnknown* pUnk, HRESULT* phr)
: m_pContainer(pContainer),
  m_dwNext(1),
  CUnknown(NAME("CP"), pUnk, phr)
{
    // don't addref container -- it owns us
}

STDMETHODIMP
ConnectionPoint::NonDelegatingQueryInterface(REFIID riid, void** ppv)
{
    if (riid == IID_IConnectionPoint) {
        return GetInterface((IConnectionPoint*) this, ppv);
    } else {
        return CUnknown::NonDelegatingQueryInterface(riid,ppv);
    }
}

STDMETHODIMP
ConnectionPoint::GetConnectionInterface(IID *pIID)
{
    CopyMemory(pIID, &m_IID, sizeof(m_IID));
    return S_OK;
}

STDMETHODIMP
ConnectionPoint::GetConnectionPointContainer(IConnectionPointContainer  **ppCPC)
{
    return m_pContainer->QueryInterface(IID_IConnectionPointContainer, (void**)ppCPC);
}

STDMETHODIMP
ConnectionPoint::Advise(IUnknown *pUnkSink, DWORD *pdwCookie)
{
    pUnkSink->AddRef();
    *pdwCookie = m_dwNext++;
    m_mapConnects.insert(std::pair<DWORD, IUnknown*>(*pdwCookie, pUnkSink));

    return S_OK;
}

STDMETHODIMP
ConnectionPoint::Unadvise(DWORD dwCookie)
{
    iterator where = m_mapConnects.find(dwCookie);
    if (where != m_mapConnects.end()) {
        where->second->Release();
        m_mapConnects.erase(where);
    }
    return S_OK;
}

STDMETHODIMP
ConnectionPoint::EnumConnections(IEnumConnections** ppEnum)
{
    HRESULT hr = S_OK;
    CPEnumConnections* pEnum = new CPEnumConnections(&hr, this, begin());
    if (SUCCEEDED(hr)) {
        hr = pEnum->QueryInterface(IID_IEnumConnections, (void**) ppEnum);
    }
    if (FAILED(hr)) {
        delete pEnum;
    }
    return hr;
}

// ---------------------

CPEnumConnections::CPEnumConnections(HRESULT* phr, ConnectionPoint* pPoint, ConnectionPoint::iterator where)
: m_pPoint(pPoint),
  m_where(where),
  CUnknown(NAME("CP"), NULL, phr)
{
    m_pPoint->AddRef();
}

CPEnumConnections::~CPEnumConnections()
{
    m_pPoint->Release();
}

STDMETHODIMP
CPEnumConnections::NonDelegatingQueryInterface(REFIID riid, void** ppv)
{
    if (riid == IID_IEnumConnections) {
        return GetInterface((IEnumConnections*) this, ppv);
    } else {
        return CUnknown::NonDelegatingQueryInterface(riid,ppv);
    }
}

STDMETHODIMP
CPEnumConnections::Next(ULONG cConnections, LPCONNECTDATA rgcd, ULONG *pcFetched)
{
    int cFetched = 0;
    HRESULT hr = S_OK;
    while (cConnections) {
        if (m_where == m_pPoint->end()) {
            hr = S_FALSE;
            break;
        } else {
            rgcd[cFetched].dwCookie = m_where->first;
            rgcd[cFetched].pUnk = m_where->second;
            rgcd[cFetched].pUnk->AddRef();

            cConnections--;
            cFetched++;
            m_where++;
        }
    }
    if (pcFetched != NULL) {
        *pcFetched = cFetched;
    }
    return hr;
}

STDMETHODIMP
CPEnumConnections::Skip(ULONG cConnections)
{
    while (cConnections) {
        if (m_where++ == m_pPoint->end()) {
            return S_FALSE;
        }
    }
    return S_OK;
}

STDMETHODIMP
CPEnumConnections::Reset()
{
    m_where = m_pPoint->begin();
    return S_OK;
}

STDMETHODIMP
CPEnumConnections::Clone(IEnumConnections **ppEnum)
{
    HRESULT hr = S_OK;
    CPEnumConnections* pClone = new CPEnumConnections(&hr, m_pPoint, m_where);
    if (SUCCEEDED(hr)) {
        hr = pClone->QueryInterface(IID_IEnumConnections, (void**)ppEnum);
    }
    if (FAILED(hr)) {
        delete pClone;
    }
    return hr;
}

// -------------------------
CPContainer::CPContainer(IUnknown* pUnk, HRESULT* phr)
: CUnknown(NAME("CPContainer"), pUnk, phr)
{
}

CPContainer::~CPContainer()
{
    for(iterator i = begin(); i != end(); i++) {
        i->second->Release();
    }
}

STDMETHODIMP
CPContainer::NonDelegatingQueryInterface(REFIID riid, void** ppv)
{
    if (riid == IID_IConnectionPointContainer) {
        return GetInterface((IConnectionPointContainer*) this, ppv);
    } else {
        return CUnknown::NonDelegatingQueryInterface(riid,ppv);
    }
}

STDMETHODIMP
CPContainer::EnumConnectionPoints(IEnumConnectionPoints** ppEnum)
{
    HRESULT hr = S_OK;
    EnumCP* pEnum = new EnumCP(&hr, this, begin());
    if (SUCCEEDED(hr)) {
        hr = pEnum->QueryInterface(IID_IEnumConnectionPoints, (void**) ppEnum);
    }
    if (FAILED(hr)) {
        delete pEnum;
    }
    return hr;
}

STDMETHODIMP
CPContainer::FindConnectionPoint(REFIID riid, IConnectionPoint** ppCP)
{
    iterator pt = m_mapPoints.find(riid);
    if (pt != m_mapPoints.end()) {
        return pt->second->QueryInterface(IID_IConnectionPoint, (void**)ppCP);
    }
    return CONNECT_E_NOCONNECTION;
}

HRESULT
CPContainer::AddCP(REFIID iid)
{
    HRESULT hr = S_OK;
    ConnectionPoint* pPt = new ConnectionPoint(this, NULL, &hr);
    if (SUCCEEDED(hr)) {
        pPt->AddRef();
        m_mapPoints.insert(std::pair<IID, ConnectionPoint*>(iid, pPt));
    } else {
        delete pPt;
    }

    return hr;
}

// ---------------------

EnumCP::EnumCP(HRESULT* phr, CPContainer* pContainer, CPContainer::iterator where)
: m_pContainer(pContainer),
  m_where(where),
  CUnknown(NAME("EnumCP"), NULL, phr)
{
    m_pContainer->AddRef();
}

EnumCP::~EnumCP()
{
    m_pContainer->Release();
}

STDMETHODIMP
EnumCP::NonDelegatingQueryInterface(REFIID riid, void** ppv)
{
    if (riid == IID_IEnumConnectionPoints) {
        return GetInterface((IEnumConnectionPoints*) this, ppv);
    } else {
        return CUnknown::NonDelegatingQueryInterface(riid,ppv);
    }
}

STDMETHODIMP
EnumCP::Next(ULONG cConnections, LPCONNECTIONPOINT *ppCP, ULONG *pcFetched)
{
    int cFetched = 0;
    HRESULT hr = S_OK;
    while (cConnections) {
        if (m_where == m_pContainer->end()) {
            hr = S_FALSE;
            break;
        } else {
            ppCP[cFetched] = m_where->second;
            ppCP[cFetched]->AddRef();
            cConnections--;
            cFetched++;
            m_where++;
        }
    }
    if (pcFetched != NULL) {
        *pcFetched = cFetched;
    }
    return hr;
}

STDMETHODIMP
EnumCP::Skip(ULONG cConnections)
{
    while (cConnections) {
        if (m_where++ == m_pContainer->end()) {
            return S_FALSE;
        }
    }
    return S_OK;
}

STDMETHODIMP
EnumCP::Reset(void)
{
    m_where = m_pContainer->begin();
    return S_OK;
}

STDMETHODIMP
EnumCP::Clone(IEnumConnectionPoints **ppEnum)
{
    HRESULT hr = S_OK;
    EnumCP* pClone = new EnumCP(&hr, m_pContainer, m_where);
    if (SUCCEEDED(hr)) {
        hr = pClone->QueryInterface(IID_IEnumConnectionPoints, (void**)ppEnum);
    }
    if (FAILED(hr)) {
        delete pClone;
    }
    return hr;
}


