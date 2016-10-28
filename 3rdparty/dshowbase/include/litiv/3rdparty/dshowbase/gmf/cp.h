//
// These classes allow filters and other objects that
// are based on the DirectShow base classes to host
// Connection Point containers, to signal events to 
// other applications such as VB.
//
// Copyright (c) Geraint Davies, 2002. All rights reserved
//

#ifndef _OQM_CP_H_
#define _OQM_CP_H_

// debug info truncation (STL template names are very long)
#pragma warning(disable:4786)

#include <map>

// Implements Connection Points for outgoing event notifications
// for DirectShow filters.

class ConnectionPoint;      // a connection point for a specific outgoing interface
class CPEnumConnections;    // enumerates connections to a specific ConnectionPoint
class EnumCP;               // enumerates all the Connection Points in a container
class CPContainer;          // contains all the Connection Points on an object

// implements the connection point for a specific outgoing interface
class ConnectionPoint : public CUnknown, public IConnectionPoint
{
public:
    ConnectionPoint(CPContainer* pContainer, IUnknown* pUnk, HRESULT* phr);

    DECLARE_IUNKNOWN
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv);
    
    STDMETHOD(GetConnectionInterface)(IID *pIID);
    STDMETHOD(GetConnectionPointContainer)(IConnectionPointContainer  **ppCPC);
    STDMETHOD(Advise)(IUnknown *pUnkSink, DWORD *pdwCookie);
    STDMETHOD(Unadvise)(DWORD dwCookie);
    STDMETHOD(EnumConnections)(IEnumConnections** ppEnum);

    typedef std::map<DWORD, IUnknown*> map;
    typedef map::iterator iterator;

    iterator begin() {
        return m_mapConnects.begin();
    }
    iterator end() {
        return m_mapConnects.end();
    }
private:
    CPContainer* m_pContainer;
    DWORD m_dwNext;
    IID m_IID;
    map m_mapConnects;
};

class CPEnumConnections : public CUnknown, public IEnumConnections
{
public:
    CPEnumConnections(HRESULT* phr, ConnectionPoint* pPoint, ConnectionPoint::iterator where);
    ~CPEnumConnections();

    DECLARE_IUNKNOWN
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv);

    STDMETHOD(Next)(ULONG cConnections, LPCONNECTDATA rgcd, ULONG *pcFetched);
    STDMETHOD(Skip)(ULONG cConnections);
    STDMETHOD(Reset)();
    STDMETHOD(Clone)(IEnumConnections **ppEnum);
private:
    ConnectionPoint* m_pPoint;
    ConnectionPoint::iterator m_where;
};


class IIDCompare
{
public:
    bool operator()(const IID& Left, const IID& Right) const
    {
        if (memcmp(&Left, &Right, sizeof(IID)) < 0) {
            return true;
        } 
        return false;
    }
};


class CPContainer : public CUnknown, public IConnectionPointContainer
{
public:
    CPContainer(IUnknown* pUnk, HRESULT* phr);
    ~CPContainer();

    DECLARE_IUNKNOWN
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv);

    // interface methods
    STDMETHOD(EnumConnectionPoints)(IEnumConnectionPoints** ppEnum);
    STDMETHOD(FindConnectionPoint)(REFIID riid, IConnectionPoint** ppCP);

    // public methods
    HRESULT AddCP(REFIID iid);
    
    typedef std::map<IID, ConnectionPoint*, IIDCompare> map;
    typedef map::iterator iterator;

    iterator begin() {
        return m_mapPoints.begin();
    }
    iterator end() {
        return m_mapPoints.end();
    }
    iterator find(REFIID iid) {
        return m_mapPoints.find(iid);
    }
private:
    map m_mapPoints;
};

class EnumCP : public CUnknown, public IEnumConnectionPoints
{
public:
    EnumCP(HRESULT* phr, CPContainer* pContainer, CPContainer::iterator where);
    ~EnumCP();

    DECLARE_IUNKNOWN
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv);
    
    STDMETHOD(Next)(ULONG cConnections, LPCONNECTIONPOINT *ppCP, ULONG *pcFetched);
    STDMETHOD(Skip)(ULONG cConnections);
    STDMETHOD(Reset)(void);
    STDMETHOD(Clone)(IEnumConnectionPoints **ppEnum);
private:
    CPContainer* m_pContainer;
    CPContainer::iterator m_where;
};



#endif // _OQM_CP_H_
