#ifndef _BASEIDISPATCH_H_
#define _BASEIDISPATCH_H_


// IDispatch template based on ATL templates
// that implements the common methods of a
// 'dual' interface by calls to the custom methods
template <class intfc, const IID* iid>
class BaseIDispatch : public intfc
{
public:
    STDMETHOD(GetTypeInfoCount)(UINT* pctinfo) {
        *pctinfo = 1;
        return S_OK;
    }
    STDMETHOD(GetTypeInfo)( 
            UINT iTInfo,
            LCID lcid,
            ITypeInfo** ppTInfo)
    {
		UNREFERENCED_PARAMETER(lcid);
		UNREFERENCED_PARAMETER(iTInfo);
#ifdef UNICODE
        TCHAR wch[MAX_PATH];
        GetModuleFileName(g_hInst, wch, MAX_PATH);
#else
		TCHAR  achPath[MAX_PATH];
        WCHAR wch[MAX_PATH];
        GetModuleFileName(g_hInst, achPath, MAX_PATH);
        MultiByteToWideChar(CP_ACP, 0, achPath, -1, wch, MAX_PATH);
#endif
        ITypeLib* ptl = NULL;
        HRESULT hr = LoadTypeLib(wch, &ptl);
        if (SUCCEEDED(hr)) {
            hr = ptl->GetTypeInfoOfGuid(*iid, ppTInfo);

            ptl->Release();
        }
        return hr;

    }
        
    STDMETHOD(GetIDsOfNames)( 
            REFIID riid,
            LPOLESTR* rgszNames,
            UINT cNames,
            LCID lcid,
            DISPID* rgDispId)
    {
		UNREFERENCED_PARAMETER(riid);
        ITypeInfo* pti;
        HRESULT hr = GetTypeInfo(0, lcid, &pti);
        if (SUCCEEDED(hr)) {
            hr = pti->GetIDsOfNames(rgszNames, cNames, rgDispId);
            pti->Release();
        }
        return hr;
    }
    STDMETHOD(Invoke)( 
            DISPID dispIdMember,
            REFIID riid,
            LCID lcid,
            WORD wFlags,
            DISPPARAMS* pDispParams,
            VARIANT* pVarResult,
            EXCEPINFO* pExcepInfo,
            UINT* puArgErr)
    {
		UNREFERENCED_PARAMETER(riid);
        ITypeInfo* pti;
        HRESULT hr = GetTypeInfo(0, lcid, &pti);
        if (SUCCEEDED(hr)) {
            hr = pti->Invoke(
                        (IDispatch*)this,
                        dispIdMember,
                        wFlags,
                        pDispParams,
                        pVarResult,
                        pExcepInfo,
                        puArgErr);
            pti->Release();
        }
        return hr;
    }
};

#endif //  _BASEIDISPATCH_H_
