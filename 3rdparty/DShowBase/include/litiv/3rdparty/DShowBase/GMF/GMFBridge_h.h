

/* this ALWAYS GENERATED file contains the definitions for the interfaces */


 /* File created by MIDL compiler version 7.00.0555 */
/* at Fri Jul 27 11:20:53 2012
 */
/* Compiler settings for GMFBridge.idl:
    Oicf, W1, Zp8, env=Win32 (32b run), target_arch=X86 7.00.0555 
    protocol : dce , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data 
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
/* @@MIDL_FILE_HEADING(  ) */

#pragma warning( disable: 4049 )  /* more than 64k source lines */


/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif // __RPCNDR_H_VERSION__

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __GMFBridge_h_h__
#define __GMFBridge_h_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */ 

#ifndef __IGMFBridgeController_FWD_DEFINED__
#define __IGMFBridgeController_FWD_DEFINED__
typedef interface IGMFBridgeController IGMFBridgeController;
#endif 	/* __IGMFBridgeController_FWD_DEFINED__ */


#ifndef __IGMFBridgeController2_FWD_DEFINED__
#define __IGMFBridgeController2_FWD_DEFINED__
typedef interface IGMFBridgeController2 IGMFBridgeController2;
#endif 	/* __IGMFBridgeController2_FWD_DEFINED__ */


#ifndef __IGMFBridgeController3_FWD_DEFINED__
#define __IGMFBridgeController3_FWD_DEFINED__
typedef interface IGMFBridgeController3 IGMFBridgeController3;
#endif 	/* __IGMFBridgeController3_FWD_DEFINED__ */


#ifndef ___IGMFBridgeEvents_FWD_DEFINED__
#define ___IGMFBridgeEvents_FWD_DEFINED__
typedef interface _IGMFBridgeEvents _IGMFBridgeEvents;
#endif 	/* ___IGMFBridgeEvents_FWD_DEFINED__ */


#ifndef __GMFBridgeController_FWD_DEFINED__
#define __GMFBridgeController_FWD_DEFINED__

#ifdef __cplusplus
typedef class GMFBridgeController GMFBridgeController;
#else
typedef struct GMFBridgeController GMFBridgeController;
#endif /* __cplusplus */

#endif 	/* __GMFBridgeController_FWD_DEFINED__ */


/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C"{
#endif 


/* interface __MIDL_itf_GMFBridge_0000_0000 */
/* [local] */ 

typedef /* [public][public] */ 
enum __MIDL___MIDL_itf_GMFBridge_0000_0000_0001
    {	eUncompressed	= 0,
	eMuxInputs	= ( eUncompressed + 1 ) ,
	eAny	= ( eMuxInputs + 1 ) 
    } 	eFormatType;



extern RPC_IF_HANDLE __MIDL_itf_GMFBridge_0000_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_GMFBridge_0000_0000_v0_0_s_ifspec;

#ifndef __IGMFBridgeController_INTERFACE_DEFINED__
#define __IGMFBridgeController_INTERFACE_DEFINED__

/* interface IGMFBridgeController */
/* [unique][helpstring][dual][uuid][object] */ 


EXTERN_C const IID IID_IGMFBridgeController;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("8C4D8054-FCBA-4783-865A-7E8B3C814011")
    IGMFBridgeController : public IDispatch
    {
    public:
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE AddStream( 
            BOOL bVideo,
            eFormatType AllowedTypes,
            BOOL bDiscardUnconnected) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE InsertSinkFilter( 
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **ppFilter) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE InsertSourceFilter( 
            /* [in] */ IUnknown *pUnkSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **ppFilter) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE CreateSourceGraph( 
            /* [in] */ BSTR strFile,
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **pSinkFilter) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE CreateRenderGraph( 
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **pRenderGraphSourceFilter) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE BridgeGraphs( 
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraphSourceFilter) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE SetNotify( 
            /* [in] */ LONG_PTR hwnd,
            /* [in] */ long msg) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE SetBufferMinimum( 
            /* [in] */ long nMillisecs) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetSegmentTime( 
            /* [retval][out] */ double *pdSeconds) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE NoMoreSegments( void) = 0;
        
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetSegmentOffset( 
            /* [retval][out] */ double *pdOffset) = 0;
        
    };
    
#else 	/* C style interface */

    typedef struct IGMFBridgeControllerVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            IGMFBridgeController * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            __RPC__deref_out  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            IGMFBridgeController * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            IGMFBridgeController * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfoCount )( 
            IGMFBridgeController * This,
            /* [out] */ UINT *pctinfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfo )( 
            IGMFBridgeController * This,
            /* [in] */ UINT iTInfo,
            /* [in] */ LCID lcid,
            /* [out] */ ITypeInfo **ppTInfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetIDsOfNames )( 
            IGMFBridgeController * This,
            /* [in] */ REFIID riid,
            /* [size_is][in] */ LPOLESTR *rgszNames,
            /* [range][in] */ UINT cNames,
            /* [in] */ LCID lcid,
            /* [size_is][out] */ DISPID *rgDispId);
        
        /* [local] */ HRESULT ( STDMETHODCALLTYPE *Invoke )( 
            IGMFBridgeController * This,
            /* [in] */ DISPID dispIdMember,
            /* [in] */ REFIID riid,
            /* [in] */ LCID lcid,
            /* [in] */ WORD wFlags,
            /* [out][in] */ DISPPARAMS *pDispParams,
            /* [out] */ VARIANT *pVarResult,
            /* [out] */ EXCEPINFO *pExcepInfo,
            /* [out] */ UINT *puArgErr);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *AddStream )( 
            IGMFBridgeController * This,
            BOOL bVideo,
            eFormatType AllowedTypes,
            BOOL bDiscardUnconnected);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *InsertSinkFilter )( 
            IGMFBridgeController * This,
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **ppFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *InsertSourceFilter )( 
            IGMFBridgeController * This,
            /* [in] */ IUnknown *pUnkSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **ppFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *CreateSourceGraph )( 
            IGMFBridgeController * This,
            /* [in] */ BSTR strFile,
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **pSinkFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *CreateRenderGraph )( 
            IGMFBridgeController * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **pRenderGraphSourceFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *BridgeGraphs )( 
            IGMFBridgeController * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraphSourceFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *SetNotify )( 
            IGMFBridgeController * This,
            /* [in] */ LONG_PTR hwnd,
            /* [in] */ long msg);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *SetBufferMinimum )( 
            IGMFBridgeController * This,
            /* [in] */ long nMillisecs);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *GetSegmentTime )( 
            IGMFBridgeController * This,
            /* [retval][out] */ double *pdSeconds);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *NoMoreSegments )( 
            IGMFBridgeController * This);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *GetSegmentOffset )( 
            IGMFBridgeController * This,
            /* [retval][out] */ double *pdOffset);
        
        END_INTERFACE
    } IGMFBridgeControllerVtbl;

    interface IGMFBridgeController
    {
        CONST_VTBL struct IGMFBridgeControllerVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define IGMFBridgeController_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define IGMFBridgeController_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define IGMFBridgeController_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define IGMFBridgeController_GetTypeInfoCount(This,pctinfo)	\
    ( (This)->lpVtbl -> GetTypeInfoCount(This,pctinfo) ) 

#define IGMFBridgeController_GetTypeInfo(This,iTInfo,lcid,ppTInfo)	\
    ( (This)->lpVtbl -> GetTypeInfo(This,iTInfo,lcid,ppTInfo) ) 

#define IGMFBridgeController_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)	\
    ( (This)->lpVtbl -> GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) ) 

#define IGMFBridgeController_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)	\
    ( (This)->lpVtbl -> Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) ) 


#define IGMFBridgeController_AddStream(This,bVideo,AllowedTypes,bDiscardUnconnected)	\
    ( (This)->lpVtbl -> AddStream(This,bVideo,AllowedTypes,bDiscardUnconnected) ) 

#define IGMFBridgeController_InsertSinkFilter(This,pGraph,ppFilter)	\
    ( (This)->lpVtbl -> InsertSinkFilter(This,pGraph,ppFilter) ) 

#define IGMFBridgeController_InsertSourceFilter(This,pUnkSourceGraphSinkFilter,pRenderGraph,ppFilter)	\
    ( (This)->lpVtbl -> InsertSourceFilter(This,pUnkSourceGraphSinkFilter,pRenderGraph,ppFilter) ) 

#define IGMFBridgeController_CreateSourceGraph(This,strFile,pGraph,pSinkFilter)	\
    ( (This)->lpVtbl -> CreateSourceGraph(This,strFile,pGraph,pSinkFilter) ) 

#define IGMFBridgeController_CreateRenderGraph(This,pSourceGraphSinkFilter,pRenderGraph,pRenderGraphSourceFilter)	\
    ( (This)->lpVtbl -> CreateRenderGraph(This,pSourceGraphSinkFilter,pRenderGraph,pRenderGraphSourceFilter) ) 

#define IGMFBridgeController_BridgeGraphs(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter)	\
    ( (This)->lpVtbl -> BridgeGraphs(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter) ) 

#define IGMFBridgeController_SetNotify(This,hwnd,msg)	\
    ( (This)->lpVtbl -> SetNotify(This,hwnd,msg) ) 

#define IGMFBridgeController_SetBufferMinimum(This,nMillisecs)	\
    ( (This)->lpVtbl -> SetBufferMinimum(This,nMillisecs) ) 

#define IGMFBridgeController_GetSegmentTime(This,pdSeconds)	\
    ( (This)->lpVtbl -> GetSegmentTime(This,pdSeconds) ) 

#define IGMFBridgeController_NoMoreSegments(This)	\
    ( (This)->lpVtbl -> NoMoreSegments(This) ) 

#define IGMFBridgeController_GetSegmentOffset(This,pdOffset)	\
    ( (This)->lpVtbl -> GetSegmentOffset(This,pdOffset) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __IGMFBridgeController_INTERFACE_DEFINED__ */


#ifndef __IGMFBridgeController2_INTERFACE_DEFINED__
#define __IGMFBridgeController2_INTERFACE_DEFINED__

/* interface IGMFBridgeController2 */
/* [unique][helpstring][dual][uuid][object] */ 


EXTERN_C const IID IID_IGMFBridgeController2;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("1CD80D64-817E-4beb-A711-A705F7CDFADB")
    IGMFBridgeController2 : public IGMFBridgeController
    {
    public:
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE BridgeAtDiscont( 
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraphSourceFilter,
            /* [in] */ BOOL bIsDiscontinuity) = 0;
        
    };
    
#else 	/* C style interface */

    typedef struct IGMFBridgeController2Vtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            IGMFBridgeController2 * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            __RPC__deref_out  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            IGMFBridgeController2 * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            IGMFBridgeController2 * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfoCount )( 
            IGMFBridgeController2 * This,
            /* [out] */ UINT *pctinfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfo )( 
            IGMFBridgeController2 * This,
            /* [in] */ UINT iTInfo,
            /* [in] */ LCID lcid,
            /* [out] */ ITypeInfo **ppTInfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetIDsOfNames )( 
            IGMFBridgeController2 * This,
            /* [in] */ REFIID riid,
            /* [size_is][in] */ LPOLESTR *rgszNames,
            /* [range][in] */ UINT cNames,
            /* [in] */ LCID lcid,
            /* [size_is][out] */ DISPID *rgDispId);
        
        /* [local] */ HRESULT ( STDMETHODCALLTYPE *Invoke )( 
            IGMFBridgeController2 * This,
            /* [in] */ DISPID dispIdMember,
            /* [in] */ REFIID riid,
            /* [in] */ LCID lcid,
            /* [in] */ WORD wFlags,
            /* [out][in] */ DISPPARAMS *pDispParams,
            /* [out] */ VARIANT *pVarResult,
            /* [out] */ EXCEPINFO *pExcepInfo,
            /* [out] */ UINT *puArgErr);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *AddStream )( 
            IGMFBridgeController2 * This,
            BOOL bVideo,
            eFormatType AllowedTypes,
            BOOL bDiscardUnconnected);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *InsertSinkFilter )( 
            IGMFBridgeController2 * This,
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **ppFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *InsertSourceFilter )( 
            IGMFBridgeController2 * This,
            /* [in] */ IUnknown *pUnkSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **ppFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *CreateSourceGraph )( 
            IGMFBridgeController2 * This,
            /* [in] */ BSTR strFile,
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **pSinkFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *CreateRenderGraph )( 
            IGMFBridgeController2 * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **pRenderGraphSourceFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *BridgeGraphs )( 
            IGMFBridgeController2 * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraphSourceFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *SetNotify )( 
            IGMFBridgeController2 * This,
            /* [in] */ LONG_PTR hwnd,
            /* [in] */ long msg);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *SetBufferMinimum )( 
            IGMFBridgeController2 * This,
            /* [in] */ long nMillisecs);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *GetSegmentTime )( 
            IGMFBridgeController2 * This,
            /* [retval][out] */ double *pdSeconds);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *NoMoreSegments )( 
            IGMFBridgeController2 * This);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *GetSegmentOffset )( 
            IGMFBridgeController2 * This,
            /* [retval][out] */ double *pdOffset);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *BridgeAtDiscont )( 
            IGMFBridgeController2 * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraphSourceFilter,
            /* [in] */ BOOL bIsDiscontinuity);
        
        END_INTERFACE
    } IGMFBridgeController2Vtbl;

    interface IGMFBridgeController2
    {
        CONST_VTBL struct IGMFBridgeController2Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define IGMFBridgeController2_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define IGMFBridgeController2_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define IGMFBridgeController2_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define IGMFBridgeController2_GetTypeInfoCount(This,pctinfo)	\
    ( (This)->lpVtbl -> GetTypeInfoCount(This,pctinfo) ) 

#define IGMFBridgeController2_GetTypeInfo(This,iTInfo,lcid,ppTInfo)	\
    ( (This)->lpVtbl -> GetTypeInfo(This,iTInfo,lcid,ppTInfo) ) 

#define IGMFBridgeController2_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)	\
    ( (This)->lpVtbl -> GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) ) 

#define IGMFBridgeController2_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)	\
    ( (This)->lpVtbl -> Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) ) 


#define IGMFBridgeController2_AddStream(This,bVideo,AllowedTypes,bDiscardUnconnected)	\
    ( (This)->lpVtbl -> AddStream(This,bVideo,AllowedTypes,bDiscardUnconnected) ) 

#define IGMFBridgeController2_InsertSinkFilter(This,pGraph,ppFilter)	\
    ( (This)->lpVtbl -> InsertSinkFilter(This,pGraph,ppFilter) ) 

#define IGMFBridgeController2_InsertSourceFilter(This,pUnkSourceGraphSinkFilter,pRenderGraph,ppFilter)	\
    ( (This)->lpVtbl -> InsertSourceFilter(This,pUnkSourceGraphSinkFilter,pRenderGraph,ppFilter) ) 

#define IGMFBridgeController2_CreateSourceGraph(This,strFile,pGraph,pSinkFilter)	\
    ( (This)->lpVtbl -> CreateSourceGraph(This,strFile,pGraph,pSinkFilter) ) 

#define IGMFBridgeController2_CreateRenderGraph(This,pSourceGraphSinkFilter,pRenderGraph,pRenderGraphSourceFilter)	\
    ( (This)->lpVtbl -> CreateRenderGraph(This,pSourceGraphSinkFilter,pRenderGraph,pRenderGraphSourceFilter) ) 

#define IGMFBridgeController2_BridgeGraphs(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter)	\
    ( (This)->lpVtbl -> BridgeGraphs(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter) ) 

#define IGMFBridgeController2_SetNotify(This,hwnd,msg)	\
    ( (This)->lpVtbl -> SetNotify(This,hwnd,msg) ) 

#define IGMFBridgeController2_SetBufferMinimum(This,nMillisecs)	\
    ( (This)->lpVtbl -> SetBufferMinimum(This,nMillisecs) ) 

#define IGMFBridgeController2_GetSegmentTime(This,pdSeconds)	\
    ( (This)->lpVtbl -> GetSegmentTime(This,pdSeconds) ) 

#define IGMFBridgeController2_NoMoreSegments(This)	\
    ( (This)->lpVtbl -> NoMoreSegments(This) ) 

#define IGMFBridgeController2_GetSegmentOffset(This,pdOffset)	\
    ( (This)->lpVtbl -> GetSegmentOffset(This,pdOffset) ) 


#define IGMFBridgeController2_BridgeAtDiscont(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter,bIsDiscontinuity)	\
    ( (This)->lpVtbl -> BridgeAtDiscont(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter,bIsDiscontinuity) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __IGMFBridgeController2_INTERFACE_DEFINED__ */


#ifndef __IGMFBridgeController3_INTERFACE_DEFINED__
#define __IGMFBridgeController3_INTERFACE_DEFINED__

/* interface IGMFBridgeController3 */
/* [unique][helpstring][dual][uuid][object] */ 


EXTERN_C const IID IID_IGMFBridgeController3;

#if defined(__cplusplus) && !defined(CINTERFACE)
    
    MIDL_INTERFACE("B344D399-F3F6-431C-882D-3DDFCFA9F968")
    IGMFBridgeController3 : public IGMFBridgeController2
    {
    public:
        virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE SetLiveTiming( 
            /* [in] */ BOOL bIsLiveTiming) = 0;
        
    };
    
#else 	/* C style interface */

    typedef struct IGMFBridgeController3Vtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            IGMFBridgeController3 * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            __RPC__deref_out  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            IGMFBridgeController3 * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            IGMFBridgeController3 * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfoCount )( 
            IGMFBridgeController3 * This,
            /* [out] */ UINT *pctinfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfo )( 
            IGMFBridgeController3 * This,
            /* [in] */ UINT iTInfo,
            /* [in] */ LCID lcid,
            /* [out] */ ITypeInfo **ppTInfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetIDsOfNames )( 
            IGMFBridgeController3 * This,
            /* [in] */ REFIID riid,
            /* [size_is][in] */ LPOLESTR *rgszNames,
            /* [range][in] */ UINT cNames,
            /* [in] */ LCID lcid,
            /* [size_is][out] */ DISPID *rgDispId);
        
        /* [local] */ HRESULT ( STDMETHODCALLTYPE *Invoke )( 
            IGMFBridgeController3 * This,
            /* [in] */ DISPID dispIdMember,
            /* [in] */ REFIID riid,
            /* [in] */ LCID lcid,
            /* [in] */ WORD wFlags,
            /* [out][in] */ DISPPARAMS *pDispParams,
            /* [out] */ VARIANT *pVarResult,
            /* [out] */ EXCEPINFO *pExcepInfo,
            /* [out] */ UINT *puArgErr);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *AddStream )( 
            IGMFBridgeController3 * This,
            BOOL bVideo,
            eFormatType AllowedTypes,
            BOOL bDiscardUnconnected);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *InsertSinkFilter )( 
            IGMFBridgeController3 * This,
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **ppFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *InsertSourceFilter )( 
            IGMFBridgeController3 * This,
            /* [in] */ IUnknown *pUnkSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **ppFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *CreateSourceGraph )( 
            IGMFBridgeController3 * This,
            /* [in] */ BSTR strFile,
            /* [in] */ IUnknown *pGraph,
            /* [retval][out] */ IUnknown **pSinkFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *CreateRenderGraph )( 
            IGMFBridgeController3 * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraph,
            /* [retval][out] */ IUnknown **pRenderGraphSourceFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *BridgeGraphs )( 
            IGMFBridgeController3 * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraphSourceFilter);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *SetNotify )( 
            IGMFBridgeController3 * This,
            /* [in] */ LONG_PTR hwnd,
            /* [in] */ long msg);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *SetBufferMinimum )( 
            IGMFBridgeController3 * This,
            /* [in] */ long nMillisecs);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *GetSegmentTime )( 
            IGMFBridgeController3 * This,
            /* [retval][out] */ double *pdSeconds);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *NoMoreSegments )( 
            IGMFBridgeController3 * This);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *GetSegmentOffset )( 
            IGMFBridgeController3 * This,
            /* [retval][out] */ double *pdOffset);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *BridgeAtDiscont )( 
            IGMFBridgeController3 * This,
            /* [in] */ IUnknown *pSourceGraphSinkFilter,
            /* [in] */ IUnknown *pRenderGraphSourceFilter,
            /* [in] */ BOOL bIsDiscontinuity);
        
        /* [helpstring][id] */ HRESULT ( STDMETHODCALLTYPE *SetLiveTiming )( 
            IGMFBridgeController3 * This,
            /* [in] */ BOOL bIsLiveTiming);
        
        END_INTERFACE
    } IGMFBridgeController3Vtbl;

    interface IGMFBridgeController3
    {
        CONST_VTBL struct IGMFBridgeController3Vtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define IGMFBridgeController3_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define IGMFBridgeController3_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define IGMFBridgeController3_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define IGMFBridgeController3_GetTypeInfoCount(This,pctinfo)	\
    ( (This)->lpVtbl -> GetTypeInfoCount(This,pctinfo) ) 

#define IGMFBridgeController3_GetTypeInfo(This,iTInfo,lcid,ppTInfo)	\
    ( (This)->lpVtbl -> GetTypeInfo(This,iTInfo,lcid,ppTInfo) ) 

#define IGMFBridgeController3_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)	\
    ( (This)->lpVtbl -> GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) ) 

#define IGMFBridgeController3_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)	\
    ( (This)->lpVtbl -> Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) ) 


#define IGMFBridgeController3_AddStream(This,bVideo,AllowedTypes,bDiscardUnconnected)	\
    ( (This)->lpVtbl -> AddStream(This,bVideo,AllowedTypes,bDiscardUnconnected) ) 

#define IGMFBridgeController3_InsertSinkFilter(This,pGraph,ppFilter)	\
    ( (This)->lpVtbl -> InsertSinkFilter(This,pGraph,ppFilter) ) 

#define IGMFBridgeController3_InsertSourceFilter(This,pUnkSourceGraphSinkFilter,pRenderGraph,ppFilter)	\
    ( (This)->lpVtbl -> InsertSourceFilter(This,pUnkSourceGraphSinkFilter,pRenderGraph,ppFilter) ) 

#define IGMFBridgeController3_CreateSourceGraph(This,strFile,pGraph,pSinkFilter)	\
    ( (This)->lpVtbl -> CreateSourceGraph(This,strFile,pGraph,pSinkFilter) ) 

#define IGMFBridgeController3_CreateRenderGraph(This,pSourceGraphSinkFilter,pRenderGraph,pRenderGraphSourceFilter)	\
    ( (This)->lpVtbl -> CreateRenderGraph(This,pSourceGraphSinkFilter,pRenderGraph,pRenderGraphSourceFilter) ) 

#define IGMFBridgeController3_BridgeGraphs(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter)	\
    ( (This)->lpVtbl -> BridgeGraphs(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter) ) 

#define IGMFBridgeController3_SetNotify(This,hwnd,msg)	\
    ( (This)->lpVtbl -> SetNotify(This,hwnd,msg) ) 

#define IGMFBridgeController3_SetBufferMinimum(This,nMillisecs)	\
    ( (This)->lpVtbl -> SetBufferMinimum(This,nMillisecs) ) 

#define IGMFBridgeController3_GetSegmentTime(This,pdSeconds)	\
    ( (This)->lpVtbl -> GetSegmentTime(This,pdSeconds) ) 

#define IGMFBridgeController3_NoMoreSegments(This)	\
    ( (This)->lpVtbl -> NoMoreSegments(This) ) 

#define IGMFBridgeController3_GetSegmentOffset(This,pdOffset)	\
    ( (This)->lpVtbl -> GetSegmentOffset(This,pdOffset) ) 


#define IGMFBridgeController3_BridgeAtDiscont(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter,bIsDiscontinuity)	\
    ( (This)->lpVtbl -> BridgeAtDiscont(This,pSourceGraphSinkFilter,pRenderGraphSourceFilter,bIsDiscontinuity) ) 


#define IGMFBridgeController3_SetLiveTiming(This,bIsLiveTiming)	\
    ( (This)->lpVtbl -> SetLiveTiming(This,bIsLiveTiming) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */




#endif 	/* __IGMFBridgeController3_INTERFACE_DEFINED__ */



#ifndef __GMFBridgeLib_LIBRARY_DEFINED__
#define __GMFBridgeLib_LIBRARY_DEFINED__

/* library GMFBridgeLib */
/* [helpstring][version][uuid] */ 


EXTERN_C const IID LIBID_GMFBridgeLib;

#ifndef ___IGMFBridgeEvents_DISPINTERFACE_DEFINED__
#define ___IGMFBridgeEvents_DISPINTERFACE_DEFINED__

/* dispinterface _IGMFBridgeEvents */
/* [helpstring][uuid] */ 


EXTERN_C const IID DIID__IGMFBridgeEvents;

#if defined(__cplusplus) && !defined(CINTERFACE)

    MIDL_INTERFACE("0732D4D6-96F5-46f6-B687-1DB7CD36D413")
    _IGMFBridgeEvents : public IDispatch
    {
    };
    
#else 	/* C style interface */

    typedef struct _IGMFBridgeEventsVtbl
    {
        BEGIN_INTERFACE
        
        HRESULT ( STDMETHODCALLTYPE *QueryInterface )( 
            _IGMFBridgeEvents * This,
            /* [in] */ REFIID riid,
            /* [annotation][iid_is][out] */ 
            __RPC__deref_out  void **ppvObject);
        
        ULONG ( STDMETHODCALLTYPE *AddRef )( 
            _IGMFBridgeEvents * This);
        
        ULONG ( STDMETHODCALLTYPE *Release )( 
            _IGMFBridgeEvents * This);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfoCount )( 
            _IGMFBridgeEvents * This,
            /* [out] */ UINT *pctinfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetTypeInfo )( 
            _IGMFBridgeEvents * This,
            /* [in] */ UINT iTInfo,
            /* [in] */ LCID lcid,
            /* [out] */ ITypeInfo **ppTInfo);
        
        HRESULT ( STDMETHODCALLTYPE *GetIDsOfNames )( 
            _IGMFBridgeEvents * This,
            /* [in] */ REFIID riid,
            /* [size_is][in] */ LPOLESTR *rgszNames,
            /* [range][in] */ UINT cNames,
            /* [in] */ LCID lcid,
            /* [size_is][out] */ DISPID *rgDispId);
        
        /* [local] */ HRESULT ( STDMETHODCALLTYPE *Invoke )( 
            _IGMFBridgeEvents * This,
            /* [in] */ DISPID dispIdMember,
            /* [in] */ REFIID riid,
            /* [in] */ LCID lcid,
            /* [in] */ WORD wFlags,
            /* [out][in] */ DISPPARAMS *pDispParams,
            /* [out] */ VARIANT *pVarResult,
            /* [out] */ EXCEPINFO *pExcepInfo,
            /* [out] */ UINT *puArgErr);
        
        END_INTERFACE
    } _IGMFBridgeEventsVtbl;

    interface _IGMFBridgeEvents
    {
        CONST_VTBL struct _IGMFBridgeEventsVtbl *lpVtbl;
    };

    

#ifdef COBJMACROS


#define _IGMFBridgeEvents_QueryInterface(This,riid,ppvObject)	\
    ( (This)->lpVtbl -> QueryInterface(This,riid,ppvObject) ) 

#define _IGMFBridgeEvents_AddRef(This)	\
    ( (This)->lpVtbl -> AddRef(This) ) 

#define _IGMFBridgeEvents_Release(This)	\
    ( (This)->lpVtbl -> Release(This) ) 


#define _IGMFBridgeEvents_GetTypeInfoCount(This,pctinfo)	\
    ( (This)->lpVtbl -> GetTypeInfoCount(This,pctinfo) ) 

#define _IGMFBridgeEvents_GetTypeInfo(This,iTInfo,lcid,ppTInfo)	\
    ( (This)->lpVtbl -> GetTypeInfo(This,iTInfo,lcid,ppTInfo) ) 

#define _IGMFBridgeEvents_GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId)	\
    ( (This)->lpVtbl -> GetIDsOfNames(This,riid,rgszNames,cNames,lcid,rgDispId) ) 

#define _IGMFBridgeEvents_Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr)	\
    ( (This)->lpVtbl -> Invoke(This,dispIdMember,riid,lcid,wFlags,pDispParams,pVarResult,pExcepInfo,puArgErr) ) 

#endif /* COBJMACROS */


#endif 	/* C style interface */


#endif 	/* ___IGMFBridgeEvents_DISPINTERFACE_DEFINED__ */


EXTERN_C const CLSID CLSID_GMFBridgeController;

#ifdef __cplusplus

class DECLSPEC_UUID("08E3287F-3A5C-47e9-8179-A9E9221A5CDE")
GMFBridgeController;
#endif
#endif /* __GMFBridgeLib_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

unsigned long             __RPC_USER  BSTR_UserSize(     unsigned long *, unsigned long            , BSTR * ); 
unsigned char * __RPC_USER  BSTR_UserMarshal(  unsigned long *, unsigned char *, BSTR * ); 
unsigned char * __RPC_USER  BSTR_UserUnmarshal(unsigned long *, unsigned char *, BSTR * ); 
void                      __RPC_USER  BSTR_UserFree(     unsigned long *, BSTR * ); 

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif


