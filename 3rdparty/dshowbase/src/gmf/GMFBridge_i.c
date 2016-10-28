

/* this ALWAYS GENERATED file contains the IIDs and CLSIDs */

/* link this file in with the server and any clients */


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


#ifdef __cplusplus
extern "C"{
#endif 


#include <rpc.h>
#include <rpcndr.h>

#ifdef _MIDL_USE_GUIDDEF_

#ifndef INITGUID
#define INITGUID
#include <guiddef.h>
#undef INITGUID
#else
#include <guiddef.h>
#endif

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        DEFINE_GUID(name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8)

#else // !_MIDL_USE_GUIDDEF_

#ifndef __IID_DEFINED__
#define __IID_DEFINED__

typedef struct _IID
{
    unsigned long x;
    unsigned short s1;
    unsigned short s2;
    unsigned char  c[8];
} IID;

#endif // __IID_DEFINED__

#ifndef CLSID_DEFINED
#define CLSID_DEFINED
typedef IID CLSID;
#endif // CLSID_DEFINED

#define MIDL_DEFINE_GUID(type,name,l,w1,w2,b1,b2,b3,b4,b5,b6,b7,b8) \
        const type name = {l,w1,w2,{b1,b2,b3,b4,b5,b6,b7,b8}}

#endif !_MIDL_USE_GUIDDEF_

MIDL_DEFINE_GUID(IID, IID_IGMFBridgeController,0x8C4D8054,0xFCBA,0x4783,0x86,0x5A,0x7E,0x8B,0x3C,0x81,0x40,0x11);


MIDL_DEFINE_GUID(IID, IID_IGMFBridgeController2,0x1CD80D64,0x817E,0x4beb,0xA7,0x11,0xA7,0x05,0xF7,0xCD,0xFA,0xDB);


MIDL_DEFINE_GUID(IID, IID_IGMFBridgeController3,0xB344D399,0xF3F6,0x431C,0x88,0x2D,0x3D,0xDF,0xCF,0xA9,0xF9,0x68);


MIDL_DEFINE_GUID(IID, LIBID_GMFBridgeLib,0x5CE27AC5,0x940C,0x4199,0x87,0x46,0x01,0xFE,0x1F,0x12,0xA1,0x2E);


MIDL_DEFINE_GUID(IID, DIID__IGMFBridgeEvents,0x0732D4D6,0x96F5,0x46f6,0xB6,0x87,0x1D,0xB7,0xCD,0x36,0xD4,0x13);


MIDL_DEFINE_GUID(CLSID, CLSID_GMFBridgeController,0x08E3287F,0x3A5C,0x47e9,0x81,0x79,0xA9,0xE9,0x22,0x1A,0x5C,0xDE);

#undef MIDL_DEFINE_GUID

#ifdef __cplusplus
}
#endif



