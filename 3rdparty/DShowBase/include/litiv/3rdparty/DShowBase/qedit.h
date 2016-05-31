
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// SOURCE : http://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/2ab5c212-5824-419d-b5d9-7f5db82f57cd/qedith-missing-in-current-windows-sdk-v70?forum=windowsdirectshowdevelopment

#pragma once

#ifndef COM_NO_WINDOWS_H
#include <windows.h>
#include <ole2.h>
#include <strmif.h>
#endif /*COM_NO_WINDOWS_H*/

interface ISampleGrabberCB : public IUnknown {
    virtual STDMETHODIMP SampleCB(double SampleTime, IMediaSample *pSample) = 0;
    virtual STDMETHODIMP BufferCB(double SampleTime, BYTE *pBuffer, long BufferLen) = 0;
};

static const IID IID_ISampleGrabberCB = {0x0579154A, 0x2B53, 0x4994, {0xB0, 0xD0, 0xE7, 0x73, 0x14, 0x8E, 0xFF, 0x85}};

interface ISampleGrabber : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE SetOneShot(BOOL OneShot) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetMediaType(const AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetConnectedMediaType(AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetBufferSamples(BOOL BufferThem) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentBuffer(long *pBufferSize, long *pBuffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentSample(IMediaSample **ppSample) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetCallback(ISampleGrabberCB *pCallback, long WhichMethodToCallback) = 0;
};

static const IID IID_ISampleGrabber = {0x6B652FFF, 0x11FE, 0x4fce, {0x92, 0xAD, 0x02, 0x66, 0xB5, 0xD7, 0xC7, 0x8F}};
static const CLSID CLSID_SampleGrabber = {0xC1F400A0, 0x3F08, 0x11d3, {0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37}};

static const CLSID CLSID_NullRenderer = {0xC1F400A4, 0x3F08, 0x11d3, {0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37}};
static const CLSID CLSID_VideoEffects1Category = {0xcc7bfb42, 0xf175, 0x11d1, {0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59}};
static const CLSID CLSID_VideoEffects2Category = {0xcc7bfb43, 0xf175, 0x11d1, {0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59}};
static const CLSID CLSID_AudioEffects1Category = {0xcc7bfb44, 0xf175, 0x11d1, {0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59}};
static const CLSID CLSID_AudioEffects2Category = {0xcc7bfb45, 0xf175, 0x11d1, {0xa3, 0x92, 0x0, 0xe0, 0x29, 0x1f, 0x39, 0x59}};
