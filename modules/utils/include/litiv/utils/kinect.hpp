
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#pragma once

#include "litiv/utils/cxx.hpp"
#if defined(_MSC_VER) && !USE_KINECTSDK_STANDALONE
#include <atlbase.h>
#include <windows.h>
#include <winerror.h>
#include <comdef.h>
#include <stdint.h>
#include <direct.h>
#include <psapi.h>
#include <Kinect.h>
#endif //!(defined(_MSC_VER) && !USE_KINECTSDK_STANDALONE)

namespace lv {

#if USE_KINECTSDK_STANDALONE
#ifndef BODY_COUNT
#define BODY_COUNT 6
#endif //ndef(BODY_COUNT)
    using TIMESPAN = int64_t;
#if !defined(_MSC_VER)
    using BOOLEAN = uchar;
    using DWORD = unsigned long;
    using UINT64 = unsigned long long;
#endif //(!defined(_MSC_VER))
#ifndef _Vector4_
#define _Vector4_
    using Vector4 = struct _Vector4 {
        float x,y,z,w;
    };
#endif //ndef(_Vector4_)
#ifndef _PointF_
#define _PointF_
    using PointF = struct _PointF {
        float X,Y;
    };
#endif //ndef(_PointF_)
#ifndef _FrameEdges_
#define _FrameEdges_
    using FrameEdges = enum _FrameEdges {
        FrameEdge_None   = 0x0,
        FrameEdge_Right  = 0x1,
        FrameEdge_Left   = 0x2,
        FrameEdge_Top    = 0x4,
        FrameEdge_Bottom = 0x8,
    };
#endif //ndef(_FrameEdges_)
#ifndef _TrackingState_
#define _TrackingState_
    using TrackingState = enum _TrackingState {
        TrackingState_NotTracked = 0,
        TrackingState_Inferred   = 1,
        TrackingState_Tracked    = 2,
    };
#endif //ndef(_TrackingState_)
#ifndef _HandState_
#define _HandState_
    using HandState = enum _HandState {
        HandState_Unknown    = 0,
        HandState_NotTracked = 1,
        HandState_Open       = 2,
        HandState_Closed     = 3,
        HandState_Lasso      = 4,
    };
#endif //ndef(_HandState_)
#ifndef _TrackingConfidence_
#define _TrackingConfidence_
    using TrackingConfidence = enum _TrackingConfidence {
        TrackingConfidence_Low  = 0,
        TrackingConfidence_High = 1,
    };
#endif //ndef(_TrackingConfidence_)
#ifndef _CameraSpacePoint_
#define _CameraSpacePoint_
    using CameraSpacePoint = struct _CameraSpacePoint {
        float X,Y,Z;
    };
#endif //ndef(_CameraSpacePoint_)
#ifndef _JointType_
#define _JointType_
    using JointType = enum _JointType {
        JointType_SpineBase     = 0,
        JointType_SpineMid      = 1,
        JointType_Neck          = 2,
        JointType_Head          = 3,
        JointType_ShoulderLeft  = 4,
        JointType_ElbowLeft     = 5,
        JointType_WristLeft     = 6,
        JointType_HandLeft      = 7,
        JointType_ShoulderRight = 8,
        JointType_ElbowRight    = 9,
        JointType_WristRight    = 10,
        JointType_HandRight     = 11,
        JointType_HipLeft       = 12,
        JointType_KneeLeft      = 13,
        JointType_AnkleLeft     = 14,
        JointType_FootLeft      = 15,
        JointType_HipRight      = 16,
        JointType_KneeRight     = 17,
        JointType_AnkleRight    = 18,
        JointType_FootRight     = 19,
        JointType_SpineShoulder = 20,
        JointType_HandTipLeft   = 21,
        JointType_ThumbLeft     = 22,
        JointType_HandTipRight  = 23,
        JointType_ThumbRight    = 24,
        JointType_Count         = 25,
    };
#endif //ndef(_JointType_)
#ifndef _Joint_
#define _Joint_
    using Joint = struct _Joint {
        _JointType JointType;
        _CameraSpacePoint Position;
        _TrackingState TrackingState;
    };
#endif //ndef(_Joint_)
#ifndef _JointOrientation_
#define _JointOrientation_
    using JointOrientation = struct _JointOrientation {
        _JointType JointType;
        _Vector4 Orientation;
    };
#endif //ndef(_JointOrientation_)
#endif //USE_KINECTSDK_STANDALONE

    /// portable structure containing kinect body data for one frame
    struct KinectBodyFrame {
        using BOOLEAN = unsigned char;
        using DWORD = unsigned long;
        using UINT64 = unsigned long long;
        bool bIsValid;
        TIMESPAN nTimeStamp;
        Vector4 vFloorClipPlane;
        size_t nFrameIdx;
        struct BodyData {
            BOOLEAN bIsTracked;
            BOOLEAN bIsRestricted;
            DWORD nClippedEdges;
            UINT64 nTrackingID;
            PointF vLean;
            TrackingState eLeanTrackState;
            HandState eLeftHandState;
            HandState eRightHandState;
            TrackingConfidence eLeftHandStateConfidence;
            TrackingConfidence eRightHandStateConfidence;
            std::array<Joint,JointType::JointType_Count> aJointData;
            std::array<JointOrientation,JointType::JointType_Count> aJointOrientationData;
        } aBodyData[BODY_COUNT];
    };

} // namespace lv