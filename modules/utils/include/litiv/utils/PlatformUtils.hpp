
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

// includes here really need cleanup @@@@

#include <opencv2/core.hpp>
#include "litiv/utils/DistanceUtils.hpp"
#include "litiv/utils/CxxUtils.hpp"
#include <queue>
#include <string>
#include <algorithm>
#include <vector>
#include <set>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <map>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <inttypes.h>
#include <csignal>
#if defined(_MSC_VER)
#include <windows.h>
#include <winerror.h>
#include <comdef.h>
#include <stdint.h>
#include <direct.h>
#include <psapi.h>
template<class T>
void SafeRelease(T **ppT) {if(*ppT) {(*ppT)->Release();*ppT = nullptr;}}
#if !USE_KINECTSDK_STANDALONE
#include <Kinect.h>
#endif //(!USE_KINECTSDK_STANDALONE)
#else //(!defined(_MSC_VER))
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <stdio.h>
#endif //(!defined(_MSC_VER))

namespace PlatformUtils {

    std::string GetCurrentWorkDirPath();
    std::string AddDirSlashIfMissing(const std::string& sDirPath);
    void GetFilesFromDir(const std::string& sDirPath, std::vector<std::string>& vsFilePaths);
    void GetSubDirsFromDir(const std::string& sDirPath, std::vector<std::string>& vsSubDirPaths);
    void FilterFilePaths(std::vector<std::string>& vsFilePaths, const std::vector<std::string>& vsRemoveTokens, const std::vector<std::string>& vsKeepTokens);
    bool CreateDirIfNotExist(const std::string& sDirPath);
    std::fstream CreateBinFileWithPrealloc(const std::string& sFilePath, size_t nPreallocBytes, bool bZeroInit=false);
    void RegisterAllConsoleSignals(void(*lHandler)(int));
    size_t GetCurrentPhysMemBytesUsed();

    inline bool compare_lowercase(const std::string& i, const std::string& j) {
        std::string i_lower(i), j_lower(j);
        std::transform(i_lower.begin(),i_lower.end(),i_lower.begin(),tolower);
        std::transform(j_lower.begin(),j_lower.end(),j_lower.begin(),tolower);
        return i_lower<j_lower;
    }

    template<typename T>
    inline int digit_count(T number) {
        // counts sign as extra digit if negative
        int digits = number<0?1:0;
        while(std::abs(number)>=1) {
            number /= 10;
            digits++;
        }
        return digits;
    }

    inline bool string_contains_token(const std::string& s, const std::vector<std::string>& tokens) {
        for(size_t i=0; i<tokens.size(); ++i)
            if(s.find(tokens[i])!=std::string::npos)
                return true;
        return false;
    }

    template<typename T>
    inline std::vector<size_t> sort_indices(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndices(voVals.size());
        std::iota(vnIndices.begin(),vnIndices.end(),0);
        std::sort(vnIndices.begin(),vnIndices.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]<voVals[n2];
        });
        return vnIndices;
    }

    template<typename T, typename P>
    inline std::vector<size_t> sort_indices(const std::vector<T>& voVals, P oSortFunctor) {
        std::vector<size_t> vnIndices(voVals.size());
        std::iota(vnIndices.begin(),vnIndices.end(),0);
        std::sort(vnIndices.begin(),vnIndices.end(),oSortFunctor);
        return vnIndices;
    }

    template<typename T>
    inline std::vector<size_t> unique_indices(const std::vector<T>& voVals) {
        std::vector<size_t> vnIndices = sort_indices(voVals);
        auto pLastIdxIter = std::unique(vnIndices.begin(),vnIndices.end(),[&voVals](size_t n1, size_t n2) {
            return voVals[n1]==voVals[n2];
        });
        return std::vector<size_t>(vnIndices.begin(),pLastIdxIter);
    }

    template<typename T, typename P1, typename P2>
    inline std::vector<size_t> unique_indices(const std::vector<T>& voVals, P1 oSortFunctor, P2 oCompareFunctor) {
        std::vector<size_t> vnIndices = sort_indices(voVals,oSortFunctor);
        auto pLastIdxIter = std::unique(vnIndices.begin(),vnIndices.end(),oCompareFunctor);
        return std::vector<size_t>(vnIndices.begin(),pLastIdxIter);
    }

    template<typename T>
    inline std::vector<T> unique(const cv::Mat_<T>& oMat, bool bSort=true) {
        const std::unordered_set<T> mMap(oMat.begin(),oMat.end());
        std::vector<T> vals(mMap.begin(),mMap.end());
        if(bSort)
            std::sort(vals.begin(),vals.end());
        return vals;
    }

    template<typename T>
    size_t find_nn_index(T oReqVal, const std::vector<T>& voRefVals) {
        decltype(DistanceUtils::L1dist(T(0),T(0))) oMinDist = std::numeric_limits<decltype(DistanceUtils::L1dist(T(0),T(0)))>::max();
        size_t nIdx = size_t(-1);
        for(size_t n=0; n<voRefVals.size(); ++n) {
            auto oCurrDist = DistanceUtils::L1dist(oReqVal,voRefVals[n]);
            if(nIdx==size_t(-1) || oCurrDist<oMinDist) {
                oMinDist = oCurrDist;
                nIdx = n;
            }
        }
        return nIdx;
    }

    template<typename Tx, typename Ty>
    std::vector<Ty> interp1(const std::vector<Tx>& vX, const std::vector<Ty>& vY, const std::vector<Tx>& vXReq) {
        // assumes that all vectors are sorted
        CV_Assert(vX.size()==vY.size());
        CV_Assert(vX.size()>1);
        std::vector<Tx> vDX;
        vDX.reserve(vX.size());
        std::vector<Ty> vDY, vSlope, vIntercept;
        vDY.reserve(vX.size());
        vSlope.reserve(vX.size());
        vIntercept.reserve(vX.size());
        for(size_t i=0; i<vX.size(); ++i) {
            if(i<vX.size()-1) {
                vDX.push_back(vX[i+1]-vX[i]);
                vDY.push_back(vY[i+1]-vY[i]);
                vSlope.push_back(Ty(vDY[i]/vDX[i]));
                vIntercept.push_back(vY[i]-Ty(vX[i]*vSlope[i]));
            }
            else {
                vDX.push_back(vDX[i-1]);
                vDY.push_back(vDY[i-1]);
                vSlope.push_back(vSlope[i-1]);
                vIntercept.push_back(vIntercept[i-1]);
            }
        }
        std::vector<Ty> vYReq;
        vYReq.reserve(vXReq.size());
        for(size_t i=0; i<vXReq.size(); ++i) {
            if(vXReq[i]>=vX.front() && vXReq[i]<=vX.back()) {
                size_t nNNIdx = find_nn_index(vXReq[i],vX);
                vYReq.push_back(vSlope[nNNIdx]*vXReq[i]+vIntercept[nNNIdx]);
            }
        }
        return vYReq;
    }

    template<typename T>
    inline std::enable_if_t<std::is_integral<T>::value,std::vector<T>> linspace(T a, T b, size_t steps, bool bIncludeInitVal=true) {
        if(steps==0)
            return std::vector<T>();
        else if(steps==1)
            return std::vector<T>(1,b);
        std::vector<T> vnResult(steps);
        const double dStep = double(b-a)/(steps-int(bIncludeInitVal));
        if(bIncludeInitVal)
            for(size_t nStepIter = 0; nStepIter<steps; ++nStepIter)
                vnResult[nStepIter] = a+T(dStep*nStepIter);
        else
            for(size_t nStepIter = 1; nStepIter<=steps; ++nStepIter)
                vnResult[nStepIter-1] = a+T(dStep*nStepIter);
        return vnResult;
    }

    template<typename T>
    inline std::enable_if_t<std::is_floating_point<T>::value,std::vector<T>> linspace(T a, T b, size_t steps, bool bIncludeInitVal=true) {
        if(steps==0)
            return std::vector<T>();
        else if(steps==1)
            return std::vector<T>(1,b);
        std::vector<T> vfResult(steps);
        const T fStep = (b-a)/(steps-int(bIncludeInitVal));
        if(bIncludeInitVal)
            for(size_t nStepIter = 0; nStepIter<steps; ++nStepIter)
                vfResult[nStepIter] = a+fStep*T(nStepIter);
        else
            for(size_t nStepIter = 1; nStepIter<=steps; ++nStepIter)
                vfResult[nStepIter] = a+fStep*T(nStepIter);
        return vfResult;
    }

    template<size_t nWorkers>
    struct WorkerPool {
        static_assert(nWorkers>0,"Worker pool must have at least one work thread");
        WorkerPool() : m_bIsActive(true) {
            CxxUtils::unroll<nWorkers>([this](size_t){m_vhWorkers.emplace_back(std::bind(&WorkerPool::entry,this));});
        }
        ~WorkerPool() {
            m_bIsActive = false;
            m_oSyncVar.notify_all();
            for(std::thread& oWorker : m_vhWorkers)
                oWorker.join();
        }
        template<typename Tfunc, typename... Targs>
        std::future<std::result_of_t<Tfunc(Targs...)>> queueTask(Tfunc&& lTaskEntryPoint, Targs&&... args) {
            using task_return_t = std::result_of_t<Tfunc(Targs...)>;
            using task_t = std::packaged_task<task_return_t()>;
            // http://stackoverflow.com/questions/28179817/how-can-i-store-generic-packaged-tasks-in-a-container
            std::shared_ptr<task_t> pSharableTask = std::make_shared<task_t>(std::bind(std::forward<Tfunc>(lTaskEntryPoint),std::forward<Targs>(args)...));
            std::future<task_return_t> oTaskRes = pSharableTask->get_future();
            {
                std::mutex_lock_guard sync_lock(m_oSyncMutex);
                m_qTasks.emplace([pSharableTask](){(*pSharableTask)();}); // lambda keeps a copy of the task in the queue
            }
            m_oSyncVar.notify_one();
            return oTaskRes;
        }
    protected:
        std::queue<std::function<void()>> m_qTasks;
        std::vector<std::thread> m_vhWorkers;
        std::mutex m_oSyncMutex;
        std::condition_variable m_oSyncVar;
        std::atomic_bool m_bIsActive;
    private:
        void entry() {
            std::mutex_unique_lock sync_lock(m_oSyncMutex);
            while(m_bIsActive || !m_qTasks.empty()) {
                if(m_qTasks.empty())
                    m_oSyncVar.wait(sync_lock);
                if(!m_qTasks.empty()) {
                    std::function<void()> task = std::move(m_qTasks.front());
                    m_qTasks.pop();
                    std::unlock_guard<std::mutex_unique_lock> oUnlock(sync_lock);
                    task();
                }
            }
        }
        WorkerPool(const WorkerPool&) = delete;
        WorkerPool& operator=(const WorkerPool&) = delete;
    };

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
    struct KinectBodyFrame {
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
} //namespace PlatformUtils
