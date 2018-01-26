
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2018 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

////////////////////////////////
#define GEN_SEGMENTATION_ANNOT  1
#define GEN_REGISTRATION_ANNOT  0
#define SAVE_C2D_MAPPING        0
////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      1
////////////////////////////////
#define DATASETS_LITIV2018_CALIB_VERSION 4
#define DATASETS_LITIV2018_DATA_VERSION 4
#define DATASETS_START_IDX 530

#if (GEN_SEGMENTATION_ANNOT+GEN_REGISTRATION_ANNOT)>1
#error "must select one type of annotation to generate"
#endif //(GEN_SEGMENTATION_ANNOT+GEN_REGISTRATION_ANNOT)>1
#if SAVE_C2D_MAPPING
#ifndef _MSC_VER
#error "must use kinect api"
#endif //ndef(_MSC_VER)
#else //!SAVE_C2D_MAPPING
#define USE_KINECTSDK_STANDALONE 1
#endif //!SAVE_C2D_MAPPING
#include "litiv/utils/kinect.hpp"
#include "litiv/datasets.hpp"
#include "litiv/imgproc.hpp"
#include <opencv2/calib3d.hpp>
#ifdef _MSC_VER
#include <windows.h>
#endif //def(_MSC_VER)
#if !USING_LZ4
#error "lz4 required."
#endif //USING_LZ4

using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::Dataset_LITIV_stcharles2018,lv::NonParallel>;
void Analyze(lv::IDataHandlerPtr pBatch);

int main(int, char**) {
    try {
        DatasetType::Ptr pDataset = DatasetType::create(
                DATASET_OUTPUT_PATH, // const std::string& sOutputDirName
                false, //bool bSaveOutput
                false, //bool bUseEvaluator
                (GEN_SEGMENTATION_ANNOT==1 || SAVE_C2D_MAPPING==1), //bool bLoadDepth
                (GEN_REGISTRATION_ANNOT==1), //bool bUndistort
                (GEN_REGISTRATION_ANNOT==1), //bool bHorizRectify
                false, //bool bEvalDisparities
                false, //bool bFlipDisparities
                false, //bool bLoadFrameSubset
                false, //bool bEvalOnlyFrameSubset
                0, //int nEvalTemporalWindowSize
                0, //int nLoadInputMasks
                1.0 //double dScaleFactor
        );
        lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false);
        const size_t nTotPackets = pDataset->getInputCount();
        const size_t nTotBatches = vpBatches.size();
        if(nTotBatches==0 || nTotPackets==0)
            lvError_("Could not parse any data for dataset '%s'",pDataset->getName().c_str());
        std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
        for(lv::IDataHandlerPtr pBatch : vpBatches)
            Analyze(pBatch);
    }
    catch(const lv::Exception&) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught lv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const cv::Exception&) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    return 0;
}

void Analyze(lv::IDataHandlerPtr pBatch) {
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        lvAssert(oBatch.getInputPacketType()==lv::ImageArrayPacket && oBatch.getInputStreamCount()>=2 && oBatch.getInputCount()>=1);
        if(DATASET_PRECACHING)
            oBatch.startPrecaching();
        const size_t nTotPacketCount = oBatch.getInputCount();
        size_t nCurrIdx = 0;
        std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // mat content becomes invalid on next getInput call
        lvAssert(!vInitInput.empty() && vInitInput.size()==oBatch.getInputStreamCount());
        std::vector<cv::Size> vOrigSizes(vInitInput.size());
        for(size_t a=0u; a<vInitInput.size(); ++a) {
            vOrigSizes[a] = vInitInput[a].size();
        #if GEN_REGISTRATION_ANNOT
            lvAssert(vOrigSizes[a]==DATASETS_LITIV2018_RECTIFIED_SIZE);
        #endif //GEN_REGISTRATION_ANNOT
            vInitInput[a] = vInitInput[a].clone();
        }
        lvLog_(1,"\ninitializing batch '%s'...\n",oBatch.getName().c_str());

        const std::string sC2DMappingDir = oBatch.getDataPath()+"coordmap_c2d/";
    #if SAVE_C2D_MAPPING
        CComPtr<IKinectSensor> pKinectSensor;
        lvAssertHR(GetDefaultKinectSensor(&pKinectSensor));
        lvAssert(pKinectSensor);
        lvAssertHR(pKinectSensor->Open());
        CComPtr<ICoordinateMapper> pCoordMapper;
        lvAssertHR(pKinectSensor->get_CoordinateMapper(&pCoordMapper));
        lv::createDirIfNotExist(sC2DMappingDir);
    #endif //SAVE_C2D_MAPPING

        lv::DisplayHelperPtr pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
        pDisplayHelper->setContinuousUpdates(true);
        pDisplayHelper->setDisplayCursor(false);
        //const cv::Size oDisplayTileSize(1024,768);
        const cv::Size oDisplayTileSize(1200,1000);
        std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs = {{
            std::make_pair(vInitInput[0].clone(),oBatch.getInputStreamName(0)),
            std::make_pair(vInitInput[1].clone(),oBatch.getInputStreamName(1))
        }};
        lvAssert(oBatch.getName().find("calib")==std::string::npos);
        lvLog(1,"Parsing metadata...");
        cv::FileStorage oMetadataFS(oBatch.getDataPath()+"metadata.yml",cv::FileStorage::READ);
        lvAssert(oMetadataFS.isOpened());
        int nMinDepthVal,nMaxDepthVal;
        cv::FileNode oDepthData = oMetadataFS["depth_metadata"];
        oDepthData["min_reliable_dist"] >> nMinDepthVal;
        oDepthData["max_reliable_dist"] >> nMaxDepthVal;
        lvAssert(nMinDepthVal>=0 && nMaxDepthVal>nMinDepthVal);
        const std::string sGTName = (GEN_REGISTRATION_ANNOT?"gt_disp":"gt_masks");
        const std::string sRGBGTDir = oBatch.getDataPath()+"rgb_"+sGTName+"/";
        const std::string sLWIRGTDir = oBatch.getDataPath()+"lwir_"+sGTName+"/";
        lv::createDirIfNotExist(sRGBGTDir);
        lv::createDirIfNotExist(sLWIRGTDir);
        std::set<int> mnSubsetIdxs;
        {
            cv::FileStorage oGTMetadataFS(oBatch.getDataPath()+sGTName+"_metadata.yml",cv::FileStorage::READ);
            if(oGTMetadataFS.isOpened()) {
                int nPrevPacketCount;
                oGTMetadataFS["npackets"] >> nPrevPacketCount;
                lvAssert(nPrevPacketCount==(int)nTotPacketCount);
                std::vector<int> vnPrevSubsetIdxs;
                oGTMetadataFS["subsetidxs"] >> vnPrevSubsetIdxs;
                mnSubsetIdxs.insert(vnPrevSubsetIdxs.begin(),vnPrevSubsetIdxs.end());
            }
        }
    #if (GEN_SEGMENTATION_ANNOT || GEN_REGISTRATION_ANNOT)
    #if GEN_SEGMENTATION_ANNOT
        double dSegmOpacity = 0.5;
        std::array<double,2> adSegmToolRadius={10,3};
        const std::array<double,2> adSegmToolRadiusMin={3,1},adSegmToolRadiusMax={150,60};
        std::array<cv::Mat,2> aSegmMasks,aPrevSegmMasks,aTempSegmMasks,aSegmMasks_3ch,aFloodFillMasks;
        bool bDragInProgress = false;
        int nShapeOriginTile;
        std::array<cv::Rect,2> aShapeDragRect;
        std::array<cv::Point,2> aShapeDragInitPos,aShapeDragInitInnerOffset;
        size_t nCurrView = 0u;
    #elif GEN_REGISTRATION_ANNOT
        std::array<std::vector<cv::Point2i>,2> avTempPts;
        std::array<std::vector<std::pair<cv::Point2i,int>>,2> avCorrespPts;
        bool bDragInProgress = false;
        size_t nSelectedPoint = SIZE_MAX;
        int nSelectedPointTile = -1, nLineCreationStep = 0;
        cv::Point2i vInitDragPt;
        const double dMaxSelectDist = 5.0;
        const int nLineSplitRatio = 3;
        const std::array<int,2> anMarkerSizes{2,2};
    #endif //GEN_REGISTRATION_ANNOT
        cv::Point2f vMousePos(0.5f,0.5f),vLastMousePos(0.5f,0.5f);
        int nCurrTile = -1;
        pDisplayHelper->setContinuousUpdates(true);
        pDisplayHelper->setMouseCallback([&](const lv::DisplayHelper::CallbackData& oData) {
            vMousePos = cv::Point2f(float(oData.oInternalPosition.x)/oData.oTileSize.width,float(oData.oInternalPosition.y)/oData.oTileSize.height);
            if(vMousePos.x>=0.0f && vMousePos.y>=0.0f && vMousePos.x<1.0f && vMousePos.y<1.0f) {
                nCurrTile = oData.oPosition.x/oData.oTileSize.width;
            #if GEN_SEGMENTATION_ANNOT
                if(nCurrView!=2u)
                    nCurrTile = (int)nCurrView;
                if(!bDragInProgress && oData.nEvent==cv::EVENT_MOUSEWHEEL && cv::getMouseWheelDelta(oData.nFlags)>0) {
                    adSegmToolRadius[nCurrTile] = std::min(adSegmToolRadius[nCurrTile]+std::max(1.0,adSegmToolRadius[nCurrTile]/4),adSegmToolRadiusMax[nCurrTile]);
                    lvLog_(2,"\tnew tool size = %f",adSegmToolRadius[nCurrTile]);
                }
                else if(!bDragInProgress && oData.nEvent==cv::EVENT_MOUSEWHEEL && cv::getMouseWheelDelta(oData.nFlags)<0) {
                    adSegmToolRadius[nCurrTile] = std::max(adSegmToolRadius[nCurrTile]-std::max(1.0,adSegmToolRadius[nCurrTile]/4),adSegmToolRadiusMin[nCurrTile]);
                    lvLog_(2,"\tnew tool size = %f",adSegmToolRadius[nCurrTile]);
                }
                if(!bDragInProgress && oData.nEvent==cv::EVENT_MBUTTONDOWN) {
                    aFloodFillMasks[nCurrTile].create(vOrigSizes[nCurrTile].height+2,vOrigSizes[nCurrTile].width+2,CV_8UC1);
                    aFloodFillMasks[nCurrTile] = 0u;
                    aShapeDragInitPos[nCurrTile] = cv::Point(int(vMousePos.x*vOrigSizes[nCurrTile].width),int(vMousePos.y*vOrigSizes[nCurrTile].height));
                    aShapeDragInitPos[nCurrTile^1] = cv::Point(int(vMousePos.x*vOrigSizes[nCurrTile^1].width),int(vMousePos.y*vOrigSizes[nCurrTile^1].height));
                    if(aSegmMasks[nCurrTile].at<uchar>(aShapeDragInitPos[nCurrTile])!=0u) {
                        cv::floodFill(aSegmMasks[nCurrTile],aFloodFillMasks[nCurrTile],aShapeDragInitPos[nCurrTile],cv::Scalar(0),&aShapeDragRect[nCurrTile],cv::Scalar(),cv::Scalar(),4|((255)<<8));
                        bDragInProgress = true;
                    }
                    else if(aPrevSegmMasks[nCurrTile].at<uchar>(aShapeDragInitPos[nCurrTile])!=0u) {
                        cv::floodFill(aPrevSegmMasks[nCurrTile],aFloodFillMasks[nCurrTile],aShapeDragInitPos[nCurrTile],cv::Scalar(0),&aShapeDragRect[nCurrTile],cv::Scalar(),cv::Scalar(),4|((255)<<8));
                        bDragInProgress = true;
                    }
                    if(bDragInProgress) {
                        const double dWidthRatio = double(vOrigSizes[nCurrTile^1].width)/vOrigSizes[nCurrTile].width;
                        const double dHeightRatio = double(vOrigSizes[nCurrTile^1].height)/vOrigSizes[nCurrTile].height;
                        const cv::Point2i oShapeDragRect2_tl((int)std::round(aShapeDragRect[nCurrTile].x*dWidthRatio),(int)std::round(aShapeDragRect[nCurrTile].y*dHeightRatio));
                        const cv::Point2i oShapeDragRect2_wh((int)std::round(aShapeDragRect[nCurrTile].width*dWidthRatio),(int)std::round(aShapeDragRect[nCurrTile].height*dHeightRatio));
                        aShapeDragRect[nCurrTile^1] = cv::Rect(oShapeDragRect2_tl.x,oShapeDragRect2_tl.y,oShapeDragRect2_wh.x,oShapeDragRect2_wh.y);
                        cv::resize(aFloodFillMasks[nCurrTile],aFloodFillMasks[nCurrTile^1],cv::Size(vOrigSizes[nCurrTile^1].width+2,vOrigSizes[nCurrTile^1].height+2),0,0,cv::INTER_NEAREST);
                        aShapeDragInitInnerOffset[nCurrTile] = aShapeDragRect[nCurrTile].tl()-aShapeDragInitPos[nCurrTile];
                        aShapeDragInitInnerOffset[nCurrTile^1] = aShapeDragRect[nCurrTile^1].tl()-aShapeDragInitPos[nCurrTile^1];
                        nShapeOriginTile = nCurrTile;
                    }
                }
                else if(bDragInProgress && oData.nEvent==cv::EVENT_MBUTTONUP) {
                    aTempSegmMasks[0u] = 0u;
                    aTempSegmMasks[1u] = 0u;
                    const cv::Point oROI_out_tl(int(vMousePos.x*vOrigSizes[nCurrTile].width)+aShapeDragInitInnerOffset[nCurrTile].x,int(vMousePos.y*vOrigSizes[nCurrTile].height)+aShapeDragInitInnerOffset[nCurrTile].y);
                    const cv::Rect oOutputRect(oROI_out_tl.x,oROI_out_tl.y,aShapeDragRect[nCurrTile].width,aShapeDragRect[nCurrTile].height);
                    if(nCurrTile!=nShapeOriginTile) {
                        const cv::Rect oInputRect(aShapeDragRect[nShapeOriginTile].x+1,aShapeDragRect[nShapeOriginTile].y+1,aShapeDragRect[nShapeOriginTile].width,aShapeDragRect[nShapeOriginTile].height);
                        lv::copyValidPixelsTo(aFloodFillMasks[nShapeOriginTile],oInputRect,aSegmMasks[nShapeOriginTile],oInputRect,nullptr,aFloodFillMasks[nShapeOriginTile]>0);
                    }
                    const cv::Rect oInputRect(aShapeDragRect[nCurrTile].x+1,aShapeDragRect[nCurrTile].y+1,aShapeDragRect[nCurrTile].width,aShapeDragRect[nCurrTile].height);
                    lv::copyValidPixelsTo(aFloodFillMasks[nCurrTile],oInputRect,aSegmMasks[nCurrTile],oOutputRect,nullptr,aFloodFillMasks[nCurrTile]>0);
                    bDragInProgress = false;
                }
                else if(bDragInProgress && oData.nEvent==cv::EVENT_MOUSEMOVE) {
                    aTempSegmMasks[0u] = 0u;
                    aTempSegmMasks[1u] = 0u;
                    const cv::Point oROI_out_tl(int(vMousePos.x*vOrigSizes[nCurrTile].width)+aShapeDragInitInnerOffset[nCurrTile].x,int(vMousePos.y*vOrigSizes[nCurrTile].height)+aShapeDragInitInnerOffset[nCurrTile].y);
                    const cv::Rect oOutputRect(oROI_out_tl.x,oROI_out_tl.y,aShapeDragRect[nCurrTile].width,aShapeDragRect[nCurrTile].height);
                    if(nCurrTile!=nShapeOriginTile) {
                        const cv::Rect oInputRect(aShapeDragRect[nShapeOriginTile].x+1,aShapeDragRect[nShapeOriginTile].y+1,aShapeDragRect[nShapeOriginTile].width,aShapeDragRect[nShapeOriginTile].height);
                        lv::copyValidPixelsTo(aFloodFillMasks[nShapeOriginTile],oInputRect,aTempSegmMasks[nShapeOriginTile],oInputRect);
                    }
                    const cv::Rect oInputRect(aShapeDragRect[nCurrTile].x+1,aShapeDragRect[nCurrTile].y+1,aShapeDragRect[nCurrTile].width,aShapeDragRect[nCurrTile].height);
                    lv::copyValidPixelsTo(aFloodFillMasks[nCurrTile],oInputRect,aTempSegmMasks[nCurrTile],oOutputRect);
                }
                else if(!bDragInProgress && (oData.nEvent==cv::EVENT_LBUTTONDOWN || (oData.nEvent==cv::EVENT_MOUSEMOVE && (oData.nFlags&cv::EVENT_FLAG_LBUTTON)))) {
                    const cv::Point2i vMousePos_FP2(int(vMousePos.x*vOrigSizes[nCurrTile].width*4),int(vMousePos.y*vOrigSizes[nCurrTile].height*4));
                    const cv::Point2i vLastMousePos_FP2(int(vLastMousePos.x*vOrigSizes[nCurrTile].width*4),int(vLastMousePos.y*vOrigSizes[nCurrTile].height*4));
                    const cv::Point2i vMousePosDiff = vMousePos_FP2-vLastMousePos_FP2;
                    const int nMoveIter = int(cv::norm(vMousePosDiff)/2.0);
                    for(int nCurrIter=1; nCurrIter<=nMoveIter; ++nCurrIter) {
                        const cv::Point2i vMouseCurrPos = vLastMousePos_FP2+cv::Point2i(vMousePosDiff.x*nCurrIter/nMoveIter,vMousePosDiff.y*nCurrIter/nMoveIter);
                        cv::circle(aSegmMasks[nCurrTile],vMouseCurrPos,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(255),-1,cv::LINE_8,2);
                    }
                    cv::circle(aSegmMasks[nCurrTile],vMousePos_FP2,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(255),-1,cv::LINE_8,2);
                }
                else if(!bDragInProgress && (oData.nEvent==cv::EVENT_RBUTTONDOWN || (oData.nEvent==cv::EVENT_MOUSEMOVE && (oData.nFlags&cv::EVENT_FLAG_RBUTTON)))) {
                    if(oData.nFlags&cv::EVENT_FLAG_CTRLKEY) {
                        const cv::Point oEraseLocPt(int(vMousePos.x*vOrigSizes[nCurrTile].width),int(vMousePos.y*vOrigSizes[nCurrTile].height));
                        if(aSegmMasks[nCurrTile].at<uchar>(oEraseLocPt)!=0u)
                            cv::floodFill(aSegmMasks[nCurrTile],oEraseLocPt,cv::Scalar(0));
                    }
                    else {
                        const cv::Point2i vMousePos_FP2(int(vMousePos.x*vOrigSizes[nCurrTile].width*4),int(vMousePos.y*vOrigSizes[nCurrTile].height*4));
                        const cv::Point2i vLastMousePos_FP2(int(vLastMousePos.x*vOrigSizes[nCurrTile].width*4),int(vLastMousePos.y*vOrigSizes[nCurrTile].height*4));
                        const cv::Point2i vMousePosDiff = vMousePos_FP2-vLastMousePos_FP2;
                        const int nMoveIter = int(cv::norm(vMousePosDiff)/2.0);
                        for(int nCurrIter=1; nCurrIter<=nMoveIter; ++nCurrIter) {
                            const cv::Point2i vMouseCurrPos = vLastMousePos_FP2+cv::Point2i(vMousePosDiff.x*nCurrIter/nMoveIter,vMousePosDiff.y*nCurrIter/nMoveIter);
                            cv::circle(aSegmMasks[nCurrTile],vMouseCurrPos,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(0),-1,cv::LINE_8,2);
                        }
                        cv::circle(aSegmMasks[nCurrTile],vMousePos_FP2,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(0),-1,cv::LINE_8,2);
                    }
                }
                if(!bDragInProgress && (oData.nEvent==cv::EVENT_LBUTTONUP || oData.nEvent==cv::EVENT_RBUTTONUP)) {
                    for(size_t a=0u; a<2u; ++a)
                        cv::compare(aSegmMasks[a],128u,aSegmMasks[a],cv::CMP_GT);
                }
            #elif GEN_REGISTRATION_ANNOT
                const cv::Point2i vCurrPt(int(vMousePos.x*vOrigSizes[nCurrTile].width),int(vMousePos.y*vOrigSizes[nCurrTile].height));
                if(nLineCreationStep==0 && !bDragInProgress && (oData.nEvent==cv::EVENT_MBUTTONDOWN || oData.nEvent==cv::EVENT_LBUTTONDOWN)) {
                    lvAssert(nSelectedPoint==SIZE_MAX);
                    double dMinDist = 9999.;
                    for(size_t nPtIdx=0u; nPtIdx<avCorrespPts[nCurrTile].size(); ++nPtIdx) {
                        const double dCurrDist = cv::norm(vCurrPt-avCorrespPts[nCurrTile][nPtIdx].first);
                        if(dCurrDist<dMaxSelectDist && dCurrDist<dMinDist) {
                            dMinDist = dCurrDist;
                            nSelectedPoint = nPtIdx;
                        }
                    }
                    if(oData.nEvent==cv::EVENT_MBUTTONDOWN) {
                        if(nSelectedPoint!=SIZE_MAX) {
                            avCorrespPts[nCurrTile].erase(avCorrespPts[nCurrTile].begin()+nSelectedPoint);
                            avCorrespPts[nCurrTile^1].erase(avCorrespPts[nCurrTile^1].begin()+nSelectedPoint);
                            nSelectedPoint = SIZE_MAX;
                        }
                    }
                    else if(oData.nEvent==cv::EVENT_LBUTTONDOWN) {
                        if(nSelectedPoint==SIZE_MAX) {
                            nSelectedPoint = avCorrespPts[nCurrTile].size();
                            avCorrespPts[nCurrTile].emplace_back(vCurrPt,0);
                            avCorrespPts[nCurrTile^1].emplace_back(vCurrPt,0);
                        }
                    }
                    if(nSelectedPoint!=SIZE_MAX) {
                        lvAssert(nSelectedPoint<avCorrespPts[nCurrTile].size() && nSelectedPoint<avCorrespPts[nCurrTile^1].size());
                        nSelectedPointTile = nCurrTile;
                        vInitDragPt = vCurrPt;
                        bDragInProgress = true;
                    }
                }
                else if(nLineCreationStep==0 && bDragInProgress && (oData.nEvent==cv::EVENT_MBUTTONUP || oData.nEvent==cv::EVENT_LBUTTONUP || oData.nEvent==cv::EVENT_MOUSEMOVE)) {
                    const int nDist = (nSelectedPointTile!=nCurrTile)?0:((vCurrPt.x-vInitDragPt.x)/8);
                    lvAssert(nSelectedPoint<avCorrespPts[nSelectedPointTile^1].size());
                    avCorrespPts[nSelectedPointTile][nSelectedPoint].second = nDist;
                    avCorrespPts[nSelectedPointTile^1][nSelectedPoint].second = -nDist;
                    avCorrespPts[nSelectedPointTile^1][nSelectedPoint].first.x = avCorrespPts[nSelectedPointTile][nSelectedPoint].first.x+nDist;
                    if(oData.nEvent==cv::EVENT_MBUTTONUP || oData.nEvent==cv::EVENT_LBUTTONUP) {
                        bDragInProgress = false;
                        nSelectedPointTile = -1;
                        nSelectedPoint = SIZE_MAX;
                    }
                }
                else if(!bDragInProgress && oData.nEvent==cv::EVENT_RBUTTONDOWN) {
                    if(nLineCreationStep==0 || (nLineCreationStep==2 && nSelectedPointTile==nCurrTile)) {
                        lvAssert(avTempPts[nCurrTile].empty());
                        avTempPts[nCurrTile].push_back(vCurrPt);
                        if(nLineCreationStep==2)
                            avTempPts[nCurrTile][0].y = avTempPts[nCurrTile^1][0].y;
                        nSelectedPointTile = nCurrTile;
                        ++nLineCreationStep;
                    }
                    else if(nLineCreationStep==1 && nSelectedPointTile==nCurrTile) {
                        const int nLineHeight = avTempPts[nCurrTile][0].y;
                        avTempPts[nCurrTile].emplace_back(vCurrPt.x,nLineHeight);
                        nSelectedPointTile = nCurrTile^1;
                        ++nLineCreationStep;
                    }
                    else if(nLineCreationStep==3 && nSelectedPointTile==nCurrTile) {
                        lvAssert(avTempPts[nCurrTile].size()==1u && avTempPts[nCurrTile^1].size()==2u);
                        const int nLineHeight = avTempPts[nCurrTile][0].y;
                        lvAssert(nLineHeight==avTempPts[nCurrTile^1][0].y && nLineHeight==avTempPts[nCurrTile^1][1].y);
                        avTempPts[nCurrTile].emplace_back(vCurrPt.x,nLineHeight);
                        const int nFirstPairDist = avTempPts[nCurrTile].back().x-avTempPts[nCurrTile].front().x;
                        int nSecondPairDist = avTempPts[nCurrTile^1].back().x-avTempPts[nCurrTile^1].front().x;
                        if((nFirstPairDist<0 && nSecondPairDist>0) || (nFirstPairDist>0 && nSecondPairDist<0)) {
                            std::swap(avTempPts[nCurrTile^1].front(),avTempPts[nCurrTile^1].back());
                            nSecondPairDist = -nSecondPairDist;
                        }
                        const int nMinLength = std::min(std::abs(nFirstPairDist),std::abs(nSecondPairDist));
                        const int nInteriorPts = nMinLength/nLineSplitRatio;
                        for(int nPtIdx=0; nPtIdx<=nInteriorPts; ++nPtIdx) {
                            const cv::Point2i oFirstPt(avTempPts[nCurrTile].front().x+int(std::round(nPtIdx*float(nFirstPairDist)/std::max(nInteriorPts,1))),nLineHeight);
                            const cv::Point2i oSecondPt(avTempPts[nCurrTile^1].front().x+int(std::round(nPtIdx*float(nSecondPairDist)/std::max(nInteriorPts,1))),nLineHeight);
                            const int nDist = oSecondPt.x-oFirstPt.x;
                            avCorrespPts[nCurrTile].emplace_back(oFirstPt,nDist);
                            avCorrespPts[nCurrTile^1].emplace_back(oSecondPt,-nDist);
                        }
                        avTempPts[nCurrTile].clear();
                        avTempPts[nCurrTile^1].clear();
                        nSelectedPointTile = -1;
                        nLineCreationStep = 0;
                    }
                }
            #endif //GEN_REGISTRATION_ANNOT
            }
            else
                nCurrTile = -1;
            vLastMousePos = vMousePos;
        });
    #endif //(GEN_SEGMENTATION_ANNOT || GEN_REGISTRATION_ANNOT)
        lvLog(1,"Annotation edit mode initialized...");
        nCurrIdx = size_t(DATASETS_START_IDX);
        while(nCurrIdx<nTotPacketCount) {
            const std::string sPacketName = oBatch.getInputName(nCurrIdx);
            const std::vector<cv::Mat>& vCurrInputs = oBatch.getInputArray(nCurrIdx);
            std::array<cv::Mat,2> aInputs;
            for(size_t a=0; a<2u; ++a) {
                aInputs[a] = vCurrInputs[a].clone();
                if(aInputs[a].channels()==1)
                    cv::cvtColor(aInputs[a],aInputs[a],cv::COLOR_GRAY2BGR);
            }
            cv::Mat& oRGBFrame=aInputs[0],oLWIRFrame=aInputs[1];
            lvAssert(!oRGBFrame.empty() && !oLWIRFrame.empty());
            lvAssert(oRGBFrame.size()==vOrigSizes[0] && oLWIRFrame.size()==vOrigSizes[1]);
            lvAssert(oRGBFrame.type()==CV_8UC3 && oLWIRFrame.type()==CV_8UC3);
            cv::Mat oDepthFrame,oBodyIdxFrame;
            if(vCurrInputs.size()>=3) {
                cv::Mat oDepthFrame_raw = vCurrInputs[2].clone();
                lvAssert(!oDepthFrame_raw.empty());
                lvAssert(oDepthFrame_raw.size()==vOrigSizes[2]);
                lvAssert(oDepthFrame_raw.type()==CV_16UC1);
                const bool bTooOldForLZ4 = (oBatch.getName()=="vid04") || (oBatch.getName()=="vid05");
                cv::Mat oBodyIdxFrame_raw = lv::read(oBatch.getDataPath()+"/bodyidx/"+sPacketName+".bin",bTooOldForLZ4?lv::MatArchive_BINARY:lv::MatArchive_BINARY_LZ4);
                lvAssert(oBodyIdxFrame_raw.size()==oDepthFrame_raw.size() && oBodyIdxFrame_raw.type()==CV_8UC1);
            #if SAVE_C2D_MAPPING
                {
                    cv::Mat oCoordMapFrame_c2d(oRGBFrame.size(),CV_32FC2);
                    lvAssertHR(pCoordMapper->MapColorFrameToDepthSpace((UINT)oDepthFrame_raw.total(),(uint16_t*)oDepthFrame_raw.data,(UINT)oRGBFrame.total(),(DepthSpacePoint*)oCoordMapFrame_c2d.data));
                    lv::write(sC2DMappingDir+sPacketName+".bin",oCoordMapFrame_c2d,lv::MatArchive_BINARY_LZ4);
                }
            #endif //SAVE_C2D_MAPPING
                const cv::Mat oCoordMapFrame_c2d = lv::read(sC2DMappingDir+sPacketName+".bin",lv::MatArchive_BINARY_LZ4);
                if(!oCoordMapFrame_c2d.empty()) {
                    lvAssert(oCoordMapFrame_c2d.size()==oRGBFrame.size() && oCoordMapFrame_c2d.type()==CV_32FC2);
                    oDepthFrame.create(oRGBFrame.size(),CV_16UC1);
                    oBodyIdxFrame.create(oRGBFrame.size(),CV_8UC1);
                    oDepthFrame = 0u; // 'dont care'
                    oBodyIdxFrame = 255u; // 'dont care'
                    for(size_t nPxIter=0u; nPxIter<oRGBFrame.total(); ++nPxIter) {
                        const cv::Vec2f vRealPt = ((cv::Vec2f*)oCoordMapFrame_c2d.data)[nPxIter];
                        if(vRealPt[0]>=0 && vRealPt[0]<oDepthFrame_raw.cols && vRealPt[1]>=0 && vRealPt[1]<oDepthFrame_raw.rows) {
                            ((ushort*)oDepthFrame.data)[nPxIter] = oDepthFrame_raw.at<ushort>((int)std::round(vRealPt[1]),(int)std::round(vRealPt[0]));
                            ((uchar*)oBodyIdxFrame.data)[nPxIter] = oBodyIdxFrame_raw.at<uchar>((int)std::round(vRealPt[1]),(int)std::round(vRealPt[0]));
                        }
                    }
                    cv::flip(oDepthFrame,oDepthFrame,1);
                    cv::flip(oBodyIdxFrame,oBodyIdxFrame,1);
                    //cv::Mat oDepthFrameDisplay,oBodyIdxFrameDisplay;
                    //cv::normalize(oDepthFrame,oDepthFrameDisplay,1.0,0.0,cv::NORM_MINMAX,CV_32FC1);
                    //cv::applyColorMap(oDepthFrameDisplay,oDepthFrameDisplay,cv::COLORMAP_BONE);
                    //oBodyIdxFrameDisplay = (oBodyIdxFrame<(BODY_COUNT));
                    //cv::resize(oDepthFrameDisplay,oDepthFrameDisplay,cv::Size(),0.5,0.5);
                    //cv::resize(oBodyIdxFrameDisplay,oBodyIdxFrameDisplay,cv::Size(),0.5,0.5);
                    //cv::imshow("oDepthFrameDisplay",oDepthFrameDisplay);
                    //cv::imshow("oBodyIdxFrameDisplay",oBodyIdxFrameDisplay);
                    //cv::waitKey(1);
                }
            }
        #if GEN_SEGMENTATION_ANNOT
            for(size_t a=0u; a<2u; ++a) {
                if(!aSegmMasks[a].empty())
                    aSegmMasks[a].copyTo(aPrevSegmMasks[a]);
                else
                    aPrevSegmMasks[a] = cv::Mat(vOrigSizes[a],CV_8UC1,cv::Scalar(0));
                aSegmMasks[a] = cv::imread(((a==0u)?sRGBGTDir:sLWIRGTDir)+lv::putf("%05d.png",(int)nCurrIdx),cv::IMREAD_GRAYSCALE);
                if(aSegmMasks[a].empty()) {
                    aSegmMasks[a].create(vOrigSizes[a],CV_8UC1);
                    aSegmMasks[a] = 0u;
                }
                lvAssert(lv::MatInfo(aSegmMasks[a])==lv::MatInfo(vOrigSizes[a],CV_8UC1));
                aTempSegmMasks[a].create(vOrigSizes[a],CV_8UC1);
                aTempSegmMasks[a] = 0u;
            }
        #elif GEN_REGISTRATION_ANNOT
            for(size_t a=0u; a<2u; ++a) {
                avCorrespPts[a].clear();
                const cv::FileStorage oGTFS(((a==0u)?sRGBGTDir:sLWIRGTDir)+lv::putf("%05d.yml",(int)nCurrIdx),cv::FileStorage::READ);
                if(oGTFS.isOpened()) {
                    int nPts;
                    oGTFS["nbpts"] >> nPts;
                    lvAssert(nPts>0);
                    for(int nPtIdx=0; nPtIdx<nPts; ++nPtIdx) {
                        const cv::FileNode oPtNode = oGTFS[lv::putf("pt%04d",nPtIdx)];
                        lvAssert(!oPtNode.empty());
                        int x,y,d;
                        oPtNode["x"] >> x;
                        oPtNode["y"] >> y;
                        oPtNode["d"] >> d;
                        avCorrespPts[a].emplace_back(cv::Point(x,y),d);
                    }
                }
            }
        #endif //GEN_REGISTRATION_ANNOT
            lvLog_(1,"\t annot @ #%d ('%s') of %d",int(nCurrIdx),sPacketName.c_str(),int(nTotPacketCount));
        #ifdef _MSC_VER
            OutputDebugString(lv::convertStrToWStr(lv::putf("\t annot @ #%d ('%s') of %d\n",int(nCurrIdx),sPacketName.c_str(),int(nTotPacketCount))).c_str());
        #endif //def(_MSC_VER)
        #if (GEN_SEGMENTATION_ANNOT || GEN_REGISTRATION_ANNOT)
            int nKeyPressed = -1;
            while(nKeyPressed!=(int)'q' &&
                  nKeyPressed!=27/*escape*/ &&
                  nKeyPressed!=8/*backspace*/ &&
                  (nKeyPressed%256)!=10/*lf*/ &&
                  (nKeyPressed%256)!=13/*enter*/) {
                for(size_t a=0u; a<2u; ++a) {
                #if GEN_SEGMENTATION_ANNOT
                    cv::cvtColor((aSegmMasks[a]|aTempSegmMasks[a]),aSegmMasks_3ch[a],cv::COLOR_GRAY2BGR);
                    aSegmMasks_3ch[a] &= cv::Scalar_<uchar>(0,0,255);
                    cv::addWeighted(aInputs[a],(1-dSegmOpacity),aSegmMasks_3ch[a],dSegmOpacity,0.0,vvDisplayPairs[0][a].first);
                #elif GEN_REGISTRATION_ANNOT
                    aInputs[a].copyTo(vvDisplayPairs[0][a].first);
                    for(size_t nPtIdx=0u; nPtIdx<avCorrespPts[a].size(); ++nPtIdx) {
                        if(bDragInProgress && nSelectedPoint==nPtIdx)
                            cv::rectangle(vvDisplayPairs[0][a].first,cv::Rect(avCorrespPts[a][nPtIdx].first.x,0,1,vOrigSizes[a].height),cv::Scalar_<uchar>(0,0,172));
                        if(nSelectedPoint==nPtIdx)
                            cv::circle(vvDisplayPairs[0][a].first,avCorrespPts[a][nPtIdx].first,anMarkerSizes[a]*2,cv::Scalar::all(1u),-1);
                        cv::circle(vvDisplayPairs[0][a].first,avCorrespPts[a][nPtIdx].first,anMarkerSizes[a],cv::Scalar_<uchar>(lv::getBGRFromHSL(360*float(nPtIdx)/avCorrespPts[a].size(),1.0f,0.5f)),-1);
                    }
                    for(size_t nPtIdx=0u; nPtIdx<avTempPts[a].size(); ++nPtIdx) {
                        cv::circle(vvDisplayPairs[0][a].first,avTempPts[a][nPtIdx],anMarkerSizes[a]*2,cv::Scalar::all(255),-1);
                        cv::circle(vvDisplayPairs[0][a].first,avTempPts[a][nPtIdx],anMarkerSizes[a],cv::Scalar::all(1u),-1);
                    }
                #endif //GEN_REGISTRATION_ANNOT
                }
            #if GEN_SEGMENTATION_ANNOT
                if(nCurrTile!=-1) {
                    const cv::Point2i vMousePos_FP2(int(vMousePos.x*vOrigSizes[nCurrTile].width*4),int(vMousePos.y*vOrigSizes[nCurrTile].height*4));
                    cv::circle(vvDisplayPairs[0][nCurrTile].first,vMousePos_FP2,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(0,0,255),1,cv::LINE_AA,2);
                }
                if(nCurrView==0u || nCurrView==1u)
                    pDisplayHelper->display(std::vector<std::vector<std::pair<cv::Mat,std::string>>>{{vvDisplayPairs[0][nCurrView]}},oDisplayTileSize);
                else
                    pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
            #else //!GEN_SEGMENTATION_ANNOT
                pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
            #endif //!GEN_SEGMENTATION_ANNOT
                nKeyPressed = pDisplayHelper->waitKey(1);
            #if GEN_SEGMENTATION_ANNOT
                if(nKeyPressed==(int)'o') {
                    dSegmOpacity = std::min(dSegmOpacity+0.05,1.0);
                    lvLog_(1,"\topacity now at %f",dSegmOpacity);
                }
                else if(nKeyPressed==(int)'p') {
                    dSegmOpacity = std::max(dSegmOpacity-0.05,0.0);
                    lvLog_(1,"\topacity now at %f",dSegmOpacity);
                }
                else if(nKeyPressed==(int)' ')
                    nCurrView = (nCurrView+1u)%3u;
                else if(nKeyPressed==(int)'z') {
                    if(!oBodyIdxFrame.empty()) {
                        aSegmMasks[0u] = (oBodyIdxFrame<(BODY_COUNT));
                        cv::medianBlur(aSegmMasks[0u],aSegmMasks[0u],5);
                    }
                    else
                        aSegmMasks[0u] = 0u;
                }
                /*if(cv::countNonZero(aSegmMasks[0])==0) {
                    nKeyPressed = 13;
                    break;
                }*/
            #endif //GEN_SEGMENTATION_ANNOT
            }
        #if GEN_SEGMENTATION_ANNOT
            const bool bCurrFrameValid = (cv::countNonZero(aSegmMasks[0])!=0 || cv::countNonZero(aSegmMasks[1])!=0);
            if(bCurrFrameValid) {
                if(cv::countNonZero(aSegmMasks[0])!=0)
                    cv::imwrite(sRGBGTDir+lv::putf("%05d.png",(int)nCurrIdx),aSegmMasks[0]);
                if(cv::countNonZero(aSegmMasks[1])!=0)
                    cv::imwrite(sLWIRGTDir+lv::putf("%05d.png",(int)nCurrIdx),aSegmMasks[1]);
            }
        #elif GEN_REGISTRATION_ANNOT
            const bool bCurrFrameValid = (!avCorrespPts[0].empty() || !avCorrespPts[1].empty());
            if(bCurrFrameValid) {
                lvAssert(avCorrespPts[0].size()==avCorrespPts[1].size());
                for(size_t a=0u; a<2u; ++a) {
                    cv::FileStorage oGTFS(((a==0u)?sRGBGTDir:sLWIRGTDir)+lv::putf("%05d.yml",(int)nCurrIdx),cv::FileStorage::WRITE);
                    lvAssert(oGTFS.isOpened());
                    oGTFS << "htag" << lv::getVersionStamp();
                    oGTFS << "date" << lv::getTimeStamp();
                    oGTFS << "nbpts" << (int)avCorrespPts[a].size();
                    for(size_t nPtIdx=0; nPtIdx<avCorrespPts[a].size(); ++nPtIdx) {
                        oGTFS << lv::putf("pt%04d",(int)nPtIdx) << "{";
                        oGTFS << "x" << avCorrespPts[a][nPtIdx].first.x;
                        oGTFS << "y" << avCorrespPts[a][nPtIdx].first.y;
                        oGTFS << "d" << avCorrespPts[a][nPtIdx].second;
                        oGTFS << "}";
                    }
                }
            }
        #endif //GEN_REGISTRATION_ANNOT
            const int nRealIdx = std::stoi(sPacketName);
            if(bCurrFrameValid && mnSubsetIdxs.find(nRealIdx)==mnSubsetIdxs.end()) {
                mnSubsetIdxs.insert(nRealIdx);
            }
            else if(!bCurrFrameValid && mnSubsetIdxs.find(nRealIdx)!=mnSubsetIdxs.end()) {
                mnSubsetIdxs.erase(nRealIdx);
            }
            if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
                break;
            else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0u)
                --nCurrIdx;
            else if(((nKeyPressed%256)==10/*lf*/ || (nKeyPressed%256)==13/*enter*/) && nCurrIdx<(nTotPacketCount-1u))
                ++nCurrIdx;
        #else //!(GEN_SEGMENTATION_ANNOT || GEN_REGISTRATION_ANNOT)
            ++nCurrIdx;
        #endif //!(GEN_SEGMENTATION_ANNOT || GEN_REGISTRATION_ANNOT)
        }
        {
            cv::FileStorage oGTMetadataFS(oBatch.getDataPath()+sGTName+"_metadata.yml",cv::FileStorage::WRITE);
            lvAssert(oGTMetadataFS.isOpened());
            oGTMetadataFS << "htag" << lv::getVersionStamp();
            oGTMetadataFS << "date" << lv::getTimeStamp();
            oGTMetadataFS << "npackets" << (int)nTotPacketCount;
            oGTMetadataFS << "subsetidxs" << std::vector<int>(mnSubsetIdxs.begin(),mnSubsetIdxs.end());
        }
        lvLog(1,"... batch done.\n");
    }
    catch(const lv::Exception&) {std::cout << "\nAnalyze caught lv::Exception (check stderr)\n" << std::endl;}
    catch(const cv::Exception&) {std::cout << "\nAnalyze caught cv::Exception (check stderr)\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyze caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyze caught unhandled exception\n" << std::endl;}
    try {
        if(pBatch->isPrecaching())
            dynamic_cast<DatasetType::WorkBatch&>(*pBatch).stopPrecaching();
    } catch(...) {
        std::cout << "\nAnalyze caught unhandled exception while attempting to stop batch precaching.\n" << std::endl;
        throw;
    }
}
