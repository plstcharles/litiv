
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

#include "litiv/datasets.hpp"
#include "litiv/imgproc.hpp"
#include "litiv/video.hpp"

////////////////////////////////
#define PROCESS_PREPROC_BGSEGM  0
#define PROCESS_PREPROC_GRABCUT 0
#define WRITE_IMG_OUTPUT        1
#define EVALUATE_OUTPUT         1
#define GLOBAL_VERBOSITY        2
////////////////////////////////
#define DATASET_VAPTRIMOD       0
#define DATASET_LITIV2014       0
#define DATASET_LITIV2018       1
#define DATASET_MINI_TESTS      0
////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      0
#define DATASET_SCALE_FACTOR    1//0.5
#define DATASET_WORKTHREADS     1
////////////////////////////////
#define DATASET_FORCE_RECALC_FEATURES      1
#define DATASET_EVAL_DISPARITY_MASKS       0
#define DATASET_EVAL_BAD_INIT_MASKS        0
#define DATASET_EVAL_APPROX_MASKS_ONLY     0
#define DATASET_EVAL_OUTPUT_MASKS_ONLY     0
#define DATASET_EVAL_INPUT_SUBSET          1
#define DATASET_EVAL_GT_SUBSET             0
#define DATASET_EVAL_FINAL_UPDATE          1
#define DATASET_BATCH_START_INDEX          0
#define DATASET_BATCH_STOP_MAX_INDEX       9999

#if (DATASET_VAPTRIMOD+DATASET_LITIV2014+DATASET_LITIV2018+DATASET_MINI_TESTS/*+...*/)!=1
#error "Must pick a single dataset."
#endif //(DATASET_+.../*+...*/)!=1
#if (DATASET_EVAL_APPROX_MASKS_ONLY+DATASET_EVAL_OUTPUT_MASKS_ONLY)>1
#error "Must pick single output source to evaluate."
#endif //(DATASET_EVAL_APPROX_MASKS_ONLY+DATASET_EVAL_OUTPUT_MASKS_ONLY)>1
#if (DATASET_EVAL_APPROX_MASKS_ONLY && DATASET_EVAL_DISPARITY_MASKS)
#error "Cannot eval approx input masks w/ disp evaluation."
#endif //(DATASET_EVAL_APPROX_MASKS_ONLY && DATASET_EVAL_DISPARITY_MASKS)
#if (((DATASET_EVAL_APPROX_MASKS_ONLY+DATASET_EVAL_OUTPUT_MASKS_ONLY)>0) && WRITE_IMG_OUTPUT)
#error "Should not overwrite output if only reevaluating results."
#endif //(((DATASET_EVAL_APPROX_MASKS_ONLY+DATASET_EVAL_OUTPUT_MASKS_ONLY)>0) && WRITE_IMG_OUTPUT)
#if ((DATASET_EVAL_APPROX_MASKS_ONLY || DATASET_EVAL_OUTPUT_MASKS_ONLY) && DATASET_EVAL_FINAL_UPDATE)
#error "Deferred eval useless here."
#endif //((DATASET_EVAL_APPROX_MASKS_ONLY || DATASET_EVAL_OUTPUT_MASKS_ONLY) && DATASET_EVAL_FINAL_UPDATE)
#define PROCESS_PREPROC (PROCESS_PREPROC_BGSEGM || PROCESS_PREPROC_GRABCUT)
#if DATASET_VAPTRIMOD
#define DATASET_ID Dataset_VAP_trimod2016
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    false,                                        /* bool bLoadDepth=true */\
    PROCESS_PREPROC?false:true,                   /* bool bUndistort=true */\
    PROCESS_PREPROC?false:true,                   /* bool bHorizRectify=false */\
    DATASET_EVAL_DISPARITY_MASKS,                 /* bool bEvalStereoDisp=false */\
    PROCESS_PREPROC_BGSEGM?false:DATASET_EVAL_INPUT_SUBSET,/* bool bLoadFrameSubset=false */\
    DATASET_EVAL_GT_SUBSET,                       /* bool bEvalOnlyFrameSubset=false */\
    PROCESS_PREPROC?0:(int)SegmMatcher::getTemporalDepth(),/* int nEvalTemporalWindowSize=0*/\
    PROCESS_PREPROC?0:1,                          /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
#elif DATASET_LITIV2014
#define DATASET_ID Dataset_LITIV_bilodeau2014
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    DATASET_EVAL_DISPARITY_MASKS,                 /* bool bEvalStereoDisp=true */\
    true,                                         /* bool bFlipDisparities=false */\
    true,                                         /* bool bLoadFrameSubset=true */\
    -1,                                           /* int nLoadPersonSets=-1 */\
    PROCESS_PREPROC?0:1,                          /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
#elif DATASET_LITIV2018
#define DATASET_ID Dataset_LITIV_stcharles2018
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    false,                                        /* bool bLoadDepth=true */\
    PROCESS_PREPROC?false:true,                   /* bool bUndistort=true */\
    PROCESS_PREPROC?false:true,                   /* bool bHorizRectify=true */\
    DATASET_EVAL_DISPARITY_MASKS,                 /* bool bEvalStereoDisp=false */\
    false,                                        /* bool bFlipDisparities=false*/\
    PROCESS_PREPROC_BGSEGM?false:DATASET_EVAL_INPUT_SUBSET,/* bool bLoadFrameSubset=false */\
    DATASET_EVAL_GT_SUBSET,                       /* bool bEvalOnlyFrameSubset=false */\
    PROCESS_PREPROC?0:(int)SegmMatcher::getTemporalDepth(),/* int nEvalTemporalWindowSize=0*/\
    PROCESS_PREPROC?0:1,                          /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
#elif DATASET_MINI_TESTS
#include "cosegm_tests.hpp"
#define DATASET_ID Dataset_CosegmTests
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    DATASET_EVAL_DISPARITY_MASKS,                 /* bool bEvalStereoDisp=false */\
    false,                                        /* bool bLoadFrameSubset=false */\
    1,                                            /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
//#elif DATASET_...
#endif //DATASET_...
#if PROCESS_PREPROC_BGSEGM
#define BGSEGM_ALGO_TYPE BackgroundSubtractorPAWCS
#endif //PROCESS_PREPROC_BGSEGM
#include "litiv/imgproc/SegmMatcher.hpp"

void Analyze(std::string sWorkerName, lv::IDataHandlerPtr pBatch);
#if DATASET_EVAL_DISPARITY_MASKS
using DatasetType = lv::Dataset_<lv::DatasetTask_StereoReg,lv::DATASET_ID,lv::NonParallel>;
#else //!DATASET_EVAL_DISPARITY_MASKS
using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::DATASET_ID,lv::NonParallel>;
#endif //!DATASET_EVAL_DISPARITY_MASKS

int main(int, char**) {
    try {
        lv::setVerbosity(GLOBAL_VERBOSITY);
        DatasetType::Ptr pDataset = DatasetType::create(DATASET_PARAMS);
        lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false);
        const size_t nTotPackets = pDataset->getInputCount();
        const size_t nTotBatches = vpBatches.size();
        if(nTotBatches==0 || nTotPackets==0)
            lvError_("Could not parse any data for dataset '%s'",pDataset->getName().c_str());
        std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
        std::cout << "Executing algorithm with " << DATASET_WORKTHREADS << " thread(s)..." << std::endl;
        lv::WorkerPool<DATASET_WORKTHREADS> oPool;
        std::vector<std::future<void>> vTaskResults;
        size_t nCurrBatchIdx = 1;
        for(lv::IDataHandlerPtr pBatch : vpBatches)
            vTaskResults.push_back(oPool.queueTask(Analyze,std::to_string(nCurrBatchIdx++)+"/"+std::to_string(nTotBatches),pBatch));
        for(std::future<void>& oTaskRes : vTaskResults)
            oTaskRes.get();
        pDataset->writeEvalReport();
    }
    catch(const lv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught lv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception (check stderr)\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    return 0;
}

void Analyze(std::string sWorkerName, lv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        lvAssert(oBatch.getInputPacketType()==lv::ImageArrayPacket && oBatch.getOutputPacketType()==lv::ImageArrayPacket); // app works with stereo heads (2 images at once)
        lvAssert(oBatch.getIOMappingType()==lv::IndexMapping && oBatch.getGTMappingType()==lv::ElemMapping); // segmentation = 1:1 mapping between inputs and gtmasks
#if PROCESS_PREPROC
        lvAssert(oBatch.getInputStreamCount()==2); // expect only one input image per camera
#else //!PROCESS_PREPROC
        lvAssert(oBatch.getInputStreamCount()==4); // expect approx fg masks to be interlaced with input images
#endif //!PROCESS_PREPROC
        lvAssert(oBatch.getOutputStreamCount()==2); // we always only eval one output type at a time (fg masks or disparity)
        if(DATASET_PRECACHING)
            oBatch.startPrecaching(!bool(EVALUATE_OUTPUT));
        const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
        std::cout << "\t\t" << sCurrBatchName << " @ init [" << sWorkerName << "]" << std::endl;
        const std::vector<cv::Mat>& vROIs = oBatch.getFrameROIArray();
        lvAssert(!vROIs.empty() && vROIs.size()==oBatch.getInputStreamCount());
        size_t nCurrIdx = DATASET_BATCH_START_INDEX, nLastTemporalBreakIdx = DATASET_BATCH_START_INDEX;
        lvIgnore(nLastTemporalBreakIdx);
        const std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // note: mat content becomes invalid on next getInput call
        lvAssert(vInitInput.size()==oBatch.getInputStreamCount());
        for(size_t nStreamIdx=0; nStreamIdx<vInitInput.size(); ++nStreamIdx) {
            lvLog_(2,"\tinput %d := %s   (roi=%s)",(int)nStreamIdx,lv::MatInfo(vInitInput[nStreamIdx]).str().c_str(),lv::MatInfo(vROIs[nStreamIdx]).str().c_str());
            if(lv::getVerbosity()>=5) {
                cv::imshow(std::string("vInitInput_")+std::to_string(nStreamIdx),vInitInput[nStreamIdx]);
                cv::imshow(std::string("vROI_")+std::to_string(nStreamIdx),vROIs[nStreamIdx]);
            }
        }
        if(lv::getVerbosity()>=5)
            cv::waitKey(0);
        const std::vector<lv::MatInfo> oInfoArray = oBatch.getInputInfoArray();
        lv::DisplayHelperPtr pDisplayHelper;
        if(lv::getVerbosity()>=1) {
            pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"../");
            pDisplayHelper->setContinuousUpdates(true);
        }
        const size_t nTotPacketCount = std::min(oBatch.getFrameCount(),size_t(DATASET_BATCH_STOP_MAX_INDEX+1));
#if PROCESS_PREPROC_BGSEGM
        lvAssert(!EVALUATE_OUTPUT || !DATASET_EVAL_DISPARITY_MASKS);
        constexpr size_t nExpectedAlgoInputCount = 2u;
        lvIgnore(nExpectedAlgoInputCount);
        std::vector<std::shared_ptr<IBackgroundSubtractor>> vAlgos = {
            std::make_shared<BGSEGM_ALGO_TYPE>(),
            std::make_shared<BGSEGM_ALGO_TYPE>(/*2,30*/)
        };
        const double dDefaultLearningRate = vAlgos[0]->getDefaultLearningRate();
        for(size_t nCamIdx=0; nCamIdx<2; ++nCamIdx) {
            cv::Mat oCurrInput = vInitInput[nCamIdx].clone();
            cv::Mat oCurrROI = vROIs[nCamIdx].clone();
            while(oCurrInput.size().area()>1000*1000) {
                cv::resize(oCurrInput,oCurrInput,cv::Size(),0.5,0.5);
                cv::resize(oCurrROI,oCurrROI,cv::Size(),0.5,0.5,cv::INTER_NEAREST);
            }
            lvAssert(oCurrInput.size()==oCurrROI.size());
            if(oCurrInput.size()!=vInitInput[nCamIdx].size())
                lvLog_(2,"\tdownsizing input #%d to %dx%d...",(int)nCamIdx,oCurrInput.cols,oCurrInput.rows);
            vAlgos[nCamIdx]->initialize(oCurrInput,oCurrROI);
        }
        std::vector<cv::Mat> vCurrFGMasks(2);
#if PROCESS_PREPROC_BGSEGM>1
        const size_t nMaxInitLoops = 50, nMaxInitLoopIdx = 6;
        //const size_t nMaxInitLoops = 8, nMaxInitLoopIdx = 87;
        size_t nCurrInitLoop = 0;
        bool bIncreasingIdxs = true;
#endif //PROCESS_PREPROC_BGSEGM>1
#elif PROCESS_PREPROC_GRABCUT
        lvAssert(!EVALUATE_OUTPUT || !DATASET_EVAL_DISPARITY_MASKS);
        constexpr size_t nExpectedAlgoInputCount = 2u;
        lvAssert(nExpectedAlgoInputCount==vInitInput.size());
        std::array<cv::Mat,nExpectedAlgoInputCount> aCurrInputs,aCurrMasks;
        std::array<cv::Size,nExpectedAlgoInputCount> aOrigSizes;
        for(size_t a=0u; a<2u; ++a)
            aOrigSizes[a] = vInitInput[a].size();
        const std::array<std::vector<cv::Rect>,nExpectedAlgoInputCount> avDefaultBBoxes = {
            std::vector<cv::Rect>(1,cv::Rect(aOrigSizes[0].width/4,aOrigSizes[0].height/4,aOrigSizes[0].width/2,aOrigSizes[0].height/2)),
            std::vector<cv::Rect>(1,cv::Rect(aOrigSizes[1].width/4,aOrigSizes[1].height/4,aOrigSizes[1].width/2,aOrigSizes[1].height/2)),
        };
        std::vector<std::array<std::vector<cv::Rect>,nExpectedAlgoInputCount>> vavBBoxes(nTotPacketCount,avDefaultBBoxes);
        {
            cv::FileStorage oBBoxesFS(oBatch.getOutputPath()+"bboxes.yml",cv::FileStorage::READ);
            if(oBBoxesFS.isOpened()) {
                int nReadPackets;
                oBBoxesFS["npackets"] >> nReadPackets;
                lvAssert(nTotPacketCount==(size_t)nReadPackets);
                for(size_t a=0u; a<2u; ++a) {
                    cv::FileNode oTopNode = oBBoxesFS[std::string("rects")+std::to_string(a)];
                    lvAssert(!oTopNode.empty());
                    for(size_t nIdx=0; nIdx<nTotPacketCount; ++nIdx) {
                        cv::FileNode oFrameNode = oTopNode[(std::string("f")+std::to_string(nIdx))];
                        lvAssert(!oFrameNode.empty());
                        size_t nRectIdx = 0u;
                        cv::FileNode oRectNode = oFrameNode[(std::string("r")+std::to_string(nRectIdx))];
                        lvAssert(!oRectNode.empty());
                        std::vector<cv::Rect> vRects;
                        while(!oRectNode.empty()) {
                            vRects.resize(vRects.size()+1);
                            oRectNode >> vRects.back();
                            oRectNode = oFrameNode[(std::string("r")+std::to_string(++nRectIdx))];
                        }
                        vavBBoxes[nIdx][a] = vRects;
                    }
                }
            }
        #if PROCESS_PREPROC_GRABCUT==1
            else
                lvError("missing bboxes file storage");
        #endif //PROCESS_PREPROC_GRABCUT==1
        }
        const cv::Size oDisplayTileSize(1024,768);
        std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs = {{
            std::make_pair(vInitInput[0].clone(),oBatch.getInputStreamName(0)),
            std::make_pair(vInitInput[1].clone(),oBatch.getInputStreamName(1))
        }};
    #if PROCESS_PREPROC_GRABCUT>1
        const auto lRectArchiver = [&]() {
            lvLog(1,"Archiving rects data to file storage...");
            cv::FileStorage oBBoxesFS(oBatch.getOutputPath()+"bboxes.yml",cv::FileStorage::WRITE);
            lvAssert(oBBoxesFS.isOpened());
            oBBoxesFS << "htag" << lv::getVersionStamp();
            oBBoxesFS << "date" << lv::getTimeStamp();
            oBBoxesFS << "npackets" << (int)nTotPacketCount;
            for(size_t a=0u; a<2u; ++a) {
                oBBoxesFS << (std::string("rects")+std::to_string(a)) << "{";
                for(size_t nIdx=0; nIdx<nTotPacketCount; ++nIdx) {
                    oBBoxesFS << (std::string("f")+std::to_string(nIdx)) << "{";
                    for(size_t nRectIdx=0; nRectIdx<vavBBoxes[nIdx][a].size(); ++nRectIdx)
                        oBBoxesFS << (std::string("r")+std::to_string(nRectIdx)) << vavBBoxes[nIdx][a][nRectIdx];
                    oBBoxesFS << "}";
                }
                oBBoxesFS << "}";
            }
        };
        if(!pDisplayHelper) {
            pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"../");
            pDisplayHelper->setContinuousUpdates(true);
        }
        bool bIsDrawingRect = false, bIsSelectingRect = false;
        size_t nSelectedRect = SIZE_MAX, nSelectedRectTile = SIZE_MAX;
        cv::Point vRectStartPt;
        pDisplayHelper->setMouseCallback([&](const lv::DisplayHelper::CallbackData& oData) {
            if(oData.nEvent==cv::EVENT_LBUTTONUP || oData.nEvent==cv::EVENT_RBUTTONUP) {
                const cv::Point2f vClickPos(float(oData.oInternalPosition.x)/oData.oTileSize.width,float(oData.oInternalPosition.y)/oData.oTileSize.height);
                if(vClickPos.x>=0.0f && vClickPos.y>=0.0f && vClickPos.x<1.0f && vClickPos.y<1.0f) {
                    const int nCurrTile = oData.oPosition.x/oData.oTileSize.width;
                    if(oData.nEvent==cv::EVENT_LBUTTONUP || oData.nEvent==cv::EVENT_RBUTTONUP) {
                        if(bIsDrawingRect && nSelectedRectTile==size_t(nCurrTile)) {
                            lvAssert(!bIsSelectingRect);
                            lvAssert(nSelectedRect!=SIZE_MAX);
                            const cv::Point2i vRectEndPt((int)std::round(vClickPos.x*aOrigSizes[nCurrTile].width),(int)std::round(vClickPos.y*aOrigSizes[nCurrTile].height));
                            vavBBoxes[nCurrIdx][nCurrTile][nSelectedRect] = cv::Rect(vRectStartPt,vRectEndPt);
                            aCurrInputs[nCurrTile].copyTo(vvDisplayPairs[0][nCurrTile].first);
                            if(vvDisplayPairs[0][nCurrTile].first.channels()==1)
                                cv::cvtColor(vvDisplayPairs[0][nCurrTile].first,vvDisplayPairs[0][nCurrTile].first,cv::COLOR_GRAY2BGR);
                            aCurrMasks[nCurrTile].create(aOrigSizes[nCurrTile],CV_8UC1);
                            aCurrMasks[nCurrTile] = uchar(0);
                            cv::Mat oCurrMask(aOrigSizes[nCurrTile],CV_8UC1,cv::Scalar_<uchar>(cv::GC_BGD));
                            for(size_t nRectIdx=0u; nRectIdx<vavBBoxes[nCurrIdx][nCurrTile].size(); ++nRectIdx) {
                                cv::rectangle(vvDisplayPairs[0][nCurrTile].first,vavBBoxes[nCurrIdx][nCurrTile][nRectIdx],cv::Scalar_<uchar>(0,0,255));
                                oCurrMask(vavBBoxes[nCurrIdx][nCurrTile][nRectIdx]) = uchar(cv::GC_PR_FGD);
                            }
                            cv::grabCut(aCurrInputs[nCurrTile],oCurrMask,cv::Rect(),cv::Mat(),cv::Mat(),3,cv::GC_INIT_WITH_MASK);
                            cv::bitwise_or(aCurrMasks[nCurrTile],(oCurrMask==cv::GC_FGD)|(oCurrMask==cv::GC_PR_FGD),aCurrMasks[nCurrTile]);
                            cv::Mat oCurrBGRMask;
                            cv::cvtColor(aCurrMasks[nCurrTile],oCurrBGRMask,cv::COLOR_GRAY2BGR);
                            cv::addWeighted(vvDisplayPairs[0][nCurrTile].first,0.5,oCurrBGRMask,0.5,0.0,vvDisplayPairs[0][nCurrTile].first);
                            bIsDrawingRect = false;
                            nSelectedRect = SIZE_MAX;
                            nSelectedRectTile = SIZE_MAX;
                        }
                        else {
                            vRectStartPt = cv::Point((int)std::round(vClickPos.x*aOrigSizes[nCurrTile].width),(int)std::round(vClickPos.y*aOrigSizes[nCurrTile].height));
                            if(oData.nEvent==cv::EVENT_LBUTTONUP) {
                                if(!bIsSelectingRect) {
                                    size_t nClosestIdx = SIZE_MAX;
                                    float fBestDistance = 9999.f;
                                    for(size_t nRectIdx=0u; nRectIdx<vavBBoxes[nCurrIdx][nCurrTile].size(); ++nRectIdx) {
                                        const float fDistance_TL = (float)cv::norm(vRectStartPt-vavBBoxes[nCurrIdx][nCurrTile][nRectIdx].tl());
                                        if(fDistance_TL<10.0f && fDistance_TL<fBestDistance) {
                                            fBestDistance = fDistance_TL;
                                            nClosestIdx = nRectIdx*2u;
                                        }
                                        const float fDistance_BR = (float)cv::norm(vRectStartPt-vavBBoxes[nCurrIdx][nCurrTile][nRectIdx].br());
                                        if(fDistance_BR<10.0f && fDistance_BR<fBestDistance) {
                                            fBestDistance = fDistance_BR;
                                            nClosestIdx = nRectIdx*2u+1u;
                                        }
                                    }
                                    if(nClosestIdx<vavBBoxes[nCurrIdx][nCurrTile].size()*2u) {
                                        bIsSelectingRect = true;
                                        nSelectedRect = nClosestIdx/2u;
                                        nSelectedRectTile = size_t(nCurrTile);
                                        cv::rectangle(vvDisplayPairs[0][nCurrTile].first,vavBBoxes[nCurrIdx][nCurrTile][nSelectedRect],cv::Scalar_<uchar>(255,0,255));
                                        cv::circle(vvDisplayPairs[0][nCurrTile].first,vavBBoxes[nCurrIdx][nCurrTile][nSelectedRect].tl(),2,cv::Scalar_<uchar>(0,255,0),-1);
                                        cv::circle(vvDisplayPairs[0][nCurrTile].first,vavBBoxes[nCurrIdx][nCurrTile][nSelectedRect].br(),2,cv::Scalar_<uchar>(0,255,0),-1);
                                    }
                                }
                                else if(bIsSelectingRect && nSelectedRectTile==size_t(nCurrTile)) {
                                    lvAssert(nSelectedRect!=SIZE_MAX);
                                    bIsSelectingRect = false;
                                    bIsDrawingRect = true;
                                    cv::circle(vvDisplayPairs[0][nCurrTile].first,vRectStartPt,2,cv::Scalar_<uchar>(255,0,0),-1);
                                }
                            }
                            else if(oData.nEvent==cv::EVENT_RBUTTONUP && !bIsSelectingRect) {
                                vavBBoxes[nCurrIdx][nCurrTile].emplace_back(vRectStartPt,vRectStartPt);
                                bIsDrawingRect = true;
                                nSelectedRect = vavBBoxes[nCurrIdx][nCurrTile].size()-1u;
                                nSelectedRectTile = size_t(nCurrTile);
                                cv::circle(vvDisplayPairs[0][nCurrTile].first,vRectStartPt,2,cv::Scalar_<uchar>(255,0,0),-1);
                            }
                            else if(oData.nEvent==cv::EVENT_RBUTTONUP && bIsSelectingRect && nSelectedRectTile==size_t(nCurrTile)) {
                                lvAssert(nSelectedRect!=SIZE_MAX);
                                if(vavBBoxes[nCurrIdx][nCurrTile].size()>1u) {
                                    vavBBoxes[nCurrIdx][nCurrTile].erase(vavBBoxes[nCurrIdx][nCurrTile].begin()+nSelectedRect);
                                    aCurrInputs[nCurrTile].copyTo(vvDisplayPairs[0][nCurrTile].first);
                                    if(vvDisplayPairs[0][nCurrTile].first.channels()==1)
                                        cv::cvtColor(vvDisplayPairs[0][nCurrTile].first,vvDisplayPairs[0][nCurrTile].first,cv::COLOR_GRAY2BGR);
                                    aCurrMasks[nCurrTile].create(aOrigSizes[nCurrTile],CV_8UC1);
                                    aCurrMasks[nCurrTile] = uchar(0);
                                    cv::Mat oCurrMask(aOrigSizes[nCurrTile],CV_8UC1,cv::Scalar_<uchar>(cv::GC_BGD));
                                    for(size_t nRectIdx=0u; nRectIdx<vavBBoxes[nCurrIdx][nCurrTile].size(); ++nRectIdx) {
                                        cv::rectangle(vvDisplayPairs[0][nCurrTile].first,vavBBoxes[nCurrIdx][nCurrTile][nRectIdx],cv::Scalar_<uchar>(0,0,255));
                                        oCurrMask(vavBBoxes[nCurrIdx][nCurrTile][nRectIdx]) = uchar(cv::GC_PR_FGD);
                                    }
                                    cv::grabCut(aCurrInputs[nCurrTile],oCurrMask,cv::Rect(),cv::Mat(),cv::Mat(),3,cv::GC_INIT_WITH_MASK);
                                    cv::bitwise_or(aCurrMasks[nCurrTile],(oCurrMask==cv::GC_FGD)|(oCurrMask==cv::GC_PR_FGD),aCurrMasks[nCurrTile]);
                                    cv::Mat oCurrBGRMask;
                                    cv::cvtColor(aCurrMasks[nCurrTile],oCurrBGRMask,cv::COLOR_GRAY2BGR);
                                    cv::addWeighted(vvDisplayPairs[0][nCurrTile].first,0.5,oCurrBGRMask,0.5,0.0,vvDisplayPairs[0][nCurrTile].first);
                                    bIsSelectingRect = false;
                                    nSelectedRect = SIZE_MAX;
                                    nSelectedRectTile = SIZE_MAX;
                                }
                            }
                        }
                    }
                }
            }
            //pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
        });
    #endif //PROCESS_PREPROC_GRABCUT>1
    #else //!PROCESS_PREPROC_...
        lvAssert(vInitInput.size()==4 && (vInitInput.size()%2)==0); // assume masks are interlaced with input images
        const size_t nMinDisp = oBatch.getMinDisparity(), nMaxDisp = oBatch.getMaxDisparity();
        lvLog_(2,"\tdisp = [%d,%d]",(int)nMinDisp,(int)nMaxDisp);
        constexpr size_t nCameraCount = SegmMatcher::getCameraCount();
        constexpr size_t nExpectedAlgoInputCount = SegmMatcher::getInputStreamCount();
        constexpr size_t nExpectedAlgoOutputCount = SegmMatcher::getOutputStreamCount();
        static_assert(nCameraCount==2,"unexpected algo internal camera head count");
        static_assert(nExpectedAlgoInputCount==4,"unexpected input stream count for instanced algo");
        static_assert(nExpectedAlgoOutputCount==4,"unexpected output stream count for instanced algo");
        using OutputType = SegmMatcher::OutputLabelType;
        std::vector<cv::Mat_<OutputType>> vCurrOutput(nExpectedAlgoOutputCount);
        std::vector<cv::Mat> vCurrFGMasks(nCameraCount);
        std::vector<cv::Mat> vCurrStereoMaps(nCameraCount);
        std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs(nCameraCount);
        if(lv::getVerbosity()>=2) {
            for(size_t nDisplayRowIdx=0; nDisplayRowIdx<nCameraCount; ++nDisplayRowIdx) {
                std::vector<std::pair<cv::Mat,std::string>> vRow(5);
                vRow[0] = std::make_pair(cv::Mat(),oBatch.getInputStreamName(nDisplayRowIdx*2));
                vRow[1] = std::make_pair(cv::Mat(),"INPUT MASK");
                vRow[2] = std::make_pair(cv::Mat(),"OUTPUT DISP");
                vRow[3] = std::make_pair(cv::Mat(),"OUTPUT MASK");
                vRow[4] = std::make_pair(cv::Mat(),"GT EVAL");
                vvDisplayPairs[nDisplayRowIdx] = vRow;
            }
        }
    #if (!DATASET_EVAL_APPROX_MASKS_ONLY && !DATASET_EVAL_OUTPUT_MASKS_ONLY)
        std::shared_ptr<SegmMatcher> pAlgo = std::make_shared<SegmMatcher>(nMinDisp,nMaxDisp);
        pAlgo->m_pDisplayHelper = pDisplayHelper;
        pAlgo->initialize(std::array<cv::Mat,2>{vROIs[0],vROIs[2]});
        oBatch.setFeaturesDirName(pAlgo->getFeatureExtractorName());
    #endif //(!DATASET_EVAL_APPROX_MASKS_ONLY && !DATASET_EVAL_OUTPUT_MASKS_ONLY)
    #endif //!PROCESS_PREPROC_...
        oBatch.startProcessing();
        while(nCurrIdx<nTotPacketCount) {
            //if(!((nCurrIdx+1)%100))
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << "   [" << sWorkerName << "]" << std::endl;
            std::vector<cv::Mat> vCurrInput = oBatch.getInputArray(nCurrIdx); // caution: should use const ref to vec, but copy here so that we can clone and change mats if needed at app level
            lvDbgAssert(vCurrInput.size()==oBatch.getInputStreamCount());
            lvDbgAssert(vCurrInput.size()==nExpectedAlgoInputCount);
            for(size_t nStreamIdx=0; nStreamIdx<vCurrInput.size(); ++nStreamIdx) {
                lvDbgAssert(oInfoArray[nStreamIdx].size()==vCurrInput[nStreamIdx].size());
            #if PROCESS_PREPROC_BGSEGM>2
                if(nCurrInitLoop<nMaxInitLoops) {
                    const float fStdDev = 0.01;
                    cv::Mat oGaussianNoise(vCurrInput[nStreamIdx].size(),CV_32FC(vCurrInput[nStreamIdx].channels())),oGaussianNoise_BYTE;
                    cv::randn(oGaussianNoise,cv::Scalar(0,0,0),cv::Scalar(fStdDev,fStdDev,fStdDev));
                    cv::add(vCurrInput[nStreamIdx].clone(),oGaussianNoise*255,vCurrInput[nStreamIdx],cv::noArray(),CV_8UC(vCurrInput[nStreamIdx].channels()));
                }
            #endif //PROCESS_PREPROC_BGSEGM>2
            }
        #if PROCESS_PREPROC_BGSEGM
            const double dCurrLearningRate = nCurrIdx<=100?1:dDefaultLearningRate;
        #if USING_OPENMP
            #pragma omp parallel sections
        #endif //USING_OPENMP
            {
            #if USING_OPENMP
                #pragma omp section
            #endif //USING_OPENMP
                {
                    cv::Mat oCurrInput = vCurrInput[0].clone();
                    while(oCurrInput.size().area()>1000*1000)
                        cv::resize(oCurrInput,oCurrInput,cv::Size(),0.5,0.5);
                    cv::Mat oCurrOutput(oCurrInput.size(),CV_8UC1);
                    vAlgos[0]->apply(oCurrInput,oCurrOutput,dCurrLearningRate);
                    if(vCurrInput[0].size()!=oCurrInput.size())
                        cv::resize(oCurrOutput,oCurrOutput,vInitInput[0].size(),0,0,cv::INTER_NEAREST);
                    oCurrOutput.copyTo(vCurrFGMasks[0]);
                }
            #if USING_OPENMP
                #pragma omp section
            #endif //USING_OPENMP
                {
                    lvAssert(vCurrInput[1].size().area()<1000*1000);
                    vAlgos[1]->apply(vCurrInput[1],vCurrFGMasks[1],dCurrLearningRate);
                }
            }
            if(lv::getVerbosity()>=3) {
                std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs(2);
                const std::vector<cv::Mat>& vCurrEvalRes = oBatch.getColoredMaskArray(vCurrFGMasks,nCurrIdx);
                for(size_t nCamIdx=0; nCamIdx<2; ++nCamIdx) {
                    vvDisplayPairs[nCamIdx].resize(3);
                    vvDisplayPairs[nCamIdx][0].first = vCurrInput[nCamIdx];
                    vvDisplayPairs[nCamIdx][0].second = std::to_string(nCurrIdx);
                    vAlgos[nCamIdx]->getBackgroundImage(vvDisplayPairs[nCamIdx][1].first);
                    if(vvDisplayPairs[nCamIdx][1].first.size()!=vCurrInput[nCamIdx].size())
                        cv::resize(vvDisplayPairs[nCamIdx][1].first,vvDisplayPairs[nCamIdx][1].first,vCurrInput[nCamIdx].size());
                    vCurrEvalRes[nCamIdx].copyTo(vvDisplayPairs[nCamIdx][2].first);
                    if(!vROIs[nCamIdx].empty()) {
                        cv::bitwise_or(vvDisplayPairs[nCamIdx][1].first,UCHAR_MAX/2,vvDisplayPairs[nCamIdx][1].first,vROIs[nCamIdx]==0);
                        cv::bitwise_or(vvDisplayPairs[nCamIdx][2].first,UCHAR_MAX/2,vvDisplayPairs[nCamIdx][2].first,vROIs[nCamIdx]==0);
                    }
                }
                lvAssert(pDisplayHelper);
                pDisplayHelper->display(vvDisplayPairs,cv::Size(640,480));
                const int nKeyPressed = pDisplayHelper->waitKey();
                if(nKeyPressed==(int)'q')
                    break;
            }
            oBatch.push(vCurrFGMasks,nCurrIdx);
        #if PROCESS_PREPROC_BGSEGM>1
            if(nCurrInitLoop<nMaxInitLoops) {
                const size_t nMaxPretrainFrames = nMaxInitLoops*(nMaxInitLoopIdx+1)*2;
                const size_t nCurrPretrainFrame = nCurrInitLoop*(nMaxInitLoopIdx+1)*2 + ((bIncreasingIdxs)?(nCurrIdx+1):(2*(nMaxInitLoopIdx+1)-nCurrIdx));
                std::cout << "\t\t\t   pretrain @ " <<  nCurrPretrainFrame << "/" << nMaxPretrainFrames << "   [" << sWorkerName << "]" << std::endl;
                if(nCurrIdx==nMaxInitLoopIdx && bIncreasingIdxs)
                    bIncreasingIdxs = false;
                else if(bIncreasingIdxs)
                    ++nCurrIdx;
                else if(nCurrIdx==0)
                    bIncreasingIdxs = bool(++nCurrInitLoop);
                else
                    --nCurrIdx;
            }
            else
                ++nCurrIdx;
        #else //!(PROCESS_PREPROC_BGSEGM>1)
            ++nCurrIdx;
        #endif //!(PROCESS_PREPROC_BGSEGM>1)
        #elif PROCESS_PREPROC_GRABCUT
            lvAssert(vCurrInput.size()==nExpectedAlgoInputCount);
            for(size_t a=0u; a<nExpectedAlgoInputCount; ++a) {
                lvAssert(!vCurrInput.empty());
                if(pDisplayHelper)
                    vCurrInput[a].copyTo(vvDisplayPairs[0][a].first);
                vCurrInput[a].copyTo(aCurrInputs[a]);
                if(aCurrInputs[a].channels()==1) {
                    if(pDisplayHelper)
                        cv::cvtColor(vvDisplayPairs[0][a].first,vvDisplayPairs[0][a].first,cv::COLOR_GRAY2BGR);
                    cv::cvtColor(aCurrInputs[a],aCurrInputs[a],cv::COLOR_GRAY2BGR);
                }
                aCurrMasks[a].create(aOrigSizes[a],CV_8UC1);
                aCurrMasks[a] = uchar(0);
                cv::Mat oCurrMask(aOrigSizes[a],CV_8UC1,cv::Scalar_<uchar>(cv::GC_BGD));
                for(size_t nRectIdx=0u; nRectIdx<vavBBoxes[nCurrIdx][a].size(); ++nRectIdx) {
                    if(pDisplayHelper)
                        cv::rectangle(vvDisplayPairs[0][a].first,vavBBoxes[nCurrIdx][a][nRectIdx],cv::Scalar_<uchar>(0,0,255));
                    oCurrMask(vavBBoxes[nCurrIdx][a][nRectIdx]) = uchar(cv::GC_PR_FGD);
                }
                cv::grabCut(aCurrInputs[a],oCurrMask,cv::Rect(),cv::Mat(),cv::Mat(),3,cv::GC_INIT_WITH_MASK);
                cv::bitwise_or(aCurrMasks[a],(oCurrMask==cv::GC_FGD)|(oCurrMask==cv::GC_PR_FGD),aCurrMasks[a]);
                if(pDisplayHelper) {
                    cv::Mat oCurrBGRMask;
                    cv::cvtColor(aCurrMasks[a],oCurrBGRMask,cv::COLOR_GRAY2BGR);
                    cv::addWeighted(vvDisplayPairs[0][a].first,0.5,oCurrBGRMask,0.5,0.0,vvDisplayPairs[0][a].first);
                }
            }
        #if PROCESS_PREPROC_GRABCUT>1
            lvLog_(1,"\t grabcut @ #%d",int(nCurrIdx));
            int nKeyPressed = -1;
            while(nKeyPressed!=(int)'q' && nKeyPressed!=27/*escape*/ && nKeyPressed!=8/*backspace*/ && (nKeyPressed%256)!=10/*lf*/ && (nKeyPressed%256)!=13/*enter*/) {
                pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
                nKeyPressed = pDisplayHelper->waitKey();
            }
            bIsDrawingRect = false;
            bIsSelectingRect = false;
            nSelectedRect = SIZE_MAX;
            nSelectedRectTile = SIZE_MAX;
            if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
                break;
            else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0u)
                --nCurrIdx;
            else if(((nKeyPressed%256)==10/*lf*/ || (nKeyPressed%256)==13/*enter*/) && nCurrIdx<(nTotPacketCount-1u))
                ++nCurrIdx;
            lRectArchiver();
        #else //PROCESS_PREPROC_GRABCUT<=1
            oBatch.push(aCurrMasks,nCurrIdx);
            ++nCurrIdx;
            if(pDisplayHelper) {
                pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
                const int nKeyPressed = pDisplayHelper->waitKey(lv::getVerbosity()>=4?0:1);
                if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
                    break;
            }
        #endif //PROCESS_PREPROC_GRABCUT<=1
        #else //!PROCESS_PREPROC_...
        #if (DATASET_EVAL_APPROX_MASKS_ONLY || DATASET_EVAL_OUTPUT_MASKS_ONLY)
        #if DATASET_EVAL_OUTPUT_MASKS_ONLY
            const std::vector<cv::Mat> vArchivedOutput = oBatch.loadOutputArray(nCurrIdx);
        #endif //DATASET_EVAL_OUTPUT_MASKS_ONLY
            for(size_t nCamIdx=0; nCamIdx<nCameraCount; ++nCamIdx) {
                const size_t nOutputMaskIdx = nCamIdx*SegmMatcher::OutputPackOffset+SegmMatcher::OutputPackOffset_Mask;
                const size_t nOutputDispIdx = nCamIdx*SegmMatcher::OutputPackOffset+SegmMatcher::OutputPackOffset_Disp;
        #if DATASET_EVAL_APPROX_MASKS_ONLY
                const size_t nInputMaskIdx = nCamIdx*SegmMatcher::InputPackOffset+SegmMatcher::InputPackOffset_Mask;
                vCurrInput[nInputMaskIdx].convertTo(vCurrOutput[nOutputDispIdx],CV_32S); // only to fool output checks
                vCurrInput[nInputMaskIdx].convertTo(vCurrOutput[nOutputMaskIdx],CV_32S,1.0/255);
        #endif //DATASET_EVAL_APPROX_MASKS_ONLY
        #if DATASET_EVAL_OUTPUT_MASKS_ONLY
                if(oBatch.isEvaluatingDisparities()) {
                    vArchivedOutput[nCamIdx].convertTo(vCurrOutput[nOutputDispIdx],CV_32S);
                    cv::Mat_<OutputType>(oInfoArray[nCamIdx].size(),OutputType(0)).copyTo(vCurrOutput[nOutputMaskIdx]); // only to fool output checks
                }
                else {
                    cv::Mat_<OutputType>(oInfoArray[nCamIdx].size(),OutputType(0)).copyTo(vCurrOutput[nOutputDispIdx]); // only to fool output checks
                    vArchivedOutput[nCamIdx].convertTo(vCurrOutput[nOutputMaskIdx],CV_32S);
                }
        #endif //DATASET_EVAL_OUTPUT_MASKS_ONLY
            }
        #else //!(DATASET_EVAL_APPROX_MASKS_ONLY || DATASET_EVAL_OUTPUT_MASKS_ONLY)
            if(oBatch.isTemporalWindowBreak(nCurrIdx)) {
                // only useful for subset processing; should never be triggered in other conditions
                lvDbgAssert(DATASET_EVAL_INPUT_SUBSET || DATASET_EVAL_GT_SUBSET);
                std::cout << "\t\t\t" << " -> temporal break detected, resetting temporal model" << std::endl;
                pAlgo->resetTemporalModel();
                nLastTemporalBreakIdx = nCurrIdx;
            }
            const bool bNextIdxIsTemporalBreak = oBatch.isTemporalWindowBreak(nCurrIdx+1u);
            lvIgnore(bNextIdxIsTemporalBreak);
        #if DATASET_EVAL_BAD_INIT_MASKS
            const size_t nModifMaskIdx = SegmMatcher::InputPack_LeftMask;
            cv::Mat oModifMask = vCurrInput[nModifMaskIdx].clone();
            lvAssert(!oModifMask.empty() && oModifMask.type()==CV_8UC1);
            oModifMask = 0;
            vCurrInput[nModifMaskIdx] = oModifMask;
        #endif //DATASET_EVAL_BAD_INIT_MASKS
        #if !DATASET_FORCE_RECALC_FEATURES
            const cv::Mat& oNextFeatsPacket = oBatch.loadFeatures(nCurrIdx);
            if(!oNextFeatsPacket.empty())
                pAlgo->setNextFeatures(oNextFeatsPacket);
            else
        #endif //!DATASET_FORCE_RECALC_FEATURES
            {
                cv::Mat oNewFeatsPacket;
                pAlgo->calcFeatures(lv::convertVectorToArray<nExpectedAlgoInputCount>(vCurrInput),&oNewFeatsPacket);
                //oBatch.saveFeatures(nCurrIdx,oNewFeatsPacket);
                pAlgo->setNextFeatures(oNewFeatsPacket);
            }
            pAlgo->apply(vCurrInput,vCurrOutput/*,dDefaultThreshold*/);
        #endif //!(DATASET_EVAL_APPROX_MASKS_ONLY || DATASET_EVAL_OUTPUT_MASKS_ONLY)
            lvDbgAssert(vCurrOutput.size()==nExpectedAlgoOutputCount);
            using OutputLabelType = SegmMatcher::LabelType;
            for(size_t nOutputArrayIdx=0; nOutputArrayIdx<vCurrOutput.size(); ++nOutputArrayIdx) {
                lvAssert(vCurrOutput[nOutputArrayIdx].type()==lv::MatRawType_<OutputLabelType>());
                lvAssert(vCurrOutput[nOutputArrayIdx].size()==oInfoArray[nOutputArrayIdx].size());
                const size_t nRealOutputArrayIdx = nOutputArrayIdx/SegmMatcher::OutputPackOffset;
                if((nOutputArrayIdx%SegmMatcher::OutputPackOffset)==SegmMatcher::OutputPackOffset_Disp) {
                    double dMin,dMax;
                    cv::minMaxIdx(vCurrOutput[nOutputArrayIdx],&dMin,&dMax);
                    lvDbgAssert_(dMin>=0 && dMax<=255,"unexpected min/max disp for 8u mats");
                    vCurrOutput[nOutputArrayIdx].convertTo(vCurrStereoMaps[nRealOutputArrayIdx],CV_8U);
                }
                else if((nOutputArrayIdx%SegmMatcher::OutputPackOffset)==SegmMatcher::OutputPackOffset_Mask)
                    cv::Mat(vCurrOutput[nOutputArrayIdx]!=0).copyTo(vCurrFGMasks[nRealOutputArrayIdx]);
            }
            lvDbgAssert(vCurrFGMasks.size()==oBatch.getOutputStreamCount());
            lvDbgAssert(vCurrStereoMaps.size()==oBatch.getOutputStreamCount());
            if(lv::getVerbosity()>=3) {
                lvAssert(oBatch.getGTStreamCount()==vCurrFGMasks.size() && oBatch.getGTStreamCount()==vCurrStereoMaps.size());
                const std::vector<cv::Mat>& vCurrEvalRes = oBatch.getColoredMaskArray(oBatch.isEvaluatingDisparities()?vCurrStereoMaps:vCurrFGMasks,nCurrIdx);
                for(size_t nDisplayRowIdx=0; nDisplayRowIdx<nCameraCount; ++nDisplayRowIdx) {
                    vCurrInput[nDisplayRowIdx*2].copyTo(vvDisplayPairs[nDisplayRowIdx][0].first);
                    vCurrInput[nDisplayRowIdx*2+1].copyTo(vvDisplayPairs[nDisplayRowIdx][1].first);
                #if !DATASET_EVAL_APPROX_MASKS_ONLY && !DATASET_EVAL_OUTPUT_MASKS_ONLY
                    pAlgo->getStereoDispMapDisplay(0,nDisplayRowIdx).copyTo(vvDisplayPairs[nDisplayRowIdx][2].first);
                    pAlgo->getResegmMapDisplay(0,nDisplayRowIdx).copyTo(vvDisplayPairs[nDisplayRowIdx][3].first);
                #endif //!DATASET_EVAL_APPROX_MASKS_ONLY && !DATASET_EVAL_OUTPUT_MASKS_ONLY
                    vCurrEvalRes[nDisplayRowIdx].copyTo(vvDisplayPairs[nDisplayRowIdx][4].first);
                }
                lvAssert(pDisplayHelper);
                pDisplayHelper->display(vvDisplayPairs,cv::Size(320,240));
                const int nKeyPressed = pDisplayHelper->waitKey(lv::getVerbosity()>=4?0:1);
                if(nKeyPressed==(int)'q')
                    break;
            }
        #if DATASET_EVAL_FINAL_UPDATE
            SegmMatcher::MatArrayOut aUpdatedOutputs;
            if(bNextIdxIsTemporalBreak) {
                // here, we have catching up to do, as the model will be reset next frame; all results need to be pushed
                const size_t nOutputsToPush = std::min(nCurrIdx-nLastTemporalBreakIdx,SegmMatcher::getTemporalDepth())+1u;
                for(size_t nPushOffsetIdx=nOutputsToPush; nPushOffsetIdx>0; --nPushOffsetIdx) {
                    lvDbgAssert(SegmMatcher::getTemporalDepth()>=(nPushOffsetIdx-1u));
                    pAlgo->getOutput(nPushOffsetIdx-1u,aUpdatedOutputs);
                    lvDbgAssert(aUpdatedOutputs.size()==nExpectedAlgoOutputCount);
                    for(size_t nOutputArrayIdx=0; nOutputArrayIdx<aUpdatedOutputs.size(); ++nOutputArrayIdx) {
                        lvAssert(aUpdatedOutputs[nOutputArrayIdx].type()==lv::MatRawType_<OutputLabelType>());
                        lvAssert(aUpdatedOutputs[nOutputArrayIdx].size()==oInfoArray[nOutputArrayIdx].size());
                        const size_t nRealOutputArrayIdx = nOutputArrayIdx/SegmMatcher::OutputPackOffset;
                        if((nOutputArrayIdx%SegmMatcher::OutputPackOffset)==SegmMatcher::OutputPackOffset_Disp) {
                            double dMin,dMax;
                            cv::minMaxIdx(aUpdatedOutputs[nOutputArrayIdx],&dMin,&dMax);
                            lvDbgAssert_(dMin>=0 && dMax<=255,"unexpected min/max disp for 8u mats");
                            aUpdatedOutputs[nOutputArrayIdx].convertTo(vCurrStereoMaps[nRealOutputArrayIdx],CV_8U);
                        }
                        else if((nOutputArrayIdx%SegmMatcher::OutputPackOffset)==SegmMatcher::OutputPackOffset_Mask)
                            cv::Mat(aUpdatedOutputs[nOutputArrayIdx]!=0).copyTo(vCurrFGMasks[nRealOutputArrayIdx]);
                    }
                    lvDbgAssert(vCurrFGMasks.size()==oBatch.getOutputStreamCount());
                    lvDbgAssert(vCurrStereoMaps.size()==oBatch.getOutputStreamCount());
                    const size_t nEvalIdx = nCurrIdx-(nPushOffsetIdx-1u);
                    if(oBatch.isEvaluatingDisparities())
                        oBatch.push(vCurrStereoMaps,nEvalIdx);
                    else
                        oBatch.push(vCurrFGMasks,nEvalIdx);
                }
            }
            else if((nCurrIdx-nLastTemporalBreakIdx)>=SegmMatcher::getTemporalDepth()) {
                // outputs can be improved over new iterations; by default, we only push the top temporal layer for eval
                pAlgo->getOutput(SegmMatcher::getTemporalDepth(),aUpdatedOutputs);
                lvDbgAssert(aUpdatedOutputs.size()==nExpectedAlgoOutputCount);
                for(size_t nOutputArrayIdx=0; nOutputArrayIdx<aUpdatedOutputs.size(); ++nOutputArrayIdx) {
                    lvAssert(aUpdatedOutputs[nOutputArrayIdx].type()==lv::MatRawType_<OutputLabelType>());
                    lvAssert(aUpdatedOutputs[nOutputArrayIdx].size()==oInfoArray[nOutputArrayIdx].size());
                    const size_t nRealOutputArrayIdx = nOutputArrayIdx/SegmMatcher::OutputPackOffset;
                    if((nOutputArrayIdx%SegmMatcher::OutputPackOffset)==SegmMatcher::OutputPackOffset_Disp) {
                        double dMin,dMax;
                        cv::minMaxIdx(aUpdatedOutputs[nOutputArrayIdx],&dMin,&dMax);
                        lvDbgAssert_(dMin>=0 && dMax<=255,"unexpected min/max disp for 8u mats");
                        aUpdatedOutputs[nOutputArrayIdx].convertTo(vCurrStereoMaps[nRealOutputArrayIdx],CV_8U);
                    }
                    else if((nOutputArrayIdx%SegmMatcher::OutputPackOffset)==SegmMatcher::OutputPackOffset_Mask)
                        cv::Mat(aUpdatedOutputs[nOutputArrayIdx]!=0).copyTo(vCurrFGMasks[nRealOutputArrayIdx]);
                }
                lvDbgAssert(vCurrFGMasks.size()==oBatch.getOutputStreamCount());
                lvDbgAssert(vCurrStereoMaps.size()==oBatch.getOutputStreamCount());
                const size_t nEvalIdx = nCurrIdx-SegmMatcher::getTemporalDepth();
                if(oBatch.isEvaluatingDisparities())
                    oBatch.push(vCurrStereoMaps,nEvalIdx);
                else
                    oBatch.push(vCurrFGMasks,nEvalIdx);
            }
            ++nCurrIdx;
        #else //!DATASET_EVAL_FINAL_UPDATE
            if(oBatch.isEvaluatingDisparities())
                oBatch.push(vCurrStereoMaps,nCurrIdx++);
            else
                oBatch.push(vCurrFGMasks,nCurrIdx++);
        #endif //!DATASET_EVAL_FINAL_UPDATE
        #endif //!PROCESS_PREPROC_...
        }
        oBatch.stopProcessing();
        const double dTimeElapsed = oBatch.getFinalProcessTime();
        const double dProcessSpeed = (double)nCurrIdx/dTimeElapsed;
        std::cout << "\t\t" << sCurrBatchName << " @ end [" << sWorkerName << "] (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oBatch.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
    }
    catch(const lv::Exception& e) {std::cout << "\nAnalyze caught lv::Exception (check stderr)\n" << std::endl;}
    catch(const cv::Exception& e) {std::cout << "\nAnalyze caught cv::Exception (check stderr)\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyze caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyze caught unhandled exception\n" << std::endl;}
    try {
        if(pBatch->isProcessing())
            dynamic_cast<DatasetType::WorkBatch&>(*pBatch).stopProcessing();
    } catch(...) {
        std::cout << "\nAnalyze caught unhandled exception while attempting to stop batch processing.\n" << std::endl;
        throw;
    }
}