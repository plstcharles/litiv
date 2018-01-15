
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
////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      1
////////////////////////////////
#define DATASETS_LITIV2018_CALIB_VERSION 2

#if (GEN_SEGMENTATION_ANNOT+GEN_REGISTRATION_ANNOT)!=1
#error "must select one type of annotation to generate"
#endif //(GEN_SEGMENTATION_ANNOT+GEN_REGISTRATION_ANNOT)!=1

#include "litiv/datasets.hpp"
#include "litiv/imgproc.hpp"
#include <opencv2/calib3d.hpp>

using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::Dataset_LITIV_stcharles2018,lv::NonParallel>;
void Analyze(lv::IDataHandlerPtr pBatch);

int main(int, char**) {
    try {
        DatasetType::Ptr pDataset = DatasetType::create(
                DATASET_OUTPUT_PATH, // const std::string& sOutputDirName
                false, //bool bSaveOutput
                false, //bool bUseEvaluator
                (GEN_SEGMENTATION_ANNOT==1), //bool bLoadDepth
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

        lv::DisplayHelperPtr pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
        pDisplayHelper->setContinuousUpdates(true);
        pDisplayHelper->setDisplayCursor(!GEN_SEGMENTATION_ANNOT);
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
        const std::string sRGBGTDir = oBatch.getDataPath()+(GEN_REGISTRATION_ANNOT?"rgb_gt_disp/":"rgb_gt_masks/");
        const std::string sLWIRGTDir = oBatch.getDataPath()+(GEN_REGISTRATION_ANNOT?"lwir_gt_disp/":"lwir_gt_masks/");
        lv::createDirIfNotExist(sRGBGTDir); lv::createDirIfNotExist(sLWIRGTDir);

        std::set<int> mnSubsetIdxs;
        {
            cv::FileStorage oGTMetadataFS(sRGBGTDir+"gtmetadata.yml",cv::FileStorage::READ);
            if(oGTMetadataFS.isOpened()) {
                int nPrevPacketCount;
                oGTMetadataFS["npackets"] >> nPrevPacketCount;
                lvAssert(nPrevPacketCount==(int)nTotPacketCount);
                std::vector<int> vnPrevSubsetIdxs;
                oGTMetadataFS["subsetidxs"] >> vnPrevSubsetIdxs;
                mnSubsetIdxs.insert(vnPrevSubsetIdxs.begin(),vnPrevSubsetIdxs.end());
            }
        }

    #if GEN_SEGMENTATION_ANNOT
        std::array<cv::Mat,2> aSegmMasks,aSegmMasks_3ch;
    #elif GEN_REGISTRATION_ANNOT
        std::array<std::vector<std::pair<cv::Point2i,int>>,2> avRegistrPts;
    #endif //GEN_REGISTRATION_ANNOT
        // @@@@@ add circle for tool size, add drag and drop via user side toggle
        double dSegmOpacity = 0.5;
        std::array<double,2> adSegmToolRadius={10,3};
        const std::array<double,2> adSegmToolRadiusMin={3,1},adSegmToolRadiusMax={80,40};
        cv::Point2f vMousePos(0.5f,0.5f),vLastMousePos(0.5f,0.5f);
        int nCurrTile = -1;
        pDisplayHelper->setMouseCallback([&](const lv::DisplayHelper::CallbackData& oData) {
            vMousePos = cv::Point2f(float(oData.oInternalPosition.x)/oData.oTileSize.width,float(oData.oInternalPosition.y)/oData.oTileSize.height);
            if(vMousePos.x>=0.0f && vMousePos.y>=0.0f && vMousePos.x<1.0f && vMousePos.y<1.0f) {
                nCurrTile = oData.oPosition.x/oData.oTileSize.width;
                if(oData.nEvent==cv::EVENT_MOUSEWHEEL && cv::getMouseWheelDelta(oData.nFlags)>0) {
                    adSegmToolRadius[nCurrTile] = std::min(adSegmToolRadius[nCurrTile]+std::max(1.0,adSegmToolRadius[nCurrTile]/6),adSegmToolRadiusMax[nCurrTile]);
                    lvLog_(2,"\tnew tool size = %f",adSegmToolRadius[nCurrTile]);
                }
                else if(oData.nEvent==cv::EVENT_MOUSEWHEEL && cv::getMouseWheelDelta(oData.nFlags)<0) {
                    adSegmToolRadius[nCurrTile] = std::max(adSegmToolRadius[nCurrTile]-std::max(1.0,adSegmToolRadius[nCurrTile]/6),adSegmToolRadiusMin[nCurrTile]);
                    lvLog_(2,"\tnew tool size = %f",adSegmToolRadius[nCurrTile]);
                }
                if(oData.nEvent==cv::EVENT_LBUTTONDOWN || (oData.nEvent==cv::EVENT_MOUSEMOVE && oData.nFlags==cv::EVENT_FLAG_LBUTTON)) {
                    const cv::Point2i vMousePos_FP2(int(vMousePos.x*vOrigSizes[nCurrTile].width*4),int(vMousePos.y*vOrigSizes[nCurrTile].height*4));
                    const cv::Point2i vLastMousePos_FP2(int(vLastMousePos.x*vOrigSizes[nCurrTile].width*4),int(vLastMousePos.y*vOrigSizes[nCurrTile].height*4));
                    const cv::Point2i vMousePosDiff = vMousePos_FP2-vLastMousePos_FP2;
                    const int nMoveIter = int(cv::norm(vMousePosDiff)/4.0);
                    for(int nCurrIter=1; nCurrIter<=nMoveIter; ++nCurrIter) {
                        const cv::Point2i vMouseCurrPos = vLastMousePos_FP2+cv::Point2i(vMousePosDiff.x*nCurrIter/nMoveIter,vMousePosDiff.y*nCurrIter/nMoveIter);
                        cv::circle(aSegmMasks[nCurrTile],vMouseCurrPos,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(255),-1,cv::LINE_8,2);
                    }
                    cv::circle(aSegmMasks[nCurrTile],vMousePos_FP2,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(255),-1,cv::LINE_8,2);
                }
                else if(oData.nEvent==cv::EVENT_RBUTTONDOWN || (oData.nEvent==cv::EVENT_MOUSEMOVE && oData.nFlags==cv::EVENT_FLAG_RBUTTON)) {
                    const cv::Point2i vMousePos_FP2(int(vMousePos.x*vOrigSizes[nCurrTile].width*4),int(vMousePos.y*vOrigSizes[nCurrTile].height*4));
                    const cv::Point2i vLastMousePos_FP2(int(vLastMousePos.x*vOrigSizes[nCurrTile].width*4),int(vLastMousePos.y*vOrigSizes[nCurrTile].height*4));
                    const cv::Point2i vMousePosDiff = vMousePos_FP2-vLastMousePos_FP2;
                    const int nMoveIter = int(cv::norm(vMousePosDiff)/4.0);
                    for(int nCurrIter=1; nCurrIter<=nMoveIter; ++nCurrIter) {
                        const cv::Point2i vMouseCurrPos = vLastMousePos_FP2+cv::Point2i(vMousePosDiff.x*nCurrIter/nMoveIter,vMousePosDiff.y*nCurrIter/nMoveIter);
                        cv::circle(aSegmMasks[nCurrTile],vMouseCurrPos,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(0),-1,cv::LINE_8,2);
                    }
                    cv::circle(aSegmMasks[nCurrTile],vMousePos_FP2,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(0),-1,cv::LINE_8,2);
                }
                if(oData.nEvent==cv::EVENT_LBUTTONUP || oData.nEvent==cv::EVENT_RBUTTONUP) {
                    for(size_t a=0u; a<2u; ++a)
                        cv::compare(aSegmMasks[a],128u,aSegmMasks[a],cv::CMP_GT);
                }
            }
            else
                nCurrTile = -1;
            vLastMousePos = vMousePos;
            //pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
        });
        lvLog(1,"Annotation edit mode initialized...");
        nCurrIdx = 0u;
        while(nCurrIdx<nTotPacketCount) {
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
            cv::Mat oDepthFrame;
            if(vCurrInputs.size()>=3) {
                oDepthFrame = vCurrInputs[2].clone();
                lvAssert(!oDepthFrame.empty());
                lvAssert(oDepthFrame.size()==vOrigSizes[2]);
                lvAssert(oDepthFrame.type()==CV_16UC1);
            }
            const std::string sPacketName = oBatch.getInputName(nCurrIdx);
    #if GEN_SEGMENTATION_ANNOT
            aSegmMasks[0] = cv::imread(sRGBGTDir+lv::putf("%05d.png",(int)nCurrIdx),cv::IMREAD_GRAYSCALE);
            if(aSegmMasks[0].empty()) {
                aSegmMasks[0].create(vOrigSizes[0],CV_8UC1);
                aSegmMasks[0] = 0u;
                // @@@ init with depth here?
            }
            aSegmMasks[1] = cv::imread(sLWIRGTDir+lv::putf("%05d.png",(int)nCurrIdx),cv::IMREAD_GRAYSCALE);
            if(aSegmMasks[1].empty()) {
                aSegmMasks[1].create(vOrigSizes[1],CV_8UC1);
                aSegmMasks[1] = 0u;
                // @@@ try to reg w/ depth here?
            }
    #elif GEN_REGISTRATION_ANNOT
            {
                cv::FileStorage oGTFS_RGB(sRGBGTDir+lv::putf("%05d.yml",(int)nCurrIdx),cv::FileStorage::READ);
                cv::FileStorage oGTFS_LWIR(sLWIRGTDir+lv::putf("%05d.yml",(int)nCurrIdx),cv::FileStorage::READ);
                @@@ missing impl
            }
    #endif //GEN_REGISTRATION_ANNOT
            lvLog_(1,"\t annot @ #%d ('%s') of %d",int(nCurrIdx),sPacketName.c_str(),int(nTotPacketCount));
            int nKeyPressed = -1;
            while(nKeyPressed!=(int)'q' &&
                  nKeyPressed!=27/*escape*/ &&
                  nKeyPressed!=8/*backspace*/ &&
                  (nKeyPressed%256)!=10/*lf*/ &&
                  (nKeyPressed%256)!=13/*enter*/) {
                for(size_t a=0u; a<2u; ++a) {
                #if GEN_SEGMENTATION_ANNOT
                    cv::cvtColor(aSegmMasks[a],aSegmMasks_3ch[a],cv::COLOR_GRAY2BGR);
                    cv::addWeighted(aInputs[a],(1-dSegmOpacity),aSegmMasks_3ch[a],dSegmOpacity,0.0,vvDisplayPairs[0][a].first);
                #elif GEN_REGISTRATION_ANNOT
                    aInputs[a].copyTo(vvDisplayPairs[0][a].first);
                    @@@@
                    for(size_t nPtIdx=0u; nPtIdx<avvImagePts[a][nCurrIdx].size(); ++nPtIdx) {
                        if(nSelectedPoint==nPtIdx)
                            cv::circle(vvDisplayPairs[0][a].first,cv::Point2i(avvImagePts[a][nCurrIdx][nPtIdx]),anMarkerSizes[a]*2,cv::Scalar::all(1u),-1);
                        cv::circle(vvDisplayPairs[0][a].first,cv::Point2i(avvImagePts[a][nCurrIdx][nPtIdx]),anMarkerSizes[a],cv::Scalar_<uchar>(lv::getBGRFromHSL(360*float(nPtIdx)/avvImagePts[a][nCurrIdx].size(),1.0f,0.5f)),-1);
                    }
                #endif //GEN_REGISTRATION_ANNOT
                }
            #if GEN_SEGMENTATION_ANNOT
                if(nCurrTile!=-1) {
                    const cv::Point2i vMousePos_FP2(int(vMousePos.x*vOrigSizes[nCurrTile].width*4),int(vMousePos.y*vOrigSizes[nCurrTile].height*4));
                    cv::circle(vvDisplayPairs[0][nCurrTile].first,vMousePos_FP2,(int)adSegmToolRadius[nCurrTile],cv::Scalar_<uchar>(0,0,255),1,cv::LINE_AA,2);
                }
            #endif //GEN_SEGMENTATION_ANNOT
                pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
                nKeyPressed = pDisplayHelper->waitKey();
                if(nKeyPressed==(int)'o') {
                    dSegmOpacity = std::min(dSegmOpacity+0.05,1.0);
                    lvLog_(1,"\topacity now at %f",dSegmOpacity);
                }
                else if(nKeyPressed==(int)'p') {
                    dSegmOpacity = std::max(dSegmOpacity-0.05,0.0);
                    lvLog_(1,"\topacity now at %f",dSegmOpacity);
                }
            }
    #if GEN_SEGMENTATION_ANNOT
            const bool bCurrFrameValid = (cv::countNonZero(aSegmMasks[0])!=0 || cv::countNonZero(aSegmMasks[1])!=0);
            if(bCurrFrameValid) {
                cv::imwrite(sRGBGTDir+lv::putf("%05d.png",(int)nCurrIdx),aSegmMasks[0]);
                cv::imwrite(sLWIRGTDir+lv::putf("%05d.png",(int)nCurrIdx),aSegmMasks[1]);
            }
    #elif GEN_REGISTRATION_ANNOT
            const bool bCurrFrameValid = (!avvRegistrPts[0][nCurrIdx].empty() || !avvRegistrPts[1][nCurrIdx].empty());
            if(bCurrFrameValid) {
                cv::FileStorage oGTFS_RGB(sRGBGTDir+lv::putf("%05d.yml",(int)nCurrIdx),cv::FileStorage::WRITE);
                cv::FileStorage oGTFS_LWIR(sLWIRGTDir+lv::putf("%05d.yml",(int)nCurrIdx),cv::FileStorage::WRITE);
                @@@@
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
        }
        {
            cv::FileStorage oGTMetadataFS(sRGBGTDir+"gtmetadata.yml",cv::FileStorage::WRITE);
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
