
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

////////////////////////////////
#define USE_OPENCV_CALIB        0
#define USE_FINE_TUNING_PASS    0
#define USE_INTRINSIC_GUESS     0
#define LOAD_CALIB_FROM_LAST    1
////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      1
////////////////////////////////
#define DATASETS_LITIV2018_LOAD_CALIB_DATA 1

#include "litiv/datasets.hpp"
#include "litiv/imgproc.hpp"
#include <opencv2/calib3d.hpp>

using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::Dataset_LITIV_stcharles2018,lv::NonParallel>;
void Analyze(lv::IDataHandlerPtr pBatch);

inline cv::Mat drawWorldPointsMap(const std::vector<cv::Point3f>& vPts, const cv::Size& oPatternSize, const cv::Size& oMapSize, size_t nSelectedPoint=SIZE_MAX) {
    lvAssert(oMapSize.area()>1 && oPatternSize.area()>1);
    lvAssert((int)vPts.size()==oPatternSize.area() || vPts.empty());
    cv::Mat oMap(oMapSize,CV_8UC3,cv::Scalar::all(66));
    if(!vPts.empty()) {
        lvAssert(nSelectedPoint==SIZE_MAX || nSelectedPoint<vPts.size());
        cv::Point3f vMin=vPts[0],vMax=vPts[0];
        for(size_t nIdx=1u; nIdx<vPts.size(); ++nIdx) {
            const double dDist = cv::norm(vPts[nIdx]);
            if(dDist<cv::norm(vMin))
                vMin = vPts[nIdx];
            if(dDist>cv::norm(vMax))
                vMax = vPts[nIdx];
        }
        vMax += vMin;
        for(size_t nPtIdx=0u; nPtIdx<vPts.size(); ++nPtIdx) {
            const cv::Point2i oImagePt((int)std::round((vPts[nPtIdx].x/vMax.x)*oMapSize.width),(int)std::round((vPts[nPtIdx].y/vMax.y)*oMapSize.height));
            if(nSelectedPoint==nPtIdx)
                cv::circle(oMap,oImagePt,18,cv::Scalar_<uchar>::all(1u),-1);
            cv::circle(oMap,oImagePt,10,cv::Scalar_<uchar>(lv::getBGRFromHSL(360*float(nPtIdx)/vPts.size(),1.0f,0.5f)),-1);
        }
    }
    return oMap;
}

int main(int, char**) {
    try {
        DatasetType::Ptr pDataset = DatasetType::create(
                DATASET_OUTPUT_PATH, // const std::string& sOutputDirName
                false, //bool bSaveOutput
                false, //bool bUseEvaluator
                false, //bool bLoadDepth
                false, //bool bUndistort
                false, //bool bHorizRectify
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
    DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
    lvAssert(oBatch.getInputPacketType()==lv::ImageArrayPacket && oBatch.getInputStreamCount()==2 && oBatch.getInputCount()>=1);
    if(DATASET_PRECACHING)
        oBatch.startPrecaching();
    const size_t nTotPacketCount = oBatch.getInputCount();
    size_t nCurrIdx = 0;
    std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // mat content becomes invalid on next getInput call
    lvAssert(!vInitInput.empty() && vInitInput.size()==oBatch.getInputStreamCount());
    std::array<cv::Size,2> aOrigSizes;
    for(size_t a=0u; a<2u; ++a) {
        aOrigSizes[a] = vInitInput[a].size();
        vInitInput[a] = vInitInput[a].clone();
    }
    const cv::Size oRGBSize=aOrigSizes[0],oLWIRSize=aOrigSizes[1];
    lvIgnore(oRGBSize);lvIgnore(oLWIRSize);

    const cv::Size oTargetSize(DATASETS_LITIV2018_RECTIFIED_SIZE);
    std::array<cv::Mat_<double>,2> aCamMats,aDistCoeffs;
    cv::Mat_<double> oRotMat,oTranslMat,oEssMat,oFundMat;

    lv::DisplayHelperPtr pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
    pDisplayHelper->setContinuousUpdates(true);
    const cv::Size oDisplayTileSize(1024,768);
    std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs = {{
        std::make_pair(vInitInput[0].clone(),oBatch.getInputStreamName(0)),
        std::make_pair(vInitInput[1].clone(),oBatch.getInputStreamName(1))
    }};

#if LOAD_CALIB_FROM_LAST
    {
        const std::string sCalibDataPath = oBatch.getCalibDataFolderPath();
        cv::FileStorage oCalibFile(sCalibDataPath+"calibdata.yml",cv::FileStorage::READ);
        lvAssert(oCalibFile.isOpened());
        std::string sVerStr;
        oCalibFile["ver"] >> sVerStr;
        lvAssert(!sVerStr.empty());
        lvCout << "Loading calib data from '" << sVerStr << "'...\n";
        oCalibFile["aCamMats0"] >> aCamMats[0];
        oCalibFile["aCamMats1"] >> aCamMats[1];
        oCalibFile["aDistCoeffs0"] >> aDistCoeffs[0];
        oCalibFile["aDistCoeffs1"] >> aDistCoeffs[1];
        oCalibFile["oRotMat"] >> oRotMat;
        oCalibFile["oTranslMat"] >> oTranslMat;
        oCalibFile["oEssMat"] >> oEssMat;
        oCalibFile["oFundMat"] >> oFundMat;
        double dStereoCalibErr;
        oCalibFile["dStereoCalibErr"] >> dStereoCalibErr;
        lvAssert(dStereoCalibErr>=0.0);
        lvCout << "\t(calib error was " << dStereoCalibErr << ")\n";
    }
#else //!LOAD_CALIB_FROM_LAST

    lvAssert(oBatch.getName().find("calib")!=std::string::npos);
    lvLog(1,"Parsing calibration metadata file storage...");
    std::array<std::vector<std::vector<cv::Point2f>>,2> avvImagePts{std::vector<std::vector<cv::Point2f>>(nTotPacketCount),std::vector<std::vector<cv::Point2f>>(nTotPacketCount)};
    std::vector<std::vector<cv::Point3f>> vvWorldPts(nTotPacketCount);
    cv::FileStorage oMetadataFS(oBatch.getDataPath()+"metadata.yml",cv::FileStorage::READ);
    lvAssert(oMetadataFS.isOpened());
    float fSquareSize_in,fSquareSizeMatlab_m;
    int nSquareCount_x,nSquareCount_y;
    cv::FileNode oCalibBoardData = oMetadataFS["calib_board"];
    oCalibBoardData["square_size_real_in"] >> fSquareSize_in;
    oCalibBoardData["square_size_matlab_m"] >> fSquareSizeMatlab_m;
    oCalibBoardData["square_count_x"] >> nSquareCount_x;
    oCalibBoardData["square_count_y"] >> nSquareCount_y;
    lvAssert(fSquareSize_in>0.0f && fSquareSizeMatlab_m>0.0f && nSquareCount_x>0 && nSquareCount_y>0);
    const float fSquareSize_m = 0.0254f*fSquareSize_in;
    const cv::Size oPatternSize(nSquareCount_x-1,nSquareCount_y-1); // -1 since we count inner corners

#if USE_OPENCV_CALIB
    // assume all calib board views have full pattern in sight
    while(nCurrIdx<nTotPacketCount) {
        std::cout << "\t\t @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << std::endl;
        const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
        lvDbgAssert(vCurrInput.size()==aOrigSizes[a]);
        bool bGotChessboards = true;
        for(size_t a=0u; a<vCurrInput.size(); ++a) {
            bool bDoubledSize = false;
            cv::Mat oOrigInput = vCurrInput[a].clone();
            if(oOrigInput.size().area()<640*480) {
                cv::resize(oOrigInput,oOrigInput,cv::Size(),2.0,2.0,cv::INTER_CUBIC);
                bDoubledSize = true;
            }
            cv::Mat oInput=oOrigInput.clone(), oInputDisplay=oOrigInput.clone(), oInputDisplay_gray=oOrigInput.clone();
            if(oInputDisplay.channels()!=1)
                cv::cvtColor(oInputDisplay,oInputDisplay_gray,cv::COLOR_BGR2GRAY);
            else
                cv::cvtColor(oInputDisplay_gray,oInputDisplay,cv::COLOR_GRAY2BGR);
            const bool bGotCurrChessboard = cv::findChessboardCorners(oInput,oPatternSize,avvImagePts[a][nCurrIdx],(cv::CALIB_CB_ADAPTIVE_THRESH+cv::CALIB_CB_NORMALIZE_IMAGE));
            if(bGotCurrChessboard)
                lvAssert_(cv::find4QuadCornerSubpix(oInputDisplay_gray,avvImagePts[a][nCurrIdx],cv::Size(15,15)),"subpix optimization failed");
            bGotChessboards &= bGotCurrChessboard;
            cv::drawChessboardCorners(oInputDisplay,oPatternSize,avvImagePts[a][nCurrIdx],bGotCurrChessboard);
            if(oInputDisplay.size().area()>1024*768)
                cv::resize(oInputDisplay,oInputDisplay,cv::Size(),0.5,0.5);
            if(oInputDisplay.size().area()<640*480)
                cv::resize(oInputDisplay,oInputDisplay,cv::Size(),2.0,2.0);
            cv::imshow(std::string("vCurrInput_")+std::to_string(a),oInputDisplay);
            if(bDoubledSize) {
                for(size_t nPtIdx=0u; nPtIdx<avvImagePts[a][nCurrIdx].size(); ++nPtIdx) {
                    avvImagePts[a][nCurrIdx][nPtIdx].x /= 2.0f;
                    avvImagePts[a][nCurrIdx][nPtIdx].y /= 2.0f;
                }
            }
        }
        if(bGotChessboards) {
            vvWorldPts[nCurrIdx].resize(size_t(nSquareCount_y*nSquareCount_x));
            // indices start at 1, we are interested in inner corners only
            for(int nSquareRowIdx=1; nSquareRowIdx<nSquareCount_y; ++nSquareRowIdx)
                for(int nSquareColIdx=1; nSquareColIdx<nSquareCount_x; ++nSquareColIdx)
                    vvWorldPts[nCurrIdx][nSquareRowIdx*nSquareCount_x+nSquareColIdx] = cv::Point3f(nSquareColIdx*fSquareSize_m,nSquareRowIdx*fSquareSize_m,0.0f);
        }
        int nKeyPressed = cv::waitKey(0);
        if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
            break;
        else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0u)
            --nCurrIdx;
        else if(nKeyPressed!=8/*backspace*/)
            ++nCurrIdx;
    }

#else //!USE_OPENCV_CALIB

    // opencv chessboard detection fails in lwir (contrast issues); use exports from matlab calib toolbox & older yml files
    lvLog(1,"Image point export parsing initialized...");
    {
        cv::FileNode oWorldPointsNode,oRGBPointsNode,oLWIRPointsNode;
        cv::FileStorage oPointsFS(oBatch.getDataPath()+"ptsdata.yml",cv::FileStorage::READ);
        std::vector<std::string> vsRGBImagePtsPaths = lv::getFilesFromDir(oBatch.getDataPath()+"rgb_subset");
        std::vector<std::string> vsLWIRImagePtsPaths = lv::getFilesFromDir(oBatch.getDataPath()+"lwir_subset");
        const std::string sImagePtsFileNamePrefix = "imagepts", sImagePtsFileNameSuffix = ".txt";
        lv::filterFilePaths(vsRGBImagePtsPaths,{},{sImagePtsFileNameSuffix});
        lv::filterFilePaths(vsRGBImagePtsPaths,{},{sImagePtsFileNamePrefix});
        lv::filterFilePaths(vsLWIRImagePtsPaths,{},{sImagePtsFileNameSuffix});
        lv::filterFilePaths(vsLWIRImagePtsPaths,{},{sImagePtsFileNamePrefix});
        const auto lIndexExtractor = [&](const std::string& sFilePath) {
            const size_t nLastInputSlashPos = sFilePath.find_last_of("/\\");
            const std::string sInputFileNameExt = nLastInputSlashPos==std::string::npos?sFilePath:sFilePath.substr(nLastInputSlashPos+1);
            const size_t nLastInputDotPos = sInputFileNameExt.find_last_of('.');
            const std::string sInputFileName = (nLastInputDotPos==std::string::npos)?sInputFileNameExt:sInputFileNameExt.substr(0,nLastInputDotPos);
            const std::string sInputFileIdxStr = sInputFileName.substr(sImagePtsFileNamePrefix.size());
            return (size_t)std::stoi(sInputFileIdxStr);
        };
        std::sort(vsRGBImagePtsPaths.begin(),vsRGBImagePtsPaths.end(),[&](const std::string& a, const std::string& b){
            return lIndexExtractor(a)<lIndexExtractor(b);
        });
        std::sort(vsLWIRImagePtsPaths.begin(),vsLWIRImagePtsPaths.end(),[&](const std::string& a, const std::string& b){
            return lIndexExtractor(a)<lIndexExtractor(b);
        });
        if(oPointsFS.isOpened()) {
            lvLog(1,"Found points data file storage, bypassing matlab export...");
            int nOldPacketCount;
            oPointsFS["npackets"] >> nOldPacketCount;
            lvAssert(nOldPacketCount==(int)nTotPacketCount);
            oWorldPointsNode = oPointsFS["world_pts"];
            oRGBPointsNode = oPointsFS["rgb_pts"];
            oLWIRPointsNode = oPointsFS["lwir_pts"];
        }
        else {
            lvAssert(vsRGBImagePtsPaths.size()<=nTotPacketCount);
            lvAssert(vsLWIRImagePtsPaths.size()<=nTotPacketCount);
        }
        nCurrIdx = 0u;
        size_t nRGBDataIdx=0u,nLWIRDataIdx=0;
        while(nCurrIdx<nTotPacketCount) {
            const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
            cv::Mat oRGBFrame=vCurrInput[0].clone(),oLWIRFrame=vCurrInput[1].clone();
            lvAssert(!oRGBFrame.empty() && !oLWIRFrame.empty());
            lvAssert(oRGBFrame.size()==oRGBSize && oLWIRFrame.size()==oLWIRSize);
            const std::vector<bool>& vInputValid = oBatch.isCalibInputValid(nCurrIdx);
            if(!oPointsFS.isOpened()) {
                float fImagePosX,fImagePosY;
                size_t nRGBPointCount=0u,nLWIRPointCount=0u,nWorldPointCount=0u;
                std::ifstream oRGBData,oLWIRData;
                if(vInputValid[0]) {
                    oRGBData.open(vsRGBImagePtsPaths[nRGBDataIdx++]);
                    lvAssert(oRGBData.is_open());
                    avvImagePts[0][nCurrIdx].clear();
                    while(oRGBData>>fImagePosX && oRGBData>>fImagePosY) {
                        avvImagePts[0][nCurrIdx].emplace_back(fImagePosX,fImagePosY);
                        ++nRGBPointCount;
                    }
                #if DATASETS_LITIV2018_FLIP_RGB
                    for(size_t nPtIdx=0u; nPtIdx<avvImagePts[0][nCurrIdx].size(); ++nPtIdx)
                        avvImagePts[0][nCurrIdx][nPtIdx] = cv::Point2f((oRGBSize.width-1)-avvImagePts[0][nCurrIdx][nPtIdx].x,avvImagePts[0][nCurrIdx][nPtIdx].y);
                #endif //DATASETS_LITIV2018_FLIP_RGB
                }
                if(vInputValid[1]) {
                    oLWIRData.open(vsLWIRImagePtsPaths[nLWIRDataIdx++]);
                    lvAssert(oLWIRData.is_open());
                    avvImagePts[1][nCurrIdx].clear();
                    while(oLWIRData>>fImagePosX && oLWIRData>>fImagePosY) {
                        avvImagePts[1][nCurrIdx].emplace_back(fImagePosX,fImagePosY);
                        ++nLWIRPointCount;
                    }
                }
                vvWorldPts[nCurrIdx].resize(size_t(oPatternSize.area()));
                // indices are offset as we are interested in inner corners only
                for(int nSquareColIdx=oPatternSize.width-1; nSquareColIdx>=0; --nSquareColIdx)
                    for(int nSquareRowIdx=0; nSquareRowIdx<oPatternSize.height; ++nSquareRowIdx)
                        vvWorldPts[nCurrIdx][nWorldPointCount++] = cv::Point3f((nSquareColIdx+1)*fSquareSize_m,(nSquareRowIdx+1)*fSquareSize_m,0.0f);
                lvAssert(!vInputValid[0] || nRGBPointCount==nWorldPointCount);
                lvAssert(!vInputValid[1] || nLWIRPointCount==nWorldPointCount);
            }
            else {
                const std::string sFrameIdxStr = std::string("f")+std::to_string(nCurrIdx);
                oWorldPointsNode[sFrameIdxStr] >> vvWorldPts[nCurrIdx];
                oRGBPointsNode[sFrameIdxStr] >> avvImagePts[0][nCurrIdx];
                oLWIRPointsNode[sFrameIdxStr] >> avvImagePts[1][nCurrIdx];
                lvAssert(avvImagePts[0][nCurrIdx].empty() || vvWorldPts[nCurrIdx].size()==avvImagePts[0][nCurrIdx].size());
                lvAssert(avvImagePts[1][nCurrIdx].empty() || vvWorldPts[nCurrIdx].size()==avvImagePts[1][nCurrIdx].size());
            }
            //cv::circle(oRGBFrame,avvImagePts[0][nCurrIdx].back(),2,cv::Scalar_<uchar>(0,0,255),-1);
            //cv::circle(oLWIRFrame,avvImagePts[1][nCurrIdx].back(),2,cv::Scalar_<uchar>(0,0,255),-1);
#if USE_CORNER_SUBPIX_OPTIM
            cv::Mat oRGBFrame_gray; cv::cvtColor(oRGBFrame,oRGBFrame_gray,cv::COLOR_BGR2GRAY);
            cv::find4QuadCornerSubpix(oRGBFrame_gray,avvImagePts[0][nCurrIdx],cv::Size(5,5));
            cv::find4QuadCornerSubpix(oLWIRFrame,avvImagePts[1][nCurrIdx],cv::Size(3,3));
#endif //USE_CORNER_SUBPIX_OPTIM
            //lvPrint(vvWorldPts[nCurrIdx]);
            //lvPrint(avvImagePts[0][nCurrIdx]);
            //cv::imshow("rgb",oRGBFrame);
            //cv::imshow("lwir",oLWIRFrame);
            //cv::waitKey(0);
            ++nCurrIdx;
        }
    }
#if USE_FINE_TUNING_PASS
    bool bIsZoomedIn = false;
    size_t nSelectedPoint = SIZE_MAX;
    const cv::Size oZoomedNeigbSize(24,16);
    const std::array<int,2> anMarkerSizes{4,1};
    std::array<cv::Mat,2> aCurrInputs,aZoomedPatches;
    pDisplayHelper->setMouseCallback([&](const lv::DisplayHelper::CallbackData& oData) {
        if(oData.nEvent==cv::EVENT_LBUTTONUP || oData.nEvent==cv::EVENT_RBUTTONUP) {
            const cv::Point2f vClickPos(float(oData.oInternalPosition.x)/oData.oTileSize.width,float(oData.oInternalPosition.y)/oData.oTileSize.height);
            if(vClickPos.x>=0.0f && vClickPos.y>=0.0f && vClickPos.x<1.0f && vClickPos.y<1.0f) {
                const int nCurrTile = oData.oPosition.x/oData.oTileSize.width;
                if(oData.nEvent==cv::EVENT_LBUTTONUP) {
                    if(bIsZoomedIn) {
                        const cv::Point2f vOffset = (vClickPos-cv::Point2f(0.5f,0.5f));
                        avvImagePts[nCurrTile][nCurrIdx][nSelectedPoint] += cv::Point2f(vOffset.x*oZoomedNeigbSize.width,vOffset.y*oZoomedNeigbSize.height);
                    }
                    else if(!bIsZoomedIn) {
                        size_t nClosestIdx = SIZE_MAX;
                        float fBestDistance = 9999.f;
                        const cv::Point2f vRealClickPos(vClickPos.x*aOrigSizes[nCurrTile].width,vClickPos.y*aOrigSizes[nCurrTile].height);
                        for(size_t nIdx=0u; nIdx<avvImagePts[nCurrTile][nCurrIdx].size(); ++nIdx) {
                            const float fDistance = (float)cv::norm(vRealClickPos-avvImagePts[nCurrTile][nCurrIdx][nIdx]);
                            if(fDistance<10.0f && fDistance<fBestDistance) {
                                fBestDistance = fDistance;
                                nClosestIdx = nIdx;
                            }
                        }
                        if(nClosestIdx<avvImagePts[nCurrTile][nCurrIdx].size()) {
                            bIsZoomedIn = true;
                            nSelectedPoint = nClosestIdx;
                        }
                    }
                    if(bIsZoomedIn) {
                        for(size_t a=0u; a<2u; ++a) {
                            if(avvImagePts[a][nCurrIdx].empty())
                                continue;
                            cv::getRectSubPix(aCurrInputs[a],oZoomedNeigbSize,avvImagePts[a][nCurrIdx][nSelectedPoint],aZoomedPatches[a]);
                            cv::resize(aZoomedPatches[a],aZoomedPatches[a],oDisplayTileSize,0,0,cv::INTER_CUBIC);
                            if(aZoomedPatches[a].channels()==1)
                                cv::cvtColor(aZoomedPatches[a],aZoomedPatches[a],cv::COLOR_GRAY2BGR);
                        }
                    }
                }
                else if(oData.nEvent==cv::EVENT_RBUTTONUP) {
                    if(bIsZoomedIn)
                        bIsZoomedIn = false;
                }
            }
        }
        //pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
    });
    lvLog(1,"Image point edit mode initialized...");
    nCurrIdx = 0u;
    while(nCurrIdx<nTotPacketCount) {
        const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
        cv::Mat oRGBFrame=vCurrInput[0].clone(),oLWIRFrame=vCurrInput[1].clone();
        lvAssert(!oRGBFrame.empty() && !oLWIRFrame.empty());
        lvAssert(oRGBFrame.size()==oRGBSize && oLWIRFrame.size()==oLWIRSize);
        const std::vector<bool>& vInputValid = oBatch.isCalibInputValid(nCurrIdx);
        lv::copyVectorToArray(vCurrInput,aCurrInputs);
        lvLog_(1,"\t calib @ #%d",int(nCurrIdx));
        int nKeyPressed = -1;
        while(nKeyPressed!=(int)'q' && nKeyPressed!=27/*escape*/ && nKeyPressed!=8/*backspace*/ && (nKeyPressed%256)!=10/*lf*/ && (nKeyPressed%256)!=13/*enter*/) {
            for(size_t a=0u; a<2u; ++a) {
                if(!vInputValid[a]) {
                    vvDisplayPairs[0][a].first = cv::Scalar::all(0);
                    continue;
                }
                if(bIsZoomedIn) {
                    aZoomedPatches[a].copyTo(vvDisplayPairs[0][a].first);
                    const cv::Size oCurrSize(vvDisplayPairs[0][a].first.size());
                    cv::rectangle(vvDisplayPairs[0][a].first,cv::Point(0,oCurrSize.height/2),cv::Point(oCurrSize.width-1,oCurrSize.height/2),cv::Scalar_<uchar>::all(0),-1);
                    cv::rectangle(vvDisplayPairs[0][a].first,cv::Point(oCurrSize.width/2,0),cv::Point(oCurrSize.width/2,oCurrSize.height-1),cv::Scalar_<uchar>::all(0),-1);
                    cv::rectangle(vvDisplayPairs[0][a].first,cv::Rect(oCurrSize.width/2-oCurrSize.height/4,oCurrSize.height/4,oCurrSize.height/2,oCurrSize.height/2),cv::Scalar_<uchar>::all(0),1);
                    cv::circle(vvDisplayPairs[0][a].first,cv::Point(oCurrSize.width/2,oCurrSize.height/2),2,cv::Scalar_<uchar>::all(0),-1);
                    cv::circle(vvDisplayPairs[0][a].first,cv::Point(oCurrSize.width/2,oCurrSize.height/2),1,cv::Scalar_<uchar>(0,0,255),-1);
                }
                else {
                    aCurrInputs[a].copyTo(vvDisplayPairs[0][a].first);
                    if(vvDisplayPairs[0][a].first.channels()==1)
                        cv::cvtColor(vvDisplayPairs[0][a].first,vvDisplayPairs[0][a].first,cv::COLOR_GRAY2BGR);
                    for(size_t nPtIdx=0u; nPtIdx<avvImagePts[a][nCurrIdx].size(); ++nPtIdx) {
                        if(nSelectedPoint==nPtIdx)
                            cv::circle(vvDisplayPairs[0][a].first,cv::Point2i(avvImagePts[a][nCurrIdx][nPtIdx]),anMarkerSizes[a]*2,cv::Scalar::all(1u),-1);
                        cv::circle(vvDisplayPairs[0][a].first,cv::Point2i(avvImagePts[a][nCurrIdx][nPtIdx]),anMarkerSizes[a],cv::Scalar_<uchar>(lv::getBGRFromHSL(360*float(nPtIdx)/avvImagePts[a][nCurrIdx].size(),1.0f,0.5f)),-1);
                    }
                }
            }
            pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
            cv::imshow("worldmap",drawWorldPointsMap(vvWorldPts[nCurrIdx],oPatternSize,cv::Size(640,480),nSelectedPoint));
            nKeyPressed = pDisplayHelper->waitKey();
        }
        if(bIsZoomedIn) {
            nSelectedPoint = SIZE_MAX;
            bIsZoomedIn = false;
        }
        if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
            break;
        else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0u)
            --nCurrIdx;
        else if(((nKeyPressed%256)==10/*lf*/ || (nKeyPressed%256)==13/*enter*/) && nCurrIdx<(nTotPacketCount-1u))
            ++nCurrIdx;
    }
    cv::destroyWindow(oBatch.getName());
    cv::destroyWindow("worldmap");
    cv::waitKey(100);
    {
        lvLog(1,"Archiving points data to file storage...");
        cv::FileStorage oPointsFS(oBatch.getDataPath()+"ptsdata.yml",cv::FileStorage::WRITE);
        lvAssert(oPointsFS.isOpened());
        oPointsFS << "htag" << lv::getVersionStamp();
        oPointsFS << "date" << lv::getTimeStamp();
        oPointsFS << "npackets" << (int)nTotPacketCount;
        oPointsFS << "patternsize" << oPatternSize;
        oPointsFS << "world_pts" << "{";
        for(size_t nIdx=0; nIdx<nTotPacketCount; ++nIdx)
            oPointsFS << (std::string("f")+std::to_string(nIdx)) << vvWorldPts[nIdx];
        oPointsFS << "}";
        oPointsFS << "rgb_pts" << "{";
        for(size_t nIdx=0; nIdx<nTotPacketCount; ++nIdx)
            oPointsFS << (std::string("f")+std::to_string(nIdx)) << avvImagePts[0][nIdx];
        oPointsFS << "}";
        oPointsFS << "lwir_pts" << "{";
        for(size_t nIdx=0; nIdx<nTotPacketCount; ++nIdx)
            oPointsFS << (std::string("f")+std::to_string(nIdx)) << avvImagePts[1][nIdx];
        oPointsFS << "}";
    }
#endif //USE_FINE_TUNING_PASS
#endif //!USE_OPENCV_CALIB
    lvDbgExceptionWatch;
    for(size_t a=0u; a<2u; ++a) {
        const cv::Size oOrigSize = aOrigSizes[a];
        if(oOrigSize!=oTargetSize) {
            for(size_t nFrameIdx=0u; nFrameIdx<avvImagePts[a].size(); ++nFrameIdx) {
                //const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nFrameIdx);
                //cv::Mat display = vCurrInput[a].clone();
                //cv::resize(display,display,oTargetSize);
                //if(display.channels()==1)
                //    cv::cvtColor(display,display,cv::COLOR_GRAY2BGR);
                for(size_t nPtIdx=0u; nPtIdx<avvImagePts[a][nFrameIdx].size(); ++nPtIdx) {
                    cv::Point2f& oPt = avvImagePts[a][nFrameIdx][nPtIdx];
                    oPt.x *= float(oTargetSize.width)/oOrigSize.width;
                    oPt.y *= float(oTargetSize.height)/oOrigSize.height;
                    lvAssert(oPt.x<oTargetSize.width);
                    lvAssert(oPt.y<oTargetSize.height);
                    //cv::circle(display,oPt,3,cv::Scalar(0,0,255),-1);
                }
                //cv::imshow("display",display);
                //cv::waitKey(0);
            }
        }
    }
    std::vector<std::vector<cv::Mat>> vvInputFrames(nTotPacketCount,std::vector<cv::Mat>(2u));
    for(size_t nFrameIdx=0u; nFrameIdx<nTotPacketCount; ++nFrameIdx) {
        const std::vector<cv::Mat>& vCurrInputs = oBatch.getInputArray(nFrameIdx);
        for(size_t a=0u; a<vCurrInputs.size(); ++a) {
            vvInputFrames[nFrameIdx][a] = vCurrInputs[a].clone();
            if(vvInputFrames[nFrameIdx][a].channels()==1)
                cv::cvtColor(vvInputFrames[nFrameIdx][a],vvInputFrames[nFrameIdx][a],cv::COLOR_GRAY2BGR);
        }
    }
    for(size_t nFrameIdx=0u; nFrameIdx<vvWorldPts.size(); ++nFrameIdx) {
        if(vvWorldPts[nFrameIdx].empty()) {
            vvInputFrames.erase(vvInputFrames.begin()+nFrameIdx);
            avvImagePts[0].erase(avvImagePts[0].begin()+nFrameIdx);
            avvImagePts[1].erase(avvImagePts[1].begin()+nFrameIdx);
            vvWorldPts.erase(vvWorldPts.begin()+nFrameIdx);
            nFrameIdx = 0u;
        }
    }
    lvAssert(vvWorldPts.size()==avvImagePts[0].size() && vvWorldPts.size()==avvImagePts[1].size() && vvWorldPts.size()==vvInputFrames.size());
    lvAssert(!vvWorldPts.empty());

#if USE_INTRINSIC_GUESS
    @@@cleanup, retest (with target scale)
    aCamMats[0] = cv::initCameraMatrix2D(vvWorldPts,avvImagePts[0],oOrigImgSize,/*1.0*/1.27);
    //aCamMats[0] = (cv::Mat_<double>(3,3) << 531.15,0,320,  0,416.35,240,  0,0,1);
    aCamMats[1] = cv::initCameraMatrix2D(vvWorldPts,avvImagePts[1],oOrigImgSize,/*1.0*/1.27);
    cv::calibrateCamera(vvWorldPts,avvImagePts[0],oOrigImgSize,aCamMats[0],aDistCoeffs[0],cv::noArray(),cv::noArray(),
                        //0,
                        cv::CALIB_USE_INTRINSIC_GUESS,
                        //cv::CALIB_FIX_ASPECT_RATIO+cv::CALIB_FIX_FOCAL_LENGTH+cv::CALIB_FIX_PRINCIPAL_POINT,
                        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,250,DBL_EPSILON));
    cv::calibrateCamera(vvWorldPts,avvImagePts[1],oOrigImgSize,aCamMats[1],aDistCoeffs[1],cv::noArray(),cv::noArray(),
                        //0,
                        cv::CALIB_USE_INTRINSIC_GUESS,
                        cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,250,DBL_EPSILON));
    for(size_t n=0; n<avCalibInputs[0].size(); ++n) {
        std::array<cv::Mat,2> aUndistortInput;
        std::array<std::vector<cv::Point2f>,2> avUndistortImagePts;
        for(size_t a=0; a<2; ++a) {
            cv::undistort(avCalibInputs[a][n],aUndistortInput[a],aCamMats[a],aDistCoeffs[a]);
            cv::undistortPoints(avvImagePts[a][n],avUndistortImagePts[a],aCamMats[a],aDistCoeffs[a],cv::noArray(),aCamMats[a]);
            for(size_t p=0; p<avUndistortImagePts[a].size(); ++p)
                cv::circle(aUndistortInput[a],avUndistortImagePts[a][p],2,cv::Scalar_<uchar>(0,0,255),-1);
            cv::imshow(std::string("aUndistortInput")+std::to_string(a),aUndistortInput[a]);
        }
        cv::waitKey(0);
    }
#else //!USE_INTRINSIC_GUESS
    for(size_t a=0u; a<2u; ++a) {
        lvDbgExceptionWatch;
        std::vector<std::vector<cv::Point2f>> vvImagePts = avvImagePts[a];
        std::vector<std::vector<cv::Point3f>> vvWorldPtsCopy = vvWorldPts;
        for(size_t nFrameIdx=0u; nFrameIdx<vvWorldPtsCopy.size(); ++nFrameIdx) {
            if(vvImagePts[nFrameIdx].empty()) {
                vvWorldPtsCopy.erase(vvWorldPtsCopy.begin()+nFrameIdx);
                vvImagePts.erase(vvImagePts.begin()+nFrameIdx);
                nFrameIdx = 0u;
            }
        }
        lvAssert(vvWorldPtsCopy.size()==vvImagePts.size());
        lvAssert(!vvWorldPtsCopy.empty());

        lvLog_(1,"Running camera calibration for head #%d with %d images...",(int)a+1,(int)vvImagePts.size());
        //cv::Mat_<double> oPerViewErrors;
        aCamMats[a] = cv::initCameraMatrix2D(vvWorldPtsCopy,vvImagePts,oTargetSize);
        aDistCoeffs[a].create(1,5);
        aDistCoeffs[a] = 0.0;
        const double dReprojErr =
            cv::calibrateCamera(vvWorldPtsCopy,vvImagePts,oTargetSize,aCamMats[a],aDistCoeffs[a],
                                cv::noArray(),cv::noArray(),/*cv::noArray(),cv::noArray(),oPerViewErrors,*/
                                cv::CALIB_USE_INTRINSIC_GUESS+cv::CALIB_ZERO_TANGENT_DIST+((a==0u)?(cv::CALIB_FIX_PRINCIPAL_POINT):(0)),
                                cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,1000,1e-7));
        //lvPrint(oPerViewErrors);
        lvPrint(aCamMats[a]);
        lvPrint(aDistCoeffs[a]);
        lvLog_(1,"\n\tcalib err for cam[%d] : %f\n",int(a),dReprojErr);
    }
#endif //!USE_INTRINSIC_GUESS

    for(size_t nFrameIdx=0u; nFrameIdx<vvWorldPts.size(); ++nFrameIdx) {
        if(avvImagePts[0][nFrameIdx].empty() || avvImagePts[1][nFrameIdx].empty()) {
            vvInputFrames.erase(vvInputFrames.begin()+nFrameIdx);
            avvImagePts[0].erase(avvImagePts[0].begin()+nFrameIdx);
            avvImagePts[1].erase(avvImagePts[1].begin()+nFrameIdx);
            vvWorldPts.erase(vvWorldPts.begin()+nFrameIdx);
            nFrameIdx = 0u;
        }
    }
    lvAssert(vvWorldPts.size()==avvImagePts[0].size() && vvWorldPts.size()==avvImagePts[1].size() && vvWorldPts.size()==vvInputFrames.size());
    lvAssert(!vvWorldPts.empty());
    lvDbgExceptionWatch;
    lvLog_(1,"Running stereo calibration with %d images...",(int)vvWorldPts.size());
    const double dStereoCalibErr = cv::stereoCalibrate(vvWorldPts,avvImagePts[0],avvImagePts[1],
                                                       aCamMats[0],aDistCoeffs[0],aCamMats[1],aDistCoeffs[1],
                                                       oTargetSize,oRotMat,oTranslMat,oEssMat,oFundMat,
                                                       //cv::CALIB_USE_INTRINSIC_GUESS+cv::CALIB_ZERO_TANGENT_DIST,
                                                       cv::CALIB_FIX_INTRINSIC,
                                                       cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,1000,1e-7));
    lvLog_(1,"\tmean stereo calib err : %f",dStereoCalibErr);
    {
        lvLog(1,"\nSaving stereo calibration...\n");
        const std::string sCalibDataPath = oBatch.getCalibDataFolderPath();
        cv::FileStorage oCalibFile(sCalibDataPath+"calibdata.yml",cv::FileStorage::WRITE);
        lvAssert(oCalibFile.isOpened());
        oCalibFile << "ver" << (lv::getVersionStamp()+" "+lv::getTimeStamp());
        oCalibFile << "aCamMats0" << aCamMats[0];
        oCalibFile << "aCamMats1" << aCamMats[1];
        oCalibFile << "aDistCoeffs0" << aDistCoeffs[0];
        oCalibFile << "aDistCoeffs1" << aDistCoeffs[1];
        oCalibFile << "oRotMat" << oRotMat;
        oCalibFile << "oTranslMat" << oTranslMat;
        oCalibFile << "oEssMat" << oEssMat;
        oCalibFile << "oFundMat" << oFundMat;
        oCalibFile << "oTargetSize" << oTargetSize;
        oCalibFile << "dStereoCalibErr" << dStereoCalibErr;
    }


    /*for(size_t n=0; n<avCalibInputs[0].size(); ++n) {
        std::array<cv::Mat,2> aUndistortInput;
        std::array<std::vector<cv::Point2f>,2> avUndistortImagePts;
        for(size_t a=0; a<2; ++a) {
            cv::undistort(avCalibInputs[a][n],aUndistortInput[a],aCamMats[a],aDistCoeffs[a]);
            cv::undistortPoints(avvImagePts[a][n],avUndistortImagePts[a],aCamMats[a],aDistCoeffs[a],cv::noArray(),aCamMats[a]);
            for(size_t p=0; p<avUndistortImagePts[a].size(); ++p)
                cv::circle(aUndistortInput[a],avUndistortImagePts[a][p],2,cv::Scalar_<uchar>(0,0,255),-1);
            cv::imshow(std::string("aUndistortInput")+std::to_string(a),aUndistortInput[a]);
        }
        cv::waitKey(0);
    }*/

    /*std::array<cv::Mat,2> aUncalibRectHoms;
    std::array<std::vector<cv::Point2f>,2> avUndistortImagePts;
    for(size_t a=0; a<2; ++a)
        cv::undistortPoints(avImagePts[a],avUndistortImagePts[a],aCamMats[a],aDistCoeffs[a]);
    const cv::Mat oRoughFundMat = cv::findFundamentalMat(avUndistortImagePts[0],avUndistortImagePts[1],cv::FM_8POINT);
    cv::stereoRectifyUncalibrated(avUndistortImagePts[0],avUndistortImagePts[1],oRoughFundMat,oOrigImgSize,aUncalibRectHoms[0],aUncalibRectHoms[1],-1);
    std::array<cv::Mat,2> aNewUndistortInput;
    std::array<cv::Mat,2> aWarpedInput;
    for(size_t a=0; a<2; ++a) {
        cv::undistort(vInitInput[a],aNewUndistortInput[a],aCamMats[a],aDistCoeffs[a]);
        cv::imshow(std::string("aNewUndistortInput")+std::to_string(a),aNewUndistortInput[a]);
        cv::warpPerspective(aNewUndistortInput[a],aWarpedInput[a],aUncalibRectHoms[a],oOrigImgSize,cv::INTER_LINEAR);
        cv::imshow(std::string("aWarpedInput")+std::to_string(a),aWarpedInput[a]);
    }
    cv::waitKey(0);*/

#endif //!LOAD_CALIB_FROM_LAST

    lvLog(1,"Running live stereo rectification...");
    nCurrIdx = 0;
    double dRectifAlpha = -1;
    bool bRectify = true;
#if LOAD_CALIB_FROM_LAST
    const size_t nTotStereoFrames = nTotPacketCount;
#else //!LOAD_CALIB_FROM_LAST
    const size_t nTotStereoFrames = vvInputFrames.size();
#endif //!LOAD_CALIB_FROM_LAST
    while(nCurrIdx<nTotStereoFrames) {
    #if LOAD_CALIB_FROM_LAST
        const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
    #else //!LOAD_CALIB_FROM_LAST
        const std::vector<cv::Mat>& vCurrInput = vvInputFrames[nCurrIdx];
    #endif //!LOAD_CALIB_FROM_LAST
        lvAssert(vCurrInput.size()==aOrigSizes.size());
        std::array<cv::Mat,2> aRectifRotMats,aRectifProjMats;
        cv::Mat oDispToDepthMap;
        cv::stereoRectify(aCamMats[0],aDistCoeffs[0],aCamMats[1],aDistCoeffs[1],
                          oTargetSize,oRotMat,oTranslMat,
                          aRectifRotMats[0],aRectifRotMats[1],
                          aRectifProjMats[0],aRectifProjMats[1],
                          oDispToDepthMap,
                          0,//cv::CALIB_ZERO_DISPARITY,
                          dRectifAlpha,oTargetSize);

        std::array<std::array<cv::Mat,2>,2> aaRectifMaps;
        cv::initUndistortRectifyMap(aCamMats[0],aDistCoeffs[0],aRectifRotMats[0],aRectifProjMats[0],
                                    oTargetSize,CV_16SC2,aaRectifMaps[0][0],aaRectifMaps[0][1]);
        cv::initUndistortRectifyMap(aCamMats[1],aDistCoeffs[1],aRectifRotMats[1],aRectifProjMats[1],
                                    oTargetSize,CV_16SC2,aaRectifMaps[1][0],aaRectifMaps[1][1]);
        std::array<cv::Mat,2> aCurrRectifInput;
        for(size_t a=0; a<2; ++a) {
            cv::Mat oCurrInput = vCurrInput[a].clone();
            lvAssert(oCurrInput.size()==aOrigSizes[a]);
            if(oCurrInput.size()!=oTargetSize)
                cv::resize(oCurrInput,oCurrInput,oTargetSize);
        #if !LOAD_CALIB_FROM_LAST
            for(size_t nPtIdx=0u; nPtIdx<avvImagePts[a][nCurrIdx].size(); ++nPtIdx) {
                const cv::Point2f& oPt = avvImagePts[a][nCurrIdx][nPtIdx];
                cv::circle(oCurrInput,oPt,5,cv::Scalar_<uchar>(lv::getBGRFromHSL(360*float(nPtIdx)/avvImagePts[a][nCurrIdx].size(),1.0f,0.5f)),-1);
            }
        #endif //!LOAD_CALIB_FROM_LAST
            if(bRectify) {
                cv::remap(oCurrInput,aCurrRectifInput[a],aaRectifMaps[a][0],aaRectifMaps[a][1],cv::INTER_LINEAR);
                aCurrRectifInput[a].copyTo(vvDisplayPairs[0][a].first);
            }
            else
                oCurrInput.copyTo(vvDisplayPairs[0][a].first);
            pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
        }
        const int nKeyPressed = pDisplayHelper->waitKey();
        const uchar cKeyPressed = uchar(nKeyPressed);
        if(cKeyPressed=='q' || nKeyPressed==27/*escape*/)
            break;
        else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0u) {
            --nCurrIdx;
            std::cout << "\t\t calib @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotStereoFrames)) << nCurrIdx+1 << "/" << nTotStereoFrames << std::endl;
        }
        else if(((nKeyPressed%256)==10/*lf*/ || (nKeyPressed%256)==13/*enter*/) && nCurrIdx<(nTotStereoFrames-1u)) {
            ++nCurrIdx;
            std::cout << "\t\t calib @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotStereoFrames)) << nCurrIdx+1 << "/" << nTotStereoFrames << std::endl;
        }
        else if(cKeyPressed=='+' || cKeyPressed=='=' || cKeyPressed=='p') {
            dRectifAlpha = std::max(std::min(dRectifAlpha+0.05,1.0),0.0);
            lvLog_(1,"dRectifAlpha = %f",dRectifAlpha);
        }
        else if(cKeyPressed=='-' || cKeyPressed=='l') {
            dRectifAlpha = std::min(std::max(dRectifAlpha-0.05,0.0),1.0);
            lvLog_(1,"dRectifAlpha = %f",dRectifAlpha);
        }
        else if(cKeyPressed=='0') {
            dRectifAlpha = -1.0;
            lvLog_(1,"dRectifAlpha = %f",dRectifAlpha);
        }
        else if(cKeyPressed=='r') {
            bRectify = !bRectify;
        }
    }
}
