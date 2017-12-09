
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
#include <opencv2/calib3d.hpp>

////////////////////////////////
#define USE_UNCALIB_FMAT_ESTIM  0
#if USE_UNCALIB_FMAT_ESTIM
#define LOAD_POINTS_FROM_LAST   1
#define USE_FMAT_RANSAC_ESTIM   0
#else //!USE_UNCALIB_FMAT_ESTIM
#define USE_INTRINSIC_GUESS     0
#define LOAD_CALIB_FROM_LAST    0
#endif //!USE_UNCALIB_FMAT_ESTIM

////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      1

using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::Dataset_VAP_trimod2016,lv::NonParallel>;
void Analyze(lv::IDataHandlerPtr pBatch);

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
    const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
    std::cout << "\t\t" << sCurrBatchName << " @ init" << std::endl;
    const size_t nTotPacketCount = oBatch.getInputCount();
    size_t nCurrIdx = 0;
    const std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // mat content becomes invalid on next getInput call
    lvAssert(!vInitInput.empty() && vInitInput.size()==oBatch.getInputStreamCount());
    const cv::Size oOrigImgSize = vInitInput[0].size();
    lvAssert(vInitInput[1].size()==oOrigImgSize);
#if USE_UNCALIB_FMAT_ESTIM
#if LOAD_POINTS_FROM_LAST
    cv::FileStorage oFS(oBatch.getOutputPath()+"/../"+oBatch.getName()+" calib.yml",cv::FileStorage::READ);
    lvAssert(oFS.isOpened());
    std::array<std::vector<cv::Point2f>,2> avMarkers;
    oFS["pts0"] >> avMarkers[0];
    oFS["pts1"] >> avMarkers[1];
#else //!LOAD_POINTS_FROM_LAST
    lv::DisplayHelperPtr pDisplayHelper = lv::DisplayHelper::create(oBatch.getName()+" calib",oBatch.getOutputPath()+"/../");
    std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs = {{
           std::make_pair(vInitInput[0].clone(),oBatch.getInputStreamName(0)),
           std::make_pair(vInitInput[1].clone(),oBatch.getInputStreamName(1))
        }
    };
    std::vector<std::vector<std::array<cv::Point2f,2>>> vvaMarkers(nTotPacketCount);
    std::array<std::vector<cv::Point2f>,2> avMarkers;
    std::array<cv::Mat,2> aMarkerMasks;
    int nNextTile = -1;
    auto lIsValidCoord = [&](const cv::Point2f& p) {
        return p.x>=0 && p.y>=0 && p.x<1 && p.y<1;
    };
    auto lUpdateMasks = [&]() {
        for(size_t m=0; m<2; ++m)
            aMarkerMasks[m] = cv::Mat(vInitInput[m].size(),vInitInput[m].type(),cv::Scalar::all(0));
        std::vector<std::array<cv::Point2f,2>>& vaMarkers = vvaMarkers[nCurrIdx];
        for(size_t a=0; a<vaMarkers.size(); ++a)
            for(size_t m=0; m<2; ++m)
                if(lIsValidCoord(vaMarkers[a][m]))
                    cv::circle(aMarkerMasks[m],cv::Point2i(int(vaMarkers[a][m].x*aMarkerMasks[m].cols),int(vaMarkers[a][m].y*aMarkerMasks[m].rows)),1,cv::Scalar::all(lIsValidCoord(vaMarkers[a][m^1])?255:127),-1);
    };
    auto lUpdateMarkers = [&](const lv::DisplayHelper::CallbackData& oData) {
        const cv::Point2f vClickPos(float(oData.oInternalPosition.x)/oData.oTileSize.width,float(oData.oInternalPosition.y)/oData.oTileSize.height);
        if(lIsValidCoord(vClickPos)) {
            const int nCurrTile = oData.oPosition.x/oData.oTileSize.width;
            std::vector<std::array<cv::Point2f,2>>& vaMarkers = vvaMarkers[nCurrIdx];
            if(oData.nFlags==cv::EVENT_FLAG_LBUTTON) {
                if(nNextTile==-1) {
                    std::array<cv::Point2f,2> aNewPair;
                    aNewPair[nCurrTile] = vClickPos;
                    nNextTile = nCurrTile^1;
                    aNewPair[nNextTile] = cv::Point2f(-1,-1);
                    vaMarkers.push_back(std::move(aNewPair));
                    lUpdateMasks();
                }
                else if(nNextTile==nCurrTile) {
                    std::array<cv::Point2f,2>& aLastPair = vaMarkers.back();
                    aLastPair[nCurrTile] = vClickPos;
                    nNextTile = -1;
                    lUpdateMasks();
                }
            }
            else if(oData.nFlags==cv::EVENT_FLAG_RBUTTON) {
                const float fMinDist = 3.0f/std::max(oData.oTileSize.width,oData.oTileSize.height);
                for(size_t a=0; a<vaMarkers.size(); ++a) {
                    if(lv::L2dist(cv::Vec2f(vaMarkers[a][nCurrTile]),cv::Vec2f(vClickPos))<fMinDist) {
                        vaMarkers.erase(vaMarkers.begin()+a);
                        lUpdateMasks();
                        break;
                    }
                }
            }
        }
    };
    const cv::Size oDisplayTileSize(960,720);
    pDisplayHelper->setMouseCallback([&](const lv::DisplayHelper::CallbackData& oData) {
        if(oData.nEvent==cv::EVENT_LBUTTONDOWN || oData.nEvent==cv::EVENT_RBUTTONDOWN)
            lUpdateMarkers(oData);
        pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
    });
    pDisplayHelper->setContinuousUpdates(true);
    while(nCurrIdx<nTotPacketCount) {
        std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << std::endl;
        const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
        lvDbgAssert(vCurrInput.size()==vInitInput.size());
        int nKeyPressed = -1;
        lUpdateMasks();
        do {
            for(size_t m = 0; m<2; ++m) {
                cv::bitwise_or(vCurrInput[m],aMarkerMasks[m],vvDisplayPairs[0][m].first);
                cv::bitwise_and(vvDisplayPairs[0][m].first,aMarkerMasks[m],vvDisplayPairs[0][m].first,aMarkerMasks[m]>0);
            }
            pDisplayHelper->display(vvDisplayPairs,oDisplayTileSize);
            nKeyPressed = pDisplayHelper->waitKey();
        } while(nKeyPressed==-1);
        if(nNextTile!=-1) {
            vvaMarkers[nCurrIdx].erase(vvaMarkers[nCurrIdx].begin()+vvaMarkers[nCurrIdx].size()-1);
            nNextTile = -1;
        }
        if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
            break;
        else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0)
            --nCurrIdx;
        else if(nKeyPressed!=8/*backspace*/)
            ++nCurrIdx;
    }
    std::cout << "\t\t" << sCurrBatchName << " @ pre-end" << std::endl;
    for(const auto& vaMarkers : vvaMarkers)
        if(!vaMarkers.empty())
            for(const auto& aMarkers : vaMarkers)
                for(size_t a=0; a<2; ++a)
                    avMarkers[a].emplace_back(aMarkers[a].x*oOrigImgSize.width,aMarkers[a].y*oOrigImgSize.height);
    pDisplayHelper->m_oFS << "verstamp" << lv::getVersionStamp();
    pDisplayHelper->m_oFS << "timestamp" << lv::getTimeStamp();
    pDisplayHelper->m_oFS << "pts0" << avMarkers[0];
    pDisplayHelper->m_oFS << "pts1" << avMarkers[1];
    pDisplayHelper = nullptr; // makes sure output is saved (we wont reuse it anyway)
#endif //!LOAD_POINTS_FROM_LAST
    lvAssert(avMarkers[0].size()==avMarkers[1].size());
    std::vector<uchar> vInlinerMask;
    std::array<std::vector<cv::Point2f>,2> avInlierMarkers;
    const int nRANSAC_MaxDist = 10;
    const double dRANSAC_conf = 0.999;
    const cv::Mat oFundMat = cv::findFundamentalMat(avMarkers[0],avMarkers[1],USE_FMAT_RANSAC_ESTIM?cv::FM_RANSAC:cv::FM_8POINT,nRANSAC_MaxDist,dRANSAC_conf,vInlinerMask);
    for(size_t p=0; p<avMarkers[0].size(); ++p) {
        if(vInlinerMask[p]) {
            const cv::Mat p0 = (cv::Mat_<double>(3,1) << avMarkers[0][p].x,avMarkers[0][p].y,1.0);
            const cv::Mat p1 = (cv::Mat_<double>(1,3) << avMarkers[1][p].x,avMarkers[1][p].y,1.0);
            std::cout << "p[" << p << "] err = " << cv::Mat(p1*oFundMat*p0).at<double>(0,0)/cv::norm(oFundMat,cv::NORM_L2) << std::endl;
            for(size_t a=0; a<2; ++a)
                avInlierMarkers[a].push_back(avMarkers[a][p]);
            /*std::vector<cv::Point2f> vpts = {avMarkers[0][p]};
            std::vector<cv::Point3f> vlines;
            cv::computeCorrespondEpilines(vpts,0,oFundMat,vlines);
            const std::vector<cv::Mat>& test = oBatch.getInputArray(0);
            cv::Mat in = test[0].clone(), out = test[1].clone();
            cv::circle(in,vpts[0],3,cv::Scalar(0,0,255),-1);
            for(int c=0; c<out.cols; ++c) {
                const cv::Point2f newpt(c,-(vlines[0].x*c+vlines[0].z)/vlines[0].y);
                if(newpt.x>=0 && newpt.x<test[1].cols && newpt.y>=0 && newpt.y<test[1].rows)
                    cv::circle(out,newpt,3,cv::Scalar(255),-1);
            }
            cv::circle(out,avMarkers[1][p],3,cv::Scalar(0,0,255),-1);
            cv::imshow("0",in);
            cv::imshow("1",out);
            cv::waitKey(0);*/
        }
        else
            std::cout << "p[" << p << "] OUTLIER" << std::endl;
    }
    std::array<cv::Mat,2> aRectifHoms;
    cv::stereoRectifyUncalibrated(avInlierMarkers[0],avInlierMarkers[1],oFundMat,oOrigImgSize,aRectifHoms[0],aRectifHoms[1],0);
    /*for(size_t a=0; a<2; ++a)
        std::cout << "aRectifHoms[" << a << "] = " << aRectifHoms[a] << std::endl;*/
    nCurrIdx = 0;
    while(nCurrIdx<nTotPacketCount) {
        std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << std::endl;
        const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
        lvDbgAssert(vCurrInput.size()==vInitInput.size());
        std::array<cv::Mat,2> aCurrRectifInput;
        for(size_t a=0; a<2; ++a) {
            cv::warpPerspective(vCurrInput[a],aCurrRectifInput[a],aRectifHoms[a],vCurrInput[a].size());
            cv::imshow(std::string("aCurrRectifInput_")+std::to_string(a),aCurrRectifInput[a]);
        }
        int nKeyPressed = cv::waitKey(0);
        if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
            break;
        else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0)
            --nCurrIdx;
        else if(nKeyPressed!=8/*backspace*/)
            ++nCurrIdx;
    }
#else //!USE_UNCALIB_FMAT_ESTIM

    const std::string sBaseCalibDataPath = oBatch.getDataPath()+"calib/";
    std::array<cv::Mat,2> aCamMats,aDistCoeffs;
    cv::Mat oRotMat,oTranslMat,oEssMat,oFundMat;

#if LOAD_CALIB_FROM_LAST

    cv::FileStorage oCalibFile(sBaseCalibDataPath+"calibdata.yml",cv::FileStorage::READ);
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

#else //!LOAD_CALIB_FROM_LAST

    std::array<std::vector<std::vector<cv::Point2f>>,2> avvImagePts;
    std::array<std::vector<cv::Point2f>,2> avImagePts;
    std::vector<std::vector<cv::Point3f>> vvWorldPts;
    std::array<std::vector<cv::Mat>,2> avCalibInputs;
    lvCout << "\tloading exported calib data...\n";
    if(oBatch.getName()=="Scene 1") {
        // scene 1 image calib data missing; use new exports from matlab calib toolbox
        std::ifstream oMetaDataFile(sBaseCalibDataPath+"export/metadata.txt");
        lvAssert(oMetaDataFile.is_open());
        size_t nFirstIdx,nLastIdx;
        lvAssert(oMetaDataFile >> nFirstIdx);
        lvAssert(oMetaDataFile >> nLastIdx);
        lvAssert(nFirstIdx<=nLastIdx);
        for(size_t nIdx=nFirstIdx; nIdx<=nLastIdx; ++nIdx) {
            const std::string sIdxStr = std::to_string(nIdx);
            cv::Mat oVisible = cv::imread(sBaseCalibDataPath+"export/RGB"+sIdxStr+".jpg",cv::IMREAD_COLOR);
            cv::Mat oThermal = cv::imread(sBaseCalibDataPath+"export/T"+sIdxStr+".jpg",cv::IMREAD_COLOR);
            std::ifstream oVisibleData(sBaseCalibDataPath+"export/RGB"+sIdxStr+".txt");
            std::ifstream oThermalData(sBaseCalibDataPath+"export/T"+sIdxStr+".txt");
            if(oVisible.empty() || oThermal.empty() || !oVisibleData.is_open() || !oThermalData.is_open()) {
                lvCout << "\t\tskipping exported pair #" << sIdxStr << "...\n";
                continue;
            }
            lvAssert(oVisible.size()==oThermal.size() && oVisible.size()==oOrigImgSize);
            avCalibInputs[0].push_back(oVisible.clone());
            avCalibInputs[1].push_back(oThermal.clone());
            size_t nVisiblePointCount = 0, nThermalPointCount = 0;
            float fWorldPosX,fWorldPosY,fWorldPosZ,fImagePosX,fImagePosY;
            std::vector<cv::Point3f> vWorldPtsValid;
            vvWorldPts.emplace_back();
            avvImagePts[0].emplace_back();
            while(oVisibleData>>fWorldPosX && oVisibleData>>fWorldPosY && oVisibleData>>fWorldPosZ && oVisibleData>>fImagePosX && oVisibleData>>fImagePosY) {
                // fix world pos exported from matlab (was 40mm, real is 35mm, and x/y inverted)
                //vvWorldPts.back().emplace_back(((fWorldPosY*35)/40)/1000,((fWorldPosX*35)/40)/1000,((fWorldPosZ*35)/40)/1000);
                vvWorldPts.back().emplace_back(fWorldPosY/1000,fWorldPosX/1000,fWorldPosZ/1000);
                avvImagePts[0].back().emplace_back(fImagePosX,fImagePosY);
                cv::circle(oVisible,avvImagePts[0].back().back(),2,cv::Scalar_<uchar>(0,0,255),-1);
                ++nVisiblePointCount;
            }
            avImagePts[0].insert(avImagePts[0].end(),avvImagePts[0].back().begin(),avvImagePts[0].back().end());
            avvImagePts[1].emplace_back();
            while(oThermalData>>fWorldPosX && oThermalData>>fWorldPosY && oThermalData>>fWorldPosZ && oThermalData>>fImagePosX && oThermalData>>fImagePosY) {
                // fix world pos exported from matlab (was 40mm, real is 35mm, and x/y inverted)
                //vWorldPtsValid.emplace_back(((fWorldPosY*35)/40)/1000,((fWorldPosX*35)/40)/1000,((fWorldPosZ*35)/40)/1000);
                vWorldPtsValid.emplace_back(fWorldPosY/1000,fWorldPosX/1000,fWorldPosZ/1000);
                avvImagePts[1].back().emplace_back(fImagePosX,fImagePosY);
                cv::circle(oThermal,avvImagePts[1].back().back(),2,cv::Scalar_<uchar>(0,0,255),-1);
                ++nThermalPointCount;
            }
            avImagePts[1].insert(avImagePts[1].end(),avvImagePts[1].back().begin(),avvImagePts[1].back().end());
            lvAssert(nThermalPointCount==nVisiblePointCount);
            lvAssert(avvImagePts[0].back().size()==avvImagePts[1].back().size());
            lvAssert(avvImagePts[0].back().size()==nVisiblePointCount);
            lvAssert(avImagePts[0].size()==avImagePts[1].size());
            lvAssert(vvWorldPts.back().size()==vWorldPtsValid.size());
            lvAssert(vWorldPtsValid.size()==nVisiblePointCount);
            for(size_t n=0; n<vWorldPtsValid.size(); ++n)
                lvAssert(cv::norm(vvWorldPts.back()[n]-vWorldPtsValid[n])<0.01);

            /*lvPrint(vvWorldPts.back());
            lvPrint(avvImagePts[0].back());
            cv::imshow("visible",oVisible);
            cv::imshow("thermal",oThermal);
            cv::waitKey(0);*/
        }
    }
    else {
        // use original image calib points ('chessboard' text files)
        std::ifstream oVisibleMetaDataFile(sBaseCalibDataPath+"lChessboard3D_nbrCorners.txt");
        lvAssert(oVisibleMetaDataFile.is_open());
        std::ifstream oThermalMetaDataFile(sBaseCalibDataPath+"rChessboard3D_nbrCorners.txt");
        lvAssert(oThermalMetaDataFile.is_open());
        std::string sIndicesLine,sIndicesLineValid;
        lvAssert(std::getline(oVisibleMetaDataFile,sIndicesLine));
        lvAssert(std::getline(oThermalMetaDataFile,sIndicesLineValid));
        lvAssert(sIndicesLine==sIndicesLineValid);
        const std::vector<std::string> vsImageIndices = lv::split(sIndicesLine,',');
        std::vector<size_t> vnImageIndices(vsImageIndices.size());
        for(size_t nImageIdx=0; nImageIdx<vsImageIndices.size(); ++nImageIdx)
            std::stringstream(vsImageIndices[nImageIdx]) >> vnImageIndices[nImageIdx];
        lvAssert(vnImageIndices.size()>0);
        std::vector<cv::Size> vImagePointCounts(vnImageIndices.size());
        for(size_t nIdx=0; nIdx<vnImageIndices.size(); ++nIdx) {
            char cDelim;
            int nCoordsX,nCoordsY;
            lvAssert(oVisibleMetaDataFile>>nCoordsX);
            lvAssert(oVisibleMetaDataFile>>cDelim);
            lvAssert(oVisibleMetaDataFile>>nCoordsY);
            vImagePointCounts[nIdx] = cv::Size(nCoordsX,nCoordsY);
            lvAssert(oThermalMetaDataFile>>nCoordsX);
            lvAssert(oThermalMetaDataFile>>cDelim);
            lvAssert(oThermalMetaDataFile>>nCoordsY);
            lvAssert(vImagePointCounts[nIdx]==cv::Size(nCoordsX,nCoordsY));
        }
        lvAssert(vnImageIndices.size()==vImagePointCounts.size());
        std::ifstream oVisibleImagePtsFile(sBaseCalibDataPath+"lChessboard3D_imagec.txt");
        std::ifstream oVisibleObjectPtsFile(sBaseCalibDataPath+"lChessboard3D_objectc.txt");
        lvAssert(oVisibleImagePtsFile.is_open() && oVisibleObjectPtsFile.is_open());
        std::ifstream oThermalImagePtsFile(sBaseCalibDataPath+"rChessboard3D_imagec.txt");
        std::ifstream oThermalObjectPtsFile(sBaseCalibDataPath+"rChessboard3D_objectc.txt");
        lvAssert(oThermalImagePtsFile.is_open() && oThermalObjectPtsFile.is_open());
        const auto lPointReader = [](std::ifstream& oFile, size_t nIdx, bool bWorldPt) {
            if(nIdx!=0) {
                std::string sDummyLine;
                lvAssert(std::getline(oFile,sDummyLine));
            }
            std::string sCoordsLineX,sCoordsLineY,sCoordsLineZ;
            lvAssert(std::getline(oFile,sCoordsLineX));
            lvAssert(std::getline(oFile,sCoordsLineY));
            if(bWorldPt)
                lvAssert(std::getline(oFile,sCoordsLineZ));
            const std::vector<std::string> vsCoordsX = lv::split(sCoordsLineX,',');
            const std::vector<std::string> vsCoordsY = lv::split(sCoordsLineY,',');
            const std::vector<std::string> vsCoordsZ = lv::split(sCoordsLineZ,',');
            lvAssert(vsCoordsX.size()==vsCoordsY.size() && (!bWorldPt || vsCoordsY.size()==vsCoordsZ.size()));
            lvAssert(!vsCoordsX.empty());
            const size_t nCoords = vsCoordsX.size();
            std::vector<cv::Point3f> vPts(nCoords);
            for(size_t nCoordIdx=0; nCoordIdx<nCoords; ++nCoordIdx) {
                vPts[nCoordIdx].x = std::stof(vsCoordsX[nCoordIdx]);
                vPts[nCoordIdx].y = std::stof(vsCoordsY[nCoordIdx]);
                if(bWorldPt) {
                    vPts[nCoordIdx].z = std::stof(vsCoordsZ[nCoordIdx]);
                    std::swap(vPts[nCoordIdx].x,vPts[nCoordIdx].y);
                    vPts[nCoordIdx] /= 1000;
                }
            }
            return vPts;
        };
        const auto lPointConv = [](const std::vector<cv::Point3f>& vPtsIn) {
            std::vector<cv::Point2f> vPtsOut(vPtsIn.size());
            for(size_t nPtIdx=0; nPtIdx<vPtsIn.size(); ++nPtIdx)
                vPtsOut[nPtIdx] = cv::Point2f(vPtsIn[nPtIdx].x,vPtsIn[nPtIdx].y);
            return vPtsOut;
        };
        for(size_t nImageIdx=0; nImageIdx<vnImageIndices.size(); ++nImageIdx) {
            const std::string sImageIdxStr = std::to_string(vnImageIndices[nImageIdx]);
            cv::Mat oVisible = cv::imread(sBaseCalibDataPath+"RGB"+sImageIdxStr+".jpg",cv::IMREAD_COLOR);
            cv::Mat oThermal = cv::imread(sBaseCalibDataPath+"T"+sImageIdxStr+".jpg",cv::IMREAD_COLOR);
            lvAssert(!oVisible.empty() && !oThermal.empty());
            lvAssert(oVisible.size()==oThermal.size() && oVisible.size()==oOrigImgSize);
            avCalibInputs[0].push_back(oVisible.clone());
            avCalibInputs[1].push_back(oThermal.clone());
            const std::vector<cv::Point3f> vWorldPts = lPointReader(oVisibleObjectPtsFile,nImageIdx,true);
            const std::vector<cv::Point3f> vWorldPtsValid = lPointReader(oThermalObjectPtsFile,nImageIdx,true);
            lvAssert(vWorldPts==vWorldPtsValid);
            const std::vector<cv::Point2f> vVisibleImagePts = lPointConv(lPointReader(oVisibleImagePtsFile,nImageIdx,false));
            const std::vector<cv::Point2f> vThermalImagePts = lPointConv(lPointReader(oThermalImagePtsFile,nImageIdx,false));
            lvAssert(vVisibleImagePts.size()==vWorldPts.size() && vThermalImagePts.size()==vWorldPts.size());
            vvWorldPts.push_back(vWorldPts);
            avvImagePts[0].push_back(vVisibleImagePts);
            avvImagePts[1].push_back(vThermalImagePts);
            const size_t nPts = vWorldPts.size();
            for(size_t nPtIdx=0; nPtIdx<nPts; ++nPtIdx) {
                avImagePts[0].push_back(vVisibleImagePts[nPtIdx]);
                avImagePts[1].push_back(vThermalImagePts[nPtIdx]);
                cv::circle(oVisible,vVisibleImagePts[nPtIdx],2,cv::Scalar_<uchar>(0,0,255),-1);
                cv::circle(oThermal,vThermalImagePts[nPtIdx],2,cv::Scalar_<uchar>(0,0,255),-1);
            }
            /*lvPrint(vvWorldPts.back());
            lvPrint(avvImagePts[0].back());
            cv::imshow("visible",oVisible);
            cv::imshow("thermal",oThermal);
            cv::waitKey(0);*/
        }
    }
    lvAssert(avCalibInputs[0].size()==avvImagePts[0].size());

#if USE_INTRINSIC_GUESS
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
#endif //USE_INTRINSIC_GUESS

    const double dStereoCalibErr = cv::stereoCalibrate(vvWorldPts,avvImagePts[0],avvImagePts[1],
                                                       aCamMats[0],aDistCoeffs[0],aCamMats[1],aDistCoeffs[1],
                                                       oOrigImgSize,oRotMat,oTranslMat,oEssMat,oFundMat,
                                                       USE_INTRINSIC_GUESS?cv::CALIB_USE_INTRINSIC_GUESS:0,
                                                       //0,
                                                       //cv::CALIB_FIX_INTRINSIC,
                                                       //CV_CALIB_USE_INTRINSIC_GUESS+CV_CALIB_FIX_PRINCIPAL_POINT+CV_CALIB_FIX_ASPECT_RATIO+CV_CALIB_ZERO_TANGENT_DIST,
                                                       //cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_USE_INTRINSIC_GUESS + cv::CALIB_SAME_FOCAL_LENGTH + cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
                                                       //cv::CALIB_RATIONAL_MODEL + cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5,
                                                       //cv::CALIB_ZERO_TANGENT_DIST,
                                                       cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,1000,1e-6));
    lvPrint(dStereoCalibErr);

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

    {
        cv::FileStorage oCalibFile(sBaseCalibDataPath+"calibdata.yml",cv::FileStorage::WRITE);
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
        oCalibFile << "dStereoCalibErr" << dStereoCalibErr;
    }

#endif //!LOAD_CALIB_FROM_LAST

    std::array<cv::Mat,2> aRectifRotMats,aRectifProjMats;
    cv::Mat oDispToDepthMap;
    cv::stereoRectify(aCamMats[0],aDistCoeffs[0],aCamMats[1],aDistCoeffs[1],oOrigImgSize,oRotMat,oTranslMat,
                      aRectifRotMats[0],aRectifRotMats[1],aRectifProjMats[0],aRectifProjMats[1],oDispToDepthMap,
                      //0,
                      cv::CALIB_ZERO_DISPARITY,
                      -1,cv::Size());

    std::array<std::array<cv::Mat,2>,2> aaRectifMaps;
    cv::initUndistortRectifyMap(aCamMats[0],aDistCoeffs[0],aRectifRotMats[0],aRectifProjMats[0],oOrigImgSize,
                                CV_16SC2,aaRectifMaps[0][0],aaRectifMaps[0][1]);
    cv::initUndistortRectifyMap(aCamMats[1],aDistCoeffs[1],aRectifRotMats[1],aRectifProjMats[1],oOrigImgSize,
                                CV_16SC2,aaRectifMaps[1][0],aaRectifMaps[1][1]);

    nCurrIdx = 0;
    while(nCurrIdx<nTotPacketCount) {
        std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << std::endl;
        const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
        lvDbgAssert(vCurrInput.size()==vInitInput.size());
        std::array<cv::Mat,2> aCurrRectifInput;
        for(size_t a=0; a<2; ++a) {
            cv::remap(vCurrInput[a],aCurrRectifInput[a],aaRectifMaps[a][0],aaRectifMaps[a][1],cv::INTER_LINEAR);
            cv::imshow(std::string("aCurrRectifInput_")+std::to_string(a),aCurrRectifInput[a]);
        }
        int nKeyPressed = cv::waitKey(0);
        if(nKeyPressed==(int)'q' || nKeyPressed==27/*escape*/)
            break;
        else if(nKeyPressed==8/*backspace*/ && nCurrIdx>0)
            --nCurrIdx;
        else if(nKeyPressed!=8/*backspace*/)
            ++nCurrIdx;
    }
#endif //!USE_UNCALIB_FMAT_ESTIM
    std::cout << "\t\t" << sCurrBatchName << " @ post-end" << std::endl;
}
