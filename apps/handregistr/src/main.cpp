
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
#define USE_FMAT_RANSAC_ESTIM   0
#define LOAD_POINTS_FROM_LAST   0

////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      1

using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::Dataset_VAPtrimod2016,lv::NonParallel>;
void Analyze(lv::IDataHandlerPtr pBatch);

int main(int, char**) {
    try {
        DatasetType::Ptr pDataset = DatasetType::create(DATASET_OUTPUT_PATH,false,false,false,1.0,false,true);
        lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false);
        const size_t nTotPackets = pDataset->getInputCount();
        const size_t nTotBatches = vpBatches.size();
        if(nTotBatches==0 || nTotPackets==0)
            lvError_("Could not parse any data for dataset '%s'",pDataset->getName().c_str());
        std::cout << "Parsing complete. [" << nTotBatches << " batch(es)]" << std::endl;
        std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
        for(lv::IDataHandlerPtr pBatch : vpBatches)
            Analyze(pBatch);
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    return 0;
}

void Analyze(lv::IDataHandlerPtr pBatch) {
    DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
    lvAssert(oBatch.getInputPacketType()==lv::ImageArrayPacket && oBatch.getInputStreamCount()==2 && oBatch.getInputCount()>=1);
    if(DATASET_PRECACHING)
        oBatch.startPrecaching(false);
    const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
    std::cout << "\t\t" << sCurrBatchName << " @ init" << std::endl;
    const size_t nTotPacketCount = oBatch.getInputCount();
    size_t nCurrIdx = 0;
    const std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // mat content becomes invalid on next getInput call
    lvAssert(!vInitInput.empty() && vInitInput.size()==oBatch.getInputStreamCount());
    const cv::Size oOrigTileSize = vInitInput[0].size();
    lvAssert(vInitInput[1].size()==oOrigTileSize);
#if LOAD_POINTS_FROM_LAST
    cv::FileStorage oFS(oBatch.getOutputPath()+"/../"+oBatch.getName()+" calib.yml",cv::FileStorage::READ);
    lvAssert(oFS.isOpened());
    std::array<std::vector<cv::Point2f>,2> avMarkers;
    oFS["pts0"] >> avMarkers[0];
    oFS["pts1"] >> avMarkers[1];
#else //!LOAD_POINTS_FROM_LAST
    cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oBatch.getName()+" calib",oBatch.getOutputPath()+"/../");
    std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs = {{
            std::make_pair(vInitInput[0].clone(),oBatch.getInputStreamName(0)),
            std::make_pair(vInitInput[1].clone(),oBatch.getInputStreamName(1))
        }
    };
    std::vector<std::vector<std::array<cv::Point2f,2>>> vvaMarkers(nTotPacketCount);
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
    auto lUpdateMarkers = [&](const cv::DisplayHelper::CallbackData& oData) {
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
    pDisplayHelper->setMouseCallback([&](const cv::DisplayHelper::CallbackData& oData) {
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
    std::array<std::vector<cv::Point2f>,2> avMarkers;
    for(const auto& vaMarkers : vvaMarkers)
        if(!vaMarkers.empty())
            for(const auto& aMarkers : vaMarkers)
                for(size_t a=0; a<2; ++a)
                    avMarkers[a].emplace_back(aMarkers[a].x*oOrigTileSize.width,aMarkers[a].y*oOrigTileSize.height);
    pDisplayHelper->m_oFS << "verstamp" << lv::getVersionStamp();
    pDisplayHelper->m_oFS << "timestamp" << lv::getTimeStamp();
    pDisplayHelper->m_oFS << "pts0" << avMarkers[0];
    pDisplayHelper->m_oFS << "pts1" << avMarkers[1];
    pDisplayHelper = nullptr; // makes sure output is saved (we wont reuse it anyway)
#endif //LOAD_POINTS_FROM_LAST
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
    cv::stereoRectifyUncalibrated(avInlierMarkers[0],avInlierMarkers[1],oFundMat,oOrigTileSize,aRectifHoms[0],aRectifHoms[1],0);
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
    std::cout << "\t\t" << sCurrBatchName << " @ post-end" << std::endl;
}
