
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
#include "litiv/imgproc/ForegroundStereoMatcher.hpp"

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define EVALUATE_OUTPUT         0
#define DISPLAY_OUTPUT          0
#define GLOBAL_VERBOSITY        2
////////////////////////////////
#define DATASET_VAPTRIMOD       1
#define DATASET_MINI_TESTS      0
////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      0
#define DATASET_SCALE_FACTOR    1//0.5
#define DATASET_WORKTHREADS     1
////////////////////////////////
#define DATASET_USE_DISPARITY_EVAL         0
#define DATASET_USE_HALF_GT_INPUT_FLAG     0
#define DATASET_USE_PRECALC_FEATURES       1

#if (DATASET_VAPTRIMOD+DATASET_MINI_TESTS/*+...*/)!=1
#error "Must pick a single dataset."
#endif //(DATASET_+.../*+...*/)!=1
#if DATASET_VAPTRIMOD
#define DATASET_ID Dataset_VAPtrimod2016
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    false,                                        /* bool bLoadDepth=true */\
    true,                                         /* bool bUndistort=true */\
    true,                                         /* bool bHorizRectify=false */\
    false,                                        /* bool bEvalStereoDisp=false */\
    7,                                            /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
#elif DATASET_MINI_TESTS
#include "cosegm_tests.hpp"
#define DATASET_ID Dataset_CosegmTests
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    false,                                        /* bool bEvalStereoDisp=false */\
    1,                                            /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
//#elif DATASET_...
#endif //DATASET_...

void Analyze(std::string sWorkerName, lv::IDataHandlerPtr pBatch);
using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::DATASET_ID,lv::NonParallel>;

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
        lvAssert(oBatch.getInputStreamCount()==4); // expect approx fg masks to be interlaced with input images
        lvAssert(oBatch.getOutputStreamCount()==2); // we always only eval one output type at a time (fg masks or disparity)
        if(DATASET_PRECACHING)
            oBatch.startPrecaching(!bool(EVALUATE_OUTPUT));
        const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
        std::cout << "\t\t" << sCurrBatchName << " @ init [" << sWorkerName << "]" << std::endl;
        const std::vector<cv::Mat>& vROIs = oBatch.getFrameROIArray();
        lvAssert(!vROIs.empty() && vROIs.size()==oBatch.getInputStreamCount());
        size_t nCurrIdx = 25;
        const std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // note: mat content becomes invalid on next getInput call
        lvAssert(vInitInput.size()==oBatch.getInputStreamCount());
        for(size_t nStreamIdx=0; nStreamIdx<vInitInput.size(); ++nStreamIdx) {
            lvAssert(vInitInput[nStreamIdx].size()==vInitInput[0].size());
            lvLog_(2,"\tinput %d := %s   (roi=%s)",(int)nStreamIdx,lv::MatInfo(vInitInput[nStreamIdx]).str().c_str(),lv::MatInfo(vROIs[nStreamIdx]).str().c_str());
            //cv::imshow(std::string("vInitInput_")+std::to_string(nStreamIdx),vInitInput[nStreamIdx]);
            //cv::imshow(std::string("vROI_")+std::to_string(nStreamIdx),vROIs[nStreamIdx]);
        }
        //cv::waitKey(0);
        const std::vector<lv::MatInfo> oInfoArray = oBatch.getInputInfoArray();
        const lv::MatSize oFrameSize = oInfoArray[0].size;
        const size_t nMinDisp = oBatch.getMinDisparity(), nMaxDisp = oBatch.getMaxDisparity();
        lvLog_(2,"\tdisp = [%d,%d]",(int)nMinDisp,(int)nMaxDisp);
        std::shared_ptr<StereoSegmMatcher> pAlgo = std::make_shared<StereoSegmMatcher>(nMinDisp,nMaxDisp);
        pAlgo->initialize(std::array<cv::Mat,2>{vROIs[0],vROIs[2]});
        oBatch.setFeaturesDirName(pAlgo->getFeatureExtractorName());
        constexpr size_t nExpectedAlgoInputCount = StereoSegmMatcher::getInputStreamCount();
        constexpr size_t nExpectedAlgoOutputCount = StereoSegmMatcher::getOutputStreamCount();
        static_assert(nExpectedAlgoInputCount==4,"unexpected input stream count for instanced algo");
        static_assert(nExpectedAlgoOutputCount==4,"unexpected output stream count for instanced algo");
        std::vector<cv::Mat> vCurrOutput(nExpectedAlgoOutputCount);
        std::vector<cv::Mat> vCurrFGMasks(nExpectedAlgoOutputCount/2);
        std::vector<cv::Mat> vCurrStereoMaps(nExpectedAlgoOutputCount/2);
    #if DISPLAY_OUTPUT>0
        lv::DisplayHelperPtr pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
        lvAssert((vInitInput.size()%2)==0); // assume masks are interlaced with input images
        std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs(vInitInput.size()/2);
        for(size_t nDisplayRowIdx=0; nDisplayRowIdx<vInitInput.size()/2; ++nDisplayRowIdx) {
            std::vector<std::pair<cv::Mat,std::string>> vRow(4);
            vRow[0] = std::make_pair(cv::Mat(),oBatch.getInputStreamName(nDisplayRowIdx*2));
            vRow[1] = std::make_pair(cv::Mat(),"INPUT MASK");
            vRow[2] = std::make_pair(cv::Mat(),"OUTPUT DISP");
            vRow[3] = std::make_pair(cv::Mat(),"OUTPUT MASK");
            vvDisplayPairs[nDisplayRowIdx] = vRow;
        }
    #endif //DISPLAY_OUTPUT>0
        const size_t nTotPacketCount = oBatch.getFrameCount();
        oBatch.startProcessing();
        while(nCurrIdx<nTotPacketCount) {
            //if(!((nCurrIdx+1)%100))
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << "   [" << sWorkerName << "]" << std::endl;
            const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
            lvDbgAssert(vCurrInput.size()==oBatch.getInputStreamCount());
            lvDbgAssert(vCurrInput.size()==nExpectedAlgoInputCount);
            for(size_t nStreamIdx=0; nStreamIdx<vCurrInput.size(); ++nStreamIdx)
                lvDbgAssert(oFrameSize==vCurrInput[nStreamIdx].size());
        #if DATASET_USE_PRECALC_FEATURES
            const cv::Mat& oNextFeatsPacket = oBatch.loadFeatures(nCurrIdx);
            lvAssert__(!oNextFeatsPacket.empty(),"could not load precalc feature packet for idx=%d",(int)nCurrIdx);
            pAlgo->setNextFeatures(oNextFeatsPacket);
        #else //!DATASET_USE_PRECALC_FEATURES
            cv::Mat oNextFeatsPacket;
            pAlgo->calcFeatures(lv::convertVectorToArray<nExpectedAlgoInputCount>(vCurrInput),&oNextFeatsPacket);
            oBatch.saveFeatures(nCurrIdx,oNextFeatsPacket);
            pAlgo->setNextFeatures(oNextFeatsPacket);
        #endif //!DATASET_USE_PRECALC_FEATURES
            pAlgo->apply(vCurrInput,vCurrOutput/*,dDefaultThreshold*/);
            lvDbgAssert(vCurrOutput.size()==nExpectedAlgoOutputCount);
            using OutputLabelType = StereoSegmMatcher::LabelType;
            for(size_t nOutputArrayIdx=0; nOutputArrayIdx<vCurrOutput.size(); ++nOutputArrayIdx) {
                lvAssert(vCurrOutput[nOutputArrayIdx].type()==lv::MatRawType_<OutputLabelType>());
                lvAssert(vCurrOutput[nOutputArrayIdx].size()==oFrameSize());
                if(nOutputArrayIdx%2)
                    vCurrStereoMaps[nOutputArrayIdx/2] = vCurrOutput[nOutputArrayIdx];
                else
                    vCurrFGMasks[nOutputArrayIdx/2] = vCurrOutput[nOutputArrayIdx];
            }
            lvDbgAssert(vCurrFGMasks.size()==oBatch.getOutputStreamCount());
            lvDbgAssert(vCurrStereoMaps.size()==oBatch.getOutputStreamCount());
        #if DISPLAY_OUTPUT>0
            const std::vector<cv::Mat>& vCurrGT = oBatch.getGTArray(nCurrIdx);
            lvAssert(vCurrGT.size()==vCurrFGMasks.size() && vCurrGT.size()==vCurrStereoMaps.size());
        #if DISPLAY_OUTPUT>1
            if(!vCurrGT[0].empty() && !vCurrGT[1].empty()) {
                cv::Mat test_rgb = vCurrInput[0].clone(), gt_rgb_mask, approx_rgb_mask;
                cv::bitwise_or(test_rgb/2,cv::Mat(vCurrInput[0].size(),CV_8UC3,cv::Scalar_<uchar>(127)),gt_rgb_mask,vCurrGT[0]);
                cv::bitwise_or(test_rgb/2,cv::Mat(vCurrInput[0].size(),CV_8UC3,cv::Scalar_<uchar>(127)),approx_rgb_mask,vCurrInput[1]);
                cv::imshow("gt_rgb_mask",gt_rgb_mask);
                cv::imshow("approx_rgb_mask",approx_rgb_mask);
                cv::Mat test_thermal = vCurrInput[2].clone(), gt_thermal_mask, approx_thermal_mask;
                cv::cvtColor(test_thermal,test_thermal,cv::COLOR_GRAY2BGR);
                cv::bitwise_or(test_thermal/2,cv::Mat(vCurrInput[2].size(),CV_8UC3,cv::Scalar_<uchar>(127)),gt_thermal_mask,vCurrGT[1]);
                cv::bitwise_or(test_thermal/2,cv::Mat(vCurrInput[2].size(),CV_8UC3,cv::Scalar_<uchar>(127)),approx_thermal_mask,vCurrInput[3]);
                cv::imshow("gt_thermal_mask",gt_thermal_mask);
                cv::imshow("approx_thermal_mask",approx_thermal_mask);
                cv::Mat test_merge = test_rgb/2+test_thermal/2;
                cv::imshow("merge",test_merge);
                cv::Mat test_gt;
                cv::merge(std::vector<cv::Mat>{vCurrGT[0]&vROIs[0],cv::Mat::zeros(vCurrGT[0].size(),CV_8UC1),vCurrGT[1]&vROIs[1]},test_gt);
                cv::imshow("merge mask",test_gt);
            }
        #endif //DISPLAY_OUTPUT>1
            for(size_t nDisplayRowIdx=0; nDisplayRowIdx<vCurrInput.size()/2; ++nDisplayRowIdx) {
                vCurrInput[nDisplayRowIdx*2].copyTo(vvDisplayPairs[nDisplayRowIdx][0].first);
                vCurrInput[nDisplayRowIdx*2+1].copyTo(vvDisplayPairs[nDisplayRowIdx][1].first);
                // @@@@@ ship color funcs to custom evaluator? or func in cosegm_tests.hpp?
                vvDisplayPairs[nDisplayRowIdx][2].first.create(vCurrStereoMaps[nDisplayRowIdx].size(),CV_8UC3);
                cv::Mat& oColoredStereoMap = vvDisplayPairs[nDisplayRowIdx][2].first;
                for(int nRowIdx=0; nRowIdx<oColoredStereoMap.rows; ++nRowIdx) {
                    for(int nColIdx=0; nColIdx<oColoredStereoMap.cols; ++nColIdx) {
                        const OutputLabelType nCurrPxLabel = vCurrStereoMaps[nDisplayRowIdx].at<OutputLabelType>(nRowIdx,nColIdx);
                        lvDbgAssert_(nCurrPxLabel<UCHAR_MAX,"cannot properly display all stereo label values with current label type");
                        if(nCurrPxLabel==StereoSegmMatcher::getStereoDontCareLabel())
                            oColoredStereoMap.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(255,0,255);
                        else if(nCurrPxLabel==StereoSegmMatcher::getStereoOccludedLabel())
                            oColoredStereoMap.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b(255,0,0);
                        else
                            oColoredStereoMap.at<cv::Vec3b>(nRowIdx,nColIdx) = cv::Vec3b((uchar)nCurrPxLabel,(uchar)nCurrPxLabel,(uchar)nCurrPxLabel);
                    }
                }
                vCurrFGMasks[nDisplayRowIdx].convertTo(vvDisplayPairs[nDisplayRowIdx][3].first,CV_8U,double(UCHAR_MAX));
            }
            pDisplayHelper->display(vvDisplayPairs,cv::Size(320,240));
            const int nKeyPressed = pDisplayHelper->waitKey();
            if(nKeyPressed==(int)'q')
                break;
        #endif //DISPLAY_OUTPUT>0
            if(oBatch.isEvaluatingStereoDisp())
                oBatch.push(vCurrStereoMaps,nCurrIdx++);
            else
                oBatch.push(vCurrFGMasks,nCurrIdx++);
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
