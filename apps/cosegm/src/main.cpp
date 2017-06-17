
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
#define WRITE_IMG_OUTPUT        1
#define EVALUATE_OUTPUT         1
#define GLOBAL_VERBOSITY        2
////////////////////////////////
#define DATASET_VAPTRIMOD       1
#define DATASET_LITIV2014       0
#define DATASET_MINI_TESTS      0
////////////////////////////////
#define DATASET_OUTPUT_PATH     "results_resegm_subset2_r001"
#define DATASET_PRECACHING      0
#define DATASET_SCALE_FACTOR    1//0.5
#define DATASET_WORKTHREADS     1
////////////////////////////////
#define DATASET_FORCE_RECALC_FEATURES      0
#define DATASET_EVAL_APPROX_MASKS_ONLY     0
#define DATASET_BATCH_START_INDEX          0
#define DATASET_BATCH_STOP_MAX_INDEX       9999

#if (DATASET_VAPTRIMOD+DATASET_LITIV2014+DATASET_MINI_TESTS/*+...*/)!=1
#error "Must pick a single dataset."
#endif //(DATASET_+.../*+...*/)!=1
#if DATASET_VAPTRIMOD
#define DATASET_ID Dataset_VAP_trimod2016
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    false,                                        /* bool bLoadDepth=true */\
    PROCESS_PREPROC_BGSEGM?false:true,            /* bool bUndistort=true */\
    PROCESS_PREPROC_BGSEGM?false:true,            /* bool bHorizRectify=false */\
    false,                                        /* bool bEvalStereoDisp=false */\
    true,                                         /* bool bLoadFrameSubset=false */\
    PROCESS_PREPROC_BGSEGM?0:1/*4*/,              /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
#elif DATASET_LITIV2014
#define DATASET_ID Dataset_LITIV_bilodeau2014
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    false,                                        /* bool bLoadFullVideos=false */\
    true,                                         /* bool bEvalStereoDisp=true */\
    true,                                         /* bool bFlipDisparities=false */\
    false,                                        /* bool bLoadFrameSubset=false */\
    16/*-1*/,                                     /* int nLoadPersonSets=-1 */\
    PROCESS_PREPROC_BGSEGM?0:1/*4*/,              /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
#elif DATASET_MINI_TESTS
#include "cosegm_tests.hpp"
#define DATASET_ID Dataset_CosegmTests
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                          /* const std::string& sOutputDirName */\
    bool(WRITE_IMG_OUTPUT),                       /* bool bSaveOutput=false */\
    bool(EVALUATE_OUTPUT),                        /* bool bUseEvaluator=true */\
    false,                                        /* bool bEvalStereoDisp=false */\
    false,                                        /* bool bLoadFrameSubset=false */\
    1,                                            /* int nLoadInputMasks=0 */\
    DATASET_SCALE_FACTOR                          /* double dScaleFactor=1.0 */
//#elif DATASET_...
#endif //DATASET_...
#if PROCESS_PREPROC_BGSEGM
#define BGSEGM_ALGO_TYPE BackgroundSubtractorPAWCS
#else //!PROCESS_PREPROC_BGSEGM
#include "litiv/imgproc/ForegroundStereoMatcher.hpp"
#endif //!PROCESS_PREPROC_BGSEGM

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
#if PROCESS_PREPROC_BGSEGM
        lvAssert(oBatch.getInputStreamCount()==2); // expect only one input image per camera
#else //!PROCESS_PREPROC_BGSEGM
        lvAssert(oBatch.getInputStreamCount()==4); // expect approx fg masks to be interlaced with input images
#endif //!PROCESS_PREPROC_BGSEGM
        lvAssert(oBatch.getOutputStreamCount()==2); // we always only eval one output type at a time (fg masks or disparity)
        if(DATASET_PRECACHING)
            oBatch.startPrecaching(!bool(EVALUATE_OUTPUT));
        const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
        std::cout << "\t\t" << sCurrBatchName << " @ init [" << sWorkerName << "]" << std::endl;
        const std::vector<cv::Mat>& vROIs = oBatch.getFrameROIArray();
        lvAssert(!vROIs.empty() && vROIs.size()==oBatch.getInputStreamCount());
        size_t nCurrIdx = DATASET_BATCH_START_INDEX;
        const std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // note: mat content becomes invalid on next getInput call
        lvAssert(vInitInput.size()==oBatch.getInputStreamCount());
        for(size_t nStreamIdx=0; nStreamIdx<vInitInput.size(); ++nStreamIdx) {
            lvAssert(vInitInput[nStreamIdx].size()==vInitInput[0].size());
            lvLog_(2,"\tinput %d := %s   (roi=%s)",(int)nStreamIdx,lv::MatInfo(vInitInput[nStreamIdx]).str().c_str(),lv::MatInfo(vROIs[nStreamIdx]).str().c_str());
            if(lv::getVerbosity()>=5) {
                cv::imshow(std::string("vInitInput_")+std::to_string(nStreamIdx),vInitInput[nStreamIdx]);
                cv::imshow(std::string("vROI_")+std::to_string(nStreamIdx),vROIs[nStreamIdx]);
            }
        }
        if(lv::getVerbosity()>=5)
            cv::waitKey(0);
        const std::vector<lv::MatInfo> oInfoArray = oBatch.getInputInfoArray();
        const lv::MatSize oFrameSize = oInfoArray[0].size;
        lv::DisplayHelperPtr pDisplayHelper;
        if(lv::getVerbosity()>=1)
            pDisplayHelper = lv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"../");
        pDisplayHelper->setContinuousUpdates(true);
#if PROCESS_PREPROC_BGSEGM
        std::vector<std::shared_ptr<IBackgroundSubtractor>> vAlgos = {std::make_shared<BGSEGM_ALGO_TYPE>(),std::make_shared<BGSEGM_ALGO_TYPE>()};
        const double dDefaultLearningRate = vAlgos[0]->getDefaultLearningRate();
        for(size_t nCamIdx=0; nCamIdx<2; ++nCamIdx)
            vAlgos[nCamIdx]->initialize(vInitInput[nCamIdx],vROIs[nCamIdx]);
        std::vector<cv::Mat> vCurrFGMasks(2);
#if PROCESS_PREPROC_BGSEGM>1
        const size_t nMaxInitLoops = 50, nMaxInitLoopIdx = 6;
        size_t nCurrInitLoop = 0;
        bool bIncreasingIdxs = true;
#endif //PROCESS_PREPROC_BGSEGM>1
#else //!PROCESS_PREPROC_BGSEGM
        lvAssert(vInitInput.size()==4 && (vInitInput.size()%2)==0); // assume masks are interlaced with input images
        const size_t nMinDisp = oBatch.getMinDisparity(), nMaxDisp = oBatch.getMaxDisparity();
        lvLog_(2,"\tdisp = [%d,%d]",(int)nMinDisp,(int)nMaxDisp);
        constexpr size_t nCameraCount = StereoSegmMatcher::getCameraCount();
        constexpr size_t nExpectedAlgoInputCount = StereoSegmMatcher::getInputStreamCount();
        constexpr size_t nExpectedAlgoOutputCount = StereoSegmMatcher::getOutputStreamCount();
        static_assert(nCameraCount==2,"unexpected algo internal camera head count");
        static_assert(nExpectedAlgoInputCount==4,"unexpected input stream count for instanced algo");
        static_assert(nExpectedAlgoOutputCount==4,"unexpected output stream count for instanced algo");
        using OutputType = StereoSegmMatcher::OutputLabelType;
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
#if !DATASET_EVAL_APPROX_MASKS_ONLY
        std::shared_ptr<StereoSegmMatcher> pAlgo = std::make_shared<StereoSegmMatcher>(nMinDisp,nMaxDisp);
        pAlgo->m_pDisplayHelper = pDisplayHelper;
        pAlgo->initialize(std::array<cv::Mat,2>{vROIs[0],vROIs[2]});
        oBatch.setFeaturesDirName(pAlgo->getFeatureExtractorName());
#endif //!DATASET_EVAL_APPROX_MASKS_ONLY
#endif //!PROCESS_PREPROC_BGSEGM
        const size_t nTotPacketCount = std::min(oBatch.getFrameCount(),size_t(DATASET_BATCH_STOP_MAX_INDEX+1));
        oBatch.startProcessing();
        while(nCurrIdx<nTotPacketCount) {
            //if(!((nCurrIdx+1)%100))
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << "   [" << sWorkerName << "]" << std::endl;
            const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
            lvDbgAssert(vCurrInput.size()==oBatch.getInputStreamCount());
            lvDbgAssert(vCurrInput.size()==nExpectedAlgoInputCount);
            for(size_t nStreamIdx=0; nStreamIdx<vCurrInput.size(); ++nStreamIdx) {
                lvDbgAssert(oFrameSize==vCurrInput[nStreamIdx].size());
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
                {vAlgos[0]->apply(vCurrInput[0],vCurrFGMasks[0],dCurrLearningRate);}
            #if USING_OPENMP
                #pragma omp section
            #endif //USING_OPENMP
                {vAlgos[1]->apply(vCurrInput[1],vCurrFGMasks[1],dCurrLearningRate);}
            }
            if(lv::getVerbosity()>=2) {
                std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs(2);
                for(size_t nCamIdx=0; nCamIdx<2; ++nCamIdx) {
                    vvDisplayPairs[nCamIdx].resize(3);
                    vvDisplayPairs[nCamIdx][0].first = vCurrInput[nCamIdx];
                    vvDisplayPairs[nCamIdx][0].second = std::to_string(nCurrIdx);
                    vAlgos[nCamIdx]->getBackgroundImage(vvDisplayPairs[nCamIdx][1].first);
                    vCurrFGMasks[nCamIdx].copyTo(vvDisplayPairs[nCamIdx][2].first);
                    if(!vROIs[nCamIdx].empty()) {
                        cv::bitwise_or(vvDisplayPairs[nCamIdx][1].first,UCHAR_MAX/2,vvDisplayPairs[nCamIdx][1].first,vROIs[nCamIdx]==0);
                        cv::bitwise_or(vvDisplayPairs[nCamIdx][2].first,UCHAR_MAX/2,vvDisplayPairs[nCamIdx][2].first,vROIs[nCamIdx]==0);
                    }
                }
                lvAssert(pDisplayHelper);
                pDisplayHelper->display(vvDisplayPairs,oFrameSize);
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
        #else //!PROCESS_PREPROC_BGSEGM
        #if DATASET_EVAL_APPROX_MASKS_ONLY
            for(size_t nCamIdx=0; nCamIdx<nCameraCount; ++nCamIdx) {
                const size_t nInputMaskIdx = nCamIdx*StereoSegmMatcher::InputPackOffset+StereoSegmMatcher::InputPackOffset_Mask;
                const size_t nOutputMaskIdx = nCamIdx*StereoSegmMatcher::OutputPackOffset+StereoSegmMatcher::OutputPackOffset_Mask;
                const size_t nOutputDispIdx = nCamIdx*StereoSegmMatcher::OutputPackOffset+StereoSegmMatcher::OutputPackOffset_Disp;
                vCurrInput[nInputMaskIdx].convertTo(vCurrOutput[nOutputDispIdx],CV_32S); // only to fool output checks
                vCurrInput[nInputMaskIdx].convertTo(vCurrOutput[nOutputMaskIdx],CV_32S,1.0/255);
            }
        #else //!DATASET_EVAL_APPROX_MASKS_ONLY
        #if !DATASET_FORCE_RECALC_FEATURES
            const cv::Mat& oNextFeatsPacket = oBatch.loadFeatures(nCurrIdx);
            if(!oNextFeatsPacket.empty())
                pAlgo->setNextFeatures(oNextFeatsPacket);
            else
        #endif //!DATASET_FORCE_RECALC_FEATURES
            {
                cv::Mat oNewFeatsPacket;
                pAlgo->calcFeatures(lv::convertVectorToArray<nExpectedAlgoInputCount>(vCurrInput),&oNewFeatsPacket);
                oBatch.saveFeatures(nCurrIdx,oNewFeatsPacket);
                pAlgo->setNextFeatures(oNewFeatsPacket);
            }
            pAlgo->apply(vCurrInput,vCurrOutput/*,dDefaultThreshold*/);
        #endif //!DATASET_EVAL_APPROX_MASKS_ONLY
            lvDbgAssert(vCurrOutput.size()==nExpectedAlgoOutputCount);
            using OutputLabelType = StereoSegmMatcher::LabelType;
            for(size_t nOutputArrayIdx=0; nOutputArrayIdx<vCurrOutput.size(); ++nOutputArrayIdx) {
                lvAssert(vCurrOutput[nOutputArrayIdx].type()==lv::MatRawType_<OutputLabelType>());
                lvAssert(vCurrOutput[nOutputArrayIdx].size()==oFrameSize());
                const size_t nRealOutputArrayIdx = nOutputArrayIdx/StereoSegmMatcher::OutputPackOffset;
                if((nOutputArrayIdx%StereoSegmMatcher::OutputPackOffset)==StereoSegmMatcher::OutputPackOffset_Disp) {
                    double dMin,dMax;
                    cv::minMaxIdx(vCurrOutput[nOutputArrayIdx],&dMin,&dMax);
                    lvDbgAssert_(dMin>=0 && dMax<=255,"unexpected min/max disp for 8u mats");
                    vCurrOutput[nOutputArrayIdx].convertTo(vCurrStereoMaps[nRealOutputArrayIdx],CV_8U);
                }
                else if((nOutputArrayIdx%StereoSegmMatcher::OutputPackOffset)==StereoSegmMatcher::OutputPackOffset_Mask)
                    vCurrFGMasks[nRealOutputArrayIdx] = vCurrOutput[nOutputArrayIdx]!=0;
            }
            lvDbgAssert(vCurrFGMasks.size()==oBatch.getOutputStreamCount());
            lvDbgAssert(vCurrStereoMaps.size()==oBatch.getOutputStreamCount());
            if(lv::getVerbosity()>=2) {
                const std::vector<cv::Mat>& vCurrGT = oBatch.getGTArray(nCurrIdx);
                lvAssert(vCurrGT.size()==vCurrFGMasks.size() && vCurrGT.size()==vCurrStereoMaps.size());
                const std::vector<cv::Mat>& vCurrEvalRes = oBatch.getColoredMaskArray(oBatch.isEvaluatingDisparities()?vCurrStereoMaps:vCurrFGMasks,nCurrIdx);
                for(size_t nDisplayRowIdx=0; nDisplayRowIdx<nCameraCount; ++nDisplayRowIdx) {
                    vCurrInput[nDisplayRowIdx*2].copyTo(vvDisplayPairs[nDisplayRowIdx][0].first);
                    vCurrInput[nDisplayRowIdx*2+1].copyTo(vvDisplayPairs[nDisplayRowIdx][1].first);
                #if !DATASET_EVAL_APPROX_MASKS_ONLY
                    pAlgo->getStereoDispMapDisplay(nDisplayRowIdx).copyTo(vvDisplayPairs[nDisplayRowIdx][2].first);
                    pAlgo->getResegmMapDisplay(nDisplayRowIdx).copyTo(vvDisplayPairs[nDisplayRowIdx][3].first);
                #endif //!DATASET_EVAL_APPROX_MASKS_ONLY
                    vCurrEvalRes[nDisplayRowIdx].copyTo(vvDisplayPairs[nDisplayRowIdx][4].first);
                }
                lvAssert(pDisplayHelper);
                pDisplayHelper->display(vvDisplayPairs,cv::Size(320,240));
                const int nKeyPressed = pDisplayHelper->waitKey();
                if(nKeyPressed==(int)'q')
                    break;
            }
            if(oBatch.isEvaluatingDisparities())
                oBatch.push(vCurrStereoMaps,nCurrIdx++);
            else
                oBatch.push(vCurrFGMasks,nCurrIdx++);
        #endif //!PROCESS_PREPROC_BGSEGM
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