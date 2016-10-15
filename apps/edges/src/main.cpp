
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

#include "litiv/datasets.hpp"
#include "litiv/imgproc.hpp"

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define EVALUATE_OUTPUT         0
#define DISPLAY_OUTPUT          1
////////////////////////////////
#define USE_CANNY               0
#define USE_LBSP                1
////////////////////////////////
#define FULL_THRESH_ANALYSIS    1
////////////////////////////////
#define DATASET_ID              Dataset_BSDS500 // comment this line to fall back to custom dataset definition
#define DATASET_OUTPUT_PATH     "results_test" // will be created in the app's working directory if using a custom dataset
#define DATASET_PRECACHING      1
#define DATASET_SCALE_FACTOR    1.0
#define DATASET_WORKTHREADS     1
////////////////////////////////
#if (USE_CANNY+USE_LBSP)!=1
#error "Must specify a single algorithm."
#endif //USE_...
#ifndef DATASET_ID
#define DATASET_ID Dataset_Custom
#define DATASET_PARAMS \
    "@@@@",                                                      /* => const std::string& sDatasetName */ \
    "@@@@",                                                      /* => const std::string& sDatasetDirPath */ \
    DATASET_OUTPUT_PATH,                                         /* => const std::string& sOutputDirPath */ \
    "edge_mask_",                                                /* => const std::string& sOutputNamePrefix */ \
    ".png",                                                      /* => const std::string& sOutputNameSuffix */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsWorkBatchDirs */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsSkippedDirTokens */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsGrayscaleDirTokens */ \
    bool(WRITE_IMG_OUTPUT),                                      /* => bool bSaveOutput */ \
    bool(EVALUATE_OUTPUT),                                       /* => bool bUseEvaluator */ \
    false,                                                       /* => bool bForce4ByteDataAlign */ \
    DATASET_SCALE_FACTOR                                         /* => double dScaleFactor */
#else //defined(DATASET_ID)
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                                         /* => const std::string& sOutputDirName */ \
    bool(WRITE_IMG_OUTPUT),                                      /* => bool bSaveOutput */ \
    bool(EVALUATE_OUTPUT),                                       /* => bool bUseEvaluator */ \
    false,                                                       /* => bool bForce4ByteDataAlign */ \
    DATASET_SCALE_FACTOR                                         /* => double dScaleFactor */
#endif //defined(DATASET_ID)

void Analyze(std::string sWorkerName, lv::IDataHandlerPtr pBatch);
using DatasetType = lv::Dataset_<lv::DatasetTask_EdgDet,lv::DATASET_ID,lv::NonParallel>;
#if USE_CANNY
using EdgeDetectorType = EdgeDetectorCanny;
#elif USE_LBSP
using EdgeDetectorType = EdgeDetectorLBSP;
#endif //USE_...

int main(int, char**) {
    try {
        DatasetType::Ptr pDataset = DatasetType::create(DATASET_PARAMS);
        lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false);
        const size_t nTotPackets = pDataset->getInputCount();
        const size_t nTotBatches = vpBatches.size();
        if(nTotBatches==0 || nTotPackets==0)
            lvError_("Could not parse any data for dataset '%s'",pDataset->getName().c_str());
        std::cout << "Parsing complete. [" << nTotBatches << " batch(es)]" << std::endl;
        std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
        std::cout << "Executing algorithm with " << DATASET_WORKTHREADS << " thread(s)..." << std::endl;
        lv::WorkerPool<DATASET_WORKTHREADS> oPool;
        std::vector<std::future<void>> vTaskResults;
        for(lv::IDataHandlerPtr pBatch : vpBatches)
            vTaskResults.push_back(oPool.queueTask(Analyze,std::to_string(nTotBatches-vpBatches.size()+1)+"/"+std::to_string(nTotBatches),pBatch));
        for(std::future<void>& oTaskRes : vTaskResults)
            oTaskRes.get();
        pDataset->writeEvalReport();
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return -1;}
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
        lvAssert(oBatch.getInputPacketType()==lv::ImagePacket && oBatch.getOutputPacketType()==lv::ImagePacket);
        lvAssert(oBatch.getImageCount()>=1);
        lvAssert(oBatch.isInputConstantSize());
        if(DATASET_PRECACHING)
            oBatch.startPrecaching(EVALUATE_OUTPUT);
        const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
        std::cout << "\t\t" << sCurrBatchName << " @ init [" << sWorkerName << "]" << std::endl;
        const size_t nTotPacketCount = oBatch.getImageCount();
        size_t nCurrIdx = 0;
        cv::Mat oCurrInput = oBatch.getInput(nCurrIdx).clone();
        lvAssert(!oCurrInput.empty() && oCurrInput.isContinuous());
        cv::Mat oCurrEdgeMask(oBatch.getInputMaxSize(),CV_8UC1,cv::Scalar_<uchar>(0));
        std::shared_ptr<IEdgeDetector> pAlgo = std::make_shared<EdgeDetectorType>();
#if !FULL_THRESH_ANALYSIS
        const double dDefaultThreshold = pAlgo->getDefaultThreshold();
#endif //(!FULL_THRESH_ANALYSIS)
#if DISPLAY_OUTPUT>0
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
#endif //DISPLAY_OUTPUT>0
        oBatch.startProcessing();
        while(nCurrIdx<nTotPacketCount) {
            //if(!((nCurrIdx+1)%100))
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << "   [" << sWorkerName << "]" << std::endl;
            oCurrInput = oBatch.getInput(nCurrIdx);
#if FULL_THRESH_ANALYSIS
            pAlgo->apply(oCurrInput,oCurrEdgeMask);
#else //(!FULL_THRESH_ANALYSIS)
            pAlgo->apply_threshold(oCurrInput,oCurrEdgeMask,dDefaultThreshold);
#endif //(!FULL_THRESH_ANALYSIS)
#if DISPLAY_OUTPUT>0
            pDisplayHelper->display(oCurrInput,oCurrEdgeMask,oBatch.getColoredMask(oCurrEdgeMask,nCurrIdx),nCurrIdx);
            const int nKeyPressed = pDisplayHelper->waitKey();
            if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT>0
            oBatch.push(oCurrEdgeMask,nCurrIdx++);
        }
        oBatch.stopProcessing();
        const double dTimeElapsed = oBatch.getFinalProcessTime();
        const double dProcessSpeed = (double)nCurrIdx/dTimeElapsed;
        std::cout << "\t\t" << sCurrBatchName << " @ end [" << sWorkerName << "] (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oBatch.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyze caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
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
