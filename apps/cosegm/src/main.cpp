
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

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define EVALUATE_OUTPUT         0
#define DISPLAY_OUTPUT          1
////////////////////////////////
#define DATASET_ID              Dataset_VAPtrimod2016
#define DATASET_OUTPUT_PATH     "results_test"
#define DATASET_PRECACHING      1
#define DATASET_SCALE_FACTOR    1.0
////////////////////////////////
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                                         /* => const std::string& sOutputDirName */ \
    bool(WRITE_IMG_OUTPUT),                                      /* => bool bSaveOutput */ \
    bool(EVALUATE_OUTPUT),                                       /* => bool bUseEvaluator */ \
    false,                                                       /* => bool bForce4ByteDataAlign */ \
    DATASET_SCALE_FACTOR                                         /* => double dScaleFactor */

void Analyze(int nThreadIdx, lv::IDataHandlerPtr pBatch);
using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::DATASET_ID,lv::NonParallel>;
std::atomic_size_t g_nActiveThreads(0);
const size_t g_nMaxThreads = 1;

int main(int, char**) {
    try {
        lv::IDatasetPtr pDataset = DatasetType::create(DATASET_PARAMS);
        lv::IDataHandlerPtrQueue vpBatches = pDataset->getSortedBatches(false);
        const size_t nTotPackets = pDataset->getInputCount();
        const size_t nTotBatches = vpBatches.size();
        if(nTotBatches==0 || nTotPackets==0)
            lvError_("Could not parse any data for dataset '%s'",pDataset->getName().c_str());
        std::cout << "Parsing complete. [" << nTotBatches << " batch(es)]" << std::endl;
        std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
        std::cout << "Executing edge detection with " << ((g_nMaxThreads>nTotBatches)?nTotBatches:g_nMaxThreads) << " thread(s)..." << std::endl;
        size_t nProcessedBatches = 0;
        while(!vpBatches.empty()) {
            while(g_nActiveThreads>=g_nMaxThreads)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            lv::IDataHandlerPtr pBatch = vpBatches.top();
            std::cout << "\tProcessing [" << ++nProcessedBatches << "/" << nTotBatches << "] (" << pBatch->getRelativePath() << ", L=" << std::scientific << std::setprecision(2) << pBatch->getExpectedLoad() << ")" << std::endl;
            if(DATASET_PRECACHING)
                pBatch->startPrecaching(EVALUATE_OUTPUT);
            std::this_thread::sleep_for(std::chrono::seconds(3));
            ++g_nActiveThreads;
            std::thread(Analyze,(int)nProcessedBatches,pBatch).detach();
            vpBatches.pop();
        }
        while(g_nActiveThreads>0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if(pDataset->getProcessedOutputCountPromise()==nTotPackets)
            pDataset->writeEvalReport();
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}

void Analyze(int nThreadIdx, lv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrIdx = 0;
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        lvAssert(oBatch.getInputPacketType()==lv::ImageArrayPacket && oBatch.getOutputPacketType()==lv::ImageArrayPacket);
        lvAssert(oBatch.getOutputStreamCount()==oBatch.getInputStreamCount()); // curr only support 1:1 cosegm
        lvAssert(oBatch.getFrameCount()>1);
        const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
        const size_t nTotPacketCount = oBatch.getFrameCount();
        std::vector<cv::Mat> vInitInput = oBatch.getInputArray(nCurrIdx); // mat content becomes invalid on next getInput call
        lvAssert(!vInitInput.empty() && vInitInput.size()==oBatch.getInputStreamCount());
        std::vector<cv::Mat> vCurrFGMasks(vInitInput.size());
        //std::shared_ptr<IEdgeDetector> pAlgo = std::make_shared<CosegmenterType>();
        // init & defaults...
#if DISPLAY_OUTPUT>0
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
        //pAlgo->m_pDisplayHelper = pDisplayHelper;
        std::vector<std::vector<std::pair<cv::Mat,std::string>>> vvDisplayPairs;
        for(size_t nDisplayRowIdx=0; nDisplayRowIdx<vInitInput.size(); ++nDisplayRowIdx) {
            std::vector<std::pair<cv::Mat,std::string>> vRow(3);
            vRow[0] = std::make_pair(vInitInput[nDisplayRowIdx].clone(),oBatch.getInputStreamName(nDisplayRowIdx));
            vRow[1] = std::make_pair(cv::Mat(vInitInput[nDisplayRowIdx].size(),CV_8UC1,cv::Scalar_<uchar>(128)),"DEBUG");
            vRow[2] = std::make_pair(cv::Mat(vInitInput[nDisplayRowIdx].size(),CV_8UC1,cv::Scalar_<uchar>(128)),"OUTPUT");
            vvDisplayPairs.push_back(vRow);
        }
#endif //DISPLAY_OUTPUT>0
        oBatch.startProcessing();
        while(nCurrIdx<nTotPacketCount) {
            if(!((nCurrIdx+1)%100) && nCurrIdx<nTotPacketCount)
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << "   [T=" << nThreadIdx << "]" << std::endl;
            const std::vector<cv::Mat>& vCurrInput = oBatch.getInputArray(nCurrIdx);
            lvAssert(vCurrInput.size()==vInitInput.size());
            //pAlgo->apply(vCurrInput,vCurrFGMasks,dDefaultThreshold);
            lvAssert(vCurrInput.size()==vCurrFGMasks.size());
#if DISPLAY_OUTPUT>0
            for(size_t nDisplayRowIdx=0; nDisplayRowIdx<vCurrInput.size(); ++nDisplayRowIdx) {
                vvDisplayPairs[nDisplayRowIdx][0].first = vCurrInput[nDisplayRowIdx];
                //vvDisplayPairs[nDisplayRowIdx][1].first = ... output;
                //vvDisplayPairs[nDisplayRowIdx][1].first = ... colored output;
            }
            pDisplayHelper->display(vvDisplayPairs,cv::Size(320,240));
            const int nKeyPressed = pDisplayHelper->waitKey();
            if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT>0
            oBatch.push(vCurrFGMasks,nCurrIdx++);
        }
        oBatch.stopProcessing();
        const double dTimeElapsed = oBatch.getProcessTime();
        const double dProcessSpeed = (double)nCurrIdx/dTimeElapsed;
        std::cout << "\t\t" << sCurrBatchName << " @ F:" << nCurrIdx << "/" << nTotPacketCount << "   [T=" << nThreadIdx << "]   (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oBatch.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyze caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyze caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyze caught unhandled exception\n" << std::endl;}
    --g_nActiveThreads;
    try {
        if(pBatch->isProcessing())
            dynamic_cast<DatasetType::WorkBatch&>(*pBatch).stopProcessing();
    } catch(...) {
        std::cout << "\nAnalyze caught unhandled exception while attempting to stop batch processing.\n" << std::endl;
        throw;
    }
}
