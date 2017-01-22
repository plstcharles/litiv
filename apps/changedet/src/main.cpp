
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

// @@@ imgproc gpu algo does not support mipmapping binding yet
// @@@ test compute shader group size vs shared mem usage
// @@@ support non-integer textures top level (alg)? need to replace all ui-stores by float-stores, rest is ok

#include "litiv/datasets.hpp"
#include "litiv/video.hpp"

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define EVALUATE_OUTPUT         0
#define DISPLAY_OUTPUT          1
////////////////////////////////
#define USE_PAWCS               1
#define USE_LOBSTER             0
#define USE_SUBSENSE            0
////////////////////////////////
#define USE_GLSL_IMPL           0
#define USE_CUDA_IMPL           0
#define USE_OPENCL_IMPL         0
////////////////////////////////
#define DATASET_ID              Dataset_CDnet // comment this line to fall back to custom dataset definition
#define DATASET_OUTPUT_PATH     "results_test" // will be created in the app's working directory if using a custom dataset
#define DATASET_PRECACHING      1
#define DATASET_SCALE_FACTOR    1.0
#define DATASET_WORKTHREADS     1
////////////////////////////////
#define USE_GPU_IMPL (USE_GLSL_IMPL||USE_CUDA_IMPL||USE_OPENCL_IMPL)
#if (USE_GLSL_IMPL+USE_CUDA_IMPL+USE_OPENCL_IMPL)>1
#error "Must specify a single impl."
#elif (USE_LOBSTER+USE_SUBSENSE+USE_PAWCS)!=1
#error "Must specify a single algorithm."
#endif //USE_...
#ifndef DATASET_ID
#define DATASET_ID Dataset_Custom
#define DATASET_PARAMS \
    "@@@@",                                                      /* => const std::string& sDatasetName */ \
    "@@@@",                                                      /* => const std::string& sDatasetDirPath */ \
    DATASET_OUTPUT_PATH,                                         /* => const std::string& sOutputDirPath */ \
    "segm_mask_",                                                /* => const std::string& sOutputNamePrefix */ \
    ".png",                                                      /* => const std::string& sOutputNameSuffix */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsWorkBatchDirs */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsSkippedDirTokens */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsGrayscaleDirTokens */ \
    bool(WRITE_IMG_OUTPUT),                                      /* => bool bSaveOutput */ \
    bool(EVALUATE_OUTPUT),                                       /* => bool bUseEvaluator */ \
    bool(USE_GPU_IMPL),                                          /* => bool bForce4ByteDataAlign */ \
    DATASET_SCALE_FACTOR                                         /* => double dScaleFactor */
#else //defined(DATASET_ID)
#define DATASET_PARAMS \
    DATASET_OUTPUT_PATH,                                         /* => const std::string& sOutputDirName */ \
    bool(WRITE_IMG_OUTPUT),                                      /* => bool bSaveOutput */ \
    bool(EVALUATE_OUTPUT),                                       /* => bool bUseEvaluator */ \
    bool(USE_GPU_IMPL),                                          /* => bool bForce4ByteDataAlign */ \
    DATASET_SCALE_FACTOR                                         /* => double dScaleFactor */
#endif //defined(DATASET_ID)

void Analyze(std::string sWorkerName, lv::IDataHandlerPtr pBatch);
#if USE_GLSL_IMPL
constexpr lv::ParallelAlgoType eImplTypeEnum = lv::GLSL;
#else // USE_..._IMPL
constexpr lv::ParallelAlgoType eImplTypeEnum = lv::NonParallel;
#endif // USE_..._IMPL
using DatasetType = lv::Dataset_<lv::DatasetTask_Segm,lv::DATASET_ID,eImplTypeEnum>;
#if USE_LOBSTER
using BackgroundSubtractorType = BackgroundSubtractorLOBSTER_<eImplTypeEnum>;
#elif USE_SUBSENSE
using BackgroundSubtractorType = BackgroundSubtractorSuBSENSE_<eImplTypeEnum>;
#elif USE_PAWCS
using BackgroundSubtractorType = BackgroundSubtractorPAWCS_<eImplTypeEnum>;
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
        std::cout << "Executing algorithm with " << (USE_GPU_IMPL?1:DATASET_WORKTHREADS) << " thread(s)..." << std::endl;
        lv::WorkerPool<(USE_GPU_IMPL?1:DATASET_WORKTHREADS)> oPool;
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

#if (HAVE_GLSL && USE_GLSL_IMPL)
void Analyze(std::string sWorkerName, lv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        lvAssert(oBatch.getInputPacketType()==lv::ImagePacket && oBatch.getOutputPacketType()==lv::ImagePacket);
        lvAssert(oBatch.getFrameCount()>1);
        if(DATASET_PRECACHING)
            oBatch.startPrecaching(EVALUATE_OUTPUT);
        const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
        std::cout << "\t\t" << sCurrBatchName << " @ init [" << sWorkerName << "]" << std::endl;
        const size_t nTotPacketCount = oBatch.getFrameCount();
        lv::gl::Context oContext(oBatch.getFrameSize(),oBatch.getName()+" [GPU]",DISPLAY_OUTPUT==0);
        std::shared_ptr<IBackgroundSubtractor_<lv::GLSL>> pAlgo = std::make_shared<BackgroundSubtractorType>();
#if DISPLAY_OUTPUT>1
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
#endif //DISPLAY_OUTPUT>1
        const double dDefaultLearningRate = pAlgo->getDefaultLearningRate();
        oBatch.initialize_gl(pAlgo);
        oContext.setWindowSize(oBatch.getIdealGLWindowSize());
        oBatch.startProcessing();
        size_t nNextIdx = 1;
        while(nNextIdx<=nTotPacketCount) {
            if(!((nNextIdx+1)%100))
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nNextIdx+1 << "/" << nTotPacketCount << " [" << sWorkerName << "]" << std::endl;
            const double dCurrLearningRate = nNextIdx<=100?1:dDefaultLearningRate;
            oBatch.apply_gl(pAlgo,nNextIdx++,false,dCurrLearningRate);
            //pGLSLAlgoEvaluator->apply_gl(oNextGTMask);
            glErrorCheck;
            if(oContext.pollEventsAndCheckIfShouldClose())
                break;
#if DISPLAY_OUTPUT>0
            if(oContext.getKeyPressed('q'))
                break;
            oContext.swapBuffers(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
#if DISPLAY_OUTPUT>1
            const int nKeyPressed = pDisplayHelper->waitKey();
            if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT>1
#endif //DISPLAY_OUTPUT>0
        }
        oBatch.stopProcessing();
        const double dTimeElapsed = oBatch.getFinalProcessTime();
        const double dProcessSpeed = (double)(nNextIdx-1)/dTimeElapsed;
        std::cout << "\t\t" << sCurrBatchName << " @ end [" << sWorkerName << "] (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oBatch.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
    }
    catch(const lv::Exception& e) {
        std::cout << "\nAnalyze caught Exception:\n" << e.what();
        const std::string sContextErrMsg = lv::gl::Context::getLatestErrorMessage();
        if(!sContextErrMsg.empty())
            std::cout << "\nContext error: " << sContextErrMsg << "\n" << std::endl;
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
#elif (HAVE_CUDA && USE_CUDA_IMPL)
static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
void Analyze(std::string sWorkerName, lv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        lvAssert(oBatch.getInputPacketType()==lv::ImagePacket && oBatch.getOutputPacketType()==lv::ImagePacket);
        lvAssert(oBatch.getFrameCount()>1);
        if(DATASET_PRECACHING)
            oBatch.startPrecaching(EVALUATE_OUTPUT);
        const std::string sCurrBatchName = lv::clampString(oBatch.getName(),12);
        std::cout << "\t\t" << sCurrBatchName << " @ init [" << sWorkerName << "]" << std::endl;
        const size_t nTotPacketCount = oBatch.getFrameCount();
        const cv::Mat& oROI = oBatch.getFrameROI();
        size_t nCurrIdx = 0;
        cv::Mat oCurrInput = oBatch.getInput(nCurrIdx).clone();
        lvAssert(!oCurrInput.empty() && oCurrInput.isContinuous());
        cv::Mat oCurrFGMask(oBatch.getFrameSize(),CV_8UC1,cv::Scalar_<uchar>(0));
        std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorType>();
        const double dDefaultLearningRate = pAlgo->getDefaultLearningRate();
        pAlgo->initialize(oCurrInput,oROI);
#if DISPLAY_OUTPUT>0
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
#endif //DISPLAY_OUTPUT>0
        oBatch.startProcessing();
        while(nCurrIdx<nTotPacketCount) {
            if(!((nCurrIdx+1)%100))
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(lv::digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << " [" << sWorkerName << "]" << std::endl;
            const double dCurrLearningRate = nCurrIdx<=100?1:dDefaultLearningRate;
            oCurrInput = oBatch.getInput(nCurrIdx);
            pAlgo->apply(oCurrInput,oCurrFGMask,dCurrLearningRate);
#if DISPLAY_OUTPUT>0
            cv::Mat oCurrBGImg;
            pAlgo->getBackgroundImage(oCurrBGImg);
            if(!oROI.empty()) {
                cv::bitwise_or(oCurrBGImg,UCHAR_MAX/2,oCurrBGImg,oROI==0);
                cv::bitwise_or(oCurrFGMask,UCHAR_MAX/2,oCurrFGMask,oROI==0);
            }
            pDisplayHelper->display(oCurrInput,oCurrBGImg,oBatch.getColoredMask(oCurrFGMask,nCurrIdx),nCurrIdx);
            const int nKeyPressed = pDisplayHelper->waitKey();
            if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT>0
            oBatch.push(oCurrFGMask,nCurrIdx++);
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
#endif //(!USE_GPU_IMPL)
