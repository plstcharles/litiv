
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
#define USE_GLSL_IMPL           0
#define USE_CUDA_IMPL           0
#define USE_OPENCL_IMPL         0
////////////////////////////////
#define DATASET_ID              eDataset_BSDS500 // comment this line to fall back to custom dataset definition
#define DATASET_OUTPUT_PATH     "results_test" // will be created in the app's working directory if using a custom dataset
#define DATASET_PRECACHING      1
#define DATASET_SCALE_FACTOR    1.0
////////////////////////////////
#if (DEBUG_OUTPUT && !DISPLAY_OUTPUT)
#undef DISPLAY_OUTPUT
#define DISPLAY_OUTPUT 1
#endif //(DEBUG_OUTPUT && !DISPLAY_OUTPUT)
#if (DEBUG_OUTPUT && FULL_THRESH_ANALYSIS)
#error "Cannot enable debug output while using all threshold values."
#endif //(DEBUG_OUTPUT && FULL_THRESH_ANALYSIS)
#define USE_GPU_IMPL (USE_GLSL_IMPL||USE_CUDA_IMPL||USE_OPENCL_IMPL)
#if (USE_GLSL_IMPL+USE_CUDA_IMPL+USE_OPENCL_IMPL)>1
#error "Must specify a single impl."
#elif (USE_CANNY+USE_LBSP)!=1
#error "Must specify a single algorithm."
#endif //USE_...
#ifndef DATASET_ROOT
#error "Dataset root path should have been specified in CMake."
#endif //ndef(DATASET_ROOT)
#ifndef DATASET_ID
#define DATASET_ID eDataset_Custom
#define DATASET_PARAMS \
    "@@@@",                                                      /* => const std::string& sDatasetName */ \
    "@@@@",                                                      /* => const std::string& sDatasetDirPath */ \
    DATASET_OUTPUT_PATH,                                         /* => const std::string& sOutputDirPath */ \
    "edge_mask_",                                                /* => const std::string& sOutputNamePrefix */ \
    ".png",                                                      /* => const std::string& sOutputNameSuffix */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsWorkBatchDirs */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsSkippedDirTokens */ \
    std::vector<std::string>{"@@@","@@@","@@@","..."},           /* => const std::vector<std::string>& vsGrayscaleDirTokens */ \
    0,                                                           /* => size_t nOutputIdxOffset */ \
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
void Analyze(int nThreadIdx, litiv::IDataHandlerPtr pBatch);
#if USE_GLSL_IMPL
static_assert(false,"Missing impl");
constexpr ParallelUtils::eParallelAlgoType eImplTypeEnum = ParallelUtils::eGLSL;
#else // USE_..._IMPL
constexpr ParallelUtils::eParallelAlgoType eImplTypeEnum = ParallelUtils::eNonParallel;
#endif // USE_..._IMPL
using DatasetType = litiv::Dataset_<litiv::eDatasetTask_EdgDet,litiv::DATASET_ID,eImplTypeEnum>;
#if USE_CANNY
using EdgeDetectorType = EdgeDetectorCanny;
#elif USE_LBSP
using EdgeDetectorType = EdgeDetectorLBSP;
#endif //USE_...
std::atomic_size_t g_nActiveThreads(0);
const size_t g_nMaxThreads = USE_GPU_IMPL?1:std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():DEFAULT_NB_THREADS;

int main(int, char**) {
    try {
        litiv::IDatasetPtr pDataset = litiv::datasets::create<litiv::eDatasetTask_EdgDet,litiv::DATASET_ID,eImplTypeEnum>(DATASET_PARAMS);
        litiv::IDataHandlerPtrQueue vpBatches = pDataset->getSortedBatches(false);
        const size_t nTotPackets = pDataset->getTotPackets();
        const size_t nTotBatches = vpBatches.size();
        if(nTotBatches==0 || nTotPackets==0)
            lvErrorExt("Could not parse any data for dataset '%s'",pDataset->getName().c_str());
        std::cout << "Parsing complete. [" << nTotBatches << " batch(es)]" << std::endl;
        std::cout << "\n[" << CxxUtils::getTimeStamp() << "]\n" << std::endl;
        std::cout << "Executing edge detection with " << ((g_nMaxThreads>nTotBatches)?nTotBatches:g_nMaxThreads) << " thread(s)..." << std::endl;
        size_t nProcessedBatches = 0;
        while(!vpBatches.empty()) {
            while(g_nActiveThreads>=g_nMaxThreads)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            litiv::IDataHandlerPtr pBatch = vpBatches.top();
            std::cout << "\tProcessing [" << ++nProcessedBatches << "/" << nTotBatches << "] (" << pBatch->getRelativePath() << ", L=" << std::scientific << std::setprecision(2) << pBatch->getExpectedLoad() << ")" << std::endl;
            if(DATASET_PRECACHING)
                dynamic_cast<DatasetType::WorkBatch&>(*pBatch).startAsyncPrecaching(EVALUATE_OUTPUT);
            ++g_nActiveThreads;
            std::thread(Analyze,(int)nProcessedBatches,pBatch).detach();
            vpBatches.pop();
        }
        while(g_nActiveThreads>0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        if(pDataset->getProcessedPacketsCountPromise()==nTotPackets)
            pDataset->writeEvalReport();
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    std::cout << "\n[" << CxxUtils::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}

#if (HAVE_GLSL && USE_GLSL_IMPL)
static_assert(false,"Missing impl");
void Analyze(int nThreadIdx, litiv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrImageIdx = 0;
    size_t nNextImageIdx = nCurrImageIdx+1;
    bool bGPUContextInitialized = false;
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        CV_Assert(oBatch.getImageCount()>1);
        const std::string sCurrBatchName = CxxUtils::clampString(oBatch.getName(),12);
        const size_t nTotPacketCount = oBatch.getImageCount();
        GLContext oContext(oBatch.getMaxImageSize(),std::string("[GPU] ")+oBatch.getRelativePath(),DISPLAY_OUTPUT==0);
        std::shared_ptr<IEdgeDetector_<ParallelUtils::eGLSL>> pAlgo = std::make_shared<EdgeDetectorType>();
#if DISPLAY_OUTPUT>1
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
#endif //DISPLAY_OUTPUT>1
        const double dDefaultLearningRate = pAlgo->getDefaultLearningRate();
        oBatch.initialize_gl(pAlgo);
        oContext.setWindowSize(oBatch.getIdealGLWindowSize());
        oBatch.startProcessing();
        size_t nNextIdx = 1;
        while(nNextIdx<=nTotPacketCount) {
            if(!(nNextIdx%100))
                std::cout << "\t\t" << CxxUtils::clampString(sCurrBatchName,12) << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nTotPacketCount)) << nNextIdx << "/" << nTotPacketCount << "   [GPU]" << std::endl;
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
        const double dTimeElapsed = oBatch.getProcessTime();
        const double dProcessSpeed = (double)(nNextIdx-1)/dTimeElapsed;
        std::cout << "\t\t" << sCurrBatchName << " @ F:" << (nNextIdx-1) << "/" << nTotPacketCount << "   [T=" << nThreadIdx << "]   (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oBatch.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
    }
    catch(const CxxUtils::Exception& e) {
        std::cout << "\nAnalyze caught Exception:\n" << e.what();
        const std::string sContextErrMsg = GLContext::getLatestErrorMessage();
        if(!sContextErrMsg.empty())
            std::cout << "\nContext error: " << sContextErrMsg << "\n" << std::endl;
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyze caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyze caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyze caught unhandled exception\n" << std::endl;}
    --g_nActiveThreads;
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        if(oBatch.isProcessing())
            oBatch.stopProcessing();
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
void Analyze(int nThreadIdx, litiv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrIdx = 0;
    try {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        CV_Assert(oBatch.getImageCount()>1);
        const std::string sCurrBatchName = CxxUtils::clampString(oBatch.getName(),12);
        const size_t nTotPacketCount = oBatch.getImageCount();
        cv::Mat oCurrInput = oBatch.getInput(nCurrIdx).clone();
        CV_Assert(!oCurrInput.empty());
        CV_Assert(oCurrInput.isContinuous());
        CV_Assert(oBatch.isInputConstantSize() && oBatch.getInputPacketType()==litiv::eImagePacket);
        cv::Mat oCurrEdgeMask(oBatch.getInputMaxSize(),CV_8UC1,cv::Scalar_<uchar>(0));
        std::shared_ptr<IEdgeDetector> pAlgo = std::make_shared<EdgeDetectorType>();
#if !FULL_THRESH_ANALYSIS
        const double dDefaultThreshold = pAlgo->getDefaultThreshold();
#endif //(!FULL_THRESH_ANALYSIS)
#if DISPLAY_OUTPUT>0
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oBatch.getName(),oBatch.getOutputPath()+"/../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
#endif //DISPLAY_OUTPUT>0
        oBatch.startProcessing();
        while(nCurrIdx<nTotPacketCount) {
            //if(!((nCurrIdx+1)%100) && nCurrIdx<nTotPacketCount)
                std::cout << "\t\t" << sCurrBatchName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nTotPacketCount)) << nCurrIdx+1 << "/" << nTotPacketCount << "   [T=" << nThreadIdx << "]" << std::endl;
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
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        if(oBatch.isProcessing())
            oBatch.stopProcessing();
    } catch(...) {
        std::cout << "\nAnalyze caught unhandled exception while attempting to stop batch processing.\n" << std::endl;
        throw;
    }
}
#endif //(!USE_GPU_IMPL)
