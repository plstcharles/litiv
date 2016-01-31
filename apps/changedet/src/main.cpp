
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
// @@@ search for ">( new " everywhere and replace by make_shared/make_unique if possible (2x blocks vs 1x block!)

#include "litiv/datasets.hpp"
#include "litiv/video.hpp"

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define EVALUATE_OUTPUT         0
#define DISPLAY_OUTPUT          1
////////////////////////////////
#define USE_PAWCS               0
#define USE_LOBSTER             1
#define USE_SUBSENSE            0
////////////////////////////////
#define USE_GLSL_IMPL           0
#define USE_CUDA_IMPL           0
#define USE_OPENCL_IMPL         0
////////////////////////////////
#define DATASET_ID              eDataset_VideoSegm_CDnet // comment this line to fall back to custom definition
#define DATASET_OUTPUT_PATH     "results_test" // always relative to the dataset root path
#define DATASET_PRECACHING      1
#define DATASET_SCALE_FACTOR    1.0
////////////////////////////////
#if EVALUATE_OUTPUT // @@@@ dataset should auto detect what is available and what to use
#if HAVE_GLSL
#define USE_GLSL_EVALUATION     1
#endif //HAVE_GLSL
#if HAVE_CUDA
#define USE_CUDA_EVALUATION     1
#endif //HAVE_CUDA
#if HAVE_OPENCL
#define USE_OPENCL_EVALUATION   1
#endif //HAVE_OPENCL
#endif //EVALUATE_OUTPUT
////////////////////////////////
#ifndef USE_GLSL_EVALUATION
#define USE_GLSL_EVALUATION 0
#endif //USE_GLSL_EVALUATION
#ifndef USE_CUDA_EVALUATION
#define USE_CUDA_EVALUATION 0
#endif //USE_CUDA_EVALUATION
#ifndef USE_OPENCL_EVALUATION
#define USE_OPENCL_EVALUATION 0
#endif //USE_OPENCL_EVALUATION
#define USE_GPU_IMPL (USE_GLSL_IMPL||USE_CUDA_IMPL||USE_OPENCL_IMPL)
#define USE_GPU_EVALUATION (USE_GLSL_EVALUATION || USE_CUDA_EVALUATION || USE_OPENCL_EVALUATION)
#define NEED_FG_MASK (DISPLAY_OUTPUT || WRITE_IMG_OUTPUT || (EVALUATE_OUTPUT && (!USE_GPU_EVALUATION || VALIDATE_GPU_EVALUATION)))
#define NEED_LAST_GT_MASK (DISPLAY_OUTPUT || (EVALUATE_OUTPUT && (!USE_GPU_EVALUATION || VALIDATE_GPU_EVALUATION)))
#define NEED_GT_MASK (DISPLAY_OUTPUT || EVALUATE_OUTPUT)
#if (USE_GLSL_IMPL+USE_CUDA_IMPL+USE_OPENCL_IMPL)>1
#error "Must specify a single impl."
#elif (USE_LOBSTER+USE_SUBSENSE+USE_PAWCS)!=1
#error "Must specify a single algorithm."
#endif //USE_...
#ifndef DATASET_ROOT
#error "Dataset root path should have been specified in CMake."
#endif //ndef(DATASET_ROOT)
#ifndef DATASET_ID
#define DATASET_ID eDataset_VideoSegm_Custom
#define DATASET_PARAMS \
    "@@@@",                                                      /* => const std::string& sDatasetName */ \
    "@@@@",                                                      /* => const std::string& sDatasetDirName */ \
    DATASET_OUTPUT_PATH,                                         /* => const std::string& sOutputDirName */ \
    "segm_mask_",                                                /* => const std::string& sOutputNamePrefix */ \
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

void AnalyzeSequence(int nThreadIdx, litiv::IDataHandlerPtr pBatch);
using DatasetType = litiv::Dataset_<litiv::eDatasetType_VideoSegm,litiv::DATASET_ID>;

std::atomic_size_t g_nActiveThreads(0);
const size_t g_nMaxThreads = USE_GPU_IMPL?1:std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():DEFAULT_NB_THREADS;

int main(int, char**) {
    try {
        litiv::IDatasetPtr pDataset = litiv::datasets::create<litiv::eDatasetType_VideoSegm,litiv::DATASET_ID>(DATASET_PARAMS);
        litiv::IDataHandlerPtrQueue vpBatches = pDataset->getSortedBatches();
        const size_t nTotPackets = pDataset->getTotPackets();
        const size_t nTotBatches = vpBatches.size();
        if(nTotBatches==0 || nTotPackets==0)
            lvErrorExt("Could not find any sequences/frames to process for dataset '%s'",pDataset->getName().c_str());
        std::cout << "Parsing complete. [" << pDataset->getBatches().size() << " batch group(s), " << nTotBatches << " sequence(s)]" << std::endl;
        std::cout << "\n[" << CxxUtils::getTimeStamp() << "]\n" << std::endl;
        std::cout << "Executing background subtraction with " << ((g_nMaxThreads>nTotBatches)?nTotBatches:g_nMaxThreads) << " thread(s)..." << std::endl;
        size_t nProcessedBatches = 0;
        while(!vpBatches.empty()) {
            while(g_nActiveThreads>=g_nMaxThreads)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            litiv::IDataHandlerPtr pBatch = vpBatches.top();
            std::cout << "\tProcessing [" << ++nProcessedBatches << "/" << nTotBatches << "] (" << pBatch->getRelativePath() << ", L=" << std::scientific << std::setprecision(2) << pBatch->getExpectedLoad() << ")" << std::endl;
            if(DATASET_PRECACHING)
                pBatch->startPrecaching(EVALUATE_OUTPUT);
            ++g_nActiveThreads;
            std::thread(AnalyzeSequence,(int)nProcessedBatches,pBatch).detach();
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
void AnalyzeSequence(int nThreadIdx, litiv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrFrameIdx = 0;
    size_t nNextFrameIdx = nCurrFrameIdx+1;
    try {
        DatasetType::WorkBatch& oCurrSequence = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        CV_Assert(oCurrSequence.getFrameCount()>1);
        const std::string sCurrSeqName = CxxUtils::clampString(oCurrSequence.getName(),12);
        const size_t nFrameCount = oCurrSequence.getFrameCount();
        const cv::Mat oROI = oCurrSequence.getROI();
        cv::Mat oCurrInputFrame = oCurrSequence.getInputFrame(nCurrFrameIdx).clone();
        cv::Mat oNextInputFrame = oCurrSequence.getInputFrame(nNextFrameIdx);
        CV_Assert(!oCurrInputFrame.empty());
        CV_Assert(oCurrInputFrame.isContinuous());
        glAssert(oCurrInputFrame.channels()==1 || oCurrInputFrame.channels()==4);
        cv::Size oWindowSize = oCurrInputFrame.size();
        GLContext oContext(oWindowSize,std::string("[GPU] ")+oCurrSequence.getRelativePath(),bool(!DISPLAY_OUTPUT));
#if USE_LOBSTER
        std::shared_ptr<BackgroundSubtractorLOBSTER_GLSL> pAlgo = std::make_shared<BackgroundSubtractorLOBSTER_GLSL>();
#elif USE_SUBSENSE
#error "Missing glsl impl." // ... @@@@@
        std::shared_ptr<BackgroundSubtractorSuBSENSE_GLSL> pAlgo = std::make_shared<BackgroundSubtractorSuBSENSE_GLSL>();
#elif USE_PAWCS
#error "Missing glsl impl." // ... @@@@@
        std::shared_ptr<BackgroundSubtractorPAWCS_GLSL> pAlgo = std::make_shared<BackgroundSubtractorPAWCS_GLSL>();
#endif //USE...
        std::shared_ptr<GLImageProcAlgo> pAlgo_glsl = pAlgo;
        std::shared_ptr<IBackgroundSubtractor> pAlgo_base = pAlgo;
        const double dDefaultLearningRate = pAlgo_base->getDefaultLearningRate();
        pAlgo_base->initialize(oCurrInputFrame,oROI);
#if USE_GLSL_EVALUATION
        std::shared_ptr<DatasetUtils::EvaluatorBase::GLEvaluatorBase> pGLSLAlgoEvaluator;
        if(pCurrSequence->m_pEvaluator!=nullptr)
            pGLSLAlgoEvaluator = std::dynamic_pointer_cast<DatasetUtils::EvaluatorBase::GLEvaluatorBase>(pCurrSequence->m_pEvaluator->CreateGLEvaluator(pAlgo_glsl,nFrameCount));
        if(pGLSLAlgoEvaluator==nullptr)
            glError("Segmentation evaluation algorithm has no GLSegmEvaluator interface");
        pGLSLAlgoEvaluator->initialize(oCurrGTMask,oROI.empty()?cv::Mat(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(255)):oROI);
        oWindowSize.width *= pGLSLAlgoEvaluator->m_nSxSDisplayCount;
#else //!USE_GLSL_EVALUATION
        oWindowSize.width *= pAlgo_glsl->m_nSxSDisplayCount; // @@@@ should pAlgo contain a real 'display size' param, and we should query it here?
#endif //!USE_GLSL_EVALUATION
        oContext.setWindowSize(oWindowSize.width,oWindowSize.height);
#if DISPLAY_OUTPUT
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oCurrSequence.getRelativePath(),oCurrSequence.getOutputPath()+"/../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
        pAlgo_glsl->setOutputFetching(true); // @@@ should be toggled in evaluator
#endif //DISPLAY_OUTPUT
        oCurrSequence.startProcessing();
        while(nNextFrameIdx<=nFrameCount) {
            if(!((nCurrFrameIdx+1)%100))
                std::cout << "\t\t" << CxxUtils::clampString(sCurrSeqName,12) << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nFrameCount)) << nCurrFrameIdx+1 << "/" << nFrameCount << "   [GPU]" << std::endl;
            const double dCurrLearningRate = nCurrFrameIdx<=100?1:dDefaultLearningRate;
            pAlgo->apply_async(oNextInputFrame,dCurrLearningRate);
#if USE_GLSL_EVALUATION
            pGLSLAlgoEvaluator->apply_async(oNextGTMask);
            just call evaluator -> apply_next_async();
#endif //USE_GLSL_EVALUATION
#if DISPLAY_OUTPUT
            cv::Mat oLastInputFrame;
            oCurrInputFrame.copyTo(oLastInputFrame);
            oNextInputFrame.copyTo(oCurrInputFrame);
#endif //DISPLAY_OUTPUT
            if(++nNextFrameIdx<nFrameCount)
                oNextInputFrame = oCurrSequence.getInputFrame(nNextFrameIdx);
            glErrorCheck;
            if(oContext.pollEventsAndCheckIfShouldClose())
                break;
#if DISPLAY_OUTPUT
            if(oContext.getKeyPressed('q'))
                break;
            oContext.swapBuffers(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
#if DISPLAY_OUTPUT // @@@@@ add define for display_output_cpu?
            cv::Mat oLastFGMask,oLastBGImg;
            pAlgo->getLatestForegroundMask(oLastFGMask);
            pAlgo->getBackgroundImage(oLastBGImg);
            if(!oROI.empty()) {
                cv::bitwise_or(oLastBGImg,UCHAR_MAX/2,oLastBGImg,oROI==0);
                cv::bitwise_or(oLastFGMask,UCHAR_MAX/2,oLastFGMask,oROI==0);
            }
            pDisplayHelper->display(oLastInputFrame,oLastBGImg,oCurrSequence.getColoredSegmMask(oLastFGMask,nCurrFrameIdx),nCurrFrameIdx);
            const int nKeyPressed = pDisplayHelper->waitKey();
            if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT
#endif //DISPLAY_OUTPUT
            //oCurrSequence.pushSegmMask_async(nCurrFrameIdx++); @@@@@ SOMETHING
        }
        oCurrSequence.stopProcessing();
        const double dTimeElapsed = oCurrSequence.getProcessTime();
        const double dProcessSpeed = (double)nCurrFrameIdx/dTimeElapsed;
        std::cout << "\t\t" << sCurrSeqName << " @ F:" << nCurrFrameIdx << "/" << nFrameCount << "   [T=" << nThreadIdx << "]   (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oCurrSequence.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
    }
    catch(const CxxUtils::Exception& e) {
        std::cout << "\nAnalyzeSequence caught Exception:\n" << e.what();
        const std::string sContextErrMsg = GLContext::getLatestErrorMessage();
        if(!sContextErrMsg.empty())
            std::cout << "\nContext error: " << sContextErrMsg << "\n" << std::endl;
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyzeSequence caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyzeSequence caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyzeSequence caught unhandled exception\n" << std::endl;}
    --g_nActiveThreads;
    try {
        DatasetType::WorkBatch& oCurrSequence = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        if(oCurrSequence.isProcessing())
            oCurrSequence.stopProcessing();
    } catch(...) {
        std::cout << "\nAnalyzeSequence caught unhandled exception while attempting to stop batch processing.\n" << std::endl;
        throw;
    }
}
#elif (HAVE_CUDA && USE_CUDA_IMPL)
static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
void AnalyzeSequence(int nThreadIdx, litiv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrFrameIdx = 0;
    try {
        DatasetType::WorkBatch& oCurrSequence = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        CV_Assert(oCurrSequence.getFrameCount()>1);
        const std::string sCurrSeqName = CxxUtils::clampString(oCurrSequence.getName(),12);
        const size_t nFrameCount = oCurrSequence.getFrameCount();
        const cv::Mat oROI = oCurrSequence.getROI();
        cv::Mat oCurrInputFrame = oCurrSequence.getInputFrame(nCurrFrameIdx).clone();
        CV_Assert(!oCurrInputFrame.empty());
        CV_Assert(oCurrInputFrame.isContinuous());
        cv::Mat oCurrFGMask(oCurrSequence.getFrameSize(),CV_8UC1,cv::Scalar_<uchar>(0));
        std::shared_ptr<IBackgroundSubtractor> pAlgo;
#if USE_LOBSTER
        pAlgo = std::make_shared<BackgroundSubtractorLOBSTER>();
#elif USE_SUBSENSE
        pAlgo = std::make_shared<BackgroundSubtractorSuBSENSE>();
#elif USE_PAWCS
        pAlgo = std::make_shared<BackgroundSubtractorPAWCS>();
#endif //USE_...
        const double dDefaultLearningRate = pAlgo->getDefaultLearningRate();
        pAlgo->initialize(oCurrInputFrame,oROI);
#if DISPLAY_OUTPUT
        cv::DisplayHelperPtr pDisplayHelper = cv::DisplayHelper::create(oCurrSequence.getRelativePath(),oCurrSequence.getOutputPath()+"/../");
        pAlgo->m_pDisplayHelper = pDisplayHelper;
#endif //DISPLAY_OUTPUT
        oCurrSequence.startProcessing();
        while(nCurrFrameIdx<nFrameCount) {
            if(!((nCurrFrameIdx+1)%100) && nCurrFrameIdx<nFrameCount)
                std::cout << "\t\t" << sCurrSeqName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nFrameCount)) << nCurrFrameIdx+1 << "/" << nFrameCount << "   [T=" << nThreadIdx << "]" << std::endl;
            const double dCurrLearningRate = nCurrFrameIdx<=100?1:dDefaultLearningRate;
            oCurrInputFrame = oCurrSequence.getInputFrame(nCurrFrameIdx);
            pAlgo->apply(oCurrInputFrame,oCurrFGMask,dCurrLearningRate);
#if DISPLAY_OUTPUT
            cv::Mat oCurrBGImg;
            pAlgo->getBackgroundImage(oCurrBGImg);
            if(!oROI.empty()) {
                cv::bitwise_or(oCurrBGImg,UCHAR_MAX/2,oCurrBGImg,oROI==0);
                cv::bitwise_or(oCurrFGMask,UCHAR_MAX/2,oCurrFGMask,oROI==0);
            }
            pDisplayHelper->display(oCurrInputFrame,oCurrBGImg,oCurrSequence.getColoredSegmMask(oCurrFGMask,nCurrFrameIdx),nCurrFrameIdx);
            const int nKeyPressed = pDisplayHelper->waitKey();
            if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT
            oCurrSequence.pushSegmMask(oCurrFGMask,nCurrFrameIdx++);
        }
        oCurrSequence.stopProcessing();
        const double dTimeElapsed = oCurrSequence.getProcessTime();
        const double dProcessSpeed = (double)nCurrFrameIdx/dTimeElapsed;
        std::cout << "\t\t" << sCurrSeqName << " @ F:" << nCurrFrameIdx << "/" << nFrameCount << "   [T=" << nThreadIdx << "]   (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oCurrSequence.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyzeSequence caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyzeSequence caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyzeSequence caught unhandled exception\n" << std::endl;}
    --g_nActiveThreads;
    try {
        DatasetType::WorkBatch& oCurrSequence = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        if(oCurrSequence.isProcessing())
            oCurrSequence.stopProcessing();
    } catch(...) {
        std::cout << "\nAnalyzeSequence caught unhandled exception while attempting to stop batch processing.\n" << std::endl;
        throw;
    }
}
#endif //!USE_GPU_IMPL
