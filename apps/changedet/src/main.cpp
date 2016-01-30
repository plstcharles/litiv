
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
// @@@ add opencv hardware instr set check, and impl some stuff in MMX/SSE/SSE2/3/4.1/4.2? (also check popcount and AVX)
// @@@ support non-integer textures top level (alg)? need to replace all ui-stores by float-stores, rest is ok
// @@@ change comma pos list & super constr everywhere
// @@@ change template formatting everywhere

#include "litiv/datasets.hpp"

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define EVALUATE_OUTPUT         1
#define DEBUG_OUTPUT            0
#define DISPLAY_OUTPUT          0
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
#if EVALUATE_OUTPUT
#if HAVE_GLSL
#define USE_GLSL_EVALUATION     1
#endif //HAVE_GLSL
#if HAVE_CUDA
#define USE_CUDA_EVALUATION     1
#endif //HAVE_CUDA
#if HAVE_OPENCL
#define USE_OPENCL_EVALUATION   1
#endif //HAVE_OPENCL
#if HAVE_GPU_SUPPORT
#define VALIDATE_GPU_EVALUATION 0
#endif //HAVE_GPU_SUPPORT
#endif //EVALUATE_OUTPUT
////////////////////////////////
#ifndef VALIDATE_GPU_EVALUATION
#define VALIDATE_GPU_EVALUATION 0
#endif //VALIDATE_GPU_EVALUATION
#ifndef USE_GLSL_EVALUATION
#define USE_GLSL_EVALUATION 0
#endif //USE_GLSL_EVALUATION
#ifndef USE_CUDA_EVALUATION
#define USE_CUDA_EVALUATION 0
#endif //USE_CUDA_EVALUATION
#ifndef USE_OPENCL_EVALUATION
#define USE_OPENCL_EVALUATION 0
#endif //USE_OPENCL_EVALUATION
#if (DEBUG_OUTPUT && !DISPLAY_OUTPUT)
#undef DISPLAY_OUTPUT
#define DISPLAY_OUTPUT 1
#endif //(DEBUG_OUTPUT && !DISPLAY_OUTPUT)
#define USE_GPU_IMPL (USE_GLSL_IMPL||USE_CUDA_IMPL||USE_OPENCL_IMPL)
#define USE_GPU_EVALUATION (USE_GLSL_EVALUATION || USE_CUDA_EVALUATION || USE_OPENCL_EVALUATION)
#define NEED_FG_MASK (DISPLAY_OUTPUT || WRITE_IMG_OUTPUT || (EVALUATE_OUTPUT && (!USE_GPU_EVALUATION || VALIDATE_GPU_EVALUATION)))
#define NEED_LAST_GT_MASK (DISPLAY_OUTPUT || (EVALUATE_OUTPUT && (!USE_GPU_EVALUATION || VALIDATE_GPU_EVALUATION)))
#define NEED_GT_MASK (DISPLAY_OUTPUT || EVALUATE_OUTPUT)
#if (USE_GLSL_IMPL+USE_CUDA_IMPL+USE_OPENCL_IMPL)>1
#error "Must specify a single impl."
#elif (USE_LOBSTER+USE_SUBSENSE+USE_PAWCS)!=1
#error "Must specify a single algorithm."
#elif USE_PAWCS
#include "litiv/video/BackgroundSubtractorPAWCS.hpp"
#elif USE_LOBSTER
#include "litiv/video/BackgroundSubtractorLOBSTER.hpp"
#elif USE_SUBSENSE
#include "litiv/video/BackgroundSubtractorSuBSENSE.hpp"
#endif //USE_...
#if (DEBUG_OUTPUT && (USE_GPU_IMPL || DEFAULT_NB_THREADS>1))
#error "Cannot debug output with GPU support or with more than one thread."
#endif //(DEBUG_OUTPUT && (USE_GPU_IMPL || DEFAULT_NB_THREADS>1))
#if (HAVE_GLSL && USE_GLSL_IMPL)
#if !HAVE_GLFW
#error "missing glfw"
#endif //!HAVE_GLFW
#endif //(HAVE_GLSL && USE_GLSL_IMPL)
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
int g_nLatestMouseX = -1, g_nLatestMouseY = -1; // @@@@ package x/y/event/code into shared_ptr struct? check opencv 3.0 callback for mouse event?
int *g_pnLatestMouseX = &g_nLatestMouseX, *g_pnLatestMouseY = &g_nLatestMouseY;
void OnMouseEvent(int event, int x, int y, int, void*) {
    if(event!=cv::EVENT_MOUSEMOVE || !x || !y)
        return;
    *g_pnLatestMouseX = x;
    *g_pnLatestMouseY = y;
}

std::atomic_size_t g_nActiveThreads(0);
const size_t g_nMaxThreads = USE_GPU_IMPL?1:std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():DEFAULT_NB_THREADS;
constexpr bool g_bIsPrecaching = DATASET_PRECACHING;

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
std::string g_sLatestGLFWErrorMessage;
void GLFWErrorCallback(int nCode, const char* acMessage) {
    std::stringstream ssStr;
    ssStr << "code: " << nCode << ", message: " << acMessage;
    g_sLatestGLFWErrorMessage = ssStr.str();
}

void AnalyzeSequence(int nThreadIdx, litiv::IDataHandlerPtr pBatch) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrFrameIdx = 0;
    size_t nNextFrameIdx = nCurrFrameIdx+1;
    bool bGPUContextInitialized = false;
    try {
        glfwSetErrorCallback(GLFWErrorCallback);
        CV_Assert(pCurrSequence.get() && pCurrSequence->GetTotalImageCount()>1);
        if(pCurrSequence->m_pEvaluator==nullptr && EVALUATE_OUTPUT)
            lvErrorExt("Missing evaluation impl for video segmentation dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str());
        const std::string sCurrSeqName = CxxUtils::clampString(pCurrSequence->m_sName,12);
        const size_t nFrameCount = pCurrSequence->GetTotalImageCount();
        const cv::Mat oROI = pCurrSequence->GetROI();
        cv::Mat oCurrInputFrame = pCurrSequence->GetInputFromIndex(nCurrFrameIdx).clone();
        CV_Assert(!oCurrInputFrame.empty());
        CV_Assert(oCurrInputFrame.isContinuous());
#if NEED_GT_MASK
        cv::Mat oCurrGTMask = pCurrSequence->GetGTFromIndex(nCurrFrameIdx).clone();
        CV_Assert(!oCurrGTMask.empty() && oCurrGTMask.isContinuous());
#endif //NEED_GT_MASK
#if DISPLAY_OUTPUT
        cv::Mat oLastInputFrame = oCurrInputFrame.clone();
#endif //DISPLAY_OUTPUT
        cv::Mat oNextInputFrame = pCurrSequence->GetInputFromIndex(nNextFrameIdx);
#if NEED_GT_MASK
#if NEED_LAST_GT_MASK
        cv::Mat oLastGTMask = oCurrGTMask.clone();
#endif // NEED_LAST_GT_MASK
        cv::Mat oNextGTMask = pCurrSequence->GetGTFromIndex(nNextFrameIdx);
#endif //NEED_GT_MASK
#if NEED_FG_MASK
        cv::Mat oLastFGMask(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(0));
#endif //NEED_FG_MASK
#if DISPLAY_OUTPUT
        cv::Mat oLastBGImg;
#endif //DISPLAY_OUTPUT
        glAssert(oCurrInputFrame.channels()==1 || oCurrInputFrame.channels()==4);
        cv::Size oWindowSize = oCurrInputFrame.size();
        // note: never construct GL classes before context initialization
        if(glfwInit()==GL_FALSE)
            glError("Failed to init GLFW");
        bGPUContextInitialized = true;
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,TARGET_GL_VER_MAJOR);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,TARGET_GL_VER_MINOR);
        glfwWindowHint(GLFW_RESIZABLE,GL_FALSE);
#if !DISPLAY_OUTPUT
        glfwWindowHint(GLFW_VISIBLE,GL_FALSE);
#endif //!DISPLAY_OUTPUT
        std::unique_ptr<GLFWwindow,void(*)(GLFWwindow*)> pWindow(glfwCreateWindow(oWindowSize.width,oWindowSize.height,(pCurrSequence->m_sRelativePath+" [GPU]").c_str(),nullptr,nullptr),glfwDestroyWindow);
        if(!pWindow)
            glError("Failed to create window via GLFW");
        glfwMakeContextCurrent(pWindow.get());
        GLContext<TARGET_GL_VER_MAJOR,TARGET_GL_VER_MINOR>::initGLEW();
#if USE_LOBSTER
        std::shared_ptr<BackgroundSubtractorLOBSTER_GLSL> pAlgo(new BackgroundSubtractorLOBSTER_GLSL());
        const double dDefaultLearningRate = BGSLOBSTER_DEFAULT_LEARNING_RATE;
        pAlgo->initialize(oCurrInputFrame,oROI);
#elif USE_SUBSENSE
#error "Missing glsl impl." // ... @@@@@
        std::shared_ptr<BackgroundSubtractorSuBSENSE_GLSL> pAlgo(new BackgroundSubtractorSuBSENSE_GLSL());
        const double dDefaultLearningRate = 0;
        pAlgo->initialize(oCurrInputFrame,oROI);
#elif USE_PAWCS
#error "Missing glsl impl." // ... @@@@@
        std::shared_ptr<BackgroundSubtractorPAWCS_GLSL> pAlgo(new BackgroundSubtractorPAWCS_GLSL());
        const double dDefaultLearningRate = 0;
        pAlgo->initialize(oCurrInputFrame,oROI);
#endif //USE...
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = pCurrSequence->m_sRelativePath;
        cv::namedWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
        std::shared_ptr<GLImageProcAlgo> pGLSLAlgo = std::dynamic_pointer_cast<GLImageProcAlgo>(pAlgo);
        if(pGLSLAlgo==nullptr)
            glError("Segmentation algorithm has no GLImageProcAlgo interface");
        pGLSLAlgo->setOutputFetching(NEED_FG_MASK);
        if(!pGLSLAlgo->getIsUsingDisplay() && DISPLAY_OUTPUT) // @@@@ determine in advance to hint window to hide? or just always hide, and show when needed?
            glfwHideWindow(pWindow.get());
#if USE_GLSL_EVALUATION
        std::shared_ptr<DatasetUtils::EvaluatorBase::GLEvaluatorBase> pGLSLAlgoEvaluator;
        if(pCurrSequence->m_pEvaluator!=nullptr)
            pGLSLAlgoEvaluator = std::dynamic_pointer_cast<DatasetUtils::EvaluatorBase::GLEvaluatorBase>(pCurrSequence->m_pEvaluator->CreateGLEvaluator(pGLSLAlgo,nFrameCount));
        if(pGLSLAlgoEvaluator==nullptr)
            glError("Segmentation evaluation algorithm has no GLSegmEvaluator interface");
        pGLSLAlgoEvaluator->initialize(oCurrGTMask,oROI.empty()?cv::Mat(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(255)):oROI);
        oWindowSize.width *= pGLSLAlgoEvaluator->m_nSxSDisplayCount;
#else //!USE_GLSL_EVALUATION
        oWindowSize.width *= pGLSLAlgo->m_nSxSDisplayCount;
#endif //!USE_GLSL_EVALUATION
        glfwSetWindowSize(pWindow.get(),oWindowSize.width,oWindowSize.height);
        glViewport(0,0,oWindowSize.width,oWindowSize.height);
        while(nNextFrameIdx<=nFrameCount) {
            if(!((nCurrFrameIdx+1)%100))
                std::cout << "\t\t" << CxxUtils::clampString(sCurrSeqName,12) << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nFrameCount)) << nCurrFrameIdx+1 << "/" << nFrameCount << "   [GPU]" << std::endl;
            const double dCurrLearningRate = nCurrFrameIdx<=100?1:dDefaultLearningRate;
            pAlgo->apply_async(oNextInputFrame,dCurrLearningRate);
#if USE_GLSL_EVALUATION
            pGLSLAlgoEvaluator->apply_async(oNextGTMask);
#endif //USE_GLSL_EVALUATION
#if DISPLAY_OUTPUT
            oCurrInputFrame.copyTo(oLastInputFrame);
            oNextInputFrame.copyTo(oCurrInputFrame);
#endif //DISPLAY_OUTPUT
            if(++nNextFrameIdx<nFrameCount)
                oNextInputFrame = pCurrSequence->GetInputFromIndex(nNextFrameIdx);
#if DEBUG_OUTPUT
            cv::imshow(sMouseDebugDisplayName,oNextInputFrame);
#endif //DEBUG_OUTPUT
#if NEED_GT_MASK
#if NEED_LAST_GT_MASK
            oCurrGTMask.copyTo(oLastGTMask);
            oNextGTMask.copyTo(oCurrGTMask);
#endif //NEED_LAST_GT_MASK
            if(nNextFrameIdx<nFrameCount)
                oNextGTMask = pCurrSequence->GetGTFromIndex(nNextFrameIdx);
#endif //NEED_GT_MASK
            glErrorCheck;
            if(glfwWindowShouldClose(pWindow.get()))
                break;
            glfwPollEvents();
#if DISPLAY_OUTPUT
            if(glfwGetKey(pWindow.get(),GLFW_KEY_ESCAPE) || glfwGetKey(pWindow.get(),GLFW_KEY_Q))
                break;
            glfwSwapBuffers(pWindow.get());
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
#endif //DISPLAY_OUTPUT
#if NEED_FG_MASK
            pAlgo->getLatestForegroundMask(oLastFGMask);
            if(!oROI.empty())
                cv::bitwise_or(oLastFGMask,UCHAR_MAX/2,oLastFGMask,oROI==0);
#endif //NEED_FG_MASK
#if DISPLAY_OUTPUT
            pAlgo->getBackgroundImage(oLastBGImg);
            if(!oROI.empty())
                cv::bitwise_or(oLastBGImg,UCHAR_MAX/2,oLastBGImg,oROI==0);
            cv::Mat oDisplayFrame = DatasetUtils::GetDisplayImage(oLastInputFrame,oLastBGImg,pCurrSequence->m_pEvaluator?pCurrSequence->m_pEvaluator->GetColoredSegmMaskFromResult(oLastFGMask,oLastGTMask,oROI):oLastFGMask,nCurrFrameIdx);
            cv::Mat oDisplayFrameResized;
            if(oDisplayFrame.cols>1920 || oDisplayFrame.rows>1080)
                cv::resize(oDisplayFrame,oDisplayFrameResized,cv::Size(oDisplayFrame.cols/2,oDisplayFrame.rows/2));
            else
                oDisplayFrameResized = oDisplayFrame;
            cv::imshow(sDisplayName,oDisplayFrameResized);
            int nKeyPressed;
            if(bContinuousUpdates)
                nKeyPressed = cv::waitKey(1);
            else
                nKeyPressed = cv::waitKey(0);
            if(nKeyPressed!=-1)
                nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
            if(nKeyPressed==' ')
                bContinuousUpdates = !bContinuousUpdates;
            else if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT
#if (EVALUATE_OUTPUT && (!USE_GLSL_EVALUATION || VALIDATE_GPU_EVALUATION))
            if(pCurrSequence->m_pEvaluator)
                pCurrSequence->m_pEvaluator->AccumulateMetricsFromResult(oCurrFGMask,oCurrGTMask,oROI);
#endif //(EVALUATE_OUTPUT && (!USE_GLSL_EVALUATION || VALIDATE_GPU_EVALUATION))
            ++nCurrFrameIdx;
        }
        const double dTimeElapsed = TIMER_ELAPSED_MS(MainLoop)/1000;
        const double dAvgFPS = (double)nCurrFrameIdx/dTimeElapsed;
        std::cout << "\t\t" << CxxUtils::clampString(sCurrSeqName,12) << " @ end, " << int(dTimeElapsed) << " sec in-thread (" << (int)floor(dAvgFPS+0.5) << " FPS)" << std::endl;
#if EVALUATE_OUTPUT
        if(pCurrSequence->m_pEvaluator) {
#if USE_GLSL_EVALUATION
#if VALIDATE_GPU_EVALUATION
            printf("cpu eval:\n\tnTP=%" PRIu64 ", nTN=%" PRIu64 ", nFP=%" PRIu64 ", nFN=%" PRIu64 ", nSE=%" PRIu64 ", tot=%" PRIu64 "\n",pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTP,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTN,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFP,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFN,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nSE,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTP+pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTN+pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFP+pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFN);
#endif //VALIDATE_USE_GLSL_EVALUATION
            pCurrSequence->m_pEvaluator->FetchGLEvaluationResults(pGLSLAlgoEvaluator);
#if VALIDATE_GPU_EVALUATION
            printf("gpu eval:\n\tnTP=%" PRIu64 ", nTN=%" PRIu64 ", nFP=%" PRIu64 ", nFN=%" PRIu64 ", nSE=%" PRIu64 ", tot=%" PRIu64 "\n",pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTP,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTN,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFP,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFN,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nSE,pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTP+pCurrSequence->m_pEvaluator->m_oBasicMetrics.nTN+pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFP+pCurrSequence->m_pEvaluator->m_oBasicMetrics.nFN);
#endif //VALIDATE_USE_GLSL_EVALUATION
#endif //USE_GLSL_EVALUATION
            pCurrSequence->m_pEvaluator->dTimeElapsed_sec = dTimeElapsed;
        }
#endif //EVALUATE_OUTPUT
#if DISPLAY_OUTPUT
        cv::destroyWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
    }
    catch(const CxxUtils::Exception& e) {
        std::cout << "\nAnalyzeSequence caught Exception:\n" << e.what();
        if(!g_sLatestGLFWErrorMessage.empty()) {
            std::cout << " (" << g_sLatestGLFWErrorMessage << ")" << "\n" << std::endl;
            g_sLatestGLFWErrorMessage = std::string();
        }
        else
            std::cout << "\n" << std::endl;
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyzeSequence caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyzeSequence caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyzeSequence caught unhandled exception\n" << std::endl;}
    if(bGPUContextInitialized)
        glfwTerminate();
    if(pCurrSequence.get()) {
#if DATASET_PRECACHING
        pCurrSequence->StopPrecaching();
#endif //DATASET_PRECACHING
        pCurrSequence->m_nImagesProcessed.set_value(nCurrFrameIdx);
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
#if USE_LOBSTER
        std::shared_ptr<BackgroundSubtractorLOBSTER> pAlgo(new BackgroundSubtractorLOBSTER());
        const double dDefaultLearningRate = pAlgo->getDefaultLearningRate();
#elif USE_SUBSENSE
        std::shared_ptr<BackgroundSubtractorSuBSENSE> pAlgo(new BackgroundSubtractorSuBSENSE());
        const double dDefaultLearningRate = 0;
#elif USE_PAWCS
        std::shared_ptr<BackgroundSubtractorPAWCS> pAlgo(new BackgroundSubtractorPAWCS());
        const double dDefaultLearningRate = 0;
#endif //USE_...
        pAlgo->initialize(oCurrInputFrame,oROI);
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = oCurrSequence.getRelativePath();
        cv::namedWindow(sDisplayName);
#if DEBUG_OUTPUT
        // @@@@@ fuse getDisplayImage output image (3 tiles) with debug in new utility struct in utils module?
        cv::FileStorage oDebugFS = cv::FileStorage(oCurrSequence.getOutputPath()+"/../"+oCurrSequence.getName()+"_debug.yml",cv::FileStorage::WRITE);
        pAlgo->m_pDebugFS = &oDebugFS;
        pAlgo->m_sDebugName = oCurrSequence.getName();
        g_pnLatestMouseX = &pAlgo->m_nDebugCoordX;
        g_pnLatestMouseY = &pAlgo->m_nDebugCoordY;
        std::string sMouseDebugDisplayName = oCurrSequence.getName() + " [MOUSE DEBUG]";
        cv::namedWindow(sMouseDebugDisplayName,0);
        cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
#endif //DEBUG_OUTPUT
#endif //DISPLAY_OUTPUT
        oCurrSequence.startProcessing();
        while(nCurrFrameIdx<nFrameCount) {
            if(!((nCurrFrameIdx+1)%100) && nCurrFrameIdx<nFrameCount)
                std::cout << "\t\t" << CxxUtils::clampString(sCurrSeqName,12) << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nFrameCount)) << nCurrFrameIdx+1 << "/" << nFrameCount << "   [T=" << nThreadIdx << "]" << std::endl;
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
            cv::Mat oDisplayFrame = litiv::getDisplayImage(oCurrInputFrame,oCurrBGImg,oCurrSequence.getColoredSegmMask(oCurrFGMask,nCurrFrameIdx),nCurrFrameIdx,cv::Point(*g_pnLatestMouseX,*g_pnLatestMouseY));
            cv::Mat oDisplayFrameResized;
            if(oDisplayFrame.cols>1920 || oDisplayFrame.rows>1080)
                cv::resize(oDisplayFrame,oDisplayFrameResized,cv::Size(oDisplayFrame.cols/2,oDisplayFrame.rows/2));
            else
                oDisplayFrameResized = oDisplayFrame;
            cv::imshow(sDisplayName,oDisplayFrameResized);
#if DEBUG_OUTPUT
            cv::imshow(sMouseDebugDisplayName,oCurrInputFrame);
#endif //DEBUG_OUTPUT
            int nKeyPressed;
            if(bContinuousUpdates)
                nKeyPressed = cv::waitKey(1);
            else
                nKeyPressed = cv::waitKey(0);
            if(nKeyPressed!=-1)
                nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
            if(nKeyPressed==' ')
                bContinuousUpdates = !bContinuousUpdates;
            else if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT
            oCurrSequence.pushSegmMask(oCurrFGMask,nCurrFrameIdx++);
        }
        oCurrSequence.stopProcessing();
        const double dTimeElapsed = oCurrSequence.getProcessTime();
        const double dProcessSpeed = (double)nCurrFrameIdx/dTimeElapsed;
        std::cout << "\t\t" << CxxUtils::clampString(sCurrSeqName,12) << " @ F:" << nCurrFrameIdx << "/" << nFrameCount << "   [T=" << nThreadIdx << "]   (" << std::fixed << std::setw(4) << dTimeElapsed << " sec, " << std::setw(4) << dProcessSpeed << " Hz)" << std::endl;
        oCurrSequence.writeEvalReport(); // this line is optional; it allows results to be read before all batches are processed
#if DISPLAY_OUTPUT
#if DEBUG_OUTPUT
        cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
        cv::destroyWindow(sMouseDebugDisplayName);
        g_pnLatestMouseX = &g_nLatestMouseX;
        g_pnLatestMouseY = &g_nLatestMouseY;
#endif //DEBUG_OUTPUT
        cv::destroyWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
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
