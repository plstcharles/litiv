
#include "litiv/utils/DatasetEvalUtils.hpp"

////////////////////////////////
#define WRITE_IMG_OUTPUT        1
#define EVALUATE_OUTPUT         1
#define DEBUG_OUTPUT            0
#define DISPLAY_OUTPUT          1
#define DISPLAY_TIMERS          0
////////////////////////////////
#define USE_CANNY               1
#define USE_LBSP                0
////////////////////////////////
#define FULL_THRESH_ANALYSIS    1
////////////////////////////////
#define USE_GLSL_IMPL           0
#define USE_CUDA_IMPL           0
#define USE_OPENCL_IMPL         0
////////////////////////////////
#define DATASET_ID              eDataset_BSDS500_edge_train
#define DATASET_ROOT_PATH       std::string("/shared2/datasets/")
#define DATASET_RESULTS_PATH    std::string("results_ceval_bsds500utils_full")
#define DATASET_PRECACHING      1
////////////////////////////////
#if (DEBUG_OUTPUT && !DISPLAY_OUTPUT)
#undef DISPLAY_OUTPUT
#define DISPLAY_OUTPUT 1
#endif //(DEBUG_OUTPUT && !DISPLAY_OUTPUT)
#if (DEBUG_OUTPUT && FULL_THRESH_ANALYSIS)
#error "Cannot enable debug output while using all threshold values."
#endif //(DEBUG_OUTPUT && FULL_THRESH_ANALYSIS)
#define USE_GPU_IMPL (USE_GLSL_IMPL||USE_CUDA_IMPL||USE_OPENCL_IMPL)
#if EVALUATE_OUTPUT
#if (HAVE_GLSL && USE_GLSL_IMPL)
#define USE_GLSL_EVALUATION 1
#endif //(HAVE_GLSL && USE_GLSL_IMPL)
#if (HAVE_CUDA && USE_CUDA_IMPL)
#define USE_CUDA_EVALUATION 1
#endif //(HAVE_CUDA && USE_CUDA_IMPL)
#if (HAVE_OPENCL && USE_OPENCL_IMPL)
#define USE_OPENCL_EVALUATION 1
#endif //(HAVE_OPENCL && USE_OPENCL_IMPL)
#if HAVE_GPU_SUPPORT
#define VALIDATE_GPU_EVALUATION 0
#endif //HAVE_GPU_SUPPORT
#endif //EVALUATE_OUTPUT
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
#define USE_GPU_EVALUATION (USE_GLSL_EVALUATION || USE_CUDA_EVALUATION || USE_OPENCL_EVALUATION)
#define NEED_EDGES_MASK (DISPLAY_OUTPUT || WRITE_IMG_OUTPUT || (EVALUATE_OUTPUT && (!USE_GPU_EVALUATION || VALIDATE_GPU_EVALUATION)))
#define NEED_LAST_GT_MASK (DISPLAY_OUTPUT || (EVALUATE_OUTPUT && (!USE_GPU_EVALUATION || VALIDATE_GPU_EVALUATION)))
#define NEED_GT_MASK (DISPLAY_OUTPUT || EVALUATE_OUTPUT)
#if (USE_GLSL_IMPL+USE_CUDA_IMPL+USE_OPENCL_IMPL)>1
#error "Must specify a single impl."
#elif (USE_CANNY+USE_LBSP)!=1
#error "Must specify a single algorithm."
#elif USE_CANNY
#include "litiv/imgproc/EdgeDetectorCanny.hpp"
#elif USE_LBSP
#include "litiv/imgproc/EdgeDetectorLBSP.hpp"
#endif //USE_...
#if DISPLAY_TIMERS
#define TIMER_INTERNAL_TIC(x) TIMER_TIC(x)
#define TIMER_INTERNAL_TOC(x) TIMER_TOC(x)
#define TIMER_INTERNAL_ELAPSED_MS(x) TIMER_ELAPSED_MS(x)
#else //!ENABLE_INTERNAL_TIMERS
#define TIMER_INTERNAL_TIC(x)
#define TIMER_INTERNAL_TOC(x)
#define TIMER_INTERNAL_ELAPSED_MS(x)
#endif //!ENABLE_INTERNAL_TIMERS
#if (HAVE_GLSL && USE_GLSL_IMPL)
void AnalyzeSet_GLSL(std::shared_ptr<DatasetUtils::Segm::Image::Set> pCurrSet);
#elif (HAVE_CUDA && USE_CUDA_IMPL)
static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
void AnalyzeSet(int nThreadIdx, std::shared_ptr<DatasetUtils::Segm::Image::Set> pCurrSet);
#else // bad config
#error "Bad config, trying to use an unavailable impl."
#endif // bad config

std::atomic_size_t g_nActiveThreads(0);
const size_t g_nMaxThreads = DEFAULT_NB_THREADS;//std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():DEFAULT_NB_THREADS;
const std::shared_ptr<DatasetUtils::Segm::Image::DatasetInfo> g_pDatasetInfo = DatasetUtils::Segm::Image::GetDatasetInfo(DatasetUtils::Segm::Image::DATASET_ID,DATASET_ROOT_PATH,DATASET_RESULTS_PATH,USE_GPU_IMPL);

int main(int, char**) {
    try {
        std::cout << "Parsing dataset '" << g_pDatasetInfo->m_sDatasetName << "'..." << std::endl;
        std::vector<std::shared_ptr<DatasetUtils::WorkGroup>> vpDatasetGroups = g_pDatasetInfo->ParseDataset();
        size_t nImagesTotal = 0;
        // @@@ check out priority_queue?
        std::multimap<double,std::shared_ptr<DatasetUtils::Segm::Image::Set>> mSetsLoads;
        for(auto ppGroupIter=vpDatasetGroups.begin(); ppGroupIter!=vpDatasetGroups.end(); ++ppGroupIter) {
            for(auto ppBatchIter=(*ppGroupIter)->m_vpBatches.begin(); ppBatchIter!=(*ppGroupIter)->m_vpBatches.end(); ++ppBatchIter) {
                auto pSet = std::dynamic_pointer_cast<DatasetUtils::Segm::Image::Set>(*ppBatchIter);
                CV_Assert(pSet!=nullptr);
                nImagesTotal += pSet->GetTotalImageCount();
                mSetsLoads.insert(std::make_pair(pSet->GetExpectedLoad(),pSet));
            }
        }
        const size_t nSetsTotal = mSetsLoads.size();
        if(nSetsTotal==0 || nImagesTotal==0)
            throw std::runtime_error(cv::format("Could not find any image to process for dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str()));
        std::cout << "Parsing complete. [" << vpDatasetGroups.size() << " group(s), " << nSetsTotal << " set(s)]\n" << std::endl;
        const time_t nStartupTime = time(nullptr);
        const std::string sStartupTimeStr(asctime(localtime(&nStartupTime)));
        std::cout << "[" << sStartupTimeStr.substr(0,sStartupTimeStr.size()-1) << "]" << std::endl;
        std::cout << "Executing edge detection with " << ((g_nMaxThreads>nSetsTotal)?nSetsTotal:g_nMaxThreads) << " thread(s)..." << std::endl;
        size_t nSetsProcessed = 0;
        for(auto pSetIter = mSetsLoads.rbegin(); pSetIter!=mSetsLoads.rend(); ++pSetIter) {
            while(g_nActiveThreads>=g_nMaxThreads)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::cout << "\tProcessing [" << ++nSetsProcessed << "/" << nSetsTotal << "] (" << pSetIter->second->m_sRelativePath << ", L=" << std::scientific << std::setprecision(2) << pSetIter->first << ")" << std::endl;
            if(DATASET_PRECACHING)
                pSetIter->second->StartPrecaching(EVALUATE_OUTPUT);
#if (HAVE_GLSL && USE_GLSL_IMPL)
            AnalyzeSet_GLSL(pSetIter->second);
#elif (HAVE_CUDA && USE_CUDA_IMPL)
            static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
            static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
            ++g_nActiveThreads;
            std::thread(AnalyzeSet,nSetsProcessed,pSetIter->second).detach();
#endif //!USE_GPU_IMPL
        }
        while(g_nActiveThreads>0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        size_t nTotImagesProcessed = 0;
        for(auto pSetIter = mSetsLoads.rbegin(); pSetIter!=mSetsLoads.rend(); ++pSetIter)
            nTotImagesProcessed += pSetIter->second->m_nImagesProcessed.get_future().get();
        const time_t nShutdownTime = time(nullptr);
        const std::string sShutdownTimeStr(asctime(localtime(&nShutdownTime)));
        std::cout << "[" << sShutdownTimeStr.substr(0,sShutdownTimeStr.size()-1) << "]\n" << std::endl;
        if(EVALUATE_OUTPUT && nTotImagesProcessed==nImagesTotal) {
            std::cout << "Writing evaluation results..." << std::endl;
            g_pDatasetInfo->WriteEvalResults(vpDatasetGroups);

        }
        else if(EVALUATE_OUTPUT && nTotImagesProcessed<nImagesTotal)
            std::cout << "Skipping eval results output, as some image sets were skipped." << std::endl;
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    std::cout << "\nAll done." << std::endl;
    return 0;
}

#if (HAVE_GLSL && USE_GLSL_IMPL)
std::string g_sLatestGLFWErrorMessage;
void GLFWErrorCallback(int nCode, const char* acMessage) {
    std::stringstream ssStr;
    ssStr << "code: " << nCode << ", message: " << acMessage;
    g_sLatestGLFWErrorMessage = ssStr.str();
}

void AnalyzeSet_GLSL(std::shared_ptr<DatasetUtils::Segm::Image::Set> pCurrSet) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrImageIdx = 0;
    size_t nNextImageIdx = nCurrImageIdx+1;
    bool bGPUContextInitialized = false;
    try {
        glfwSetErrorCallback(GLFWErrorCallback);
        CV_Assert(pCurrSet.get() && pCurrSet->GetTotalImageCount()>0);
        if(pCurrSet->m_pEvaluator==nullptr && EVALUATE_OUTPUT)
            throw std::runtime_error(cv::format("Missing evaluation impl for image segmentation dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str()));
        const std::string sCurrSetName = pCurrSet->m_sName.size()>12?pCurrSet->m_sName.substr(0,12):pCurrSet->m_sName;
        const size_t nImageCount = pCurrSet->GetTotalImageCount();
        cv::Mat oCurrInputImage = pCurrSet->GetInputFromIndex(nCurrImageIdx).clone();
        CV_Assert(!oCurrInputImage.empty());
        CV_Assert(oCurrInputImage.isContinuous());
#if NEED_GT_MASK
        cv::Mat oCurrGTMask = pCurrSet->GetGTFromIndex(nCurrImageIdx).clone();
        CV_Assert(!oCurrGTMask.empty() && oCurrGTMask.isContinuous());
#endif //NEED_GT_MASK
#if DISPLAY_OUTPUT
        cv::Mat oLastInputImage = oCurrInputImage.clone();
#endif //DISPLAY_OUTPUT
        cv::Mat oNextInputImage = pCurrSet->GetInputFromIndex(nNextImageIdx);
#if NEED_GT_MASK
#if NEED_LAST_GT_MASK
        cv::Mat oLastGTMask = oCurrGTMask.clone();
#endif // NEED_LAST_GT_MASK
        cv::Mat oNextGTMask = pCurrSet->GetGTFromIndex(nNextImageIdx);
#endif //NEED_GT_MASK
#if NEED_EDGES_MASK
        cv::Mat oLastEdgeMask(oCurrInputImage.size(),CV_8UC1,cv::Scalar_<uchar>(0));
#endif //NEED_EDGES_MASK
        glAssert(oCurrInputImage.channels()==1 || oCurrInputImage.channels()==4);
        cv::Size oWindowSize = pCurrSet->GetMaxImageSize();
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
        std::unique_ptr<GLFWwindow,void(*)(GLFWwindow*)> pWindow(glfwCreateWindow(oWindowSize.width,oWindowSize.height,(pCurrSet->m_sRelativePath+" [GPU]").c_str(),nullptr,nullptr),glfwDestroyWindow);
        if(!pWindow)
            glError("Failed to create window via GLFW");
        glfwMakeContextCurrent(pWindow.get());
        glewInitErrorCheck;
#if USE_CANNY
#error "Missing glsl impl." // ... @@@@@
        std::shared_ptr<BackgroundSubtractorLOBSTER_GLSL> pAlgo(new BackgroundSubtractorLOBSTER_GLSL());
#elif USE_LBSP
#error "Missing glsl impl." // ... @@@@@
        std::shared_ptr<BackgroundSubtractorSuBSENSE_GLSL> pAlgo(new BackgroundSubtractorSuBSENSE_GLSL());
#endif //USE_...
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = pCurrSet->m_sRelativePath;
        cv::namedWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
        std::shared_ptr<GLImageProcAlgo> pGLSLAlgo = std::dynamic_pointer_cast<GLImageProcAlgo>(pAlgo);
        if(pGLSLAlgo==nullptr)
            glError("Segmentation algorithm has no GLImageProcAlgo interface");
        pGLSLAlgo->setOutputFetching(NEED_EDGES_MASK);
        if(!pGLSLAlgo->getIsUsingDisplay() && DISPLAY_OUTPUT) // @@@@ determine in advance to hint window to hide? or just always hide, and show when needed?
            glfwHideWindow(pWindow.get());
#if USE_GLSL_EVALUATION
        std::shared_ptr<DatasetUtils::EvaluatorBase::GLEvaluatorBase> pGLSLAlgoEvaluator;
        if(pCurrSet->m_pEvaluator!=nullptr)
            pGLSLAlgoEvaluator = std::dynamic_pointer_cast<DatasetUtils::EvaluatorBase::GLEvaluatorBase>(pCurrSet->m_pEvaluator->CreateGLEvaluator(pGLSLAlgo,nImageCount));
        if(pGLSLAlgoEvaluator==nullptr)
            glError("Segmentation evaluation algorithm has no GLSegmEvaluator interface");
        pGLSLAlgoEvaluator->initialize(oCurrGTMask,cv::Mat(oCurrInputImage.size(),CV_8UC1,cv::Scalar_<uchar>(255)));
        oWindowSize.width *= pGLSLAlgoEvaluator->m_nSxSDisplayCount;
#else //!USE_GLSL_EVALUATION
        oWindowSize.width *= pGLSLAlgo->m_nSxSDisplayCount;
#endif //!USE_GLSL_EVALUATION
        glfwSetWindowSize(pWindow.get(),oWindowSize.width,oWindowSize.height);
        glViewport(0,0,oWindowSize.width,oWindowSize.height);
        TIMER_TIC(MainLoop);
        while(nNextImageIdx<=nImageCount) {
            //if(!((nCurrImageIdx+1)%100))
                std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSetName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nImageCount)) << nCurrImageIdx+1 << "/" << nImageCount << "   [GPU]" << std::endl;
            TIMER_INTERNAL_TIC(OverallLoop);
            TIMER_INTERNAL_TIC(PipelineUpdate);
            pAlgo->apply_async(oNextInputImage);
            TIMER_INTERNAL_TOC(PipelineUpdate);
#if USE_GLSL_EVALUATION
            pGLSLAlgoEvaluator->apply_async(oNextGTMask);
#endif //USE_GLSL_EVALUATION
            TIMER_INTERNAL_TIC(ImageQuery);
#if DISPLAY_OUTPUT
            oCurrInputImage.copyTo(oLastInputImage);
            oNextInputImage.copyTo(oCurrInputImage);
#endif //DISPLAY_OUTPUT
            if(++nNextImageIdx<nImageCount)
                oNextInputImage = pCurrSet->GetInputFromIndex(nNextImageIdx);
#if NEED_GT_MASK
#if NEED_LAST_GT_MASK
            oCurrGTMask.copyTo(oLastGTMask);
            oNextGTMask.copyTo(oCurrGTMask);
#endif //NEED_LAST_GT_MASK
            if(nNextImageIdx<nImageCount)
                oNextGTMask = pCurrSet->GetGTFromIndex(nNextImageIdx);
#endif //NEED_GT_MASK
            TIMER_INTERNAL_TOC(ImageQuery);
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
#if NEED_EDGES_MASK
            pAlgo->getLatestEdgeMask(oLastEdgeMask);
#endif //NEED_EDGES_MASK
#if DISPLAY_OUTPUT
            @@@ update via copy of non-parallel impl
            cv::Mat oDisplayImage = DatasetUtils::GetDisplayImage(oLastInputImage,oLastEdgeMask,pCurrSet->m_pEvaluator?pCurrSet->m_pEvaluator->GetColoredSegmMaskFromResult(oLastEdgeMask,oLastGTMask,cv::Mat()):oLastEdgeMask,nCurrImageIdx);
            cv::Mat oDisplayImageResized;
            if(oDisplayImage.cols>1280 || oDisplayImage.rows>960)
                cv::resize(oDisplayImage,oDisplayImageResized,cv::Size(oDisplayImage.cols/2,oDisplayImage.rows/2));
            else
                oDisplayImageResized = oDisplayImage;
            cv::imshow(sDisplayName,oDisplayImageResized);
            int nKeyPressed;
            if(bContinuousUpdates)
                nKeyPressed = cv::waitKey(1);
            else
                nKeyPressed = cv::waitKey(0);
            if(nKeyPressed!=-1) {
                nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
                std::cout << "nKeyPressed = " << nKeyPressed%(UCHAR_MAX+1) << std::endl;
            }
            if(nKeyPressed==' ')
                bContinuousUpdates = !bContinuousUpdates;
            else if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_OUTPUT
#if WRITE_IMG_OUTPUT
            pCurrSet->WriteResult(nCurrImageIdx,oLastEdgeMask);
#endif //WRITE_IMG_OUTPUT
#if (EVALUATE_OUTPUT && (!USE_GLSL_EVALUATION || VALIDATE_GPU_EVALUATION))
            if(pCurrSet->m_pEvaluator)
                pCurrSet->m_pEvaluator->AccumulateMetricsFromResult(oCurrEdgeMask,oCurrGTMask,cv::Mat());
#endif //(EVALUATE_OUTPUT && (!USE_GLSL_EVALUATION || VALIDATE_GPU_EVALUATION))
            TIMER_INTERNAL_TOC(OverallLoop);
#if DISPLAY_TIMERS
            std::cout << "ImageQuery=" << TIMER_INTERNAL_ELAPSED_MS(ImageQuery) << "ms,  "
            << "PipelineUpdate=" << TIMER_INTERNAL_ELAPSED_MS(PipelineUpdate) << "ms,  "
            << "OverallLoop=" << TIMER_INTERNAL_ELAPSED_MS(OverallLoop) << "ms" << std::endl;
#endif //ENABLE_INTERNAL_TIMERS
            ++nCurrImageIdx;
        }
        TIMER_TOC(MainLoop);
        const double dTimeElapsed = TIMER_ELAPSED_MS(MainLoop)/1000;
        const double dAvgFPS = (double)nCurrImageIdx/dTimeElapsed;
        std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSetName << " @ end, " << int(dTimeElapsed) << " sec in-thread (" << (int)floor(dAvgFPS+0.5) << " FPS)" << std::endl;
#if EVALUATE_OUTPUT
        if(pCurrSet->m_pEvaluator) {
#if USE_GLSL_EVALUATION
#if VALIDATE_GPU_EVALUATION
            printf("cpu eval:\n\tnTP=%" PRIu64 ", nTN=%" PRIu64 ", nFP=%" PRIu64 ", nFN=%" PRIu64 ", nSE=%" PRIu64 ", tot=%" PRIu64 "\n",pCurrSet->m_pEvaluator->m_oBasicMetrics.nTP,pCurrSet->m_pEvaluator->m_oBasicMetrics.nTN,pCurrSet->m_pEvaluator->m_oBasicMetrics.nFP,pCurrSet->m_pEvaluator->m_oBasicMetrics.nFN,pCurrSet->m_pEvaluator->m_oBasicMetrics.nSE,pCurrSet->m_pEvaluator->m_oBasicMetrics.nTP+pCurrSet->m_pEvaluator->m_oBasicMetrics.nTN+pCurrSet->m_pEvaluator->m_oBasicMetrics.nFP+pCurrSet->m_pEvaluator->m_oBasicMetrics.nFN);
#endif //VALIDATE_USE_GLSL_EVALUATION
            pCurrSet->m_pEvaluator->FetchGLEvaluationResults(pGLSLAlgoEvaluator);
#if VALIDATE_GPU_EVALUATION
            printf("gpu eval:\n\tnTP=%" PRIu64 ", nTN=%" PRIu64 ", nFP=%" PRIu64 ", nFN=%" PRIu64 ", nSE=%" PRIu64 ", tot=%" PRIu64 "\n",pCurrSet->m_pEvaluator->m_oBasicMetrics.nTP,pCurrSet->m_pEvaluator->m_oBasicMetrics.nTN,pCurrSet->m_pEvaluator->m_oBasicMetrics.nFP,pCurrSet->m_pEvaluator->m_oBasicMetrics.nFN,pCurrSet->m_pEvaluator->m_oBasicMetrics.nSE,pCurrSet->m_pEvaluator->m_oBasicMetrics.nTP+pCurrSet->m_pEvaluator->m_oBasicMetrics.nTN+pCurrSet->m_pEvaluator->m_oBasicMetrics.nFP+pCurrSet->m_pEvaluator->m_oBasicMetrics.nFN);
#endif //VALIDATE_USE_GLSL_EVALUATION
#endif //USE_GLSL_EVALUATION
            pCurrSet->m_pEvaluator->dTimeElapsed_sec = dTimeElapsed;
        }
#endif //EVALUATE_OUTPUT
#if DISPLAY_OUTPUT
        cv::destroyWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
    }
    catch(const GLException& e) {
        std::cout << "\nAnalyzeSequence caught GLException:\n" << e.what();
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
    if(pCurrSet.get()) {
        if(DATASET_PRECACHING)
            pCurrSet->StopPrecaching();
        pCurrSet->m_nImagesProcessed.set_value(nCurrImageIdx);
    }
}
#elif (HAVE_CUDA && USE_CUDA_IMPL)
static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
void AnalyzeSet(int nThreadIdx, std::shared_ptr<DatasetUtils::Segm::Image::Set> pCurrSet) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrImageIdx = 0;
    try {
        CV_Assert(pCurrSet.get() && pCurrSet->GetTotalImageCount()>0);
        if(pCurrSet->m_pEvaluator==nullptr && EVALUATE_OUTPUT)
            throw std::runtime_error(cv::format("Missing evaluation impl for image segmentation dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str()));
        const std::string sCurrSetName = pCurrSet->m_sName.size()>12?pCurrSet->m_sName.substr(0,12):pCurrSet->m_sName;
        const size_t nImageCount = pCurrSet->GetTotalImageCount();
        cv::Mat oCurrInputImage = pCurrSet->GetInputFromIndex(nCurrImageIdx).clone();
        CV_Assert(!oCurrInputImage.empty());
        CV_Assert(oCurrInputImage.isContinuous());
#if NEED_GT_MASK
        cv::Mat oCurrGTMask = pCurrSet->GetGTFromIndex(nCurrImageIdx).clone();
        CV_Assert(!oCurrGTMask.empty() && oCurrGTMask.isContinuous());
#endif //NEED_GT_MASK
        cv::Mat oCurrEdgeMask(oCurrInputImage.size(),CV_8UC1,cv::Scalar_<uchar>(0));
#if USE_CANNY
        std::shared_ptr<EdgeDetectorImpl> pAlgo(new EdgeDetectorCanny());
#elif USE_LBSP
        std::shared_ptr<EdgeDetectorImpl> pAlgo(new EdgeDetectorLBSP());
#endif //USE_...
#if !FULL_THRESH_ANALYSIS
        const double dThreshold = pAlgo->getDefaultThreshold();
#endif //!FULL_THRESH_ANALYSIS
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = pCurrSet->m_sRelativePath;
        cv::namedWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
#if DEBUG_OUTPUT
        cv::FileStorage oDebugFS = cv::FileStorage(pCurrSet->m_sResultsPath+"../"+pCurrSet->m_sName+"_debug.yml",cv::FileStorage::WRITE);
        pAlgo->m_pDebugFS = &oDebugFS;
        pAlgo->m_sDebugName = pCurrSet->m_sName;
        int nCurrThreshold = int((pAlgo->getDefaultThreshold())*UCHAR_MAX);
        cv::createTrackbar("Threshold:",sDisplayName,&nCurrThreshold,UCHAR_MAX);
#endif //DEBUG_OUTPUT
        bool bExit = false;
        TIMER_TIC(MainLoop);
        while(nCurrImageIdx<nImageCount && !bExit) {
            //if(!((nCurrImageIdx+1)%100))
                std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSetName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nImageCount)) << nCurrImageIdx+1 << "/" << nImageCount << "   [T=" << nThreadIdx << "]" << std::endl;
            TIMER_INTERNAL_TIC(OverallLoop);
            TIMER_INTERNAL_TIC(ImageQuery);
            oCurrInputImage = pCurrSet->GetInputFromIndex(nCurrImageIdx);
#if NEED_GT_MASK
            oCurrGTMask = pCurrSet->GetGTFromIndex(nCurrImageIdx);
#endif //NEED_GT_MASK
            TIMER_INTERNAL_TOC(ImageQuery);
            TIMER_INTERNAL_TIC(PipelineUpdate);
#if FULL_THRESH_ANALYSIS
            pAlgo->apply(oCurrInputImage,oCurrEdgeMask);
#else //!FULL_THRESH_ANALYSIS
            pAlgo->apply_threshold(oCurrInputImage,oCurrEdgeMask,dThreshold);
#endif //!FULL_THRESH_ANALYSIS
            TIMER_INTERNAL_TOC(PipelineUpdate);
#if DISPLAY_OUTPUT
#if DEBUG_OUTPUT
            while(!bExit) {
                pAlgo->apply(oCurrInputImage,oCurrEdgeMask,double(nCurrThreshold)/UCHAR_MAX);
#endif //DEBUG_OUTPUT
                cv::Mat oDisplayImage = DatasetUtils::GetDisplayImage(oCurrInputImage,oCurrEdgeMask,pCurrSet->m_pEvaluator?pCurrSet->m_pEvaluator->GetColoredSegmMaskFromResult(oCurrEdgeMask,oCurrGTMask,cv::Mat()):oCurrEdgeMask,nCurrImageIdx);
                cv::Mat oDisplayImageResized;
                if(oDisplayImage.cols>1280 || oDisplayImage.rows>960)
                    cv::resize(oDisplayImage,oDisplayImageResized,cv::Size(oDisplayImage.cols/2,oDisplayImage.rows/2));
                else
                    oDisplayImageResized = oDisplayImage;
                cv::imshow(sDisplayName,oDisplayImageResized);
                int nKeyPressed;
                if(bContinuousUpdates)
                    nKeyPressed = cv::waitKey(1);
                else
                    nKeyPressed = cv::waitKey(DEBUG_OUTPUT*100);
                if(nKeyPressed!=-1) {
                    nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
                    std::cout << "nKeyPressed = " << nKeyPressed%(UCHAR_MAX+1) << std::endl;
                }
                if(nKeyPressed==' ')
                    bContinuousUpdates = !bContinuousUpdates;
                else if(nKeyPressed==(int)'q')
                    bExit = true;
#if DEBUG_OUTPUT
                else if(nKeyPressed==' ')
                    break;
            }
#endif //DEBUG_OUTPUT
#endif //DISPLAY_OUTPUT
#if WRITE_IMG_OUTPUT
            pCurrSet->WriteResult(nCurrImageIdx,oCurrEdgeMask);
#endif //WRITE_IMG_OUTPUT
#if EVALUATE_OUTPUT
            //@@@ FULL_THRESH_ANALYSIS oCurrEdgeMask
            if(pCurrSet->m_pEvaluator)
                pCurrSet->m_pEvaluator->AccumulateMetricsFromResult(oCurrEdgeMask,oCurrGTMask,cv::Mat());
#endif //EVALUATE_OUTPUT
            TIMER_INTERNAL_TOC(OverallLoop);
#if DISPLAY_TIMERS
            std::cout << "ImageQuery=" << TIMER_INTERNAL_ELAPSED_MS(ImageQuery) << "ms,  "
                      << "PipelineUpdate=" << TIMER_INTERNAL_ELAPSED_MS(PipelineUpdate) << "ms,  "
                      << "OverallLoop=" << TIMER_INTERNAL_ELAPSED_MS(OverallLoop) << "ms" << std::endl;
#endif //ENABLE_INTERNAL_TIMERS
            ++nCurrImageIdx;
        }
        TIMER_TOC(MainLoop);
        const double dTimeElapsed = TIMER_ELAPSED_MS(MainLoop)/1000;
        const double dAvgFPS = (double)nCurrImageIdx/dTimeElapsed;
        std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSetName << " @ end, " << int(dTimeElapsed) << " sec in-thread (" << (int)floor(dAvgFPS+0.5) << " FPS)" << std::endl;
#if EVALUATE_OUTPUT
        if(pCurrSet->m_pEvaluator) {
            pCurrSet->m_pEvaluator->dTimeElapsed_sec = dTimeElapsed;
            //pCurrSet->m_pEvaluator->WriteMetrics(pCurrSet->m_sResultsPath+"../"+pCurrSet->m_sName+".txt",*pCurrSet);
        }
#endif //EVALUATE_OUTPUT
#if DISPLAY_OUTPUT
        cv::destroyWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyzeSequence caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyzeSequence caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyzeSequence caught unhandled exception\n" << std::endl;}
    g_nActiveThreads--;
    if(pCurrSet.get()) {
        if(DATASET_PRECACHING)
            pCurrSet->StopPrecaching();
        pCurrSet->m_nImagesProcessed.set_value(nCurrImageIdx);
    }
}
#endif //!USE_GPU_IMPL
