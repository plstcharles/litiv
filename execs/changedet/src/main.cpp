// @@@ imgproc gpu algo does not support mipmapping binding yet
// @@@ test compute shader group size vs shared mem usage
// @@@ add opencv hardware instr set check, and impl some stuff in MMX/SSE/SSE2/3/4.1/4.2? (also check popcount and AVX)
// @@@ support non-integer textures top level (alg)? need to replace all ui-stores by float-stores, rest is ok
// @@@ change comma pos list & super constr everywhere
// @@@ change template formatting everywhere

#include "litiv/utils/DatasetEvalUtils.hpp"

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define WRITE_AVI_OUTPUT        0
#define EVALUATE_OUTPUT         1
#define DEBUG_OUTPUT            0
#define DISPLAY_OUTPUT          0
#define DISPLAY_TIMERS          0
////////////////////////////////
#define USE_VIBE                0
#define USE_PBAS                0
#define USE_PAWCS               0
#define USE_LOBSTER             1
#define USE_SUBSENSE            0
////////////////////////////////
#define USE_GLSL_IMPL           1
#define USE_CUDA_IMPL           0
#define USE_OPENCL_IMPL         0
////////////////////////////////
#define DATASET_ID              eDataset_CDnet2014
#ifndef GLOBAL_DATASET_ROOT_PATH
#define DATASET_ROOT_PATH       std::string("/some/dataset/root/path/")
#else //def(GLOBAL_DATASET_ROOT_PATH)
#define DATASET_ROOT_PATH       std::string(GLOBAL_DATASET_ROOT_PATH)
#endif //def(GLOBAL_DATASET_ROOT_PATH)
#define DATASET_RESULTS_PATH    std::string("results")
#define DATASET_PRECACHING      1
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
#define LIMIT_MODEL_TO_SEQUENCE_ROI (USE_LOBSTER||USE_SUBSENSE||USE_PAWCS)
#define BOOTSTRAP_100_FIRST_FRAMES  (USE_LOBSTER||USE_SUBSENSE||USE_PAWCS)
#if (USE_GLSL_IMPL+USE_CUDA_IMPL+USE_OPENCL_IMPL)>1
#error "Must specify a single impl."
#elif (USE_LOBSTER+USE_SUBSENSE+USE_VIBE+USE_PBAS+USE_PAWCS)!=1
#error "Must specify a single algorithm."
#elif USE_VIBE
#include "litiv/video/BackgroundSubtractorViBe.hpp"
#elif USE_PBAS
#include "litiv/video/BackgroundSubtractorPBAS.hpp"
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
#define TARGET_GL_VER_STR "GL_VERSION_" XSTR(TARGET_GL_VER_MAJOR) "_" XSTR(TARGET_GL_VER_MINOR)
#define glewInitErrorCheck { \
    glErrorCheck; \
    glewExperimental = GLEW_EXPERIMENTAL?GL_TRUE:GL_FALSE; \
    if(GLenum __glewerrn=glewInit()!=GLEW_OK) \
        glErrorExt("Failed to init GLEW (code=%d)",__glewerrn); \
    else if(GLenum __errn=glGetError()!=GL_INVALID_ENUM) \
        glErrorExt("Unexpected GLEW init error (code=%d)",__errn); \
    if(!glewIsSupported(TARGET_GL_VER_STR)) \
        glErrorExt("Bad GL core/ext version detected (target is %s)",TARGET_GL_VER_STR); \
}
#if !HAVE_GLFW
#error "missing glfw"
#endif //!HAVE_GLFW
void AnalyzeSequence_GLSL(std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence);
#elif (HAVE_CUDA && USE_CUDA_IMPL)
static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
void AnalyzeSequence(int nThreadIdx, std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence);
#else // bad config
#error "Bad config, trying to use an unavailable impl."
#endif // bad config
#ifndef DATASET_ID
const std::shared_ptr<DatasetUtils::Segm::Video::DatasetInfo> g_pDatasetInfo(new DatasetUtils::Segm::Video::DatasetInfo(
        "GeekSys2D measy02", // m_sDatasetName
        DATASET_ROOT_PATH+"/geeksys/measy02/", // m_sDatasetRootPath
        DATASET_ROOT_PATH+"/geeksys/measy02/"+DATASET_RESULTS_PATH+"/", // m_sResultsRootPath
        "fg_", // m_sResultNamePrefix
        ".png", // m_sResultNameSuffix
        {"original"}, // m_vsWorkBatchPaths
        {}, // m_vsSkippedNameTokens
        {"original"}, // m_vsGrayscaleNameTokens
        USE_GPU_IMPL, // m_bForce4ByteDataAlign
        0.5, // m_dScaleFactor
        DatasetUtils::Segm::Video::eDataset_Custom, // m_eDatasetID
        -1 // m_nResultIdxOffset
));
#else //defined(DATASET_ID)
const std::shared_ptr<DatasetUtils::Segm::Video::DatasetInfo> g_pDatasetInfo = DatasetUtils::Segm::Video::GetDatasetInfo(DatasetUtils::Segm::Video::DATASET_ID,DATASET_ROOT_PATH,DATASET_RESULTS_PATH,USE_GPU_IMPL);
#endif //defined(DATASET_ID)

// @@@@ package x/y/event/code into shared_ptr struct? check opencv 3.0 callback for mouse event?
int g_nLatestMouseX = -1, g_nLatestMouseY = -1;
int *g_pnLatestMouseX = &g_nLatestMouseX, *g_pnLatestMouseY = &g_nLatestMouseY;
void OnMouseEvent(int event, int x, int y, int, void*) {
    if(event!=cv::EVENT_MOUSEMOVE || !x || !y)
        return;
    *g_pnLatestMouseX = x;
    *g_pnLatestMouseY = y;
}

std::atomic_size_t g_nActiveThreads(0);
const size_t g_nMaxThreads = DEFAULT_NB_THREADS;//std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():DEFAULT_NB_THREADS;

int main(int, char**) {
    try {
        std::cout << "Parsing dataset '" << g_pDatasetInfo->m_sDatasetName << "'..." << std::endl;
        std::vector<std::shared_ptr<DatasetUtils::WorkGroup>> vpDatasetGroups = g_pDatasetInfo->ParseDataset();
        size_t nFramesTotal = 0;
        // @@@ check out priority_queue?
        std::multimap<double,std::shared_ptr<DatasetUtils::Segm::Video::Sequence>> mSeqLoads;
        for(auto ppGroupIter=vpDatasetGroups.begin(); ppGroupIter!=vpDatasetGroups.end(); ++ppGroupIter) {
            for(auto ppBatchIter=(*ppGroupIter)->m_vpBatches.begin(); ppBatchIter!=(*ppGroupIter)->m_vpBatches.end(); ++ppBatchIter) {
                auto pSeq = std::dynamic_pointer_cast<DatasetUtils::Segm::Video::Sequence>(*ppBatchIter);
                CV_Assert(pSeq!=nullptr);
                nFramesTotal += pSeq->GetTotalImageCount();
                mSeqLoads.insert(std::make_pair(pSeq->GetExpectedLoad(),pSeq));
            }
        }
        const size_t nSeqTotal = mSeqLoads.size();
        if(nSeqTotal==0 || nFramesTotal==0)
            throw std::runtime_error(cv::format("Could not find any sequences/frames to process for dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str()));
        std::cout << "Parsing complete. [" << vpDatasetGroups.size() << " group(s), " << nSeqTotal << " sequence(s)]\n" << std::endl;
        const time_t nStartupTime = time(nullptr);
        const std::string sStartupTimeStr(asctime(localtime(&nStartupTime)));
        std::cout << "[" << sStartupTimeStr.substr(0,sStartupTimeStr.size()-1) << "]" << std::endl;
        std::cout << "Executing background subtraction with " << ((g_nMaxThreads>nSeqTotal)?nSeqTotal:g_nMaxThreads) << " thread(s)..." << std::endl;
        size_t nSeqProcessed = 0;
        for(auto pSeqIter = mSeqLoads.rbegin(); pSeqIter!=mSeqLoads.rend(); ++pSeqIter) {
            while(g_nActiveThreads>=g_nMaxThreads)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::cout << "\tProcessing [" << ++nSeqProcessed << "/" << nSeqTotal << "] (" << pSeqIter->second->m_sRelativePath << ", L=" << std::scientific << std::setprecision(2) << pSeqIter->first << ")" << std::endl;
            if(DATASET_PRECACHING)
                pSeqIter->second->StartPrecaching(EVALUATE_OUTPUT);
#if (HAVE_GLSL && USE_GLSL_IMPL)
            AnalyzeSequence_GLSL(pSeqIter->second);
#elif (HAVE_CUDA && USE_CUDA_IMPL)
            static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
            static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
            ++g_nActiveThreads;
            std::thread(AnalyzeSequence,nSeqProcessed,pSeqIter->second).detach();
#endif //!USE_GPU_IMPL
        }
        while(g_nActiveThreads>0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        size_t nTotFramesProcessed = 0;
        for(auto pSeqIter = mSeqLoads.rbegin(); pSeqIter!=mSeqLoads.rend(); ++pSeqIter)
            nTotFramesProcessed += pSeqIter->second->m_nImagesProcessed.get_future().get();
        const time_t nShutdownTime = time(nullptr);
        const std::string sShutdownTimeStr(asctime(localtime(&nShutdownTime)));
        std::cout << "[" << sShutdownTimeStr.substr(0,sShutdownTimeStr.size()-1) << "]\n" << std::endl;
        if(EVALUATE_OUTPUT && nTotFramesProcessed==nFramesTotal) {
            std::cout << "Writing evaluation results..." << std::endl;
            g_pDatasetInfo->WriteEvalResults(vpDatasetGroups);

        }
        else if(EVALUATE_OUTPUT && nTotFramesProcessed<nFramesTotal)
            std::cout << "Skipping eval results output, as some sequences were skipped." << std::endl;
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

void AnalyzeSequence_GLSL(std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrFrameIdx = 0;
    size_t nNextFrameIdx = nCurrFrameIdx+1;
    bool bGPUContextInitialized = false;
    try {
        glfwSetErrorCallback(GLFWErrorCallback);
        CV_Assert(pCurrSequence.get() && pCurrSequence->GetTotalImageCount()>1);
        if(pCurrSequence->m_pEvaluator==nullptr && EVALUATE_OUTPUT)
            throw std::runtime_error(cv::format("Missing evaluation impl for video segmentation dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str()));
        const std::string sCurrSeqName = pCurrSequence->m_sName.size()>12?pCurrSequence->m_sName.substr(0,12):pCurrSequence->m_sName;
        const size_t nFrameCount = pCurrSequence->GetTotalImageCount();
        const cv::Mat oROI = LIMIT_MODEL_TO_SEQUENCE_ROI?pCurrSequence->GetROI():cv::Mat();
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
        glewInitErrorCheck;
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
#else //USE_VIBE || USE_PBAS
#error "Missing glsl impl." // ... @@@@@
        const size_t m_nInputChannels = (size_t)oCurrInputFrame.channels();
#if USE_VIBE
        std::shared_ptr<cv::BackgroundSubtractorViBe_GLSL> pAlgo;
        if(m_nInputChannels==3)
            pAlgo = std::shared_ptr<cv::BackgroundSubtractorViBe_GLSL>(new BackgroundSubtractorViBe_GLSL_3ch());
        else
            pAlgo = std::shared_ptr<cv::BackgroundSubtractorViBe_GLSL>(new BackgroundSubtractorViBe_GLSL_1ch());
        const double dDefaultLearningRate = BGSVIBE_DEFAULT_LEARNING_RATE;
#else //USE_PBAS
        std::shared_ptr<cv::BackgroundSubtractorPBAS_GLSL> pAlgo;
        if(m_nInputChannels==3)
            pAlgo = std::shared_ptr<cv::BackgroundSubtractorPBAS_GLSL>(new BackgroundSubtractorPBAS_GLSL_3ch());
        else
            pAlgo = std::shared_ptr<cv::BackgroundSubtractorPBAS_GLSL>(new BackgroundSubtractorPBAS_GLSL_1ch());
        const double dDefaultLearningRate = BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE;
#endif //USE_PBAS
        pAlgo->initialize(oCurrInputFrame);
#endif //USE_VIBE || USE_PBAS
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = pCurrSequence->m_sRelativePath;
        cv::namedWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
#if (WRITE_IMG_OUTPUT || WRITE_AVI_OUTPUT)
#if WRITE_AVI_OUTPUT
        cv::VideoWriter oSegmWriter(pCurrSequence->m_sResultsPath+"../"+pCurrSequence->m_sName+"_segm.avi",CV_FOURCC('F','F','V','1'),30,pCurrSequence->GetImageSize(),false);
#endif //WRITE_AVI_OUTPUT
#endif //(WRITE_IMG_OUTPUT || WRITE_AVI_OUTPUT)
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
        TIMER_TIC(MainLoop);
        while(nNextFrameIdx<=nFrameCount) {
            if(!((nCurrFrameIdx+1)%100))
                std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nFrameCount)) << nCurrFrameIdx+1 << "/" << nFrameCount << "   [GPU]" << std::endl;
            const double dCurrLearningRate = (BOOTSTRAP_100_FIRST_FRAMES&&nCurrFrameIdx<=100)?1:dDefaultLearningRate;
            TIMER_INTERNAL_TIC(OverallLoop);
            TIMER_INTERNAL_TIC(PipelineUpdate);
            pAlgo->apply_async(oNextInputFrame,dCurrLearningRate);
            TIMER_INTERNAL_TOC(PipelineUpdate);
#if USE_GLSL_EVALUATION
            pGLSLAlgoEvaluator->apply_async(oNextGTMask);
#endif //USE_GLSL_EVALUATION
            TIMER_INTERNAL_TIC(VideoQuery);
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
            TIMER_INTERNAL_TOC(VideoQuery);
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
            if(oDisplayImage.cols>1920 || oDisplayImage.rows>1080)
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
#if WRITE_AVI_OUTPUT
            oSegmWriter.write(oLastFGMask);
#endif //WRITE_AVI_OUTPUT
#if WRITE_IMG_OUTPUT
            pCurrSequence->WriteResult(nCurrFrameIdx,oLastFGMask);
#endif //WRITE_IMG_OUTPUT
#if (EVALUATE_OUTPUT && (!USE_GLSL_EVALUATION || VALIDATE_GPU_EVALUATION))
            if(pCurrSequence->m_pEvaluator)
                pCurrSequence->m_pEvaluator->AccumulateMetricsFromResult(oCurrFGMask,oCurrGTMask,oROI);
#endif //(EVALUATE_OUTPUT && (!USE_GLSL_EVALUATION || VALIDATE_GPU_EVALUATION))
            TIMER_INTERNAL_TOC(OverallLoop);
#if DISPLAY_TIMERS
            std::cout << "VideoQuery=" << TIMER_INTERNAL_ELAPSED_MS(VideoQuery) << "ms,  "
            << "PipelineUpdate=" << TIMER_INTERNAL_ELAPSED_MS(PipelineUpdate) << "ms,  "
            << "OverallLoop=" << TIMER_INTERNAL_ELAPSED_MS(OverallLoop) << "ms" << std::endl;
#endif //ENABLE_INTERNAL_TIMERS
            ++nCurrFrameIdx;
        }
        TIMER_TOC(MainLoop);
        const double dTimeElapsed = TIMER_ELAPSED_MS(MainLoop)/1000;
        const double dAvgFPS = (double)nCurrFrameIdx/dTimeElapsed;
        std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ end, " << int(dTimeElapsed) << " sec in-thread (" << (int)floor(dAvgFPS+0.5) << " FPS)" << std::endl;
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
        if(DATASET_PRECACHING)
            pCurrSequence->StopPrecaching();
        pCurrSequence->m_nImagesProcessed.set_value(nCurrFrameIdx);
    }
}
#elif (HAVE_CUDA && USE_CUDA_IMPL)
static_assert(false,"missing impl");
#elif (HAVE_OPENCL && USE_OPENCL_IMPL)
static_assert(false,"missing impl");
#elif !USE_GPU_IMPL
void AnalyzeSequence(int nThreadIdx, std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrFrameIdx = 0;
    try {
        CV_Assert(pCurrSequence.get() && pCurrSequence->GetTotalImageCount()>1);
        if(pCurrSequence->m_pEvaluator==nullptr && EVALUATE_OUTPUT)
            throw std::runtime_error(cv::format("Missing evaluation impl for video segmentation dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str()));
        const std::string sCurrSeqName = pCurrSequence->m_sName.size()>12?pCurrSequence->m_sName.substr(0,12):pCurrSequence->m_sName;
        const size_t nFrameCount = pCurrSequence->GetTotalImageCount();
        const cv::Mat oROI = LIMIT_MODEL_TO_SEQUENCE_ROI?pCurrSequence->GetROI():cv::Mat();
        cv::Mat oCurrInputFrame = pCurrSequence->GetInputFromIndex(nCurrFrameIdx).clone();
        CV_Assert(!oCurrInputFrame.empty());
        CV_Assert(oCurrInputFrame.isContinuous());
#if NEED_GT_MASK
        cv::Mat oCurrGTMask = pCurrSequence->GetGTFromIndex(nCurrFrameIdx).clone();
        CV_Assert(!oCurrGTMask.empty() && oCurrGTMask.isContinuous());
#endif //NEED_GT_MASK
        cv::Mat oCurrFGMask(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(0));
#if DISPLAY_OUTPUT
        cv::Mat oCurrBGImg;
#endif //DISPLAY_OUTPUT
#if USE_LOBSTER
        std::shared_ptr<BackgroundSubtractorLOBSTER> pAlgo(new BackgroundSubtractorLOBSTER());
        const double dDefaultLearningRate = BGSLOBSTER_DEFAULT_LEARNING_RATE;
        pAlgo->initialize(oCurrInputFrame,oROI);
#elif USE_SUBSENSE
        std::shared_ptr<BackgroundSubtractorSuBSENSE> pAlgo(new BackgroundSubtractorSuBSENSE());
        const double dDefaultLearningRate = 0;
        pAlgo->initialize(oCurrInputFrame,oROI);
#elif USE_PAWCS
        std::shared_ptr<BackgroundSubtractorPAWCS> pAlgo(new BackgroundSubtractorPAWCS());
        const double dDefaultLearningRate = 0;
        pAlgo->initialize(oCurrInputFrame,oROI);
#else //USE_VIBE || USE_PBAS
        const size_t m_nInputChannels = (size_t)oCurrInputFrame.channels();
#if USE_VIBE
        std::shared_ptr<BackgroundSubtractorViBe> pAlgo;
        if(m_nInputChannels==3)
            pAlgo = std::shared_ptr<BackgroundSubtractorViBe>(new BackgroundSubtractorViBe_3ch());
        else
            pAlgo = std::shared_ptr<BackgroundSubtractorViBe>(new BackgroundSubtractorViBe_1ch());
        const double dDefaultLearningRate = BGSVIBE_DEFAULT_LEARNING_RATE;
#else //USE_PBAS
        std::shared_ptr<BackgroundSubtractorPBAS> pAlgo;
        if(m_nInputChannels==3)
            pAlgo = std::shared_ptr<BackgroundSubtractorPBAS>(new BackgroundSubtractorPBAS_3ch());
        else
            pAlgo = std::shared_ptr<BackgroundSubtractorPBAS>(new BackgroundSubtractorPBAS_1ch());
        const double dDefaultLearningRate = BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE;
#endif //USE_PBAS
        pAlgo->initialize(oCurrInputFrame);
#endif //USE_VIBE || USE_PBAS
#if (DEBUG_OUTPUT && (USE_LOBSTER || USE_SUBSENSE || USE_PAWCS))
        cv::FileStorage oDebugFS = cv::FileStorage(pCurrSequence->m_sResultsPath+"../"+pCurrSequence->m_sName+"_debug.yml",cv::FileStorage::WRITE);
        pAlgo->m_pDebugFS = &oDebugFS;
        pAlgo->m_sDebugName = pCurrSequence->m_sName;
        g_pnLatestMouseX = &pAlgo->m_nDebugCoordX;
        g_pnLatestMouseY = &pAlgo->m_nDebugCoordY;
        std::string sMouseDebugDisplayName = pCurrSequence->m_sName + " [MOUSE DEBUG]";
        cv::namedWindow(sMouseDebugDisplayName,0);
        cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
#endif //(DEBUG_OUTPUT && (USE_LOBSTER || USE_SUBSENSE || USE_PAWCS))
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = pCurrSequence->m_sRelativePath;
        cv::namedWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
#if (WRITE_IMG_OUTPUT || WRITE_AVI_OUTPUT)
#if WRITE_AVI_OUTPUT
        cv::VideoWriter oSegmWriter(pCurrSequence->m_sResultsPath+"../"+pCurrSequence->m_sName+"_segm.avi",CV_FOURCC('F','F','V','1'),30,pCurrSequence->GetImageSize(),false);
#endif //WRITE_AVI_OUTPUT
#endif //(WRITE_IMG_OUTPUT || WRITE_AVI_OUTPUT)
        TIMER_TIC(MainLoop);
        while(nCurrFrameIdx<nFrameCount) {
            if(!((nCurrFrameIdx+1)%100))
                std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nFrameCount)) << nCurrFrameIdx+1 << "/" << nFrameCount << "   [T=" << nThreadIdx << "]" << std::endl;
            const double dCurrLearningRate = (BOOTSTRAP_100_FIRST_FRAMES&&nCurrFrameIdx<=100)?1:dDefaultLearningRate;
            TIMER_INTERNAL_TIC(OverallLoop);
            TIMER_INTERNAL_TIC(VideoQuery);
            oCurrInputFrame = pCurrSequence->GetInputFromIndex(nCurrFrameIdx);
#if DEBUG_OUTPUT
            cv::imshow(sMouseDebugDisplayName,oCurrInputFrame);
#endif //DEBUG_OUTPUT
#if NEED_GT_MASK
            oCurrGTMask = pCurrSequence->GetGTFromIndex(nCurrFrameIdx);
#endif //NEED_GT_MASK
            TIMER_INTERNAL_TOC(VideoQuery);
            TIMER_INTERNAL_TIC(PipelineUpdate);
            pAlgo->apply(oCurrInputFrame,oCurrFGMask,dCurrLearningRate);
            TIMER_INTERNAL_TOC(PipelineUpdate);
            if(!oROI.empty())
                cv::bitwise_or(oCurrFGMask,UCHAR_MAX/2,oCurrFGMask,oROI==0);
#if DISPLAY_OUTPUT
            pAlgo->getBackgroundImage(oCurrBGImg);
            if(!oROI.empty())
                cv::bitwise_or(oCurrBGImg,UCHAR_MAX/2,oCurrBGImg,oROI==0);
            cv::Mat oDisplayFrame = DatasetUtils::GetDisplayImage(oCurrInputFrame,oCurrBGImg,pCurrSequence->m_pEvaluator?pCurrSequence->m_pEvaluator->GetColoredSegmMaskFromResult(oCurrFGMask,oCurrGTMask,oROI):oCurrFGMask,nCurrFrameIdx,cv::Point(*g_pnLatestMouseX,*g_pnLatestMouseY));
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
#if WRITE_AVI_OUTPUT
            oSegmWriter.write(oCurrFGMask);
#endif //WRITE_AVI_OUTPUT
#if WRITE_IMG_OUTPUT
            pCurrSequence->WriteResult(nCurrFrameIdx,oCurrFGMask);
#endif //WRITE_IMG_OUTPUT
#if EVALUATE_OUTPUT
            if(pCurrSequence->m_pEvaluator)
                pCurrSequence->m_pEvaluator->AccumulateMetricsFromResult(oCurrFGMask,oCurrGTMask,oROI);
#endif //EVALUATE_OUTPUT
            TIMER_INTERNAL_TOC(OverallLoop);
#if DISPLAY_TIMERS
            std::cout << "VideoQuery=" << TIMER_INTERNAL_ELAPSED_MS(VideoQuery) << "ms,  "
                      << "PipelineUpdate=" << TIMER_INTERNAL_ELAPSED_MS(PipelineUpdate) << "ms,  "
                      << "OverallLoop=" << TIMER_INTERNAL_ELAPSED_MS(OverallLoop) << "ms" << std::endl;
#endif //ENABLE_INTERNAL_TIMERS
            ++nCurrFrameIdx;
        }
        TIMER_TOC(MainLoop);
        const double dTimeElapsed = TIMER_ELAPSED_MS(MainLoop)/1000;
        const double dAvgFPS = (double)nCurrFrameIdx/dTimeElapsed;
        std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ end, " << int(dTimeElapsed) << " sec in-thread (" << (int)floor(dAvgFPS+0.5) << " FPS)" << std::endl;
#if EVALUATE_OUTPUT
        if(pCurrSequence->m_pEvaluator) {
            pCurrSequence->m_pEvaluator->dTimeElapsed_sec = dTimeElapsed;
            //pCurrSequence->m_pEvaluator->WriteMetrics(pCurrSequence->m_sResultsPath+"../"+pCurrSequence->m_sName+".txt",*pCurrSequence);
        }
#endif //EVALUATE_OUTPUT
#if DISPLAY_OUTPUT
        cv::destroyWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
#if DEBUG_OUTPUT
        cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
        cv::destroyWindow(sMouseDebugDisplayName);
        g_pnLatestMouseX = &g_nLatestMouseX;
        g_pnLatestMouseY = &g_nLatestMouseY;
#endif //DEBUG_OUTPUT
    }
    catch(const cv::Exception& e) {std::cout << "\nAnalyzeSequence caught cv::Exception:\n" << e.what() << "\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\nAnalyzeSequence caught std::exception:\n" << e.what() << "\n" << std::endl;}
    catch(...) {std::cout << "\nAnalyzeSequence caught unhandled exception\n" << std::endl;}
    g_nActiveThreads--;
    if(pCurrSequence.get()) {
        if(DATASET_PRECACHING)
            pCurrSequence->StopPrecaching();
        pCurrSequence->m_nImagesProcessed.set_value(nCurrFrameIdx);
    }
}
#endif //!USE_GPU_IMPL
