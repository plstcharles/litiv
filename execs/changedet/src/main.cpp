// @@@ imgproc gpu algo does not support mipmapping binding yet
// @@@ test compute shader group size vs shared mem usage
// @@@ add opencv hardware instr set check, and impl some stuff in MMX/SSE/SSE2/3/4.1/4.2? (also check popcount and AVX)
// @@@ support non-integer textures top level (alg)? need to replace all ui-stores by float-stores, rest is ok

#include <litiv/utils/DatasetEvalUtils.hpp>

////////////////////////////////
#define WRITE_IMG_OUTPUT        0
#define WRITE_AVI_OUTPUT        0
#define EVALUATE_OUTPUT         1
#define DEBUG_OUTPUT            0
#define DISPLAY_OUTPUT          1
#define DISPLAY_TIMERS          0
////////////////////////////////
#define USE_VIBE                0
#define USE_PBAS                0
#define USE_PAWCS               0
#define USE_LOBSTER             1
#define USE_SUBSENSE            0
////////////////////////////////
#if EVALUATE_OUTPUT
#define WRITE_METRICS           1
#if HAVE_GLSL
#define GLSL_EVALUATION         1
#define VALIDATE_EVALUATION     0
#endif //HAVE_GLSL
#endif //EVALUATE_OUTPUT
////////////////////////////////
#define DATASET_ID              eDataset_CDnet2014
#define DATASET_ROOT_PATH       std::string("/shared2/datasets/")
#define DATASET_RESULTS_PATH    std::string("results")
#define DATASET_PRECACHING      1
////////////////////////////////

#define NEED_LAST_GT_MASK (DISPLAY_OUTPUT || (WRITE_METRICS && (!GLSL_EVALUATION || VALIDATE_EVALUATION)))
#define NEED_GT_MASK (DISPLAY_OUTPUT || WRITE_METRICS)
#define NEED_FG_MASK (DISPLAY_OUTPUT || WRITE_AVI_OUTPUT || WRITE_IMG_OUTPUT || ((!GLSL_EVALUATION || VALIDATE_EVALUATION) && WRITE_METRICS))
#define LIMIT_MODEL_TO_SEQUENCE_ROI (USE_LOBSTER||USE_SUBSENSE||USE_PAWCS)
#define BOOTSTRAP_100_FIRST_FRAMES  (USE_LOBSTER||USE_SUBSENSE||USE_PAWCS)
#if (USE_LOBSTER+USE_SUBSENSE+USE_VIBE+USE_PBAS+USE_PAWCS)!=1
#error "Must specify a single algorithm."
#elif USE_VIBE
#include "litiv/video/BackgroundSubtractorViBe_1ch.hpp"
#include "litiv/video/BackgroundSubtractorViBe_3ch.hpp"
#elif USE_PBAS
#include "litiv/video/BackgroundSubtractorPBAS_1ch.hpp"
#include "litiv/video/BackgroundSubtractorPBAS_3ch.hpp"
#elif USE_PAWCS
#include "litiv/video/BackgroundSubtractorPAWCS.hpp"
#elif USE_LOBSTER
#include "litiv/video/BackgroundSubtractorLOBSTER.hpp"
#elif USE_SUBSENSE
#include "litiv/video/BackgroundSubtractorSuBSENSE.hpp"
#endif //USE_...
#if (HAVE_GPU_SUPPORT && DEFAULT_NB_THREADS>1)
#error "Cannot support multithreading + gpu exec"
#endif //(HAVE_GPU_SUPPORT && DEFAULT_NB_THREADS>1)
#if (DEBUG_OUTPUT && (HAVE_GPU_SUPPORT || DEFAULT_NB_THREADS>1))
#error "Cannot debug output with GPU support or with more than one thread."
#endif //(DEBUG_OUTPUT && (HAVE_GPU_SUPPORT || DEFAULT_NB_THREADS>1))
#if DISPLAY_TIMERS
#define TIMER_INTERNAL_TIC(x) TIMER_TIC(x)
#define TIMER_INTERNAL_TOC(x) TIMER_TOC(x)
#define TIMER_INTERNAL_ELAPSED_MS(x) TIMER_ELAPSED_MS(x)
#else //!ENABLE_INTERNAL_TIMERS
#define TIMER_INTERNAL_TIC(x)
#define TIMER_INTERNAL_TOC(x)
#define TIMER_INTERNAL_ELAPSED_MS(x)
#endif //!ENABLE_INTERNAL_TIMERS

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
const std::shared_ptr<DatasetUtils::Segm::Video::DatasetInfo> g_pDatasetInfo = DatasetUtils::Segm::Video::GetDatasetInfo(DatasetUtils::Segm::Video::DATASET_ID,DATASET_ROOT_PATH,DATASET_RESULTS_PATH,HAVE_GPU_SUPPORT);
const std::shared_ptr<DatasetUtils::Segm::SegmEvaluator> g_pEvaluator = std::dynamic_pointer_cast<DatasetUtils::Segm::SegmEvaluator>(g_pDatasetInfo->m_pEvaluator);

int AnalyzeSequence_GPU(std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence, const std::string& sCurrResultsPath);
int AnalyzeSequence(int nThreadIdx, std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence, const std::string& sCurrResultsPath);

int main(int, char**) {
    try {
        if(g_pEvaluator==nullptr && EVALUATE_OUTPUT)
            throw std::runtime_error(cv::format("Missing evaluation impl for video segmentation dataset '%s'",g_pDatasetInfo->m_sDatasetName.c_str()));
        std::cout << "Parsing dataset '" << g_pDatasetInfo->m_sDatasetName << "'..." << std::endl;
        std::vector<std::shared_ptr<DatasetUtils::WorkGroup>> vpDatasetGroups = DatasetUtils::DatasetInfoBase::ParseDataset(*g_pDatasetInfo);
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
        const std::string sCurrResultsPath = g_pDatasetInfo->m_sResultsRootPath;
        PlatformUtils::CreateDirIfNotExist(sCurrResultsPath);
        for(size_t c = 0; c<vpDatasetGroups.size(); ++c)
            PlatformUtils::CreateDirIfNotExist(sCurrResultsPath+vpDatasetGroups[c]->m_sName+"/");
        const time_t nStartupTime = time(nullptr);
        const std::string sStartupTimeStr(asctime(localtime(&nStartupTime)));
        std::cout << "[" << sStartupTimeStr.substr(0,sStartupTimeStr.size()-1) << "]" << std::endl;
        std::cout << "Executing background subtraction with " << ((g_nMaxThreads>nSeqTotal)?nSeqTotal:g_nMaxThreads) << " thread(s)..." << std::endl;
        size_t nSeqProcessed = 0;
        for(auto pSeqIter = mSeqLoads.rbegin(); pSeqIter!=mSeqLoads.rend(); ++pSeqIter) {
            while(g_nActiveThreads>=g_nMaxThreads)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::cout << "\tProcessing [" << ++nSeqProcessed << "/" << nSeqTotal << "] (" << pSeqIter->second->m_sGroupName << ":" << pSeqIter->second->m_sName << ", L=" << std::scientific << std::setprecision(2) << pSeqIter->first << ")" << std::endl;
#if HAVE_GPU_SUPPORT
            AnalyzeSequence_GPU(pSeqIter->second,sCurrResultsPath);
#else //!HAVE_GPU_SUPPORT
            ++g_nActiveThreads;
            std::thread(AnalyzeSequence,nSeqProcessed,pSeqIter->second,sCurrResultsPath).detach();
#endif //!HAVE_GPU_SUPPORT
        }
        while(g_nActiveThreads>0)
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        const time_t nShutdownTime = time(nullptr);
        const std::string sShutdownTimeStr(asctime(localtime(&nShutdownTime)));
        std::cout << "[" << sShutdownTimeStr.substr(0,sShutdownTimeStr.size()-1) << "]\n" << std::endl;
        if(WRITE_METRICS) {
            std::cout << "Summing and writing metrics results..." << std::endl;
            for(size_t c=0; c<vpDatasetGroups.size(); ++c) {
                if(!vpDatasetGroups[c]->m_vpBatches.empty()) {
                    for(size_t s=0; s<vpDatasetGroups[c]->m_vpBatches.size(); ++s)
                        DatasetUtils::Segm::WriteMetrics(sCurrResultsPath+vpDatasetGroups[c]->m_sName+"/"+vpDatasetGroups[c]->m_vpBatches[s]->m_sName+".txt",dynamic_cast<const DatasetUtils::Segm::SegmWorkBatch&>(*vpDatasetGroups[c]->m_vpBatches[s]));
                    std::sort(vpDatasetGroups[c]->m_vpBatches.begin(),vpDatasetGroups[c]->m_vpBatches.end(),DatasetUtils::WorkBatch::compare<DatasetUtils::WorkBatch>);
                    DatasetUtils::Segm::WriteMetrics(sCurrResultsPath+vpDatasetGroups[c]->m_sName+".txt",*vpDatasetGroups[c]);
                    std::cout << std::endl;
                }
            }
            std::sort(vpDatasetGroups.begin(),vpDatasetGroups.end(),&DatasetUtils::WorkBatch::compare<DatasetUtils::WorkBatch>);
            DatasetUtils::Segm::WriteMetrics(sCurrResultsPath+"overall.txt",vpDatasetGroups);
        }
        std::cout << "All done." << std::endl;
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl;}
    return 0;
}

#if HAVE_GPU_SUPPORT
int AnalyzeSequence_GPU(std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence, const std::string& sCurrResultsPath) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrFrameIdx = 0;
    size_t nNextFrameIdx = nCurrFrameIdx+1;
    bool bGPUContextInitialized = false;
    try {
        CV_Assert(pCurrSequence.get() && pCurrSequence->GetTotalImageCount()>1 && !sCurrResultsPath.empty());
        if(DATASET_PRECACHING)
            pCurrSequence->StartPrecaching();
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
#if HAVE_GLSL
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
        std::unique_ptr<GLFWwindow,void(*)(GLFWwindow*)> pWindow(glfwCreateWindow(oWindowSize.width,oWindowSize.height,"changedet_gpu",nullptr,nullptr),glfwDestroyWindow);
        if(!pWindow)
            glError("Failed to create window via GLFW");
        glfwMakeContextCurrent(pWindow.get());
        glewInitErrorCheck;
#endif //HAVE_GLSL
#if USE_LOBSTER
        std::shared_ptr<BackgroundSubtractorLOBSTER> pBGS(new BackgroundSubtractorLOBSTER());
        const double dDefaultLearningRate = BGSLOBSTER_DEFAULT_LEARNING_RATE;
        pBGS->initialize(oCurrInputFrame,oROI);
#elif USE_SUBSENSE
        std::shared_ptr<BackgroundSubtractorSuBSENSE> pBGS(new BackgroundSubtractorSuBSENSE());
        const double dDefaultLearningRate = 0;
        pBGS->initialize(oCurrInputFrame,oROI);
#elif USE_PAWCS
        std::shared_ptr<BackgroundSubtractorPAWCS> pBGS(new BackgroundSubtractorPAWCS());
        const double dDefaultLearningRate = 0;
        pBGS->initialize(oCurrInputFrame,oROI);
#else //USE_VIBE || USE_PBAS
        const size_t m_nInputChannels = (size_t)oCurrInputFrame.channels();
#if USE_VIBE
        std::shared_ptr<cv::BackgroundSubtractorViBe> pBGS;
        if(m_nInputChannels==3)
            pBGS = std::shared_ptr<cv::BackgroundSubtractorViBe>(new BackgroundSubtractorViBe_3ch());
        else
            pBGS = std::shared_ptr<cv::BackgroundSubtractorViBe>(new BackgroundSubtractorViBe_1ch());
        const double dDefaultLearningRate = BGSVIBE_DEFAULT_LEARNING_RATE;
#else //USE_PBAS
        std::shared_ptr<cv::BackgroundSubtractorPBAS> pBGS;
        if(m_nInputChannels==3)
            pBGS = std::shared_ptr<cv::BackgroundSubtractorPBAS>(new BackgroundSubtractorPBAS_3ch());
        else
            pBGS = std::shared_ptr<cv::BackgroundSubtractorPBAS>(new BackgroundSubtractorPBAS_1ch());
        const double dDefaultLearningRate = BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE;
#endif //USE_PBAS
        pBGS->initialize(oCurrInputFrame);
#endif //USE_VIBE || USE_PBAS
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = pCurrSequence->m_sGroupName + std::string(" -- ") + pCurrSequence->m_sName;
        cv::namedWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
#if (WRITE_IMG_OUTPUT || WRITE_AVI_OUTPUT)
        PlatformUtils::CreateDirIfNotExist(sCurrResultsPath+pCurrSequence->m_sGroupName+"/"+pCurrSequence->m_sName+"/");
#if WRITE_AVI_OUTPUT
        cv::VideoWriter oSegmWriter(sCurrResultsPath+pCurrSequence->m_sGroupName+"/"+pCurrSequence->m_sName+"_segm.avi",CV_FOURCC('F','F','V','1'),30,pCurrSequence->GetImageSize(),false);
#endif //WRITE_AVI_OUTPUT
#endif //(WRITE_IMG_OUTPUT || WRITE_AVI_OUTPUT)
#if HAVE_GLSL
        std::shared_ptr<GLImageProcAlgo> pBGS_GPU = std::dynamic_pointer_cast<GLImageProcAlgo>(pBGS);
        if(pBGS_GPU==nullptr)
            glError("Video segmentation algorithm has no GLImageProcAlgo interface");
        pBGS_GPU->setOutputFetching(NEED_FG_MASK);
        if(!pBGS_GPU->getIsUsingDisplay() && DISPLAY_OUTPUT) // @@@@ determine in advance to hint window to hide? or just always hide, and show when needed?
            glfwHideWindow(pWindow.get());
#if (GLSL_EVALUATION && WRITE_METRICS)
        std::shared_ptr<DatasetUtils::Segm::SegmEvaluator::GLSegmEvaluator> pBGS_GPU_EVAL = std::dynamic_pointer_cast<DatasetUtils::Segm::SegmEvaluator::GLSegmEvaluator>(g_pEvaluator->CreateGLEvaluator(pBGS_GPU,nFrameCount));
        if(pBGS_GPU_EVAL==nullptr)
            glError("Video segmentation evaluation algorithm has no GLSegmEvaluator interface");
        pBGS_GPU_EVAL->initialize(oCurrGTMask,oROI.empty()?cv::Mat(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(255)):oROI);
        oWindowSize.width *= pBGS_GPU_EVAL->m_nSxSDisplayCount;
#else //!(GLSL_EVALUATION && WRITE_METRICS)
        oWindowSize.width *= pBGS_GPU->m_nSxSDisplayCount;
#endif //!(GLSL_EVALUATION && WRITE_METRICS)
        glfwSetWindowSize(pWindow.get(),oWindowSize.width,oWindowSize.height);
        glViewport(0,0,oWindowSize.width,oWindowSize.height);
#endif //HAVE_GLSL
        TIMER_TIC(MainLoop);
        while(nNextFrameIdx<=nFrameCount) {
            if(!((nCurrFrameIdx+1)%100))
                std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nFrameCount)) << nCurrFrameIdx+1 << "/" << nFrameCount << "   [GPU]" << std::endl;
            const double dCurrLearningRate = (BOOTSTRAP_100_FIRST_FRAMES&&nCurrFrameIdx<=100)?1:dDefaultLearningRate;
            TIMER_INTERNAL_TIC(OverallLoop);
            TIMER_INTERNAL_TIC(PipelineUpdate);
            pBGS->apply_async(oNextInputFrame,dCurrLearningRate);
            TIMER_INTERNAL_TOC(PipelineUpdate);
#if (GLSL_EVALUATION && WRITE_METRICS)
            pBGS_GPU_EVAL->apply_async(oNextGTMask);
#endif //(GLSL_EVALUATION && WRITE_METRICS)
            TIMER_INTERNAL_TIC(VideoQuery);
#if DISPLAY_OUTPUT
            oCurrInputFrame.copyTo(oLastInputFrame);
            const cv::Mat& oInputFrame = oLastInputFrame;
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
            const cv::Mat& oGTMask = oLastGTMask;
            oNextGTMask.copyTo(oCurrGTMask);
#endif //NEED_LAST_GT_MASK
            if(nNextFrameIdx<nFrameCount)
                oNextGTMask = pCurrSequence->GetGTFromIndex(nNextFrameIdx);
#endif //NEED_GT_MASK
            TIMER_INTERNAL_TOC(VideoQuery);
#if HAVE_GLSL
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
#endif //HAVE_GLSL
#if NEED_FG_MASK
            pBGS->getLatestForegroundMask(oLastFGMask);
            if(!oROI.empty())
                cv::bitwise_or(oLastFGMask,UCHAR_MAX/2,oLastFGMask,oROI==0);
            const cv::Mat& oFGMask = oLastFGMask;
#endif //NEED_FG_MASK
#if DISPLAY_OUTPUT
            pBGS->getBackgroundImage(oLastBGImg);
            if(!oROI.empty())
                cv::bitwise_or(oLastBGImg,UCHAR_MAX/2,oLastBGImg,oROI==0);
            const cv::Mat& oBGImg = oLastBGImg;
            cv::Mat oDisplayFrame = DatasetUtils::Segm::GetDisplayImage(oInputFrame,oBGImg,g_pEvaluator?g_pEvaluator->GetColoredSegmMaskFromResult(oFGMask,oGTMask,oROI):oFGMask,oROI,nCurrFrameIdx);
            cv::Mat oDisplayFrameResized;
            if(oDisplayFrame.cols>1280 || oDisplayFrame.rows>960)
                cv::resize(oDisplayFrame,oDisplayFrameResized,cv::Size(oDisplayFrame.cols/2,oDisplayFrame.rows/2));
            else
                oDisplayFrameResized = oDisplayFrame;
            cv::imshow(sDisplayName,oDisplayFrameResized);
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
#if WRITE_AVI_OUTPUT
            oSegmWriter.write(oFGMask);
#endif //WRITE_AVI_OUTPUT
#if WRITE_IMG_OUTPUT
            DatasetUtils::Segm::Video::WriteResult(sCurrResultsPath,pCurrSequence->m_sGroupName,pCurrSequence->m_sName,g_pDatasetInfo->m_sResultFrameNamePrefix,nCurrFrameIdx+g_pDatasetInfo->m_nResultIdxOffset,g_pDatasetInfo->m_sResultFrameNameSuffix,oFGMask);
#endif //WRITE_IMG_OUTPUT
#if (WRITE_METRICS && (!GLSL_EVALUATION || VALIDATE_EVALUATION))
            DatasetUtils::Segm::Video::CDnet::AccumulateMetricsFromResult(oFGMask,oGTMask,oROI,pCurrSequence->m_oMetrics);
#endif //(WRITE_METRICS && (!GLSL_EVALUATION || VALIDATE_GLSL_EVALUATION))
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
        const double dAvgFPS = (double)nFrameCount/dTimeElapsed;
        std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ end, " << int(dTimeElapsed) << " sec in-thread (" << (int)floor(dAvgFPS+0.5) << " FPS)" << std::endl;
#if WRITE_METRICS
#if GLSL_EVALUATION
#if VALIDATE_EVALUATION
        printf("cpu eval:\n\tnTP=%" PRIu64 ", nTN=%" PRIu64 ", nFP=%" PRIu64 ", nFN=%" PRIu64 ", nSE=%" PRIu64 ", tot=%" PRIu64 "\n",pCurrSequence->nTP,pCurrSequence->nTN,pCurrSequence->nFP,pCurrSequence->nFN,pCurrSequence->nSE,pCurrSequence->nTP+pCurrSequence->nTN+pCurrSequence->nFP+pCurrSequence->nFN);
#endif //VALIDATE_GLSL_EVALUATION
        pCurrSequence->m_oMetrics = pBGS_GPU_EVAL->getCumulativeMetrics();
#if VALIDATE_EVALUATION
        printf("gpu eval:\n\tnTP=%" PRIu64 ", nTN=%" PRIu64 ", nFP=%" PRIu64 ", nFN=%" PRIu64 ", nSE=%" PRIu64 ", tot=%" PRIu64 "\n",pCurrSequence->nTP,pCurrSequence->nTN,pCurrSequence->nFP,pCurrSequence->nFN,pCurrSequence->nSE,pCurrSequence->nTP+pCurrSequence->nTN+pCurrSequence->nFP+pCurrSequence->nFN);
#endif //VALIDATE_GLSL_EVALUATION
#endif //GLSL_EVALUATION
        pCurrSequence->m_oMetrics.dTimeElapsed_sec = dTimeElapsed;
        DatasetUtils::Segm::WriteMetrics(sCurrResultsPath+pCurrSequence->m_sGroupName+"/"+pCurrSequence->m_sName+".txt",*pCurrSequence);
#endif //WRITE_METRICS
#if DISPLAY_OUTPUT
        cv::destroyWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
    }
#if HAVE_GLSL
    catch(const GLException& e) {std::cout << "\n!!!!!!!!!!!!!!\nAnalyzeSequence caught GLException:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl;}
#endif //HAVE_GLSL
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nAnalyzeSequence caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nAnalyzeSequence caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nAnalyzeSequence caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl;}
    if(bGPUContextInitialized) {
#if HAVE_GLSL
        glfwTerminate();
#endif //HAVE_GLSL
    }
    return 0;
}
#else //!HAVE_GPU_SUPPORT
int AnalyzeSequence(int nThreadIdx, std::shared_ptr<DatasetUtils::Segm::Video::Sequence> pCurrSequence, const std::string& sCurrResultsPath) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
    size_t nCurrFrameIdx = 0;
    try {
        CV_Assert(pCurrSequence.get() && pCurrSequence->GetTotalImageCount()>1 && !sCurrResultsPath.empty());
        if(DATASET_PRECACHING)
            pCurrSequence->StartPrecaching();
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
        std::shared_ptr<BackgroundSubtractorLOBSTER> pBGS(new BackgroundSubtractorLOBSTER());
        const double dDefaultLearningRate = BGSLOBSTER_DEFAULT_LEARNING_RATE;
        pBGS->initialize(oCurrInputFrame,oROI);
#elif USE_SUBSENSE
        std::shared_ptr<BackgroundSubtractorSuBSENSE> pBGS(new BackgroundSubtractorSuBSENSE());
        const double dDefaultLearningRate = 0;
        pBGS->initialize(oCurrInputFrame,oROI);
#elif USE_PAWCS
        std::shared_ptr<BackgroundSubtractorPAWCS> pBGS(new BackgroundSubtractorPAWCS());
        const double dDefaultLearningRate = 0;
        pBGS->initialize(oCurrInputFrame,oROI);
#else //USE_VIBE || USE_PBAS
        const size_t m_nInputChannels = (size_t)oCurrInputFrame.channels();
#if USE_VIBE
        std::shared_ptr<cv::BackgroundSubtractorViBe> pBGS;
        if(m_nInputChannels==3)
            pBGS = std::shared_ptr<cv::BackgroundSubtractorViBe>(new BackgroundSubtractorViBe_3ch());
        else
            pBGS = std::shared_ptr<cv::BackgroundSubtractorViBe>(new BackgroundSubtractorViBe_1ch());
        const double dDefaultLearningRate = BGSVIBE_DEFAULT_LEARNING_RATE;
#else //USE_PBAS
        std::shared_ptr<cv::BackgroundSubtractorPBAS> pBGS;
        if(m_nInputChannels==3)
            pBGS = std::shared_ptr<cv::BackgroundSubtractorPBAS>(new BackgroundSubtractorPBAS_3ch());
        else
            pBGS = std::shared_ptr<cv::BackgroundSubtractorPBAS>(new BackgroundSubtractorPBAS_1ch());
        const double dDefaultLearningRate = BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE;
#endif //USE_PBAS
        pBGS->initialize(oCurrInputFrame);
#endif //USE_VIBE || USE_PBAS
#if (DEBUG_OUTPUT && (USE_LOBSTER || USE_SUBSENSE || USE_PAWCS))
        cv::FileStorage oDebugFS = cv::FileStorage(sCurrResultsPath+"_debug.yml",cv::FileStorage::WRITE);
        pBGS->m_pDebugFS = &oDebugFS;
        pBGS->m_sDebugName = pCurrSequence->m_sGroupName+"_"+pCurrSequence->m_sName;
        g_pnLatestMouseX = &pBGS->m_nDebugCoordX;
        g_pnLatestMouseY = &pBGS->m_nDebugCoordY;
        std::string sMouseDebugDisplayName = pCurrSequence->m_sGroupName + std::string(" -- ") + pCurrSequence->m_sName + " [MOUSE DEBUG]";
        cv::namedWindow(sMouseDebugDisplayName,0);
        cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
#endif //(DEBUG_OUTPUT && (USE_LOBSTER || USE_SUBSENSE || USE_PAWCS))
#if DISPLAY_OUTPUT
        bool bContinuousUpdates = false;
        std::string sDisplayName = pCurrSequence->m_sGroupName + std::string(" -- ") + pCurrSequence->m_sName;
        cv::namedWindow(sDisplayName);
#endif //DISPLAY_OUTPUT
#if (WRITE_IMG_OUTPUT || WRITE_AVI_OUTPUT)
        PlatformUtils::CreateDirIfNotExist(sCurrResultsPath+pCurrSequence->m_sGroupName+"/"+pCurrSequence->m_sName+"/");
#if WRITE_AVI_OUTPUT
        cv::VideoWriter oSegmWriter(sCurrResultsPath+pCurrSequence->m_sGroupName+"/"+pCurrSequence->m_sName+"_segm.avi",CV_FOURCC('F','F','V','1'),30,pCurrSequence->GetImageSize(),false);
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
#if DISPLAY_OUTPUT
            const cv::Mat& oInputFrame = oCurrInputFrame;
#endif //DISPLAY_OUTPUT
#if DEBUG_OUTPUT
            cv::imshow(sMouseDebugDisplayName,oCurrInputFrame);
#endif //DEBUG_OUTPUT
#if NEED_GT_MASK
            oCurrGTMask = pCurrSequence->GetGTFromIndex(nCurrFrameIdx);
            const cv::Mat oGTMask = oCurrGTMask;
#endif //NEED_GT_MASK
            TIMER_INTERNAL_TOC(VideoQuery);
            TIMER_INTERNAL_TIC(PipelineUpdate);
            pBGS->apply(oCurrInputFrame,oCurrFGMask,dCurrLearningRate);
            TIMER_INTERNAL_TOC(PipelineUpdate);
            if(!oROI.empty())
                cv::bitwise_or(oCurrFGMask,UCHAR_MAX/2,oCurrFGMask,oROI==0);
#if NEED_FG_MASK
            const cv::Mat& oFGMask = oCurrFGMask;
#endif //NEED_FG_MASK
#if DISPLAY_OUTPUT
            pBGS->getBackgroundImage(oCurrBGImg);
            if(!oROI.empty())
                cv::bitwise_or(oCurrBGImg,UCHAR_MAX/2,oCurrBGImg,oROI==0);
            const cv::Mat& oBGImg = oCurrBGImg;
            cv::Mat oDisplayFrame = DatasetUtils::Segm::GetDisplayImage(oInputFrame,oBGImg,g_pEvaluator?g_pEvaluator->GetColoredSegmMaskFromResult(oFGMask,oGTMask,oROI):oFGMask,oROI,nCurrFrameIdx,cv::Point(*g_pnLatestMouseX,*g_pnLatestMouseY));
            cv::Mat oDisplayFrameResized;
            if(oDisplayFrame.cols>1280 || oDisplayFrame.rows>960)
                cv::resize(oDisplayFrame,oDisplayFrameResized,cv::Size(oDisplayFrame.cols/2,oDisplayFrame.rows/2));
            else
                oDisplayFrameResized = oDisplayFrame;
            cv::imshow(sDisplayName,oDisplayFrameResized);
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
#if WRITE_AVI_OUTPUT
            oSegmWriter.write(oFGMask);
#endif //WRITE_AVI_OUTPUT
#if WRITE_IMG_OUTPUT
            DatasetUtils::Segm::Video::WriteResult(sCurrResultsPath,pCurrSequence->m_sGroupName,pCurrSequence->m_sName,g_pDatasetInfo->m_sResultFrameNamePrefix,nCurrFrameIdx+g_pDatasetInfo->m_nResultIdxOffset,g_pDatasetInfo->m_sResultFrameNameSuffix,oFGMask);
#endif //WRITE_IMG_OUTPUT
#if WRITE_METRICS
            if(g_pEvaluator)
                g_pEvaluator->AccumulateMetricsFromResult(oFGMask,oGTMask,oROI,pCurrSequence->m_oMetrics);
#endif //WRITE_METRICS
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
        const double dAvgFPS = (double)nFrameCount/dTimeElapsed;
        std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ end, " << int(dTimeElapsed) << " sec in-thread (" << (int)floor(dAvgFPS+0.5) << " FPS)" << std::endl;
#if WRITE_METRICS
        pCurrSequence->m_oMetrics.dTimeElapsed_sec = dTimeElapsed;
        DatasetUtils::Segm::WriteMetrics(sCurrResultsPath+pCurrSequence->m_sGroupName+"/"+pCurrSequence->m_sName+".txt",*pCurrSequence);
#endif //WRITE_METRICS
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
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nAnalyzeSequence caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nAnalyzeSequence caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nAnalyzeSequence caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl;}
    g_nActiveThreads--;
    return 0;
}
#endif //!HAVE_GPU_SUPPORT
