#include "PlatformUtils.h"
#include "DatasetUtils.h"
#include "BackgroundSubtractorPAWCS.h"
#include "BackgroundSubtractorSuBSENSE.h"
#include "BackgroundSubtractorLOBSTER.h"
#include "BackgroundSubtractorViBe_1ch.h"
#include "BackgroundSubtractorViBe_3ch.h"
#include "BackgroundSubtractorPBAS_1ch.h"
#include "BackgroundSubtractorPBAS_3ch.h"

// @@@@ OPENCV: CHECK HARDWARE SUPPORT FOR SSEX USING BUILT-IN UTILITY FUNCTIONS

//////////////////////////////////////////
// USER/ENVIRONMENT-SPECIFIC VARIABLES :
//////////////////////////////////////////
#define DEFAULT_NB_THREADS               1
//////////////////////////////////////////
#define EVAL_RESULTS_ONLY                0
#define WRITE_BGSUB_IMG_OUTPUT           0
#define WRITE_BGSUB_DEBUG_IMG_OUTPUT     0
#define WRITE_BGSUB_METRICS_ANALYSIS     0
//////////////////////////////////////////
#if DEFAULT_NB_THREADS==1
#define DISPLAY_BGSUB_DEBUG_OUTPUT       1
#define ENABLE_DISPLAY_MOUSE_DEBUG       0
#define ENABLE_FRAME_TIMERS              0
#define WRITE_BGSUB_SEGM_AVI_OUTPUT      0
#endif //DEFAULT_NB_THREADS==1
//////////////////////////////////////////
#define USE_CB_LBSP_BG_SUBTRACTOR        0
#define USE_VIBE_LBSP_BG_SUBTRACTOR      1
#define USE_PBAS_LBSP_BG_SUBTRACTOR      0
#define USE_VIBE_BG_SUBTRACTOR           0
#define USE_PBAS_BG_SUBTRACTOR           0
//////////////////////////////////////////
#if (USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR)
#define LIMIT_MODEL_TO_SEQUENCE_ROI      1
#endif
//////////////////////////////////////////
#define USE_CDNET2012_DATASET            0
#define USE_CDNET2014_DATASET            0
#define USE_WALLFLOWER_DATASET           0
#define USE_PETS2001_D3TC1_DATASET       0
#define USE_SINGLE_AVI_FILE              0
#define USE_VIDEO_REGISTRATION           1
/////////////////////////////////////////////////////////////////////
#define DATASET_ROOT_DIR                 std::string("/shared/datasets/")
#define RESULTS_ROOT_DIR                 std::string("/shared/datasets/")
#define RESULTS_OUTPUT_DIR_NAME          std::string("results_test")
#define TOTAL_NB_ITERS                   1
#define TOTAL_NB_PASSES                  1
/////////////////////////////////////////////////////////////////////

#if EVAL_RESULTS_ONLY && (DEFAULT_NB_THREADS>1 || !WRITE_BGSUB_METRICS_ANALYSIS)
#error "Eval-only mode must run with 1 thread & write results somewhere."
#elif (TOTAL_NB_ITERS<=0 || TOTAL_NB_PASSES<=0)
#error "Must run at least 1 iteration & 1 pass."
#endif //(TOTAL_NB_ITERS<=0 || TOTAL_NB_PASSES<=0)
#if (USE_VIBE_LBSP_BG_SUBTRACTOR+USE_PBAS_LBSP_BG_SUBTRACTOR+USE_VIBE_BG_SUBTRACTOR+USE_PBAS_BG_SUBTRACTOR+USE_CB_LBSP_BG_SUBTRACTOR)!=1
#error "Must specify a single algorithm."
#elif (USE_CDNET2012_DATASET+USE_CDNET2014_DATASET+USE_WALLFLOWER_DATASET+USE_PETS2001_D3TC1_DATASET+USE_SINGLE_AVI_FILE+USE_VIDEO_REGISTRATION)!=1
#error "Must specify a single dataset."
#elif USE_CDNET2012_DATASET
const DatasetUtils::eAvailableDatasetsID g_eDatasetID = DatasetUtils::eDataset_CDnet;
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/CDNet/dataset/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/CDNet/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetFolders[] = {"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"};
const char* g_asDatasetGrayscaleDirNameTokens = {"thermal"};
const size_t g_nResultIdxOffset = 1;
#elif USE_CDNET2014_DATASET
const DatasetUtils::eAvailableDatasetsID g_eDatasetID = DatasetUtils::eDataset_CDnet;
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/CDNet2014/dataset/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/CDNet2014/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetFolders[] = {"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"};
const char* g_asDatasetGrayscaleDirNameTokens[] = {"thermal","turbulence"};
const size_t g_nResultIdxOffset = 1;
#elif USE_WALLFLOWER_DATASET
const DatasetUtils::eAvailableDatasetsID g_eDatasetID = DatasetUtils::eDataset_Wallflower;
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/Wallflower/dataset/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/Wallflower/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetFolders[] = {"global"};
const size_t g_nResultIdxOffset = 0;
#elif USE_PETS2001_D3TC1_DATASET
const DatasetUtils::eAvailableDatasetsID g_eDatasetID = DatasetUtils::eDataset_PETS2001_D3TC1;
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/PETS2001/DATASET3/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/PETS2001/DATASET3/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetFolders[] = {"TESTING"};
const size_t g_nResultIdxOffset = 0;
#elif USE_SINGLE_AVI_FILE
const DatasetUtils::eAvailableDatasetsID g_eDatasetID = DatasetUtils::eDataset_GenericSegmentationTest;
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/avitest/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/avitest/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("");
const std::string g_sResultSuffix("");
const char* g_asDatasetFolders[] = {"gait_analysis"};
const size_t g_nResultIdxOffset = 0;
#elif USE_VIDEO_REGISTRATION
const DatasetUtils::eAvailableDatasetsID g_eDatasetID = DatasetUtils::eDataset_LITIV_Registr01;
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/litiv/registration_set01/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/litiv/registration_set01/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("");
const std::string g_sResultSuffix("");
const char* g_asDatasetFolders[] = {"SEQUENCE1"};
const char* g_asDatasetGrayscaleDirNameTokens[] = {"THERMAL"};
const size_t g_nResultIdxOffset = 0;
#endif //USE_VIDEO_REGISTRATION
#if ENABLE_DISPLAY_MOUSE_DEBUG
static int *pnLatestMouseX=nullptr, *pnLatestMouseY=nullptr;
void OnMouseEvent(int event, int x, int y, int, void*) {
    if(event!=cv::EVENT_MOUSEMOVE || !x || !y)
        return;
    *pnLatestMouseX = x;
    *pnLatestMouseY = y;
}
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
#if WRITE_BGSUB_METRICS_ANALYSIS
cv::FileStorage g_oDebugFS;
#endif //!WRITE_BGSUB_METRICS_ANALYSIS
#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
cv::Size g_oDisplayOutputSize(960,240);
bool g_bContinuousUpdates = false;
cv::Mat GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oFGMask, const cv::Mat& oGTFGMask, const cv::Mat& oROI, size_t nFrame);
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if DEFAULT_NB_THREADS<1
#error "Bad default number of threads specified."
#endif //DEFAULT_NB_THREADS<1
int AnalyzeSequence(int nThreadIdx, DatasetUtils::CategoryInfo* pCurrCategory, DatasetUtils::SequenceInfo* pCurrSequence, const std::string& sCurrResultsPath);
#if PLATFORM_SUPPORTS_CPP11
const size_t g_nMaxThreads = DEFAULT_NB_THREADS;//std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():DEFAULT_NB_THREADS;
std::atomic_size_t g_nActiveThreads(0);
#if WRITE_BGSUB_IMG_OUTPUT
const std::vector<int> g_vnResultsComprParams = {CV_IMWRITE_PNG_COMPRESSION,9}; // when writing output bin files, lower to increase processing speed
#endif //WRITE_BGSUB_IMG_OUTPUT
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
const size_t g_nMaxThreads = DEFAULT_NB_THREADS;
HANDLE g_hThreadEvent[g_nMaxThreads] = {0};
HANDLE g_hThreads[g_nMaxThreads] = {0};
void* g_apThreadDataStruct[g_nMaxThreads][3] = {0};
DWORD WINAPI AnalyzeSequenceEntryPoint(LPVOID lpParam) {
    return AnalyzeSequence((int)(lpParam),(CategoryInfo*)g_apThreadDataStruct[(int)(lpParam)][0],(SequenceInfo*)g_apThreadDataStruct[(int)(lpParam)][1],std::string((const char*)(g_apThreadDataStruct[(int)(lpParam)][2])));
}
#if WRITE_BGSUB_IMG_OUTPUT
const int g_anResultsComprParams[2] = {CV_IMWRITE_PNG_COMPRESSION,9}; // when writing output bin files, lower to increase processing speed
const std::vector<int> g_vnResultsComprParams(g_anResultsComprParams,g_anResultsComprParams+2);
#endif //WRITE_BGSUB_IMG_OUTPUT
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for threads/mutexes/atomic variables on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#if TOTAL_NB_ITERS>1
const int g_nBGSamplesIncrPerIter = 5;
int g_nCurrIter;
#endif //TOTAL_NB_ITERS>1

int main() {
#if PLATFORM_USES_WIN32API
    SetConsoleWindowSize(80,40,1000);
#endif //PLATFORM_USES_WIN32API
    std::vector<DatasetUtils::CategoryInfo*> vpCategories;
    std::cout << "Parsing dataset..." << std::endl;
    try {
        for(size_t p=0; p<TOTAL_NB_PASSES; ++p)
            for(size_t i=0; i<sizeof(g_asDatasetFolders)/sizeof(char*); ++i)
                vpCategories.push_back(new DatasetUtils::CategoryInfo(g_asDatasetFolders[i],g_sDatasetPath+g_asDatasetFolders[i],g_eDatasetID,g_asDatasetGrayscaleDirNameTokens));
    } catch(std::runtime_error& e) { std::cout << e.what() << std::endl; }
    size_t nSeqTotal = 0;
    size_t nFramesTotal = 0;
    std::multimap<double,DatasetUtils::SequenceInfo*> mSeqLoads;
    for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory) {
        nSeqTotal += (*pCurrCategory)->m_vpSequences.size();
        for(auto pCurrSequence=(*pCurrCategory)->m_vpSequences.begin(); pCurrSequence!=(*pCurrCategory)->m_vpSequences.end(); ++pCurrSequence) {
            nFramesTotal += (*pCurrSequence)->GetNbInputFrames();
            mSeqLoads.insert(std::pair<double,DatasetUtils::SequenceInfo*>((*pCurrSequence)->m_dExpectedROILoad,(*pCurrSequence)));
        }
    }
    CV_Assert(mSeqLoads.size()==nSeqTotal);
    std::cout << "Parsing complete. [" << vpCategories.size() << " category(ies), "  << nSeqTotal  << " sequence(s)]" << std::endl << std::endl;
    if(nSeqTotal) {
#if DEFAULT_NB_THREADS>1
        // since the algorithm isn't implemented to be parallelised yet, we parallelise the sequence treatment instead
        std::cout << "Running background subtraction with " << ((g_nMaxThreads>nSeqTotal)?nSeqTotal:g_nMaxThreads) << " thread(s)..." << std::endl;
#else //DEFAULT_NB_THREADS==1
        std::cout << "Running background subtraction..." << std::endl;
#endif //DEFAULT_NB_THREADS==1
        size_t nSeqProcessed = 1;
#if TOTAL_NB_ITERS==1
        const std::string sCurrResultsPath = g_sResultsPath;
#else //TOTAL_NB_ITERS>1
        for(g_nCurrIter=1; g_nCurrIter<=TOTAL_NB_ITERS; ++g_nCurrIter) {
            std::cout << std::endl << std::endl << "Iteration [" << g_nCurrIter << "/" << TOTAL_NB_ITERS << "]" << std::endl << std::endl;
            std::stringstream ssCurrResultsPath;
            ssCurrResultsPath << g_sResultsPath << "_iter" << std::setfill('0') << std::setw(3) << g_nCurrIter << "/";
            const std::string sCurrResultsPath = ssCurrResultsPath.str();
            for(size_t c=0; c<vpCategories.size(); ++c) {
                vpCategories[c]->nTP = 0;
                vpCategories[c]->nTN = 0;
                vpCategories[c]->nFP = 0;
                vpCategories[c]->nFN = 0;
                vpCategories[c]->nSE = 0;
                for(size_t s=0; s<vpCategories[c]->m_vpSequences.size(); ++s) {
                    vpCategories[c]->m_vpSequences[s]->nTP = 0;
                    vpCategories[c]->m_vpSequences[s]->nTN = 0;
                    vpCategories[c]->m_vpSequences[s]->nFP = 0;
                    vpCategories[c]->m_vpSequences[s]->nFN = 0;
                    vpCategories[c]->m_vpSequences[s]->nSE = 0;
                }
            }
#endif //TOTAL_NB_ITERS>1
#if WRITE_BGSUB_METRICS_ANALYSIS
            g_oDebugFS = cv::FileStorage(sCurrResultsPath+"framelevel_debug.yml",cv::FileStorage::WRITE);
#endif //WRITE_BGSUB_METRICS_ANALYSIS
#if EVAL_RESULTS_ONLY
            std::cout << "Running background subtraction evaluation..." << std::endl;
            for(auto oSeqIter=mSeqLoads.rbegin(); oSeqIter!=mSeqLoads.rend(); ++oSeqIter) {
                std::cout << "\tProcessing [" << nSeqProcessed << "/" << nSeqTotal << "] (" << oSeqIter->second->m_pParent->m_sName << ":" << oSeqIter->second->m_sName << ", L=" << std::scientific << std::setprecision(2) << oSeqIter->first << ")" << std::endl;
                for(size_t k=0; k<oSeqIter->second->GetNbGTFrames(); ++k) {
                    cv::Mat oGTImg = oSeqIter->second->GetGTFrameFromIndex(k);
                    cv::Mat oFGMask = ReadResult(sCurrResultsPath,oSeqIter->second->m_pParent->m_sName,oSeqIter->second->m_sName,g_sResultPrefix,k+g_nResultIdxOffset,g_sResultSuffix);
                    CalcMetricsFromResult(oFGMask,oGTImg,oSeqIter->second->GetSequenceROI(),oSeqIter->second->nTP,oSeqIter->second->nTN,oSeqIter->second->nFP,oSeqIter->second->nFN,oSeqIter->second->nSE);
                }
                ++nSeqProcessed;
            }
            const double dFinalFPS = 0.0;
#else //!EVAL_RESULTS_ONLY
            PlatformUtils::CreateDirIfNotExist(sCurrResultsPath);
            for(size_t c=0; c<vpCategories.size(); ++c)
                PlatformUtils::CreateDirIfNotExist(sCurrResultsPath+vpCategories[c]->m_sName+"/");
            time_t startup = time(nullptr);
            tm* startup_tm = localtime(&startup);
            std::cout << "[" << (startup_tm->tm_year + 1900) << '/' << (startup_tm->tm_mon + 1) << '/' <<  startup_tm->tm_mday << " -- ";
            std::cout << startup_tm->tm_hour << ':' << startup_tm->tm_min << ':' << startup_tm->tm_sec << ']' << std::endl;
#if PLATFORM_SUPPORTS_CPP11
            for(auto oSeqIter=mSeqLoads.rbegin(); oSeqIter!=mSeqLoads.rend(); ++oSeqIter) {
                while(g_nActiveThreads>=g_nMaxThreads)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                std::cout << "\tProcessing [" << nSeqProcessed << "/" << nSeqTotal << "] (" << oSeqIter->second->m_pParent->m_sName << ":" << oSeqIter->second->m_sName << ", L=" << std::scientific << std::setprecision(2) << oSeqIter->first << ")" << std::endl;
                g_nActiveThreads++;
                nSeqProcessed++;
                std::thread(AnalyzeSequence,nSeqProcessed,oSeqIter->second->m_pParent,oSeqIter->second,sCurrResultsPath).detach();
            }
            while(g_nActiveThreads>0)
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
            for(size_t n=0; n<g_nMaxThreads; ++n)
                g_hThreadEvent[n] = CreateEvent(NULL,FALSE,TRUE,NULL);
            for(auto oSeqIter=mSeqLoads.rbegin(); oSeqIter!=mSeqLoads.rend(); ++oSeqIter) {
                DWORD ret = WaitForMultipleObjects(g_nMaxThreads,g_hThreadEvent,FALSE,INFINITE);
                std::cout << "\tProcessing [" << nSeqProcessed << "/" << nSeqTotal << "] (" << oSeqIter->second->m_pParent->m_sName << ":" << oSeqIter->second->m_sName << ", L=" << std::scientific << std::setprecision(2) << oSeqIter->first << ")" << std::endl;
                nSeqProcessed++;
                g_apThreadDataStruct[ret][0] = oSeqIter->second->m_pParent;
                g_apThreadDataStruct[ret][1] = oSeqIter->second;
                g_apThreadDataStruct[ret][2] = (void*)sCurrResultsPath.c_str();
                g_hThreads[ret] = CreateThread(NULL,NULL,AnalyzeSequenceEntryPoint,(LPVOID)ret,0,NULL);
            }
            WaitForMultipleObjects((DWORD)((g_nMaxThreads>nSeqTotal)?nSeqTotal:g_nMaxThreads),g_hThreads,TRUE,INFINITE);
            for(size_t n=0; n<g_nMaxThreads; ++n) {
                CloseHandle(g_hThreadEvent[n]);
                CloseHandle(g_hThreads[n]);
            }
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for threads/mutexes/atomic variables on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
            time_t shutdown = time(nullptr);
            tm* shutdown_tm = localtime(&shutdown);
            std::cout << "[" << (shutdown_tm->tm_year + 1900) << '/' << (shutdown_tm->tm_mon + 1) << '/' <<  shutdown_tm->tm_mday << " -- ";
            std::cout << shutdown_tm->tm_hour << ':' << shutdown_tm->tm_min << ':' << shutdown_tm->tm_sec << ']' << std::endl;
            const double dFinalFPS = ((double)nFramesTotal)/(shutdown-startup);
            std::cout << "\t ... session completed at a total of " << dFinalFPS << " fps." << std::endl;
#endif //!EVAL_RESULTS_ONLY
#if WRITE_BGSUB_METRICS_ANALYSIS
            std::cout << "Summing and writing metrics results...\n" << std::endl;
            for(size_t c=0; c<vpCategories.size(); ++c) {
                if(!vpCategories[c]->m_vpSequences.empty()) {
                    for(size_t s=0; s<vpCategories[c]->m_vpSequences.size(); ++s) {
                        vpCategories[c]->nTP += vpCategories[c]->m_vpSequences[s]->nTP;
                        vpCategories[c]->nTN += vpCategories[c]->m_vpSequences[s]->nTN;
                        vpCategories[c]->nFP += vpCategories[c]->m_vpSequences[s]->nFP;
                        vpCategories[c]->nFN += vpCategories[c]->m_vpSequences[s]->nFN;
                        vpCategories[c]->nSE += vpCategories[c]->m_vpSequences[s]->nSE;
                        WriteMetrics(sCurrResultsPath+vpCategories[c]->m_sName+"/"+vpCategories[c]->m_vpSequences[s]->m_sName+".txt",vpCategories[c]->m_vpSequences[s]);
                    }
                    WriteMetrics(sCurrResultsPath+vpCategories[c]->m_sName+".txt",vpCategories[c]);
                    std::cout << std::endl;
                }
            }
            WriteMetrics(sCurrResultsPath+"METRICS_TOTAL.txt",vpCategories,dFinalFPS);
            g_oDebugFS.release();
#endif //WRITE_BGSUB_METRICS_ANALYSIS
#if TOTAL_NB_ITERS>1
        }
#endif //TOTAL_NB_ITERS>1
        std::cout << std::endl << "All done." << std::endl;
    }
    else
        std::cout << "No sequences found, all done." << std::endl;
    // let memory 'leak' here, exits faster once job is done...
    //for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory)
    //    delete *pCurrCategory;
    //vpCategories.clear();
}

int AnalyzeSequence(int nThreadIdx, DatasetUtils::CategoryInfo* pCurrCategory, DatasetUtils::SequenceInfo* pCurrSequence, const std::string& sCurrResultsPath) {
    srand(0); // for now, assures that two consecutive runs on the same data return the same results
    //srand((unsigned int)time(NULL));
#if USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR
    BackgroundSubtractorLBSP* pBGS = nullptr;
#if LIMIT_MODEL_TO_SEQUENCE_ROI
    cv::Mat oSequenceROI = pCurrSequence->GetSequenceROI();
#else //!LIMIT_MODEL_TO_SEQUENCE_ROI
    cv::Mat oSequenceROI;
#endif //!LIMIT_MODEL_TO_SEQUENCE_ROI
#else //USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
    cv::BackgroundSubtractor* pBGS = nullptr;
#endif //USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
    try {
        CV_Assert(pCurrCategory && pCurrSequence);
        CV_Assert(pCurrSequence->GetNbInputFrames()>1);
#if USE_PRECACHED_IO
        pCurrSequence->StartPrecaching();
#endif //USE_PRECACHED_IO
        cv::Mat oFGMask, oInitImg = pCurrSequence->GetInputFrameFromIndex(0);
#if USE_VIBE_LBSP_BG_SUBTRACTOR
#if TOTAL_NB_ITERS>1
        pBGS = new BackgroundSubtractorLOBSTER( BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
                                                BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD,
                                                BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD,
                                                BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD,
                                                g_nBGSamplesIncrPerIter*g_nCurrIter,
                                                BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES);
#else //TOTAL_NB_ITERS==1
        pBGS = new BackgroundSubtractorLOBSTER();
#endif //TOTAL_NB_ITERS==1
        const double dDefaultLearningRate = BGSLOBSTER_DEFAULT_LEARNING_RATE;
        pBGS->initialize(oInitImg,oSequenceROI);
#elif USE_PBAS_LBSP_BG_SUBTRACTOR
        pBGS = new BackgroundSubtractorSuBSENSE();
        const double dDefaultLearningRate = 0;
        pBGS->initialize(oInitImg,oSequenceROI);
#elif USE_CB_LBSP_BG_SUBTRACTOR
        pBGS = new BackgroundSubtractorPAWCS();
        const double dDefaultLearningRate = 0;
        pBGS->initialize(oInitImg,oSequenceROI);
#else //USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
        const size_t m_nInputChannels = (size_t)oInitImg.channels();
#if USE_VIBE_BG_SUBTRACTOR
        if(m_nInputChannels==3)
            pBGS = new BackgroundSubtractorViBe_3ch();
        else
            pBGS = new BackgroundSubtractorViBe_1ch();
        ((BackgroundSubtractorPBAS*)pBGS)->initialize(oInitImg);
        const double dDefaultLearningRate = BGSVIBE_DEFAULT_LEARNING_RATE;
#else //USE_PBAS_BG_SUBTRACTOR
        if(m_nInputChannels==3)
            pBGS = new BackgroundSubtractorPBAS_3ch();
        else
            pBGS = new BackgroundSubtractorPBAS_1ch();
        ((BackgroundSubtractorPBAS*)pBGS)->initialize(oInitImg);
        const double dDefaultLearningRate = BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE;
#endif //USE_PBAS_BG_SUBTRACTOR
#endif //USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
#if USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR
        pBGS->m_sDebugName = pCurrCategory->m_sName+"_"+pCurrSequence->m_sName;
#if WRITE_BGSUB_METRICS_ANALYSIS
        pBGS->m_pDebugFS = &g_oDebugFS;
#endif //WRITE_BGSUB_METRICS_ANALYSIS
#if ENABLE_DISPLAY_MOUSE_DEBUG
        pnLatestMouseX = &pBGS->m_nDebugCoordX;
        pnLatestMouseY = &pBGS->m_nDebugCoordY;
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
#endif //USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR
#if DISPLAY_BGSUB_DEBUG_OUTPUT
        std::string sDebugDisplayName = pCurrCategory->m_sName + std::string(" -- ") + pCurrSequence->m_sName;
        cv::namedWindow(sDebugDisplayName);
#endif //DISPLAY_ANALYSIS_DEBUG_RESULTS
        CV_Assert((!WRITE_BGSUB_DEBUG_IMG_OUTPUT && !WRITE_BGSUB_SEGM_AVI_OUTPUT && !WRITE_BGSUB_IMG_OUTPUT) || !sCurrResultsPath.empty());
#if ENABLE_DISPLAY_MOUSE_DEBUG
        std::string sMouseDebugDisplayName = pCurrCategory->m_sName + std::string(" -- ") + pCurrSequence->m_sName + " [MOUSE DEBUG]";
        cv::namedWindow(sMouseDebugDisplayName,0);
        cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
#if WRITE_BGSUB_DEBUG_IMG_OUTPUT
        cv::VideoWriter oDebugWriter(sCurrResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+".avi",CV_FOURCC('X','V','I','D'),30,g_oDisplayOutputSize,true);
#endif //WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if WRITE_BGSUB_SEGM_AVI_OUTPUT
        cv::VideoWriter oSegmWriter(sCurrResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+"_segm.avi",CV_FOURCC('F','F','V','1'),30,pCurrSequence->GetSize(),false);
#endif //WRITE_BGSUB_SEGM_AVI_OUTPUT
#if WRITE_BGSUB_IMG_OUTPUT
        CreateDirIfNotExist(sCurrResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+"/");
#endif //WRITE_BGSUB_IMG_OUTPUT
#if WRITE_BGSUB_METRICS_ANALYSIS
        time_t startup = time(nullptr);
#endif //WRITE_BGSUB_METRICS_ANALYSIS
        const size_t nNbInputFrames = pCurrSequence->GetNbInputFrames();
        for(size_t k=0; k<nNbInputFrames; k++) {
            if(!(k%100)) {
                const std::string sCurrSeqName = pCurrSequence->m_sName.size()>12?pCurrSequence->m_sName.substr(0,12):pCurrSequence->m_sName;
                std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " @ F:" << std::setfill('0') << std::setw(PlatformUtils::decimal_integer_digit_count((int)nNbInputFrames)) << k << "/" << nNbInputFrames << "   [T=" << nThreadIdx << "]" << std::endl;
            }
#if ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
            std::chrono::high_resolution_clock::time_point pre_query = std::chrono::high_resolution_clock::now();
#endif //ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
            const cv::Mat& oInputImg = pCurrSequence->GetInputFrameFromIndex(k);
#if ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
            std::chrono::high_resolution_clock::time_point post_query = std::chrono::high_resolution_clock::now();
            std::cout << "t = " << k << ", query=" << std::fixed << std::setprecision(1) << (float)(std::chrono::duration_cast<std::chrono::microseconds>(post_query-pre_query).count())/1000 << ", ";
#endif //ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
#if ENABLE_DISPLAY_MOUSE_DEBUG
            cv::imshow(sMouseDebugDisplayName,oInputImg);
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
            cv::Mat oLastBGImg;
            pBGS->getBackgroundImage(oLastBGImg);
#if (USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR)
            if(!oSequenceROI.empty())
                cv::bitwise_or(oLastBGImg,UCHAR_MAX/2,oLastBGImg,oSequenceROI==0);
#endif //(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR)
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
            std::chrono::high_resolution_clock::time_point pre_process = std::chrono::high_resolution_clock::now();
#endif //ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
            pBGS->apply(oInputImg, oFGMask, k<=100?1:dDefaultLearningRate);
#if ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
            std::chrono::high_resolution_clock::time_point post_process = std::chrono::high_resolution_clock::now();
            std::cout << "proc=" << std::fixed << std::setprecision(1) << (float)(std::chrono::duration_cast<std::chrono::microseconds>(post_process-pre_process).count())/1000 << "." << std::endl;
#endif //ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
#if (WRITE_BGSUB_IMG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT || DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_SEGM_AVI_OUTPUT)
#if (USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR)
            if(!oSequenceROI.empty())
                cv::bitwise_or(oFGMask,UCHAR_MAX/2,oFGMask,oSequenceROI==0);
#endif //(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR)
#endif //(WRITE_BGSUB_IMG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT || DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_SEGM_AVI_OUTPUT)
#if WRITE_BGSUB_SEGM_AVI_OUTPUT
            oSegmWriter.write(oFGMask);
#endif //WRITE_BGSUB_SEGM_AVI_OUTPUT
#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT || WRITE_BGSUB_METRICS_ANALYSIS
            cv::Mat oGTImg = pCurrSequence->GetGTFrameFromIndex(k);
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT || WRITE_BGSUB_METRICS_ANALYSIS
#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
            cv::Mat oDebugDisplayFrame = GetDisplayResult(oInputImg,oLastBGImg,oFGMask,oGTImg,pCurrSequence->GetSequenceROI(),k);
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if WRITE_BGSUB_DEBUG_IMG_OUTPUT
            oDebugWriter.write(oDebugDisplayFrame);
#endif //WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if DISPLAY_BGSUB_DEBUG_OUTPUT
            cv::Mat oDebugDisplayFrameResized;
            if(oDebugDisplayFrame.cols>1280 || oDebugDisplayFrame.rows>960)
                cv::resize(oDebugDisplayFrame,oDebugDisplayFrameResized,cv::Size(oDebugDisplayFrame.cols/2,oDebugDisplayFrame.rows/2));
            else
                oDebugDisplayFrameResized = oDebugDisplayFrame;
            cv::imshow(sDebugDisplayName, oDebugDisplayFrameResized);
            int nKeyPressed;
            if(g_bContinuousUpdates)
                nKeyPressed = cv::waitKey(1);
            else
                nKeyPressed = cv::waitKey(0);
            if(nKeyPressed!=-1) {
                nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
                std::cout << "nKeyPressed = " << nKeyPressed%(UCHAR_MAX+1) << std::endl;
            }
            if(nKeyPressed==' ')
                g_bContinuousUpdates = !g_bContinuousUpdates;
            else if(nKeyPressed==(int)'q')
                break;
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT
#if WRITE_BGSUB_IMG_OUTPUT
            WriteResult(sCurrResultsPath,pCurrCategory->m_sName,pCurrSequence->m_sName,g_sResultPrefix,k+g_nResultIdxOffset,g_sResultSuffix,oFGMask,g_vnResultsComprParams);
#endif //WRITE_BGSUB_IMG_OUTPUT
#if WRITE_BGSUB_METRICS_ANALYSIS
            DatasetUtils::CalcMetricsFromResult(oFGMask,oGTImg,pCurrSequence->GetSequenceROI(),pCurrSequence->nTP,pCurrSequence->nTN,pCurrSequence->nFP,pCurrSequence->nFN,pCurrSequence->nSE);
        }
        time_t shutdown = time(nullptr);
        pCurrSequence->m_dAvgFPS = ((double)nNbInputFrames)/(shutdown-startup);
        WriteMetrics(sCurrResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+".txt",pCurrSequence);
#else //!WRITE_BGSUB_METRICS_ANALYSIS
        }
#endif //!WRITE_BGSUB_METRICS_ANALYSIS
#if DISPLAY_BGSUB_DEBUG_OUTPUT
        cv::destroyWindow(sDebugDisplayName);
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT
#if ENABLE_DISPLAY_MOUSE_DEBUG
        cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
        pnLatestMouseX = nullptr;
        pnLatestMouseY = nullptr;
        cv::destroyWindow(sMouseDebugDisplayName);
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
    }
    catch(cv::Exception& e) {std::cout << e.what() << std::endl;}
    catch(std::runtime_error& e) {std::cout << e.what() << std::endl;}
    catch(...) {std::cout << "Caught unknown exception." << std::endl;}
    if(pBGS) delete pBGS;
#if PLATFORM_SUPPORTS_CPP11
    g_nActiveThreads--;
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
    SetEvent(g_hThreadEvent[nThreadIdx]);
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for threads/mutexes/atomic variables on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
    return 0;
}

#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
// NOTE : current impl is most likely broken for pure vibe/pbas subtractors.
cv::Mat GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oFGMask, const cv::Mat& oGTFGMask, const cv::Mat& oROI, size_t nFrame) {
    // note: this function is definitely NOT efficient in any way; it is only intended for debug purposes.
    cv::Mat oInputImgBYTE3, oBGImgBYTE3, oFGMaskBYTE3;
    if(oInputImg.channels()!=3) {
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_GRAY2RGB);
        cv::cvtColor(oBGImg,oBGImgBYTE3,cv::COLOR_GRAY2RGB);
    }
    else {
        oInputImgBYTE3 = oInputImg;
        oBGImgBYTE3 = oBGImg;
    }
    oFGMaskBYTE3 = DatasetUtils::GetColoredSegmFrameFromResult(oFGMask,oGTFGMask,oROI);
#if ENABLE_DISPLAY_MOUSE_DEBUG
    if(pnLatestMouseX&&pnLatestMouseY) {
        cv::Point dbgpt(*pnLatestMouseX,*pnLatestMouseY);
        cv::circle(oInputImgBYTE3,dbgpt,5,cv::Scalar(255,255,255));
        cv::circle(oFGMaskBYTE3,dbgpt,5,cv::Scalar(255,255,255));
    }
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
    cv::Mat displayH,displayV1,displayV2;
    cv::resize(oInputImgBYTE3,oInputImgBYTE3,cv::Size(320,240));
    cv::resize(oBGImgBYTE3,oBGImgBYTE3,cv::Size(320,240));
    cv::resize(oFGMaskBYTE3,oFGMaskBYTE3,cv::Size(320,240));

    std::stringstream sstr;
    sstr << "Input Image #" << nFrame;
    DatasetUtils::WriteOnImage(oInputImgBYTE3,sstr.str());
    DatasetUtils::WriteOnImage(oBGImgBYTE3,"Reference Image");
    DatasetUtils::WriteOnImage(oFGMaskBYTE3,"Segmentation Result");

    cv::hconcat(oInputImgBYTE3,oBGImgBYTE3,displayH);
    cv::hconcat(displayH,oFGMaskBYTE3,displayH);
    return displayH;
}
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
