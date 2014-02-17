#include "PlatformUtils.h"
#include "DatasetUtils.h"
#include "BackgroundSubtractorCBLBSP.h"
#include "BackgroundSubtractorPBASLBSP.h"
#include "BackgroundSubtractorLOBSTER.h"
#include "BackgroundSubtractorViBe_1ch.h"
#include "BackgroundSubtractorViBe_3ch.h"
#include "BackgroundSubtractorPBAS_1ch.h"
#include "BackgroundSubtractorPBAS_3ch.h"

/////////////////////////////////////////
// USER/ENVIRONMENT-SPECIFIC VARIABLES :
/////////////////////////////////////////
#define DEFAULT_NB_THREADS				4
/////////////////////////////////////////
#define WRITE_BGSUB_IMG_OUTPUT			0
#define WRITE_BGSUB_DEBUG_IMG_OUTPUT	0
#define WRITE_BGSUB_METRICS_ANALYSIS	1
/////////////////////////////////////////
#if DEFAULT_NB_THREADS==1
#define DISPLAY_BGSUB_DEBUG_OUTPUT		0
#define ENABLE_DISPLAY_MOUSE_DEBUG		0
#define ENABLE_FRAME_TIMERS				0
#endif //DEFAULT_NB_THREADS==1
/////////////////////////////////////////
#define USE_CB_LBSP_BG_SUBTRACTOR		1
#define USE_VIBE_LBSP_BG_SUBTRACTOR		0
#define USE_PBAS_LBSP_BG_SUBTRACTOR		0
#define USE_VIBE_BG_SUBTRACTOR			0
#define USE_PBAS_BG_SUBTRACTOR			0
/////////////////////////////////////////
#if USE_VIBE_LBSP_BG_SUBTRACTOR
#define USE_RELATIVE_LBSP_COMPARISONS	1
#endif //USE_VIBE_LBSP_BG_SUBTRACTOR
#if USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR
#define LIMIT_KEYPTS_TO_SEQUENCE_ROI	1
#endif
/////////////////////////////////////////
#define USE_CDNET_DATASET				1
#define USE_WALLFLOWER_DATASET			0
#define USE_PETS2001_D3TC1_DATASET		0
/////////////////////////////////////////////////////////////////////
#define DATASET_ROOT_DIR 				std::string("/tmp/datasets/")
#define RESULTS_ROOT_DIR 				std::string("/tmp/datasets/")
#define RESULTS_OUTPUT_DIR_NAME			std::string("results_test")
/////////////////////////////////////////////////////////////////////

#if (USE_VIBE_LBSP_BG_SUBTRACTOR+USE_PBAS_LBSP_BG_SUBTRACTOR+USE_VIBE_BG_SUBTRACTOR+USE_PBAS_BG_SUBTRACTOR+USE_CB_LBSP_BG_SUBTRACTOR)!=1
#error "Must specify a single algorithm."
#elif (USE_CDNET_DATASET+USE_WALLFLOWER_DATASET+USE_PETS2001_D3TC1_DATASET)!=1
#error "Must specify a single dataset."
#elif USE_CDNET_DATASET
const std::string g_sDatasetName(CDNET_DB_NAME);
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/CDNet/dataset/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/CDNet/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
//const char* g_asDatasetCategories[] = {"baseline"};
//const char* g_asDatasetCategories[] = {"baseline_office"};
//const char* g_asDatasetCategories[] = {"cameraJitter_boulevard"};
//const char* g_asDatasetCategories[] = {"baseline","shadow_cubicle"};
//const char* g_asDatasetCategories[] = {"dynamicBackground_fountain01"};
//const char* g_asDatasetCategories[] = {"shadow_bungalows"};
//const char* g_asDatasetCategories[] = {"intermittentObjectMotion_streetLight"};
//const char* g_asDatasetCategories[] = {"dynamicBackground_boats","dynamicBackground_fountain01","dynamicBackground_fountain02","dynamicBackground_overpass","cameraJitter_sidewalk"};
//const char* g_asDatasetCategories[] = {"shadow_cubicle","intermittentObjectMotion_tramstop","intermittentObjectMotion_winterDriveway"};
//const char* g_asDatasetCategories[] = {"thermal_lakeSide"};
const char* g_asDatasetCategories[] = {"dynamicBackground","shadow","baseline","intermittentObjectMotion","cameraJitter","thermal"};
const size_t g_nResultIdxOffset = 1;
#elif USE_WALLFLOWER_DATASET
const std::string g_sDatasetName(WALLFLOWER_DB_NAME);
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/Wallflower/dataset/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/Wallflower/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetCategories[] = {"global"};
const size_t g_nResultIdxOffset = 0;
#elif USE_PETS2001_D3TC1_DATASET
const std::string g_sDatasetName(PETS2001_D3TC1_DB_NAME);
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/PETS2001/DATASET3/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/PETS2001/DATASET3/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetCategories[] = {"TESTING"};
const size_t g_nResultIdxOffset = 0;
#endif //USE_PETS2001_D3TC1_DATASET
#if ENABLE_DISPLAY_MOUSE_DEBUG
static int *pnLatestMouseX=nullptr, *pnLatestMouseY=nullptr;
void OnMouseEvent(int event, int x, int y, int, void*) {
	if(event!=cv::EVENT_MOUSEMOVE || !x || !y)
		return;
	*pnLatestMouseX = x;
	*pnLatestMouseY = y;
}
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
cv::Size g_oDisplayOutputSize(960,240);
bool g_bContinuousUpdates = false;
cv::Mat GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oFGMask, const cv::Mat& oGTFGMask, const cv::Mat& oROI, size_t nFrame);
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if DEFAULT_NB_THREADS<1
#error "Bad default number of threads specified."
#endif //DEFAULT_NB_THREADS<1
#if PLATFORM_SUPPORTS_CPP11
int AnalyzeSequence(CategoryInfo* pCurrCategory, SequenceInfo* pCurrSequence);
const size_t g_nMaxThreads = /*std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():*/DEFAULT_NB_THREADS;
std::atomic_size_t g_nActiveThreads(0);
#if WRITE_BGSUB_IMG_OUTPUT
const std::vector<int> g_vnResultsComprParams = {CV_IMWRITE_PNG_COMPRESSION,9}; // when writing output bin files, lower to increase processing speed
#endif //WRITE_BGSUB_IMG_OUTPUT
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
int AnalyzeSequence(int nThreadIdx, CategoryInfo* pCurrCategory, SequenceInfo* pCurrSequence);
const size_t g_nMaxThreads = DEFAULT_NB_THREADS;
HANDLE g_hThreadEvent[g_nMaxThreads] = {0};
HANDLE g_hThreads[g_nMaxThreads] = {0};
void* g_apThreadDataStruct[g_nMaxThreads][2] = {0};
DWORD WINAPI AnalyzeSequenceEntryPoint(LPVOID lpParam) {
	return AnalyzeSequence((int)(lpParam),(CategoryInfo*)g_apThreadDataStruct[(int)(lpParam)][0],(SequenceInfo*)g_apThreadDataStruct[(int)(lpParam)][1]);
}
#if WRITE_BGSUB_IMG_OUTPUT
const int g_anResultsComprParams[2] = {CV_IMWRITE_PNG_COMPRESSION,9}; // when writing output bin files, lower to increase processing speed
const std::vector<int> g_vnResultsComprParams(g_anResultsComprParams,g_anResultsComprParams+2);
#endif //WRITE_BGSUB_IMG_OUTPUT
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for threads/mutexes/atomic variables on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11

int main() {
	srand(0); // for now, assures that two consecutive runs on the same data return the same results
	std::vector<CategoryInfo*> vpCategories;
	std::cout << "Parsing dataset '"<< g_sDatasetName << "'..." << std::endl;
	try {
		for(size_t i=0; i<sizeof(g_asDatasetCategories)/sizeof(char*); ++i) {
			bool bIsThermal = (std::string(g_asDatasetCategories[i]).find("thermal")!=std::string::npos);
			vpCategories.push_back(new CategoryInfo(g_asDatasetCategories[i], g_sDatasetPath+g_asDatasetCategories[i], g_sDatasetName, bIsThermal));
		}
	} catch(std::runtime_error& e) { std::cout << e.what() << std::endl; }
	size_t nSeqTotal = 0;
	size_t nFramesTotal = 0;
	std::multimap<double,SequenceInfo*> mSeqLoads;
	for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory) {
		nSeqTotal += (*pCurrCategory)->m_vpSequences.size();
		for(auto pCurrSequence=(*pCurrCategory)->m_vpSequences.begin(); pCurrSequence!=(*pCurrCategory)->m_vpSequences.end(); ++pCurrSequence) {
			nFramesTotal += (*pCurrSequence)->GetNbInputFrames();
			mSeqLoads.insert(std::pair<double,SequenceInfo*>((*pCurrSequence)->m_dExpectedROILoad,(*pCurrSequence)));
		}
	}
	CV_Assert(mSeqLoads.size()==nSeqTotal);
	std::cout << "Parsing complete. [" << vpCategories.size() << " category(ies), "  << nSeqTotal  << " sequence(s)]" << std::endl << std::endl;
	time_t startup = time(nullptr);
	tm* startup_tm = localtime(&startup);
	std::cout << "[" << (startup_tm->tm_year + 1900) << '/' << (startup_tm->tm_mon + 1) << '/' <<  startup_tm->tm_mday << " -- ";
	std::cout << startup_tm->tm_hour << ':' << startup_tm->tm_min << ':' << startup_tm->tm_sec << ']' << std::endl;
	if(nSeqTotal) {
		// since the algorithm isn't implemented to be parallelised yet, we parallelise the sequence treatment instead
		std::cout << "Running LBSP background subtraction with " << ((g_nMaxThreads>nSeqTotal)?nSeqTotal:g_nMaxThreads) << " thread(s)..." << std::endl;
		size_t nSeqProcessed = 1;
#if PLATFORM_SUPPORTS_CPP11
		for(auto oSeqIter=mSeqLoads.rbegin(); oSeqIter!=mSeqLoads.rend(); ++oSeqIter) {
			while(g_nActiveThreads>=g_nMaxThreads)
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			std::cout << "\tProcessing sequence " << nSeqProcessed << "/" << nSeqTotal << "... (" << oSeqIter->second->m_pParent->m_sName << ":" << oSeqIter->second->m_sName << ", L=" << std::scientific <<  oSeqIter->first << ")" << std::endl;
			g_nActiveThreads++;
			nSeqProcessed++;
			std::thread(AnalyzeSequence,oSeqIter->second->m_pParent,oSeqIter->second).detach();
		}
		while(g_nActiveThreads>0)
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
		for(size_t n=0; n<g_nMaxThreads; ++n)
			g_hThreadEvent[n] = CreateEvent(NULL,FALSE,TRUE,NULL);
		for(auto oSeqIter=mSeqLoads.rbegin(); oSeqIter!=mSeqLoads.rend(); ++oSeqIter) {
			DWORD ret = WaitForMultipleObjects(g_nMaxThreads,g_hThreadEvent,FALSE,INFINITE);
			std::cout << "\tProcessing sequence " << nSeqProcessed << "/" << nSeqTotal << "... (" << oSeqIter->second->m_pParent->m_sName << ":" << oSeqIter->second->m_sName << ", L=" << std::scientific << oSeqIter->first << ")" << std::endl;
			nSeqProcessed++;
			g_apThreadDataStruct[ret][0] = oSeqIter->second->m_pParent;
			g_apThreadDataStruct[ret][1] = oSeqIter->second;
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
		double dFinalFPS = ((double)nFramesTotal)/(shutdown-startup);
		std::cout << "\t ... session completed at a total of " << dFinalFPS << " fps." << std::endl;
#if WRITE_BGSUB_METRICS_ANALYSIS
		std::cout << "Summing and writing metrics results..." << std::endl;
		for(size_t c=0; c<vpCategories.size(); ++c) {
			if(!vpCategories[c]->m_vpSequences.empty()) {
				for(size_t s=0; s<vpCategories[c]->m_vpSequences.size(); ++s) {
					vpCategories[c]->nTP += vpCategories[c]->m_vpSequences[s]->nTP;
					vpCategories[c]->nTN += vpCategories[c]->m_vpSequences[s]->nTN;
					vpCategories[c]->nFP += vpCategories[c]->m_vpSequences[s]->nFP;
					vpCategories[c]->nFN += vpCategories[c]->m_vpSequences[s]->nFN;
					vpCategories[c]->nSE += vpCategories[c]->m_vpSequences[s]->nSE;
				}
				WriteMetrics(g_sResultsPath+vpCategories[c]->m_sName+".txt",vpCategories[c]);
			}
		}
		WriteMetrics(g_sResultsPath+"METRICS_TOTAL.txt",vpCategories,dFinalFPS);
#endif
		std::cout << "All done." << std::endl;
	}
	else
		std::cout << "No sequences found, all done." << std::endl;

	// let memory 'leak' here, exits faster once job is done...
	//for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory)
	//	delete *pCurrCategory;
	//vpCategories.clear();
}

#if PLATFORM_SUPPORTS_CPP11
int AnalyzeSequence(CategoryInfo* pCurrCategory, SequenceInfo* pCurrSequence) {
#elif PLATFORM_USES_WIN32API
int AnalyzeSequence(int nThreadIdx, CategoryInfo* pCurrCategory, SequenceInfo* pCurrSequence) {
#endif //PLATFORM_USES_WIN32API
#if USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR
		BackgroundSubtractorLBSP* pBGS = nullptr;
#if LIMIT_KEYPTS_TO_SEQUENCE_ROI
		std::vector<cv::KeyPoint> voKPs = pCurrSequence->GetKeyPointsFromROI();
#else //!LIMIT_KEYPTS_TO_SEQUENCE_ROI
		std::vector<cv::KeyPoint> voKPs;
#endif //!LIMIT_KEYPTS_TO_SEQUENCE_ROI
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
#if USE_RELATIVE_LBSP_COMPARISONS
		pBGS = new BackgroundSubtractorLOBSTER(BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD);
#else //!USE_RELATIVE_LBSP_COMPARISONS
		pBGS = new BackgroundSubtractorLOBSTER(BGSLOBSTER_DEFAULT_LBSP_ABS_SIMILARITY_THRESHOLD);
#endif //!USE_RELATIVE_LBSP_COMPARISONS
		const double dDefaultLearningRate = BGSLOBSTER_DEFAULT_LEARNING_RATE;
		pBGS->initialize(oInitImg,voKPs);
#elif USE_PBAS_LBSP_BG_SUBTRACTOR
		pBGS = new BackgroundSubtractorPBASLBSP();
		const double dDefaultLearningRate = 0;
		pBGS->initialize(oInitImg,voKPs);
#elif USE_CB_LBSP_BG_SUBTRACTOR
		pBGS = new BackgroundSubtractorCBLBSP();
		const double dDefaultLearningRate = 0;
		pBGS->initialize(oInitImg,voKPs);
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
#if ENABLE_DISPLAY_MOUSE_DEBUG
		pnLatestMouseX = &pBGS->nDebugCoordX;
		pnLatestMouseY = &pBGS->nDebugCoordY;
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
#endif //USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR || USE_CB_LBSP_BG_SUBTRACTOR
#if DISPLAY_BGSUB_DEBUG_OUTPUT
		std::string sDebugDisplayName = pCurrCategory->m_sName + std::string(" -- ") + pCurrSequence->m_sName;
		cv::namedWindow(sDebugDisplayName);
#endif //DISPLAY_ANALYSIS_DEBUG_RESULTS
#if ENABLE_DISPLAY_MOUSE_DEBUG
		std::string sMouseDebugDisplayName = pCurrCategory->m_sName + std::string(" -- ") + pCurrSequence->m_sName + " [MOUSE DEBUG]";
		cv::namedWindow(sMouseDebugDisplayName,0);
		cv::setMouseCallback(sMouseDebugDisplayName,OnMouseEvent,nullptr);
#endif //ENABLE_DISPLAY_MOUSE_DEBUG
#if WRITE_BGSUB_DEBUG_IMG_OUTPUT
		cv::VideoWriter oDebugWriter(g_sResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+".avi",CV_FOURCC('X','V','I','D'),30,g_oDisplayOutputSize,true);
#endif //WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if WRITE_BGSUB_METRICS_ANALYSIS
		time_t startup = time(nullptr);
#endif //WRITE_BGSUB_METRICS_ANALYSIS
		const size_t nNbInputFrames = pCurrSequence->GetNbInputFrames();
		for(size_t k=0; k<nNbInputFrames; k++) {
			if(!(k%100))
				std::cout << "\t\t" << std::setw(12) << pCurrSequence->m_sName << " @ F:" << k << "/" << nNbInputFrames << std::endl;
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
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
			std::chrono::high_resolution_clock::time_point pre_process = std::chrono::high_resolution_clock::now();
#endif //ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
			(*pBGS)(oInputImg, oFGMask, k<=100?1:dDefaultLearningRate);
#if ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
			std::chrono::high_resolution_clock::time_point post_process = std::chrono::high_resolution_clock::now();
			std::cout << "proc=" << std::fixed << std::setprecision(1) << (float)(std::chrono::duration_cast<std::chrono::microseconds>(post_process-pre_process).count())/1000 << "." << std::endl;
#endif //ENABLE_FRAME_TIMERS && PLATFORM_SUPPORTS_CPP11
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
			if(nKeyPressed==32)
				g_bContinuousUpdates = !g_bContinuousUpdates;

			// highway
			// prev best:			Rcl=0.931093, Prc=0.930421, FMs=0.930757
			// descdistthrs^2 :		Rcl=0.932013, Prc=0.917029, FMs=0.92446
			// rel_lbsp = 275/3:	Rcl=0.939192, Prc=0.926686, FMs=0.932897
			// t(x) via Dmix:		Rcl=0.934150, Prc=0.930690, FMs=0.93242
			// t(x) via Dlast:		Rcl=0.934270, Prc=0.930470, FMs=0.93237
			// 2014/01/31 eod:		Rcl=0.950739, Prc=0.929408, FMs=0.939952
			// 2014/02/03 16:19:	Rcl=0.948456, Prc=0.934025, FMs=0.941185
			// ^ + extrablinks/2:	Rcl=0.949355, Prc=0.934115, FMs=0.941673
			// ^ + sc=curr/2:		Rcl=0.949667, Prc=0.934234, FMs=0.941887			#tag_B
			// ^ + r2offst=150:		Rcl=0.950451, Prc=0.933824, FMs=0.942064			#tag_A
			// ^ + r2offst=200:		Rcl=0.960891, Prc=0.924784, FMs=0.942492			#tag_C
			// ^ + r2offst=075:		Rcl=0.949670, Prc=0.934314, FMs=0.941929			#tag_D
			// tag_A + r2incr=1.5:	Rcl=0.944220, Prc=0.933833, FMs=0.938998			#tag_E
			// ^ + r2decr=150(/8):	Rcl=0.950329, Prc=0.934202, FMs=0.942197			#tag_F
			// tag_F + tincr=16:	Rcl=0.950273, Prc=0.934078, FMs=0.942106			#tag_G
			// tag_G+bits+gw_cdist:	Rcl=0.920550, Prc=0.933174, FMs=0.926819			#tag_H
			// tag_G+gw_cdist:		Rcl=0.929573, Prc=0.934089, FMs=0.931825			#tag_I
			// tag_G+cdist+norm_sc:	Rcl=0.908795, Prc=0.931999, FMs=0.920251
			// tag_G+all_cdist_tr:	Rcl=0.900433, Prc=0.931819, FMs=0.915857
			// tag_G+subsenseR/R2:	Rcl=0.944790, Prc=0.934720, FMs=0.939730			#tag_J
			// tag_J+unst_no_wupdt:	Rcl=0.945965, Prc=0.934780, FMs=0.940339			#tag_K
			// #K+24/6+r_halt_unst:	Rcl=0.858190, Prc=0.930450, FMs=0.892860
			// #K+24/6+r_half_unst: Rcl=0.957600, Prc=0.934190, FMs=0.945750			#L
			// #K+24/6+r_qurt_unst: Rcl=0.957539, Prc=0.934152, FMs=0.945701			#M
			// #L + 1/2 R2 decr fg: Rcl=0.958552, Prc=0.934039, FMs=0.946137			#N
			// #L + 1/4 R2 decr fg: Rcl=0.957937, Prc=0.934159, FMs=0.945899			#O
			// #M + cdist_gwords:	Rcl=0.958708, Prc=0.934256, FMs=0.946325			#P
			// #P + 26/6 thres:		Rcl=0.938563, Prc=0.934272, FMs=0.936412			#Q
			// #Q + 2dilate:		Rcl=0.925830, Prc=0.933810, FMs=0.929803			#R
			// #Q + T_var_dilated:	Rcl=0.939083, Prc=0.934175, FMs=0.936623			#S
			// #S + poly01prms - T: Rcl=0.941722, Prc=0.932715, FMs=0.937197			#T
			// #753bddf				Rcl=0.952214, Prc=0.934343, FMs=0.943194			#U
			// ^ + T(x)_meanmind:	Rcl=0.947081, Prc=0.934162, FMs=0.940577			#V
			// #V + wgt=1/sqrt(R):  Rcl=0.952034, Prc=0.933851, FMs=0.942855            #W
			// #1dc437f4			Rcl=0.948100, Prc=0.934004, FMs=0.940999			#X
			// #X + const_Tx_1_20:  Rcl=0.947506, Prc=0.934002, FMs=0.940706
			// #X + T_non_dilated:  Rcl=0.947704, Prc=0.933961, FMs=0.940782
			// #X + 1500wo:        .Rcl=0.949670, Prc=0.933803, FMs=0.941669			#Z
			// #Z + 325rel:        .Rcl=0.944531, Prc=0.934109, FMs=0.939291
			// #Z + T_non_dilated: .Rcl=0.950269, Prc=0.933923, FMs=0.942025			#AA
			// #AA + 5x5 nghb:     .Rcl=0.952195, Prc=0.933429, FMs=0.942719
			// ^ + no neighb updt: .Rcl=0.951750, Prc=0.933454, FMs=0.942513
			// ^ + nbupd cdist .5: .Rcl=0.950941, Prc=0.933493, FMs=0.942136
			// ^ + nbupd cdist:    .Rcl=0.952254, Prc=0.933941, FMs=0.943009
			// ^ + nbupd cdist-var:.Rcl=0.938155, Prc=0.933723, FMs=0.935933
			// @@@@ 4/28/365+3     .Rcl=0.945168, Prc=0.931436, FMs=0.938252
			// @@@@ 4/28/365+0     .Rcl=0.943129, Prc=0.928360, FMs=0.935687
			// @@@@ 5/28/325+3     .Rcl=0.940768, Prc=0.934186, FMs=0.937465
			// ^ with fg_mmd=1.0   .Rcl=0.934910, Prc=0.933533, FMs=0.934221
			// ^ w fg_mmd=max(clsc).Rcl=0.916206, Prc=0.932744, FMs=0.924401
			// ^ w mmd=mrsr        .Rcl=0.944148, Prc=0.934423, FMs=0.939260
			// ^ w 3.5s & ok_mmd   .Rcl=0.932723, Prc=0.934001, FMs=0.933361


			// office:
			// #T:					Rcl=0.881170, Prc=0.945362, FMs=0.912138
			//	 ghost_dmax=0080:	Rcl=0.888345, Prc=0.947286, FMs=0.916870
			//   4occincr ghost:	Rcl=0.898930, Prc=0.946446, FMs=0.922076
			// #U:					Rcl=0.949649, Prc=0.942024, FMs=0.945821
			// #V:					Rcl=0.959248, Prc=0.947236, FMs=0.953204
			// #W:					Rcl=0.956419, Prc=0.943360, FMs=0.949845
			// #X:					Rcl=0.958136, Prc=0.942816, FMs=0.950414
			// #X + const_Tx_1_20:  Rcl=0.957556, Prc=0.944135, FMs=0.950798
			// #X + T_non_dilated:  Rcl=0.954482, Prc=0.943574, FMs=0.948997
			// #X + 1500wo:        .Rcl=0.959961, Prc=0.937377, FMs=0.948534
			// #Z + 325rel:        .Rcl=0.951381, Prc=0.944445, FMs=0.947900
			// #Z + T_non_dilated: .Rcl=0.955784, Prc=0.940786, FMs=0.948226
			// #AA + 5x5 nghb:     .Rcl=0.960838, Prc=0.942902, FMs=0.951786
			// ^ + no neighb updt: .Rcl=0.960214, Prc=0.943282, FMs=0.951673
			// ^ + nbupd cdist .5: .Rcl=0.957881, Prc=0.944487, FMs=0.951137
			// ^ + nbupd cdist:    .Rcl=0.962896, Prc=0.938753, FMs=0.950671
			// ^ + nbupd cdist-var:.Rcl=0.958422, Prc=0.930467, FMs=0.944238
			// @@@@@@              .Rcl=0.936757, Prc=0.963799, FMs=0.950086
			// @@@@ 4/28/365+3     .Rcl=0.943839, Prc=0.961624, FMs=0.952649
			// @@@@ 4/28/365+0     .Rcl=0.954291, Prc=0.953103, FMs=0.953697
			// @@@@ 5/28/325+3     .Rcl=0.936819, Prc=0.961758, FMs=0.949125
			// ^ with fg_mmd=1.0   .Rcl=0.939968, Prc=0.961773, FMs=0.950745
			// ^ w fg_mmd=max(clsc).Rcl=0.942422, Prc=0.961160, FMs=0.951699
			// ^ w mmd=mrsr        .Rcl=0.949341, Prc=0.952398, FMs=0.950867
			// ^ w 3.5s & ok_mmd   .Rcl=0.940379, Prc=0.963501, FMs=0.951799


			// pedestrians
			// #AA + 5x5 nghb:     .Rcl=0.930446, Prc=0.970780, FMs=0.950185
			// ^ + no neighb updt: .Rcl=0.935010, Prc=0.970319, FMs=0.952338
			// ^ + nbupd cdist .5: .Rcl=0.931584, Prc=0.971455, FMs=0.951101
			// ^ + nbupd cdist:    .Rcl=0.934204, Prc=0.969504, FMs=0.951527
			// ^ + nbupd cdist-var:.Rcl=0.938700, Prc=0.967485, FMs=0.952875
			// @@@@ 4/28/365+3     .Rcl=0.917150, Prc=0.979489, FMs=0.947295
			// @@@@ 4/28/365+0     .Rcl=0.914798, Prc=0.979124, FMs=0.945868
			// @@@@ 5/28/325+3     .Rcl=0.926150, Prc=0.974762, FMs=0.949834
			// ^ with fg_mmd=1.0   .Rcl=0.924473, Prc=0.974321, FMs=0.948743
			// ^ w fg_mmd=max(clsc).Rcl=0.920207, Prc=0.975755, FMs=0.947167
			// ^ w mmd=mrsr        .Rcl=0.927528, Prc=0.974658, FMs=0.950509
			// ^ w 3.5s & ok_mmd   .


			// PETS2006
			// #Q:					Rcl=0.916534, Prc=0.919027, FMs=0.917779
			// #S:					Rcl=0.919912, Prc=0.915745, FMs=0.917824
			// #V					Rcl=0.939782, Prc=0.901046, FMs=0.920007
			// #X:					Rcl=0.942293, Prc=0.897195, FMs=0.919191
			// #X + const_Tx_1_20:  Rcl=0.942630, Prc=0.895940, FMs=0.918692
			// #X + T_non_dilated:  Rcl=0.941642, Prc=0.902070, FMs=0.921431
			// #X + 1500wo:        .Rcl=0.942165, Prc=0.898186, FMs=0.919650
			// #Z + 325rel:        .Rcl=0.934542, Prc=0.899725, FMs=0.916803
			// #Z + T_non_dilated: .Rcl=0.941465, Prc=0.901419, FMs=0.921007
			// #AA + 5x5 nghb:     .Rcl=0.948637, Prc=0.891855, FMs=0.919370
			// ^ + no neighb updt: .Rcl=0.945858, Prc=0.896351, FMs=0.920440
			// ^ + nbupd cdist .5: .Rcl=0.949153, Prc=0.895178, FMs=0.921376
			// ^ + nbupd cdist:    .Rcl=0.961363, Prc=0.890554, FMs=0.924605
			// ^ + nbupd cdist-var:.Rcl=0.956397, Prc=0.896259, FMs=0.925352
			// @@@@ 4/28/365+3     .Rcl=0.928400, Prc=0.912024, FMs=0.920139
			// @@@@ 4/28/365+0     .Rcl=0.947138, Prc=0.901892, FMs=0.923962
			// @@@@ 5/28/325+3     .Rcl=0.927637, Prc=0.918975, FMs=0.923286
			// ^ with fg_mmd=1.0   .Rcl=0.929308, Prc=0.910527, FMs=0.919821
			// ^ w fg_mmd=max(clsc).Rcl=0.930500, Prc=0.908129, FMs=0.919178
			// ^ w mmd=mrsr        .Rcl=0.932657, Prc=0.901628, FMs=0.916880
			// ^ w 3.5s & ok_mmd   .


			// sidewalk
			// #Q:					Rcl=0.298911, Prc=0.944698, FMs=0.454131
			// #R:					Rcl=0.266669, Prc=0.955489, FMs=0.416966
			// #S:					Rcl=0.300202, Prc=0.948541, FMs=0.456065
			// #T:					Rcl=0.470909, Prc=0.919542, FMs=0.622849
			// #V:					Rcl=0.394588, Prc=0.958474, FMs=0.559032
			// #X:					Rcl=0.432202, Prc=0.945773, FMs=0.593283
			// #X + const_Tx_1_20:	Rcl=0.405685, Prc=0.948797, FMs=0.568354
			// #X + T_non_dilated:  Rcl=0.391017, Prc=0.946734, FMs=0.553450
			// #X + 1500wo:        .Rcl=0.420258, Prc=0.951138, FMs=0.582943
			// #Z + 325rel:        .Rcl=0.207853, Prc=0.969814, FMs=0.342336
			// #Z + T_non_dilated: .Rcl=0.377907, Prc=0.948782, FMs=0.540520
			// #AA + 5x5 nghb:     .Rcl=0.440248, Prc=0.944813, FMs=0.600626
			// ^ + no neighb updt: .Rcl=0.458798, Prc=0.942124, FMs=0.617086
			// ^ + nbupd cdist .5: .Rcl=0.429816, Prc=0.940732, FMs=0.590044
			// ^ + nbupd cdist:    .Rcl=0.491887, Prc=0.943443, FMs=0.646635
			// ^ + nbupd cdist-var:.Rcl=0.488130, Prc=0.944099, FMs=0.643533
			// @@@@ 4/28/365+3     .


			// boats
			// tag_J:				Rcl=0.470857, Prc=0.939418, FMs=0.627298
			// tag_J + 24/6 thrs:	Rcl=0.565080, Prc=0.869930, FMs=0.685120
			// #L:					Rcl=0.553656, Prc=0.872358, FMs=0.677393
			// #M:					Rcl=0.527570, Prc=0.874200, FMs=0.658030
			// #N:					Rcl=0.529364, Prc=0.863368, FMs=0.656316
			// #Q:					Rcl=0.496703, Prc=0.933441, FMs=0.648386
			// #R:					Rcl=0.488918, Prc=0.939676, FMs=0.643184
			// #S:					Rcl=0.562238, Prc=0.931999, FMs=0.701368
			// #X:					Rcl=0.660993, Prc=0.898521, FMs=0.761668
			// #X + const_Tx_1_20:  Rcl=0.669569, Prc=0.897442, FMs=0.766937
			// #X + T_non_dilated:  Rcl=0.619758, Prc=0.897892, FMs=0.733339
			// #X + 1500wo:        .Rcl=0.670614, Prc=0.919972, FMs=0.775747
			// #Z + 325rel:        .Rcl=0.679892, Prc=0.912957, FMs=0.779373
			// #Z + T_non_dilated: .Rcl=0.618060, Prc=0.924392, FMs=0.740807
			// #AA + 5x5 nghb:     .Rcl=0.674509, Prc=0.905729, FMs=0.773203
			// ^ + no neighb updt: .Rcl=0.653268, Prc=0.890369, FMs=0.753610
			// ^ + nbupd cdist .5: .Rcl=0.672591, Prc=0.907597, FMs=0.772619
			// ^ + nbupd cdist:    .Rcl=0.729282, Prc=0.887098, FMs=0.800485
			// ^ + nbupd cdist-var:.Rcl=0.720913, Prc=0.886185, FMs=0.795051
			// @@@@ 4/28/365+3     .


			// overpass
			// tag_A:				Rcl=0.830823, Prc=0.892503, FMs=0.860559
			// tag_D:				Rcl=0.830510, Prc=0.890189, FMs=0.859315
			// tag_C:				Rcl=0.861188, Prc=0.247715, FMs=0.384757
			// tag_A + r2incr=1.5:	Rcl=0.816230, Prc=0.916030, FMs=0.863250
			// tag_F:				Rcl=0.840330, Prc=0.875020, FMs=0.857320
			// tag_G:				Rcl=0.841790, Prc=0.876165, FMs=0.858634
			// tag_H:				Rcl=0.843985, Prc=0.897190, FMs=0.869775
			// tag_I:				Rcl=0.872832, Prc=0.856349, FMs=0.864512
			// tag_J:				Rcl=0.829180, Prc=0.867960, FMs=0.848130
			// #M:					Rcl=0.867367, Prc=0.929980, FMs=0.897583
			// #N:					Rcl=0.888656, Prc=0.759753, FMs=0.819165
			// #O:					Rcl=0.872233, Prc=0.904826, FMs=0.888230
			// #P:					Rcl=0.927732, Prc=0.746578, FMs=0.827355
			// #Q:					Rcl=0.879180, Prc=0.916280, FMs=0.897347
			// #R:					Rcl=0.861482, Prc=0.939683, FMs=0.898885
			// #S:					Rcl=0.923283, Prc=0.918754, FMs=0.921013
			// #T:					Rcl=0.931775, Prc=0.928482, FMs=0.930126
			// #X:					Rcl=0.951544, Prc=0.904461, FMs=0.927406
			// #X + const_Tx_1_20:  Rcl=0.955936, Prc=0.791134, FMs=0.865762
			// #X + T_non_dilated:  Rcl=0.945041, Prc=0.898851, FMs=0.921368
			// #X + 1500wo:        .Rcl=0.917736, Prc=0.932141, FMs=0.924882
			// #Z + 325rel:        .Rcl=0.905045, Prc=0.932449, FMs=0.918543
			// #Z + T_non_dilated: .Rcl=0.909279, Prc=0.925521, FMs=0.917328
			// #AA + 5x5 nghb:     .Rcl=0.919610, Prc=0.907380, FMs=0.913454
			// ^ + no neighb updt: .Rcl=0.941998, Prc=0.765543, FMs=0.844653
			// ^ + nbupd cdist .5: .Rcl=0.926528, Prc=0.764415, FMs=0.837700
			// ^ + nbupd cdist:    .Rcl=0.916914, Prc=0.909016, FMs=0.912948
			// ^ + nbupd cdist-var:.Rcl=0.875142, Prc=0.921976, FMs=0.897949
			// @@@@ 4/28/365+3     .


			// fountain01
			// #Q:					Rcl=0.801221, Prc=0.795687, FMs=0.798444
			// #S:					Rcl=0.796137, Prc=0.792670, FMs=0.794400
			// #T:					Rcl=0.841555, Prc=0.761737, FMs=0.799659
			// #V:					Rcl=0.812265, Prc=0.697703, FMs=0.750638
			// #W:					Rcl=0.808936, Prc=0.759893, FMs=0.783648
			// #X:					Rcl=0.816523, Prc=0.710933, FMs=0.760079
			// #X + const_Tx_1_20:  Rcl=0.823620, Prc=0.697667, FMs=0.755429
			// #X + T_non_dilated:  Rcl=0.821013, Prc=0.708925, FMs=0.760863
			// #X + 1500wo:        .Rcl=0.841787, Prc=0.711764, FMs=0.771334
			// #Z + 325rel:        .Rcl=0.746242, Prc=0.814202, FMs=0.778742
			// #Z + T_non_dilated: .Rcl=0.836045, Prc=0.713667, FMs=0.770024
			// #AA + 5x5 nghb:     .Rcl=0.868444, Prc=0.700954, FMs=0.775761
			// ^ + no neighb updt: .Rcl=0.857618, Prc=0.691000, FMs=0.765346
			// ^ + nbupd cdist .5: .Rcl=0.865928, Prc=0.694741, FMs=0.770946
			// ^ + nbupd cdist:    .Rcl=0.838264, Prc=0.803165, FMs=0.820339
			// ^ + nbupd cdist-var:.Rcl=0.847451, Prc=0.763499, FMs=0.803288
			// @@@@ 4/28/365+3     .


			// fountain02
			// tag_G:				Rcl=0.910792, Prc=0.783915, FMs=0.842604
			// tag_H:				Rcl=0.908098, Prc=0.917706, FMs=0.912877
			// tag_I:				Rcl=0.902608, Prc=0.855943, FMs=0.878656
			// tag_J:				Rcl=0.860610, Prc=0.976250, FMs=0.914790
			// #M:					Rcl=0.920896, Prc=0.797068, FMs=0.854519
			// #O:					Rcl=0.918067, Prc=0.791589, FMs=0.850149
			// #P:					Rcl=0.925588, Prc=0.833369, FMs=0.877061
			// #Q:					Rcl=0.918620, Prc=0.943247, FMs=0.930771
			// #S:					Rcl=0.918602, Prc=0.923016, FMs=0.920803
			// #T:					Rcl=0.933888, Prc=0.955309, FMs=0.944477
			// #V:					Rcl=0.922003, Prc=0.940379, FMs=0.931101
			// #X:					Rcl=0.924301, Prc=0.885098, FMs=0.904275
			// #X + const_Tx_1_20:  Rcl=0.921869, Prc=0.885861, FMs=0.903506
			// #X + T_non_dilated:  Rcl=0.920319, Prc=0.873296, FMs=0.896191
			// #X + 1500wo:        .Rcl=0.913292, Prc=0.893864, FMs=0.903473
			// #Z + 325rel:        .Rcl=0.911196, Prc=0.923842, FMs=0.917476
			// #Z + T_non_dilated: .Rcl=0.907454, Prc=0.944988, FMs=0.925841
			// #AA + 5x5 nghb:     .Rcl=0.930273, Prc=0.828201, FMs=0.876274
			// ^ + no neighb updt: .Rcl=0.916959, Prc=0.852176, FMs=0.883381
			// ^ + nbupd cdist .5: .Rcl=0.923025, Prc=0.885889, FMs=0.904076
			// ^ + nbupd cdist:    .Rcl=0.930801, Prc=0.804833, FMs=0.863246
			// ^ + nbupd cdist-var:.Rcl=0.931486, Prc=0.790862, FMs=0.855433
			// @@@@ 4/28/365+3     .


			// tramstop
			// #Q:					Rcl=0.545644, Prc=0.508917, FMs=0.526641
			// #R:					Rcl=0.514407, Prc=0.528120, FMs=0.521173
			// #S:					Rcl=0.572501, Prc=0.475996, FMs=0.519807
			// #T:					Rcl=0.438114, Prc=0.490211, FMs=0.462701
			// #V:					Rcl=0.613563, Prc=0.465263, FMs=0.52922
			// #X:					Rcl=0.670684, Prc=0.471854, FMs=0.553968
			// #X + const_Tx_1_20:  Rcl=0.663521, Prc=0.471381, FMs=0.551186
			// #X + T_non_dilated:  Rcl=0.677659, Prc=0.511894, FMs=0.583227
			// #X + 1500wo:        .Rcl=0.715466, Prc=0.471783, FMs=0.568616
			// #Z + 325rel:        .Rcl=0.686587, Prc=0.496836, FMs=0.576499
			// #Z + T_non_dilated: .Rcl=0.739224, Prc=0.481555, FMs=0.583196
			// #AA + 5x5 nghb:     .Rcl=0.712121, Prc=0.459814, FMs=0.558807
			// ^ + no neighb updt: .Rcl=0.763005, Prc=0.445172, FMs=0.562283
			// ^ + nbupd cdist .5: .Rcl=0.749791, Prc=0.455051, FMs=0.566370
			// ^ + nbupd cdist:    .Rcl=0.733018, Prc=0.450582, FMs=0.558102
			// ^ + nbupd cdist-var:.Rcl=0.762600, Prc=0.479459, FMs=0.588757
			// @@@@ 4/28/365+3     .


			// winterDriveway
			// #Q:					Rcl=0.665845, Prc=0.182505, FMs=0.286486
			// #S:					Rcl=0.669613, Prc=0.148552, FMs=0.243159
			// #T:					Rcl=0.725818, Prc=0.186702, FMs=0.297005
			// #V:					Rcl=0.688900, Prc=0.110206, FMs=0.190015
			// #X:					Rcl=0.713443, Prc=0.109719, FMs=0.190189
			// #X + const_Tx_1_20:  Rcl=0.691151, Prc=0.110000, FMs=0.189793
			// #X + T_non_dilated:  Rcl=0.692605, Prc=0.120976, FMs=0.205975
			// #X + 1500wo:        .Rcl=0.713356, Prc=0.101121, FMs=0.177132
			// #Z + 325rel:        .Rcl=0.698821, Prc=0.124364, FMs=0.211150
			// #Z + T_non_dilated: .Rcl=0.680084, Prc=0.111589, FMs=0.191720
			// #AA + 5x5 nghb:     .Rcl=0.716479, Prc=0.125520, FMs=0.213617
			// ^ + no neighb updt: .Rcl=0.715192, Prc=0.105221, FMs=0.183452
			// ^ + nbupd cdist .5: .Rcl=0.724181, Prc=0.121361, FMs=0.207885
			// ^ + nbupd cdist:    .Rcl=0.730866, Prc=0.143700, FMs=0.240177
			// ^ + nbupd cdist-var:.Rcl=0.720769, Prc=0.137018, FMs=0.230263
			// @@@@ 4/28/365+3     .


			// cubicle
			// tag_G:  				Rcl=0.968970, Prc=0.710990, FMs=0.820170
			// tag_G + cdistmix:	Rcl=0.956708, Prc=0.741298, FMs=0.835339
			// tag_J:				Rcl=0.962832, Prc=0.788824, FMs=0.867185
			// #P:					Rcl=0.965794, Prc=0.713589, FMs=0.820754
			// #Q:					Rcl=0.960422, Prc=0.748846, FMs=0.841540
			// #S:					Rcl=0.962342, Prc=0.719896, FMs=0.823648
			// #T:					Rcl=0.967106, Prc=0.728511, FMs=0.831021
			// #X:					Rcl=0.962703, Prc=0.733938, FMs=0.832898
			// #X + const_Tx_1_20:  Rcl=0.961532, Prc=0.708199, FMs=0.815648
			// #X + T_non_dilated:  Rcl=0.962072, Prc=0.735192, FMs=0.833468
			// #X + 1500wo:        .Rcl=0.956824, Prc=0.725477, FMs=0.825244
			// #Z + 325rel:        .Rcl=0.958707, Prc=0.730822, FMs=0.829396
			// #Z + T_non_dilated: .Rcl=0.955358, Prc=0.736223, FMs=0.831597
			// #AA + 5x5 nghb:     .Rcl=0.965044, Prc=0.734526, FMs=0.834152
			// ^ + no neighb updt: .Rcl=0.960935, Prc=0.721335, FMs=0.824072
			// ^ + nbupd cdist .5: .Rcl=0.964017, Prc=0.731921, FMs=0.832087
			// ^ + nbupd cdist:    .Rcl=0.968105, Prc=0.733679, FMs=0.834746
			// ^ + nbupd cdist-var:.Rcl=0.969891, Prc=0.736067, FMs=0.836955
			// @@@@ 4/28/365+3     .



			// lakeSide
			// prev best:			Rcl=0.200375, Prc=0.973144, FMs=0.332323   (3ch bug?)
			// no_mod_grad_prop:	Rcl=0.200375, Prc=0.973144, FMs=0.332323   (3ch bug?)
			// 2014/01/31 lakeSide:	Rcl=0.189571, Prc=0.973639, FMs=0.317352   (3ch bug?) -- 500mod
			// 2014/01/31 lakeSide: Rcl=0.569957, Prc=0.718582, FMs=0.635698   (1ch fixd) -- 500mod
			// w modgradprop:		Rcl=0.519131, Prc=0.771384, FMs=0.620604   500mod
			// inverted:			bad (+++++recall, ------prec)
			// w/o modgradprop:		Rcl=0.767510, Prc=0.597617, FMs=0.671992   400mod
			// w/o modgrap+modlbsp: Rcl=0.582556, Prc=0.819366, FMs=0.680961   400mod
			// ^, but 1chlbsp=200:  Rcl=0.626148, Prc=0.770467, FMs=0.690851   400mod
			// ^, but 1chlbsp=150:  Rcl=0.675936, Prc=0.698206, FMs=0.686891   400mod
			// tag_G:				Rcl=0.462617, Prc=0.803432, FMs=0.587151
			// tag_J:				Rcl=0.362808, Prc=0.858112, FMs=0.509992
			// tag_J + 350mod:		Rcl=0.421901, Prc=0.819222, FMs=0.556964
			// tag_J+350m+rel_mod:	Rcl=0.828399, Prc=0.524355, FMs=0.642209
			// tag_J+400m+rel_mod:	Rcl=0.709958, Prc=0.576107, FMs=0.636067
			// tag_J+500m+rel_mod:	Rcl=0.540496, Prc=0.666046, FMs=0.596739
			// #Q:					Rcl=0.855502, Prc=0.514804, FMs=0.642799
			// #AA + 5x5 nghb:     . BAAAAAAAAAAAAAAD (recall way too high)


			// f3805d5:
			//	baseline:			Rcl=0.935799, Prc=0.937523, FMs=0.936082
			//	shadow_cubicle:		Rcl=0.955704, Prc=0.765950, FMs=0.850370
			// + 333/0 cfg:
			//	baseline:			Rcl=0.938634, Prc=0.937781, FMs=0.937626
			//	shadow_cubicle:		Rcl=0.963893, Prc=0.704842, FMs=0.814260
			// + 300/3 cfg:
			//	baseline:			Rcl=0.940739, Prc=0.938287, FMs=0.939119
			//	shadow_cubicle:		Rcl=0.964089, Prc=0.753964, FMs=0.846177
			// + 300/0 cfg:
			//	baseline:			Rcl=0.940689, Prc=0.935401, FMs=0.937448
			//	shadow_cubicle:		Rcl=0.965565, Prc=0.707248, FMs=0.816462
			// + 300/3/28 cfg:
			//	baseline:			Rcl=0.940165, Prc=0.934881, FMs=0.937140
			//	shadow_cubicle:		Rcl=0.962358, Prc=0.756256, FMs=0.846949
			// // + 300/3 - 1500wo/25/25 cfg:
			//	baseline:			Rcl=0.940584, Prc=0.939082, FMs=0.939389
			//	shadow_cubicle:		Rcl=0.967217, Prc=0.693129, FMs=0.807550

#endif //DISPLAY_BGSUB_DEBUG_OUTPUT
#if WRITE_BGSUB_IMG_OUTPUT
			WriteResult(g_sResultsPath,pCurrCategory->m_sName,pCurrSequence->m_sName,g_sResultPrefix,k+g_nResultIdxOffset,g_sResultSuffix,oFGMask,g_vnResultsComprParams);
#endif //WRITE_BGSUB_IMG_OUTPUT
#if WRITE_BGSUB_METRICS_ANALYSIS
			CalcMetricsFromResult(oFGMask,oGTImg,pCurrSequence->GetSequenceROI(),pCurrSequence->nTP,pCurrSequence->nTN,pCurrSequence->nFP,pCurrSequence->nFN,pCurrSequence->nSE);
		}
		time_t shutdown = time(nullptr);
		pCurrSequence->m_dAvgFPS = ((double)nNbInputFrames)/(shutdown-startup);
		WriteMetrics(g_sResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+".txt",pCurrSequence);
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
		cv::cvtColor(oInputImg,oInputImgBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oBGImg,oBGImgBYTE3,CV_GRAY2RGB);
	}
	else {
		oInputImgBYTE3 = oInputImg;
		oBGImgBYTE3 = oBGImg;
	}
	oFGMaskBYTE3 = GetColoredSegmFrameFromResult(oFGMask,oGTFGMask,oROI);
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
	WriteOnImage(oInputImgBYTE3,sstr.str());
	WriteOnImage(oBGImgBYTE3,"Reference Image");
	WriteOnImage(oFGMaskBYTE3,"Segmentation Result");

	cv::hconcat(oInputImgBYTE3,oBGImgBYTE3,displayH);
	cv::hconcat(displayH,oFGMaskBYTE3,displayH);
	return displayH;
}
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
