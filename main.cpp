#include "PlatformUtils.h"
#include "DatasetUtils.h"
#include "BackgroundSubtractorPBASLBSP.h"
#include "BackgroundSubtractorViBeLBSP.h"
#include "BackgroundSubtractorViBe_1ch.h"
#include "BackgroundSubtractorViBe_3ch.h"
#include "BackgroundSubtractorPBAS.h"

/////////////////////////////////////////
// USER/ENVIRONMENT-SPECIFIC VARIABLES :
/////////////////////////////////////////
#define WRITE_BGSUB_IMG_OUTPUT			0
#define WRITE_BGSUB_DEBUG_IMG_OUTPUT	0
#define WRITE_BGSUB_METRICS_ANALYSIS	1
/////////////////////////////////////////
#define DISPLAY_BGSUB_DEBUG_OUTPUT		0
/////////////////////////////////////////
#define USE_VIBE_LBSP_BG_SUBTRACTOR		0
#define USE_PBAS_LBSP_BG_SUBTRACTOR		0
#define USE_VIBE_BG_SUBTRACTOR			0
#define USE_PBAS_BG_SUBTRACTOR			1
/////////////////////////////////////////
#if (USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
#define USE_RELATIVE_LBSP_COMPARISONS	1
#endif //(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
/////////////////////////////////////////
#define USE_CDNET_DATASET				1
#define USE_WALLFLOWER_DATASET			0
#define USE_PETS2001_D3TC1_DATASET		0
/////////////////////////////////////////
#define DEFAULT_NB_THREADS				8
/////////////////////////////////////////
#define FORCE_STDOUT_FLUSH_FOR_ECLIPSE	0
/////////////////////////////////////////////////////////////////////
#define DATASET_ROOT_DIR 				std::string("/tmp/datasets/")
#define RESULTS_ROOT_DIR 				std::string("/usagers/pistc/datasets/")
#define RESULTS_OUTPUT_DIR_NAME			std::string("results_testpbas_iocache")
/////////////////////////////////////////////////////////////////////

#if (USE_VIBE_LBSP_BG_SUBTRACTOR+USE_PBAS_LBSP_BG_SUBTRACTOR+USE_VIBE_BG_SUBTRACTOR+USE_PBAS_BG_SUBTRACTOR)!=1
#error "Must specify a single algorithm."
#elif (USE_CDNET_DATASET+USE_WALLFLOWER_DATASET+USE_PETS2001_D3TC1_DATASET)!=1
#error "Must specify a single dataset."
#elif USE_CDNET_DATASET
const std::string g_sDatasetName(CDNET_DB_NAME);
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/CDNet/dataset/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/CDNet/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetCategories[] = {"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"};
const int g_nResultIdxOffset = 1;
#elif USE_WALLFLOWER_DATASET
const std::string g_sDatasetName(WALLFLOWER_DB_NAME);
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/Wallflower/dataset/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/Wallflower/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetCategories[] = {"global"};
const int g_nResultIdxOffset = 0;
#elif USE_PETS2001_D3TC1_DATASET
const std::string g_sDatasetName(PETS2001_D3TC1_DB_NAME);
const std::string g_sDatasetPath(DATASET_ROOT_DIR+"/PETS2001/DATASET3/");
const std::string g_sResultsPath(RESULTS_ROOT_DIR+"/PETS2001/DATASET3/"+RESULTS_OUTPUT_DIR_NAME+"/");
const std::string g_sResultPrefix("bin");
const std::string g_sResultSuffix(".png");
const char* g_asDatasetCategories[] = {"TESTING"};
const int g_nResultIdxOffset = 0;
#endif

int AnalyzeSequence(int nThreadIdx, CategoryInfo* pCurrCategory, SequenceInfo* pCurrSequence);

#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if !USE_VIBE_BG_SUBTRACTOR && !USE_PBAS_BG_SUBTRACTOR
cv::Size g_oDisplayOutputSize(960,240);
cv::Mat GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oBGDesc, const cv::Mat& oFGMask, std::vector<cv::KeyPoint> voKeyPoints, size_t nFrame);
#else //USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
cv::Size g_oDisplayOutputSize(800,240);
cv::Mat GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oFGMask, size_t nFrame);
#endif //USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT

#if PLATFORM_SUPPORTS_CPP11
const size_t g_nMaxThreads = /*std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():*/DEFAULT_NB_THREADS;
std::atomic_size_t g_nActiveThreads(0);
#if WRITE_BGSUB_IMG_OUTPUT
const std::vector<int> g_vnResultsComprParams = {CV_IMWRITE_PNG_COMPRESSION,9}; // when writing output bin files, lower to increase processing speed
#endif //WRITE_BGSUB_IMG_OUTPUT
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
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

int main( int argc, char** argv ) {
	srand(0); // for now, assures that two consecutive runs on the same data return the same results
#if FORCE_STDOUT_FLUSH_FOR_ECLIPSE
	setvbuf(stdout, NULL, _IONBF, 0); // fixes output flush problems when using the eclipse built-in console
#endif //FORCE_STDOUT_FLUSH_FOR_ECLIPSE
	std::vector<CategoryInfo*> vpCategories;
	std::cout << "Parsing dataset '"<< g_sDatasetName << "'..." << std::endl;
	try {
		for(size_t i=0; i<sizeof(g_asDatasetCategories)/sizeof(char*); ++i)
			vpCategories.push_back(new CategoryInfo(g_asDatasetCategories[i], g_sDatasetPath+g_asDatasetCategories[i], g_sDatasetName, (std::string("thermal")==g_asDatasetCategories[i])?true:false));
	} catch(std::runtime_error& e) { std::cout << e.what() << std::endl; }
	size_t nSeqTotal = 0;
	size_t nFramesTotal = 0;
	for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory) {
		nSeqTotal += (*pCurrCategory)->m_vpSequences.size();
		for(auto pCurrSequence=(*pCurrCategory)->m_vpSequences.begin(); pCurrSequence!=(*pCurrCategory)->m_vpSequences.end(); ++pCurrSequence)
			nFramesTotal += (*pCurrSequence)->GetNbInputFrames();
	}
	std::cout << "Parsing complete. [" << vpCategories.size() << " category(ies), "  << nSeqTotal  << " sequence(s)]" << std::endl << std::endl;
	time_t startup = time(NULL);
	tm* startup_tm = localtime(&startup);
	std::cout << "[" << (startup_tm->tm_year + 1900) << '/' << (startup_tm->tm_mon + 1) << '/' <<  startup_tm->tm_mday << " -- ";
	std::cout << startup_tm->tm_hour << ':' << startup_tm->tm_min << ':' << startup_tm->tm_sec << ']' << std::endl;
	if(nSeqTotal) {
		// since the algorithm isn't implemented to be parallelised yet, we parallelise the sequence treatment instead
		std::cout << "Running LBSP background subtraction with " << ((g_nMaxThreads>nSeqTotal)?nSeqTotal:g_nMaxThreads) << " thread(s)..." << std::endl;
		size_t nSeqProcessed = 1;
#if PLATFORM_SUPPORTS_CPP11
		for(auto& pCurrCategory : vpCategories) {
			for(auto& pCurrSequence : pCurrCategory->m_vpSequences) {
				while(g_nActiveThreads>=g_nMaxThreads)
					std::this_thread::sleep_for(std::chrono::milliseconds(1000));
				std::cout << "\tProcessing sequence " << nSeqProcessed << "/" << nSeqTotal << "... (" << pCurrCategory->m_sName << ":" << pCurrSequence->m_sName << ")" << std::endl;
				g_nActiveThreads++;
				nSeqProcessed++;
				std::thread(AnalyzeSequence,-1,pCurrCategory,pCurrSequence).detach();
			}
		}
		while(g_nActiveThreads>0)
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#elif PLATFORM_USES_WIN32API //&& !PLATFORM_SUPPORTS_CPP11
		for(size_t n=0; n<g_nMaxThreads; ++n)
			g_hThreadEvent[n] = CreateEvent(NULL,FALSE,TRUE,NULL);
		for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory) {
			for(auto pCurrSequence=(*pCurrCategory)->m_vpSequences.begin(); pCurrSequence!=(*pCurrCategory)->m_vpSequences.end(); ++pCurrSequence) {
				DWORD ret = WaitForMultipleObjects(g_nMaxThreads,g_hThreadEvent,FALSE,INFINITE);
				std::cout << "\tProcessing sequence " << nSeqProcessed << "/" << nSeqTotal << "... (" << (*pCurrCategory)->m_sName << ":" << (*pCurrSequence)->m_sName << ")" << std::endl;
				nSeqProcessed++;
				g_apThreadDataStruct[ret][0] = (*pCurrCategory);
				g_apThreadDataStruct[ret][1] = (*pCurrSequence);
				g_hThreads[ret] = CreateThread(NULL,NULL,AnalyzeSequenceEntryPoint,(LPVOID)ret,0,NULL);
			}
		}
		WaitForMultipleObjects((g_nMaxThreads>nSeqTotal)?nSeqTotal:g_nMaxThreads,g_hThreads,TRUE,INFINITE);
		for(size_t n=0; n<g_nMaxThreads; ++n) {
			CloseHandle(g_hThreadEvent[n]);
			CloseHandle(g_hThreads[n]);
		}
#else //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#error "Missing implementation for threads/mutexes/atomic variables on this platform."
#endif //!PLATFORM_USES_WIN32API && !PLATFORM_SUPPORTS_CPP11
#if WRITE_BGSUB_METRICS_ANALYSIS
		std::cout << "Summing and writing metrics results..." << std::endl;
		for(size_t c=0; c<vpCategories.size(); ++c) {
			for(size_t s=0; s<vpCategories[c]->m_vpSequences.size(); ++s) {
				vpCategories[c]->nTP += vpCategories[c]->m_vpSequences[s]->nTP;
				vpCategories[c]->nTN += vpCategories[c]->m_vpSequences[s]->nTN;
				vpCategories[c]->nFP += vpCategories[c]->m_vpSequences[s]->nFP;
				vpCategories[c]->nFN += vpCategories[c]->m_vpSequences[s]->nFN;
				vpCategories[c]->nSE += vpCategories[c]->m_vpSequences[s]->nSE;
			}
			WriteMetrics(g_sResultsPath+vpCategories[c]->m_sName+".txt",vpCategories[c]);
		}
		WriteMetrics(g_sResultsPath+"METRICS_TOTAL.txt",vpCategories);
#endif
		std::cout << "All done." << std::endl;
	}
	else
		std::cout << "No sequences found, all done." << std::endl;
	time_t shutdown = time(NULL);
	tm* shutdown_tm = localtime(&shutdown);
	std::cout << "[" << (shutdown_tm->tm_year + 1900) << '/' << (shutdown_tm->tm_mon + 1) << '/' <<  shutdown_tm->tm_mday << " -- ";
	std::cout << shutdown_tm->tm_hour << ':' << shutdown_tm->tm_min << ':' << shutdown_tm->tm_sec << ']' << std::endl;
	std::cout << "\t ... session averaged " << ((double)nFramesTotal)/(shutdown-startup) << " fps." << std::endl;
	// let memory 'leak' here, exits faster once job is done...
	//for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory)
	//	delete *pCurrCategory;
	//vpCategories.clear();
}

int AnalyzeSequence(int nThreadIdx, CategoryInfo* pCurrCategory, SequenceInfo* pCurrSequence) {
	try {
		CV_DbgAssert(pCurrCategory && pCurrSequence);
		CV_DbgAssert(pCurrSequence->GetNbInputFrames()>1);
#if USE_PRECACHED_IO
		pCurrSequence->StartPrecaching();
#endif //USE_PRECACHED_IO
		cv::Mat oFGMask;
		cv::Mat oInitImg = pCurrSequence->GetInputFrameFromIndex(0);
#if USE_VIBE_LBSP_BG_SUBTRACTOR
		BackgroundSubtractorViBeLBSP
#elif USE_PBAS_LBSP_BG_SUBTRACTOR
		BackgroundSubtractorPBASLBSP
#elif USE_VIBE_BG_SUBTRACTOR
		BackgroundSubtractorViBe_1ch
#elif USE_PBAS_BG_SUBTRACTOR
		BackgroundSubtractorPBAS
#endif //USE_PBAS_BG_SUBTRACTOR
#if !(USE_VIBE_BG_SUBTRACTOR||USE_PBAS_BG_SUBTRACTOR)
#if USE_RELATIVE_LBSP_COMPARISONS
		oBGSubtr(LBSP_DEFAULT_REL_SIMILARITY_THRESHOLD);
#else //!USE_RELATIVE_LBSP_COMPARISONS
		oBGSubtr;
#endif //!USE_RELATIVE_LBSP_COMPARISONS
		oBGSubtr.initialize(oInitImg);
#else //(USE_VIBE_BG_SUBTRACTOR||USE_PBAS_BG_SUBTRACTOR))
		oBGSubtr[3];
		const int m_nInputChannels = oInitImg.channels();
		std::vector<cv::Mat> voInputImg;
		cv::split(oInitImg,voInputImg);
		for(int c=0; c<m_nInputChannels; ++c)
			oBGSubtr[c].initialize(voInputImg[c]);
#endif //(USE_VIBE_BG_SUBTRACTOR||USE_PBAS_BG_SUBTRACTOR))
#if DISPLAY_BGSUB_DEBUG_OUTPUT
		std::string sDebugDisplayName = pCurrCategory->m_sName + std::string(" -- ") + pCurrSequence->m_sName;
#endif //DISPLAY_ANALYSIS_DEBUG_RESULTS
#if WRITE_BGSUB_DEBUG_IMG_OUTPUT
		cv::VideoWriter oDebugWriter(g_sResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+".avi",CV_FOURCC('X','V','I','D'),30,g_oDisplayOutputSize,true);
#endif //WRITE_BGSUB_DEBUG_IMG_OUTPUT
		const size_t nNbInputFrames = pCurrSequence->GetNbInputFrames();
		for(size_t k=0; k<nNbInputFrames; k++) {
			if(!(k%100))
				std::cout << "\t\t" << std::setw(12) << pCurrSequence->m_sName << " @ F:" << k << "/" << nNbInputFrames << std::endl;
			const cv::Mat& oInputImg = pCurrSequence->GetInputFrameFromIndex(k);
#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
			cv::Mat oLastBGImg;
#if USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR
			oBGSubtr.getBackgroundImage(oLastBGImg);
			cv::Mat oLastBGDescImg;
			oBGSubtr.getBackgroundDescriptorsImage(oLastBGDescImg);
#else //!(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
			std::vector<cv::Mat> voLastBGImg(m_nInputChannels);
			for(int c=0; c<m_nInputChannels; ++c)
				oBGSubtr[c].getBackgroundImage(voLastBGImg[c]);
			cv::merge(voLastBGImg,oLastBGImg);
#endif //!(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if USE_VIBE_LBSP_BG_SUBTRACTOR
			oBGSubtr(oInputImg, oFGMask, k<=100?1:BGSVIBELBSP_DEFAULT_LEARNING_RATE);
#elif USE_PBAS_LBSP_BG_SUBTRACTOR
			oBGSubtr(oInputImg, oFGMask, k<=100?1:BGSPBASLBSP_DEFAULT_LEARNING_RATE_OVERRIDE);
#elif USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
			cv::split(oInputImg,voInputImg);
			std::vector<cv::Mat> voFGMask(m_nInputChannels);
			for(int c=0; c<m_nInputChannels; ++c) {
#if USE_PBAS_BG_SUBTRACTOR
				oBGSubtr[c](voInputImg[c], voFGMask[c], k<=100?1:BGSPBAS_DEFAULT_LEARNING_RATE_OVERRIDE);
#else //!USE_PBAS_BG_SUBTRACTOR
				oBGSubtr[c](voInputImg[c], voFGMask[c], k<=100?1:BGSVIBE_DEFAULT_LEARNING_RATE);
#endif //!USE_PBAS_BG_SUBTRACTOR
				if(c==0)
					oFGMask = voFGMask[0];
				else
					cv::bitwise_or(oFGMask,voFGMask[c],oFGMask);
			}
#endif //USE_VIBE_BG_SUBTRACTOR || USE_PBAS_BG_SUBTRACTOR
#if DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
#if USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR
			cv::Mat oDebugDisplayFrame = GetDisplayResult(oInputImg,oLastBGImg,oLastBGDescImg,oFGMask,oBGSubtr.getBGKeyPoints(),k);
#else //!(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
			cv::Mat oDebugDisplayFrame = GetDisplayResult(oInputImg,oLastBGImg,oFGMask,k);
#endif //!(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
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
			cv::waitKey(1);
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT
#if WRITE_BGSUB_IMG_OUTPUT
			WriteResult(g_sResultsPath,pCurrCategory->m_sName,pCurrSequence->m_sName,g_sResultPrefix,k+g_nResultIdxOffset,g_sResultSuffix,oFGMask,g_vnResultsComprParams);
#endif //WRITE_BGSUB_IMG_OUTPUT
#if WRITE_BGSUB_METRICS_ANALYSIS
			CalcMetricsFromResult(oFGMask,pCurrSequence->GetGTFrameFromIndex(k),pCurrSequence->GetSequenceROI(),pCurrSequence->nTP,pCurrSequence->nTN,pCurrSequence->nFP,pCurrSequence->nFN,pCurrSequence->nSE);
		}
		WriteMetrics(g_sResultsPath+pCurrCategory->m_sName+"/"+pCurrSequence->m_sName+".txt",pCurrSequence);
#else //!WRITE_BGSUB_METRICS_ANALYSIS
		}
#endif //!WRITE_BGSUB_METRICS_ANALYSIS
	}
	catch(cv::Exception& e) {std::cout << e.what() << std::endl;}
	catch(std::runtime_error& e) {std::cout << e.what() << std::endl;}
	catch(...) {std::cout << "Caught unknown exception." << std::endl;}
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
#if USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR
cv::Mat GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oBGDesc, const cv::Mat& oFGMask, std::vector<cv::KeyPoint> voKeyPoints, size_t nFrame) {
	// note: this function is definitely NOT efficient in any way; it is only intended for debug purposes.
	cv::Mat oInputImgBYTE3, oBGImgBYTE3, oBGDescBYTE, oBGDescBYTE3, oFGMaskBYTE3;
	cv::Mat oInputDesc, oInputDescBYTE, oInputDescBYTE3;
	cv::Mat oDescDiff, oDescDiffBYTE, oDescDiffBYTE3;
	cv::Mat oImgDiffBYTE3;
	LBSP oExtractor;
	oExtractor.setReference(oBGImg);
	oExtractor.compute2(oInputImg,voKeyPoints,oInputDesc);
	LBSP::calcDescImgDiff(oInputDesc,oBGDesc,oDescDiff);
	oInputDesc.convertTo(oInputDescBYTE,CV_8U,(double)UCHAR_MAX/USHRT_MAX);
	oBGDesc.convertTo(oBGDescBYTE,CV_8U,(double)UCHAR_MAX/USHRT_MAX);
	oDescDiffBYTE = oDescDiff;
	cv::cvtColor(oFGMask,oFGMaskBYTE3,CV_GRAY2RGB);
	if(oInputImg.channels()!=3) {
		cv::cvtColor(oInputImg,oInputImgBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oBGImg,oBGImgBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oInputDescBYTE,oInputDescBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oBGDescBYTE,oBGDescBYTE3,CV_GRAY2RGB);
		cv::cvtColor(oDescDiffBYTE,oDescDiffBYTE3,CV_GRAY2RGB);
	}
	else {
		oInputImgBYTE3 = oInputImg;
		oBGImgBYTE3 = oBGImg;
		oInputDescBYTE3 = oInputDescBYTE;
		oBGDescBYTE3 = oBGDescBYTE;
		oDescDiffBYTE3 = oDescDiffBYTE;
	}

	cv::absdiff(oInputImgBYTE3,oBGImgBYTE3,oImgDiffBYTE3);
	cv::Mat displayH,displayV1,displayV2;
	cv::resize(oInputImgBYTE3,oInputImgBYTE3,cv::Size(320,240));
	cv::resize(oBGImgBYTE3,oBGImgBYTE3,cv::Size(160,120));
	cv::resize(oImgDiffBYTE3,oImgDiffBYTE3,cv::Size(160,120));
	cv::resize(oBGDescBYTE3,oBGDescBYTE3,cv::Size(160,120));
	cv::resize(oDescDiffBYTE3,oDescDiffBYTE3,cv::Size(160,120));
	cv::resize(oFGMaskBYTE3,oFGMaskBYTE3,cv::Size(320,240));

	std::stringstream sstr;
	sstr << "Input Image #" << nFrame;
	WriteOnImage(oInputImgBYTE3,sstr.str());
	WriteOnImage(oBGImgBYTE3,"Reference Image");
	WriteOnImage(oImgDiffBYTE3,"Diff Image");
	WriteOnImage(oBGDescBYTE3,"Reference DescImage");
	WriteOnImage(oDescDiffBYTE3,"Diff DescImage");
	WriteOnImage(oFGMaskBYTE3,"Segmentation Result");

	cv::vconcat(oBGImgBYTE3,oImgDiffBYTE3,displayV1);
	cv::vconcat(oBGDescBYTE3,oDescDiffBYTE3,displayV2);
	cv::hconcat(oInputImgBYTE3,displayV1,displayH);
	cv::hconcat(displayH,displayV2,displayH);
	cv::hconcat(displayH,oFGMaskBYTE3,displayH);
	return displayH;
}
#else //!(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
cv::Mat GetDisplayResult(const cv::Mat& oInputImg, const cv::Mat& oBGImg, const cv::Mat& oFGMask, size_t nFrame) {
	// note: this function is definitely NOT efficient in any way; it is only intended for debug purposes.
	CV_Assert(oInputImg.type()==oBGImg.type() && oBGImg.type()==CV_8UC3);
	CV_Assert(oFGMask.type()==CV_8UC1);
	CV_Assert(oInputImg.size()==oBGImg.size() && oBGImg.size()==oFGMask.size());
	cv::Mat oInputImgBYTE3,oBGImgBYTE3,oFGMaskBYTE3,oImgDiffBYTE3;
	cv::cvtColor(oFGMask,oFGMaskBYTE3,CV_GRAY2RGB);
	oInputImgBYTE3 = oInputImg;
	oBGImgBYTE3 = oBGImg;
	cv::absdiff(oInputImgBYTE3,oBGImgBYTE3,oImgDiffBYTE3);
	cv::Mat displayH,displayV;

	cv::resize(oInputImgBYTE3,oInputImgBYTE3,cv::Size(320,240));
	cv::resize(oBGImgBYTE3,oBGImgBYTE3,cv::Size(160,120));
	cv::resize(oImgDiffBYTE3,oImgDiffBYTE3,cv::Size(160,120));
	cv::resize(oFGMaskBYTE3,oFGMaskBYTE3,cv::Size(320,240));


	std::stringstream sstr;
	sstr << "Input Image #" << nFrame;
	WriteOnImage(oInputImgBYTE3,sstr.str());
	WriteOnImage(oBGImgBYTE3,"Reference Image");
	WriteOnImage(oImgDiffBYTE3,"Diff Image");
	WriteOnImage(oFGMaskBYTE3,"Segmentation Result");

	cv::vconcat(oBGImgBYTE3,oImgDiffBYTE3,displayV);
	cv::hconcat(oInputImgBYTE3,displayV,displayH);
	cv::hconcat(displayH,oFGMaskBYTE3,displayH);
	return displayH;
}
#endif //!(USE_VIBE_LBSP_BG_SUBTRACTOR || USE_PBAS_LBSP_BG_SUBTRACTOR)
#endif //DISPLAY_BGSUB_DEBUG_OUTPUT || WRITE_BGSUB_DEBUG_IMG_OUTPUT
