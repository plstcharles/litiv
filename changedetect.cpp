#include "BackgroundSubtractorLBSP.h"
#include "DatasetUtils.h"

int AnalyzeSequence(int nThreadIdx, const CategoryInfo* pCurrCategory, const SequenceInfo* pCurrSequence);

////////////////////////////////////////
// USER/ENVIRONMENT-SPECIFIC VARIABLES :
#define WRITE_ANALYSIS_RESULTS 0
#define DISPLAY_ANALYSIS_DEBUG_RESULTS 0
#define WRITE_ANALYSIS_DEBUG_RESULTS 0
#define USE_RELATIVE_LBSP_COMPARISONS 1
const std::string g_sResultPrefix("bin"); // based on the CDNet result image template
const std::string g_sResultSuffix(".png"); // based on the CDNet result image template
#if WIN32 && _MSC_VER <= 1600
#define USE_WINDOWS_API
#include <windows.h>
#include <process.h>
const std::string g_sDatasetPath("C:/datasets/CDNet/dataset/");
const std::string g_sResultsPath("C:/datasets/CDNet/results_test/");
const int g_anResultsComprParams[2] = {CV_IMWRITE_PNG_COMPRESSION,9}; // lower to increase processing speed
const std::vector<int> g_vnResultsComprParams(g_anResultsComprParams,g_anResultsComprParams+2);
const size_t g_nMaxThreads = 4;
HANDLE g_hThreadEvent[g_nMaxThreads] = {0};
HANDLE g_hThreads[g_nMaxThreads] = {0};
void* g_apThreadDataStruct[g_nMaxThreads][2] = {0};
DWORD WINAPI AnalyzeSequenceEntryPoint(LPVOID lpParam) {
	return AnalyzeSequence((int)(lpParam),(CategoryInfo*)g_apThreadDataStruct[(int)(lpParam)][0],(SequenceInfo*)g_apThreadDataStruct[(int)(lpParam)][1]);
}
#else //!WIN32 || _MSC_VER > 1600
#include <thread>
#include <chrono>
#include <atomic>
const std::string g_sDatasetPath("/shared/datasets/CDNet/dataset/");
const std::string g_sResultsPath("/shared/datasets/CDNet/results_test/");
const std::vector<int> g_vnResultsComprParams = {CV_IMWRITE_PNG_COMPRESSION,9}; // lower to increase processing speed
const size_t g_nMaxThreads = (std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():1);
std::atomic_size_t g_nActiveThreads(0);
#endif //!WIN32 || _MSC_VER > 1600

///////////////////////////////////
int main( int argc, char** argv ) {
	srand(0); // for now, assures that two consecutive runs on the same data return the same results
	setvbuf(stdout, NULL, _IONBF, 0); // fixes output flush problems when using the eclipse built-in console
	std::vector<CategoryInfo*> vpCategories;
	std::cout << "Parsing dataset..." << std::endl;
	try {
		vpCategories.push_back(new CategoryInfo("baseline", g_sDatasetPath+"baseline"));
		vpCategories.push_back(new CategoryInfo("cameraJitter", g_sDatasetPath+"cameraJitter"));
		vpCategories.push_back(new CategoryInfo("dynamicBackground", g_sDatasetPath+"dynamicBackground"));
		vpCategories.push_back(new CategoryInfo("intermittentObjectMotion", g_sDatasetPath+"intermittentObjectMotion"));
		vpCategories.push_back(new CategoryInfo("shadow", g_sDatasetPath+"shadow"));
		vpCategories.push_back(new CategoryInfo("thermal", g_sDatasetPath+"thermal"));
	} catch(std::runtime_error& e) { std::cout << e.what() << std::endl; }
	size_t nSeqTotal = 0;
	for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory)
		nSeqTotal += (*pCurrCategory)->vpSequences.size();
	std::cout << "Parsing complete. [" << vpCategories.size() << " categories, "  << nSeqTotal  << " sequences]" << std::endl << std::endl;
	// since the algorithm isn't implemented to be parallelized yet, we parallelize the sequence treatment instead
	std::cout << "Running LBSP background subtraction with " << g_nMaxThreads << " threads..." << std::endl;
	size_t nSeqProcessed = 1;
#ifndef USE_WINDOWS_API
	for(auto& pCurrCategory : vpCategories) {
		for(auto& pCurrSequence : pCurrCategory->vpSequences) {
			while(g_nActiveThreads>=g_nMaxThreads)
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			std::cout << "\tProcessing sequence " << nSeqProcessed << "/" << nSeqTotal << "... (" << pCurrCategory->sName << ":" << pCurrSequence->sName << ")" << std::endl;
			g_nActiveThreads++;
			nSeqProcessed++;
			std::thread(AnalyzeSequence,-1,pCurrCategory,pCurrSequence).detach();
		}
	}
	while(g_nActiveThreads>0)
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
#else //USE_WINDOWS_API
	for(size_t n=0; n<g_nMaxThreads; ++n)
		g_hThreadEvent[n] = CreateEvent(NULL,FALSE,TRUE,NULL);
	for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory) {
		for(auto pCurrSequence=(*pCurrCategory)->vpSequences.begin(); pCurrSequence!=(*pCurrCategory)->vpSequences.end(); ++pCurrSequence) {
			DWORD ret = WaitForMultipleObjects(g_nMaxThreads,g_hThreadEvent,FALSE,INFINITE);
			std::cout << "\tProcessing sequence " << nSeqProcessed << "/" << nSeqTotal << "... (" << (*pCurrCategory)->sName << ":" << (*pCurrSequence)->sName << ")" << std::endl;
			nSeqProcessed++;
			g_apThreadDataStruct[ret][0] = (*pCurrCategory);
			g_apThreadDataStruct[ret][1] = (*pCurrSequence);
			g_hThreads[ret] = CreateThread(NULL,NULL,AnalyzeSequenceEntryPoint,(LPVOID)ret,0,NULL);
		}
	}
	WaitForMultipleObjects(g_nMaxThreads,g_hThreads,TRUE,INFINITE);
	for(size_t n=0; n<g_nMaxThreads; ++n) {
		CloseHandle(g_hThreadEvent[n]);
		CloseHandle(g_hThreads[n]);
	}
#endif //USE_WINDOWS_API
	// let memory 'leak' here, exits faster once job is done...
	//for(auto pCurrCategory=vpCategories.begin(); pCurrCategory!=vpCategories.end(); ++pCurrCategory)
	//	delete *pCurrCategory;
	//vpCategories.clear();
}

int AnalyzeSequence(int nThreadIdx, const CategoryInfo* pCurrCategory, const SequenceInfo* pCurrSequence) {
	try {
		CV_DbgAssert(pCurrCategory && pCurrSequence);
		CV_DbgAssert(pCurrSequence->vsInputFramePaths.size()>1);
		const int nInputFlags = (pCurrCategory->sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR; // force thermal sequences to be loaded as grayscale (faster processing, better noise compensation)
		cv::Mat oFGMask, oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[0],nInputFlags);
#if USE_RELATIVE_LBSP_COMPARISONS
		BackgroundSubtractorLBSP oBGSubtr(LBSP_DEFAULT_REL_SIMILARITY_THRESHOLD);
#else
		BackgroundSubtractorLBSP oBGSubtr;
#endif
		oBGSubtr.initialize(oInputImg);
#if DISPLAY_ANALYSIS_DEBUG_RESULTS
		std::string sDebugDisplayName = pCurrCategory->sName + std::string(" -- ") + pCurrSequence->sName;
#if WRITE_ANALYSIS_DEBUG_RESULTS
		cv::Size oWriterInputSize = oInputImg.size();
		oWriterInputSize.height*=3;
		oWriterInputSize.width*=2;
		cv::VideoWriter oWriter(g_sResultsPath+"/"+pCurrCategory->sName+"/"+pCurrSequence->sName+".avi",CV_FOURCC('X','V','I','D'),30,oWriterInputSize,true);
#endif //WRITE_ANALYSIS_DEBUG_RESULTS
#endif //DISPLAY_ANALYSIS_DEBUG_RESULTS
		for(size_t k=0; k<pCurrSequence->vsInputFramePaths.size(); k++) {
			if(!(k%100))
				std::cout << "\t\t" << std::setw(12) << pCurrSequence->sName << " @ F:" << k << "/" << pCurrSequence->vsInputFramePaths.size() << std::endl;
			oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[k],nInputFlags);
#if DISPLAY_ANALYSIS_DEBUG_RESULTS
			cv::Mat oLastBGImg = oBGSubtr.getCurrentBGImage();
			cv::Mat oLastBGDesc = oBGSubtr.getCurrentBGDescriptors();
#endif //DISPLAY_ANALYSIS_DEBUG_RESULTS
			oBGSubtr(oInputImg, oFGMask, k<=100?1:BGSLBSP_DEFAULT_LEARNING_RATE);
#if DISPLAY_ANALYSIS_DEBUG_RESULTS
			cv::Mat oDebugDisplayFrame = GetDisplayResult(oInputImg,oLastBGImg,oLastBGDesc,oFGMask,oBGSubtr.getBGKeyPoints(),k);
			cv::imshow(sDebugDisplayName, oDebugDisplayFrame);
#if WRITE_ANALYSIS_DEBUG_RESULTS
			oWriter.write(oDebugDisplayFrame);
#endif //WRITE_ANALYSIS_DEBUG_RESULTS
			cv::waitKey(1);
#endif //DISPLAY_ANALYSIS_DEBUG_RESULTS
#if WRITE_ANALYSIS_RESULTS
			WriteResult(g_sResultsPath,pCurrCategory->sName,pCurrSequence->sName,g_sResultPrefix,k+1,g_sResultSuffix,oFGMask,g_vnResultsComprParams);
#endif //WRITE_ANALYSIS_RESULTS
		}
	}
	catch(cv::Exception& e) {std::cout << e.what() << std::endl;}
	catch(std::runtime_error& e) {std::cout << e.what() << std::endl;}
	catch(...) {std::cout << "Caught unknown exception." << std::endl;}
#ifndef USE_WINDOWS_API
	g_nActiveThreads--;
#else //USE_WINDOWS_API
	SetEvent(g_hThreadEvent[nThreadIdx]);
#endif //USE_WINDOWS_API
	return 0;
}
