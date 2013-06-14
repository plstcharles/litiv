#include "BackgroundSubtractorLBSP.h"
#include "DatasetUtils.h"
#include <thread>
#include <chrono>
#include <atomic>

// USER/ENVIRONMENT-SPECIFIC VARIABLES
#define g_bWriteResults 0
#define g_bDisplayDebugResults 1
#define g_bWriteDebugResults 0
#ifndef WIN32
const std::string g_sDatasetPath("/shared/datasets/CDNet/dataset/");
const std::string g_sResultsPath("/shared/datasets/CDNet/results/");
#else //WIN32
const std::string g_sDatasetPath("C:/datasets/CDNet/dataset/");
const std::string g_sResultsPath("C:/datasets/CDNet/results/");
#endif //WIN32
const std::string g_sResultPrefix("bin"); // based on the CDNet result image template
const std::string g_sResultSuffix(".png"); // based on the CDNet result image template
const std::vector<int> g_vnResultsComprParams = {CV_IMWRITE_PNG_COMPRESSION,9}; // lower to increase processing speed
const unsigned int g_nMaxThreads = (std::thread::hardware_concurrency()>0?std::thread::hardware_concurrency():2);
std::atomic_size_t g_nActiveThreads(0);

int AnalyzeSequence(const CategoryInfo* pCurrCategory, const SequenceInfo* pCurrSequence);

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
	for(auto& pCurrCategory : vpCategories)
		nSeqTotal += pCurrCategory->vpSequences.size();
	std::cout << "Parsing complete. [" << vpCategories.size() << " categories, << "  << nSeqTotal  << " sequences]" << std::endl << std::endl;
	// since the algorithm isn't implemented to be parallelized yet, we parallelize the sequence treatment instead
	std::cout << "Running LBSP background subtraction with " << g_nMaxThreads << " threads..." << std::endl;
	size_t nSeqProcessed = 1;
	for(auto pCurrCategory : vpCategories) {
		for(auto pCurrSequence : pCurrCategory->vpSequences) {
			while(g_nActiveThreads>=g_nMaxThreads)
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			std::cout << "\tProcessing sequence " << nSeqProcessed << "/" << nSeqTotal << "... (" << pCurrCategory->sName << ":" << pCurrSequence->sName << ")" << std::endl;
			g_nActiveThreads++;
			nSeqProcessed++;
			std::thread(AnalyzeSequence,pCurrCategory,pCurrSequence).detach();
		}
	}
	while(g_nActiveThreads>0)
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	for(auto pCurrCategory : vpCategories)
		delete pCurrCategory;
	vpCategories.clear();
}

int AnalyzeSequence(const CategoryInfo* pCurrCategory, const SequenceInfo* pCurrSequence) {
	try {
		CV_DbgAssert(pCurrCategory && pCurrSequence);
		CV_DbgAssert(pCurrSequence->vsInputFramePaths.size()>1);
		const int nInputFlags = (pCurrCategory->sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR; // force thermal sequences to be loaded as grayscale (faster processing, better noise compensation)
		cv::Mat oFGMask, oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[0],nInputFlags);
		BackgroundSubtractorLBSP oBGSubtr;
		oBGSubtr.initialize(oInputImg);
		#if g_bDisplayDebugResults
			const std::string sDebugDisplayName = pCurrCategory->sName + std::string(" -- ") + pCurrSequence->sName;
			#if g_bWriteDebugResults
				cv::Size oWriterInputSize = oInputImg.size();
				oWriterInputSize.height*=3;
				oWriterInputSize.width*=2;
				cv::VideoWriter oWriter(g_sResultsPath+"/"+pCurrCategory->sName+"/"+pCurrSequence->sName+".avi",CV_FOURCC('X','V','I','D'),30,oWriterInputSize,true);
			#endif
		#endif
		for(size_t k=0; k<pCurrSequence->vsInputFramePaths.size(); k++) {
			if(!(k%100))
				std::cout << "\t\t[" << pCurrSequence->sName << "\t F:" << k << "/" << pCurrSequence->vsInputFramePaths.size() << "]" << std::endl;
			oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[k],nInputFlags);
			#if g_bDisplayDebugResults
				cv::Mat oLastBGImg = oBGSubtr.getCurrentBGImage();
				cv::Mat oLastBGDesc = oBGSubtr.getCurrentBGDescriptors();
			#endif
			oBGSubtr(oInputImg, oFGMask, k<=100?1:BGSLBSP_DEFAULT_LEARNING_RATE);
			#if g_bDisplayDebugResults
				cv::Mat oDebugDisplayFrame = GetDisplayResult(oInputImg,oLastBGImg,oLastBGDesc,oFGMask,oBGSubtr.getBGKeyPoints(),k);
				cv::imshow(sDebugDisplayName, oDebugDisplayFrame);
				#if g_bWriteDebugResults
					oWriter.write(oDebugDisplayFrame);
				#endif
				cv::waitKey(1);
			#endif
			#if g_bWriteResults
				WriteResult(g_sResultsPath,pCurrCategory->sName,pCurrSequence->sName,g_sResultPrefix,k+1,g_sResultSuffix,oFGMask,g_vnResultsComprParams);
			#endif
		}
	}
	catch(cv::Exception& e) {std::cout << e.what() << std::endl;}
	catch(std::runtime_error& e) {std::cout << e.what() << std::endl;}
	catch(...) {std::cout << "Caught unknown exception." << std::endl;}
	g_nActiveThreads--;
	return 0;
}
