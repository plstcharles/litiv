#include "BackgroundSubtractorLBSP.h"
#include "DatasetUtils.h"

#define WRITE_OUTPUT 0
#define DISPLAY_OUTPUT 1
#define WRITE_DISPLAY_OUTPUT 0

int main( int argc, char** argv ) {
	srand(0);
	setvbuf(stdout, NULL, _IONBF, 0);
	setvbuf(stderr, NULL, _IONBF, 0);
#ifndef WIN32
	std::string sDatasetPath = "/shared/datasets/CDNet/dataset/";
	std::string sResultsPath = "/shared/datasets/CDNet/results/";
#else //WIN32
	std::string sDatasetPath = "C:/datasets/CDNet/dataset/";
	std::string sResultsPath = "C:/datasets/CDNet/results/";
#endif //WIN32
	std::string sInputPrefix = "input/";
	std::string sInputSuffix = ".jpg";
	std::string sGroundtruthPrefix = "groundtruth/gt";
	std::string sGroundtruthSuffix = ".png";
	std::string sResultPrefix = "bin";
	std::string sResultSuffix = ".png";
	std::vector<CategoryInfo*> vpCategories;
	std::cout << "Parsing dataset..." << std::endl;
	try {
		vpCategories.push_back(new CategoryInfo("baseline", sDatasetPath+"baseline"));
		vpCategories.push_back(new CategoryInfo("cameraJitter", sDatasetPath+"cameraJitter"));
		vpCategories.push_back(new CategoryInfo("dynamicBackground", sDatasetPath+"dynamicBackground"));
		vpCategories.push_back(new CategoryInfo("intermittentObjectMotion", sDatasetPath+"intermittentObjectMotion"));
		vpCategories.push_back(new CategoryInfo("shadow", sDatasetPath+"shadow"));
		vpCategories.push_back(new CategoryInfo("thermal", sDatasetPath+"thermal"));
	} catch(std::runtime_error& e) { std::cout << e.what() << std::endl; }
	std::cout << "Parsing complete. [" << vpCategories.size() << " categories]" << std::endl << std::endl;
	std::vector<int> vnCompressionParams;
	vnCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	vnCompressionParams.push_back(9);
	for(size_t i=0; i<vpCategories.size(); i++) {
		CategoryInfo* pCurrCategory = vpCategories[i];
		std::cout << "Processing category " << i+1 << "/" << vpCategories.size() << "... (" << pCurrCategory->sName << ")" << std::endl;
		for(size_t j=0; j<pCurrCategory->vpSequences.size(); j++) {
			try {
				SequenceInfo* pCurrSequence = pCurrCategory->vpSequences[j];
				std::cout << "\tProcessing sequence " << j+1 << "/" << pCurrCategory->vpSequences.size() << "... (" << pCurrSequence->sName << ")" << std::endl;
				assert(pCurrSequence->vsInputFramePaths.size()>1);
				cv::Mat oFGMask, oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[0], (pCurrCategory->sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
				BackgroundSubtractorLBSP oBGSubtr;
				oBGSubtr.initialize(oInputImg);
#if DISPLAY_OUTPUT && WRITE_DISPLAY_OUTPUT
				cv::Size oWriterInputSize = oInputImg.size();
				oWriterInputSize.height*=3;
				oWriterInputSize.width*=2;
				cv::VideoWriter oWriter(sResultsPath+"/"+pCurrCategory->sName+"/"+pCurrSequence->sName+".avi",CV_FOURCC('X','V','I','D'),30,oWriterInputSize,true);
#endif //DISPLAY_OUTPUT && WRITE_DISPLAY_OUTPUT
				for(size_t k=0; k<pCurrSequence->vsInputFramePaths.size(); k++) {
					std::cout << "\t\t[F:" << k << "/" << pCurrSequence->vsInputFramePaths.size() << "]" << std::endl;
					oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[k], (pCurrCategory->sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
#if DISPLAY_OUTPUT
					cv::Mat oLastBGImg = oBGSubtr.getCurrentBGImage();
					cv::Mat oLastBGDesc = oBGSubtr.getCurrentBGDescriptors();
#endif //DISPLAY_OUTPUT
					oBGSubtr(oInputImg, oFGMask, k<=100?1:BGSLBSP_DEFAULT_LEARNING_RATE);
#if DISPLAY_OUTPUT
					cv::Mat display = GetDisplayResult(oInputImg,oLastBGImg,oLastBGDesc,oFGMask,oBGSubtr.getBGKeyPoints(),k);
					cv::imshow("display", display);
#if WRITE_DISPLAY_OUTPUT
					oWriter.write(display);
#endif //WRITE_DISPLAY_OUTPUT
					cv::waitKey(1);
#endif //DISPLAY_OUTPUT
#if WRITE_OUTPUT
					WriteResult(sResultsPath,pCurrCategory->sName,pCurrSequence->sName,sResultPrefix,k+1,sResultSuffix,oFGMask,vnCompressionParams);
#endif //WRITE_OUTPUT
				}
			}
			catch(cv::Exception& e) {std::cout << e.what() << std::endl;}
			catch(std::runtime_error& e) {std::cout << e.what() << std::endl;}
			catch(...) {std::cout << "Caught unknown exception." << std::endl;}
		}
	}
	for(size_t i=0; i<vpCategories.size(); i++) {
		delete vpCategories[i];
	}
	vpCategories.clear();
}
