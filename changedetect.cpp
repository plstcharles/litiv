#include "BackgroundSubtractorLBSP.h"
#include "DatasetUtils.h"

#define WRITE_OUTPUT 0
#define DISPLAY_OUTPUT 1
#define WRITE_DISPLAY_OUTPUT 0

inline void WriteOnImage(cv::Mat& oImg, const std::string& sText, bool bBottom=false) {
	cv::putText(oImg,sText,cv::Point(10,bBottom?(oImg.rows-10):10),cv::FONT_HERSHEY_PLAIN,0.7,cv::Scalar(255,0,0),1,CV_AA);
}

inline void writeResult(	const std::string& sResultsPath,
							const std::string& sCatName,
							const std::string& sSeqName,
							const std::string& sResultPrefix,
							int framenum,
							const std::string& sResultSuffix,
							const cv::Mat& res,
							const std::vector<int>& vnComprParams) {
	char buffer[10];
	sprintf(buffer,"%06d",framenum);
	std::stringstream sResultFilePath;
	sResultFilePath << sResultsPath << sCatName << "/" << sSeqName << "/" << sResultPrefix << buffer << sResultSuffix;
	cv::imwrite(sResultFilePath.str(), res, vnComprParams);
}

inline cv::Mat getDisplayResult(const cv::Mat& oInputImg,
								const cv::Mat& oBGImg,
								const cv::Mat& oBGDesc,
								const cv::Mat& oFGMask,
								std::vector<cv::KeyPoint> voKeyPoints) {
	// note: this function is definitely NOT efficient in any way; it is only intended for debug purposes.
	cv::Mat oInputImgBYTE3, oBGImgBYTE3, oBGDescBYTE, oBGDescBYTE3, oFGMaskBYTE3;
	cv::Mat oInputDesc, oInputDescBYTE, oInputDescBYTE3;
	cv::Mat oDescDiff, oDescDiffBYTE, oDescDiffBYTE3;
	LBSP oExtractor;
	oExtractor.compute2(oInputImg,voKeyPoints,oInputDesc);
	LBSP::calcDescImgDiff(oInputDesc,oBGDesc,oDescDiff);
	oInputDesc.convertTo(oInputDescBYTE,CV_8U);
	oBGDesc.convertTo(oBGDescBYTE,CV_8U);
	oDescDiff.convertTo(oDescDiffBYTE,CV_8U);
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
	cv::Mat display1H,display2H,display3H;
	cv::hconcat(oInputImgBYTE3,oBGImgBYTE3,display1H);
	cv::hconcat(oInputDescBYTE3,oBGDescBYTE3,display2H);
	cv::hconcat(oFGMaskBYTE3,oDescDiffBYTE3,display3H);
	cv::Mat display;
	cv::vconcat(display1H,display2H,display);
	cv::vconcat(display,display3H,display);
	return display;
}

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
		//voCategories.push_back(CategoryInfo("cameraJitter", sDatasetPath+"cameraJitter"));
		//voCategories.push_back(CategoryInfo("dynamicBackground", sDatasetPath+"dynamicBackground"));
		//voCategories.push_back(CategoryInfo("intermittentObjectMotion", sDatasetPath+"intermittentObjectMotion"));
		//voCategories.push_back(CategoryInfo("shadow", sDatasetPath+"shadow"));
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
					//cv::GaussianBlur(oInputImg, oInputImg, cv::Size2i(5,5), 3, 3);
					oBGSubtr(oInputImg, oFGMask, k<=50?1:BGSLBSP_DEFAULT_LEARNING_RATE);
#if DISPLAY_OUTPUT
					cv::Mat display = getDisplayResult(oInputImg,oLastBGImg,oLastBGDesc,oFGMask,oBGSubtr.getBGKeyPoints());
					cv::imshow("display", display);
#if WRITE_DISPLAY_OUTPUT
					oWriter.write(display);
#endif //WRITE_DISPLAY_OUTPUT
					cv::waitKey(1);
#endif //DISPLAY_OUTPUT
#if WRITE_OUTPUT
					writeResult(sResultsPath,pCurrCategory->sName,pCurrSequence->sName,sResultPrefix,k+1,sResultSuffix,oFGMask,vnCompressionParams);
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
