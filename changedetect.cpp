#include <opencv2/opencv.hpp>
#include "LBSP.h"
#include "BackgroundSubtractorLBSP.h"
#include <stdio.h>
#include "DatasetUtils.h"

#define WRITE_OUTPUT 1
#define DISPLAY_OUTPUT 1
#define WRITE_DISPLAY_OUTPUT 1

const double dLearningRate = -1;
const int nFGThreshold = 9;
const int nFGSCThreshold = 11;
const int nDescThreshold = 30;

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
	LBSP oExtractor(nDescThreshold);
	oExtractor.setReference(oBGImg);
	cv::Mat oInputDesc, oInputDescImg, oInputDescImgBYTE, oBGDescImg, oBGDescImgBYTE;
	cv::Mat oDescDiffImg, oDescImgDiffBYTE, oFGMaskBYTE;
	oExtractor.compute(oInputImg,voKeyPoints,oInputDesc);
	LBSP::recreateDescImage(oInputImg.size(),voKeyPoints,oInputDesc,oInputDescImg);
	LBSP::recreateDescImage(oInputImg.size(),voKeyPoints,oBGDesc,oBGDescImg);
	LBSP::calcDescImgDiff(oInputDescImg,oBGDescImg,oDescDiffImg);
	oInputDescImg.convertTo(oInputDescImgBYTE,CV_8UC3);
	oBGDescImg.convertTo(oBGDescImgBYTE,CV_8UC3);
	oDescDiffImg.convertTo(oDescImgDiffBYTE,CV_8UC3);
	if(oInputImg.channels()==3)
		cv::cvtColor(oFGMask,oFGMaskBYTE,CV_GRAY2BGR);
	else
		oFGMaskBYTE = oFGMask;
	cv::Mat display1H,display2H,display3H;
	cv::hconcat(oInputImg,oBGImg,display1H);
	cv::hconcat(oInputDescImgBYTE,oBGDescImgBYTE,display2H);
	cv::hconcat(oFGMaskBYTE,oDescImgDiffBYTE,display3H);
	cv::Mat display;
	cv::vconcat(display1H,display2H,display);
	cv::vconcat(display,display3H,display);
	return display;
}

int main( int argc, char** argv ) {
#ifndef WIN32
	std::string sDatasetPath = "/shared/datasets/CDNet/dataset/";
	std::string sResultsPath = "/shared/datasets/CDNet/results/";
#else //WIN32
	std::string sDatasetPath = "E:/datasets/CDNet/dataset/";
	std::string sResultsPath = "E:/datasets/CDNet/results/";
#endif //WIN32
	std::string sInputPrefix = "input/";
	std::string sInputSuffix = ".jpg";
	std::string sGroundtruthPrefix = "groundtruth/gt";
	std::string sGroundtruthSuffix = ".png";
	std::string sResultPrefix = "bin";
	std::string sResultSuffix = ".png";
	std::vector<CategoryInfo> voCategories;
	std::cout << "Parsing dataset..." << std::endl;
	try {
		//voCategories.push_back(CategoryInfo("baseline", sDatasetPath+"baseline"));
		//voCategories.push_back(CategoryInfo("cameraJitter", sDatasetPath+"cameraJitter"));
		//voCategories.push_back(CategoryInfo("dynamicBackground", sDatasetPath+"dynamicBackground"));
		//voCategories.push_back(CategoryInfo("intermittentObjectMotion", sDatasetPath+"intermittentObjectMotion"));
		//voCategories.push_back(CategoryInfo("shadow", sDatasetPath+"shadow"));
		voCategories.push_back(CategoryInfo("thermal", sDatasetPath+"thermal"));
	} catch(std::runtime_error& e) { std::cout << e.what() << std::endl; }
	std::cout << "Parsing complete. [" << voCategories.size() << " categories]" << std::endl << std::endl;
	std::vector<int> vnCompressionParams;
	vnCompressionParams.push_back(CV_IMWRITE_PNG_COMPRESSION);
	vnCompressionParams.push_back(9);
	for(size_t i=0; i<voCategories.size(); i++) {
		CategoryInfo& oCurrCategory = voCategories[i];
		std::cout << "Processing category " << i+1 << "/" << voCategories.size() << "... (" << oCurrCategory.sName << ")" << std::endl;
		for(size_t j=0; j<oCurrCategory.vpSequences.size(); j++) {
			try {
				SequenceInfo* pCurrSequence = oCurrCategory.vpSequences[j];
				std::cout << "\tProcessing sequence " << j+1 << "/" << oCurrCategory.vpSequences.size() << "... (" << pCurrSequence->sName << ")" << std::endl;
				assert(pCurrSequence->vsInputFramePaths.size()>1);
				cv::Mat oFGMask, oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[0], (oCurrCategory.sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
				BackgroundSubtractorLBSP oBGSubtr(nDescThreshold,nFGThreshold,nFGSCThreshold);
				oBGSubtr.initialize(oInputImg.size(),oInputImg.type());
#if DISPLAY_OUTPUT && WRITE_DISPLAY_OUTPUT
				cv::VideoWriter oWriter(sResultsPath+"/"+oCurrCategory.sName+"/"+pCurrSequence->sName+".avi",CV_FOURCC('X','V','I','D'),30,oInputImg.size()*2,true);
#endif //DISPLAY_OUTPUT && WRITE_DISPLAY_OUTPUT
				for(size_t k=0; k<pCurrSequence->vsInputFramePaths.size(); k++) {
					std::cout << "\t\t[F:" << k << "/" << pCurrSequence->vsInputFramePaths.size() << "]" << std::endl;
					oInputImg = cv::imread(pCurrSequence->vsInputFramePaths[k], (oCurrCategory.sName=="thermal")?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
#if DISPLAY_OUTPUT
					cv::Mat oLastBGImg = oBGSubtr.getCurrentBGImage();
					cv::Mat oLastBGDesc = oBGSubtr.getCurrentBGDescriptors();
#endif //DISPLAY_OUTPUT
					cv::GaussianBlur(oInputImg, oInputImg, cv::Size2i(5,5), 3, 3);
					oBGSubtr(oInputImg, oFGMask, dLearningRate);
#if DISPLAY_OUTPUT
					cv::Mat display = getDisplayResult(oInputImg,oLastBGImg,oLastBGDesc,oFGMask,oBGSubtr.getBGKeyPoints());
					cv::imshow("display", display);
#if WRITE_DISPLAY_OUTPUT
					oWriter.write(display);
#endif //WRITE_DISPLAY_OUTPUT
					cv::waitKey(1);
#endif //DISPLAY_OUTPUT
#if WRITE_OUTPUT
					writeResult(sResultsPath,oCurrCategory.sName,pCurrSequence->sName,sResultPrefix,k+1,sResultSuffix,oFGMask,vnCompressionParams);
#endif //WRITE_OUTPUT
				}
			}
			catch(cv::Exception& e) {std::cout << e.what() << std::endl;}
			catch(std::runtime_error& e) {std::cout << e.what() << std::endl;}
			catch(...) {std::cout << "Caught unknown exception." << std::endl;}
		}
	}
}
