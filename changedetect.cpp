#include <opencv2/opencv.hpp>
#include "LBSP.h"
#include "DetectChange.h"
#include <stdio.h>
#include "DatasetUtils.h"

#define WRITE_OUTPUT
#define DISPLAY_OUTPUT
#define WRITE_DISPLAY_OUTPUT

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

inline cv::Mat getDisplayResult(const cv::Mat& bgDescImg,
								const cv::Mat& currDescImg,
								const cv::Mat& currImg,
								const cv::Mat& currResult) {
	cv::Mat currDescDiff, currDescDiffBYTE, currDescImgBYTE, currResultBYTE;
	LBSP::calcDescImgDiff(bgDescImg,currDescImg,currDescDiff);
	currDescDiff.convertTo(currDescDiffBYTE,CV_8UC3);
	currDescImg.convertTo(currDescImgBYTE,CV_8UC3);
	cv::cvtColor(currResult,currResultBYTE,CV_GRAY2BGR);
	cv::Mat display1H,display2H;
	cv::hconcat(currImg,currDescImgBYTE,display1H);
	cv::hconcat(currResultBYTE,currDescDiffBYTE,display2H);
	cv::Mat display;
	cv::vconcat(display1H,display2H,display);
	return display;
}

int main( int argc, char** argv ) {
	
	bool training=true;
	std::vector<CategoryInfo> voCategories;
#ifndef WIN32
	std::string sDatasetPath = "/shared/datasets/CDNet/dataset/";
	std::string sResultsPath = "/shared/datasets/CDNet/results/";
#else
	std::string sDatasetPath = "E:/datasets/CDNet/dataset/";
	std::string sResultsPath = "E:/datasets/CDNet/results/";
#endif
	std::string sInputPrefix = "input/";
	std::string sInputSuffix = ".jpg";
	std::string sGroundtruthPrefix = "groundtruth/gt";
	std::string sGroundtruthSuffix = ".png";
	std::string sResultPrefix = "bin";
	std::string sResultSuffix = ".png";
	std::cout << "Parsing dataset..." << std::endl;
	try {
		voCategories.push_back(CategoryInfo("baseline", sDatasetPath+"baseline"));
		//voCategories.push_back(CategoryInfo("cameraJitter", sDatasetPath+"cameraJitter"));
		//voCategories.push_back(CategoryInfo("dynamicBackground", sDatasetPath+"dynamicBackground"));
		//voCategories.push_back(CategoryInfo("intermittentObjectMotion", sDatasetPath+"intermittentObjectMotion"));
		//voCategories.push_back(CategoryInfo("shadow", sDatasetPath+"shadow"));
		voCategories.push_back(CategoryInfo("thermal", sDatasetPath+"thermal"));
	} catch(std::runtime_error& e) { std::cout << e.what() << std::endl; }
	std::cout << "Parsing complete. [" << voCategories.size() << " categories]" << std::endl << std::endl;
	for(size_t i=0; i<voCategories.size(); i++) {
		CategoryInfo& oCurrCategory = voCategories[i];
		std::cout << "Processing category " << i+1 << "/" << voCategories.size() << "... (" << oCurrCategory.sName << ")" << std::endl;
		for(size_t j=0; j<oCurrCategory.vpSequences.size(); j++) {
			try {
				SequenceInfo* pCurrSequence = oCurrCategory.vpSequences[j];
				std::cout << "\tProcessing sequence " << j+1 << "/" << oCurrCategory.vpSequences.size() << "... (" << pCurrSequence->sName << ")" << std::endl;
				assert(pCurrSequence->vsInputFramePaths.size()>1);
				std::vector<cv::KeyPoint> keypointsA;
				cv::Mat descriptorsA;
				cv::Mat descimage, descimage2, descimage3, imgprev, imgres;
				std::vector<int> compression_params;
				compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
				compression_params.push_back(9);
				cv::DenseFeatureDetector detector(	1.0f,	// init feature scale
													1,		// feature scale levels
													1.0f,	// feature scale mult
													1,		// init xy step
													0,		// init img bound
													true,	// var xy step with scale
													false	// var img bound with scale
													);		// note: the extractor will remove keypoints that are out of bounds itself
				LBSP extractor(30);
				DetectChange DC(28);
				cv::Mat imgA = cv::imread(pCurrSequence->vsInputFramePaths[0], cv::IMREAD_COLOR);
#ifdef WRITE_DISPLAY_OUTPUT
				cv::VideoWriter oWriter(sResultsPath+"/"+oCurrCategory.sName+"/"+pCurrSequence->sName+".avi",CV_FOURCC('X','V','I','D'),30,imgA.size()*2,true);
#endif
				cv::GaussianBlur(imgA, imgA, cv::Size2i(5,5), 3,3);
				if(keypointsA.capacity()<(size_t)(imgA.cols*imgA.rows))
					keypointsA.reserve(imgA.cols*imgA.rows);
				detector.detect(imgA, keypointsA);
				extractor.setReference(cv::Mat());
				extractor.compute(imgA, keypointsA, descriptorsA);
				LBSP::recreateDescImage(imgA.channels(),imgA.rows,imgA.cols,keypointsA,descriptorsA,descimage);
				DC.setBGModel(descimage, imgA);
				DC.compute(descimage, imgres);
#ifdef DISPLAY_OUTPUT
				cv::Mat display = getDisplayResult(DC.getBGDesc(),descimage,imgA,imgres);
				cv::imshow("display", display);
#ifdef WRITE_DISPLAY_OUTPUT
				oWriter.write(display);
#endif
				cv::waitKey(1);
#endif
#ifdef WRITE_OUTPUT
				writeResult(sResultsPath,oCurrCategory.sName,pCurrSequence->sName,sResultPrefix,1,sResultSuffix,imgres,compression_params);
#endif
				for(size_t k=1; k<pCurrSequence->vsInputFramePaths.size(); k++) {
					//if(!(k%100))
						std::cout << "\t\t[F:" << k << "/" << pCurrSequence->vsInputFramePaths.size() << "]" << std::endl;
					imgA = cv::imread(pCurrSequence->vsInputFramePaths[k], cv::IMREAD_COLOR);
					cv::GaussianBlur(imgA, imgA, cv::Size2i(5,5), 3,3);
					if(k==500) training=false;
					LBSP::computeImpl(imgA,DC.getBGImage(),keypointsA,descriptorsA,extractor.getAbsThreshold());
					LBSP::recreateDescImage(imgA.channels(),imgA.rows,imgA.cols,keypointsA,descriptorsA,descimage);
					if(training) {
						LBSP::computeImpl(imgA,DC.getBGImage2(),keypointsA,descriptorsA,extractor.getAbsThreshold());
						LBSP::recreateDescImage(imgA.channels(),imgA.rows,imgA.cols,keypointsA,descriptorsA,descimage2);
						LBSP::computeImpl(imgA,cv::Mat(),keypointsA,descriptorsA,extractor.getAbsThreshold());
						LBSP::recreateDescImage(imgA.channels(),imgA.rows,imgA.cols,keypointsA,descriptorsA,descimage3);
						training = DC.trainandcompute(descimage, descimage2, descimage3, imgA, imgres);
						if(!training)
							std::cout << "all code stable!" << std::endl;
					}
					else
						DC.compute(descimage, imgres);
#ifdef DISPLAY_OUTPUT
					cv::Mat display = getDisplayResult(DC.getBGDesc(),descimage,imgA,imgres);
					cv::imshow("display", display);
#ifdef WRITE_DISPLAY_OUTPUT
					oWriter.write(display);
#endif
					cv::waitKey(1);
#endif
#ifdef WRITE_OUTPUT
					writeResult(sResultsPath,oCurrCategory.sName,pCurrSequence->sName,sResultPrefix,k+1,sResultSuffix,imgres,compression_params);
#endif
				}
			}
			catch(cv::Exception& e) {std::cout << e.what() << std::endl;}
			catch(std::runtime_error& e) {std::cout << e.what() << std::endl;}
			catch(...) {std::cout << "Caught unknown exception." << std::endl;}
		}
	}
}
