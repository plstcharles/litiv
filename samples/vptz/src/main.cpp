
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2017 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
/////////////////////////////////////////////////////////////////////////////
//
// This sample demonstrates how to use the Virtual PTZ module to evaluate a
// tracking algorithm on all test videos. No actual algorithm implementation
// is used here, so a static bounding box is used instead. This is the code
// you should start from if you wish to evaluate your own method.
//
// Note that the input scenrio/gt paths should still be redefined below.
//
/////////////////////////////////////////////////////////////////////////////

#include "litiv/vptz/virtualptz.hpp"
#include "litiv/utils/platform.hpp"

////////////////////////////////////////
#define VPTZ_USE_WAITSLEEP         0       // = 'simulate' full delays by 'sleeping' between frames
#define VPTZ_EXEC_DELAY_RATIO      1.0     // = normal processing delay penalty (100% time lost considered)
#define VPTZ_COMMUNICATION_DELAY   0.125   // = 125ms network ping delay between server and client
////////////////////////////////////////////////////////////////////////////////////////////////////////
#define VPTZ_DATASET_ROOT_DIR_PATH       std::string("/some/root/directory/litiv_vptz_icip2015/")
#define INPUT_TEST_SETS_PATH_PREFIX      std::string("testsets/")
#define OUTPUT_EVAL_FILES_PATH_PREFIX    std::string("results_test/")
#define INTPUT_TEST_SETS_NAMES           {"all","articulated_objects","cluttered_background", \
                                          "distractors","fast_motion","illumination_variation", \
                                          "low_resolution","occlusion"}
/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int /*argc*/, char** /*argv*/) {
    if(!lv::checkIfExists(VPTZ_DATASET_ROOT_DIR_PATH)) {
        std::cerr << "Cannot find dataset root directory at '" << VPTZ_DATASET_ROOT_DIR_PATH << "'!" << std::endl;
        return -1;
    }
    const char* asTestSets[] = INTPUT_TEST_SETS_NAMES;
    for(size_t nTestSetIdx=0; nTestSetIdx<sizeof(asTestSets)/sizeof(char*); ++nTestSetIdx) {
        const std::string sCurrTestSetPath = VPTZ_DATASET_ROOT_DIR_PATH+INPUT_TEST_SETS_PATH_PREFIX+asTestSets[nTestSetIdx]+".yml";
        const std::string sCurrResultFilePath = VPTZ_DATASET_ROOT_DIR_PATH+OUTPUT_EVAL_FILES_PATH_PREFIX+asTestSets[nTestSetIdx]+".yml";
        std::cout << "\n\n===============================\n\n  Setting up testset#" << nTestSetIdx+1 << " [" << asTestSets[nTestSetIdx] << "]\n\n===============================\n" << std::endl;
        try {
            vptz::Evaluator oTestEval(sCurrTestSetPath,sCurrResultFilePath,VPTZ_COMMUNICATION_DELAY,VPTZ_EXEC_DELAY_RATIO);
            for(int nTestIdx=0; nTestIdx<oTestEval.GetTestSetSize(); ++nTestIdx) {
                std::cout << "\nSetting up seq#" << nTestIdx+1 << "..." << std::endl;
                oTestEval.SetupTesting(nTestIdx);
                std::cout << "Processing seq#" << nTestIdx+1 << " [" << oTestEval.GetCurrTestSequenceName() << "]..." << std::endl;
                cv::Mat oCurrImg = oTestEval.GetInitTargetFrame();
                lvAssert(!oCurrImg.empty() && oCurrImg.type()==CV_8UC4);
                cv::Mat oTargetImg = oTestEval.GetInitTarget();
                lvAssert(!oTargetImg.empty() && oTargetImg.type()==CV_8UC4);
                cv::Rect oTargetBBox((oCurrImg.cols-oTargetImg.cols)/2,(oCurrImg.rows-oTargetImg.rows)/2,oTargetImg.cols,oTargetImg.rows); // assuming target starts centered in first frame

                ///////////////////////////////////////////////////////////////////
                // initialize your own tracker here using oCurrImg and oTargetBBox
                ///////////////////////////////////////////////////////////////////

                oTestEval.BeginTesting();
                while(!oCurrImg.empty()) {
                    // try to reposition the camera on the target's location using the previous result
                    const cv::Point oNewCenterPos((oTargetBBox.tl()+oTargetBBox.br())*0.5);
                    oCurrImg = oTestEval.GetNextFrame(oNewCenterPos);
                    if(oCurrImg.empty())
                        break;
                    lvAssert(oCurrImg.type()==CV_8UC4);

                    //////////////////////////////////////////////////////////////////
                    // feed oCurrImg to your own tracker here, and update oTargetBBox
                    //////////////////////////////////////////////////////////////////

                    oTestEval.UpdateCurrentResult(oTargetBBox);
                    cv::Mat oCurrImg_display = oCurrImg.clone();
                    if(oTargetBBox.width>0 && oTargetBBox.height>0)
                        cv::rectangle(oCurrImg_display,oTargetBBox,cv::Scalar(0,255,0),3);
                    cv::rectangle(oCurrImg_display,oTestEval.GetLastGTBoundingBox(),cv::Scalar(0,255,255));
                    while(oCurrImg_display.cols>1024 || oCurrImg_display.rows>768)
                        cv::resize(oCurrImg_display,oCurrImg_display,cv::Size(0,0),0.5,0.5);
                    cv::imshow("display",oCurrImg_display);
                    cv::waitKey(1);
                }
                oTestEval.EndTesting();
                cv::destroyAllWindows();
            }
            std::cout << std::endl;
        }
        catch(const cv::Exception& e) {
            std::cerr << "top level caught cv::Exception:\n" << e.what() << std::endl;
            break;
        }
        catch(const std::exception& e) {
            std::cerr << "top level caught std::exception:\n" << e.what() << std::endl;
            break;
        }
        catch(...) {
            std::cerr << "top level caught unknown exception." << std::endl;
            break;
        }
    }
}