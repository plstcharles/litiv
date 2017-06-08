
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
// Copyright 2014 Gengjie Chen; chgengj <at> mail2.sysu.edu.cn
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
//======================================================================================
//
// This program is used to get the ground truth of the dataset.
//
//  MOVING BOUNDING BOXES / CHANGING CAMERA ANGLE:
//  => 'a', 'd', 'w' and 's' move the camera around to navigate the scenario
//  => 'left arrow' and 'down arrow' shrink the bbox width and height, respectively
//  => 'right arrow' and 'up arrow' expand the bbox width and height, respectively
//  => double clicking any position with the left mouse button moves the box there
//  => dragging the bbox center with the left mouse button moves the bbox
//  => dragging the bottom-right point with left mouse button changes the bbox size
//  => dragging any bbox boundary with the left mouse button changes its size
//
//  SAVING OUTPUT:
//  => pressing any key but the ones listed above saves the current bbox configuration
//     for the current frame and advances to the next frame
//  => if no starting frame index is given (i.e. OUTPUT_GT_SEQUENCE_START_IDX=-1),
//     pre-seeking can be achieved by pressing any key but space; this mode is shown
//     by a red bbox. Once the desired starting frame is reached, press space to begin
//     annotating; the bbox boundary will turn green to show that it is now 'recording'.
//  => 'esc' quits the application, always saving (and overwriting) the output
//  => when rewriting annotations, a yellow bounding box indicates that annotations
//     are being directly imported from the last GT file to help initial positioning
//     (note that in this mode, all frames from the first one are always 'recorded')
//  => '-' disables the rewriting mode for the remainder of a GT file, and lets you
//     position all boxes by yourself (same as double-right-clicking)
//
//======================================================================================

#include "litiv/vptz/virtualptz.hpp"
#include <iostream>
#include <sys/stat.h>

///////////////////////////////////////////
#define USE_COMPILE_TIME_DIRS         0
#define OUTPUT_GT_SEQUENCE_WIDTH      (640)
#define OUTPUT_GT_SEQUENCE_HEIGHT     (480)
#define OUTPUT_GT_SEQUENCE_V_FOV      (90)
#define OUTPUT_GT_SEQUENCE_START_IDX  (-1)
///////////////////////////////////////////////////////////////////////////////////////
#if USE_COMPILE_TIME_DIRS /////////////////////////////////////////////////////////////
#define INPUT_SCENARIO_PATH         "W:/virtualptz/scenario5/frames/scenario5_%06d.jpg"
#define OUTPUT_GT_SEQUENCE_PATH     "W:/virtualptz/scenario5/gt/scenario5_NEW_PL.yml"
#else //!USE_COMPILE_TIME_DIRS ////////////////////////////////////////////////////////
#define INPUT_SCENARIO_PATH         sInputPath.c_str()
#define OUTPUT_GT_SEQUENCE_PATH     sOutputPath.c_str()
#endif //!USE_COMPILE_TIME_DIRS ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
#ifndef WIN32
#define tolower(c) std::tolower(c)
#endif //!WIN32

class BBox {
public:
    cv::Point top_left, bottom_right;
    BBox(cv::Point tl=cv::Point(0,0), cv::Point br=cv::Point(50,100)) {
        top_left = tl;
        bottom_right = br;
    }
    cv::Point center() { return 0.5*(top_left+bottom_right); }
    void changeCenter(cv::Point cen) {
        cv::Point offset = cen-center();
        top_left += offset;
        bottom_right += offset;
    }
};

BBox g_oTargetBB;
bool g_bQuit = false;
bool g_bWriting = false;
bool g_bRewriting = false;
bool g_bReadjusting = false;
bool g_bMouseDragging = false;
bool g_bRepositioning = false;
cv::Mat g_oCurrFrame;
cv::Mat g_oCurrDrawFrame;

void displayRect() {
    cv::rectangle(g_oCurrDrawFrame,cv::Rect(g_oTargetBB.top_left,g_oTargetBB.bottom_right),cv::Scalar(0,(g_bRewriting||g_bWriting)?255:0,(g_bRewriting&&!g_bWriting)?255:g_bWriting?0:255));
    cv::circle(g_oCurrDrawFrame,g_oTargetBB.center(), 2, cv::Scalar(255,0,0), 3);
    cv::circle(g_oCurrDrawFrame,g_oTargetBB.bottom_right, 2, cv::Scalar(255,0,0), 3);
    cv::imshow("Current View",g_oCurrDrawFrame);
    cv::waitKey(1);
}

void onMouse(int nEventCode, int x, int y, int, void*) {
    if(g_bRepositioning) {
        int error = 10;
        if(nEventCode==cv::EVENT_LBUTTONDOWN)
            g_bReadjusting = true;
        else if(nEventCode==cv::EVENT_LBUTTONUP)
            g_bReadjusting = false;
        else if(nEventCode==cv::EVENT_RBUTTONDOWN && !g_bRewriting)
            g_bMouseDragging = !g_bMouseDragging;
        else if(nEventCode==cv::EVENT_RBUTTONDBLCLK) {
            g_bMouseDragging = !g_bMouseDragging;
            g_bRewriting = false;
            g_bWriting = true;
        }
        else if(nEventCode==cv::EVENT_MOUSEMOVE) {
            if(g_bReadjusting) {
                if(cv::norm(cv::Point(x,y)-g_oTargetBB.center()) < error)
                    g_oTargetBB.changeCenter(cv::Point(x,y));
                else if(cv::norm(cv::Point(x,y)-g_oTargetBB.bottom_right) < error)
                    g_oTargetBB.bottom_right = cv::Point(x,y);
                else if(abs(x-g_oTargetBB.top_left.x) < error)
                    g_oTargetBB.top_left.x = x;
                else if(abs(x-g_oTargetBB.bottom_right.x) < error)
                    g_oTargetBB.bottom_right.x = x;
                else if(abs(y-g_oTargetBB.top_left.y) < error)
                    g_oTargetBB.top_left.y = y;
                else if(abs(y-g_oTargetBB.bottom_right.y) < error)
                    g_oTargetBB.bottom_right.y = y;
            }
            else if(g_bMouseDragging) {
                g_oTargetBB.changeCenter(cv::Point(x,y));
            }
        }
        else if(nEventCode==cv::EVENT_LBUTTONDBLCLK)
            g_oTargetBB.changeCenter(cv::Point(x,y));
        g_oCurrFrame.copyTo(g_oCurrDrawFrame);
        displayRect();
    }
}

bool exists(const char* path) {
    struct stat buffer;
    return (stat(path,&buffer)==0);
}

bool getUserInput_YesNoQuestion() {
    std::string sInput;
    while(true) {
        std::cin >> sInput;
        std::transform(sInput.begin(), sInput.end(), sInput.begin(), tolower);
        if(sInput=="no" || sInput=="n" || sInput=="non")
            return false;
        else if(sInput=="y" || sInput=="yes")
            return true;
        std::cout << " Huh? (" << sInput << ")\n >> ";
    }
}

int getUserInput_IntegerFromRange(int min, int max) {
    int nInput;
    while(true) {
        std::cin >> nInput;
        if(nInput>=min && nInput<=max)
            return nInput;
        std::cout << " Huh? (" << nInput << ")\n >> ";
    }
}

std::string getUserInput_string() {
    std::string sStr;
    while(true) {
        std::cin >> sStr;
        if(!sStr.empty())
            break;
        std::cout << "Huh?\n >> ";
    }
    return sStr;
}

int main(int /*argc*/, char** /*argv*/) {
#if !USE_COMPILE_TIME_DIRS
    std::cout << "\n Please provide the path to an .avi file or to an image folder: \n >> ";
    const std::string sInputPath = getUserInput_string();
    std::cout << "\n Please provide the path for the desired output YML gt file: \n >> ";
    const std::string sOutputPath = getUserInput_string();
    std::cout << std::endl << std::endl;
#endif //!USE_COMPILE_TIME_DIRS
    cv::FileStorage outputGT;
    try {
        if(exists(OUTPUT_GT_SEQUENCE_PATH)) {
            std::cout << "\n GT file already exists; overwrite? [y/n]\n >> ";
            if(!getUserInput_YesNoQuestion())
                return 0;
            std::cout << "\n Restart annotating using old GT file as base? [y/n]\n >> ";
            g_bRewriting = getUserInput_YesNoQuestion();
            std::cout << std::endl << std::endl;
        }
        vptz::Camera testPTZ( INPUT_SCENARIO_PATH,
                                  OUTPUT_GT_SEQUENCE_V_FOV,
                                  OUTPUT_GT_SEQUENCE_WIDTH,
                                  OUTPUT_GT_SEQUENCE_HEIGHT,
                                  0,90);
        g_oTargetBB.changeCenter(cv::Point(OUTPUT_GT_SEQUENCE_WIDTH/2,OUTPUT_GT_SEQUENCE_HEIGHT/2));
        const int nTotFrameCount = (int)testPTZ.Get(vptz::PTZ_CAM_FRAME_NUM);
        int nFirstFrameIdx = OUTPUT_GT_SEQUENCE_START_IDX;
        if(nFirstFrameIdx>=nTotFrameCount)
            lvError("cannot seek video to the desired first frame");
        cv::FileStorage inputGT;
        cv::FileNode inputGTAnnotations;
        int nTotPreviousGTAnnotations = 0;
        int nLastPreviousGTAnnotationIdx = -1;
        if(nFirstFrameIdx<0)
            nFirstFrameIdx = 0;
        int nCurrFrameIdx = nFirstFrameIdx;
        if(g_bRewriting && !inputGT.open(OUTPUT_GT_SEQUENCE_PATH,cv::FileStorage::READ))
            lvError("could not open old YML file at given location");
        else if(g_bRewriting) {
            inputGTAnnotations = inputGT["basicGroundTruth"];
            if(inputGTAnnotations.empty())
                lvError("previous annotation file was empty");
            nFirstFrameIdx = inputGTAnnotations[(int)0]["framePos"];
            nTotPreviousGTAnnotations = (int)(inputGTAnnotations.size());
            nLastPreviousGTAnnotationIdx = inputGTAnnotations[nTotPreviousGTAnnotations-1]["framePos"];
            std::cout << "\n Restart annotation at which frame index? [" << nFirstFrameIdx << "--"<< nFirstFrameIdx+nTotPreviousGTAnnotations-1 << "]\n >> ";
            nCurrFrameIdx = getUserInput_IntegerFromRange(nFirstFrameIdx,nFirstFrameIdx+nTotPreviousGTAnnotations-1);
            std::cout << std::endl << std::endl;
        }
        if(!outputGT.open(OUTPUT_GT_SEQUENCE_PATH,cv::FileStorage::WRITE))
            lvError("could not create YML file at given location");
        if(!g_bRewriting) {
            outputGT << "Name" << "Ground Truth"
                     << "Function" << "This is the ground-truth data (bounding box) of PTZ camera benchmark"
                     << "Notation" << "In each image, the target is at center of the view"
                     << "InputPath" << INPUT_SCENARIO_PATH
                     << "totalFrameNum" << testPTZ.Get(vptz::PTZ_CAM_FRAME_NUM)
                     << "frameImageWidth" << testPTZ.Get(vptz::PTZ_CAM_OUTPUT_WIDTH)
                     << "frameImageHeight" << testPTZ.Get(vptz::PTZ_CAM_OUTPUT_HEIGHT)
                     << "verticalFOV" << testPTZ.Get(vptz::PTZ_CAM_VERTI_FOV);
        }
        else {
            std::string sName, sFunction, sNotation;//, sInputPath;
            inputGT["Name"] >> sName;
            inputGT["Function"] >> sFunction;
            inputGT["Notation"] >> sNotation;
            //inputGT["InputPath"] >> sInputPath;
            double dImageWidth, dImageHeight, dFOV;
            inputGT["frameImageWidth"] >> dImageWidth;
            inputGT["frameImageHeight"] >> dImageHeight;
            inputGT["verticalFOV"] >> dFOV;
            outputGT << "Name" << sName
                     << "Function" << sFunction
                     << "Notation" << sNotation
                     << "InputPath" << INPUT_SCENARIO_PATH
                     << "totalFrameNum" << nTotFrameCount
                     << "frameImageWidth" << dImageWidth
                     << "frameImageHeight" << dImageHeight
                     << "verticalFOV" << dFOV;
            testPTZ.Set(vptz::PTZ_CAM_OUTPUT_WIDTH,dImageWidth);
            testPTZ.Set(vptz::PTZ_CAM_OUTPUT_HEIGHT,dImageHeight);
            testPTZ.Set(vptz::PTZ_CAM_VERTI_FOV,dFOV);
        }
        testPTZ.Set(vptz::PTZ_CAM_FRAME_POS,nFirstFrameIdx);
        g_oCurrFrame = testPTZ.GetFrame();
        g_oCurrDrawFrame = g_oCurrFrame.clone();
        cv::namedWindow("Current View");
        cv::setMouseCallback("Current View", onMouse, 0);
        std::string imageName;
        outputGT << "basicGroundTruth" << "[" ;
        for(int nPreviousFrameIdx=nFirstFrameIdx; nPreviousFrameIdx<nCurrFrameIdx; ++nPreviousFrameIdx) {
            std::cout << "  frame#" << nPreviousFrameIdx << ";\trewriting before start frame..." << std::endl;
            int nPreviousGTArrayIdx = nPreviousFrameIdx-nFirstFrameIdx;
            g_oTargetBB.top_left = cv::Point(0,0);
            g_oTargetBB.bottom_right = cv::Point((int)inputGTAnnotations[nPreviousGTArrayIdx]["width"],(int)inputGTAnnotations[nPreviousGTArrayIdx]["height"]);
            g_oTargetBB.changeCenter(cv::Point(OUTPUT_GT_SEQUENCE_WIDTH/2,OUTPUT_GT_SEQUENCE_HEIGHT/2));
            testPTZ.Set(vptz::PTZ_CAM_HORI_ANGLE,inputGTAnnotations[nPreviousGTArrayIdx]["horizontalAngle"]);
            testPTZ.Set(vptz::PTZ_CAM_VERTI_ANGLE,inputGTAnnotations[nPreviousGTArrayIdx]["verticalAngle"]);
            outputGT << "{:" << "framePos" << nPreviousFrameIdx << "height" << g_oTargetBB.bottom_right.y - g_oTargetBB.top_left.y << "width" << g_oTargetBB.bottom_right.x - g_oTargetBB.top_left.x
                     << "horizontalAngle" << testPTZ.Get(vptz::PTZ_CAM_HORI_ANGLE) << "verticalAngle" << testPTZ.Get(vptz::PTZ_CAM_VERTI_ANGLE) << "}";
        }
        for(; nCurrFrameIdx<nTotFrameCount; ++nCurrFrameIdx) {
            std::cout << "  frame#" << nCurrFrameIdx;
            testPTZ.Set(vptz::PTZ_CAM_FRAME_POS, nCurrFrameIdx);
            g_bRewriting &= nCurrFrameIdx<=nLastPreviousGTAnnotationIdx;
            if(g_bRewriting) {
                int nCurrGTArrayIdx = nCurrFrameIdx-nFirstFrameIdx;
                int bgtBBoxwidth = inputGTAnnotations[nCurrGTArrayIdx]["width"];
                int bgtBBoxheight = inputGTAnnotations[nCurrGTArrayIdx]["height"];
                g_oTargetBB.top_left = cv::Point(0,0);
                g_oTargetBB.bottom_right = cv::Point(bgtBBoxwidth,bgtBBoxheight);
                g_oTargetBB.changeCenter(cv::Point(OUTPUT_GT_SEQUENCE_WIDTH/2,OUTPUT_GT_SEQUENCE_HEIGHT/2));
                testPTZ.Set(vptz::PTZ_CAM_HORI_ANGLE,inputGTAnnotations[nCurrGTArrayIdx]["horizontalAngle"]);
                testPTZ.Set(vptz::PTZ_CAM_VERTI_ANGLE,inputGTAnnotations[nCurrGTArrayIdx]["verticalAngle"]);
            }
            else if(nLastPreviousGTAnnotationIdx>=0 && !g_bWriting)
                g_bWriting = true;
            if(!g_bQuit) {
                g_bRepositioning = true;
                std::cout << ";\tsetting bounding box..." << std::endl;
                g_oCurrFrame = testPTZ.GetFrame();
                while(true) {
                    g_oCurrFrame.copyTo(g_oCurrDrawFrame);
                    displayRect();
                    int key_orig = cv::waitKey();
                    double horiAngleVar = 0;
                    double vertiAngleVar = 0;
                    // warning: arrow/keypad keys might be platform-specific
                    if(key_orig==2555904 || key_orig==1113939) { // arrow_right
                        g_oTargetBB.bottom_right.x = std::min(g_oCurrDrawFrame.cols-1,g_oTargetBB.bottom_right.x+1);
                        g_oTargetBB.top_left.x = std::max(0,g_oTargetBB.top_left.x-1);
                    }
                    else if(key_orig==2424832 || key_orig==1113937) { // arrow_left
                        g_oTargetBB.bottom_right.x = std::max(g_oTargetBB.top_left.x+2,g_oTargetBB.bottom_right.x-1);
                        g_oTargetBB.top_left.x = std::min(g_oTargetBB.top_left.x+1,g_oTargetBB.bottom_right.x-2);
                    }
                    else if(key_orig==2490368 || key_orig==1113938) { // arrow_up
                        g_oTargetBB.bottom_right.y = std::min(g_oCurrDrawFrame.rows-1,g_oTargetBB.bottom_right.y+1);
                        g_oTargetBB.top_left.y = std::max(0,g_oTargetBB.top_left.y-1);
                    }
                    else if(key_orig==2621440 || key_orig==1113940) { // arrow_down
                        g_oTargetBB.bottom_right.y = std::max(g_oTargetBB.top_left.y+2,g_oTargetBB.bottom_right.y-1);
                        g_oTargetBB.top_left.y = std::min(g_oTargetBB.top_left.y+1,g_oTargetBB.bottom_right.y-2);
                    }
                    else {
                        unsigned char key = tolower(key_orig);
                        switch(key) {
                            case 'a':
                                horiAngleVar += 1.0;
                                break;
                            case 'd':
                                horiAngleVar -= 1.0;
                                break;
                            case 's':
                                vertiAngleVar += 1.0;
                                break;
                            case 'w':
                                vertiAngleVar -= 1.0;
                                break;
                            case '-': // disables 'rewriting' mode and goes to regular 'writing' mode
                                g_bRewriting = false;
                                g_bWriting = true;
                                break;
                            case 27: // escape
                                g_bQuit = true;
                                cv::destroyWindow("Current View");
                            case ' ': // activates 'writing' mode if the user was previously 'seeking' for a first frame
                                g_bWriting = !g_bRewriting;
                            default:
                                g_bRepositioning = false;
                        }
                    }
                    if(!g_bRepositioning)
                        break;
                    testPTZ.Set(vptz::PTZ_CAM_HORI_ANGLE, horiAngleVar+testPTZ.Get(vptz::PTZ_CAM_HORI_ANGLE));
                    testPTZ.Set(vptz::PTZ_CAM_VERTI_ANGLE, vertiAngleVar+testPTZ.Get(vptz::PTZ_CAM_VERTI_ANGLE));
                    g_oCurrFrame = testPTZ.GetFrame();
                }
            }
            if(g_bQuit && !g_bRewriting)
                break;
            else if(g_bQuit)
                std::cout << ";\trewriting before quitting..." << std::endl;
            else {
                testPTZ.GoToPosition(g_oTargetBB.center());
                if(!g_bMouseDragging)
                    g_oTargetBB.changeCenter(cv::Point(OUTPUT_GT_SEQUENCE_WIDTH/2,OUTPUT_GT_SEQUENCE_HEIGHT/2));
                g_oCurrFrame = testPTZ.GetFrame();
            }
            if(g_bRewriting || g_bWriting) {
                outputGT << "{:" << "framePos" << nCurrFrameIdx << "height" << g_oTargetBB.bottom_right.y - g_oTargetBB.top_left.y << "width" << g_oTargetBB.bottom_right.x - g_oTargetBB.top_left.x
                         << "horizontalAngle" << testPTZ.Get(vptz::PTZ_CAM_HORI_ANGLE) << "verticalAngle" << testPTZ.Get(vptz::PTZ_CAM_VERTI_ANGLE) << "}";
            }
        }
        outputGT.release();
    }
    catch(const std::exception& e) {
        std::cerr << "top level caught std::exception:\n" << e.what() << std::endl;
    }
    catch(...) {
        std::cerr << "top level caught unknown exception." << std::endl;
    }
    return 0;
}
