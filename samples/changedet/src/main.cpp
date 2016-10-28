
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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
// This sample demonstrates how to extract foreground masks using a change
// detection via background subtraction algo, using either a webcam or a
// user-provided video as input. It will also display the processing speed.
//
/////////////////////////////////////////////////////////////////////////////

#include "litiv/video.hpp" // includes all background subtraction algos, along with most core utility & opencv headers

#define USE_WEBCAM 1 // defines whether OpenCV's first detected camera should be used, or a given video should be read
#if !USE_WEBCAM
#define VIDEO_FILE_PATH "/path/to/some/user/specified/video/file.avi" // if you want to use your own file, put it here!
#endif //(!USE_WEBCAM)

int main(int, char**) { // this sample uses no command line argument
    try { // its always a good idea to scope your app's top level in some try/catch blocks!
#if USE_WEBCAM
        cv::VideoCapture oCap(0); // will attempt to connect to the first available webcam
#else //(!USE_WEBCAM)
        cv::VideoCapture oCap(VIDEO_FILE_PATH); // will attempt to load the given video file (make sure you have the codec installed!)
#endif //(!USE_WEBCAM)
        if(!oCap.isOpened()) // check if the video capture object is initialized
            CV_Error(-1,"Could not open video capture object");
        cv::Mat oInput; // this matrix will be used to fetch input frames (no need to preallocate)
        oCap >> oInput; // minimalistic way to fetch a frame from a video capture object
        if(oInput.empty()) // check if the fetched frame is empty (i.e. if the video failed to load/seek)
            CV_Error(-1,"Could not fetch video frame from video capture object");
        cv::Mat oForegroundMask; // no need to preallocate the output matrix (the algo will make sure it is allocated at some point)
        //std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorLOBSTER>(); // instantiate a background subtractor algo with default parameters
        std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorSuBSENSE>(); // instantiate a background subtractor algo with default parameters
        //std::shared_ptr<IBackgroundSubtractor> pAlgo = std::make_shared<BackgroundSubtractorPAWCS>(); // instantiate a background subtractor algo with default parameters
        const double dDefaultLearningRate = pAlgo->getDefaultLearningRate(); // gets the suggested learning rate to use post-initialization (algo-dependent, some will totally ignore it)
        cv::Mat oROI; // specifies the segmentation region of interest (when the matrix is left unallocated, the full frame is used by default)
        pAlgo->initialize(oInput,oROI); // initialize the background model using the video's first frame (it may already contain foreground; the algo should adapt to any case)
        size_t nCurrInputIdx = 0; // use a frame counter to report average processing speed
        lv::StopWatch oStopWatch;
        while(true) { // loop, as long as we can still fetch frames
            const double dCurrLearningRate = nCurrInputIdx<=50?1:dDefaultLearningRate; // boost the learning rate in the first ~50 frames for initialization, and switch to default after
            pAlgo->apply(oInput,oForegroundMask,dCurrLearningRate); // apply background subtraction on a new frame (oInput), and fetch the result simultaneously (oForegroundMask)
            cv::imshow("Video input",oInput); // display the input video frame
            cv::imshow("Segmentation output",oForegroundMask); // display the output segmentation mask (white = foreground)
            if(cv::waitKey(1)==(int)27) // immediately refresh the display with a 1ms pause, and continue processing (or quit if the escape key is pressed)
                break;
            oCap >> oInput; // fetch the next frame from the video capture object
            if(oInput.empty()) // if the frame is empty, we hit the end of the video, or the webcam was shut off
                break;
            if((++nCurrInputIdx%30)==0) // every 30 frames, display the total average processing speed
                std::cout << " avgFPS = " << nCurrInputIdx/oStopWatch.elapsed() << std::endl;
        }

    }
    catch(const cv::Exception& e) {std::cout << "\nmain caught cv::Exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}
