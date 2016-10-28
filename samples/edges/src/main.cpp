
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
// This sample demonstrates how to obtain edge masks using an edge detector
// algo, either with a full detection sensitivity threshold sweep (i.e. with
// "confidence" levels), or with a single hard-coded threshold. The only
// required input (the image) is located in the sample data directory.
//
/////////////////////////////////////////////////////////////////////////////

#include "litiv/imgproc.hpp" // includes all edge detection algos, along with most core utility & opencv headers

#define FULL_THRESH_ANALYSIS 1 // defines whether we want a full threshold sweep or not (1/0)

int main(int, char**) { // this sample uses no command line argument
    try { // its always a good idea to scope your app's top level in some try/catch blocks!
        const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg"); // load a training image taken from the BSDS500 dataset
        if(oInput.empty()) // check if the mat is empty (i.e. if the image failed to load)
            CV_Error(-1,"Could not load test image from internal sample data folder");
        cv::Mat oEdgeMask; // no need to preallocate the output matrix (the algo will make sure it is allocated at some point)
        //std::shared_ptr<IEdgeDetector> pAlgo = std::make_shared<EdgeDetectorCanny>(); // instantiate an edge detector algo with default parameters
        std::shared_ptr<IEdgeDetector> pAlgo = std::make_shared<EdgeDetectorLBSP>(); // instantiate an edge detector algo with default parameters
#if FULL_THRESH_ANALYSIS
        pAlgo->apply(oInput,oEdgeMask); // apply the edge detector threshold sweep on an image (oInput), and fetch the result simultaneously (oEdgeMask)
#else //(!FULL_THRESH_ANALYSIS)
        const double dDefaultThreshold = pAlgo->getDefaultThreshold(); // defines the single threshold to use for non-sweep mode
        pAlgo->apply_threshold(oInput,oEdgeMask,dDefaultThreshold); // apply the edge detector on an image (oInput), and fetch the result simultaneously (oEdgeMask)
#endif //(!FULL_THRESH_ANALYSIS)
        cv::imshow("Edge detection output",oEdgeMask); // display the output edge mask
        cv::waitKey(0); // wait for the user to press a key before shutting down
    }
    catch(const cv::Exception& e) {std::cout << "\nmain caught cv::Exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}
