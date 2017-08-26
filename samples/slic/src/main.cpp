
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
// This sample demonstrates how to compute superpixel masks on GPU via CUDA
// using the SLIC segmentation algorithm. The only required input (the image)
// is located in the sample data directory.
//
/////////////////////////////////////////////////////////////////////////////

#include "litiv/imgproc.hpp" // includes all image segmentation algos, along with most core utility & opencv headers

int main(int, char**) { // this sample uses no command line argument
    try { // its always a good idea to scope your app's top level in some try/catch blocks!
        // side note: CUDA device code will print debug info and call std::exit if an error is encountered
        const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg"); // load a training image taken from the BSDS500 dataset
        if(oInput.empty()) // check if the mat is empty (i.e. if the image failed to load)
            CV_Error(-1,"Could not load test image from internal sample data folder");
        cv::imshow("oInput",oInput);
        SLIC oAlgo; // instantiate SLIC segmentation algo (with default empty constructor)
        oAlgo.initialize(oInput.size()/*, ...*/); // use default parameters for SLIC constructor initialization
        oAlgo.segment(oInput); // run SLIC segmentation algo
        oAlgo.enforceConnectivity(); // enforce superpixel connectivity
        const cv::Mat& oSPXMask = oAlgo.getLabels(); // returns segmentation labels for the input image
        lv::doNotOptimize(oSPXMask); // for some reason, unless we pass the algo output to another lib call, kernels don't execute on MSVC2015 in release...
        cv::imshow("Segmentation output",SLIC::displayBound(oInput,oSPXMask,cv::Scalar(255,0,0)));
        cv::imshow("Superpixel RGB Mean", SLIC::displayMean(oInput, oSPXMask));
        cv::waitKey(0); // wait for the user to press a key before shutting down
    }
    catch(const lv::Exception&) {std::cout << "\nmain caught lv::Exception (check stderr)\n" << std::endl; return -1;}
    catch(const cv::Exception&) {std::cout << "\nmain caught cv::Exception (check stderr)\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}
