
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
// This sample demonstrates how to use various 2d image feature descriptors
// in a simplistic multimodal stereo disparity benchmark. The required inputs
// (the stereo image pair) is located in the sample data directory.
//
/////////////////////////////////////////////////////////////////////////////

#include "litiv/features2d.hpp" // includes all feature extractors, along with most core utility & opencv headers

int main(int, char**) { // this sample uses no command line argument
    try { // its always a good idea to scope your app's top level in some try/catch blocks!

        const cv::Mat oInputRGB = cv::imread(SAMPLES_DATA_ROOT "/multispectral_stereo_ex/img2.png"); // load the color image from the stereo pair
        const cv::Mat oInputNIR = cv::imread(SAMPLES_DATA_ROOT "/multispectral_stereo_ex/img1_corr_h0v8.png",cv::IMREAD_GRAYSCALE); // load the infrared image from the stereo pair
        if(oInputRGB.empty() || oInputNIR.empty() || oInputRGB.size()!=oInputNIR.size() || oInputRGB.size()!=cv::Size(800,600)) // check if any image failed to load
            lvError("Could not load test image from internal sample data folder");
        const std::vector<cv::Point> vTargetPointsRGB = {cv::Point(603,122)}; // target points in the RGB image (known, treated as input)
        const std::vector<cv::Point> vTargetPointNIR = {cv::Point(613,122)}; // target points in the NIR image (unknown, treated as gt)
        const int nMinDisparityOffset = 0; // sets the minimum disparity offset for objects in the scene
        const int nMaxDisparityOffset = 40; // sets the maximum disparity offset for objects in the scene

        std::unique_ptr<LSS> pLSS = std::make_unique<LSS>(); // instantiation of LSS feature descriptor with default parameters
        const cv::Size oWindowSize_LSS = pLSS->windowSize(); // minimum size required for LSS description
        std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(size_t(2),0.09f); // instantiation of DASC feature descriptor with default parameters
        const cv::Size oWindowSize_DASC = pDASC->windowSize(); // minimum size required for DASC description
        std::unique_ptr<MutualInfo> pMI = std::make_unique<MutualInfo>(); // instantiation of MI score extractor helper with default parameters
        const cv::Size oWindowSize_MI = pMI->windowSize(); // minimum size required for MI scoring

        const auto lStereoMatcher = [&](const cv::Point& oTargetPoint, const cv::Mat& oRefImg, const cv::Mat& oSearchImg, int nMinOffset, int nMaxOffset) {
            if(nMinOffset>nMaxOffset)
                std::swap(nMinOffset,nMaxOffset);
            const int nTestCount = nMaxOffset-nMinOffset;
            const cv::Rect oRefZone(0,0,oRefImg.cols,oRefImg.rows);

            const cv::Rect oRefZone_LSS(oTargetPoint.x-oWindowSize_LSS.width/2,oTargetPoint.y-oWindowSize_LSS.height/2,oWindowSize_LSS.width,oWindowSize_LSS.height);
            lvAssert(oRefZone.contains(oRefZone_LSS.tl()) && oRefZone.contains(oRefZone_LSS.br()));
            const cv::Rect oSearchZone_LSS(oRefZone_LSS.x+nMinOffset,oRefZone_LSS.y,oRefZone_LSS.width+nTestCount,oRefZone_LSS.height);
            lvAssert(oRefZone.contains(oSearchZone_LSS.tl()) && oRefZone.contains(oSearchZone_LSS.br()));
            const cv::Mat oRef_LSS = oRefImg(oRefZone_LSS), oSearch_LSS = oSearchImg(oSearchZone_LSS);
            cv::Mat oRefDesc_LSS,oSearchDesc_LSS;
            pLSS->compute2(oRef_LSS,oRefDesc_LSS);
            lvAssert(oRefDesc_LSS.dims==3 && oSearchDesc_LSS.size[0]==1 && oSearchDesc_LSS.size[1]==1);
            pLSS->compute2(oSearch_LSS,oSearchDesc_LSS);
            lvAssert(oRefDesc_LSS.dims==3 && oSearchDesc_LSS.size[0]==1 && oSearchDesc_LSS.size[1]==size_t(nTestCount));
            pLSS->calcDistance(oRefDesc_LSS,)

            const cv::Rect oRefZone_DASC(oTargetPoint.x-oWindowSize_DASC.width/2,oTargetPoint.y-oWindowSize_DASC.height/2,oWindowSize_DASC.width,oWindowSize_DASC.height);
            lvAssert(oRefZone.contains(oRefZone_DASC.tl()) && oRefZone.contains(oRefZone_DASC.br()));
            const cv::Rect oSearchZone_DASC(oRefZone_DASC.x+nMinOffset,oRefZone_DASC.y,oRefZone_DASC.width+nTestCount,oRefZone_DASC.height);
            lvAssert(oRefZone.contains(oSearchZone_DASC.tl()) && oRefZone.contains(oSearchZone_DASC.br()));
            const cv::Mat oRef_DASC = oRefImg(oRefZone_DASC), oSearch_DASC = oSearchImg(oSearchZone_DASC);
            cv::Mat oRefDesc_DASC,oSearchDesc_DASC;
            pDASC->compute2(oRef_DASC,oRefDesc_DASC);
            pDASC->compute2(oSearch_DASC,oSearchDesc_DASC);

            const cv::Rect oRefZone_MI(oTargetPoint.x-oWindowSize_MI.width/2,oTargetPoint.y-oWindowSize_MI.height/2,oWindowSize_MI.width,oWindowSize_MI.height);
            lvAssert(oRefZone.contains(oRefZone_MI.tl()) && oRefZone.contains(oRefZone_MI.br()));
            const cv::Mat oRef_MI = oRefImg(oRefZone_MI);
            std::vector<double> vdMatchRes_MI(size_t(nMaxOffset-nMinOffset),0.0);
            for(int nOffset=nMinOffset; nOffset<nMaxOffset; ++nOffset) {
                const cv::Rect oCurrSearchZone_MI(oRefZone_MI.x+nOffset,oRefZone_MI.y,oRefZone_MI.width,oRefZone_MI.height);
                lvAssert(oRefZone.contains(oCurrSearchZone_MI.tl()) && oRefZone.contains(oCurrSearchZone_MI.br()));
                const cv::Mat oCurrSearch_MI = oSearchImg(oCurrSearchZone_MI);
                vdMatchRes_MI[nOffset-nMinOffset] = pMI->compute(oRef_MI,oCurrSearch_MI);
            }
        };


        for(const cv::Point& oCurrTargetPointRGB : vTargetPointsRGB) // for each target point in the RGB image
            lStereoMatcher(oCurrTargetPointRGB,oInputRGB,oInputNIR,nMinDisparityOffset,nMaxDisparityOffset);

        // @@@ display match curves

    }
    catch(const cv::Exception& e) {std::cout << "\nmain caught cv::Exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}
