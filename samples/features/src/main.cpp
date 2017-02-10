
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
        const int nMaxDisparityOffset = 20; // sets the maximum disparity offset for objects in the scene
        lvAssert(vTargetPointNIR.size()==vTargetPointsRGB.size());

        std::unique_ptr<LSS> pLSS = std::make_unique<LSS>(); // instantiation of LSS feature descriptor with default parameters
        const cv::Size oWindowSize_LSS = pLSS->windowSize(); // minimum size required for LSS description
        std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(size_t(2),0.09f); // instantiation of DASC feature descriptor with default parameters
        const cv::Size oWindowSize_DASC = pDASC->windowSize(); // minimum size required for DASC description
        std::unique_ptr<MutualInfo> pMI = std::make_unique<MutualInfo>(); // instantiation of MI score extractor helper with default parameters
        const cv::Size oWindowSize_MI = pMI->windowSize(); // minimum size required for MI scoring

        // this lambda will be called for each search point --- inside it is the disparity lookup loop
        const auto lStereoMatcher = [&](const cv::Point& oTargetPoint, const cv::Mat& oRefImg, const cv::Mat& oSearchImg, int nMinOffset, int nMaxOffset) {
            if(nMinOffset>nMaxOffset) // flip min/max disparity offsets if needed
                std::swap(nMinOffset,nMaxOffset);
            const int nTestCount = nMaxOffset-nMinOffset+1; // number of tests = number of different disparity offsets to verify
            const cv::Rect oRefZone(0,0,oRefImg.cols,oRefImg.rows); // default reference zone = entire input (reference) image

            // below is the lookup loop for the LSS matcher
            const cv::Rect oRefZone_LSS(oTargetPoint.x-oWindowSize_LSS.width/2,oTargetPoint.y-oWindowSize_LSS.height/2,oWindowSize_LSS.width,oWindowSize_LSS.height); // lookup window for the reference image
            lvAssert(oRefZone.contains(oRefZone_LSS.tl()) && oRefZone.contains(oRefZone_LSS.br()-cv::Point2i(1,1))); // lookup window should be contained in the reference image
            const cv::Rect oSearchZone_LSS(oRefZone_LSS.x+nMinOffset,oRefZone_LSS.y,oRefZone_LSS.width+(nMaxOffset-nMinOffset),oRefZone_LSS.height); // lookup window for the 'search' image
            lvAssert(oRefZone.contains(oSearchZone_LSS.tl()) && oRefZone.contains(oSearchZone_LSS.br()-cv::Point2i(1,1))); // search window should be contained in the reference image
            const cv::Mat oRef_LSS = oRefImg(oRefZone_LSS), oSearch_LSS = oSearchImg(oSearchZone_LSS); // grabs reference & search subimages using preset windows
            cv::Mat_<float> oRefDescMap_LSS,oSearchDescMap_LSS; // 3D maps where descriptor values will be generated for valid pixels (i.e. pixels far enough from image borders)
            pLSS->compute2(oRef_LSS,oRefDescMap_LSS); // extracts the actual descriptors for each (valid) input image pixel
            lvAssert(oRefDescMap_LSS.dims==3 && oRefDescMap_LSS.size[0]==oRefZone_LSS.height && oRefDescMap_LSS.size[1]==oRefZone_LSS.width); // output dense desc map should have the first two dims of the input
            const cv::Mat_<float> oRefDesc_LSS(1,oRefDescMap_LSS.size[2],oRefDescMap_LSS.ptr<float>(oWindowSize_LSS.height/2,oWindowSize_LSS.width/2)); // extract the only relevant reference descriptor (mid pixel)
            pLSS->compute2(oSearch_LSS,oSearchDescMap_LSS); // extracts the actual descriptors for each (valid) input image pixel
            lvAssert(oSearchDescMap_LSS.dims==3 && oSearchDescMap_LSS.size[0]==oSearchZone_LSS.height && oSearchDescMap_LSS.size[1]==oSearchZone_LSS.width); // output dense desc map should have the first two dims of the input
            std::vector<double> vdMatchRes_LSS((size_t)nTestCount,0.0); // output match score vector (contains one double per disparity test)
            for(size_t nTestIdx=0; nTestIdx<size_t(nTestCount); ++nTestIdx) {// for each disparity test
                const cv::Mat_<float> oSearchDesc_LSS(1,oSearchDescMap_LSS.size[2],oSearchDescMap_LSS.ptr<float>(oWindowSize_LSS.height/2,oWindowSize_LSS.width/2+int(nTestIdx))); // extract the only relevant descriptor for this test
                vdMatchRes_LSS[nTestIdx] = pLSS->calcDistance(oRefDesc_LSS,oSearchDesc_LSS); // calculate the distance between the reference image and search image descriptors
            }
            std::cout << "\nLSS Match Results (offset = " << (oTargetPoint.x+nMinOffset) << ") : \n";
            lv::print(cv::Mat_<double>(1,nTestCount,vdMatchRes_LSS.data())); // prints the match score matrix (one line, and one match score per column)
            const auto pMinScoreIter_LSS = std::min_element(vdMatchRes_LSS.begin(),vdMatchRes_LSS.end());
            std::cout << "... best match score = '" << *pMinScoreIter_LSS << "', at x = " << int((oTargetPoint.x+nMinOffset)+std::distance(vdMatchRes_LSS.begin(),pMinScoreIter_LSS)) << std::endl;

            // below is the lookup loop for the DASC matcher
            const cv::Rect oRefZone_DASC(oTargetPoint.x-oWindowSize_DASC.width/2,oTargetPoint.y-oWindowSize_DASC.height/2,oWindowSize_DASC.width,oWindowSize_DASC.height); // lookup window for the reference image
            lvAssert(oRefZone.contains(oRefZone_DASC.tl()) && oRefZone.contains(oRefZone_DASC.br()-cv::Point2i(1,1))); // lookup window should be contained in the reference image
            const cv::Rect oSearchZone_DASC(oRefZone_DASC.x+nMinOffset,oRefZone_DASC.y,oRefZone_DASC.width+(nMaxOffset-nMinOffset),oRefZone_DASC.height); // lookup window for the 'search' image
            lvAssert(oRefZone.contains(oSearchZone_DASC.tl()) && oRefZone.contains(oSearchZone_DASC.br()-cv::Point2i(1,1))); // search window should be contained in the reference image
            const cv::Mat oRef_DASC = oRefImg(oRefZone_DASC), oSearch_DASC = oSearchImg(oSearchZone_DASC); // grabs reference & search subimages using preset windows
            cv::Mat_<float> oRefDescMap_DASC,oSearchDescMap_DASC; // 3D maps where descriptor values will be generated for valid pixels (i.e. pixels far enough from image borders)
            pDASC->compute2(oRef_DASC,oRefDescMap_DASC); // extracts the actual descriptors for each (valid) input image pixel
            lvAssert(oRefDescMap_DASC.dims==3 && oRefDescMap_DASC.size[0]==oRefZone_DASC.height && oRefDescMap_DASC.size[1]==oRefZone_DASC.width); // output dense desc map should have the first two dims of the input
            const cv::Mat_<float> oRefDesc_DASC(1,oRefDescMap_DASC.size[2],oRefDescMap_DASC.ptr<float>(oWindowSize_DASC.height/2,oWindowSize_DASC.width/2)); // extract the only relevant reference descriptor (mid pixel)
            pDASC->compute2(oSearch_DASC,oSearchDescMap_DASC); // extracts the actual descriptors for each (valid) input image pixel
            lvAssert(oSearchDescMap_DASC.dims==3 && oSearchDescMap_DASC.size[0]==oSearchZone_DASC.height && oSearchDescMap_DASC.size[1]==oSearchZone_DASC.width); // output dense desc map should have the first two dims of the input
            std::vector<double> vdMatchRes_DASC((size_t)nTestCount,0.0); // output match score vector (contains one double per disparity test)
            for(size_t nTestIdx=0; nTestIdx<size_t(nTestCount); ++nTestIdx) { // for each disparity test
                const cv::Mat_<float> oSearchDesc_DASC(1,oSearchDescMap_DASC.size[2],oSearchDescMap_DASC.ptr<float>(oWindowSize_DASC.height/2,oWindowSize_DASC.width/2+int(nTestIdx))); // extract the only relevant descriptor for this test
                vdMatchRes_DASC[nTestIdx] = pDASC->calcDistance(oRefDesc_DASC,oSearchDesc_DASC); // calculate the distance between the reference image and search image descriptors
            }
            std::cout << "\nDASC Match Results (offset = " << (oTargetPoint.x+nMinOffset) << ") : \n";
            lv::print(cv::Mat_<double>(1,nTestCount,vdMatchRes_DASC.data())); // prints the match score matrix (one line, and one match score per column)
            const auto pMinScoreIter_DASC = std::min_element(vdMatchRes_DASC.begin(),vdMatchRes_DASC.end());
            std::cout << "... best match score = '" << *pMinScoreIter_DASC << "', at x = " << int((oTargetPoint.x+nMinOffset)+std::distance(vdMatchRes_DASC.begin(),pMinScoreIter_DASC)) << std::endl;

            // below is the lookup loop for the MI matcher
            const cv::Rect oRefZone_MI(oTargetPoint.x-oWindowSize_MI.width/2,oTargetPoint.y-oWindowSize_MI.height/2,oWindowSize_MI.width,oWindowSize_MI.height); // lookup window for the reference image
            lvAssert(oRefZone.contains(oRefZone_MI.tl()) && oRefZone.contains(oRefZone_MI.br()-cv::Point2i(1,1))); // lookup window should be contained in the reference image
            const cv::Mat oRef_MI = oRefImg(oRefZone_MI); // grabs reference subimage using preset window
            std::vector<double> vdMatchRes_MI((size_t)nTestCount,0.0); // output match score vector (contains one double per disparity test)
            for(int nOffset=nMinOffset; nOffset<=nMaxOffset; ++nOffset) { // for each disparity test
                const cv::Rect oCurrSearchZone_MI(oRefZone_MI.x+nOffset,oRefZone_MI.y,oRefZone_MI.width,oRefZone_MI.height); // lookup window for the 'search' image
                lvAssert(oRefZone.contains(oCurrSearchZone_MI.tl()) && oRefZone.contains(oCurrSearchZone_MI.br()-cv::Point2i(1,1))); // search window should be contained in the reference image
                const cv::Mat oCurrSearch_MI = oSearchImg(oCurrSearchZone_MI); // grabs search subimage using preset window
                vdMatchRes_MI[nOffset-nMinOffset] = pMI->compute(oRef_MI,oCurrSearch_MI); // calculate the mutual info score between the reference image and search image
            }
            std::cout << "\nMI Match Results (offset = " << (oTargetPoint.x+nMinOffset) << ") : \n";
            lv::print(cv::Mat_<double>(1,nTestCount,vdMatchRes_MI.data())); // prints the match score matrix (one line, and one match score per column)
            const auto pMaxScoreIter_MI = std::max_element(vdMatchRes_MI.begin(),vdMatchRes_MI.end());
            std::cout << "... best match score = '" << *pMaxScoreIter_MI << "', at x = " << int((oTargetPoint.x+nMinOffset)+std::distance(vdMatchRes_MI.begin(),pMaxScoreIter_MI)) << std::endl;
        };

        for(size_t nTargetPtIdx=0; nTargetPtIdx<vTargetPointsRGB.size(); ++nTargetPtIdx) { // for each test point, run the stereo matcher
            const cv::Point& oCurrTargetPointRGB = vTargetPointsRGB[nTargetPtIdx]; // get ref to current RGB point (used as 'reference')
            const cv::Point& oCurrTargetPointNIR = vTargetPointNIR[nTargetPtIdx]; // get ref to current NIR point (used as 'groundtruth')
            std::cout << "\n-------------------------------------\n" << std::endl;
            std::cout << "Target RGB Point = " << oCurrTargetPointRGB << std::endl;
            std::cout << "Target NIR Point = " << oCurrTargetPointNIR << "   (gt)" << std::endl;
            std::cout << "Min Disp Offset = " << nMinDisparityOffset << std::endl;
            std::cout << "Max Disp Offset = " << nMaxDisparityOffset << std::endl;
            std::cout << "Ideal Disp Offset = " << (oCurrTargetPointNIR.x-oCurrTargetPointRGB.x) << "\n" << std::endl;
            lStereoMatcher(oCurrTargetPointRGB,oInputRGB,oInputNIR,nMinDisparityOffset,nMaxDisparityOffset); // run the stereo matching algos
            std::cout << std::endl;
        }
        std::cout << "\nAll done.\n" << std::endl;
    }
    catch(const cv::Exception& e) {std::cout << "\nmain caught cv::Exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(const std::exception& e) {std::cout << "\nmain caught std::exception:\n" << e.what() << "\n" << std::endl; return -1;}
    catch(...) {std::cout << "\nmain caught unhandled exception\n" << std::endl; return -1;}
    return 0;
}