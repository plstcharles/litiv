
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

//#include "litiv/datasets.hpp"
#include "litiv/imgproc/StereoGraphMatcher.hpp"
//#include "litiv/video.hpp"

int main(int, char**) {
    try {
        // here, assume prerectified, and base->offset is positive disparity (going left)
        const cv::Mat_<uchar> oTestImgBase = cv::imread(SAMPLES_DATA_ROOT "/stereo/middlebury-2005-art/view1.png",cv::IMREAD_GRAYSCALE);
        lvAssert(!oTestImgBase.empty() && oTestImgBase.channels()==1);
        const cv::Mat_<uchar> oTestImgOffset = cv::imread(SAMPLES_DATA_ROOT "/stereo/middlebury-2005-art/view5.png",cv::IMREAD_GRAYSCALE);
        lvAssert(!oTestImgOffset.empty() && oTestImgOffset.channels()==1 && oTestImgOffset.size()==oTestImgBase.size());
        const cv::Mat_<uchar> oTestGTBase = cv::imread(SAMPLES_DATA_ROOT "/stereo/middlebury-2005-art/disp1.png",cv::IMREAD_GRAYSCALE);
        lvAssert(!oTestGTBase.empty() && oTestGTBase.channels()==1 && oTestGTBase.size()==oTestImgBase.size());
        const std::vector<uchar> vDisparities = lv::filter_out(lv::unique(oTestGTBase),std::vector<uchar>{0}); // remove 0 ('dc' in middlebury)
        const std::vector<StereoGraphMatcher::LabelType> vLabels(vDisparities.begin(),vDisparities.end());
        StereoGraphMatcher oMatcher(oTestImgBase.size(),vLabels);
        StereoGraphMatcher::MatArray aInput = {oTestImgBase,oTestImgOffset};
        StereoGraphMatcher::MatArray aOutput;
        oMatcher.apply(aInput,aOutput);
        cv::imshow("out0",aOutput[0]*3);
        cv::waitKey(0);
    }
    catch(const cv::Exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught cv::Exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(const std::exception& e) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught std::exception:\n" << e.what() << "\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    catch(...) {std::cout << "\n!!!!!!!!!!!!!!\nTop level caught unhandled exception\n!!!!!!!!!!!!!!\n" << std::endl; return 1;}
    std::cout << "\n[" << lv::getTimeStamp() << "]\n" << std::endl;
    std::cout << "All done." << std::endl;
    return 0;
}
