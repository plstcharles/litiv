
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/imgproc.hpp"
#include "litiv/utils/cxx.hpp"

void thinning_internal_ZhangSuen(cv::Mat& oInput, cv::Mat& oTempMarker, bool bIter) {
    oTempMarker.create(oInput.size(),CV_8UC1);
    oTempMarker = cv::Scalar_<uchar>(0);

    const uchar* pAbove = nullptr;
    const uchar* pCurr = oInput.ptr<uchar>(0);
    const uchar* pBelow = oInput.ptr<uchar>(1);
    const uchar* nw, *no, *ne;
    const uchar* we, *me, *ea;
    const uchar* sw, *so, *se;

    for(int y=1; y<oInput.rows-1; ++y) {
        // shift the rows up by one
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = oInput.ptr<uchar>(y+1);
        uchar* pDst = oTempMarker.ptr<uchar>(y);

        // initialize col pointers
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);

        for(int x=1; x<oInput.cols-1; ++x) {
            // shift col pointers left by one (scan left to right)
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);

            int A  = (!*no && *ne>0) + (!*ne && *ea>0) +
                     (!*ea && *se>0) + (!*se && *so>0) +
                     (!*so && *sw>0) + (!*sw && *we>0) +
                     (!*we && *nw>0) + (!*nw && *no>0);
            int B  = (*no>0)+(*ne>0)+(*ea>0)+(*se>0)+(*so>0)+(*sw>0)+(*we>0)+(*nw>0);
            int m1 = !bIter?((*no>0)*(*ea>0)*(*so>0)):((*no>0)*(*ea>0)*(*we>0));
            int m2 = !bIter?((*ea>0)*(*so>0)*(*we>0)):((*no>0)*(*so>0)*(*we>0));
            if(A==1 && B>=2 && B<=6 && !m1 && !m2)
                pDst[x] = UCHAR_MAX;
        }
    }
    oInput &= ~oTempMarker;
}

void thinning_internal_LamLeeSuen(cv::Mat& oInput, bool bIter) {
    for(int i=1; i<oInput.rows-1; ++i) {
        for(int j=1; j<oInput.cols-1; ++j) {
            if(!oInput.at<uchar>(i,j))
                continue;
            const std::array<uchar,8> anLUT{
                oInput.at<uchar>(i+1,j  ),
                oInput.at<uchar>(i+1,j-1),
                oInput.at<uchar>(i  ,j-1),
                oInput.at<uchar>(i-1,j-1),
                oInput.at<uchar>(i-1,j  ),
                oInput.at<uchar>(i-1,j+1),
                oInput.at<uchar>(i  ,j+1),
                oInput.at<uchar>(i+1,j+1)
            };
            size_t x_h = 0, n1 = 0, n2 = 0;
            for(size_t k=0; k<4; ++k) {
#if defined(_MSC_VER) || USE_IMGPROC_THINNING_MATLAB_IMPL_FIX
                // G1:
                x_h += bool(!anLUT[2*(k+1)-2] && (anLUT[2*(k+1)-1] || anLUT[(2*(k+1))%8]));
                // G2:
                n1 += bool(anLUT[2*(k+1)-2] || anLUT[2*(k+1)-1]);
                n2 += bool(anLUT[2*(k+1)-1] || anLUT[(2*(k+1))%8]);
#elif defined(__clang__)
#pragma clang diagnostic push
// but no warnings to push...?
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif //(defined(__GNUC__) || defined(__GNUG__))
                // G1:
                x_h += bool(!anLUT[2*(k+1)-2] && (anLUT[2*(k+1)-1] || anLUT[2*(k+1)]));
                // G2:
                n1 += bool(anLUT[2*(k+1)-2] || anLUT[2*(k+1)-1]);
                n2 += bool(anLUT[2*(k+1)-1] || anLUT[2*(k+1)]);
#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#endif //(defined(__GNUC__) || defined(__GNUG__))
            }
            size_t n_min = std::min(n1,n2);
            if(x_h==1 && n_min>=2 && n_min<=3) {
                // G3 || G3' :
                if( (!bIter && !((anLUT[1] || anLUT[2] || !anLUT[7]) && anLUT[0])) ||
                    (bIter && !((anLUT[5] || anLUT[6] || !anLUT[3]) && anLUT[4]))) {
                    oInput.at<uchar>(i,j) = 0;
                }
            }
        }
    }
}

void lv::thinning(const cv::Mat& oInput, cv::Mat& oOutput, ThinningMode eMode) {
    lvAssert_(!oInput.empty() && oInput.isContinuous(),"input image must be non-empty and continuous");
    lvAssert_(oInput.type()==CV_8UC1,"input image type must be 8UC1");
    lvAssert_(oInput.rows>3 && oInput.cols>3,"input image size must be greater than 3x3");
    oOutput.create(oInput.size(),CV_8UC1);
    oInput.copyTo(oOutput);
    cv::Mat oPrevious(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    cv::Mat oTempMarker;
    bool bEq;
    do {
        if(eMode==ThinningMode_ZhangSuen) {
            thinning_internal_ZhangSuen(oOutput,oTempMarker,false);
            thinning_internal_ZhangSuen(oOutput,oTempMarker,true);
        }
        else { //eMode==ThinningMode_LamLeeSuen
            thinning_internal_LamLeeSuen(oOutput,false);
            thinning_internal_LamLeeSuen(oOutput,true);
        }
        bEq = std::equal(oOutput.begin<uchar>(),oOutput.end<uchar>(),oPrevious.begin<uchar>());
        oOutput.copyTo(oPrevious);
    }
    while(!bEq);
}
