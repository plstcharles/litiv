
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

#pragma once

#include "litiv/imgproc/EdgeDetectorCanny.hpp"
#include "litiv/imgproc/EdgeDetectorLBSP.hpp"

namespace litiv {

    enum eThinningMode {
        eThinningMode_ZhangSuen=0,
        eThinningMode_LamLeeSuen
    };

    //! 'thins' the provided image (currently only works on 1ch 8UC1 images, treated as binary)
    void thinning(const cv::Mat& oInput, cv::Mat& oOutput, eThinningMode eMode=eThinningMode_LamLeeSuen);

    //! performs non-maximum suppression on the input image, with a (nWinSize)x(nWinSize) window
    template<int nWinSize>
    void nonMaxSuppression(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oMask=cv::Mat());

    template<int aperture_size, bool L2gradient=false>
    void cv_canny(cv::InputArray _src, cv::OutputArray _dst, double low_thresh, double high_thresh);

} //namespace litiv

template<int nWinSize>
void litiv::nonMaxSuppression(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oMask) {
    //  http://code.opencv.org/attachments/994/nms.cpp
    //  Copyright (c) 2012, Willow Garage, Inc.
    //  All rights reserved.
    //
    //  Redistribution and use in source and binary forms, with or without
    //  modification, are permitted provided that the following conditions
    //  are met:
    //
    //   * Redistributions of source code must retain the above copyright
    //     notice, this list of conditions and the following disclaimer.
    //   * Redistributions in binary form must reproduce the above
    //     copyright notice, this list of conditions and the following
    //     disclaimer in the documentation and/or other materials provided
    //     with the distribution.
    //   * Neither the name of Willow Garage, Inc. nor the names of its
    //     contributors may be used to endorse or promote products derived
    //     from this software without specific prior written permission.
    //
    //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    //  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    //  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    //  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    //  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    //  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    //  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    //  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    //  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    //  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    //  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    //  POSSIBILITY OF SUCH DAMAGE.
    CV_Assert(oInput.channels()==1);
    // initialise the block oMask and destination
    const int M = oInput.rows;
    const int N = oInput.cols;
    const bool masked = !oMask.empty();
    cv::Mat block = 255*cv::Mat_<uint8_t>::ones(cv::Size(2*nWinSize+1,2*nWinSize+1));
    oOutput = cv::Mat_<uint8_t>::zeros(oInput.size());
    // iterate over image blocks
    for(int m = 0; m < M; m+=nWinSize+1) {
        for(int n = 0; n < N; n+=nWinSize+1) {
            cv::Point  ijmax;
            double vcmax, vnmax;
            // get the maximal candidate within the block
            cv::Range ic(m,std::min(m+nWinSize+1,M));
            cv::Range jc(n,std::min(n+nWinSize+1,N));
            cv::minMaxLoc(oInput(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? oMask(ic,jc) : cv::noArray());
            cv::Point cc = ijmax + cv::Point(jc.start,ic.start);
            // search the neighbours centered around the candidate for the true maxima
            cv::Range in(std::max(cc.y-nWinSize,0),std::min(cc.y+nWinSize+1,M));
            cv::Range jn(std::max(cc.x-nWinSize,0),std::min(cc.x+nWinSize+1,N));
            // mask out the block whose maxima we already know
            cv::Mat_<uint8_t> blockmask;
            block(cv::Range(0,in.size()),cv::Range(0,jn.size())).copyTo(blockmask);
            cv::Range iis(ic.start-in.start,std::min(ic.start-in.start+nWinSize+1, in.size()));
            cv::Range jis(jc.start-jn.start,std::min(jc.start-jn.start+nWinSize+1, jn.size()));
            blockmask(iis, jis) = cv::Mat_<uint8_t>::zeros(cv::Size(jis.size(),iis.size()));
            cv::minMaxLoc(oInput(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? oMask(in,jn).mul(blockmask) : blockmask);
            //cv::Point cn = ijmax + cv::Point(jn.start, in.start);
            // if the block centre is also the neighbour centre, then it's a local maxima
            if(vcmax > vnmax)
                oOutput.at<uint8_t>(cc.y, cc.x) = 255;
        }
    }
}

//
// @@@@@@ code below is currently a copy of opencv's canny impl (need to extract good bits)
// see https://github.com/Itseez/opencv for copyright notice & license info
//
// note: canny nms always winsize=3
//
template<int aperture_size, bool L2gradient>
void litiv::cv_canny(cv::InputArray _src, cv::OutputArray _dst, double low_thresh, double high_thresh) {
    static_assert(!((aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7))),"Aperture size should be odd");
    const int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    const cv::Size size = _src.size();

    CV_Assert( depth == CV_8U );
    _dst.create(size, CV_8U);

    if(low_thresh > high_thresh)
        std::swap(low_thresh, high_thresh);

    cv::Mat src = _src.getMat(), dst = _dst.getMat();

    cv::Mat dx(src.rows, src.cols, CV_16SC(cn));
    cv::Mat dy(src.rows, src.cols, CV_16SC(cn));

    cv::Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, cv::BORDER_REPLICATE);
    cv::Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, cv::BORDER_REPLICATE);

    /*cv::Mat dxa,dya;
    cv::convertScaleAbs(dx,dxa);
    cv::convertScaleAbs(dy,dya);
    cv::Mat test_grad = (cv::abs(dxa)/2+cv::abs(dya)/2);
    std::vector<cv::Mat> votest;
    cv::split(test_grad,votest);
    cv::max(votest[0],votest[1],votest[0]);
    cv::max(votest[0],votest[2],votest[0]);
    cv::normalize(votest[0],votest[0],0,255,cv::NORM_MINMAX);
    cv::imshow("votest[0]",votest[0]);*/

    if(L2gradient) {
        low_thresh = std::min(32767.0, low_thresh);
        high_thresh = std::min(32767.0, high_thresh);
        if(low_thresh > 0) low_thresh *= low_thresh;
        if(high_thresh > 0) high_thresh *= high_thresh;
    }
    int low = cvFloor(low_thresh);
    int high = cvFloor(high_thresh);

    ptrdiff_t mapstep = src.cols + 2;
    cv::AutoBuffer<uchar> buffer((src.cols+2)*(src.rows+2) + cn * mapstep * 3 * sizeof(int));
    // buffer contains 3x lines of cn*(img width + 2) in int, and a full img map + 1px border in uchar

    int* mag_buf[3]; // the pointers to the 3 lines of cn*(img width + 2) in int (ring buffer)
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep*cn;
    mag_buf[2] = mag_buf[1] + mapstep*cn;
    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int)); // initializes 1/cn-th of the first int line to zero

    uchar* map = (uchar*)(mag_buf[2] + mapstep*cn); // the pointer to the first uchar map element
    memset(map, 1, mapstep); // initializes the first map row to 1 (i.e. pixels cannot belong to an edge)
    memset(map + mapstep*(src.rows + 1), 1, mapstep); // initializes the last map row to 1 (i.e. pixels cannot belong to an edge)

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
    std::vector<uchar*> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
       (Top-Left Origin)

        1   2   3
         *  *  *
          * * *
        0*******0
          * * *
         *  *  *
        3   2   1
    */

#define CANNY_PUSH(d)    *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d)     (d) = *--stack_top

    // calculate magnitude and angle of gradient, perform non-maxima suppression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; ++i) {
        int* _norm = mag_buf[(i > 0) + 1] + 1;
        if(i < src.rows) {
            // fills the next magnitude row in mag_buf[2] (or fills mag_buf[1] if first iter)
            short* _dx = dx.ptr<short>(i);
            short* _dy = dy.ptr<short>(i);
            if(!L2gradient) {
                int j = 0, width = src.cols * cn;
#if HAVE_SSE2
                __m128i v_zero = _mm_setzero_si128();
                for (; j <= width - 8; j += 8) {
                    __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                    __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));
                    v_dx = _mm_max_epi16(v_dx, _mm_sub_epi16(v_zero, v_dx));
                    v_dy = _mm_max_epi16(v_dy, _mm_sub_epi16(v_zero, v_dy));

                    __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx, v_zero), _mm_unpacklo_epi16(v_dy, v_zero));
                    _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                    v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx, v_zero), _mm_unpackhi_epi16(v_dy, v_zero));
                    _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                }
#endif //HAVE_SSE2
                for (; j < width; ++j)
                    _norm[j] = std::abs(int(_dx[j])) + std::abs(int(_dy[j]));
            }
            else
            {
                int j = 0, width = src.cols * cn;
#if HAVE_SSE2
                for (; j <= width - 8; j += 8) {
                    __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                    __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                    __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx), v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                    __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy), v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                    __m128i v_norm = _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh), _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                    v_norm = _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh), _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                }
#endif //HAVE_SSE2
                for (; j < width; ++j)
                    _norm[j] = int(_dx[j])*_dx[j] + int(_dy[j])*_dy[j];
            }

            if(cn > 1)
            {
                for(int j = 0, jn = 0; j < src.cols; ++j, jn += cn)
                {
                    int maxIdx = jn;
                    for(int k = 1; k < cn; ++k)
                        if(_norm[jn + k] > _norm[maxIdx]) maxIdx = jn + k;
                    _norm[j] = _norm[maxIdx];
                    _dx[j] = _dx[maxIdx];
                    _dy[j] = _dy[maxIdx];
                }
            }
            _norm[-1] = _norm[src.cols] = 0;
        }
        else
            memset(_norm-1, 0, /* cn* */mapstep*sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if(i>0) {
            // if not first iter, then mag_buf[0] is before-last, mag_buf[1] is last, and mag_buf[2] was just filled
            // orientation analysis is done for magnitude values contained in mag_buf[1] (= defined 3x3 neighborhood)
            uchar* _map = map+mapstep*i+1;
            _map[-1] = _map[src.cols] = 1; // border pixels cannot belong to an edge

            const int* _mag = mag_buf[1]+1; // take the central row
            ptrdiff_t magstep_down = mag_buf[2]-mag_buf[1];
            ptrdiff_t magstep_up = mag_buf[0]-mag_buf[1];

            const short* _x = dx.ptr<short>(i-1);
            const short* _y = dy.ptr<short>(i-1);

            if((stack_top-stack_bottom)+src.cols>maxsize) {
                int sz = (int)(stack_top-stack_bottom);
                maxsize = std::max(maxsize*3/2,sz+src.cols);
                stack.resize(maxsize);
                stack_bottom = &stack[0];
                stack_top = stack_bottom+sz;
            }

            int prev_flag = 0;
            for(int j = 0; j<src.cols; j++) {
#define CANNY_SHIFT 15
                const int TG22 = (int)(0.4142135623730950488016887242097*(1 << CANNY_SHIFT)+0.5); // TG22 = tan(pi/8)

                const int m = _mag[j];

                if(m>low) {
                    int xs = _x[j];
                    int ys = _y[j];
                    int x = std::abs(xs);
                    int y = std::abs(ys) << CANNY_SHIFT;

                    int tg22x = x*TG22; // tg22x = 0.4142135623730950488016887242097*std::abs(xs)

                    if(y<tg22x) { // if(std::abs(ys) < 0.4142135623730950488016887242097*std::abs(xs))
                        // flat gradient (sector 0)
                        if(m>_mag[j-1] && m>=_mag[j+1]) // if current magnitude is a peak among neighbors
                            goto __ocv_canny_push; // push as 'edge'
                    }
                    else { // else(std::abs(ys) >= 0.4142135623730950488016887242097*std::abs(xs))
                        // not a flat gradient (sectors 1, 2 or 3)
                        int tg67x = tg22x+(x << (CANNY_SHIFT+1)); // tg67x = 2.4142135623730950488016887242097*std::abs(xs) = tan(3*pi/8)*std::abs(xs)
                        if(y>tg67x) { // if(std::abs(ys) > 2.4142135623730950488016887242097*std::abs(xs)
                            // vertical gradient (sector 2)
                            if(m>_mag[j+magstep_up] && m>=_mag[j+magstep_down])
                                goto __ocv_canny_push;
                        }
                        else { // else(std::abs(ys) <= 2.4142135623730950488016887242097*std::abs(xs)
                            // diagonal gradient (sector sign(xs!=ys)?3:1)
                            int s = (xs^ys)<0?-1:1; //int s = (std::signbit(xs)!=std::signbit(ys))?-1:1;
                            if(m>_mag[j+magstep_up-s] && m>_mag[j+magstep_down+s])
                                goto __ocv_canny_push;
                        }
                    }
                }
                prev_flag = 0;
                _map[j] = uchar(1); // cannot belong to an edge (gradmag is below min threshold)
                continue;
                __ocv_canny_push:
                if(!prev_flag && m>high && _map[j-mapstep]!=2) { // if not neighbor to identified edge (top/left), and gradmag above max threshold
                    CANNY_PUSH(_map+j); // then pixel belongs to an edge
                    prev_flag = 1; // set for neighbor check
                }
                else
                    _map[j] = 0; // might belong to an edge (gradmag is local maximum above min threshold)
            }

            // scroll the ring buffer
            int* const _mag_tmp = mag_buf[0];
            mag_buf[0] = mag_buf[1];
            mag_buf[1] = mag_buf[2];
            mag_buf[2] = _mag_tmp;
        }
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom)
    {
        uchar* m;
        if((stack_top - stack_bottom) + 8 > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3/2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if(!m[-1])         CANNY_PUSH(m - 1);
        if(!m[1])          CANNY_PUSH(m + 1);
        if(!m[-mapstep-1]) CANNY_PUSH(m - mapstep - 1);
        if(!m[-mapstep])   CANNY_PUSH(m - mapstep);
        if(!m[-mapstep+1]) CANNY_PUSH(m - mapstep + 1);
        if(!m[mapstep-1])  CANNY_PUSH(m + mapstep - 1);
        if(!m[mapstep])    CANNY_PUSH(m + mapstep);
        if(!m[mapstep+1])  CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image
    const uchar* pmap = map + mapstep + 1;
    uchar* pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step)
    {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar)-(pmap[j] >> 1);
    }
}
