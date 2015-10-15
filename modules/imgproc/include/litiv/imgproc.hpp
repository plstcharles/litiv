
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


/*! @brief suppress non-maximal values
 *
 * Taken from http://code.opencv.org/attachments/994/nms.cpp
 * (see copyright notice and licensing information there)
 *
 * nonMaximaSuppression produces a oMask (oOutput) such that every non-zero
 * value of the oMask corresponds to a local maxima of oInput. The criteria
 * for local maxima is as follows:
 *
 * 	For every possible (nWinSize x nWinSize) region within oInput, an element is a
 * 	local maxima of oInput iff it is strictly greater than all other elements
 * 	of windows which intersect the given element
 *
 * Intuitively, this means that all maxima must be at least nWinSize+1 pixels
 * apart, though the spacing may be greater
 *
 * A gradient image or a constant image has no local maxima by the definition
 * given above
 *
 * The method is derived from the following paper:
 * A. Neubeck and L. Van Gool. "Efficient Non-Maximum Suppression," ICPR 2006
 *
 * Example:
 * \code
 * 	// create a random test image
 * 	Mat random(Size(2000,2000), DataType<float>::type);
 * 	randn(random, 1, 1);
 *
 * 	// only look for local maxima above the value of 1
 * 	Mat oMask = (random > 1);
 *
 * 	// find the local maxima with a window of 50
 * 	Mat maxima;
 * 	nonMaximaSuppression(random, 50, maxima, oMask);
 *
 * 	// optionally set all non-maxima to zero
 * 	random.setTo(0, maxima == 0);
 * \endcode
 *
 * @param oInput the input image/matrix, of any valid cv type
 * @param nWinSize the size of the window
 * @param oOutput the oMask of type CV_8U, where non-zero elements correspond to
 * local maxima of the oInput
 * @param oMask an input oMask to skip particular elements
 */
template<int nWinSize>
void litiv::nonMaxSuppression(const cv::Mat& oInput, cv::Mat& oOutput, const cv::Mat& oMask) {
    CV_Assert(oInput.channels()==1);

    // initialise the block oMask and destination
    const int M = oInput.rows;
    const int N = oInput.cols;
    const bool masked = !oMask.empty();
    cv::Mat block = 255*cv::Mat_<uint8_t>::ones(cv::Size(2*nWinSize+1,2*nWinSize+1));
    oOutput = cv::Mat_<uint8_t>::zeros(oInput.size());

    // iterate over image blocks
    for (int m = 0; m < M; m+=nWinSize+1) {
        for (int n = 0; n < N; n+=nWinSize+1) {
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

            // oMask out the block whose maxima we already know
            cv::Mat_<uint8_t> blockmask;
            block(cv::Range(0,in.size()),cv::Range(0,jn.size())).copyTo(blockmask);
            cv::Range iis(ic.start-in.start,std::min(ic.start-in.start+nWinSize+1, in.size()));
            cv::Range jis(jc.start-jn.start,std::min(jc.start-jn.start+nWinSize+1, jn.size()));
            blockmask(iis, jis) = cv::Mat_<uint8_t>::zeros(cv::Size(jis.size(),iis.size()));

            cv::minMaxLoc(oInput(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? oMask(in,jn).mul(blockmask) : blockmask);
            //cv::Point cn = ijmax + cv::Point(jn.start, in.start);

            // if the block centre is also the neighbour centre, then it's a local maxima
            if(vcmax > vnmax) {
                oOutput.at<uint8_t>(cc.y, cc.x) = 255;
            }
        }
    }
}

//
// @@@@@@ code below is currently a copy of opencv's canny impl (need to extract good bits)
// see https://github.com/Itseez/opencv for copyright notice & license info
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

    int* mag_buf[3];
    mag_buf[0] = (int*)(uchar*)buffer;
    mag_buf[1] = mag_buf[0] + mapstep*cn;
    mag_buf[2] = mag_buf[1] + mapstep*cn;
    memset(mag_buf[0], 0, /* cn* */mapstep*sizeof(int));

    uchar* map = (uchar*)(mag_buf[2] + mapstep*cn);
    memset(map, 1, mapstep);
    memset(map + mapstep*(src.rows + 1), 1, mapstep);

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
        if(i==0)
            continue;

        uchar* _map = map + mapstep*i + 1;
        _map[-1] = _map[src.cols] = 1;

        int* _mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short* _x = dx.ptr<short>(i-1);
        const short* _y = dy.ptr<short>(i-1);

        if((stack_top - stack_bottom) + src.cols > maxsize)
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3/2, sz + src.cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++)
        {
#define CANNY_SHIFT 15
            const int TG22 = (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5);

            int m = _mag[j];

            if(m > low)
            {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if(y < tg22x)
                {
                    if(m > _mag[j-1] && m >= _mag[j+1]) goto __ocv_canny_push;
                }
                else
                {
                    int tg67x = tg22x + (x << (CANNY_SHIFT+1));
                    if(y > tg67x)
                    {
                        if(m > _mag[j+magstep2] && m >= _mag[j+magstep1]) goto __ocv_canny_push;
                    }
                    else
                    {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if(m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s]) goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
            __ocv_canny_push:
            if(!prev_flag && m > high && _map[j-mapstep] != 2)
            {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            }
            else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
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
