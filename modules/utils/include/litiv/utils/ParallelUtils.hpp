#pragma once

#define HAVE_GPU_SUPPORT    0
#define HAVE_SIMD_SUPPORT   1
#define DEFAULT_NB_THREADS  1

#if HAVE_GPU_SUPPORT
#define HAVE_GLSL           1
#define HAVE_CUDA           0
#if (HAVE_GLSL+HAVE_CUDA)!=1
#error "GPUUtils: must pick a single GPU impl approach."
#endif //(HAVE_GLSL+HAVE_CUDA)!=1
#endif //HAVE_GPU_SUPPORT
#if DEFAULT_NB_THREADS<1
#error "Bad default number of threads specified."
#endif //DEFAULT_NB_THREADS<1
#if HAVE_GLSL
#define GLSL_RENDERING      0
#include "litiv/utils/GLUtils.hpp"
#elif HAVE_CUDA
#include @@@@@
#endif //HAVE_CUDA
#if HAVE_SIMD_SUPPORT
#include <opencv2/core.hpp>
#define HAVE_MMX 1
#define HAVE_SSE 1
#define HAVE_SSE2 1
#define HAVE_SSE3 1
#define HAVE_SSSE3 1
#define HAVE_SSE4_1 1
#define HAVE_SSE4_2 1
#define HAVE_POPCNT 1
#define HAVE_AVX 1
#define HAVE_AVX2 1
#define HAVE_CV_MMX cv::checkHardwareSupport(cv::CPU_MMX)
#define HAVE_CV_SSE cv::checkHardwareSupport(cv::CPU_SSE)
#define HAVE_CV_SSE2 cv::checkHardwareSupport(cv::CPU_SSE2)
#define HAVE_CV_SSE3 cv::checkHardwareSupport(cv::CPU_SSE3)
#define HAVE_CV_SSSE3 cv::checkHardwareSupport(cv::CPU_SSSE3)
#define HAVE_CV_SSE4_1 cv::checkHardwareSupport(cv::CPU_SSE4_1)
#define HAVE_CV_SSE4_2 cv::checkHardwareSupport(cv::CPU_SSE4_2)
#define HAVE_CV_POPCNT cv::checkHardwareSupport(cv::CPU_POPCNT)
#define HAVE_CV_AVX cv::checkHardwareSupport(cv::CPU_AVX)
#define HAVE_CV_AVX2 cv::checkHardwareSupport(cv::CPU_AVX2)
namespace ParallelUtils {
    static inline void checkSIMDSupport() {
        CV_Assert(!HAVE_MMX || HAVE_CV_MMX);
        CV_Assert(!HAVE_SSE || HAVE_CV_SSE);
        CV_Assert(!HAVE_SSE2 || HAVE_CV_SSE2);
        CV_Assert(!HAVE_SSE3 || HAVE_CV_SSE3);
        CV_Assert(!HAVE_SSSE3 || HAVE_CV_SSSE3);
        CV_Assert(!HAVE_SSE4_1 || HAVE_CV_SSE4_1);
        CV_Assert(!HAVE_SSE4_2 || HAVE_CV_SSE4_2);
        CV_Assert(!HAVE_POPCNT || HAVE_CV_POPCNT);
        CV_Assert(!HAVE_AVX || HAVE_CV_AVX);
        CV_Assert(!HAVE_AVX2 || HAVE_CV_AVX2);
    }
}; //namespace ParallelUtils
#if defined(_MSC_VER)
#include <intrin.h>
#else //!defined(_MSC_VER)
#include <x86intrin.h>
#endif //!defined(_MSC_VER)
#endif //HAVE_SIMD_SUPPORT
