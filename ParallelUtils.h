#pragma once

#define HAVE_GPU_SUPPORT    1
// cpu:  highway : Rcl=0.8707 Prc=0.9197 FM=0.8945 MCC=0.8885
// nopp: highway : Rcl=0.8279 Prc=0.8327 FM=0.8303 MCC=0.8197
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
#define GLSL_RENDERING      1
#include "GLUtils.h"
#elif HAVE_CUDA
#include @@@@@
#endif //HAVE_CUDA
