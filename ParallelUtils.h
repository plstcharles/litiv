#pragma once

#define HAVE_GPU_SUPPORT    1
#define DEFAULT_NB_THREADS  1

#if HAVE_GPU_SUPPORT
#define HAVE_GLSL           1
#define HAVE_CUDA           0
#if (HAVE_GLSL+HAVE_CUDA)!=1
#error "GPUUtils: must pick a single GPU impl approach."
#endif //(HAVE_GLSL+HAVE_CUDA)!=1
#define GPU_RENDERING       0
#define GPU_EVALUATION      1
#define ASYNC_PROCESS       1
#endif //HAVE_GPU_SUPPORT
#if DEFAULT_NB_THREADS<1
#error "Bad default number of threads specified."
#endif //DEFAULT_NB_THREADS<1
#if HAVE_GLSL
#include "GLUtils.h"
#elif HAVE_CUDA
#include @@@@@
#endif //HAVE_CUDA
