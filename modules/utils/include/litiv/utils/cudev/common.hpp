
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

#pragma once

#include <array>
#include <cstdio>
#include <sstream>
#include <cassert>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/limits.hpp>
#include <opencv2/cudev.hpp>

#ifdef CUDA_EXIT_ON_ERROR
#define CUDA_ERROR_HANDLER(errn,msg) do { printf("%s",msg); std::exit(errn); } while(false)
#else //ndef(CUDA_EXIT_ON_ERROR)
#define CUDA_ERROR_HANDLER(errn,msg) do { (void)errn; throw std::runtime_error(msg); } while(false)
#endif //ndef(CUDA_..._ON_ERROR)

#define cudaKernelWrap(func,kparams,...) do { \
        impl::func<<<kparams.vGridSize,kparams.vBlockSize,kparams.nSharedMemSize,kparams.pStream>>>(__VA_ARGS__); \
        const cudaError_t __errn = cudaGetLastError(); \
        if(__errn!=cudaSuccess) { \
            std::array<char,1024> acBuffer; \
            snprintf(acBuffer.data(),acBuffer.size(),"cuda kernel '" #func "' execution failed [code=%d, msg=%s]\n\t... in function '%s'\n\t... from %s(%d)\n\t... with kernel params = %s\n", \
                     (int)__errn,cudaGetErrorString(__errn),__PRETTY_FUNCTION__,__FILE__,__LINE__,kparams.str().c_str()); \
            CUDA_ERROR_HANDLER((int)__errn,acBuffer.data()); \
        } \
    } while(false)

#define cudaErrorCheck do { \
        const cudaError_t __errn = cudaGetLastError(); \
        if(__errn!=cudaSuccess) { \
            std::array<char,1024> acBuffer; \
            snprintf(acBuffer.data(),acBuffer.size(),"cudaErrorCheck failed [code=%d, msg=%s]\n\t... in function '%s'\n\t... from %s(%d)\n", \
                     (int)__errn,cudaGetErrorString(__errn),__PRETTY_FUNCTION__,__FILE__,__LINE__); \
            CUDA_ERROR_HANDLER((int)__errn,acBuffer.data()); \
        } \
    } while(false)

#define cudaErrorCheck_(test) do { \
        const cudaError_t __errn = test; \
        if(__errn!=cudaSuccess) { \
            std::array<char,1024> acBuffer; \
            snprintf(acBuffer.data(),acBuffer.size(),"cudaErrorCheck failed [code=%d, msg=%s]\n\t... in function '%s'\n\t... from %s(%d)\n", \
                     (int)__errn,cudaGetErrorString(__errn),__PRETTY_FUNCTION__,__FILE__,__LINE__); \
            CUDA_ERROR_HANDLER((int)__errn,acBuffer.data()); \
        } \
    } while(false)
