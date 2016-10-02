
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

#include "litiv/utils/defines.hpp"
#include "litiv/utils/cxx.hpp"
#include "litiv/utils/parallel.hpp"
#include "litiv/utils/distances.hpp"
#include "litiv/utils/platform.hpp"
#include "litiv/utils/console.hpp"
#include "litiv/utils/opencv.hpp"
#if HAVE_GLSL
#include "litiv/utils/opengl.hpp"
#include "litiv/utils/opengl-draw.hpp"
#include "litiv/utils/opengl-shaders.hpp"
#include "litiv/utils/opengl-imgproc.hpp"
#endif //HAVE_GLSL
#if HAVE_CUDA
// ...
#endif //HAVE_CUDA
#if HAVE_OPENCL
// ...
#endif //HAVE_OPENCL
