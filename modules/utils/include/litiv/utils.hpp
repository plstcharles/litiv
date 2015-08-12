#pragma once

#include "litiv/utils/DefineUtils.hpp"
#include "litiv/utils/CxxUtils.hpp"
#include "litiv/utils/ParallelUtils.hpp"
#include "litiv/utils/DatasetUtils.hpp"
#include "litiv/utils/DatasetEvalUtils.hpp"
#include "litiv/utils/DistanceUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"
#include "litiv/utils/RandUtils.hpp"
#if HAVE_GLSL
#include "litiv/utils/GLUtils.hpp"
#include "litiv/utils/GLDrawUtils.hpp"
#include "litiv/utils/GLShaderUtils.hpp"
#include "litiv/utils/GLImageProcUtils.hpp"
#endif //HAVE_GLSL
#if HAVE_CUDA
// ... @@@
#endif //HAVE_CUDA
#if HAVE_OPENCL
// ... @@@
#endif //HAVE_OPENCL
