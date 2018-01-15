
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

option(USE_OPENCV_x264_TEST "Use internal unit test to see if x264-encoded videos can be parsed by ffmpeg" OFF)
option(USE_SIGNEXT_SHIFT_TRICK "Use sign-extended right shift bit trick in float absolute value computation" OFF)
option(USE_FAST_SQRT_FOR_CDIST "Use fast approximate square root inside floating point version of color distortion" OFF)
option(USE_RLUTIL_ANSI_DEFINE "Use ANSI escape sequences also on Windows in console utils (defaults to using WinAPI otherwise)" OFF)
set(RLUTIL_STRING_TYPE "const char*" CACHE STRING "Preferred string type for console utils (defaults to std::string otherwise)")

mark_as_advanced(
    USE_OPENCV_x264_TEST
    USE_SIGNEXT_SHIFT_TRICK
    USE_FAST_SQRT_FOR_CDIST
    USE_RLUTIL_ANSI_DEFINE
    RLUTIL_STRING_TYPE
)

set_eval(USE_OPENCV_MAT_CONSTR_FIX ((${OpenCV_VERSION_MAJOR} LESS 4) AND (${OpenCV_VERSION_MINOR} LESS 2)))
file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/defines.hpp")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/utils/defines.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/defines.hpp"
)

add_files(SOURCE_FILES
    "src/cxx.cpp"
    "src/platform.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/utils/algo.hpp"
    "include/litiv/utils/console.hpp"
    "include/litiv/utils/cxx.hpp"
    "include/litiv/utils/defines.hpp"
    "include/litiv/utils/kinect.hpp"
    "include/litiv/utils/math.hpp"
    "include/litiv/utils/platform.hpp"
    "include/litiv/utils/simd.hpp"
    "include/litiv/utils.hpp"
)
if(USE_CVCORE_WITH_UTILS)
    add_files(SOURCE_FILES
        "src/opencv.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/utils/opencv.hpp"
    )
    if(USE_GLSL)
        add_files(SOURCE_FILES
            "src/opengl-imgproc.cpp"
            "src/opengl-shaders.cpp"
            "src/opengl-draw.cpp"
            "src/opengl.cpp"
        )
        add_files(INCLUDE_FILES
            "include/litiv/utils/opengl-imgproc.hpp"
            "include/litiv/utils/opengl-draw.hpp"
            "include/litiv/utils/opengl-shaders.hpp"
            "include/litiv/utils/opengl.hpp"
        )
    endif(USE_GLSL)
    if(USE_OPENGM)
        add_files(INCLUDE_FILES
            "include/litiv/utils/opengm.hpp"
        )
    endif(USE_OPENGM)
endif()
if(USE_CUDA)
    add_files(SOURCE_FILES
        "src/cuda.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/utils/cuda.hpp"
    )
    add_files(CUDA_SOURCE_FILES
        "cuda/test.cu"
    )
    add_files(CUDA_INCLUDE_FILES
        "cuda/test.cuh"
    )
endif()

litiv_module(utils
    SOURCE_FILES
    INCLUDE_FILES
    CUDA_SOURCE_FILES
    CUDA_INCLUDE_FILES
)

if(USE_CVCORE_WITH_UTILS)
    # opencv include dirs auto-added due to proper packaging
    target_link_libraries(litiv_utils
        PUBLIC
            "${OpenCV_LIBS}"
    )
    if(USE_GLSL)
        target_include_directories(litiv_utils
            PUBLIC
                "$<BUILD_INTERFACE:${OpenGL_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${GLEW_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>"
        )
        target_link_libraries(litiv_utils
            PUBLIC
                "${OPENGL_LIBRARIES}"
                "${GLEW_LIBRARIES}"
                "${GLM_LIBRARIES}"
        )
        if(USE_GLFW)
            target_include_directories(litiv_utils
                PUBLIC
                    "$<BUILD_INTERFACE:${GLFW_INCLUDE_DIR}>"
            )
            target_link_libraries(litiv_utils
                PUBLIC
                    "${GLFW_LIBRARIES}"
            )
        elseif(USE_FREEGLUT)
            target_include_directories(litiv_utils
                PUBLIC
                    "$<BUILD_INTERFACE:${FREEGLUT_INCLUDE_DIR}>"
            )
            target_link_libraries(litiv_utils
                PUBLIC
                    "${FREEGLUT_LIBRARY}"
            )
        endif()
    endif()
endif()
if(USE_CUDA)
    target_include_directories(litiv_utils
        PUBLIC
            "$<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>"
    )
    target_link_libraries(litiv_utils
        PUBLIC
            "${CUDA_LIBRARIES}"
    )
endif()
if(KinectSDK2_FOUND)
    target_include_directories(litiv_utils
        PUBLIC
            "$<BUILD_INTERFACE:${KinectSDK2_INCLUDE_DIRS}>"
    )
    target_link_libraries(litiv_utils
        PUBLIC
            "${KinectSDK2_LIBRARIES}"
    )
endif()
if(USE_OPENGM)
    target_include_directories(litiv_utils
        PUBLIC
            "$<BUILD_INTERFACE:${OpenGM_INCLUDE_DIRS}>"
    )
    target_link_libraries(litiv_utils
        PUBLIC
            "${OpenGM_LIBRARIES}"
    )
endif()
if(USE_LZ4)
	target_link_libraries(litiv_utils PRIVATE litiv_3rdparty_lz4)
endif()
if(UNIX AND NOT ANDROID AND NOT (APPLE AND CMAKE_COMPILER_IS_CLANGCXX))
    # build fails without this for some platforms... broken external dependency? (see PR#5)
    target_link_libraries(litiv_utils
        PUBLIC
            "pthread"
    )
endif()