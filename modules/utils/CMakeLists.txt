
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

option(USE_SIGNEXT_SHIFT_TRICK "Use sign-extended right shift bit trick in float absolute value computation" OFF)

mark_as_advanced(
    USE_SIGNEXT_SHIFT_TRICK
)

file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/defines.hpp")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/utils/defines.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/defines.hpp"
)

add_files(SOURCE_FILES
    "src/platform.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/utils/algo.hpp"
    "include/litiv/utils/console.hpp"
    "include/litiv/utils/cxx.hpp"
    "include/litiv/utils/defines.hpp"
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
endif()

litiv_module(utils SOURCE_FILES INCLUDE_FILES)

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
if(UNIX AND NOT ANDROID AND NOT (APPLE AND CMAKE_COMPILER_IS_CLANGCXX))
    # build fails without this for some platforms... broken external dependency? (see PR#5)
    target_link_libraries(litiv_utils
        PUBLIC
            "pthread"
    )
endif()