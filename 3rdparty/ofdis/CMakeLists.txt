
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2017 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

if(NOT USE_3RDPARTY_OPTFLOW_KROEGER2016)
    return()
endif()

if(NOT USE_SSE4_2)
    message(FATAL_ERROR "ofdis 3rdparty module requires sse4")
endif()

if((NOT DEFINED $ENV{EIGEN3_ROOT}) OR (NOT $ENV{EIGEN3_ROOT}) OR (NOT EXISTS $ENV{EIGEN3_ROOT}))
    set(ENV{EIGEN3_ROOT} "/store/pistc/litiv/build/3rdparty/eigen3/eigen3-src/")
endif()

find_package(Eigen3 REQUIRED)
mark_as_advanced(Eigen3_DIR)

add_files(SOURCE_FILES
    "src/fdf/image.c"
    "src/fdf/opticalflow_aux.c"
    "src/fdf/solver.c"
    "src/oflow.cpp"
    "src/patch.cpp"
    "src/patchgrid.cpp"
    "src/refine_variational.cpp"
    "src/ofdis.cpp"
)

add_files(INCLUDE_FILES
    "include/litiv/3rdparty/ofdis/fdf/image.h"
    "include/litiv/3rdparty/ofdis/fdf/opticalflow_aux.h"
    "include/litiv/3rdparty/ofdis/fdf/solver.h"
    "include/litiv/3rdparty/ofdis/oflow.hpp"
    "include/litiv/3rdparty/ofdis/patch.hpp"
    "include/litiv/3rdparty/ofdis/patchgrid.hpp"
    "include/litiv/3rdparty/ofdis/refine_variational.hpp"
    "include/litiv/3rdparty/ofdis/ofdis.hpp"
)

litiv_3rdparty_module(ofdis SOURCE_FILES INCLUDE_FILES)

target_include_directories(litiv_3rdparty_ofdis
    PUBLIC
        "$<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>"
    PRIVATE
        # to use define utils internally (i.e. to toggle openmp)
        "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/modules/utils/include>"
        "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/utils/include>"
)
target_link_libraries(litiv_3rdparty_ofdis
    PUBLIC
        "${OpenCV_LIBS}"
)
