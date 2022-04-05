
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

#find_package(Eigen3)
if(Eigen3_FOUND)
    return()
endif()

if(CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

if((NOT DEFINED eigen3_SOURCE_DIR) OR (NOT EXISTS ${eigen3_SOURCE_DIR}))
    download_project(
        PROJ
            eigen3
        PREFIX
            "${CMAKE_CURRENT_BINARY_DIR}"
        URL
            "https://gitlab.com/libeigen/eigen/-/archive/3.3.4/eigen-3.3.4.zip"
        QUIET
    #    ${UPDATE_DISCONNECTED_IF_AVAILABLE} # bugged in some cmake versions, needs patch
    )
    set(eigen3_SOURCE_DIR ${eigen3_SOURCE_DIR} CACHE INTERNAL "")
    set(eigen3_BINARY_DIR ${eigen3_BINARY_DIR} CACHE INTERNAL "")
endif()

set(EIGEN3_FOUND ON CACHE BOOL "")
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/eigen3-src" CACHE PATH "")
set(EIGEN_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/eigen3-src" CACHE PATH "")
set(EIGEN_VERSION "3.3.4" CACHE STRING "")
mark_as_advanced(EIGEN3_FOUND EIGEN_INCLUDE_DIR EIGEN_INCLUDE_DIRS EIGEN_VERSION)
