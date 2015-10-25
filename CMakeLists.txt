
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

cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(FrameworkUtils)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#set(CMAKE_VERBOSE_MAKEFILE ON)

project(litiv_framework)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(LITIV_FRAMEWORK_VERSION_MAJOR 1)
set(LITIV_FRAMEWORK_VERSION_MINOR 0)
set(LITIV_FRAMEWORK_VERSION_PATCH 0)
set(LITIV_FRAMEWORK_VERSION ${LITIV_FRAMEWORK_VERSION_MAJOR}.${LITIV_FRAMEWORK_VERSION_MINOR}.${LITIV_FRAMEWORK_VERSION_PATCH})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>)

### OPENCV CHECK
find_package(OpenCV 3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

### GLSL CHECK
find_package(GLFW)
find_package(FREEGLUT)
find_package(OpenGL)
find_package(GLEW)
find_package(GLM)
set_eval(USE_GLSL ((${GLFW_FOUND} OR ${FREEGLUT_FOUND}) AND ${OPENGL_FOUND} AND ${GLEW_FOUND} AND ${GLM_FOUND}))
if(USE_GLSL)
    if(${GLFW_FOUND})
        set(USE_GLFW 1 CACHE BOOL "Use GLFW as the OpenGL window manager")
        set(USE_FREEGLUT 0 CACHE BOOL "Use FREEGLUT as the OpenGL window manager")
    elseif(${FREEGLUT_FOUND})
        set(USE_GLFW 0 CACHE BOOL "Use GLFW as the OpenGL window manager")
        set(USE_FREEGLUT 1 CACHE BOOL "Use FREEGLUT as the OpenGL window manager")
    endif(${GLFW_FOUND})
    if(${USE_GLFW} EQUAL ${USE_FREEGLUT})
        message(FATAL_ERROR "Need to select one window manager.")
    endif(${USE_GLFW} EQUAL ${USE_FREEGLUT})
    if(USE_GLFW)
        include_directories(${GLFW_INCLUDE_DIRS})
    elseif(USE_FREEGLUT)
        include_directories(${FREEGLUT_INCLUDE_DIR})
    endif(USE_GLFW)
    include_directories(${OpenGL_INCLUDE_DIRS})
    include_directories(${GLEW_INCLUDE_DIRS})
    include_directories(${GLM_INCLUDE_DIRS})
    set(TARGET_GL_VER_MAJOR 4 CACHE STRING "Target OpenGL profile major version")
    set(TARGET_GL_VER_MINOR 4 CACHE STRING "Target OpenGL profile minor version")
    option(GLEW_EXPERIMENTAL "Use experimental GLEW features" ON)
else(NOT USE_GLSL)
    message(WARNING "Missing OpenGL library(ies), vptz & GLSL implementations will be disabled")
endif(USE_GLSL)

### CUDA CHECK @@@@ add later for parallel utils & impls
set_eval(USE_CUDA 0)

### OPENCL CHECK @@@@ add later for parallel utils & impls
set_eval(USE_OPENCL 0)

### HARDWARE SIMD CHECK
if(CMAKE_CROSSCOMPILING)
    option(USE_MMX "Use MMX instructions support" OFF)
    option(USE_SSE "Use SSE instructions support" OFF)
    option(USE_SSE2 "Use SSE2 instructions support" OFF)
    option(USE_SSE3 "Use SSE3 instructions support" OFF)
    option(USE_SSSE3 "Use SSSE3 instructions support" OFF)
    option(USE_SSE4_1 "Use SSE4_1 instructions support" OFF)
    option(USE_SSE4_2 "Use SSE4_2 instructions support" OFF)
    option(USE_POPCNT "Use POPCNT instructions support" OFF)
    option(USE_AVX "Use AVX instructions support" OFF)
    option(USE_AVX2 "Use AVX2 instructions support" OFF)
else(CMAKE_CROSSCOMPILING)
    ## @@@@ these might not work on msvc toolchain (to be tested)
    try_cvhardwaresupport_runcheck_and_set_success(MMX)
    try_cvhardwaresupport_runcheck_and_set_success(SSE)
    try_cvhardwaresupport_runcheck_and_set_success(SSE2)
    try_cvhardwaresupport_runcheck_and_set_success(SSE3)
    try_cvhardwaresupport_runcheck_and_set_success(SSSE3)
    try_cvhardwaresupport_runcheck_and_set_success(SSE4_1)
    try_cvhardwaresupport_runcheck_and_set_success(SSE4_2)
    try_cvhardwaresupport_runcheck_and_set_success(POPCNT)
    try_cvhardwaresupport_runcheck_and_set_success(AVX)
    try_cvhardwaresupport_runcheck_and_set_success(AVX2)
endif(CMAKE_CROSSCOMPILING)

option(USE_FAST_MATH "Enable fast math optimizations" OFF)
set(GLOBAL_DATASET_ROOT_PATH "/shared2/datasets/" CACHE STRING "Global dataset root folder for exec projects")
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    if(NOT CMAKE_CROSSCOMPILING)
        add_definitions(-march=native)
    endif(NOT CMAKE_CROSSCOMPILING)
    if(USE_FAST_MATH)
        add_definitions(-ffast-math)
    endif(USE_FAST_MATH)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-Wshadow)
    add_definitions(-Wfatal-errors)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    if(USE_FAST_MATH)
        add_definitions(/fp:fast)
    else(NOT USE_FAST_MATH)
        add_definitions(/fp:precise)
    endif(USE_FAST_MATH)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/openmp)
    add_definitions(/arch:AVX) # check performance difference? vs 387? @@@
    add_definitions(/Wall)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(SEND_ERROR "Intel compiler still unsupported; please edit the main CMakeList.txt file to add proper configuration")
endif()

add_subdirectory(3rdparty)

initialize_internal_list(litiv_modules)
initialize_internal_list(litiv_projects)
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    add_definitions(-Werror)
    add_definitions(-pedantic-errors)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions(/W4)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # ... @@@
endif()

add_subdirectory(modules)
add_subdirectory(execs)
