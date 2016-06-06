
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
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/ ${CMAKE_SOURCE_DIR}/cmake/Modules/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
include(FrameworkUtils)
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(DATASET_ROOT "${CMAKE_SOURCE_DIR}/data/" CACHE STRING "Dataset root folder (this is where test data should be located for apps benchmarks to be used out-of-the-box)")
set(SAMPLES_ROOT "${CMAKE_SOURCE_DIR}/samples/" CACHE STRING "Samples root folder (should contain a data subdirectory so that samples can be used out-of-the-box)")
if(WIN32)
    # add hardcoded guesses for find_package here (caution: top level project is not defined yet!)
    list(APPEND CMAKE_PREFIX_PATH
        $ENV{USER_DEVELOP}/opencv/ $ENV{USER_DEVELOP}/opencv/build/
        $ENV{USER_DEVELOP}/freeglut/ $ENV{USER_DEVELOP}/freeglut/build/
        $ENV{USER_DEVELOP}/glfw/ $ENV{USER_DEVELOP}/glfw/build/
        $ENV{USER_DEVELOP}/glew/ $ENV{USER_DEVELOP}/glew/build/
        $ENV{USER_DEVELOP}/glm/ $ENV{USER_DEVELOP}/glm/build/
    )
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE STRING "Install path prefix, prepended onto install directories (optional)")
else(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/usr/local/" CACHE STRING "Install path prefix, prepended onto install directories (optional)")
endif()
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

project(litiv)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC AND (MSVC_VERSION LESS 1900))
    message(FATAL_ERROR "MSVC toolchains older than 2015 (v140) are not supported!")
elseif(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

set(LITIV_VERSION_MAJOR 1) # last change: 2015/10
set(LITIV_VERSION_MINOR 2) # last change: 2016/04
set(LITIV_VERSION_PATCH 0) # last change: 2016/04
set(LITIV_VERSION ${LITIV_VERSION_MAJOR}.${LITIV_VERSION_MINOR}.${LITIV_VERSION_PATCH})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>)

### OPENCV CHECK
find_package(OpenCV 3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

### GLSL CHECK
find_package(OpenGL)
find_package(FREEGLUT)
find_package(GLFW)
find_package(GLEW)
find_package(GLM)
set_eval(USE_GLSL ((${GLFW_FOUND} OR ${FREEGLUT_FOUND}) AND ${OPENGL_FOUND} AND ${GLEW_FOUND} AND ${GLM_FOUND}))
if(USE_GLSL)
    if(${GLFW_FOUND} AND ${FREEGLUT_FOUND})
        set(USE_GLFW 1 CACHE BOOL "Use GLFW as the OpenGL window manager")
        set(USE_FREEGLUT 0 CACHE BOOL "Use FREEGLUT as the OpenGL window manager")
    elseif(${GLFW_FOUND})
        set(USE_GLFW 1)
        set(USE_FREEGLUT 0)
    elseif(${FREEGLUT_FOUND})
        set(USE_GLFW 0)
        set(USE_FREEGLUT 1)
    endif()
    if(${USE_GLFW} EQUAL ${USE_FREEGLUT})
        message(FATAL_ERROR "Need to select one window manager.")
    endif()
    if(USE_GLFW)
        include_directories(${GLFW_INCLUDE_DIR})
    elseif(USE_FREEGLUT)
        include_directories(${FREEGLUT_INCLUDE_DIR})
    endif()
    include_directories(${OpenGL_INCLUDE_DIRS})
    include_directories(${GLEW_INCLUDE_DIRS})
    include_directories(${GLM_INCLUDE_DIRS})
    set(TARGET_GL_VER_MAJOR 4 CACHE STRING "Target OpenGL profile major version")
    set(TARGET_GL_VER_MINOR 4 CACHE STRING "Target OpenGL profile minor version")
    option(GLEW_EXPERIMENTAL "Use experimental GLEW features" ON)
else(NOT USE_GLSL)
    message(WARNING "Missing OpenGL library(ies), vptz & GLSL implementations will be disabled")
endif()

### CUDA CHECK @@@@ add later for parallel utils & impls
set_eval(USE_CUDA 0)

### OPENCL CHECK @@@@ add later for parallel utils & impls
set_eval(USE_OPENCL 0)

### SIMD CHECK
try_cvhardwaresupport_runcheck_and_set_success(MMX OFF)
try_cvhardwaresupport_runcheck_and_set_success(SSE ON)
try_cvhardwaresupport_runcheck_and_set_success(SSE2 ON)
try_cvhardwaresupport_runcheck_and_set_success(SSE3 ON)
try_cvhardwaresupport_runcheck_and_set_success(SSSE3 ON)
try_cvhardwaresupport_runcheck_and_set_success(SSE4_1 ON)
try_cvhardwaresupport_runcheck_and_set_success(SSE4_2 ON)
try_cvhardwaresupport_runcheck_and_set_success(POPCNT ON)
try_cvhardwaresupport_runcheck_and_set_success(AVX ON)
try_cvhardwaresupport_runcheck_and_set_success(AVX2 OFF)

option(USE_FAST_MATH "Enable fast math optimizations" OFF)
if(CMAKE_SIZEOF_VOID_P EQUAL 8) # 64-bit toolchain/platform
    set(DATASETS_CACHE_SIZE 6 CACHE STRING "Cache size to be used for dataset preloading, if needed (in GB)")
    set(TARGET_PLATFORM_IS_x64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32-bit toolchain/platform
    set(DATASETS_CACHE_SIZE 1 CACHE STRING "Cache size to be used for dataset preloading, if needed (in GB)")
    set(TARGET_PLATFORM_IS_x64 FALSE CACHE INTERNAL "" FORCE)
else()
    message(FATAL_ERROR "Could not detect x64/x86 platform identity using void pointer size.")
endif()
mark_as_advanced(DATASETS_CACHE_SIZE)
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    if(NOT CMAKE_CROSSCOMPILING)
        add_definitions(-march=native)
    endif()
    if(USE_FAST_MATH)
        add_definitions(-ffast-math)
    endif()
    add_definitions(-Wall)
    add_definitions(-Wfatal-errors)
    add_definitions(-fPIC) # @@@@ for shared libs only?
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions(-DUNICODE -D_UNICODE)
    if(USE_FAST_MATH)
        add_definitions(/fp:fast)
    else(NOT USE_FAST_MATH)
        add_definitions(/fp:precise)
    endif()
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/W1)
    add_definitions(/openmp)
    add_definitions(/arch:AVX) # check performance difference? vs 387? @@@
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(SEND_ERROR "Intel compiler still unsupported; please edit the main CMakeList.txt file to add proper configuration")
    # ... @@@
endif()

add_subdirectory(3rdparty)

initialize_internal_list(litiv_modules)
initialize_internal_list(litiv_projects)
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"))
    add_definitions(-Wextra)
    add_definitions(-Wshadow)
    add_definitions(-Werror)
    add_definitions(-pedantic-errors)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions(/W4)
    add_definitions(/wd4201) # disables C4201, "nonstandard extension used : nameless struct/union"
    add_definitions(/wd4505) # disables C4505, "unreferenced local function has been removed"
    add_definitions(/wd4514) # disables C4514, "unreferenced inline function has been removed"
    add_definitions(/wd4250) # disables C4250, "'class1' : inherits 'class2::member' via dominance" (such behavior is expected in datasets module due to diamond struct patterns)
    add_definitions(/wd4268) # disables C4268, "'variable': 'const' static/global data initialized with compiler generated default constructor fills the object with zeros
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # ... @@@
endif()

add_subdirectory(modules)
add_subdirectory(samples)
add_subdirectory(apps)
add_subdirectory(doc)
