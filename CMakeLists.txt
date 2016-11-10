
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
set(EXTERNAL_DATA_ROOT "${CMAKE_SOURCE_DIR}/data/" CACHE PATH "External data root folder (this is where all datasets should be located for applications to find them automatically)")
set(SAMPLES_DATA_ROOT "${CMAKE_SOURCE_DIR}/samples/data/" CACHE PATH "Sample data root folder (should contain necessary files so that samples can be used out-of-the-box)")
if(WIN32)
    if(MSVC AND (MSVC_VERSION LESS 1900))
        message(FATAL_ERROR "MSVC toolchains older than 2015 (v140) are not supported!")
    elseif(MSVC)
        set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
    endif()
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install CACHE PATH "Install path prefix, prepended onto install directories (optional)")
else(NOT WIN32)
    set(CMAKE_INSTALL_PREFIX "/usr/local/" CACHE PATH "Install path prefix, prepended onto install directories (optional)")
endif()
# add hardcoded guesses for find_package here (caution: top level project is not defined yet!)
list(APPEND CMAKE_PREFIX_PATH
    $ENV{USER_DEVELOP}/opencv/build-litiv/install/ $ENV{USER_DEVELOP}/opencv/build/ $ENV{USER_DEVELOP}/opencv/
    $ENV{USER_DEVELOP}/freeglut/build-litiv/install/ $ENV{USER_DEVELOP}/freeglut/build/ $ENV{USER_DEVELOP}/freeglut/
    $ENV{USER_DEVELOP}/glfw/build-litiv/install/ $ENV{USER_DEVELOP}/glfw/build/ $ENV{USER_DEVELOP}/glfw/
    $ENV{USER_DEVELOP}/glew/build-litiv/install/ $ENV{USER_DEVELOP}/glew/build/ $ENV{USER_DEVELOP}/glew/
    $ENV{USER_DEVELOP}/glm/build-litiv/install/ $ENV{USER_DEVELOP}/glm/build/ $ENV{USER_DEVELOP}/glm/
    $ENV{USER_DEVELOP}/opengm/build-litiv/install/ $ENV{USER_DEVELOP}/opengm/build/ $ENV{USER_DEVELOP}/opengm/
)

################################################################

project(litiv)
set(LITIV_VERSION_MAJOR 1) # last change: 2015/10
set(LITIV_VERSION_MINOR 3) # last change: 2016/10
set(LITIV_VERSION_PATCH 1) # last change: 2016/10
set(LITIV_VERSION ${LITIV_VERSION_MAJOR}.${LITIV_VERSION_MINOR}.${LITIV_VERSION_PATCH})
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Forced project build type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>)
if(CMAKE_SIZEOF_VOID_P EQUAL 8) # 64-bit toolchain/platform
    set(DATASETS_CACHE_SIZE 6 CACHE STRING "Cache size to be used for dataset preloading, if needed (in GB)")
    set(TARGET_PLATFORM_x64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32-bit toolchain/platform
    set(DATASETS_CACHE_SIZE 1 CACHE STRING "Cache size to be used for dataset preloading, if needed (in GB)")
    set(TARGET_PLATFORM_x64 FALSE CACHE INTERNAL "" FORCE)
else()
    message(FATAL_ERROR "Could not detect x64/x86 platform identity using void pointer size (s=${CMAKE_SIZEOF_VOID_P}).")
endif()
option(USE_FAST_MATH "Enable fast math optimizations" OFF)
mark_as_advanced(USE_FAST_MATH DATASETS_CACHE_SIZE)

### OPENCV CHECK
find_package(OpenCV 3.0 REQUIRED)

### GLSL CHECK
find_package(OpenGL)
find_package(FREEGLUT)
find_package(GLFW)
find_package(GLEW)
find_package(GLM)
set_eval(USE_GLSL ((${GLFW_FOUND} OR ${FREEGLUT_FOUND}) AND ${OPENGL_FOUND} AND ${GLEW_FOUND} AND ${GLM_FOUND}))
option(USE_GLSL "Specifies whether GLSL support should be enabled or not; if false, projects that depend on it will be disabled" ${USE_GLSL})
if(USE_GLSL)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(GLM REQUIRED)
    option(USE_GLFW "Use GLFW as the OpenGL window manager for GLSL implementations" ${GLFW_FOUND})
    option(USE_FREEGLUT "Use FREEGLUT as the OpenGL window manager for GLSL implementations" ${FREEGLUT_FOUND})
    if((${GLFW_FOUND} AND ${FREEGLUT_FOUND}) OR ((NOT ${GLFW_FOUND}) AND (NOT ${FREEGLUT_FOUND})))
        set(USE_GLFW ON)
        set(USE_FREEGLUT OFF)
    endif()
    if(${USE_GLFW})
        find_package(GLFW REQUIRED)
    elseif(${FREEGLUT_FOUND})
        find_package(FREEGLUT REQUIRED)
    endif()
    set(TARGET_GL_VER_MAJOR 4 CACHE STRING "Target OpenGL profile major version")
    set(TARGET_GL_VER_MINOR 4 CACHE STRING "Target OpenGL profile minor version")
    option(GLEW_EXPERIMENTAL "Use experimental GLEW features" ON)
else()
    message("Without GLSL support enabled, vptz module & shader-based implementations will be disabled")
endif()

### OPENGM CHECK
find_package(OpenGM COMPONENTS ext)
option(USE_OPENGM "Specifies whether OpenGM should be included/linked or not; if false, projects that depend on it will be disabled" ${OpenGM_FOUND})
if(USE_OPENGM)
    find_package(OpenGM REQUIRED COMPONENTS ext)
else()
    message("Without OpenGM support w/ external dependencies enabled, cosegm project & utilities will be disabled")
endif()

### CUDA CHECK @@@@ add later for parallel utils & impls
set_eval(USE_CUDA 0)

### OPENCL CHECK @@@@ add later for parallel utils & impls
set_eval(USE_OPENCL 0)

### Kinect SDK CHECK
if(WIN32)
    find_package(KinectSDK2)
    mark_as_advanced(
        KinectSDK2_FACE
        KinectSDK2_FUSION
        KinectSDK2_VGB
    )
    option(USE_KINECTSDK "Specifies whether the Kinectv2 SDK should be included/linked or not; if false, projects that depend on it will be disabled" ${KinectSDK2_FOUND})
    if(USE_KINECTSDK)
        find_package(KinectSDK2 REQUIRED)
    else()
        message("Without full Kinectv2 SDK support, capture app will be disabled")
    endif()
endif()
if(KinectSDK2_FOUND)
    message(STATUS "Kinectv2 SDK found, will disable internal SDK standalone utilities.")
    set(USE_KINECTSDK_STANDALONE FALSE CACHE INTERNAL "" FORCE)
else()
    set(USE_KINECTSDK_STANDALONE TRUE CACHE INTERNAL "" FORCE)
endif()

initialize_internal_list(litiv_projects)
initialize_internal_list(litiv_3rdparty_modules)
initialize_internal_list(litiv_modules)
initialize_internal_list(litiv_apps)
initialize_internal_list(litiv_samples)

################################################################

add_subdirectory(cmake/checks/simd)
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    if(NOT CMAKE_CROSSCOMPILING)
        add_definitions(-march=native)
    endif()
    if(USE_FAST_MATH)
        add_definitions(-ffast-math)
    endif()
    add_definitions(-Wall)
    add_definitions(-Wfatal-errors)
    add_definitions(-fPIC) # @@@@ for shared libs only?
    check_function_exists(aligned_alloc USE_STL_ALIGNED_ALLOC)
    check_function_exists(posix_memalign USE_POSIX_ALIGNED_ALLOC)
elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
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
    message(FATAL_ERROR "Intel compiler still unsupported; please edit the main CMakeList.txt file to add proper configuration")
    # ... @@@
else()
    message(FATAL_ERROR "Unknown compiler; please edit the main CMakeList.txt file to add proper configuration")
endif()

add_subdirectory(3rdparty)
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    add_definitions(-Wextra)
    add_definitions(-Wshadow)
    add_definitions(-Werror)
    add_definitions(-pedantic-errors)
    add_definitions(-Wno-missing-braces)
elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
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

################################################################

install(FILES "LICENSE.txt" DESTINATION "share/litiv/")
if(EXISTS "${CMAKE_ROOT}/Modules/CPack.cmake")
    option(USE_CPACK "Use CPack to build installation package" OFF)
    if(USE_CPACK)
        include(InstallRequiredSystemLibraries)
        #set(CPACK_SET_DESTDIR ON)
        #set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/")
        set(CPACK_PACKAGE_DESCRIPTION "LITIV Computer Vision Framework")
        set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "C++ implementation pool for computer vision projects undertaken in the LITIV lab at Polytechnique Montreal.")
        set(CPACK_PACKAGE_VENDOR "Pierre-Luc St-Charles")
        set(CPACK_PACKAGE_CONTACT "pierre-luc.st-charles@polymtl.ca")
        set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_SOURCE_DIR}/README.md")
        set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
        set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
        #set(CPACK_RESOURCE_FILE_WELCOME "${CMAKE_SOURCE_DIR}/README.md")
        set(CPACK_PACKAGE_VERSION_MAJOR ${LITIV_VERSION_MAJOR})
        set(CPACK_PACKAGE_VERSION_MINOR ${LITIV_VERSION_MINOR})
        set(CPACK_PACKAGE_VERSION_PATCH ${LITIV_VERSION_PATCH})
        set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${LITIV_VERSION_MAJOR}.${LITIV_VERSION_MINOR}.${LITIV_VERSION_PATCH}")
        set(CPACK_PACKAGE_INSTALL_DIRECTORY "LITIV Framework v${LITIV_VERSION_MAJOR}.${LITIV_VERSION_MINOR}")
        #set(CPACK_OUTPUT_CONFIG_FILE "litivConfig.cmake")
        if(WIN32 AND NOT UNIX)
            set(CPACK_GENERATOR "NSIS")
            #set(CPACK_PACKAGE_ICON "${CMAKE_SOURCE_DIR}/doc/images\\\\litiv.ico") # @@@ 4x backslash bug?
            set(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} LITIV Framework")
            set(CPACK_NSIS_HELP_LINK "https://github.com/plstcharles/litiv")
            set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/plstcharles/litiv")
            set(CPACK_NSIS_CONTACT ${CPACK_PACKAGE_CONTACT})
            set(CPACK_NSIS_MODIFY_PATH ON)
        else()
            set(CPACK_GENERATOR "DEB")
            set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/litiv/${CPACK_PACKAGE_FILE_NAME}" CACHE STRING "LITIV package install path prefix")
            #set(CPACK_DEBIAN_PACKAGE_DEPENDS " kate , plasma-desktop, libc6 (>= 2.1.3) ")
            set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
            set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
            set(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})
            set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/plstcharles/litiv")
            set(CPACK_DEBIAN_COMPRESSION_TYPE "gzip")
            #set(CPACK_STRIP_FILES "bin/<something>") # @@@ or true/false?
            set(CPACK_SOURCE_STRIP_FILES "")
        endif()
        #set(CPACK_PACKAGE_EXECUTABLES "<exec>" "<execname>")
        #set(CPACK_COMPONENTS_ALL "<component1>" "<component2>")
        include(CPack)
    endif()
endif()
