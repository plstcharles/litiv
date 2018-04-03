
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
list(APPEND CMAKE_MODULE_PATH
    "${CMAKE_SOURCE_DIR}/cmake/"
    "${CMAKE_SOURCE_DIR}/cmake/Modules/"
)
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   message(FATAL_ERROR "In-source builds are not supported; generation should be done in a local subfolder, e.g. '${CMAKE_SOURCE_DIR}/build'.")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
cmake_policy(SET CMP0007 NEW)
include(FrameworkUtils)
include(ExternalProject)
include(DownloadProject)
include(CMakePackageConfigHelpers)
include(GetGitRevisionDescription)
include(CheckFunctionExists) # should be pre-packaged with CMake...
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(EXTERNAL_DATA_ROOT "${CMAKE_SOURCE_DIR}/data/" CACHE PATH "External data root folder (this is where all datasets should be located for applications to find them automatically)")
set(SAMPLES_DATA_ROOT "${CMAKE_SOURCE_DIR}/samples/data/" CACHE PATH "Sample data root folder (should contain necessary files so that samples can be used out-of-the-box)")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install path prefix, prepended onto install directories (optional)")
# add hardcoded guesses for find_package here (caution: top level project is not defined yet!)
list(APPEND CMAKE_PREFIX_PATH
    "$ENV{USER_DEVELOP}/cuda-litiv/" "$ENV{USER_DEVELOP}/cuda-latest/" "$ENV{USER_DEVELOP}/cuda/"
    "$ENV{USER_DEVELOP}/opencv/build-litiv/install/" "$ENV{USER_DEVELOP}/opencv/build/install/" "$ENV{USER_DEVELOP}/opencv/"
    "$ENV{USER_DEVELOP}/freeglut/build-litiv/install/" "$ENV{USER_DEVELOP}/freeglut/build/install/" "$ENV{USER_DEVELOP}/freeglut/"
    "$ENV{USER_DEVELOP}/glfw/build-litiv/install/" "$ENV{USER_DEVELOP}/glfw/build/install/" "$ENV{USER_DEVELOP}/glfw/"
    "$ENV{USER_DEVELOP}/glew/build-litiv/install/" "$ENV{USER_DEVELOP}/glew/build/install/" "$ENV{USER_DEVELOP}/glew/"
    "$ENV{USER_DEVELOP}/glm/build-litiv/install/" "$ENV{USER_DEVELOP}/glm/build/install/" "$ENV{USER_DEVELOP}/glm/"
    "$ENV{USER_DEVELOP}/opengm/build-litiv/install/" "$ENV{USER_DEVELOP}/opengm/build/install/" "$ENV{USER_DEVELOP}/opengm/"
    "$ENV{USER_DEVELOP}/eigen3/build-litiv/install/" "$ENV{USER_DEVELOP}/eigen3/build/install/" "$ENV{USER_DEVELOP}/eigen3/"
    "$ENV{USER_DEVELOP}/llvm/build-litiv/install/" "$ENV{USER_DEVELOP}/llvm/build/install/" "$ENV{USER_DEVELOP}/llvm/"
)

################################################################

project(litiv)
set(LITIV_VERSION_MAJOR 1) # last change: 2015/10
set(LITIV_VERSION_MINOR 6) # last change: 2018/02
set(LITIV_VERSION_PATCH 0) # last change: 2018/02
set(LITIV_VERSION "${LITIV_VERSION_MAJOR}.${LITIV_VERSION_MINOR}.${LITIV_VERSION_PATCH}")
set(LITIV_VERSION_PLAIN "${LITIV_VERSION_MAJOR}${LITIV_VERSION_MINOR}${LITIV_VERSION_PATCH}")
set(LITIV_VERSION_ABI 0) # still unstable
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Configs" FORCE)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Forced project build type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS "$<$<CONFIG:Debug>:DEBUG>")
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS "$<$<CONFIG:Debug>:_DEBUG>")
if(CMAKE_SIZEOF_VOID_P EQUAL 8) # 64-bit toolchain/platform
    set(DATASETS_CACHE_SIZE 4096 CACHE STRING "Cache size to be used for dataset preloading, if needed (in MB)")
    set(TARGET_PLATFORM_x64 TRUE CACHE INTERNAL "" FORCE)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4) # 32-bit toolchain/platform
    set(DATASETS_CACHE_SIZE 512 CACHE STRING "Cache size to be used for dataset preloading, if needed (in MB)")
    set(TARGET_PLATFORM_x64 FALSE CACHE INTERNAL "" FORCE)
else()
    message(FATAL_ERROR "Could not detect x64/x86 platform identity using void pointer size (s=${CMAKE_SIZEOF_VOID_P}).")
endif()
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    option(USE_3RDPARTY_OPTFLOW_KROEGER2016 "Link & use 3rdparty GPLv3 optical flow implementation from Kroeger et al. (not fully MSVC-compat)" OFF)
    mark_as_advanced(USE_3RDPARTY_OPTFLOW_KROEGER2016)
else()
    option(USE_3RDPARTY_OPTFLOW_KROEGER2016 "Link & use 3rdparty GPLv3 optical flow implementation from Kroeger et al." ON)
endif()
option(USE_LZ4 "Download and compile LZ4 fast compression lib for use in framework utils" ON)
option(USE_WORLD_SOURCE_GLOB "Compile all LITIV modules as part of the world module instead of soft-linking them" OFF)
option(USE_BSDS500_BENCHMARK "Link & use BSDS500 dataset benchmark implementation in litiv_datasets (cleaned 3rd party code; will use an approximative solution otherwise)" ON)
option(USE_DEBUG_SYMBOLS "Enable debug symbols in all configs, for all targets" OFF)
option(USE_CVCORE_WITH_UTILS "Include OpenCV core components in utils module" ON)
option(USE_LINK_TIME_OPTIM "Enable link time optimization" ON)
option(USE_INLINE_INTRINSIC_FUNCS "Enable use of built-in inline intrinsic functions" ON)
option(USE_FAST_MATH "Enable fast math optimization" OFF)
option(USE_OPENMP "Enable OpenMP in internal implementations" ON)
option(USE_PROFILING "Enable gperftools profiling of litiv applications" OFF)
if(NOT CMAKE_CROSSCOMPILING)
    option(BUILD_TESTS "Build regression and performance tests with Google Test/Benchmark frameworks" ON)
    option(BUILD_TESTS_PERF "Build & run performance tests alongside regression tests in CTest" OFF)
    option(BUILD_TESTS_FULL_FLOAT "Run full floating point test suite to check for perfect platform compatibility" OFF)
    mark_as_advanced(BUILD_TESTS_FULL_FLOAT)
    if(NOT BUILD_TESTS)
        unset(BUILD_TESTS_PERF CACHE)
    endif()
endif()
mark_as_advanced(
    USE_WORLD_SOURCE_GLOB
    USE_BSDS500_BENCHMARK
    USE_CVCORE_WITH_UTILS
    USE_LINK_TIME_OPTIM
    USE_INLINE_INTRINSIC_FUNCS
    USE_FAST_MATH
    USE_OPENMP
    DATASETS_CACHE_SIZE
)

### OPENCV CHECK
find_package(OpenCV 3.0 REQUIRED)
message(STATUS "Found OpenCV >=3.0 at '${OpenCV_DIR}'")

### GLSL CHECK
find_package(OpenGL QUIET)
find_package(FREEGLUT QUIET)
find_package(GLFW QUIET)
find_package(GLEW QUIET)
find_package(GLM QUIET)
set_eval(USE_GLSL_INIT ((${GLFW_FOUND} OR ${FREEGLUT_FOUND}) AND ${OPENGL_FOUND} AND ${GLEW_FOUND} AND ${GLM_FOUND}))
option(USE_GLSL "Specifies whether GLSL support should be enabled or not; if false, projects that depend on it will be disabled" ${USE_GLSL_INIT})
if(USE_GLSL)
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(GLM REQUIRED)
    option(USE_GLFW "Use GLFW as the OpenGL window manager for GLSL implementations" ${GLFW_FOUND})
    option(USE_FREEGLUT "Use FREEGLUT as the OpenGL window manager for GLSL implementations" ${FREEGLUT_FOUND})
    if((NOT ${GLFW_FOUND}) AND (NOT ${FREEGLUT_FOUND}))
        message(FATAL_ERROR "Missing OpenGL window manager; default (and suggested) library is GLFW -- enable one via USE_**** option.")
    endif()
    if(${USE_GLFW})
        find_package(GLFW REQUIRED)
    elseif(${FREEGLUT_FOUND})
        find_package(FREEGLUT REQUIRED)
    endif()
    set(TARGET_GL_VER_MAJOR 4 CACHE STRING "Target OpenGL profile major version for LITIV modules")
    set(TARGET_GL_VER_MINOR 4 CACHE STRING "Target OpenGL profile minor version for LITIV modules")
    option(USE_VPTZ_STANDALONE "Build VPTZ library as standalone lib from LITIV framework" ON)
    option(USE_GLEW_EXPERIMENTAL "Use experimental GLEW features" ON)
    mark_as_advanced(USE_GLEW_EXPERIMENTAL)
else()
    message("Without GLSL support enabled, vptz module & shader-based implementations will be disabled.")
    unset(USE_GLFW CACHE)
    unset(USE_FREEGLUT CACHE)
    unset(USE_GLEW_EXPERIMENTAL CACHE)
    unset(USE_VPTZ_STANDALONE CACHE)
endif()

### BOOST CHECK
find_package(Boost QUIET)
option(USE_BOOST "Specifies whether Boost should be included/linked or not; if false, projects that depend on it (e.g. SoSPD, cosegmenters) will be disabled" ${Boost_FOUND})
if(USE_BOOST)
    find_package(Boost COMPONENTS chrono date_time system filesystem program_options regex serialization)
    if(NOT Boost_FOUND)
        message(FATAL_ERROR "Unable to find boost include/library directories; user must set BOOST_ROOT, BOOST_INCLUDEDIR or BOOST_LIBRARYDIR to be used as hints.")
    endif()
else()
    message("Without Boost support, projects that depend on it (e.g. SoSPD, cosegmenters) will be disabled.")
endif()

### OPENGM CHECK
find_package(OpenGM QUIET COMPONENTS ext)
option(USE_OPENGM "Specifies whether OpenGM should be included/linked or not; if false, projects that depend on it (e.g. cosegmenters) will be disabled" ${OpenGM_FOUND})
if(USE_OPENGM)
    find_package(OpenGM COMPONENTS ext)
    if(NOT OpenGM_FOUND)
        message(FATAL_ERROR "Unable to find OpenGM include/library directories; user must set OpenGM_INCLUDEDIR and/or OpenGM_LIBRARYDIR to be used as hints.")
    endif()
else()
    message("Without OpenGM support w/ external dependencies enabled, cosegmentation project & utilities will be disabled.")
endif()

### CUDA CHECK
if(CMAKE_CROSSCOMPILING OR (WIN32 AND NOT MSVC) OR (CMAKE_COMPILER_IS_GNUCXX AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")))
    set_eval(USE_CUDA 0)
    message("Cannot use CUDA, invalid platform/compiler combination.")
else()
    find_package(CUDA ${OpenCV_CUDA_VERSION} QUIET)
    option(USE_CUDA "Specifies whether CUDA should be included/linked or not; if false, projects that depend on it will be disabled" ${CUDA_FOUND})
    if(USE_CUDA)
        if(NOT OpenCV_CUDA_VERSION)
            message(FATAL_ERROR "Setting 'USE_CUDA' requires OpenCV to also be built with cuda support")
        endif()
        find_package(CUDA ${OpenCV_CUDA_VERSION} REQUIRED)
        if(CUDA_VERSION VERSION_LESS 7.0)
            message(FATAL_ERROR "Framework requires a CUDA version of at least 7.0")
        endif()
        set(CUDA_DEFAULT_GENCODE "-gencode arch=compute_30,code=sm_30 -gencode arch=compute_30,code=compute_30")
        set(CUDA_CURRENT_GENCODE "${CUDA_DEFAULT_GENCODE}" CACHE STRING "Gencode command line args which will be passed to nvcc; change here to target different/more architectures)")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "${CUDA_CURRENT_GENCODE}")
        if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11) # ideally, we would use std=c++14, but it is not supported
        endif()
        if(USE_FAST_MATH)
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math)
        endif()
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr) # required by utilities in opencv2/core/cuda/vec_math.hpp
        option(CUDA_EXIT_ON_ERROR "Specifies whether CUDA calls should cause program abortion on error, or throw an exception" ON)
        mark_as_advanced(CUDA_EXIT_ON_ERROR)
    else()
        message("Without CUDA support enabled, some imgproc utilities will be disabled.")
    endif()
endif()

### OPENCL CHECK @@@@ add later for parallel utils & impls
set_eval(USE_OPENCL 0)

### OPENMP CHECK
if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

### Kinect SDK CHECK
if(WIN32)
    find_package(KinectSDK2 QUIET)
    mark_as_advanced(
        KinectSDK2_FACE
        KinectSDK2_FUSION
        KinectSDK2_VGB
    )
    option(USE_KINECTSDK "Specifies whether the Kinectv2 SDK should be included/linked or not; if false, projects that depend on it will be disabled" ${KinectSDK2_FOUND})
    if(USE_KINECTSDK)
        find_package(KinectSDK2 REQUIRED)
    else()
        message("Without full Kinectv2 SDK support, capture app will be disabled.")
    endif()
endif()
if(KinectSDK2_FOUND)
    message(STATUS "Kinectv2 SDK found, will disable internal SDK standalone utilities.")
    set(USE_KINECTSDK_STANDALONE FALSE CACHE INTERNAL "" FORCE)
else()
    set(USE_KINECTSDK_STANDALONE TRUE CACHE INTERNAL "" FORCE)
endif()

### GoogleTest CHECK
if(BUILD_TESTS)
    enable_testing()
    find_package(Threads REQUIRED)
    # we do not check for GTest/Benchmark packages, they will be downloaded/built in-tree
    if(UNIX)
        add_custom_target(check
            COMMAND +env ${CMAKE_MAKE_PROGRAM} all
            COMMAND ${CMAKE_COMMAND} -E echo CMD=${CMAKE_CTEST_COMMAND} -C $<CONFIG>
            COMMAND ${CMAKE_COMMAND} -E echo ----------------------------------
            COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
                    ${CMAKE_CTEST_COMMAND} -C $<CONFIG>
            WORKING_DIRECTORY
                ${CMAKE_BINARY_DIR}
        )
        set_target_properties(check
            PROPERTIES
                EXCLUDE_FROM_ALL
                    TRUE
        )
    elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
        add_custom_target(ALL_BUILD_AND_TEST
            ${CMAKE_COMMAND} -E echo CMD=${CMAKE_CTEST_COMMAND} -C $<CONFIG>
            COMMAND ${CMAKE_COMMAND} -E echo ----------------------------------
            COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
                    ${CMAKE_CTEST_COMMAND} -C $<CONFIG>
            DEPENDS
                ALL_BUILD
            WORKING_DIRECTORY
                ${CMAKE_BINARY_DIR}
        )
        set_target_properties(ALL_BUILD_AND_TEST
            PROPERTIES
                EXCLUDE_FROM_ALL
                    TRUE
        )
    endif()
    set(TEST_INPUT_DATA_ROOT "${CMAKE_SOURCE_DIR}/modules/test/data" CACHE PATH "Test input data root folder (used internally for testing, contains binary reference files)")
    set(TEST_OUTPUT_DATA_ROOT "${CMAKE_BINARY_DIR}/Testing/data" CACHE PATH "Test output data root folder (used internally for testing, light external data may be downloaded and kept there)")
    mark_as_advanced(TEST_INPUT_DATA_ROOT TEST_OUTPUT_DATA_ROOT)
    file(MAKE_DIRECTORY ${TEST_OUTPUT_DATA_ROOT})
endif()

if(USE_PROFILING)
    find_package(Gperftools REQUIRED)
    if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
        if(NOT USE_DEBUG_SYMBOLS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
        endif()
        set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -lprofiler") # can apply to all build targets...
    else()
        message(FATAL_ERROR "Missing Gperftools compiler/linker options for this toolchain...")
    endif()
endif()

### clang-tidy CHECK (will always use if available)
find_program(CLANG_TIDY_BIN NAMES "clang-tidy")
if(CLANG_TIDY_BIN)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_BIN}")
    option(USE_CLANG_TIDY_IN_BUILD "Specifies whether the clang-tidy C++ linter should be used or not at compile time" OFF)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CLANG_TIDY_CHECKS "*,-boost-*,-hicpp-*,-llvm-*,-mpi-*,-objc-*")
    #set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cert-dcl58-cpp") # (std extension) not disabled, but using NOLINT hints
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cert-msc50-cpp,-cert-msc30-c") # disable checks for std::rand()
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-clang-analyzer-osx*") # disable osx-specific tests
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-owning-memory")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-bounds-array-to-pointer-decay")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-bounds-constant-array-index")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-bounds-pointer-arithmetic")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-type-const-cast")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-type-cstyle-cast")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-type-member-init") # https://llvm.org/bugs/show_bug.cgi?id=31039
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-type-reinterpret-cast")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-type-static-cast-downcast")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-type-union-access")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-pro-type-vararg")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-cppcoreguidelines-special-member-functions")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-google-readability-braces-around-statements")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-google-readability-casting")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-google-readability-function-size")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-google-readability-todo")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-google-runtime-int")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-google-runtime-references")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-misc-misplaced-widening-cast") # should be used...
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-modernize-raw-string-literal")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-modernize-return-braced-init-list")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-modernize-use-auto")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-modernize-use-bool-literals")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-modernize-use-default-member-init") # should be used...
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-modernize-use-transparent-functors")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-modernize-use-using") # should be used...
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-readability-braces-around-statements")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-readability-else-after-return")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-readability-function-size")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-readability-identifier-naming")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-readability-named-parameter")
    set(CLANG_TIDY_CHECKS "${CLANG_TIDY_CHECKS},-readability-redundant-member-init") # https://bugs.llvm.org/show_bug.cgi?id=32966
    set(CLANG_TIDY_ARGS "-checks='${CLANG_TIDY_CHECKS}'" CACHE STRING "Argument list given to clang-tidy")
    mark_as_advanced(CLANG_TIDY_ARGS USE_CLANG_TIDY_IN_BUILD)
    add_custom_target(tidy
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running clang-tidy on all '$<CONFIG>' targets..."
    )
    set_target_properties(tidy
        PROPERTIES
            EXCLUDE_FROM_ALL
                TRUE
    )
endif()

set_eval(USE_SOSPD_EVAL (${USE_OPENGM} AND ${USE_BOOST}))
set(USE_SOSPD ${USE_SOSPD_EVAL} CACHE INTERNAL "")

initialize_internal_list(litiv_projects)
initialize_internal_list(litiv_3rdparty_modules)
initialize_internal_list(litiv_3rdparty_modules_sourcelist)
initialize_internal_list(litiv_modules)
initialize_internal_list(litiv_modules_sourcelist)
initialize_internal_list(litiv_apps)
initialize_internal_list(litiv_samples)
initialize_internal_list(litiv_tests)
add_subdirectory(cmake/checks/simd)
add_definitions(-DLITIV_BUILD)
if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    option(BUILD_SHARED_LIBS "Build shared libraries (.so) instead of static ones (.a)" ON)
    option(USE_ASAN_RELEASE "Enable address sanitizer in release builds" OFF)
    option(USE_ASAN_DEBUG "Enable address sanitizer in debug builds" OFF)
    if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=900") # already default for gcc, but not for clang
    endif()
    if(NOT CMAKE_CROSSCOMPILING)
        add_definitions(-march=native)
        if(USE_NEON)
            add_definitions(-mfpu=neon)
        endif()
    endif()
    if(USE_FAST_MATH)
        add_definitions(-ffast-math)
    endif()
    if(USE_ASAN_RELEASE)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -fsanitize=address")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fsanitize=address")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_LINKER_FLAGS_RELEASE "${CMAKE_LINKER_FLAGS_RELEASE} -fsanitize=address")
        endif()
    endif()
    if(USE_ASAN_DEBUG)
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fsanitize=address")
        endif()
    endif()
    add_definitions(-Wall)
    add_definitions(-Wfatal-errors)
    check_function_exists(aligned_alloc USE_STL_ALIGNED_ALLOC)
    if(NOT USE_STL_ALIGNED_ALLOC)
        set(USE_STL_ALIGNED_ALLOC 0 CACHE INTERNAL "Have function aligned_alloc")
    endif()
    check_function_exists(posix_memalign USE_POSIX_ALIGNED_ALLOC)
    if(NOT USE_POSIX_ALIGNED_ALLOC)
        set(USE_POSIX_ALIGNED_ALLOC 0 CACHE INTERNAL "Have function posix_memalign")
    endif()
    if(USE_LINK_TIME_OPTIM)
        add_definitions(-flto)
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
          find_program(GCC_AR gcc-ar)
          if(GCC_AR)
            set(CMAKE_AR ${GCC_AR})
          endif()
          find_program(GCC_RANLIB gcc-ranlib)
          if(GCC_RANLIB)
            set(CMAKE_RANLIB ${GCC_RANLIB})
          endif()
        endif()
    endif()
    if(USE_DEBUG_SYMBOLS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    endif()
elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    if(MSVC_VERSION LESS 1900)
        message(FATAL_ERROR "MSVC toolchains older than 2015 (v140) are not supported!")
    endif()
    #set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
    set(CMAKE_DEBUG_POSTFIX "d" CACHE INTERNAL "" FORCE)
    option(USE_VERSION_TAGS "Apply version tags suffixes on built libraries" ON)
    option(BUILD_SHARED_LIBS "Build shared libraries (.dll) instead of static ones (.lib)" OFF)
    if(BUILD_SHARED_LIBS)
        message("LITIV DLLs are still missing symbol exports, and might be incomplete for some targets.")
    endif()
    add_definitions(-DUNICODE -D_UNICODE)
    if(USE_FAST_MATH)
        add_definitions(/fp:fast)
    else(NOT USE_FAST_MATH)
        add_definitions(/fp:precise)
    endif()
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/W1)
    add_definitions(/arch:AVX) # check performance difference? vs 387? @@@
    if(USE_LINK_TIME_OPTIM)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
        set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    endif()
    if(USE_INLINE_INTRINSIC_FUNCS)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Oi")
    endif()
    if(USE_DEBUG_SYMBOLS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(FATAL_ERROR "Intel compiler still unsupported; please edit the main CMakeList.txt file to add proper configuration.")
    # ... @@@
else()
    message(FATAL_ERROR "Unknown compiler; please edit the main CMakeList.txt file to add proper configuration.")
endif()
if(NOT USE_SSE4_1 AND USE_3RDPARTY_OPTFLOW_KROEGER2016)
    message("SSE4 disabled, cannot use ofdis 3rdparty library (missing impl).")
    set(USE_3RDPARTY_OPTFLOW_KROEGER2016 OFF CACHE INTERNAL "" FORCE)
endif()
set(USE_OFDIS ${USE_3RDPARTY_OPTFLOW_KROEGER2016} CACHE INTERNAL "")

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

include(FrameworkPackGen)
