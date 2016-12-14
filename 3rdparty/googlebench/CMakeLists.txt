
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

if(NOT BUILD_TESTS)
    return()
endif()

if(CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

string(REPLACE " " "\\ " CMAKE_COMMAND_ESCAPED ${CMAKE_COMMAND})

set(BENCHMARK_ENABLE_LTO ${USE_LINK_TIME_OPTIM} CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_USE_LIBCXX OFF CACHE BOOL "" FORCE)
mark_as_advanced(
    BENCHMARK_ENABLE_LTO
    BENCHMARK_ENABLE_TESTING
    BENCHMARK_USE_LIBCXX
)

download_project(
    PROJ
        googlebench
    PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}"
    URL
        "https://github.com/google/benchmark/archive/v1.1.0.zip"
    PATCH_COMMAND
        "${CMAKE_COMMAND_ESCAPED} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/changes-v1.1.0 ${CMAKE_CURRENT_BINARY_DIR}/googlebench-src"
    QUIET
#    ${UPDATE_DISCONNECTED_IF_AVAILABLE} # bugged in some cmake versions, needs patch
)

add_subdirectory(
    ${googlebench_SOURCE_DIR}
    ${googlebench_BINARY_DIR}
    EXCLUDE_FROM_ALL
)

add_library(benchmark_main STATIC "src/main.cpp")
target_link_libraries(benchmark_main
    benchmark
    ${CMAKE_THREAD_LIBS_INIT}
)
set_target_properties(benchmark_main
    PROPERTIES
        FOLDER "3rdparty/googlebench"
        DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
        EXCLUDE_FROM_ALL ON
)
set_target_properties(benchmark
    PROPERTIES
        FOLDER "3rdparty/googlebench"
        EXCLUDE_FROM_ALL ON
)