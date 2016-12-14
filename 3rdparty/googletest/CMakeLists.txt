
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
set(BUILD_GTEST ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(gtest_build_samples OFF CACHE BOOL "" FORCE)
set(gtest_build_tests OFF CACHE BOOL "" FORCE)
if(MINGW)
    set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
else()
    set(gtest_disable_pthreads OFF CACHE BOOL "" FORCE)
endif()
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(gtest_hide_internal_symbols OFF CACHE BOOL "" FORCE)
set(gmock_build_tests OFF CACHE BOOL "" FORCE)

mark_as_advanced(
    BUILD_GTEST
    BUILD_GMOCK
    gtest_build_samples
    gtest_build_tests
    gtest_disable_pthreads
    gtest_force_shared_crt
    gtest_hide_internal_symbols
    gmock_build_tests
)

download_project(
    PROJ
        googletest
    PREFIX
        "${CMAKE_CURRENT_BINARY_DIR}"
    URL
        "https://github.com/google/googletest/archive/release-1.8.0.zip"
    QUIET
    ${UPDATE_DISCONNECTED_IF_AVAILABLE}
)

add_subdirectory(
    ${googletest_SOURCE_DIR}
    ${googletest_BINARY_DIR}
    EXCLUDE_FROM_ALL
)

set_target_properties(gtest
    PROPERTIES
        FOLDER "3rdparty/googletest"
)
set_target_properties(gtest_main
    PROPERTIES
        FOLDER "3rdparty/googletest"
)