
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

if(GTEST_FOUND OR (NOT BUILD_TESTS))
    return()
endif()

set(GTEST_DISABLE_PTHREADS OFF)
if(MINGW)
    set(GTEST_DISABLE_PTHREADS ON)
endif()

ExternalProject_Add(googletest
    GIT_REPOSITORY
        https://github.com/google/googletest.git
    GIT_TAG
        "release-1.8.0"
    SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/external"
    CMAKE_ARGS
        -DBUILD_GTEST=ON
        -DBUILD_GMOCK=ON
        -DBUILD_SHARED_LIBS=OFF
        -Dgtest_force_shared_crt=ON
        -Dgtest_disable_pthreads=${GTEST_DISABLE_PTHREADS}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/install
    BINARY_DIR
        "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_DIR
        ""
)

set(GTEST_ROOT "${CMAKE_CURRENT_BINARY_DIR}/install" CACHE INTERNAL "")
set(GTEST_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/install/include" CACHE INTERNAL "")
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    set(GTEST_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/install/lib/gtest.lib" CACHE INTERNAL "")
    set(GTEST_MAIN_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/install/lib/gtest_main.lib" CACHE INTERNAL "")
else()
    set(GTEST_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/install/lib/libgtest.a" CACHE INTERNAL "")
    set(GTEST_MAIN_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/install/lib/libgtest_main.a" CACHE INTERNAL "")
endif()
set(GTEST_BOTH_LIBRARIES "${GTEST_LIBRARIES}" "${GTEST_MAIN_LIBRARIES}" CACHE INTERNAL "")