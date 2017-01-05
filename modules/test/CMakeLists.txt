
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

if(BUILD_TESTS)

    add_files(SOURCE_FILES
        "src/test.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/test.hpp"
    )

    litiv_library(test "module" FALSE SOURCE_FILES INCLUDE_FILES)
    target_include_directories(litiv_test
        PRIVATE
            "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/modules/utils/include>"
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/modules/utils/include>"
    )
    target_link_libraries(litiv_test
        PUBLIC
            gtest
            benchmark
            ${CMAKE_THREAD_LIBS_INIT}
    )
    
endif()