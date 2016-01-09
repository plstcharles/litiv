
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

project(BSDS500)

add_files(SOURCE_FILES
    "src/kofn.cpp"
    "src/csa.cpp"
)

add_files(INCLUDE_FILES
    "include/3rdparty/BSDS500/csa.hpp"
    "include/3rdparty/BSDS500/csa_defs.hpp"
    "include/3rdparty/BSDS500/csa_types.hpp"
    "include/3rdparty/BSDS500/kofn.hpp"
    "include/3rdparty/BSDS500/match.hpp"
)

add_library(BSDS500 STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

target_include_directories(BSDS500
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
set_target_properties(BSDS500 PROPERTIES FOLDER "3rdparty")
#install(TARGETS BSDS500 DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
