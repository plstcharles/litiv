
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

litiv_3rdparty_module(bsds500)

add_files(SOURCE_FILES
    "src/kofn.cpp"
    "src/csa.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/3rdparty/bsds500/csa.hpp"
    "include/litiv/3rdparty/bsds500/csa_defs.hpp"
    "include/litiv/3rdparty/bsds500/csa_types.hpp"
    "include/litiv/3rdparty/bsds500/kofn.hpp"
    "include/litiv/3rdparty/bsds500/match.hpp"
)

add_library(litiv_bsds500 STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
set_target_properties(litiv_bsds500 PROPERTIES FOLDER "3rdparty")
target_include_directories(litiv_bsds500
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
        "$<INSTALL_INTERFACE:include/>"
)

install(TARGETS litiv_bsds500
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT 3rdparty
)
install(DIRECTORY "include/"
    DESTINATION "include/"
    COMPONENT 3rdparty
    FILES_MATCHING PATTERN "*.hpp"
)
