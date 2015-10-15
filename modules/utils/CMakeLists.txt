
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

litiv_module(utils)

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/DefineUtils.hpp)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/DefineUtils.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/DefineUtils.hpp"
)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/${LITIV_CURRENT_MODULE_NAME}/*.hpp"
    "litiv/${LITIV_CURRENT_MODULE_NAME}/*.ipp"
)

add_library(${LITIV_CURRENT_PROJECT_NAME} STATIC ${SOURCE_FILES})

option(USE_BSDS500_BENCHMARK "Link & use BSDS500 dataset benchmark impl in litiv_utils" ON)
if(USE_BSDS500_BENCHMARK)
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} BSDS500)
endif(USE_BSDS500_BENCHMARK)
target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} litiv_imgproc) # needed for thinning in BSDS500 eval
target_link_litiv_dependencies(${LITIV_CURRENT_PROJECT_NAME})
target_include_directories(${LITIV_CURRENT_PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

#install(TARGETS ${LITIV_CURRENT_PROJECT_NAME} DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
