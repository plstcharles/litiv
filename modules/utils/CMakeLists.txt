
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

option(USE_BSDS500_BENCHMARK "Link & use BSDS500 dataset benchmark implementation in litiv_utils (will use an approximative solution otherwise)" ON)

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/DefineUtils.hpp)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/utils/DefineUtils.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/DefineUtils.hpp"
)

add_files(SOURCE_FILES
    "src/DatasetEvalUtils.cpp"
    "src/DatasetUtils.cpp"
    "src/PlatformUtils.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/utils/ConsoleUtils.hpp"
    "include/litiv/utils/CxxUtils.hpp"
    "include/litiv/utils/DatasetEvalUtils.hpp"
    "include/litiv/utils/DatasetUtils.hpp"
    "include/litiv/utils/DefineUtils.hpp"
    "include/litiv/utils/DistanceUtils.hpp"
    "include/litiv/utils/ParallelUtils.hpp"
    "include/litiv/utils/PlatformUtils.hpp"
    "include/litiv/utils/RandUtils.hpp"
    "include/litiv/utils.hpp"
)
if(USE_GLSL)
    add_files(SOURCE_FILES
        "src/GLDrawUtils.cpp"
        "src/GLImageProcUtils.cpp"
        "src/GLShaderUtils.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/utils/GLDrawUtils.hpp"
        "include/litiv/utils/GLImageProcUtils.hpp"
        "include/litiv/utils/GLShaderUtils.hpp"
        "include/litiv/utils/GLUtils.hpp"
    )
endif(USE_GLSL)

add_library(${LITIV_CURRENT_PROJECT_NAME} STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

if(USE_BSDS500_BENCHMARK)
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} BSDS500)
endif()
target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} litiv_imgproc) # needed for thinning in BSDS500 eval
target_link_litiv_dependencies(${LITIV_CURRENT_PROJECT_NAME})
target_include_directories(${LITIV_CURRENT_PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
set_target_properties(${LITIV_CURRENT_PROJECT_NAME} PROPERTIES FOLDER "modules")
#install(TARGETS ${LITIV_CURRENT_PROJECT_NAME} DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
