
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

litiv_module(datasets)

option(USE_BSDS500_BENCHMARK "Link & use BSDS500 dataset benchmark implementation in litiv_datasets (will use an approximative solution otherwise)" ON)

add_files(SOURCE_FILES
    "src/datasets.cpp"
    "src/eval.cpp"
    "src/utils.cpp"
    "src/metrics.cpp"
    "src/impl/BSDS500.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/datasets.hpp"
    "include/litiv/datasets/eval.hpp"
    "include/litiv/datasets/utils.hpp"
    "include/litiv/datasets/metrics.hpp"
    "include/litiv/datasets/impl/all.hpp"
    "include/litiv/datasets/impl/BSDS500.hpp"
    "include/litiv/datasets/impl/CDnet.hpp"
    "include/litiv/datasets/impl/LITIV2012b.hpp"
    "include/litiv/datasets/impl/PETS2001.hpp"
    "include/litiv/datasets/impl/Wallflower.hpp"
)

add_library(${LITIV_CURRENT_PROJECT_NAME} STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} litiv_utils litiv_imgproc)
if(USE_BSDS500_BENCHMARK)
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} BSDS500)
endif()
target_link_litiv_dependencies(${LITIV_CURRENT_PROJECT_NAME})
if(UNIX)
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} pthread)
endif()
target_include_directories(${LITIV_CURRENT_PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
set_target_properties(${LITIV_CURRENT_PROJECT_NAME} PROPERTIES FOLDER "modules")

install(TARGETS ${LITIV_CURRENT_PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT modules
)
install(DIRECTORY "include/"
    DESTINATION "include/"
    COMPONENT modules
    FILES_MATCHING PATTERN "*.hpp"
)
