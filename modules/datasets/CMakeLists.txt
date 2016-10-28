
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

option(USE_BSDS500_BENCHMARK "Link & use BSDS500 dataset benchmark implementation in litiv_datasets (cleaned 3rd party code; will use an approximative solution otherwise)" ON)

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
    "include/litiv/datasets/impl/vap-trimod2016.hpp"
    "include/litiv/datasets/impl/Wallflower.hpp"
)

add_library(litiv_datasets STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
set_target_properties(litiv_datasets PROPERTIES FOLDER "modules")
#target_compile_definitions(litiv_datasets PRIVATE BUILDING_DLL)
target_include_directories(litiv_datasets
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
        "$<INSTALL_INTERFACE:include/>"
)
target_link_libraries(litiv_datasets litiv_imgproc)
if(USE_BSDS500_BENCHMARK)
    target_link_libraries(litiv_datasets litiv_bsds500)
endif()

install(TARGETS litiv_datasets
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
