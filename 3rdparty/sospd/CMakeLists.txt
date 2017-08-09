
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2017 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

if(NOT USE_OPENGM)
    return()
endif()

if(NOT USE_BOOST)
    return()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/QPBO")
    file(DOWNLOAD
        "http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz"
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src.tar.gz"
    )
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf 
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src.tar.gz"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
    file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src"
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO")
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src.tar.gz")
    file(COPY
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO_extra.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO_maxflow.cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO_postprocessing.cpp"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src/")
    file(COPY
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/block.h"
        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO.h"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/3rdparty/QPBO/")
endif()

add_files(SOURCE_FILES
    "src/QPBO.cpp"
    "src/QPBO_extra.cpp"
    "src/QPBO_maxflow.cpp"
    "src/QPBO_postprocessing.cpp"
)

set_source_files_properties(${SOURCE_FILES} PROPERTIES COMPILE_FLAGS "-w")

add_files(SOURCE_FILES
    "src/alpha-expansion.cpp"
    "src/bidirectional-ibfs.cpp"
    "src/gen-random.cpp"
    "src/parametric-ibfs.cpp"
    "src/sospd.cpp"
    "src/source-ibfs.cpp"
    "src/submodular-functions.cpp"
    "src/submodular-ibfs.cpp"
)

add_files(INCLUDE_FILES
    "include/litiv/3rdparty/qpbo.hpp"
    "include/litiv/3rdparty/QPBO/block.h"
    "include/litiv/3rdparty/QPBO/QPBO.h"
    "include/litiv/3rdparty/HOCR/HOCR.h"
    "include/litiv/3rdparty/HOCR/HOCR0.h"
    "include/litiv/3rdparty/HOCR/Image.h"
    "include/litiv/3rdparty/sospd/alpha-expansion.hpp"
    "include/litiv/3rdparty/sospd/energy-common.hpp"
    "include/litiv/3rdparty/sospd/flow-solver.hpp"
    "include/litiv/3rdparty/sospd/fusion-move.hpp"
    "include/litiv/3rdparty/sospd/generic-higher-order.hpp"
    "include/litiv/3rdparty/sospd/gen-random.hpp"
    "include/litiv/3rdparty/sospd/higher-order-energy.hpp"
    "include/litiv/3rdparty/sospd/multilabel-energy.hpp"
    "include/litiv/3rdparty/sospd/sos-graph.hpp"
    "include/litiv/3rdparty/sospd/sospd.hpp"
    "include/litiv/3rdparty/sospd/submodular-functions.hpp"
    "include/litiv/3rdparty/sospd/submodular-ibfs.hpp"
)

litiv_3rdparty_module(sospd SOURCE_FILES INCLUDE_FILES)

target_include_directories(litiv_3rdparty_sospd
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/litiv/3rdparty/QPBO>"
        "$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>"
        "$<INSTALL_INTERFACE:include/litiv/3rdparty/QPBO>"
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
)
