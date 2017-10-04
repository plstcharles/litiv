
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

if(NOT USE_SOSPD)
    return()
endif()

##################################################################################
##
## as of v1.5.0, some sospd utilities have been disabled in favor of equivalent
## ones already in opengm; the code to fetch QPBO is kept here for reference only
##
#if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/QPBO")
#    file(DOWNLOAD
#        "http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz"
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src.tar.gz"
#    )
#    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src.tar.gz"
#        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
#    )
#    file(RENAME "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src"
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO")
#    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/QPBO-v1.3.src.tar.gz")
#    file(COPY
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO.cpp"
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO_extra.cpp"
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO_maxflow.cpp"
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO_postprocessing.cpp"
#        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/src/")
#    file(COPY
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/block.h"
#        "${CMAKE_CURRENT_BINARY_DIR}/QPBO/QPBO.h"
#        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/3rdparty/QPBO/")
#endif()
#
#add_files(SOURCE_FILES
#    "src/QPBO.cpp"
#    "src/QPBO_extra.cpp"
#    "src/QPBO_maxflow.cpp"
#    "src/QPBO_postprocessing.cpp"
#)
#
#set_source_files_properties(${SOURCE_FILES} PROPERTIES COMPILE_FLAGS "-w")
#
#add_files(INCLUDE_FILES
#    "include/litiv/3rdparty/qpbo.hpp"
#    "include/litiv/3rdparty/QPBO/block.h"
#    "include/litiv/3rdparty/QPBO/QPBO.h"
#)

set(SOURCE_FILES "") # module is header-only as of v1.5.0

add_files(INCLUDE_FILES
    "include/litiv/3rdparty/sospd/energy-common.hpp"
    "include/litiv/3rdparty/sospd/flow-solver.hpp"
    "include/litiv/3rdparty/sospd/multilabel-energy.hpp"
    "include/litiv/3rdparty/sospd/sos-graph.hpp"
    "include/litiv/3rdparty/sospd/sospd.hpp"
    "include/litiv/3rdparty/sospd/submodular-functions.hpp"
    "include/litiv/3rdparty/sospd/submodular-ibfs.hpp"
)

litiv_3rdparty_module(sospd SOURCE_FILES INCLUDE_FILES)

target_include_directories(litiv_3rdparty_sospd
    INTERFACE
        #"$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/litiv/3rdparty/QPBO>"
        "$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>"
        #"$<INSTALL_INTERFACE:include/litiv/3rdparty/QPBO>"
)
