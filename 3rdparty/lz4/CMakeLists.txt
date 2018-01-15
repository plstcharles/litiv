
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2018 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

if(NOT USE_LZ4)
    return()
endif()

if(CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED 1")
endif()

if((NOT DEFINED lz4_SOURCE_DIR) OR (NOT EXISTS ${lz4_SOURCE_DIR}))
    download_project(
        PROJ
            lz4
        PREFIX
            "${CMAKE_CURRENT_BINARY_DIR}"
        URL
            "https://github.com/lz4/lz4/archive/v1.8.1.2.zip"
        QUIET
    #    ${UPDATE_DISCONNECTED_IF_AVAILABLE} # bugged in some cmake versions, needs patch
    )
    set(lz4_SOURCE_DIR ${lz4_SOURCE_DIR} CACHE INTERNAL "")
    set(lz4_BINARY_DIR ${lz4_BINARY_DIR} CACHE INTERNAL "")
endif()

set(LZ4_FOUND ON CACHE BOOL "")
set(LZ4_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib" CACHE PATH "")
set(LZ4_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib" CACHE PATH "")
set(LZ4_VERSION "1.8.1.2" CACHE STRING "")
mark_as_advanced(LZ4_FOUND LZ4_INCLUDE_DIR LZ4_INCLUDE_DIRS LZ4_VERSION)

add_files(SOURCE_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4.c"
	"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4frame.c"
	"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4hc.c"
	"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/xxhash.c"
)

add_files(INCLUDE_FILES
    "${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4.h"
	"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4frame.h"
	#"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4frame_static.h"
	"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4hc.h"
	"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/lz4opt.h"
	"${CMAKE_CURRENT_BINARY_DIR}/lz4-src/lib/xxhash.h"
)

litiv_3rdparty_module(lz4 SOURCE_FILES INCLUDE_FILES)

target_include_directories(litiv_3rdparty_lz4
    PUBLIC
        "$<BUILD_INTERFACE:${LZ4_INCLUDE_DIR}>"
)