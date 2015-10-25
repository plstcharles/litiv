
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

project(litiv_world)

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/litiv_modules.hpp)
set(CMAKE_CONFIGURABLE_FILE_CONTENT "#pragma once\n")
foreach(l ${litiv_modules})
    set(CMAKE_CONFIGURABLE_FILE_CONTENT "${CMAKE_CONFIGURABLE_FILE_CONTENT}\n#include \"litiv/${l}.hpp\"")
endforeach(l ${${litiv_modules}})
set(CMAKE_CONFIGURABLE_FILE_CONTENT "${CMAKE_CONFIGURABLE_FILE_CONTENT}\n")
configure_file(
    "${CMAKE_ROOT}/Modules/CMakeConfigurableFile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/litiv_modules.hpp"
)
unset(CMAKE_CONFIGURABLE_FILE_CONTENT)

add_files(SOURCE_FILES
    "src/world.cpp"
)

add_files(INCLUDE_FILES
    "include/litiv/world.hpp"
)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_library(litiv_world STATIC ${SOURCE_FILES} ${INCLUDE_FILES}) # need to add API export/import config later @@@@
else(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_library(litiv_world SHARED ${SOURCE_FILES} ${INCLUDE_FILES})
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")

target_link_libraries(litiv_world ${litiv_projects})
target_include_directories(litiv_world
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
set_target_properties(litiv_world PROPERTIES FOLDER "modules")
install(TARGETS litiv_world DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
