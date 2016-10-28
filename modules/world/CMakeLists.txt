
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
set(CMAKE_CONFIGURABLE_FILE_CONTENT "${CMAKE_CONFIGURABLE_FILE_CONTENT}")
configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/CMakeConfigurableFile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/litiv_modules.hpp"
)
unset(CMAKE_CONFIGURABLE_FILE_CONTENT)

add_files(SOURCE_FILES
    "src/world.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/litiv.hpp"
)
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    add_library(litiv_world STATIC ${SOURCE_FILES} ${INCLUDE_FILES}) # need to add API export/import config later @@@@
else()
    add_library(litiv_world SHARED ${SOURCE_FILES} ${INCLUDE_FILES})
endif()
set_target_properties(litiv_world PROPERTIES FOLDER "modules")
#target_compile_definitions(litiv_world PRIVATE BUILDING_DLL)
target_include_directories(litiv_world
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
        "$<INSTALL_INTERFACE:include/>"
)
set(litiv_module_names ${litiv_modules})
xfix_list_tokens(litiv_module_names "litiv_" "")
list(LENGTH litiv_module_names litiv_module_count)
message(STATUS "litiv_world will link to ${litiv_module_count} modules...")
foreach(litiv_module_name IN LISTS litiv_module_names)
    message(STATUS " - ${litiv_module_name}")
endforeach()
target_link_libraries(litiv_world ${litiv_module_names})

install(TARGETS litiv_world
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
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/litiv_modules.hpp"
    DESTINATION "include/litiv/"
    COMPONENT modules
)
