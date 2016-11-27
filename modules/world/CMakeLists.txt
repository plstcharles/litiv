
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

file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/litiv_modules.hpp")
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
option(USE_WORLD_SOURCE_GLOB "Compile all LITIV modules as part of the world module instead of soft-linking them" OFF)
mark_as_advanced(USE_WORLD_SOURCE_GLOB)
set(litiv_3rdparty_module_names ${litiv_3rdparty_modules})
xfix_list_tokens(litiv_3rdparty_module_names "litiv_3rdparty_" "")
list(LENGTH litiv_3rdparty_module_names litiv_3rdparty_module_count)
set(litiv_module_names ${litiv_modules})
xfix_list_tokens(litiv_module_names "litiv_" "")
list(LENGTH litiv_module_names litiv_module_count)
math(EXPR total_module_count "${litiv_3rdparty_module_count}+${litiv_module_count}")
if(USE_WORLD_SOURCE_GLOB)
    # remove super-strict compiler defs at this level, as we are using 3rdparty sources in here, and everything gets compiled twice anyway
    if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
        remove_definitions(-Wextra)
        remove_definitions(-Wshadow)
        remove_definitions(-Werror)
        remove_definitions(-pedantic-errors)
    elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
        remove_definitions(/W4)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # ... @@@
    endif()
    set(WORLD_SOURCE_FILES "${SOURCE_FILES};${litiv_modules_sourcelist};${litiv_3rdparty_modules_sourcelist};")
    list(LENGTH WORLD_SOURCE_FILES world_source_count)
    litiv_module(world WORLD_SOURCE_FILES INCLUDE_FILES)
    message(STATUS "litiv_world will pull ${world_source_count} source files from ${total_module_count} modules...")
    foreach(litiv_3rdparty_module_name ${litiv_3rdparty_module_names})
        add_dependencies(litiv_world "${litiv_3rdparty_module_name}")
        get_property(libs
            TARGET
                ${litiv_3rdparty_module_name}
            PROPERTY
                INTERFACE_LINK_LIBRARIES
        )
        foreach(lib ${libs})
            if(NOT(";${litiv_3rdparty_module_names};" MATCHES ";${lib};") AND NOT(";${litiv_module_names};" MATCHES ";${lib};"))
                #message("module '${litiv_3rdparty_module_name}' had lib '${lib}'")
                target_link_libraries(litiv_world
                    PUBLIC
                        "${lib}"
                )
            endif()
        endforeach()
        get_property(includedirs
            TARGET
                ${litiv_3rdparty_module_name}
            PROPERTY
                INCLUDE_DIRECTORIES
        )
        target_include_directories(litiv_world
            PUBLIC
                "$<BUILD_INTERFACE:${includedirs}>"
        )
    endforeach()
    foreach(litiv_module_name ${litiv_module_names})
        add_dependencies(litiv_world "${litiv_module_name}")
        get_property(libs
            TARGET
                ${litiv_module_name}
            PROPERTY
                INTERFACE_LINK_LIBRARIES
        )
        foreach(lib ${libs})
            if(NOT(";${litiv_3rdparty_module_names};" MATCHES ";${lib};") AND NOT(";${litiv_module_names};" MATCHES ";${lib};"))
                #message("module '${litiv_module_name}' had lib '${lib}'")
                target_link_libraries(litiv_world
                    PUBLIC
                        "${lib}"
                )
            endif()
        endforeach()
        get_property(includedirs
            TARGET
                ${litiv_module_name}
            PROPERTY
                INCLUDE_DIRECTORIES
        )
        target_include_directories(litiv_world
            PUBLIC
                "$<BUILD_INTERFACE:${includedirs}>"
        )
    endforeach()
else()
    litiv_module(world SOURCE_FILES INCLUDE_FILES)
    message(STATUS "litiv_world will have soft links to ${total_module_count} modules...")
    target_link_libraries(litiv_world
        PUBLIC
            "${litiv_3rdparty_module_names}"
            "${litiv_module_names}"
    )
endif()
foreach(litiv_3rdparty_module_name IN LISTS litiv_3rdparty_module_names)
    message(STATUS " - ${litiv_3rdparty_module_name}")
endforeach()
foreach(litiv_module_name IN LISTS litiv_module_names)
    message(STATUS " - ${litiv_module_name}")
endforeach()