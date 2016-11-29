
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

macro(xfix_list_tokens list_name prefix suffix)
    set(${list_name}_TMP)
    foreach(l ${${list_name}})
        list(APPEND ${list_name}_TMP ${prefix}${l}${suffix} )
    endforeach(l ${${list_name}})
    set(${list_name} "${${list_name}_TMP}")
    unset(${list_name}_TMP)
endmacro(xfix_list_tokens)

macro(append_internal_list list_name value)
    if(${list_name})
        set(${list_name} "${${list_name}};${value}" CACHE INTERNAL "Internal list variable")
    else(NOT ${list_name})
        set(${list_name} "${value}" CACHE INTERNAL "Internal list variable")
    endif()
endmacro(append_internal_list)

macro(initialize_internal_list list_name)
    set(${list_name} "" CACHE INTERNAL "Internal list variable")
endmacro(initialize_internal_list)

macro(litiv_module name sourcelist headerlist)
    project(litiv_${name})
    if(NOT(${name} STREQUAL "world"))
        append_internal_list(litiv_modules ${name})
        foreach(source ${${sourcelist}})
            append_internal_list(litiv_modules_sourcelist "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/${source}")
        endforeach()
    endif()
    append_internal_list(litiv_projects ${PROJECT_NAME})
    if(BUILD_SHARED_LIBS)
        add_library(${PROJECT_NAME} SHARED ${${sourcelist}} ${${headerlist}})
        set_target_properties(${PROJECT_NAME}
            PROPERTIES
                VERSION "${LITIV_VERSION}"
                SOVERSION "${LITIV_VERSION_ABI}"
        )
        target_compile_definitions(${PROJECT_NAME}
            PRIVATE
                "LV_EXPORT_API"
            INTERFACE
                "LV_IMPORT_API"
        )
        if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
            # disables C4251 + C4275 to allow STL/template classes to be used in exported classes/members
            # need to eliminate these using pImpl idiom in exported classes to add abstraction layer @@@@
            target_compile_options(${PROJECT_NAME}
                PUBLIC
                    /wd4251 # disables C4251, "'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'"
                    /wd4275 # disables C4275, "non DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'"
            )
        endif()
    else()
        add_library(${PROJECT_NAME} STATIC ${${sourcelist}} ${${headerlist}})
    endif()
    if(WIN32)
        if(USE_VERSION_TAGS)
            set_target_properties(${PROJECT_NAME}
                PROPERTIES
                    OUTPUT_NAME "${PROJECT_NAME}${LITIV_VERSION_PLAIN}"
            )
        endif()
    endif()
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            FOLDER "modules"
    )
    target_compile_definitions(${PROJECT_NAME}
        PUBLIC
            "LITIV_DEBUG=$<CONFIG:Debug>"
    )
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/include")
        target_include_directories(${PROJECT_NAME}
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        )
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_include_directories(${PROJECT_NAME}
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        )
    endif()
    install(
        TARGETS ${PROJECT_NAME}
        EXPORT "litiv-targets"
        COMPONENT "modules"
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        INCLUDES DESTINATION "include"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
        DESTINATION "."
        COMPONENT "modules"
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.hxx"
            PATTERN "*.h"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include"
        DESTINATION "."
        COMPONENT "modules"
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.hxx"
            PATTERN "*.h"
    )
endmacro(litiv_module)

macro(litiv_3rdparty_module name sourcelist headerlist)
    project(litiv_3rdparty_${name})
    if(NOT(${name} STREQUAL "world"))
        append_internal_list(litiv_3rdparty_modules ${name})
        foreach(source ${${sourcelist}})
            append_internal_list(litiv_3rdparty_modules_sourcelist "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/${source}")
        endforeach()
    endif()
    append_internal_list(litiv_projects ${PROJECT_NAME})
#    if(BUILD_SHARED_LIBS)
#        add_library(${PROJECT_NAME} SHARED ${${sourcelist}} ${${headerlist}})
#        set_target_properties(${PROJECT_NAME}
#            PROPERTIES
#                VERSION "${LITIV_VERSION}"
#                SOVERSION "${LITIV_VERSION_ABI}"
#        )
#        target_compile_definitions(${PROJECT_NAME}
#            PRIVATE
#                "LV_EXPORT_API"
#            INTERFACE
#                "LV_IMPORT_API"
#        )
#        if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
#            # disables C4251 + C4275 to allow STL/template classes to be used in exported classes/members
#            # need to eliminate these using pImpl idiom in exported classes to add abstraction layer @@@@
#            target_compile_options(${PROJECT_NAME}
#                PUBLIC
#                    /wd4251 # disables C4251, "'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'"
#                    /wd4275 # disables C4275, "non DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'"
#            )
#        endif()
#    else()
        add_library(${PROJECT_NAME} STATIC ${${sourcelist}} ${${headerlist}})
#    endif()
    if(WIN32)
        if(USE_VERSION_TAGS)
            set_target_properties(${PROJECT_NAME}
                PROPERTIES
                    OUTPUT_NAME "${PROJECT_NAME}${LITIV_VERSION_PLAIN}"
            )
        endif()
    endif()
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            FOLDER "3rdparty"
    )
    target_compile_definitions(${PROJECT_NAME}
        PUBLIC
            "LITIV_DEBUG=$<CONFIG:Debug>"
    )
    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/include")
        target_include_directories(${PROJECT_NAME}
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
        )
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_include_directories(${PROJECT_NAME}
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        )
    endif()
    install(
        TARGETS ${PROJECT_NAME}
        EXPORT "litiv-targets"
        COMPONENT "3rdparty"
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        INCLUDES DESTINATION "include"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
        DESTINATION "."
        COMPONENT "3rdparty"
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.hxx"
            PATTERN "*.h"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include"
        DESTINATION "."
        COMPONENT "3rdparty"
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.hxx"
            PATTERN "*.h"
    )
endmacro(litiv_3rdparty_module)

macro(litiv_app name sources)
    project(litiv_app_${name})
    append_internal_list(litiv_apps ${name})
    append_internal_list(litiv_projects ${PROJECT_NAME})
    add_executable(${PROJECT_NAME} ${sources})
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            FOLDER "apps"
            DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
    )
    target_link_libraries(${PROJECT_NAME} litiv_world)
    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION "bin"
        COMPONENT "apps"
    )
endmacro(litiv_app)

macro(litiv_sample name sources)
    project(litiv_sample_${name})
    append_internal_list(litiv_samples ${name})
    append_internal_list(litiv_projects ${PROJECT_NAME})
    add_executable(${PROJECT_NAME} ${sources})
    set_target_properties(${PROJECT_NAME}
        PROPERTIES
            FOLDER "samples"
            DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
    )
    target_link_libraries(${PROJECT_NAME} litiv_world)
    install(
        TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION "bin"
        COMPONENT "samples"
    )
endmacro(litiv_sample)

macro(set_eval name)
    if(${ARGN})
        set(${name} 1)
    else(NOT ${ARGN})
        set(${name} 0)
    endif()
endmacro(set_eval)

macro(add_files list_name)
    foreach(filepath ${ARGN})
        list(APPEND ${list_name} "${filepath}")
    endforeach()
endmacro()

macro(get_subdirectory_list result dir)
    file(GLOB children RELATIVE ${dir} ${dir}/*)
    set(dirlisttemp "")
    foreach(child ${children})
        if(IS_DIRECTORY ${dir}/${child})
            list(APPEND dirlisttemp ${child})
        endif()
    endforeach(child ${children})
    set(${result} ${dirlisttemp})
endmacro(get_subdirectory_list result dir)
