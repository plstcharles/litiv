
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

macro(litiv_library libname groupname canbeshared sourcelist headerlist)
    if(${groupname} STREQUAL "module")
        project(litiv_${libname})
        if(NOT(${libname} STREQUAL "world"))
            append_internal_list(litiv_modules ${libname})
            foreach(source ${${sourcelist}})
                append_internal_list(litiv_modules_sourcelist "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/${source}")
            endforeach()
        endif()
    else()
        project(litiv_${groupname}_${libname})
        append_internal_list(litiv_${groupname}_modules ${libname})
        foreach(source ${${sourcelist}})
            append_internal_list(litiv_${groupname}_modules_sourcelist "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/${source}")
        endforeach()
        set(libname ${groupname}_${libname})
    endif()
    append_internal_list(litiv_projects ${PROJECT_NAME})
    set(project_install_targets "${PROJECT_NAME}")
    if(${BUILD_SHARED_LIBS} AND ${canbeshared})
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
        if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
            target_compile_options(${PROJECT_NAME}
                PRIVATE
                    -fPIC
            )
        endif()
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
            FOLDER "${groupname}"
    )
    target_compile_definitions(${PROJECT_NAME}
        PUBLIC
            "LITIV_DEBUG=$<CONFIG:Debug>"
    )
    target_include_directories(${PROJECT_NAME}
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    )
    if((${groupname} STREQUAL "module") AND NOT(${libname} STREQUAL "utils"))
        target_link_libraries(${PROJECT_NAME} PUBLIC litiv_utils)
    endif()
    if(USE_CUDA AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda"))
        file(GLOB cudasources ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu)
        if(cudasources)
            # needed here since cuda_add_library ignores target properties (i.e. include dirs)
            #cuda_include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
            #cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
            #cuda_include_directories(${CMAKE_BINARY_DIR}/modules/utils/cuda)
            #cuda_include_directories(${CMAKE_SOURCE_DIR}/modules/utils/cuda)
            #cuda_include_directories($<$<BOOL:$<TARGET_PROPERTY:litiv_utils,INTERFACE_INCLUDE_DIRECTORIES>>:$<JOIN:$<TARGET_PROPERTY:litiv_utils,INTERFACE_INCLUDE_DIRECTORIES>,;>>)
            cuda_include_directories(${CMAKE_SOURCE_DIR}/modules/utils/include) # for litiv/utils/cuda.hpp only
            file(GLOB cudaheaders ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cuh)
            if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
                cuda_add_library(${PROJECT_NAME}_cuda "${cudasources};${cudaheaders}" STATIC OPTIONS "-Xcompiler -fPIC")
            else()
                cuda_add_library(${PROJECT_NAME}_cuda "${cudasources};${cudaheaders}" STATIC)
            endif()
            target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_cuda)
            target_include_directories(${PROJECT_NAME}
                PRIVATE
                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/cuda>"
                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda>"
            )
            set(project_install_targets ${project_install_targets} ${PROJECT_NAME}_cuda)
            if(WIN32)
                if(USE_VERSION_TAGS)
                    set_target_properties(${PROJECT_NAME}_cuda
                        PROPERTIES
                            OUTPUT_NAME "${PROJECT_NAME}_cuda${LITIV_VERSION_PLAIN}"
                    )
                endif()
            endif()
            set_target_properties(${PROJECT_NAME}_cuda
                PROPERTIES
                    FOLDER "${groupname}/cuda"
            )
        endif()
    endif()
    if(BUILD_TESTS AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test"))
        file(GLOB testfiles ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
        foreach(testfile ${testfiles})
            get_filename_component(testname "${testfile}" NAME_WE)
            set(testname "${libname}_${testname}")
            append_internal_list(litiv_tests "${testname}")
            add_executable(litiv_utest_app_${testname} "${testfile}")
            target_compile_definitions(litiv_utest_app_${testname}
                PUBLIC
                    PERFTEST=0
            )
            target_link_libraries(litiv_utest_app_${testname}
                ${PROJECT_NAME}
                benchmark gtest gtest_main
                ${CMAKE_THREAD_LIBS_INIT}
            )
            set_target_properties(litiv_utest_app_${testname}
                PROPERTIES
                    FOLDER "tests/${libname}"
                    DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
            )
            set(utestbinname "litiv_utest_app_${testname}$<$<CONFIG:Debug>:${CMAKE_DEBUG_POSTFIX}>")
            add_test(
                NAME
                    "litiv_utest_${testname}"
                COMMAND
                    "${CMAKE_BINARY_DIR}/bin/${utestbinname}"
                    "--gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/${utestbinname}.xml"
            )
            add_executable(litiv_ptest_app_${testname} "${testfile}")
            target_compile_definitions(litiv_ptest_app_${testname}
                PUBLIC
                    PERFTEST=1
            )
            target_link_libraries(litiv_ptest_app_${testname}
                ${PROJECT_NAME}
                gtest benchmark benchmark_main
                ${CMAKE_THREAD_LIBS_INIT}
            )
            set_target_properties(litiv_ptest_app_${testname}
                PROPERTIES
                    FOLDER "tests/${libname}"
                    DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
            )
            if(BUILD_TESTS_PERF)
                set(ptestbinname "litiv_ptest_app_${testname}$<$<CONFIG:Debug>:${CMAKE_DEBUG_POSTFIX}>")
                add_test(
                    NAME
                        "litiv_ptest_${testname}"
                    COMMAND
                        "${CMAKE_BINARY_DIR}/bin/${ptestbinname}"
                        "--benchmark_format=console"
                        "--benchmark_out_format=console"
                        "--benchmark_out=${CMAKE_BINARY_DIR}/Testing/${ptestbinname}.txt"
                )
            endif()
        endforeach()
    endif()
    install(
        TARGETS ${project_install_targets}
        EXPORT "litiv-targets"
        COMPONENT "${groupname}"
        RUNTIME DESTINATION "bin"
        LIBRARY DESTINATION "lib"
        ARCHIVE DESTINATION "lib"
        INCLUDES DESTINATION "include"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
        DESTINATION "."
        COMPONENT "${groupname}"
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.hxx"
            PATTERN "*.h"
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include"
        DESTINATION "."
        COMPONENT "${groupname}"
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.hxx"
            PATTERN "*.h"
    )
endmacro(litiv_library)

macro(litiv_module name sourcelist headerlist)
    litiv_library(${name} "module" TRUE ${sourcelist} ${headerlist})
endmacro(litiv_module)

macro(litiv_3rdparty_module name sourcelist headerlist)
    litiv_library(${name} "3rdparty" FALSE ${sourcelist} ${headerlist})
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
