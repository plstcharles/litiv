
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
        list(APPEND ${list_name}_TMP "${prefix}${l}${suffix}")
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
    if(USE_CUDA AND (${ARGC} EQUAL 7)) # got cuda_sourcelist and cuda_headerlist; will copy internally
        set(cuda_sourcelist_internal "${${ARGV5}}")
        if(${ARGV6})
            set(cuda_headerlist_internal "${${ARGV6}}")
        else()
            set(cuda_headerlist_internal "")
        endif()
    elseif((NOT USE_CUDA) OR (${ARGC} EQUAL 5)) # no cuda_sourcelist and cuda_headerlist; will init as empty
        set(cuda_sourcelist_internal "")
        set(cuda_headerlist_internal "")
    else()
        message(FATAL_ERROR "bad number of args passed to litiv_library")
    endif()
    if(${groupname} STREQUAL "module")
        project(litiv_${libname})
        if((NOT(${libname} STREQUAL "world")) AND (NOT(${libname} STREQUAL "test")))
            append_internal_list(litiv_modules ${libname})
            foreach(source ${${sourcelist}})
                append_internal_list(litiv_modules_sourcelist
                    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/${source}"
                )
            endforeach()
        endif()
    else()
        project(litiv_${groupname}_${libname})
        append_internal_list(litiv_${groupname}_modules ${libname})
        foreach(source ${${sourcelist}})
            append_internal_list(litiv_${groupname}_modules_sourcelist
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/${source}"
            )
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
        if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR
           ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
           ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
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
    else()
        if(NOT CMAKE_CROSSCOMPILING)
            target_compile_options(${PROJECT_NAME}
                INTERFACE
                    "-march=native"
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
    if(USE_CUDA AND cuda_sourcelist_internal)
        if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR
           ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
           ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
            cuda_add_library(${PROJECT_NAME}_cuda
                "${cuda_sourcelist_internal};${cuda_headerlist_internal}"
                STATIC
                OPTIONS
                    "-Xcompiler -fPIC"
            )
        else()
            cuda_add_library(${PROJECT_NAME}_cuda
                "${cuda_sourcelist_internal};${cuda_headerlist_internal}"
                STATIC
            )
        endif()
        if(CUDA_EXIT_ON_ERROR)
            target_compile_definitions(${PROJECT_NAME}_cuda
                PRIVATE
                    CUDA_EXIT_ON_ERROR
            )
        endif()
        target_include_directories(${PROJECT_NAME}_cuda
            PRIVATE
                "${OpenCV_INCLUDE_DIRS}" # for opencv cuda utils in core module only
                "${CMAKE_SOURCE_DIR}/modules/utils/include" # for litiv/utils/cuda.hpp only
        )
        target_link_libraries(${PROJECT_NAME}
            PUBLIC
                ${PROJECT_NAME}_cuda
        )
        target_include_directories(${PROJECT_NAME}
            PRIVATE
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/cuda>"
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cuda>"
        )
        set(project_install_targets
            ${project_install_targets}
            ${PROJECT_NAME}_cuda
        )
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
    if(NOT(${libname} STREQUAL "test"))
        if(BUILD_TESTS AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test"))
            file(GLOB testfiles ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)
            if(USE_CUDA AND (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/cuda"))
                file(GLOB cuda_cpp_testfiles ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda/*.cpp)
                list(APPEND testfiles ${cuda_cpp_testfiles})
                #file(GLOB cuda_cu_testfiles ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda/*.cu)
                # @@@@ add new cuda execs if cuda_cu_testfiles contains sources?
            endif()
            foreach(testfile ${testfiles})
                get_filename_component(testname "${testfile}" NAME_WE)
                set(testname "${libname}_${testname}")
                set(TEST_CURR_INPUT_DATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/test/data/")
                append_internal_list(litiv_tests "${testname}")
                add_executable(litiv_utest_app_${testname} "${testfile}")
                target_compile_definitions(litiv_utest_app_${testname}
                    PUBLIC
                        PERFTEST=0
                        UNITTEST=1
                        "TEST_CURR_INPUT_DATA_ROOT=\"${TEST_CURR_INPUT_DATA_ROOT}\""
                )
                target_link_libraries(litiv_utest_app_${testname}
                    ${PROJECT_NAME}
                    litiv_test
                    gtest_main
                )
                if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
                    target_compile_options(litiv_utest_app_${testname}
                        PRIVATE "/bigobj"
                    )
                endif()
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
                if(BUILD_TESTS_PERF)
                    add_executable(litiv_ptest_app_${testname} "${testfile}")
                    target_compile_definitions(litiv_ptest_app_${testname}
                        PUBLIC
                            PERFTEST=1
                            UNITTEST=0
                            "TEST_CURR_INPUT_DATA_ROOT=\"${TEST_CURR_INPUT_DATA_ROOT}\""
                    )
                    target_link_libraries(litiv_ptest_app_${testname}
                        ${PROJECT_NAME}
                        litiv_test
                        benchmark_main
                    )
                    if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
                        target_compile_options(litiv_ptest_app_${testname}
                            PRIVATE "/bigobj"
                        )
                    endif()
                    set_target_properties(litiv_ptest_app_${testname}
                        PROPERTIES
                            FOLDER "tests/${libname}"
                            DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}"
                    )
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
    endif()
endmacro(litiv_library)

macro(litiv_module name sourcelist headerlist)
    if(USE_CUDA AND (${ARGC} EQUAL 5)) # ok, expect cuda_sourcelist and cuda_headerlist
        litiv_library(${name} "module" TRUE ${sourcelist} ${headerlist} ${ARGV3} ${ARGV4})
    else()
        litiv_library(${name} "module" TRUE ${sourcelist} ${headerlist})
    endif()
endmacro(litiv_module)

macro(litiv_3rdparty_module name sourcelist headerlist)
    if(USE_CUDA AND (${ARGC} EQUAL 5)) # ok, expect cuda_sourcelist and cuda_headerlist
        litiv_library(${name} "3rdparty" FALSE ${sourcelist} ${headerlist} ${ARGV3} ${ARGV4})
    else()
        litiv_library(${name} "3rdparty" FALSE ${sourcelist} ${headerlist})
    endif()
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
    target_link_libraries(${PROJECT_NAME} PUBLIC litiv_world)
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

function(get_link_libraries output_list target)
    list(APPEND visited_targets ${target})
    get_target_property(target_libs ${target} INTERFACE_LINK_LIBRARIES)
    set(output_list_tmp "")
    if(NOT ("${target_libs}" STREQUAL "target_libs-NOTFOUND"))
        foreach(lib ${target_libs})
            if(TARGET ${lib})
                list(FIND visited_targets ${lib} visited)
                if(${visited} EQUAL -1)
                    get_link_libraries(linked_libs ${lib})
                    list(APPEND output_list_tmp ${linked_libs})
                endif()
            else()
                if(${lib} MATCHES "^-l")
                    string(SUBSTRING ${lib} 2 -1 lib)
                endif()
                string(STRIP ${lib} lib)
                list(FIND output_list_tmp ${lib} found_new)
                list(FIND output_list ${lib} found_old)
                if((${found_new} EQUAL -1) AND (${found_old} EQUAL -1))
                    list(APPEND output_list_tmp ${lib})
                endif()
            endif()
        endforeach()
    else()
        list(APPEND output_list_tmp ${target})
    endif()
    list(REMOVE_DUPLICATES output_list_tmp)
    set(visited_targets ${visited_targets} PARENT_SCOPE)
    set(${output_list} ${output_list_tmp} PARENT_SCOPE)
endfunction()