
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

macro(litiv_module name)
    project(litiv_${name})
    append_internal_list(litiv_modules ${name})
    append_internal_list(litiv_projects litiv_${name})
    set(LITIV_CURRENT_MODULE_NAME ${name})
    set(LITIV_CURRENT_PROJECT_NAME litiv_${name})
endmacro(litiv_module)

macro(litiv_3rdparty_module name)
    project(litiv_${name})
    append_internal_list(litiv_3rdparty_modules ${name})
    append_internal_list(litiv_projects litiv_${name})
    set(LITIV_CURRENT_3RDPARTY_MODULE_NAME ${name})
    set(LITIV_CURRENT_PROJECT_NAME litiv_${name})
endmacro(litiv_3rdparty_module)

macro(litiv_app name)
    project(litiv_app_${name})
    append_internal_list(litiv_apps ${name})
    append_internal_list(litiv_projects litiv_app_${name})
    set(LITIV_CURRENT_APP_NAME ${name})
    set(LITIV_CURRENT_PROJECT_NAME litiv_app_${name})
endmacro(litiv_app)

macro(litiv_sample name)
    project(litiv_sample_${name})
    append_internal_list(litiv_samples ${name})
    append_internal_list(litiv_projects litiv_sample_${name})
    set(LITIV_CURRENT_SAMPLE_NAME ${name})
    set(LITIV_CURRENT_PROJECT_NAME litiv_sample_${name})
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
