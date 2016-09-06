
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
#
# FindOpenGM.cmake
#
# This module defines the following variables:
#   OpenGM_INCLUDE_DIRS
#   OpenGM_LIBRARIES      (may be empty)
#   OpenGM_FOUND

macro(_set_eval name)
    if(${ARGN})
        set(${name} 1)
    else(NOT ${ARGN})
        set(${name} 0)
    endif()
endmacro(_set_eval)

find_path(OpenGM_INCLUDE_DIR
    NAMES
        opengm/opengm.hxx
    HINTS
        "${OpenGM_ROOT_DIR}/include/"
        "$ENV{OPENGM_ROOT}/include/"
        "$ENV{USER_DEVELOP}/OpenGM/include/"
        "$ENV{USER_DEVELOP}/vendor/OpenGM/include/"
)

include(FindPackageHandleStandardArgs)

set(_supported_components gurobi cplex hdf5 ext)
set(_opengm_gurobi_default OFF)
set(_opengm_cplex_default OFF)
set(_opengm_hdf5_default OFF)
set(_opengm_ext_default OFF)
foreach(_comp ${OpenGM_FIND_COMPONENTS})
    if(NOT ";${_supported_components};" MATCHES ";${_comp};")
        message(WARNING "Specified unsupported OpenGM component: ${_comp}")
    elseif("${_comp}" STREQUAL gurobi)
        set(_opengm_gurobi_default ON)
    elseif("${_comp}" STREQUAL cplex)
        set(_opengm_cplex_default ON)
    elseif("${_comp}" STREQUAL hdf5)
        set(_opengm_hdf5_default ON)
    elseif("${_comp}" STREQUAL ext)
        set(_opengm_ext_default ON)
    endif()
endforeach()

option(USE_OPENGM_WITH_EXTLIB "Specifies whether OpenGM should be linked with its external/3rd-party library." ${_opengm_ext_default})
if(USE_OPENGM_WITH_EXTLIB)
    find_library(OpenGM_EXT_LIBRARY
        NAMES
            "opengm-external"
        HINTS
            "${OpenGM_INCLUDE_DIR}/../lib/"
            "${OpenGM_INCLUDE_DIR}/../lib64/"
            "${OpenGM_LIBRARY_DIR}"
            "${OpenGM_ROOT_DIR}/lib/"
            "${OpenGM_ROOT_DIR}/build/lib/"
            "${OpenGM_ROOT_DIR}/build/lib/Release/"
            "$ENV{OPENGM_ROOT}/lib/"
            "$ENV{OPENGM_ROOT}/build/lib/"
            "$ENV{OPENGM_ROOT}/build/lib/Release/"
            "$ENV{USER_DEVELOP}/opengm/build/Release/"
            "$ENV{USER_DEVELOP}/vendor/opengm/build/Release/"
    )
    find_package_handle_standard_args(OpenGM
        REQUIRED_VARS
            OpenGM_INCLUDE_DIR
            OpenGM_EXT_LIBRARY
    )
    list(APPEND OpenGM_LIBRARIES OpenGM_EXT_LIBRARY)
else()
    set(OpenGM_EXT_LIBRARY "")
    find_package_handle_standard_args(OpenGM
        REQUIRED_VARS
            OpenGM_INCLUDE_DIR
    )
    message("Will use OpenGM without its external dependencies, some inference algos might be disabled.")
endif()

if(OpenGM_FOUND)
    set(OpenGM_INCLUDE_DIRS ${OpenGM_INCLUDE_DIR})
    set(OpenGM_LIBRARIES ${OpenGM_EXT_LIBRARY})

    # @@@ might be able to deduce USE_OPENGM_WITH... vals using header presence

    find_package(CPLEX QUIET)
    _set_eval(_opengm_cplex_current (${_opengm_cplex_default} OR ${CPLEX_FOUND}))
    option(USE_OPENGM_WITH_CPLEX "Specifies whether OpenGM was built with IBM CPLEX support or not" ${_opengm_cplex_current})
    if(USE_OPENGM_WITH_CPLEX)
        find_package(CPLEX REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${CPLEX_INCLUDE_DIRS})
        list(APPEND OpenGM_LIBRARIES ${CPLEX_LIBRARIES})
    else()
        message("Will use OpenGM without IBM CPLEX support, some inference algos might be disabled.")
    endif()

    find_package(GUROBI QUIET)
    _set_eval(_opengm_gurobi_current (${_opengm_gurobi_default} OR ${GUROBI_FOUND}))
    option(USE_OPENGM_WITH_GUROBI "Specifies whether OpenGM was built with GUROBI support or not" ${_opengm_gurobi_current})
    if(USE_OPENGM_WITH_GUROBI)
        find_package(GUROBI REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${GUROBI_INCLUDE_DIRS})
        list(APPEND OpenGM_LIBRARIES ${GUROBI_LIBRARIES})
    else()
        message("Will use OpenGM without GUROBI support, some inference algos might be disabled.")
    endif()

    find_package(HDF5 QUIET)
    _set_eval(_opengm_hdf5_current (${_opengm_hdf5_default} OR ${HDF5_FOUND}))
    option(USE_OPENGM_WITH_HDF5 "Specifies whether OpenGM was built with HDF5 support or not" ${_opengm_hdf5_current})
    if(USE_OPENGM_WITH_HDF5)
        find_package(HDF5 REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
        list(APPEND OpenGM_LIBRARIES ${HDF5_LIBRARIES})
    else()
        message("Will use OpenGM without HDF5 support, some I/O methods might be disabled.")
    endif()

    mark_as_advanced(
        OpenGM_INCLUDE_DIR
        OpenGM_INCLUDE_DIRS
        OpenGM_EXT_LIBRARY
        OpenGM_LIBRARIES
    )
    mark_as_advanced(CLEAR
        USE_OPENGM_WITH_EXTLIB
    )
else()
    mark_as_advanced(
        USE_OPENGM_WITH_EXTLIB
    )
endif()
