
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
option(USE_OPENGM_WITH_EXTLIB "Specifies whether OpenGM should be linked with its external/3rd-party library." OFF)
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
endif()

if(OpenGM_FOUND)
    set(OpenGM_INCLUDE_DIRS ${OpenGM_INCLUDE_DIR})
    set(OpenGM_LIBRARIES ${OpenGM_EXT_LIBRARY})

    # @@@ might be able to deduce USE_OPENGM_WITH... vals using header presence

    find_package(CPLEX QUIET)
    option(USE_OPENGM_WITH_CPLEX "Specifies whether OpenGM was built with IBM CPLEX support or not" ${CPLEX_FOUND})
    if(USE_OPENGM_WITH_CPLEX)
        find_package(CPLEX REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${CPLEX_INCLUDE_DIRS})
        list(APPEND OpenGM_LIBRARIES ${CPLEX_LIBRARIES})
    else()
        message(WARNING "Will use OpenGM without IBM CPLEX support, some inference algos might be disabled.")
    endif()

    find_package(GUROBI QUIET)
    option(USE_OPENGM_WITH_GUROBI "Specifies whether OpenGM was built with GUROBI support or not" ${GUROBI_FOUND})
    if(USE_OPENGM_WITH_GUROBI)
        find_package(GUROBI REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${GUROBI_INCLUDE_DIRS})
        list(APPEND OpenGM_LIBRARIES ${GUROBI_LIBRARIES})
    else()
        message(WARNING "Will use OpenGM without GUROBI support, some inference algos might be disabled.")
    endif()

    find_package(HDF5 QUIET)
    option(USE_OPENGM_WITH_HDF5 "Specifies whether OpenGM was built with HDF5 support or not" ${HDF5_FOUND})
    if(USE_OPENGM_WITH_GUROBI)
        find_package(GUROBI REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
        list(APPEND OpenGM_LIBRARIES ${HDF5_LIBRARIES})
    else()
        message(WARNING "Will use OpenGM without HDF5 support, some I/O methods might be disabled.")
    endif()

    mark_as_advanced(
        OpenGM_INCLUDE_DIR
        OpenGM_INCLUDE_DIRS
        OpenGM_EXT_LIBRARY
        OpenGM_LIBRARIES
    )
endif()
