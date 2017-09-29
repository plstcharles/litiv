
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
#   OpenGM_LIBRARIES          (may be empty)
#   OpenGM_FOUND
#   USE_OPENGM_WITH_EXTLIB    (option)
#   USE_OPENGM_WITH_CPLEX     (option)
#   USE_OPENGM_WITH_GUROBI    (option)
#   USE_OPENGM_WITH_HDF5      (option)
#   HAVE_OPENGM_EXTLIB_FASTPD (internal)
#   HAVE_OPENGM_EXTLIB_QPBO   (internal)
#   ...

macro(_set_eval name)
    if(${ARGN})
        set(${name} 1)
    else(NOT ${ARGN})
        set(${name} 0)
    endif()
endmacro(_set_eval)

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
set(HAVE_OPENGM_EXTLIB_FASTPD OFF CACHE INTERNAL "Specifies if FastPD was found among OpenGM's external libs.")
set(HAVE_OPENGM_EXTLIB_QPBO OFF CACHE INTERNAL "Specifies if QPBO was found among OpenGM's external libs.")
if(USE_OPENGM_WITH_EXTLIB)
    find_library(OpenGM_EXT_LIBRARY
        NAMES
            "opengm-external"
        HINTS
            "${OpenGM_LIBRARYDIR}"
            "${OpenGM_LIBRARY_DIR}"
            "${OpenGM_ROOT_DIR}/build/install/lib/"
            "${OpenGM_ROOT_DIR}/build/lib/Release/"
            "${OpenGM_ROOT_DIR}/build/lib/"
            "${OpenGM_ROOT_DIR}/lib/"
            "$ENV{OPENGM_ROOT}/build/install/lib/"
            "$ENV{OPENGM_ROOT}/build/lib/Release/"
            "$ENV{OPENGM_ROOT}/build/lib/"
            "$ENV{OPENGM_ROOT}/lib/"
            "$ENV{USER_DEVELOP}/opengm/build/install/lib/"
            "$ENV{USER_DEVELOP}/opengm/build/lib/Release/"
            "$ENV{USER_DEVELOP}/opengm/build/lib/"
    )
    get_filename_component(OpenGM_LIBRARY_ROOT ${OpenGM_EXT_LIBRARY} DIRECTORY)
    find_path(OpenGM_INCLUDE_DIR
        NAMES
            opengm/opengm.hxx
        HINTS
            "${OpenGM_INCLUDEDIR}"
            "${OpenGM_LIBRARY_ROOT}/../include"
            "${OpenGM_LIBRARYDIR}/../include"
            "${OpenGM_LIBRARY_DIR}/../include"
            "${OpenGM_ROOT_DIR}/build/install/include/"
            "${OpenGM_ROOT_DIR}/include/"
            "$ENV{OPENGM_ROOT}/build/install/include/"
            "$ENV{OPENGM_ROOT}/include/"
            "$ENV{USER_DEVELOP}/opengm/build/install/include/"
            "$ENV{USER_DEVELOP}/opengm/include/"
    )
    if(NOT OpenGM_EXT_LIBRARY)
        set(OpenGM_INCLUDE_DIR "OpenGM_INCLUDE_DIR-NOTFOUND" CACHE PATH "Include directory where opengm/opengm.hxx can be found" FORCE)
    endif()
    find_package_handle_standard_args(OpenGM
        REQUIRED_VARS
            OpenGM_INCLUDE_DIR
            OpenGM_EXT_LIBRARY
    )
    list(APPEND OpenGM_LIBRARIES OpenGM_EXT_LIBRARY)

    find_path(OpenGM_EXTLIB_FASTPD_TEST NAMES opengm/inference/external/fastpd/Fast_PD.h HINTS "${OpenGM_INCLUDE_DIR}")
    if(OpenGM_EXTLIB_FASTPD_TEST)
        set(HAVE_OPENGM_EXTLIB_FASTPD ON CACHE INTERNAL "Specifies if FastPD was found among OpenGM's external libs.")
        if(NOT OpenGM_FIND_QUIETLY)
            message(STATUS "Found FastPD in OpenGM external headers.")
        endif()
    endif()
    mark_as_advanced(OpenGM_EXTLIB_FASTPD_TEST)

    find_path(OpenGM_EXTLIB_QPBO_TEST NAMES opengm/inference/external/qpbo/QPBO.h HINTS "${OpenGM_INCLUDE_DIR}")
    if(OpenGM_EXTLIB_QPBO_TEST)
        set(HAVE_OPENGM_EXTLIB_QPBO ON CACHE INTERNAL "Specifies if QPBO was found among OpenGM's external libs.")
        if(NOT OpenGM_FIND_QUIETLY)
            message(STATUS "Found QPBO in OpenGM external headers.")
        endif()
    endif()
    mark_as_advanced(OpenGM_EXTLIB_QPBO_TEST)

    # @@@ add missing external libs checks here

else()
    set(OpenGM_EXT_LIBRARY "")
    find_path(OpenGM_INCLUDE_DIR
        NAMES
            opengm/opengm.hxx
        HINTS
            "${OpenGM_INCLUDEDIR}"
            "${OpenGM_ROOT_DIR}/build/install/include/"
            "${OpenGM_ROOT_DIR}/include/"
            "$ENV{OPENGM_ROOT}/build/install/include/"
            "$ENV{OPENGM_ROOT}/include/"
            "$ENV{USER_DEVELOP}/opengm/build/install/include/"
            "$ENV{USER_DEVELOP}/opengm/include/"
    )
    find_package_handle_standard_args(OpenGM
        REQUIRED_VARS
            OpenGM_INCLUDE_DIR
    )
    if(NOT OpenGM_FIND_QUIETLY)
        message(STATUS "Will use OpenGM without its external dependencies, some inference algos might be disabled.")
    endif()
endif()

if(OpenGM_FOUND)
    set(OpenGM_INCLUDE_DIRS ${OpenGM_INCLUDE_DIR})
    set(OpenGM_LIBRARIES ${OpenGM_EXT_LIBRARY})

    find_package(CPLEX QUIET)
    _set_eval(_opengm_cplex_current (${_opengm_cplex_default} OR ${CPLEX_FOUND}))
    option(USE_OPENGM_WITH_CPLEX "Specifies whether OpenGM was built with IBM CPLEX support or not" ${_opengm_cplex_current})
    if(USE_OPENGM_WITH_CPLEX)
        find_package(CPLEX REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${CPLEX_INCLUDE_DIRS})
        list(APPEND OpenGM_LIBRARIES ${CPLEX_LIBRARIES})
    elseif(NOT OpenGM_FIND_QUIETLY)
        message(STATUS "Will use OpenGM without IBM CPLEX support, some inference algos might be disabled.")
    endif()

    find_package(GUROBI QUIET)
    _set_eval(_opengm_gurobi_current (${_opengm_gurobi_default} OR ${GUROBI_FOUND}))
    option(USE_OPENGM_WITH_GUROBI "Specifies whether OpenGM was built with GUROBI support or not" ${_opengm_gurobi_current})
    if(USE_OPENGM_WITH_GUROBI)
        find_package(GUROBI REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${GUROBI_INCLUDE_DIRS})
        list(APPEND OpenGM_LIBRARIES ${GUROBI_LIBRARIES})
    elseif(NOT OpenGM_FIND_QUIETLY)
        message(STATUS "Will use OpenGM without GUROBI support, some inference algos might be disabled.")
    endif()

    find_package(HDF5 QUIET)
    _set_eval(_opengm_hdf5_current (${_opengm_hdf5_default} OR ${HDF5_FOUND}))
    option(USE_OPENGM_WITH_HDF5 "Specifies whether OpenGM was built with HDF5 support or not" ${_opengm_hdf5_current})
    if(USE_OPENGM_WITH_HDF5)
        find_package(HDF5 REQUIRED)
        list(APPEND OpenGM_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
        list(APPEND OpenGM_LIBRARIES ${HDF5_LIBRARIES})
    elseif(NOT OpenGM_FIND_QUIETLY)
        message(STATUS "Will use OpenGM without HDF5 support, some I/O methods might be disabled.")
    endif()
    mark_as_advanced(
        HDF5_CORE_LIBRARY
        HDF5_CPP_LIBRARY
        HDF5_HL_LIBRARY
        HDF5_INCLUDE_DIR
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
