
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2018 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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
# FindGperftools.cmake
#
# Uses 'Gperftools_ROOT_DIR' as a hint variable.
#
# This module then defines the following variables:
#    Gperftools_FOUND
#    Gperftools_LIBRARIES
#    Gperftools_INCLUDE_DIR

find_library(Gperftools_TCMALLOC
    NAMES
        tcmalloc
    HINTS
        "${Gperftools_ROOT_DIR}/lib"
        "$ENV{Gperftools_LOCATION}/lib"
        "$ENV{USER_DEVELOP}/vendor/Gperftools/lib"
        "$ENV{USER_DEVELOP}/Gperftools/lib"
    PATHS
        "$ENV{PROGRAMFILES}/Gperftools/lib"
        /usr/local/lib
        /usr/lib
)

find_library(Gperftools_PROFILER
    NAMES
        profiler
    HINTS
        "${Gperftools_ROOT_DIR}/lib"
        "$ENV{Gperftools_LOCATION}/lib"
        "$ENV{USER_DEVELOP}/vendor/Gperftools/lib"
        "$ENV{USER_DEVELOP}/Gperftools/lib"
    PATHS
        "$ENV{PROGRAMFILES}/Gperftools/lib"
        /usr/local/lib
        /usr/lib
)

find_library(Gperftools_TCMALLOC_AND_PROFILER
    NAMES
        tcmalloc_and_profiler
    HINTS
        "${Gperftools_ROOT_DIR}/lib"
        "$ENV{Gperftools_LOCATION}/lib"
        "$ENV{USER_DEVELOP}/vendor/Gperftools/lib"
        "$ENV{USER_DEVELOP}/Gperftools/lib"
    PATHS
        "$ENV{PROGRAMFILES}/Gperftools/lib"
        /usr/local/lib
        /usr/lib
)

find_path(Gperftools_INCLUDE_DIR
    NAMES
        gperftools/heap-profiler.h
    HINTS
        "${Gperftools_ROOT_DIR}/include"
        "$ENV{Gperftools_LOCATION}/include"
        "$ENV{USER_DEVELOP}/vendor/Gperftools/include"
        "$ENV{USER_DEVELOP}/Gperftools/include"
    PATHS
        "$ENV{PROGRAMFILES}/Gperftools/include"
        /usr/local/include
        /usr/include
)

set(Gperftools_LIBRARIES ${Gperftools_TCMALLOC_AND_PROFILER})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gperftools
    REQUIRED_VARS
        Gperftools_LIBRARIES
        Gperftools_INCLUDE_DIR
)

mark_as_advanced(
    Gperftools_ROOT_DIR
    Gperftools_TCMALLOC
    Gperftools_PROFILER
    Gperftools_TCMALLOC_AND_PROFILER
    Gperftools_LIBRARIES
    Gperftools_INCLUDE_DIR
)