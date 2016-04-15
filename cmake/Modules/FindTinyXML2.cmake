
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
# FindTinyXML2.cmake
#
# This module defines the following variables:
#   TinyXML2_INCLUDE_DIRS
#   TinyXML2_LIBRARIES
#   TinyXML2_FOUND

find_path(TinyXML2_INCLUDE_DIR
    NAMES
        tinyxml2.h
    HINTS
        "${TinyXML2_ROOT_DIR}/include/"
        "$ENV{TINYXML2_ROOT}/include/"
        "$ENV{USER_DEVELOP}/tinyxml2/include/"
        "$ENV{USER_DEVELOP}/vendor/tinyxml2/include/"
)

find_library(TinyXML2_LIBRARY
    NAMES
        tinyxml2
    HINTS
        "${TinyXML2_LIBRARY_DIR}"
        "${TinyXML2_ROOT_DIR}/lib/"
        "${TinyXML2_ROOT_DIR}/build/lib/"
        "${TinyXML2_ROOT_DIR}/build/lib/Release/"
        "$ENV{TINYXML2_ROOT}/lib/"
        "$ENV{TINYXML2_ROOT}/build/lib/"
        "$ENV{TINYXML2_ROOT}/build/lib/Release/"
        "$ENV{USER_DEVELOP}/tinyxml2/build/Release/"
        "$ENV{USER_DEVELOP}/vendor/tinyxml2/build/Release/"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TinyXML2
    REQUIRED_VARS
        TinyXML2_INCLUDE_DIR
        TinyXML2_LIBRARY
)

if(TinyXML2_FOUND)
    set(TinyXML2_INCLUDE_DIRS ${TinyXML2_INCLUDE_DIR})
    set(TinyXML2_LIBRARIES ${TinyXML2_LIBRARY})
    mark_as_advanced(
        TinyXML2_INCLUDE_DIR
        TinyXML2_INCLUDE_DIRS
        TinyXML2_LIBRARY
        TinyXML2_LIBRARIES
    )
endif()
