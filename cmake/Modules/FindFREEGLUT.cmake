
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
# FindFREEGLUT.cmake
#
# This module defines the following variables:
#   FREEGLUT_INCLUDE_DIRS
#   FREEGLUT_LIBRARIES
#   FREEGLUT_FOUND

find_path(FREEGLUT_INCLUDE_DIR
    NAMES
        GL/freeglut.h
    HINTS
        "${OPENGL_INCLUDE_DIR}"
        "${OPENGL_ROOT_DIR}/include/"
        "$ENV{OPENGL_ROOT}/include/"
        "${FREEGLUT_ROOT_DIR}/include/"
        "$ENV{FREEGLUT_ROOT}/include/"
        "$ENV{USER_DEVELOP}/freeglut/include/"
        "$ENV{USER_DEVELOP}/vendor/freeglut/include/"
)

find_library(FREEGLUT_LIBRARY
    NAMES
        freeglut_static
        freeglut
        glut
    HINTS
        "${OPENGL_LIBRARY_DIR}"
        "${OPENGL_ROOT_DIR}/lib/"
        "${OPENGL_ROOT_DIR}/build/lib/"
        "${OPENGL_ROOT_DIR}/build/lib/Release/"
        "$ENV{OPENGL_ROOT}/lib/"
        "$ENV{OPENGL_ROOT}/build/lib/"
        "$ENV{OPENGL_ROOT}/build/lib/Release/"
        "${FREEGLUT_LIBRARY_DIR}"
        "${FREEGLUT_ROOT_DIR}/lib/"
        "${FREEGLUT_ROOT_DIR}/build/lib/"
        "${FREEGLUT_ROOT_DIR}/build/lib/Release/"
        "$ENV{FREEGLUT_ROOT}/lib/"
        "$ENV{FREEGLUT_ROOT}/build/lib/"
        "$ENV{FREEGLUT_ROOT}/build/lib/Release/"
        "$ENV{USER_DEVELOP}/freeglut/build/Release/"
        "$ENV{USER_DEVELOP}/vendor/freeglut/build/Release/"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FREEGLUT
    REQUIRED_VARS
        FREEGLUT_INCLUDE_DIR
        FREEGLUT_LIBRARY
)

if(FREEGLUT_FOUND)
    set(FREEGLUT_INCLUDE_DIRS ${FREEGLUT_INCLUDE_DIR})
    set(FREEGLUT_LIBRARIES ${FREEGLUT_LIBRARY})
    mark_as_advanced(
        FREEGLUT_INCLUDE_DIR
        FREEGLUT_INCLUDE_DIRS
        FREEGLUT_LIBRARY
        FREEGLUT_LIBRARIES
    )
endif()
