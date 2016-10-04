# Copyright (C) 2013 Bjoern Andres, Thorsten Beier and Joerg H.~Kappes.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
#
# FindCPLEX.cmake -- This module finds cplex.
#
# User can give CPLEX_ROOT_DIR as a hint stored in the cmake cache.
#
# It sets the following variables:
#  CPLEX_FOUND              - Set to false, or undefined, if cplex isn't found.
#  CPLEX_INCLUDE_DIRS       - include directory
#  CPLEX_LIBRARIES          - library files

if(WIN32)
    execute_process(COMMAND cmd /C set CPLEX_STUDIO_DIR OUTPUT_VARIABLE CPLEX_STUDIO_DIR_VAR ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT CPLEX_STUDIO_DIR_VAR)
        MESSAGE(FATAL_ERROR "Unable to find CPLEX: environment variable CPLEX_STUDIO_DIR<VERSION> not set.")
    endif()
    STRING(REGEX REPLACE "^CPLEX_STUDIO_DIR" "" CPLEX_STUDIO_DIR_VAR ${CPLEX_STUDIO_DIR_VAR})
    STRING(REGEX MATCH "^[0-9]+" CPLEX_WIN_VERSION ${CPLEX_STUDIO_DIR_VAR})
    STRING(REGEX REPLACE "^[0-9]+=" "" CPLEX_STUDIO_DIR_VAR ${CPLEX_STUDIO_DIR_VAR})
    file(TO_CMAKE_PATH "${CPLEX_STUDIO_DIR_VAR}" CPLEX_ROOT_DIR_GUESS)
    set(CPLEX_WIN_VERSION ${CPLEX_WIN_VERSION} CACHE STRING "CPLEX version to be used.")
    set(CPLEX_ROOT_DIR "${CPLEX_ROOT_DIR_GUESS}" CACHE PATH "CPLEX root directory.")
    MESSAGE(STATUS "Found CLPEX version ${CPLEX_WIN_VERSION} at '${CPLEX_ROOT_DIR}'")
    STRING(REGEX REPLACE "/VC/bin/.*" "" VISUAL_STUDIO_PATH ${CMAKE_C_COMPILER})
    STRING(REGEX MATCH "Studio [0-9]+" CPLEX_WIN_VS_VERSION ${VISUAL_STUDIO_PATH})
    STRING(REGEX REPLACE "Studio " "" CPLEX_WIN_VS_VERSION ${CPLEX_WIN_VS_VERSION})
    if(CPLEX_WIN_VS_VERSION STREQUAL "10")
        set(CPLEX_WIN_VS_VERSION "2010")
    elseif(CPLEX_WIN_VS_VERSION STREQUAL "11")
        set(CPLEX_WIN_VS_VERSION "2012")
    elseif(CPLEX_WIN_VS_VERSION STREQUAL "12")
        set(CPLEX_WIN_VS_VERSION "2013")
    elseif(CPLEX_WIN_VS_VERSION STREQUAL "14")
        set(CPLEX_WIN_VS_VERSION "2015")
    else()
        message(FATAL_ERROR "Unsupported MSVC compiler version: ${CPLEX_WIN_VS_VERSION}")
    endif()
    set(CPLEX_WIN_VS_VERSION ${CPLEX_WIN_VS_VERSION} CACHE STRING "Visual Studio Version")
    if("${CMAKE_C_COMPILER}" MATCHES "amd64")
        set(CPLEX_WIN_BITNESS x64)
    else()
        set(CPLEX_WIN_BITNESS x86)
    endif()
    set(CPLEX_WIN_BITNESS ${CPLEX_WIN_BITNESS} CACHE STRING "On Windows: x86 or x64 (32bit resp. 64bit)")
    MESSAGE(STATUS "CPLEX: using Visual Studio ${CPLEX_WIN_VS_VERSION} ${CPLEX_WIN_BITNESS} at '${VISUAL_STUDIO_PATH}'")
    if(NOT CPLEX_WIN_LINKAGE)
        set(CPLEX_WIN_LINKAGE mda CACHE STRING "CPLEX linkage variant on Windows. One of these: mda (dll, release), mdd (dll, debug), mta (static, release), mtd (static, debug)")
    endif(NOT CPLEX_WIN_LINKAGE)
    set(CPLEX_WIN_PLATFORM "${CPLEX_WIN_BITNESS}_windows_vs${CPLEX_WIN_VS_VERSION}/stat_${CPLEX_WIN_LINKAGE}")
else()
    set(CPLEX_ROOT_DIR "" CACHE PATH "CPLEX root directory.")
    set(CPLEX_WIN_PLATFORM "")
endif()

FIND_PATH(CPLEX_INCLUDE_DIR
    NAMES
        ilcplex/cplex.h
    HINTS
        ${CPLEX_ROOT_DIR}/cplex/include
        ${CPLEX_ROOT_DIR}/include
        $ENV{CPLEX_ROOT}/cplex/include
        $ENV{CPLEX_ROOT}/include
    PATHS
        ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
)

FIND_PATH(CPLEX_CONCERT_INCLUDE_DIR
    NAMES
        ilconcert/iloenv.h
    HINTS
        ${CPLEX_ROOT_DIR}/concert/include
        ${CPLEX_ROOT_DIR}/include
        $ENV{CPLEX_ROOT}/concert/include
        $ENV{CPLEX_ROOT}/include
        $ENV{CPLEX_ROOT}/../concert/include
    PATHS
        ENV C_INCLUDE_PATH
        ENV C_PLUS_INCLUDE_PATH
        ENV INCLUDE_PATH
)

FIND_LIBRARY(CPLEX_LIBRARY
    NAMES
        cplex
        cplex${CPLEX_WIN_VERSION}
    HINTS
        ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM}
        ${CPLEX_ROOT_DIR}/lib/${CPLEX_WIN_PLATFORM}
        $ENV{CPLEX_ROOT}/cplex/lib/${CPLEX_WIN_PLATFORM}
        $ENV{CPLEX_ROOT}/lib/${CPLEX_WIN_PLATFORM}
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_debian4.0_4.1/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_debian4.0_4.1/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_debian4.0_4.1/static_pic
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_sles10_4.1/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_sles10_4.1/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_sles10_4.1/static_pic
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_linux/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_linux/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_linux/static_pic
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_osx/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_osx/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_osx/static_pic
    PATHS
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
)
message(STATUS "CPLEX Library: ${CPLEX_LIBRARY}")

FIND_LIBRARY(CPLEX_ILOCPLEX_LIBRARY
    NAMES
        ilocplex
    HINTS
        ${CPLEX_ROOT_DIR}/cplex/lib/${CPLEX_WIN_PLATFORM}
        ${CPLEX_ROOT_DIR}/lib/${CPLEX_WIN_PLATFORM}
        $ENV{CPLEX_ROOT}/cplex/lib/${CPLEX_WIN_PLATFORM}
        $ENV{CPLEX_ROOT}/lib/${CPLEX_WIN_PLATFORM}
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_debian4.0_4.1/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_debian4.0_4.1/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_debian4.0_4.1/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_debian4.0_4.1/static_pic
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_sles10_4.1/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_sles10_4.1/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_sles10_4.1/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_sles10_4.1/static_pic
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_linux/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_linux/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_linux/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_linux/static_pic
        ${CPLEX_ROOT_DIR}/cplex/lib/x86-64_osx/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_osx/static_pic
        $ENV{CPLEX_ROOT}/cplex/lib/x86-64_osx/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_osx/static_pic
    PATHS
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
)
message(STATUS "ILOCPLEX Library: ${CPLEX_ILOCPLEX_LIBRARY}")

FIND_LIBRARY(CPLEX_CONCERT_LIBRARY
    NAMES
        concert
    HINTS
        ${CPLEX_ROOT_DIR}/concert/lib/${CPLEX_WIN_PLATFORM}
        ${CPLEX_ROOT_DIR}/../concert/lib/${CPLEX_WIN_PLATFORM}
        ${CPLEX_ROOT_DIR}/lib/${CPLEX_WIN_PLATFORM}
        $ENV{CPLEX_ROOT}/concert/lib/${CPLEX_WIN_PLATFORM}
        $ENV{CPLEX_ROOT}/../concert/lib/${CPLEX_WIN_PLATFORM}
        $ENV{CPLEX_ROOT}/lib/${CPLEX_WIN_PLATFORM}
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_debian4.0_4.1/static_pic
        ${CPLEX_ROOT_DIR}/../concert/lib/x86-64_debian4.0_4.1/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_debian4.0_4.1/static_pic
        $ENV{CPLEX_ROOT}/concert/lib/x86-64_debian4.0_4.1/static_pic
        $ENV{CPLEX_ROOT}/../concert/lib/x86-64_debian4.0_4.1/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_debian4.0_4.1/static_pic
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_sles10_4.1/static_pic
        ${CPLEX_ROOT_DIR}/../concert/lib/x86-64_sles10_4.1/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_sles10_4.1/static_pic
        $ENV{CPLEX_ROOT}/concert/lib/x86-64_sles10_4.1/static_pic
        $ENV{CPLEX_ROOT}/../concert/lib/x86-64_sles10_4.1/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_sles10_4.1/static_pic
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_linux/static_pic
        ${CPLEX_ROOT_DIR}/../concert/lib/x86-64_linux/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_linux/static_pic
        $ENV{CPLEX_ROOT}/concert/lib/x86-64_linux/static_pic
        $ENV{CPLEX_ROOT}/../concert/lib/x86-64_linux/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_linux/static_pic
        ${CPLEX_ROOT_DIR}/concert/lib/x86-64_osx/static_pic
        ${CPLEX_ROOT_DIR}/../concert/lib/x86-64_osx/static_pic
        ${CPLEX_ROOT_DIR}/lib/x86-64_osx/static_pic
        $ENV{CPLEX_ROOT}/concert/lib/x86-64_osx/static_pic
        $ENV{CPLEX_ROOT}/../concert/lib/x86-64_osx/static_pic
        $ENV{CPLEX_ROOT}/lib/x86-64_osx/static_pic
    PATHS
        ENV LIBRARY_PATH
        ENV LD_LIBRARY_PATH
)
message(STATUS "CONCERT Library: ${CPLEX_CONCERT_LIBRARY}")

if(WIN32)
    FIND_PATH(CPLEX_BIN_DIR
        NAMES
            cplex${CPLEX_WIN_VERSION}.dll
        HINTS
            ${CPLEX_ROOT_DIR}/cplex/bin/${CPLEX_WIN_PLATFORM}
            ${CPLEX_ROOT_DIR}/bin/${CPLEX_WIN_PLATFORM}
            $ENV{CPLEX_ROOT}/cplex/bin/${CPLEX_WIN_PLATFORM}
            $ENV{CPLEX_ROOT}/bin/${CPLEX_WIN_PLATFORM}
    )
else()
    FIND_PATH(CPLEX_BIN_DIR
        NAMES
            cplex
        HINTS
            ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_sles10_4.1
            ${CPLEX_ROOT_DIR}/bin/x86-64_sles10_4.1
            ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_debian4.0_4.1
            ${CPLEX_ROOT_DIR}/bin/x86-64_debian4.0_4.1
            ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_linux
            ${CPLEX_ROOT_DIR}/bin/x86-64_linux
            ${CPLEX_ROOT_DIR}/cplex/bin/x86-64_osx
            ${CPLEX_ROOT_DIR}/bin/x86-64_osx
            $ENV{CPLEX_ROOT}/cplex/bin/x86-64_sles10_4.1
            $ENV{CPLEX_ROOT}/bin/x86-64_sles10_4.1
            $ENV{CPLEX_ROOT}/cplex/bin/x86-64_debian4.0_4.1
            $ENV{CPLEX_ROOT}/bin/x86-64_debian4.0_4.1
            $ENV{CPLEX_ROOT}/cplex/bin/x86-64_linux
            $ENV{CPLEX_ROOT}/bin/x86-64_linux
            $ENV{CPLEX_ROOT}/cplex/bin/x86-64_osx
            $ENV{CPLEX_ROOT}/bin/x86-64_osx
        PATHS
            ENV LIBRARY_PATH
            ENV LD_LIBRARY_PATH
    )
endif()
message(STATUS "CPLEX Bin Dir: ${CPLEX_BIN_DIR}")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CPLEX DEFAULT_MSG
    CPLEX_LIBRARY
    CPLEX_INCLUDE_DIR
    CPLEX_ILOCPLEX_LIBRARY
    CPLEX_CONCERT_LIBRARY
    CPLEX_CONCERT_INCLUDE_DIR
)

IF(CPLEX_FOUND)
    SET(CPLEX_INCLUDE_DIRS ${CPLEX_INCLUDE_DIR} ${CPLEX_CONCERT_INCLUDE_DIR})
    SET(CPLEX_LIBRARIES ${CPLEX_CONCERT_LIBRARY} ${CPLEX_ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY} )
    IF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        SET(CPLEX_LIBRARIES "${CPLEX_LIBRARIES};m;pthread")
    ENDIF(CMAKE_SYSTEM_NAME STREQUAL "Linux")
ENDIF(CPLEX_FOUND)

MARK_AS_ADVANCED(CPLEX_LIBRARY CPLEX_INCLUDE_DIR CPLEX_ILOCPLEX_LIBRARY CPLEX_CONCERT_INCLUDE_DIR CPLEX_CONCERT_LIBRARY)
