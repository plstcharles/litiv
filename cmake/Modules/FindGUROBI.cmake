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
#### Taken from http://www.openflipper.org/svnrepo/CoMISo/trunk/CoMISo/cmake/FindGUROBI.cmake
#
# FindGUROBI.cmake -- Try to find GUROBI
#
# Once done this will define
#   GUROBI_FOUND - System has Gurobi
#   GUROBI_INCLUDE_DIRS - The Gurobi include directories
#   GUROBI_LIBRARIES - The libraries needed to use Gurobi

if(GUROBI_INCLUDE_DIR)
    # in cache already
    set(GUROBI_FOUND TRUE)
    set(GUROBI_INCLUDE_DIRS "${GUROBI_INCLUDE_DIR}" )
    set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY};${GUROBI_LIBRARY}" )
else(GUROBI_INCLUDE_DIR)

    find_path(GUROBI_INCLUDE_DIR
        NAMES
            gurobi_c++.h
        HINTS
            "$ENV{GUROBI_HOME}/include"
            "$ENV{GUROBI_ROOT}/include"
            "/Library/gurobi651/mac64/include"
            "C:\\libs\\gurobi651\\include"
            "/opt/gurobi/include"
            "/opt/gurobi/gurobi651/include"
            "/opt/gurobi651/include"
    )

    find_library(GUROBI_LIBRARY
        NAMES
            gurobi
            gurobi45
            gurobi46
            gurobi50
            gurobi51
            gurobi52
            gurobi55
            gurobi60
            gurobi65
        PATHS
            "$ENV{GUROBI_HOME}/lib"
            "$ENV{GUROBI_ROOT}/lib"
            "/Library/gurobi651/mac64/lib"
            "C:\\libs\\gurobi651\\lib"
            "/opt/gurobi/lib"
            "/opt/gurobi/gurobi651/lib"
            "/opt/gurobi651/lib"
    )

    if(MSVC)
        STRING(REGEX REPLACE "/VC/bin/.*" "" VISUAL_STUDIO_PATH ${CMAKE_C_COMPILER})
        STRING(REGEX MATCH "Studio [0-9]+" VISUAL_STUDIO_VERSION ${VISUAL_STUDIO_PATH})
        STRING(REGEX REPLACE "Studio " "" VISUAL_STUDIO_VERSION ${VISUAL_STUDIO_VERSION})
        if(VISUAL_STUDIO_VERSION STREQUAL "10")
            set(VISUAL_STUDIO_YEAR "2010")
        elseif(VISUAL_STUDIO_VERSION STREQUAL "11")
            set(VISUAL_STUDIO_YEAR "2012")
        elseif(VISUAL_STUDIO_VERSION STREQUAL "12")
            set(VISUAL_STUDIO_YEAR "2013")
        elseif(VISUAL_STUDIO_VERSION STREQUAL "14")
            set(VISUAL_STUDIO_YEAR "2015")
        else()
            message(FATAL_ERROR "Unsupported MSVC compiler version: ${VISUAL_STUDIO_VERSION}")
        endif()
        set(GUROBI_LIB_NAME gurobi_c++md${VISUAL_STUDIO_YEAR})
    else()
        set(GUROBI_LIB_NAME gurobi_c++)
    endif()

    find_library(GUROBI_CXX_LIBRARY
        NAMES
            ${GUROBI_LIB_NAME}
        PATHS
            "$ENV{GUROBI_HOME}/lib"
            "$ENV{GUROBI_ROOT}/lib"
            "/Library/gurobi651/mac64/lib"
            "C:\\libs\\gurobi651\\lib"
            "/opt/gurobi/lib"
            "/opt/gurobi/gurobi651/lib"
            "/opt/gurobi651/lib"
    )

    set(GUROBI_INCLUDE_DIRS "${GUROBI_INCLUDE_DIR}" )
    set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY};${GUROBI_LIBRARY}" )

    # use c++ headers as default
    # set(GUROBI_COMPILER_FLAGS "-DIL_STD" CACHE STRING "Gurobi Compiler Flags")

    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set LIBGUROBI_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARY GUROBI_CXX_LIBRARY GUROBI_INCLUDE_DIR)
    mark_as_advanced(GUROBI_INCLUDE_DIR GUROBI_LIBRARY GUROBI_CXX_LIBRARY)

endif(GUROBI_INCLUDE_DIR)
