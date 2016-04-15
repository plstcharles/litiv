# FindGLEW.cmake -- Finds the OpenGL Extension Wrangler Library (GLEW)
#
# This module defines the :prop_tgt:`IMPORTED` target ``GLEW::GLEW``,
# if GLEW has been found.
#
# This module defines the following variables:
#   GLEW_INCLUDE_DIRS - include directories for GLEW
#   GLEW_LIBRARIES - libraries to link against GLEW
#   GLEW_FOUND - true if GLEW has been found and can be used

#=============================================================================
# Copyright 2012 Benjamin Eikel
#
# Distributed under the OSI-approved BSD License (the "License").
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================

find_path(GLEW_INCLUDE_DIR GL/glew.h)
find_library(GLEW_LIBRARY
    NAMES
        GLEW
        glew32
        glew
        glew32s
    HINTS
        "$ENV{USER_DEVELOP}/glew/lib/Release/x64/"
        "$ENV{USER_DEVELOP}/vendor/glew/lib/Release/x64/"
    PATH_SUFFIXES
        lib64
)

set(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIR})
set(GLEW_LIBRARIES ${GLEW_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW
    REQUIRED_VARS
        GLEW_INCLUDE_DIR
        GLEW_LIBRARY
)

if(GLEW_FOUND)
    if(NOT TARGET GLEW::GLEW)
        add_library(GLEW::GLEW UNKNOWN IMPORTED)
        set_target_properties(GLEW::GLEW PROPERTIES
            IMPORTED_LOCATION "${GLEW_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")
    endif()
    mark_as_advanced(GLEW_INCLUDE_DIR GLEW_LIBRARY)
endif()
