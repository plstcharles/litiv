
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

if(USE_GLSL)

    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/defines.hpp")
    configure_file(
        "${CMAKE_SOURCE_DIR}/modules/utils/include/litiv/utils/defines.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/defines.hpp"
    )
    file(REMOVE "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/utils.hpp")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/vptz/utils.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/utils.hpp"
    )

    add_files(SOURCE_FILES
        "src/virtualptz.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/vptz/defines.hpp"
        "include/litiv/vptz/utils.hpp"
        "include/litiv/vptz/virtualptz.hpp"
        "include/litiv/vptz.hpp"
    )

    if(USE_VPTZ_STANDALONE)

        project(vptz)
        add_library(vptz SHARED "${SOURCE_FILES};${INCLUDE_FILES}")
        set_target_properties(vptz
            PROPERTIES
                FOLDER "vptz"
        )
        target_compile_definitions(vptz
            PUBLIC
                "LITIV_DEBUG=$<CONFIG:Debug>"
            PRIVATE
                "LV_EXPORT_API"
            INTERFACE
                "LV_IMPORT_API"
        )
        if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
            # disables C4251 + C4275 to allow STL/template classes to be used in exported classes/members
            target_compile_options(vptz
                PUBLIC
                    # need to eliminate these using pImpl idiom in exported classes to add abstraction layer @@@@
                    /wd4251 # disables C4251, "'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'"
                    /wd4275 # disables C4275, "non DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'"
            )
        endif()
        target_include_directories(vptz
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                "$<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${OpenGL_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${GLEW_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>"
        )
        target_link_libraries(vptz
            PUBLIC
                "${OpenCV_LIBRARIES}"
                "${OPENGL_LIBRARIES}"
                "${GLEW_LIBRARIES}"
                "${GLM_LIBRARIES}"
        )
        if(USE_GLFW)
            target_include_directories(vptz
                PUBLIC
                    "$<BUILD_INTERFACE:${GLFW_INCLUDE_DIR}>"
            )
            target_link_libraries(vptz
                PUBLIC
                    "${GLFW_LIBRARIES}"
            )
        elseif(USE_FREEGLUT)
            target_include_directories(vptz
                PUBLIC
                    "$<BUILD_INTERFACE:${FREEGLUT_INCLUDE_DIR}>"
            )
            target_link_libraries(vptz
                PUBLIC
                    "${FREEGLUT_LIBRARY}"
            )
        endif()
        install(
            TARGETS vptz
            EXPORT "litiv-targets"
            COMPONENT "vptz"
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib"
            INCLUDES DESTINATION "include"
        )
        install(
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include"
            DESTINATION "."
            COMPONENT "vptz"
            FILES_MATCHING
                PATTERN "*.hpp"
                PATTERN "*.hxx"
                PATTERN "*.h"
        )
        install(
            DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include"
            DESTINATION "."
            COMPONENT "vptz"
            FILES_MATCHING
                PATTERN "*.hpp"
                PATTERN "*.hxx"
                PATTERN "*.h"
        )

    else(NOT USE_VPTZ_STANDALONE)

        litiv_module(vptz SOURCE_FILES INCLUDE_FILES)
        target_link_libraries(litiv_vptz
            PUBLIC
                litiv_utils
        )

    endif(USE_VPTZ_STANDALONE)

endif(USE_GLSL)
