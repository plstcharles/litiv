
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
    option(USE_VPTZ_STANDALONE "Build VPTZ library as standalone lib from LITIV framework" ON)

    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/utils.hpp)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/vptz/utils.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/utils.hpp"
    )

    add_files(SOURCE_FILES
        "src/virtualptz.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/vptz/utils.hpp"
        "include/litiv/vptz/virtualptz.hpp"
        "include/litiv/vptz.hpp"
    )

    if(USE_VPTZ_STANDALONE)

        project(vptz)

        add_library(vptz SHARED ${SOURCE_FILES} ${INCLUDE_FILES})
        set_target_properties(vptz PROPERTIES FOLDER "vptz")
        target_compile_definitions(vptz PRIVATE VPTZ_EXPORT)
        if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
            # disables C4251 + C4275 to allow STL/template classes to be used in exported classes/members
            # need to eliminate these using pImpl idiom in exported classes to add abstraction layer @@@@
            target_compile_options(vptz PUBLIC /wd4251) # disables C4251, "'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'"
            target_compile_options(vptz PUBLIC /wd4275) # disables C4275, "non DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'"
        endif()
        target_include_directories(vptz
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
                "$<INSTALL_INTERFACE:include/>"
                "$<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${OpenGL_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${GLEW_INCLUDE_DIRS}>"
                "$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>"
        )
        target_link_libraries(vptz
            ${OpenCV_LIBRARIES}
            ${OPENGL_LIBRARIES}
            ${GLEW_LIBRARIES}
            ${GLM_LIBRARIES}
        )
        if(USE_GLFW)
            target_include_directories(vptz
                PUBLIC
                    "$<BUILD_INTERFACE:${GLFW_INCLUDE_DIR}>"
            )
            target_link_libraries(vptz
                ${GLFW_LIBRARIES}
            )
        elseif(USE_FREEGLUT)
            target_include_directories(vptz
                PUBLIC
                    "$<BUILD_INTERFACE:${FREEGLUT_INCLUDE_DIR}>"
            )
            target_link_libraries(vptz
                ${FREEGLUT_LIBRARY}
            )
        endif()

        install(TARGETS vptz
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            COMPONENT vptz
        )

    else(NOT USE_VPTZ_STANDALONE)

        litiv_module(vptz)

        add_library(litiv_vptz STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
        set_target_properties(litiv_vptz PROPERTIES FOLDER "modules")
        #target_compile_definitions(litiv_vptz PRIVATE BUILDING_DLL)
        target_include_directories(litiv_vptz
            PUBLIC
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
                "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
                "$<INSTALL_INTERFACE:include/>"
        )
        target_link_libraries(litiv_vptz litiv_utils)

        install(TARGETS litiv_vptz
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            COMPONENT modules
        )
        
    endif(USE_VPTZ_STANDALONE)
    
    install(DIRECTORY "include/"
        DESTINATION "include/"
        COMPONENT modules
        FILES_MATCHING PATTERN "*.hpp"
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/utils.hpp"
        DESTINATION "include/litiv/vptz/"
        COMPONENT modules
    )

endif(USE_GLSL)
