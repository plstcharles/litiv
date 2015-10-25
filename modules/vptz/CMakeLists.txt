
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
    option(BUILD_VPTZ "Build VPTZ library; requires full OpenGL support" ON)
endif(USE_GLSL)
if(BUILD_VPTZ)
    if(NOT ${USE_GLSL})
        message(ERROR "VirtualPTZ library requires full OpenGL support")
    endif(NOT ${USE_GLSL})
    option(USE_VPTZ_STANDALONE "Build VPTZ library as standalone lib from LITIV framework" ON)
    if(USE_VPTZ_STANDALONE)
        project(litiv_vptz)
        set(LITIV_CURRENT_MODULE_NAME vptz)
        set(LITIV_CURRENT_PROJECT_NAME litiv_vptz)
    else(NOT USE_VPTZ_STANDALONE)
        litiv_module(vptz)
    endif(USE_VPTZ_STANDALONE)

    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/StandaloneUtils.hpp)

    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/vptz/StandaloneUtils.hpp.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/vptz/StandaloneUtils.hpp"
    )

    add_files(SOURCE_FILES
        "src/virtualptz.cpp"
    )

    add_files(INCLUDE_FILES
        "include/litiv/vptz/StandaloneUtils.hpp"
        "include/litiv/vptz/virtualptz.hpp"
        "include/litiv/vptz.hpp"
    )

    if(USE_VPTZ_STANDALONE)
        add_definitions(-DVPTZ_EXPORT)
        add_library(${LITIV_CURRENT_PROJECT_NAME} SHARED ${SOURCE_FILES} ${INCLUDE_FILES})
        target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${OpenCV_LIBRARIES})
        if(USE_GLFW)
            target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${GLFW_LIBRARIES})
        elseif(USE_FREEGLUT)
            target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${FREEGLUT_LIBRARY})
        endif(USE_GLFW)
        target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${OPENGL_LIBRARIES})
        target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${GLEW_LIBRARIES})
    else(NOT USE_VPTZ_STANDALONE)
        add_library(${LITIV_CURRENT_PROJECT_NAME} STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
        target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} litiv_utils)
    endif(USE_VPTZ_STANDALONE)
    target_include_directories(${LITIV_CURRENT_PROJECT_NAME}
        PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
        PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
        INTERFACE "$<INSTALL_INTERFACE:include/>"
    )
    set_target_properties(${LITIV_CURRENT_PROJECT_NAME} PROPERTIES FOLDER "modules")
    #install(TARGETS ${LITIV_CURRENT_PROJECT_NAME} DESTINATION lib)
    install(FILES ${INCLUDE_FILES} DESTINATION include)

endif(BUILD_VPTZ)
