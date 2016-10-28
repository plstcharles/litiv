
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

litiv_module(utils)

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/defines.hpp)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/utils/defines.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/defines.hpp"
)

add_files(SOURCE_FILES
    "src/platform.cpp"
    "src/opencv.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/utils/console.hpp"
    "include/litiv/utils/cxx.hpp"
    "include/litiv/utils/defines.hpp"
    "include/litiv/utils/distances.hpp"
    "include/litiv/utils/parallel.hpp"
    "include/litiv/utils/platform.hpp"
    "include/litiv/utils/opencv.hpp"
    "include/litiv/utils.hpp"
)
if(USE_GLSL)
    add_files(SOURCE_FILES
        "src/opengl-imgproc.cpp"
        "src/opengl-shaders.cpp"
        "src/opengl-draw.cpp"
        "src/opengl.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/utils/opengl-imgproc.hpp"
        "include/litiv/utils/opengl-draw.hpp"
        "include/litiv/utils/opengl-shaders.hpp"
        "include/litiv/utils/opengl.hpp"
    )
endif(USE_GLSL)

add_library(litiv_utils STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
set_target_properties(litiv_utils PROPERTIES FOLDER "modules")
#target_compile_definitions(litiv_utils PRIVATE BUILDING_DLL)
#if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
#    # disables C4251 + C4275 to allow STL/template classes to be used in exported classes/members
#    # need to eliminate these using pImpl idiom in exported classes to add abstraction layer @@@@
#    target_compile_options(litiv_utils PUBLIC /wd4251) # disables C4251, "'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'"
#    target_compile_options(litiv_utils PUBLIC /wd4275) # disables C4275, "non DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'"
#endif()
target_include_directories(litiv_utils
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
        "$<INSTALL_INTERFACE:include/>"
        "$<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>"
)
target_link_libraries(litiv_utils
    ${OpenCV_LIBRARIES}
)
if(USE_GLSL)
    target_include_directories(litiv_utils
        PUBLIC
            "$<BUILD_INTERFACE:${OpenGL_INCLUDE_DIRS}>"
            "$<BUILD_INTERFACE:${GLEW_INCLUDE_DIRS}>"
            "$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>"
    )
    target_link_libraries(litiv_utils
        ${OPENGL_LIBRARIES}
        ${GLEW_LIBRARIES}
        ${GLM_LIBRARIES}
    )
    if(USE_GLFW)
        target_include_directories(litiv_utils
            PUBLIC
                "$<BUILD_INTERFACE:${GLFW_INCLUDE_DIR}>"
        )
        target_link_libraries(litiv_utils
            ${GLFW_LIBRARIES}
        )
    elseif(USE_FREEGLUT)
        target_include_directories(litiv_utils
            PUBLIC
                "$<BUILD_INTERFACE:${FREEGLUT_INCLUDE_DIR}>"
        )
        target_link_libraries(litiv_utils
            ${FREEGLUT_LIBRARY}
        )
    endif()
endif()
if(KinectSDK2_FOUND)
    target_include_directories(litiv_utils
        PUBLIC
            "$<BUILD_INTERFACE:${KinectSDK2_INCLUDE_DIRS}>"
    )
    target_link_libraries(litiv_utils
        ${KinectSDK2_LIBRARIES}
    )
endif()
if(UNIX AND NOT ANDROID AND NOT (APPLE AND CMAKE_COMPILER_IS_CLANGCXX))
    # build fails without this for some platforms... broken external dependency? (see PR#5)
    target_link_libraries(litiv_utils
        pthread
    )
endif()

install(TARGETS litiv_utils
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT modules
)
install(DIRECTORY "include/"
    DESTINATION "include/"
    COMPONENT modules
    FILES_MATCHING PATTERN "*.hpp"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/defines.hpp"
    DESTINATION "include/litiv/utils/"
    COMPONENT modules
)
