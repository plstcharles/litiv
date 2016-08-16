
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

add_library(${LITIV_CURRENT_PROJECT_NAME} STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

target_link_litiv_dependencies(${LITIV_CURRENT_PROJECT_NAME})
target_include_directories(${LITIV_CURRENT_PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
set_target_properties(${LITIV_CURRENT_PROJECT_NAME} PROPERTIES FOLDER "modules")

install(TARGETS ${LITIV_CURRENT_PROJECT_NAME}
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
