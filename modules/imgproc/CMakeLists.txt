
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

litiv_module(imgproc)

add_files(SOURCE_FILES
    "src/EdgeDetectionUtils.cpp"
    "src/EdgeDetectorCanny.cpp"
    "src/EdgeDetectorLBSP.cpp"
    "src/imgproc.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/imgproc/EdgeDetectionUtils.hpp"
    "include/litiv/imgproc/EdgeDetectorCanny.hpp"
    "include/litiv/imgproc/EdgeDetectorLBSP.hpp"
    "include/litiv/imgproc.hpp"
)

add_library(litiv_imgproc STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
set_target_properties(litiv_imgproc PROPERTIES FOLDER "modules")
#target_compile_definitions(litiv_imgproc PRIVATE BUILDING_DLL)
target_include_directories(litiv_imgproc
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
        "$<INSTALL_INTERFACE:include/>"
)
target_link_libraries(litiv_imgproc litiv_features2d)

install(TARGETS litiv_imgproc
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
