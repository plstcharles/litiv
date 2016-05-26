
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

if(WIN32) # project based on Kinectv2 SDK & DirectShow for FLIR A40
    find_package(KinectSDK2)
    option(BUILD_CAPTURE_APP "Sets whether the RGBDT stereo capture app should be built or not (win32 only due to Microsoft SDK dependencies)" ${KinectSDK2_FOUND})
    if(BUILD_CAPTURE_APP)
        find_package(KinectSDK2 REQUIRED)
        include_directories(${KinectSDK2_INCLUDE_DIRS})
        link_directories(${KinectSDK2_LIBRARY_DIRS})
        project(capture_rgbdt)
        add_executable(capture_rgbdt src/main.cpp)
        target_link_libraries(capture_rgbdt litiv_world DShowBase ${KinectSDK2_LIBRARIES})
        set_target_properties(capture_rgbdt PROPERTIES FOLDER "apps")
        install(TARGETS capture_rgbdt RUNTIME DESTINATION bin COMPONENT apps)
    endif()
endif()