
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

if(WIN32 AND KinectSDK2_FOUND) # only for WIN32 due to Kinectv2SDK & DirectShow

    litiv_app(capture)

    add_executable(litiv_app_capture src/main.cpp)
    set_target_properties(litiv_app_capture PROPERTIES FOLDER "apps")
    target_include_directories(litiv_app_capture
        PUBLIC
            "$<BUILD_INTERFACE:${KinectSDK2_INCLUDE_DIRS}>"
    )
    target_link_libraries(litiv_app_capture
        litiv_world
        litiv_dshowbase
        ${KinectSDK2_LIBRARIES}
    )

    install(TARGETS litiv_app_capture
        RUNTIME DESTINATION bin
        COMPONENT apps
    )

endif()
