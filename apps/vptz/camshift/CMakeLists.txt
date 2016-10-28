
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

    if(USE_VPTZ_STANDALONE)

        project(vptz_camshift)

        add_executable(vptz_camshift src/main.cpp)
        set_target_properties(vptz_camshift PROPERTIES FOLDER "vptz")
        target_compile_definitions(vptz_camshift PRIVATE VPTZ_IMPORT)
        target_link_libraries(vptz_camshift vptz)

        install(TARGETS vptz_camshift
            RUNTIME DESTINATION bin
            COMPONENT vptz
        )

    else(NOT USE_VPTZ_STANDALONE)

        litiv_app(camshift)

        add_executable(litiv_app_camshift src/main.cpp)
        set_target_properties(litiv_app_camshift PROPERTIES FOLDER "apps/vptz")
        target_link_libraries(litiv_app_camshift litiv_vptz)

        install(TARGETS litiv_app_camshift
            RUNTIME DESTINATION bin
            COMPONENT apps
        )

    endif(USE_VPTZ_STANDALONE)

endif(USE_GLSL)
