
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

if(0) ## (disabled as of march 2016, incomplete)

    litiv_app(vidreg)

    add_executable(litiv_app_vidreg src/main.cpp)
    set_target_properties(litiv_app_vidreg PROPERTIES FOLDER "apps")
    target_link_libraries(litiv_app_vidreg litiv_world)

    install(TARGETS litiv_app_vidreg
        RUNTIME DESTINATION bin
        COMPONENT apps
    )

endif()
