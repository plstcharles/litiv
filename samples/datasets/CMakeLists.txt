
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

litiv_sample(datasets) # macro defines the project as part of the 'sample' group with correct prefix/suffix

add_executable(litiv_sample_datasets src/main.cpp src/middlebury2005.hpp) # pass all project source dependencies
set_target_properties(litiv_sample_datasets PROPERTIES FOLDER "samples") # groups this project with other samples in the IDE
target_link_libraries(litiv_sample_datasets litiv_world) # litiv_world indirectly links all subdependencies (opencv, ...)

install(TARGETS litiv_sample_datasets
    RUNTIME DESTINATION bin
    COMPONENT samples
) # exports the executable binary, once built, to the 'bin' install directory
