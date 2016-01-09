
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

if(BUILD_VPTZ)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # disables C4251 + C4275 to allow STL/template classes to be used in exported classes/members
        # need to eliminate these using pImpl idiom in exported classes to add abstraction layer @@@@
        add_definitions(/wd4251) # disables C4251, "'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'"
        add_definitions(/wd4275) # disables C4275, "non DLL-interface classkey 'identifier' used as base for DLL-interface classkey 'identifier'"
    endif()
    if(USE_VPTZ_STANDALONE)
        add_definitions(-DVPTZ_IMPORT)
        include_directories(${litiv_vptz_INCLUDE_DIR})
        set(VPTZ_LINK_LIBS litiv_vptz)
    else(NOT USE_VPTZ_STANDALONE)
        include_directories(${litiv_vptz_INCLUDE_DIR} ${litiv_utils_INCLUDE_DIR})
        set(VPTZ_LINK_LIBS litiv_world)
    endif()
    get_subdirectory_list(subdirs ${CMAKE_CURRENT_SOURCE_DIR})
    foreach(subdir ${subdirs})
        add_subdirectory(${subdir})
    endforeach()
endif()
