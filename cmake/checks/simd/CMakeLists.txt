
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

macro(try_runcheck_and_set_success name defaultval)
    set(CV_HARDWARE_SUPPORT_CHECK_FLAG_NAME ${name})
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/checks/simd/cvhardwaresupport_check.cpp.in"
        "${CMAKE_BINARY_DIR}/cmake/checks/simd/${name}.cpp"
    )
    if(NOT (DEFINED USE_${name}))
        if(CMAKE_CROSSCOMPILING) # all code intrinsics should be user-defined
            option(USE_${name} "Allow implementations to use ${name} instructions (cannot guarantee remote support)" OFF)
        elseif(WIN32) # try_run often fails on windows due to linking/dll issues (thus should be user-defined)
            option(USE_${name} "Allow implementations to use ${name} instructions (cannot guarantee local support)" ${defaultval})
        else()
            message(STATUS "Testing local support for ${name} instructions via OpenCV...")
            try_run(${name}_RUN_RESULT ${name}_COMPILE_RESULT ${CMAKE_BINARY_DIR}/cmake/checks/simd/ ${CMAKE_BINARY_DIR}/cmake/checks/simd/${name}.cpp CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${OpenCV_INCLUDE_DIRS}" LINK_LIBRARIES ${OpenCV_LIBRARIES} COMPILE_OUTPUT_VARIABLE ${name}_COMPILE_OUTPUT)
            set_eval(USE_${name} (${name}_RUN_RESULT AND ${name}_COMPILE_RESULT))
            if(${USE_${name}})
                option(USE_${name} "Allow implementations to use ${name} instructions (local support tested successfully)" ${USE_${name}})
                message(STATUS "Testing local support for ${name} instructions via OpenCV... OK")
            else()
                option(USE_${name} "Allow implementations to use ${name} instructions (cannot guarantee local support)" ${defaultval})
                if(NOT ${${name}_COMPILE_RESULT})
                    message(STATUS "Testing local support for ${name} instructions via OpenCV... FAILED (compilation error, output: ${${name}_COMPILE_OUTPUT})")
                else()
                    message(STATUS "Testing local support for ${name} instructions via OpenCV... FAILED (runtime error)")
                endif()
            endif()
            file(REMOVE ${CMAKE_BINARY_DIR}/cmake/checks/${name}.cpp)
        endif()
    endif()
endmacro(try_runcheck_and_set_success)

if(NOT TARGET_PLATFORM_x64)
    try_runcheck_and_set_success(MMX ON)
else()
    option(USE_MMX "Allow implementations to use MMX instructions" FALSE)
    mark_as_advanced(USE_MMX)
endif()
try_runcheck_and_set_success(SSE ON)
try_runcheck_and_set_success(SSE2 ON)
try_runcheck_and_set_success(SSE3 ON)
try_runcheck_and_set_success(SSSE3 ON)
try_runcheck_and_set_success(SSE4_1 ON)
try_runcheck_and_set_success(SSE4_2 ON)
try_runcheck_and_set_success(POPCNT ON)
try_runcheck_and_set_success(AVX ON)
try_runcheck_and_set_success(AVX2 OFF)
