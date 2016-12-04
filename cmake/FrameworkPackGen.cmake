
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

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/uninstall.cmake.in"
    "${CMAKE_BINARY_DIR}/cmake/generated/uninstall.cmake"
    @ONLY
)
set(uninstall_target_name "uninstall")
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    set(uninstall_target_name "UNINSTALL")
endif()
add_custom_target(${uninstall_target_name} "${CMAKE_COMMAND}" -P "${CMAKE_BINARY_DIR}/cmake/generated/uninstall.cmake")
set_target_properties(${uninstall_target_name} PROPERTIES FOLDER "CMakePredefinedTargets")

set(LITIV_COMPONENTS "")
set(litiv_build_includedirs "")
foreach(litiv_3rdparty_module_name ${litiv_3rdparty_modules})
    set(includedir "${CMAKE_BINARY_DIR}/3rdparty/${litiv_3rdparty_module_name}/include")
    if(EXISTS "${includedir}")
        list(APPEND litiv_build_includedirs "${includedir}")
    endif()
    set(includedir "${CMAKE_SOURCE_DIR}/3rdparty/${litiv_3rdparty_module_name}/include")
    if(EXISTS "${includedir}")
        list(APPEND litiv_build_includedirs "${includedir}")
    endif()
    list(APPEND LITIV_COMPONENTS "litiv_3rdparty_${litiv_3rdparty_module_name}")
endforeach()
foreach(litiv_module_name ${litiv_modules})
    set(includedir "${CMAKE_BINARY_DIR}/modules/${litiv_module_name}/include")
    if(EXISTS "${includedir}")
        list(APPEND litiv_build_includedirs "${includedir}")
    endif()
    set(includedir "${CMAKE_SOURCE_DIR}/modules/${litiv_module_name}/include")
    if(EXISTS "${includedir}")
        list(APPEND litiv_build_includedirs "${includedir}")
    endif()
    list(APPEND LITIV_COMPONENTS "${litiv_module_name}")
endforeach()

set(RUNTIME_INSTALL_DIR "bin")
set(INCLUDE_INSTALL_DIR "include")
set(LIBRARY_INSTALL_DIR "lib")
set(MODULES_INSTALL_DIR "lib/cmake/Modules")
set(CONFIG_INSTALL_DIR "etc/litiv")
write_basic_package_version_file(
    "${CMAKE_BINARY_DIR}/cmake/generated/litiv-config-version.cmake"
    VERSION
        "${LITIV_VERSION}"
    COMPATIBILITY
        ExactVersion # will switch to 'SameMajorVersion' later, when api more stable
)
file(
    COPY
        "${CMAKE_BINARY_DIR}/cmake/generated/litiv-config-version.cmake"
    DESTINATION
        "${CMAKE_BINARY_DIR}"
)
file(
    COPY
        "${CMAKE_SOURCE_DIR}/cmake/checks"
    DESTINATION
        "${CMAKE_BINARY_DIR}/cmake/"
)
file(
    COPY
        "${CMAKE_SOURCE_DIR}/cmake/Modules"
    DESTINATION
        "${CMAKE_BINARY_DIR}/cmake/"
)
if(WIN32)
    set(package_config_install_dirs
        "lib/cmake/litiv"
        "lib"
        "${CMAKE_INSTALL_PREFIX}"
    )
else()
    set(package_config_install_dirs
        "lib/cmake/litiv"
    )
    if("${CMAKE_INSTALL_PREFIX}" STREQUAL "${CMAKE_BINARY_DIR}/install")
        list(APPEND package_config_install_dirs "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()
set(CURRENT_CONFIG_INSTALL 1)
list(LENGTH package_config_install_dirs package_config_install_dir_count)
set(package_config_install_dir_index 0)
while(package_config_install_dir_index LESS package_config_install_dir_count)
    list(GET package_config_install_dirs ${package_config_install_dir_index} package_config_install_dir)
    math(EXPR package_config_install_dir_index "${package_config_install_dir_index}+1")
    configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/cmake/litiv-config.cmake.in"
        "${CMAKE_BINARY_DIR}/cmake/generated/litiv-config-${package_config_install_dir_index}.cmake"
        INSTALL_DESTINATION
            "${package_config_install_dir}"
        PATH_VARS
            RUNTIME_INSTALL_DIR
            INCLUDE_INSTALL_DIR
            LIBRARY_INSTALL_DIR
            MODULES_INSTALL_DIR
            CONFIG_INSTALL_DIR
    )
    install(
        FILES
            "${CMAKE_BINARY_DIR}/cmake/generated/litiv-config-version.cmake"
        DESTINATION
            "${package_config_install_dir}"
    )
    install(
        FILES
            "${CMAKE_BINARY_DIR}/cmake/generated/litiv-config-${package_config_install_dir_index}.cmake"
        DESTINATION
            "${package_config_install_dir}"
        RENAME
            "litiv-config.cmake"
    )
    install(
        EXPORT
            "litiv-targets"
        DESTINATION
            "${package_config_install_dir}"
    )
endwhile()
export(
    EXPORT
        "litiv-targets"
    FILE
        "${CMAKE_BINARY_DIR}/litiv-targets.cmake"
)
install(
    DIRECTORY
        "${CMAKE_SOURCE_DIR}/cmake/checks"
    DESTINATION
        "lib/cmake"
)
install(
    DIRECTORY
        "${CMAKE_SOURCE_DIR}/cmake/Modules"
    DESTINATION
        "lib/cmake"
)

set(CURRENT_CONFIG_INSTALL 0)
set(LITIV_BUILD_RUNTIME_DIR "${CMAKE_BINARY_DIR}/bin")
set(LITIV_BUILD_LIBRARY_DIR "${CMAKE_BINARY_DIR}/lib")
set(LITIV_BUILD_MODULES_DIR "${CMAKE_BINARY_DIR}/cmake/Modules")
set(LITIV_BUILD_CONFIG_DIR "${CMAKE_BINARY_DIR}/etc/litiv")
set(LITIV_BUILD_INCLUDE_DIRS "${litiv_build_includedirs}")

configure_file(
    "${CMAKE_SOURCE_DIR}/cmake/litiv-config.cmake.in"
    "${CMAKE_BINARY_DIR}/litiv-config.cmake"
    @ONLY
)