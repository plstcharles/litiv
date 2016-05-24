#.rst:
# FindKinectSDK
# -------------
#
# Find Kinect for Windows SDK v1 (Kinect SDK v1) and Kinect for Windows Developer Toolkit v1 (Kinect Developer Toolkit v1) include dirs, library dirs, libraries and post-build commands
#
# Use this module by invoking find_package with the form::
#
#    find_package( KinectSDK [REQUIRED] )
#
# Results for users are reported in following variables::
#
#    KinectSDK_FOUND                 - Return "TRUE" when Kinect SDK v1 found. Otherwise, Return "FALSE".
#    KinectSDK_INCLUDE_DIRS          - Kinect SDK v1 include directories. (${KinectSDK_DIR}/inc)
#    KinectSDK_LIBRARY_DIRS          - Kinect SDK v1 library directories. (${KinectSDK_DIR}/lib/x86 or ${KinectSDK_DIR}/lib/amd64)
#    KinectSDK_LIBRARIES             - Kinect SDK v1 library files. (${KinectSDK_LIBRARY_DIRS}/Kinect10.lib (If check the box of any application festures, corresponding library will be added.))
#
#    KinectToolkit_FOUND             - Return "TRUE" when Kinect Developer Toolkit v1 found. Otherwise, Return "FALSE".
#    KinectToolkit_INCLUDE_DIRS      - Kinect Developer Toolkit v1 include directories. (${KinectToolkit_DIR}/inc)
#    KinectToolkit_LIBRARY_DIRS      - Kinect Developer Toolkit v1 library directories. (${KinectToolkit_DIR}/lib/x86 or ${KinectToolkit_DIR}/lib/amd64)
#    KinectToolkit_LIBRARIES         - Kinect Developer Toolkit v1 library files. (If check the box of any application festures, corresponding library will be added.)
#    KinectToolkit_COMMANDS          - Copy commands of redist files for application functions of Kinect Developer Toolkit v1.
#
# This module reads hints about search locations from following environment variables::
#
#    KINECTSDK10_DIR                 - Kinect SDK v1 root directory. (This environment variable has been set by installer of Kinect SDK v1.)
#    KINECT_TOOLKIT_DIR              - Kinect Developer Toolkit v1 root directory. (This environment variable has been set by installer of Kinect Developer Toolkit v1.)
#
# CMake entries::
#
#    KinectSDK_DIR                   - Kinect SDK v1 root directory. (Default $ENV{KINECTSDK10_DIR})
#
#    KinectToolkit_DIR               - Kinect Developer Toolkit v1 root directory. (Default $ENV{KINECT_TOOLKIT_DIR})
#    KinectToolkit_FACE              - Check the box when using Face features. (Default uncheck)
#    KinectToolkit_FUSION            - Check the box when using Fusion features. (Default uncheck)
#    KinectToolkit_BACKGROUNDREMOVAL - Check the box when using Back Ground Removal features. (Default uncheck)
#
# Example to find Kinect SDK v1 and Kinect Developer Toolkit v1::
#
#    cmake_minimum_required( VERSION 2.8 )
#
#    project( project )
#    add_executable( project main.cpp )
#
#    # Find package using this module.
#    find_package( KinectSDK REQUIRED )
#
#    if(KinectSDK_FOUND)
#      # [C/C++]>[General]>[Additional Include Directories]
#      include_directories( ${KinectSDK_INCLUDE_DIRS} )
#
#      # [Linker]>[General]>[Additional Library Directories]
#      link_directories( ${KinectSDK_LIBRARY_DIRS} )
#
#      # [Linker]>[Input]>[Additional Dependencies]
#      target_link_libraries( project ${KinectSDK_LIBRARIES} )
#    endif()
#
#    if(KinectToolkit_FOUND)
#      # [C/C++]>[General]>[Additional Include Directories]
#      include_directories( ${KinectToolkit_INCLUDE_DIRS} )
#
#      # [Linker]>[General]>[Additional Library Directories]
#      link_directories( ${KinectToolkit_LIBRARY_DIRS} )
#
#      # [Linker]>[Input]>[Additional Dependencies]
#      target_link_libraries( project ${KinectToolkit_LIBRARIES} )
#
#      # [Build Events]>[Post-Build Event]>[Command Line]
#      add_custom_command( TARGET project POST_BUILD ${KinectToolkit_COMMANDS} )
#    endif()
#
# =============================================================================
#
# Copyright (c) 2016 Tsukasa SUGIURA
# Distributed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# =============================================================================

##### Utility #####

# Check Directory Macro
macro(CHECK_DIR _DIR)
  if(NOT EXISTS "${${_DIR}}")
    message(WARNING "Directory \"${${_DIR}}\" not found.")
    if("${_DIR}" MATCHES KinectSDK_.*)
      set(KinectSDK_FOUND FALSE)
    else()
      set(KinectToolkit_FOUND FALSE)
    endif()
    unset(_DIR)
  endif()
endmacro()

# Check Files Macro
macro(CHECK_FILES _FILES _DIR)
  set(_MISSING_FILES)
  foreach(_FILE ${${_FILES}})
    if(NOT EXISTS "${_FILE}")
      get_filename_component(_FILE ${_FILE} NAME)
      set(_MISSING_FILES "${_MISSING_FILES}${_FILE}, ")
    endif()
  endforeach()
  if(_MISSING_FILES)
    message(WARNING "In directory \"${${_DIR}}\" not found files: ${_MISSING_FILES}")
    if("${_FILES}" MATCHES KinectSDK_.*)
      set(KinectSDK_FOUND FALSE)
    else()
      set(KinectToolkit_FOUND FALSE)
    endif()
    unset(_FILES)
  endif()
endmacro()

# Target Platform
set(TARGET_PLATFORM)
if(NOT CMAKE_CL_64)
  set(TARGET_PLATFORM x86)
else()
  set(TARGET_PLATFORM amd64)
endif()

##### Find Kinect SDK v1 #####

# Found
set(KinectSDK_FOUND TRUE)
if(MSVC_VERSION LESS 1600)
  message(WARNING "Kinect for Windows SDK v1 supported Visual Studio 2010 or later.")
  set(KinectSDK_FOUND FALSE)
endif()

# Root Directoty
set(KinectSDK_DIR)
if(KinectSDK_FOUND)
  set(KinectSDK_DIR $ENV{KINECTSDK10_DIR} CACHE PATH "Kinect for Windows SDK v1 Install Path." FORCE)
  check_dir(KinectSDK_DIR)
endif()

# Include Directories
set(KinectSDK_INCLUDE_DIRS)
if(KinectSDK_FOUND)
  set(KinectSDK_INCLUDE_DIRS ${KinectSDK_DIR}/inc)
  check_dir(KinectSDK_INCLUDE_DIRS)
endif()

# Library Directories
set(KinectSDK_LIBRARY_DIRS)
if(KinectSDK_FOUND)
  set(KinectSDK_LIBRARY_DIRS ${KinectSDK_DIR}/lib/${TARGET_PLATFORM})
  check_dir(KinectSDK_LIBRARY_DIRS)
endif()

# Dependencies
set(KinectSDK_LIBRARIES)
if(KinectSDK_FOUND)
  set(KinectSDK_LIBRARIES ${KinectSDK_LIBRARY_DIRS}/Kinect10.lib)
  check_files(KinectSDK_LIBRARIES KinectSDK_LIBRARY_DIRS)
endif()

message(STATUS "KinectSDK_FOUND : ${KinectSDK_FOUND}")

##### Find Kinect Developer Toolkit v1 #####

# Options
option(KinectToolkit_FACE "Face features" FALSE)
option(KinectToolkit_FUSION "Fusion features" FALSE)
option(KinectToolkit_BACKGROUNDREMOVAL "Back Ground Removal features" FALSE)

set(KinectToolkit FALSE)
if(KinectToolkit_FACE OR KinectToolkit_FUSION OR KinectToolkit_BACKGROUNDREMOVAL)
  set(KinectToolkit TRUE)
endif()

set(KinectToolkit_DIR)
set(KinectToolkit_INCLUDE_DIRS)
set(KinectToolkit_LIBRARY_DIRS)
set(KinectToolkit_LIBRARIES)
set(KinectToolkit_COMMANDS COMMAND)

if(KinectToolkit)
  # Found
  set(KinectToolkit_FOUND TRUE)
  if(MSVC_VERSION LESS 1600)
    message(WARNING "Kinect for Windows Developer Toolkit v1 supported Visual Studio 2010 or later.")
    set(KinectToolkit_FOUND FALSE)
  elseif(NOT KinectSDK_FOUND)
    message(WARNING "Kinect for Windows Developer Toolkit v1 need Kinect for Windows SDK v1.")
    set(KinectToolkit_FOUND FALSE)
  endif()

  # Root Directory
  if(KinectToolkit_FOUND)
    if(EXISTS $ENV{KINECT_TOOLKIT_DIR})
      set(KinectToolkit_DIR $ENV{KINECT_TOOLKIT_DIR} CACHE PATH "Kinect for Windows Developer Toolkit v1 Install Path." FORCE)
    elseif(EXISTS $ENV{FTSDK_DIR})
      set(KinectToolkit_DIR $ENV{FTSDK_DIR} CACHE PATH "Kinect for Windows Developer Toolkit v1 Install Path." FORCE)
    endif()
    check_dir(KinectToolkit_DIR)
  endif()

  # Include Directories
  if(KinectToolkit_FOUND)
    set(KinectToolkit_INCLUDE_DIRS ${KinectToolkit_DIR}/inc)
    check_dir(KinectToolkit_INCLUDE_DIRS)
  endif()

  # Library Directories
  if(KinectToolkit_FOUND)
    set(KinectToolkit_LIBRARY_DIRS ${KinectToolkit_DIR}/lib/${TARGET_PLATFORM})
    check_dir(KinectToolkit_LIBRARY_DIRS)
  endif()

  # Dependencies
  if(KinectToolkit_FOUND)
    if(KinectToolkit_FACE)
      find_library(KinectToolkit_FACE_LIBRARY
                   NAMES FaceTrackLib
                   PATHS ${KinectToolkit_LIBRARY_DIRS})
      if(KinectToolkit_FACE_LIBRARY)
        list(APPEND KinectToolkit_LIBRARIES ${KinectToolkit_FACE_LIBRARY})
      else()
        message(WARNING "In directory Face Tracking Library not found files: ${KinectToolkit_LIBRARY_DIRS}")
      endif()
    endif()

    if(KinectToolkit_FUSION)
      find_library(KinectToolkit_FUSION_LIBRARY
                   NAMES KinectFusion180_64 KinectFusion180_32 KinectFusion170_64 KinectFusion170_32
                   PATHS ${KinectToolkit_LIBRARY_DIRS})
      if(KinectToolkit_FUSION_LIBRARY)
        list(APPEND KinectToolkit_LIBRARIES ${KinectToolkit_FUSION_LIBRARY})
      else()
        message(WARNING "In directory Fusion Library not found files: ${KinectToolkit_LIBRARY_DIRS}")
      endif()
    endif()

    if(KinectToolkit_BACKGROUNDREMOVAL)
      find_library(KinectToolkit_BACKGROUNDREMOVAL_LIBRARY
                   NAMES KinectBackgroundRemoval180_64 KinectBackgroundRemoval180_32
                   PATHS ${KinectToolkit_LIBRARY_DIRS})
      if(KinectToolkit_BACKGROUNDREMOVAL_LIBRARY)
        list(APPEND KinectToolkit_LIBRARIES ${KinectToolkit_BACKGROUNDREMOVAL_LIBRARY})
      else()
        message(WARNING "In directory Back Ground Removal Library not found files: ${KinectToolkit_LIBRARY_DIRS}")
      endif()
    endif()

    check_files(KinectToolkit_LIBRARIES KinectToolkit_LIBRARY_DIRS)
  endif()

  # Custom Commands
  if(KinectToolkit_FOUND)
    set(KinectToolkit_REDIST_DIR ${KinectToolkit_DIR}/Redist/${TARGET_PLATFORM})
    check_dir(KinectToolkit_REDIST_DIR)
    list(APPEND KinectToolkit_COMMANDS COMMAND cd /d "${KinectToolkit_REDIST_DIR}" > NUL)

    if(KinectToolkit_FACE)
      list(APPEND KinectToolkit_COMMANDS COMMAND copy "FaceTrack*.dll" "$(OutDir)" /y > NUL)
    endif()

    if(KinectToolkit_FUSION)
      list(APPEND KinectToolkit_COMMANDS COMMAND xcopy "KinectFusion*.dll" "$(OutDir)" /e /y /i /r > NUL)
    endif()

    if(KinectToolkit_BACKGROUNDREMOVAL)
      list(APPEND KinectToolkit_COMMANDS COMMAND xcopy "KinectBackgroundRemoval*.dll" "$(OutDir)" /e /y /i /r > NUL)
    endif()
  endif()

  message(STATUS "KinectToolkit_FOUND : ${KinectToolkit_FOUND}")
endif()