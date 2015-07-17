cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)
#set(CMAKE_VERBOSE_MAKEFILE ON)

project(litiv_framework)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(LITIV_FRAMEWORK_VERSION_MAJOR 1)
set(LITIV_FRAMEWORK_VERSION_MINOR 0)
set(LITIV_FRAMEWORK_VERSION_PATCH 0)

add_compile_options(-std=c++11 -march=native -Wall -Wextra -Wfatal-errors)

option(USE_TEST "Use some useless global test define" ON)

find_package(OpenCV 3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

#if using glsl
find_package(GLFW REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${OpenGL_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})
#endif using glsl
add_subdirectory(modules)
add_subdirectory(execs)
