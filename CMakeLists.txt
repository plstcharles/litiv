cmake_minimum_required(VERSION 3.3)
project(SLIC_CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#CUDA
FIND_PACKAGE(CUDA QUIET REQUIRED)
set(CUDA_HOST_COMPILER /usr/bin/gcc-4.9)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
include_directories(usr/local/include; usr/local/cuda)
link_directories(usr/local/lib;usr/local/cuda)


#OpenCV
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(usr/local/include; usr/local/cuda)
link_directories(usr/local/lib;usr/local/cuda)


#OpenGL
find_package(GLUT)
find_package(OpenGL)
find_package(GLEW)
if (GLEW_FOUND)
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif()
include_directories(${FREEGLUT_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIRS})



file(GLOB SOURCE_FILES *.cpp *.cu)
cuda_add_executable(SLIC_CUDA ${SOURCE_FILES} SLIC_cuda.cpp SLIC_cuda.h)

# always after add_executable
target_link_libraries(SLIC_CUDA ${OpenCV_LIBS})
target_link_libraries(SLIC_CUDA ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

