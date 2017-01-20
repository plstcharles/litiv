cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(SuperpixelSegmentation)



set(EXTERNAL_DATA_ROOT "${CMAKE_SOURCE_DIR}/data/" CACHE PATH "External data root folder (this is where all datasets should be located for applications to find them automatically)")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Forced project build type" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


### OPENCV CHECK
find_package(OpenCV 3.0 REQUIRED)
message(STATUS "Found OpenCV >=3.0 at '${OpenCV_DIR}'")

### CUDA CHECK 
find_package(CUDA REQUIRED)
set(CUDA_COMPUTE_CAPABILITY "20")
set(CUDA_GENERATE_CODE "arch=compute_${CUDA_COMPUTE_CAPABILITY},code=sm_${CUDA_COMPUTE_CAPABILITY}" CACHE STRING "Which GPU architectures to generate code for (each arch/code pair will be passed as --generate-code option to nvcc, separate multiple pairs by ;)")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch compute_${CUDA_COMPUTE_CAPABILITY})


### Compilation 

if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang"))
    option(BUILD_SHARED_LIBS "Build shared libraries (.so) instead of static ones (.a)" ON)
    if(NOT CMAKE_CROSSCOMPILING)
        add_definitions(-march=native)
    endif()
    if(USE_FAST_MATH)
        add_definitions(-ffast-math)
    endif()
    add_definitions(-Wall)
    add_definitions(-Wfatal-errors)
    add_definitions(-ftemplate-depth=900) # already default for gcc, but not for clang
    check_function_exists(aligned_alloc USE_STL_ALIGNED_ALLOC)
    if(NOT USE_STL_ALIGNED_ALLOC)
        set(USE_STL_ALIGNED_ALLOC 0 CACHE INTERNAL "Have function aligned_alloc")
    endif()
    check_function_exists(posix_memalign USE_POSIX_ALIGNED_ALLOC)
    if(NOT USE_POSIX_ALIGNED_ALLOC)
        set(USE_POSIX_ALIGNED_ALLOC 0 CACHE INTERNAL "Have function posix_memalign")
    endif()
    if(USE_LINK_TIME_OPTIM)
        add_definitions(-flto)
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
          find_program(GCC_AR gcc-ar)
          if(GCC_AR)
            set(CMAKE_AR ${GCC_AR})
          endif()
          find_program(GCC_RANLIB gcc-ranlib)
          if(GCC_RANLIB)
            set(CMAKE_RANLIB ${GCC_RANLIB})
          endif()
        endif()
    endif()
elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    #set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
    set(CMAKE_DEBUG_POSTFIX "d" CACHE INTERNAL "" FORCE)
    option(USE_VERSION_TAGS "Apply version tags suffixes on built libraries" ON)
    option(BUILD_SHARED_LIBS "Build shared libraries (.dll) instead of static ones (.lib)" OFF)
    if(BUILD_SHARED_LIBS)
        message("LITIV DLLs are still missing symbol exports, and might be incomplete for some targets.")
    endif()
    add_definitions(-DUNICODE -D_UNICODE)
    if(USE_FAST_MATH)
        add_definitions(/fp:fast)
    else(NOT USE_FAST_MATH)
        add_definitions(/fp:precise)
    endif()
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/W1)
    add_definitions(/openmp)
    add_definitions(/arch:AVX) # check performance difference? vs 387? @@@
    if(USE_LINK_TIME_OPTIM)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL")
        set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS_RELEASE} /LTCG")
        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    message(FATAL_ERROR "Intel compiler still unsupported; please edit the main CMakeList.txt file to add proper configuration.")
    # ... @@@
else()
    message(FATAL_ERROR "Unknown compiler; please edit the main CMakeList.txt file to add proper configuration.")
endif()


#add_custom_command(COMMAND set "PATH=${opencvDLL_DIRS};%PATH%")



CUDA_INCLUDE_DIRECTORIES("${OpenCV_INCLUDE_DIRS}" "${CMAKE_SOURCE_DIR}/include")

### Link 
set(SOURCE_FILES "src/SlicCudaHost.cu" "src/SlicCudaDevice.cu" "include/SlicCudaHost.h" "include/SlicCudaDevice.h")
cuda_add_executable(SlicCuda ${SOURCE_FILES} "src/main.cpp")
target_include_directories(SlicCuda PUBLIC "include")
target_link_libraries(SlicCuda ${OpenCV_LIBS})