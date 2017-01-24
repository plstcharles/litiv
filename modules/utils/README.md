LITIV *utils* Module
--------------------
This module is the spiritual equivalent of OpenCV’s “core”, and was at the root of the framework's creation. It contains miscellaneous utilities needed by other modules (e.g. OpenGL wrappers, platform wrappers, distance functions, C++11/14 compatibility tricks). In other words, this is a compacted gold mine for other programmers who might dabble in related computer vision work.

Here is a non-exhaustive (and probably not fully up-to-date) list of features:
  - OpenGL:
    - GLFW/GLUT window handler wrappers
    - Object-based GL context creation
    - OpenCV <=> OpenGL image type conversion functions
    - Texture/matrices deep copy functions
    - Templated array-based glGetInteger wrapper
    - 32-bit Tiny Mersenne Twister GLSL implementation
    - Packed GL matrix/vertex data containers
    - Object-based wrappers for VAOs, PBOs, (dynamic)textures(arrays), billboards, etc.
    - Object-based wrapper for GLSL shader compilation, linking, and global program management
    - GLSL source code for passthrough fragment/vertex/compute shaders
    - Pre-coded shaders for various compute operations (parallel sums, shared data preloading, ...)
    - Image processing GLSL algorithm interface (hides all the ugly binding/prep calls)
  - CUDA:
    - helper functions for error handling, + basic test kernel
  - OpenCV:
    - Display helper object with mouse feedback for debugging
    - Random pixel lookup functions (based on 4-NN, 8-NN, Gaussian kernels, ...)
    - Generic cv::Mat archiver/loader (binary, text, or YML format)
    - Aligned memory allocator for OpenCV matrices
  - SIMD:
    - Wrappers for missing SSE calls (e.g. _mm_mullo_epi32, _mm_extract_epi32)
    - Horizontal SSE sums/min/max functions
  - Platform:
    - Files/subdirectories query functions with name filtering
    - Aligned memory allocator for STL containers
    - Default-init memory allocator for STL containers
  - Kinect:
    - Data structures re-implementation for body joint tracking
  - Console:
    - Various line/symbol/block drawing & updating utilities (most taken from [rlutil](https://github.com/tapio/rlutil))
    - Console window/buffer resizing utilities (win32 only)
    - Progress bar
  - Cxx:
    - Exception extenders & loggers
    - STL unique_pointer type casting helpers
    - Loop unroller
    - String manipulation (caseless comparison, token search, ...)
    - Vector manipulation (concat, filter, member accumulation, ...)
    - Sort utilities (sort indices, uniques, nearest neighbor, ...)
    - Worker thread pool implementation for generic tasking/queueing
    - High-precision stopwatch
    - Time, version stamp generators for loggers
    - enable_shared_from_this wrapper with cast function
    - has_const_iterator type traits helper
    - get_bigger_integer type traits helper
    - Tuple for_each, unpack, unpack_call
    - Static array in-place transformation/reduction w/ constexpr support
    - Lookup Table impl w/ support for generic functions
    - Mutex unlock guards
    - C++11 semaphore implementation
    - C++11 make_unique
  - Math:
    - Type-templated L1/L2sqr/L2/Hamming/Gradient/ColorDistortion distance functions
    - LUT-based Hamming weight functions for processors that do not support Intel's POPCNT
    - GLSL implementations of various distance functions
    - Nearest neighbor helper function
    - L1 vector/pointset interpolation function (matlab-compat)
    - Linearly separated vector generator function (matlab-compat)
    - Integer flag bit expander


This module also links all external dependencies for the framework, meaning that executables only have to link to this target (along with other LITIV targets, as needed) to fully link to, e.g., OpenCV. Also, note that most important CMake variables have symbol entrypoints in the [defines.hpp.in](./include/litiv/utils/defines.hpp.in) file, which is converted by CMake into an actual header file.
