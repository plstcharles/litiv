LITIV Computer Vision Framework
===============================

This framework contains various libraries, executables and scripts originating from R&D projects undertaken in the [LITIV lab (Laboratoire d'Interprétation et de Traitement d'Images et Vidéo)](http://www.polymtl.ca/litiv/en/), at Polytechnique Montreal. For now, it primarily consists of C++ algorithm implementations and utilities, most of which rely on [OpenCV](http://opencv.org/). Its build system is based on [CMake](https://cmake.org/), and its structure is inspired by OpenCV's. The framework should be compilable on most Unix/Windows systems given proper configuration (I personally use it on Ubuntu w/ CLion and on Windows 10 w/ MSVC Community 2015).

Most of the source code behind the LITIV framework is available under the [Apache 2.0 license](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)); see the LICENSE file for more information. Some third-party libraries and utilities are provided under their own BSD ([2-clause](https://tldrlegal.com/license/bsd-2-clause-license-(freebsd)) or [3-clause](https://tldrlegal.com/license/bsd-3-clause-license-(revised))) licenses. Specific licensing details are available in each source file (see the 3rdparty folder for more info). While this means most of the LITIV framework source code may be used in distributed commercial applications, be aware that some algorithms therein (e.g. PBAS, VIBE) may be covered by patents in your country. More information is provided in the header files for these algorithms. Note that we will offer no legal advice on possible patent infringements cases; the LITIV framework should be primarily used for testing, evaluation, research, and development purposes in an academic setting.

Since the LITIV framework is still in its infancy, it has very little documentation outside header files. If you want to learn more about the algorithms, your best bet is to read the papers that introduced them. More minimalist code samples, automated testing & unit testing are also on the 'big TODO list'; for now, module behavior validation solely relies on assertions, and most of these are only enabled for debug builds. If you are looking for a place to start digging, I would recommend the *samples* and *apps* folders; they contain executable projects providing a high-level look at some features of the framework. The *apps* folder contains mostly development sandboxes and testbenches (mostly uncommented), while the *samples* folder contains cleaner use-case examples with more adequate description.

Structure Overview
------------------
As stated before, the LITIV framework structure is inspired by OpenCV's structure. This means libraries are split into *modules*, and they are all assembled under a global library (*world*) for easier linking. Third-party modules are kept in their own folder (*3rdparty*) at the root level. The *apps* and *samples* folders contain various executables which rely on LITIV algorithms and utilities for testing and evaluation. The *scripts* folder contains evaluation/test scripts that once had some purpose in the lab (it needs cleaning).

All internal modules can be dynamically or statically linked on Unix systems, and only statically linked on Windows (symbol exports are still missing).

Requirements
------------
* [CMake](https://cmake.org/) >= 3.1.0 (required)
* [OpenCV](http://opencv.org/) >= 3.0.0 (required)
* OpenGL >= 4.3 (optional, for GLSL impl)
* [GLFW](http://www.glfw.org/) >= 3.0.0 or [FreeGLUT](http://freeglut.sourceforge.net/) >= 2.8.0 (optional, for GLSL impl)
* [GLEW](http://glew.sourceforge.net/) >= 1.9.0 (optional, for GLSL impl)
* (CUDA/OpenCL will eventually be added as optional)
* (OpenGM + Gurobi/CPLEX/HDF5 will eventually be added as optional)

Module List
-----------
##### _BSDS500 (3rd party)_
Contains a cleaned/optimized version of the [BSDS500 boundary detection evaluation utilities](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html); provided with no license. These utilities are optionally linked into the *datasets* module via a CMake option. If left out, a more naïve evaluation approach is used directly in *datasets* which gives comparable results, but way faster.

Anyone studying edge detection who feels the BSDS500 matlab scripts are too slow should definitely check these out.
##### _datasets_
Standalone dataset parsing and result evaluation framework with support for data precaching, async processing and on-the-fly custom work batch generation. Largely based on template specialization and multiple interface inheritance --- the only header required to use the module is “datasets.hpp”.

As of version 1.1, the interfaces support image-based and video-based data parsing and saving, and evaluation for binary classification problems (including BSDS500 via a custom evaluator). Upcoming versions should add support for image-array-based data parsing/saving, and evaluation for image/video registration and cosegmentation problems.

List of (non-LITIV) datasets with out-of-the-box specialization (more should be added over time):
  - Foreground-background video segmentation & background subtraction:
    - [ChangeDetection.net 2012](http://wordpress-jodoin.dmi.usherb.ca/cdw2012)
    - [ChangeDetection.net 2014](http://wordpress-jodoin.dmi.usherb.ca/cdw2014)
    - Wallflower
    - PETS2006 (dataset#3)
  - Boundary segmentation & edge detection:
    - [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html)

##### _features2d_
For now, only contains a feature descriptor class for Local Binary Similarity Patterns (LBSPs, [CRV2013](http://dx.doi.org/10.1109/CRV.2013.29)). These can be used for dense image description (e.g. texture analysis) or keypoint-based description. If SSE2+ is available, intrinsics will be used to drastically accelerate their computation.

Besides, an implementation of Felzenszwalb’s HOG features with proper optimization might be added later...
##### _imgproc_
Contains various image processing utilities and classes for edge detection. The main header file contains two implementations of image thinning algorithms along with non-max suppression utilities. The module itself contains a wrapper interface for edge detection, and two versions of Canny's method: one based on OpenCV's implementation, and one using binary feature convolutions via LBSPs (CVPRW2016).
##### _utils_
Equivalent of OpenCV’s “core” module; contains miscellaneous utilities required in other modules (e.g. OpenGL wrappers, platform wrappers, distance functions, C++11 utilities, etc.). Here is a non-exhaustive list of features:
  - OpenGL:
    - GLFW/GLUT window handler wrappers
    - Object-based GL context creation utilities
    - OpenCV <=> OpenGL type conversion utilities
    - Texture/matrices deep copy utilities
    - Templated array-based glGetInteger wrapper
    - 32-bit Tiny Mersenne Twister GLSL implementation
    - GL matrices/vertex data containers
    - Object-based wrappers for VAOs, PBOs, (dynamic)textures(arrays), billboards, etc.
    - Object-based wrapper for GLSL shader compilation, linking, and global program management
    - GLSL source code for passthrough fragment/vertex/compute shaders
    - GLSL source code for various simple compute shader operations (parallel sums, shared data preloading, ...)
    - GLSL image processing algorithm & evaluator interfaces (hides all the ugly binding/prep calls)
  - OpenCV:
    - Display helper object with mouse feedback for algo debugging
    - Random pixel lookup utilities (based on 4-NN, 8-NN, Gaussian kernels, ...)
  - Platform:
    - Files/subdirectories query utilities with name filtering
    - Various string parsing/transformation utilities
    - Various sorting/unique/linear interpolation utilities (for matlab compat)
  - Console:
    - Various line/symbol/block drawing & updating utilities (most taken from [rlutil](https://github.com/tapio/rlutil))
    - Console window/buffer resizing utilities (win32 only)
    - Progress bar
  - Cxx:
    - STL pointer casting utilities
    - Loop unroller
    - Aligned memory allocator for STL containers
    - Exception extenders & loggers
    - Stopwatch
    - Time, version stamps for loggers
    - enable_shared_from_this wrapper with cast function
    - C++11 missing make_unique
  - Distance:
    - Type-templated versions of L1/L2/cdist/Hamming distance functions
    - GLSL implementations of various distance functions

##### _video_
Contains multiple background subtraction algorithm implementations, and might eventually also inherit tracking and registration algorithms. List of background subtraction methods currently available:
  - LOBSTER ([WACV2014](http://dx.doi.org/10.1109/WACV.2014.6836059))
  - SuBSENSE ([CVPRW2014](http://www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W12/papers/St-Charles_Flexible_Background_Subtraction_2014_CVPR_paper.pdf), [TIP2015](http://dx.doi.org/10.1109/TIP.2014.2378053))
  - PAWCS ([WACV2015](http://dx.doi.org/10.1109/WACV.2015.137))
  - PBAS (CVPRW2012)
  - ViBe (ICASSP 2009, TIP2011)

##### _vptz_
Contains a compact version of the [VirtualPTZ library](https://bitbucket.org/pierre_luc_st_charles/virtualptz_standalone) used to evaluate PTZ trackers --- builds as a standalone dynamic library (or DLL with proper export symbols) if specified via CMake option.
##### _world_
Contains nothing, simply wraps other modules into one big spaghetti ball that's easier/faster to link elsewhere.

Notes
-----
For a more user-friendly, stable and documented background subtraction/video segmentation framework with various other utilities, see [Andrews Sobral's BGSLibrary](https://github.com/andrewssobral/bgslibrary).

Release notes and versions for the framework are maintained through [Github Releases](https://github.com/plstcharles/litiv/releases). **Expect both major and minor version changes to break backwards compatibility for some time, as the API is not yet stable**. For now, no binaries (executables or libraries) are provided; you are expected to configure and compile the framework yourself based on your own needs & your hardware's capabilities.

Citation
--------
If you use a module from this framework in your own work, please cite its related LITIV publication(s) as acknowledgement. See [this page](http://www.polymtl.ca/litiv/pub/index.php) for a full list.

Contributing
------------
If you have a something you'd like to contribute to the LITIV framework, send me a message first; I'll be happy to merge your code in if it fits with the rest of the framework.

