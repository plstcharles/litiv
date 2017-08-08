LITIV Computer Vision Framework
===============================

[![License](https://img.shields.io/badge/license-Apache%202-green.svg)](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0))
[![Language](https://img.shields.io/badge/lang-C%2B%2B14-f34b7d.svg)](http://en.cppreference.com/w/cpp/compiler_support)
[![Build Status](https://travis-ci.org/plstcharles/litiv.svg?branch=master)](https://travis-ci.org/plstcharles/litiv)
[![Stable Release](https://img.shields.io/github/release/plstcharles/litiv.svg)](https://github.com/plstcharles/litiv/releases)

This framework contains various libraries, executables and scripts originating from computer vision R&D projects undertaken in the [LITIV lab (Laboratoire d'Interprétation et de Traitement d'Images et Vidéo)](http://www.polymtl.ca/litiv/en/), at Polytechnique Montréal. For now, it primarily consists of C++ algorithm implementations and utilities, most of which rely on [OpenCV](http://opencv.org/). Its build system is based on [CMake](https://cmake.org/), and its structure is inspired by OpenCV. The framework should be compatible with most Unix/Windows systems given proper configuration; it has been developed in part on Windows 7/8/10 with MSVC2015v3, and Ubuntu/Fedora with CLion.

Most of the source code behind the LITIV framework is available under the [Apache 2.0 license](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)); see the [LICENSE](./LICENSE.txt) file for more information. Some third-party libraries and utilities are provided under their own BSD ([2-clause](https://tldrlegal.com/license/bsd-2-clause-license-(freebsd)) or [3-clause](https://tldrlegal.com/license/bsd-3-clause-license-(revised))) licenses. Specific licensing details are available in each source file or folder. While this means the LITIV framework source code can be used in distributed commercial applications, be aware that external algorithms that may be used therein (e.g. ViBe, FastPD) are covered by patents in some countries. Note that we will offer no legal advice on possible patent infringements cases; the LITIV framework should be primarily used for testing, evaluation, research, and development purposes in an academic setting.

Unfortunately, the framework currently has little documentation outside doxygen comments in header files. If you want to learn more about the algorithms, your best bet is to read the papers that introduced them. More minimalist code samples & unit tests are also on the 'big TODO list'. For now, module behavior validation primarily relies on assertions, and most of these are only enabled for debug builds. If you are looking for a place to start digging, it is recommended to start in the [*apps*](./apps/) and [*samples*](./samples/) folders; their executable projects can provide a high-level look at some features of the framework. The [*apps*](./apps/) folder contains mostly development sandboxes and testbenches (mostly uncommented), while the [*samples*](./samples/) folder contains cleaner use-case examples with more adequate descriptions.

Structure Overview
------------------
As stated before, the LITIV framework structure is inspired by OpenCV's structure. This means libraries are split into [*modules*](./modules/), and they are all assembled under a global library ([*world*](./modules/world/)) for easier linking. Third-party components are kept in their own folder ([*3rdparty*](./3rdparty/)) at the root level. The [*apps*](./apps/) and [*samples*](./samples/) folders contain various executables which rely on LITIV algorithms and utilities for testing and evaluation. The [*scripts*](./scripts/) folder contains evaluation/test scripts that once had some purpose in the lab (it needs cleaning). For testing, each subproject can have its own *test* directory, and it will automatically be parsed and added via CTest.

All internal modules can be dynamically or statically linked on Unix systems, and only statically linked on Windows (symbol exports are still missing).

Requirements
------------

The primary goal here is to have the framework core only depend on OpenCV/CMake, and have as many OpenCV-based implementations as possible. This is not always possible however, so building without all dependencies will disable some features. Here is the list of required and optional dependencies:

* **[CMake](https://cmake.org/) >= 3.1.0 (required)**
* **[OpenCV](http://opencv.org/) >= 3.0.0 (required)**
* [GLFW](http://www.glfw.org/) >= 3.0.0 or [FreeGLUT](http://freeglut.sourceforge.net/) >= 2.8.0 (optional, for GLSL implementations)
* [GLEW](http://glew.sourceforge.net/) >= 1.9.0 (optional, for GLSL implementations)
* [GLM](http://glm.g-truc.net/) (optional, for GLSL implementations)
* [CUDA](https://developer.nvidia.com/cuda-toolkit) >= 7.0 with compute >=3.0 (optional, for some algo implementations)
* [Boost](http://www.boost.org/) >= 1.49 (optional, for some 3rdparty algo implementations)
* [OpenGM](https://github.com/plstcharles/opengm) (optional, for graph-based algo implementations)

A dockerfile which builds an Ubuntu image including all these dependencies is available [here](./Dockerfile). The images built by Travis are periodically uploaded to Docker Hub [here](https://hub.docker.com/r/plstcharles/litiv-base/) and [here](https://hub.docker.com/r/plstcharles/litiv/).

Modules Overview
----------------

A list of all internal modules is presented below; for more information, refer to their README files.

* [**datasets**](./modules/datasets/) : Provides dataset parsing and evaluation utilities with precaching and async processing support.
* [**features2d**](./modules/features2d/) : Provides feature descriptors & matchers ([LBSP](./modules/features2d/include/litiv/features2d/LBSP.hpp), [DASC](./modules/features2d/include/litiv/features2d/DASC.hpp), [LSS](./modules/features2d/include/litiv/features2d/LSS.hpp), [MI](./modules/features2d/include/litiv/features2d/MI.hpp), [ShapeContext](./modules/features2d/include/litiv/features2d/SC.hpp)).
* [**imgproc**](./modules/imgproc/) : Provides image processing algos and utilities ([edge detectors](./modules/imgproc/include/litiv/imgproc/EdgeDetectorLBSP.hpp), [shape cosegmenters](./modules/imgproc/include/litiv/imgproc/ForegroundStereoMatcher.hpp), [superpixels extractors](./modules/imgproc/include/litiv/imgproc/SLIC.hpp)).
* [**test**](./modules/test/) : Provides utilities for project-wide unit/performance testing; never exported for external usage.
* [**utils**](./modules/utils/) : Equivalent of OpenCV's "core" module; contains miscellaneous common utilities ([see list here](./modules/utils/README.md)).
* [**video**](./modules/video/) : Provides background subtraction algos & utilities ([LOBSTER](./modules/video/include/litiv/video/BackgroundSubtractorLOBSTER.hpp), [SuBSENSE](./modules/video/include/litiv/video/BackgroundSubtractorSuBSENSE.hpp), [PAWCS](./modules/video/include/litiv/video/BackgroundSubtractorPAWCS.hpp) and [others](./modules/video/include/litiv/video/)).
* [**vptz**](./modules/vptz/) : Provides a compact version of the [VirtualPTZ library](https://bitbucket.org/pierre_luc_st_charles/virtualptz_standalone) used to evaluate PTZ trackers.
* [**world**](./modules/world/) : Provides nothing, and pre-links other modules to make linking easier/faster elsewhere.

Notes
-----
For a more user-friendly, stable and documented background subtraction/video segmentation framework with various other utilities, see [Andrews Sobral's BGSLibrary](https://github.com/andrewssobral/bgslibrary).

Release notes and versions for the LITIV framework are maintained through [Github Releases](https://github.com/plstcharles/litiv/releases). You are expected to configure and compile the framework yourself based on your own needs & your machine's capabilities.

Citation
--------
If you use a module from this framework in your own work, please cite its related LITIV publication(s) as acknowledgment. See [this page](http://www.polymtl.ca/litiv/pub/index.php) for a full list.

Contributing
------------
If you have something you'd like to contribute to the LITIV framework (algorithm, improvement or bugfix), you can send it in as a pull request.
