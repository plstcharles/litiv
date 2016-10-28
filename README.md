LITIV Computer Vision Framework
===============================

This framework contains various libraries, executables and scripts originating from R&D projects undertaken in the [LITIV lab (Laboratoire d'Interprétation et de Traitement d'Images et Vidéo)](http://www.polymtl.ca/litiv/en/), at Polytechnique Montreal. For now, it primarily consists of C++ algorithm implementations and utilities, most of which rely on [OpenCV](http://opencv.org/). Its build system is based on [CMake](https://cmake.org/), and its structure is inspired by OpenCV's. The framework should be compilable on most Unix/Windows systems given proper configuration (I personally use it on Ubuntu w/ CLion and on Windows 10 w/ MSVC Community 2015).

Most of the source code behind the LITIV framework is available under the [Apache 2.0 license](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)); see the [LICENSE](./LICENSE.txt) file for more information. Some third-party libraries and utilities are provided under their own BSD ([2-clause](https://tldrlegal.com/license/bsd-2-clause-license-(freebsd)) or [3-clause](https://tldrlegal.com/license/bsd-3-clause-license-(revised))) licenses. Specific licensing details are available in each source file or folder. While this means most of the LITIV framework source code may be used in distributed commercial applications, be aware that some 3rd-party algorithms therein (e.g. PBAS, VIBE) may be covered by patents in your country. More information is provided in the header files for these algorithms. Note that we will offer no legal advice on possible patent infringements cases; the LITIV framework should be primarily used for testing, evaluation, research, and development purposes in an academic setting.

Since the LITIV framework is still in its infancy, it has very little documentation outside header files. If you want to learn more about the algorithms, your best bet is to read the papers that introduced them. More minimalist code samples, automated testing & unit testing are also on the 'big TODO list'; for now, module behavior validation solely relies on assertions, and most of these are only enabled for debug builds. If you are looking for a place to start digging, I would recommend the [*apps*](./apps/) and [*samples*](./samples/) folders; they contain executable projects providing a high-level look at some features of the framework. The [*apps*](./apps/) folder contains mostly development sandboxes and testbenches (mostly uncommented), while the [*samples*](./samples/) folder contains cleaner use-case examples with more adequate descriptions.

Structure Overview
------------------
As stated before, the LITIV framework structure is inspired by OpenCV's structure. This means libraries are split into [*modules*](./modules/), and they are all assembled under a global library ([*world*](./modules/world/)) for easier linking. Third-party modules are kept in their own folder ([*3rdparty*](./3rdparty/)) at the root level. The [*apps*](./apps/) and [*samples*](./samples/) folders contain various executables which rely on LITIV algorithms and utilities for testing and evaluation. The [*scripts*](./scripts/) folder contains evaluation/test scripts that once had some purpose in the lab (it needs cleaning).

All internal modules can be dynamically or statically linked on Unix systems, and only statically linked on Windows (symbol exports are still missing).

Requirements
------------

My primary goal is to have the framework core only depend on OpenCV/CMake, and have as many OpenCV-based implementations as possible. This is not always possible however (let's not reinvent the wheel here), so building without the optional dependencies will disable some framework functionalities/features. More specifically, here is the list of required and optional dependencies:

* **[CMake](https://cmake.org/) >= 3.1.0 (required)**
* **[OpenCV](http://opencv.org/) >= 3.0.0 (required)**
* OpenGL >= 4.3 (optional, for GLSL impl)
* [GLFW](http://www.glfw.org/) >= 3.0.0 or [FreeGLUT](http://freeglut.sourceforge.net/) >= 2.8.0 (optional, for GLSL implementations)
* [GLEW](http://glew.sourceforge.net/) >= 1.9.0 (optional, for GLSL implementations)
* [GLM](http://glm.g-truc.net/) (optional, for GLSL implementations)
* (CUDA/OpenCL will eventually be added as optional)
* (OpenGM + Gurobi/CPLEX/HDF5 will eventually be added as optional)

Modules Overview
----------------

A list of all modules (including 3rd party ones) is presented below; for more information, refer to their README files.

* [**datasets**](./modules/datasets/) : Contains dataset parsing and evaluation utilities with data precaching and async processing support.
* [**features2d**](./modules/features2d/) : Contains feature descriptors (e.g. LBSP, FHOG).
* [**imgproc**](./modules/imgproc/) : Contains image processing algos and utilities (e.g. for edge detection, non-max suppresion).
* [**utils**](./modules/utils/) : Equivalent of OpenCV's "core" module; contains miscellaneous utilities required by other modules.
* [**video**](./modules/video/) : Contains video processing algos and utilities (e.g. for background subtraction/change detection, registration).
* [**vptz**](./modules/vptz/) : Contains a compact version of the [VirtualPTZ library](https://bitbucket.org/pierre_luc_st_charles/virtualptz_standalone) used to evaluate PTZ trackers.
* [**world**](./modules/world/) : Contains nothing, and pre-links other modules to make linking easier/faster elsewhere.
* [**bsds500 (3rd party)**](./3rdparty/bsds500/) : Contains a cleaned version of the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) boundary detection evaluation utilities.
* [**dshowbase (3rd party)**](./3rdparty/dshowbase/) : Contains the [DirectShow base classes](https://msdn.microsoft.com/en-us/library/windows/desktop/dd375456(v=vs.85).aspx) as distributed in the Windows 7 SDK, and more.

Notes
-----
For a more user-friendly, stable and documented background subtraction/video segmentation framework with various other utilities, see [Andrews Sobral's BGSLibrary](https://github.com/andrewssobral/bgslibrary).

Release notes and versions for the framework are maintained through [Github Releases](https://github.com/plstcharles/litiv/releases). **Expect both major and minor version changes to break backwards compatibility for some time, as the API is not yet stable**. For now, no binaries (executables or libraries) are provided; you are expected to configure and compile the framework yourself based on your own needs & your hardware's capabilities.

Citation
--------
If you use a module from this framework in your own work, please cite its related LITIV publication(s) as acknowledgement. See [this page](http://www.polymtl.ca/litiv/pub/index.php) for a full list.

Contributing
------------
If you have a something you'd like to contribute to the LITIV framework, send me a message first; I'll be happy to merge your code in if it fits with the rest of the framework. Bugfixes are always welcomed; you can send them directly in as pull requests.
