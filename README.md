LITIV Computer Vision Framework
===============================

This framework contains various libraries, executables and scripts originating from R&D projects undertaken in the [LITIV lab (Laboratoire d'Interprétation et de Traitement d'Images et Vidéo)](http://www.polymtl.ca/litiv/en/), at Polytechnique Montreal. For now, it primarily consists of C++ implementations and utilities, most of which rely on [OpenCV](http://opencv.org/). Its build system is based on [CMake](https://cmake.org/), and its structure is inspired by OpenCV's. The framework should be compilable on most Unix/Windows systems given proper configuration (I personally use it on Ubuntu & Windows 10 w/ MSVC Community 2015).

Most of the source code behind the LITIV framework is available under the [Apache 2.0 license](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)); see the LICENSE file for more information. Some third-party modules and utilities are provided under their own BSD ([2-clause](https://tldrlegal.com/license/bsd-2-clause-license-(freebsd)) or [3-clause](https://tldrlegal.com/license/bsd-3-clause-license-(revised))) licenses. Specific licensing details are available in each source file. While this means most of the LITIV framework source code may be used in distributed commercial applications, be aware that some algorithms therein (e.g. PBAS, VIBE) may be covered by patents in your country. More information is provided in the header files for these algorithms. Note that we will offer no legal advice on possible patent infrigements cases; the LITIV framework should be primarily used for testing, evaluation, research, and development purposes in an academic setting.

Since the LITIV framework is still in its infancy, it has very little documentation outside header files. If you want to learn more about the algorithms, your best bet is to read the papers that introduced them. Minimalist code samples, automated testing & unit testing are also on the 'big TODO list'; for now, module behavior validation solely relies on assertions, and most of these are only enabled for debug builds.

Structure Overview
------------------
As stated before, the LITIV framework structure is inspired by OpenCV's structure. This means libraries are split into *modules*, and they are all assembled under a global library ("*world*") for easier linking. Third-party modules are kept in their own folder at the root level. The *execs* folder contains various executables which rely on LITIV algorithms and utilities for testing and evaluation. The *scripts* folder contains (should contain?) useful evaluation/test scripts; for now, it is a desolate code dump area that really needs cleaning.

Module List
-----------
* BSDS500 (3rd party)
  - Contains a cleaned version of the BSDS500 boundary detection evaluation utilities; provided with no license (??!)
* features2d
  - For now, only contains a Local Binary Similarity Pattern (LBSP) feature descriptor
* imgproc
  - Contains various image processing utilities and classes for edge detection (work in progress)
* utils
  - Contains miscellaneous utilities required in other modules (e.g. dataset parsers, OpenGL wrappers, platform wrappers, distance functions, etc.)
* video
  - For now, contains multiple background subtraction algorithm implementations (LOBSTER, SuBSENSE, PAWCS...), but might eventually inherit tracking algorithms
* vptz
  - Contains a compact version of the [VirtualPTZ library](https://bitbucket.org/pierre_luc_st_charles/virtualptz_standalone) used to evaluate PTZ trackers
* world
  - Contains nothing, simply wraps other modules into one big spaghetti ball that's easier/faster to link elsewhere

Framework Requirements
----------------------
* [CMake](https://cmake.org/) >= 3.1.0 (required)
* [OpenCV](http://opencv.org/) >= 3.0.0 (required)
* OpenGL >= 4.3 (optional, for GLSL impl)
* [GLFW](http://www.glfw.org/) >= 3.0.0 or [FreeGLUT](http://freeglut.sourceforge.net/) >= 2.8.0 (optional, for GLSL impl)
* [GLEW](http://glew.sourceforge.net/) >= 1.9.0 (optional, for GLSL impl)
* (CUDA/OpenCL will eventually be added as optional)

Notes
-----
For a more user-friendly, stable and documented background subtraction/video segmentation framework with various other utilities, see [Andrews Sobral's BGSLibrary](https://github.com/andrewssobral/bgslibrary).

Release notes and versions for the framework are maintained through [Github Releases](https://github.com/plstcharles/litiv/releases). For now, no binaries (executables or libraries) are provided; you are expected to configure and compile the framework yourself based on your own needs & your hardware's capabilities.

Citation
--------
If you use a module from this framework in your own work, please cite its related LITIV publication(s) as acknowledgement. See [this page](http://www.polymtl.ca/litiv/pub/index.php) for a full list.

Contributing
------------
If you have a something you'd like to contribute to the LITIV framework, send me a message first; I'll be happy to merge your code in if it fits with the rest of the framework.