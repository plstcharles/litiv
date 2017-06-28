LITIV Modules
-------------
This directory contains the main modules (libraries) of the LITIV framework. These modules are all licensed under the Apache License, Version 2.0 (see [LICENSE.txt](../LICENSE.txt) for more information).

Module list:
* [**datasets**](./datasets/) : Contains dataset parsing and evaluation utilities with data precaching and async processing support.
* [**features2d**](./features2d/) : Contains feature descriptors (e.g. LBSP, DASC, LSS, FHOG).
* [**imgproc**](./imgproc/) : Contains image processing algos and utilities (e.g. for edge detection and superpixel segmentation).
* [**test**](./test/) : Contains common utilities for project-wide unit/performance testing; never exported for external usage.
* [**utils**](./utils/) : Equivalent of OpenCV's "core" module; contains miscellaneous utilities required by other modules.
* [**video**](./video/) : Contains video processing algos and utilities (e.g. for background subtraction/change detection, registration).
* [**vptz**](./vptz/) : Contains a compact version of the [VirtualPTZ library](https://bitbucket.org/pierre_luc_st_charles/virtualptz_standalone) used to evaluate PTZ trackers.
* [**world**](./world/) : Contains nothing, and pre-links other modules to make linking easier/faster elsewhere.

For more information on these modules, see their respective README files.
