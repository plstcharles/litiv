LITIV Modules
-------------
This directory contains the main modules (libraries) of the LITIV framework. These modules are all licensed under the Apache License, Version 2.0 (see [LICENSE.txt](../LICENSE.txt) for more information).

Module list:
* [**datasets**](./datasets/) : Provides dataset parsing and evaluation utilities with precaching and async processing support.
* [**features2d**](./features2d/) : Provides feature descriptors & matchers ([LBSP](./features2d/include/litiv/features2d/LBSP.hpp), [DASC](./features2d/include/litiv/features2d/DASC.hpp), [LSS](./features2d/include/litiv/features2d/LSS.hpp), [MI](./features2d/include/litiv/features2d/MI.hpp), [ShapeContext](./features2d/include/litiv/features2d/SC.hpp)).
* [**imgproc**](./imgproc/) : Provides image processing algos and utilities ([edge detectors](./imgproc/include/litiv/imgproc/EdgeDetectorLBSP.hpp), [shape cosegmenters](./imgproc/include/litiv/imgproc/SegmMatcher.hpp), [superpixels extractors](./imgproc/include/litiv/imgproc/SLIC.hpp)).
* [**test**](./test/) : Provides utilities for project-wide unit/performance testing; never exported for external usage.
* [**utils**](./utils/) : Equivalent of OpenCV's "core" module; contains miscellaneous common utilities ([see list here](./utils/README.md)).
* [**video**](./video/) : Provides background subtraction algos & utilities ([LOBSTER](./video/include/litiv/video/BackgroundSubtractorLOBSTER.hpp), [SuBSENSE](./video/include/litiv/video/BackgroundSubtractorSuBSENSE.hpp), [PAWCS](./video/include/litiv/video/BackgroundSubtractorPAWCS.hpp) and [others](./video/include/litiv/video/)).
* [**vptz**](./vptz/) : Provides a compact version of the [VirtualPTZ library](https://bitbucket.org/pierre_luc_st_charles/virtualptz_standalone) used to evaluate PTZ trackers.
* [**world**](./world/) : Provides nothing, and pre-links other modules to make linking easier/faster elsewhere.

For more information on these modules, see their respective README files.
