3rd Party
---------
This directory contains libraries & frameworks developed in part or entirely by 3rd-parties, some of which may have been adapted to be used as modules in the LITIV framework. These may be licensed on terms different from the LITIV framework; see each subdirectory's LICENSE file (if any) for more information.

Module list:
* [**bsds500 (module)**](./bsds500/) : Contains a cleaned/optimized version of the [BSDS500](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) boundary detection evaluation utilities, used in the datasets module.
* [**dshowbase (module)**](./dshowbase/) : Contains the [DirectShow base classes](https://msdn.microsoft.com/en-us/library/windows/desktop/dd375456(v=vs.85).aspx) as distributed in the Windows 7 SDK, and more DirectShow utility classes for filter usage.
* [**googlebench**](./googlebench) : Contains an importation script for the [Google Benchmark](https://github.com/google/benchmark) framework used for performance testing (allows in-tree use of original targets).
* [**googletest**](./googletest) : Contains an importation script for the [Google Test](https://github.com/google/googletest) framework used for regression testing (allows in-tree use of original targets).
* [**sospd**](./sospd) : Contains [A. Fix's optimizers](https://github.com/letterx/sospd) for higher-order multilabel Markov Random Fields (requires boost); used in the imgproc module for inference.

For more information on these modules, see their respective README files.
