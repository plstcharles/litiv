LITIV *imgproc* Module
----------------------
This module contains various image processing utilities and algorithms. The main header file, [imgproc.hpp](./include/litiv/imgproc.hpp), contains implementations for image thinning, non-max suppression, affinity map computation and Gaussian mixture model learning. The module itself contains wrapper interfaces for image warping, edge detection, and mutual segmentation algorithms.

For edge detection, two versions of Canny's method are included: one based on OpenCV's implementation, and one using binary feature convolutions via LBSPs (see our [CVPRW2016 paper](http://www.polymtl.ca/litiv/doc/StCharlesCVPRW2016.pdf) for more information).

If CUDA is found and enabled via CMake, a GPU version of [Achanta et al.'s](https://doi.org/10.1109/TPAMI.2012.120) SLIC superpixels will also be available included (taken from [fderue/SLIC_CUDA](https://github.com/fderue/SLIC_CUDA)).

The mutual segmentation method [included here](./include/litiv/imgproc/SegmMatcher.hpp) was published in an ICCV workshop in 2017; see the associated publication [here](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w6/St-Charles_Mutual_Foreground_Segmentation_ICCV_2017_paper.pdf) for more details.

Finally, the image warping algorithm available [here](./include/litiv/imgproc/imwarp.hpp) is based on a Mean Least Square strategy, and inspired from the implementation of [imgwarp-opencv](https://github.com/cxcxcxcx/imgwarp-opencv).
