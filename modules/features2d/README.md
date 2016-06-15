LITIV *features2d* Module
-------------------------
For now, this module only contains a feature descriptor class for Local Binary Similarity Patterns (LBSPs, [CRV2013](http://dx.doi.org/10.1109/CRV.2013.29)). These features can be used for dense image description (e.g. texture analysis) or keypoint-based description. If SSE2+ is available, intrinsics will automatically be used to drastically accelerate their computation.

Besides, an implementation of Felzenszwalb’s HOG features with proper optimization under an OpenCV interface should eventually be added to this module.
