
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_files(SOURCE_FILES
    "src/BackgroundSubtractionUtils.cpp"
    "src/BackgroundSubtractorLBSP.cpp"
    "src/BackgroundSubtractorLOBSTER.cpp"
    "src/BackgroundSubtractorPAWCS.cpp"
    "src/BackgroundSubtractorPBAS.cpp"
    "src/BackgroundSubtractorSuBSENSE.cpp"
    "src/BackgroundSubtractorViBe.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/video/BackgroundSubtractionUtils.hpp"
    "include/litiv/video/BackgroundSubtractorLBSP.hpp"
    "include/litiv/video/BackgroundSubtractorLOBSTER.hpp"
    "include/litiv/video/BackgroundSubtractorPAWCS.hpp"
    "include/litiv/video/BackgroundSubtractorPBAS.hpp"
    "include/litiv/video/BackgroundSubtractorSuBSENSE.hpp"
    "include/litiv/video/BackgroundSubtractorViBe.hpp"
    "include/litiv/video/VideoCosegmentationUtils.hpp"
    "include/litiv/video/VideoCosegmentationUtils.inl.hpp"
    "include/litiv/video.hpp"
)

litiv_module(video SOURCE_FILES INCLUDE_FILES)
target_link_libraries(litiv_video
    PUBLIC
        litiv_imgproc
        litiv_features2d
)
