
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
    "src/EdgeDetectionUtils.cpp"
    "src/EdgeDetectorCanny.cpp"
    "src/EdgeDetectorLBSP.cpp"
    "src/imgproc.cpp"
)
add_files(INCLUDE_FILES
    "include/litiv/imgproc/CosegmentationUtils.hpp"
    "include/litiv/imgproc/EdgeDetectionUtils.hpp"
    "include/litiv/imgproc/EdgeDetectorCanny.hpp"
    "include/litiv/imgproc/EdgeDetectorLBSP.hpp"
    "include/litiv/imgproc/StereoMatcher.hpp"
    "include/litiv/imgproc.hpp"
)

if(USE_CUDA)
    add_files(SOURCE_FILES
        "src/SLIC.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/imgproc/SLIC.hpp"
    )
endif()

litiv_module(imgproc SOURCE_FILES INCLUDE_FILES)
target_link_libraries(litiv_imgproc
    PUBLIC
        litiv_features2d
        litiv_utils
)