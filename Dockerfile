
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

FROM plstcharles/litiv-base
MAINTAINER Pierre-Luc St-Charles <pierre-luc.st-charles@polymtl.ca>
LABEL Description="LITIV framework test build"

ARG CMAKECFG_BUILD_SHARED_LIBS=OFF
ENV CMAKECFG_BUILD_SHARED_LIBS=${CMAKECFG_BUILD_SHARED_LIBS}
ARG CMAKECFG_USE_WORLD_SOURCE_GLOB=OFF
ENV CMAKECFG_USE_WORLD_SOURCE_GLOB=${CMAKECFG_USE_WORLD_SOURCE_GLOB}

WORKDIR /litiv/build
ADD . /litiv
RUN cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_SHARED_LIBS=${CMAKECFG_BUILD_SHARED_LIBS} \
    -D USE_WORLD_SOURCE_GLOB=${CMAKECFG_USE_WORLD_SOURCE_GLOB} \
    .. && make -j${nbthreads} && make install
CMD ["/bin/bash"]