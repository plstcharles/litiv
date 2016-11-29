FROM ubuntu:16.04
MAINTAINER Pierre-Luc St-Charles <pierre-luc.st-charles@polymtl.ca>
LABEL Description="This base image contains all LITIV framework dependencies"

ENV opencvtag=3.1.0
ENV nbthreads=4

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev \
    pkg-config \
    wget \
    curl \
    python \
    python-dev \
    python-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    zlib1g-dev \
    libdc1394-22-dev \
    libeigen3-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libglm-dev \
    freeglut3-dev \
    libglew-dev \
    unzip \
 && rm -rf /var/lib/apt/lists/*

RUN git clone -b ${opencvtag} --progress --verbose --single-branch https://github.com/opencv/opencv.git /opencv && git clone -b ${opencvtag} --progress --verbose --single-branch https://github.com/opencv/opencv_contrib.git /opencv_contrib
WORKDIR /opencv/build
RUN cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
    .. && make -j${nbthreads} install && make clean

RUN git clone -b master --progress --verbose --single-branch https://github.com/plstcharles/opengm.git /opengm
WORKDIR /opengm/build
RUN cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTING=OFF \
    -D BUILD_TUTORIALS=OFF \
    -D INSTALL_EXTERNAL_LIB=ON \
    -D WITH_GCO=ON \
    -D WITH_MAXFLOW=ON \
    -D WITH_OPENMP=ON \
    -D WITH_QPBO=ON \
    -D WITH_TRWS=ON \
    .. \
 && make externalLibs && cmake .. && make -j${nbthreads} install && make clean

RUN ldconfig
CMD ["bash"]
