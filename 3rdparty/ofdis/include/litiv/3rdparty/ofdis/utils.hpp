
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <cstdio>
#include <sys/time.h>

namespace ofdis {

    /// input configuration type list for ofdis algorithm
    enum FlowInputType {
        FlowInput_Grayscale,
        FlowInput_Gradient,
        FlowInput_RGB,
    };

    /// output configuration type list for ofdis algorithm
    enum FlowOutputType {
        FlowOutput_OpticalFlow,
        FlowOutput_StereoDepth,
    };

    /// type shortcut used in internal impl
    typedef __v4sf v4sf;

} // namespace ofdis