
#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <cstdio>
#include <sys/time.h>
#ifdef OFDIS_INTERNAL
#if defined(_MSC_VER)
#include <intrin.h>
#else //(!defined(_MSC_VER))
#include <x86intrin.h>
#endif //(!defined(_MSC_VER))
#else //ndef(OFDIS_INTERNAL)
#ifndef OFDIS_API
#error "must only include 'ofdis.hpp' header for API"
#endif //ndef(OFDIS_API
#endif //ndef(OFDIS_INTERNAL)

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

#ifdef OFDIS_INTERNAL

    /// type shortcut used in internal impl
    typedef __v4sf v4sf;

    /// type shortcut for multi-channel images
    template<FlowInputType eInput>
    using InputImageType = std::conditional_t<(eInput==ofdis::FlowInput_RGB),color_image_t,image_t>;

#endif //OFDIS_INTERNAL

} // namespace ofdis