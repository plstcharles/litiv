#pragma once

namespace RandUtils {

    /*// gaussian 3x3 pattern, based on 'floor(fspecial('gaussian', 3, 1)*256)'
    static const int s_nSamplesInitPatternWidth = 3;
    static const int s_nSamplesInitPatternHeight = 3;
    static const int s_nSamplesInitPatternTot = 256;
    static const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
        {19,    32,    19,},
        {32,    52,    32,},
        {19,    32,    19,},
    };*/

    // gaussian 7x7 pattern, based on 'floor(fspecial('gaussian',7,2)*512)'
    static const int s_nSamplesInitPatternWidth = 7;
    static const int s_nSamplesInitPatternHeight = 7;
    static const int s_nSamplesInitPatternTot = 512;
    static const int s_anSamplesInitPattern[s_nSamplesInitPatternHeight][s_nSamplesInitPatternWidth] = {
        {2,     4,     6,     7,     6,     4,     2,},
        {4,     8,    12,    14,    12,     8,     4,},
        {6,    12,    21,    25,    21,    12,     6,},
        {7,    14,    25,    28,    25,    14,     7,},
        {6,    12,    21,    25,    21,    12,     6,},
        {4,     8,    12,    14,    12,     8,     4,},
        {2,     4,     6,     7,     6,     4,     2,},
    };

    //! returns a random init/sampling position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    static inline void getRandSamplePosition(int& x_sample, int& y_sample, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
        int r = 1+rand()%s_nSamplesInitPatternTot;
        for(x_sample=0; x_sample<s_nSamplesInitPatternWidth; ++x_sample) {
            for(y_sample=0; y_sample<s_nSamplesInitPatternHeight; ++y_sample) {
                r -= s_anSamplesInitPattern[y_sample][x_sample];
                if(r<=0)
                    goto stop;
            }
        }
        stop:
        x_sample += x_orig-s_nSamplesInitPatternWidth/2;
        y_sample += y_orig-s_nSamplesInitPatternHeight/2;
        if(x_sample<border)
            x_sample = border;
        else if(x_sample>=imgsize.width-border)
            x_sample = imgsize.width-border-1;
        if(y_sample<border)
            y_sample = border;
        else if(y_sample>=imgsize.height-border)
            y_sample = imgsize.height-border-1;
    }

    // simple 8-connected (3x3) neighbors pattern
    static const int s_anNeighborPatternSize_3x3 = 8;
    static const int s_anNeighborPattern_3x3[8][2] = {
        {-1, 1},  { 0, 1},  { 1, 1},
        {-1, 0},            { 1, 0},
        {-1,-1},  { 0,-1},  { 1,-1},
    };

    //! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    static inline void getRandNeighborPosition_3x3(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
        int r = rand()%s_anNeighborPatternSize_3x3;
        x_neighbor = x_orig+s_anNeighborPattern_3x3[r][0];
        y_neighbor = y_orig+s_anNeighborPattern_3x3[r][1];
        if(x_neighbor<border)
            x_neighbor = border;
        else if(x_neighbor>=imgsize.width-border)
            x_neighbor = imgsize.width-border-1;
        if(y_neighbor<border)
            y_neighbor = border;
        else if(y_neighbor>=imgsize.height-border)
            y_neighbor = imgsize.height-border-1;
    }

    // 5x5 neighbors pattern
    static const int s_anNeighborPatternSize_5x5 = 24;
    static const int s_anNeighborPattern_5x5[24][2] = {
        {-2, 2},  {-1, 2},  { 0, 2},  { 1, 2},  { 2, 2},
        {-2, 1},  {-1, 1},  { 0, 1},  { 1, 1},  { 2, 1},
        {-2, 0},  {-1, 0},            { 1, 0},  { 2, 0},
        {-2,-1},  {-1,-1},  { 0,-1},  { 1,-1},  { 2,-1},
        {-2,-2},  {-1,-2},  { 0,-2},  { 1,-2},  { 2,-2},
    };

    //! returns a random neighbor position for the specified pixel position; also guards against out-of-bounds values via image/border size check.
    static inline void getRandNeighborPosition_5x5(int& x_neighbor, int& y_neighbor, const int x_orig, const int y_orig, const int border, const cv::Size& imgsize) {
        int r = rand()%s_anNeighborPatternSize_5x5;
        x_neighbor = x_orig+s_anNeighborPattern_5x5[r][0];
        y_neighbor = y_orig+s_anNeighborPattern_5x5[r][1];
        if(x_neighbor<border)
            x_neighbor = border;
        else if(x_neighbor>=imgsize.width-border)
            x_neighbor = imgsize.width-border-1;
        if(y_neighbor<border)
            y_neighbor = border;
        else if(y_neighbor>=imgsize.height-border)
            y_neighbor = imgsize.height-border-1;
    }

#if HAVE_GLSL

    struct TMT32GenParams {
        uint status[4];
        uint mat1;
        uint mat2;
        uint tmat;
        uint pad;
    };

    static inline void initTinyMT32Generators(glm::uvec3 vGeneratorLayout, std::vector<TMT32GenParams>& voData) {
        glAssert(vGeneratorLayout.x>0 && vGeneratorLayout.y>0 && vGeneratorLayout.z>0);
        voData.resize(vGeneratorLayout.x*vGeneratorLayout.y*vGeneratorLayout.z);
        TMT32GenParams* pData = voData.data();
        // tinymt32dc:cecf43a2417bd5c41e5d6f80cf2ce903,32,1337,f20d1b78,ff90ffe5,30fbdfff,65,0
        for(size_t z=0; z<vGeneratorLayout.z; ++z) {
            const size_t nStepSize_Z = z*vGeneratorLayout.y*vGeneratorLayout.x;
            for(size_t y=0; y<vGeneratorLayout.y; ++y) {
                const size_t nStepSize_Y = y*vGeneratorLayout.x + nStepSize_Z;
                for(size_t x=0; x<vGeneratorLayout.x; ++x) {
                    const size_t nStepSize_X = x + nStepSize_Y;
                    TMT32GenParams* pCurrGenParams = pData+nStepSize_X;
                    pCurrGenParams->status[0] = (uint)rand();
                    pCurrGenParams->status[1] = pCurrGenParams->mat1 = 0xF20D1B78;
                    pCurrGenParams->status[2] = pCurrGenParams->mat2 = 0xFF90FFE5;
                    pCurrGenParams->status[3] = pCurrGenParams->tmat = 0x30FBDFFF;
                    pCurrGenParams->pad = 1337;
                    for(int nLoop=1; nLoop<8; ++nLoop)
                        pCurrGenParams->status[nLoop&3] ^= nLoop+UINT32_C(1812433253)*((pCurrGenParams->status[(nLoop-1)&3])^(pCurrGenParams->status[(nLoop-1)&3]>>30));
                    for(int nLoop=0; nLoop<8; ++nLoop) {
                        uint s0 = pCurrGenParams->status[3];
                        uint s1 = (pCurrGenParams->status[0]&UINT32_C(0x7fffffff))^(pCurrGenParams->status[1])^(pCurrGenParams->status[2]);
                        s1 ^= (s1<<1);
                        s0 ^= (s0>>1)^s1;
                        pCurrGenParams->status[0] = pCurrGenParams->status[1];
                        pCurrGenParams->status[1] = pCurrGenParams->status[2];
                        pCurrGenParams->status[2] = s1^(s0<<10);
                        pCurrGenParams->status[3] = s0;
                        pCurrGenParams->status[1] ^= -((int)(s0&1))&(pCurrGenParams->mat1);
                        pCurrGenParams->status[2] ^= -((int)(s0&1))&(pCurrGenParams->mat2);
                    }
                }
            }
        }
    }

    static inline std::string getShaderFunctionSource_getRandNeighbor3x3(size_t nBorderSize, const cv::Size& oFrameSize) {
        std::stringstream ssSrc;
        ssSrc << "const ivec2 _avNeighborPattern3x3[8] = ivec2[8](\n"
                 "    ivec2(-1, 1),ivec2(0, 1),ivec2(1, 1),\n"
                 "    ivec2(-1, 0),            ivec2(1, 0),\n"
                 "    ivec2(-1,-1),ivec2(0,-1),ivec2(1,-1)\n"
                 ");\n"
                 "ivec2 getRandNeighbor3x3(in ivec2 vCurrPos, in uint nRandVal) {\n";
        if(nBorderSize>0) ssSrc <<
                 "    const int nBorderSize = " << nBorderSize << ";\n";
        ssSrc << "    const int nFrameWidth = " << oFrameSize.width << ";\n"
                 "    const int nFrameHeight = " << oFrameSize.height << ";\n"
                 "    ivec2 vNeighborPos = vCurrPos+_avNeighborPattern3x3[nRandVal%8];\n";
        if(nBorderSize>0) ssSrc <<
                 "    clamp(vNeighborPos,ivec2(nBorderSize),ivec2(nFrameWidth-nBorderSize-1,nFrameHeight-nBorderSize-1));\n";
        else ssSrc <<
                 "    clamp(vNeighborPos,ivec2(0),ivec2(nFrameWidth-1,nFrameHeight-1));\n";
        ssSrc << "    return vNeighborPos;\n"
                 "}\n";
        return ssSrc.str();
    }

    static inline std::string getShaderFunctionSource_frand() {
        std::stringstream ssSrc;
        ssSrc << "float frand(inout vec2 vSeed) {\n"
                 "    float fRandVal = 0.5 + 0.5 * fract(sin(dot(vSeed.xy, vec2(12.9898, 78.233)))* 43758.5453);\n"
                 "    vSeed *= fRandVal;\n"
                 "    return fRandVal;\n"
                 "}\n";
        return ssSrc.str();
    }

    static inline std::string getShaderFunctionSource_urand() {
        std::stringstream ssSrc;
        // 1x iter of Bob Jenkins' "One-At-A-Time" hashing algorithm
        ssSrc << "uint urand(inout uint nSeed) {\n"
                 "   nSeed += (nSeed<<10u);\n"
                 "   nSeed ^= (nSeed>>6u);\n"
                 "   nSeed += (nSeed<<3u);\n"
                 "   nSeed ^= (nSeed>>11u);\n"
                 "   nSeed += (nSeed<<15u);\n"
                 "   return nSeed;\n"
                 "}\n";
        return ssSrc.str();
    }

    static inline std::string getShaderFunctionSource_urand_tinymt32() {
        std::stringstream ssSrc;
        //
        //                  32-bit Tiny Mersenne Twister
        //
        // Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
        // University and The University of Tokyo. All rights reserved.
        //
        // Redistribution and use in source and binary forms, with or without
        // modification, are permitted provided that the following conditions are
        // met:
        //
        //     * Redistributions of source code must retain the above copyright
        //       notice, this list of conditions and the following disclaimer.
        //     * Redistributions in binary form must reproduce the above
        //       copyright notice, this list of conditions and the following
        //       disclaimer in the documentation and/or other materials provided
        //       with the distribution.
        //     * Neither the name of the Hiroshima University nor the names of
        //       its contributors may be used to endorse or promote products
        //       derived from this software without specific prior written
        //       permission.
        //
        // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
        // "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
        // LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
        // A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
        // OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
        // SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
        // LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
        // DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
        // THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        // (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        ssSrc << "struct TMT32Model {\n"
                 "    uvec4 status;\n"
                 "    uint mat1;\n"
                 "    uint mat2;\n"
                 "    uint tmat;\n"
                 "    uint pad;\n"
                 "};\n"
                 "uint urand(inout TMT32Model p) {\n"
                 "    uint s0 = p.status[3];\n"
                 "    uint s1 = (p.status[0]&0x7fffffff)^p.status[1]^p.status[2];\n"
                 "    s1 ^= (s1<<1);\n"
                 "    s0 ^= (s0>>1)^s1;\n"
                 "    p.status[0] = p.status[1];\n"
                 "    p.status[1] = p.status[2];\n"
                 "    p.status[2] = s1^(s0<<10);\n"
                 "    p.status[3] = s0;\n"
                 "    p.status[1] ^= -int(s0&1)&p.mat1;\n"
                 "    p.status[2] ^= -int(s0&1)&p.mat2;\n"
                 "    uint t0 = p.status[3];\n"
                 "    uint t1 = p.status[0]+(p.status[2]>>8);\n"
                 "    t0 ^= t1;\n"
                 "    t0 ^= -int(t1&1)&p.tmat;\n"
                 "    return t0;\n"
                 "}\n";
        return ssSrc.str();
    }

#endif //HAVE_GLSL

}; //namespace RandUtils
