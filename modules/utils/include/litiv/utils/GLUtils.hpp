
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// @@@@@ TODO: move all exception/log/assert defines to defineutils & cxxutils

#include "litiv/utils/DefineUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"
#include "litiv/utils/CxxUtils.hpp"
#include <GL/glew.h>
#include <GL/glu.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>
#include <sstream>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#if HAVE_GLFW
#include <GLFW/glfw3.h>
struct glfwWindowDeleter {
    void operator()(GLFWwindow* pWindow) {
        glfwDestroyWindow(pWindow);
    }
};
#endif //HAVE_GLFW
#if HAVE_FREEGLUT
#include <GL/freeglut.h>
struct glutHandle {
    glutHandle() : m_nHandle(0) {}
    glutHandle(std::nullptr_t) : m_nHandle(0) {}
    explicit glutHandle(int v) : m_nHandle(v) {}
    glutHandle& operator=(std::nullptr_t) {m_nHandle = 0; return *this;}
    explicit operator bool() const {return m_nHandle!=0;}
    int m_nHandle;
};
inline bool operator==(const glutHandle& lhs, const glutHandle& rhs) {return lhs.m_nHandle==rhs.m_nHandle;}
inline bool operator!=(const glutHandle& lhs, const glutHandle& rhs) {return lhs.m_nHandle!=rhs.m_nHandle;}
struct glutWindowDeleter {
    void operator()(const glutHandle& oWindowHandle) {
        glutDestroyWindow(oWindowHandle.m_nHandle);
    }
    typedef glutHandle pointer;
};
#endif //HAVE_FREEGLUT

#define glError(msg) throw CxxUtils::Exception(std::string("[glError] ")+msg,__PRETTY_FUNCTION__,__FILE__,__LINE__)
#define glErrorExt(msg,...) throw CxxUtils::Exception(std::string("[glError] ")+msg,__PRETTY_FUNCTION__,__FILE__,__LINE__,__VA_ARGS__)
#define glAssert(expr) {if(!!(expr)); else glError("assertion failed ("#expr")");}
#define glErrorCheck { \
    GLenum __errn = glGetError(); \
    if(__errn!=GL_NO_ERROR) \
        glErrorExt("glErrorCheck failed [code=%d, msg=%s]",__errn,gluErrorString(__errn)); \
}
#ifdef _DEBUG
#define glDbgAssert(expr) glAssert(expr)
#define glDbgErrorCheck glErrorCheck
#define glDbgExceptionWatch CxxUtils::UncaughtExceptionLogger __logger(__PRETTY_FUNCTION__,__FILE__,__LINE__)
#else //!defined(_DEBUG)
#define glDbgAssert(expr)
#define glDbgErrorCheck
#define glDbgExceptionWatch
#endif //!defined(_DEBUG)

template<size_t nGLVerMajor=TARGET_GL_VER_MAJOR,size_t nGLVerMinor=TARGET_GL_VER_MINOR>
class GLContext {
public:
    GLContext(const cv::Size& oWinSize, const std::string& sWinName, bool bHide=true) {
#if HAVE_GLFW
        std::call_once(GetInitFlag(),[](){
            if(glfwInit()==GL_FALSE)
                glError("Failed to init GLFW");
            std::atexit(glfwTerminate);
        });
        if(nGLVerMajor>3 || (nGLVerMajor==3 && nGLVerMinor>=2))
            glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,nGLVerMajor);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,nGLVerMinor);
        glfwWindowHint(GLFW_RESIZABLE,GL_FALSE);
        if(bHide)
            glfwWindowHint(GLFW_VISIBLE,GL_FALSE);
        m_pWindowHandle = std::unique_ptr<GLFWwindow,glfwWindowDeleter>(glfwCreateWindow(oWinSize.width,oWinSize.height,sWinName.c_str(),nullptr,nullptr),glfwWindowDeleter());
        if(!m_pWindowHandle.get())
            glErrorExt("Failed to create [%d,%d] window via GLFW for core GL profile v%d.%d",oWinSize.width,oWinSize.height,nGLVerMajor,nGLVerMinor);
        glfwMakeContextCurrent(m_pWindowHandle.get());
#elif HAVE_FREEGLUT
        std::call_once(GetInitFlag(),[](){
            int argc = 0;
            glutInit(&argc,NULL);
        });
        glutInitDisplayMode(GLUT_SINGLE);
        glutInitWindowSize(oWinSize.width,oWinSize.height);
        glutInitWindowPosition(0,0);
        m_oWindowHandle = std::unique_ptr<glutHandle,glutWindowDeleter>(glutHandle(glutCreateWindow(sWinName.c_str())),glutWindowDeleter());
        if(!(m_oWindowHandle.get().m_nHandle))
            glError("Failed to create window via glut");
        glutSetWindow(m_oWindowHandle.get().m_nHandle);
        if(bHide)
            glutHideWindow();
#endif //HAVE_FREEGLUT
        initGLEW();
    }

    void SetAsActive() {
#if HAVE_GLFW
        glfwMakeContextCurrent(m_pWindowHandle.get());
#elif HAVE_FREEGLUT
        glutSetWindow(m_oWindowHandle.get().m_nHandle);
#endif //HAVE_FREEGLUT
    }

    //constexpr static std::string GetGLEWVersionString() {return std::string("GL_VERSION_")+CxxUtils::MetaStrConcatenator<typename CxxUtils::MetaITOA<nGLVerMajor>::type,CxxUtils::MetaStr<'_'>,typename CxxUtils::MetaITOA<nGLVerMinor>::type,CxxUtils::MetaStr<'\0'>>::type::value;}
    constexpr static std::string GetGLEWVersionString() {return std::string("GL_VERSION_")+std::to_string(nGLVerMajor)+"_"+std::to_string(nGLVerMinor);}

    static void initGLEW() {
        glErrorCheck;
        glewExperimental = GLEW_EXPERIMENTAL?GL_TRUE:GL_FALSE;
        const GLenum glewerrn = glewInit();
        if(glewerrn!=GLEW_OK)
            glErrorExt("Failed to init GLEW [code=%d, msg=%s]",glewerrn,glewGetErrorString(glewerrn));
        const GLenum errn = glGetError();
        // see glew init GL_INVALID_ENUM bug discussion at https://www.opengl.org/wiki/OpenGL_Loading_Library
        if(errn!=GL_NO_ERROR && errn!=GL_INVALID_ENUM)
            glErrorExt("Unexpected GLEW init error [code=%d, msg=%s]",errn,gluErrorString(errn));
        if(!glewIsSupported(GetGLEWVersionString().c_str()))
            glErrorExt("Bad GL core/ext version detected (target is %s)",GetGLEWVersionString().c_str());
    }

private:

#if HAVE_GLFW
    std::unique_ptr<GLFWwindow,glfwWindowDeleter> m_pWindowHandle;
#elif HAVE_FREEGLUT
    std::unique_ptr<glutHandle,glutWindowDeleter> m_oWindowHandle;
#endif //HAVE_FREEGLUT
    GLContext& operator=(const GLContext&) = delete;
    GLContext(const GLContext&) = delete;
    static std::once_flag& GetInitFlag() {static std::once_flag oInitFlag; return oInitFlag;}
};
using GLDefaultContext = GLContext<>;

namespace GLUtils {

    static inline bool isInternalFormatSupported(GLenum eInternalFormat) {
        switch(eInternalFormat) {
            case GL_R8:
            case GL_R8UI:
            case GL_R32UI:
            case GL_R32F:
            case GL_RGB32UI:
            case GL_RGB32F:
            case GL_RGBA8:
            case GL_RGBA8UI:
            case GL_RGBA32UI:
            case GL_RGBA32F:
                return true;
            default:
                return false;
        }
    }

    static inline bool isMatTypeSupported(int nType) {
        switch(nType) {
            case CV_8UC1:
            case CV_8UC3:
            case CV_8UC4:
            case CV_32FC1:
            case CV_32FC3:
            case CV_32FC4:
            case CV_32SC1:
            case CV_32SC3:
            case CV_32SC4:
                return true;
            default:
                return false;
        }
    }

    static inline GLenum getInternalFormatFromMatType(int nTextureType, bool bUseIntegralFormat=true) {
        switch(nTextureType) {
            case CV_8UC1:
                return bUseIntegralFormat?GL_R8UI:GL_R8;
            case CV_8UC3:
            case CV_8UC4:
                return bUseIntegralFormat?GL_RGBA8UI:GL_RGBA8;
            case CV_32FC1:
                return GL_R32F;
            case CV_32FC3:
                return GL_RGB32F;
            case CV_32FC4:
                return GL_RGBA32F;
            case CV_32SC1:
                return bUseIntegralFormat?GL_R32UI:glError("opencv mat input format type cannot be matched to a predefined setup");
            case CV_32SC3:
                return bUseIntegralFormat?GL_RGB32UI:glError("opencv mat input format type cannot be matched to a predefined setup");
            case CV_32SC4:
                return bUseIntegralFormat?GL_RGBA32UI:glError("opencv mat input format type cannot be matched to a predefined setup");
            default:
                glError("opencv mat input format type did not match any predefined setup");
        }
    }

    static inline GLenum getNormalizedIntegralFormatFromInternalFormat(GLenum eInternalFormat) {
        switch(eInternalFormat) {
            case GL_R8UI:
                return GL_R8;
            case GL_RGBA8UI:
                return GL_RGBA8;
            case GL_R8:
            case GL_R32F:
            case GL_RGBA8:
            case GL_RGB32F:
            case GL_RGBA32F:
                return eInternalFormat;
            case GL_R32UI:
            case GL_RGB32UI:
            case GL_RGBA32UI:
            default:
                glError("input internal format did not match any predefined normalized format");
        }
    }

    static inline GLenum getIntegralFormatFromInternalFormat(GLenum eInternalFormat) {
        switch(eInternalFormat) {
            case GL_R8:
                return GL_R8UI;
            case GL_RGBA8:
                return GL_RGBA8UI;
            case GL_R8UI:
            case GL_R32UI:
            case GL_RGBA8UI:
            case GL_RGB32UI:
            case GL_RGBA32UI:
                return eInternalFormat;
            case GL_R32F:
            case GL_RGB32F:
            case GL_RGBA32F:
            default:
                glError("input normalized format did not match any predefined integral format");
                return 0;
        }
    }

    static inline const char* getGLSLFormatNameFromInternalFormat(GLenum eInternalFormat) {
        switch(eInternalFormat) {
            case GL_R8UI:
                return "r8ui";
            case GL_R8:
                return "r8";
            case GL_RGBA8UI:
                return "rgba8ui";
            case GL_RGBA8:
                return "rgba8";
            case GL_R32F:
                return "r32f";
            case GL_RGB32F:
                return "rgb32f";
            case GL_RGBA32F:
                return "rgba32f";
            case GL_R32UI:
                return "r32ui";
            case GL_RGB32UI:
                return "rgb32ui";
            case GL_RGBA32UI:
                return "rgba32ui";
            default:
                glError("input internal format did not match any predefined glsl layout");
        }
    }

    static inline bool isInternalFormatIntegral(GLenum eInternalFormat) {
        switch(eInternalFormat) {
            case GL_R8UI:
            case GL_R32UI:
            case GL_RGBA8UI:
            case GL_RGB32UI:
            case GL_RGBA32UI:
                return true;
            case GL_R8:
            case GL_RGBA8:
            case GL_R32F:
            case GL_RGB32F:
            case GL_RGBA32F:
                return false;
            default:
                glError("input internal format did not match any predefined integer format setup");
        }
    }

    static inline int getMatTypeFromInternalFormat(GLenum eInternalFormat) {
        switch(eInternalFormat) {
            case GL_R8UI:
            case GL_R8:
                return CV_8UC1;
            case GL_RGBA8UI:
            case GL_RGBA8:
                return CV_8UC4;
            case GL_R32F:
                return CV_32FC1;
            case GL_RGB32F:
                return CV_32FC3;
            case GL_RGBA32F:
                return CV_32FC4;
            case GL_R32UI:
                return CV_32SC1;
            case GL_RGB32UI:
                return CV_32SC3;
            case GL_RGBA32UI:
                return CV_32SC4;
            default:
                glError("input internal format did not match any predefined opencv mat setup");
                return 0;
        }
    }

    static inline GLenum getDataFormatFromChannels(int nTextureChannels, bool bUseIntegralFormat=true) {
        switch(nTextureChannels) {
            case 1:
                return bUseIntegralFormat?GL_RED_INTEGER:GL_RED;
            case 3:
                return bUseIntegralFormat?GL_BGR_INTEGER:GL_BGR;
            case 4:
                return bUseIntegralFormat?GL_BGRA_INTEGER:GL_BGRA;
            default:
                glError("opencv mat input format type did not match any predefined setup");
        }
    }

    static inline int getChannelsFromDataFormat(GLenum eDataFormat) {
        switch(eDataFormat) {
            case GL_RED:
            case GL_RED_INTEGER:
                return 1;
            case GL_BGR:
            case GL_BGR_INTEGER:
                return 3;
            case GL_BGRA:
            case GL_BGRA_INTEGER:
                return 4;
            default:
                glError("input data format did not match any predefined opencv mat setup");
        }
    }

    static inline GLenum getDataTypeFromMatDepth(int nTextureDepth, int nTextureChannels) {
        switch(nTextureDepth) {
            case CV_8U:
                return nTextureChannels==4?GL_UNSIGNED_INT_8_8_8_8_REV:GL_UNSIGNED_BYTE;
            case CV_32F:
                return GL_FLOAT;
            case CV_32S:
                return GL_UNSIGNED_INT;
            default:
                glError("opencv mat input format type did not match any predefined setup");
        }
    }

    static inline int getMatDepthFromDataType(GLenum eDataType) {
        switch(eDataType) {
            case GL_UNSIGNED_BYTE:
            case GL_UNSIGNED_INT_8_8_8_8_REV:
                return CV_8U;
            case GL_FLOAT:
                return CV_32F;
            case GL_UNSIGNED_INT:
                return CV_32S;
            default:
                glError("input data type did not match any predefined opencv mat setup");
        }
    }

    static inline int getByteSizeFromMatDepth(int nDepth) {
        switch(nDepth) {
            case CV_8U:
                return 1;
            case CV_32F:
            case CV_32S:
                return 4;
            default:
                glError("input depth did not match any predefined byte size");
        }
    }

    static inline int getChannelsFromMatType(int nType) {
        switch(nType) {
            case CV_8UC1:
            case CV_32FC1:
            case CV_32SC1:
                return 1;
            case CV_8UC3:
            case CV_32FC3:
            case CV_32SC3:
                return 3;
            case CV_8UC4:
            case CV_32FC4:
            case CV_32SC4:
                return 4;
            default:
                glError("input type did not match any predefined channel size");
        }
    }

    static inline int getChannelsFromInternalFormat(GLenum eInternalFormat) {
        switch(eInternalFormat) {
            case GL_R8:
            case GL_R8UI:
            case GL_R32F:
            case GL_R32UI:
                return 1;
            case GL_RGB32F:
            case GL_RGB32UI:
                return 3;
            case GL_RGBA8:
            case GL_RGBA8UI:
            case GL_RGBA32F:
            case GL_RGBA32UI:
                return 4;
            default:
                glError("input internal format did not match any predefined channel size");
                return 0;
        }
    }

    template<uint N>
    static inline typename std::enable_if<(N==1),int>::type getIntegerVal(GLenum eParamName) {
        int nVal;
        glGetIntegerv(eParamName,&nVal);
        glErrorCheck;
        return nVal;
    }

    template<uint N>
    static inline typename std::enable_if<(N>1),std::array<int,N>>::type getIntegerVal(GLenum eParamName) {
        std::array<int,N> anVal;
        for(uint n=0; n<N; ++n)
            glGetIntegeri_v(eParamName,n,&anVal[n]);
        glErrorCheck;
        return anVal;
    }

    static inline cv::Mat deepCopyImage(GLsizei nWidth,
                                        GLsizei nHeight,
                                        GLvoid* pData,
                                        GLenum eDataFormat,
                                        GLenum eDataType) {
        glAssert(nWidth>0 && nHeight>0 && pData);
        const int nDepth = getMatDepthFromDataType(eDataType);
        const int nChannels = getChannelsFromDataFormat(eDataFormat);
        return cv::Mat(nHeight,nWidth,CV_MAKETYPE(nDepth,nChannels),pData,nWidth*nChannels*getByteSizeFromMatDepth(nDepth)).clone();
    }

    static inline std::vector<cv::Mat> deepCopyImages(const std::vector<cv::Mat>& voInputMats) {
        glAssert(!voInputMats.empty() && !voInputMats[0].empty());
        std::vector<cv::Mat> voOutputMats;
        for(size_t nMatIter=0; nMatIter<voInputMats.size(); ++nMatIter)
            voOutputMats.push_back(voInputMats[nMatIter].clone());
        return voOutputMats;
    }

    static inline std::vector<cv::Mat> deepCopyImages( GLsizei nTextureCount,
                                                       GLsizei nWidth,
                                                       GLsizei nHeight,
                                                       GLvoid* pData,
                                                       GLenum eDataFormat,
                                                       GLenum eDataType) {
        glAssert(nTextureCount>0 && nWidth>0 && nHeight>0 && pData);
        std::vector<cv::Mat> voOutputMats;
        const int nDepth = getMatDepthFromDataType(eDataType);
        const int nChannels = getChannelsFromDataFormat(eDataFormat);
        const int nImageSize = nHeight*nWidth*nChannels*getByteSizeFromMatDepth(nDepth);
        for(int nMatIter=0; nMatIter<nTextureCount; ++nMatIter)
            voOutputMats.push_back(deepCopyImage(nWidth,nHeight,((char*)pData)+(nMatIter*nImageSize),eDataFormat,eDataType));
        return voOutputMats;
    }

    static inline std::string addLineNumbersToString(const std::string& sSrc, bool bPrefixTab) {
        if(sSrc.empty())
            return std::string();
        std::stringstream ssRes;
        int nCurrLine = 1;
        ssRes << (bPrefixTab?cv::format("\t%06d: ",nCurrLine):cv::format("%06d: ",nCurrLine));
        for(auto pcSrcCharIter=sSrc.begin(); pcSrcCharIter!=sSrc.end(); ++pcSrcCharIter) {
            if(*pcSrcCharIter=='\n') {
                ++nCurrLine;
                ssRes << (bPrefixTab?"\n\t":"\n") << cv::format("%06d: ",nCurrLine);
            }
            else
                ssRes << *pcSrcCharIter;

        }
        return ssRes.str();
    }

    struct alignas(32) TMT32GenParams {
        uint status[4];
        uint mat1;
        uint mat2;
        uint tmat;
        uint pad;
    };

    template<typename Talloc>
    static inline void initTinyMT32Generators(glm::uvec3 vGeneratorLayout,std::vector<TMT32GenParams,Talloc>& voData) {
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

    static inline std::string getShaderFunctionSource_getRandNeighbor3x3(size_t nBorderSize,const cv::Size& oFrameSize) {
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

} //namespace GLUtils
