
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

} //namespace GLUtils
