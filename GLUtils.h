#pragma once

#include <GL/glew.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <sstream>
#include <exception>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#ifndef STR
#ifndef XSTR
#define XSTR(s) STR(s)
#endif //XSTR
#define STR(s) #s
#endif //STR
#define TARGET_GL_VER_MAJOR  4
#define TARGET_GL_VER_MINOR  4
#define TARGET_GLEW_EXPERIM  1
#define TARGET_GL_VER_STR "GL_VERSION_" XSTR(TARGET_GL_VER_MAJOR) "_" XSTR(TARGET_GL_VER_MINOR)
#ifdef _MSC_VER
    #define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#define glError(msg) throw GLException(msg,__PRETTY_FUNCTION__,__FILE__,__LINE__)
#define glErrorExt(msg,...) throw GLException(msg,__PRETTY_FUNCTION__,__FILE__,__LINE__,__VA_ARGS__)
#define glAssert(expr) {if(!!(expr)); else glError("assertion failed ("#expr")");}
#define glErrorCheck { \
    GLenum __errn = glGetError(); \
    if(__errn!=GL_NO_ERROR) \
        glErrorExt("glErrorCheck failed (code=%d)",__errn); \
}
// see glew init GL_INVALID_ENUM bug discussion at https://www.opengl.org/wiki/OpenGL_Loading_Library
#define glewInitErrorCheck { \
    glErrorCheck; \
    glewExperimental = TARGET_GLEW_EXPERIM?GL_TRUE:GL_FALSE; \
    if(GLenum __errn=glewInit()!=GLEW_OK) \
        glErrorExt("Failed to init GLEW (code=%d)",__errn); \
    else if(GLenum __errn=glGetError()!=GL_INVALID_ENUM) \
        glErrorExt("Unexpected GLEW init error (code=%d)",__errn); \
    if(!glewIsSupported(TARGET_GL_VER_STR)) \
        glErrorExt("Bad GL core/ext version detected (target is %s)",TARGET_GL_VER_STR); \
}
#ifdef _DEBUG
#define glDbgAssert(expr) glAssert(expr)
#define glDbgErrorCheck glErrorCheck
#else
#define glDbgAssert(expr)
#define glDbgErrorCheck
#endif

class GLException : public std::runtime_error {
public:
    template<typename... VALIST> GLException(const char* sErrMsg, const char* sFunc, const char* sFile, int nLine, VALIST... vArgs) : std::runtime_error(cv::format((std::string("GLException in function '%s' from %s(%d) : \n")+sErrMsg).c_str(),sFunc,sFile,nLine,vArgs...)), m_eErrn(GL_NO_ERROR), m_acErrMsg(sErrMsg), m_acFuncName(sFunc), m_acFileName(sFile), m_nLineNumber(nLine) {};
    const GLenum m_eErrn;
    const char* const m_acErrMsg;
    const char* const m_acFuncName;
    const char* const m_acFileName;
    const int m_nLineNumber;
};

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
            return 0;
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
            return 0;
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
            return 0;
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
            return 0;
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
            return 0;
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
            return 0;
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
            return 0;
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
            return 0;
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
            return 0;
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
            return 0;
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
    for(auto oSrcCharIter = sSrc.begin(); oSrcCharIter!=sSrc.end(); ++oSrcCharIter) {
        if(*oSrcCharIter=='\n') {
            ++nCurrLine;
            ssRes << (bPrefixTab?"\n\t":"\n") << cv::format("%06d: ",nCurrLine);
        }
        else
            ssRes << *oSrcCharIter;

    }
    return ssRes.str();
}
