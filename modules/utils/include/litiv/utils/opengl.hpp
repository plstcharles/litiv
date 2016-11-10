
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

#include <GL/glew.h>
#include <GL/glu.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#elif defined(_MSC_VER)
#pragma warning(push,0)
#endif //defined(_MSC_VER)
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif //defined(_MSC_VER)
#include <opencv2/opencv.hpp>
#include "litiv/utils/cxx.hpp"
#include "litiv/utils/platform.hpp"

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

namespace lv {

    namespace gl {

        class Context {
        public:
            Context(const cv::Size& oWinSize,
                    const std::string& sWinName,
                    bool bHide=true,
                    size_t nGLVerMajor=TARGET_GL_VER_MAJOR,
                    size_t nGLVerMinor=TARGET_GL_VER_MINOR);
            void setAsActive();
            void setWindowVisibility(bool bVal);
            void setWindowSize(const cv::Size& oSize, bool bUpdateViewport=true);
            static std::string getLatestErrorMessage();
            bool pollEventsAndCheckIfShouldClose();
            bool getKeyPressed(char nKeyID);
            void swapBuffers(int nClearFlags=0);
            static void initGLEW(size_t nGLVerMajor, size_t nGLVerMinor);

        private:
#if HAVE_GLFW
            std::unique_ptr<GLFWwindow,glfwWindowDeleter> m_pWindowHandle;
            static std::mutex s_oGLFWErrorMessageMutex;
            static std::string s_sLatestGLFWErrorMessage;
            static void onGLFWErrorCallback(int nCode, const char* acMessage) {
                std::stringstream ssStr;
                ssStr << "code: " << nCode << ", message: " << acMessage;
                std::lock_guard<std::mutex> oLock(s_oGLFWErrorMessageMutex);
                s_sLatestGLFWErrorMessage = ssStr.str();
            }
#elif HAVE_FREEGLUT
            std::unique_ptr<glutHandle,glutWindowDeleter> m_oWindowHandle;
#endif //HAVE_FREEGLUT
            const size_t m_nGLVerMajor;
            const size_t m_nGLVerMinor;
            static std::once_flag s_oInitFlag;
            Context& operator=(const Context&) = delete;
            Context(const Context&) = delete;
        };

        inline bool isInternalFormatSupported(GLenum eInternalFormat) {
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

        inline bool isMatTypeSupported(int nType) {
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

        inline GLenum getInternalFormatFromMatType(int nTextureType, bool bUseIntegralFormat=true) {
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
                    return bUseIntegralFormat?GL_R32UI:lvError("opencv mat input format type cannot be matched to a predefined setup");
                case CV_32SC3:
                    return bUseIntegralFormat?GL_RGB32UI:lvError("opencv mat input format type cannot be matched to a predefined setup");
                case CV_32SC4:
                    return bUseIntegralFormat?GL_RGBA32UI:lvError("opencv mat input format type cannot be matched to a predefined setup");
                default:
                    lvError("opencv mat input format type did not match any predefined setup");
            }
        }

        inline GLenum getNormalizedIntegralFormatFromInternalFormat(GLenum eInternalFormat) {
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
                    lvError("input internal format did not match any predefined normalized format");
            }
        }

        inline GLenum getIntegralFormatFromInternalFormat(GLenum eInternalFormat) {
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
                    lvError("input normalized format did not match any predefined integral format");
                    return 0;
            }
        }

        inline const char* getGLSLFormatNameFromInternalFormat(GLenum eInternalFormat) {
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
                    lvError("input internal format did not match any predefined glsl layout");
            }
        }

        inline bool isInternalFormatIntegral(GLenum eInternalFormat) {
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
                    lvError("input internal format did not match any predefined integer format setup");
            }
        }

        inline int getMatTypeFromInternalFormat(GLenum eInternalFormat) {
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
                    lvError("input internal format did not match any predefined opencv mat setup");
                    return 0;
            }
        }

        inline GLenum getDataFormatFromChannels(int nTextureChannels, bool bUseIntegralFormat=true) {
            switch(nTextureChannels) {
                case 1:
                    return bUseIntegralFormat?GL_RED_INTEGER:GL_RED;
                case 3:
                    return bUseIntegralFormat?GL_BGR_INTEGER:GL_BGR;
                case 4:
                    return bUseIntegralFormat?GL_BGRA_INTEGER:GL_BGRA;
                default:
                    lvError("opencv mat input format type did not match any predefined setup");
            }
        }

        inline int getChannelsFromDataFormat(GLenum eDataFormat) {
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
                    lvError("input data format did not match any predefined opencv mat setup");
            }
        }

        inline GLenum getDataTypeFromMatDepth(int nTextureDepth, int nTextureChannels) {
            switch(nTextureDepth) {
                case CV_8U:
                    return nTextureChannels==4?GL_UNSIGNED_INT_8_8_8_8_REV:GL_UNSIGNED_BYTE;
                case CV_32F:
                    return GL_FLOAT;
                case CV_32S:
                    return GL_UNSIGNED_INT;
                default:
                    lvError("opencv mat input format type did not match any predefined setup");
            }
        }

        inline int getMatDepthFromDataType(GLenum eDataType) {
            switch(eDataType) {
                case GL_UNSIGNED_BYTE:
                case GL_UNSIGNED_INT_8_8_8_8_REV:
                    return CV_8U;
                case GL_FLOAT:
                    return CV_32F;
                case GL_UNSIGNED_INT:
                    return CV_32S;
                default:
                    lvError("input data type did not match any predefined opencv mat setup");
            }
        }

        inline int getByteSizeFromMatDepth(int nDepth) {
            switch(nDepth) {
                case CV_8U:
                    return 1;
                case CV_32F:
                case CV_32S:
                    return 4;
                default:
                    lvError("input depth did not match any predefined byte size");
            }
        }

        inline int getChannelsFromMatType(int nType) {
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
                    lvError("input type did not match any predefined channel size");
            }
        }

        inline int getChannelsFromInternalFormat(GLenum eInternalFormat) {
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
                    lvError("input internal format did not match any predefined channel size");
                    return 0;
            }
        }

        template<uint N>
        inline std::enable_if_t<(N==1),int> getIntegerVal(GLenum eParamName) {
            int nVal;
            glGetIntegerv(eParamName,&nVal);
            glErrorCheck;
            return nVal;
        }

        template<uint N>
        inline std::enable_if_t<(N>1),std::array<int,N>> getIntegerVal(GLenum eParamName) {
            std::array<int,N> anVal;
            for(uint n=0; n<N; ++n)
                glGetIntegeri_v(eParamName,n,&anVal[n]);
            glErrorCheck;
            return anVal;
        }

        cv::Mat deepCopyImage(GLsizei nWidth,GLsizei nHeight,GLvoid* pData,GLenum eDataFormat,GLenum eDataType);
        std::vector<cv::Mat> deepCopyImages(const std::vector<cv::Mat>& voInputMats);
        std::vector<cv::Mat> deepCopyImages(GLsizei nTextureCount,GLsizei nWidth,GLsizei nHeight,GLvoid* pData,GLenum eDataFormat,GLenum eDataType);

        std::string addLineNumbersToString(const std::string& sSrc, bool bPrefixTab);

        struct alignas(32) TMT32GenParams {
            uint status[4];
            uint mat1;
            uint mat2;
            uint tmat;
            uint pad;
            static void initTinyMT32Generators(glm::uvec3 vGeneratorLayout,std::aligned_vector<TMT32GenParams,32>& voData);
        };

    } // namespace gl

} // namespace lv
