
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

#include "litiv/utils/opengl.hpp"

std::mutex lv::gl::Context::s_oGLFWErrorMessageMutex;
std::string lv::gl::Context::s_sLatestGLFWErrorMessage;
std::once_flag lv::gl::Context::s_oInitFlag;

lv::gl::Context::Context(const cv::Size& oWinSize,
                         const std::string& sWinName,
                         bool bHide,
                         size_t nGLVerMajor,
                         size_t nGLVerMinor) :
        m_nGLVerMajor(nGLVerMajor),
        m_nGLVerMinor(nGLVerMinor) {
#if HAVE_GLFW
    std::call_once(s_oInitFlag,[](){
        glfwSetErrorCallback(onGLFWErrorCallback);
        if(glfwInit()==GL_FALSE)
            lvError("Failed to init GLFW");
        std::atexit(glfwTerminate);
    });
    if(nGLVerMajor>3 || (nGLVerMajor==3 && nGLVerMinor>=2))
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,int(nGLVerMajor));
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,int(nGLVerMinor));
    glfwWindowHint(GLFW_RESIZABLE,GL_FALSE);
    if(bHide)
        glfwWindowHint(GLFW_VISIBLE,GL_FALSE);
    m_pWindowHandle = std::unique_ptr<GLFWwindow,glfwWindowDeleter>(glfwCreateWindow(oWinSize.width,oWinSize.height,sWinName.c_str(),nullptr,nullptr),glfwWindowDeleter());
    if(!m_pWindowHandle.get())
        lvError_("Failed to create [%d,%d] window via GLFW for core GL profile v%d.%d",oWinSize.width,oWinSize.height,nGLVerMajor,nGLVerMinor);
    glfwMakeContextCurrent(m_pWindowHandle.get());
#elif HAVE_FREEGLUT
    std::call_once(s_oInitFlag,[](){
        int argc = 0;
        glutInit(&argc,NULL);
    });
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(oWinSize.width,oWinSize.height);
    glutInitWindowPosition(0,0);
    m_oWindowHandle = std::unique_ptr<glutHandle,glutWindowDeleter>(glutHandle(glutCreateWindow(sWinName.c_str())),glutWindowDeleter());
    if(!(m_oWindowHandle.get().m_nHandle))
        lvError("Failed to create window via glut");
    glutSetWindow(m_oWindowHandle.get().m_nHandle);
    if(bHide)
        glutHideWindow();
#endif //HAVE_FREEGLUT
    initGLEW(m_nGLVerMajor,m_nGLVerMinor);
}

void lv::gl::Context::setAsActive() {
#if HAVE_GLFW
    glfwMakeContextCurrent(m_pWindowHandle.get());
#elif HAVE_FREEGLUT
    glutSetWindow(m_oWindowHandle.get().m_nHandle);
#endif //HAVE_FREEGLUT
}

void lv::gl::Context::setWindowVisibility(bool bVal) {
#if HAVE_GLFW
    if(bVal)
        glfwShowWindow(m_pWindowHandle.get());
    else
        glfwHideWindow(m_pWindowHandle.get());
#elif HAVE_FREEGLUT
    glutSetWindow(m_oWindowHandle.get().m_nHandle);
        if(bVal)
            glutShowWindow();
        else
            glutHideWindow();
#endif //HAVE_FREEGLUT
}

void lv::gl::Context::setWindowSize(const cv::Size& oSize, bool bUpdateViewport) {
#if HAVE_GLFW
    glfwSetWindowSize(m_pWindowHandle.get(),oSize.width,oSize.height);
#elif HAVE_FREEGLUT
    glutSetWindow(m_oWindowHandle.get().m_nHandle);
        glutReshapeWindow(oSize.width,oSize.height);
#endif //HAVE_FREEGLUT
    if(bUpdateViewport)
        glViewport(0,0,oSize.width,oSize.height);
}

std::string lv::gl::Context::getLatestErrorMessage() { // also clears the latest error message
#if HAVE_GLFW
    std::lock_guard<std::mutex> oLock(s_oGLFWErrorMessageMutex);
    std::string sErrMsg;
    std::swap(sErrMsg,s_sLatestGLFWErrorMessage);
    return sErrMsg;
#elif HAVE_FREEGLUT
    return std::string(); // glut gives no custom error messages...
#endif //HAVE_FREEGLUT
}

bool lv::gl::Context::pollEventsAndCheckIfShouldClose() {
#if HAVE_GLFW
    glfwPollEvents();
    return glfwWindowShouldClose(m_pWindowHandle.get())!=0;
#elif HAVE_FREEGLUT
    return glutGetWindow()!=0; // not ideal, but there is nothing else...
#endif //HAVE_FREEGLUT
}

bool lv::gl::Context::getKeyPressed(char nKeyID) {
#if HAVE_GLFW
    return glfwGetKey(m_pWindowHandle.get(),nKeyID)==GLFW_PRESS; // will not capture special keys (need custom define)
#elif HAVE_FREEGLUT
    return false; // seriously, ditch glut
#endif //HAVE_FREEGLUT
}

void lv::gl::Context::swapBuffers(int nClearFlags) {
#if HAVE_GLFW
    glfwSwapBuffers(m_pWindowHandle.get());
#elif HAVE_FREEGLUT
    glutSetWindow(m_oWindowHandle.get().m_nHandle);
        glutSwapBuffers();
#endif //HAVE_FREEGLUT
    if(nClearFlags!=0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
}

void lv::gl::Context::initGLEW(size_t nGLVerMajor, size_t nGLVerMinor) {
    glErrorCheck;
    glewExperimental = GLEW_EXPERIMENTAL?GL_TRUE:GL_FALSE;
    const GLenum glewerrn = glewInit();
    if(glewerrn!=GLEW_OK)
        lvError_("Failed to init GLEW [code=%d, msg=%s]",glewerrn,glewGetErrorString(glewerrn));
    const GLenum errn = glGetError();
    // see glew init GL_INVALID_ENUM bug discussion at https://www.opengl.org/wiki/OpenGL_Loading_Library
    if(errn!=GL_NO_ERROR && errn!=GL_INVALID_ENUM)
        lvError_("Unexpected GLEW init error [code=%d, msg=%s]",errn,gluErrorString(errn));
    const std::string sGLEWVersionString = std::string("GL_VERSION_")+std::to_string(nGLVerMajor)+"_"+std::to_string(nGLVerMinor);
    if(!glewIsSupported(sGLEWVersionString.c_str()))
        lvError_("Bad GL core/ext version detected (target is %s)",sGLEWVersionString.c_str());
}

cv::Mat lv::gl::deepCopyImage(GLsizei nWidth,GLsizei nHeight,GLvoid* pData,GLenum eDataFormat,GLenum eDataType) {
    lvAssert(nWidth>0 && nHeight>0 && pData);
    const int nDepth = getMatDepthFromDataType(eDataType);
    const int nChannels = getChannelsFromDataFormat(eDataFormat);
    return cv::Mat(nHeight,nWidth,CV_MAKETYPE(nDepth,nChannels),pData,nWidth*nChannels*getByteSizeFromMatDepth(nDepth)).clone();
}

std::vector<cv::Mat> lv::gl::deepCopyImages(const std::vector<cv::Mat>& voInputMats) {
    lvAssert(!voInputMats.empty() && !voInputMats[0].empty());
    std::vector<cv::Mat> voOutputMats(voInputMats.size());
    for(size_t nMatIter=0; nMatIter<voInputMats.size(); ++nMatIter)
        voInputMats[nMatIter].copyTo(voOutputMats[nMatIter]);
    return voOutputMats;
}

std::vector<cv::Mat> lv::gl::deepCopyImages(GLsizei nTextureCount,GLsizei nWidth,GLsizei nHeight,GLvoid* pData,GLenum eDataFormat,GLenum eDataType) {
    lvAssert(nTextureCount>0 && nWidth>0 && nHeight>0 && pData);
    std::vector<cv::Mat> voOutputMats(nTextureCount);
    const int nDepth = getMatDepthFromDataType(eDataType);
    const int nChannels = getChannelsFromDataFormat(eDataFormat);
    const int nImageSize = nHeight*nWidth*nChannels*getByteSizeFromMatDepth(nDepth);
    for(int nMatIter=0; nMatIter<nTextureCount; ++nMatIter)
        voOutputMats[nMatIter] = deepCopyImage(nWidth,nHeight,((char*)pData)+(nMatIter*nImageSize),eDataFormat,eDataType);
    return voOutputMats;
}

std::string lv::gl::addLineNumbersToString(const std::string& sSrc, bool bPrefixTab) {
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

void lv::gl::TMT32GenParams::initTinyMT32Generators(glm::uvec3 vGeneratorLayout,std::aligned_vector<lv::gl::TMT32GenParams,32>& voData) {
    static_assert(sizeof(TMT32GenParams)==sizeof(uint)*8,"Hmmm...?");
    lvAssert(vGeneratorLayout.x>0 && vGeneratorLayout.y>0 && vGeneratorLayout.z>0);
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
