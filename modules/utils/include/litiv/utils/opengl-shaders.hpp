
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

#include "litiv/utils/opengl-draw.hpp"

// @@@@ rewrite all classes as part of lv::gl namespace?

class GLShader {
public:
    GLShader(bool bFixedFunct=false);
    ~GLShader();

    GLuint addSource(const std::string& sSource, GLenum eType);
    bool removeSource(GLuint id);

    bool compile();
    bool link(bool bDiscardSources=true);
    bool activate();
    void clear();

    bool setUniform1f(const std::string& sName, GLfloat fVal);
    bool setUniform1i(const std::string& sName, GLint nVal);
    bool setUniform1ui(const std::string& sName, GLuint nVal);
    bool setUniform4fv(const std::string& sName, const glm::vec4& afVals);
    bool setUniform4fm(const std::string& sName, const glm::mat4& mfVals);
    bool setUniform1f(GLint nLoc, GLfloat fVal);
    bool setUniform1i(GLint nLoc, GLint nVal);
    bool setUniform1ui(GLint nLoc, GLuint nVal);
    bool setUniform4fv(GLint nLoc, const glm::vec4& afVals);
    bool setUniform4fm(GLint nLoc, const glm::mat4& mfVals);

    std::string getUniformNameFromLoc(GLint nLoc);
    GLint getUniformLocFromName(const std::string& sName);

    bool isCompiled()  {return m_bIsCompiled;}
    bool isLinked()    {return m_bIsLinked;}
    bool isActive()    {return m_bIsActive;}
    bool isEmpty()     {return m_bIsEmpty;}
    GLuint getProgID() {return m_nProgID;}

    //static const char* getDefaultVertexAttribVarName(GLVertex::VertexAttribList eVar); @@@@ todo?

    static std::string getVertexShaderSource_PassThrough(bool bPassNormals, bool bPassColors, bool bPassTexCoords);
    static std::string getVertexShaderSource_PassThrough_ConstArray(GLuint nVertexCount, const GLVertex* aVertices, bool bPassNormals, bool bPassColors, bool bPassTexCoords);
    static std::string getFragmentShaderSource_PassThrough_ConstColor(glm::vec4 vColor);
    static std::string getFragmentShaderSource_PassThrough_PassedColor();
    static std::string getFragmentShaderSource_PassThrough_TexSampler2D(GLuint nSamplerBinding, bool bUseIntegralFormat);
    static std::string getFragmentShaderSource_PassThrough_SxSTexSampler2D(const std::vector<GLuint>& vnSamplerBindings, GLint nTextureLayer, bool bUseIntegralFormat); // @@@ to be tested
    static std::string getFragmentShaderSource_PassThrough_SxSTexSampler2DArray(GLuint nSamplerBinding, GLint nTextureCount, bool bUseIntegralFormat);
    static std::string getFragmentShaderSource_PassThrough_TexelFetchSampler2D(bool bUseTopLeftFragCoordOrigin, GLuint nSamplerBinding, float fTextureLevel, bool bUseIntegralFormat);
    static std::string getFragmentShaderSource_PassThrough_ImgLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, GLuint nImageBinding, bool bUseIntegralFormat);
    static std::string getFragmentShaderSource_PassThrough_SxSImgLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, const std::vector<GLuint> vnImageBindings, GLint nImageLayer, bool bUseIntegralFormat); // @@@ to be tested
    static std::string getFragmentShaderSource_PassThrough_SxSImgArrayLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, GLuint nImageBinding, GLint nImageCount, bool bUseIntegralFormat);
    static std::string getComputeShaderSource_PassThrough_ImgLoadCopy(const glm::uvec2& vWorkGroupSize, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding, bool bUseIntegralFormat);
    static std::string getComputeShaderSource_ParallelPrefixSum(size_t nMaxRowSize, bool bBinaryProc, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding);
    static std::string getComputeShaderSource_ParallelPrefixSum_BlockMerge(size_t nColumns, size_t nMaxRowSize, size_t nRows, GLenum eInternalFormat, GLuint nImageBinding);
    static std::string getComputeShaderSource_Transpose(size_t nBlockSize, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding);
    static std::string getComputeShaderFunctionSource_SharedDataPreLoad(size_t nChannels, const glm::uvec2& vWorkGroupSize, size_t nExternalBorderSize);
    static std::string getComputeShaderFunctionSource_BinaryMedianBlur(size_t nKernelSize, bool bUseSharedDataPreload, const glm::uvec2& vWorkGroupSize);
    static std::string getShaderFunctionSource_getRandNeighbor3x3(size_t nBorderSize,const cv::Size& oFrameSize);
    static std::string getShaderFunctionSource_frand();
    static std::string getShaderFunctionSource_urand();
    static std::string getShaderFunctionSource_urand_tinymt32();

private:
    static bool useShaderProgram(GLShader* pNewShader);
    GLShader& operator=(const GLShader&) = delete;
    GLShader(const GLShader&) = delete;
    std::map<GLuint,std::string> m_mShaderSources;
    std::map<std::string,GLint> m_mShaderUniformLocations;
    bool m_bIsCompiled;
    bool m_bIsLinked;
    bool m_bIsActive;
    bool m_bIsEmpty;
    const GLuint m_nProgID;
    static GLShader* s_pCurrActiveShader; // add mutex? manage via shared_ptr w/ enable_... interf?
};
