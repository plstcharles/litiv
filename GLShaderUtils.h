#pragma once

#include "GLDrawUtils.h"
#include <stdio.h>
#include <string>
#include <map>
#include <vector>
#include <cstdlib>
#include <cassert>

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
    bool setUniform1f(GLint nLoc, GLfloat fVal);
    bool setUniform1i(GLint nLoc, GLint nVal);
    bool setUniform1ui(GLint nLoc, GLuint nVal);
    bool setUniform4fv(GLint nLoc, const glm::vec4& afVals);

    std::string getUniformNameFromLoc(GLint nLoc);
    GLint getUniformLocFromName(const std::string& sName);

    bool isCompiled()  {return m_bIsCompiled;}
    bool isLinked()    {return m_bIsLinked;}
    bool isActive()    {return m_bIsActive;}
    bool isEmpty()     {return m_bIsEmpty;}
    GLuint getProgID() {return m_nProgID;}

    static const char* getDefaultVertexAttribVarName(GLVertex::eVertexAttribList eVar);

    static std::string getPassThroughVertexShaderSource(bool bPassNormals, bool bPassColors, bool bPassTexCoords);
    static std::string getPassThroughVertexShaderSource_ConstArray(GLuint nVertexCount, const GLVertex* aVertices, bool bPassNormals, bool bPassColors, bool bPassTexCoords);
    static std::string getPassThroughFragmentShaderSource_ConstColor(glm::vec4 vColor);
    static std::string getPassThroughFragmentShaderSource_PassedColor();
    static std::string getPassThroughFragmentShaderSource_TexSampler2D(GLuint nSamplerBinding, bool bUseIntegralFormat);
    static std::string getPassThroughFragmentShaderSource_SxSTexSampler2D(const std::vector<GLuint>& vnSamplerBindings, GLint nTextureLayer, bool bUseIntegralFormat); // @@@ to be tested
    static std::string getPassThroughFragmentShaderSource_SxSTexSampler2DArray(GLuint nSamplerBinding, GLint nTextureCount, bool bUseIntegralFormat);
    static std::string getPassThroughFragmentShaderSource_TexelFetchSampler2D(bool bUseTopLeftFragCoordOrigin, GLuint nSamplerBinding, float fTextureLevel, bool bUseIntegralFormat);
    static std::string getPassThroughFragmentShaderSource_ImgLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, GLuint nImageBinding, bool bUseIntegralFormat);
    static std::string getPassThroughFragmentShaderSource_SxSImgLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, const std::vector<GLuint> vnImageBindings, GLint nImageLayer, bool bUseIntegralFormat); // @@@ to be tested
    static std::string getPassThroughFragmentShaderSource_SxSImgArrayLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, GLuint nImageBinding, GLint nImageCount, bool bUseIntegralFormat);
    static std::string getPassThroughComputeShaderSource_ImgLoadCopy(const glm::uvec2& vWorkGroupSize, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding, bool bUseIntegralFormat);

private:
    static bool useShaderProgram(GLShader* pNewShader);
    GLShader& operator=(const GLShader&)=delete;
    GLShader(const GLShader&)=delete;
    std::map<GLuint,std::string> m_mShaderSources;
    std::map<std::string,GLint> m_mShaderUniformLocations;
    bool m_bIsCompiled;
    bool m_bIsLinked;
    bool m_bIsActive;
    bool m_bIsEmpty;
    const GLuint m_nProgID;
    static GLShader* s_pCurrActiveShader; // add mutex?
};

namespace NBodySimulationUtils {
    // @@@@ not necessary?
    std::string getVertexShaderSource();
    std::string getGeometryShaderSource();
    std::string getFragmentShaderSource();
    std::string getAccelerationComputeShaderSource();
    std::string getTiledAccelerationComputeShaderSource();
    std::string getIntegrateComputeShaderSource();
}; // namespace NBodySimulationUtils

namespace ComputeShaderUtils {
    std::string getComputeShaderSource_ParallelPrefixSum(int nMaxRowSize, bool bBinaryProc, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding);
    std::string getComputeShaderSource_ParallelPrefixSum_BlockMerge(int nColumns, int nMaxRowSize, int nRows, GLenum eInternalFormat, GLuint nImageBinding);
    std::string getComputeShaderSource_Transpose(int nBlockSize, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding);
}; // namespace ComputeShaderUtils

namespace GLSLFunctionUtils {
    // @@@@@@ transfer all to their respective utils files
    std::string getShaderFunctionSource_absdiff(bool bUseBuiltinDistance); // @@@@ test with/without
    std::string getShaderFunctionSource_L1dist();
    std::string getShaderFunctionSource_L2dist(bool bUseBuiltinDistance);
    std::string getShaderFunctionSource_hdist();
    std::string getShaderFunctionSource_getRandNeighbor3x3(const int nBorderSize, const cv::Size& oFrameSize);
    std::string getShaderFunctionSource_frand();
    std::string getShaderFunctionSource_urand();
    std::string getShaderFunctionSource_urand_tinymt32();
    struct TMT32GenParams {
        uint status[4];
        uint mat1;
        uint mat2;
        uint tmat;
        uint pad;
    };
    void initTinyMT32Generators(glm::uvec3 vGeneratorLayout, std::vector<TMT32GenParams>& voData);
}; // namespace DistanceFunctionUtils

