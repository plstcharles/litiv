#include "GLShaderUtils.h"

GLShader* GLShader::s_pCurrActiveShader = nullptr;

bool GLShader::useShaderProgram(GLShader* pNewShader) {
    if(pNewShader) {
        if(pNewShader->m_bIsActive && s_pCurrActiveShader==pNewShader)
            return true;
        if(!pNewShader->m_bIsLinked)
            pNewShader->link();
    }
    if(!pNewShader || (pNewShader->m_bIsEmpty && !pNewShader->m_nProgID) || pNewShader->m_bIsLinked) {
        if(pNewShader) {
            glUseProgram(pNewShader->m_nProgID);
            glErrorCheck;
            if(s_pCurrActiveShader)
                s_pCurrActiveShader->m_bIsActive = false;
            pNewShader->m_bIsActive = true;
            s_pCurrActiveShader = pNewShader;
        }
        else {
            glUseProgram(0);
            glErrorCheck;
            if(s_pCurrActiveShader)
                s_pCurrActiveShader->m_bIsActive = false;
            s_pCurrActiveShader = nullptr;
        }
        return true;
    }
    return false;
}

GLShader::GLShader(bool bFixedFunct)
  :  m_bIsCompiled(false)
    ,m_bIsLinked(false)
    ,m_bIsActive(false)
    ,m_bIsEmpty(true)
    ,m_nProgID(bFixedFunct?0:glCreateProgram()) {
    if(bFixedFunct&&!m_nProgID)
        glError("glCreateProgram failed");
    glErrorCheck;
}

GLShader::~GLShader() {
    if(m_nProgID) {
        while(!m_mShaderSources.empty())
            removeSource(m_mShaderSources.begin()->first);
        glDeleteProgram(m_nProgID);
    }
}

GLuint GLShader::addSource(const std::string& sSource, GLenum eType) {
    if(!m_nProgID)
        glError("attempted to add source to default shader pipeline program");
    GLuint shaderID = glCreateShader(eType);
    glErrorCheck;
    try {
        const char* acSourcePtr = sSource.c_str();
        glShaderSource(shaderID,1,&acSourcePtr,nullptr);
        glErrorCheck;
    } catch(GLException& e) {
        glDeleteShader(shaderID);
        throw;
    }
    m_mShaderSources.insert(std::make_pair(shaderID,sSource));
    m_bIsCompiled = m_bIsLinked = false;
    m_bIsEmpty = false;
    return shaderID;
}

bool GLShader::removeSource(GLuint id) {
    auto oSrcIter = m_mShaderSources.find(id);
    if(oSrcIter!=m_mShaderSources.end()) {
        if(m_bIsActive)
            useShaderProgram(nullptr);
        glDetachShader(m_nProgID,oSrcIter->first);
        glDeleteShader(oSrcIter->first);
        m_mShaderSources.erase(oSrcIter);
        m_bIsCompiled = m_bIsLinked = false;
        m_bIsEmpty = m_mShaderSources.empty();
        return true;
    }
    return false;
}

bool GLShader::compile() {
    GLboolean bCompilerSupport;
    glGetBooleanv(GL_SHADER_COMPILER,&bCompilerSupport);
    if(bCompilerSupport==GL_FALSE)
        glError("shader compiler not supported");
    GLint nCompiled = GL_TRUE;
    for(auto oSrcIter=m_mShaderSources.begin(); oSrcIter!=m_mShaderSources.end(); ++oSrcIter) {
        glGetShaderiv(oSrcIter->first,GL_COMPILE_STATUS,&nCompiled);
        glErrorCheck;
        if(nCompiled==GL_TRUE)
            continue;
        glCompileShader(oSrcIter->first);
        glErrorCheck;
        glGetShaderiv(oSrcIter->first,GL_COMPILE_STATUS,&nCompiled);
        if(nCompiled==GL_FALSE) {
            GLint nLogSize;
            glGetShaderiv(oSrcIter->first, GL_INFO_LOG_LENGTH, &nLogSize);
            std::vector<char> vcLog(nLogSize);
            glGetShaderInfoLog(oSrcIter->first, nLogSize, &nLogSize, &vcLog[0]);
            std::cerr << "GLShader::compile(); error in shaderID#" << oSrcIter->first << " :\n\n" << addLineNumbersToString(oSrcIter->second,true) << "\n\n" << &vcLog[0] << std::endl;
            return false;
        }
    }
    return (m_bIsCompiled=!m_mShaderSources.empty());
}

bool GLShader::link(bool bDiscardSources) {
    if(!compile() && !m_bIsEmpty)
        return false;
    if(m_bIsLinked && m_bIsEmpty)
        return true;
    for(auto oSrcIter=m_mShaderSources.begin(); oSrcIter!=m_mShaderSources.end(); ++oSrcIter) {
        glAttachShader(m_nProgID,oSrcIter->first);
        glErrorCheck;
    }
    glLinkProgram(m_nProgID);
    glErrorCheck;
    m_mShaderUniformLocations.clear();
    GLint nLinked;
    glGetProgramiv(m_nProgID, GL_LINK_STATUS, &nLinked);
    if(nLinked==GL_FALSE) {
        GLint nLogSize;
        glGetProgramiv(m_nProgID, GL_INFO_LOG_LENGTH, &nLogSize);
        std::vector<char> vcLog(nLogSize);
        glGetProgramInfoLog(m_nProgID, nLogSize, &nLogSize, &vcLog[0]);
        std::cerr << "GLShader::link(); error in program #" << m_nProgID << " :\n" << &vcLog[0] << std::endl;
        return false;
    }
    else if(bDiscardSources) {
        while(!m_mShaderSources.empty())
            removeSource(m_mShaderSources.begin()->first);
    }
    return (m_bIsLinked=true);
}

bool GLShader::activate() {
    return useShaderProgram(this);
}

void GLShader::clear() {
    while(!m_mShaderSources.empty())
        removeSource(m_mShaderSources.begin()->first);
}

bool GLShader::setUniform1f(const std::string& sName, GLfloat fVal) {
    GLint nLoc = getUniformLocFromName(sName);
    if(nLoc==-1)
        return false;
    glProgramUniform1f(m_nProgID,nLoc,fVal);
    glErrorCheck;
    return true;
}

bool GLShader::setUniform1i(const std::string& sName, GLint nVal) {
    GLint nLoc = getUniformLocFromName(sName);
    if(nLoc==-1)
        return false;
    glProgramUniform1i(m_nProgID,nLoc,nVal);
    glErrorCheck;
    return true;
}

bool GLShader::setUniform1ui(const std::string& sName, GLuint nVal) {
    GLint nLoc = getUniformLocFromName(sName);
    if(nLoc==-1)
        return false;
    glProgramUniform1ui(m_nProgID,nLoc,nVal);
    glErrorCheck;
    return true;
}

bool GLShader::setUniform4fv(const std::string& sName, const glm::vec4& afVals) {
    GLint nLoc = getUniformLocFromName(sName);
    if(nLoc==-1)
        return false;
    glProgramUniform4fv(m_nProgID,nLoc,1,&afVals[0]);
    glErrorCheck;
    return true;
}

bool GLShader::setUniform1f(GLint nLoc, GLfloat fVal) {
    if(nLoc<0)
        return false;
    glProgramUniform1f(m_nProgID,nLoc,fVal);
    glErrorCheck;
    return true;
}

bool GLShader::setUniform1i(GLint nLoc, GLint nVal) {
    if(nLoc<0)
        return false;
    glProgramUniform1i(m_nProgID,nLoc,nVal);
    glErrorCheck;
    return true;
}

bool GLShader::setUniform1ui(GLint nLoc, GLuint nVal) {
    if(nLoc<0)
        return false;
    glProgramUniform1ui(m_nProgID,nLoc,nVal);
    glErrorCheck;
    return true;
}

bool GLShader::setUniform4fv(GLint nLoc, const glm::vec4& afVals) {
    if(nLoc<0)
        return false;
    glProgramUniform4fv(m_nProgID,nLoc,1,&afVals[0]);
    glErrorCheck;
    return true;
}

std::string GLShader::getUniformNameFromLoc(GLint nLoc) {
    const GLsizei nNameMaxSize = 256;
    GLchar acName[nNameMaxSize];
    GLsizei nNameSize, nSize;
    GLenum eType;
    glGetActiveUniform(m_nProgID,(GLuint)nLoc,nNameMaxSize,&nNameSize,&nSize,&eType,acName);
    glErrorCheck;
    return std::string(acName);
}

GLint GLShader::getUniformLocFromName(const std::string& sName) {
    if(!m_bIsLinked)
        return -1;
    auto oFindRes = m_mShaderUniformLocations.find(sName);
    if(oFindRes==m_mShaderUniformLocations.end()) {
        GLint nLoc = glGetUniformLocation(m_nProgID,sName.c_str());
        m_mShaderUniformLocations.insert(std::make_pair(sName,nLoc));
        return nLoc;
    }
    return oFindRes->second;
}

std::string GLShader::getPassThroughVertexShaderSource(bool bPassNormals, bool bPassColors, bool bPassTexCoords) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=" << GLVertex::eVertexAttrib_PositionIdx << ") in vec4 in_position;\n";
    if(bPassNormals) ssSrc <<
             "layout(location=" << GLVertex::eVertexAttrib_NormalIdx << ") in vec4 in_normal;\n"
             "layout(location=" << GLVertex::eVertexAttrib_NormalIdx << ") out vec4 out_normal;\n";
    if(bPassColors) ssSrc <<
             "layout(location=" << GLVertex::eVertexAttrib_ColorIdx << ") in vec4 in_color;\n"
             "layout(location=" << GLVertex::eVertexAttrib_ColorIdx << ") out vec4 out_color;\n";
    if(bPassTexCoords) ssSrc <<
             "layout(location=" << GLVertex::eVertexAttrib_TexCoordIdx << ") in vec4 in_texCoord;\n"
             "layout(location=" << GLVertex::eVertexAttrib_TexCoordIdx << ") out vec4 out_texCoord;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    gl_Position = in_position;\n";
    if(bPassNormals) ssSrc <<
             "    out_normal = in_normal;\n";
    if(bPassColors) ssSrc <<
             "    out_color = in_color;\n";
    if(bPassTexCoords) ssSrc <<
             "    out_texCoord = in_texCoord;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughVertexShaderSource_ConstArray(GLuint nVertexCount, const GLVertex* aVertices, bool bPassNormals, bool bPassColors, bool bPassTexCoords) {
    glAssert(nVertexCount>0 && aVertices);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "const vec4 positions[" << nVertexCount << "] = vec4[" << nVertexCount << "](\n";
    for(size_t nVertexIter=0; nVertexIter<nVertexCount; ++nVertexIter) ssSrc <<
                 "\tvec4(" << aVertices[nVertexIter].vPosition[0] << "," <<
                              aVertices[nVertexIter].vPosition[1] << "," <<
                              aVertices[nVertexIter].vPosition[2] << "," <<
                              aVertices[nVertexIter].vPosition[3] << (nVertexIter==nVertexCount-1?")\n":"),\n");
    ssSrc << ");\n";
    if(bPassNormals) { ssSrc <<
             "const vec4 normals[" << nVertexCount << "] = vec4[" << nVertexCount << "](\n";
        for(size_t nVertexIter=0; nVertexIter<nVertexCount; ++nVertexIter) ssSrc <<
                 "\tvec4(" << aVertices[nVertexIter].vNormal[0] << "," <<
                              aVertices[nVertexIter].vNormal[1] << "," <<
                              aVertices[nVertexIter].vNormal[2] << "," <<
                              aVertices[nVertexIter].vNormal[3] << (nVertexIter==nVertexCount-1?")\n":"),\n");
        ssSrc <<
             ");\n"
             "layout(location=" << GLVertex::eVertexAttrib_NormalIdx << ") out vec4 out_normal;\n";
    }
    if(bPassColors) { ssSrc <<
             "const vec4 colors[" << nVertexCount << "] = vec4[" << nVertexCount << "](\n";
        for(size_t nVertexIter=0; nVertexIter<nVertexCount; ++nVertexIter) ssSrc <<
                 "\tvec4(" << aVertices[nVertexIter].vColor[0] << "," <<
                              aVertices[nVertexIter].vColor[1] << "," <<
                              aVertices[nVertexIter].vColor[2] << "," <<
                              aVertices[nVertexIter].vColor[3] << (nVertexIter==nVertexCount-1?")\n":"),\n");
        ssSrc <<
             ");\n"
             "layout(location=" << GLVertex::eVertexAttrib_ColorIdx << ") out vec4 out_color;\n";
    }
    if(bPassTexCoords) { ssSrc <<
             "const vec4 texCoords[" << nVertexCount << "] = vec4[" << nVertexCount << "](\n";
        for(size_t nVertexIter=0; nVertexIter<nVertexCount; ++nVertexIter) ssSrc <<
                 "\tvec4(" << aVertices[nVertexIter].vTexCoord[0] << "," <<
                              aVertices[nVertexIter].vTexCoord[1] << "," <<
                              aVertices[nVertexIter].vTexCoord[2] << "," <<
                              aVertices[nVertexIter].vTexCoord[3] << (nVertexIter==nVertexCount-1?")\n":"),\n");
        ssSrc <<
             ");\n"
             "layout(location=" << GLVertex::eVertexAttrib_TexCoordIdx << ") out vec4 out_texCoord;\n";
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    gl_Position = positions[gl_VertexID];\n";
    if(bPassNormals) ssSrc <<
             "    out_normal = normals[gl_VertexID];\n";
    if(bPassColors) ssSrc <<
             "    out_color = colors[gl_VertexID];\n";
    if(bPassTexCoords) ssSrc <<
             "    out_texCoord = texCoords[gl_VertexID];\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_ConstColor(glm::vec4 vColor) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    out_color = vec4(" << vColor[0] << "," <<
                                        vColor[1] << "," <<
                                        vColor[2] << "," <<
                                        vColor[3] << ");\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_PassedColor() {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n"
             "layout(location=" << GLVertex::eVertexAttrib_ColorIdx << ") in vec4 in_color;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    out_color = in_color\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_TexSampler2D(GLuint nSamplerBinding, bool bUseIntegralFormat) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n"
             "layout(location=" << GLVertex::eVertexAttrib_TexCoordIdx << ") in vec4 texCoord2D;\n"
             "layout(binding=" << nSamplerBinding << ") uniform" << (bUseIntegralFormat?" u":" ") << "sampler2D texSampler2D;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    out_color = texture(texSampler2D,texCoord2D.xy);\n";
    if(bUseIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_SxSTexSampler2D(const std::vector<GLuint>& vnSamplerBindings, GLint nTextureLayer, bool bUseIntegralFormat) {
    glAssert(vnSamplerBindings.size()>1 && nTextureLayer>=0);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n"
             "layout(location=" << GLVertex::eVertexAttrib_TexCoordIdx << ") in vec4 texCoord2D;\n";
    for(size_t nSamplerBindingIter=0; nSamplerBindingIter<vnSamplerBindings.size(); ++nSamplerBindingIter) ssSrc <<
             "layout(binding=" << vnSamplerBindings[nSamplerBindingIter] << ") uniform" << (bUseIntegralFormat?" u":" ") << "sampler2DArray texSampler2DArray" << nSamplerBindingIter << ";\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    float texCoord2DArrayIdx = 1;\n"
             "    vec3 texCoord3D = vec3(modf(texCoord2D.x*texSampler2DArrayLayers,texCoord2DArrayIdx),texCoord2D.y,texCoord2DArrayIdx);\n";
    for(size_t nSamplerBindingIter=0; nSamplerBindingIter<vnSamplerBindings.size(); ++nSamplerBindingIter) ssSrc <<
             "    vec4 out_color" << nSamplerBindingIter << " = texture(texSampler2DArray" << nSamplerBindingIter << ",vec3(texCoord3D.xy," << nTextureLayer << "));\n";
    ssSrc << "    if(texCoord3D.z==0) { \n"
             "        out_color = out_color0;\n"
             "    }\n";
    for(size_t nSamplerBindingIter=1; nSamplerBindingIter<vnSamplerBindings.size(); ++nSamplerBindingIter) ssSrc <<
             "    else if(texCoord3D.z==" << nSamplerBindingIter << ") {\n"
             "        out_color = out_color" << nSamplerBindingIter << ";\n"
             "    }\n";
    if(bUseIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_SxSTexSampler2DArray(GLuint nSamplerBinding, GLint nTextureCount, bool bUseIntegralFormat) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n"
             "layout(location=" << GLVertex::eVertexAttrib_TexCoordIdx << ") in vec4 texCoord2D;\n"
             "layout(binding=" << nSamplerBinding << ") uniform" << (bUseIntegralFormat?" u":" ") << "sampler2DArray texSampler2DArray;\n"
             "const int texSampler2DArrayLayers = " << nTextureCount << ";\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    float texCoord2DArrayIdx = 1;\n"
             "    vec3 texCoord3D = vec3(modf(texCoord2D.x*texSampler2DArrayLayers,texCoord2DArrayIdx),texCoord2D.y,texCoord2DArrayIdx);\n"
             "    out_color = texture(texSampler2DArray,texCoord3D);\n";
    if(bUseIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}
std::string GLShader::getPassThroughFragmentShaderSource_TexelFetchSampler2D(bool bUseTopLeftFragCoordOrigin, GLuint nSamplerBinding, float fTextureLevel, bool bUseIntegralFormat) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n";
    if(bUseTopLeftFragCoordOrigin) ssSrc <<
             "layout(origin_upper_left) in vec4 gl_FragCoord;\n";
    ssSrc << "layout(binding=" << nSamplerBinding << ") uniform" << (bUseIntegralFormat?" u":" ") << "sampler2D texSampler2D;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    ivec2 imgCoord = ivec2(gl_FragCoord.xy);\n"
             "    out_color = texelFetch(texSampler2D,imgCoord," << fTextureLevel << ");\n";
    if(bUseIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_ImgLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, GLuint nImageBinding, bool bUseIntegralFormat) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n";
    if(bUseTopLeftFragCoordOrigin) ssSrc <<
             "layout(origin_upper_left) in vec4 gl_FragCoord;\n";
    const bool bInternalFormatIsIntegral = isInternalFormatIntegral(eInternalFormat);
    if(!bUseIntegralFormat && bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nImageBinding << ", " << getGLSLFormatNameFromInternalFormat(getNormalizedIntegralFormatFromInternalFormat(eInternalFormat)) << ") readonly uniform image2D img;\n";
    else if(bUseIntegralFormat && bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform uimage2D img;\n";
    else if(!bUseIntegralFormat && !bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform image2D img;\n";
    else
        glError("bad internal format & useintegral setup");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    ivec2 imgCoord = ivec2(gl_FragCoord.xy);\n"
             "    out_color = imageLoad(img,imgCoord);\n";
    if(bUseIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_SxSImgLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, const std::vector<GLuint> vnImageBindings, GLint nImageLayer, bool bUseIntegralFormat) {
    glAssert(vnImageBindings.size()>1 && nImageLayer>=0);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n";
    if(bUseTopLeftFragCoordOrigin) ssSrc <<
             "layout(origin_upper_left) in vec4 gl_FragCoord;\n";
    const bool bInternalFormatIsIntegral = isInternalFormatIntegral(eInternalFormat);
    if(!bUseIntegralFormat && bInternalFormatIsIntegral)
        for(size_t nImageBindingIter=0; nImageBindingIter<vnImageBindings.size(); ++nImageBindingIter) ssSrc <<
             "layout(binding=" << nImageBindingIter << ", " << getGLSLFormatNameFromInternalFormat(getNormalizedIntegralFormatFromInternalFormat(eInternalFormat)) << ") readonly uniform image2D" << (nImageLayer>0?"Array imgArray":" img") << nImageBindingIter << ";\n";
    else if(bUseIntegralFormat && bInternalFormatIsIntegral)
        for(size_t nImageBindingIter=0; nImageBindingIter<vnImageBindings.size(); ++nImageBindingIter) ssSrc <<
             "layout(binding=" << nImageBindingIter << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform uimage2D" << (nImageLayer>0?"Array imgArray":" img") << nImageBindingIter << ";\n";
    else if(!bUseIntegralFormat && !bInternalFormatIsIntegral)
        for(size_t nImageBindingIter=0; nImageBindingIter<vnImageBindings.size(); ++nImageBindingIter) ssSrc <<
             "layout(binding=" << nImageBindingIter << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform image2D" << (nImageLayer>0?"Array imgArray":" img") << nImageBindingIter << ";\n";
    else
        glError("bad internal format & useintegral setup");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n";
    if(nImageLayer>0) { ssSrc <<
             "    ivec3 imgSize = imageSize(imgArray1);\n"
             "    ivec3 imgCoord = ivec3(mod(int(gl_FragCoord.x),imgSize.x),gl_FragCoord.y,int(gl_FragCoord.x)/imgSize.x);\n"
             "    if(imgCoord.z==0) { \n"
             "        out_color = imageLoad(imgArray0,ivec3(imgCoord.xy," << nImageLayer << "));\n"
             "    }\n";
        for(size_t nImageBindingIter=1; nImageBindingIter<vnImageBindings.size(); ++nImageBindingIter) ssSrc <<
             "    else if(imgCoord.z==" << nImageBindingIter << ") {\n"
             "        out_color = imageLoad(imgArray" << nImageBindingIter << ",ivec3(imgCoord.xy," << nImageLayer << "));\n"
             "    }\n";
    }
    else { ssSrc <<
             "    ivec2 imgSize = imageSize(img1);\n"
             "    ivec2 imgCoord = ivec3(mod(int(gl_FragCoord.x),imgSize.x),gl_FragCoord.y,int(gl_FragCoord.x)/imgSize.x);\n"
             "    if(imgCoord.z==0) { \n"
             "        out_color = imageLoad(img0,imgCoord.xy);\n"
             "    }\n";
        for(size_t nImageBindingIter=1; nImageBindingIter<vnImageBindings.size(); ++nImageBindingIter) ssSrc <<
             "    else if(imgCoord.z==" << nImageBindingIter << ") {\n"
             "        out_color = imageLoad(img" << nImageBindingIter << ",imgCoord.xy);\n"
             "    }\n";
    }
    if(bUseIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughFragmentShaderSource_SxSImgArrayLoad(bool bUseTopLeftFragCoordOrigin, GLenum eInternalFormat, GLuint nImageBinding, GLint nImageCount, bool bUseIntegralFormat) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n";
    if(bUseTopLeftFragCoordOrigin) ssSrc <<
             "layout(origin_upper_left) in vec4 gl_FragCoord;\n";
    const bool bInternalFormatIsIntegral = isInternalFormatIntegral(eInternalFormat);
    if(!bUseIntegralFormat && bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nImageBinding << ", " << getGLSLFormatNameFromInternalFormat(getNormalizedIntegralFormatFromInternalFormat(eInternalFormat)) << ") readonly uniform image2D" << (nImageCount>1?"Array imgArray":" img") << ";\n";
    else if(bUseIntegralFormat && bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform uimage2D" << (nImageCount>1?"Array imgArray":" img") << ";\n";
    else if(!bUseIntegralFormat && !bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform image2D" << (nImageCount>1?"Array imgArray":" img") << ";\n";
    else
        glError("bad internal format & useintegral setup");
    ssSrc << "const int imgArrayLayers = " << nImageCount << ";\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n";
    if(nImageCount>1) ssSrc <<
             "    ivec3 imgSize = imageSize(imgArray);\n"
             "    ivec3 imgCoord = ivec3(mod(int(gl_FragCoord.x),imgSize.x),gl_FragCoord.y,int(gl_FragCoord.x)/imgSize.x);\n"
             "    out_color = imageLoad(imgArray,imgCoord);\n";
    else ssSrc <<
             "    out_color = imageLoad(img,ivec2(gl_FragCoord.xy));\n";
    if(bUseIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLShader::getPassThroughComputeShaderSource_ImgLoadCopy(const glm::ivec2& vWorkGroupSize, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding, bool bUseIntegralFormat) {
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(local_size_x=" << vWorkGroupSize.x << ",local_size_y=" << vWorkGroupSize.y << ") in;\n";
    const bool bInternalFormatIsIntegral = isInternalFormatIntegral(eInternalFormat);
    if(!bUseIntegralFormat && bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nInputImageBinding << ", " << getGLSLFormatNameFromInternalFormat(getNormalizedIntegralFormatFromInternalFormat(eInternalFormat)) << ") readonly uniform image2D imgInput;\n"
             "layout(binding=" << nOutputImageBinding << ", " << getGLSLFormatNameFromInternalFormat(getNormalizedIntegralFormatFromInternalFormat(eInternalFormat)) << ") writeonly uniform image2D imgOutput;\n";
    else if(bUseIntegralFormat && bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nInputImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform uimage2D imgInput;\n"
             "layout(binding=" << nOutputImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") writeonly uniform uimage2D imgOutput;\n";
    else if(!bUseIntegralFormat && !bInternalFormatIsIntegral) ssSrc <<
             "layout(binding=" << nInputImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") readonly uniform image2D imgInput;\n"
             "layout(binding=" << nOutputImageBinding << ", " << getGLSLFormatNameFromInternalFormat(eInternalFormat) << ") writeonly uniform image2D imgOutput;\n";
    else
        glError("bad internal format & useintegral setup");
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    ivec2 imgCoord = ivec2(gl_GlobalInvocationID.xy);\n"
             "    imageStore(imgOutput,imgCoord,imageLoad(imgInput,imgCoord));\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string NBodySimulationUtils::getVertexShaderSource() {
    // the vertex shader simply passes through data
    return
    "#version 430\n"
    "layout(location = 0) in vec4 vposition;\n"
    "void main() {\n"
    "   gl_Position = vposition;\n"
    "}\n";
}

std::string NBodySimulationUtils::getGeometryShaderSource() {
    // the geometry shader creates the billboard quads
    return
    "#version 430\n"
    "layout(location = 0) uniform mat4 View;\n"
    "layout(location = 1) uniform mat4 Projection;\n"
    "layout (points) in;\n"
    "layout (triangle_strip, max_vertices = 4) out;\n"
    "out vec2 txcoord;\n"
    "void main() {\n"
    "   vec4 pos = View*gl_in[0].gl_Position;\n"
    "   txcoord = vec2(-1,-1);\n"
    "   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));\n"
    "   EmitVertex();\n"
    "   txcoord = vec2( 1,-1);\n"
    "   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));\n"
    "   EmitVertex();\n"
    "   txcoord = vec2(-1, 1);\n"
    "   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));\n"
    "   EmitVertex();\n"
    "   txcoord = vec2( 1, 1);\n"
    "   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));\n"
    "   EmitVertex();\n"
    "}\n";
}

std::string NBodySimulationUtils::getFragmentShaderSource() {
    // the fragment shader creates a bell like radial color distribution
    return
    "#version 330\n"
    "in vec2 txcoord;\n"
    "layout(location = 0) out vec4 FragColor;\n"
    "void main() {\n"
    "   float s = (1/(1+15.*dot(txcoord, txcoord))-1/16.);\n"
    "   FragColor = s*vec4(0.3,0.3,1.0,1);\n"
    "}\n";
}

std::string NBodySimulationUtils::getAccelerationComputeShaderSource() {
    // straight forward implementation of the nbody kernel
    return
    "#version 430\n"
    "layout(local_size_x=256) in;\n"
    "layout(location = 0) uniform float dt;\n"
    "layout(std430, binding=0) buffer pblock { vec4 positions[]; };\n"
    "layout(std430, binding=1) buffer vblock { vec4 velocities[]; };\n"
    "void main() {\n"
    "   int N = int(gl_NumWorkGroups.x*gl_WorkGroupSize.x);\n"
    "   int index = int(gl_GlobalInvocationID);\n"

    "   vec3 position = positions[index].xyz;\n"
    "   vec3 velocity = velocities[index].xyz;\n"
    "   vec3 acceleration = vec3(0,0,0);\n"
    "   for(int i = 0;i<N;++i) {\n"
    "       vec3 other = positions[i].xyz;\n"
    "       vec3 diff = position - other;\n"
    "       float invdist = 1.0/(length(diff)+0.001);\n"
    "       acceleration -= diff*0.1*invdist*invdist*invdist;\n"
    "   }\n"
    "   velocities[index] = vec4(velocity+dt*acceleration,0);\n"
    "}\n";
}

std::string NBodySimulationUtils::getTiledAccelerationComputeShaderSource() {
    // tiled version of the nbody shader that makes use of shared memory to reduce global memory transactions
    return
    "#version 430\n"
    "layout(local_size_x=256) in;\n"
    "layout(location = 0) uniform float dt;\n"
    "layout(std430, binding=0) buffer pblock { vec4 positions[]; };\n"
    "layout(std430, binding=1) buffer vblock { vec4 velocities[]; };\n"
    "shared vec4 tmp[gl_WorkGroupSize.x];\n"
    "void main() {\n"
    "   int N = int(gl_NumWorkGroups.x*gl_WorkGroupSize.x);\n"
    "   int index = int(gl_GlobalInvocationID);\n"
    "   vec3 position = positions[index].xyz;\n"
    "   vec3 velocity = velocities[index].xyz;\n"
    "   vec3 acceleration = vec3(0,0,0);\n"
    "   for(int tile = 0;tile<N;tile+=int(gl_WorkGroupSize.x)) {\n"
    "       tmp[gl_LocalInvocationIndex] = positions[tile + int(gl_LocalInvocationIndex)];\n"
    "       groupMemoryBarrier();\n"
    "       barrier();\n"
    "       for(int i = 0;i<gl_WorkGroupSize.x;++i) {\n"
    "           vec3 other = tmp[i].xyz;\n"
    "           vec3 diff = position - other;\n"
    "           float invdist = 1.0/(length(diff)+0.001);\n"
    "           acceleration -= diff*0.1*invdist*invdist*invdist;\n"
    "       }\n"
    "       groupMemoryBarrier();\n"
    "       barrier();\n"
    "   }\n"
    "   velocities[index] = vec4(velocity+dt*acceleration,0);\n"
    "}\n";
}

std::string NBodySimulationUtils::getIntegrateComputeShaderSource() {
    // the integrate shader does the second part of the euler integration
    return
    "#version 430\n"
    "layout(local_size_x=256) in;\n"
    "layout(location = 0) uniform float dt;\n"
    "layout(std430, binding=0) buffer pblock { vec4 positions[]; };\n"
    "layout(std430, binding=1) buffer vblock { vec4 velocities[]; };\n"
    "void main() {\n"
    "   int index = int(gl_GlobalInvocationID);\n"
    "   vec4 position = positions[index];\n"
    "   vec4 velocity = velocities[index];\n"
    "   position.xyz += dt*velocity.xyz;\n"
    "   positions[index] = position;\n"
    "}\n";
}

std::string ComputeShaderUtils::getComputeShaderSource_ParallelPrefixSum(int nMaxRowSize, bool bBinaryProc, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding) {
    // dispatch must use x=ceil(ceil(nColumns/2)/nMaxRowSize), y=nRows, z=1
    glAssert(isInternalFormatIntegral(eInternalFormat));
    glAssert(nMaxRowSize>1);
    const int nInvocations = (int)ceil((float)nMaxRowSize/2);
    glAssert((!(nMaxRowSize%2) && nInvocations==nMaxRowSize/2) || ((nMaxRowSize%2) && nInvocations-1==nMaxRowSize/2));
    int nMaxSharedMem;
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE,&nMaxSharedMem);
    glAssert(nMaxRowSize*4*4<nMaxSharedMem);
    const char* acInternalFormatName = getGLSLFormatNameFromInternalFormat(eInternalFormat);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "#define nRowSize " << nMaxRowSize << "\n"
             "#define nInvocations " << nInvocations << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(local_size_x=nInvocations) in;\n";
    if(nInputImageBinding!=nOutputImageBinding) ssSrc <<
             "layout(binding=" << nInputImageBinding << ", " << acInternalFormatName << ") readonly uniform uimage2D imgInput;\n"
             "layout(binding=" << nOutputImageBinding << ") writeonly uniform uimage2D imgOutput;\n";
    else ssSrc <<
             "layout(binding=" << nInputImageBinding << ", " << acInternalFormatName << ") uniform uimage2D imgInput;\n"
             "#define imgOutput imgInput\n";
    ssSrc << "shared uvec4 tmp[nRowSize];\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    tmp[gl_LocalInvocationID.x*2] = imageLoad(imgInput,ivec2(gl_WorkGroupID.x*nRowSize+gl_LocalInvocationID.x*2,gl_WorkGroupID.y))" << (bBinaryProc?"&1;":";") << "\n"
             "    tmp[gl_LocalInvocationID.x*2+1] = imageLoad(imgInput,ivec2(gl_WorkGroupID.x*nRowSize+gl_LocalInvocationID.x*2+1,gl_WorkGroupID.y))" << (bBinaryProc?"&1;":";") << "\n"
             "    uint offset = 1;\n"
             "    for(int depth=nInvocations; depth>0; depth>>=1) {\n"
             "        barrier();\n"
             "        if(gl_LocalInvocationID.x<depth) {\n"
             "            uint idx_lower = offset*(gl_LocalInvocationID.x*2+1)-1;\n"
             "            uint idx_upper = offset*(gl_LocalInvocationID.x*2+2)-1;\n"
             "            tmp[idx_upper] += tmp[idx_lower];\n"
             "        }\n"
             "        offset <<= 1;\n"
             "    }\n"
             "    if(gl_LocalInvocationID.x==0)\n"
             "        tmp[nRowSize-1] = uvec4(0);\n"
             "    for(int depth=1; depth<nRowSize; depth<<=1) {\n"
             "        offset >>= 1;\n"
             "        barrier();\n"
             "        if(gl_LocalInvocationID.x<depth) {\n"
             "            uint idx_upper = offset*(gl_LocalInvocationID.x*2+1)-1;\n"
             "            uint idx_lower = offset*(gl_LocalInvocationID.x*2+2)-1;\n"
             "            uvec4 swapsum = tmp[idx_upper];\n"
             "            tmp[idx_upper] = tmp[idx_lower];\n"
             "            tmp[idx_lower] += swapsum;\n"
             "        }\n"
             "    }\n"
             "    barrier();\n"
             "    imageStore(imgOutput,ivec2(gl_WorkGroupID.x*nRowSize+gl_LocalInvocationID.x*2,gl_WorkGroupID.y),tmp[gl_LocalInvocationID.x*2]);\n"
             "    imageStore(imgOutput,ivec2(gl_WorkGroupID.x*nRowSize+gl_LocalInvocationID.x*2+1,gl_WorkGroupID.y),tmp[gl_LocalInvocationID.x*2+1]);\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string ComputeShaderUtils::getComputeShaderSource_ParallelPrefixSum_BlockMerge(int nColumns, int nMaxRowSize, int nRows, GLenum eInternalFormat, GLuint nImageBinding) {
    // dispatch must use x=1, y=1, z=1
    glAssert(nMaxRowSize>0);
    glAssert(nColumns>nMaxRowSize); // shader step is useless otherwise
    glAssert(isInternalFormatIntegral(eInternalFormat));
    const char* acInternalFormatName = getGLSLFormatNameFromInternalFormat(eInternalFormat);
    const int nBlockCount = (int)ceil((float)nColumns/nMaxRowSize);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "#define nRowSize " << nMaxRowSize << "\n"
             "#define nBlockCount " << nBlockCount << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(local_size_x=1,local_size_y=" << nRows << ") in;\n"
             "layout(binding=" << nImageBinding << ", " << acInternalFormatName << ") uniform uimage2D img;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    for(int blockidx=1; blockidx<nBlockCount; ++blockidx) {\n"
             "        uvec4 last_block_sum = imageLoad(img,ivec2(blockidx*nRowSize-1,gl_LocalInvocationID.y));\n"
             "        for(int x=0; x<nRowSize; ++x) {\n"
             "            ivec2 imgCoord = ivec2(blockidx*nRowSize+x,gl_LocalInvocationID.y);\n"
             "            imageStore(img,imgCoord,imageLoad(img,imgCoord)+last_block_sum);\n"
             "        }\n"
             "    }\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string ComputeShaderUtils::getComputeShaderSource_Transpose(int nBlockSize, GLenum eInternalFormat, GLuint nInputImageBinding, GLuint nOutputImageBinding) {
    // dispatch must use x=ceil(nCols/nBlockSize), y=ceil(nRows/nBlockSize), z=1
    int nMaxSharedMem;
    glGetIntegerv(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE,&nMaxSharedMem);
    glAssert(nBlockSize*nBlockSize*4*4<nMaxSharedMem);
    glAssert(nInputImageBinding!=nOutputImageBinding);
    const bool bUsingIntegralFormat = isInternalFormatIntegral(eInternalFormat);
    const char* acInternalFormatName = getGLSLFormatNameFromInternalFormat(eInternalFormat);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "#define gvec " << (bUsingIntegralFormat?"uvec4":"vec4") << "\n"
             "#define blocksize " << nBlockSize << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(local_size_x=blocksize,local_size_y=blocksize) in;\n"
             "layout(binding=" << nInputImageBinding << ", " << acInternalFormatName << ") readonly uniform " << (bUsingIntegralFormat?"uimage2D":"image2D") << " imgInput;\n"
             "layout(binding=" << nOutputImageBinding << ") writeonly uniform " << (bUsingIntegralFormat?"uimage2D":"image2D") << " imgOutput;\n"
             "shared gvec tmp[blocksize][blocksize];\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    tmp[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = imageLoad(imgInput,ivec2(gl_GlobalInvocationID.xy));\n"
             "    barrier();\n"
             "    imageStore(imgOutput,ivec2(gl_GlobalInvocationID.yx),tmp[gl_LocalInvocationID.y][gl_LocalInvocationID.x]);\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string GLSLFunctionUtils::getShaderFunctionSource_absdiff(bool bUseBuiltinDistance) {
    std::stringstream ssSrc;
    ssSrc << "uint absdiff(in uint a, in uint b) {\n"
             "    return uint(" << (bUseBuiltinDistance?"distance(a,b)":"abs((int)a-(int)b)") << ");\n"
             "}\n";
    return ssSrc.str();
}

std::string GLSLFunctionUtils::getShaderFunctionSource_L1dist() {
    std::stringstream ssSrc;
    ssSrc << "uint L1dist(in uvec3 a, in uvec3 b) {\n"
             "    ivec3 absdiffs = abs(ivec3(a)-ivec3(b));\n"
             "    return uint(absdiffs.b+absdiffs.g+absdiffs.r);\n"
             "}\n";
    return ssSrc.str();
}

std::string GLSLFunctionUtils::getShaderFunctionSource_L2dist(bool bUseBuiltinDistance) {
    std::stringstream ssSrc;
    ssSrc << "uint L2dist(in uvec3 a, in uvec3 b) {\n"
             "    return uint(" << (bUseBuiltinDistance?"distance(a,b)":"sqrt(dot(ivec3(a)-ivec3(b)))") << ");\n"
             "}\n";
    return ssSrc.str();
}

std::string GLSLFunctionUtils::getShaderFunctionSource_hdist() {
    std::stringstream ssSrc;
    ssSrc << "uint hdist(in uvec3 a, in uvec3 b) {\n"
             "    return uint(bitCount(a^b));\n"
             "}\n";
    return ssSrc.str();
}

std::string GLSLFunctionUtils::getShaderFunctionSource_getRandNeighbor3x3(const int nBorderSize, const cv::Size& oFrameSize) {
    std::stringstream ssSrc;
    ssSrc << "const ivec2 _avNeighborPattern3x3[8] = ivec2[8](\n"
             "    ivec2(-1, 1),ivec2(0, 1),ivec2(1, 1),\n"
             "    ivec2(-1, 0),            ivec2(1, 0),\n"
             "    ivec2(-1,-1),ivec2(0,-1),ivec2(1,-1)\n"
             ");\n"
             "ivec2 getRandNeighbor3x3(in ivec2 vCurrPos, in uint nRandVal) {\n"
             "    const int nBorderSize = " << nBorderSize << ";\n"
             "    const int nFrameWidth = " << oFrameSize.width << ";\n"
             "    const int nFrameHeight = " << oFrameSize.height << ";\n"
             "    ivec2 vNeighborPos = vCurrPos+_avNeighborPattern3x3[nRandVal%8];\n"
             "    clamp(vNeighborPos,ivec2(nBorderSize),ivec2(nFrameWidth-nBorderSize-1,nFrameHeight-nBorderSize-1));\n"
             "    return vNeighborPos;\n"
             "}\n";
    return ssSrc.str();
}

std::string GLSLFunctionUtils::getShaderFunctionSource_frand() {
    std::stringstream ssSrc;
    ssSrc << "float frand(inout vec2 vSeed) {\n"
             "    float fRandVal = 0.5 + 0.5 * fract(sin(dot(vSeed.xy, vec2(12.9898, 78.233)))* 43758.5453);\n"
             "    vSeed *= fRandVal;\n"
             "    return fRandVal;\n"
             "}\n";
    return ssSrc.str();
}

std::string GLSLFunctionUtils::getShaderFunctionSource_urand() {
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

std::string GLSLFunctionUtils::getShaderFunctionSource_urand_tinymt32() {
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

void GLSLFunctionUtils::initTinyMT32Generators(glm::uvec3 vGeneratorLayout, std::vector<TMT32GenParams>& voData) {
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
