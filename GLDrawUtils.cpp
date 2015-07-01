#include "GLDrawUtils.h"

GLVertexArrayObject::GLVertexArrayObject() {
    glGenVertexArrays(1,&m_nVAO);
}

GLVertexArrayObject::~GLVertexArrayObject() {
    glDeleteVertexArrays(1,&m_nVAO);
}

GLPixelBufferObject::GLPixelBufferObject(const cv::Mat& oInitBufferData, GLenum eBufferTarget, GLenum eBufferUsage)
    :    m_eBufferTarget(eBufferTarget)
        ,m_eBufferUsage(eBufferUsage)
        ,m_nBufferSize(oInitBufferData.rows*oInitBufferData.cols*oInitBufferData.channels()*GLUtils::getByteSizeFromMatDepth(oInitBufferData.depth()))
        ,m_nFrameType(oInitBufferData.type())
        ,m_oFrameSize(oInitBufferData.size()) {
    glAssert(m_eBufferTarget==GL_PIXEL_PACK_BUFFER || (m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER && oInitBufferData.isContinuous()));
    glAssert(m_nBufferSize>0);
    glGenBuffers(1,&m_nPBO);
    glBindBuffer(m_eBufferTarget,m_nPBO);
    glBufferData(m_eBufferTarget,m_nBufferSize,(m_eBufferTarget==GL_PIXEL_PACK_BUFFER)?nullptr:oInitBufferData.data,m_eBufferUsage);
    glBindBuffer(m_eBufferTarget,0);
}

GLPixelBufferObject::~GLPixelBufferObject() {
    glDeleteBuffers(1,&m_nPBO);
}

bool GLPixelBufferObject::updateBuffer(const cv::Mat& oBufferData, bool bRealloc, bool bRebindAll) {
    glDbgAssert(m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER);
    glDbgAssert(oBufferData.type()==m_nFrameType && oBufferData.size()==m_oFrameSize && oBufferData.isContinuous());
    glBindBuffer(m_eBufferTarget,m_nPBO);
    if(bRealloc)
        glBufferData(m_eBufferTarget,m_nBufferSize,nullptr,m_eBufferUsage);
    void* pBufferClientPtr = glMapBuffer(m_eBufferTarget,GL_WRITE_ONLY);
    if(pBufferClientPtr) {
        memcpy(pBufferClientPtr,oBufferData.data,m_nBufferSize);
        glUnmapBuffer(m_eBufferTarget);
    }
    if(bRebindAll)
        glBindBuffer(m_eBufferTarget,0);
    return bool(pBufferClientPtr);
}

bool GLPixelBufferObject::fetchBuffer(cv::Mat& oBufferData, bool bRebindAll) {
    glDbgAssert(m_eBufferTarget==GL_PIXEL_PACK_BUFFER);
    glDbgAssert(oBufferData.type()==m_nFrameType && oBufferData.size()==m_oFrameSize && oBufferData.isContinuous());
    glBindBuffer(m_eBufferTarget,m_nPBO);
    void* pBufferClientPtr = glMapBuffer(m_eBufferTarget,GL_READ_ONLY);
    if(pBufferClientPtr) {
        memcpy(oBufferData.data,pBufferClientPtr,m_nBufferSize);
        glUnmapBuffer(m_eBufferTarget);
    }
    if(bRebindAll)
        glBindBuffer(m_eBufferTarget,0);
    return bool(pBufferClientPtr);
}

GLTexture::GLTexture() {
    glGenTextures(1,&m_nTex);
}

GLTexture::~GLTexture() {
    glDeleteTextures(1,&m_nTex);
}

GLTexture2D::GLTexture2D( GLsizei nLevels,
                          GLenum eInternalFormat,
                          GLsizei nWidth,
                          GLsizei nHeight,
                          GLvoid* pData,
                          GLenum eDataFormat,
                          GLenum eDataType)
    :    GLTexture()
        ,m_bUseIntegralFormat(GLUtils::isInternalFormatIntegral(eInternalFormat))
        ,m_nWidth(nWidth)
        ,m_nHeight(nHeight)
        ,m_nLevels(nLevels)
        ,m_eInternalFormat(eInternalFormat)
        ,m_eDataFormat(eDataFormat)
        ,m_eDataType(eDataType)
        ,m_oInitTexture(GLUtils::deepCopyImage(nWidth,nHeight,pData,eDataFormat,eDataType)) {
    glAssert(m_nLevels>=1 && m_nWidth>0 && m_nHeight>0);
    glBindTexture(GL_TEXTURE_2D,getTexId());
    glTexStorage2D(GL_TEXTURE_2D,m_nLevels,m_eInternalFormat,m_nWidth,m_nHeight);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_nWidth,m_nHeight,m_eDataFormat,m_eDataType,m_oInitTexture.data);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_MIRRORED_REPEAT);
    if(m_nLevels>1) {
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    }
    else {
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    }
    glErrorCheck;
}

GLTexture2D::GLTexture2D(GLsizei nLevels, const cv::Mat& oTexture, bool bUseIntegralFormat)
    :    GLTexture()
        ,m_bUseIntegralFormat(bUseIntegralFormat)
        ,m_nWidth(oTexture.cols)
        ,m_nHeight(oTexture.rows)
        ,m_nLevels(nLevels)
        ,m_eInternalFormat(GLUtils::getInternalFormatFromMatType(oTexture.type(),bUseIntegralFormat))
        ,m_eDataFormat(GLUtils::getDataFormatFromChannels(oTexture.channels(),bUseIntegralFormat))
        ,m_eDataType(GLUtils::getDataTypeFromMatDepth(oTexture.depth(),oTexture.channels()))
        ,m_oInitTexture(oTexture.clone()) {
    glAssert(m_nLevels>=1 && !m_oInitTexture.empty());
    glBindTexture(GL_TEXTURE_2D,getTexId());
    glTexStorage2D(GL_TEXTURE_2D,m_nLevels,m_eInternalFormat,m_nWidth,m_nHeight);
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_nWidth,m_nHeight,m_eDataFormat,m_eDataType,m_oInitTexture.data);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_MIRRORED_REPEAT);
    if(m_nLevels>1) {
        glGenerateMipmap(GL_TEXTURE_2D);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    }
    else {
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    }
    glErrorCheck;
}

GLTexture2D::~GLTexture2D() {}

void GLTexture2D::bindToImage(GLuint nUnit, int nLevel, GLenum eAccess) {
    glDbgAssert(nLevel>=0 && nLevel<m_nLevels);
    glBindImageTexture(nUnit,getTexId(),nLevel,GL_FALSE,0,eAccess,m_eInternalFormat);
}

void GLTexture2D::bindToSampler(GLuint nUnit) {
    glActiveTexture(GL_TEXTURE0+nUnit);
    glBindTexture(GL_TEXTURE_2D,getTexId());
}

GLTexture2DArray::GLTexture2DArray( GLsizei nTextureCount,
                                    GLsizei nLevels,
                                    GLenum eInternalFormat,
                                    GLsizei nWidth,
                                    GLsizei nHeight,
                                    GLvoid* pData,
                                    GLenum eDataFormat,
                                    GLenum eDataType)
    :    GLTexture()
        ,m_bUseIntegralFormat(GLUtils::isInternalFormatIntegral(eInternalFormat))
        ,m_nTextureCount(nTextureCount)
        ,m_nWidth(nWidth)
        ,m_nHeight(nHeight)
        ,m_nLevels(nLevels)
        ,m_eInternalFormat(eInternalFormat)
        ,m_eDataFormat(eDataFormat)
        ,m_eDataType(eDataType)
        ,m_voInitTextures(GLUtils::deepCopyImages(nTextureCount,nWidth,nHeight,pData,eDataFormat,eDataType)) {
    glAssert(m_nTextureCount>0 && m_nLevels>=1 && m_nWidth>0 && m_nHeight>0);
    glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
    glTexStorage3D(GL_TEXTURE_2D_ARRAY,m_nLevels,m_eInternalFormat,m_nWidth,m_nHeight,m_nTextureCount);
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,0,m_nWidth,m_nHeight,m_nTextureCount,m_eDataFormat,m_eDataType,pData);
    glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_S,GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_T,GL_MIRRORED_REPEAT);
    if(m_nLevels>1) {
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    }
    else {
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    }
    glErrorCheck;
}

GLTexture2DArray::GLTexture2DArray(GLsizei nLevels, const std::vector<cv::Mat>& voTextures, bool bUseIntegralFormat)
:    GLTexture()
    ,m_bUseIntegralFormat(bUseIntegralFormat)
    ,m_nTextureCount((int)voTextures.size())
    ,m_nWidth(voTextures[0].cols)
    ,m_nHeight(voTextures[0].rows)
    ,m_nLevels(nLevels)
    ,m_eInternalFormat(GLUtils::getInternalFormatFromMatType(voTextures[0].type(),bUseIntegralFormat))
    ,m_eDataFormat(GLUtils::getDataFormatFromChannels(voTextures[0].channels(),bUseIntegralFormat))
    ,m_eDataType(GLUtils::getDataTypeFromMatDepth(voTextures[0].depth(),voTextures[0].channels()))
    ,m_voInitTextures(GLUtils::deepCopyImages(voTextures)) {
    glAssert(m_nLevels>=1);
    glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
    glTexStorage3D(GL_TEXTURE_2D_ARRAY,m_nLevels,m_eInternalFormat,m_nWidth,m_nHeight,m_nTextureCount);
    for(int nTexIter=0; nTexIter<m_nTextureCount; ++nTexIter) {
        glAssert(m_voInitTextures[nTexIter].size()==m_voInitTextures[0].size() && m_voInitTextures[nTexIter].type()==m_voInitTextures[0].type());
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,nTexIter,m_nWidth,m_nHeight,1,m_eDataFormat,m_eDataType,m_voInitTextures[nTexIter].data);
    }
    glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_S,GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_WRAP_T,GL_MIRRORED_REPEAT);
    if(m_nLevels>1) {
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR);
    }
    else {
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    }
    glErrorCheck;
}

GLTexture2DArray::~GLTexture2DArray() {}

void GLTexture2DArray::bindToImage(GLuint nUnit, int nLevel, int nLayer, GLenum eAccess) {
    glDbgAssert(nLevel>=0 && nLevel<m_nLevels && nLayer>=0 && nLayer<m_nTextureCount);
    glBindImageTexture(nUnit,getTexId(),nLevel,GL_FALSE,nLayer,eAccess,m_eInternalFormat);
}

void GLTexture2DArray::bindToImageArray(GLuint nUnit, int nLevel, GLenum eAccess) {
    glDbgAssert(nLevel>=0 && nLevel<m_nLevels);
    glBindImageTexture(nUnit,getTexId(),nLevel,GL_TRUE,0,eAccess,m_eInternalFormat);
}

void GLTexture2DArray::bindToSamplerArray(GLuint nUnit) {
    glActiveTexture(GL_TEXTURE0+nUnit);
    glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
}

GLDynamicTexture2D::GLDynamicTexture2D(GLsizei nLevels, const cv::Mat& oInitTexture, bool bUseIntegralFormat)
    :    GLTexture2D(nLevels,oInitTexture,bUseIntegralFormat) {}

GLDynamicTexture2D::~GLDynamicTexture2D() {}

void GLDynamicTexture2D::updateTexture(const cv::Mat& oTexture, bool bRebindAll) {
    glDbgAssert(!oTexture.empty());
    glDbgAssert(oTexture.size()==m_oInitTexture.size() && oTexture.type()==m_oInitTexture.type());
    glDbgAssert(oTexture.isContinuous());
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D,getTexId());
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_nWidth,m_nHeight,m_eDataFormat,m_eDataType,oTexture.data);
    if(m_nLevels>1)
        glGenerateMipmap(GL_TEXTURE_2D);
}

void GLDynamicTexture2D::updateTexture(const GLPixelBufferObject& oPBO, bool bRebindAll) {
    glDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER);
    glDbgAssert(oPBO.size()==m_oInitTexture.size() && oPBO.type()==m_oInitTexture.type());
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D,getTexId());
    glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_nWidth,m_nHeight,m_eDataFormat,m_eDataType,nullptr);
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

void GLDynamicTexture2D::fetchTexture(cv::Mat& oTexture) {
    glDbgAssert(!oTexture.empty());
    glDbgAssert(oTexture.size()==m_oInitTexture.size() && oTexture.type()==m_oInitTexture.type());
    glDbgAssert(oTexture.isContinuous());
    glBindTexture(GL_TEXTURE_2D,getTexId());
    glGetTexImage(GL_TEXTURE_2D,0,m_eDataFormat,m_eDataType,oTexture.data);
}

void GLDynamicTexture2D::fetchTexture(const GLPixelBufferObject& oPBO, bool bRebindAll) {
    glDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_PACK_BUFFER);
    glDbgAssert(oPBO.size()==m_oInitTexture.size() && oPBO.type()==m_oInitTexture.type());
    glBindTexture(GL_TEXTURE_2D,getTexId());
    glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
    glGetTexImage(GL_TEXTURE_2D,0,m_eDataFormat,m_eDataType,nullptr);
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

GLDynamicTexture2DArray::GLDynamicTexture2DArray(GLsizei nLevels, const std::vector<cv::Mat>& voInitTextures, bool bUseIntegralFormat)
    :    GLTexture2DArray(nLevels,voInitTextures,bUseIntegralFormat)
        ,m_oTextureArrayFetchBuffer(glGetTextureSubImage?cv::Mat():cv::Mat(m_nHeight*m_nTextureCount,m_nWidth,voInitTextures[0].type())) {
    static bool s_bAlreadyWarned = false;
    if(!glGetTextureSubImage && !s_bAlreadyWarned) {
        std::cout << "\tWarning: glGetTextureSubImage not supported, performance might be affected (full arrays will be transferred)" << std::endl;
        s_bAlreadyWarned = true;
    }
}

GLDynamicTexture2DArray::~GLDynamicTexture2DArray() {}

void GLDynamicTexture2DArray::updateTexture(const cv::Mat& oTexture, int nLayer, bool bRebindAll, bool bRegenMipmaps) {
    glDbgAssert(!oTexture.empty());
    glDbgAssert(oTexture.size()==m_voInitTextures[0].size() && oTexture.type()==m_voInitTextures[0].type());
    glDbgAssert(oTexture.isContinuous());
    glDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,nLayer,m_nWidth,m_nHeight,1,m_eDataFormat,m_eDataType,oTexture.data);
    if(m_nLevels>1 && bRegenMipmaps)
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
}

void GLDynamicTexture2DArray::updateTexture(const GLPixelBufferObject& oPBO, int nLayer, bool bRebindAll) {
    glDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER);
    glDbgAssert(oPBO.size()==m_voInitTextures[0].size() && oPBO.type()==m_voInitTextures[0].type());
    glDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
    glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,nLayer,m_nWidth,m_nHeight,1,m_eDataFormat,m_eDataType,nullptr);
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

void GLDynamicTexture2DArray::fetchTexture(cv::Mat& oTexture, int nLayer) {
    glDbgAssert(!oTexture.empty());
    glDbgAssert(oTexture.size()==m_voInitTextures[0].size() && oTexture.type()==m_voInitTextures[0].type());
    glDbgAssert(oTexture.isContinuous());
    glDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(glGetTextureSubImage)
        glGetTextureSubImage(getTexId(),0,0,0,nLayer,m_nWidth,m_nHeight,1,m_eDataFormat,m_eDataType,m_voInitTextures[0].step.p[0]*m_nHeight,oTexture.data);
    else {
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
        glGetTexImage(GL_TEXTURE_2D_ARRAY,0,m_eDataFormat,m_eDataType,m_oTextureArrayFetchBuffer.data);
        m_oTextureArrayFetchBuffer(cv::Rect(0,m_nHeight*nLayer,m_nWidth,m_nHeight)).copyTo(oTexture);
    }
}

void GLDynamicTexture2DArray::fetchTexture(const GLPixelBufferObject& oPBO, int nLayer, bool bRebindAll) {
    glDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_PACK_BUFFER);
    glDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(glGetTextureSubImage) {
        glDbgAssert(oPBO.size()==m_voInitTextures[0].size() && oPBO.type()==m_voInitTextures[0].type());
        glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
        glGetTextureSubImage(getTexId(),0,0,0,nLayer,m_nWidth,m_nHeight,1,m_eDataFormat,m_eDataType,m_voInitTextures[0].step.p[0]*m_nHeight,nullptr);
    }
    else {
        glDbgAssert(oPBO.size()==m_oTextureArrayFetchBuffer.size() && oPBO.type()==m_oTextureArrayFetchBuffer.type());
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
        glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
        glGetTexImage(GL_TEXTURE_2D_ARRAY,0,m_eDataFormat,m_eDataType,nullptr);
    }
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

GLScreenBillboard::GLScreenBillboard()
    :    GLVertexArrayObject() {
    glBindVertexArray(getVAOId());
    glGenBuffers(1,&m_nVBO);
    glBindBuffer(GL_ARRAY_BUFFER,m_nVBO);
    glBufferData(GL_ARRAY_BUFFER,s_nVertexCount*sizeof(GLVertex),s_aVertices,GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)GLVertex::eVertexAttrib_PositionIdx);
    glVertexAttribPointer((GLuint)GLVertex::eVertexAttrib_PositionIdx,sizeof(GLVertex::vPosition)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vPosition));
    glEnableVertexAttribArray((GLuint)GLVertex::eVertexAttrib_NormalIdx);
    glVertexAttribPointer((GLuint)GLVertex::eVertexAttrib_NormalIdx,sizeof(GLVertex::vNormal)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vNormal));
    glEnableVertexAttribArray((GLuint)GLVertex::eVertexAttrib_ColorIdx);
    glVertexAttribPointer((GLuint)GLVertex::eVertexAttrib_ColorIdx,sizeof(GLVertex::vColor)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vColor));
    glEnableVertexAttribArray((GLuint)GLVertex::eVertexAttrib_TexCoordIdx);
    glVertexAttribPointer((GLuint)GLVertex::eVertexAttrib_TexCoordIdx,sizeof(GLVertex::vTexCoord)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vTexCoord));
    glGenBuffers(1,&m_nIBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_nIBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,s_nIndexCount*sizeof(GLubyte),s_anIndices,GL_STATIC_DRAW);
    glBindVertexArray(0);
    glErrorCheck;
}

GLScreenBillboard::~GLScreenBillboard() {
    glDeleteBuffers(1,&m_nVBO);
    glDeleteBuffers(1,&m_nIBO);
}

void GLScreenBillboard::render() {
    glBindVertexArray(getVAOId());
    glDrawElements(GL_TRIANGLES,GLScreenBillboard::s_nIndexCount,GL_UNSIGNED_BYTE,0);
    glDbgErrorCheck;
}

const GLuint GLScreenBillboard::s_nVertexCount = 4;

const GLuint GLScreenBillboard::s_nIndexCount = 6;

const GLVertex GLScreenBillboard::s_aVertices[GLScreenBillboard::s_nVertexCount] = {
    //       vPosition                  vNormal                vColor                vTexCoord
#if GLScreenBillboard_FLIP_TEX_Y_COORDS
    {{-1.0f,-1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {0.0f,1.0f,0.0f,0.0f}}, // 0
    {{ 1.0f,-1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {1.0f,1.0f,0.0f,0.0f}}, // 1
    {{ 1.0f, 1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {1.0f,0.0f,0.0f,0.0f}}, // 2
    {{-1.0f, 1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {0.0f,0.0f,0.0f,0.0f}}, // 3
#else //!GLScreenBillboard_FLIP_TEX_Y_COORDS
    {{-1.0f,-1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {0.0f,0.0f,0.0f,0.0f}}, // 0
    {{ 1.0f,-1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {1.0f,0.0f,0.0f,0.0f}}, // 1
    {{ 1.0f, 1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {1.0f,1.0f,0.0f,0.0f}}, // 2
    {{-1.0f, 1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {0.0f,1.0f,0.0f,0.0f}}, // 3
#endif //GLScreenBillboard_FLIP_TEX_Y_COORDS
};

const GLubyte GLScreenBillboard::s_anIndices[GLScreenBillboard::s_nIndexCount] = {0,1,3,3,1,2};

