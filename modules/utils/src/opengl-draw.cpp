
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

#include "litiv/utils/opengl-draw.hpp"

GLVertexArrayObject::GLVertexArrayObject() {
    glGenVertexArrays(1,&m_nVAO);
}

GLVertexArrayObject::~GLVertexArrayObject() {
    glDeleteVertexArrays(1,&m_nVAO);
}

GLPixelBufferObject::GLPixelBufferObject(const cv::Mat& oInitBufferData, GLenum eBufferTarget, GLenum eBufferUsage) :
        m_eBufferTarget(eBufferTarget),
        m_eBufferUsage(eBufferUsage),
        m_nBufferSize(oInitBufferData.rows*oInitBufferData.cols*oInitBufferData.channels()*lv::gl::getByteSizeFromMatDepth(oInitBufferData.depth())),
        m_nFrameType(oInitBufferData.type()),
        m_oFrameSize(oInitBufferData.size()) {
    lvAssert(m_eBufferTarget==GL_PIXEL_PACK_BUFFER || (m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER && oInitBufferData.isContinuous()));
    lvAssert(m_nBufferSize>0);
    glGenBuffers(1,&m_nPBO);
    glBindBuffer(m_eBufferTarget,m_nPBO);
    glBufferData(m_eBufferTarget,m_nBufferSize,(m_eBufferTarget==GL_PIXEL_PACK_BUFFER)?nullptr:oInitBufferData.data,m_eBufferUsage);
    glBindBuffer(m_eBufferTarget,0);
}

GLPixelBufferObject::~GLPixelBufferObject() {
    glDeleteBuffers(1,&m_nPBO);
}

bool GLPixelBufferObject::updateBuffer(const cv::Mat& oBufferData, bool bRealloc, bool bRebindAll) {
    lvDbgAssert(m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER);
    lvDbgAssert(oBufferData.type()==m_nFrameType && oBufferData.size()==m_oFrameSize && oBufferData.isContinuous());
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
    return pBufferClientPtr!=nullptr;
}

bool GLPixelBufferObject::fetchBuffer(cv::Mat& oBufferData, bool bRebindAll) {
    lvDbgAssert(m_eBufferTarget==GL_PIXEL_PACK_BUFFER);
    lvDbgAssert(oBufferData.type()==m_nFrameType && oBufferData.size()==m_oFrameSize && oBufferData.isContinuous());
    glBindBuffer(m_eBufferTarget,m_nPBO);
    void* pBufferClientPtr = glMapBuffer(m_eBufferTarget,GL_READ_ONLY);
    if(pBufferClientPtr) {
        memcpy(oBufferData.data,pBufferClientPtr,m_nBufferSize);
        glUnmapBuffer(m_eBufferTarget);
    }
    if(bRebindAll)
        glBindBuffer(m_eBufferTarget,0);
    return pBufferClientPtr!=nullptr;
}

GLTexture::GLTexture() {
    glGenTextures(1,&m_nTex);
}

GLTexture::~GLTexture() {
    glDeleteTextures(1,&m_nTex);
}

GLTexture2D::GLTexture2D( GLsizei nLevels, GLenum eInternalFormat, GLsizei nWidth, GLsizei nHeight,
                          GLvoid* pData, GLenum eDataFormat, GLenum eDataType) :
        GLTexture(),
        m_bUseIntegralFormat(lv::gl::isInternalFormatIntegral(eInternalFormat)),
        m_nWidth(nWidth),
        m_nHeight(nHeight),
        m_nLevels(nLevels),
        m_eInternalFormat(eInternalFormat),
        m_eDataFormat(eDataFormat),
        m_eDataType(eDataType),
        m_oInitTexture(lv::gl::deepCopyImage(nWidth,nHeight,pData,eDataFormat,eDataType)) {
    lvAssert(m_nLevels>=1 && m_nWidth>0 && m_nHeight>0);
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

GLTexture2D::GLTexture2D(GLsizei nLevels, const cv::Mat& oTexture, bool bUseIntegralFormat) :
        GLTexture(),
        m_bUseIntegralFormat(bUseIntegralFormat),
        m_nWidth(oTexture.cols),
        m_nHeight(oTexture.rows),
        m_nLevels(nLevels),
        m_eInternalFormat(lv::gl::getInternalFormatFromMatType(oTexture.type(),bUseIntegralFormat)),
        m_eDataFormat(lv::gl::getDataFormatFromChannels(oTexture.channels(),bUseIntegralFormat)),
        m_eDataType(lv::gl::getDataTypeFromMatDepth(oTexture.depth(),oTexture.channels())),
        m_oInitTexture(oTexture.clone()) {
    lvAssert(m_nLevels>=1 && !m_oInitTexture.empty());
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
    lvDbgAssert(nLevel>=0 && nLevel<m_nLevels);
    glBindImageTexture(nUnit,getTexId(),nLevel,GL_FALSE,0,eAccess,m_eInternalFormat);
}

void GLTexture2D::bindToSampler(GLuint nUnit) {
    glActiveTexture(GL_TEXTURE0+nUnit);
    glBindTexture(GL_TEXTURE_2D,getTexId());
}

GLTexture2DArray::GLTexture2DArray( GLsizei nTextureCount, GLsizei nLevels, GLenum eInternalFormat, GLsizei nWidth,
                                    GLsizei nHeight, GLvoid* pData, GLenum eDataFormat, GLenum eDataType) :
        GLTexture(),
        m_bUseIntegralFormat(lv::gl::isInternalFormatIntegral(eInternalFormat)),
        m_nTextureCount(nTextureCount),
        m_nWidth(nWidth),
        m_nHeight(nHeight),
        m_nLevels(nLevels),
        m_eInternalFormat(eInternalFormat),
        m_eDataFormat(eDataFormat),
        m_eDataType(eDataType),
        m_voInitTextures(lv::gl::deepCopyImages(nTextureCount,nWidth,nHeight,pData,eDataFormat,eDataType)) {
    lvAssert(m_nTextureCount>0 && m_nLevels>=1 && m_nWidth>0 && m_nHeight>0);
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

GLTexture2DArray::GLTexture2DArray(GLsizei nLevels, const std::vector<cv::Mat>& voTextures, bool bUseIntegralFormat) :
        GLTexture(),
        m_bUseIntegralFormat(bUseIntegralFormat),
        m_nTextureCount((int)voTextures.size()),
        m_nWidth(voTextures[0].cols),
        m_nHeight(voTextures[0].rows),
        m_nLevels(nLevels),
        m_eInternalFormat(lv::gl::getInternalFormatFromMatType(voTextures[0].type(),bUseIntegralFormat)),
        m_eDataFormat(lv::gl::getDataFormatFromChannels(voTextures[0].channels(),bUseIntegralFormat)),
        m_eDataType(lv::gl::getDataTypeFromMatDepth(voTextures[0].depth(),voTextures[0].channels())),
        m_voInitTextures(lv::gl::deepCopyImages(voTextures)) {
    lvAssert(m_nLevels>=1);
    glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
    glTexStorage3D(GL_TEXTURE_2D_ARRAY,m_nLevels,m_eInternalFormat,m_nWidth,m_nHeight,m_nTextureCount);
    for(int nTexIter=0; nTexIter<m_nTextureCount; ++nTexIter) {
        lvAssert(m_voInitTextures[nTexIter].size()==m_voInitTextures[0].size() && m_voInitTextures[nTexIter].type()==m_voInitTextures[0].type());
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
    lvDbgAssert(nLevel>=0 && nLevel<m_nLevels && nLayer>=0 && nLayer<m_nTextureCount);
    glBindImageTexture(nUnit,getTexId(),nLevel,GL_FALSE,nLayer,eAccess,m_eInternalFormat);
}

void GLTexture2DArray::bindToImageArray(GLuint nUnit, int nLevel, GLenum eAccess) {
    lvDbgAssert(nLevel>=0 && nLevel<m_nLevels);
    glBindImageTexture(nUnit,getTexId(),nLevel,GL_TRUE,0,eAccess,m_eInternalFormat);
}

void GLTexture2DArray::bindToSamplerArray(GLuint nUnit) {
    glActiveTexture(GL_TEXTURE0+nUnit);
    glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
}

GLDynamicTexture2D::GLDynamicTexture2D(GLsizei nLevels, const cv::Mat& oInitTexture, bool bUseIntegralFormat) :
        GLTexture2D(nLevels,oInitTexture,bUseIntegralFormat) {}

GLDynamicTexture2D::~GLDynamicTexture2D() {}

void GLDynamicTexture2D::updateTexture(const cv::Mat& oTexture, bool bRebindAll) {
    lvDbgAssert(!oTexture.empty());
    lvDbgAssert(oTexture.size()==m_oInitTexture.size() && oTexture.type()==m_oInitTexture.type());
    lvDbgAssert(oTexture.isContinuous());
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D,getTexId());
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_nWidth,m_nHeight,m_eDataFormat,m_eDataType,oTexture.data);
    if(m_nLevels>1)
        glGenerateMipmap(GL_TEXTURE_2D);
}

void GLDynamicTexture2D::updateTexture(const GLPixelBufferObject& oPBO, bool bRebindAll) {
    lvDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER);
    lvDbgAssert(oPBO.size()==m_oInitTexture.size() && oPBO.type()==m_oInitTexture.type());
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D,getTexId());
    glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,m_nWidth,m_nHeight,m_eDataFormat,m_eDataType,nullptr);
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

void GLDynamicTexture2D::fetchTexture(cv::Mat& oTexture) {
    lvDbgAssert(!oTexture.empty());
    lvDbgAssert(oTexture.size()==m_oInitTexture.size() && oTexture.type()==m_oInitTexture.type());
    lvDbgAssert(oTexture.isContinuous());
    glBindTexture(GL_TEXTURE_2D,getTexId());
    glGetTexImage(GL_TEXTURE_2D,0,m_eDataFormat,m_eDataType,oTexture.data);
}

void GLDynamicTexture2D::fetchTexture(const GLPixelBufferObject& oPBO, bool bRebindAll) {
    lvDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_PACK_BUFFER);
    lvDbgAssert(oPBO.size()==m_oInitTexture.size() && oPBO.type()==m_oInitTexture.type());
    glBindTexture(GL_TEXTURE_2D,getTexId());
    glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
    glGetTexImage(GL_TEXTURE_2D,0,m_eDataFormat,m_eDataType,nullptr);
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

GLDynamicTexture2DArray::GLDynamicTexture2DArray(GLsizei nLevels, const std::vector<cv::Mat>& voInitTextures, bool bUseIntegralFormat) :
        GLTexture2DArray(nLevels,voInitTextures,bUseIntegralFormat),
        m_oTextureArrayFetchBuffer(glGetTextureSubImage?cv::Mat():cv::Mat(m_nHeight*m_nTextureCount,m_nWidth,voInitTextures[0].type())) {
    static bool s_bAlreadyWarned = false;
    if(!glGetTextureSubImage && !s_bAlreadyWarned) {
        std::cout << "\tWarning: glGetTextureSubImage not supported, performance might be affected (full arrays will be transferred)" << std::endl;
        s_bAlreadyWarned = true;
    }
}

GLDynamicTexture2DArray::~GLDynamicTexture2DArray() {}

void GLDynamicTexture2DArray::updateTexture(const cv::Mat& oTexture, int nLayer, bool bRebindAll, bool bRegenMipmaps) {
    lvDbgAssert(!oTexture.empty());
    lvDbgAssert(oTexture.size()==m_voInitTextures[0].size() && oTexture.type()==m_voInitTextures[0].type());
    lvDbgAssert(oTexture.isContinuous());
    lvDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,nLayer,m_nWidth,m_nHeight,1,m_eDataFormat,m_eDataType,oTexture.data);
    if(m_nLevels>1 && bRegenMipmaps)
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
}

void GLDynamicTexture2DArray::updateTexture(const GLPixelBufferObject& oPBO, int nLayer, bool bRebindAll) {
    lvDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_UNPACK_BUFFER);
    lvDbgAssert(oPBO.size()==m_voInitTextures[0].size() && oPBO.type()==m_voInitTextures[0].type());
    lvDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(bRebindAll)
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
    glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,0,0,0,nLayer,m_nWidth,m_nHeight,1,m_eDataFormat,m_eDataType,nullptr);
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

void GLDynamicTexture2DArray::fetchTexture(cv::Mat& oTexture, int nLayer) {
    lvDbgAssert(!oTexture.empty());
    lvDbgAssert(oTexture.size()==m_voInitTextures[0].size() && oTexture.type()==m_voInitTextures[0].type());
    lvDbgAssert(oTexture.isContinuous());
    lvDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(glGetTextureSubImage)
        glGetTextureSubImage(getTexId(),0,0,0,nLayer,(GLsizei)m_nWidth,(GLsizei)m_nHeight,1,m_eDataFormat,m_eDataType,(GLsizei)(m_voInitTextures[0].step.p[0]*m_nHeight),oTexture.data);
    else {
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
        glGetTexImage(GL_TEXTURE_2D_ARRAY,0,m_eDataFormat,m_eDataType,m_oTextureArrayFetchBuffer.data);
        m_oTextureArrayFetchBuffer(cv::Rect(0,m_nHeight*nLayer,m_nWidth,m_nHeight)).copyTo(oTexture);
    }
}

void GLDynamicTexture2DArray::fetchTexture(const GLPixelBufferObject& oPBO, int nLayer, bool bRebindAll) {
    lvDbgAssert(oPBO.m_eBufferTarget==GL_PIXEL_PACK_BUFFER);
    lvDbgAssert(nLayer>=0 && nLayer<m_nTextureCount);
    if(glGetTextureSubImage) {
        lvDbgAssert(oPBO.size()==m_voInitTextures[0].size() && oPBO.type()==m_voInitTextures[0].type());
        glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
        glGetTextureSubImage(getTexId(),0,0,0,nLayer,(GLsizei)m_nWidth,(GLsizei)m_nHeight,1,m_eDataFormat,m_eDataType,(GLsizei)(m_voInitTextures[0].step.p[0]*m_nHeight),nullptr);
    }
    else {
        lvDbgAssert(oPBO.size()==m_oTextureArrayFetchBuffer.size() && oPBO.type()==m_oTextureArrayFetchBuffer.type());
        glBindTexture(GL_TEXTURE_2D_ARRAY,getTexId());
        glBindBuffer(oPBO.m_eBufferTarget,oPBO.getPBOId());
        glGetTexImage(GL_TEXTURE_2D_ARRAY,0,m_eDataFormat,m_eDataType,nullptr);
    }
    if(bRebindAll)
        glBindBuffer(oPBO.m_eBufferTarget,0);
}

GLScreenBillboard::GLScreenBillboard() : GLVertexArrayObject() {
    glBindVertexArray(getVAOId());
    glGenBuffers(1,&m_nVBO);
    glBindBuffer(GL_ARRAY_BUFFER,m_nVBO);
    glBufferData(GL_ARRAY_BUFFER,s_nVertexCount*sizeof(GLVertex),s_aVertices,GL_STATIC_DRAW);
    glEnableVertexAttribArray((GLuint)GLVertex::VertexAttrib_PositionIdx);
    glVertexAttribPointer((GLuint)GLVertex::VertexAttrib_PositionIdx,sizeof(GLVertex::vPosition)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vPosition));
    glEnableVertexAttribArray((GLuint)GLVertex::VertexAttrib_NormalIdx);
    glVertexAttribPointer((GLuint)GLVertex::VertexAttrib_NormalIdx,sizeof(GLVertex::vNormal)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vNormal));
    glEnableVertexAttribArray((GLuint)GLVertex::VertexAttrib_ColorIdx);
    glVertexAttribPointer((GLuint)GLVertex::VertexAttrib_ColorIdx,sizeof(GLVertex::vColor)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vColor));
    glEnableVertexAttribArray((GLuint)GLVertex::VertexAttrib_TexCoordIdx);
    glVertexAttribPointer((GLuint)GLVertex::VertexAttrib_TexCoordIdx,sizeof(GLVertex::vTexCoord)/sizeof(GLfloat),GL_FLOAT,GL_FALSE,sizeof(GLVertex),(GLvoid*)offsetof(GLVertex,vTexCoord));
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

void GLScreenBillboard::render(GLMatrices /*oMats*/) {
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
#else //(!GLScreenBillboard_FLIP_TEX_Y_COORDS)
    {{-1.0f,-1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {0.0f,0.0f,0.0f,0.0f}}, // 0
    {{ 1.0f,-1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {1.0f,0.0f,0.0f,0.0f}}, // 1
    {{ 1.0f, 1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {1.0f,1.0f,0.0f,0.0f}}, // 2
    {{-1.0f, 1.0f, 0.0f, 1.0f}, {0.0f,0.0f,-1.0f,0.0f}, {1.0f,1.0f,1.0f,1.0f}, {0.0f,1.0f,0.0f,0.0f}}, // 3
#endif //GLScreenBillboard_FLIP_TEX_Y_COORDS
};

const GLubyte GLScreenBillboard::s_anIndices[GLScreenBillboard::s_nIndexCount] = {0,1,3,3,1,2};
