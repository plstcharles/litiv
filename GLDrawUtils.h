#pragma once

#include "GLUtils.h"

#define GLScreenBillboard_FLIP_TEX_Y_COORDS 1

struct GLVertex {
    enum eVertexAttribList {
        eVertexAttrib_PositionIdx=0,
        eVertexAttrib_NormalIdx,
        eVertexAttrib_ColorIdx,
        eVertexAttrib_TexCoordIdx,
        eVertexAttribsCount,
    };
    glm::vec4 vPosition;
    glm::vec4 vNormal;
    glm::vec4 vColor;
    glm::vec4 vTexCoord;
};

struct GLVertexArrayObject {
    GLVertexArrayObject();
    virtual ~GLVertexArrayObject();
    virtual void render()=0;
    inline GLuint getVAOId() {return m_nVAO;}
private:
    GLVertexArrayObject& operator=(const GLVertexArrayObject&)=delete;
    GLVertexArrayObject(const GLVertexArrayObject&)=delete;
    GLuint m_nVAO;
};

struct GLPixelBufferObject {
    GLPixelBufferObject(const cv::Mat& oInitBufferData, GLenum eBufferTarget, GLenum eBufferUsage);
    ~GLPixelBufferObject();
    inline GLuint getPBOId() const {return m_nPBO;}
    inline int type() const {return m_nFrameType;}
    inline cv::Size size() const {return m_oFrameSize;}
    bool updateBuffer(const cv::Mat& oBufferData, bool bRealloc=false, bool bRebindAll=false);
    bool fetchBuffer(cv::Mat& oBufferData, bool bRebindAll=false);
    const GLenum m_eBufferTarget;
    const GLenum m_eBufferUsage;
private:
    GLPixelBufferObject& operator=(const GLPixelBufferObject&)=delete;
    GLPixelBufferObject(const GLPixelBufferObject&)=delete;
    GLuint m_nPBO;
    const int m_nBufferSize;
    const int m_nFrameType;
    const cv::Size m_oFrameSize;
};

struct GLTexture {
    // glPixelStore modifs might affect underlying behavior of this class
    enum eDefaultImageLayoutList {
        eDefaultImage_InputBinding=0,
        eDefaultImage_OutputBinding,
        eDefaultImageBindingsCount,
    };
    enum eDefaultTextureLayoutList {
        eDefaultTexture_ColorBinding=0,
        eDefaultTexture_NormalBinding,
        eDefaultTexture_ShadowBinding,
        eDefaultTextureBindingsCount,
    };
    GLTexture();
    virtual ~GLTexture();
    inline GLuint getTexId() {return m_nTex;}
private:
    GLTexture& operator=(const GLTexture&)=delete;
    GLTexture(const GLTexture&)=delete;
    GLuint m_nTex;
};

struct GLTexture2D : GLTexture {
    GLTexture2D(GLsizei nLevels, GLenum eInternalFormat, GLsizei nWidth, GLsizei nHeight,
                GLvoid* pData=nullptr, GLenum eDataFormat=GL_BGRA, GLenum eDataType=GL_UNSIGNED_BYTE);
    GLTexture2D(GLsizei nLevels, const cv::Mat& oTexture, bool bUseIntegralFormat);
    virtual ~GLTexture2D();
    virtual void bindToImage(GLuint nUnit, int nLevel, GLenum eAccess);
    virtual void bindToSampler(GLuint nUnit);
    const bool m_bUseIntegralFormat;
    const int m_nWidth;
    const int m_nHeight;
    const int m_nLevels;
    const GLenum m_eInternalFormat;
    const GLenum m_eDataFormat;
    const GLenum m_eDataType;
    const cv::Mat m_oInitTexture;
};

struct GLTexture2DArray : GLTexture {
    GLTexture2DArray(GLsizei nTextureCount, GLsizei nLevels, GLenum eInternalFormat, GLsizei nWidth, GLsizei nHeight,
                     GLvoid* pData=nullptr, GLenum eDataFormat=GL_BGRA, GLenum eDataType=GL_UNSIGNED_BYTE);
    GLTexture2DArray(GLsizei nLevels, const std::vector<cv::Mat>& voTextures, bool bUseIntegralFormat);
    virtual ~GLTexture2DArray();
    virtual void bindToImage(GLuint nUnit, int nLevel, int nLayer, GLenum eAccess);
    virtual void bindToImageArray(GLuint nUnit, int nLevel, GLenum eAccess);
    virtual void bindToSamplerArray(GLuint nUnit);
    const bool m_bUseIntegralFormat;
    const int m_nTextureCount;
    const int m_nWidth;
    const int m_nHeight;
    const int m_nLevels;
    const GLenum m_eInternalFormat;
    const GLenum m_eDataFormat;
    const GLenum m_eDataType;
    const std::vector<cv::Mat> m_voInitTextures;
};

struct GLDynamicTexture2D : GLTexture2D {
    GLDynamicTexture2D(GLsizei nLevels, const cv::Mat& oInitTexture, bool bUseIntegralFormat);
    virtual ~GLDynamicTexture2D();
    virtual void updateTexture(const cv::Mat& oTexture, bool bRebindAll=false);
    virtual void updateTexture(const GLPixelBufferObject& oPBO, bool bRebindAll=false);
    virtual void fetchTexture(cv::Mat& oTexture);
    virtual void fetchTexture(const GLPixelBufferObject& oPBO, bool bRebindAll=false);
};

struct GLDynamicTexture2DArray : GLTexture2DArray {
    GLDynamicTexture2DArray(GLsizei nLevels, const std::vector<cv::Mat>& voInitTextures, bool bUseIntegralFormat);
    virtual ~GLDynamicTexture2DArray();
    virtual void updateTexture(const cv::Mat& oTexture, int nLayer, bool bRebindAll=false, bool bRegenMipmaps=false);
    virtual void updateTexture(const GLPixelBufferObject& oPBO, int nLayer, bool bRebindAll=false);
    virtual void fetchTexture(cv::Mat& oTexture, int nLayer);
    virtual void fetchTexture(const GLPixelBufferObject& oPBO, int nLayer, bool bRebindAll=false);
private:
    cv::Mat m_oTextureArrayFetchBuffer;
};

struct GLScreenBillboard : GLVertexArrayObject {
    GLScreenBillboard();
    virtual ~GLScreenBillboard();
    virtual void render();
    inline GLuint getVBOId() {return m_nVBO;}
    inline GLuint getIBOId() {return m_nIBO;}
    static const GLuint s_nVertexCount;
    static const GLuint s_nIndexCount;
    static const GLVertex s_aVertices[];
    static const GLubyte s_anIndices[];
private:
    GLuint m_nVBO;
    GLuint m_nIBO;
};
