#pragma once

#include "litiv/utils/GLShaderUtils.hpp"

#define GLUTILS_IMGPROC_DEFAULT_WORKGROUP           glm::uvec2(12,8)
#define GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT         2
#define GLUTILS_IMGPROC_TEXTURE_ARRAY_SIZE          4
#define GLUTILS_IMGPROC_USE_TEXTURE_ARRAYS          0
#define GLUTILS_IMGPROC_USE_DOUBLE_PBO_INPUT        0
#define GLUTILS_IMGPROC_USE_DOUBLE_PBO_OUTPUT       1
#define GLUTILS_IMGPROC_USE_PBO_UPDATE_REALLOC      1 // @@@@@ unused?

class GLImageProcAlgo {
public:
    GLImageProcAlgo( size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures,
                     int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat);
    virtual ~GLImageProcAlgo();

    virtual std::string getVertexShaderSource() const;
    virtual std::string getFragmentShaderSource() const;
    virtual std::string getComputeShaderSource(size_t nStage) const = 0;

    inline bool setOutputFetching(bool b) {return (m_bFetchingOutput=(b&&m_bUsingOutput));}
    inline bool setDebugFetching(bool b) {return (m_bFetchingDebug=(b&&m_bUsingDebug));}
    inline bool getIsUsingDisplay() const {return m_bUsingDisplay;}
    inline GLuint getACBOId(size_t n) const {glAssert(n<m_nACBOs); return m_vnACBO[n];}
    inline GLuint getSSBOId(size_t n) const {glAssert(n<m_nSSBOs); return m_vnSSBO[n];}
    inline size_t getTextureBinding(size_t nLayer, size_t eTexID) const {return (m_bUsingTexArrays?0:nLayer)*m_nTextures+eTexID;}

    size_t fetchLastOutput(cv::Mat& oOutput) const;
    size_t fetchLastDebug(cv::Mat& oDebug) const;

    virtual void initialize(const cv::Mat& oInitInput, const cv::Mat& oROI);
    virtual void apply(const cv::Mat& oNextInput, bool bRebindAll=false);

    const size_t m_nLevels;
    const size_t m_nComputeStages;
    const size_t m_nSSBOs;
    const size_t m_nACBOs;
    const size_t m_nImages;
    const size_t m_nTextures;
    const size_t m_nSxSDisplayCount;
    const bool m_bUsingOutputPBOs;
    const bool m_bUsingDebugPBOs;
    const bool m_bUsingInputPBOs;
    const bool m_bUsingOutput;
    const bool m_bUsingDebug;
    const bool m_bUsingInput;
    const bool m_bUsingTexArrays;
    const bool m_bUsingTimers;
    const bool m_bUsingIntegralFormat;
    const glm::uvec2 m_vDefaultWorkGroupSize; // make dynamic? @@@@@
protected:
    enum eImageDefaultLayoutList {
        eImage_OutputBinding,
        eImage_DebugBinding,
        eImage_InputBinding,
        eImage_ROIBinding,
        eImage_GTBinding,
        // reserved here
        eImageDefaultBindingsCount
    };
    enum eTextureDefaultLayoutList {
        eTexture_OutputBinding,
        eTexture_DebugBinding,
        eTexture_InputBinding,
        eTexture_GTBinding,
        // reserved here
        eTextureDefaultBindingsCount
    };
    enum eStorageBufferDefaultBindingList {
        // reserved here
        eStorageBufferDefaultBindingsCount
    };
    enum eAtomicCounterBufferDefaultBindingList {
        eAtomicCounterBuffer_EvalBinding,
        // reserved here
        eAtomicCounterBufferDefaultBindingsCount
    };
    enum eGLTimersList {
        eGLTimer_TextureUpdate,
        eGLTimer_ComputeDispatch,
        eGLTimer_DisplayUpdate,
        eGLTimersCount
    };
    bool m_bUsingDisplay;
    bool m_bGLInitialized;
    cv::Size m_oFrameSize;
    size_t m_nInternalFrameIdx;
    size_t m_nLastOutputInternalIdx, m_nLastDebugInternalIdx;
    bool m_bFetchingOutput, m_bFetchingDebug;
    cv::Mat m_oLastOutput, m_oLastDebug;
    size_t m_nNextLayer,m_nCurrLayer,m_nLastLayer;
    size_t m_nCurrPBO, m_nNextPBO;
    std::array<GLuint,eGLTimersCount> m_nGLTimers;
    std::array<GLuint64,eGLTimersCount> m_nGLTimerVals;
    std::vector<std::unique_ptr<GLDynamicTexture2D>> m_vpInputArray;
    std::vector<std::unique_ptr<GLDynamicTexture2D>> m_vpDebugArray;
    std::vector<std::unique_ptr<GLDynamicTexture2D>> m_vpOutputArray;
    std::unique_ptr<GLDynamicTexture2DArray> m_pInputArray;
    std::unique_ptr<GLDynamicTexture2DArray> m_pDebugArray;
    std::unique_ptr<GLDynamicTexture2DArray> m_pOutputArray;
    std::vector<std::unique_ptr<GLShader>> m_vpImgProcShaders;
    std::unique_ptr<GLPixelBufferObject> m_apInputPBOs[2];
    std::unique_ptr<GLPixelBufferObject> m_apDebugPBOs[2];
    std::unique_ptr<GLPixelBufferObject> m_apOutputPBOs[2];
    std::unique_ptr<GLTexture2D> m_pROITexture;
    std::unique_ptr<GLTexture2D> m_apCustomTextures[3];
    GLScreenBillboard m_oDisplayBillboard;
    GLShader m_oDisplayShader;
    const int m_nOutputType;
    const int m_nDebugType;

    virtual void dispatch(size_t nStage, GLShader& oShader);
    static const char* getCurrTextureLayerUniformName();
    static const char* getLastTextureLayerUniformName();
    static const char* getFrameIndexUniformName();
    std::string getFragmentShaderSource_internal(int nOutputType,int nDebugType,int nInputType) const;
private:
    GLImageProcAlgo& operator=(const GLImageProcAlgo&)=delete;
    GLImageProcAlgo(const GLImageProcAlgo&)=delete;
    friend class GLImageProcEvaluatorAlgo;
    std::vector<GLuint> m_vnSSBO;
    std::vector<GLuint> m_vnACBO;
    int m_nInputType;
};

class GLImageProcEvaluatorAlgo : public GLImageProcAlgo {
public:
    GLImageProcEvaluatorAlgo(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount, size_t nCountersPerFrame,
                             int nDebugType, int nGroundtruthType, bool bUseIntegralFormat);
    virtual ~GLImageProcEvaluatorAlgo();
    const cv::Mat& getEvaluationAtomicCounterBuffer();
    virtual std::string getFragmentShaderSource() const;

    virtual void initialize(const cv::Mat& oInitInput, const cv::Mat& oInitGT, const cv::Mat& oROI);
    virtual void initialize(const cv::Mat& oInitGT, const cv::Mat& oROI);
    virtual void apply(const cv::Mat& oNextInput, const cv::Mat& oNextGT, bool bRebindAll=false);
    virtual void apply(const cv::Mat& oNextGT, bool bRebindAll=false);

protected:
    const int m_nGroundtruthType;
    const size_t m_nTotFrameCount;
    const size_t m_nEvalBufferFrameSize;
    const size_t m_nEvalBufferTotSize;
    const size_t m_nEvalBufferMaxSize;
    size_t m_nCurrEvalBufferSize;
    size_t m_nCurrEvalBufferOffsetPtr;
    size_t m_nCurrEvalBufferOffsetBlock;
    std::shared_ptr<GLImageProcAlgo> m_pParent;
private:
    cv::Mat m_oEvalQueryBuffer;
};

class GLImagePassThroughAlgo : public GLImageProcAlgo {
public:
    GLImagePassThroughAlgo(int nFrameType, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat);
    virtual std::string getComputeShaderSource(size_t nStage) const;
};

class BinaryMedianFilter : public GLImageProcAlgo {
public:
    // @@@ add support for variable kernels? per-px kernel size could be provided via image load/store
    // @@@ currently not using ROI
    // via integral image: O(n) (where n is the total image size --- does not depend on r, the kernel size)
    BinaryMedianFilter( size_t nKernelSize, size_t nBorderSize, const cv::Mat& oROI,
                        bool bUseOutputPBOs, bool bUseInputPBOs, bool bUseTexArrays,
                        bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat);
    virtual std::string getComputeShaderSource(size_t nStage) const;
    const size_t m_nKernelSize;
    const size_t m_nBorderSize;
    static const size_t m_nPPSMaxRowSize;
    static const size_t m_nTransposeBlockSize;
protected:
    std::vector<std::string> m_vsComputeShaderSources;
    std::vector<glm::uvec3> m_vvComputeShaderDispatchSizes;
    static const GLuint eImage_PPSAccumulator;
    static const GLuint eImage_PPSAccumulator_T;
    virtual void dispatch(size_t nStage, GLShader& oShader);
};
