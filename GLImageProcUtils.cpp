#include "GLImageProcUtils.h"

GLImageProcAlgo::GLImageProcAlgo( int nLevels, int nLayers, int nComputeStages,
                                  int nOutputType, int nDebugType, int nInputType,
                                  bool bUseOutputPBOs, bool bUseDebugPBOs, bool bUseInputPBOs,
                                  bool bUseTexArrays, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat)
    :    m_nLevels(nLevels)
        ,m_nLayers(nLayers)
        ,m_nSxSDisplayCount(int(nOutputType>=0)+int(nDebugType>=0)+int(nInputType>=0))
        ,m_nComputeStages(nComputeStages)
        ,m_nOutputType(nOutputType)
        ,m_nDebugType(nDebugType)
        ,m_nInputType(nInputType)
        ,m_bUsingOutputPBOs(bUseOutputPBOs&&nOutputType>=0)
        ,m_bUsingDebugPBOs(bUseDebugPBOs&&nDebugType>=0)
        ,m_bUsingInputPBOs(bUseInputPBOs&&nInputType>=0)
        ,m_bUsingOutput(nOutputType>=0)
        ,m_bUsingDebug(nDebugType>=0)
        ,m_bUsingInput(nInputType>=0)
        ,m_bUsingTexArrays(bUseTexArrays&&nLevels==1)
        ,m_bUsingDisplay(bUseDisplay)
        ,m_bUsingTimers(bUseTimers)
        ,m_bUsingIntegralFormat(bUseIntegralFormat)
        ,m_vDefaultWorkGroupSize(GLUTILS_IMGPROC_DEFAULT_WORKGROUP)
        ,m_bGLInitialized(false)
        ,m_nInternalFrameIdx(-1)
        ,m_nLastOutputInternalIdx(-1)
        ,m_nLastDebugInternalIdx(-1)
        ,m_bFetchingOutput(false)
        ,m_bFetchingDebug(false)
        ,m_nNextLayer(1)
        ,m_nCurrLayer(0)
        ,m_nLastLayer(nLayers-1)
        ,m_nCurrPBO(0)
        ,m_nNextPBO(1)
        ,m_nAtomicBufferSize(0)
        ,m_nAtomicBufferRangeSize(0)
        ,m_nAtomicBufferOffsetVar(0)
        ,m_nCurrAtomicBufferOffset(0) {
    glAssert(m_nLevels>0 && m_nLayers>1);
    if(m_bUsingTexArrays && !glGetTextureSubImage && (m_bUsingDebugPBOs || m_bUsingOutputPBOs))
        glError("missing impl for texture arrays pbo fetch when glGetTextureSubImage is not available");
    int nMaxComputeInvocs;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&nMaxComputeInvocs);
    const int nCurrComputeStageInvocs = m_vDefaultWorkGroupSize.x*m_vDefaultWorkGroupSize.y;
    glAssert(nCurrComputeStageInvocs>0 && nCurrComputeStageInvocs<nMaxComputeInvocs);
    GLint nMaxImageUnits;
    glGetIntegerv(GL_MAX_IMAGE_UNITS,&nMaxImageUnits);
    if(nMaxImageUnits<GLImageProcAlgo::eImageBindingsCount)
        glError("image units limit is too small for the current impl");
    GLint nMaxTextureUnits;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS,&nMaxTextureUnits);
    if(nMaxTextureUnits<GLImageProcAlgo::eTextureBindingsCount)
        glError("texture units limit is too small for the current impl");
    if(m_bUsingTimers)
        glGenQueries(GLImageProcAlgo::eGLTimersCount,m_nGLTimers);
    glGenBuffers(1,&m_nAtomicBuffer);
    glGenBuffers(GLImageProcAlgo::eBufferBindingsCount,m_anSSBO);
}

GLImageProcAlgo::~GLImageProcAlgo() {
    if(m_bUsingTimers)
        glDeleteQueries(GLImageProcAlgo::eGLTimersCount,m_nGLTimers);
    glDeleteBuffers(1,&m_nAtomicBuffer);
    glDeleteBuffers(GLImageProcAlgo::eBufferBindingsCount,m_anSSBO);
}

std::string GLImageProcAlgo::getVertexShaderSource() const {
    return GLShader::getPassThroughVertexShaderSource(false,false,true);
}

std::string GLImageProcAlgo::getFragmentShaderSource() const {
    return getFragmentShaderSource_internal(m_nLayers,m_nOutputType,m_nDebugType,m_nInputType,m_bUsingOutput,m_bUsingDebug,m_bUsingInput,m_bUsingTexArrays,m_bUsingIntegralFormat);
}

void GLImageProcAlgo::initialize(const cv::Mat& oInitInput, const cv::Mat& oROI) {
    glAssert(oInitInput.type()==m_nInputType && oInitInput.size()==oROI.size() && oInitInput.isContinuous() && oROI.type()==CV_8UC1);
    m_bGLInitialized = false;
    m_oFrameSize = oROI.size();
    for(int nPBOIter=0; nPBOIter<2; ++nPBOIter) {
        if(m_bUsingOutputPBOs)
            m_apOutputPBOs[nPBOIter] = std::unique_ptr<GLPixelBufferObject>(new GLPixelBufferObject(cv::Mat(m_oFrameSize,m_nOutputType),GL_PIXEL_PACK_BUFFER,GL_STREAM_READ));
        if(m_bUsingDebugPBOs)
            m_apDebugPBOs[nPBOIter] = std::unique_ptr<GLPixelBufferObject>(new GLPixelBufferObject(cv::Mat(m_oFrameSize,m_nDebugType),GL_PIXEL_PACK_BUFFER,GL_STREAM_READ));
        if(m_bUsingInputPBOs)
            m_apInputPBOs[nPBOIter] = std::unique_ptr<GLPixelBufferObject>(new GLPixelBufferObject(oInitInput,GL_PIXEL_UNPACK_BUFFER,GL_STREAM_DRAW));
    }
    if(m_bUsingTexArrays) {
        if(m_bUsingOutput) {
            m_pOutputArray = std::unique_ptr<GLDynamicTexture2DArray>(new GLDynamicTexture2DArray(1,std::vector<cv::Mat>(m_nLayers,cv::Mat(m_oFrameSize,m_nOutputType)),m_bUsingIntegralFormat));
            m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
        }
        if(m_bUsingDebug) {
            m_pDebugArray = std::unique_ptr<GLDynamicTexture2DArray>(new GLDynamicTexture2DArray(1,std::vector<cv::Mat>(m_nLayers,cv::Mat(m_oFrameSize,m_nDebugType)),m_bUsingIntegralFormat));
            m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::eTexture_DebugBinding);
        }
        if(m_bUsingInput) {
            m_pInputArray = std::unique_ptr<GLDynamicTexture2DArray>(new GLDynamicTexture2DArray(1,std::vector<cv::Mat>(m_nLayers,cv::Mat(m_oFrameSize,m_nInputType)),m_bUsingIntegralFormat));
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
            if(m_bUsingInputPBOs) {
                m_pInputArray->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),m_nCurrLayer,true);
                m_pInputArray->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),m_nNextLayer,true);
            }
            else {
                m_pInputArray->updateTexture(oInitInput,m_nCurrLayer,true);
                m_pInputArray->updateTexture(oInitInput,m_nNextLayer,true);
            }
        }
    }
    else {
        m_vpOutputArray.clear();
        m_vpInputArray.clear();
        m_vpDebugArray.clear();
        for(int nLayerIter=0; nLayerIter<m_nLayers; ++nLayerIter) {
            if(m_bUsingOutput) {
                m_vpOutputArray.push_back(std::unique_ptr<GLDynamicTexture2D>(new GLDynamicTexture2D(1,cv::Mat(m_oFrameSize,m_nOutputType),m_bUsingIntegralFormat)));
                m_vpOutputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
            }
            if(m_bUsingDebug) {
                m_vpDebugArray.push_back(std::unique_ptr<GLDynamicTexture2D>(new GLDynamicTexture2D(1,cv::Mat(m_oFrameSize,m_nDebugType),m_bUsingIntegralFormat)));
                m_vpDebugArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_DebugBinding);
            }
            if(m_bUsingInput) {
                m_vpInputArray.push_back(std::unique_ptr<GLDynamicTexture2D>(new GLDynamicTexture2D(m_nLevels,cv::Mat(m_oFrameSize,m_nInputType),m_bUsingIntegralFormat)));
                m_vpInputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
                if(m_bUsingInputPBOs) {
                    if(nLayerIter==m_nCurrLayer)
                        m_vpInputArray[m_nCurrLayer]->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),true);
                    else if(nLayerIter==m_nNextLayer)
                        m_vpInputArray[m_nNextLayer]->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),true);
                }
                else {
                    if(nLayerIter==m_nCurrLayer)
                        m_vpInputArray[m_nCurrLayer]->updateTexture(oInitInput,true);
                    else if(nLayerIter==m_nNextLayer)
                        m_vpInputArray[m_nNextLayer]->updateTexture(oInitInput,true);
                }
            }
        }
    }
    m_pROITexture = std::unique_ptr<GLTexture2D>(new GLTexture2D(1,oROI,m_bUsingIntegralFormat));
    m_pROITexture->bindToImage(GLImageProcAlgo::eImage_ROIBinding,0,GL_READ_ONLY);
    if(!m_bUsingOutputPBOs && m_bUsingOutput)
        m_oLastOutput = cv::Mat(m_oFrameSize,m_nOutputType);
    if(!m_bUsingDebugPBOs && m_bUsingDebug)
        m_oLastDebug = cv::Mat(m_oFrameSize,m_nDebugType);
    m_vpImgProcShaders.clear();
    for(int nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        m_vpImgProcShaders.push_back(std::unique_ptr<GLShader>(new GLShader()));
        m_vpImgProcShaders.back()->addSource(getComputeShaderSource(nCurrStageIter),GL_COMPUTE_SHADER);
        if(!m_vpImgProcShaders.back()->link())
            glError("Could not link image processing shader");
    }
    m_oDisplayShader.clear();
    m_oDisplayShader.addSource(getVertexShaderSource(),GL_VERTEX_SHADER);
    m_oDisplayShader.addSource(getFragmentShaderSource(),GL_FRAGMENT_SHADER);
    if(!m_oDisplayShader.link())
        glError("Could not link display shader");
    glErrorCheck;
    m_nInternalFrameIdx = 0;
    m_bGLInitialized = true;
}

void GLImageProcAlgo::apply(const cv::Mat& oNextInput, bool bRebindAll) {
    glAssert(m_bGLInitialized && (oNextInput.empty() || (oNextInput.type()==m_nInputType && oNextInput.size()==m_oFrameSize && oNextInput.isContinuous())));
    m_nLastLayer = m_nCurrLayer;
    m_nCurrLayer = m_nNextLayer;
    ++m_nNextLayer %= m_nLayers;
    m_nCurrPBO = m_nNextPBO;
    ++m_nNextPBO %= 2;
    if(m_nAtomicBufferSize && (bRebindAll || m_nAtomicBufferOffsetVar)) {
        glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER,0,m_nAtomicBuffer,m_nCurrAtomicBufferOffset,m_nAtomicBufferRangeSize);
        m_nCurrAtomicBufferOffset += m_nAtomicBufferOffsetVar;
    }
    if(bRebindAll) {
        for(int nBufferBindingIter=0; nBufferBindingIter<eBufferBindingsCount; ++nBufferBindingIter) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,m_anSSBO[nBufferBindingIter]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,(GLImageProcAlgo::eBufferBindingList)nBufferBindingIter,m_anSSBO[nBufferBindingIter]);
        }
    }
    if(m_bUsingTimers)
        glBeginQuery(GL_TIME_ELAPSED,m_nGLTimers[GLImageProcAlgo::eGLTimer_TextureUpdate]);
    if(m_bUsingInputPBOs && !oNextInput.empty())
        m_apInputPBOs[m_nNextPBO]->updateBuffer(oNextInput,false,bRebindAll);
    if(m_bUsingTexArrays) {
        if(m_bUsingOutput) {
            if(bRebindAll)
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
            m_pOutputArray->bindToImage(GLImageProcAlgo::eImage_OutputBinding,0,m_nCurrLayer,GL_READ_WRITE);
        }
        if(m_bUsingDebug) {
            if(bRebindAll)
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::eTexture_DebugBinding);
            m_pDebugArray->bindToImage(GLImageProcAlgo::eImage_DebugBinding,0,m_nCurrLayer,GL_WRITE_ONLY);
        }
        if(m_bUsingInput) {
            if(bRebindAll || !m_bUsingInputPBOs) {
                m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
                if(!m_bUsingInputPBOs && !oNextInput.empty())
                    m_pInputArray->updateTexture(oNextInput,m_nNextLayer,bRebindAll);
            }
            m_pInputArray->bindToImage(GLImageProcAlgo::eImage_InputBinding,0,m_nCurrLayer,GL_READ_ONLY);
        }
    }
    else {
        for(int nLayerIter=0; nLayerIter<m_nLayers; ++nLayerIter) {
            if(bRebindAll) {
                if(m_bUsingOutput)
                    m_vpOutputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
                if(m_bUsingDebug)
                    m_vpDebugArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_DebugBinding);
                if(m_bUsingInput)
                    m_vpInputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
            }
            if(nLayerIter==m_nNextLayer && !m_bUsingInputPBOs) {
                if(!bRebindAll)
                    m_vpInputArray[m_nNextLayer]->bindToSampler((m_nNextLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
                if(!oNextInput.empty())
                    m_vpInputArray[m_nNextLayer]->updateTexture(oNextInput,bRebindAll);
            }
            else if(nLayerIter==m_nCurrLayer) {
                if(m_bUsingOutput)
                    m_vpOutputArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::eImage_OutputBinding,0,GL_READ_WRITE);
                if(m_bUsingDebug)
                    m_vpDebugArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::eImage_DebugBinding,0,GL_WRITE_ONLY);
                if(m_bUsingInput)
                    m_vpInputArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::eImage_InputBinding,0,GL_READ_ONLY);
            }
        }
    }
    if(bRebindAll)
        m_pROITexture->bindToImage(GLImageProcAlgo::eImage_ROIBinding,0,GL_READ_ONLY);
    if(m_bUsingTimers) {
        glEndQuery(GL_TIME_ELAPSED);
        glBeginQuery(GL_TIME_ELAPSED,m_nGLTimers[GLImageProcAlgo::eGLTimer_ComputeDispatch]);
    }
    for(int nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        m_vpImgProcShaders[nCurrStageIter]->activate();
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getCurrTextureLayerUniformName(),m_nCurrLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getLastTextureLayerUniformName(),m_nLastLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getFrameIndexUniformName(),m_nInternalFrameIdx);
        dispatch(nCurrStageIter,m_vpImgProcShaders[nCurrStageIter].get()); // add timer for stage? can reuse the same @@@@@@@@@@@@@@@
    }
    if(m_bUsingTimers)
        glEndQuery(GL_TIME_ELAPSED);
    if(m_bUsingInputPBOs) {
        if(m_bUsingTexArrays) {
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
            m_pInputArray->updateTexture(m_apInputPBOs[m_nNextPBO].get(),m_nNextLayer,bRebindAll);
        }
        else {
            m_vpInputArray[m_nNextLayer]->bindToSampler((m_nNextLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
            m_vpInputArray[m_nNextLayer]->updateTexture(m_apInputPBOs[m_nNextPBO].get(),bRebindAll);
        }
    }
    if(m_bFetchingDebug) {
        m_nLastDebugInternalIdx = m_nInternalFrameIdx;
        glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
        if(m_bUsingDebugPBOs) {
            if(m_bUsingTexArrays) {
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::eTexture_DebugBinding);
                m_pDebugArray->fetchTexture(m_apDebugPBOs[m_nNextPBO].get(),m_nCurrLayer,bRebindAll);
            }
            else {
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_DebugBinding);
                m_vpDebugArray[m_nCurrLayer]->fetchTexture(m_apDebugPBOs[m_nNextPBO].get(),bRebindAll);
            }
        }
        else {
            if(m_bUsingTexArrays) {
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::eTexture_DebugBinding);
                m_pDebugArray->fetchTexture(m_oLastDebug,m_nCurrLayer);
            }
            else {
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_DebugBinding);
                m_vpDebugArray[m_nCurrLayer]->fetchTexture(m_oLastDebug);
            }
        }
    }
    if(m_bFetchingOutput) {
        m_nLastOutputInternalIdx = m_nInternalFrameIdx;
        if(!m_bFetchingDebug)
            glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
        if(m_bUsingOutputPBOs) {
            if(m_bUsingTexArrays) {
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
                m_pOutputArray->fetchTexture(m_apOutputPBOs[m_nNextPBO].get(),m_nCurrLayer,bRebindAll);
            }
            else {
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
                m_vpOutputArray[m_nCurrLayer]->fetchTexture(m_apOutputPBOs[m_nNextPBO].get(),bRebindAll);
            }
        }
        else {
            if(m_bUsingTexArrays) {
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
                m_pOutputArray->fetchTexture(m_oLastOutput,m_nCurrLayer);
            }
            else {
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
                m_vpOutputArray[m_nCurrLayer]->fetchTexture(m_oLastOutput);
            }
        }
    }
    if(m_bUsingDisplay) {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        if(m_bUsingDebug) {
            if(m_bUsingTexArrays)
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::eTexture_DebugBinding);
            else
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_DebugBinding);
        }
        if(m_bUsingOutput) {
            if(m_bUsingTexArrays)
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
            else
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
        }
        if(m_bUsingTimers)
            glBeginQuery(GL_TIME_ELAPSED,m_nGLTimers[GLImageProcAlgo::eGLTimer_DisplayUpdate]);
        m_oDisplayShader.activate();
        m_oDisplayShader.setUniform1ui(getCurrTextureLayerUniformName(),m_nCurrLayer);
        m_oDisplayBillboard.render();
        if(m_bUsingTimers)
            glEndQuery(GL_TIME_ELAPSED);
    }
    if(m_bUsingTimers) {
        GLuint64 nGLTimerValTot = 0;
        std::cout << "\t\tGPU: ";
        glGetQueryObjectui64v(m_nGLTimers[GLImageProcAlgo::eGLTimer_TextureUpdate],GL_QUERY_RESULT,&m_nGLTimerVals[GLImageProcAlgo::eGLTimer_TextureUpdate]);
        std::cout << "TextureUpdate=" << m_nGLTimerVals[GLImageProcAlgo::eGLTimer_TextureUpdate]*1.e-6 << "ms,  ";
        nGLTimerValTot += m_nGLTimerVals[GLImageProcAlgo::eGLTimer_TextureUpdate];
        glGetQueryObjectui64v(m_nGLTimers[GLImageProcAlgo::eGLTimer_ComputeDispatch],GL_QUERY_RESULT,&m_nGLTimerVals[GLImageProcAlgo::eGLTimer_ComputeDispatch]);
        std::cout << "ComputeDispatch=" << m_nGLTimerVals[GLImageProcAlgo::eGLTimer_ComputeDispatch]*1.e-6 << "ms,  ";
        nGLTimerValTot += m_nGLTimerVals[GLImageProcAlgo::eGLTimer_ComputeDispatch];
        if(m_bUsingDisplay) {
            glGetQueryObjectui64v(m_nGLTimers[GLImageProcAlgo::eGLTimer_DisplayUpdate],GL_QUERY_RESULT,&m_nGLTimerVals[GLImageProcAlgo::eGLTimer_DisplayUpdate]);
            std::cout << "DisplayUpdate=" << m_nGLTimerVals[GLImageProcAlgo::eGLTimer_DisplayUpdate]*1.e-6 << "ms,  ";
            nGLTimerValTot += m_nGLTimerVals[GLImageProcAlgo::eGLTimer_DisplayUpdate];
        }
        std::cout << " tot=" << nGLTimerValTot*1.e-6 << "ms" << std::endl;
    }
    ++m_nInternalFrameIdx;
}

int GLImageProcAlgo::fetchLastOutput(cv::Mat& oOutput) const {
    glAssert(oOutput.size()==m_oFrameSize && oOutput.type()==m_nOutputType && oOutput.isContinuous());
    glAssert(m_bFetchingOutput);
    if(m_bUsingOutputPBOs)
        m_apOutputPBOs[m_nNextPBO]->fetchBuffer(oOutput,true);
    else
        m_oLastOutput.copyTo(oOutput);
    return m_nLastOutputInternalIdx;
}

int GLImageProcAlgo::fetchLastDebug(cv::Mat& oDebug) const {
    glAssert(oDebug.size()==m_oFrameSize && oDebug.type()==m_nDebugType && oDebug.isContinuous());
    glAssert(m_bFetchingDebug);
    if(m_bUsingDebugPBOs)
        m_apDebugPBOs[m_nNextPBO]->fetchBuffer(oDebug,true);
    else
        m_oLastDebug.copyTo(oDebug);
    return m_nLastDebugInternalIdx;
}

const char* GLImageProcAlgo::getCurrTextureLayerUniformName() {
    return "nCurrLayerIdx";
}

const char* GLImageProcAlgo::getLastTextureLayerUniformName() {
    return "nLastLayerIdx";
}

const char* GLImageProcAlgo::getFrameIndexUniformName() {
    return "nFrameIdx";
}

std::string GLImageProcAlgo::getFragmentShaderSource_internal( int nLayers, int nOutputType, int nDebugType, int nInputType,
                                                               bool bUsingOutput, bool bUsingDebug, bool bUsingInput,
                                                               bool bUsingTexArrays, bool bUsingIntegralFormat) {
    // @@@ replace else-if ladders by switch statements?
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n"
             "layout(location=" << GLVertex::eVertexAttrib_TexCoordIdx << ") in vec4 texCoord2D;\n";
    if(bUsingTexArrays) {
        if(bUsingOutput) ssSrc <<
             "layout(binding=" << GLImageProcAlgo::eTexture_OutputBinding << ") uniform" << (bUsingIntegralFormat?" u":" ") << "sampler2DArray texArrayOutput;\n";
        if(bUsingDebug) ssSrc <<
             "layout(binding=" << GLImageProcAlgo::eTexture_DebugBinding << ") uniform" << (bUsingIntegralFormat?" u":" ") << "sampler2DArray texArrayDebug;\n";
        if(bUsingInput) ssSrc <<
             "layout(binding=" << GLImageProcAlgo::eTexture_InputBinding << ") uniform" << (bUsingIntegralFormat?" u":" ") << "sampler2DArray texArrayInput;\n";
    }
    else
        for(int nLayerIter=0; nLayerIter<nLayers; ++nLayerIter) {
            if(bUsingOutput) ssSrc <<
             "layout(binding=" << (nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding << ") uniform" << (bUsingIntegralFormat?" u":" ") << "sampler2D texOutput" << nLayerIter << ";\n";
            if(bUsingDebug) ssSrc <<
             "layout(binding=" << (nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_DebugBinding << ") uniform" << (bUsingIntegralFormat?" u":" ") << "sampler2D texDebug" << nLayerIter << ";\n";
            if(bUsingInput) ssSrc <<
             "layout(binding=" << (nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding << ") uniform" << (bUsingIntegralFormat?" u":" ") << "sampler2D texInput" << nLayerIter << ";\n";
        }
    ssSrc << "uniform uint nCurrLayerIdx;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    float texCoord2DArrayIdx = 1;\n"
             "    vec3 texCoord3D = vec3(modf(texCoord2D.x*" << int(bUsingInput)+int(bUsingDebug)+int(bUsingOutput)<< ",texCoord2DArrayIdx),texCoord2D.y,texCoord2DArrayIdx);\n"
             "    vec2 texCoord2D_dPdx = dFdx(texCoord3D.xy);\n"
             "    vec2 texCoord2D_dPdy = dFdy(texCoord3D.xy);\n";
    if(bUsingTexArrays) {
        if(bUsingInput) { ssSrc <<
             "    if(texCoord2DArrayIdx==0) {\n";
            if(getChannelsFromMatType(nInputType)==1) ssSrc <<
             "        out_color = vec4(textureGrad(texArrayInput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (bUsingIntegralFormat?"255":"1") << ");\n";
            else ssSrc <<
             "        out_color = textureGrad(texArrayInput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy);\n";
            ssSrc <<
             "    }\n";
        }
        if(bUsingDebug) { ssSrc <<
             "    " << (bUsingInput?"else if":"if") << "(texCoord2DArrayIdx==" << int(bUsingInput) << ") {\n";
            if(getChannelsFromMatType(nDebugType)==1) ssSrc <<
             "        out_color = vec4(textureGrad(texArrayDebug,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (bUsingIntegralFormat?"255":"1") << ");\n";
            else ssSrc <<
             "        out_color = textureGrad(texArrayDebug,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy);\n";
            ssSrc <<
             "    }\n";
        }
        if(bUsingOutput) { ssSrc <<
             "    " << ((bUsingInput||bUsingDebug)?"else if":"if") << "(texCoord2DArrayIdx==" << int(bUsingInput)+int(bUsingDebug) << ") {\n";
            if(getChannelsFromMatType(nOutputType)==1) ssSrc <<
             "        out_color = vec4(textureGrad(texArrayOutput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (bUsingIntegralFormat?"255":"1") << ");\n";
            else ssSrc <<
             "        out_color = textureGrad(texArrayOutput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy);\n";
            ssSrc <<
             "    }\n";
        }
        ssSrc <<
             "    " << ((bUsingInput||bUsingDebug||bUsingOutput)?"else {":"{") << "\n"
             "        out_color = vec4(" << (bUsingIntegralFormat?"255":"1") << ");\n"
             "    }\n";
    }
    else {
        if(bUsingInput) { ssSrc <<
             "    if(texCoord2DArrayIdx==0) {\n";
            for(int nLayerIter=0; nLayerIter<nLayers; ++nLayerIter) { ssSrc <<
             "        " << ((nLayerIter>0)?"else if":"if") << "(nCurrLayerIdx==" << nLayerIter << ") {\n";
                if(getChannelsFromMatType(nInputType)==1) ssSrc <<
             "            out_color = vec4(textureGrad(texInput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (bUsingIntegralFormat?"255":"1") << ");\n";
                else ssSrc <<
             "            out_color = textureGrad(texInput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy);\n";
                ssSrc <<
             "        }\n";
            }
            ssSrc <<
             "        else {\n"
             "            out_color = vec4(" << (bUsingIntegralFormat?"255":"1") << ");\n"
             "        }\n"
             "    }\n";
        }
        if(bUsingDebug) { ssSrc <<
             "    " << (bUsingInput?"else if":"if") << "(texCoord2DArrayIdx==" << int(bUsingInput) << ") {\n";
            for(int nLayerIter=0; nLayerIter<nLayers; ++nLayerIter) { ssSrc <<
             "        " << ((nLayerIter>0)?"else if":"if") << "(nCurrLayerIdx==" << nLayerIter << ") {\n";
                if(getChannelsFromMatType(nDebugType)==1) ssSrc <<
             "            out_color = vec4(textureGrad(texDebug" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (bUsingIntegralFormat?"255":"1") << ");\n";
                else ssSrc <<
             "            out_color = textureGrad(texDebug" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy);\n";
                ssSrc <<
             "        }\n";
            }
            ssSrc <<
             "        else {\n"
             "            out_color = vec4(" << (bUsingIntegralFormat?"255":"1") << ");\n"
             "        }\n"
             "    }\n";
        }
        if(bUsingOutput) { ssSrc <<
             "    " << ((bUsingInput||bUsingDebug)?"else if":"if") << "(texCoord2DArrayIdx==" << int(bUsingInput)+int(bUsingDebug) << ") {\n";
            for(int nLayerIter=0; nLayerIter<nLayers; ++nLayerIter) { ssSrc <<
             "        " << ((nLayerIter>0)?"else if":"if") << "(nCurrLayerIdx==" << nLayerIter << ") {\n";
                if(getChannelsFromMatType(nOutputType)==1) ssSrc <<
             "            out_color = vec4(textureGrad(texOutput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (bUsingIntegralFormat?"255":"1") << ");\n";
                else ssSrc <<
             "            out_color = textureGrad(texOutput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy);\n";
                ssSrc <<
             "        }\n";
            }
            ssSrc <<
             "        else {\n"
             "            out_color = vec4(" << (bUsingIntegralFormat?"255":"1") << ");\n"
             "        }\n"
             "    }\n";
        }
        ssSrc <<
             "    " << ((bUsingInput||bUsingDebug||bUsingOutput)?"else {":"{") << "\n"
             "        out_color = vec4(" << (bUsingIntegralFormat?"255":"1") << ");\n"
             "    }\n";
    }
    if(bUsingIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

GLEvaluatorAlgo::GLEvaluatorAlgo(GLImageProcAlgo* pParent, int nComputeStages, int nOutputType, int nGroundtruthType, bool bUseDisplay)
    :    GLImageProcAlgo(1,pParent->m_nLayers,nComputeStages,nOutputType,-1,nGroundtruthType,pParent->m_bUsingOutputPBOs,false,pParent->m_bUsingInputPBOs,pParent->m_bUsingTexArrays,bUseDisplay,false,pParent->m_bUsingIntegralFormat)
        ,m_pParent(pParent) {
    glAssert(m_bUsingInput);
    glAssert(!dynamic_cast<GLEvaluatorAlgo*>(m_pParent));
}

GLEvaluatorAlgo::~GLEvaluatorAlgo() {}

std::string GLEvaluatorAlgo::getFragmentShaderSource() const {
    return getFragmentShaderSource_internal(m_nLayers,m_nOutputType,m_pParent->m_nDebugType,m_pParent->m_nInputType,m_bUsingOutput,m_pParent->m_bUsingDebug,m_bUsingInput,m_bUsingTexArrays,m_bUsingIntegralFormat);
}

void GLEvaluatorAlgo::initialize(const cv::Mat oInitInput, const cv::Mat& oROI, const cv::Mat& oInitGT) {
    glAssert(oInitGT.type()==m_nInputType && oInitGT.size()==oROI.size() && oInitGT.isContinuous() && oROI.type()==CV_8UC1);
    m_bGLInitialized = false;
    m_oFrameSize = oROI.size();
    m_pParent->GLImageProcAlgo::initialize(oInitInput,oROI);
    for(int nPBOIter=0; nPBOIter<2; ++nPBOIter) {
        if(m_bUsingOutputPBOs)
            m_apOutputPBOs[nPBOIter] = std::unique_ptr<GLPixelBufferObject>(new GLPixelBufferObject(cv::Mat(m_oFrameSize,m_nOutputType),GL_PIXEL_PACK_BUFFER,GL_STREAM_READ));
        if(m_bUsingInputPBOs)
            m_apInputPBOs[nPBOIter] = std::unique_ptr<GLPixelBufferObject>(new GLPixelBufferObject(oInitGT,GL_PIXEL_UNPACK_BUFFER,GL_STREAM_DRAW));
    }
    if(m_bUsingTexArrays) {
        if(m_bUsingOutput) {
            m_pOutputArray = std::unique_ptr<GLDynamicTexture2DArray>(new GLDynamicTexture2DArray(1,std::vector<cv::Mat>(m_nLayers,cv::Mat(m_oFrameSize,m_nOutputType)),m_bUsingIntegralFormat));
            m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
        }
        m_pParent->m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_EvalBinding);
        m_pInputArray = std::unique_ptr<GLDynamicTexture2DArray>(new GLDynamicTexture2DArray(1,std::vector<cv::Mat>(m_nLayers,cv::Mat(m_oFrameSize,m_nInputType)),m_bUsingIntegralFormat));
        m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
        if(m_bUsingInputPBOs) {
            m_pInputArray->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),m_nCurrLayer,true);
            m_pInputArray->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),m_nNextLayer,true);
        }
        else {
            m_pInputArray->updateTexture(oInitGT,m_nCurrLayer,true);
            m_pInputArray->updateTexture(oInitGT,m_nNextLayer,true);
        }
    }
    else {
        m_vpOutputArray.clear();
        m_vpInputArray.clear();
        for(int nLayerIter=0; nLayerIter<m_nLayers; ++nLayerIter) {
            if(m_bUsingOutput) {
                m_vpOutputArray.push_back(std::unique_ptr<GLDynamicTexture2D>(new GLDynamicTexture2D(1,cv::Mat(m_oFrameSize,m_nOutputType),m_bUsingIntegralFormat)));
                m_vpOutputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
            }
            m_pParent->m_vpOutputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_EvalBinding);
            m_vpInputArray.push_back(std::unique_ptr<GLDynamicTexture2D>(new GLDynamicTexture2D(m_nLevels,cv::Mat(m_oFrameSize,m_nInputType),m_bUsingIntegralFormat)));
            m_vpInputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
            if(m_bUsingInputPBOs) {
                if(nLayerIter==m_nCurrLayer)
                    m_vpInputArray[m_nCurrLayer]->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),true);
                else if(nLayerIter==m_nNextLayer)
                    m_vpInputArray[m_nNextLayer]->updateTexture(m_apInputPBOs[m_nCurrPBO].get(),true);
            }
            else {
                if(nLayerIter==m_nCurrLayer)
                    m_vpInputArray[m_nCurrLayer]->updateTexture(oInitGT,true);
                else if(nLayerIter==m_nNextLayer)
                    m_vpInputArray[m_nNextLayer]->updateTexture(oInitGT,true);
            }
        }
    }
    m_pROITexture = std::unique_ptr<GLTexture2D>(new GLTexture2D(1,oROI,m_bUsingIntegralFormat));
    m_pROITexture->bindToImage(GLImageProcAlgo::eImage_ROIBinding,0,GL_READ_ONLY);
    if(!m_bUsingOutputPBOs && m_bUsingOutput)
        m_oLastOutput = cv::Mat(m_oFrameSize,m_nOutputType);
    m_vpImgProcShaders.clear();
    for(int nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        m_vpImgProcShaders.push_back(std::unique_ptr<GLShader>(new GLShader()));
        m_vpImgProcShaders.back()->addSource(getComputeShaderSource(nCurrStageIter),GL_COMPUTE_SHADER);
        if(!m_vpImgProcShaders.back()->link())
            glError("Could not link image processing shader");
    }
    m_oDisplayShader.clear();
    m_oDisplayShader.addSource(getVertexShaderSource(),GL_VERTEX_SHADER);
    m_oDisplayShader.addSource(getFragmentShaderSource(),GL_FRAGMENT_SHADER);
    if(!m_oDisplayShader.link())
        glError("Could not link display shader");
    glErrorCheck;
    m_nInternalFrameIdx = 0;
    m_bGLInitialized = true;
}

void GLEvaluatorAlgo::apply(const cv::Mat oNextInput, const cv::Mat& oNextGT, bool bRebindAll) {
    glAssert(m_bGLInitialized && (oNextGT.empty() || (oNextGT.type()==m_nInputType && oNextGT.size()==m_oFrameSize && oNextGT.isContinuous())));
    m_pParent->GLImageProcAlgo::apply(oNextInput,bRebindAll);
    m_nLastLayer = m_nCurrLayer;
    m_nCurrLayer = m_nNextLayer;
    ++m_nNextLayer %= m_nLayers;
    m_nCurrPBO = m_nNextPBO;
    ++m_nNextPBO %= 2;
    if(m_nAtomicBufferSize && (bRebindAll || m_nAtomicBufferOffsetVar)) {
        glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER,0,m_nAtomicBuffer,m_nCurrAtomicBufferOffset,m_nAtomicBufferRangeSize);
        m_nCurrAtomicBufferOffset += m_nAtomicBufferOffsetVar;
    }
    if(bRebindAll) {
        for(int nBufferBindingIter=0; nBufferBindingIter<eBufferBindingsCount; ++nBufferBindingIter) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,m_anSSBO[nBufferBindingIter]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,(GLImageProcAlgo::eBufferBindingList)nBufferBindingIter,m_anSSBO[nBufferBindingIter]);
        }
    }
    if(m_bUsingInputPBOs && !oNextGT.empty())
        m_apInputPBOs[m_nNextPBO]->updateBuffer(oNextGT,false,bRebindAll);
    if(m_bUsingTexArrays) {
        if(m_bUsingOutput) {
            if(bRebindAll || (m_pParent->m_bFetchingOutput && (m_bUsingDisplay||m_bFetchingOutput)))
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
            m_pOutputArray->bindToImage(GLImageProcAlgo::eImage_OutputBinding,0,m_nCurrLayer,GL_READ_WRITE);
        }
        if(bRebindAll)
            m_pParent->m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_EvalBinding);
        if(!m_bUsingInputPBOs && !oNextGT.empty()) {
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
            m_pInputArray->updateTexture(oNextGT,m_nNextLayer,bRebindAll);
            if(m_bUsingDisplay || bRebindAll)
                m_pParent->m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
        }
        m_pInputArray->bindToImage(GLImageProcAlgo::eImage_InputBinding,0,m_nCurrLayer,GL_READ_ONLY);
        m_pParent->m_pOutputArray->bindToImage(GLImageProcAlgo::eImage_EvalBinding,0,m_pParent->m_nCurrLayer,GL_READ_ONLY);
    }
    else {
        for(int nLayerIter=0; nLayerIter<m_nLayers; ++nLayerIter) {
            if(bRebindAll || (m_pParent->m_bFetchingOutput && (m_bUsingDisplay||m_bFetchingOutput))) {
                if(m_bUsingOutput)
                    m_vpOutputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
                if(bRebindAll)
                    m_pParent->m_vpOutputArray[nLayerIter]->bindToSampler((nLayerIter*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_EvalBinding);
            }
            if(nLayerIter==m_nNextLayer && !m_bUsingInputPBOs && !oNextGT.empty()) {
                m_vpInputArray[m_nNextLayer]->bindToSampler((m_nNextLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
                m_vpInputArray[m_nNextLayer]->updateTexture(oNextGT,bRebindAll);
                if(m_bUsingDisplay || bRebindAll)
                    m_pParent->m_vpInputArray[m_nNextLayer]->bindToSampler((m_nNextLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
            }
            else if(nLayerIter==m_nCurrLayer) {
                if(m_bUsingOutput)
                    m_vpOutputArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::eImage_OutputBinding,0,GL_READ_WRITE);
                m_vpInputArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::eImage_InputBinding,0,GL_READ_ONLY);
            }
        }
        m_pParent->m_vpOutputArray[m_pParent->m_nCurrLayer]->bindToImage(GLImageProcAlgo::eImage_EvalBinding,0,GL_READ_ONLY);
    }
    if(bRebindAll)
        m_pROITexture->bindToImage(GLImageProcAlgo::eImage_ROIBinding,0,GL_READ_ONLY);
    for(int nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        m_vpImgProcShaders[nCurrStageIter]->activate();
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getCurrTextureLayerUniformName(),m_nCurrLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getLastTextureLayerUniformName(),m_nLastLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getFrameIndexUniformName(),m_nInternalFrameIdx);
        dispatch(nCurrStageIter,m_vpImgProcShaders[nCurrStageIter].get());
    }
    if(m_bUsingInputPBOs) {
        if(m_bUsingTexArrays) {
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
            m_pInputArray->updateTexture(m_apInputPBOs[m_nNextPBO].get(),m_nNextLayer,bRebindAll);
            if(m_bUsingDisplay || bRebindAll)
                m_pParent->m_pInputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_InputBinding);
        }
        else {
            m_vpInputArray[m_nNextLayer]->bindToSampler((m_nNextLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
            m_vpInputArray[m_nNextLayer]->updateTexture(m_apInputPBOs[m_nNextPBO].get(),bRebindAll);
            if(m_bUsingDisplay || bRebindAll)
                m_pParent->m_vpInputArray[m_nNextLayer]->bindToSampler((m_nNextLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_InputBinding);
        }
    }
    if(m_bFetchingOutput) {
        m_nLastOutputInternalIdx = m_nInternalFrameIdx;
        glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
        if(m_bUsingOutputPBOs) {
            if(m_bUsingTexArrays) {
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
                m_pOutputArray->fetchTexture(m_apOutputPBOs[m_nNextPBO].get(),m_nCurrLayer,bRebindAll);
            }
            else {
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
                m_vpOutputArray[m_nCurrLayer]->fetchTexture(m_apOutputPBOs[m_nNextPBO].get(),bRebindAll);
            }
        }
        else {
            if(m_bUsingTexArrays) {
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::eTexture_OutputBinding);
                m_pOutputArray->fetchTexture(m_oLastOutput,m_nCurrLayer);
            }
            else {
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((m_nCurrLayer*GLImageProcAlgo::eTextureBindingsCount)+GLImageProcAlgo::eTexture_OutputBinding);
                m_vpOutputArray[m_nCurrLayer]->fetchTexture(m_oLastOutput);
            }
        }
    }
    if(m_bUsingDisplay) {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        m_oDisplayShader.activate();
        m_oDisplayShader.setUniform1ui(getCurrTextureLayerUniformName(),m_nCurrLayer);
        m_oDisplayBillboard.render();
    }
    ++m_nInternalFrameIdx;
}

GLImagePassThroughAlgo::GLImagePassThroughAlgo( int nLayers, const cv::Mat& oExampleFrame, bool bUseOutputPBOs, bool bUseInputPBOs,
                                                bool bUseTexArrays, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat)
    :    GLImageProcAlgo(1,nLayers,1,oExampleFrame.type(),-1,oExampleFrame.type(),bUseOutputPBOs,false,bUseInputPBOs,bUseTexArrays,bUseDisplay,bUseTimers,bUseIntegralFormat) {
    glAssert(!oExampleFrame.empty());
    int nMaxComputeInvocs;
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&nMaxComputeInvocs);
    const int nCurrComputeStageInvocs = m_vDefaultWorkGroupSize.x*m_vDefaultWorkGroupSize.y;
    glAssert(nCurrComputeStageInvocs>0 && nCurrComputeStageInvocs<nMaxComputeInvocs);
}

std::string GLImagePassThroughAlgo::getComputeShaderSource(int nStage) const {
    glAssert(nStage>=0 && nStage<m_nComputeStages);
    return GLShader::getPassThroughComputeShaderSource_ImgLoadCopy(m_vDefaultWorkGroupSize,getInternalFormatFromMatType(m_nInputType,m_bUsingIntegralFormat),GLImageProcAlgo::eImage_InputBinding,GLImageProcAlgo::eImage_OutputBinding,m_bUsingIntegralFormat);
}

void GLImagePassThroughAlgo::dispatch(int nStage, GLShader*) {
    glAssert(nStage>=0 && nStage<m_nComputeStages);
    glDispatchCompute((GLuint)ceil((float)m_oFrameSize.width/m_vDefaultWorkGroupSize.x), (GLuint)ceil((float)m_oFrameSize.height/m_vDefaultWorkGroupSize.y), 1);
}
