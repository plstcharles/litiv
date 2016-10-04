
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

#include "litiv/utils/opengl-imgproc.hpp"

GLImageProcAlgo::GLImageProcAlgo( size_t nLevels, size_t nComputeStages, size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures,
                                  int nOutputType, int nDebugType, bool bUseInput, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
        m_nLevels(nLevels),
        m_nComputeStages(nComputeStages),
        m_nSSBOs(GLImageProcAlgo::nStorageBufferDefaultBindingsCount+nExtraSSBOs),
        m_nACBOs(GLImageProcAlgo::nAtomicCounterBufferDefaultBindingsCount+nExtraACBOs),
        m_nImages(GLImageProcAlgo::nImageDefaultBindingsCount+nExtraImages),
        m_nTextures(GLImageProcAlgo::nTextureDefaultBindingsCount+nExtraTextures),
        m_nSxSDisplayCount(size_t(nOutputType>=0)+size_t(nDebugType>=0)+size_t(bUseInput)),
        m_bUsingOutputPBOs(nOutputType>=0&&GLUTILS_IMGPROC_USE_DOUBLE_PBO_OUTPUT),
        m_bUsingDebugPBOs(nDebugType>=0&&GLUTILS_IMGPROC_USE_DOUBLE_PBO_OUTPUT),
        m_bUsingInputPBOs(bUseInput&&GLUTILS_IMGPROC_USE_DOUBLE_PBO_INPUT),
        m_bUsingOutput(nOutputType>=0),
        m_bUsingDebug(nDebugType>=0),
        m_bUsingInput(bUseInput),
        m_bUsingTexArrays(GLUTILS_IMGPROC_USE_TEXTURE_ARRAYS&&nLevels==1), /// && levels==1??? @@@@
        m_bUsingTimers(bUseTimers),
        m_bUsingIntegralFormat(bUseIntegralFormat),
        m_vDefaultWorkGroupSize(GLUTILS_IMGPROC_DEFAULT_WORKGROUP),
        m_bUsingDisplay(bUseDisplay),
        m_bGLInitialized(false),
        m_nInternalFrameIdx(size_t(-1)),
        m_nLastOutputInternalIdx(size_t(-1)),
        m_nLastDebugInternalIdx(size_t(-1)),
        m_bFetchingOutput(false),
        m_bFetchingDebug(false),
        m_nNextLayer(1),
        m_nCurrLayer(0),
        m_nLastLayer(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT-1),
        m_nCurrPBO(0),
        m_nNextPBO(1),
        m_nOutputType(nOutputType),
        m_nDebugType(nDebugType),
        m_nInputType(-1) {
    static_assert(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT>1,"texture arrays must have at least one layer each");
    lvAssert_(m_nLevels>0,"textures must have at least one level each");
    lvAssert_(m_nComputeStages>0,"image processing pipeline must have at least one compute stage");
    if(m_bUsingTexArrays && !glGetTextureSubImage && (m_bUsingDebugPBOs || m_bUsingOutputPBOs))
        lvError("missing impl for texture arrays pbo fetch when glGetTextureSubImage is not available");
    const std::array<int,3> anMaxWorkGroupSize = lv::gl::getIntegerVal<3>(GL_MAX_COMPUTE_WORK_GROUP_SIZE);
    if(anMaxWorkGroupSize[0]<(int)m_vDefaultWorkGroupSize.x || anMaxWorkGroupSize[1]<(int)m_vDefaultWorkGroupSize.y)
        lvError_("workgroup size limit is too small for the current impl (curr=[%d,%d], req=[%d,%d])",anMaxWorkGroupSize[0],anMaxWorkGroupSize[1],(int)m_vDefaultWorkGroupSize.x,(int)m_vDefaultWorkGroupSize.y);
    const size_t nCurrComputeStageInvocs = m_vDefaultWorkGroupSize.x*m_vDefaultWorkGroupSize.y;
    lvAssert_(nCurrComputeStageInvocs>0,"work group size dimensions must be non-null");
    if((size_t)lv::gl::getIntegerVal<1>(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS)<nCurrComputeStageInvocs)
        lvError_("compute invoc limit is too small for the current impl (curr=%lu, req=%lu)",(size_t)lv::gl::getIntegerVal<1>(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS),nCurrComputeStageInvocs);
    if((size_t)lv::gl::getIntegerVal<1>(GL_MAX_IMAGE_UNITS)<m_nImages || (size_t)lv::gl::getIntegerVal<1>(GL_MAX_COMPUTE_IMAGE_UNIFORMS)<m_nImages)
        lvError("image units limit is too small for the current impl");
    if((size_t)lv::gl::getIntegerVal<1>(GL_MAX_TEXTURE_UNITS)<m_nTextures)
        lvError("texture units limit is too small for the current impl");
    if((size_t)lv::gl::getIntegerVal<1>(GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS)<m_nSSBOs)
        lvError("ssbo bindings limit is too small for the current impl");
    if((size_t)lv::gl::getIntegerVal<1>(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)<m_nSSBOs)
        lvError("ssbo blocks limit is too small for the current impl");
    if((size_t)lv::gl::getIntegerVal<1>(GL_MAX_ATOMIC_COUNTER_BUFFER_BINDINGS)<m_nACBOs)
        lvError("atomic bo bindings limit is too small for the current impl");
    if(m_bUsingTimers)
        glGenQueries((GLsizei)m_nGLTimers.size(),m_nGLTimers.data());
    if(m_nSSBOs) {
        m_vnSSBO.resize(m_nSSBOs);
        glGenBuffers((GLsizei)m_nSSBOs,m_vnSSBO.data());
    }
    if(m_nACBOs) {
        m_vnACBO.resize(m_nACBOs);
        glGenBuffers((GLsizei)m_nACBOs,m_vnACBO.data());
    }
}

GLImageProcAlgo::~GLImageProcAlgo() {
    if(m_bUsingTimers)
        glDeleteQueries((GLsizei)m_nGLTimers.size(),m_nGLTimers.data());
    if(m_nACBOs)
        glDeleteBuffers((GLsizei)m_nACBOs,m_vnACBO.data());
    if(m_nSSBOs)
        glDeleteBuffers((GLsizei)m_nSSBOs,m_vnSSBO.data());
}

std::string GLImageProcAlgo::getVertexShaderSource() const {
    return GLShader::getVertexShaderSource_PassThrough(false,false,true);
}

std::string GLImageProcAlgo::getFragmentShaderSource() const {
    return getFragmentShaderSource_internal(m_nOutputType,m_nDebugType,m_nInputType);
}

void GLImageProcAlgo::initialize_gl(const cv::Mat& oInitInput, const cv::Mat& oROI) {
    m_bGLInitialized = false;
    lvAssert_(!oROI.empty() && oROI.isContinuous() && oROI.type()==CV_8UC1,"provided ROI must be non-empty, continuous, and of type 8UC1");
    if(m_bUsingInput) {
        lvAssert_(!oInitInput.empty() && oInitInput.size()==oROI.size() && oInitInput.isContinuous(),"provided input must be non-empty, continuous, and of the same size as the ROI");
        m_nInputType = oInitInput.type();
    }
    m_oFrameSize = oROI.size();
    const std::array<int,3> anMaxWorkGroupCount = lv::gl::getIntegerVal<3>(GL_MAX_COMPUTE_WORK_GROUP_COUNT);
    if(anMaxWorkGroupCount[0]<(int)ceil((float)m_oFrameSize.width/m_vDefaultWorkGroupSize.x) || anMaxWorkGroupCount[1]<(int)ceil((float)m_oFrameSize.height/m_vDefaultWorkGroupSize.y))
        lvError("workgroup count dispatch limit is too small for the current impl");
    for(size_t nPBOIter=0; nPBOIter<2; ++nPBOIter) {
        if(m_bUsingOutputPBOs)
            m_apOutputPBOs[nPBOIter] = std::make_unique<GLPixelBufferObject>(cv::Mat(m_oFrameSize,m_nOutputType),GL_PIXEL_PACK_BUFFER,GL_STREAM_READ);
        if(m_bUsingDebugPBOs)
            m_apDebugPBOs[nPBOIter] = std::make_unique<GLPixelBufferObject>(cv::Mat(m_oFrameSize,m_nDebugType),GL_PIXEL_PACK_BUFFER,GL_STREAM_READ);
        if(m_bUsingInputPBOs)
            m_apInputPBOs[nPBOIter] = std::make_unique<GLPixelBufferObject>(oInitInput,GL_PIXEL_UNPACK_BUFFER,GL_STREAM_DRAW);
    }
    if(m_bUsingTexArrays) {
        if(m_bUsingOutput) {
            m_pOutputArray = std::make_unique<GLDynamicTexture2DArray>(1,std::vector<cv::Mat>(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT,cv::Mat(m_oFrameSize,m_nOutputType)),m_bUsingIntegralFormat);
            m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::Texture_OutputBinding);
        }
        if(m_bUsingDebug) {
            m_pDebugArray = std::make_unique<GLDynamicTexture2DArray>(1,std::vector<cv::Mat>(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT,cv::Mat(m_oFrameSize,m_nDebugType)),m_bUsingIntegralFormat);
            m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
        }
        if(m_bUsingInput) {
            m_pInputArray = std::make_unique<GLDynamicTexture2DArray>(1,std::vector<cv::Mat>(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT,cv::Mat(m_oFrameSize,m_nInputType)),m_bUsingIntegralFormat);
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::Texture_InputBinding);
            if(m_bUsingInputPBOs) {
                m_pInputArray->updateTexture(*m_apInputPBOs[m_nCurrPBO],(int)m_nCurrLayer,true);
                m_pInputArray->updateTexture(*m_apInputPBOs[m_nCurrPBO],(int)m_nNextLayer,true);
            }
            else {
                m_pInputArray->updateTexture(oInitInput,(int)m_nCurrLayer,true);
                m_pInputArray->updateTexture(oInitInput,(int)m_nNextLayer,true);
            }
        }
    }
    else {
        m_vpOutputArray.resize(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT);
        m_vpInputArray.resize(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT);
        m_vpDebugArray.resize(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT);
        for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) {
            if(m_bUsingOutput) {
                m_vpOutputArray[nLayerIter] = std::make_unique<GLDynamicTexture2D>(1,cv::Mat(m_oFrameSize,m_nOutputType),m_bUsingIntegralFormat);
                m_vpOutputArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_OutputBinding));
            }
            if(m_bUsingDebug) {
                m_vpDebugArray[nLayerIter] = std::make_unique<GLDynamicTexture2D>(1,cv::Mat(m_oFrameSize,m_nDebugType),m_bUsingIntegralFormat);
                m_vpDebugArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_DebugBinding));
            }
            if(m_bUsingInput) {
                m_vpInputArray[nLayerIter] = std::make_unique<GLDynamicTexture2D>((GLsizei)m_nLevels,cv::Mat(m_oFrameSize,m_nInputType),m_bUsingIntegralFormat);
                m_vpInputArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_InputBinding));
                if(m_bUsingInputPBOs) {
                    if(nLayerIter==m_nCurrLayer)
                        m_vpInputArray[m_nCurrLayer]->updateTexture(*m_apInputPBOs[m_nCurrPBO],true);
                    else if(nLayerIter==m_nNextLayer)
                        m_vpInputArray[m_nNextLayer]->updateTexture(*m_apInputPBOs[m_nCurrPBO],true);
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
    m_pROITexture = std::make_unique<GLTexture2D>(1,oROI,m_bUsingIntegralFormat);
    m_pROITexture->bindToImage(GLImageProcAlgo::Image_ROIBinding,0,GL_READ_ONLY);
    if(!m_bUsingOutputPBOs && m_bUsingOutput)
        m_oLastOutput = cv::Mat(m_oFrameSize,m_nOutputType);
    if(!m_bUsingDebugPBOs && m_bUsingDebug)
        m_oLastDebug = cv::Mat(m_oFrameSize,m_nDebugType);
    m_vpImgProcShaders.resize(m_nComputeStages);
    for(size_t nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        m_vpImgProcShaders[nCurrStageIter] = std::make_unique<GLShader>();
        m_vpImgProcShaders[nCurrStageIter]->addSource(getComputeShaderSource(nCurrStageIter),GL_COMPUTE_SHADER);
        if(!m_vpImgProcShaders[nCurrStageIter]->link())
            lvError("Could not link image processing shader");
    }
    m_oDisplayShader.clear();
    m_oDisplayShader.addSource(this->getVertexShaderSource(),GL_VERTEX_SHADER);
    m_oDisplayShader.addSource(this->getFragmentShaderSource(),GL_FRAGMENT_SHADER);
    if(!m_oDisplayShader.link())
        lvError("Could not link display shader");
    glErrorCheck;
    m_nInternalFrameIdx = 0;
    m_bGLInitialized = true;
}

void GLImageProcAlgo::apply_gl(const cv::Mat& oNextInput, bool bRebindAll) {
    lvAssert_(m_bGLInitialized,"algo must be initialized first");
    lvAssert_(oNextInput.empty() || (oNextInput.type()==m_nInputType && oNextInput.size()==m_oFrameSize && oNextInput.isContinuous()),"input must be the same size/type as initially provided, and continuous");
    m_nLastLayer = m_nCurrLayer;
    m_nCurrLayer = m_nNextLayer;
    ++m_nNextLayer %= GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT;
    m_nCurrPBO = m_nNextPBO;
    ++m_nNextPBO %= 2;
    if(bRebindAll) {
        for(size_t nSSBOIter=GLImageProcAlgo::nStorageBufferDefaultBindingsCount; nSSBOIter<m_nSSBOs; ++nSSBOIter)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,(GLuint)nSSBOIter,m_vnSSBO[nSSBOIter]);
        for(size_t nACBOIter=GLImageProcAlgo::nAtomicCounterBufferDefaultBindingsCount; nACBOIter<m_nACBOs; ++nACBOIter)
            glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER,(GLuint)nACBOIter,m_vnACBO[nACBOIter]);
        // rebind extra images, textures? @@@@@
    }
    if(m_bUsingTimers)
        glBeginQuery(GL_TIME_ELAPSED,m_nGLTimers[GLImageProcAlgo::GLTimer_TextureUpdate]);
    if(m_bUsingInputPBOs && !oNextInput.empty())
        m_apInputPBOs[m_nNextPBO]->updateBuffer(oNextInput,false,bRebindAll);
    if(m_bUsingTexArrays) {
        if(m_bUsingOutput) {
            if(bRebindAll)
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::Texture_OutputBinding);
            m_pOutputArray->bindToImage(GLImageProcAlgo::Image_OutputBinding,0,(int)m_nCurrLayer,GL_READ_WRITE);
        }
        if(m_bUsingDebug) {
            if(bRebindAll)
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
            m_pDebugArray->bindToImage(GLImageProcAlgo::Image_DebugBinding,0,(int)m_nCurrLayer,GL_WRITE_ONLY);
        }
        if(m_bUsingInput) {
            if(bRebindAll || !m_bUsingInputPBOs) {
                m_pInputArray->bindToSamplerArray(GLImageProcAlgo::Texture_InputBinding);
                if(!m_bUsingInputPBOs && !oNextInput.empty())
                    m_pInputArray->updateTexture(oNextInput,(int)m_nNextLayer,bRebindAll);
            }
            m_pInputArray->bindToImage(GLImageProcAlgo::Image_InputBinding,0,(int)m_nCurrLayer,GL_READ_ONLY);
        }
    }
    else {
        for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) {
            if(bRebindAll) {
                if(m_bUsingOutput)
                    m_vpOutputArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_OutputBinding));
                if(m_bUsingDebug)
                    m_vpDebugArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_DebugBinding));
                if(m_bUsingInput)
                    m_vpInputArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_InputBinding));
            }
            if(nLayerIter==m_nNextLayer && !m_bUsingInputPBOs) {
                if(!bRebindAll)
                    m_vpInputArray[m_nNextLayer]->bindToSampler((GLuint)getTextureBinding(m_nNextLayer,GLImageProcAlgo::Texture_InputBinding));
                if(!oNextInput.empty())
                    m_vpInputArray[m_nNextLayer]->updateTexture(oNextInput,bRebindAll);
            }
            else if(nLayerIter==m_nCurrLayer) {
                if(m_bUsingOutput)
                    m_vpOutputArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::Image_OutputBinding,0,GL_READ_WRITE);
                if(m_bUsingDebug)
                    m_vpDebugArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::Image_DebugBinding,0,GL_WRITE_ONLY);
                if(m_bUsingInput)
                    m_vpInputArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::Image_InputBinding,0,GL_READ_ONLY);
            }
        }
    }
    if(bRebindAll)
        m_pROITexture->bindToImage(GLImageProcAlgo::Image_ROIBinding,0,GL_READ_ONLY);
    if(m_bUsingTimers) {
        glEndQuery(GL_TIME_ELAPSED);
        glBeginQuery(GL_TIME_ELAPSED,m_nGLTimers[GLImageProcAlgo::GLTimer_ComputeDispatch]);
    }
    for(size_t nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        lvAssert(m_vpImgProcShaders[nCurrStageIter]->activate());
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getCurrTextureLayerUniformName(),(GLuint)m_nCurrLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getLastTextureLayerUniformName(),(GLuint)m_nLastLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getFrameIndexUniformName(),(GLuint)m_nInternalFrameIdx);
        dispatch(nCurrStageIter,*m_vpImgProcShaders[nCurrStageIter]); // add timer for stage? can reuse the same @@@@
    }
    if(m_bUsingTimers)
        glEndQuery(GL_TIME_ELAPSED);
    if(m_bUsingInputPBOs) {
        if(m_bUsingTexArrays) {
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::Texture_InputBinding);
            m_pInputArray->updateTexture(*m_apInputPBOs[m_nNextPBO],(int)m_nNextLayer,bRebindAll);
        }
        else {
            m_vpInputArray[m_nNextLayer]->bindToSampler((GLuint)getTextureBinding(m_nNextLayer,GLImageProcAlgo::Texture_InputBinding));
            m_vpInputArray[m_nNextLayer]->updateTexture(*m_apInputPBOs[m_nNextPBO],bRebindAll);
        }
    }
    if(m_bFetchingDebug) {
        m_nLastDebugInternalIdx = m_nInternalFrameIdx;
        glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
        if(m_bUsingDebugPBOs) {
            if(m_bUsingTexArrays) {
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
                m_pDebugArray->fetchTexture(*m_apDebugPBOs[m_nNextPBO],(int)m_nCurrLayer,bRebindAll);
            }
            else {
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_DebugBinding));
                m_vpDebugArray[m_nCurrLayer]->fetchTexture(*m_apDebugPBOs[m_nNextPBO],bRebindAll);
            }
        }
        else {
            if(m_bUsingTexArrays) {
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
                m_pDebugArray->fetchTexture(m_oLastDebug,(int)m_nCurrLayer);
            }
            else {
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_DebugBinding));
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
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::Texture_OutputBinding);
                m_pOutputArray->fetchTexture(*m_apOutputPBOs[m_nNextPBO],(int)m_nCurrLayer,bRebindAll);
            }
            else {
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_OutputBinding));
                m_vpOutputArray[m_nCurrLayer]->fetchTexture(*m_apOutputPBOs[m_nNextPBO],bRebindAll);
            }
        }
        else {
            if(m_bUsingTexArrays) {
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::Texture_OutputBinding);
                m_pOutputArray->fetchTexture(m_oLastOutput,(int)m_nCurrLayer);
            }
            else {
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_OutputBinding));
                m_vpOutputArray[m_nCurrLayer]->fetchTexture(m_oLastOutput);
            }
        }
    }
    if(m_bUsingDisplay) {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        if(m_bUsingDebug) {
            if(m_bUsingTexArrays)
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
            else
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_DebugBinding));
        }
        if(m_bUsingOutput) {
            if(m_bUsingTexArrays)
                m_pOutputArray->bindToSamplerArray(GLImageProcAlgo::Texture_OutputBinding);
            else
                m_vpOutputArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_OutputBinding));
        }
        if(m_bUsingTimers)
            glBeginQuery(GL_TIME_ELAPSED,m_nGLTimers[GLImageProcAlgo::GLTimer_DisplayUpdate]);
        lvAssert(m_oDisplayShader.activate());
        m_oDisplayShader.setUniform1ui(getCurrTextureLayerUniformName(),(GLuint)m_nCurrLayer);
        m_oDisplayBillboard.render();
        if(m_bUsingTimers)
            glEndQuery(GL_TIME_ELAPSED);
    }
    if(m_bUsingTimers) {
        GLuint64 nGLTimerValTot = 0;
        std::cout << "\t\tGPU: ";
        glGetQueryObjectui64v(m_nGLTimers[GLImageProcAlgo::GLTimer_TextureUpdate],GL_QUERY_RESULT,&m_nGLTimerVals[GLImageProcAlgo::GLTimer_TextureUpdate]);
        std::cout << "TextureUpdate=" << m_nGLTimerVals[GLImageProcAlgo::GLTimer_TextureUpdate]*1.e-6 << "ms,  ";
        nGLTimerValTot += m_nGLTimerVals[GLImageProcAlgo::GLTimer_TextureUpdate];
        glGetQueryObjectui64v(m_nGLTimers[GLImageProcAlgo::GLTimer_ComputeDispatch],GL_QUERY_RESULT,&m_nGLTimerVals[GLImageProcAlgo::GLTimer_ComputeDispatch]);
        std::cout << "ComputeDispatch=" << m_nGLTimerVals[GLImageProcAlgo::GLTimer_ComputeDispatch]*1.e-6 << "ms,  ";
        nGLTimerValTot += m_nGLTimerVals[GLImageProcAlgo::GLTimer_ComputeDispatch];
        if(m_bUsingDisplay) {
            glGetQueryObjectui64v(m_nGLTimers[GLImageProcAlgo::GLTimer_DisplayUpdate],GL_QUERY_RESULT,&m_nGLTimerVals[GLImageProcAlgo::GLTimer_DisplayUpdate]);
            std::cout << "DisplayUpdate=" << m_nGLTimerVals[GLImageProcAlgo::GLTimer_DisplayUpdate]*1.e-6 << "ms,  ";
            nGLTimerValTot += m_nGLTimerVals[GLImageProcAlgo::GLTimer_DisplayUpdate];
        }
        std::cout << " tot=" << nGLTimerValTot*1.e-6 << "ms" << std::endl;
    }
    ++m_nInternalFrameIdx;
}

size_t GLImageProcAlgo::fetchLastOutput(cv::Mat& oOutput) const {
    lvAssert_(m_bFetchingOutput,"algo is not configured for cpu-side output mat fetching");
    oOutput.create(m_oFrameSize,m_nOutputType);
    if(m_bUsingOutputPBOs)
        m_apOutputPBOs[m_nNextPBO]->fetchBuffer(oOutput,true);
    else
        m_oLastOutput.copyTo(oOutput);
    return m_nLastOutputInternalIdx;
}

size_t GLImageProcAlgo::fetchLastDebug(cv::Mat& oDebug) const {
    lvAssert_(m_bFetchingDebug,"algo is not configured for cpu-side debug mat fetching");
    oDebug.create(m_oFrameSize,m_nDebugType);
    if(m_bUsingDebugPBOs)
        m_apDebugPBOs[m_nNextPBO]->fetchBuffer(oDebug,true);
    else
        m_oLastDebug.copyTo(oDebug);
    return m_nLastDebugInternalIdx;
}

void GLImageProcAlgo::dispatch(size_t nStage, GLShader&) {
    lvAssert_(nStage<m_nComputeStages,"required compute stage does not exist");
    glDispatchCompute((GLuint)ceil((float)m_oFrameSize.width/m_vDefaultWorkGroupSize.x),(GLuint)ceil((float)m_oFrameSize.height/m_vDefaultWorkGroupSize.y),1);
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

std::string GLImageProcAlgo::getFragmentShaderSource_internal(int nOutputType, int nDebugType, int nInputType) const {
    // @@@ replace else-if ladders by switch statements?
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(location=0) out vec4 out_color;\n"
             "layout(location=" << GLVertex::VertexAttrib_TexCoordIdx << ") in vec4 texCoord2D;\n";
    if(m_bUsingTexArrays) {
        if(nOutputType>=0) ssSrc <<
             "layout(binding=" << GLImageProcAlgo::Texture_OutputBinding << ") uniform" << (m_bUsingIntegralFormat?" u":" ") << "sampler2DArray texArrayOutput;\n";
        if(nDebugType>=0) ssSrc <<
             "layout(binding=" << GLImageProcAlgo::Texture_DebugBinding << ") uniform" << (m_bUsingIntegralFormat?" u":" ") << "sampler2DArray texArrayDebug;\n";
        if(nInputType>=0) ssSrc <<
             "layout(binding=" << GLImageProcAlgo::Texture_InputBinding << ") uniform" << (m_bUsingIntegralFormat?" u":" ") << "sampler2DArray texArrayInput;\n";
    }
    else
        for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) {
            if(nOutputType>=0) ssSrc <<
             "layout(binding=" << getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_OutputBinding) << ") uniform" << (m_bUsingIntegralFormat?" u":" ") << "sampler2D texOutput" << nLayerIter << ";\n";
            if(nDebugType>=0) ssSrc <<
             "layout(binding=" << getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_DebugBinding) << ") uniform" << (m_bUsingIntegralFormat?" u":" ") << "sampler2D texDebug" << nLayerIter << ";\n";
            if(nInputType>=0) ssSrc <<
             "layout(binding=" << getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_InputBinding) << ") uniform" << (m_bUsingIntegralFormat?" u":" ") << "sampler2D texInput" << nLayerIter << ";\n";
        }
    ssSrc << "uniform uint nCurrLayerIdx;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    float texCoord2DArrayIdx = 1;\n"
             "    vec3 texCoord3D = vec3(modf(texCoord2D.x*" << int(nOutputType>=0)+int(nDebugType>=0)+int(nInputType>=0)<< ",texCoord2DArrayIdx),texCoord2D.y,texCoord2DArrayIdx);\n"
             "    vec2 texCoord2D_dPdx = dFdx(texCoord3D.xy);\n"
             "    vec2 texCoord2D_dPdy = dFdy(texCoord3D.xy);\n";
    if(m_bUsingTexArrays) {
        if(nInputType>=0) { ssSrc <<
             "    if(texCoord2DArrayIdx==0) {\n";
            if(lv::gl::getChannelsFromMatType(nInputType)==1) ssSrc <<
             "        out_color = vec4(textureGrad(texArrayInput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (m_bUsingIntegralFormat?"255":"1") << ");\n";
            else ssSrc <<
             "        out_color = textureGrad(texArrayInput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy);\n";
            ssSrc <<
             "    }\n";
        }
        if(nDebugType>=0) { ssSrc <<
             "    " << (nInputType>=0?"else if":"if") << "(texCoord2DArrayIdx==" << int(nInputType>=0) << ") {\n";
            if(lv::gl::getChannelsFromMatType(nDebugType)==1) ssSrc <<
             "        out_color = vec4(textureGrad(texArrayDebug,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (m_bUsingIntegralFormat?"255":"1") << ");\n";
            else ssSrc <<
             "        out_color = textureGrad(texArrayDebug,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy);\n";
            ssSrc <<
             "    }\n";
        }
        if(nOutputType>=0) { ssSrc <<
             "    " << ((nInputType>=0||nDebugType>=0)?"else if":"if") << "(texCoord2DArrayIdx==" << int(nInputType>=0)+int(nDebugType>=0) << ") {\n";
            if(lv::gl::getChannelsFromMatType(nOutputType)==1) ssSrc <<
             "        out_color = vec4(textureGrad(texArrayOutput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (m_bUsingIntegralFormat?"255":"1") << ");\n";
            else ssSrc <<
             "        out_color = textureGrad(texArrayOutput,vec3(texCoord3D.xy,nCurrLayerIdx),texCoord2D_dPdx,texCoord2D_dPdy);\n";
            ssSrc <<
             "    }\n";
        }
        ssSrc <<
             "    " << ((nInputType>=0||nDebugType>=0||nOutputType>=0)?"else {":"{") << "\n"
             "        out_color = vec4(" << (m_bUsingIntegralFormat?"255":"1") << ");\n"
             "    }\n";
    }
    else {
        if(nInputType>=0) { ssSrc <<
             "    if(texCoord2DArrayIdx==0) {\n";
            for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) { ssSrc <<
             "        " << ((nLayerIter>0)?"else if":"if") << "(nCurrLayerIdx==" << nLayerIter << ") {\n";
                if(lv::gl::getChannelsFromMatType(nInputType)==1) ssSrc <<
             "            out_color = vec4(textureGrad(texInput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (m_bUsingIntegralFormat?"255":"1") << ");\n";
                else ssSrc <<
             "            out_color = textureGrad(texInput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy);\n";
                ssSrc <<
             "        }\n";
            }
            ssSrc <<
             "        else {\n"
             "            out_color = vec4(" << (m_bUsingIntegralFormat?"255":"1") << ");\n"
             "        }\n"
             "    }\n";
        }
        if(nDebugType>=0) { ssSrc <<
             "    " << (nInputType>=0?"else if":"if") << "(texCoord2DArrayIdx==" << int(nInputType>=0) << ") {\n";
            for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) { ssSrc <<
             "        " << ((nLayerIter>0)?"else if":"if") << "(nCurrLayerIdx==" << nLayerIter << ") {\n";
                if(lv::gl::getChannelsFromMatType(nDebugType)==1) ssSrc <<
             "            out_color = vec4(textureGrad(texDebug" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (m_bUsingIntegralFormat?"255":"1") << ");\n";
                else ssSrc <<
             "            out_color = textureGrad(texDebug" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy);\n";
                ssSrc <<
             "        }\n";
            }
            ssSrc <<
             "        else {\n"
             "            out_color = vec4(" << (m_bUsingIntegralFormat?"255":"1") << ");\n"
             "        }\n"
             "    }\n";
        }
        if(nOutputType>=0) { ssSrc <<
             "    " << ((nInputType>=0||nDebugType>=0)?"else if":"if") << "(texCoord2DArrayIdx==" << int(nInputType>=0)+int(nDebugType>=0) << ") {\n";
            for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) { ssSrc <<
             "        " << ((nLayerIter>0)?"else if":"if") << "(nCurrLayerIdx==" << nLayerIter << ") {\n";
                if(lv::gl::getChannelsFromMatType(nOutputType)==1) ssSrc <<
             "            out_color = vec4(textureGrad(texOutput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy).xxx," << (m_bUsingIntegralFormat?"255":"1") << ");\n";
                else ssSrc <<
             "            out_color = textureGrad(texOutput" << nLayerIter << ",texCoord3D.xy,texCoord2D_dPdx,texCoord2D_dPdy);\n";
                ssSrc <<
             "        }\n";
            }
            ssSrc <<
             "        else {\n"
             "            out_color = vec4(" << (m_bUsingIntegralFormat?"255":"1") << ");\n"
             "        }\n"
             "    }\n";
        }
        ssSrc <<
             "    " << ((nInputType>=0||nDebugType>=0||nOutputType>=0)?"else {":"{") << "\n"
             "        out_color = vec4(" << (m_bUsingIntegralFormat?"255":"1") << ");\n"
             "    }\n";
    }
    if(m_bUsingIntegralFormat) ssSrc <<
             "    out_color = out_color/255;\n";
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

GLImageProcEvaluatorAlgo::GLImageProcEvaluatorAlgo( const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount, size_t nCountersPerFrame,
                                                    int nDebugType, int nGroundtruthType, bool bUseIntegralFormat) :
        // note: using extra buffers/images/textures would force rebinding for each iterations due to diamond dependency over enum lists
        GLImageProcAlgo(1,1,0,0,0,0,-1,nDebugType,true,pParent->m_bUsingDisplay,false,bUseIntegralFormat),
        m_nGroundtruthType(nGroundtruthType),
        m_nTotFrameCount(nTotFrameCount),
        m_nEvalBufferFrameSize(nCountersPerFrame*4),
        m_nEvalBufferTotSize(nTotFrameCount*nCountersPerFrame*4),
        m_nEvalBufferMaxSize((size_t)lv::gl::getIntegerVal<1>(GL_MAX_ATOMIC_COUNTER_BUFFER_SIZE)),
        m_nCurrEvalBufferSize(m_nEvalBufferTotSize),
        m_nCurrEvalBufferOffsetPtr(0),
        m_nCurrEvalBufferOffsetBlock(0),
        m_pParent(pParent) {
    lvAssert_(m_bUsingInput && m_pParent->m_bUsingOutput,"gpu evaluator must use an input, and the parent must provide an output");
    lvAssert_(m_nGroundtruthType>=0 && m_nGroundtruthType==m_pParent->m_nOutputType,"gpu evaluator gt type must be the same as the parent's output type"); // @@@ relax constraint later?
    lvAssert_(!dynamic_cast<GLImageProcEvaluatorAlgo*>(m_pParent.get()),"parent algo must not be an evaluator algo");
    lvAssert_(nTotFrameCount>0 && nCountersPerFrame>0,"total frame count & counters per frame must be known and positive");
    lvAssert_(m_nEvalBufferMaxSize>nCountersPerFrame*4,"evaluation max buffer size is too small for required counters");
    if(m_nEvalBufferMaxSize<=m_nCurrEvalBufferSize) {
        while(m_nEvalBufferMaxSize<=m_nCurrEvalBufferSize)
            m_nCurrEvalBufferSize /= 2;
        m_nCurrEvalBufferSize -= (m_nCurrEvalBufferSize%(nCountersPerFrame*4));
        std::cout << "\tWarning: atomic counter buffer size limit (" << m_nEvalBufferMaxSize/1024 << "kb) is smaller than required (" << m_nEvalBufferTotSize/1024 << "kb), will use " << m_nCurrEvalBufferSize/1024 << "kb instead, performance might be affected" << std::endl;
    }
    m_pParent->m_bUsingDisplay = false;
}

GLImageProcEvaluatorAlgo::~GLImageProcEvaluatorAlgo() {}

const cv::Mat& GLImageProcEvaluatorAlgo::getEvaluationAtomicCounterBuffer() {
    lvAssert_(m_bGLInitialized,"algo must be initialized first");
    glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER,getACBOId(GLImageProcAlgo::AtomicCounterBuffer_EvalBinding));
    if(m_nCurrEvalBufferOffsetPtr)
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER,0,m_nCurrEvalBufferOffsetPtr,m_oEvalQueryBuffer.data+m_nCurrEvalBufferOffsetBlock);
    return m_oEvalQueryBuffer;
}

std::string GLImageProcEvaluatorAlgo::getFragmentShaderSource() const {
    return GLImageProcAlgo::getFragmentShaderSource_internal(-1,m_nDebugType,m_pParent->m_nInputType);
}

void GLImageProcEvaluatorAlgo::initialize_gl(const cv::Mat& oInitInput, const cv::Mat& oInitGT, const cv::Mat& oROI) {
    lvAssert_(oROI.empty() || (oROI.size()==oInitInput.size() && oROI.size()==oInitGT.size()),"roi size mismatch w/ input+gt");
    m_pParent->initialize_gl(oInitInput,oROI);
    this->initialize_gl(oInitGT,oROI);
}

void GLImageProcEvaluatorAlgo::initialize_gl(const cv::Mat& oInitGT, const cv::Mat& oROI) {
    lvAssert_(!oROI.empty() && oROI.isContinuous() && oROI.type()==CV_8UC1,"provided ROI must be non-empty, continuous, and of type 8UC1");
    lvAssert_(oROI.size()==m_pParent->m_oFrameSize,"provided ROI size must match parent algo frame size");
    lvAssert_(oInitGT.type()==m_nGroundtruthType && oInitGT.size()==oROI.size() && oInitGT.isContinuous(),"provided init gt mat must match original type/size, and be continuous");
    m_bGLInitialized = false;
    m_oFrameSize = oROI.size();
    for(size_t nPBOIter=0; nPBOIter<2; ++nPBOIter) {
        if(m_bUsingDebugPBOs)
            m_apDebugPBOs[nPBOIter] = std::make_unique<GLPixelBufferObject>(cv::Mat(m_oFrameSize,m_nDebugType),GL_PIXEL_PACK_BUFFER,GL_STREAM_READ);
        if(m_bUsingInputPBOs)
            m_apInputPBOs[nPBOIter] = std::make_unique<GLPixelBufferObject>(oInitGT,GL_PIXEL_UNPACK_BUFFER,GL_STREAM_DRAW);
    }
    if(m_bUsingTexArrays) {
        if(m_bUsingDebug) {
            m_pDebugArray = std::make_unique<GLDynamicTexture2DArray>(1,std::vector<cv::Mat>(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT,cv::Mat(m_oFrameSize,m_nDebugType)),m_bUsingIntegralFormat);
            m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
        }
        m_pInputArray = std::make_unique<GLDynamicTexture2DArray>(1,std::vector<cv::Mat>(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT,cv::Mat(m_oFrameSize,m_nGroundtruthType)),m_bUsingIntegralFormat);
        m_pInputArray->bindToSamplerArray(GLImageProcAlgo::Texture_GTBinding);
        if(m_bUsingInputPBOs) {
            m_pInputArray->updateTexture(*m_apInputPBOs[m_nCurrPBO],(int)m_nCurrLayer,true);
            m_pInputArray->updateTexture(*m_apInputPBOs[m_nCurrPBO],(int)m_nNextLayer,true);
        }
        else {
            m_pInputArray->updateTexture(oInitGT,(int)m_nCurrLayer,true);
            m_pInputArray->updateTexture(oInitGT,(int)m_nNextLayer,true);
        }
    }
    else {
        m_vpDebugArray.resize(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT);
        m_vpInputArray.resize(GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT);
        for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) {
            if(m_bUsingDebug) {
                m_vpDebugArray[nLayerIter] = std::make_unique<GLDynamicTexture2D>(1,cv::Mat(m_oFrameSize,m_nDebugType),m_bUsingIntegralFormat);
                m_vpDebugArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_DebugBinding));
            }
            m_vpInputArray[nLayerIter] = std::make_unique<GLDynamicTexture2D>((GLsizei)m_nLevels,cv::Mat(m_oFrameSize,m_nGroundtruthType),m_bUsingIntegralFormat);
            m_vpInputArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_GTBinding));
            if(m_bUsingInputPBOs) {
                if(nLayerIter==m_nCurrLayer)
                    m_vpInputArray[m_nCurrLayer]->updateTexture(*m_apInputPBOs[m_nCurrPBO],true);
                else if(nLayerIter==m_nNextLayer)
                    m_vpInputArray[m_nNextLayer]->updateTexture(*m_apInputPBOs[m_nCurrPBO],true);
            }
            else {
                if(nLayerIter==m_nCurrLayer)
                    m_vpInputArray[m_nCurrLayer]->updateTexture(oInitGT,true);
                else if(nLayerIter==m_nNextLayer)
                    m_vpInputArray[m_nNextLayer]->updateTexture(oInitGT,true);
            }
        }
    }
    m_pROITexture = std::make_unique<GLTexture2D>(1,oROI,m_bUsingIntegralFormat);
    m_pROITexture->bindToImage(GLImageProcAlgo::Image_ROIBinding,0,GL_READ_ONLY);
    if(!m_bUsingDebugPBOs && m_bUsingDebug)
        m_oLastDebug = cv::Mat(m_oFrameSize,m_nDebugType);
    m_vpImgProcShaders.resize(m_nComputeStages);
    for(size_t nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        m_vpImgProcShaders[nCurrStageIter] = std::make_unique<GLShader>();
        m_vpImgProcShaders[nCurrStageIter]->addSource(getComputeShaderSource(nCurrStageIter),GL_COMPUTE_SHADER);
        if(!m_vpImgProcShaders[nCurrStageIter]->link())
            lvError("Could not link image processing shader");
    }
    m_oDisplayShader.clear();
    m_oDisplayShader.addSource(this->getVertexShaderSource(),GL_VERTEX_SHADER);
    m_oDisplayShader.addSource(this->getFragmentShaderSource(),GL_FRAGMENT_SHADER);
    if(!m_oDisplayShader.link())
        lvError("Could not link display shader");
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER,getACBOId(GLImageProcAlgo::AtomicCounterBuffer_EvalBinding));
    glBufferData(GL_ATOMIC_COUNTER_BUFFER,m_nCurrEvalBufferSize,NULL,GL_DYNAMIC_READ);
    glClearBufferData(GL_ATOMIC_COUNTER_BUFFER,GL_R32UI,GL_RED_INTEGER,GL_INT,NULL);
    /*GLuint* pAtomicCountersPtr = (GLuint*)glMapBufferRange(GL_ATOMIC_COUNTER_BUFFER,0,m_nCurrEvalBufferSize,GL_MAP_WRITE_BIT|GL_MAP_INVALIDATE_BUFFER_BIT|GL_MAP_UNSYNCHRONIZED_BIT);
    if(!pAtomicCountersPtr)
        lvError("Could not init atomic counters");
    memset(pAtomicCountersPtr,0,m_nCurrEvalBufferSize);
    glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);*/
    m_nCurrEvalBufferOffsetPtr = 0;
    m_nCurrEvalBufferOffsetBlock = 0;
    m_oEvalQueryBuffer.create((int)m_nTotFrameCount,(int)m_nEvalBufferFrameSize/4,CV_32SC1);
    m_oEvalQueryBuffer = cv::Scalar_<int>(0);
    glErrorCheck;
    m_nInternalFrameIdx = 0;
    m_bGLInitialized = true;
}

void GLImageProcEvaluatorAlgo::apply_gl(const cv::Mat& oNextInput, const cv::Mat& oNextGT, bool bRebindAll) {
    m_pParent->apply_gl(oNextInput,bRebindAll);
    this->apply_gl(oNextGT,bRebindAll);
}

void GLImageProcEvaluatorAlgo::apply_gl(const cv::Mat& oNextGT, bool bRebindAll) {
    lvAssert_(m_bGLInitialized,"algo must be initialized first");
    lvAssert_(oNextGT.empty() || (oNextGT.type()==m_nGroundtruthType && oNextGT.size()==m_oFrameSize && oNextGT.isContinuous()),"input gt must be the same size/type as initially provided, and continuous");
    lvAssert_(m_nInternalFrameIdx<m_nTotFrameCount,"internal frame counter exceeded the sequence's total frame count");
    m_nLastLayer = m_nCurrLayer;
    m_nCurrLayer = m_nNextLayer;
    ++m_nNextLayer %= GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT;
    m_nCurrPBO = m_nNextPBO;
    ++m_nNextPBO %= 2;
    if(m_nCurrEvalBufferOffsetPtr+m_nEvalBufferFrameSize>m_nCurrEvalBufferSize) {
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER,getACBOId(GLImageProcAlgo::AtomicCounterBuffer_EvalBinding));
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER,0,m_nCurrEvalBufferSize,m_oEvalQueryBuffer.data+m_nCurrEvalBufferOffsetBlock);
        glClearBufferData(GL_ATOMIC_COUNTER_BUFFER,GL_R32UI,GL_RED_INTEGER,GL_INT,NULL);
        m_nCurrEvalBufferOffsetBlock += m_nCurrEvalBufferSize;
        m_nCurrEvalBufferOffsetPtr = 0;
    }
    glBindBufferRange(GL_ATOMIC_COUNTER_BUFFER,GLImageProcAlgo::AtomicCounterBuffer_EvalBinding,getACBOId(GLImageProcAlgo::AtomicCounterBuffer_EvalBinding),m_nCurrEvalBufferOffsetPtr,m_nEvalBufferFrameSize);
    m_nCurrEvalBufferOffsetPtr += m_nEvalBufferFrameSize;
    if(m_bUsingInputPBOs && !oNextGT.empty())
        m_apInputPBOs[m_nNextPBO]->updateBuffer(oNextGT,false,bRebindAll);
    if(m_bUsingTexArrays) {
        if(m_bUsingDebug) {
            if(bRebindAll || (m_pParent->m_bFetchingDebug && (m_bUsingDisplay||m_bFetchingDebug)))
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
            m_pDebugArray->bindToImage(GLImageProcAlgo::Image_DebugBinding,0,(int)m_nCurrLayer,GL_READ_WRITE);
        }
        if(!m_bUsingInputPBOs && !oNextGT.empty()) {
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::Texture_GTBinding);
            m_pInputArray->updateTexture(oNextGT,(int)m_nNextLayer,bRebindAll);
        }
        m_pInputArray->bindToImage(GLImageProcAlgo::Image_GTBinding,0,(int)m_nCurrLayer,GL_READ_ONLY);
        m_pParent->m_pOutputArray->bindToImage(GLImageProcAlgo::Image_OutputBinding,0,(int)m_pParent->m_nCurrLayer,GL_READ_ONLY);
    }
    else {
        for(size_t nLayerIter=0; nLayerIter<GLUTILS_IMGPROC_DEFAULT_LAYER_COUNT; ++nLayerIter) {
            if(bRebindAll || (m_pParent->m_bFetchingDebug && (m_bUsingDisplay||m_bFetchingDebug))) {
                if(m_bUsingDebug)
                    m_vpDebugArray[nLayerIter]->bindToSampler((GLuint)getTextureBinding(nLayerIter,GLImageProcAlgo::Texture_DebugBinding));
            }
            if(nLayerIter==m_nNextLayer && !m_bUsingInputPBOs && !oNextGT.empty()) {
                m_vpInputArray[m_nNextLayer]->bindToSampler((GLuint)getTextureBinding(m_nNextLayer,GLImageProcAlgo::Texture_GTBinding));
                m_vpInputArray[m_nNextLayer]->updateTexture(oNextGT,bRebindAll);
            }
            else if(nLayerIter==m_nCurrLayer) {
                if(m_bUsingDebug)
                    m_vpDebugArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::Image_DebugBinding,0,GL_READ_WRITE);
                m_vpInputArray[m_nCurrLayer]->bindToImage(GLImageProcAlgo::Image_GTBinding,0,GL_READ_ONLY);
            }
        }
        m_pParent->m_vpOutputArray[m_pParent->m_nCurrLayer]->bindToImage(GLImageProcAlgo::Image_OutputBinding,0,GL_READ_ONLY);
    }
    m_pROITexture->bindToImage(GLImageProcAlgo::Image_ROIBinding,0,GL_READ_ONLY);
    for(size_t nCurrStageIter=0; nCurrStageIter<m_nComputeStages; ++nCurrStageIter) {
        lvAssert(m_vpImgProcShaders[nCurrStageIter]->activate());
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getCurrTextureLayerUniformName(),(GLuint)m_nCurrLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getLastTextureLayerUniformName(),(GLuint)m_nLastLayer);
        m_vpImgProcShaders[nCurrStageIter]->setUniform1ui(getFrameIndexUniformName(),(GLuint)m_nInternalFrameIdx);
        dispatch(nCurrStageIter,*m_vpImgProcShaders[nCurrStageIter]);
    }
    if(m_bUsingInputPBOs) {
        if(m_bUsingTexArrays) {
            m_pInputArray->bindToSamplerArray(GLImageProcAlgo::Texture_GTBinding);
            m_pInputArray->updateTexture(*m_apInputPBOs[m_nNextPBO],(int)m_nNextLayer,bRebindAll);
        }
        else {
            m_vpInputArray[m_nNextLayer]->bindToSampler((GLuint)getTextureBinding(m_nNextLayer,GLImageProcAlgo::Texture_GTBinding));
            m_vpInputArray[m_nNextLayer]->updateTexture(*m_apInputPBOs[m_nNextPBO],bRebindAll);
        }
    }
    if(m_bFetchingDebug) {
        m_nLastDebugInternalIdx = m_nInternalFrameIdx;
        glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
        if(m_bUsingDebugPBOs) {
            if(m_bUsingTexArrays) {
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
                m_pDebugArray->fetchTexture(*m_apDebugPBOs[m_nNextPBO],(int)m_nCurrLayer,bRebindAll);
            }
            else {
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_DebugBinding));
                m_vpDebugArray[m_nCurrLayer]->fetchTexture(*m_apDebugPBOs[m_nNextPBO],bRebindAll);
            }
        }
        else {
            if(m_bUsingTexArrays) {
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
                m_pDebugArray->fetchTexture(m_oLastDebug,(int)m_nCurrLayer);
            }
            else {
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_DebugBinding));
                m_vpDebugArray[m_nCurrLayer]->fetchTexture(m_oLastDebug);
            }
        }
    }
    if(m_bUsingDisplay) {
        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        if(m_bUsingDebug) {
            if(m_bUsingTexArrays)
                m_pDebugArray->bindToSamplerArray(GLImageProcAlgo::Texture_DebugBinding);
            else
                m_vpDebugArray[m_nCurrLayer]->bindToSampler((GLuint)getTextureBinding(m_nCurrLayer,GLImageProcAlgo::Texture_DebugBinding));
        }
        lvAssert(m_oDisplayShader.activate());
        m_oDisplayShader.setUniform1ui(getCurrTextureLayerUniformName(),(GLuint)m_nCurrLayer);
        m_oDisplayBillboard.render();
    }
    m_pParent->m_pROITexture->bindToImage(GLImageProcAlgo::Image_ROIBinding,0,GL_READ_ONLY);
    ++m_nInternalFrameIdx;
}

GLImagePassThroughAlgo::GLImagePassThroughAlgo(int nFrameType, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
        GLImageProcAlgo(1,1,0,0,0,0,nFrameType,-1,true,bUseDisplay,bUseTimers,bUseIntegralFormat) {
    lvAssert_(nFrameType>=0,"must provide a valid frame type for pass-through usage");
}

std::string GLImagePassThroughAlgo::getComputeShaderSource(size_t nStage) const {
    lvAssert_(nStage<m_nComputeStages,"required compute stage does not exist");
    return GLShader::getComputeShaderSource_PassThrough_ImgLoadCopy(m_vDefaultWorkGroupSize,lv::gl::getInternalFormatFromMatType(m_nOutputType,m_bUsingIntegralFormat),GLImageProcAlgo::Image_InputBinding,GLImageProcAlgo::Image_OutputBinding,m_bUsingIntegralFormat);
}

/* @@@@ CLEANUP REQUIRED, NEED TO RE-VALIDATE WITH NEW PROJ STRUCTURE
const size_t BinaryMedianFilter::m_nPPSMaxRowSize = 512;
const size_t BinaryMedianFilter::m_nTransposeBlockSize = 32;
const GLenum BinaryMedianFilter::Image_PPSAccumulator = GLImageProcAlgo::Image_CustomBinding1;
const GLenum BinaryMedianFilter::Image_PPSAccumulator_T = GLImageProcAlgo::Image_CustomBinding2;

BinaryMedianFilter::BinaryMedianFilter( size_t nKernelSize, size_t nBorderSize, const cv::Mat& oROI,
                                        bool bUseOutputPBOs, bool bUseInputPBOs, bool bUseTexArrays,
                                        bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat) :
        GLImageProcAlgo(1,4+bool(oROI.cols>m_nPPSMaxRowSize)+bool(oROI.rows>m_nPPSMaxRowSize),CV_8UC1,-1,CV_8UC1,bUseOutputPBOs,false,bUseInputPBOs,bUseTexArrays,bUseDisplay,bUseTimers,bUseIntegralFormat),
        m_nKernelSize(nKernelSize),
        m_nBorderSize(nBorderSize) {
    lvAssert((m_nKernelSize%2)==1 && m_nKernelSize>1 && m_nKernelSize<m_oFrameSize.width && m_nKernelSize<m_oFrameSize.height);
    lvAssert(m_nBorderSize<(m_oFrameSize.width-m_nKernelSize) && m_nBorderSize<(m_oFrameSize.height-m_nKernelSize));
    int nMaxComputeInvocs;@@@@ recheck for new workgroup sizes?
    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,&nMaxComputeInvocs);
    const size_t nCurrComputeStageInvocs = m_vDefaultWorkGroupSize.x*m_vDefaultWorkGroupSize.y;
    lvAssert(nCurrComputeStageInvocs>0 && nCurrComputeStageInvocs<nMaxComputeInvocs);
    lvAssert(m_nTransposeBlockSize*m_nTransposeBlockSize>0 && m_nTransposeBlockSize*m_nTransposeBlockSize<nMaxComputeInvocs);
    int nMaxWorkGroupCount_X, nMaxWorkGroupCount_Y;
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,0,&nMaxWorkGroupCount_X);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT,1,&nMaxWorkGroupCount_Y);
    lvAssert(m_oFrameSize.width<nMaxWorkGroupCount_X && m_oFrameSize.width<nMaxWorkGroupCount_Y);
    lvAssert(m_oFrameSize.height<nMaxWorkGroupCount_X && m_oFrameSize.height<nMaxWorkGroupCount_Y);
    const GLenum eInputInternalFormat = getIntegralFormatFromInternalFormat(getInternalFormatFromMatType(m_nInputType,m_bUsingIntegralFormat));
    const GLenum eAccumInternalFormat = getIntegralFormatFromInternalFormat(getInternalFormatFromMatType(CV_32SC1));
    if(m_oFrameSize.width>m_nPPSMaxRowSize) {
        m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_ParallelPrefixSum(m_nPPSMaxRowSize,true,eInputInternalFormat,GLImageProcAlgo::Image_InputBinding,BinaryMedianFilter::Image_PPSAccumulator));
        m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.width/m_nPPSMaxRowSize),m_oFrameSize.height,1));
        m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_ParallelPrefixSum_BlockMerge(m_oFrameSize.width,m_nPPSMaxRowSize,m_oFrameSize.height,getInternalFormatFromMatType(CV_32SC1),BinaryMedianFilter::Image_PPSAccumulator));
        m_vvComputeShaderDispatchSizes.push_back(glm::uvec3(1,1,1));
        m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_Transpose(m_nTransposeBlockSize,eAccumInternalFormat,BinaryMedianFilter::Image_PPSAccumulator,BinaryMedianFilter::Image_PPSAccumulator_T));
        m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.width/m_nTransposeBlockSize), (GLuint)ceil((float)m_oFrameSize.height/m_nTransposeBlockSize), 1));
        m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_ParallelPrefixSum(m_nPPSMaxRowSize,false,getInternalFormatFromMatType(CV_32SC1),BinaryMedianFilter::Image_PPSAccumulator_T,BinaryMedianFilter::Image_PPSAccumulator_T));
        m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.height/m_nPPSMaxRowSize),m_oFrameSize.width,1));
        if(m_oFrameSize.height>m_nPPSMaxRowSize) {
            m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_ParallelPrefixSum_BlockMerge(m_oFrameSize.height,m_nPPSMaxRowSize,m_oFrameSize.width,getInternalFormatFromMatType(CV_32SC1),BinaryMedianFilter::Image_PPSAccumulator_T));
            m_vvComputeShaderDispatchSizes.push_back(glm::uvec3(1,1,1));
        }
        //m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_Transpose(m_nTransposeBlockSize,getInternalFormatFromMatType(CV_32SC1),BinaryMedianFilter::Image_PPSAccumulator_T,ImageProcShaderAlgo::Image_OutputBinding));
        //m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.height/m_nTransposeBlockSize),(GLuint)ceil((float)m_oFrameSize.width/m_nTransposeBlockSize),1));
    }
    else {
        m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_ParallelPrefixSum(m_nPPSMaxRowSize,true,eInputInternalFormat,GLImageProcAlgo::Image_InputBinding,BinaryMedianFilter::Image_PPSAccumulator));
        m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.width/m_nPPSMaxRowSize),m_oFrameSize.height,1));
        m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_Transpose(m_nTransposeBlockSize,getInternalFormatFromMatType(CV_32SC1),BinaryMedianFilter::Image_PPSAccumulator,BinaryMedianFilter::Image_PPSAccumulator_T));
        m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.width/m_nTransposeBlockSize),(GLuint)ceil((float)m_oFrameSize.height/m_nTransposeBlockSize),1));
        m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_ParallelPrefixSum(m_nPPSMaxRowSize,false,getInternalFormatFromMatType(CV_32SC1),BinaryMedianFilter::Image_PPSAccumulator_T,BinaryMedianFilter::Image_PPSAccumulator_T));
        m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.height/m_nPPSMaxRowSize),m_oFrameSize.width,1));
        //m_vsComputeShaderSources.push_back(GLShader::getComputeShaderSource_Transpose(m_nTransposeBlockSize,getInternalFormatFromMatType(CV_32SC1),BinaryMedianFilter::Image_PPSAccumulator_T,ImageProcShaderAlgo::Image_OutputBinding));
        //m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.height/m_nTransposeBlockSize),(GLuint)ceil((float)m_oFrameSize.width/m_nTransposeBlockSize),1));
    }
    // area lookup with clamped coords & final output write
    const char* acAccumInternalFormatName = getGLSLFormatNameFromInternalFormat(eAccumInternalFormat);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "#define radius " << m_nKernelSize/2 << "\n"
             "#define imgWidth " << m_oFrameSize.width << "\n"
             "#define imgHeight " << m_oFrameSize.height << "\n"
             "#define halfkernelarea " << (m_nKernelSize*m_nKernelSize)/2 << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
             "layout(binding=" << BinaryMedianFilter::Image_PPSAccumulator_T << ", " << acAccumInternalFormatName << ") readonly uniform uimage2D imgInput;\n"
             "layout(binding=" << GLImageProcAlgo::Image_OutputBinding << ") writeonly uniform uimage2D imgOutput;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
            // @@@@@@@@@@ could also benefit from shared mem, fetch area sums in and around local workgroup
             "    ivec2 center = ivec2(gl_GlobalInvocationID.xy);\n"
             "    // area sum = D - C - B + A\n"
             "    ivec2 D = min(center+ivec2(radius),ivec2(imgWidth-1,imgHeight-1));\n"
             "    ivec2 C = ivec2(max(center.x-radius,0),min(center.y+radius,imgHeight-1));\n"
             "    ivec2 B = ivec2(min(center.x+radius,imgWidth-1),max(center.y-radius,0));\n"
             "    ivec2 A = max(center-ivec2(radius),ivec2(0));\n"
             "    uint areasum = imageLoad(imgInput,D.yx).r-imageLoad(imgInput,C.yx).r-imageLoad(imgInput,B.yx).r+imageLoad(imgInput,A.yx).r;\n"
             "    imageStore(imgOutput,center,uvec4(255*uint(areasum>halfkernelarea)));\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    m_vsComputeShaderSources.push_back(ssSrc.str());
    m_vvComputeShaderDispatchSizes.push_back(glm::uvec3((GLuint)ceil((float)m_oFrameSize.width/m_vDefaultWorkGroupSize.x),(GLuint)ceil((float)m_oFrameSize.height/m_vDefaultWorkGroupSize.y),1));
    lvAssert((int)m_vsComputeShaderSources.size()==m_nComputeStages && (int)m_vvComputeShaderDispatchSizes.size()==m_nComputeStages);
}

std::string BinaryMedianFilter::getComputeShaderSource(size_t nStage) const {
    // @@@@ go check how opencv handles borders (sets as 0...?)
    lvAssert_(nStage<m_nComputeStages,"required compute stage does not exist");
    return m_vsComputeShaderSources[nStage];
}

void BinaryMedianFilter::dispatchCompute(size_t nStage, GLShader*) {
    lvAssert_(nStage<m_nComputeStages,"required compute stage does not exist");
    if(nCurrStageIter>0)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT); // add barrier for acbo? ssbo? @@@@@
    glDispatchCompute(m_vvComputeShaderDispatchSizes[nStage].x,m_vvComputeShaderDispatchSizes[nStage].y,m_vvComputeShaderDispatchSizes[nStage].z);
}

void BinaryMedianFilter::postInitialize(const cv::Mat&) {
    m_apCustomTextures[0] = std::unique_ptr<GLTexture2D>(new GLTexture2D(1,cv::Mat(m_oFrameSize,CV_32SC1),m_bUsingIntegralFormat));
    m_apCustomTextures[0]->bindToImage(BinaryMedianFilter::Image_PPSAccumulator,0,GL_READ_WRITE);
    m_apCustomTextures[1] = std::unique_ptr<GLTexture2D>(new GLTexture2D(1,cv::Mat(cv::Size(m_oFrameSize.height,m_oFrameSize.width),CV_32SC1),m_bUsingIntegralFormat));
    m_apCustomTextures[1]->bindToImage(BinaryMedianFilter::Image_PPSAccumulator_T,0,GL_READ_WRITE);
    glErrorCheck;
}

void BinaryMedianFilter::preProcess(bool bRebindAll) {
    if(bRebindAll) {
        m_apCustomTextures[0]->bindToImage(BinaryMedianFilter::Image_PPSAccumulator,0,GL_READ_WRITE);
        m_apCustomTextures[1]->bindToImage(BinaryMedianFilter::Image_PPSAccumulator_T,0,GL_READ_WRITE);
        glErrorCheck;
    }
}
*/
