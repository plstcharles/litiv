
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

#include "litiv/video/BackgroundSubtractorLOBSTER.hpp"

template<>
IBackgroundSubtractorLOBSTER::IBackgroundSubtractorLOBSTER_(size_t nDescDistThreshold, size_t nColorDistThreshold, size_t nBGSamples,
                                                            size_t nRequiredBGSamples, size_t nLBSPThresholdOffset, float fRelLBSPThreshold) :
        IBackgroundSubtractorLBSP(fRelLBSPThreshold,nLBSPThresholdOffset),
        m_nColorDistThreshold(nColorDistThreshold),
        m_nDescDistThreshold(nDescDistThreshold),
        m_nBGSamples(nBGSamples),
        m_nRequiredBGSamples(nRequiredBGSamples) {
    lvAssert_(m_nBGSamples>0 && m_nRequiredBGSamples<=m_nBGSamples,"algo cannot require more sample matches than sample count in model");
    lvAssert_(m_nColorDistThreshold>0 || m_nDescDistThreshold>0,"distance thresholds must be positive values");
}

#if HAVE_GLSL
template<>
IBackgroundSubtractorLOBSTER_GLSL::IBackgroundSubtractorLOBSTER_(size_t nDescDistThreshold, size_t nColorDistThreshold, size_t nBGSamples,
                                                                 size_t nRequiredBGSamples, size_t nLBSPThresholdOffset, float fRelLBSPThreshold) :
        IBackgroundSubtractorLBSP_GLSL(1,1+BGSLOBSTER_GLSL_USE_POSTPROC,2,0,0,0,BGSLOBSTER_GLSL_USE_DEBUG?CV_8UC4:-1,BGSLOBSTER_GLSL_USE_DEBUG,BGSLOBSTER_GLSL_USE_TIMERS,true,fRelLBSPThreshold,nLBSPThresholdOffset),
        m_nColorDistThreshold(nColorDistThreshold),
        m_nDescDistThreshold(nDescDistThreshold),
        m_nBGSamples(nBGSamples),
        m_nRequiredBGSamples(nRequiredBGSamples) {
    lvAssert_(m_nRequiredBGSamples<=m_nBGSamples,"algo cannot require more sample matches than sample count in model");
    lvAssert_(m_nColorDistThreshold>0 || m_nDescDistThreshold>0,"distance thresholds must be positive values");
    glErrorCheck;
}

void BackgroundSubtractorLOBSTER_GLSL::refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate) {
    lvDbgExceptionWatch;
    // == refresh
    lvAssert_(m_bInitialized,"algo must be initialized first");
    lvAssert_(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f,"model refresh must be given as a non-null fraction");
    const size_t nModelSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
    const size_t nRefreshSampleStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
    if(!bForceFGUpdate)
        getLatestForegroundMask(m_oLastFGMask);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,getSSBOId(BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_BGModelBinding));
    for(size_t nRowIdx=0; nRowIdx<(size_t)m_oFrameSize.height; ++nRowIdx) {
        const size_t nRowOffset = nRowIdx*m_oFrameSize.height;
        const size_t nModelRowOffset = nRowIdx*m_nRowStepSize;
        for(size_t nColIdx=0; nColIdx<(size_t)m_oFrameSize.width; ++nColIdx) {
            const size_t nColOffset = nColIdx+nRowOffset;
            const size_t nModelColOffset = nColIdx*m_nColStepSize+nModelRowOffset;
            if(bForceFGUpdate || !m_oLastFGMask.data[nColOffset]) {
                for(size_t nCurrModelSampleIdx=nRefreshSampleStartPos; nCurrModelSampleIdx<nRefreshSampleStartPos+nModelSamplesToRefresh; ++nCurrModelSampleIdx) {
                    int nSampleRowIdx, nSampleColIdx;
                    cv::getRandSamplePosition_7x7_std2(nSampleColIdx,nSampleRowIdx,(int)nColIdx,(int)nRowIdx,(int)LBSP::PATCH_SIZE/2,m_oFrameSize);
                    const size_t nSamplePxIdx = nSampleColIdx + nSampleRowIdx*m_oFrameSize.width;
                    if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
                        const size_t nCurrRealModelSampleIdx = nCurrModelSampleIdx%m_nBGSamples;
                        const size_t nSampleOffset_color = nSampleColIdx*m_oLastColorFrame.step.p[1]+(nSampleRowIdx*m_oLastColorFrame.step.p[0]);
                        const size_t nSampleOffset_desc = nSampleColIdx*m_oLastDescFrame.step.p[1]+(nSampleRowIdx*m_oLastDescFrame.step.p[0]);
                        const size_t nModelPxOffset_color = nCurrRealModelSampleIdx*m_nSampleStepSize+nModelColOffset;
                        const size_t nModelPxOffset_desc = nModelPxOffset_color+(m_nBGSamples*m_nSampleStepSize);
                        for(size_t nChannelIdx=0; nChannelIdx<m_nImgChannels; ++nChannelIdx) {
                            const size_t nModelTotOffset_color = nChannelIdx+nModelPxOffset_color;
                            const size_t nModelTotOffset_desc = nChannelIdx+nModelPxOffset_desc;
                            const size_t nSampleChannelIdx = ((nChannelIdx==3||m_nImgChannels==1)?nChannelIdx:2-nChannelIdx);
                            const size_t nSampleTotOffset_color = nSampleOffset_color+nSampleChannelIdx;
                            const size_t nSampleTotOffset_desc = nSampleOffset_desc+(nSampleChannelIdx*2);
                            m_vnBGModelData[nModelTotOffset_color] = (uint)m_oLastColorFrame.data[nSampleTotOffset_color];
                            if(m_nImgChannels==1)
                                LBSP::computeDescriptor<1>(m_oLastColorFrame,m_oLastColorFrame.data[nSampleTotOffset_color],nSampleColIdx,nSampleRowIdx,0,m_anLBSPThreshold_8bitLUT[m_oLastColorFrame.data[nSampleTotOffset_color]],*((ushort*)(m_oLastDescFrame.data+nSampleTotOffset_desc)));
                            else if(m_nImgChannels==3)
                                LBSP::computeDescriptor<3>(m_oLastColorFrame,m_oLastColorFrame.data[nSampleTotOffset_color],nSampleColIdx,nSampleRowIdx,nChannelIdx,m_anLBSPThreshold_8bitLUT[m_oLastColorFrame.data[nSampleTotOffset_color]],*((ushort*)(m_oLastDescFrame.data+nSampleTotOffset_desc)));
                            else //m_nImgChannels==4
                                LBSP::computeDescriptor<4>(m_oLastColorFrame,m_oLastColorFrame.data[nSampleTotOffset_color],nSampleColIdx,nSampleRowIdx,nChannelIdx,m_anLBSPThreshold_8bitLUT[m_oLastColorFrame.data[nSampleTotOffset_color]],*((ushort*)(m_oLastDescFrame.data+nSampleTotOffset_desc)));
                            m_vnBGModelData[nModelTotOffset_desc] = (uint)*(ushort*)(m_oLastDescFrame.data+nSampleTotOffset_desc);
                        }
                    }
                }
            }
        }
    }
    glBufferData(GL_SHADER_STORAGE_BUFFER,m_nBGModelSize*sizeof(uint),m_vnBGModelData.data(),GL_STATIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_BGModelBinding,getSSBOId(BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_BGModelBinding));
    glErrorCheck;
}

void BackgroundSubtractorLOBSTER_GLSL::initialize_gl(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    lvDbgExceptionWatch;
    // == init
    initialize_common(oInitImg,oROI);
    // not considering relevant pixels via LUT: it would ruin shared mem usage
    m_nTMT32ModelSize = size_t(m_oROI.cols*m_oROI.rows);
    m_nSampleStepSize = m_nImgChannels;
    m_nPxModelSize = m_nImgChannels*m_nBGSamples*2;
    m_nPxModelPadding = (m_nPxModelSize%4)?4-m_nPxModelSize%4:0;
    m_nColStepSize = m_nPxModelSize+m_nPxModelPadding;
    m_nRowStepSize = m_nColStepSize*m_oROI.cols;
    m_nBGModelSize = m_nRowStepSize*m_oROI.rows;
    const int nMaxSSBOBlockSize = lv::gl::getIntegerVal<1>(GL_MAX_SHADER_STORAGE_BLOCK_SIZE);
    lvAssert_(nMaxSSBOBlockSize>(int)(m_nBGModelSize*sizeof(uint)) && nMaxSSBOBlockSize>(int)(m_nTMT32ModelSize*sizeof(lv::gl::TMT32GenParams)),"max ssbo block size is tool small for the predicted model size");
    m_vnBGModelData.resize(m_nBGModelSize,0);
    lv::gl::TMT32GenParams::initTinyMT32Generators(glm::uvec3(uint(m_oROI.cols),uint(m_oROI.rows),1),m_voTMT32ModelData);
    m_bInitialized = true;
    GLImageProcAlgo::initialize_gl(oInitImg,m_oROI);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,getSSBOId(BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_TMT32ModelBinding));
    glBufferData(GL_SHADER_STORAGE_BUFFER,m_nTMT32ModelSize*sizeof(lv::gl::TMT32GenParams),m_voTMT32ModelData.data(),GL_STATIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_TMT32ModelBinding,getSSBOId(BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_TMT32ModelBinding));
    refreshModel(1.0f,true);
    m_bModelInitialized = true;
}

std::string BackgroundSubtractorLOBSTER_GLSL::getComputeShaderSource_LOBSTER() const {
    lvDbgExceptionWatch;
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    if(m_nImgChannels==4) { ssSrc <<
             "#define COLOR_DIST_THRESHOLD    " << m_nColorDistThreshold*3 << "\n"
             "#define DESC_DIST_THRESHOLD     " << m_nDescDistThreshold*3 << "\n"
             "#define COLOR_DIST_SC_THRESHOLD (COLOR_DIST_THRESHOLD/2)\n"
             "#define DESC_DIST_SC_THRESHOLD  (DESC_DIST_THRESHOLD/2)\n";
    }
    else { ssSrc << //m_nImgChannels==1
             "#define COLOR_DIST_THRESHOLD    " << m_nColorDistThreshold << "\n"
             "#define DESC_DIST_THRESHOLD     " << m_nDescDistThreshold << "\n";
    }
    ssSrc << "#define NB_SAMPLES              " << m_nBGSamples << "\n"
             "#define NB_REQ_SAMPLES          " << m_nRequiredBGSamples << "\n"
             "#define MODEL_STEP_SIZE         " << m_oFrameSize.width << "\n"
             "layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
             "layout(binding=" << GLImageProcAlgo::Image_ROIBinding << ", r8ui) readonly uniform uimage2D mROI;\n"
             "layout(binding=" << GLImageProcAlgo::Image_InputBinding << ", " << (m_nImgChannels==4?"rgba8ui":"r8ui") << ") readonly uniform uimage2D mInput;\n"
#if BGSLOBSTER_GLSL_USE_DEBUG
             //"layout(binding=" << GLImageProcAlgo::Image_DebugBinding << ", rgba8) writeonly uniform image2D mDebug;\n"
             "layout(binding=" << GLImageProcAlgo::Image_DebugBinding << ", rgba8ui) writeonly uniform uimage2D mDebug;\n"
#endif //BGSLOBSTER_GLSL_USE_DEBUG
             "layout(binding=" << GLImageProcAlgo::Image_OutputBinding << ", r8ui) writeonly uniform uimage2D mOutput;\n" <<
             (m_nImgChannels==4?lv::getShaderFunctionSource_L1dist():std::string())+lv::getShaderFunctionSource_absdiff(true) <<
             lv::getShaderFunctionSource_hdist() <<
             GLShader::getShaderFunctionSource_urand_tinymt32() <<
             GLShader::getShaderFunctionSource_getRandNeighbor3x3(0,m_oFrameSize) <<
             IBackgroundSubtractorLBSP_GLSL::getLBSPThresholdLUTShaderSource() <<
             LBSP::getShaderFunctionSource(m_nImgChannels,BGSLOBSTER_GLSL_USE_SHAREDMEM,m_vDefaultWorkGroupSize) <<
#if !BGSLOBSTER_GLSL_USE_SHAREDMEM
             "#define lbsp(t,ref,vCoords) lbsp(t,ref,mInput,vCoords)\n"
#endif //(!BGSLOBSTER_GLSL_USE_SHAREDMEM)
             "struct PxModel {\n"
             "    " << (m_nImgChannels==4?"uvec4":"uint") << " color_samples[" << m_nBGSamples << "];\n"
             "    " << (m_nImgChannels==4?"uvec4":"uint") << " lbsp_samples[" << m_nBGSamples << "];\n";
    if(m_nPxModelPadding>0) ssSrc <<
             "    uint pad[" << m_nPxModelPadding << "];\n";
    ssSrc << "};\n"
             "layout(binding=" << BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_BGModelBinding << ", std430) coherent buffer bBGModel {\n"
             "    PxModel aoPxModels[];\n"
             "};\n"
             "layout(binding=" << BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_TMT32ModelBinding << ", std430) buffer bTMT32Model {\n"
             "    TMT32Model aoTMT32Models[];\n"
             "};\n"
             "#define urand() urand(aoTMT32Models[nModelIdx])\n"
             "uniform uint nFrameIdx;\n"
             "uniform uint nResamplingRate;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
#if BGSLOBSTER_GLSL_USE_SHAREDMEM
             "    preload_data(mInput);\n"
             "    barrier();\n"
             //"        if(uvec2(gl_LocalInvocationID.xy)==uvec2(0,0)) {\n"
             //"            for(int y=0; y<" << m_vDefaultWorkGroupSize.y+(LBSP::PATCH_SIZE/2)*2 << "; ++y) {\n"
             //"                for(int x=0; x<" << m_vDefaultWorkGroupSize.x+(LBSP::PATCH_SIZE/2)*2 << "; ++x) {\n"
             //"                    ivec2 globalcoord = ivec2(gl_GlobalInvocationID.xy)-ivec2(gl_LocalInvocationID.xy)-ivec2("<<(LBSP::PATCH_SIZE/2)<<")+ivec2(x,y);\n"
             //"                    if(x<"<<(LBSP::PATCH_SIZE/2)<<" || y<"<<(LBSP::PATCH_SIZE/2)<<" || x>="<<m_vDefaultWorkGroupSize.x<<" || y>="<<m_vDefaultWorkGroupSize.y<<")\n"
             //"                    imageStore(mDebug,globalcoord,avLBSPData[y][x]);\n"//imageLoad(mInput,globalcoord)
             //"                }\n"
             //"            }\n"
             //"        }\n"
#endif //BGSLOBSTER_GLSL_USE_SHAREDMEM
             "    ivec2 vImgCoords = ivec2(gl_GlobalInvocationID.xy);\n"
             "    uvec4 vSegmResult = uvec4(0);\n"
             "    uint nROIVal = imageLoad(mROI,vImgCoords).r;\n"
             "    uint nModelIdx = gl_GlobalInvocationID.y*MODEL_STEP_SIZE + gl_GlobalInvocationID.x;\n"
             "    uvec3 vInputColor = imageLoad(mInput,vImgCoords).rgb;\n"
             "    uvec3 vInputDescThres = uvec3(anLBSPThresLUT[vInputColor.r],anLBSPThresLUT[vInputColor.g],anLBSPThresLUT[vInputColor.b]);\n"
             "    uvec3 vInputIntraDesc = lbsp(vInputDescThres,vInputColor,vImgCoords);\n"
             "    uint nGoodSamplesCount=0, nSampleIdx=0;\n"
             "    if(bool(nROIVal)) {\n"
             "        while(/*nGoodSamplesCount<NB_REQ_SAMPLES && */nSampleIdx<NB_SAMPLES) {\n"; // NOTE: CHECKING TWO CONDITIONS AT ONCE IN WHILE EXPRESSION STILL BROKEN AS F*CK (prop 4.4.0 NVIDIA 331.113)
    if(m_nImgChannels==4) { ssSrc <<
             "            uvec3 vCurrBGColorSample = aoPxModels[nModelIdx].color_samples[nSampleIdx].rgb;\n"
             "            uvec3 vCurrBGIntraDescSample = aoPxModels[nModelIdx].lbsp_samples[nSampleIdx].rgb;\n"
             "            uvec3 vCurrColorDist = absdiff(vInputColor,vCurrBGColorSample);\n"
#if BGSLOBSTER_GLSL_USE_BASIC_IMPL
             "            uvec3 vCurrDescDist = hdist(vInputIntraDesc,vCurrBGIntraDescSample);\n"
             "            if((vCurrColorDist.r+vCurrColorDist.g+vCurrColorDist.b)<=COLOR_DIST_THRESHOLD && (vCurrDescDist.r+vCurrDescDist.g+vCurrDescDist.b)<=DESC_DIST_THRESHOLD)\n";
#else //(!BGSLOBSTER_GLSL_USE_BASIC_IMPL)
             "            uvec3 vCurrDescThres = uvec3(anLBSPThresLUT[vCurrBGColorSample.r],anLBSPThresLUT[vCurrBGColorSample.g],anLBSPThresLUT[vCurrBGColorSample.b]);\n"
             "            uvec3 vCurrDescDist = hdist(lbsp(vCurrDescThres,vCurrBGColorSample,vImgCoords),vCurrBGIntraDescSample);\n"
             "            if(all(lessThanEqual(vCurrColorDist,uvec3(COLOR_DIST_SC_THRESHOLD))) && (vCurrColorDist.r+vCurrColorDist.g+vCurrColorDist.b)<=COLOR_DIST_THRESHOLD &&\n"
             "               all(lessThanEqual(vCurrDescDist,uvec3(DESC_DIST_SC_THRESHOLD))) && (vCurrDescDist.r+vCurrDescDist.g+vCurrDescDist.b)<=DESC_DIST_THRESHOLD)\n";
#endif //BGSLOBSTER_GLSL_USE_BASIC_IMPL
    }
    else { ssSrc << //m_nImgChannels==1
             "            uint nCurrBGColorSample = aoPxModels[nModelIdx].color_samples[nSampleIdx];\n"
             "            uint nCurrBGDescSample = aoPxModels[nModelIdx].lbsp_samples[nSampleIdx];\n"
#if BGSLOBSTER_GLSL_USE_BASIC_IMPL
             "            if(absdiff(vInputColor.r,nCurrBGColorSample)<=COLOR_DIST_THRESHOLD/2 && hdist(vInputIntraDesc.r,nCurrBGDescSample)<=DESC_DIST_THRESHOLD)\n";
#else //(!BGSLOBSTER_GLSL_USE_BASIC_IMPL)
             "            if(absdiff(vInputColor.r,nCurrBGColorSample)<=COLOR_DIST_THRESHOLD/2 &&\n"
             "               hdist(lbsp(uvec3(anLBSPThresLUT[nCurrBGColorSample]),uvec3(nCurrBGColorSample),vImgCoords).r,nCurrBGDescSample)<=DESC_DIST_THRESHOLD)\n";
#endif //(!BGSLOBSTER_GLSL_USE_BASIC_IMPL)
    }
    ssSrc << "                ++nGoodSamplesCount;\n"
             "            ++nSampleIdx;\n"
             "            if(nGoodSamplesCount>=NB_REQ_SAMPLES)\n"
             "                break;\n"
             "        }\n"
             "    }\n"
             //"    barrier();\n"
             "    if(bool(nROIVal)) {\n"
             "        if(nGoodSamplesCount<NB_REQ_SAMPLES)\n"
             "            vSegmResult.r = 255;\n"
             "        else {\n"
             "            if((urand()%nResamplingRate)==0) {\n"
             "                aoPxModels[nModelIdx].color_samples[(urand()%NB_SAMPLES)] = " << (m_nImgChannels==1?"vInputColor.r;\n":"uvec4(vInputColor,0);\n") <<
             "                aoPxModels[nModelIdx].lbsp_samples[(urand()%NB_SAMPLES)] = " << (m_nImgChannels==1?"vInputIntraDesc.r;\n":"uvec4(vInputIntraDesc,0);\n") <<
             "                memoryBarrier();\n"
             "            }\n"
             "            if((urand()%nResamplingRate)==0) {\n"
             "                ivec2 vNeighbCoords = getRandNeighbor3x3(vImgCoords,urand());\n"
             "                uint nNeighbPxModelIdx = uint(vNeighbCoords.y)*MODEL_STEP_SIZE + uint(vNeighbCoords.x);\n"
             "                aoPxModels[nNeighbPxModelIdx].color_samples[(urand()%NB_SAMPLES)] = " << (m_nImgChannels==1?"vInputColor.r;\n":"uvec4(vInputColor,0);\n") <<
             "                aoPxModels[nNeighbPxModelIdx].lbsp_samples[(urand()%NB_SAMPLES)] = " << (m_nImgChannels==1?"vInputIntraDesc.r;\n":"uvec4(vInputIntraDesc,0);\n") <<
             "                memoryBarrier();\n"
             "            }\n"
             "        }\n"
             "    }\n"
#if BGSLOBSTER_GLSL_USE_DEBUG
             //"    barrier();\n"
             "    if(bool(nROIVal)) {\n"
             //"        vec4 vAvgBGColor = vec4(0);\n"
             //"        for(uint nSampleIdx=0; nSampleIdx<NB_SAMPLES; ++nSampleIdx)\n"
             //"            vAvgBGColor += vec4(aoPxModels[nModelIdx].color_samples[nSampleIdx]);\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(vAvgBGColor/NB_SAMPLES));\n"
             //"        uvec4 test = uvec4(0);\n"
             //"        float fVar = float(gl_WorkGroupSize.y)/gl_WorkGroupSize.x;\n"
             //"        if(int(gl_LocalInvocationID.x*fVar)>=gl_LocalInvocationID.y && (gl_WorkGroupSize.y-int(fVar*gl_LocalInvocationID.x))>gl_LocalInvocationID.y)\n"
             //"            test = uvec4(255,0,0,0);\n"
             //"        else if(int(gl_LocalInvocationID.x*fVar)>gl_LocalInvocationID.y && (gl_WorkGroupSize.y-int(fVar*gl_LocalInvocationID.x))<=gl_LocalInvocationID.y)\n"
             //"            test = uvec4(255,255,0,0);\n"
             //"        else if(int(gl_LocalInvocationID.x*fVar)<=gl_LocalInvocationID.y && (gl_WorkGroupSize.y-int(fVar*gl_LocalInvocationID.x))<gl_LocalInvocationID.y)\n"
             //"            test = uvec4(255,0,255,0);\n"
             //"        else if(int(gl_LocalInvocationID.x*fVar)<gl_LocalInvocationID.y && (gl_WorkGroupSize.y-int(fVar*gl_LocalInvocationID.x))>=gl_LocalInvocationID.y)\n"
             //"            test = uvec4(255,255,255,0);\n"
             //"        imageStore(mDebug,vImgCoords,test);\n"
             //"        imageStore(mDebug,vImgCoords,aoPxModels[nModelIdx].color_samples[0]);\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(nResamplingRate*15));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(128));\n"
             //"        imageStore(mDebug,vImgCoords,vec4(0.5));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(nSampleIdx*(255/NB_SAMPLES)));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(hdist(uvec3(0),vInputIntraDesc)*15,0));\n"
             "    }\n"
#endif //BGSLOBSTER_GLSL_USE_DEBUG
             "    imageStore(mOutput,vImgCoords,vSegmResult);\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string BackgroundSubtractorLOBSTER_GLSL::getComputeShaderSource_PostProc() const {
    lvDbgExceptionWatch;
    lvAssert_(m_nDefaultMedianBlurKernelSize>0,"postproc median blur kernel size must be positive");
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
             //"layout(binding=" << GLImageProcAlgo::Image_ROIBinding << ", r8ui) readonly uniform uimage2D mROI;\n"
             "layout(binding=" << GLImageProcAlgo::Image_OutputBinding << ", r8ui) uniform uimage2D mOutput;\n" <<
             GLShader::getComputeShaderFunctionSource_BinaryMedianBlur(size_t(m_nDefaultMedianBlurKernelSize),BGSLOBSTER_GLSL_USE_SHAREDMEM,m_vDefaultWorkGroupSize);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "void main() {\n"
             "    ivec2 vImgCoords = ivec2(gl_GlobalInvocationID.xy);\n"
           //"    uint nROIVal = imageLoad(mROI,vImgCoords).r;\n"
#if BGSLOBSTER_GLSL_USE_SHAREDMEM
             "    preload_data(mOutput);\n"
             "    barrier();\n"
             "    uint nFinalSegmRes = BinaryMedianBlur(vImgCoords);\n"
#else //(!BGSLOBSTER_GLSL_USE_SHAREDMEM)
             "    uint nFinalSegmRes = BinaryMedianBlur(mOutput,vImgCoords);\n"
             "    barrier();\n"
#endif //(!BGSLOBSTER_GLSL_USE_SHAREDMEM)
             "    imageStore(mOutput,vImgCoords,uvec4(nFinalSegmRes));\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::string BackgroundSubtractorLOBSTER_GLSL::getComputeShaderSource(size_t nStage) const {
    lvDbgExceptionWatch;
    lvAssert_(m_bInitialized,"algo must be initialized first");
    lvAssert_(nStage<m_nComputeStages,"required compute stage does not exist");
    if(nStage==0)
        return getComputeShaderSource_LOBSTER();
    else //nStage==1 && BGSLOBSTER_GLSL_USE_POSTPROC
        return getComputeShaderSource_PostProc();
}

void BackgroundSubtractorLOBSTER_GLSL::dispatch(size_t nStage, GLShader& oShader) {
    lvDbgExceptionWatch;
    lvAssert_(nStage<m_nComputeStages,"required compute stage does not exist");
    if(nStage==0) {
        if(m_dCurrLearningRate>0)
            oShader.setUniform1ui("nResamplingRate",(GLuint)ceil(m_dCurrLearningRate));
        else
            oShader.setUniform1ui("nResamplingRate",BGSLOBSTER_DEFAULT_LEARNING_RATE);
    }
    else //nStage==1 && BGSLOBSTER_GLSL_USE_POSTPROC
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    glDispatchCompute((GLuint)ceil((float)m_oFrameSize.width/m_vDefaultWorkGroupSize.x),(GLuint)ceil((float)m_oFrameSize.height/m_vDefaultWorkGroupSize.y),1);
}

void BackgroundSubtractorLOBSTER_GLSL::getBackgroundImage(cv::OutputArray oBGImg) const {
    lvDbgExceptionWatch;
    lvAssert_(m_bInitialized,"algo must be initialized first");
    lvAssert_(m_bGLInitialized && !m_vnBGModelData.empty(),"algo gpu bg model not initialized");
    oBGImg.create(m_oFrameSize,CV_8UC(int(m_nImgChannels)));
    cv::Mat oOutputImg = oBGImg.getMatRef();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,getSSBOId(BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_BGModelBinding));
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,m_nBGModelSize*sizeof(uint),(void*)m_vnBGModelData.data());
    glErrorCheck;
    for(size_t nRowIdx=0; nRowIdx<(size_t)m_oFrameSize.height; ++nRowIdx) {
        const size_t nModelRowOffset = nRowIdx*m_nRowStepSize;
        const size_t nImgRowOffset = nRowIdx*oOutputImg.step.p[0];
        for(size_t nColIdx=0; nColIdx<(size_t)m_oFrameSize.width; ++nColIdx) {
            const size_t nModelColOffset = nColIdx*m_nColStepSize+nModelRowOffset;
            const size_t nImgColOffset = nColIdx*oOutputImg.step.p[1]+nImgRowOffset;
            std::array<float,4> afCurrPxSum = {0.0f,0.0f,0.0f,0.0f};
            for(size_t nSampleIdx=0; nSampleIdx<m_nBGSamples; ++nSampleIdx) {
                const size_t nModelPxOffset = nSampleIdx*m_nSampleStepSize+nModelColOffset;
                for(size_t nChannelIdx=0; nChannelIdx<m_nImgChannels; ++nChannelIdx) {
                    const size_t nModelTotOffset = nChannelIdx+nModelPxOffset;
                    afCurrPxSum[nChannelIdx] += m_vnBGModelData[nModelTotOffset];
                }
            }
            for(size_t nChannelIdx=0; nChannelIdx<m_nImgChannels; ++nChannelIdx) {
                const size_t nSampleChannelIdx = ((nChannelIdx==3||m_nImgChannels==1)?nChannelIdx:2-nChannelIdx);
                const size_t nImgTotOffset = nSampleChannelIdx+nImgColOffset;
                oOutputImg.data[nImgTotOffset] = (uchar)(afCurrPxSum[nChannelIdx]/m_nBGSamples);
            }
        }
    }
}

void BackgroundSubtractorLOBSTER_GLSL::getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const {
    lvDbgExceptionWatch;
    lvAssert_(m_bInitialized,"algo must be initialized first");
    lvAssert_(m_bGLInitialized && !m_vnBGModelData.empty(),"algo gpu bg model not initialized");
    static_assert(LBSP::DESC_SIZE==2,"Some assumptions are breaking below");
    oBGDescImg.create(m_oFrameSize,CV_16UC(int(m_nImgChannels)));
    cv::Mat oOutputImg = oBGDescImg.getMatRef();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,getSSBOId(BackgroundSubtractorLOBSTER_::LOBSTERStorageBuffer_BGModelBinding));
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,m_nBGModelSize*sizeof(uint),(void*)m_vnBGModelData.data());
    glErrorCheck;
    for(size_t nRowIdx=0; nRowIdx<(size_t)m_oFrameSize.height; ++nRowIdx) {
        const size_t nModelRowOffset = nRowIdx*m_nRowStepSize;
        const size_t nImgRowOffset = nRowIdx*oOutputImg.step.p[0];
        for(size_t nColIdx=0; nColIdx<(size_t)m_oFrameSize.width; ++nColIdx) {
            const size_t nModelColOffset = nColIdx*m_nColStepSize+nModelRowOffset;
            const size_t nImgColOffset = nColIdx*oOutputImg.step.p[1]+nImgRowOffset;
            std::array<float,4> afCurrPxSum = {0.0f,0.0f,0.0f,0.0f};
            for(size_t nSampleIdx=0; nSampleIdx<m_nBGSamples; ++nSampleIdx) {
                const size_t nModelPxOffset_color = nSampleIdx*m_nSampleStepSize+nModelColOffset;
                const size_t nModelPxOffset_desc = nModelPxOffset_color+(m_nBGSamples*m_nSampleStepSize);
                for(size_t nChannelIdx=0; nChannelIdx<m_nImgChannels; ++nChannelIdx) {
                    const size_t nModelTotOffset = nChannelIdx+nModelPxOffset_desc;
                    afCurrPxSum[nChannelIdx] += m_vnBGModelData[nModelTotOffset];
                }
            }
            for(size_t nChannelIdx=0; nChannelIdx<m_nImgChannels; ++nChannelIdx) {
                const size_t nSampleChannelIdx = ((nChannelIdx==3||m_nImgChannels==1)?nChannelIdx:2-nChannelIdx);
                const size_t nImgTotOffset = nSampleChannelIdx*2+nImgColOffset;
                *(ushort*)(oOutputImg.data+nImgTotOffset) = (ushort)(afCurrPxSum[nChannelIdx]/m_nBGSamples);
            }
        }
    }
}

template struct BackgroundSubtractorLOBSTER_<lv::GLSL>;
#endif //HAVE_GLSL

#if HAVE_CUDA
// ... @@@ add impl later
//template struct BackgroundSubtractorLOBSTER_<lv::CUDA>;
#endif //HAVE_CUDA

#if HAVE_OPENCL
// ... @@@ add impl later
//template struct BackgroundSubtractorLOBSTER_<lv::OpenCL>;
#endif //HAVE_OPENCL

void BackgroundSubtractorLOBSTER::refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate) {
    lvDbgExceptionWatch;
    // == refresh
    lvAssert_(m_bInitialized,"algo must be initialized first");
    lvAssert_(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f,"model refresh must be given as a non-null fraction");
    const size_t nModelSamplesToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
    const size_t nRefreshSampleStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
    for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
        const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
        if(bForceFGUpdate || !m_oLastFGMask.data[nPxIter]) {
            for(size_t nCurrModelSampleIdx=nRefreshSampleStartPos; nCurrModelSampleIdx<nRefreshSampleStartPos+nModelSamplesToRefresh; ++nCurrModelSampleIdx) {
                int nSampleImgCoord_Y, nSampleImgCoord_X;
                cv::getRandSamplePosition_7x7_std2(nSampleImgCoord_X,nSampleImgCoord_Y,m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
                if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
                    const size_t nCurrRealModelSampleIdx = nCurrModelSampleIdx%m_nBGSamples;
                    for(size_t c=0; c<m_nImgChannels; ++c) {
                        m_voBGColorSamples[nCurrRealModelSampleIdx].data[nPxIter*m_nImgChannels+c] = m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c];
                        if(m_nImgChannels==1)
                            LBSP::computeDescriptor<1>(m_oLastColorFrame,m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c],nSampleImgCoord_X,nSampleImgCoord_Y,0,m_anLBSPThreshold_8bitLUT[m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c]],*((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*m_nImgChannels+c)*2)));
                        else if(m_nImgChannels==3)
                            LBSP::computeDescriptor<3>(m_oLastColorFrame,m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c],nSampleImgCoord_X,nSampleImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c]],*((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*m_nImgChannels+c)*2)));
                        else //m_nImgChannels==4
                            LBSP::computeDescriptor<4>(m_oLastColorFrame,m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c],nSampleImgCoord_X,nSampleImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c]],*((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*m_nImgChannels+c)*2)));
                        *((ushort*)(m_voBGDescSamples[nCurrRealModelSampleIdx].data+(nPxIter*m_nImgChannels+c)*2)) = *((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*m_nImgChannels+c)*2));
                    }
                }
            }
        }
    }
}

void BackgroundSubtractorLOBSTER::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    lvDbgExceptionWatch;
    // == init
    IBackgroundSubtractorLBSP::initialize_common(oInitImg,oROI);
    m_voBGColorSamples.resize(m_nBGSamples);
    m_voBGDescSamples.resize(m_nBGSamples);
    for(size_t s=0; s<m_nBGSamples; ++s) {
        m_voBGColorSamples[s].create(m_oImgSize,CV_8UC((int)m_nImgChannels));
        m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);
        m_voBGDescSamples[s].create(m_oImgSize,CV_16UC((int)m_nImgChannels));
        m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
    }
    m_bInitialized = true;
    refreshModel(1.0f,true);
    m_bModelInitialized = true;
}

void BackgroundSubtractorLOBSTER::apply(cv::InputArray _oInputImg, cv::OutputArray _oFGMask, double dLearningRate) {
    lvDbgExceptionWatch;
    // == process_sync
    lvAssert_(m_bInitialized && m_bModelInitialized,"algo & model must be initialized first");
    lvAssert_(dLearningRate>0,"learning rate must be a positive value; faster learning is achieved with smaller values");
    cv::Mat oInputImg = _oInputImg.getMat();
    lvAssert_(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize,"input image type/size mismatch with initialization type/size");
    lvAssert_(oInputImg.isContinuous(),"input image data must be continuous");
    _oFGMask.create(m_oImgSize,CV_8UC1);
    cv::Mat oCurrFGMask = _oFGMask.getMat();
    oCurrFGMask = cv::Scalar_<uchar>(0);
    const size_t nLearningRate = std::isinf(dLearningRate)?SIZE_MAX:(size_t)ceil(dLearningRate);
    if(m_nImgChannels==1) {
        for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
            const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
            const size_t nDescIter = nPxIter*2;
            const int nCurrImgCoord_X = m_voPxInfoLUT[nPxIter].nImgCoord_X;
            const int nCurrImgCoord_Y = m_voPxInfoLUT[nPxIter].nImgCoord_Y;
            const uchar nCurrColor = oInputImg.data[nPxIter];
            alignas(16) std::array<uchar,LBSP::DESC_SIZE_BITS> anLBSPLookupVals;
            LBSP::computeDescriptor_lookup<1>(oInputImg,nCurrImgCoord_X,nCurrImgCoord_Y,0,anLBSPLookupVals);
            size_t nGoodSamplesCount=0, nModelIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nModelIdx<m_nBGSamples) {
                const uchar nBGColor = m_voBGColorSamples[nModelIdx].data[nPxIter];
                {
                    const size_t nColorDist = lv::L1dist(nCurrColor,nBGColor);
                    if(nColorDist>m_nColorDistThreshold/2)
                        goto failedcheck1ch;
                    const ushort nCurrInputDesc = LBSP::computeDescriptor_threshold(anLBSPLookupVals,nBGColor,m_anLBSPThreshold_8bitLUT[nBGColor]);
                    const size_t nDescDist = lv::hdist(nCurrInputDesc,*((ushort*)(m_voBGDescSamples[nModelIdx].data+nDescIter)));
                    if(nDescDist>m_nDescDistThreshold)
                        goto failedcheck1ch;
                    nGoodSamplesCount++;
                }
                failedcheck1ch:
                nModelIdx++;
            }
            if(nGoodSamplesCount<m_nRequiredBGSamples)
                oCurrFGMask.data[nPxIter] = UCHAR_MAX;
            else {
                if((rand()%nLearningRate)==0) {
                    const size_t nSampleModelIdx = rand()%m_nBGSamples;
                    ushort& nRandInputDesc = *((ushort*)(m_voBGDescSamples[nSampleModelIdx].data+nDescIter));
                    nRandInputDesc = LBSP::computeDescriptor_threshold(anLBSPLookupVals,nCurrColor,m_anLBSPThreshold_8bitLUT[nCurrColor]);
                    m_voBGColorSamples[nSampleModelIdx].data[nPxIter] = nCurrColor;
                }
                if((rand()%nLearningRate)==0) {
                    int nSampleImgCoord_Y, nSampleImgCoord_X;
                    cv::getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                    const size_t nSampleModelIdx = rand()%m_nBGSamples;
                    ushort& nRandInputDesc = m_voBGDescSamples[nSampleModelIdx].at<ushort>(nSampleImgCoord_Y,nSampleImgCoord_X);
                    nRandInputDesc = LBSP::computeDescriptor_threshold(anLBSPLookupVals,nCurrColor,m_anLBSPThreshold_8bitLUT[nCurrColor]);
                    m_voBGColorSamples[nSampleModelIdx].at<uchar>(nSampleImgCoord_Y,nSampleImgCoord_X) = nCurrColor;
                }
            }
        }
    }
    else { //m_nImgChannels==3
        const size_t nCurrDescDistThreshold = m_nDescDistThreshold*3;
        const size_t nCurrColorDistThreshold = m_nColorDistThreshold*3;
        const size_t nCurrSCDescDistThreshold = nCurrDescDistThreshold/2;
        const size_t nCurrSCColorDistThreshold = nCurrColorDistThreshold/2;
        const size_t desc_row_step = m_voBGDescSamples[0].step.p[0];
        const size_t img_row_step = m_voBGColorSamples[0].step.p[0];
        for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
            const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
            const int nCurrImgCoord_X = m_voPxInfoLUT[nPxIter].nImgCoord_X;
            const int nCurrImgCoord_Y = m_voPxInfoLUT[nPxIter].nImgCoord_Y;
            const size_t nPxIterRGB = nPxIter*3;
            const size_t nDescIterRGB = nPxIterRGB*2;
            const uchar* const anCurrColor = oInputImg.data+nPxIterRGB;
            alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,3> aanLBSPLookupVals;
            LBSP::computeDescriptor_lookup(oInputImg,nCurrImgCoord_X,nCurrImgCoord_Y,aanLBSPLookupVals);
            size_t nGoodSamplesCount=0, nModelIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nModelIdx<m_nBGSamples) {
                const ushort* const anBGDesc = (ushort*)(m_voBGDescSamples[nModelIdx].data+nDescIterRGB);
                const uchar* const anBGColor = m_voBGColorSamples[nModelIdx].data+nPxIterRGB;
                size_t nTotColorDist = 0;
                size_t nTotDescDist = 0;
                for(size_t c=0;c<3; ++c) {
                    const size_t nColorDist = lv::L1dist(anCurrColor[c],anBGColor[c]);
                    if(nColorDist>nCurrSCColorDistThreshold)
                        goto failedcheck3ch;
                    const ushort nCurrInputDesc = LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],anBGColor[c],m_anLBSPThreshold_8bitLUT[anBGColor[c]]);
                    const size_t nDescDist = lv::hdist(nCurrInputDesc,anBGDesc[c]);
                    if(nDescDist>nCurrSCDescDistThreshold)
                        goto failedcheck3ch;
                    nTotColorDist += nColorDist;
                    nTotDescDist += nDescDist;
                }
                if(nTotDescDist<=nCurrDescDistThreshold && nTotColorDist<=nCurrColorDistThreshold)
                    nGoodSamplesCount++;
                failedcheck3ch:
                nModelIdx++;
            }
            if(nGoodSamplesCount<m_nRequiredBGSamples)
                oCurrFGMask.data[nPxIter] = UCHAR_MAX;
            else {
                if((rand()%nLearningRate)==0) {
                    const size_t nSampleModelIdx = rand()%m_nBGSamples;
                    ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[nSampleModelIdx].data+nDescIterRGB));
                    for(size_t c=0; c<3; ++c) {
                        *(m_voBGColorSamples[nSampleModelIdx].data+nPxIterRGB+c) = anCurrColor[c];
                        anRandInputDesc[c] = LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],anCurrColor[c],m_anLBSPThreshold_8bitLUT[anCurrColor[c]]);
                    }
                }
                if((rand()%nLearningRate)==0) {
                    int nSampleImgCoord_Y, nSampleImgCoord_X;
                    cv::getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                    const size_t nSampleModelIdx = rand()%m_nBGSamples;
                    ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[nSampleModelIdx].data + desc_row_step*nSampleImgCoord_Y + 6*nSampleImgCoord_X));
                    for(size_t c=0; c<3; ++c) {
                        *(m_voBGColorSamples[nSampleModelIdx].data+img_row_step*nSampleImgCoord_Y+3*nSampleImgCoord_X+c) = anCurrColor[c];
                        anRandInputDesc[c] = LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],anCurrColor[c],m_anLBSPThreshold_8bitLUT[anCurrColor[c]]);
                    }
                }
            }
        }
    }
    cv::medianBlur(oCurrFGMask,m_oLastFGMask,m_nDefaultMedianBlurKernelSize);
    m_oLastFGMask.copyTo(oCurrFGMask);
    oInputImg.copyTo(m_oLastColorFrame);
}

void BackgroundSubtractorLOBSTER::getBackgroundImage(cv::OutputArray oBGImg) const {
    lvDbgExceptionWatch;
    lvAssert_(m_bInitialized,"algo must be initialized first");
    cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
    for(size_t s=0; s<m_nBGSamples; ++s) {
        for(int y=0; y<m_oImgSize.height; ++y) {
            for(int x=0; x<m_oImgSize.width; ++x) {
                const size_t idx_nimg = m_voBGColorSamples[s].step.p[0]*y + m_voBGColorSamples[s].step.p[1]*x;
                const size_t idx_flt32 = idx_nimg*4;
                float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+idx_flt32);
                const uchar* const oBGImgPtr = m_voBGColorSamples[s].data+idx_nimg;
                for(size_t c=0; c<m_nImgChannels; ++c)
                    oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_nBGSamples;
            }
        }
    }
    oAvgBGImg.convertTo(oBGImg,CV_8U);
}

void BackgroundSubtractorLOBSTER::getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const {
    static_assert(LBSP::DESC_SIZE==2,"bad assumptions in impl below");
    lvDbgExceptionWatch;
    lvAssert_(m_bInitialized,"algo must be initialized first");
    cv::Mat oAvgBGDesc = cv::Mat::zeros(m_oImgSize,CV_32FC((int)m_nImgChannels));
    for(size_t n=0; n<m_voBGDescSamples.size(); ++n) {
        for(int y=0; y<m_oImgSize.height; ++y) {
            for(int x=0; x<m_oImgSize.width; ++x) {
                const size_t idx_ndesc = m_voBGDescSamples[n].step.p[0]*y + m_voBGDescSamples[n].step.p[1]*x;
                const size_t idx_flt32 = idx_ndesc*2;
                float* oAvgBgDescPtr = (float*)(oAvgBGDesc.data+idx_flt32);
                const ushort* const oBGDescPtr = (ushort*)(m_voBGDescSamples[n].data+idx_ndesc);
                for(size_t c=0; c<m_nImgChannels; ++c)
                    oAvgBgDescPtr[c] += ((float)oBGDescPtr[c])/m_voBGDescSamples.size();
            }
        }
    }
    oAvgBGDesc.convertTo(oBGDescImg,CV_16U);
}

template struct BackgroundSubtractorLOBSTER_<lv::NonParallel>;
