#include "BackgroundSubtractorLOBSTER.h"
#include "RandUtils.h"
#include <iostream>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#if HAVE_GLSL
#define LOBSTER_GLSL_DEBUG       1
#define LOBSTER_GLSL_TIMERS      0
#define LOBSTER_GLSL_BASIC       0
#if (!GLSL_RENDERING && LOBSTER_GLSL_DEBUG)
#undef LOBSTER_GLSL_DEBUG
#define LOBSTER_GLSL_DEBUG 0
#endif //(!LOBSTER_GLSL_DISPLAY && LOBSTER_GLSL_DEBUG)
#endif //HAVE_GLSL

BackgroundSubtractorLOBSTER::BackgroundSubtractorLOBSTER(  float fRelLBSPThreshold
                                                          ,size_t nLBSPThresholdOffset
                                                          ,size_t nDescDistThreshold
                                                          ,size_t nColorDistThreshold
                                                          ,size_t nBGSamples
                                                          ,size_t nRequiredBGSamples)
    :    BackgroundSubtractorLBSP(fRelLBSPThreshold,nLBSPThresholdOffset)
#if HAVE_GLSL
        ,GLImageProcAlgo(1,2,1,CV_8UC1,LOBSTER_GLSL_DEBUG?CV_8UC4:-1,true,GLSL_RENDERING,LOBSTER_GLSL_TIMERS,true)
#endif //HAVE_GLSL
        ,m_nColorDistThreshold(nColorDistThreshold)
        ,m_nDescDistThreshold(nDescDistThreshold)
        ,m_nBGSamples(nBGSamples)
        ,m_nRequiredBGSamples(nRequiredBGSamples)
        ,m_bModelInitialized(false)
#if HAVE_GLSL
        ,m_nCurrResamplingRate(BGSLOBSTER_DEFAULT_LEARNING_RATE)
        ,m_nTMT32ModelSize(0)
        ,m_nSampleStepSize(0)
        ,m_nPxModelSize(0)
        ,m_nPxModelPadding(0)
        ,m_nColStepSize(0)
        ,m_nRowStepSize(0)
        ,m_nBGModelSize(0)
#endif //HAVE_GLSL
{
    CV_Assert(m_nRequiredBGSamples<=m_nBGSamples);
    CV_Assert(m_nColorDistThreshold>0);
    m_bAutoModelResetEnabled = false; // @@@@@@ not supported here for now
#if HAVE_GLSL
    glAssert(m_nCurrResamplingRate>0);
    glAssert(GLUtils::getIntegerVal<1>(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)>=2);
    glErrorCheck;
#endif //HAVE_GLSL
}

BackgroundSubtractorLOBSTER::~BackgroundSubtractorLOBSTER() {}

void BackgroundSubtractorLOBSTER::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    // == init
    BackgroundSubtractorLBSP::initialize(oInitImg,oROI);
    m_bModelInitialized = false;
#if HAVE_GLSL
    // not considering relevant pixels via LUT @@@@@@@@ should it?
    m_nTMT32ModelSize = m_oROI.cols*m_oROI.rows;
    m_nSampleStepSize = m_nImgChannels;
    m_nPxModelSize = m_nImgChannels*m_nBGSamples*2;
    m_nPxModelPadding = (m_nPxModelSize%4)?4-m_nPxModelSize%4:0;
    m_nColStepSize = m_nPxModelSize+m_nPxModelPadding;
    m_nRowStepSize = m_nColStepSize*m_oROI.cols;
    m_nBGModelSize = m_nRowStepSize*m_oROI.rows;
    const int nMaxSSBOBlockSize = GLUtils::getIntegerVal<1>(GL_MAX_SHADER_STORAGE_BLOCK_SIZE);
    glAssert(nMaxSSBOBlockSize>(int)(m_nBGModelSize*sizeof(uint)) && nMaxSSBOBlockSize>(int)(m_nTMT32ModelSize*sizeof(GLSLFunctionUtils::TMT32GenParams)));
    m_vnBGModelData.resize(m_nBGModelSize,0);
    GLSLFunctionUtils::initTinyMT32Generators(glm::uvec3(m_oROI.cols,m_oROI.rows,1),m_voTMT32ModelData);
    m_bModelInitialized = true;
    GLImageProcAlgo::initialize(oInitImg,m_oROI);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,getSSBOId(BackgroundSubtractorLOBSTER::eBuffer_TMT32ModelBinding));
    glBufferData(GL_SHADER_STORAGE_BUFFER,m_nTMT32ModelSize*sizeof(GLSLFunctionUtils::TMT32GenParams),m_voTMT32ModelData.data(),GL_STATIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,BackgroundSubtractorLOBSTER::eBuffer_TMT32ModelBinding,getSSBOId(BackgroundSubtractorLOBSTER::eBuffer_TMT32ModelBinding));
#else //!HAVE_GLSL
    m_voBGColorSamples.resize(m_nBGSamples);
    m_voBGDescSamples.resize(m_nBGSamples);
    for(size_t s=0; s<m_nBGSamples; ++s) {
        m_voBGColorSamples[s].create(m_oImgSize,CV_8UC((int)m_nImgChannels));
        m_voBGColorSamples[s] = cv::Scalar_<uchar>::all(0);
        m_voBGDescSamples[s].create(m_oImgSize,CV_16UC((int)m_nImgChannels));
        m_voBGDescSamples[s] = cv::Scalar_<ushort>::all(0);
    }
    m_bModelInitialized = true;
#endif //!HAVE_GLSL
    refreshModel(1.0f);
}

void BackgroundSubtractorLOBSTER::refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate) {
    // == refresh
    CV_Assert(m_bInitialized);
    CV_Assert(fSamplesRefreshFrac>0.0f && fSamplesRefreshFrac<=1.0f);
    const size_t nModelsToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*m_nBGSamples):m_nBGSamples;
    const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%m_nBGSamples:0;
#if HAVE_GLSL
    // full clears model every time, should it consider refresh frac? @@@@@@@@
    glAssert(nModelsToRefresh==m_nBGSamples && nRefreshStartPos==0 && (bForceFGUpdate||true));
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,getSSBOId(BackgroundSubtractorLOBSTER::eBuffer_BGModelBinding));
    for(size_t nRowIdx=0; nRowIdx<(size_t)m_oFrameSize.height; ++nRowIdx) {
        const size_t nModelRowOffset = nRowIdx*m_nRowStepSize;
        for(size_t nColIdx=0; nColIdx<(size_t)m_oFrameSize.width; ++nColIdx) {
            const size_t nModelColOffset = nColIdx*m_nColStepSize+nModelRowOffset;
            for(size_t nSampleIdx=0; nSampleIdx<m_nBGSamples; ++nSampleIdx) {
                int nSampleRowIdx, nSampleColIdx;
                getRandNeighborPosition_3x3(nSampleColIdx,nSampleRowIdx,nColIdx,nRowIdx,LBSP::PATCH_SIZE/2,m_oFrameSize);
                const size_t nSampleOffset_color = nSampleColIdx*m_oLastColorFrame.step.p[1]+(nSampleRowIdx*m_oLastColorFrame.step.p[0]);
                const size_t nSampleOffset_desc = nSampleColIdx*m_oLastDescFrame.step.p[1]+(nSampleRowIdx*m_oLastDescFrame.step.p[0]);
                const size_t nModelPxOffset_color = nSampleIdx*m_nSampleStepSize+nModelColOffset;
                const size_t nModelPxOffset_desc = nModelPxOffset_color+(m_nBGSamples*m_nSampleStepSize);
                for(size_t nChannelIdx=0; nChannelIdx<m_nImgChannels; ++nChannelIdx) {
                    const size_t nModelTotOffset_color = nChannelIdx+nModelPxOffset_color;
                    const size_t nModelTotOffset_desc = nChannelIdx+nModelPxOffset_desc;
                    const size_t nSampleChannelIdx = ((nChannelIdx==3||m_nImgChannels==1)?nChannelIdx:2-nChannelIdx);
                    const size_t nSampleTotOffset_color = nSampleOffset_color+nSampleChannelIdx;
                    const size_t nSampleTotOffset_desc = nSampleOffset_desc+(nSampleChannelIdx*2);
                    m_vnBGModelData[nModelTotOffset_color] = (uint)m_oLastColorFrame.data[nSampleTotOffset_color];
                    m_vnBGModelData[nModelTotOffset_desc] = (uint)*(ushort*)(m_oLastDescFrame.data+nSampleTotOffset_desc);
                    // @@@@@@ LOBSTER does not update m_oLastDescFrame
                }
            }
        }
    }
    glBufferData(GL_SHADER_STORAGE_BUFFER,m_nBGModelSize*sizeof(uint),m_vnBGModelData.data(),GL_STATIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,BackgroundSubtractorLOBSTER::eBuffer_BGModelBinding,getSSBOId(BackgroundSubtractorLOBSTER::eBuffer_BGModelBinding));
    glErrorCheck;
#else //!HAVE_GLSL
    for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
        const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
        if(bForceFGUpdate || !m_oLastFGMask.data[nPxIter]) {
            for(size_t nCurrModelIdx=nRefreshStartPos; nCurrModelIdx<nRefreshStartPos+nModelsToRefresh; ++nCurrModelIdx) {
                int nSampleImgCoord_Y, nSampleImgCoord_X;
                getRandSamplePosition(nSampleImgCoord_X,nSampleImgCoord_Y,m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                const size_t nSamplePxIdx = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
                if(bForceFGUpdate || !m_oLastFGMask.data[nSamplePxIdx]) {
                    const size_t nCurrRealModelIdx = nCurrModelIdx%m_nBGSamples;
                    for(size_t c=0; c<m_nImgChannels; ++c) {
                        m_voBGColorSamples[nCurrRealModelIdx].data[nPxIter*m_nImgChannels+c] = m_oLastColorFrame.data[nSamplePxIdx*m_nImgChannels+c];
                        *((ushort*)(m_voBGDescSamples[nCurrRealModelIdx].data+(nPxIter*m_nImgChannels+c)*2)) = *((ushort*)(m_oLastDescFrame.data+(nSamplePxIdx*m_nImgChannels+c)*2));
                    }
                }
            }
        }
    }
#endif //!HAVE_GLSL
}

#if HAVE_GLSL

const GLuint BackgroundSubtractorLOBSTER::eBuffer_BGModelBinding = GLImageProcAlgo::eBuffer_CustomBinding1;
const GLuint BackgroundSubtractorLOBSTER::eBuffer_TMT32ModelBinding = GLImageProcAlgo::eBuffer_CustomBinding2;

void BackgroundSubtractorLOBSTER::apply(cv::InputArray _oNextInputImg, double dLearningRate) {
    this->apply_glimpl(_oNextInputImg,false,dLearningRate);
}

void BackgroundSubtractorLOBSTER::apply_glimpl(cv::InputArray _oNextInputImg, bool bRebindAll, double dLearningRate) {
    // == process_GLSL_async
    CV_Assert(m_bInitialized && m_bModelInitialized);
    CV_Assert(dLearningRate>0);
    m_nCurrResamplingRate = (size_t)ceil(dLearningRate);
    cv::Mat oNextInputImg = _oNextInputImg.getMat();
    CV_Assert(oNextInputImg.type()==m_nImgType && oNextInputImg.size()==m_oImgSize);
    CV_Assert(oNextInputImg.isContinuous());
    ++m_nFrameIdx;
    this->GLImageProcAlgo::apply(oNextInputImg,bRebindAll);
}

void BackgroundSubtractorLOBSTER::getLatestForegroundMask(cv::OutputArray _oLastFGMask) {
    _oLastFGMask.create(m_oImgSize,CV_8UC1);
    cv::Mat oLastFGMask = _oLastFGMask.getMat();
    if(!m_bFetchingOutput)
        glAssert(setOutputFetching(true))
    else if(m_nFrameIdx>0)
        fetchLastOutput(oLastFGMask);
    else
        oLastFGMask = cv::Scalar_<uchar>(0);
}

std::string BackgroundSubtractorLOBSTER::getComputeShaderSource(size_t nStage) const {
    glAssert(m_bModelInitialized && m_bInitialized);
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n";
    if(m_nImgChannels==4) { ssSrc <<
             "#define COLOR_DIST_THRESHOLD    " << m_nColorDistThreshold*3 << "\n"
             "#define DESC_DIST_THRESHOLD     " << m_nDescDistThreshold*3 << "\n"
             "#define COLOR_DIST_SC_THRESHOLD (COLOR_DIST_THRESHOLD/2)\n"
             "#define DESC_DIST_SC_THRESHOLD  (DESC_DIST_THRESHOLD/2)\n";
    }
    else { ssSrc <<
             "#define COLOR_DIST_THRESHOLD    " << m_nColorDistThreshold << "\n"
             "#define DESC_DIST_THRESHOLD     " << m_nDescDistThreshold << "\n";
    }
    ssSrc << "#define NB_SAMPLES              " << m_nBGSamples << "\n"
             "#define NB_REQ_SAMPLES          " << m_nRequiredBGSamples << "\n"
             "#define MODEL_STEP_SIZE         " << m_oFrameSize.width << "\n"
             "struct PxModel {\n"
             "    " << (m_nImgChannels==4?"uvec4":"uint") << " color_samples[" << m_nBGSamples << "];\n"
             "    " << (m_nImgChannels==4?"uvec4":"uint") << " lbsp_samples[" << m_nBGSamples << "];\n";
    if(m_nPxModelPadding>0) ssSrc <<
             "    uint pad[" << m_nPxModelPadding << "];\n";
    ssSrc << "};\n";
    ssSrc << (m_nImgChannels==4?GLSLFunctionUtils::getShaderFunctionSource_L1dist():std::string())+GLSLFunctionUtils::getShaderFunctionSource_absdiff(true);
    ssSrc << GLSLFunctionUtils::getShaderFunctionSource_hdist(); // @@@@ transfer to distanceutils
    ssSrc << GLSLFunctionUtils::getShaderFunctionSource_urand_tinymt32(); // @@@@ transfer to randutils
    ssSrc << GLSLFunctionUtils::getShaderFunctionSource_getRandNeighbor3x3(0,m_oFrameSize); // @@@@ transfer to randutils
    ssSrc << LBSP::getShaderFunctionSource(m_nImgChannels);
    ssSrc << BackgroundSubtractorLBSP::getLBSPThresholdLUTShaderSource();
    ssSrc << "#define urand() urand(aoTMT32Models[nModelIdx])\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
             "layout(binding=" << GLImageProcAlgo::eImage_ROIBinding << ", r8ui) readonly uniform uimage2D mROI;\n"
             "layout(binding=" << GLImageProcAlgo::eImage_InputBinding << ", " << (m_nImgChannels==4?"rgba8ui":"r8ui") << ") readonly uniform uimage2D mInput;\n"
#if LOBSTER_GLSL_DEBUG
             //"layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ", rgba8) writeonly uniform image2D mDebug;\n"
             "layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ", rgba8ui) writeonly uniform uimage2D mDebug;\n"
#endif //LOBSTER_GLSL_DEBUG
             "layout(binding=" << GLImageProcAlgo::eImage_OutputBinding << ", r8ui) writeonly uniform uimage2D mOutput;\n"
             "layout(binding=" << BackgroundSubtractorLOBSTER::eBuffer_BGModelBinding << ", std430) coherent buffer bBGModel {\n"
             "    PxModel aoPxModels[];\n"
             "};\n"
             "layout(binding=" << BackgroundSubtractorLOBSTER::eBuffer_TMT32ModelBinding << ", std430) buffer bTMT32Model {\n"
             "    TMT32Model aoTMT32Models[];\n"
             "};\n"
             "uniform uint nFrameIdx;\n"
             "uniform uint nResamplingRate;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // @@@@@ try impl without conditions?
    ssSrc << "void main() {\n"
             "    ivec2 vImgCoords = ivec2(gl_GlobalInvocationID.xy);\n"
             "    uvec4 vSegmResult = uvec4(0);\n"
             "    uint nROIVal = imageLoad(mROI,vImgCoords).r;\n"
             "    uint nModelIdx = gl_GlobalInvocationID.y*MODEL_STEP_SIZE + gl_GlobalInvocationID.x;\n"
             "    uint nGoodSamplesCount=0, nSampleIdx=0;\n"
             "    uvec3 vInputColor = imageLoad(mInput,vImgCoords).rgb;\n"
             "    uvec3 vInputIntraDesc = lbsp(uvec3(anLBSPThresLUT[vInputColor.r],anLBSPThresLUT[vInputColor.g],anLBSPThresLUT[vInputColor.b]),vInputColor,mInput,vImgCoords);\n" // @@@@ preload & memshare input img?
             "    if(bool(nROIVal)) {\n"
             "        while(/*nGoodSamplesCount<NB_REQ_SAMPLES && */nSampleIdx<NB_SAMPLES) {\n"; // NOTE: CHECKING TWO CONDITIONS AT ONCE IN WHILE EXPRESSION STILL BROKEN AS F*CK (prop 4.4.0 NVIDIA 331.113)
    if(m_nImgChannels==4) { ssSrc <<
             "            uvec3 vCurrBGColorSample = aoPxModels[nModelIdx].color_samples[nSampleIdx].rgb;\n"
             "            uvec3 vCurrBGIntraDescSample = aoPxModels[nModelIdx].lbsp_samples[nSampleIdx].rgb;\n"
             "            uvec3 vCurrColorDist = absdiff(vInputColor,vCurrBGColorSample);\n"
#if LOBSTER_GLSL_BASIC
             "            uvec3 vCurrDescDist = hdist(vInputIntraDesc,vCurrBGIntraDescSample);\n"
             "            if((vCurrColorDist.r+vCurrColorDist.g+vCurrColorDist.b)<=COLOR_DIST_THRESHOLD && (vCurrDescDist.r+vCurrDescDist.g+vCurrDescDist.b)<=DESC_DIST_THRESHOLD)\n"
#else //!LOBSTER_GLSL_BASIC
             "            uvec3 vCurrDescDist = hdist(lbsp(uvec3(anLBSPThresLUT[vCurrBGColorSample.r],anLBSPThresLUT[vCurrBGColorSample.g],anLBSPThresLUT[vCurrBGColorSample.b]),vCurrBGColorSample,mInput,vImgCoords),vCurrBGIntraDescSample);\n"
             "            if(all(lessThanEqual(vCurrColorDist,uvec3(COLOR_DIST_SC_THRESHOLD))) && (vCurrColorDist.r+vCurrColorDist.g+vCurrColorDist.b)<=COLOR_DIST_THRESHOLD &&\n"
             "               all(lessThanEqual(vCurrDescDist,uvec3(DESC_DIST_SC_THRESHOLD))) && (vCurrDescDist.r+vCurrDescDist.g+vCurrDescDist.b)<=DESC_DIST_THRESHOLD)\n";
#endif //LOBSTER_GLSL_BASIC
    }
    else { ssSrc <<
             "            uint nCurrBGColorSample = aoPxModels[nModelIdx].color_samples[nSampleIdx];\n"
             "            uint nCurrBGDescSample = aoPxModels[nModelIdx].lbsp_samples[nSampleIdx];\n"
#if LOBSTER_GLSL_BASIC
             "            if(absdiff(vInputColor.r,nCurrBGColorSample)<=COLOR_DIST_THRESHOLD/2 && hdist(vInputIntraDesc.r,nCurrBGDescSample)<=DESC_DIST_THRESHOLD)\n"
#else //!LOBSTER_GLSL_BASIC
             "            if(absdiff(vInputColor.r,nCurrBGColorSample)<=COLOR_DIST_THRESHOLD/2 && hdist(lbsp(uvec4(anLBSPThresLUT[nCurrBGColorSample]),uvec4(nCurrBGColorSample),mInput,vImgCoords).r,nCurrBGDescSample)<=DESC_DIST_THRESHOLD)\n";
#endif //!LOBSTER_GLSL_BASIC
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
#if LOBSTER_GLSL_DEBUG
             //"    barrier();\n"
             "    if(bool(nROIVal)) {\n"
             "        vec4 vAvgBGColor = vec4(0);\n"
             "        for(uint nSampleIdx=0; nSampleIdx<NB_SAMPLES; ++nSampleIdx)\n"
             "            vAvgBGColor += vec4(aoPxModels[nModelIdx].color_samples[nSampleIdx]);\n"
             "        imageStore(mDebug,vImgCoords,uvec4(vAvgBGColor/NB_SAMPLES));\n"
             //"        imageStore(mDebug,vImgCoords,aoPxModels[nModelIdx].color_samples[0]);\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(nResamplingRate*15));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(128));\n"
             //"        imageStore(mDebug,vImgCoords,vec4(0.5));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(nSampleIdx*(255/NB_SAMPLES)));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(hdist(uvec3(0),vInputIntraDesc)*15));\n"
             "    }\n"
#endif //LOBSTER_GLSL_DEBUG
             "    imageStore(mOutput,vImgCoords,vSegmResult);\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

void BackgroundSubtractorLOBSTER::dispatch(size_t nStage, GLShader& oShader) {
    glAssert(nStage<m_nComputeStages);
    oShader.setUniform1ui("nResamplingRate",m_nCurrResamplingRate);
    glDispatchCompute((GLuint)ceil((float)m_oFrameSize.width/m_vDefaultWorkGroupSize.x),(GLuint)ceil((float)m_oFrameSize.height/m_vDefaultWorkGroupSize.y),1);
}

void BackgroundSubtractorLOBSTER::getBackgroundImage(cv::OutputArray oBGImg) const {
    CV_Assert(m_bInitialized);
    glAssert(m_bGLInitialized && !m_vnBGModelData.empty());
    oBGImg.create(m_oFrameSize,CV_8UC(m_nImgChannels));
    cv::Mat oOutputImg = oBGImg.getMatRef();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,getSSBOId(BackgroundSubtractorLOBSTER::eBuffer_BGModelBinding));
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,0,m_nBGModelSize*sizeof(uint),(void*)m_vnBGModelData.data());
    glErrorCheck;
    for(size_t nRowIdx=0; nRowIdx<(size_t)m_oFrameSize.height; ++nRowIdx) {
        const size_t nModelRowOffset = nRowIdx*m_nRowStepSize;
        const size_t nImgRowOffset = nRowIdx*oOutputImg.step.p[0];
        for(size_t nColIdx=0; nColIdx<(size_t)m_oFrameSize.width; ++nColIdx) {
            const size_t nModelColOffset = nColIdx*m_nColStepSize+nModelRowOffset;
            const size_t nImgColOffset = nColIdx*oOutputImg.step.p[1]+nImgRowOffset;
            float afCurrPxSum[4] = {0.0f,0.0f,0.0f,0.0f};
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

void BackgroundSubtractorLOBSTER::getBackgroundDescriptorsImage(cv::OutputArray oBGDescImg) const {
    CV_Assert(LBSP::DESC_SIZE==2);
    CV_Assert(m_bInitialized);
    glAssert(m_bGLInitialized);
    oBGDescImg.create(m_oFrameSize,CV_16UC(m_nImgChannels));
    glAssert(false); // @@@@ missing impl
}

#else //!HAVE_GLSL

void BackgroundSubtractorLOBSTER::apply(cv::InputArray _oInputImg, cv::OutputArray _oFGMask, double dLearningRate) {
    // == process_sync
    CV_Assert(m_bInitialized && m_bModelInitialized);
    CV_Assert(dLearningRate>0);
    cv::Mat oInputImg = _oInputImg.getMat();
    CV_Assert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
    CV_Assert(oInputImg.isContinuous());
    _oFGMask.create(m_oImgSize,CV_8UC1);
    cv::Mat oCurrFGMask = _oFGMask.getMat();
    oCurrFGMask = cv::Scalar_<uchar>(0);
    const size_t nLearningRate = (size_t)ceil(dLearningRate);
    if(m_nImgChannels==1) {
        for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
            const size_t nPxIter = m_vnPxIdxLUT[nModelIter];
            const size_t nDescIter = nPxIter*2;
            const int nCurrImgCoord_X = m_voPxInfoLUT[nPxIter].nImgCoord_X;
            const int nCurrImgCoord_Y = m_voPxInfoLUT[nPxIter].nImgCoord_Y;
            const uchar nCurrColor = oInputImg.data[nPxIter];
            size_t nGoodSamplesCount=0, nModelIdx=0;
            ushort nCurrInputDesc;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nModelIdx<m_nBGSamples) {
                const uchar nBGColor = m_voBGColorSamples[nModelIdx].data[nPxIter];
                {
                    const size_t nColorDist = L1dist(nCurrColor,nBGColor);
                    if(nColorDist>m_nColorDistThreshold/2)
                        goto failedcheck1ch;
                    LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInputDesc);
                    const size_t nDescDist = hdist(nCurrInputDesc,*((ushort*)(m_voBGDescSamples[nModelIdx].data+nDescIter)));
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
                    LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nCurrColor],nRandInputDesc);
                    m_voBGColorSamples[nSampleModelIdx].data[nPxIter] = nCurrColor;
                }
                if((rand()%nLearningRate)==0) {
                    int nSampleImgCoord_Y, nSampleImgCoord_X;
                    getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                    const size_t nSampleModelIdx = rand()%m_nBGSamples;
                    ushort& nRandInputDesc = m_voBGDescSamples[nSampleModelIdx].at<ushort>(nSampleImgCoord_Y,nSampleImgCoord_X);
                    LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nCurrColor],nRandInputDesc);
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
            size_t nGoodSamplesCount=0, nModelIdx=0;
            ushort anCurrInputDesc[3];
            while(nGoodSamplesCount<m_nRequiredBGSamples && nModelIdx<m_nBGSamples) {
                const ushort* const anBGDesc = (ushort*)(m_voBGDescSamples[nModelIdx].data+nDescIterRGB);
                const uchar* const anBGColor = m_voBGColorSamples[nModelIdx].data+nPxIterRGB;
                size_t nTotColorDist = 0;
                size_t nTotDescDist = 0;
                for(size_t c=0;c<3; ++c) {
                    const size_t nColorDist = L1dist(anCurrColor[c],anBGColor[c]);
                    if(nColorDist>nCurrSCColorDistThreshold)
                        goto failedcheck3ch;
                    LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],nCurrImgCoord_X,nCurrImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInputDesc[c]);
                    const size_t nDescDist = hdist(anCurrInputDesc[c],anBGDesc[c]);
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
                    const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
                    LBSP::computeRGBDescriptor(oInputImg,anCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,anCurrIntraLBSPThresholds,anRandInputDesc);
                    for(size_t c=0; c<3; ++c)
                        *(m_voBGColorSamples[nSampleModelIdx].data+nPxIterRGB+c) = anCurrColor[c];
                }
                if((rand()%nLearningRate)==0) {
                    int nSampleImgCoord_Y, nSampleImgCoord_X;
                    getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                    const size_t nSampleModelIdx = rand()%m_nBGSamples;
                    ushort* anRandInputDesc = ((ushort*)(m_voBGDescSamples[nSampleModelIdx].data + desc_row_step*nSampleImgCoord_Y + 6*nSampleImgCoord_X));
                    const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
                    LBSP::computeRGBDescriptor(oInputImg,anCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,anCurrIntraLBSPThresholds,anRandInputDesc);
                    for(size_t c=0; c<3; ++c)
                        *(m_voBGColorSamples[nSampleModelIdx].data + img_row_step*nSampleImgCoord_Y + 3*nSampleImgCoord_X + c) = anCurrColor[c];
                }
            }
        }
    }
    cv::medianBlur(oCurrFGMask,m_oLastFGMask,m_nDefaultMedianBlurKernelSize);
    m_oLastFGMask.copyTo(oCurrFGMask);
}

void BackgroundSubtractorLOBSTER::getBackgroundImage(cv::OutputArray oBGImg) const {
    CV_Assert(m_bInitialized);
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
    CV_Assert(LBSP::DESC_SIZE==2);
    CV_Assert(m_bInitialized);
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

#endif //!HAVE_GLSL
