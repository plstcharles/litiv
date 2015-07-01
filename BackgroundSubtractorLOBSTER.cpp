#include "BackgroundSubtractorLOBSTER.h"
#include "DistanceUtils.h"
#include "RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

#if HAVE_GLSL
#define LOBSTER_GLSL_DEBUG       0
#define LOBSTER_GLSL_TIMERS      0
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
        ,GLImageProcAlgo(1,2,1,CV_8UC1,LOBSTER_GLSL_DEBUG?CV_8UC4:-1,CV_8UC4,GLUTILS_IMGPROC_USE_DOUBLE_PBO_OUTPUT,LOBSTER_GLSL_DEBUG?GLUTILS_IMGPROC_USE_DOUBLE_PBO_OUTPUT:0,GLUTILS_IMGPROC_USE_DOUBLE_PBO_INPUT,GLUTILS_IMGPROC_USE_TEXTURE_ARRAYS,GLSL_RENDERING,LOBSTER_GLSL_TIMERS,GLUTILS_IMGPROC_USE_INTEGER_TEX_FORMAT)
#endif //HAVE_GLSL
        ,m_nColorDistThreshold(nColorDistThreshold)
        ,m_nDescDistThreshold(nDescDistThreshold)
        ,m_nBGSamples(nBGSamples)
        ,m_nRequiredBGSamples(nRequiredBGSamples)
        ,m_nResamplingRate(BGSLOBSTER_DEFAULT_LEARNING_RATE)
        ,m_bModelInitialized(false)
#if HAVE_GLSL
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
    CV_Assert(m_nResamplingRate>0);
    m_bAutoModelResetEnabled = false; // @@@@@@ not supported here for now
#if HAVE_GLSL
    glAssert(GLUtils::getIntegerVal<1>(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS)>=2);
    glErrorCheck;
#endif //HAVE_GLSL
}

BackgroundSubtractorLOBSTER::~BackgroundSubtractorLOBSTER() {}

void BackgroundSubtractorLOBSTER::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    // == init
    BackgroundSubtractorLBSP::initialize(oInitImg,oROI);
    m_bModelInitialized = false;
    m_vnPxIdxLUT.resize(m_nTotRelevantPxCount);
    m_voPxInfoLUT.resize(m_nTotPxCount);
    if(m_nImgChannels==1) {
        CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width && m_oLastColorFrame.step.p[1]==1);
        CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset)/2);
        for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
            if(m_oROI.data[nPxIter]) {
                m_vnPxIdxLUT[nModelIter] = nPxIter;
                m_voPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nModelIdx = nModelIter;
                m_oLastColorFrame.data[nPxIter] = oInitImg.data[nPxIter];
                const size_t nDescIter = nPxIter*2;
                LBSP::computeGrayscaleDescriptor(oInitImg,oInitImg.data[nPxIter],m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxIter]],*((ushort*)(m_oLastDescFrame.data+nDescIter)));
                ++nModelIter;
            }
        }
    }
    else { //(m_nImgChannels==3 || m_nImgChannels==4)
        CV_Assert(m_oLastColorFrame.step.p[0]==(size_t)m_oImgSize.width*m_nImgChannels && m_oLastColorFrame.step.p[1]==m_nImgChannels);
        CV_Assert(m_oLastDescFrame.step.p[0]==m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset);
        for(size_t nPxIter=0, nModelIter=0; nPxIter<m_nTotPxCount; ++nPxIter) {
            if(m_oROI.data[nPxIter]) {
                m_vnPxIdxLUT[nModelIter] = nPxIter;
                m_voPxInfoLUT[nPxIter].nImgCoord_Y = (int)nPxIter/m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nImgCoord_X = (int)nPxIter%m_oImgSize.width;
                m_voPxInfoLUT[nPxIter].nModelIdx = nModelIter;
                const size_t nPxRGBIter = nPxIter*m_nImgChannels;
                const size_t nDescRGBIter = nPxRGBIter*2;
                for(size_t c=0; c<m_nImgChannels; ++c) {
                    m_oLastColorFrame.data[nPxRGBIter+c] = oInitImg.data[nPxRGBIter+c];
                    if(m_nImgChannels==3)
                        LBSP::computeSingleRGBDescriptor(oInitImg,oInitImg.data[nPxRGBIter+c],m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
                    else //m_nImgChannels==4
                        LBSP::computeSingleRGBADescriptor(oInitImg,oInitImg.data[nPxRGBIter+c],m_voPxInfoLUT[nPxIter].nImgCoord_X,m_voPxInfoLUT[nPxIter].nImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
                }
                ++nModelIter;
            }
        }
    }
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
#endif //!HAVE_GLSL
    refreshModel(1.0f);
    m_bModelInitialized = true;
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
    int nSampleRowIdx, nSampleColIdx;
    ushort nLBSPDesc;
    for(size_t nRowIdx=0; nRowIdx<(size_t)m_oFrameSize.height; ++nRowIdx) {
        const size_t nModelRowOffset = nRowIdx*m_nRowStepSize;
        for(size_t nColIdx=0; nColIdx<(size_t)m_oFrameSize.width; ++nColIdx) {
            const size_t nModelColOffset = nColIdx*m_nColStepSize+nModelRowOffset;
            for(size_t nSampleIdx=0; nSampleIdx<m_nBGSamples; ++nSampleIdx) {
                getRandNeighborPosition_3x3(nSampleColIdx,nSampleRowIdx,nColIdx,nRowIdx,LBSP::PATCH_SIZE/2,m_oFrameSize);
                const size_t nSampleOffset = nSampleColIdx*m_oLastColorFrame.step.p[1]+(nSampleRowIdx*m_oLastColorFrame.step.p[0]);
                const size_t nModelPxOffset_color = nSampleIdx*m_nSampleStepSize+nModelColOffset;
                const size_t nModelPxOffset_lbsp = nModelPxOffset_color+(m_nBGSamples*m_nSampleStepSize);
                for(size_t nChannelIdx=0; nChannelIdx<m_nImgChannels; ++nChannelIdx) {
                    const size_t nModelTotOffset_color = nChannelIdx+nModelPxOffset_color;
                    const size_t nModelTotOffset_lbsp = nChannelIdx+nModelPxOffset_lbsp;
                    const size_t nSampleChannelIdx = ((nChannelIdx==3||m_nImgChannels==1)?nChannelIdx:2-nChannelIdx);
                    const size_t nSampleTotOffset = nSampleChannelIdx+nSampleOffset;
                    m_vnBGModelData[nModelTotOffset_color] = (uint)m_oLastColorFrame.data[nSampleTotOffset];
                    LBSP::computeSingleRGBADescriptor(m_oLastColorFrame,m_oLastColorFrame.data[nSampleTotOffset],nSampleColIdx,nSampleRowIdx,nSampleChannelIdx,m_anLBSPThreshold_8bitLUT[m_oLastColorFrame.data[nSampleTotOffset]],nLBSPDesc);
                    m_vnBGModelData[nModelTotOffset_lbsp] = (uint)nLBSPDesc;
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
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc << "#version 430\n"
             "#define COLOR_DIST_THRESHOLD   " << (m_nImgChannels==4?m_nColorDistThreshold*3:m_nColorDistThreshold) << "\n"
             "#define DESC_DIST_THRESHOLD    " << (m_nImgChannels==4?m_nDescDistThreshold*3:m_nDescDistThreshold) << "\n"
             "#define NB_SAMPLES             " << m_nBGSamples << "\n"
             "#define NB_REQ_SAMPLES         " << m_nRequiredBGSamples << "\n"
             "#define MODEL_STEP_SIZE        " << m_oFrameSize.width << "\n"
             "struct PxModel {\n"
             "    " << (m_nImgChannels==4?"uvec4":"uint") << " color_samples[" << m_nBGSamples << "];\n"
             "    " << (m_nImgChannels==4?"uvec4":"uint") << " lbsp_samples[" << m_nBGSamples << "];\n";
    if(m_nPxModelPadding>0) ssSrc <<
             "    uint pad[" << m_nPxModelPadding << "];\n";
    ssSrc << "};\n";
    ssSrc << (m_nImgChannels==4?GLSLFunctionUtils::getShaderFunctionSource_L1dist():GLSLFunctionUtils::getShaderFunctionSource_absdiff(true));
    ssSrc << GLSLFunctionUtils::getShaderFunctionSource_absdiff(true);
    ssSrc << GLSLFunctionUtils::getShaderFunctionSource_hdist();
    ssSrc << GLSLFunctionUtils::getShaderFunctionSource_urand_tinymt32();
    ssSrc << GLSLFunctionUtils::getShaderFunctionSource_getRandNeighbor3x3(0,m_oFrameSize);
    ssSrc << "uvec4 lbsp(in uvec4 t, in uvec4 ref, in layout(" << (m_nImgChannels==4?"rgba8ui":"r8ui") << ") readonly uimage2D mData, in ivec2 vCoords) {\n"
             "    return (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2(-1, 1)))-ivec4(ref)),t)) << 15)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 1,-1)))-ivec4(ref)),t)) << 14)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 1, 1)))-ivec4(ref)),t)) << 13)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2(-1,-1)))-ivec4(ref)),t)) << 12)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 1, 0)))-ivec4(ref)),t)) << 11)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 0,-1)))-ivec4(ref)),t)) << 10)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2(-1, 0)))-ivec4(ref)),t)) << 9)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 0, 1)))-ivec4(ref)),t)) << 8)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2(-2,-2)))-ivec4(ref)),t)) << 7)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 2, 2)))-ivec4(ref)),t)) << 6)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 2,-2)))-ivec4(ref)),t)) << 5)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2(-2, 2)))-ivec4(ref)),t)) << 4)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 0, 2)))-ivec4(ref)),t)) << 3)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 0,-2)))-ivec4(ref)),t)) << 2)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2( 2, 0)))-ivec4(ref)),t)) << 1)\n"
             "         + (uvec4(greaterThan(abs(ivec4(imageLoad(mData,vCoords+ivec2(-2, 0)))-ivec4(ref)),t)));\n"
             "}\n";
    ssSrc << "#define urand() urand(aoTMT32Models[nModelIdx])\n"
             "#define dist(a,b) " << (m_nImgChannels==4?"L1dist(a,b)":"absdiff(a,b)") << "\n";
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
    ssSrc << "void main() {\n"
             "    ivec2 vImgCoords = ivec2(gl_GlobalInvocationID.xy);\n"
             "    uvec4 vSegmResult = uvec4(0);\n"
             "    uint nROIVal = imageLoad(mROI,vImgCoords).r;\n"
             "    uint nModelIdx = gl_GlobalInvocationID.y*MODEL_STEP_SIZE + gl_GlobalInvocationID.x;\n"
             "    uint nGoodSamplesCount=0, nSampleIdx=0;\n"
             "    uvec4 vInputColor = imageLoad(mInput,vImgCoords);\n"
             "    uvec4 vInputDesc = lbsp(uvec4(30),vInputColor,mInput,vImgCoords);\n" // preload & memshare input img?
             "    if(bool(nROIVal)) {\n"
             "        while(nSampleIdx<NB_SAMPLES) {\n";
    if(m_nImgChannels==4) { ssSrc <<
             "            uvec4 vCurrColorSample = aoPxModels[nModelIdx].color_samples[nSampleIdx];\n"
             "            uvec4 vCurrDescSample = aoPxModels[nModelIdx].lbsp_samples[nSampleIdx];\n"
             "            if(dist(vInputColor.bgr,vCurrColorSample.bgr)<COLOR_DIST_THRESHOLD && hdist(vInputDesc.bgr,vCurrDescSample.bgr)<DESC_DIST_THRESHOLD)\n"
             "                ++nGoodSamplesCount;\n"
             "            ++nSampleIdx;\n";
    }
    else { ssSrc <<
             "            uint nCurrColorSample = aoPxModels[nModelIdx].color_samples[nSampleIdx];\n"
             "            uint nCurrDescSample = aoPxModels[nModelIdx].lbsp_samples[nSampleIdx];\n"
             "            if(dist(vInputColor.r,nCurrColorSample)<COLOR_DIST_THRESHOLD && hdist(vInputDesc.r,nCurrDescSample)<DESC_DIST_THRESHOLD)\n"
             "                ++nGoodSamplesCount;\n"
             "            ++nSampleIdx;\n";
    }
    ssSrc << "            if(nGoodSamplesCount>=NB_REQ_SAMPLES)\n"
             "                break;\n"
             "        }\n"
             "    }\n"
             "    barrier();\n"
             "    if(bool(nROIVal)) {\n"
             "        if(nGoodSamplesCount<NB_REQ_SAMPLES)\n"
             "            vSegmResult.r = 255;\n"
             "        else {\n"
             "            if((urand()%nResamplingRate)==0) {\n"
             "                aoPxModels[nModelIdx].color_samples[(urand()%NB_SAMPLES)] = vInputColor" << (m_nImgChannels==1?".r;":";") << "\n"
             "                aoPxModels[nModelIdx].lbsp_samples[(urand()%NB_SAMPLES)] = vInputDesc" << (m_nImgChannels==1?".r;":";") << "\n"
             "            }\n"
             "            if((urand()%nResamplingRate)==0) {\n"
             "                ivec2 vNeighbCoords = getRandNeighbor3x3(vImgCoords,urand());\n"
             "                uint nNeighbPxModelIdx = uint(vNeighbCoords.y)*MODEL_STEP_SIZE + uint(vNeighbCoords.x);\n"
             "                aoPxModels[nNeighbPxModelIdx].color_samples[(urand()%NB_SAMPLES)] = vInputColor" << (m_nImgChannels==1?".r;":";") << "\n"
             "                aoPxModels[nNeighbPxModelIdx].lbsp_samples[(urand()%NB_SAMPLES)] = vInputDesc" << (m_nImgChannels==1?".r;":";") << "\n"
             "            }\n"
             "        }\n"
             "    }\n"
             "    barrier();\n"
             "    if(bool(nROIVal)) {\n"
#if LOBSTER_GLSL_DEBUG
             //"        imageStore(mDebug,vImgCoords,uvec4(nResamplingRate*15));\n"
             "        imageStore(mDebug,vImgCoords,aoPxModels[nModelIdx].color_samples[0]);\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(128));\n"
             //"        imageStore(mDebug,vImgCoords,vec4(0.5));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(nSampleIdx*(255/NB_SAMPLES)));\n"
             //"        imageStore(mDebug,vImgCoords,uvec4(hdist(uvec3(0),vInputDesc.bgr)*15));\n"
#endif //LOBSTER_GLSL_DEBUG
             "    }\n"
             "    imageStore(mOutput,vImgCoords,bool(nROIVal)?vSegmResult:uvec4(0));\n"
             "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

void BackgroundSubtractorLOBSTER::dispatch(size_t nStage, GLShader& oShader) {
    glAssert(nStage<m_nComputeStages);
    oShader.setUniform1ui("nResamplingRate",m_nResamplingRate);
    glMemoryBarrier(GL_ALL_BARRIER_BITS); // @@@@@@?
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
