#include "litiv/video/BackgroundSubtractorLBSP.hpp"

// local define used to determine the default median blur kernel size
#define DEFAULT_MEDIAN_BLUR_KERNEL_SIZE (9)

template<>
template<>
BackgroundSubtractorLBSP<ParallelUtils::eNonParallel>::BackgroundSubtractorLBSP<ParallelUtils::eNonParallel>(float fRelLBSPThreshold, size_t nLBSPThresholdOffset, void* /*pUnused*/) :
        ::BackgroundSubtractor(LBSP::PATCH_SIZE/2),
        m_nLBSPThresholdOffset(nLBSPThresholdOffset),
        m_fRelLBSPThreshold(fRelLBSPThreshold),
        m_nDefaultMedianBlurKernelSize(DEFAULT_MEDIAN_BLUR_KERNEL_SIZE) {
    CV_Assert(m_fRelLBSPThreshold>=0);
}

#if HAVE_GLSL

template<>
template<>
BackgroundSubtractorLBSP<ParallelUtils::eGLSL>::BackgroundSubtractorLBSP<ParallelUtils::eGLSL>( float fRelLBSPThreshold, size_t nLBSPThresholdOffset, size_t nLevels, size_t nComputeStages,
                                                                                                size_t nExtraSSBOs, size_t nExtraACBOs, size_t nExtraImages, size_t nExtraTextures,
                                                                                                int nDebugType, bool bUseDisplay, bool bUseTimers, bool bUseIntegralFormat, void* /*pUnused*/) :
    BackgroundSubtractor_GLSL(nLevels,nComputeStages,nExtraSSBOs,nExtraACBOs,nExtraImages,nExtraTextures,nDebugType,bUseDisplay,bUseTimers,bUseIntegralFormat),
    m_nLBSPThresholdOffset(nLBSPThresholdOffset),
    m_fRelLBSPThreshold(fRelLBSPThreshold),
    m_nDefaultMedianBlurKernelSize(DEFAULT_MEDIAN_BLUR_KERNEL_SIZE) {
    CV_Assert(m_fRelLBSPThreshold>=0);
}

template<>
template<>
std::string BackgroundSubtractorLBSP<ParallelUtils::eGLSL>::getLBSPThresholdLUTShaderSource<ParallelUtils::eGLSL>() const {
    glAssert(m_bInitialized);
    std::stringstream ssSrc;
    ssSrc << "const uint anLBSPThresLUT[256] = uint[256](\n    ";
    for(size_t t=0; t<=UCHAR_MAX; ++t) {
        if(t>0 && (t%((UCHAR_MAX+1)/8))==(((UCHAR_MAX+1)/8)-1) && t<UCHAR_MAX)
            ssSrc << m_anLBSPThreshold_8bitLUT[t] << ",\n    ";
        else if(t<UCHAR_MAX)
            ssSrc << m_anLBSPThreshold_8bitLUT[t] << ",";
        else
            ssSrc << m_anLBSPThreshold_8bitLUT[t] << "\n";
    }
    ssSrc << ");\n";
    return ssSrc.str();
}

template class BackgroundSubtractorLBSP<ParallelUtils::eGLSL>;

#endif //HAVE_GLSL

#if HAVE_CUDA
// ... @@@ add impl later
//template class BackgroundSubtractorLBSP<ParallelUtils::eCUDA>;
#endif //HAVE_CUDA

#if HAVE_OPENCL
// ... @@@ add impl later
//template class BackgroundSubtractorLBSP<ParallelUtils::eOpenCL>;
#endif //HAVE_OPENCL

template<ParallelUtils::eParallelAlgoType eImpl>
void BackgroundSubtractorLBSP<eImpl>::initialize(const cv::Mat& oInitImg, const cv::Mat& oROI) {
    ::BackgroundSubtractor_<eImpl>::initialize(oInitImg,oROI);
    m_oLastDescFrame.create(this->m_oImgSize,CV_16UC((int)this->m_nImgChannels));
    m_oLastDescFrame = cv::Scalar_<ushort>::all(0);
    if(this->m_nImgChannels==1) {
        CV_Assert(m_oLastDescFrame.step.p[0]==this->m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==this->m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>((t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset)/3);
        for(size_t nPxIter=0; nPxIter<this->m_nTotPxCount; ++nPxIter) {
            if(this->m_oROI.data[nPxIter]) {
                const size_t nDescIter = nPxIter*2;
                LBSP::computeDescriptor<1>(oInitImg,oInitImg.data[nPxIter],this->m_voPxInfoLUT[nPxIter].nImgCoord_X,this->m_voPxInfoLUT[nPxIter].nImgCoord_Y,0,m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxIter]],*((ushort*)(m_oLastDescFrame.data+nDescIter)));
            }
        }
    }
    else { //(m_nImgChannels==3 || m_nImgChannels==4)
        CV_Assert(m_oLastDescFrame.step.p[0]==this->m_oLastColorFrame.step.p[0]*2 && m_oLastDescFrame.step.p[1]==this->m_oLastColorFrame.step.p[1]*2);
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            m_anLBSPThreshold_8bitLUT[t] = cv::saturate_cast<uchar>(t*m_fRelLBSPThreshold+m_nLBSPThresholdOffset);
        for(size_t nPxIter=0; nPxIter<this->m_nTotPxCount; ++nPxIter) {
            if(this->m_oROI.data[nPxIter]) {
                const size_t nPxRGBIter = nPxIter*this->m_nImgChannels;
                const size_t nDescRGBIter = nPxRGBIter*2;
                if(this->m_nImgChannels==3) {
                    alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,3> aanLBSPLookupVals;
                    LBSP::computeDescriptor_lookup(oInitImg,this->m_voPxInfoLUT[nPxIter].nImgCoord_X,this->m_voPxInfoLUT[nPxIter].nImgCoord_Y,aanLBSPLookupVals);
                    for(size_t c=0; c<3; ++c)
                        LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],oInitImg.data[nPxRGBIter+c],m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
                }
                else { //m_nImgChannels==4
                    alignas(16) std::array<std::array<uchar,LBSP::DESC_SIZE_BITS>,4> aanLBSPLookupVals;
                    LBSP::computeDescriptor_lookup(oInitImg,this->m_voPxInfoLUT[nPxIter].nImgCoord_X,this->m_voPxInfoLUT[nPxIter].nImgCoord_Y,aanLBSPLookupVals);
                    for(size_t c=0; c<4; ++c)
                        LBSP::computeDescriptor_threshold(aanLBSPLookupVals[c],oInitImg.data[nPxRGBIter+c],m_anLBSPThreshold_8bitLUT[oInitImg.data[nPxRGBIter+c]],((ushort*)(m_oLastDescFrame.data+nDescRGBIter))[c]);
                }
            }
        }
    }
}

template class BackgroundSubtractorLBSP<ParallelUtils::eNonParallel>;
