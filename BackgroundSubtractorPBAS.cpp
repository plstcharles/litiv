#include "BackgroundSubtractorPBAS.h"

BackgroundSubtractorPBAS::BackgroundSubtractorPBAS(  size_t nInitColorDistThreshold
                                                    ,float fInitUpdateRate
                                                    ,size_t nBGSamples
                                                    ,size_t nRequiredBGSamples)
    :    m_nBGSamples(nBGSamples)
        ,m_nRequiredBGSamples(nRequiredBGSamples)
        ,m_voBGImg(nBGSamples)
        ,m_voBGGrad(nBGSamples)
        ,m_nDefaultColorDistThreshold(nInitColorDistThreshold)
        ,m_fDefaultUpdateRate(fInitUpdateRate)
        ,m_fFormerMeanGradDist(20)
        ,m_bInitialized(false) {
    CV_Assert(m_nRequiredBGSamples<=m_nBGSamples);
    CV_Assert(m_fDefaultUpdateRate>0 && m_fDefaultUpdateRate<=UCHAR_MAX);
}

BackgroundSubtractorPBAS::~BackgroundSubtractorPBAS() {}

void BackgroundSubtractorPBAS::getBackgroundImage(cv::OutputArray backgroundImage) const {
    CV_DbgAssert(m_bInitialized);
    cv::Mat oAvgBGImg = cv::Mat::zeros(m_oImgSize,CV_32FC(m_voBGImg[0].channels()));
    for(size_t n=0; n<m_voBGImg.size(); ++n) {
        for(int y=0; y<m_oImgSize.height; ++y) {
            for(int x=0; x<m_oImgSize.width; ++x) {
                const size_t idx_uchar = m_voBGImg[n].step.p[0]*y + m_voBGImg[n].step.p[1]*x;
                const size_t idx_flt32 = idx_uchar*4;
                float* oAvgBgImgPtr = (float*)(oAvgBGImg.data+idx_flt32);
                const uchar* const oBGImgPtr = m_voBGImg[n].data+idx_uchar;
                for(size_t c=0; c<(size_t)m_voBGImg[n].channels(); ++c)
                    oAvgBgImgPtr[c] += ((float)oBGImgPtr[c])/m_voBGImg.size();
            }
        }
    }
    oAvgBGImg.convertTo(backgroundImage,CV_8U);
}
