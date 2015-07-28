#include "litiv/imgproc/EdgeDetectorCanny.hpp"

EdgeDetectorCanny::EdgeDetectorCanny(size_t nDefaultBaseHystThreshold, double dHystThresholdMultiplier,
                                     size_t nSobelKernelSize, bool bUseL2GradientNorm) :
        m_nDefaultBaseHystThreshold(nDefaultBaseHystThreshold),
        m_dHystThresholdMultiplier(dHystThresholdMultiplier),
        m_nSobelKernelSize(nSobelKernelSize),
        m_bUsingL2GradientNorm(bUseL2GradientNorm) {
    CV_Assert(nDefaultBaseHystThreshold<UCHAR_MAX);
    CV_Assert(dHystThresholdMultiplier>0);
    CV_Assert(m_nSobelKernelSize>0 && (m_nSobelKernelSize%2)==1);
}

void EdgeDetectorCanny::apply(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dThreshold) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.channels()==1 || oInputImg.channels()==3 || oInputImg.channels()==4);
    cv::Mat oInputImage_gray;
    if(oInputImg.channels()==3)
        cv::cvtColor(oInputImg,oInputImage_gray,cv::COLOR_BGR2GRAY);
    else if(oInputImg.channels()==4)
        cv::cvtColor(oInputImg,oInputImage_gray,cv::COLOR_BGRA2GRAY);
    else
        oInputImage_gray = oInputImg;
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    const size_t nCurrBaseHystThreshold = (dThreshold<0||dThreshold>1)?m_nDefaultBaseHystThreshold:(size_t)(dThreshold*UCHAR_MAX);
    //cv::blur(oInputImage_gray,oInputImage_gray,cv::Size(3,3));
    cv::Canny(oInputImage_gray,oEdgeMask,(double)nCurrBaseHystThreshold,nCurrBaseHystThreshold*m_dHystThresholdMultiplier,m_nSobelKernelSize,m_bUsingL2GradientNorm);
}
