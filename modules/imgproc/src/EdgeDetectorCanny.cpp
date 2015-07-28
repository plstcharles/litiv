#include "litiv/imgproc/EdgeDetectorCanny.hpp"

EdgeDetectorCanny::EdgeDetectorCanny(size_t nDefaultBaseHystThreshold, double dHystThresholdMultiplier,
                                     size_t nKernelSize, bool bUseL2GradientNorm) :
        m_nDefaultBaseHystThreshold(nDefaultBaseHystThreshold),
        m_dHystThresholdMultiplier(dHystThresholdMultiplier),
        m_nKernelSize(nKernelSize),
        m_bUsingL2GradientNorm(bUseL2GradientNorm) {
    CV_Assert(nDefaultBaseHystThreshold<UCHAR_MAX);
    CV_Assert(dHystThresholdMultiplier>0);
    CV_Assert(m_nKernelSize>0 && (m_nKernelSize%2)==1);
}

void EdgeDetectorCanny::apply_threshold(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask, double dThreshold) {
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
    cv::GaussianBlur(oInputImage_gray,oInputImage_gray,cv::Size(m_nKernelSize,m_nKernelSize),0,0);
    cv::Canny(oInputImage_gray,oEdgeMask,(double)nCurrBaseHystThreshold,nCurrBaseHystThreshold*m_dHystThresholdMultiplier,m_nKernelSize,m_bUsingL2GradientNorm);
}

void EdgeDetectorCanny::apply(cv::InputArray _oInputImage, cv::OutputArray _oEdgeMask) {
    cv::Mat oInputImg = _oInputImage.getMat();
    CV_Assert(!oInputImg.empty());
    CV_Assert(oInputImg.channels()==1 || oInputImg.channels()==3 || oInputImg.channels()==4);
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    oEdgeMask = cv::Scalar_<uchar>(0);
    cv::Mat oTempEdgeMask = oEdgeMask.clone();
    for(size_t nCurrThreshold=0; nCurrThreshold<UCHAR_MAX; ++nCurrThreshold) {
        apply_threshold(oInputImg,oTempEdgeMask,double(nCurrThreshold)/UCHAR_MAX);
        oEdgeMask += oTempEdgeMask/UCHAR_MAX;
    }
    cv::normalize(oEdgeMask,oEdgeMask,0,UCHAR_MAX,cv::NORM_MINMAX);
}
