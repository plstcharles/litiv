#include "litiv/imgproc/EdgeDetectorCanny.hpp"

EdgeDetectorCanny::EdgeDetectorCanny(double dHystLowThrshFactor, double dGaussianKernelSigma, bool bUseL2GradientNorm) :
        m_dHystLowThrshFactor(dHystLowThrshFactor),
        m_dGaussianKernelSigma(dGaussianKernelSigma),
        m_bUsingL2GradientNorm(bUseL2GradientNorm) {
    CV_Assert(m_dHystLowThrshFactor>0 && m_dHystLowThrshFactor<1);
    CV_Assert(m_dGaussianKernelSigma>=0);
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
    if(m_dGaussianKernelSigma>0) {
        const int nDefaultKernelSize = int(8*ceil(m_dGaussianKernelSigma));
        const int nRealKernelSize = nDefaultKernelSize%2==0?nDefaultKernelSize+1:nDefaultKernelSize;
        cv::GaussianBlur(oInputImage_gray,oInputImage_gray,cv::Size(nRealKernelSize,nRealKernelSize),m_dGaussianKernelSigma,m_dGaussianKernelSigma);
    }
    _oEdgeMask.create(oInputImg.size(),CV_8UC1);
    cv::Mat oEdgeMask = _oEdgeMask.getMat();
    if(dThreshold<0||dThreshold>1)
        dThreshold = getDefaultThreshold();
    const size_t nCurrBaseHystThreshold = (size_t)(dThreshold*UCHAR_MAX);
    cv::Canny(oInputImage_gray,oEdgeMask,nCurrBaseHystThreshold*m_dHystLowThrshFactor,(double)nCurrBaseHystThreshold,3,m_bUsingL2GradientNorm);
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
