
#include <opencv2/imgproc.hpp>
#include "litiv/3rdparty/ofdis/ofdis.hpp"
#include "litiv/3rdparty/ofdis/oflow.hpp"

inline int autoSelectFirstScale(int imgwidth, int fratio, int patchsize) {
    return std::max(0,(int)std::floor(log2((2.0f*(float)imgwidth)/((float)fratio*(float)patchsize))));
}

ofdis::FlowParams::FlowParams(int sel_oppoint_) {
    mindprate = 0.05;
    mindrrate = 0.95;
    minimgerr = 0.0;
    usefbcon = 0;
    patnorm = 1;
    costfct = 0;
    tv_alpha = 10.0;
    tv_gamma = 10.0;
    tv_delta = 5.0;
    tv_innerit = 1;
    tv_solverit = 3;
    tv_sor = 1.6;
    verbosity = 2; // Default: Plot detailed timings
    sel_oppoint = sel_oppoint_;
}

void ofdis::FlowParams::setOpPointParams(const cv::Size& oImageSize) {
    int fratio = 5; // For automatic selection of coarsest scale: 1/fratio * width = maximum expected motion magnitude in image. Set lower to restrict search space.
    switch(sel_oppoint) {
        case 1:
            patchsz = 8;
            poverl = 0.3;
            lv_f = autoSelectFirstScale(oImageSize.width,fratio,patchsz);
            lv_l = std::max(lv_f-2,0);
            maxiter = 16;
            miniter = 16;
            usetvref = 0;
            break;
        case 3:
            patchsz = 12;
            poverl = 0.75;
            lv_f = autoSelectFirstScale(oImageSize.width,fratio,patchsz);
            lv_l = std::max(lv_f-4,0);
            maxiter = 16;
            miniter = 16;
            usetvref = 1;
            break;
        case 4:
            patchsz = 12;
            poverl = 0.75;
            lv_f = autoSelectFirstScale(oImageSize.width,fratio,patchsz);
            lv_l = std::max(lv_f-5,0);
            maxiter = 128;
            miniter = 128;
            usetvref = 1;
            break;
        case 2:
        default:
            patchsz = 8;
            poverl = 0.4;
            lv_f = autoSelectFirstScale(oImageSize.width,fratio,patchsz);
            lv_l = std::max(lv_f-2,0);
            maxiter = 12;
            miniter = 12;
            usetvref = 1;
            break;
    }
}

template<ofdis::FlowInputType eInput>
void inline constructImgPyramid(const cv::Mat & img_ao_fmat,
                                std::vector<cv::Mat>& img_ao_fmat_pyr,
                                std::vector<cv::Mat>& img_ao_dx_fmat_pyr,
                                std::vector<cv::Mat>& img_ao_dy_fmat_pyr,
                                std::vector<const float*>& img_ao_pyr,
                                std::vector<const float*>& img_ao_dx_pyr,
                                std::vector<const float*>& img_ao_dy_pyr,
                                const int lv_f,
                                const int lv_l,
                                const int rpyrtype,
                                const bool getgrad,
                                const int imgpadding,
                                const int padw,
                                const int padh) {
    for(int i=0; i<=lv_f; ++i) { // Construct image and gradient pyramides
        if(i==0) { // At finest scale: copy directly, for all other: downscale previous scale by .5
            if(eInput==ofdis::FlowInput_Gradient) {
                cv::Mat dx,dy,dx2,dy2,dmag;
                cv::Sobel( img_ao_fmat, dx, CV_32F, 1, 0, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
                cv::Sobel( img_ao_fmat, dy, CV_32F, 0, 1, 3, 1/8.0, 0, cv::BORDER_DEFAULT );
                dx2 = dx.mul(dx);
                dy2 = dy.mul(dy);
                dmag = dx2+dy2;
                cv::sqrt(dmag,dmag);
                img_ao_fmat_pyr[i] = dmag.clone();
            }
            else
                img_ao_fmat.copyTo(img_ao_fmat_pyr[i]);
        }
        else
            cv::resize(img_ao_fmat_pyr[i-1],img_ao_fmat_pyr[i],cv::Size(),.5,.5,cv::INTER_LINEAR);
        img_ao_fmat_pyr[i].convertTo(img_ao_fmat_pyr[i],rpyrtype);
        if(getgrad) {
            cv::Sobel(img_ao_fmat_pyr[i],img_ao_dx_fmat_pyr[i],CV_32F,1,0,3,1/8.0,0,cv::BORDER_DEFAULT);
            cv::Sobel(img_ao_fmat_pyr[i],img_ao_dy_fmat_pyr[i],CV_32F,0,1,3,1/8.0,0,cv::BORDER_DEFAULT);
            img_ao_dx_fmat_pyr[i].convertTo(img_ao_dx_fmat_pyr[i],CV_32F);
            img_ao_dy_fmat_pyr[i].convertTo(img_ao_dy_fmat_pyr[i],CV_32F);
        }
    }
    // pad images
    for(int i=0; i<=lv_f; ++i) { // Construct image and gradient pyramides
        copyMakeBorder(img_ao_fmat_pyr[i],img_ao_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_REPLICATE);  // Replicate border for image padding
        img_ao_pyr[i] = (float*)img_ao_fmat_pyr[i].data;
        if(getgrad) {
            copyMakeBorder(img_ao_dx_fmat_pyr[i],img_ao_dx_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT,0); // Zero padding for gradients
            copyMakeBorder(img_ao_dy_fmat_pyr[i],img_ao_dy_fmat_pyr[i],imgpadding,imgpadding,imgpadding,imgpadding,cv::BORDER_CONSTANT,0);
            img_ao_dx_pyr[i] = (float*)img_ao_dx_fmat_pyr[i].data;
            img_ao_dy_pyr[i] = (float*)img_ao_dy_fmat_pyr[i].data;
        }
    }
}

template<ofdis::FlowInputType eInput, ofdis::FlowOutputType eOutput>
void ofdis::computeFlow(const cv::Mat& oInput1, const cv::Mat& oInput2, cv::Mat& oOutput, FlowParams oParams) {
    CV_Assert(!oInput1.empty() && !oInput2.empty());
    const int rpyrtype = (eInput==FlowInput_RGB)?CV_32FC3:CV_32FC1;
    const int nochannels = (eInput==FlowInput_RGB)?3:1;
    CV_Assert(oInput1.channels()==nochannels && oInput2.channels()==nochannels);
    cv::Size sz = oInput1.size();
    if(oParams.sel_oppoint>=1)
        oParams.setOpPointParams(sz);
    const int width_org = sz.width; // unpadded original image size
    const int height_org = sz.height; // unpadded original image size
    // pad image such that width and height are restless divisible on all scales (except last)
    const int scfct = 1<<oParams.lv_f; // enforce restless division by this number on coarsest scale
    const int padw = (scfct-(width_org%scfct))%scfct;
    const int padh = (scfct-(height_org%scfct))%scfct;
    static thread_local cv::Mat img_ao_fmat,img_bo_fmat;
    if(padh>0 || padw>0) {
        static thread_local cv::Mat s_oEnlargedInput1,s_oEnlargedInput2;
        copyMakeBorder(oInput1,s_oEnlargedInput1,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
        copyMakeBorder(oInput2,s_oEnlargedInput2,floor((float)padh/2.0f),ceil((float)padh/2.0f),floor((float)padw/2.0f),ceil((float)padw/2.0f),cv::BORDER_REPLICATE);
        s_oEnlargedInput1.convertTo(img_ao_fmat,CV_32F);
        s_oEnlargedInput2.convertTo(img_bo_fmat,CV_32F);
        sz = oInput1.size();
    }
    else {
        oInput1.convertTo(img_ao_fmat,CV_32F);
        oInput2.convertTo(img_bo_fmat,CV_32F);
    }
    std::vector<const float*>img_ao_pyr((size_t)oParams.lv_f+1);
    std::vector<const float*>img_bo_pyr((size_t)oParams.lv_f+1);
    std::vector<const float*>img_ao_dx_pyr((size_t)oParams.lv_f+1);
    std::vector<const float*>img_ao_dy_pyr((size_t)oParams.lv_f+1);
    std::vector<const float*>img_bo_dx_pyr((size_t)oParams.lv_f+1);
    std::vector<const float*>img_bo_dy_pyr((size_t)oParams.lv_f+1);
    std::vector<cv::Mat> img_ao_fmat_pyr((size_t)oParams.lv_f+1);
    std::vector<cv::Mat> img_bo_fmat_pyr((size_t)oParams.lv_f+1);
    std::vector<cv::Mat> img_ao_dx_fmat_pyr((size_t)oParams.lv_f+1);
    std::vector<cv::Mat> img_ao_dy_fmat_pyr((size_t)oParams.lv_f+1);
    std::vector<cv::Mat> img_bo_dx_fmat_pyr((size_t)oParams.lv_f+1);
    std::vector<cv::Mat> img_bo_dy_fmat_pyr((size_t)oParams.lv_f+1);
    constructImgPyramid<eInput>(img_ao_fmat,img_ao_fmat_pyr,img_ao_dx_fmat_pyr,img_ao_dy_fmat_pyr,img_ao_pyr,img_ao_dx_pyr,img_ao_dy_pyr,oParams.lv_f,oParams.lv_l,rpyrtype,true,oParams.patchsz,padw,padh);
    constructImgPyramid<eInput>(img_bo_fmat,img_bo_fmat_pyr,img_bo_dx_fmat_pyr,img_bo_dy_fmat_pyr,img_bo_pyr,img_bo_dx_pyr,img_bo_dy_pyr,oParams.lv_f,oParams.lv_l,rpyrtype,true,oParams.patchsz,padw,padh);
    const int scfct2 = 1<<oParams.lv_l;
    oOutput.create(sz.height/scfct2,sz.width/scfct2,(eOutput==FlowOutput_OpticalFlow)?CV_32FC2:CV_32FC1);
    OFClass<eInput,eOutput> ofc(img_ao_pyr,img_ao_dx_pyr,img_ao_dy_pyr,img_bo_pyr,img_bo_dx_pyr,img_bo_dy_pyr,oParams.patchsz,  // extra image padding to avoid border violation check
                                (float*)oOutput.data,   // pointer to n-band output float array
                                nullptr,  // pointer to n-band input float array of size of first (coarsest) scale, pass as nullptr to disable
                                sz.width,sz.height,oParams.lv_f,oParams.lv_l,oParams.maxiter,oParams.miniter,oParams.mindprate,
                                oParams.mindrrate,oParams.minimgerr,oParams.patchsz,oParams.poverl,oParams.usefbcon,oParams.costfct,
                                nochannels,oParams.patnorm,oParams.usetvref,oParams.tv_alpha,oParams.tv_gamma,oParams.tv_delta,
                                oParams.tv_innerit,oParams.tv_solverit,oParams.tv_sor,oParams.verbosity);
    if(oParams.lv_l!=0) {
        oOutput *= scfct2;
        cv::resize(oOutput,oOutput,cv::Size(),scfct2,scfct2,cv::INTER_LINEAR);
    }
    oOutput = oOutput(cv::Rect((int)floor((float)padw/2.0f),(int)floor((float)padh/2.0f),width_org,height_org));
}

template void ofdis::computeFlow<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_OpticalFlow>(const cv::Mat&,const cv::Mat&,cv::Mat&,FlowParams);
template void ofdis::computeFlow<ofdis::FlowInput_Gradient,ofdis::FlowOutput_OpticalFlow>(const cv::Mat&,const cv::Mat&,cv::Mat&,FlowParams);
template void ofdis::computeFlow<ofdis::FlowInput_RGB,ofdis::FlowOutput_OpticalFlow>(const cv::Mat&,const cv::Mat&,cv::Mat&,FlowParams);
template void ofdis::computeFlow<ofdis::FlowInput_Grayscale,ofdis::FlowOutput_StereoDepth>(const cv::Mat&,const cv::Mat&,cv::Mat&,FlowParams);
template void ofdis::computeFlow<ofdis::FlowInput_Gradient,ofdis::FlowOutput_StereoDepth>(const cv::Mat&,const cv::Mat&,cv::Mat&,FlowParams);
template void ofdis::computeFlow<ofdis::FlowInput_RGB,ofdis::FlowOutput_StereoDepth>(const cv::Mat&,const cv::Mat&,cv::Mat&,FlowParams);