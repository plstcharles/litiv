
#include "litiv/imgproc.hpp"
#include "litiv/test.hpp"

TEST(calcMedianValue,regression) {
    std::vector<uchar> vTestVals0 = {0,1,2,3,4,5,6,7,8,9};
    cv::Mat_<uchar> vTestMat0(1,(int)vTestVals0.size(),vTestVals0.data());
    ASSERT_EQ(lv::calcMedianValue(vTestMat0),5);
    std::vector<uchar> vTestVals1 = {0,1,2,3,4,5,6,7,8,9,10};
    cv::Mat_<uchar> vTestMat1(1,(int)vTestVals1.size(),vTestVals1.data());
    ASSERT_EQ(lv::calcMedianValue(vTestMat1),5);
    std::vector<uchar> vTestVals2 = {0,0,0,0,0,0,1,1,1};
    cv::Mat_<uchar> vTestMat2(1,(int)vTestVals2.size(),vTestVals2.data());
    ASSERT_EQ(lv::calcMedianValue(vTestMat2),0);
    std::vector<uchar> vTestVals3 = {0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1};
    cv::Mat_<uchar> vTestMat3(1,(int)vTestVals3.size(),vTestVals3.data());
    ASSERT_EQ(lv::calcMedianValue(vTestMat3),1);
    for(size_t i=0u; i<50u; ++i) {
        cv::Mat_<uchar> vTestMat4((rand()%10)+1,(rand()%10)+1);
        cv::randu(vTestMat4,0,256);
        const int nMedian = lv::calcMedianValue(vTestMat4);
        std::nth_element(vTestMat4.begin(),vTestMat4.begin()+(vTestMat4.total())/2,vTestMat4.end());
        const int nMedianGT = (int)*(vTestMat4.begin()+(vTestMat4.total())/2);
        ASSERT_EQ(nMedian,nMedianGT);
    }
    std::vector<uchar> vTestVals5 = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    cv::Mat_<uchar> vTestMat5(1,(int)vTestVals5.size(),vTestVals5.data());
    ASSERT_EQ(lv::calcMedianValue(vTestMat5),1);
    std::vector<uchar> vTestVals6 = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    cv::Mat_<uchar> vTestMat6(1,(int)vTestVals6.size(),vTestVals6.data());
    ASSERT_EQ(lv::calcMedianValue(vTestMat6),0);
}

TEST(medianBlur,regression) {
    for(size_t n=0u; n<50u; ++n) {
        cv::Mat_<uchar> vTestMat((rand()%100)+1,(rand()%100)+1),oMask(vTestMat.size());
        cv::randu(vTestMat,0u,256u);
        oMask = 255u;
        cv::Mat_<uchar> oRegularOutput,oOCVOutput;
        const int nKernelSize = (((rand()%7)+1)*2)+1;
        lv::medianBlur(vTestMat,oRegularOutput,oMask,nKernelSize);
        cv::medianBlur(vTestMat,oOCVOutput,nKernelSize);
        for(int i=nKernelSize/2; i<vTestMat.rows-nKernelSize/2; ++i)
            for(int j=nKernelSize/2; j<vTestMat.cols-nKernelSize/2; ++j)
                ASSERT_EQ(oRegularOutput(i,j),oOCVOutput(i,j)) << "i=" << i << ", j=" << j;
    }
    // toy example, generalize later
    cv::Mat_<uchar> vTestMat(10,10),oMask(10,10),oOutput(10,10);
    for(int i=0; i<vTestMat.rows; ++i) {
        for(int j=0; j<vTestMat.cols; ++j) {
            vTestMat(i,j) = uchar(i*vTestMat.cols+j);
            oMask(i,j) = uchar((((i%2)^(j%2))|int((i>=5)&&(i<9)))&int((j!=6)&&(j!=7)));
        }
    }
    lv::medianBlur(vTestMat,oOutput,oMask,3);
    ASSERT_EQ((int)oOutput(0,1),10);
    ASSERT_EQ((int)oOutput(1,1),10);
    ASSERT_EQ((int)oOutput(1,4),14);
    ASSERT_EQ((int)oOutput(4,1),41);
    ASSERT_EQ((int)oOutput(4,2),43);
    ASSERT_EQ((int)oOutput(4,8),49);
    ASSERT_EQ((int)oOutput(5,2),52);
    ASSERT_EQ((int)oOutput(5,5),55);
    ASSERT_EQ((int)oOutput(5,6),55);
}

TEST(binaryMedianBlur,regression) {
    for(size_t i=0u; i<200u; ++i) {
        cv::Mat_<uchar> oMask,oInput((rand()%100)+1,(rand()%100)+1);
        cv::randu(oInput,0u,256u);
        oInput = oInput>128u;
        if((rand()%4)!=0) {
            oMask.create(oInput.size());
            cv::randu(oMask,0u,256u);
            oMask = oMask>128u;
        }
        cv::Mat_<uchar> oBinaryOutput_conv,oBinaryOutput_raw,oRegularOutput;
        const int nKernelSize = (((rand()%7)+1)*2)+1;
        lv::medianBlur(oInput,oRegularOutput,oMask,nKernelSize);
        lv::binaryMedianBlur(oInput,oBinaryOutput_conv,oMask,nKernelSize,true);
        oInput.setTo(1u,oInput!=0);
        lv::binaryMedianBlur(oInput,oBinaryOutput_raw,oMask,nKernelSize,false);
        const int nOffset = nKernelSize/2;
        const cv::Rect oROI(nOffset,nOffset,oInput.cols-nOffset*2,oInput.rows-nOffset*2);
        if(oROI.width>0 && oROI.height>0) {
            ASSERT_TRUE(lv::isEqual<uchar>(oBinaryOutput_conv(oROI),oRegularOutput(oROI)));
            ASSERT_TRUE(lv::isEqual<uchar>(oBinaryOutput_raw(oROI),oRegularOutput(oROI)));
        }
    }
}

TEST(binaryConsensus,regression) {
    for(size_t i=0u; i<200u; ++i) {
        cv::Mat_<uchar> oInput((rand()%100)+1,(rand()%100)+1);
        cv::randu(oInput,0u,3u);
        cv::Mat_<uchar> oMask(oInput.size());
        cv::randu(oMask,0u,2u);
        cv::Mat_<int> oThreshMap(oInput.size());
        const int nKernelSize = (((rand()%7)+1)*2)+1;
        const int nOffset=nKernelSize/2, nRows=oInput.rows, nCols=oInput.cols;
        for(int nRowIdx=0; nRowIdx<nRows; ++nRowIdx) {
            for(int nColIdx=0; nColIdx<nCols; ++nColIdx) {
                int nHits = 0;
                for(int nOffsetRowIdx=std::max(nRowIdx-nOffset,0); nOffsetRowIdx<=std::min(nRowIdx+nOffset,nRows-1); ++nOffsetRowIdx) {
                    for(int nOffsetColIdx=std::max(nColIdx-nOffset,0); nOffsetColIdx<=std::min(nColIdx+nOffset,nCols-1); ++nOffsetColIdx) {
                        nHits += (oMask(nOffsetRowIdx,nOffsetColIdx)?1:0);
                    }
                }
                oThreshMap(nRowIdx,nColIdx) = nHits/2+1;
            }
        }
        cv::Mat_<uchar> oOutput_conv,oOutput_raw,oOutput_old;
        lv::binaryMedianBlur(oInput,oOutput_old,oMask,nKernelSize,false);
        cv::Mat_<uchar> oInput_masked;
        oInput.copyTo(oInput_masked);
        oInput_masked.setTo(0u,oMask==0);
        lv::binaryConsensus(oInput_masked,oOutput_conv,oThreshMap,nKernelSize,true);
        oInput_masked.setTo(1u,oInput_masked!=0);
        lv::binaryConsensus(oInput_masked,oOutput_raw,oThreshMap,nKernelSize,false);
        const cv::Rect oROI(nOffset,nOffset,oInput.cols-nOffset*2,oInput.rows-nOffset*2);
        if(oROI.width>0 && oROI.height>0) {
            ASSERT_TRUE(lv::isEqual<uchar>(oOutput_conv(oROI),oOutput_raw(oROI)));
            ASSERT_TRUE(lv::isEqual<uchar>(oOutput_conv(oROI),oOutput_old(oROI)));
        }
    }
}

TEST(gmm_init,regression_opencv) {
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty() && oInput.size()==cv::Size(481,321) && oInput.channels()==3);
    const cv::Rect oROIRect(9,124,377,111);
    cv::Mat oMask(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(cv::GC_BGD));
    oMask(oROIRect) = cv::GC_PR_FGD;
    cv::RNG& oRNG = cv::theRNG();
    oRNG.state = 0xffffffff;
    cv::Mat oBGModelData,oFGModelData;
    cv::grabCut(oInput,oMask,oROIRect,oBGModelData,oFGModelData,0,cv::GC_INIT_WITH_MASK);
    ASSERT_TRUE(oBGModelData.total()==size_t(65) && oBGModelData.type()==CV_64FC1);
    ASSERT_TRUE(oFGModelData.total()==size_t(65) && oFGModelData.type()==CV_64FC1);
    oMask = 0;
    oMask(oROIRect) = 255;
    oRNG.state = 0xffffffff;
    lv::GMM<5,3> oBGModel,oFGModel;
    lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
    ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
    ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
    for(size_t nModelIdx=0; nModelIdx<size_t(65); ++nModelIdx) {
        ASSERT_FLOAT_EQ((float)oBGModelData.at<double>(0,int(nModelIdx)),(float)oBGModel.getModelData()[nModelIdx]);
        ASSERT_FLOAT_EQ((float)oFGModelData.at<double>(0,int(nModelIdx)),(float)oFGModel.getModelData()[nModelIdx]);
    }
}

TEST(gmm_init,regression_local) {
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty() && oInput.size()==cv::Size(481,321) && oInput.channels()==3);
    const cv::Rect oROIRect(9,124,377,111);
    cv::Mat oMask(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    oMask(oROIRect) = 255;
    cv::RNG& oRNG = cv::theRNG();
    oRNG.state = 0xffffffff;
    lv::GMM<5,3> oBGModel,oFGModel;
    lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
    ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
    ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
#if (defined(_MSC_VER) && _MSC_VER==1900)
    // seems ocv's RNG state is not platform-independent; use specific bin version here
    const std::string sInitBGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_init_bg.msc1900.bin";
    const std::string sInitFGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_init_fg.msc1900.bin";
#else //!(defined(_MSC_VER) && _MSC_VER==1500)
    const std::string sInitBGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_init_bg.bin";
    const std::string sInitFGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_init_fg.bin";
#endif //!(defined(_MSC_VER) && _MSC_VER==1500)
    if(lv::checkIfExists(sInitBGDataPath) && lv::checkIfExists(sInitFGDataPath)) {
        const cv::Mat_<double> oRefBGData = lv::read(sInitBGDataPath);
        const cv::Mat_<double> oRefFGData = lv::read(sInitFGDataPath);
        for(size_t nModelIdx=0; nModelIdx<size_t(120); ++nModelIdx) {
            ASSERT_FLOAT_EQ((float)oRefBGData.at<double>(0,int(nModelIdx)),(float)oBGModel.getModelData()[nModelIdx]) << "nModelIdx=" << nModelIdx;
            ASSERT_FLOAT_EQ((float)oRefFGData.at<double>(0,int(nModelIdx)),(float)oFGModel.getModelData()[nModelIdx]) << "nModelIdx=" << nModelIdx;
        }
    }
    else {
        lv::write(sInitBGDataPath,cv::Mat_<double>(1,(int)oBGModel.getModelSize(),oBGModel.getModelData()));
        lv::write(sInitFGDataPath,cv::Mat_<double>(1,(int)oFGModel.getModelSize(),oFGModel.getModelData()));
    }
}

TEST(gmm_init,regression_badmask_local) {
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty() && oInput.size()==cv::Size(481,321) && oInput.channels()==3);
    cv::Mat oMask(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    // try several times (rng will use different seeds, none should cause crashes)
    for(size_t n = 0u; n<10u; ++n) {
        lv::GMM<5,3> oBGModel,oFGModel;
        lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
    }
    oMask = uchar(255);
    for(size_t n = 0u; n<10u; ++n) {
        lv::GMM<5,3> oBGModel,oFGModel;
        lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
    }
}

TEST(gmm_learn,regression_opencv) {
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty() && oInput.size()==cv::Size(481,321) && oInput.channels()==3);
    const cv::Rect oROIRect(9,124,377,111);
    cv::Mat oMask(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(cv::GC_BGD));
    oMask(oROIRect) = cv::GC_PR_FGD;
    cv::RNG& oRNG = cv::theRNG();
    oRNG.state = 0xffffffff;
    cv::Mat oBGModelData,oFGModelData;
    cv::grabCut(oInput,oMask,oROIRect,oBGModelData,oFGModelData,1,cv::GC_INIT_WITH_MASK);
    ASSERT_TRUE(oBGModelData.total()==size_t(65) && oBGModelData.type()==CV_64FC1);
    ASSERT_TRUE(oFGModelData.total()==size_t(65) && oFGModelData.type()==CV_64FC1);
    oMask = 0;
    oMask(oROIRect) = 255;
    oRNG.state = 0xffffffff;
    lv::GMM<5,3> oBGModel,oFGModel;
    lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
    ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
    ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
    cv::Mat oAssignMap(oInput.size(),CV_32SC1);
    lv::assignGaussianMixtureComponents(oInput,oMask,oAssignMap,oBGModel,oFGModel);
    lv::learnGaussianMixtureParams(oInput,oMask,oAssignMap,oBGModel,oFGModel);
    for(size_t nModelIdx=0; nModelIdx<size_t(65); ++nModelIdx) {
        ASSERT_FLOAT_EQ((float)oBGModelData.at<double>(0,int(nModelIdx)),(float)oBGModel.getModelData()[nModelIdx]);
        ASSERT_FLOAT_EQ((float)oFGModelData.at<double>(0,int(nModelIdx)),(float)oFGModel.getModelData()[nModelIdx]);
    }
}

TEST(gmm_learn,regression_local) {
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty() && oInput.size()==cv::Size(481,321) && oInput.channels()==3);
    const cv::Rect oROIRect(9,124,377,111);
    cv::Mat oMask(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    oMask(oROIRect) = 255;
    cv::RNG& oRNG = cv::theRNG();
    oRNG.state = 0xffffffff;
    lv::GMM<5,3> oBGModel,oFGModel;
    lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
    ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
    ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
    cv::Mat oAssignMap(oInput.size(),CV_32SC1);
    lv::assignGaussianMixtureComponents(oInput,oMask,oAssignMap,oBGModel,oFGModel);
    lv::learnGaussianMixtureParams(oInput,oMask,oAssignMap,oBGModel,oFGModel);
#if (defined(_MSC_VER) && _MSC_VER==1900)
    // seems ocv's RNG state is not platform-independent; use specific bin version here
    const std::string sBGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_learn_bg.msc1900.bin";
    const std::string sFGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_learn_fg.msc1900.bin";
#else //!(defined(_MSC_VER) && _MSC_VER==1500)
    const std::string sBGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_learn_bg.bin";
    const std::string sFGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_learn_fg.bin";
#endif //!(defined(_MSC_VER) && _MSC_VER==1500)
    if(lv::checkIfExists(sBGDataPath) && lv::checkIfExists(sFGDataPath)) {
        const cv::Mat_<double> oRefBGData = lv::read(sBGDataPath);
        const cv::Mat_<double> oRefFGData = lv::read(sFGDataPath);
        for(size_t nModelIdx=0; nModelIdx<size_t(120); ++nModelIdx) {
            ASSERT_FLOAT_EQ((float)oRefBGData.at<double>(0,int(nModelIdx)),(float)oBGModel.getModelData()[nModelIdx]) << "nModelIdx=" << nModelIdx;
            ASSERT_FLOAT_EQ((float)oRefFGData.at<double>(0,int(nModelIdx)),(float)oFGModel.getModelData()[nModelIdx]) << "nModelIdx=" << nModelIdx;
        }
    }
    else {
        lv::write(sBGDataPath,cv::Mat_<double>(1,(int)oBGModel.getModelSize(),oBGModel.getModelData()));
        lv::write(sFGDataPath,cv::Mat_<double>(1,(int)oFGModel.getModelSize(),oFGModel.getModelData()));
    }
}

TEST(gmm_learn,regression_badmask_local) {
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty() && oInput.size()==cv::Size(481,321) && oInput.channels()==3);
    cv::Mat oMask(oInput.size(),CV_8UC1,cv::Scalar_<uchar>(0));
    // try several times (rng will use different seeds, none should cause crashes)
    for(size_t n = 0u; n<10u; ++n) {
        lv::GMM<5,3> oBGModel,oFGModel;
        lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
        cv::Mat oAssignMap(oInput.size(),CV_32SC1);
        lv::assignGaussianMixtureComponents(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(cv::countNonZero(oMask!=0u),0);
        lv::learnGaussianMixtureParams(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
        lv::assignGaussianMixtureComponents(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(cv::countNonZero(oMask!=0u),0);
        lv::learnGaussianMixtureParams(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
    }
    oMask = uchar(255);
    for(size_t n = 0u; n<10u; ++n) {
        lv::GMM<5,3> oBGModel,oFGModel;
        lv::initGaussianMixtureParams(oInput,oMask,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
        cv::Mat oAssignMap(oInput.size(),CV_32SC1);
        lv::assignGaussianMixtureComponents(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(cv::countNonZero(oMask!=255u),0);
        lv::learnGaussianMixtureParams(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
        lv::assignGaussianMixtureComponents(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(cv::countNonZero(oMask!=255u),0);
        lv::learnGaussianMixtureParams(oInput,oMask,oAssignMap,oBGModel,oFGModel);
        ASSERT_EQ(oBGModel.getModelSize(),size_t(120));
        ASSERT_EQ(oFGModel.getModelSize(),size_t(120));
    }
}

#if USING_OFDIS

#include "litiv/3rdparty/ofdis/ofdis.hpp"

TEST(ofdis_optflow,regression) {
    const cv::Mat oImage1 = cv::imread(SAMPLES_DATA_ROOT "/middlebury2005_dataset_ex/dolls/view1.png");
    ASSERT_FALSE(oImage1.empty());
    ASSERT_EQ(oImage1.type(),CV_8UC3);
    ASSERT_EQ(oImage1.size(),cv::Size(463,370));
    const cv::Mat oImage2 = cv::imread(SAMPLES_DATA_ROOT "/middlebury2005_dataset_ex/dolls/view0.png");
    ASSERT_FALSE(oImage2.empty());
    ASSERT_EQ(oImage1.type(),oImage2.type());
    ASSERT_EQ(oImage1.size(),oImage2.size());
    cv::Mat oFlowMap;
    ofdis::computeFlow<ofdis::FlowInput_RGB,ofdis::FlowOutput_OpticalFlow>(oImage1,oImage2,oFlowMap);
    ASSERT_EQ(oFlowMap.size(),oImage1.size());
    ASSERT_EQ(oFlowMap.type(),CV_32FC2);
    oFlowMap = oFlowMap(cv::Rect(200,200,100,100));
    const std::string sFlowMapBinPath = TEST_CURR_INPUT_DATA_ROOT "/test_flowmap_ofdis.bin";
    if(lv::checkIfExists(sFlowMapBinPath)) {
        cv::Mat oRefMap = lv::read(sFlowMapBinPath);
        ASSERT_EQ(oFlowMap.size,oRefMap.size);
        ASSERT_EQ(oFlowMap.type(),oRefMap.type());
        ASSERT_EQ(oFlowMap.total(),oRefMap.total());
        for(int i=0; i<oFlowMap.size[0]; ++i)
            for(int j=0; j<oFlowMap.size[1]; ++j)
                for(int k=0; k<2; ++k)
                    ASSERT_NEAR(oFlowMap.at<cv::Vec2f>(i,j)[k],oRefMap.at<cv::Vec2f>(i,j)[k],0.0001f) << "ijk=[" << i << "," << j << "," << k << "]";
    }
    else
        lv::write(sFlowMapBinPath,oFlowMap);
}

#endif //USING_OFDIS

#include "litiv/features2d/SC.hpp"

TEST(descriptor_affinity,regression_L2_sc) {
    std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(size_t(2),size_t(40),12,5);
    const int nSize = 257;
    const cv::Size oSize(nSize,nSize);
    cv::Mat oInput1(oSize,CV_8UC1),oInput2(oSize,CV_8UC1);
    oInput1 = 0; oInput2 = 0;
    cv::circle(oInput1,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput1,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput1,cv::Point(151,19),cv::Point(194,193),cv::Scalar_<uchar>(255),-1);
    cv::circle(oInput2,cv::Point(52,62),5,cv::Scalar_<uchar>(255),-1);
    cv::circle(oInput2,cv::Point(124,124),8,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput2,cv::Point(185,185),cv::Point(195,195),cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput2,cv::Point(161,19),cv::Point(204,193),cv::Scalar_<uchar>(255),-1);
    oInput1 = oInput1>0; oInput2 = oInput2>0;
    //cv::imshow("in1",oInput);
    //cv::imshow("in2",oInput);
    cv::Mat_<float> oDescMap1,oDescMap2;
    pShapeContext->compute2(oInput1,oDescMap1);
    pShapeContext->compute2(oInput2,oDescMap2);
    ASSERT_EQ(oDescMap1.dims,oDescMap2.dims);
    ASSERT_EQ(oDescMap1.size,oDescMap2.size);
    ASSERT_EQ(oDescMap1.type(),oDescMap2.type());
    ASSERT_EQ(oDescMap1.size[0],nSize);
    ASSERT_EQ(oDescMap1.size[1],nSize);
    ASSERT_FALSE(std::equal(oDescMap1.begin(),oDescMap1.end(),oDescMap2.begin()));
    cv::Mat_<float> oAffMap;
    const std::vector<int> vDispRange = {0,1,4,9,15};
    cv::Mat_<uchar> oROI1(oInput2.size(),uchar(0)),oROI2(oInput2.size(),uchar(0));
    oROI1(cv::Rect(0,1,205,204)) = uchar(255);
    //cv::imshow("roi1",oROI1);
    oROI2(cv::Rect(51,55,169,173)) = uchar(255);
    //cv::imshow("roi2",oROI2);
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,1,oAffMap,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),false);
    ASSERT_EQ(oAffMap.size[0],nSize);
    ASSERT_EQ(oAffMap.size[1],nSize);
    ASSERT_EQ(oAffMap.size[2],(int)vDispRange.size());
#if HAVE_CUDA
    cv::cuda::GpuMat oDescMap1_dev,oDescMap2_dev;
    pShapeContext->compute2(oInput1,oDescMap1_dev);
    pShapeContext->compute2(oInput2,oDescMap2_dev);
    ASSERT_EQ(oDescMap1_dev.size(),oDescMap2_dev.size());
    ASSERT_EQ(oDescMap1_dev.type(),oDescMap2_dev.type());
    ASSERT_EQ(oDescMap1_dev.rows,nSize*nSize);
    ASSERT_EQ(oDescMap1_dev.cols,oDescMap1.size[2]);
    cv::cuda::GpuMat oAffMap_dev,oROI1_dev,oROI2_dev;
    oROI1_dev.upload(oROI1);
    oROI2_dev.upload(oROI2);
    lv::computeDescriptorAffinity(oDescMap1_dev,oDescMap2_dev,oSize,1,oAffMap_dev,vDispRange,lv::AffinityDist_L2,oROI1_dev,oROI2_dev);
    ASSERT_EQ(oAffMap_dev.rows,nSize*nSize);
    ASSERT_EQ(oAffMap_dev.cols,(int)vDispRange.size());
    cv::Mat_<float> oAffMapTmp,oDescMap1Tmp,oDescMap2Tmp;
    oAffMap_dev.download(oAffMapTmp);
    oDescMap1_dev.download(oDescMap1Tmp);
    oDescMap2_dev.download(oDescMap2Tmp);
    ASSERT_EQ(oDescMap1Tmp.dims,oDescMap2Tmp.dims);
    ASSERT_EQ(oDescMap1Tmp.size,oDescMap2Tmp.size);
    ASSERT_EQ(oDescMap1Tmp.type(),oDescMap2Tmp.type());
    ASSERT_EQ(oDescMap1Tmp.size[0],nSize*nSize);
    ASSERT_FALSE(std::equal(oDescMap1Tmp.begin(),oDescMap1Tmp.end(),oDescMap2Tmp.begin()));
    const int nDescSize = oDescMap1.size[2];
    oDescMap1Tmp = oDescMap1Tmp.reshape(0,3,std::array<int,3>{nSize,nSize,nDescSize}.data());
    oDescMap2Tmp = oDescMap2Tmp.reshape(0,3,std::array<int,3>{nSize,nSize,nDescSize}.data());
    ASSERT_EQ(lv::MatInfo(oDescMap1),lv::MatInfo(oDescMap1Tmp));
    ASSERT_EQ(lv::MatInfo(oDescMap2),lv::MatInfo(oDescMap2Tmp));
    oAffMapTmp = oAffMapTmp.reshape(0,3,std::array<int,3>{nSize,nSize,(int)vDispRange.size()}.data());
    ASSERT_EQ(lv::MatInfo(oAffMap),lv::MatInfo(oAffMapTmp));
    for(int i=0; i<oAffMap.size[0]; ++i) {
        for(int j=0; j<oAffMap.size[1]; ++j) {
            for(int k=0; k<oDescMap1Tmp.size[2]; ++k)
                ASSERT_FLOAT_EQ(oDescMap1(i,j,k),oDescMap1Tmp(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
            for(int k=0; k<oAffMap.size[2]; ++k)
                ASSERT_FLOAT_EQ(oAffMap(i,j,k),oAffMapTmp(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
        }
    }
#endif //HAVE_CUDA
    oAffMap = lv::getSubMat(oAffMap,0,cv::Range(45,70));
    const std::string sAffMapBinPath_p1 = TEST_CURR_INPUT_DATA_ROOT "/test_affmap_p1_r5_L2.bin";
    if(lv::checkIfExists(sAffMapBinPath_p1)) {
        cv::Mat_<float> oRefMap = lv::read(sAffMapBinPath_p1);
        ASSERT_EQ(oAffMap.total(),oRefMap.total());
        ASSERT_EQ(oAffMap.size,oRefMap.size);
        for(int i=0; i<oAffMap.size[0]; ++i)
            for(int j=0; j<oAffMap.size[1]; ++j)
                for(int k=0; k<oAffMap.size[2]; ++k)
                    ASSERT_FLOAT_EQ(oAffMap(i,j,k),oRefMap(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
    }
    else
        lv::write(sAffMapBinPath_p1,oAffMap);
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,7,oAffMap,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),false);
    ASSERT_EQ(oAffMap.size[0],nSize);
    ASSERT_EQ(oAffMap.size[1],nSize);
    ASSERT_EQ(oAffMap.size[2],(int)vDispRange.size());
#if HAVE_CUDA
    lv::computeDescriptorAffinity(oDescMap1_dev,oDescMap2_dev,oSize,7,oAffMap_dev,vDispRange,lv::AffinityDist_L2,oROI1_dev,oROI2_dev);
    ASSERT_EQ(oAffMap_dev.rows,nSize*nSize);
    ASSERT_EQ(oAffMap_dev.cols,(int)vDispRange.size());
    oAffMap_dev.download(oAffMapTmp);
    oAffMapTmp = oAffMapTmp.reshape(0,3,std::array<int,3>{nSize,nSize,(int)vDispRange.size()}.data());
    for(int i=0; i<oAffMap.size[0]; ++i)
        for(int j=0; j<oAffMap.size[1]; ++j)
            for(int k=0; k<oAffMap.size[2]; ++k)
                ASSERT_FLOAT_EQ(oAffMap(i,j,k),oAffMapTmp(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
#endif //HAVE_CUDA
    oAffMap = lv::getSubMat(oAffMap,0,cv::Range(160,185));
    const std::string sAffMapBinPath_p7 = TEST_CURR_INPUT_DATA_ROOT "/test_affmap_p7_r5_L2.bin";
    if(lv::checkIfExists(sAffMapBinPath_p7)) {
        cv::Mat_<float> oRefMap = lv::read(sAffMapBinPath_p7);
        ASSERT_EQ(oAffMap.total(),oRefMap.total());
        ASSERT_EQ(oAffMap.size,oRefMap.size);
        for(int i=0; i<oAffMap.size[0]; ++i)
            for(int j=0; j<oAffMap.size[1]; ++j)
                for(int k=0; k<oAffMap.size[2]; ++k)
                    ASSERT_FLOAT_EQ(oAffMap(i,j,k),oRefMap(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
    }
    else
        lv::write(sAffMapBinPath_p7,oAffMap);
    //cv::waitKey(0);
}

#ifndef _MSC_VER

    // large-scale DASC computation with MSVC differs from result with other compilers, which breaks the test below
    // (see DASC test for more info; we disable the affinity test below, but it would pass given good desc maps)

#include "litiv/features2d/DASC.hpp"

TEST(descriptor_affinity,regression_L2_dasc) {
    std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(size_t(2),0.09f);
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty());
    cv::Mat oInput1=oInput.clone(),oInput2=oInput.clone();
    lv::shift(oInput2,oInput2,cv::Point2f(-1.5f,4.5f),cv::BORDER_REFLECT101);
    cv::GaussianBlur(oInput2,oInput2,cv::Size(5,5),0);
    lvAssert(oInput2.depth()==CV_32F);
    oInput2.convertTo(oInput2,CV_8U);
    //cv::imshow("in1",oInput);
    //cv::imshow("in2",oInput);
    cv::Mat_<float> oDescMap1,oDescMap2;
    pDASC->compute2(oInput1,oDescMap1);
    pDASC->compute2(oInput2,oDescMap2);
    ASSERT_EQ(lv::MatInfo(oDescMap1),lv::MatInfo(oDescMap2));
    ASSERT_FALSE(std::equal(oDescMap1.begin(),oDescMap1.end(),oDescMap2.begin()));
    cv::Mat_<float> oAffMap;
    const std::vector<int> vDispRange = lv::make_range(-3,0);
    cv::Mat_<uchar> oROI1(oInput2.size(),uchar(0)),oROI2(oInput2.size(),uchar(0));
    const int nBorderSize = pDASC->borderSize();
    oROI1(cv::Rect(nBorderSize,nBorderSize,oInput.cols-nBorderSize*2,oInput.rows-nBorderSize*2)) = uchar(255);
    oROI2(cv::Rect(nBorderSize,nBorderSize,oInput.cols-nBorderSize*2,oInput.rows-nBorderSize*2)) = uchar(255);
    //cv::imshow("roi1",oROI1);
    //cv::imshow("roi2",oROI2);
    //cv::waitKey(0);
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,1,oAffMap,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),false);
    ASSERT_EQ(oAffMap.size[0],oInput.rows);
    ASSERT_EQ(oAffMap.size[1],oInput.cols);
    ASSERT_EQ(oAffMap.size[2],(int)vDispRange.size());
#if HAVE_CUDA
    cv::Mat_<float> oAffMap_gpu;
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,1,oAffMap_gpu,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),true);
    ASSERT_EQ(lv::MatInfo(oAffMap_gpu),lv::MatInfo(oAffMap));
    for(int i=0; i<oAffMap.size[0]; ++i)
        for(int j=0; j<oAffMap.size[1]; ++j)
            for(int k=0; k<oAffMap.size[2]; ++k)
                ASSERT_FLOAT_EQ(oAffMap(i,j,k),oAffMap_gpu(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
#endif //HAVE_CUDA
    oAffMap = lv::getSubMat(oAffMap,0,cv::Range(45,70));
#if USE_FULL_FLOAT_TEST
    const std::string sAffMapBinPath_p1 = TEST_CURR_INPUT_DATA_ROOT "/test_affmap_p1_r4_L2.bin";
    if(lv::checkIfExists(sAffMapBinPath_p1)) {
        cv::Mat_<float> oRefMap = lv::read(sAffMapBinPath_p1);
        ASSERT_EQ(oAffMap.total(),oRefMap.total());
        ASSERT_EQ(oAffMap.size,oRefMap.size);
        for(int i=0; i<oAffMap.size[0]; ++i)
            for(int j=0; j<oAffMap.size[1]; ++j)
                for(int k=0; k<oAffMap.size[2]; ++k)
                    ASSERT_NEAR(oAffMap(i,j,k),oRefMap(i,j,k),0.001f) << "ijk=[" << i << "," << j << "," << k << "]";
    }
    else
        lv::write(sAffMapBinPath_p1,oAffMap);
#endif //USE_FULL_FLOAT_TEST
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,7,oAffMap,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),false);
    ASSERT_EQ(oAffMap.size[0],oInput.rows);
    ASSERT_EQ(oAffMap.size[1],oInput.cols);
    ASSERT_EQ(oAffMap.size[2],(int)vDispRange.size());
#if HAVE_CUDA
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,7,oAffMap_gpu,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),true);
    ASSERT_EQ(lv::MatInfo(oAffMap_gpu),lv::MatInfo(oAffMap));
    for(int i=0; i<oAffMap.size[0]; ++i)
        for(int j=0; j<oAffMap.size[1]; ++j)
            for(int k=0; k<oAffMap.size[2]; ++k)
                ASSERT_NEAR(oAffMap(i,j,k),oAffMap_gpu(i,j,k),0.001f) << "ijk=[" << i << "," << j << "," << k << "]";
#endif //HAVE_CUDA
    oAffMap = lv::getSubMat(oAffMap,0,cv::Range(160,185));
    const std::string sAffMapBinPath_p7 = TEST_CURR_INPUT_DATA_ROOT "/test_affmap_p7_r4_L2.bin";
    if(lv::checkIfExists(sAffMapBinPath_p7)) {
        cv::Mat_<float> oRefMap = lv::read(sAffMapBinPath_p7);
        ASSERT_EQ(oAffMap.total(),oRefMap.total());
        ASSERT_EQ(oAffMap.size,oRefMap.size);
        for(int i=0; i<oAffMap.size[0]; ++i)
            for(int j=0; j<oAffMap.size[1]; ++j)
                for(int k=0; k<oAffMap.size[2]; ++k)
                    ASSERT_NEAR(oAffMap(i,j,k),oRefMap(i,j,k),0.001f) << "ijk=[" << i << "," << j << "," << k << "]";
    }
    else
        lv::write(sAffMapBinPath_p7,oAffMap);
}

#endif //ndef(_MSC_VER)

TEST(computeIntegral,regression) {
    for(size_t i=0u; i<20u; ++i) {
        cv::Mat oTestMat((rand()%500)+1,(rand()%500)+1,CV_8UC((rand()%4)+1));
        cv::randu(oTestMat,0,256);
        cv::Mat oLocalOutput,oCVOutput;
        lv::integral(oTestMat,oLocalOutput,CV_32S);
        cv::integral(oTestMat,oCVOutput,CV_32S);
        ASSERT_TRUE(lv::isEqual<int>(oLocalOutput,oCVOutput));
    }
}

namespace {

    void medianBlur_perftest(benchmark::State& st) {
        const volatile int nMatSize = st.range(0);
        const volatile int nKernelSize = st.range(1);
        std::unique_ptr<uint8_t[]> aVals = lv::test::genarray<uint8_t>((size_t)nMatSize*nMatSize,0u,255u);
        cv::Mat_<uchar> oOutput(nMatSize,nMatSize);
        while(st.KeepRunning()) {
            benchmark::DoNotOptimize(aVals.get());
            cv::Mat oInput(nMatSize,nMatSize,CV_8UC1,aVals.get());
            cv::medianBlur(oInput,oOutput,nKernelSize);
            benchmark::DoNotOptimize(oOutput.data);
        }
    }

    void binaryMedianBlur_conv_perftest(benchmark::State& st) {
        const volatile int nMatSize = st.range(0);
        const volatile int nKernelSize = st.range(1);
        std::unique_ptr<uint8_t[]> aVals = lv::test::genarray<uint8_t>((size_t)nMatSize*nMatSize,0u,255u);
        cv::Mat_<uchar> oOutput(nMatSize,nMatSize);
        while(st.KeepRunning()) {
            benchmark::DoNotOptimize(aVals.get());
            cv::Mat oInput(nMatSize,nMatSize,CV_8UC1,aVals.get());
            lv::binaryMedianBlur(oInput,oOutput,cv::Mat(),nKernelSize,true);
            benchmark::DoNotOptimize(oOutput.data);
        }
    }

    void binaryMedianBlur_raw_perftest(benchmark::State& st) {
        const volatile int nMatSize = st.range(0);
        const volatile int nKernelSize = st.range(1);
        std::unique_ptr<uint8_t[]> aVals = lv::test::genarray<uint8_t>((size_t)nMatSize*nMatSize,0u,1u);
        cv::Mat_<uchar> oOutput(nMatSize,nMatSize);
        while(st.KeepRunning()) {
            benchmark::DoNotOptimize(aVals.get());
            cv::Mat oInput(nMatSize,nMatSize,CV_8UC1,aVals.get());
            lv::binaryMedianBlur(oInput,oOutput,cv::Mat(),nKernelSize,false);
            benchmark::DoNotOptimize(oOutput.data);
        }
    }

    void binaryConsensus_perftest(benchmark::State& st) {
        const volatile int nMatSize = st.range(0);
        const volatile int nKernelSize = st.range(1);
        std::unique_ptr<uint8_t[]> aVals1 = lv::test::genarray<uint8_t>((size_t)nMatSize*nMatSize,0u,1u);
        std::unique_ptr<int[]> aVals2 = lv::test::genarray<int>((size_t)nMatSize*nMatSize,0,nKernelSize*nKernelSize);
        cv::Mat_<uchar> oOutput(nMatSize,nMatSize);
        while(st.KeepRunning()) {
            benchmark::DoNotOptimize(aVals1.get());
            benchmark::DoNotOptimize(aVals2.get());
            cv::Mat oInput(nMatSize,nMatSize,CV_8UC1,aVals1.get());
            cv::Mat oThresh(nMatSize,nMatSize,CV_32SC1,aVals2.get());
            lv::binaryConsensus(oInput,oOutput,oThresh,nKernelSize,false);
            benchmark::DoNotOptimize(oOutput.data);
        }
    }

}

BENCHMARK(medianBlur_perftest)->Args({50,3})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_conv_perftest)->Args({50,3})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_raw_perftest)->Args({50,3})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryConsensus_perftest)->Args({50,3})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);

BENCHMARK(medianBlur_perftest)->Args({200,5})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_conv_perftest)->Args({200,5})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_raw_perftest)->Args({200,5})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryConsensus_perftest)->Args({200,5})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);

BENCHMARK(medianBlur_perftest)->Args({400,7})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_conv_perftest)->Args({400,7})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_raw_perftest)->Args({400,7})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryConsensus_perftest)->Args({400,7})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);

BENCHMARK(medianBlur_perftest)->Args({800,11})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_conv_perftest)->Args({800,11})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryMedianBlur_raw_perftest)->Args({800,11})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(binaryConsensus_perftest)->Args({800,11})->Unit(benchmark::kMicrosecond)->Repetitions(10)->ReportAggregatesOnly(true);
