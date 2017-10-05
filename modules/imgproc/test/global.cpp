
#include "litiv/imgproc.hpp"
#include "litiv/test.hpp"

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
                    ASSERT_FLOAT_EQ(oFlowMap.at<cv::Vec2f>(i,j)[k],oRefMap.at<cv::Vec2f>(i,j)[k]) << "ijk=[" << i << "," << j << "," << k << "]";
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
    const std::string sAffMapBinPath_p1 = TEST_CURR_INPUT_DATA_ROOT "/test_affmap_p1_r4_L2.bin";
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
    ASSERT_EQ(oAffMap.size[0],oInput.rows);
    ASSERT_EQ(oAffMap.size[1],oInput.cols);
    ASSERT_EQ(oAffMap.size[2],(int)vDispRange.size());
#if HAVE_CUDA
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,7,oAffMap_gpu,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),true);
    ASSERT_EQ(lv::MatInfo(oAffMap_gpu),lv::MatInfo(oAffMap));
    for(int i=0; i<oAffMap.size[0]; ++i)
        for(int j=0; j<oAffMap.size[1]; ++j)
            for(int k=0; k<oAffMap.size[2]; ++k)
                ASSERT_FLOAT_EQ(oAffMap(i,j,k),oAffMap_gpu(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
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
                    ASSERT_FLOAT_EQ(oAffMap(i,j,k),oRefMap(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
    }
    else
        lv::write(sAffMapBinPath_p7,oAffMap);
}