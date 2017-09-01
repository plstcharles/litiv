
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

#include "litiv/features2d/SC.hpp"

TEST(descriptor_affinity,regression_L2) {
    std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(size_t(2),size_t(40),12,5);
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(151,19),cv::Point(194,193),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    //cv::imshow("in1",oInput);
    cv::Mat_<float> oDescMap1;
    pShapeContext->compute2(oInput,oDescMap1);
#if HAVE_CUDA
    cv::cuda::GpuMat oDescMap1_dev;
    pShapeContext->compute2(oInput,oDescMap1_dev);
#endif //HAVE_CUDA
    oInput = 0;
    cv::circle(oInput,cv::Point(52,62),5,cv::Scalar_<uchar>(255),-1);
    cv::circle(oInput,cv::Point(124,124),8,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(185,185),cv::Point(195,195),cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(161,19),cv::Point(204,193),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    //cv::imshow("in2",oInput);
    cv::Mat_<float> oDescMap2;
    pShapeContext->compute2(oInput,oDescMap2);
#if HAVE_CUDA
    cv::cuda::GpuMat oDescMap2_dev;
    pShapeContext->compute2(oInput,oDescMap2_dev);
#endif //HAVE_CUDA
    ASSERT_EQ(oDescMap1.dims,oDescMap2.dims);
    ASSERT_EQ(oDescMap1.size,oDescMap2.size);
    ASSERT_EQ(oDescMap1.type(),oDescMap2.type());
    ASSERT_EQ(oDescMap1.size[0],oInput.rows);
    ASSERT_EQ(oDescMap1.size[1],oInput.cols);
    cv::Mat_<float> oAffMap;
    const std::vector<int> vDispRange = {0,1,4,9,15};
    cv::Mat_<uchar> oROI1(oInput.size(),uchar(0)),oROI2(oInput.size(),uchar(0));
    oROI1(cv::Rect(0,1,205,204)) = uchar(255);
    //cv::imshow("roi1",oROI1);
    oROI2(cv::Rect(51,55,169,173)) = uchar(255);
    //cv::imshow("roi2",oROI2);
    lv::computeDescriptorAffinity(oDescMap1,oDescMap2,1,oAffMap,vDispRange,lv::AffinityDist_L2,oROI1,oROI2,cv::Mat(),false);
    ASSERT_EQ(oAffMap.size[0],oInput.rows);
    ASSERT_EQ(oAffMap.size[1],oInput.cols);
    ASSERT_EQ(oAffMap.size[2],(int)vDispRange.size());
#if HAVE_CUDA
    ASSERT_EQ(oDescMap1_dev.size(),oDescMap2_dev.size());
    ASSERT_EQ(oDescMap1_dev.type(),oDescMap2_dev.type());
    ASSERT_EQ(oDescMap1_dev.rows,oDescMap1.size[0]*oDescMap1.size[1]);
    ASSERT_EQ(oDescMap1_dev.cols,oDescMap1.size[2]);
    cv::cuda::GpuMat oAffMap_dev,oROI1_dev,oROI2_dev;
    oROI1_dev.upload(oROI1);
    oROI2_dev.upload(oROI2);
    lv::computeDescriptorAffinity(oDescMap1_dev,oDescMap2_dev,cv::Size(oInput.cols,oInput.rows),1,oAffMap_dev,vDispRange,lv::AffinityDist_L2,oROI1_dev,oROI2_dev);
    ASSERT_EQ(oAffMap_dev.rows,oInput.rows*oInput.cols);
    ASSERT_EQ(oAffMap_dev.cols,(int)vDispRange.size());
    cv::Mat_<float> oAffMapTmp;
    oAffMap_dev.download(oAffMapTmp);
    oAffMapTmp = oAffMapTmp.reshape(0,3,std::array<int,3>{oInput.rows,oInput.cols,(int)vDispRange.size()}.data());
    for(int i=0; i<oAffMap.size[0]; ++i)
        for(int j=0; j<oAffMap.size[1]; ++j)
            for(int k=0; k<oAffMap.size[2]; ++k)
                ASSERT_FLOAT_EQ(oAffMap(i,j,k),oAffMapTmp(i,j,k)) << "ijk=[" << i << "," << j << "," << k << "]";
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
    ASSERT_EQ(oAffMap.size[0],oInput.rows);
    ASSERT_EQ(oAffMap.size[1],oInput.cols);
    ASSERT_EQ(oAffMap.size[2],(int)vDispRange.size());
#if HAVE_CUDA
    lv::computeDescriptorAffinity(oDescMap1_dev,oDescMap2_dev,cv::Size(oInput.cols,oInput.rows),7,oAffMap_dev,vDispRange,lv::AffinityDist_L2,oROI1_dev,oROI2_dev);
    ASSERT_EQ(oAffMap_dev.rows,oInput.rows*oInput.cols);
    ASSERT_EQ(oAffMap_dev.cols,(int)vDispRange.size());
    oAffMap_dev.download(oAffMapTmp);
    oAffMapTmp = oAffMapTmp.reshape(0,3,std::array<int,3>{oInput.rows,oInput.cols,(int)vDispRange.size()}.data());
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