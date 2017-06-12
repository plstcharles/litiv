
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
        ASSERT_FLOAT_EQ(oBGModelData.at<double>(0,int(nModelIdx)),oBGModel.getModelData()[nModelIdx]);
        ASSERT_FLOAT_EQ(oFGModelData.at<double>(0,int(nModelIdx)),oFGModel.getModelData()[nModelIdx]);
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
    const std::string sInitBGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_init_bg.bin";
    const std::string sInitFGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_init_fg.bin";
    if(lv::checkIfExists(sInitBGDataPath) && lv::checkIfExists(sInitFGDataPath)) {
        const cv::Mat_<double> oRefBGData = lv::read(sInitBGDataPath);
        const cv::Mat_<double> oRefFGData = lv::read(sInitFGDataPath);
        for(size_t nModelIdx=0; nModelIdx<size_t(120); ++nModelIdx) {
            ASSERT_FLOAT_EQ(oRefBGData.at<double>(0,int(nModelIdx)),oBGModel.getModelData()[nModelIdx]);
            ASSERT_FLOAT_EQ(oRefFGData.at<double>(0,int(nModelIdx)),oFGModel.getModelData()[nModelIdx]);
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
        ASSERT_FLOAT_EQ(oBGModelData.at<double>(0,int(nModelIdx)),oBGModel.getModelData()[nModelIdx]);
        ASSERT_FLOAT_EQ(oFGModelData.at<double>(0,int(nModelIdx)),oFGModel.getModelData()[nModelIdx]);
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
    const std::string sBGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_learn_bg.bin";
    const std::string sFGDataPath = TEST_CURR_INPUT_DATA_ROOT "/test_gmm_learn_fg.bin";
    if(lv::checkIfExists(sBGDataPath) && lv::checkIfExists(sFGDataPath)) {
        const cv::Mat_<double> oRefBGData = lv::read(sBGDataPath);
        const cv::Mat_<double> oRefFGData = lv::read(sFGDataPath);
        for(size_t nModelIdx=0; nModelIdx<size_t(120); ++nModelIdx) {
            ASSERT_FLOAT_EQ(oRefBGData.at<double>(0,int(nModelIdx)),oBGModel.getModelData()[nModelIdx]);
            ASSERT_FLOAT_EQ(oRefFGData.at<double>(0,int(nModelIdx)),oFGModel.getModelData()[nModelIdx]);
        }
    }
    else {
        lv::write(sBGDataPath,cv::Mat_<double>(1,(int)oBGModel.getModelSize(),oBGModel.getModelData()));
        lv::write(sFGDataPath,cv::Mat_<double>(1,(int)oFGModel.getModelSize(),oFGModel.getModelData()));
    }
}