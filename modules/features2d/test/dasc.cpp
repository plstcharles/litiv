
#include "litiv/features2d/DASC.hpp"
#include "litiv/test.hpp"

TEST(dasc_rf,regression_constr) {
    EXPECT_THROW_LV_QUIET(std::make_unique<DASC>(0.0f,0.05f));
    EXPECT_THROW_LV_QUIET(std::make_unique<DASC>(1.0f,0.0f));
    EXPECT_THROW_LV_QUIET(std::make_unique<DASC>(1.0f,0.1f,size_t(0)));
}

TEST(dasc_rf,regression_default_params) {
    std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR);
    EXPECT_GT(pDASC->windowSize().width,0);
    EXPECT_GT(pDASC->windowSize().height,0);
    EXPECT_TRUE((pDASC->windowSize().width%2)==1);
    EXPECT_TRUE((pDASC->windowSize().height%2)==1);
    EXPECT_EQ(pDASC->windowSize().width,pDASC->windowSize().height);
    EXPECT_EQ(pDASC->windowSize().width/2,pDASC->borderSize());
    EXPECT_EQ(pDASC->windowSize().width/2,pDASC->borderSize(1));
    ASSERT_THROW_LV_QUIET(pDASC->borderSize(2));
    EXPECT_GT(pDASC->descriptorSize(),0);
    EXPECT_EQ(size_t(pDASC->descriptorSize())%sizeof(float),size_t(0));
    EXPECT_EQ(pDASC->descriptorType(),CV_32F);
    EXPECT_EQ(pDASC->descriptorType(),CV_32FC1);
    EXPECT_EQ(pDASC->defaultNorm(),cv::NORM_L2);
    EXPECT_TRUE(pDASC->isUsingRF());
    EXPECT_EQ(pDASC->isPreProcessing(),DASC_DEFAULT_PREPROCESS);
}

TEST(dasc_rf,regression_single_compute) {
    std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR);
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty());
    const cv::Point2i oTargetPt(364,135);
    const cv::Size oWindowSize = pDASC->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2,oTargetPt.y-oWindowSize.height/2,oWindowSize.width,oWindowSize.height)).clone();
    cv::Mat_<float> oOutputDescMap;
    pDASC->compute2(oInputCrop,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    const cv::Mat_<float> oOutputDesc = cv::Mat_<float>(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2)).clone();
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_dasc_rf.bin")) {
        const cv::Mat_<float> oRefDesc = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_dasc_rf.bin");
        ASSERT_EQ(oOutputDesc.total(),oRefDesc.total());
        ASSERT_EQ(oOutputDesc.size,oRefDesc.size);
        for(int nDescIdx=0; nDescIdx<oRefDesc.size[2]; ++nDescIdx)
            ASSERT_NEAR_MINRATIO(oOutputDesc.at<float>(0,0,nDescIdx),oRefDesc.at<float>(0,0,nDescIdx),0.04f);
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_dasc_rf.bin",oOutputDesc);
    const std::vector<cv::KeyPoint> vKeyPoints_orig = {cv::KeyPoint(cv::Point2f(float(oWindowSize.width/2),float(oWindowSize.height/2)),float(std::max(oWindowSize.height,oWindowSize.width)))};
    std::vector<cv::KeyPoint> vKeyPoints = vKeyPoints_orig;
    pDASC->compute2(oInputCrop,vKeyPoints,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(vKeyPoints.size(),size_t(1));
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.x,vKeyPoints_orig[0].pt.x);
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.y,vKeyPoints_orig[0].pt.y);
    const cv::Mat_<float> oOutputKPDesc(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2));
    ASSERT_EQ(oOutputDesc.total(),oOutputKPDesc.total());
    ASSERT_EQ(oOutputDesc.size,oOutputKPDesc.size);
    for(int nDescIdx=0; nDescIdx<oOutputDesc.size[2]; ++nDescIdx) {
        ASSERT_FLOAT_EQ(oOutputDesc.at<float>(0,0,nDescIdx),oOutputKPDesc.at<float>(0,0,nDescIdx));
        ASSERT_FALSE(std::isnan(oOutputDesc.at<float>(0,0,nDescIdx)));
        ASSERT_GE(oOutputDesc.at<float>(0,0,nDescIdx),0.0f);
    }
    ASSERT_FLOAT_EQ(float(pDASC->calcDistance(oOutputDesc,oOutputKPDesc)),0.0f);
}

TEST(dasc_rf,regression_large_compute) {
    std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(DASC_DEFAULT_RF_SIGMAS,DASC_DEFAULT_RF_SIGMAR);
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty());
    const cv::Point2i oTargetPt(364,135);
    const int nPatchSize = 10;
    const cv::Size oWindowSize = pDASC->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2-nPatchSize,oTargetPt.y-oWindowSize.height/2-nPatchSize,oWindowSize.width+2*nPatchSize,oWindowSize.height+2*nPatchSize)).clone();
    cv::Mat_<float> oOutputDescMap1;
    pDASC->compute2(oInputCrop,oOutputDescMap1);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap1.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap1.size[1]);
    std::vector<cv::KeyPoint> vKeyPoints;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            vKeyPoints.emplace_back(cv::Point2f(float(nColIdx),float(nRowIdx)),float(std::max(oWindowSize.height,oWindowSize.width)));
    std::vector<cv::KeyPoint> vKeyPoints_modif = vKeyPoints;
    cv::Mat_<float> oOutputDescMap2;
    pDASC->compute2(oInputCrop,vKeyPoints_modif,oOutputDescMap2);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap2.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap2.size[1]);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    ASSERT_EQ(oOutputDescMap1.size[2],oOutputDescMap2.size[2]);
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx) {
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx) {
            for(int nDescIdx=0; nDescIdx<oOutputDescMap1.size[2]; ++nDescIdx) {
                ASSERT_FLOAT_EQ(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx),oOutputDescMap2.at<float>(nRowIdx,nColIdx,nDescIdx));
                ASSERT_FALSE(std::isnan(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx)));
                ASSERT_GE(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx),0.0f);
            }
        }
    }
    cv::Mat_<float> oDistMap;
    pDASC->calcDistances(oOutputDescMap1,oOutputDescMap2,oDistMap);
    ASSERT_EQ(oDistMap.rows,oInputCrop.rows);
    ASSERT_EQ(oDistMap.cols,oInputCrop.cols);
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            ASSERT_FLOAT_EQ(oDistMap.at<float>(nRowIdx,nColIdx),0.0f);
    cv::Mat_<float> oOutputDescs;
    pDASC->compute(oInputCrop,vKeyPoints_modif,oOutputDescs);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    ASSERT_EQ(size_t(oOutputDescs.rows),vKeyPoints.size());
    int nKeyPointIdx = 0;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx) {
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx) {
            for(int nDescIdx=0; nDescIdx<oOutputDescMap1.size[2]; ++nDescIdx)
                ASSERT_FLOAT_EQ(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx),oOutputDescs.at<float>(nKeyPointIdx,nDescIdx));
            ++nKeyPointIdx;
        }
    }
}

TEST(dasc_gf,regression_constr) {
    EXPECT_THROW_LV_QUIET(lv::doNotOptimize(std::make_unique<DASC>(size_t(0),0.05f)));
    EXPECT_THROW_LV_QUIET(lv::doNotOptimize(std::make_unique<DASC>(size_t(1),0.0f)));
    EXPECT_THROW_LV_QUIET(lv::doNotOptimize(std::make_unique<DASC>(size_t(1),0.05f,size_t(0))));
}

TEST(dasc_gf,regression_default_params) {
    std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS);
    EXPECT_GT(pDASC->windowSize().width,0);
    EXPECT_GT(pDASC->windowSize().height,0);
    EXPECT_TRUE((pDASC->windowSize().width%2)==1);
    EXPECT_TRUE((pDASC->windowSize().height%2)==1);
    EXPECT_EQ(pDASC->windowSize().width,pDASC->windowSize().height);
    EXPECT_EQ(pDASC->windowSize().width/2,pDASC->borderSize());
    EXPECT_EQ(pDASC->windowSize().width/2,pDASC->borderSize(1));
    ASSERT_THROW_LV_QUIET(pDASC->borderSize(2));
    EXPECT_GT(pDASC->descriptorSize(),0);
    EXPECT_EQ(size_t(pDASC->descriptorSize())%sizeof(float),size_t(0));
    EXPECT_EQ(pDASC->descriptorType(),CV_32F);
    EXPECT_EQ(pDASC->descriptorType(),CV_32FC1);
    EXPECT_EQ(pDASC->defaultNorm(),cv::NORM_L2);
    EXPECT_FALSE(pDASC->isUsingRF());
    EXPECT_EQ(pDASC->isPreProcessing(),DASC_DEFAULT_PREPROCESS);
}

TEST(dasc_gf,regression_single_compute) {
    std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS);
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty());
    const cv::Point2i oTargetPt(364,135);
    const cv::Size oWindowSize = pDASC->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2,oTargetPt.y-oWindowSize.height/2,oWindowSize.width,oWindowSize.height)).clone();
    cv::Mat_<float> oOutputDescMap;
    pDASC->compute2(oInputCrop,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    const cv::Mat_<float> oOutputDesc = cv::Mat_<float>(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2)).clone();
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_dasc_gf.bin")) {
        const cv::Mat_<float> oRefDesc = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_dasc_gf.bin");
        ASSERT_EQ(oOutputDesc.total(),oRefDesc.total());
        ASSERT_EQ(oOutputDesc.size,oRefDesc.size);
        for(int nDescIdx=0; nDescIdx<oRefDesc.size[2]; ++nDescIdx)
            ASSERT_NEAR_MINRATIO(oOutputDesc.at<float>(0,0,nDescIdx),oRefDesc.at<float>(0,0,nDescIdx),0.04f);
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_dasc_gf.bin",oOutputDesc);
    const std::vector<cv::KeyPoint> vKeyPoints_orig = {cv::KeyPoint(cv::Point2f(float(oWindowSize.width/2),float(oWindowSize.height/2)),float(std::max(oWindowSize.height,oWindowSize.width)))};
    std::vector<cv::KeyPoint> vKeyPoints = vKeyPoints_orig;
    pDASC->compute2(oInputCrop,vKeyPoints,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(vKeyPoints.size(),size_t(1));
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.x,vKeyPoints_orig[0].pt.x);
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.y,vKeyPoints_orig[0].pt.y);
    const cv::Mat_<float> oOutputKPDesc(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2));
    ASSERT_EQ(oOutputDesc.total(),oOutputKPDesc.total());
    ASSERT_EQ(oOutputDesc.size,oOutputKPDesc.size);
    for(int nDescIdx=0; nDescIdx<oOutputDesc.size[2]; ++nDescIdx) {
        ASSERT_FLOAT_EQ(oOutputDesc.at<float>(0,0,nDescIdx),oOutputKPDesc.at<float>(0,0,nDescIdx));
        ASSERT_FALSE(std::isnan(oOutputDesc.at<float>(0,0,nDescIdx)));
        ASSERT_GE(oOutputDesc.at<float>(0,0,nDescIdx),0.0f);
    }
    ASSERT_FLOAT_EQ(float(pDASC->calcDistance(oOutputDesc,oOutputKPDesc)),0.0f);
}

TEST(dasc_gf,regression_large_compute) {
    std::unique_ptr<DASC> pDASC = std::make_unique<DASC>(DASC_DEFAULT_GF_RADIUS,DASC_DEFAULT_GF_EPS);
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty());
    const cv::Point2i oTargetPt(364,135);
    const int nPatchSize = 10;
    const cv::Size oWindowSize = pDASC->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2-nPatchSize,oTargetPt.y-oWindowSize.height/2-nPatchSize,oWindowSize.width+2*nPatchSize,oWindowSize.height+2*nPatchSize)).clone();
    cv::Mat_<float> oOutputDescMap1;
    pDASC->compute2(oInputCrop,oOutputDescMap1);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap1.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap1.size[1]);
    std::vector<cv::KeyPoint> vKeyPoints;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            vKeyPoints.emplace_back(cv::Point2f(float(nColIdx),float(nRowIdx)),float(std::max(oWindowSize.height,oWindowSize.width)));
    std::vector<cv::KeyPoint> vKeyPoints_modif = vKeyPoints;
    cv::Mat_<float> oOutputDescMap2;
    pDASC->compute2(oInputCrop,vKeyPoints_modif,oOutputDescMap2);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap2.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap2.size[1]);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    ASSERT_EQ(oOutputDescMap1.size[2],oOutputDescMap2.size[2]);
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx) {
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx) {
            for(int nDescIdx=0; nDescIdx<oOutputDescMap1.size[2]; ++nDescIdx) {
                ASSERT_FLOAT_EQ(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx),oOutputDescMap2.at<float>(nRowIdx,nColIdx,nDescIdx));
                ASSERT_FALSE(std::isnan(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx)));
                ASSERT_GE(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx),0.0f);
            }
        }
    }
    cv::Mat_<float> oDistMap;
    pDASC->calcDistances(oOutputDescMap1,oOutputDescMap2,oDistMap);
    ASSERT_EQ(oDistMap.rows,oInputCrop.rows);
    ASSERT_EQ(oDistMap.cols,oInputCrop.cols);
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            ASSERT_FLOAT_EQ(oDistMap.at<float>(nRowIdx,nColIdx),0.0f);
    cv::Mat_<float> oOutputDescs;
    pDASC->compute(oInputCrop,vKeyPoints_modif,oOutputDescs);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    ASSERT_EQ(size_t(oOutputDescs.rows),vKeyPoints.size());
    int nKeyPointIdx = 0;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx) {
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx) {
            for(int nDescIdx=0; nDescIdx<oOutputDescMap1.size[2]; ++nDescIdx)
                ASSERT_FLOAT_EQ(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx),oOutputDescs.at<float>(nKeyPointIdx,nDescIdx));
            ++nKeyPointIdx;
        }
    }
}