
#include "litiv/features2d/LSS.hpp"
#include "litiv/utils/opencv.hpp"
#include "litiv/test.hpp"

TEST(lss,regression_constr) {
    EXPECT_THROW_LVQUIET(std::make_unique<LSS>(0),lv::Exception);
    EXPECT_THROW_LVQUIET(std::make_unique<LSS>(2),lv::Exception);
    EXPECT_THROW_LVQUIET(std::make_unique<LSS>(3,1),lv::Exception);
    EXPECT_THROW_LVQUIET(std::make_unique<LSS>(1,3,0),lv::Exception);
    EXPECT_THROW_LVQUIET(std::make_unique<LSS>(1,3,2,0),lv::Exception);
}

TEST(lss,regression_default_params) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>();
    EXPECT_GT(pLSS->windowSize().width,0);
    EXPECT_GT(pLSS->windowSize().height,0);
    EXPECT_TRUE((pLSS->windowSize().width%2)==1);
    EXPECT_TRUE((pLSS->windowSize().height%2)==1);
    EXPECT_EQ(pLSS->windowSize().width,pLSS->windowSize().height);
    EXPECT_EQ(pLSS->windowSize().width/2,pLSS->borderSize());
    EXPECT_EQ(pLSS->windowSize().width/2,pLSS->borderSize(1));
    ASSERT_THROW_LVQUIET(pLSS->borderSize(2),lv::Exception);
    EXPECT_GT(pLSS->descriptorSize(),0);
    EXPECT_EQ(size_t(pLSS->descriptorSize())%sizeof(float),size_t(0));
    EXPECT_EQ(pLSS->descriptorType(),CV_32F);
    EXPECT_EQ(pLSS->descriptorType(),CV_32FC1);
    EXPECT_EQ(pLSS->defaultNorm(),cv::NORM_L2);
}

TEST(lss,regression_single_compute) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>();
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    const cv::Point2i oTargetPt(364,135);
    const cv::Size oWindowSize = pLSS->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2,oTargetPt.y-oWindowSize.height/2,oWindowSize.width,oWindowSize.height)).clone();
    cv::Mat_<float> oOutputDescMap;
    pLSS->compute2(oInputCrop,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    const cv::Mat_<float> oOutputDesc = cv::Mat_<float>(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.width/2,oWindowSize.height/2)).clone();
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_lss.bin")) {
        const cv::Mat_<float> oRefDesc = cv::read(TEST_CURR_INPUT_DATA_ROOT "/test_lss.bin");
        ASSERT_EQ(oOutputDesc.total(),oRefDesc.total());
        ASSERT_EQ(oOutputDesc.size,oRefDesc.size);
        for(int nDescIdx=0; nDescIdx<oRefDesc.size[2]; ++nDescIdx)
            ASSERT_NEAR_MINRATIO(oOutputDesc.at<float>(0,0,nDescIdx),oRefDesc.at<float>(0,0,nDescIdx),0.05f);
    }
    else
        cv::write(TEST_CURR_INPUT_DATA_ROOT "/test_lss.bin",oOutputDesc);
    const std::vector<cv::KeyPoint> vKeyPoints_orig = {cv::KeyPoint(cv::Point2f(float(oWindowSize.width/2),float(oWindowSize.height/2)),float(std::max(oWindowSize.height,oWindowSize.width)))};
    std::vector<cv::KeyPoint> vKeyPoints = vKeyPoints_orig;
    pLSS->compute2(oInputCrop,vKeyPoints,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(vKeyPoints.size(),size_t(1));
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.x,vKeyPoints_orig[0].pt.x);
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.y,vKeyPoints_orig[0].pt.y);
    const cv::Mat_<float> oOutputKPDesc(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.width/2,oWindowSize.height/2));
    ASSERT_EQ(oOutputDesc.total(),oOutputKPDesc.total());
    ASSERT_EQ(oOutputDesc.size,oOutputKPDesc.size);
    for(int nDescIdx=0; nDescIdx<oOutputDesc.size[2]; ++nDescIdx)
        ASSERT_FLOAT_EQ(oOutputDesc.at<float>(0,0,nDescIdx),oOutputKPDesc.at<float>(0,0,nDescIdx));
    ASSERT_FLOAT_EQ(float(pLSS->calcDistance(oOutputDesc,oOutputKPDesc)),0.0f);
}

TEST(lss,regression_large_compute) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>();
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    const cv::Point2i oTargetPt(364,135);
    const int nPatchSize = 10;
    const cv::Size oWindowSize = pLSS->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2-nPatchSize,oTargetPt.y-oWindowSize.height/2-nPatchSize,oWindowSize.width+2*nPatchSize,oWindowSize.height+2*nPatchSize)).clone();
    cv::Mat_<float> oOutputDescMap1;
    pLSS->compute2(oInputCrop,oOutputDescMap1);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap1.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap1.size[1]);
    std::vector<cv::KeyPoint> vKeyPoints;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            vKeyPoints.emplace_back(cv::Point2f(float(nColIdx),float(nRowIdx)),float(std::max(oWindowSize.height,oWindowSize.width)));
    std::vector<cv::KeyPoint> vKeyPoints_modif = vKeyPoints;
    cv::Mat_<float> oOutputDescMap2;
    pLSS->compute2(oInputCrop,vKeyPoints_modif,oOutputDescMap2);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap2.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap2.size[1]);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    ASSERT_EQ(oOutputDescMap1.size[2],oOutputDescMap2.size[2]);
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            for(int nDescIdx=0; nDescIdx<oOutputDescMap1.size[2]; ++nDescIdx)
                ASSERT_FLOAT_EQ(oOutputDescMap1.at<float>(nRowIdx,nColIdx,nDescIdx),oOutputDescMap2.at<float>(nRowIdx,nColIdx,nDescIdx));
    cv::Mat_<float> oDistMap;
    pLSS->calcDistance(oOutputDescMap1,oOutputDescMap2,oDistMap);
    ASSERT_EQ(oDistMap.rows,oInputCrop.rows);
    ASSERT_EQ(oDistMap.cols,oInputCrop.cols);
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            ASSERT_FLOAT_EQ(oDistMap.at<float>(nRowIdx,nColIdx),0.0f);
    cv::Mat_<float> oOutputDescs;
    pLSS->compute(oInputCrop,vKeyPoints_modif,oOutputDescs);
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