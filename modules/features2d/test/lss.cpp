
#include "litiv/features2d/LSS.hpp"
#include "litiv/utils/opencv.hpp"
#include "litiv/test.hpp"

TEST(lss,regression_default_constr) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>();
    EXPECT_GT(pLSS->windowSize().width,0);
    EXPECT_GT(pLSS->windowSize().height,0);
    EXPECT_TRUE((pLSS->windowSize().width%2)==1);
    EXPECT_TRUE((pLSS->windowSize().height%2)==1);
    EXPECT_EQ(pLSS->windowSize().width,pLSS->windowSize().height);
    EXPECT_EQ(pLSS->windowSize().width/2,pLSS->borderSize());
    EXPECT_EQ(pLSS->windowSize().width/2,pLSS->borderSize(1));
    ASSERT_THROW_LV_QUIET(pLSS->borderSize(2));
    EXPECT_GT(pLSS->descriptorSize(),0);
    EXPECT_EQ(size_t(pLSS->descriptorSize())%sizeof(float),size_t(0));
    EXPECT_EQ(pLSS->descriptorType(),CV_32F);
    EXPECT_EQ(pLSS->descriptorType(),CV_32FC1);
    EXPECT_EQ(pLSS->defaultNorm(),cv::NORM_L2);
}

TEST(lss,regression_single_compute) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>(
        /*LSS_DEFAULT_INNER_RADIUS*/0,
        /*LSS_DEFAULT_OUTER_RADIUS*/40,
        /*LSS_DEFAULT_PATCH_SIZE*/5,
        /*LSS_DEFAULT_ANGULAR_BINS*/12,
        /*LSS_DEFAULT_RADIAL_BINS*/3,
        /*LSS_DEFAULT_STATNOISE_VAR*/300000.f,
        /*LSS_DEFAULT_NORM_BINS*/false,
        /*LSS_DEFAULT_PREPROCESS*/false,
        /*LSS_DEFAULT_USE_LIENH_MASK*/false
    );
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    const cv::Point2i oTargetPt(364,135);
    const cv::Size oWindowSize = pLSS->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2,oTargetPt.y-oWindowSize.height/2,oWindowSize.width,oWindowSize.height)).clone();
    cv::Mat_<float> oOutputDescMap;
    pLSS->compute2(oInputCrop,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    const cv::Mat_<float> oOutputDesc = cv::Mat_<float>(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2)).clone();
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_lss.bin")) {
        const cv::Mat_<float> oRefDesc = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_lss.bin");
        ASSERT_EQ(oOutputDesc.total(),oRefDesc.total());
        ASSERT_EQ(oOutputDesc.size,oRefDesc.size);
        for(int nDescIdx=0; nDescIdx<oRefDesc.size[2]; ++nDescIdx)
            ASSERT_NEAR_MINRATIO(oOutputDesc.at<float>(0,0,nDescIdx),oRefDesc.at<float>(0,0,nDescIdx),0.05f);
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_lss.bin",oOutputDesc);
    const std::vector<cv::KeyPoint> vKeyPoints_orig = {cv::KeyPoint(cv::Point2f(float(oWindowSize.width/2),float(oWindowSize.height/2)),float(std::max(oWindowSize.height,oWindowSize.width)))};
    std::vector<cv::KeyPoint> vKeyPoints = vKeyPoints_orig;
    pLSS->compute2(oInputCrop,vKeyPoints,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(vKeyPoints.size(),size_t(1));
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.x,vKeyPoints_orig[0].pt.x);
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.y,vKeyPoints_orig[0].pt.y);
    const cv::Mat_<float> oOutputKPDesc(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2));
    ASSERT_EQ(oOutputDesc.total(),oOutputKPDesc.total());
    ASSERT_EQ(oOutputDesc.size,oOutputKPDesc.size);
    for(int nDescIdx=0; nDescIdx<oOutputDesc.size[2]; ++nDescIdx)
        ASSERT_FLOAT_EQ(oOutputDesc.at<float>(0,0,nDescIdx),oOutputKPDesc.at<float>(0,0,nDescIdx));
    ASSERT_FLOAT_EQ(float(pLSS->calcDistance(oOutputDesc,oOutputKPDesc)),0.0f);
}

TEST(lss,regression_single_normalized_compute) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>(
        /*LSS_DEFAULT_INNER_RADIUS*/0,
        /*LSS_DEFAULT_OUTER_RADIUS*/40,
        /*LSS_DEFAULT_PATCH_SIZE*/5,
        /*LSS_DEFAULT_ANGULAR_BINS*/12,
        /*LSS_DEFAULT_RADIAL_BINS*/3,
        /*LSS_DEFAULT_STATNOISE_VAR*/300000.f,
        /*LSS_DEFAULT_NORM_BINS*/true,
        /*LSS_DEFAULT_PREPROCESS*/false,
        /*LSS_DEFAULT_USE_LIENH_MASK*/false
    );
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    const cv::Point2i oTargetPt(364,135);
    const cv::Size oWindowSize = pLSS->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2,oTargetPt.y-oWindowSize.height/2,oWindowSize.width,oWindowSize.height)).clone();
    cv::Mat_<float> oOutputDescMap;
    pLSS->compute2(oInputCrop,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    const cv::Mat_<float> oOutputDesc = cv::Mat_<float>(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2)).clone();
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_lss_norm.bin")) {
        const cv::Mat_<float> oRefDesc = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_lss_norm.bin");
        ASSERT_EQ(oOutputDesc.total(),oRefDesc.total());
        ASSERT_EQ(oOutputDesc.size,oRefDesc.size);
        for(int nDescIdx=0; nDescIdx<oRefDesc.size[2]; ++nDescIdx)
            ASSERT_NEAR_MINRATIO(oOutputDesc.at<float>(0,0,nDescIdx),oRefDesc.at<float>(0,0,nDescIdx),0.05f);
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_lss_norm.bin",oOutputDesc);
    const std::vector<cv::KeyPoint> vKeyPoints_orig = {cv::KeyPoint(cv::Point2f(float(oWindowSize.width/2),float(oWindowSize.height/2)),float(std::max(oWindowSize.height,oWindowSize.width)))};
    std::vector<cv::KeyPoint> vKeyPoints = vKeyPoints_orig;
    pLSS->compute2(oInputCrop,vKeyPoints,oOutputDescMap);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(vKeyPoints.size(),size_t(1));
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.x,vKeyPoints_orig[0].pt.x);
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.y,vKeyPoints_orig[0].pt.y);
    const cv::Mat_<float> oOutputKPDesc(3,std::array<int,3>{1,1,oOutputDescMap.size[2]}.data(),oOutputDescMap.ptr<float>(oWindowSize.height/2,oWindowSize.width/2));
    ASSERT_EQ(oOutputDesc.total(),oOutputKPDesc.total());
    ASSERT_EQ(oOutputDesc.size,oOutputKPDesc.size);
    for(int nDescIdx=0; nDescIdx<oOutputDesc.size[2]; ++nDescIdx)
        ASSERT_FLOAT_EQ(oOutputDesc.at<float>(0,0,nDescIdx),oOutputKPDesc.at<float>(0,0,nDescIdx));
    ASSERT_FLOAT_EQ(float(pLSS->calcDistance(oOutputDesc,oOutputKPDesc)),0.0f);
    ASSERT_FLOAT_EQ((float)cv::norm(oOutputKPDesc,cv::NORM_L2),1.0f);
}

TEST(lss,regression_single_invert_compute) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>();
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    const cv::Mat oInput_new = cv::Vec3b(255,255,255)-oInput;
    const cv::Point2i oTargetPt(364,135);
    cv::Mat_<float> oOutputDescMap1,oOutputDescMap2;
    const std::vector<cv::KeyPoint> vKeyPoints_orig = {cv::KeyPoint(cv::Point2f(float(oTargetPt.x),float(oTargetPt.y)),1.0f)};
    std::vector<cv::KeyPoint> vKeyPoints = vKeyPoints_orig;
    pLSS->compute2(oInput,vKeyPoints,oOutputDescMap1);
    ASSERT_EQ(oInput.size[0],oOutputDescMap1.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap1.size[1]);
    ASSERT_EQ(vKeyPoints.size(),size_t(1));
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.x,vKeyPoints_orig[0].pt.x);
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.y,vKeyPoints_orig[0].pt.y);
    pLSS->compute2(oInput_new,vKeyPoints,oOutputDescMap2);
    ASSERT_EQ(oInput_new.size[0],oOutputDescMap2.size[0]);
    ASSERT_EQ(oInput_new.size[1],oOutputDescMap2.size[1]);
    ASSERT_EQ(vKeyPoints.size(),size_t(1));
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.x,vKeyPoints_orig[0].pt.x);
    ASSERT_FLOAT_EQ(vKeyPoints[0].pt.y,vKeyPoints_orig[0].pt.y);
    const cv::Mat_<float> oOutputKPDesc1(3,std::array<int,3>{1,1,oOutputDescMap1.size[2]}.data(),oOutputDescMap1.ptr<float>(oTargetPt.y,oTargetPt.x));
    ASSERT_FLOAT_EQ((float)cv::norm(oOutputKPDesc1,cv::NORM_L2),1.0f);
    const cv::Mat_<float> oOutputKPDesc2(3,std::array<int,3>{1,1,oOutputDescMap2.size[2]}.data(),oOutputDescMap2.ptr<float>(oTargetPt.y,oTargetPt.x));
    ASSERT_FLOAT_EQ((float)cv::norm(oOutputKPDesc2,cv::NORM_L2),1.0f);
    ASSERT_NEAR(float(pLSS->calcDistance(oOutputKPDesc1,oOutputKPDesc2)),0.0f,(float)1e-5);
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
    pLSS->calcDistances(oOutputDescMap1,oOutputDescMap2,oDistMap);
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

TEST(lss,regression_large_transl_invert_compute) {
    std::unique_ptr<LSS> pLSS = std::make_unique<LSS>();
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    cv::Mat oInput_new;
    lv::shift(cv::Vec3b(255,255,255)-oInput,oInput_new,cv::Point2f(10.0f,0.0f));
    ASSERT_EQ(lv::MatInfo(oInput),lv::MatInfo(oInput_new));
    const cv::Point2i oTargetPt(364,135);
    const cv::Point2i oTargetPt_new(374,135);
    cv::Mat_<float> oOutputDescMap1,oOutputDescMap2;
    pLSS->compute2(oInput,oOutputDescMap1);
    ASSERT_EQ(oInput.size[0],oOutputDescMap1.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap1.size[1]);
    pLSS->compute2(oInput_new,oOutputDescMap2);
    ASSERT_EQ(oInput_new.size[0],oOutputDescMap2.size[0]);
    ASSERT_EQ(oInput_new.size[1],oOutputDescMap2.size[1]);
    const cv::Mat_<float> oOutputKPDesc1(3,std::array<int,3>{1,1,oOutputDescMap1.size[2]}.data(),oOutputDescMap1.ptr<float>(oTargetPt.y,oTargetPt.x));
    ASSERT_FLOAT_EQ((float)cv::norm(oOutputKPDesc1,cv::NORM_L2),1.0f);
    const cv::Mat_<float> oOutputKPDesc2(3,std::array<int,3>{1,1,oOutputDescMap2.size[2]}.data(),oOutputDescMap2.ptr<float>(oTargetPt_new.y,oTargetPt_new.x));
    ASSERT_FLOAT_EQ((float)cv::norm(oOutputKPDesc2,cv::NORM_L2),1.0f);
    ASSERT_NEAR(float(pLSS->calcDistance(oOutputKPDesc1,oOutputKPDesc2)),0.0f,(float)1e-5);
}