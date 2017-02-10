
#include "litiv/features2d/MI.hpp"
#include "litiv/utils/opencv.hpp"
#include "litiv/test.hpp"

#if __cplusplus>=201402L

TEST(mi,regression_constr) {
    EXPECT_THROW_LV_QUIET(std::make_unique<MutualInfo>(cv::Size(0,0)));
    EXPECT_THROW_LV_QUIET(std::make_unique<MutualInfo>(cv::Size(2,2)));
}

TEST(mi,regression_default_params) {
    std::unique_ptr<MutualInfo> pMI = std::make_unique<MutualInfo>();
    EXPECT_GT(pMI->windowSize().width,0);
    EXPECT_GT(pMI->windowSize().height,0);
    EXPECT_TRUE((pMI->windowSize().width%2)==1);
    EXPECT_TRUE((pMI->windowSize().height%2)==1);
    EXPECT_EQ(pMI->windowSize().width,pMI->windowSize().height);
    EXPECT_EQ(pMI->windowSize().width/2,pMI->borderSize());
    EXPECT_EQ(pMI->windowSize().width/2,pMI->borderSize(1));
    ASSERT_THROW_LV_QUIET(pMI->borderSize(2));
}

TEST(mi,regression_compute) {
    std::unique_ptr<MutualInfo> pMI = std::make_unique<MutualInfo>();
    const cv::Mat oInput1 = cv::imread(SAMPLES_DATA_ROOT "/multispectral_stereo_ex/img2.png");
    ASSERT_TRUE(!oInput1.empty());
    const cv::Mat oInput2 = cv::imread(SAMPLES_DATA_ROOT "/multispectral_stereo_ex/img1_corr_h0v8.png",cv::IMREAD_GRAYSCALE);
    ASSERT_TRUE(!oInput2.empty());
    const cv::Point oTargetPoint(603,122);
    const cv::Size oWindowSize = pMI->windowSize();
    const cv::Rect oCropZone(oTargetPoint.x-oWindowSize.width/2,oTargetPoint.y-oWindowSize.height/2,oWindowSize.width,oWindowSize.height);
    const cv::Mat oInputCrop1 = oInput1(oCropZone).clone();
    const cv::Mat oInputCrop2 = oInput2(oCropZone).clone();
    const double dSingleScore = pMI->compute(oInputCrop1,oInputCrop2);
    ASSERT_GT(dSingleScore,0.0);
    const cv::Size oPatchSize(100,60);
    std::vector<cv::KeyPoint> vKeyPoints;
    size_t nCurrIdx=0, nSingleTestIdx=0;
    for(int nRowIdx=oTargetPoint.y-oWindowSize.height/2-oPatchSize.height/2; nRowIdx<oTargetPoint.y+oWindowSize.height/2+oPatchSize.height/2; ++nRowIdx) {
        for(int nColIdx=oTargetPoint.x-oWindowSize.width/2-oPatchSize.width/2; nColIdx<oTargetPoint.x+oWindowSize.width/2+oPatchSize.width/2; ++nColIdx) {
            vKeyPoints.emplace_back(cv::Point2f(float(nColIdx),float(nRowIdx)),float(std::max(oWindowSize.height,oWindowSize.width)));
            if(nColIdx==oTargetPoint.x && nRowIdx==oTargetPoint.y)
                nSingleTestIdx = nCurrIdx;
            ++nCurrIdx;
        }
    }
    std::vector<double> vOutputScores = pMI->compute(oInput1,oInput2,vKeyPoints);
    ASSERT_EQ(vOutputScores.size(),vKeyPoints.size());
    ASSERT_DOUBLE_EQ(dSingleScore,vOutputScores[nSingleTestIdx]);
    cv::Mat_<double> oOutputScoresMat(int(vOutputScores.size()),1,vOutputScores.data());
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_mi.bin")) {
        const cv::Mat_<double> oRefScoresMat = cv::read(TEST_CURR_INPUT_DATA_ROOT "/test_mi.bin");
        ASSERT_EQ(oOutputScoresMat.total(),oRefScoresMat.total());
        ASSERT_EQ(oOutputScoresMat.type(),oRefScoresMat.type());
        for(size_t nIdx=0; nIdx<oOutputScoresMat.total(); ++nIdx)
            ASSERT_NEAR_MINRATIO(oOutputScoresMat(int(nIdx)),oRefScoresMat(int(nIdx)),0.05f);
    }
    else
        cv::write(TEST_CURR_INPUT_DATA_ROOT "/test_mi.bin",oOutputScoresMat);
}

#endif //__cplusplus>=201402L