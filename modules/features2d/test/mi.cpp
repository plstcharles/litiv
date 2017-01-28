
#include "litiv/features2d/MI.hpp"
#include "litiv/utils/opencv.hpp"
#include "litiv/test.hpp"

TEST(mi,regression_constr) {
    EXPECT_THROW_LVQUIET(std::make_unique<MutualInfo>(cv::Size(0,0)),lv::Exception);
    EXPECT_THROW_LVQUIET(std::make_unique<MutualInfo>(cv::Size(2,2)),lv::Exception);
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
    ASSERT_THROW_LVQUIET(pMI->borderSize(2),lv::Exception);
}

TEST(mi,regression_compute) {
    std::unique_ptr<MutualInfo> pMI = std::make_unique<MutualInfo>();
    const cv::Mat oInput1 = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    cv::Mat oInput2;
    cv::cvtColor(oInput1,oInput2,cv::COLOR_BGR2GRAY);
    cv::Mat(cv::Scalar_<uchar>(255)-oInput2).copyTo(oInput2);
    cv::GaussianBlur(oInput2,oInput2,cv::Size(7,7),0.0);
    const cv::Point2i oTargetPt(364,135);
    const cv::Size oWindowSize = pMI->windowSize();
    const cv::Rect oCropZone(oTargetPt.x-oWindowSize.width/2,oTargetPt.y-oWindowSize.height/2,oWindowSize.width,oWindowSize.height);
    const cv::Mat oInputCrop1 = oInput1(oCropZone).clone();
    const cv::Mat oInputCrop2 = oInput2(oCropZone).clone();
    const double dSingleScore = pMI->compute(oInputCrop1,oInputCrop2);
    ASSERT_GT(dSingleScore,0.0);
    std::vector<cv::KeyPoint> vKeyPoints;
    size_t nCurrIdx=0, nSingleTestIdx=0;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<oInput1.rows-oWindowSize.height/2; ++nRowIdx) {
        for(int nColIdx=oWindowSize.width/2; nColIdx<oInput1.cols-oWindowSize.width/2; ++nColIdx) {
            vKeyPoints.emplace_back(cv::Point2f(float(nColIdx),float(nRowIdx)),float(std::max(oWindowSize.height,oWindowSize.width)));
            if(nColIdx==oTargetPt.x && nRowIdx==oTargetPt.y)
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
        ASSERT_TRUE(cv::isEqual<double>(oOutputScoresMat,oRefScoresMat));
    }
    else
        cv::write(TEST_CURR_INPUT_DATA_ROOT "/test_mi.bin",oOutputScoresMat);
}