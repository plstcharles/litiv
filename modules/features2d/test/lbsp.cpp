
#include "litiv/features2d/LBSP.hpp"
#include "litiv/utils/opencv.hpp"
#include "litiv/test.hpp"

TEST(lbsp,regression_constr) {
    EXPECT_THROW_LVQUIET(std::make_unique<LBSP>(-0.5f),lv::Exception);
}

TEST(lbsp,regression_default_params) {
    std::unique_ptr<LBSP> pLBSP = std::make_unique<LBSP>(size_t(20));
    EXPECT_EQ(pLBSP->windowSize().width,5);
    EXPECT_EQ(pLBSP->windowSize().height,5);
    EXPECT_EQ(pLBSP->windowSize().width/2,pLBSP->borderSize());
    EXPECT_EQ(pLBSP->windowSize().width/2,pLBSP->borderSize(1));
    ASSERT_THROW_LVQUIET(pLBSP->borderSize(2),lv::Exception);
    EXPECT_EQ(pLBSP->descriptorSize(),2);
    EXPECT_EQ(pLBSP->descriptorType(),CV_16U);
    EXPECT_EQ(pLBSP->descriptorType(),CV_16UC1);
    EXPECT_EQ(pLBSP->defaultNorm(),cv::NORM_HAMMING);
}

TEST(lbsp,regression_compute) {
    std::unique_ptr<LBSP> pLBSP = std::make_unique<LBSP>(size_t(20));
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    const cv::Point2i oTargetPt(364,135);
    const int nPatchSize = 10;
    const cv::Size oWindowSize = pLBSP->windowSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-oWindowSize.width/2-nPatchSize,oTargetPt.y-oWindowSize.height/2-nPatchSize,oWindowSize.width+2*nPatchSize,oWindowSize.height+2*nPatchSize)).clone();
    cv::Mat oOutputDescMap1;
    pLBSP->compute2(oInputCrop,oOutputDescMap1);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap1.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap1.size[1]);
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_lbsp.bin"))
        ASSERT_TRUE(cv::isEqual<ushort>(oOutputDescMap1,cv::read(TEST_CURR_INPUT_DATA_ROOT "/test_lbsp.bin")));
    else
        cv::write(TEST_CURR_INPUT_DATA_ROOT "/test_lbsp.bin",oOutputDescMap1);
    std::vector<cv::KeyPoint> vKeyPoints;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            vKeyPoints.emplace_back(cv::Point2f(float(nColIdx),float(nRowIdx)),float(std::max(oWindowSize.height,oWindowSize.width)));
    std::vector<cv::KeyPoint> vKeyPoints_modif = vKeyPoints;
    cv::Mat oOutputDescMap2;
    pLBSP->compute2(oInputCrop,vKeyPoints_modif,oOutputDescMap2);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap2.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap2.size[1]);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            for(int nChIdx=0; nChIdx<oInput.channels(); ++nChIdx)
                ASSERT_EQ(oOutputDescMap1.ptr<ushort>(nRowIdx,nColIdx)[nChIdx],oOutputDescMap2.ptr<ushort>(nRowIdx,nColIdx)[nChIdx]);
    cv::Mat_<uchar> oDistMap;
    pLBSP->calcDistance(oOutputDescMap1,oOutputDescMap2,oDistMap);
    ASSERT_EQ(oDistMap.rows,oInputCrop.rows);
    ASSERT_EQ(oDistMap.cols,oInputCrop.cols);
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx)
            ASSERT_EQ(oDistMap.at<uchar>(nRowIdx,nColIdx),uchar(0));
    cv::Mat oOutputDescs;
    pLBSP->compute(oInputCrop,vKeyPoints_modif,oOutputDescs);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    ASSERT_EQ(size_t(oOutputDescs.rows),vKeyPoints.size());
    int nKeyPointIdx = 0;
    for(int nRowIdx=oWindowSize.height/2; nRowIdx<=oWindowSize.height/2+2*nPatchSize; ++nRowIdx) {
        for(int nColIdx=oWindowSize.width/2; nColIdx<=oWindowSize.width/2+2*nPatchSize; ++nColIdx) {
            for(int nChIdx=0; nChIdx<oInput.channels(); ++nChIdx)
                ASSERT_EQ(oOutputDescMap1.ptr<ushort>(nRowIdx,nColIdx)[nChIdx],oOutputDescs.ptr<ushort>(nKeyPointIdx)[nChIdx]);
            ++nKeyPointIdx;
        }
    }
}