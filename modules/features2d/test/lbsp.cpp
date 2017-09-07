
#include "litiv/features2d/LBSP.hpp"
#include "litiv/test.hpp"

TEST(lbsp,regression_constr) {
    EXPECT_THROW_LV_QUIET(std::make_unique<LBSP>(-0.5f));
}

TEST(lbsp,regression_default_params) {
    std::unique_ptr<LBSP> pLBSP = std::make_unique<LBSP>(size_t(20));
    EXPECT_EQ(pLBSP->windowSize().width,5);
    EXPECT_EQ(pLBSP->windowSize().height,5);
    EXPECT_EQ(pLBSP->windowSize().width/2,pLBSP->borderSize());
    EXPECT_EQ(pLBSP->windowSize().width/2,pLBSP->borderSize(1));
    ASSERT_THROW_LV_QUIET(pLBSP->borderSize(2));
    EXPECT_EQ(pLBSP->descriptorSize(),2);
    EXPECT_EQ(pLBSP->descriptorType(),CV_16U);
    EXPECT_EQ(pLBSP->descriptorType(),CV_16UC1);
    EXPECT_EQ(pLBSP->defaultNorm(),cv::NORM_HAMMING);
}

TEST(lbsp,regression_compute) {
    std::unique_ptr<LBSP> pLBSP = std::make_unique<LBSP>(size_t(20));
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/multispectral_stereo_ex/img2.png");
    const cv::Point2i oTargetPt(371,371);
    const int nPatchSize = 30;
    const cv::Size oWindowSize = pLBSP->windowSize();
    const int nBorderSize = pLBSP->borderSize();
    const cv::Mat oInputCrop = oInput(cv::Rect(oTargetPt.x-nBorderSize-nPatchSize,oTargetPt.y-nBorderSize-nPatchSize,oWindowSize.width+2*nPatchSize,oWindowSize.height+2*nPatchSize)).clone();
    cv::Mat oOutputDescMap1;
    pLBSP->compute2(oInputCrop,oOutputDescMap1);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap1.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap1.size[1]);
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_lbsp.bin")) {
        const cv::Rect oValidZone(nBorderSize,nBorderSize,oInputCrop.cols-nBorderSize*2,oInputCrop.rows-nBorderSize*2);
        const cv::Mat oOutputDescMap1_valid = oOutputDescMap1(oValidZone).clone();
        const cv::Mat oDescRefMap = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_lbsp.bin");
        const cv::Mat oDescRefMap_valid = oDescRefMap(oValidZone).clone();
        for(int i=0; i<oDescRefMap_valid.rows; ++i)
            for(int j=0; j<oDescRefMap_valid.cols; ++j)
                for(int k=0; k<oDescRefMap_valid.channels(); ++k)
                    EXPECT_EQ(oOutputDescMap1_valid.ptr<ushort>(i,j)[k],oDescRefMap_valid.ptr<ushort>(i,j)[k]) << "i=" << i << ", j=" << j << ", k=" << k;
        ASSERT_TRUE(lv::isEqual<uchar>(oOutputDescMap1_valid,oDescRefMap_valid));
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_lbsp.bin",oOutputDescMap1);
    std::vector<cv::KeyPoint> vKeyPoints;
    for(int nRowIdx=nBorderSize; nRowIdx<=nBorderSize+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=nBorderSize; nColIdx<=nBorderSize+2*nPatchSize; ++nColIdx)
            vKeyPoints.emplace_back(cv::Point2f(float(nColIdx),float(nRowIdx)),float(std::max(oWindowSize.height,oWindowSize.width)));
    std::vector<cv::KeyPoint> vKeyPoints_modif = vKeyPoints;
    cv::Mat oOutputDescMap2;
    pLBSP->compute2(oInputCrop,vKeyPoints_modif,oOutputDescMap2);
    ASSERT_EQ(oInputCrop.size[0],oOutputDescMap2.size[0]);
    ASSERT_EQ(oInputCrop.size[1],oOutputDescMap2.size[1]);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    for(int nRowIdx=nBorderSize; nRowIdx<=nBorderSize+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=nBorderSize; nColIdx<=nBorderSize+2*nPatchSize; ++nColIdx)
            for(int nChIdx=0; nChIdx<oInput.channels(); ++nChIdx)
                ASSERT_EQ(oOutputDescMap1.ptr<ushort>(nRowIdx,nColIdx)[nChIdx],oOutputDescMap2.ptr<ushort>(nRowIdx,nColIdx)[nChIdx]);
    cv::Mat_<uchar> oDistMap;
    pLBSP->calcDistances(oOutputDescMap1,oOutputDescMap2,oDistMap);
    ASSERT_EQ(oDistMap.rows,oInputCrop.rows);
    ASSERT_EQ(oDistMap.cols,oInputCrop.cols);
    for(int nRowIdx=nBorderSize; nRowIdx<=nBorderSize+2*nPatchSize; ++nRowIdx)
        for(int nColIdx=nBorderSize; nColIdx<=nBorderSize+2*nPatchSize; ++nColIdx)
            ASSERT_EQ(oDistMap.at<uchar>(nRowIdx,nColIdx),uchar(0));
    cv::Mat oOutputDescs;
    pLBSP->compute(oInputCrop,vKeyPoints_modif,oOutputDescs);
    ASSERT_EQ(vKeyPoints_modif.size(),vKeyPoints.size());
    ASSERT_EQ(size_t(oOutputDescs.rows),vKeyPoints.size());
    int nKeyPointIdx = 0;
    for(int nRowIdx=nBorderSize; nRowIdx<=nBorderSize+2*nPatchSize; ++nRowIdx) {
        for(int nColIdx=nBorderSize; nColIdx<=nBorderSize+2*nPatchSize; ++nColIdx) {
            for(int nChIdx=0; nChIdx<oInput.channels(); ++nChIdx)
                ASSERT_EQ(oOutputDescMap1.ptr<ushort>(nRowIdx,nColIdx)[nChIdx],oOutputDescs.ptr<ushort>(nKeyPointIdx)[nChIdx]);
            ++nKeyPointIdx;
        }
    }
}