
#include "litiv/features2d.hpp"
#include "litiv/test.hpp"

TEST(localDiff,regression) {
    // ... @@@@ TODO
}

TEST(calcJointProbHist,regression_mixed3d) {
    std::array<int,3> vals = {10,356,-5};
    cv::Mat_<uchar> test1(3,3); test1 = uchar(vals[0]);
    cv::Mat_<short> test2(3,3); test2 = short(vals[1]);
    cv::Mat_<int> test3(3,3); test3 = vals[2];
    const auto oHistData = lv::calcJointProbHist(std::make_tuple(test1,test2,test3));
    ASSERT_EQ(oHistData.nDims,3);
    for(size_t i=0; i<oHistData.nDims; ++i) {
        EXPECT_EQ(oHistData.aMinVals[i],vals[i]);
        EXPECT_EQ(oHistData.aMaxVals[i],vals[i]);
        ASSERT_EQ(oHistData.aStates[i],size_t(1));
        ASSERT_EQ(oHistData.aMargHists[i].nzcount(),size_t(1));
        EXPECT_FLOAT_EQ(oHistData.aMargHists[i](0),1.0f);
    }
    ASSERT_EQ(oHistData.nJointStates,size_t(1));
    EXPECT_FLOAT_EQ(oHistData.oJointHist(0,0,0),1.0f);
}

TEST(calcJointProbHist,regression_inv2d) {
    cv::Mat_<uchar> test1(2,2); test1 = 255; test1(0,0) = 0;
    cv::Mat_<uchar> test2(2,2); test2 = 0; test2(0,0) = 255;
    const auto oHistData = lv::calcJointProbHist(std::make_tuple(test1,test2));
    ASSERT_EQ(oHistData.nDims,2);
    for(size_t i=0; i<oHistData.nDims; ++i) {
        EXPECT_EQ(oHistData.aMinVals[i],0);
        EXPECT_EQ(oHistData.aMaxVals[i],255);
        ASSERT_EQ(oHistData.aStates[i],size_t(2));
        ASSERT_EQ(oHistData.aMargHists[i].nzcount(),size_t(2));
    }
    EXPECT_FLOAT_EQ(oHistData.aMargHists[0](0),0.25f);
    EXPECT_FLOAT_EQ(oHistData.aMargHists[0](255),0.75f);
    EXPECT_FLOAT_EQ(oHistData.aMargHists[1](0),0.75f);
    EXPECT_FLOAT_EQ(oHistData.aMargHists[1](255),0.25f);
    ASSERT_EQ(oHistData.nJointStates,size_t(4));
    EXPECT_FLOAT_EQ(oHistData.oJointHist(0,255),0.25f);
    EXPECT_FLOAT_EQ(oHistData.oJointHist(255,0),0.75f);
}
/*
TEST(calcJointProbHist,regression_inv2d_fullrange) {
    cv::Mat_<uchar> test1(2,2); test1 = 255; test1(0,0) = 0;
    cv::Mat_<uchar> test2(2,2); test2 = 0; test2(0,0) = 255;
    const auto oHistData = lv::calcJointProbHist<1,true>(std::make_tuple(test1,test2));
    ASSERT_EQ(oHistData.nDims,2);
    for(size_t i=0; i<oHistData.nDims; ++i) {
        EXPECT_EQ(oHistData.aMinVals[i],0);
        EXPECT_EQ(oHistData.aMaxVals[i],255);
        ASSERT_EQ(oHistData.aStates[i],size_t(256));
        ASSERT_EQ(oHistData.aMargHists[i].total(),size_t(256));
    }
    EXPECT_FLOAT_EQ(oHistData.aMargHists[0](0),0.25f);
    EXPECT_FLOAT_EQ(oHistData.aMargHists[0](255),0.75f);
    EXPECT_FLOAT_EQ(oHistData.aMargHists[1](0),0.75f);
    EXPECT_FLOAT_EQ(oHistData.aMargHists[1](255),0.25f);
    ASSERT_EQ(oHistData.nJointStates,size_t(256*256));
    EXPECT_FLOAT_EQ(oHistData.oJointHist(0,255),0.25f);
    EXPECT_FLOAT_EQ(oHistData.oJointHist(255,0),0.75f);
}*/