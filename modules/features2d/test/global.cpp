
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
    auto lTester = [&](auto oHistData) {
        ASSERT_EQ(oHistData.dims(),size_t(3));
        for(size_t i=0; i<oHistData.dims(); ++i) {
            EXPECT_EQ(oHistData.aMinVals[i],vals[i]);
            EXPECT_EQ(oHistData.aMaxVals[i],vals[i]);
            ASSERT_EQ(oHistData.aStates[i],size_t(1));
            ASSERT_EQ(oHistData.aMargHists[i].nzcount(),size_t(1));
            EXPECT_FLOAT_EQ(oHistData.aMargHists[i](0),1.0f);
        }
        ASSERT_EQ(oHistData.nJointStates,size_t(1));
        EXPECT_FLOAT_EQ(oHistData.oJointHist(0,0,0),1.0f);
    };
    lTester(lv::calcJointProbHist(std::make_tuple(test1,test2,test3)));
    lTester(lv::calcJointProbHist<1,true,true>(std::make_tuple(test1,test2,test3)));
}

TEST(calcJointProbHist,regression_2d2v) {
    cv::Mat_<uchar> test1(2,2); test1 = 255; test1(0,0) = 0;
    cv::Mat_<uchar> test2(2,2); test2 = 0; test2(0,0) = 255;
    auto lTester = [&](auto oHistData) {
        ASSERT_EQ(oHistData.dims(),size_t(2));
        for(size_t i=0; i<oHistData.dims(); ++i) {
            EXPECT_EQ(oHistData.aMinVals[i],0);
            EXPECT_EQ(oHistData.aMaxVals[i],255);
            ASSERT_EQ(oHistData.aStates[i],size_t(2));
            ASSERT_EQ(oHistData.aMargHists[i].nzcount(),size_t(2));
        }
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0](0),0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0](255),0.75f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1](0),0.75f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1](255),0.25f);
        ASSERT_EQ(oHistData.nJointStates,size_t(2));
        EXPECT_FLOAT_EQ(oHistData.oJointHist(0,255),0.25f);
        EXPECT_FLOAT_EQ(oHistData.oJointHist(255,0),0.75f);
    };
    lTester(lv::calcJointProbHist<1,true,false,false>(std::make_tuple(test1,test2)));
    lTester(lv::calcJointProbHist<1,true,true,false>(std::make_tuple(test1,test2)));
    lTester(lv::calcJointProbHist<1,true,true,true>(std::make_tuple(test1,test2)));
}

TEST(calcJointProbHist,regression_2d3vnz) {
    cv::Mat_<uchar> test1(2,2); test1 = 255; test1(0,0) = 1; test1(0,1) = 32;
    cv::Mat_<uchar> test2(2,2); test2 = 1; test2(0,0) = 255; test2(0,1) = 32;
    auto lTester = [&](auto oHistData, bool bSparse, bool bNoMinMax) {
        ASSERT_EQ(oHistData.dims(),size_t(2));
        for(size_t i=0; i<oHistData.dims(); ++i) {
            EXPECT_EQ(oHistData.aMinVals[i],bNoMinMax?0:1);
            EXPECT_EQ(oHistData.aMaxVals[i],255);
            ASSERT_EQ(oHistData.aStates[i],size_t(bSparse?3:bNoMinMax?256:255));
        }
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0](1-oHistData.aMinVals[0]),0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0](32-oHistData.aMinVals[0]),0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0](255-oHistData.aMinVals[0]),0.50f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1](1-oHistData.aMinVals[1]),0.50f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1](32-oHistData.aMinVals[1]),0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1](255-oHistData.aMinVals[1]),0.25f);
        ASSERT_EQ(oHistData.nJointStates,size_t(bSparse?3:bNoMinMax?256*256:255*255));
        EXPECT_FLOAT_EQ(oHistData.oJointHist(1-oHistData.aMinVals[0],255-oHistData.aMinVals[1]),0.25f);
        EXPECT_FLOAT_EQ(oHistData.oJointHist(32-oHistData.aMinVals[0],32-oHistData.aMinVals[1]),0.25f);
        EXPECT_FLOAT_EQ(oHistData.oJointHist(255-oHistData.aMinVals[0],1-oHistData.aMinVals[1]),0.50f);
    };
    lTester(lv::calcJointProbHist<1,true,false,false>(std::make_tuple(test1,test2)),true,false);
    lTester(lv::calcJointProbHist<1,true,true,false>(std::make_tuple(test1,test2)),true,false);
    lTester(lv::calcJointProbHist<1,true,true,true>(std::make_tuple(test1,test2)),true,true);
    lTester(lv::calcJointProbHist<1,false,false,false>(std::make_tuple(test1,test2)),false,false);
    lTester(lv::calcJointProbHist<1,false,true,false>(std::make_tuple(test1,test2)),false,false);
    lTester(lv::calcJointProbHist<1,false,true,true>(std::make_tuple(test1,test2)),false,true);
}

TEST(calcJointProbHist,regression_2d3vnz_quant) {
    const cv::Mat_<uchar> test1 = (cv::Mat_<uchar>(2,2) << uchar(1),uchar(32),uchar(33),uchar(255));
    const cv::Mat_<uchar> test2 = (cv::Mat_<uchar>(2,2) << uchar(255),uchar(32),uchar(33),uchar(1));
    auto lTester = [&](auto oHistData, bool bNoMinMax, bool bQuant64) {
        ASSERT_EQ(oHistData.dims(),size_t(2));
        for(size_t i=0; i<oHistData.dims(); ++i) {
            EXPECT_EQ(oHistData.aMinVals[i],bNoMinMax?0:1);
            EXPECT_EQ(oHistData.aMaxVals[i],255);
            ASSERT_EQ(oHistData.aStates[i],size_t(bQuant64?2:4));
        }
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0]((1-oHistData.aMinVals[0])/(bQuant64?64:1)),bQuant64?0.75f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0]((32-oHistData.aMinVals[0])/(bQuant64?64:1)),bQuant64?0.75f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0]((33-oHistData.aMinVals[0])/(bQuant64?64:1)),bQuant64?0.75f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[0]((255-oHistData.aMinVals[0])/(bQuant64?64:1)),0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1]((1-oHistData.aMinVals[1])/(bQuant64?64:1)),bQuant64?0.75f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1]((32-oHistData.aMinVals[1])/(bQuant64?64:1)),bQuant64?0.75f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1]((33-oHistData.aMinVals[1])/(bQuant64?64:1)),bQuant64?0.75f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.aMargHists[1]((255-oHistData.aMinVals[1])/(bQuant64?64:1)),0.25f);
        ASSERT_EQ(oHistData.nJointStates,size_t(bQuant64?3:4));
        EXPECT_FLOAT_EQ(oHistData.oJointHist((1-oHistData.aMinVals[0])/(bQuant64?64:1),(255-oHistData.aMinVals[1])/(bQuant64?64:1)),0.25f);
        EXPECT_FLOAT_EQ(oHistData.oJointHist((32-oHistData.aMinVals[0])/(bQuant64?64:1),(32-oHistData.aMinVals[1])/(bQuant64?64:1)),bQuant64?0.50f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.oJointHist((33-oHistData.aMinVals[0])/(bQuant64?64:1),(33-oHistData.aMinVals[1])/(bQuant64?64:1)),bQuant64?0.50f:0.25f);
        EXPECT_FLOAT_EQ(oHistData.oJointHist((255-oHistData.aMinVals[0])/(bQuant64?64:1),(1-oHistData.aMinVals[1])/(bQuant64?64:1)),0.25f);
    };
    lTester(lv::calcJointProbHist<1,true,false,false>(std::make_tuple(test1,test2)),false,false);
    lTester(lv::calcJointProbHist<1,true,true,false>(std::make_tuple(test1,test2)),false,false);
    lTester(lv::calcJointProbHist<1,true,true,true>(std::make_tuple(test1,test2)),true,false);
    lTester(lv::calcJointProbHist<64,true,false,false>(std::make_tuple(test1,test2)),false,true);
    lTester(lv::calcJointProbHist<64,true,true,false>(std::make_tuple(test1,test2)),false,true);
    lTester(lv::calcJointProbHist<64,true,true,true>(std::make_tuple(test1,test2)),true,true);
}