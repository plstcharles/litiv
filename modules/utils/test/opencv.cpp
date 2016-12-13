
#include "litiv/utils/opencv.hpp"
#include "common.hpp"

TEST(clampImageCoords,regression) {
    const int nBS0 = 0;
    const int nBS5 = 5;
    const cv::Size oIMS(640,480);
    int nX=0,nY=0;
    cv::clampImageCoords(nX,nY,nBS0,oIMS);
    EXPECT_TRUE(nX==0 && nY==0);
    cv::clampImageCoords(nX,nY,nBS5,oIMS);
    EXPECT_TRUE(nX==5 && nY==5);
    nX=640,nY=480;
    cv::clampImageCoords(nX,nY,nBS0,oIMS);
    EXPECT_TRUE(nX==639 && nY==479);
    cv::clampImageCoords(nX,nY,nBS5,oIMS);
    EXPECT_TRUE(nX==634 && nY==474);
    nX=320,nY=240;
    cv::clampImageCoords(nX,nY,nBS0,oIMS);
    EXPECT_TRUE(nX==320 && nY==240);
    cv::clampImageCoords(nX,nY,nBS5,oIMS);
    EXPECT_TRUE(nX==320 && nY==240);
}

TEST(getRandSamplePosition,regression) {
    int nX,nY;
    const std::array<std::array<int,1>,1> anTestPattern1u = {{1}};
    cv::getRandSamplePosition(anTestPattern1u,1,nX,nY,0,0,0,cv::Size(640,480));
    EXPECT_TRUE(nX==0 && nY==0);
    cv::getRandSamplePosition(anTestPattern1u,1,nX,nY,320,240,0,cv::Size(640,480));
    EXPECT_TRUE(nX==320 && nY==240);
    cv::getRandSamplePosition(anTestPattern1u,1,nX,nY,640,480,0,cv::Size(640,480));
    EXPECT_TRUE(nX==639 && nY==479);
    const std::array<std::array<int,3>,3> anTestPattern3i = {std::array<int,3>{0,0,100},std::array<int,3>{0,0,0},std::array<int,3>{0,0,0}};
    cv::getRandSamplePosition(anTestPattern3i,5,nX,nY,0,0,0,cv::Size(640,480));
    EXPECT_TRUE(nX==1 && nY==0);
    cv::getRandSamplePosition(anTestPattern3i,5,nX,nY,320,240,0,cv::Size(640,480));
    EXPECT_TRUE(nX==321 && nY==239);
    cv::getRandSamplePosition(anTestPattern3i,5,nX,nY,640,480,0,cv::Size(640,480));
    EXPECT_TRUE(nX==639 && nY==479);
    const std::array<std::array<int,3>,3> anTestPattern3u = {std::array<int,3>{1,1,1},std::array<int,3>{1,1,1},std::array<int,3>{1,1,1}};
    for(size_t i=0; i<10000; ++i) {
        cv::getRandSamplePosition(anTestPattern3u,9,nX,nY,320,240,0,cv::Size(640,480));
        ASSERT_TRUE(nX>=319 && nX<=321);
        ASSERT_TRUE(nY>=239 && nY<=241);
    }
}

TEST(getRandNeighborPosition,regression) {
    typedef std::array<int,2> Nb;
    const std::array<std::array<int,2>,8> anTestPattern8 ={
            Nb{-1, 1},Nb{0, 1},Nb{1, 1},
            Nb{-1, 0},         Nb{1, 0},
            Nb{-1,-1},Nb{0,-1},Nb{1,-1},
    };
    for(size_t i=0; i<10000; ++i) {
        int nX, nY;
        cv::getRandNeighborPosition(anTestPattern8,nX,nY,320,240,0,cv::Size(640,480));
        ASSERT_TRUE(nX>=319 && nX<=321);
        ASSERT_TRUE(nY>=239 && nY<=241);
        ASSERT_FALSE(nX==320 && nY==240);
    }
}

TEST(validateKeyPoints,regression) {
    auto lComp = [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.pt==b.pt;
    };
    std::vector<cv::KeyPoint> vKPs0;
    cv::Mat oROI;
    cv::validateKeyPoints(oROI,vKPs0);
    ASSERT_TRUE(vKPs0.empty());
    oROI.create(640,480,CV_8UC1);
    oROI = cv::Scalar_<uchar>(255);
    cv::validateKeyPoints(oROI,vKPs0);
    ASSERT_TRUE(vKPs0.empty());
    std::vector<cv::KeyPoint> vKPs1_orig = {cv::KeyPoint(cv::Point2f(10.0f,15.0f),-1.0f)}, vKPs1=vKPs1_orig;
    cv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs1_orig.begin(),lComp));
    oROI = cv::Scalar_<uchar>(0);
    oROI.at<uchar>(15,10) = 255;
    cv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs1_orig.begin(),lComp));
    vKPs1.push_back(cv::KeyPoint(cv::Point2f(25.0f,25.0f),-1.0f));
    cv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs1_orig.begin(),lComp));
    oROI = cv::Scalar_<uchar>(0);
    cv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs0.begin(),lComp));
}

namespace {
    template<typename T>
    struct cvunique_fixture : testing::Test {};
    typedef testing::Types<char, int, float> cvunique_types;
}
TYPED_TEST_CASE(cvunique_fixture,cvunique_types);
TYPED_TEST(cvunique_fixture,regression) {
    const cv::Mat_<TypeParam> oNoVal;
    EXPECT_EQ(cv::unique(oNoVal),std::vector<TypeParam>());
    const cv::Mat_<TypeParam> oSingleVal(1,1,TypeParam(55));
    EXPECT_EQ(cv::unique(oSingleVal),std::vector<TypeParam>{TypeParam(55)});
    const cv::Mat_<TypeParam> oSingleVals(32,24,TypeParam(55));
    EXPECT_EQ(cv::unique(oSingleVals),std::vector<TypeParam>{TypeParam(55)});
    const cv::Mat_<TypeParam> oVals = (cv::Mat_<TypeParam>(3,3) << TypeParam(1),TypeParam(7),TypeParam(1),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(8));
    EXPECT_EQ(cv::unique(oVals),(std::vector<TypeParam>{TypeParam(-100),TypeParam(-1),TypeParam(0),TypeParam(1),TypeParam(3),TypeParam(7),TypeParam(8)}));
    EXPECT_EQ(cv::unique(oVals),lv::unique((TypeParam*)oVals.datastart,(TypeParam*)oVals.dataend));
}

namespace {
    template<typename T>
    struct readwrite_fixture : testing::Test {};
    typedef testing::Types<char, short, int, uint8_t, uint16_t, float, double> readwrite_types;
}
TYPED_TEST_CASE(readwrite_fixture,readwrite_types);
TYPED_TEST(readwrite_fixture,regression) {
    cv::RNG rng((unsigned int)time(NULL));
    const std::string sArchivePath = TEST_DATA_ROOT "/test_readwrite.mat";
    for(size_t i=0; i<100; ++i) {
        cv::Mat_<TypeParam> oMat(rng.uniform(100,200),rng.uniform(100,200));
        rng.fill(oMat,cv::RNG::UNIFORM,-200,200,true);
        cv::write(sArchivePath,oMat,cv::MatArchive_BINARY);
        const cv::Mat oNewMat = cv::read(sArchivePath,cv::MatArchive_BINARY);
        ASSERT_EQ(cv::countNonZero(oNewMat!=oNewMat),0);
    }
    const std::string sYMLPath = TEST_DATA_ROOT "/test_readwrite.yml";
    for(size_t i=0; i<100; ++i) {
        cv::Mat_<TypeParam> oMat(rng.uniform(10,20),rng.uniform(10,20));
        rng.fill(oMat,cv::RNG::UNIFORM,-200,200,true);
        cv::write(sYMLPath,oMat,cv::MatArchive_FILESTORAGE);
        const cv::Mat oNewMat = cv::read(sYMLPath,cv::MatArchive_FILESTORAGE);
        ASSERT_EQ(cv::countNonZero(oNewMat!=oNewMat),0);
    }
    const std::string sTextFilePath = TEST_DATA_ROOT "/test_readwrite.txt";
    for(size_t i=0; i<100; ++i) {
        cv::Mat_<TypeParam> oMat(rng.uniform(10,20),rng.uniform(10,20));
        rng.fill(oMat,cv::RNG::UNIFORM,-200,200,true);
        cv::write(sTextFilePath,oMat,cv::MatArchive_PLAINTEXT);
        const cv::Mat oNewMat = cv::read(sTextFilePath,cv::MatArchive_PLAINTEXT);
        ASSERT_EQ(cv::countNonZero(oNewMat!=oNewMat),0);
    }
}

TEST(empty_stuff,regression) {
    ASSERT_TRUE(cv::emptyMat().empty());
    ASSERT_TRUE(cv::emptySize().area()==0);
    ASSERT_TRUE(cv::emptyMatArray().empty());
    ASSERT_TRUE(cv::emptySizeArray().empty());
}

namespace {
    template<typename T>
    struct AlignedMatAllocator_fixture : testing::Test {};
    typedef testing::Types<char, short, int, uint8_t, uint16_t, float, double> AlignedMatAllocator_types;
}
TYPED_TEST_CASE(AlignedMatAllocator_fixture,AlignedMatAllocator_types);
TYPED_TEST(AlignedMatAllocator_fixture,regression) {
    cv::AlignedMatAllocator<16,false> alloc16a;
    cv::Mat_<TypeParam> oTest;
    oTest.allocator = &alloc16a;
    oTest.create(45,23);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%16)==0);
    const int aDims1[4] = {4,5,6,7};
    oTest.create(4,aDims1);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%16)==0);
    cv::AlignedMatAllocator<32,false> alloc32a;
    oTest = cv::Mat_<TypeParam>();
    oTest.allocator = &alloc32a;
    oTest.create(43,75);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%32)==0);
    oTest = cv::Mat_<TypeParam>();
}

TEST(AlignedMatAllocatorPremade,regression) {
    cv::Mat_<float> oTest;
    cv::MatAllocator* pAlloc16a = cv::getMatAllocator16a();
    ASSERT_TRUE(pAlloc16a!=nullptr);
    oTest.allocator = pAlloc16a;
    oTest.create(12,13);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%16)==0);
    cv::MatAllocator* pAlloc32a = cv::getMatAllocator32a();
    ASSERT_TRUE(pAlloc32a!=nullptr);
    oTest = cv::Mat_<float>();
    oTest.allocator = pAlloc32a;
    oTest.create(12,13);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%32)==0);
}