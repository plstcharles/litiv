
#include "litiv/utils/opencv.hpp"
#include "litiv/test.hpp"

#define TEST_MAT_TYPE_INFO(ch) \
    do { \
        volatile const int chv = ch; \
        const lv::MatType oType8U(CV_8UC(ch)); \
        ASSERT_TRUE(oType8U.channels()==ch);\
        ASSERT_TRUE(oType8U.depth()==CV_8U);\
        ASSERT_TRUE(oType8U.depthBytes()==size_t(1));\
        ASSERT_TRUE(oType8U.type()==CV_8UC(ch));\
        ASSERT_TRUE((oType8U.isTypeCompat<uint8_t>())); \
        ASSERT_FALSE((oType8U.isTypeCompat<int8_t>())); \
        if(chv>1) { \
            ASSERT_TRUE((oType8U.isTypeCompat<cv::Vec<uint8_t,ch>,true>())); \
            ASSERT_FALSE((oType8U.isTypeCompat<cv::Vec<int8_t,ch>,true>())); \
        } \
        ASSERT_TRUE((std::is_same<lv::MatCVType_<CV_8UC(ch)>::data_type,uint8_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatCVType_<CV_8UC(ch)>::elem_type,cv::Vec<uint8_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatCVType_<CV_8UC(ch)>::elem_type,uint8_t>::value));\
        ASSERT_TRUE(lv::MatCVType_<CV_8UC(ch)>::channels()==ch);\
        ASSERT_TRUE(lv::MatCVType_<CV_8UC(ch)>::depth()==CV_8U);\
        ASSERT_TRUE(lv::MatCVType_<CV_8UC(ch)>::depthBytes()==size_t(1));\
        ASSERT_TRUE(lv::MatCVType_<CV_8UC(ch)>::type()==CV_8UC(ch));\
        ASSERT_TRUE((std::is_same<lv::MatRawType_<uint8_t,ch>::data_type,uint8_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatRawType_<uint8_t,ch>::elem_type,cv::Vec<uint8_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatRawType_<uint8_t,ch>::elem_type,uint8_t>::value));\
        ASSERT_TRUE((lv::MatRawType_<uint8_t,ch>::channels()==ch));\
        ASSERT_TRUE((lv::MatRawType_<uint8_t,ch>::depth()==CV_8U));\
        ASSERT_TRUE((lv::MatRawType_<uint8_t,ch>::depthBytes()==size_t(1)));\
        ASSERT_TRUE((lv::MatRawType_<uint8_t,ch>::type()==CV_8UC(ch)));\
        const lv::MatType oType8S(CV_8SC(ch)); \
        ASSERT_TRUE(oType8S.channels()==ch);\
        ASSERT_TRUE(oType8S.depth()==CV_8S);\
        ASSERT_TRUE(oType8S.depthBytes()==size_t(1));\
        ASSERT_TRUE(oType8S.type()==CV_8SC(ch));\
        ASSERT_TRUE((oType8S.isTypeCompat<int8_t>())); \
        ASSERT_FALSE((oType8S.isTypeCompat<uint8_t>())); \
        if(chv>1) { \
            ASSERT_TRUE((oType8S.isTypeCompat<cv::Vec<int8_t,ch>,true>())); \
            ASSERT_FALSE((oType8S.isTypeCompat<cv::Vec<uint8_t,ch>,true>())); \
        } \
        ASSERT_TRUE((std::is_same<lv::MatCVType_<CV_8SC(ch)>::data_type,int8_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatCVType_<CV_8SC(ch)>::elem_type,cv::Vec<int8_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatCVType_<CV_8SC(ch)>::elem_type,int8_t>::value));\
        ASSERT_TRUE(lv::MatCVType_<CV_8SC(ch)>::channels()==ch);\
        ASSERT_TRUE(lv::MatCVType_<CV_8SC(ch)>::depth()==CV_8S);\
        ASSERT_TRUE(lv::MatCVType_<CV_8SC(ch)>::depthBytes()==size_t(1));\
        ASSERT_TRUE(lv::MatCVType_<CV_8SC(ch)>::type()==CV_8SC(ch));\
        ASSERT_TRUE((std::is_same<lv::MatRawType_<int8_t,ch>::data_type,int8_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatRawType_<int8_t,ch>::elem_type,cv::Vec<int8_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatRawType_<int8_t,ch>::elem_type,int8_t>::value));\
        ASSERT_TRUE((lv::MatRawType_<int8_t,ch>::channels()==ch));\
        ASSERT_TRUE((lv::MatRawType_<int8_t,ch>::depth()==CV_8S));\
        ASSERT_TRUE((lv::MatRawType_<int8_t,ch>::depthBytes()==size_t(1)));\
        ASSERT_TRUE((lv::MatRawType_<int8_t,ch>::type()==CV_8SC(ch)));\
        const lv::MatType oType16U(CV_16UC(ch)); \
        ASSERT_TRUE(oType16U.channels()==ch);\
        ASSERT_TRUE(oType16U.depth()==CV_16U);\
        ASSERT_TRUE(oType16U.depthBytes()==size_t(2));\
        ASSERT_TRUE(oType16U.type()==CV_16UC(ch));\
        ASSERT_TRUE((oType16U.isTypeCompat<uint16_t>())); \
        ASSERT_FALSE((oType16U.isTypeCompat<uint8_t>())); \
        if(chv>1) { \
            ASSERT_TRUE((oType16U.isTypeCompat<cv::Vec<uint16_t,ch>,true>())); \
            ASSERT_FALSE((oType16U.isTypeCompat<cv::Vec<uint8_t,ch>,true>())); \
        } \
        ASSERT_TRUE((std::is_same<lv::MatCVType_<CV_16UC(ch)>::data_type,uint16_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatCVType_<CV_16UC(ch)>::elem_type,cv::Vec<uint16_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatCVType_<CV_16UC(ch)>::elem_type,uint16_t>::value));\
        ASSERT_TRUE(lv::MatCVType_<CV_16UC(ch)>::channels()==ch);\
        ASSERT_TRUE(lv::MatCVType_<CV_16UC(ch)>::depth()==CV_16U);\
        ASSERT_TRUE(lv::MatCVType_<CV_16UC(ch)>::depthBytes()==size_t(2));\
        ASSERT_TRUE(lv::MatCVType_<CV_16UC(ch)>::type()==CV_16UC(ch));\
        ASSERT_TRUE((std::is_same<lv::MatRawType_<uint16_t,ch>::data_type,uint16_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatRawType_<uint16_t,ch>::elem_type,cv::Vec<uint16_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatRawType_<uint16_t,ch>::elem_type,uint16_t>::value));\
        ASSERT_TRUE((lv::MatRawType_<uint16_t,ch>::channels()==ch));\
        ASSERT_TRUE((lv::MatRawType_<uint16_t,ch>::depth()==CV_16U));\
        ASSERT_TRUE((lv::MatRawType_<uint16_t,ch>::depthBytes()==size_t(2)));\
        ASSERT_TRUE((lv::MatRawType_<uint16_t,ch>::type()==CV_16UC(ch)));\
        const lv::MatType oType16S(CV_16SC(ch)); \
        ASSERT_TRUE(oType16S.channels()==ch);\
        ASSERT_TRUE(oType16S.depth()==CV_16S);\
        ASSERT_TRUE(oType16S.depthBytes()==size_t(2));\
        ASSERT_TRUE(oType16S.type()==CV_16SC(ch));\
        ASSERT_TRUE((oType16S.isTypeCompat<int16_t>())); \
        ASSERT_FALSE((oType16S.isTypeCompat<uint8_t>())); \
        if(chv>1) { \
            ASSERT_TRUE((oType16S.isTypeCompat<cv::Vec<int16_t,ch>,true>())); \
            ASSERT_FALSE((oType16S.isTypeCompat<cv::Vec<uint8_t,ch>,true>())); \
        } \
        ASSERT_TRUE((std::is_same<lv::MatCVType_<CV_16SC(ch)>::data_type,int16_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatCVType_<CV_16SC(ch)>::elem_type,cv::Vec<int16_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatCVType_<CV_16SC(ch)>::elem_type,int16_t>::value));\
        ASSERT_TRUE(lv::MatCVType_<CV_16SC(ch)>::channels()==ch);\
        ASSERT_TRUE(lv::MatCVType_<CV_16SC(ch)>::depth()==CV_16S);\
        ASSERT_TRUE(lv::MatCVType_<CV_16SC(ch)>::depthBytes()==size_t(2));\
        ASSERT_TRUE(lv::MatCVType_<CV_16SC(ch)>::type()==CV_16SC(ch));\
        ASSERT_TRUE((std::is_same<lv::MatRawType_<int16_t,ch>::data_type,int16_t>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatRawType_<int16_t,ch>::elem_type,cv::Vec<int16_t,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatRawType_<int16_t,ch>::elem_type,int16_t>::value));\
        ASSERT_TRUE((lv::MatRawType_<int16_t,ch>::channels()==ch));\
        ASSERT_TRUE((lv::MatRawType_<int16_t,ch>::depth()==CV_16S));\
        ASSERT_TRUE((lv::MatRawType_<int16_t,ch>::depthBytes()==size_t(2)));\
        ASSERT_TRUE((lv::MatRawType_<int16_t,ch>::type()==CV_16SC(ch)));\
        const lv::MatType oType32S(CV_32SC(ch)); \
        ASSERT_TRUE(oType32S.channels()==ch);\
        ASSERT_TRUE(oType32S.depth()==CV_32S);\
        ASSERT_TRUE(oType32S.depthBytes()==size_t(4));\
        ASSERT_TRUE(oType32S.type()==CV_32SC(ch));\
        ASSERT_TRUE((oType32S.isTypeCompat<int32_t>())); \
        ASSERT_FALSE((oType32S.isTypeCompat<uint8_t>())); \
        if(chv>1) { \
            ASSERT_TRUE((oType32S.isTypeCompat<cv::Vec<int32_t,ch>,true>())); \
            ASSERT_FALSE((oType32S.isTypeCompat<cv::Vec<uint8_t,ch>,true>())); \
        } \
        ASSERT_TRUE((std::is_same<lv::MatCVType_<CV_32SC(ch)>::data_type,int>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatCVType_<CV_32SC(ch)>::elem_type,cv::Vec<int,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatCVType_<CV_32SC(ch)>::elem_type,int>::value));\
        ASSERT_TRUE(lv::MatCVType_<CV_32SC(ch)>::channels()==ch);\
        ASSERT_TRUE(lv::MatCVType_<CV_32SC(ch)>::depth()==CV_32S);\
        ASSERT_TRUE(lv::MatCVType_<CV_32SC(ch)>::depthBytes()==size_t(4));\
        ASSERT_TRUE(lv::MatCVType_<CV_32SC(ch)>::type()==CV_32SC(ch));\
        ASSERT_TRUE((std::is_same<lv::MatRawType_<int,ch>::data_type,int>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatRawType_<int,ch>::elem_type,cv::Vec<int,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatRawType_<int,ch>::elem_type,int>::value));\
        ASSERT_TRUE((lv::MatRawType_<int,ch>::channels()==ch));\
        ASSERT_TRUE((lv::MatRawType_<int,ch>::depth()==CV_32S));\
        ASSERT_TRUE((lv::MatRawType_<int,ch>::depthBytes()==size_t(4)));\
        ASSERT_TRUE((lv::MatRawType_<int,ch>::type()==CV_32SC(ch)));\
        const lv::MatType oType32F(CV_32FC(ch)); \
        ASSERT_TRUE(oType32F.channels()==ch);\
        ASSERT_TRUE(oType32F.depth()==CV_32F);\
        ASSERT_TRUE(oType32F.depthBytes()==size_t(4));\
        ASSERT_TRUE(oType32F.type()==CV_32FC(ch));\
        ASSERT_TRUE((oType32F.isTypeCompat<float>())); \
        ASSERT_FALSE((oType32F.isTypeCompat<uint8_t>())); \
        if(chv>1) { \
            ASSERT_TRUE((oType32F.isTypeCompat<cv::Vec<float,ch>,true>())); \
            ASSERT_FALSE((oType32F.isTypeCompat<cv::Vec<uint8_t,ch>,true>())); \
        } \
        ASSERT_TRUE((std::is_same<lv::MatCVType_<CV_32FC(ch)>::data_type,float>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatCVType_<CV_32FC(ch)>::elem_type,cv::Vec<float,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatCVType_<CV_32FC(ch)>::elem_type,float>::value));\
        ASSERT_TRUE(lv::MatCVType_<CV_32FC(ch)>::channels()==ch);\
        ASSERT_TRUE(lv::MatCVType_<CV_32FC(ch)>::depth()==CV_32F);\
        ASSERT_TRUE(lv::MatCVType_<CV_32FC(ch)>::depthBytes()==size_t(4));\
        ASSERT_TRUE(lv::MatCVType_<CV_32FC(ch)>::type()==CV_32FC(ch));\
        ASSERT_TRUE((std::is_same<lv::MatRawType_<float,ch>::data_type,float>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatRawType_<float,ch>::elem_type,cv::Vec<float,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatRawType_<float,ch>::elem_type,float>::value));\
        ASSERT_TRUE((lv::MatRawType_<float,ch>::channels()==ch));\
        ASSERT_TRUE((lv::MatRawType_<float,ch>::depth()==CV_32F));\
        ASSERT_TRUE((lv::MatRawType_<float,ch>::depthBytes()==size_t(4)));\
        ASSERT_TRUE((lv::MatRawType_<float,ch>::type()==CV_32FC(ch)));\
        const lv::MatType oType64F(CV_64FC(ch)); \
        ASSERT_TRUE(oType64F.channels()==ch);\
        ASSERT_TRUE(oType64F.depth()==CV_64F);\
        ASSERT_TRUE(oType64F.depthBytes()==size_t(8));\
        ASSERT_TRUE(oType64F.type()==CV_64FC(ch));\
        ASSERT_TRUE((oType64F.isTypeCompat<double>())); \
        ASSERT_FALSE((oType64F.isTypeCompat<uint8_t>())); \
        if(chv>1) { \
            ASSERT_TRUE((oType64F.isTypeCompat<cv::Vec<double,ch>,true>())); \
            ASSERT_FALSE((oType64F.isTypeCompat<cv::Vec<uint8_t,ch>,true>())); \
        } \
        ASSERT_TRUE((std::is_same<lv::MatCVType_<CV_64FC(ch)>::data_type,double>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatCVType_<CV_64FC(ch)>::elem_type,cv::Vec<double,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatCVType_<CV_64FC(ch)>::elem_type,double>::value));\
        ASSERT_TRUE(lv::MatCVType_<CV_64FC(ch)>::channels()==ch);\
        ASSERT_TRUE(lv::MatCVType_<CV_64FC(ch)>::depth()==CV_64F);\
        ASSERT_TRUE(lv::MatCVType_<CV_64FC(ch)>::depthBytes()==size_t(8));\
        ASSERT_TRUE(lv::MatCVType_<CV_64FC(ch)>::type()==CV_64FC(ch));\
        ASSERT_TRUE((std::is_same<lv::MatRawType_<double,ch>::data_type,double>::value));\
        ASSERT_TRUE((chv==1 || std::is_same<lv::MatRawType_<double,ch>::elem_type,cv::Vec<double,ch>>::value));\
        ASSERT_TRUE((chv!=1 || std::is_same<lv::MatRawType_<double,ch>::elem_type,double>::value));\
        ASSERT_TRUE((lv::MatRawType_<double,ch>::channels()==ch));\
        ASSERT_TRUE((lv::MatRawType_<double,ch>::depth()==CV_64F));\
        ASSERT_TRUE((lv::MatRawType_<double,ch>::depthBytes()==size_t(8)));\
        ASSERT_TRUE((lv::MatRawType_<double,ch>::type()==CV_64FC(ch)));\
    } while(false)

TEST(MatType,regression) {
#if defined(_MSC_VER)
#pragma warning(push,0)
#endif //defined(_MSC_VER)
    TEST_MAT_TYPE_INFO(1);
    TEST_MAT_TYPE_INFO(2);
    TEST_MAT_TYPE_INFO(3);
    TEST_MAT_TYPE_INFO(4);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif //defined(_MSC_VER)
}

namespace {
    template<typename T>
    struct MatSize_fixture : testing::Test {};
    typedef testing::Types<uchar,uint8_t,short,ushort,uint16_t,int,uint,uint32_t,size_t,uint64_t,int64_t> Size_types;
}

TYPED_TEST_CASE(MatSize_fixture,Size_types);
TYPED_TEST(MatSize_fixture,regression_2d) {
    lv::MatSize_<TypeParam> test;
    ASSERT_EQ(test.dims(),TypeParam(0));
#ifdef _DEBUG
    ASSERT_THROW_LV_QUIET(test.size(0));
    ASSERT_THROW_LV_QUIET(test.size(1));
    ASSERT_THROW_LV_QUIET(test.size(2));
#endif //def(_DEBUG)
    ASSERT_EQ(((cv::MatSize)test).p[-1],0);
    ASSERT_EQ(((cv::Size)test),cv::Size());
    std::array<int,1> arr = {0};
    ASSERT_TRUE(((cv::MatSize)test)==cv::MatSize(arr.data()+1));
    std::array<int,3> arr_alt = {2,34,12};
    ASSERT_TRUE(((cv::MatSize)test)!=cv::MatSize(arr_alt.data()+1));
    ASSERT_TRUE(((cv::Size)test)==cv::Size());
    ASSERT_TRUE(((cv::Size)test)!=cv::Size(12,34));
    lv::MatSize_<TypeParam> test_alt1;
    lv::MatSize_<TypeParam> test_alt2(cv::Size(12,34));
    ASSERT_TRUE(test==test_alt1);
    ASSERT_TRUE(test!=test_alt2);
    ASSERT_EQ(test.total(),size_t(0));
    ASSERT_TRUE(test.empty());
    std::stringstream sstr0;
    sstr0 << test;
    ASSERT_EQ(sstr0.str(),std::string("0-d:[]<empty>"));
    test = test_alt2;
    ASSERT_TRUE(test==test_alt2);
    ASSERT_TRUE(test!=test_alt1);
    ASSERT_EQ(test.dims(),TypeParam(2));
    ASSERT_EQ(test.size(0),TypeParam(34));
    ASSERT_EQ(test.size(1),TypeParam(12));
    ASSERT_NE(test,(const TypeParam*)nullptr);
    ASSERT_EQ(((cv::MatSize)test).p[-1],2);
    ASSERT_EQ(((cv::Size)test),cv::Size(12,34));
    ASSERT_TRUE(((cv::MatSize)test)!=cv::MatSize(arr.data()+1));
    ASSERT_TRUE(((cv::MatSize)test)==cv::MatSize(arr_alt.data()+1));
    ASSERT_TRUE(((cv::Size)test)!=cv::Size());
    ASSERT_TRUE(((cv::Size)test)==cv::Size(12,34));
    ASSERT_TRUE(((cv::Size)test.transpose())==cv::Size(34,12));
    ASSERT_EQ(test.total(),size_t(12*34));
    ASSERT_FALSE(test.empty());
    lv::MatSize test_alt3 = test;
    ASSERT_TRUE(test==test_alt3);
    lv::MatSize_<int> test_alt4(uint8_t(34),uint16_t(12));
    ASSERT_TRUE(test==test_alt4);
    lv::MatSize_<int> test_alt6{uint8_t(34),uint8_t(12)};
    ASSERT_TRUE(test==test_alt6);
    std::stringstream sstr1;
    sstr1 << test;
    ASSERT_EQ(sstr1.str(),std::string("2-d:[34,12]"));
    test[1] = 0;
    std::stringstream sstr2;
    sstr2 << test;
    ASSERT_EQ(sstr2.str(),std::string("2-d:[34,0]<empty>"));
    std::array<int,4> arr_nd = {3,4,5,6};
    lv::MatSize_<TypeParam> test_alt5(cv::MatSize(arr_nd.data()+1));
    ASSERT_THROW_LV_QUIET(((cv::Size)test_alt5).area());
    std::array<int,3> arr_neg1 = {-1,3,4};
    ASSERT_THROW_LV_QUIET(lv::MatSize_<TypeParam>(cv::MatSize(arr_neg1.data()+1)));
    std::array<int,3> arr_neg2 = {2,3,-4};
    ASSERT_THROW_LV_QUIET(lv::MatSize_<TypeParam>(cv::MatSize(arr_neg2.data()+1)));
}

TYPED_TEST(MatSize_fixture,regression_nd) {
    for(size_t i=0; i<100000; ++i) {
        const TypeParam nDims = TypeParam(rand()%6);
        std::vector<int> vnDimsPaddedInt(nDims+1);
        std::vector<int> vnDimsInt(nDims);
        std::vector<TypeParam> vnDimsPadded(nDims+1);
        std::vector<TypeParam> vnDims(nDims);
        vnDimsPaddedInt[0] = int(nDims);
        vnDimsPadded[0] = nDims;
        size_t nElems = 0;
        for(int n=0; n<int(nDims); ++n) {
            vnDims[n] = vnDimsPadded[n+1] = TypeParam(rand()%40);
            vnDimsInt[n] = vnDimsPaddedInt[n+1] = (int)vnDims[n];
            if(n==TypeParam(0))
                nElems = size_t(vnDims[n]);
            else
                nElems *= size_t(vnDims[n]);
        }
        lv::MatSize_<TypeParam> a(vnDims);
        lv::MatSize_<TypeParam> b(vnDimsInt);
        ASSERT_EQ(a,b);
        ASSERT_EQ(a.total(),nElems);
        ASSERT_EQ(b.total(),nElems);
        ASSERT_EQ(a.dims(),vnDimsPadded[0]);
        lv::MatSize_<TypeParam> c(cv::MatSize(vnDimsPaddedInt.data()+1));
        ASSERT_EQ(c,a);
        ASSERT_EQ(c.total(),nElems);
        ASSERT_EQ(c.total(),nElems);
        if(nDims>TypeParam(1) && b.total()>size_t(0)) {
            const size_t prev_tot_a = a.total();
            const TypeParam prev_sn_a = a[nDims-1];
            a[nDims-1]++;
            ASSERT_TRUE(a!=b);
            ASSERT_EQ(a.total(),((prev_tot_a/prev_sn_a)*a[nDims-1]));
            const size_t prev_tot_c = c.total();
            const TypeParam prev_sn_c = c[0];
            c[0]++;
            ASSERT_TRUE(c!=b);
            ASSERT_EQ(c.total(),((prev_tot_c/prev_sn_c)*c[0]));
        }
    }
}

TEST(squeeze,regression_trivial) {
    cv::Mat_<int> testmat(1,1);
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    cv::Mat resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(testmat),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
    testmat.create(2,1);
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(testmat),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
    testmat.create(1,231);
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(testmat),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
}

TEST(squeeze,regression_single) {
    cv::Mat_<int> testmat(3,std::array<int,3>{2,3,1}.data());
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    cv::Mat resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(std::array<int,2>{2,3},CV_32SC1),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
    testmat.create(3,std::array<int,3>{5,1,1}.data());
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    resmat = lv::squeeze(testmat);
    ASSERT_TRUE(lv::MatInfo(std::array<int,2>{5,1},CV_32SC1)==lv::MatInfo(resmat) || lv::MatInfo(std::array<int,2>{1,5},CV_32SC1)==lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
    testmat.create(5,std::array<int,5>{5,3,4,2,1}.data());
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(std::array<int,4>{5,3,4,2},CV_32SC1),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
    testmat.create(5,std::array<int,5>{2,5,1,2,4}.data());
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(std::array<int,4>{2,5,2,4},CV_32SC1),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
}

TEST(squeeze,regression_multi) {
    cv::Mat_<int> testmat(4,std::array<int,4>{2,1,3,1}.data());
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    cv::Mat resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(std::array<int,2>{2,3},CV_32SC1),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
    testmat.create(5,std::array<int,5>{5,1,4,2,1}.data());
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(std::array<int,3>{5,4,2},CV_32SC1),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
    testmat.create(7,std::array<int,7>{1,5,1,2,1,1,1}.data());
    lv::test::fillarray(testmat.begin(),testmat.end(),0,10000);
    resmat = lv::squeeze(testmat);
    ASSERT_EQ(lv::MatInfo(std::array<int,2>{5,2},CV_32SC1),lv::MatInfo(resmat));
    ASSERT_TRUE(std::equal(testmat.begin(),testmat.end(),resmat.begin<int>(),resmat.end<int>()));
}

TEST(clampImageCoords,regression) {
    const int nBS0 = 0;
    const int nBS5 = 5;
    const cv::Size oIMS(640,480);
    int nX=0,nY=0;
    lv::clampImageCoords(nX,nY,nBS0,oIMS);
    EXPECT_TRUE(nX==0 && nY==0);
    lv::clampImageCoords(nX,nY,nBS5,oIMS);
    EXPECT_TRUE(nX==5 && nY==5);
    nX=640,nY=480;
    lv::clampImageCoords(nX,nY,nBS0,oIMS);
    EXPECT_TRUE(nX==639 && nY==479);
    lv::clampImageCoords(nX,nY,nBS5,oIMS);
    EXPECT_TRUE(nX==634 && nY==474);
    nX=320,nY=240;
    lv::clampImageCoords(nX,nY,nBS0,oIMS);
    EXPECT_TRUE(nX==320 && nY==240);
    lv::clampImageCoords(nX,nY,nBS5,oIMS);
    EXPECT_TRUE(nX==320 && nY==240);
}

TEST(getRandSamplePosition,regression) {
    int nX,nY;
    const std::array<std::array<int,1>,1> anTestPattern1u = {{1}};
    lv::getRandSamplePosition<1,1>(anTestPattern1u,1,nX,nY,0,0,0,cv::Size(640,480));
    EXPECT_TRUE(nX==0 && nY==0);
    lv::getRandSamplePosition<1,1>(anTestPattern1u,1,nX,nY,320,240,0,cv::Size(640,480));
    EXPECT_TRUE(nX==320 && nY==240);
    lv::getRandSamplePosition<1,1>(anTestPattern1u,1,nX,nY,640,480,0,cv::Size(640,480));
    EXPECT_TRUE(nX==639 && nY==479);
    const std::array<std::array<int,3>,3> anTestPattern3i = {std::array<int,3>{0,0,100},std::array<int,3>{0,0,0},std::array<int,3>{0,0,0}};
    lv::getRandSamplePosition<3,3>(anTestPattern3i,5,nX,nY,0,0,0,cv::Size(640,480));
    EXPECT_TRUE(nX==1 && nY==0);
    lv::getRandSamplePosition<3,3>(anTestPattern3i,5,nX,nY,320,240,0,cv::Size(640,480));
    EXPECT_TRUE(nX==321 && nY==239);
    lv::getRandSamplePosition<3,3>(anTestPattern3i,5,nX,nY,640,480,0,cv::Size(640,480));
    EXPECT_TRUE(nX==639 && nY==479);
    const std::array<std::array<int,3>,3> anTestPattern3u = {std::array<int,3>{1,1,1},std::array<int,3>{1,1,1},std::array<int,3>{1,1,1}};
    for(size_t i=0; i<10000; ++i) {
        lv::getRandSamplePosition<3,3>(anTestPattern3u,9,nX,nY,320,240,0,cv::Size(640,480));
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
        lv::getRandNeighborPosition<8>(anTestPattern8,nX,nY,320,240,0,cv::Size(640,480));
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
    lv::validateKeyPoints(oROI,vKPs0);
    ASSERT_TRUE(vKPs0.empty());
    oROI.create(640,480,CV_8UC1);
    oROI = cv::Scalar_<uint8_t>(255);
    lv::validateKeyPoints(oROI,vKPs0);
    ASSERT_TRUE(vKPs0.empty());
    std::vector<cv::KeyPoint> vKPs1_orig = {cv::KeyPoint(cv::Point2f(10.0f,15.0f),-1.0f)}, vKPs1=vKPs1_orig;
    lv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs1_orig.begin(),lComp));
    oROI = cv::Scalar_<uint8_t>(0);
    oROI.at<uint8_t>(15,10) = 255;
    lv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs1_orig.begin(),lComp));
    vKPs1.push_back(cv::KeyPoint(cv::Point2f(25.0f,25.0f),-1.0f));
    lv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs1_orig.begin(),lComp));
    oROI = cv::Scalar_<uint8_t>(0);
    lv::validateKeyPoints(oROI,vKPs1);
    ASSERT_TRUE(std::equal(vKPs1.begin(),vKPs1.end(),vKPs0.begin(),lComp));
}

namespace {
    template<typename T>
    struct cvunique_fixture : testing::Test {};
    typedef testing::Types<int8_t, int, float> cvunique_types;
}
TYPED_TEST_CASE(cvunique_fixture,cvunique_types);
TYPED_TEST(cvunique_fixture,regression) {
    const cv::Mat_<TypeParam> oNoVal;
    EXPECT_EQ(lv::unique(oNoVal),std::vector<TypeParam>());
    const cv::Mat_<TypeParam> oSingleVal(1,1,TypeParam(55));
    EXPECT_EQ(lv::unique(oSingleVal),std::vector<TypeParam>{TypeParam(55)});
    const cv::Mat_<TypeParam> oSingleVals(32,24,TypeParam(55));
    EXPECT_EQ(lv::unique(oSingleVals),std::vector<TypeParam>{TypeParam(55)});
    const cv::Mat_<TypeParam> oVals = (cv::Mat_<TypeParam>(3,3) << TypeParam(1),TypeParam(7),TypeParam(1),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(8));
    EXPECT_EQ(lv::unique(oVals),(std::vector<TypeParam>{TypeParam(-100),TypeParam(-1),TypeParam(0),TypeParam(1),TypeParam(3),TypeParam(7),TypeParam(8)}));
    EXPECT_EQ(lv::unique(oVals),lv::unique((TypeParam*)oVals.datastart,(TypeParam*)oVals.dataend));
}

namespace {
    template<typename T>
    struct isEqual_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int,float> isEqual_types;
}

TYPED_TEST_CASE(isEqual_fixture,isEqual_types);
TYPED_TEST(isEqual_fixture,regression_mdims) {
    cv::Mat_<TypeParam> a,b;
    EXPECT_TRUE(lv::isEqual<TypeParam>(a,b));
    cv::RNG rng((unsigned int)time(NULL));
    for(size_t i=0; i<1000; ++i) {
        const int nDims = rand()%4+2;
        std::vector<int> vnDims(nDims);
        for(int n=0; n<nDims; ++n)
            vnDims[n] = rand()%10+1;
        a.create(nDims,vnDims.data());
        rng.fill(a,cv::RNG::UNIFORM,-200,200,true);
        b = a.clone();
        ASSERT_TRUE(lv::isEqual<TypeParam>(a,b));
        TypeParam& oVal = *(((TypeParam*)(b.data))+rand()%b.total());
        oVal += TypeParam(1);
        ASSERT_FALSE(lv::isEqual<TypeParam>(a,b));
    }
}

TEST(isEqual,regression_mchannels) {
    cv::Mat a,b;
    EXPECT_TRUE(lv::isEqual<uint8_t>(a,b));
    EXPECT_TRUE(lv::isEqual<float>(a,b));
    cv::RNG rng((unsigned int)time(NULL));
    for(size_t i=0; i<1000; ++i) {
        const int nDims = rand()%4+2;
        std::vector<int> vnDims(nDims);
        for(int n=0; n<nDims; ++n)
            vnDims[n] = rand()%10+1;
        a.create(nDims,vnDims.data(),CV_8UC3);
        rng.fill(a,cv::RNG::UNIFORM,-200,200,true);
        b = a.clone();
        ASSERT_TRUE(lv::isEqual<cv::Vec3b>(a,b));
        int8_t& oVal = *(((int8_t*)(b.data))+rand()%b.total());
        oVal += int8_t(1);
        ASSERT_FALSE(lv::isEqual<cv::Vec3b>(a,b));
    }
}

TEST(cvtcolor,hsl2bgr) {
    {
        const cv::Vec3b& vBGR1 = lv::getBGRFromHSL(0.0f,0.0f,0.0f);
        EXPECT_EQ(vBGR1[0],uint8_t(0));
        EXPECT_EQ(vBGR1[1],uint8_t(0));
        EXPECT_EQ(vBGR1[2],uint8_t(0));
        const cv::Vec3b& vBGR2 = lv::getBGRFromHSL(127.0f,0.0f,0.0f);
        EXPECT_EQ(vBGR1,vBGR2);
    }
    {
        const cv::Vec3b& vBGR1 = lv::getBGRFromHSL(0.0f,1.0f,1.0f);
        EXPECT_EQ(vBGR1[0],uint8_t(255));
        EXPECT_EQ(vBGR1[1],uint8_t(255));
        EXPECT_EQ(vBGR1[2],uint8_t(255));
        const cv::Vec3b& vBGR2 = lv::getBGRFromHSL(278.0f,1.0f,1.0f);
        EXPECT_EQ(vBGR1,vBGR2);
    }
    {
        const cv::Vec3b& vBGR = lv::getBGRFromHSL(0.0f,0.0f,0.5f);
        EXPECT_EQ(vBGR[0],uint8_t(128));
        EXPECT_EQ(vBGR[1],uint8_t(128));
        EXPECT_EQ(vBGR[2],uint8_t(128));
    }
    {
        const cv::Vec3b& vBGR = lv::getBGRFromHSL(0.0f,1.0f,0.5f);
        EXPECT_EQ(vBGR[0],uint8_t(0));
        EXPECT_EQ(vBGR[1],uint8_t(0));
        EXPECT_EQ(vBGR[2],uint8_t(255));
    }
    {
        const cv::Vec3b& vBGR = lv::getBGRFromHSL(120.0f,1.0f,0.5f);
        EXPECT_EQ(vBGR[0],uint8_t(0));
        EXPECT_EQ(vBGR[1],uint8_t(255));
        EXPECT_EQ(vBGR[2],uint8_t(0));
    }
    {
        const cv::Vec3b& vBGR = lv::getBGRFromHSL(240.0f,1.0f,0.5f);
        EXPECT_EQ(vBGR[0],uint8_t(255));
        EXPECT_EQ(vBGR[1],uint8_t(0));
        EXPECT_EQ(vBGR[2],uint8_t(0));
    }
}

TEST(cvtcolor,bgr2hsl) {
    {
        const cv::Vec3f& vHSL = lv::getHSLFromBGR(cv::Vec3b(0,0,0));
        EXPECT_FLOAT_EQ(vHSL[2],0.0f);
    }
    {
        const cv::Vec3f& vHSL = lv::getHSLFromBGR(cv::Vec3b(255,255,255));
        EXPECT_FLOAT_EQ(vHSL[2],1.0f);
    }
    {
        const cv::Vec3f& vHSL = lv::getHSLFromBGR(cv::Vec3b(128,128,128));
        EXPECT_FLOAT_EQ(vHSL[1],0.0f);
        EXPECT_NEAR(vHSL[2],0.5f,1.0f/255);
    }
    {
        const cv::Vec3f& vHSL = lv::getHSLFromBGR(cv::Vec3b(0,0,255));
        EXPECT_FLOAT_EQ(vHSL[0],0.0f);
        EXPECT_FLOAT_EQ(vHSL[1],1.0f);
        EXPECT_FLOAT_EQ(vHSL[2],0.5f);
    }
    {
        const cv::Vec3f& vHSL = lv::getHSLFromBGR(cv::Vec3b(0,255,0));
        EXPECT_FLOAT_EQ(vHSL[0],120.0f);
        EXPECT_FLOAT_EQ(vHSL[1],1.0f);
        EXPECT_FLOAT_EQ(vHSL[2],0.5f);
    }
    {
        const cv::Vec3f& vHSL = lv::getHSLFromBGR(cv::Vec3b(255,0,0));
        EXPECT_FLOAT_EQ(vHSL[0],240.0f);
        EXPECT_FLOAT_EQ(vHSL[1],1.0f);
        EXPECT_FLOAT_EQ(vHSL[2],0.5f);
    }
}

TEST(cvtcolor,bgr2hsl2bgr) {
    srand((uint)time((time_t*)nullptr));
    for(size_t i=0; i<10000; ++i) {
        const cv::Vec3b vBGR = cv::Vec3b(uint8_t(rand()%256),uint8_t(rand()%256),uint8_t(rand()%256));
        ASSERT_EQ(vBGR,lv::getBGRFromHSL(lv::getHSLFromBGR(vBGR))) << " @ " << i << "/10000";
    }
}

TEST(cvtcolor,bgr2packedycbcr_gray) {
    const cv::Mat test(20,20,CV_8UC3,cv::Scalar_<uint8_t>::all(128));
    cv::Mat_<uint16_t> oOutput;
    lv::cvtBGRToPackedYCbCr(test,oOutput);
    ASSERT_EQ(cv::countNonZero(oOutput==uint16_t(34944)),test.size().area());
    cv::Mat_<cv::Vec3b> rebuilt;
    lv::cvtPackedYCbCrToBGR(oOutput,rebuilt);
    ASSERT_TRUE(lv::isEqual<cv::Vec3b>(test,rebuilt));
}

TEST(cvtcolor,bgr2packedycbcr_red) {
    cv::Mat test(20,20,CV_8UC3,cv::Scalar_<uint8_t>(0,0,128));
    test.at<cv::Vec3b>(2,5) = cv::Vec3b(160,160,160);
    cv::Mat_<uint16_t> oOutput;
    lv::cvtBGRToPackedYCbCr(test,oOutput);
    {
        cv::Mat_<cv::Vec3b> rebuilt;
        lv::cvtPackedYCbCrToBGR(oOutput,rebuilt);
        std::vector<cv::Mat> vrebuilt;
        cv::split(rebuilt,vrebuilt);
        ASSERT_EQ(vrebuilt.size(),size_t(3));
        ASSERT_EQ(cv::countNonZero(vrebuilt[0]==uint16_t(0)),test.size().area()-1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[1]==uint16_t(3)),test.size().area()-1); // noise in green channel due to quantification
        ASSERT_EQ(cv::countNonZero(vrebuilt[2]==uint16_t(128)),test.size().area()-1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[0]==uint16_t(160)),1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[1]==uint16_t(160)),1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[2]==uint16_t(160)),1);
    }
    {
        std::unique_ptr<uint16_t[]> pData(new uint16_t[20*20+7]);
        cv::Mat_<uint16_t> oOutput_unaligned(oOutput.rows,oOutput.cols,pData.get()+7);
        oOutput.copyTo(oOutput_unaligned);
        cv::Mat_<cv::Vec3b> rebuilt;
        lv::cvtPackedYCbCrToBGR(oOutput_unaligned,rebuilt);
        std::vector<cv::Mat> vrebuilt;
        cv::split(rebuilt,vrebuilt);
        ASSERT_EQ(vrebuilt.size(),size_t(3));
        ASSERT_EQ(cv::countNonZero(vrebuilt[0]==uint16_t(0)),test.size().area()-1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[1]==uint16_t(3)),test.size().area()-1); // noise in green channel due to quantification
        ASSERT_EQ(cv::countNonZero(vrebuilt[2]==uint16_t(128)),test.size().area()-1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[0]==uint16_t(160)),1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[1]==uint16_t(160)),1);
        ASSERT_EQ(cv::countNonZero(vrebuilt[2]==uint16_t(160)),1);
    }

}

namespace {
    template<typename T>
    struct readwrite_fixture : testing::Test {};
    typedef testing::Types<int8_t, int16_t, int, uint8_t, uint16_t, float, double> readwrite_types;
}
TYPED_TEST_CASE(readwrite_fixture,readwrite_types);
TYPED_TEST(readwrite_fixture,regression) {
    cv::RNG rng((unsigned int)time(NULL));
    const std::string sArchivePath = TEST_OUTPUT_DATA_ROOT "/test_readwrite.mat";
    for(size_t i=0; i<100; ++i) {
        cv::Mat_<TypeParam> oMat(rng.uniform(100,200),rng.uniform(100,200));
        rng.fill(oMat,cv::RNG::UNIFORM,-200,200,true);
        lv::write(sArchivePath,oMat,lv::MatArchive_BINARY);
        const cv::Mat oNewMat = lv::read(sArchivePath,lv::MatArchive_BINARY);
        ASSERT_EQ(cv::countNonZero(oNewMat!=oNewMat),0);
    }
    const std::string sYMLPath = TEST_OUTPUT_DATA_ROOT "/test_readwrite.yml";
    for(size_t i=0; i<100; ++i) {
        cv::Mat_<TypeParam> oMat(rng.uniform(10,20),rng.uniform(10,20));
        rng.fill(oMat,cv::RNG::UNIFORM,-200,200,true);
        lv::write(sYMLPath,oMat,lv::MatArchive_FILESTORAGE);
        const cv::Mat oNewMat = lv::read(sYMLPath,lv::MatArchive_FILESTORAGE);
        ASSERT_EQ(cv::countNonZero(oNewMat!=oNewMat),0);
    }
    const std::string sTextFilePath = TEST_OUTPUT_DATA_ROOT "/test_readwrite.txt";
    for(size_t i=0; i<100; ++i) {
        cv::Mat_<TypeParam> oMat(rng.uniform(10,20),rng.uniform(10,20));
        rng.fill(oMat,cv::RNG::UNIFORM,-200,200,true);
        lv::write(sTextFilePath,oMat,lv::MatArchive_PLAINTEXT);
        const cv::Mat oNewMat = lv::read(sTextFilePath,lv::MatArchive_PLAINTEXT);
        ASSERT_EQ(cv::countNonZero(oNewMat!=oNewMat),0);
    }
}

TEST(pack_unpack,regression) {
    srand((uint)time(nullptr));
    cv::RNG rng((unsigned int)time(NULL));
    for(size_t i=0; i<10000; ++i) {
        const size_t nMats = (size_t(rand()%10));
        std::vector<cv::Mat> vMats(nMats);
        size_t nTotalPacketSize = 0;
        for(size_t nMatIdx=0; nMatIdx<nMats; ++nMatIdx) {
            cv::Mat& oCurrMat = vMats[nMatIdx];
            if(rand()%10) {
                const int nDims = rand()%4+1;
                std::vector<int> vSizes((size_t)nDims);
                std::generate(vSizes.begin(),vSizes.end(),[](){return rand()%20+1;});
                oCurrMat.create(nDims,vSizes.data(),CV_MAKETYPE(rand()%7,rand()%4+1));
                rng.fill(oCurrMat,cv::RNG::UNIFORM,-200,200,true);
                nTotalPacketSize += oCurrMat.total()*oCurrMat.elemSize();
            }
        }
        std::vector<lv::MatInfo> vPackInfo;
        const cv::Mat oPacket = lv::packData(vMats,&vPackInfo);
        ASSERT_TRUE(oPacket.empty() || oPacket.isContinuous());
        ASSERT_EQ(vPackInfo.size(),vMats.size());
        ASSERT_EQ(oPacket.total()*oPacket.elemSize(),nTotalPacketSize);
        for(size_t nMatIdx=0; nMatIdx<nMats; ++nMatIdx) {
            ASSERT_TRUE(vPackInfo[nMatIdx].size==vMats[nMatIdx].size);
            ASSERT_TRUE(vPackInfo[nMatIdx].type()==vMats[nMatIdx].type());
        }
        const std::vector<cv::Mat> vNewMats = lv::unpackData(oPacket,vPackInfo);
        ASSERT_EQ(vPackInfo.size(),vNewMats.size());
        for(size_t nMatIdx=0; nMatIdx<nMats; ++nMatIdx)
            ASSERT_TRUE(lv::isEqual<uint8_t>(vMats[nMatIdx],vNewMats[nMatIdx]));
    }
}

TEST(shift,regression_intconstborder) {
    const cv::Mat oInput = (cv::Mat_<int>(4,5) << 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19);
    {
        cv::Mat oTestOut; lv::shift(oInput,oTestOut,cv::Point2f(1.0f,0.0f),cv::BORDER_CONSTANT,cv::Scalar(-1));
        const cv::Mat oTestGT = (cv::Mat_<int>(4,5) << -1,0,1,2,3,-1,5,6,7,8,-1,10,11,12,13,-1,15,16,17,18);
        ASSERT_TRUE(lv::isEqual<int>(oTestOut,oTestGT));
    }
    {
        cv::Mat oTestOut; lv::shift(oInput,oTestOut,cv::Point2f(0.0f,2.0f),cv::BORDER_CONSTANT,cv::Scalar(-1));
        const cv::Mat oTestGT = (cv::Mat_<int>(4,5) << -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9);
        ASSERT_TRUE(lv::isEqual<int>(oTestOut,oTestGT));
    }
    {
        cv::Mat oTestOut; lv::shift(oInput,oTestOut,cv::Point2f(1.0f,1.0f),cv::BORDER_CONSTANT,cv::Scalar(-1));
        const cv::Mat oTestGT = (cv::Mat_<int>(4,5) << -1,-1,-1,-1,-1,-1,0,1,2,3,-1,5,6,7,8,-1,10,11,12,13);
        ASSERT_TRUE(lv::isEqual<int>(oTestOut,oTestGT));
    }
    {
        cv::Mat oTestOut; lv::shift(oInput,oTestOut,cv::Point2f(-2.0f,-2.0f),cv::BORDER_CONSTANT,cv::Scalar(-1));
        const cv::Mat oTestGT = (cv::Mat_<int>(4,5) << 12,13,14,-1,-1,17,18,19,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
        ASSERT_TRUE(lv::isEqual<int>(oTestOut,oTestGT));
    }
}

namespace {
    template<typename T>
    struct AlignedMatAllocator_fixture : testing::Test {};
    typedef testing::Types<int8_t, int16_t, int, uint8_t, uint16_t, float, double> AlignedMatAllocator_types;
}
TYPED_TEST_CASE(AlignedMatAllocator_fixture,AlignedMatAllocator_types);
TYPED_TEST(AlignedMatAllocator_fixture,regression) {
    lv::AlignedMatAllocator<16,false> alloc16a;
    cv::Mat_<TypeParam> oTest;
    oTest.allocator = &alloc16a;
    oTest.create(45,23);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%16)==size_t(0));
    const int aDims1[4] = {4,5,6,7};
    oTest.create(4,aDims1);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%16)==size_t(0));
    lv::AlignedMatAllocator<32,false> alloc32a;
    oTest = cv::Mat_<TypeParam>();
    oTest.allocator = &alloc32a;
    oTest.create(43,75);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%32)==0);
    oTest = cv::Mat_<TypeParam>();
}

TEST(AlignedMatAllocatorPremade,regression) {
    cv::Mat_<float> oTest;
    cv::MatAllocator* pAlloc16a = lv::getMatAllocator16a();
    ASSERT_TRUE(pAlloc16a!=nullptr);
    oTest.allocator = pAlloc16a;
    oTest.create(12,13);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%16)==size_t(0));
    cv::MatAllocator* pAlloc32a = lv::getMatAllocator32a();
    ASSERT_TRUE(pAlloc32a!=nullptr);
    oTest = cv::Mat_<float>();
    oTest.allocator = pAlloc32a;
    oTest.create(12,13);
    ASSERT_TRUE(((uintptr_t)oTest.datastart%32)==size_t(0));
}

#if USE_OPENCV_x264_TEST

TEST(ffmpeg_compat,read_x264) {
    const std::string sFileLocation = SAMPLES_DATA_ROOT "/tractor.mp4";
    cv::VideoCapture oCap(sFileLocation);
    ASSERT_TRUE(lv::checkIfExists(sFileLocation));
    ASSERT_TRUE(oCap.isOpened()) << " --- make sure your OpenCV installation can read x264 videos!";
    cv::Mat oFrame;
    oCap >> oFrame;
    ASSERT_TRUE(!oFrame.empty() && oFrame.type()==CV_8UC3 && oFrame.size()==cv::Size(1920,1080));
    oCap >> oFrame;
    ASSERT_TRUE(!oFrame.empty() && oFrame.type()==CV_8UC3 && oFrame.size()==cv::Size(1920,1080));
}

#endif //USE_OPENCV_x264_TEST