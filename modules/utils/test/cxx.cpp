
#include "litiv/utils/cxx.hpp"
#include "litiv/test.hpp"

TEST(putf,regression) {
    EXPECT_EQ(std::string(),lv::putf(""));
    EXPECT_EQ(std::string("test"),lv::putf("test"));
    EXPECT_EQ(std::string("test 0.50 test"),lv::putf("test %.2f %s",0.5f,"test"));
}

TEST(unroll,regression) {
    constexpr size_t nIter = 10;
    size_t nIdx_ext = 0;
    lv::unroll<nIter>([&](size_t nIdx){
        EXPECT_EQ(nIdx,nIdx_ext++);
    });
    EXPECT_EQ(nIdx_ext,nIter);
}

TEST(compare_lowercase,regression) {
    EXPECT_EQ(lv::compare_lowercase("",""),std::string()<std::string());
    EXPECT_EQ(lv::compare_lowercase("abc","ABC"),std::string("abc")<std::string("abc"));
    const std::string sTestStr1("TESTstr1##$./*1STR"),sTestStr2("teststr1##$./*1str");
    EXPECT_EQ(lv::compare_lowercase(sTestStr1,sTestStr2),sTestStr2<sTestStr2);
    const std::string sCompStr("tasdgTESTstr1##$./*1STR");
    EXPECT_EQ(lv::compare_lowercase(sTestStr1,sCompStr),sTestStr2<sCompStr);
}

TEST(digit_count,regression) {
    EXPECT_EQ(lv::digit_count(0),1);
    EXPECT_EQ(lv::digit_count(size_t(0)),1);
    EXPECT_EQ(lv::digit_count(0.5f),1);
    EXPECT_EQ(lv::digit_count(1),1);
    EXPECT_EQ(lv::digit_count(size_t(1)),1);
    EXPECT_EQ(lv::digit_count(5),1);
    EXPECT_EQ(lv::digit_count(10),2);
    EXPECT_EQ(lv::digit_count(size_t(10)),2);
    EXPECT_EQ(lv::digit_count(555),3);
    EXPECT_EQ(lv::digit_count(-1.5),2);
    EXPECT_EQ(lv::digit_count(-1),2);
    EXPECT_EQ(lv::digit_count(-10),3);
    EXPECT_EQ(lv::digit_count(-1000),5);
    EXPECT_EQ(lv::digit_count(-1000),5);
    EXPECT_EQ(lv::digit_count(std::numeric_limits<float>::quiet_NaN()),3);
}

TEST(string_contains_token,regression) {
    const std::string sStrEmpty;
    const std::vector<std::string> vsNoTokens = {};
    EXPECT_FALSE(lv::string_contains_token(sStrEmpty,vsNoTokens));
    const std::string sStr("this is art");
    const std::vector<std::string> vsTokens = {"siht","##","<>/","art","a"};
    EXPECT_FALSE(lv::string_contains_token(sStrEmpty,vsTokens));
    EXPECT_TRUE(lv::string_contains_token(sStr,vsTokens));
    EXPECT_FALSE(lv::string_contains_token(sStr,std::vector<std::string>(vsTokens.begin(),vsTokens.begin()+3)));
    EXPECT_TRUE(lv::string_contains_token(sStr,std::vector<std::string>(vsTokens.begin(),vsTokens.begin()+4)));
}

TEST(clampString,regression) {
    EXPECT_EQ(lv::clampString("",0),std::string());
    EXPECT_EQ(lv::clampString("test",3),std::string("tes"));
    EXPECT_EQ(lv::clampString("test",5),std::string(" test"));
    EXPECT_EQ(lv::clampString("test",6,'x'),std::string("xxtest"));
}

namespace {
    template<typename T>
    struct concat_fixture : testing::Test {};
    typedef testing::Types<char, int, ushort, size_t, float> concat_types;
}
TYPED_TEST_CASE(concat_fixture,concat_types);
TYPED_TEST(concat_fixture,regression) {
    EXPECT_EQ((lv::concat<TypeParam>(std::vector<TypeParam>(),std::vector<TypeParam>())),(std::vector<TypeParam>{}));
    EXPECT_EQ((lv::concat<TypeParam>(std::vector<TypeParam>{TypeParam(1)},std::vector<TypeParam>{})),(std::vector<TypeParam>{TypeParam(1)}));
    EXPECT_EQ((lv::concat<TypeParam>(std::vector<TypeParam>{},std::vector<TypeParam>{TypeParam(1)})),(std::vector<TypeParam>{TypeParam(1)}));
    EXPECT_EQ((lv::concat<TypeParam>(std::vector<TypeParam>{TypeParam(1)},std::vector<TypeParam>{TypeParam(2)})),(std::vector<TypeParam>{TypeParam(1),TypeParam(2)}));
    EXPECT_EQ((lv::concat<TypeParam>(std::vector<TypeParam>{TypeParam(1),TypeParam(3)},std::vector<TypeParam>{TypeParam(2)})),(std::vector<TypeParam>{TypeParam(1),TypeParam(3),TypeParam(2)}));
}

namespace {
    template<typename T>
    struct cvtarrayvec_fixture : testing::Test {};
    typedef testing::Types<char, int, ushort, size_t, float> cvtarrayvec_types;
}
TYPED_TEST_CASE(cvtarrayvec_fixture,cvtarrayvec_types);
TYPED_TEST(cvtarrayvec_fixture,regression) {
    {
        auto a = std::array<TypeParam,0>{};
        auto v = lv::convertArrayToVector(a);
        EXPECT_EQ(a.size(),v.size());
        EXPECT_TRUE(std::equal(a.begin(),a.end(),v.begin()));
        auto a2 = lv::convertVectorToArray<0>(v);
        EXPECT_EQ(a.size(),a2.size());
        EXPECT_TRUE(std::equal(a.begin(),a.end(),a2.begin()));
    }{
        auto a = std::array<TypeParam,1>{TypeParam(2)};
        auto v = lv::convertArrayToVector(a);
        EXPECT_EQ(a.size(),v.size());
        EXPECT_TRUE(std::equal(a.begin(),a.end(),v.begin()));
        auto a2 = lv::convertVectorToArray<1>(v);
        EXPECT_EQ(a.size(),a2.size());
        EXPECT_TRUE(std::equal(a.begin(),a.end(),a2.begin()));
    }{
        auto a = std::array<TypeParam,4>{TypeParam(0),TypeParam(2),TypeParam(1),TypeParam(5)};
        auto v = lv::convertArrayToVector(a);
        EXPECT_EQ(a.size(),v.size());
        EXPECT_TRUE(std::equal(a.begin(),a.end(),v.begin()));
        auto a2 = lv::convertVectorToArray<4>(v);
        EXPECT_EQ(a.size(),a2.size());
        EXPECT_TRUE(std::equal(a.begin(),a.end(),a2.begin()));
    }
}

TEST(filter_out,regression) {
    std::vector<std::string> vNoVals;
    EXPECT_EQ((lv::filter_out(vNoVals,vNoVals)),(vNoVals));
    std::vector<std::string> vTokens{"sadf","//","12"};
    EXPECT_EQ((lv::filter_out(vNoVals,vTokens)),(vNoVals));
    std::vector<std::string> vVals = {"test","test2","12","test3"};
    EXPECT_EQ((lv::filter_out(vVals,vTokens)),(std::vector<std::string>{"test","test2","test3"}));
    EXPECT_EQ((lv::filter_out(vVals,vNoVals)),(vVals));
}

TEST(filter_in,regression) {
    std::vector<std::string> vNoVals;
    EXPECT_EQ((lv::filter_in(vNoVals,vNoVals)),(vNoVals));
    std::vector<std::string> vTokens{"sadf","//","12"};
    EXPECT_EQ((lv::filter_in(vNoVals,vTokens)),(vNoVals));
    std::vector<std::string> vVals = {"test","test2","12","test3"};
    EXPECT_EQ((lv::filter_in(vVals,vTokens)),(std::vector<std::string>{"12"}));
    EXPECT_EQ((lv::filter_in(vVals,vNoVals)),(vNoVals));
}

TEST(accumulateMembers,regression) {
    auto lObjEval = [](const float& a) {return a;};
    EXPECT_EQ((lv::accumulateMembers(std::vector<float>{},lObjEval,0.0f)),0.0f);
    EXPECT_EQ((lv::accumulateMembers(std::vector<float>{0.0f},lObjEval,0.0f)),0.0f);
    EXPECT_EQ((lv::accumulateMembers(std::vector<float>{1.0f},lObjEval,0.0f)),1.0f);
    EXPECT_EQ((lv::accumulateMembers(std::vector<float>{1.0f,2.0f,3.0f},lObjEval,0.0f)),6.0f);
    EXPECT_EQ((lv::accumulateMembers(std::vector<float>{1.0f,2.0f,3.0f},lObjEval,2.0f)),8.0f);
}

namespace {
    template<typename T>
    struct indices_of_fixture : testing::Test {};
    typedef testing::Types<char, int, size_t, float> indices_of_types;
}
TYPED_TEST_CASE(indices_of_fixture,indices_of_types);
TYPED_TEST(indices_of_fixture,regression) {
    const std::vector<TypeParam> vNoVal;
    const std::vector<TypeParam> vNoRef;
    EXPECT_EQ(lv::indices_of(vNoVal,vNoRef),std::vector<size_t>{});
    const std::vector<TypeParam> vSingleVal = {TypeParam(0)};
    EXPECT_EQ(lv::indices_of(vSingleVal,vNoRef),std::vector<size_t>{1});
    const std::vector<TypeParam> vSingleRef = {TypeParam(0)};
    EXPECT_EQ(lv::indices_of(vSingleVal,vSingleRef),std::vector<size_t>{0});
    const std::vector<TypeParam> vVals = {TypeParam(1),TypeParam(7),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(100)};
    EXPECT_EQ(lv::indices_of(vVals,vSingleRef),(std::vector<size_t>{1,1,1,1,0,1,1,1}));
    const std::vector<TypeParam> vRefs = {TypeParam(3),TypeParam(1),TypeParam(10),TypeParam(0)};
    EXPECT_EQ(lv::indices_of(vVals,vRefs),(std::vector<size_t>{1,4,0,4,3,4,4,4}));
}

namespace {
    template<typename T>
    struct sort_indices_fixture : testing::Test {
        struct custom_sort {
            const std::vector<T>& m_vVals;
            custom_sort(const std::vector<T>& vVals) : m_vVals(vVals) {}
            bool operator()(const size_t& a, const size_t& b) {
                return m_vVals[a]>m_vVals[b];
            }
        };
    };
    typedef testing::Types<char, int, float> sort_indices_types;
}
TYPED_TEST_CASE(sort_indices_fixture,sort_indices_types);
TYPED_TEST(sort_indices_fixture,regression) {
    const std::vector<TypeParam> vNoVal;
    EXPECT_EQ(lv::sort_indices(vNoVal),std::vector<size_t>{});
    const std::vector<TypeParam> vSingleVal = {TypeParam(0)};
    EXPECT_EQ(lv::sort_indices(vSingleVal),std::vector<size_t>{0});
    const std::vector<TypeParam> vVals = {TypeParam(1),TypeParam(7),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(100)};
    EXPECT_EQ(lv::sort_indices(vVals),(std::vector<size_t>{6,5,4,0,2,1,3,7}));
    typedef typename sort_indices_fixture<TypeParam>::custom_sort Sorter;
    EXPECT_EQ((lv::sort_indices(vVals,Sorter(vVals))),(std::vector<size_t>{7,3,1,2,0,4,5,6}));
}

namespace {
    template<typename T>
    struct unique_indices_fixture : testing::Test {
        struct custom_comp {
            const std::vector<T>& m_vVals;
            custom_comp(const std::vector<T>& vVals) : m_vVals(vVals) {}
            bool operator()(const size_t& a, const size_t& b) {
                return m_vVals[a]==m_vVals[b];
            }
        };
        struct custom_sort {
            const std::vector<T>& m_vVals;
            custom_sort(const std::vector<T>& vVals) : m_vVals(vVals) {}
            bool operator()(const size_t& a, const size_t& b) {
                return m_vVals[a]<m_vVals[b];
            }
        };
    };
    typedef testing::Types<char, int, float> unique_indices_types;
}
TYPED_TEST_CASE(unique_indices_fixture,unique_indices_types);
TYPED_TEST(unique_indices_fixture,regression) {
    const std::vector<TypeParam> vNoVal;
    EXPECT_EQ(lv::unique_indices(vNoVal),std::vector<size_t>{});
    const std::vector<TypeParam> vSingleVal = {TypeParam(0)};
    EXPECT_EQ(lv::unique_indices(vSingleVal),std::vector<size_t>{0});
    const std::vector<TypeParam> vVals = {TypeParam(1),TypeParam(7),TypeParam(1),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(8),TypeParam(100)};
    EXPECT_EQ(lv::unique_indices(vVals),(std::vector<size_t>{7,6,5,0,3,1,4,9}));
    typedef typename unique_indices_fixture<TypeParam>::custom_sort Sorter;
    typedef typename unique_indices_fixture<TypeParam>::custom_comp Comparer;
    EXPECT_EQ((lv::unique_indices(vVals,Sorter(vVals),Comparer(vVals))),(std::vector<size_t>{7,6,5,0,3,1,4,9}));
}

namespace {
    template<typename T>
    struct unique_fixture : testing::Test {};
    typedef testing::Types<char, int, float> unique_types;
}
TYPED_TEST_CASE(unique_fixture,unique_types);
TYPED_TEST(unique_fixture,regression) {
    const std::vector<TypeParam> vNoVal;
    EXPECT_EQ(lv::unique(vNoVal.begin(),vNoVal.end()),vNoVal);
    const std::vector<TypeParam> vSingleVal = {TypeParam(55)};
    EXPECT_EQ(lv::unique(vSingleVal.begin(),vSingleVal.end()),vSingleVal);
    const std::vector<TypeParam> vVals = {TypeParam(1),TypeParam(7),TypeParam(1),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(8),TypeParam(100)};
    EXPECT_EQ(lv::unique(vVals.begin(),vVals.end()),(std::vector<TypeParam>{TypeParam(-100),TypeParam(-1),TypeParam(0),TypeParam(1),TypeParam(3),TypeParam(7),TypeParam(8),TypeParam(100)}));
}

namespace {
    template<typename T>
    struct make_range_fixture : testing::Test {};
    typedef testing::Types<char, uchar, short, ushort, int, uint, size_t> make_range_types;
}
TYPED_TEST_CASE(make_range_fixture,make_range_types);
TYPED_TEST(make_range_fixture,regression) {
    EXPECT_EQ((lv::make_range(TypeParam(0),TypeParam(0))),(std::vector<TypeParam>{0}));
    EXPECT_EQ((lv::make_range(TypeParam(0),TypeParam(1))),(std::vector<TypeParam>{0,1}));
    EXPECT_EQ((lv::make_range(TypeParam(1),TypeParam(0))),(std::vector<TypeParam>{}));
    EXPECT_EQ((lv::make_range(TypeParam(0),TypeParam(5))),(std::vector<TypeParam>{0,1,2,3,4,5}));
    EXPECT_EQ((lv::make_range(TypeParam(5),TypeParam(9))),(std::vector<TypeParam>{5,6,7,8,9}));
    EXPECT_EQ((lv::make_range(TypeParam(5),TypeParam(9),TypeParam(2))),(std::vector<TypeParam>{5,7,9}));
    EXPECT_EQ((lv::make_range(TypeParam(0),TypeParam(6),TypeParam(3))),(std::vector<TypeParam>{0,3,6}));
    if(std::is_signed<TypeParam>::value) {
        EXPECT_EQ((lv::make_range(TypeParam(-1),TypeParam(-10))),(std::vector<TypeParam>{}));
        EXPECT_EQ((lv::make_range(TypeParam(-1),TypeParam(-1))),(std::vector<TypeParam>{TypeParam(-1)}));
        EXPECT_EQ((lv::make_range(TypeParam(-1),TypeParam(0))),(std::vector<TypeParam>{TypeParam(-1),0}));
        EXPECT_EQ((lv::make_range(TypeParam(-2),TypeParam(2))),(std::vector<TypeParam>{TypeParam(-2),TypeParam(-1),0,1,2}));
        EXPECT_EQ((lv::make_range(TypeParam(-2),TypeParam(2),TypeParam(2))),(std::vector<TypeParam>{TypeParam(-2),0,2}));
        EXPECT_EQ((lv::make_range(TypeParam(-2),TypeParam(2),TypeParam(4))),(std::vector<TypeParam>{TypeParam(-2),2}));
        EXPECT_EQ((lv::make_range(TypeParam(-10),TypeParam(-1),TypeParam(3))),(std::vector<TypeParam>{TypeParam(-10),TypeParam(-7),TypeParam(-4),TypeParam(-1)}));
    }
}

TEST(WorkerPool,regression_1thread) {
    lv::WorkerPool<1> wp1;
    std::future<size_t> nRes1 = wp1.queueTask([](){std::this_thread::sleep_for(std::chrono::seconds(1));return size_t(13);});
    std::future<size_t> nRes2 = wp1.queueTask([](){std::this_thread::sleep_for(std::chrono::milliseconds(200));return size_t(13);});
    ASSERT_TRUE(nRes1.valid() && nRes2.valid());
    ASSERT_EQ(nRes1.wait_for(std::chrono::seconds(2)),std::future_status::ready);
    ASSERT_EQ(nRes2.wait_for(std::chrono::seconds(2)),std::future_status::ready);
    ASSERT_EQ(nRes1.get(),size_t(13));
    ASSERT_EQ(nRes2.get(),size_t(13));
}

TEST(WorkerPool,regression_4threads) {
    lv::WorkerPool<4> wp4;
    std::array<std::future<size_t>,5> vRes;
    for(size_t i=0; i<vRes.size(); ++i) {
        vRes[i] = wp4.queueTask([](){std::this_thread::sleep_for(std::chrono::milliseconds(200));return size_t(13);});
        ASSERT_TRUE(vRes[i].valid());
    }
    for(size_t i=0; i<vRes.size(); ++i)
        ASSERT_EQ(vRes[i].wait_for(std::chrono::seconds(2)),std::future_status::ready);
    for(size_t i=0; i<vRes.size(); ++i)
        ASSERT_EQ(vRes[i].get(),size_t(13));
}

TEST(StopWatch,regression) {
    lv::StopWatch sw;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    const double dELapsed0 = sw.elapsed();
    const double dElapsed = sw.tock();
    EXPECT_NEAR(dElapsed,dELapsed0,0.05f);
    EXPECT_NEAR(dElapsed,1.0f,0.05f);
}

TEST(getTimeStamp,regression) {
    EXPECT_TRUE(!lv::getTimeStamp().empty());
}

TEST(getVersionStamp,regression) {
    EXPECT_TRUE(lv::getVersionStamp().find(LITIV_VERSION_STR)!=std::string::npos);
    EXPECT_TRUE(lv::getVersionStamp().find(LITIV_VERSION_SHA1)!=std::string::npos);
}

TEST(getLogStamp,regression) {
    EXPECT_TRUE(lv::getLogStamp().find(lv::getTimeStamp())!=std::string::npos);
    EXPECT_TRUE(lv::getLogStamp().find(lv::getLogStamp())!=std::string::npos);
}

TEST(enable_shared_from_this,regression) {
    struct enable_shared_from_this_test :
        lv::enable_shared_from_this<enable_shared_from_this_test> {
        virtual ~enable_shared_from_this_test() {}
    };
    struct enable_shared_from_this_test2 :
        enable_shared_from_this_test {};
    struct enable_shared_from_this_test3 {};
    std::shared_ptr<const enable_shared_from_this_test> test1 = std::make_shared<enable_shared_from_this_test2>();
    EXPECT_TRUE(test1->shared_from_this_cast<enable_shared_from_this_test2>()!=nullptr);
    std::shared_ptr<enable_shared_from_this_test> test2 = std::make_shared<enable_shared_from_this_test2>();
    EXPECT_TRUE(test2->shared_from_this_cast<enable_shared_from_this_test2>()!=nullptr);
    EXPECT_TRUE(test2->shared_from_this_cast<enable_shared_from_this_test3>()==nullptr);
}

TEST(has_const_iterator,regression) {
    static_assert(lv::has_const_iterator<std::vector<int>>::value,"gtest:has_const_iterator:regression failed");
    static_assert(!lv::has_const_iterator<int>::value,"gtest:has_const_iterator:regression failed");
}

TEST(for_each,regression) {
    const auto test1 = std::make_tuple(uchar(1),int(-34),3.2f,size_t(52),-13.34);
    float tot = 0.0f;
    lv::for_each(test1,[&tot](const auto& v){tot += float(v);});
    EXPECT_FLOAT_EQ(tot,1.0f-34.0f+3.2f+52.0f-13.34f);
    tot = 0.0f;
    lv::for_each_w_idx(test1,[&tot](const auto& v, size_t idx){tot += float(v)+float(idx);});
    EXPECT_FLOAT_EQ(tot,1.0f-34.0f+3.2f+52.0f-13.34f+(1.0f+2.0f+3.0f+4.0f));
    const std::array<float,5> test2 = {1.0f,-34.0f,3.2f,52.0f,-13.34f};
    tot = 0.0f;
    lv::for_each(test2,[&tot](const auto& v){tot += v;});
    EXPECT_FLOAT_EQ(tot,1.0f-34.0f+3.2f+52.0f-13.34f);
    tot = 0.0f;
    lv::for_each_w_idx(test2,[&tot](const auto& v, size_t idx){tot += v+float(idx);});
    EXPECT_FLOAT_EQ(tot,1.0f-34.0f+3.2f+52.0f-13.34f+(1.0f+2.0f+3.0f+4.0f));
}

TEST(unpack_and_call,regression) {
    const auto test1 = std::make_tuple(uchar(1),int(-34),3.2f,size_t(52),-13.34);
    const auto testfunc = [](auto v1, auto v2, auto v3, auto v4, auto v5) {
        return double(v1)+double(v2)+double(v3)+double(v4)+double(v5);
    };
    const std::array<float,5> test2 = {1.0f,-34.0f,3.2f,52.0f,-13.34f};
    EXPECT_FLOAT_EQ((float)lv::unpack_and_call(test1,testfunc),1.0f-34.0f+3.2f+52.0f-13.34f);
    EXPECT_FLOAT_EQ((float)lv::unpack_and_call(test2,testfunc),1.0f-34.0f+3.2f+52.0f-13.34f);
}

namespace {
    constexpr float static_transform_absdiff(float a, float b) {
        return a>b?a-b:b-a;
    }
}
TEST(static_transform,regression_static) {
    constexpr std::array<float,5> a1 = {5.0f,6.0f,7.0f,8.0f,9.0f};
    constexpr std::array<float,5> a2 = {10.0f,11.0f,12.0f,13.0f,14.0f};
    constexpr std::array<float,5> a3 = lv::static_transform(a1,a2,static_transform_absdiff);
    EXPECT_EQ(a3,(std::array<float,5>{5.0f,5.0f,5.0f,5.0f,5.0f}));
}

TEST(static_transform,regression_runtime) {
    const std::array<float,5> a1 = {5.0f,6.0f,7.0f,8.0f,9.0f};
    const std::array<float,5> a2 = {10.0f,11.0f,12.0f,13.0f,14.0f};
    const std::array<float,5> a3 = lv::static_transform(a1,a2,[](float a, float b){return std::abs(a-b);});
    EXPECT_EQ(a3,(std::array<float,5>{5.0f,5.0f,5.0f,5.0f,5.0f}));
}

namespace {
    constexpr float static_transform_abssum(float a, float b) {
        return (a>=0.0f?a:-a)+(b>=0.0f?b:-b);
    }
}
TEST(static_reduce,regression_static) {
    constexpr std::array<float,5> a1 = {5.0f,6.0f,7.0f,8.0f,9.0f};
    constexpr float res1 = lv::static_reduce(a1,static_transform_abssum);
    EXPECT_FLOAT_EQ(res1,35.0f);
}

TEST(static_reduce,regression_runtime) {
    const std::array<float,5> a1 = {5.0f,6.0f,7.0f,8.0f,9.0f};
    EXPECT_FLOAT_EQ((lv::static_reduce(a1,[](float a, float b){return std::abs(a)+std::abs(b);})),35.0f);
    EXPECT_FLOAT_EQ((lv::static_reduce(&a1[0],&a1[0]+a1.size(),[](float a, float b){return std::abs(a)+std::abs(b);})),35.0f);
}

namespace {
    template<typename T>
    struct AlignedMemAllocator_fixture : testing::Test {};
    typedef testing::Types<char, short, int, uint8_t, uint16_t, float, double> AlignedMemAllocator_types;
}
TYPED_TEST_CASE(AlignedMemAllocator_fixture,AlignedMemAllocator_types);
TYPED_TEST(AlignedMemAllocator_fixture,regression) {
    std::vector<TypeParam,lv::AlignedMemAllocator<TypeParam,16,false>> vVec16a(100);
    EXPECT_EQ(((uintptr_t)vVec16a.data()%16),size_t(0));
    ASSERT_EQ(vVec16a[0],TypeParam(0));
    ASSERT_TRUE(std::equal(vVec16a.begin()+1,vVec16a.end(),vVec16a.begin()));
    std::vector<TypeParam,lv::AlignedMemAllocator<TypeParam,32,false>> vVec32a(100);
    EXPECT_EQ(((uintptr_t)vVec32a.data()%32),size_t(0));
    ASSERT_EQ(vVec32a[0],TypeParam(0));
    ASSERT_TRUE(std::equal(vVec32a.begin()+1,vVec32a.end(),vVec32a.begin()));
}

namespace {
    template<typename T>
    struct AutoBuffer_fixture : testing::Test {};
    typedef testing::Types<char, short, int, uint8_t, uint16_t, float, double> AutoBuffer_types;
}
TYPED_TEST_CASE(AutoBuffer_fixture,AutoBuffer_types);
TYPED_TEST(AutoBuffer_fixture,regression) {
    lv::AutoBuffer<TypeParam,10,16> buff;
    ASSERT_FALSE(buff.empty());
    ASSERT_EQ(buff.size(),size_t(10));
    ASSERT_EQ(buff.max_static_size(),size_t(10));
    ASSERT_TRUE((uintptr_t(buff.data())%16)==0);
    ASSERT_EQ(buff.begin()+buff.size(),buff.end());
    ASSERT_THROW_LV_QUIET(buff.at(10));
    for(size_t i=0; i<buff.size(); ++i) {
        buff[i] = TypeParam(rand());
        ASSERT_EQ(buff.at(i),buff[i]);
        ASSERT_EQ(buff.data()[i],buff[i]);
        ASSERT_EQ(((TypeParam*)buff)[i],buff[i]);
        ASSERT_EQ((*(buff.begin()+i)),buff[i]);
    }
    buff.resize(5);
    ASSERT_EQ(buff.size(),size_t(5));
    const lv::AutoBuffer<TypeParam,16,16> buff2(buff);
    ASSERT_EQ(buff.size(),buff2.size());
    ASSERT_TRUE((uintptr_t(buff2.data())%16)==0);
    for(size_t i=0; i<buff2.size(); ++i)
        ASSERT_EQ(buff.at(i),buff2.at(i));
    const lv::AutoBuffer<TypeParam,10,16> buff3(std::move(buff2));
    ASSERT_EQ(buff.size(),buff3.size());
    ASSERT_TRUE((uintptr_t(buff3.data())%16)==0);
    for(size_t i=0; i<buff3.size(); ++i)
        ASSERT_EQ(buff.at(i),buff3.at(i));
    buff.resize(buff.size()*4);
    ASSERT_EQ(buff.size(),size_t(20));
    ASSERT_TRUE((uintptr_t(buff.data())%16)==0);
    for(size_t i=0; i<buff3.size(); ++i)
        ASSERT_EQ(buff.at(i),buff3.at(i));
    for(size_t i=0; i<buff.size(); ++i) {
        buff[i] = TypeParam(rand());
        ASSERT_EQ(buff.at(i),buff[i]);
        ASSERT_EQ(buff.data()[i],buff[i]);
        ASSERT_EQ(((TypeParam*)buff)[i],buff[i]);
        ASSERT_EQ((*(buff.begin()+i)),buff[i]);
    }
    lv::AutoBuffer<TypeParam,3,16> buff4;
    ASSERT_EQ(buff4.size(),size_t(3));
    ASSERT_TRUE((uintptr_t(buff4.data())%16)==0);
    buff4 = buff;
    ASSERT_EQ(buff.size(),buff4.size());
    ASSERT_TRUE((uintptr_t(buff4.data())%16)==0);
    for(size_t i=0; i<buff4.size(); ++i)
        ASSERT_EQ(buff.at(i),buff4.at(i));
    lv::AutoBuffer<TypeParam,6,16> buff5;
    ASSERT_EQ(buff5.size(),size_t(6));
    ASSERT_TRUE((uintptr_t(buff5.data())%16)==0);
    buff5 = std::move(buff4);
    ASSERT_EQ(buff.size(),buff5.size());
    ASSERT_TRUE((uintptr_t(buff5.data())%16)==0);
    for(size_t i=0; i<buff5.size(); ++i)
        ASSERT_EQ(buff.at(i),buff5.at(i));
}

TEST(LUT,regression_identity9) {
    constexpr size_t bins = 9;
    constexpr size_t safety = 0;
    const auto func_identity = [&](float x){return x;};
    lv::LUT<float,float,bins,safety> lut(0.0f,1.0f,func_identity);
    ASSERT_GE(lut.m_vLUT.size(),bins);
    ASSERT_FLOAT_EQ(lut.m_vLUT[0],0.0f);
    ASSERT_FLOAT_EQ(lut.m_vLUT[lut.m_vLUT.size()-1],1.0f);
    ASSERT_FLOAT_EQ(lut.m_tMin,0.0f);
    ASSERT_FLOAT_EQ(lut.m_tMax,1.0f);
    ASSERT_FLOAT_EQ(lut.m_tMidOffset,0.5f);
    ASSERT_FLOAT_EQ(lut.m_tLowOffset,0.0f);
    ASSERT_FLOAT_EQ(lut.m_tScale,float(bins-1));
    ASSERT_FLOAT_EQ(lut.m_tStep,1.0f/(bins-1));
    ASSERT_FLOAT_EQ(*lut.m_pMid,0.5f);
    ASSERT_FLOAT_EQ(*lut.m_pLow,0.0f);
    const float step = lut.m_tStep;
    for(size_t i=0; i<bins; ++i) {
        ASSERT_FLOAT_EQ(lut.eval(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_round(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_noffset(step*i-lut.m_tLowOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_noffset_round(step*i-lut.m_tLowOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_raw(i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_round(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_noffset(step*i-lut.m_tMidOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_noffset_round(step*i-lut.m_tMidOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_raw(i-bins/2),(step*i)) << "@ i=[" << i << "]";
    }
}

TEST(LUT,regression_identity9_safe) {
    constexpr size_t bins = 9;
    constexpr size_t safety = 2;
    const auto func_identity = [&](float x){return x;};
    lv::LUT<float,float,bins,safety> lut(0.0f,1.0f,func_identity);
    ASSERT_GE(lut.m_vLUT.size(),bins+safety);
    ASSERT_FLOAT_EQ(lut.m_vLUT[0],0.0f);
    ASSERT_FLOAT_EQ(lut.m_vLUT[lut.m_vLUT.size()-1],1.0f);
    ASSERT_FLOAT_EQ(lut.m_tMin,0.0f);
    ASSERT_FLOAT_EQ(lut.m_tMax,1.0f);
    ASSERT_FLOAT_EQ(lut.m_tMidOffset,0.5f);
    ASSERT_FLOAT_EQ(lut.m_tLowOffset,0.0f);
    ASSERT_FLOAT_EQ(lut.m_tScale,float(bins-1));
    ASSERT_FLOAT_EQ(lut.m_tStep,1.0f/(bins-1));
    ASSERT_FLOAT_EQ(*lut.m_pMid,0.5f);
    ASSERT_FLOAT_EQ(*lut.m_pLow,0.0f);
    const float step = lut.m_tStep;
    for(size_t i=0; i<bins; ++i) {
        ASSERT_FLOAT_EQ(lut.eval(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_round(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_noffset(step*i-lut.m_tLowOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_noffset_round(step*i-lut.m_tLowOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_raw(ptrdiff_t(i)),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_round(step*i),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_noffset(step*i-lut.m_tMidOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_noffset_round(step*i-lut.m_tMidOffset),(step*i)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_raw(ptrdiff_t(i)-bins/2),(step*i)) << "@ i=[" << i << "]";
    }
}

TEST(LUT,regression_arccos1000_safe) {
    constexpr size_t bins = 999;
    constexpr size_t safety = 10;
    const auto func_acos = [&](float x){return std::acos(x);};
    lv::LUT<float,float,bins,safety> lut(-1.0f,1.0f,func_acos);
    ASSERT_GE(lut.m_vLUT.size(),bins+safety);
    ASSERT_FLOAT_EQ(lut.m_vLUT[0],func_acos(-1.0f));
    ASSERT_FLOAT_EQ(lut.m_vLUT[lut.m_vLUT.size()-1],func_acos(1.0f));
    ASSERT_FLOAT_EQ(lut.m_tMin,-1.0f);
    ASSERT_FLOAT_EQ(lut.m_tMax,1.0f);
    ASSERT_FLOAT_EQ(lut.m_tMidOffset,0.0f);
    ASSERT_FLOAT_EQ(lut.m_tLowOffset,-1.0f);
    ASSERT_FLOAT_EQ(lut.m_tScale,float(bins-1)/2);
    ASSERT_FLOAT_EQ(lut.m_tStep,2.0f/(bins-1));
    ASSERT_NEAR(*lut.m_pMid,func_acos(0.0f),lut.m_tStep);
    ASSERT_NEAR(*lut.m_pLow,func_acos(-1.0f),lut.m_tStep);
    const float fMinX = -1.0f;
    for(size_t i=0; i<bins/2; ++i) {
        const float fLastY = func_acos(i==0?fMinX:fMinX+lut.m_tStep*(i-1));
        const float fX = fMinX+lut.m_tStep*i;
        const float fY = func_acos(fX);
        const float fDelta = (std::abs(fLastY-fY))*1.2f; // 20% added generosity for strange fp on travis
        ASSERT_NEAR(lut.eval(fX),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_round(fX),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_NEAR(lut.eval_noffset(fX-lut.m_tLowOffset),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_noffset_round(fX-lut.m_tLowOffset),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_raw(ptrdiff_t(i)),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_NEAR(lut.eval_mid(fX),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_round(fX),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_NEAR(lut.eval_mid_noffset(fX-lut.m_tMidOffset),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_noffset_round(fX-lut.m_tMidOffset),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_raw(ptrdiff_t(i)-bins/2),func_acos(fX)) << "@ i=[" << i << "]";
    }
    const float fMaxX = 1.0f;
    for(size_t i=bins/2; i<bins; ++i) {
        const float fNextY = func_acos(i==(bins-1)?fMaxX:fMinX+lut.m_tStep*(i+1));
        const float fX = fMinX+lut.m_tStep*i;
        const float fY = func_acos(fX);
        const float fDelta = (std::abs(fY-fNextY))*1.2f; // 20% added generosity for strange fp on travis
        ASSERT_NEAR(lut.eval(fX),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_round(fX),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_NEAR(lut.eval_noffset(fX-lut.m_tLowOffset),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_noffset_round(fX-lut.m_tLowOffset),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_raw(ptrdiff_t(i)),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_NEAR(lut.eval_mid(fX),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_round(fX),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_NEAR(lut.eval_mid_noffset(fX-lut.m_tMidOffset),func_acos(fX),fDelta) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_noffset_round(fX-lut.m_tMidOffset),func_acos(fX)) << "@ i=[" << i << "]";
        ASSERT_FLOAT_EQ(lut.eval_mid_raw(ptrdiff_t(i)-bins/2),func_acos(fX)) << "@ i=[" << i << "]";
    }
}

TEST(unlock_guard,regression_tuple) {
    std::mutex m1,m2,m3;
    std::lock_guard<std::mutex> g1(m1); // cannot merge locks without c++17 on gcc
    std::lock_guard<std::mutex> g2(m2);
    std::lock_guard<std::mutex> g3(m3);
    {
        ASSERT_FALSE(m1.try_lock() || m2.try_lock() || m3.try_lock());
        lv::unlock_guard<std::mutex,std::mutex,std::mutex> g4(m1,m2,m3);
        {
            ASSERT_TRUE(m1.try_lock() && m2.try_lock() && m3.try_lock());
            m1.unlock();
            m2.unlock();
            m3.unlock();
        }
    }
    ASSERT_FALSE(m1.try_lock() || m2.try_lock() || m3.try_lock());
}

TEST(unlock_guard,regression_single) {
    std::mutex m;
    std::lock_guard<std::mutex> g1(m);
    {
        ASSERT_FALSE(m.try_lock());
        lv::unlock_guard<std::mutex> g2(m);
        {
            ASSERT_TRUE(m.try_lock());
            m.unlock();
        }
    }
    ASSERT_FALSE(m.try_lock());
}

TEST(Semaphore,regression_simple) {
    lv::Semaphore s0(0);
    ASSERT_EQ(s0.count(),size_t(0));
    ASSERT_FALSE(s0.try_wait());
    s0.notify();
    ASSERT_EQ(s0.count(),size_t(1));
    ASSERT_TRUE(s0.try_wait());
    ASSERT_EQ(s0.count(),size_t(0));
}

TEST(Semaphore,regression_workers) {
    lv::Semaphore s2(2);
    ASSERT_EQ(s2.count(),size_t(2));
    {
        lv::WorkerPool<4> wp4;
        const auto work = [](){std::this_thread::sleep_for(std::chrono::milliseconds(rand()%100));};
        const auto lockwork = [&](){s2.wait();work();s2.notify();};
        for(size_t i=0; i<10; ++i) {
            wp4.queueTask(lockwork);
        }
    }
    ASSERT_EQ(s2.count(),size_t(2));
}