
#include <gtest/gtest.h>
#include <litiv/utils.hpp>

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
    EXPECT_EQ(lv::digit_count(0.5f),1);
    EXPECT_EQ(lv::digit_count(1),1);
    EXPECT_EQ(lv::digit_count(5),1);
    EXPECT_EQ(lv::digit_count(10),2);
    EXPECT_EQ(lv::digit_count(555),3);
    EXPECT_EQ(lv::digit_count(-1.5),2);
    EXPECT_EQ(lv::digit_count(-1),2);
    EXPECT_EQ(lv::digit_count(-10),3);
    EXPECT_EQ(lv::digit_count(-1000),5);
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

namespace {
    template<typename T>
    struct indices_of_fixture : ::testing::Test {};
    typedef ::testing::Types<char, int, size_t, float> indices_of_types;
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
    struct sort_indices_fixture : ::testing::Test {};
    template<typename T>
    struct sort_indices_custom_comp {
        const std::vector<T>& m_vVals;
        sort_indices_custom_comp(const std::vector<T>& vVals) : m_vVals(vVals) {}
        bool operator()(const size_t& a, const size_t& b) {
            return m_vVals[a]>m_vVals[b];
        }
    };
    typedef ::testing::Types<char, int, float> sort_indices_types;
}
TYPED_TEST_CASE(sort_indices_fixture,sort_indices_types);
TYPED_TEST(sort_indices_fixture,regression) {
    const std::vector<TypeParam> vNoVal;
    EXPECT_EQ(lv::sort_indices(vNoVal),std::vector<size_t>{});
    const std::vector<TypeParam> vSingleVal = {TypeParam(0)};
    EXPECT_EQ(lv::sort_indices(vSingleVal),std::vector<size_t>{0});
    const std::vector<TypeParam> vVals = {TypeParam(1),TypeParam(7),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(100)};
    EXPECT_EQ(lv::sort_indices(vVals),(std::vector<size_t>{6,5,4,0,2,1,3,7}));
    EXPECT_EQ((lv::sort_indices(vVals,sort_indices_custom_comp<TypeParam>(vVals))),(std::vector<size_t>{7,3,1,2,0,4,5,6}));
}