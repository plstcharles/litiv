
#include "gtest/gtest.h"
#include "benchmark/benchmark.h"
#include "litiv/utils/distances.hpp"
#include <random>

#if USE_SIGNEXT_SHIFT_TRICK
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-aliasing"
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif //defined(...GCC)

TEST(bittrick_signextshift,regression) {
    float fVal = -123.45f;
    int MAY_ALIAS nCast = reinterpret_cast<int&>(fVal);
    nCast &= 0x7FFFFFFF;
    const float fRes = reinterpret_cast<float&>(nCast);
    ASSERT_EQ(fRes,123.45f) << "sign-extended right shift not supported, bit trick for floating point abs value will fail";
}

#if defined(__clang__)
#pragma clang diagnostic pop
#elif (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#endif //defined(...GCC)
#endif //USE_SIGNEXT_SHIFT_TRICK

namespace {

    template<typename T>
    std::unique_ptr<T[]> genarray(size_t n, T min, T max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        typedef std::conditional_t<std::is_integral<T>::value,std::uniform_int_distribution<int64_t>,std::uniform_real_distribution<double>> unif_distr;
        unif_distr uniform_dist(min,max);
        std::unique_ptr<T[]> v(new T[n]);
        for(size_t i=0; i<n; ++i)
            v[i] = (T)uniform_dist(gen);
        return std::move(v);
    }

    template<typename T>
    void L1dist_perftest(benchmark::State& st) {
        const size_t nArraySize = size_t(st.range(0));
        const size_t nLoopSize = size_t(st.range(1));
        const T tMinVal = (T)st.range(2);
        const T tMaxVal = (T)st.range(3);
        std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        decltype(lv::L1dist(T(),T())) tLast = 0;
        volatile size_t nArrayIdx = 0;
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                ++nArrayIdx %= nArraySize;
                tLast = lv::L1dist(aVals[nArrayIdx],(T)tLast)/2;
            }
        }
        benchmark::DoNotOptimize(tLast);
    }

}
BENCHMARK_TEMPLATE(L1dist_perftest,double)->Args({1000000,100,-10000,10000})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,float)->Args({1000000,100,-1000,1000})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,short)->Args({1000000,100,-100,100})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,char)->Args({1000000,100,-10,10})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct L1dist_signed_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int32_t,float,double> L1dist_signed_types;
}
TYPED_TEST_CASE(L1dist_signed_fixture,L1dist_signed_types);

TYPED_TEST(L1dist_signed_fixture,regression_single) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(0))),double(0));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(1),TypeParam(0))),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(1))),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(1),TypeParam(1))),double(0));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-1),TypeParam(0))),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(-1))),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-1),TypeParam(-1))),double(0));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-10),TypeParam(10))),double(20));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-20),TypeParam(0))),double(20));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(20))),double(20));
    using TestTypeParam = std::conditional_t<std::is_same<double,TypeParam>::value,float,TypeParam>;
    const TestTypeParam tMax = std::numeric_limits<TestTypeParam>::max();
    const TestTypeParam tMin = std::numeric_limits<TestTypeParam>::min();
    EXPECT_DOUBLE_EQ(double(lv::L1dist(tMax,tMin)),(double(tMax)-tMin));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(tMin,tMax)),(double(tMax)-tMin));
}

TYPED_TEST(L1dist_signed_fixture,regression_array) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),double(3));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),double(6));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),double(12));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),double(18));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,-6}.data())),double(6));
}

TYPED_TEST(L1dist_signed_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),double(3));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),double(6));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,0})),double(12));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,6})),double(18));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,-6})),double(6));
}

TYPED_TEST(L1dist_signed_fixture,regression_mat11) {
    constexpr int nCols = 1;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

TYPED_TEST(L1dist_signed_fixture,regression_mat31) {
    constexpr int nCols = 3;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

TYPED_TEST(L1dist_signed_fixture,regression_mat14) {
    constexpr int nCols = 1;
    constexpr size_t nChannels = 4;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

TYPED_TEST(L1dist_signed_fixture,regression_mat5003) {
    constexpr int nCols = 500;
    constexpr size_t nChannels = 3;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

namespace {
    template<typename T>
    struct L1dist_unsigned_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t,uint32_t> L1dist_unsigned_types;
}
TYPED_TEST_CASE(L1dist_unsigned_fixture,L1dist_unsigned_types);

TYPED_TEST(L1dist_unsigned_fixture,regression_single) {
    EXPECT_EQ((lv::L1dist(TypeParam(0),TypeParam(0))),TypeParam(0));
    EXPECT_EQ((lv::L1dist(TypeParam(1),TypeParam(0))),TypeParam(1));
    EXPECT_EQ((lv::L1dist(TypeParam(0),TypeParam(1))),TypeParam(1));
    EXPECT_EQ((lv::L1dist(TypeParam(1),TypeParam(1))),TypeParam(0));;
    EXPECT_EQ((lv::L1dist(TypeParam(0),TypeParam(20))),TypeParam(20));
    EXPECT_EQ((lv::L1dist(TypeParam(20),TypeParam(0))),TypeParam(20));
    EXPECT_EQ((lv::L1dist(TypeParam(20),TypeParam(20))),TypeParam(0));
    const TypeParam tMax = std::numeric_limits<TypeParam>::max();
    const TypeParam tMin = std::numeric_limits<TypeParam>::min();
    EXPECT_DOUBLE_EQ(double(lv::L1dist(tMax,tMin)),(double(tMax)-tMin));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(tMin,tMax)),(double(tMax)-tMin));
}

TYPED_TEST(L1dist_unsigned_fixture,regression_array) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),double(3));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),double(6));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),double(12));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),double(6));
}

TYPED_TEST(L1dist_unsigned_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),double(1));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),double(3));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),double(6));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,0})),double(12));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,6})),double(6));
}

TYPED_TEST(L1dist_unsigned_fixture,regression_mat11) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 1;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

TYPED_TEST(L1dist_unsigned_fixture,regression_mat31) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 3;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

TYPED_TEST(L1dist_unsigned_fixture,regression_mat14) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 1;
    constexpr size_t nChannels = 4;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

TYPED_TEST(L1dist_unsigned_fixture,regression_mat5003) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 500;
    constexpr size_t nChannels = 3;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(1));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*nChannels));
    EXPECT_DOUBLE_EQ(double(lv::L1dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*nChannels));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// add other dist funcs tests here...