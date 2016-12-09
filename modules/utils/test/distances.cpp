
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

BENCHMARK_TEMPLATE(L1dist_perftest,float)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,int32_t)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,int16_t)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,int8_t)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,uint32_t)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,uint16_t)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L1dist_perftest,uint8_t)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct L1dist_signed_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int32_t,float,double> L1dist_signed_types;
}
TYPED_TEST_CASE(L1dist_signed_fixture,L1dist_signed_types);

TYPED_TEST(L1dist_signed_fixture,regression_single) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(0))),0.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(1),TypeParam(0))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(1))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(1),TypeParam(1))),0.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-1),TypeParam(0))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(-1))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-1),TypeParam(-1))),0.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-10),TypeParam(10))),20.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(-20),TypeParam(0))),20.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(TypeParam(0),TypeParam(20))),20.0);
    using TestTypeParam = std::conditional_t<std::is_same<double,TypeParam>::value,float,TypeParam>;
    const TestTypeParam tMax = std::numeric_limits<TestTypeParam>::max();
    const TestTypeParam tMin = std::numeric_limits<TestTypeParam>::min();
    EXPECT_DOUBLE_EQ(double(lv::L1dist(tMax,tMin)),(double(tMax)-tMin));
    EXPECT_DOUBLE_EQ(double(lv::L1dist(tMin,tMax)),(double(tMax)-tMin));
}

TYPED_TEST(L1dist_signed_fixture,regression_array) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),3.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),6.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),12.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),18.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,-6}.data())),6.0);
}

TYPED_TEST(L1dist_signed_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),3.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),6.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,0})),12.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,6})),18.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,-6})),6.0);
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
    EXPECT_DOUBLE_EQ(double(lv::L1dist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),3.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),6.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),12.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),6.0);
}

TYPED_TEST(L1dist_unsigned_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),3.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),6.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,0})),12.0);
    EXPECT_DOUBLE_EQ(double(lv::L1dist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,6})),6.0);
}

namespace {
    template<typename T>
    struct L1dist_fixture : testing::Test {};
    typedef testing::Types<int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,float,double> L1dist_types;
}
TYPED_TEST_CASE(L1dist_fixture,L1dist_types);

TYPED_TEST(L1dist_fixture,regression_mat_1_1) {
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

TYPED_TEST(L1dist_fixture,regression_mat_3_1) {
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

TYPED_TEST(L1dist_fixture,regression_mat_1_4) {
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

TYPED_TEST(L1dist_fixture,regression_mat_500_3) {
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

namespace {

    template<typename T>
    void L2sqrdist_perftest(benchmark::State& st) {
        const size_t nArraySize = size_t(st.range(0));
        const size_t nLoopSize = size_t(st.range(1));
        const T tMinVal = (T)st.range(2);
        const T tMaxVal = (T)st.range(3);
        std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        decltype(lv::L2sqrdist(T(),T())) tLast = 0;
        volatile size_t nArrayIdx = 0;
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                ++nArrayIdx %= nArraySize;
                tLast = lv::L2sqrdist(aVals[nArrayIdx],(T)tLast)/8;
            }
        }
        benchmark::DoNotOptimize(tLast);
    }

}

BENCHMARK_TEMPLATE(L2sqrdist_perftest,float)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L2sqrdist_perftest,int32_t)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L2sqrdist_perftest,int16_t)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L2sqrdist_perftest,int8_t)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L2sqrdist_perftest,uint32_t)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L2sqrdist_perftest,uint16_t)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(L2sqrdist_perftest,uint8_t)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct L2sqrdist_signed_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int32_t,float,double> L2sqrdist_signed_types;
}
TYPED_TEST_CASE(L2sqrdist_signed_fixture,L2sqrdist_signed_types);

TYPED_TEST(L2sqrdist_signed_fixture,regression_single) {
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(0),TypeParam(0))),0.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(1),TypeParam(0))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(0),TypeParam(1))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(1),TypeParam(1))),0.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(-1),TypeParam(0))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(0),TypeParam(-1))),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(-1),TypeParam(-1))),0.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(-10),TypeParam(10))),400.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(-20),TypeParam(0))),400.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(0),TypeParam(20))),400.0);
}

TYPED_TEST(L2sqrdist_signed_fixture,regression_array) {
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),5.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),14.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),50.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),158.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,-6}.data())),14.0);
}

TYPED_TEST(L2sqrdist_signed_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),5.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),14.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,0})),50.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,6})),158.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,-6})),14.0);
}

namespace {
    template<typename T>
    struct L2sqrdist_unsigned_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t,uint32_t> L2sqrdist_unsigned_types;
}
TYPED_TEST_CASE(L2sqrdist_unsigned_fixture,L2sqrdist_unsigned_types);

TYPED_TEST(L2sqrdist_unsigned_fixture,regression_single) {
    EXPECT_EQ(double(lv::L2sqrdist(TypeParam(0),TypeParam(0))),0.0);
    EXPECT_EQ(double(lv::L2sqrdist(TypeParam(1),TypeParam(0))),1.0);
    EXPECT_EQ(double(lv::L2sqrdist(TypeParam(0),TypeParam(1))),1.0);
    EXPECT_EQ(double(lv::L2sqrdist(TypeParam(1),TypeParam(1))),0.0);
    EXPECT_EQ(double(lv::L2sqrdist(TypeParam(0),TypeParam(20))),400.0);
    EXPECT_EQ(double(lv::L2sqrdist(TypeParam(20),TypeParam(0))),400.0);
    EXPECT_EQ(double(lv::L2sqrdist(TypeParam(20),TypeParam(20))),0.0);
    const TypeParam tMax = std::numeric_limits<TypeParam>::max();
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(tMax,TypeParam(0))),(double(tMax)*tMax));
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(TypeParam(0),tMax)),(double(tMax)*tMax));
}

TYPED_TEST(L2sqrdist_unsigned_fixture,regression_array) {
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),5.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),14.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),50.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),14.0);
}

TYPED_TEST(L2sqrdist_unsigned_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),1.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),5.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),14.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,0})),50.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,6})),14.0);
}

namespace {
    template<typename T>
    struct L2sqrdist_fixture : testing::Test {};
    typedef testing::Types<int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,float,double> L2sqrdist_types;
}
TYPED_TEST_CASE(L2sqrdist_fixture,L2sqrdist_types);

TYPED_TEST(L2sqrdist_fixture,regression_mat_1_1) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 1;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),nCols*nCols*nChannels*4.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),nCols*nChannels*4.0);
}

TYPED_TEST(L2sqrdist_fixture,regression_mat_3_1) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 3;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),nCols*nCols*nChannels*4.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),nCols*nChannels*4.0);
}

TYPED_TEST(L2sqrdist_fixture,regression_mat_1_4) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 1;
    constexpr size_t nChannels = 4;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),nCols*nCols*nChannels*4.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),nCols*nChannels*4.0);
}

TYPED_TEST(L2sqrdist_fixture,regression_mat_500_3) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 500;
    constexpr size_t nChannels = 3;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),nCols*nCols*nChannels*4.0);
    EXPECT_DOUBLE_EQ(double(lv::L2sqrdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),nCols*nChannels*4.0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    template<typename T, size_t nChannels>
    void L2dist_perftest(benchmark::State& st) {
        static_assert(nChannels>0,"need at least one channel!");
        const size_t nArraySize = size_t(st.range(0));
        lvAssert(nArraySize>nChannels*2);
        const size_t nLoopSize = size_t(st.range(1));
        const T tMinVal = (T)st.range(2);
        const T tMaxVal = (T)st.range(3);
        std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        decltype(lv::L2dist<nChannels>((T*)0,(T*)0)) tLast = 0;
        volatile size_t nArrayIdx1 = 0;
        volatile size_t nArrayIdx2 = nArraySize/2;
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nArraySize;
                tLast = (tLast + lv::L2dist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]))/2;
            }
        }
        benchmark::DoNotOptimize(tLast);
    }

}

BENCHMARK_TEMPLATE2(L2dist_perftest,float,3)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,int32_t,3)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,int16_t,3)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,int8_t,3)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,uint32_t,3)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,uint16_t,3)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,uint8_t,3)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct L2dist_signed_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int32_t,float,double> L2dist_signed_types;
}
TYPED_TEST_CASE(L2dist_signed_fixture,L2dist_signed_types);

#define L2DIST_EPS 0.000001

TYPED_TEST(L2dist_signed_fixture,regression_array) {
    EXPECT_DOUBLE_EQ(double(lv::L2dist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),1.0);
    EXPECT_NEAR(double(lv::L2dist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),std::sqrt(5.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),std::sqrt(14.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),std::sqrt(50.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),std::sqrt(158.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<4>(std::array<TypeParam,4>{1,2,3,-6}.data(),std::array<TypeParam,4>{0,0,0,-6}.data())),std::sqrt(14.0),L2DIST_EPS);
}

TYPED_TEST(L2dist_signed_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L2dist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),1.0);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),std::sqrt(5.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),std::sqrt(14.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,0})),std::sqrt(50.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,6})),std::sqrt(158.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,4>{1,2,3,-6},std::array<TypeParam,4>{0,0,0,-6})),std::sqrt(14.0),L2DIST_EPS);
}

namespace {
    template<typename T>
    struct L2dist_unsigned_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t,uint32_t> L2dist_unsigned_types;
}
TYPED_TEST_CASE(L2dist_unsigned_fixture,L2dist_unsigned_types);

TYPED_TEST(L2dist_unsigned_fixture,regression_array) {
    EXPECT_DOUBLE_EQ(double(lv::L2dist<1>(std::array<TypeParam,1>{1}.data(),std::array<TypeParam,1>{0}.data())),1.0);
    EXPECT_NEAR(double(lv::L2dist<2>(std::array<TypeParam,2>{1,2}.data(),std::array<TypeParam,2>{0,0}.data())),std::sqrt(5.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<3>(std::array<TypeParam,3>{1,2,3}.data(),std::array<TypeParam,3>{0,0,0}.data())),std::sqrt(14.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,0}.data())),std::sqrt(50.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<4>(std::array<TypeParam,4>{1,2,3,6}.data(),std::array<TypeParam,4>{0,0,0,6}.data())),std::sqrt(14.0),L2DIST_EPS);
}

TYPED_TEST(L2dist_unsigned_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::L2dist(std::array<TypeParam,1>{1},std::array<TypeParam,1>{0})),1.0);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,2>{1,2},std::array<TypeParam,2>{0,0})),std::sqrt(5.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,3>{1,2,3},std::array<TypeParam,3>{0,0,0})),std::sqrt(14.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,0})),std::sqrt(50.0),L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist(std::array<TypeParam,4>{1,2,3,6},std::array<TypeParam,4>{0,0,0,6})),std::sqrt(14.0),L2DIST_EPS);
}

namespace {
    template<typename T>
    struct L2dist_fixture : testing::Test {};
    typedef testing::Types<int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,float,double> L2dist_types;
}
TYPED_TEST_CASE(L2dist_fixture,L2dist_types);

TYPED_TEST(L2dist_fixture,regression_mat_1_1) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 1;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),std::sqrt(nCols*nCols*nChannels*4.0),nCols*L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),std::sqrt(nCols*nChannels*4.0),nCols*L2DIST_EPS);
}

TYPED_TEST(L2dist_fixture,regression_mat_3_1) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 3;
    constexpr size_t nChannels = 1;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),std::sqrt(nCols*nCols*nChannels*4.0),nCols*L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),std::sqrt(nCols*nChannels*4.0),nCols*L2DIST_EPS);
}

TYPED_TEST(L2dist_fixture,regression_mat_1_4) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 1;
    constexpr size_t nChannels = 4;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),std::sqrt(nCols*nCols*nChannels*4.0),nCols*L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),std::sqrt(nCols*nChannels*4.0),nCols*L2DIST_EPS);
}

TYPED_TEST(L2dist_fixture,regression_mat_500_3) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 500;
    constexpr size_t nChannels = 3;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a = cv::Mat_<cv::Vec<TypeParam,nChannels>>::zeros(nCols,nCols);
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>::all(2));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),std::sqrt(nCols*nCols*nChannels*4.0),nCols*L2DIST_EPS);
    EXPECT_NEAR(double(lv::L2dist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),std::sqrt(nCols*nChannels*4.0),nCols*L2DIST_EPS);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    template<typename T, size_t nChannels>
    void cdist_perftest(benchmark::State& st) {
        static_assert(nChannels>0,"need at least one channel!");
        const size_t nArraySize = size_t(st.range(0));
        lvAssert(nArraySize>nChannels*2);
        const size_t nLoopSize = size_t(st.range(1));
        const T tMinVal = (T)st.range(2);
        const T tMaxVal = (T)st.range(3);
        std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        decltype(lv::cdist<nChannels>((T*)0,(T*)0)) tLast = 0;
        volatile size_t nArrayIdx1 = 0;
        volatile size_t nArrayIdx2 = nArraySize/2;
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nArraySize;
                tLast = (tLast + lv::cdist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]))/2;
            }
        }
        benchmark::DoNotOptimize(tLast);
    }

}

BENCHMARK_TEMPLATE2(cdist_perftest,float,3)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(cdist_perftest,uint32_t,3)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(cdist_perftest,uint16_t,3)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(cdist_perftest,uint8_t,3)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct cdist_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t,uint32_t,float,double> cdist_types;
}
TYPED_TEST_CASE(cdist_fixture,L2dist_unsigned_types);

TYPED_TEST(cdist_fixture,regression_array) {
    constexpr size_t nArraySize = 1000;
    std::unique_ptr<TypeParam[]> aVals_0_1 = genarray(nArraySize,TypeParam(0),TypeParam(1));
    std::unique_ptr<TypeParam[]> aVals_0_255 = genarray(nArraySize,TypeParam(0),TypeParam(255));
    #define __cdist_mch(c) \
        ASSERT_GE(double(lv::cdist<c>(aVals_0_1.get()+i,aVals_0_1.get()+i+nArraySize/2)),0.0); \
        ASSERT_LE(double(lv::cdist<c>(aVals_0_1.get()+i,aVals_0_1.get()+i+nArraySize/2)),1.0*(c-1)); \
        ASSERT_GE(double(lv::cdist<c>(aVals_0_255.get()+i,aVals_0_255.get()+i+nArraySize/2)),0.0); \
        ASSERT_LE(double(lv::cdist<c>(aVals_0_255.get()+i,aVals_0_255.get()+i+nArraySize/2)),255.0*(c-1)); \
        ASSERT_DOUBLE_EQ(double(lv::cdist<c>(aVals_0_1.get()+i,aVals_0_1.get()+i)),0.0); \
        ASSERT_DOUBLE_EQ(double(lv::cdist<c>(aVals_0_255.get()+i,aVals_0_255.get()+i)),0.0)
    for(size_t i=0; i<nArraySize/2-4; ++i) {
        __cdist_mch(2);
        __cdist_mch(3);
        __cdist_mch(4);
    }
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,0}.data(),std::array<TypeParam,2>{0,0}.data())),0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,0}.data(),std::array<TypeParam,2>{0,0}.data())),1.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,0}.data(),std::array<TypeParam,2>{0,0}.data())),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,0}.data(),std::array<TypeParam,2>{1,0}.data())),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,0}.data(),std::array<TypeParam,2>{255,0}.data())),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,0}.data(),std::array<TypeParam,2>{0,1}.data())),1.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,1}.data(),std::array<TypeParam,2>{1,0}.data())),1.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,0}.data(),std::array<TypeParam,2>{0,255}.data())),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,255}.data(),std::array<TypeParam,2>{255,0}.data())),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,0}.data(),std::array<TypeParam,2>{0,1}.data())),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,1}.data(),std::array<TypeParam,2>{1,1}.data())),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,255}.data(),std::array<TypeParam,2>{255,255}.data())),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,1}.data(),std::array<TypeParam,2>{255,255}.data())),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,255}.data(),std::array<TypeParam,2>{1,1}.data())),0.0);
}

TYPED_TEST(cdist_fixture,regression_stdarray) {
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,0},std::array<TypeParam,2>{0,0})),0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,0},std::array<TypeParam,2>{0,0})),1.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,0},std::array<TypeParam,2>{0,0})),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,0},std::array<TypeParam,2>{1,0})),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,0},std::array<TypeParam,2>{255,0})),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,0},std::array<TypeParam,2>{0,1})),1.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,1},std::array<TypeParam,2>{1,0})),1.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,0},std::array<TypeParam,2>{0,255})),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{0,255},std::array<TypeParam,2>{255,0})),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,0},std::array<TypeParam,2>{0,1})),255.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,1},std::array<TypeParam,2>{1,1})),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,255},std::array<TypeParam,2>{255,255})),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{1,1},std::array<TypeParam,2>{255,255})),0.0);
    EXPECT_DOUBLE_EQ(double(lv::cdist<2>(std::array<TypeParam,2>{255,255},std::array<TypeParam,2>{1,1})),0.0);
}

TYPED_TEST(cdist_fixture,regression_mat_1_4) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 1;
    constexpr size_t nChannels = 4;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a(nCols,nCols,cv::Vec<TypeParam,nChannels>(0,255,0,255));
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>(255,0,255,0));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    typedef decltype(lv::cdist<nChannels>((TypeParam*)0,(TypeParam*)0,size_t(),0)) Tout;
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)b.data,(TypeParam*)a.data,a.total(),nullptr)),double(Tout(std::sqrt((255*255)*2.0))));
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)b.data,(TypeParam*)a.data,a.total(),m.data)),double(Tout(std::sqrt((255*255)*2.0))));
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(Tout(std::sqrt((255*255)*2.0))));
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(Tout(std::sqrt((255*255)*2.0))));
}

TYPED_TEST(cdist_fixture,regression_mat_500_3) {
    if(std::is_same<uint32_t,TypeParam>::value)
        return; // opencv mats do not support uint32_t, throws at runtime
    constexpr int nCols = 500;
    constexpr size_t nChannels = 3;
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> a(nCols,nCols,cv::Vec<TypeParam,nChannels>(255,0,255));
    const cv::Mat_<cv::Vec<TypeParam,nChannels>> b(a.size(),cv::Vec<TypeParam,nChannels>(0,255,0));
    const cv::Mat_<uint8_t> m = cv::Mat_<uint8_t>::eye(a.size());
    typedef decltype(lv::cdist<nChannels>((TypeParam*)0,(TypeParam*)0,size_t(),0)) Tout;
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)b.data,(TypeParam*)a.data,a.total(),nullptr)),double(nCols*nCols*255));
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)b.data,(TypeParam*)a.data,a.total(),m.data)),double(nCols*255));
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),nullptr)),double(nCols*nCols*Tout(std::sqrt((255*255)*2.0))));
    EXPECT_DOUBLE_EQ(double(lv::cdist<nChannels>((TypeParam*)a.data,(TypeParam*)b.data,a.total(),m.data)),double(nCols*Tout(std::sqrt((255*255)*2.0))));
}