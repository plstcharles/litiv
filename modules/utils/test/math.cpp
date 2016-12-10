
#include "gtest/gtest.h"
#include "benchmark/benchmark.h"
#include "litiv/utils/math.hpp"
#include <random>

#define BENCHMARK_NB_CHANNELS 3

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
        typedef std::conditional_t<
            std::is_integral<T>::value,
            std::uniform_int_distribution<
                std::conditional_t<
                    std::is_same<uint64_t,T>::value,
                    uint64_t,
                    int64_t
                >
            >,
            std::uniform_real_distribution<double>
        > unif_distr;
        unif_distr uniform_dist(min,max);
        std::unique_ptr<T[]> v(new T[n]);
        for(size_t i=0; i<n; ++i)
            v[i] = (T)uniform_dist(gen);
        return std::move(v);
    }

    template<typename T, size_t nChannels>
    void L1dist_perftest(benchmark::State& st) {
        static_assert(nChannels>0,"need at least one channel!");
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile T tMinVal = (T)st.range(2);
        const volatile T tMaxVal = (T)st.range(3);
        const std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        size_t nArrayIdx1 = 0;
        size_t nArrayIdx2 = nArraySize/2;
        lvAssert(nArraySize>nChannels*2);
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nArraySize;
                auto tLast = lv::L1dist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE2(L1dist_perftest,float,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L1dist_perftest,int32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L1dist_perftest,int16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L1dist_perftest,int8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L1dist_perftest,uint32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L1dist_perftest,uint16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L1dist_perftest,uint8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct L1dist_signed_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int32_t,float,double> L1dist_signed_types;
}
TYPED_TEST_CASE(L1dist_signed_fixture,L1dist_signed_types);

TYPED_TEST(L1dist_signed_fixture,regression_base) {
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

TYPED_TEST(L1dist_unsigned_fixture,regression_base) {
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

    template<typename T, size_t nChannels>
    void L2sqrdist_perftest(benchmark::State& st) {
        static_assert(nChannels>0,"need at least one channel!");
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile T tMinVal = (T)st.range(2);
        const volatile T tMaxVal = (T)st.range(3);
        const std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        size_t nArrayIdx1 = 0;
        size_t nArrayIdx2 = nArraySize/2;
        lvAssert(nArraySize>nChannels*2);
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nArraySize;
                auto tLast = lv::L2sqrdist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE2(L2sqrdist_perftest,float,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2sqrdist_perftest,int32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2sqrdist_perftest,int16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2sqrdist_perftest,int8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2sqrdist_perftest,uint32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2sqrdist_perftest,uint16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2sqrdist_perftest,uint8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct L2sqrdist_signed_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int32_t,float,double> L2sqrdist_signed_types;
}
TYPED_TEST_CASE(L2sqrdist_signed_fixture,L2sqrdist_signed_types);

TYPED_TEST(L2sqrdist_signed_fixture,regression_base) {
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

TYPED_TEST(L2sqrdist_unsigned_fixture,regression_base) {
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
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile T tMinVal = (T)st.range(2);
        const volatile T tMaxVal = (T)st.range(3);
        const std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        size_t nArrayIdx1 = 0;
        size_t nArrayIdx2 = nArraySize/2;
        lvAssert(nArraySize>nChannels*2);
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nArraySize;
                auto tLast = lv::L2dist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE2(L2dist_perftest,float,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,int32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,int16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,int8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,-10,10})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,uint32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,uint16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(L2dist_perftest,uint8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,20})->Repetitions(10)->ReportAggregatesOnly(true);

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
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile T tMinVal = (T)st.range(2);
        const volatile T tMaxVal = (T)st.range(3);
        const std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        size_t nArrayIdx1 = 0;
        size_t nArrayIdx2 = nArraySize/2;
        lvAssert(nArraySize>nChannels*2);
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nArraySize;
                auto tLast = lv::cdist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE2(cdist_perftest,float,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(cdist_perftest,uint32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(cdist_perftest,uint16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(cdist_perftest,uint8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100,0,255})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct cdist_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t,uint32_t,float,double> cdist_types;
}
TYPED_TEST_CASE(cdist_fixture,L2dist_unsigned_types);

TYPED_TEST(cdist_fixture,regression_array) {
    constexpr size_t nArraySize = 1000;
    const std::unique_ptr<TypeParam[]> aVals_0_1 = genarray(nArraySize,TypeParam(0),TypeParam(1));
    const std::unique_ptr<TypeParam[]> aVals_0_255 = genarray(nArraySize,TypeParam(0),TypeParam(255));
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

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    template<typename T, size_t nChannels>
    void popcount_perftest(benchmark::State& st) {
        static_assert(nChannels>0,"need at least one channel!");
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile T tMinVal = std::numeric_limits<T>::min();
        const volatile T tMaxVal = std::numeric_limits<T>::max();
        const std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        size_t nArrayIdx = 0;
        lvAssert(nArraySize>nChannels);
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx = (nArrayIdx+nChannels)%nArraySize;
                auto tLast = lv::popcount<nChannels>(&aVals[nArrayIdx]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE2(popcount_perftest,int64_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(popcount_perftest,uint64_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(popcount_perftest,int32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(popcount_perftest,uint32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(popcount_perftest,int16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(popcount_perftest,uint16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(popcount_perftest,int8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(popcount_perftest,uint8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct popcount_fixture : testing::Test {};
    typedef testing::Types<int8_t,uint8_t,int16_t,uint16_t,int32_t,uint32_t,int64_t,uint64_t> popcount_types;
}
TYPED_TEST_CASE(popcount_fixture,popcount_types);

TYPED_TEST(popcount_fixture,regression_base) {
    EXPECT_EQ(size_t(lv::popcount(TypeParam(0))),size_t(0));
    EXPECT_EQ(size_t(lv::popcount(TypeParam(1))),size_t(1));
    EXPECT_EQ(size_t(lv::popcount(TypeParam(3))),size_t(2));
    EXPECT_EQ(size_t(lv::popcount(TypeParam(8))),size_t(1));
    EXPECT_EQ(size_t(lv::popcount(TypeParam(-1))),size_t(sizeof(TypeParam)*8));
}

TYPED_TEST(popcount_fixture,regression_array) {
    EXPECT_EQ(size_t(lv::popcount<1>(std::array<TypeParam,1>{0}.data())),size_t(0));
    EXPECT_EQ(size_t(lv::popcount<1>(std::array<TypeParam,1>{1}.data())),size_t(1));
    EXPECT_EQ(size_t(lv::popcount<2>(std::array<TypeParam,2>{1,0}.data())),size_t(1));
    EXPECT_EQ(size_t(lv::popcount<2>(std::array<TypeParam,2>{1,3}.data())),size_t(3));
    EXPECT_EQ(size_t(lv::popcount<3>(std::array<TypeParam,3>{1,3,TypeParam(-1)}.data())),size_t(3+sizeof(TypeParam)*8));
}

TYPED_TEST(popcount_fixture,regression_stdarray) {
    EXPECT_EQ(size_t(lv::popcount<1>(std::array<TypeParam,1>{0})),size_t(0));
    EXPECT_EQ(size_t(lv::popcount<1>(std::array<TypeParam,1>{1})),size_t(1));
    EXPECT_EQ(size_t(lv::popcount<2>(std::array<TypeParam,2>{1,0})),size_t(1));
    EXPECT_EQ(size_t(lv::popcount<2>(std::array<TypeParam,2>{1,3})),size_t(3));
    EXPECT_EQ(size_t(lv::popcount<3>(std::array<TypeParam,3>{1,3,TypeParam(-1)})),size_t(3+sizeof(TypeParam)*8));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    template<typename T, size_t nChannels>
    void hdist_perftest(benchmark::State& st) {
        static_assert(nChannels>0,"need at least one channel!");
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile T tMinVal = std::numeric_limits<T>::min();
        const volatile T tMaxVal = std::numeric_limits<T>::max();
        const std::unique_ptr<T[]> aVals = genarray<T>(nArraySize,tMinVal,tMaxVal);
        size_t nArrayIdx1 = 0;
        size_t nArrayIdx2 = nArraySize/2;
        lvAssert(nArraySize>nChannels*2);
        while(st.KeepRunning()) {
            for(size_t nLoopIdx=0; nLoopIdx<nLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nArraySize;
                auto tLast = lv::hdist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE2(hdist_perftest,int64_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(hdist_perftest,uint64_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(hdist_perftest,int32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(hdist_perftest,uint32_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(hdist_perftest,int16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(hdist_perftest,uint16_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(hdist_perftest,int8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE2(hdist_perftest,uint8_t,BENCHMARK_NB_CHANNELS)->Args({1000000,100})->Repetitions(10)->ReportAggregatesOnly(true);

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
    template<typename T>
    struct find_nn_index_fixture : testing::Test {
        struct custom_dist {
            double operator()(const T& a, const T& b) {
                return double(a)>=double(b)?double(a)-double(b):double(b)-double(a);
            }
        };
    };
    typedef testing::Types<char, int, float> find_nn_index_types;
}
TYPED_TEST_CASE(find_nn_index_fixture,find_nn_index_types);

TYPED_TEST(find_nn_index_fixture,regression) {
    const std::vector<TypeParam> vNoVal;
    typedef typename find_nn_index_fixture<TypeParam>::custom_dist Dist;
    Dist dist;
    EXPECT_EQ(lv::find_nn_index(TypeParam(-10),vNoVal,dist),size_t(-1));
    const std::vector<TypeParam> vSingleVal = {TypeParam(55)};
    EXPECT_EQ(lv::find_nn_index(TypeParam(-10),vSingleVal,dist),size_t(0));
    const std::vector<TypeParam> vVals = {TypeParam(1),TypeParam(7),TypeParam(3),TypeParam(8),TypeParam(0),TypeParam(-1),TypeParam(-100),TypeParam(100)};
    EXPECT_EQ(lv::find_nn_index(TypeParam(-10),vVals,dist),size_t(5));
    EXPECT_EQ(lv::find_nn_index(TypeParam(10),vVals,dist),size_t(3));
    EXPECT_EQ(lv::find_nn_index(TypeParam(100),vVals,dist),size_t(7));
    EXPECT_EQ(lv::find_nn_index(TypeParam(7.2),vVals,dist),size_t(1));
    EXPECT_EQ(lv::find_nn_index(TypeParam(5),vVals,dist),size_t(1));
}

TEST(interp1,regression) {
    const std::vector<int> vX = {1,3,4,6};
    const std::vector<float> vY = {2.0f,6.0f,8.0f,12.0f};
    const std::vector<int> vXReq = {2,5};
    const std::vector<float> vYAnsw = lv::interp1(vX,vY,vXReq);
    ASSERT_EQ(vYAnsw.size(),size_t(2));
    EXPECT_FLOAT_EQ(vYAnsw[0],4.0f);
    EXPECT_FLOAT_EQ(vYAnsw[1],10.0f);
}

TEST(linspace,regression) {
    EXPECT_EQ(lv::linspace(5.0f,10.0f,0,true),std::vector<float>());
    EXPECT_EQ(lv::linspace(5.0f,10.0f,1,true),std::vector<float>{10.0f});
    EXPECT_EQ(lv::linspace(5.0f,10.0f,2,true),(std::vector<float>{5.0f,10.0f}));
    EXPECT_EQ(lv::linspace(5.0f,5.0f,100,false),std::vector<float>(100,5.0f));
    EXPECT_EQ(lv::linspace(5.0f,5.0f,100,true),std::vector<float>(100,5.0f));
    const std::vector<float> vTest1 = lv::linspace(4.0f,5.0f,2,false);
    ASSERT_EQ(vTest1.size(),size_t(2));
    EXPECT_FLOAT_EQ(vTest1[0],4.5f);
    EXPECT_FLOAT_EQ(vTest1[1],5.0f);
}

TEST(expand_bits,regression) {
    const uint32_t nTest0 = 0;
    EXPECT_EQ(lv::expand_bits<4>(nTest0),uint32_t(0));
    const uint32_t nTest1 = 0b1111;
    EXPECT_EQ(lv::expand_bits<4>(nTest1),uint32_t(0b0001000100010001));
    const uint32_t nTest2 = 0b101010;
    EXPECT_EQ(lv::expand_bits<4>(nTest2),uint32_t(0b000100000001000000010000));
}

TEST(isnan,regression) {
    EXPECT_EQ(lv::isnan(std::numeric_limits<float>::quiet_NaN()),true);
    EXPECT_EQ(lv::isnan(std::numeric_limits<double>::quiet_NaN()),true);
    EXPECT_EQ(lv::isnan(std::numeric_limits<float>::max()),false);
    EXPECT_EQ(lv::isnan(std::numeric_limits<double>::max()),false);
}