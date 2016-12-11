
#include "gtest/gtest.h"
#include "benchmark/benchmark.h"
#include "litiv/utils/math.hpp"
#include <random>

#define BENCHMARK_NB_CHANNELS 3

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

}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(isnan,regression) {
    EXPECT_EQ(lv::isnan(std::numeric_limits<float>::quiet_NaN()),true);
    EXPECT_EQ(lv::isnan(std::numeric_limits<double>::quiet_NaN()),true);
    EXPECT_EQ(lv::isnan(std::numeric_limits<float>::max()),false);
    EXPECT_EQ(lv::isnan(std::numeric_limits<double>::max()),false);
}

TEST(ispow2,regression) {
    EXPECT_EQ(lv::ispow2(1),true);
    EXPECT_EQ(lv::ispow2(2),true);
    EXPECT_EQ(lv::ispow2(3),false);
    EXPECT_EQ(lv::ispow2(4),true);
    EXPECT_EQ(lv::ispow2(16),true);
    EXPECT_EQ(lv::ispow2(17),false);
    EXPECT_EQ(lv::ispow2(32),true);
    EXPECT_EQ(lv::ispow2(std::numeric_limits<int>::max()),false);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(abs_fast,regression) {
    ASSERT_EQ(lv::abs_fast(0.0f),0.0f) << "sign-extended right shift not supported, bit trick for floating point abs value will fail";
    ASSERT_EQ(lv::abs_fast(-123.45f),123.45f) << "sign-extended right shift not supported, bit trick for floating point abs value will fail";
}

namespace {

    template<bool bUseFast>
    void abs_fast_perftest(benchmark::State& st) {
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile float fMinVal = std::numeric_limits<float>::min();
        const volatile float fMaxVal = std::numeric_limits<float>::max();
        const std::unique_ptr<float[]> afVals = genarray<float>(nArraySize,fMinVal,fMaxVal);
        size_t nArrayIdx = 0;
        while(st.KeepRunning()) {
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx = (nArrayIdx+1)%nCurrArraySize;
                volatile auto tLast = bUseFast?(lv::abs_fast(afVals[nArrayIdx])):(std::abs(afVals[nArrayIdx]));
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE1(abs_fast_perftest,true)->Args({1000000,250})->Repetitions(15)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE1(abs_fast_perftest,false)->Args({1000000,250})->Repetitions(15)->ReportAggregatesOnly(true);

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(inv_fast,regression) {
    constexpr float fErr = 0.15f; // allow +/- 15% deviation off result
    EXPECT_FLOAT_EQ(lv::inv_fast(1.0f),1.0f);
    EXPECT_FLOAT_EQ(lv::inv_fast(0.5f),2.0f);
    EXPECT_FLOAT_EQ(lv::inv_fast(2.0f),0.5f);
    EXPECT_NEAR(lv::inv_fast(10.0f),0.1f,0.1f*fErr);
    EXPECT_NEAR(lv::inv_fast(0.1f),10.0f,10.0f*fErr);
    constexpr size_t nArraySize = 100000;
    const std::unique_ptr<float[]> afVals = genarray(nArraySize,-10000.0f,10000.0f);
    for(size_t i=0; i<nArraySize; ++i)
        ASSERT_NEAR(lv::inv_fast(afVals[i]),1.0f/afVals[i],std::abs((1.0f/afVals[i])*fErr));
}

namespace {

    template<bool bUseFast>
    void inv_fast_perftest(benchmark::State& st) {
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile float fMinVal = std::numeric_limits<float>::min();
        const volatile float fMaxVal = std::numeric_limits<float>::max();
        const std::unique_ptr<float[]> afVals = genarray<float>(nArraySize,fMinVal,fMaxVal);
        size_t nArrayIdx = 0;
        while(st.KeepRunning()) {
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx = (nArrayIdx+1)%nCurrArraySize;
                volatile auto tLast = bUseFast?(lv::inv_fast(afVals[nArrayIdx])):(1.0f/afVals[nArrayIdx]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE1(inv_fast_perftest,true)->Args({1000000,250})->Repetitions(15)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE1(inv_fast_perftest,false)->Args({1000000,250})->Repetitions(15)->ReportAggregatesOnly(true);

///////////////////////////////////////////////////////////////////////////////////////////////////

TEST(invsqrt_fast_0iter,regression) {
    constexpr float fErr = 0.05f; // allow +/- 5% deviation off result
    EXPECT_NEAR(lv::invsqrt_fastest(1.0f),1.0f,1.0f*fErr);
    EXPECT_NEAR(lv::invsqrt_fastest(4.0f),0.5f,0.5f*fErr);
    EXPECT_NEAR(lv::invsqrt_fastest(16.0f),0.25f,0.25f*fErr);
    EXPECT_NEAR(lv::invsqrt_fastest(0.5f),1.0f/0.7071067f,(1.0f/0.7071067f)*fErr);
    EXPECT_NEAR(lv::invsqrt_fastest(223.31f),1.0f/14.94356f,(1.0f/14.94356f)*fErr);
    constexpr size_t nArraySize = 100000;
    const std::unique_ptr<float[]> afVals = genarray(nArraySize,0.0f,10000.0f);
    for(size_t i=0; i<nArraySize; ++i)
        ASSERT_NEAR(lv::invsqrt_fastest(afVals[i]),1.0f/std::sqrt(afVals[i]),(1.0f/std::sqrt(afVals[i]))*fErr);
}

#define INVSQRT_REGRESSION_TEST(n) \
TEST(invsqrt_fast_##n##iter,regression) { \
    const float fErr = 0.005f/std::pow(2.0f,(float)n); \
    EXPECT_NEAR(lv::invsqrt_fast<n>(1.0f),1.0f,1.0f*fErr); \
    EXPECT_NEAR(lv::invsqrt_fast<n>(4.0f),0.5f,0.5f*fErr); \
    EXPECT_NEAR(lv::invsqrt_fast<n>(16.0f),0.25f,0.25f*fErr); \
    EXPECT_NEAR(lv::invsqrt_fast<n>(0.5f),1.0f/0.7071067f,(1.0f/0.7071067f)*fErr); \
    EXPECT_NEAR(lv::invsqrt_fast<n>(223.31f),1.0f/14.94356f,(1.0f/14.94356f)*fErr); \
    constexpr size_t nArraySize = 100000; \
    const std::unique_ptr<float[]> afVals = genarray(nArraySize,0.0f,10000.0f); \
    for(size_t i=0; i<nArraySize; ++i) \
        ASSERT_NEAR(lv::invsqrt_fast<n>(afVals[i]),1.0f/std::sqrt(afVals[i]),(1.0f/std::sqrt(afVals[i]))*fErr); \
}

INVSQRT_REGRESSION_TEST(1)
INVSQRT_REGRESSION_TEST(2)
INVSQRT_REGRESSION_TEST(4)
INVSQRT_REGRESSION_TEST(8)

namespace {

    template<int nSpeedup>
    void invsqrt_fast_perftest(benchmark::State& st) {
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile float fMinVal = 0.0f;
        const volatile float fMaxVal = std::numeric_limits<float>::max();
        const std::unique_ptr<float[]> afVals = genarray<float>(nArraySize,fMinVal,fMaxVal);
        size_t nArrayIdx = 0;
        while(st.KeepRunning()) {
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx = (nArrayIdx+1)%nCurrArraySize;
                volatile auto tLast = (nSpeedup>=2)?(lv::invsqrt_fastest(afVals[nArrayIdx])):((nSpeedup==1)?(lv::invsqrt_fast(afVals[nArrayIdx])):(1.0f/std::sqrt(afVals[nArrayIdx])));
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE1(invsqrt_fast_perftest,2)->Args({1000000,250})->Repetitions(15)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE1(invsqrt_fast_perftest,1)->Args({1000000,250})->Repetitions(15)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE1(invsqrt_fast_perftest,0)->Args({1000000,250})->Repetitions(15)->ReportAggregatesOnly(true);

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

    template<bool bUseFast>
    void L1dist_perftest(benchmark::State& st) {
        const volatile size_t nArraySize = size_t(st.range(0));
        const volatile size_t nLoopSize = size_t(st.range(1));
        const volatile float fMinVal = (float)st.range(2);
        const volatile float fMaxVal = (float)st.range(3);
        const std::unique_ptr<float[]> afVals = genarray<float>(nArraySize,fMinVal,fMaxVal);
        size_t nArrayIdx = 0;
        while(st.KeepRunning()) {
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx = (nArrayIdx+2)%nCurrArraySize;
                volatile auto tLast = bUseFast?(lv::_L1dist_cheat(afVals[nArrayIdx],afVals[nArrayIdx+1])):(lv::_L1dist_nocheat(afVals[nArrayIdx],afVals[nArrayIdx+1]));
                benchmark::DoNotOptimize(tLast);
            }
        }
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
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nCurrArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nCurrArraySize;
                volatile auto tLast = lv::L1dist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
                benchmark::DoNotOptimize(tLast);
            }
        }
    }

}

BENCHMARK_TEMPLATE1(L1dist_perftest,true)->Args({1000000,250,-10,10})->Repetitions(15)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE1(L1dist_perftest,false)->Args({1000000,250,-10,10})->Repetitions(15)->ReportAggregatesOnly(true);
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
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nCurrArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nCurrArraySize;
                volatile auto tLast = lv::L2sqrdist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
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
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nCurrArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nCurrArraySize;
                volatile auto tLast = lv::L2dist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
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
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nCurrArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nCurrArraySize;
                volatile auto tLast = lv::cdist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
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
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx = (nArrayIdx+nChannels)%nCurrArraySize;
                volatile auto tLast = lv::popcount<nChannels>(&aVals[nArrayIdx]);
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
            const size_t nCurrLoopSize = nLoopSize;
            const size_t nCurrArraySize = nArraySize;
            for(size_t nLoopIdx=0; nLoopIdx<nCurrLoopSize; ++nLoopIdx) {
                nArrayIdx1 = (nArrayIdx1+nChannels)%nCurrArraySize;
                nArrayIdx2 = (nArrayIdx2+nChannels)%nCurrArraySize;
                volatile auto tLast = lv::hdist<nChannels>(&aVals[nArrayIdx1],&aVals[nArrayIdx2]);
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

TEST(extend_bits,regression) {
    EXPECT_EQ(uint32_t(lv::extend_bits(0b0u,5,8)),uint32_t(0b0));
    EXPECT_EQ(uint32_t(lv::extend_bits(0b111u,3,3)),uint32_t(0b111));
    EXPECT_EQ(uint32_t(lv::extend_bits(0b11111u,5,8)),uint32_t(0b11111111));
    EXPECT_NEAR(uint32_t(lv::extend_bits(0b10101u,5,8)),uint32_t(0b10101*256/32),256/32);
}

TEST(expand_bits,regression) {
    EXPECT_EQ(uint32_t(lv::expand_bits<4>(0)),uint32_t(0));
    EXPECT_EQ(uint32_t(lv::expand_bits<4>(0b1111)),uint32_t(0b0001000100010001));
    EXPECT_EQ(uint32_t(lv::expand_bits<4>(0b101010)),uint32_t(0b000100000001000000010000));
}