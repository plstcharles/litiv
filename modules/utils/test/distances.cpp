
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
    std::unique_ptr<T[]> l1dist_genarray(size_t n, T min, T max) {
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
    void l1dist_perftest(benchmark::State& st) {
        const size_t nArraySize = size_t(st.range(0));
        const size_t nLoopSize = size_t(st.range(1));
        const T tMinVal = (T)st.range(2);
        const T tMaxVal = (T)st.range(3);
        std::unique_ptr<T[]> aVals = l1dist_genarray<T>(nArraySize,tMinVal,tMaxVal);
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
BENCHMARK_TEMPLATE(l1dist_perftest,double)->Args({1000000,100,-10000,10000})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(l1dist_perftest,float)->Args({1000000,100,-1000,1000})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(l1dist_perftest,short)->Args({1000000,100,-100,100})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(l1dist_perftest,char)->Args({1000000,100,-10,10})->Repetitions(10)->MinTime(1.0)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct l1dist_signed_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t,int32_t,float,double> l1dist_signed_types;
}
TYPED_TEST_CASE(l1dist_signed_fixture,l1dist_signed_types);
TYPED_TEST(l1dist_signed_fixture,regression) {
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

namespace {
    template<typename T>
    struct l1dist_unsigned_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t,uint32_t> l1dist_unsigned_types;
}
TYPED_TEST_CASE(l1dist_unsigned_fixture,l1dist_unsigned_types);
TYPED_TEST(l1dist_unsigned_fixture,regression) {
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
