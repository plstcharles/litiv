
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
    std::unique_ptr<T[]> l1dist_signedarray(size_t n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> uniform_dist(T(-10),T(10));
        std::unique_ptr<T[]> v(new T[n]);
        for(size_t i=0; i<n; ++i)
            v[i] = uniform_dist(gen);
        return std::move(v);
    }

    template<typename T>
    void l1dist_perftest(benchmark::State& st) {
        const size_t nArraySize = size_t(st.range(0));
        std::unique_ptr<T[]> aVals = l1dist_signedarray<T>(nArraySize);
        volatile T tLast = T(0);
        volatile size_t nIdx = 0;
        while(st.KeepRunning()) {
            ++nIdx %= nArraySize;
            tLast = lv::L1dist(aVals[nIdx],tLast)/2;
        }
        benchmark::DoNotOptimize(tLast);
    }

    template<typename T>
    struct l1dist_float_fixture : testing::Test {};
    typedef testing::Types<float,double> l1dist_float_types;
}

TYPED_TEST_CASE(l1dist_float_fixture,l1dist_float_types);
TYPED_TEST(l1dist_float_fixture,regression) {
    EXPECT_EQ((lv::L1dist(TypeParam(0),TypeParam(0))),float(0));
}

BENCHMARK_TEMPLATE(l1dist_perftest,float)->Arg(10)->Repetitions(2)->MinTime(5.0)->ReportAggregatesOnly(true);
BENCHMARK_TEMPLATE(l1dist_perftest,double)->Arg(10)->Repetitions(2)->MinTime(5.0)->ReportAggregatesOnly(true);

namespace {
    template<typename T>
    struct l1dist_int_fixture : testing::Test {};
    typedef testing::Types<int8_t,int16_t/*,int32_t,int64_t*/> l1dist_int_types;
}
TYPED_TEST_CASE(l1dist_int_fixture,l1dist_int_types);
TYPED_TEST(l1dist_int_fixture,regression) {
    EXPECT_EQ((lv::L1dist(TypeParam(0),TypeParam(0))),size_t(0));
}

namespace {
    template<typename T>
    struct l1dist_uint_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t/*,uint32_t,uint64_t*/> l1dist_uint_types;
}
TYPED_TEST_CASE(l1dist_uint_fixture,l1dist_uint_types);
TYPED_TEST(l1dist_uint_fixture,regression) {
    EXPECT_EQ((lv::L1dist(TypeParam(0),TypeParam(0))),size_t(0));
}