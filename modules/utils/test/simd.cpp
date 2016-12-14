
#include "litiv/utils/simd.hpp"
#include "common.hpp"

#if HAVE_MMX

TEST(hsum_8ub,regression) {
    // @@@@ todo & verify on 32bit build
}

#endif //HAVE_MMX

#if HAVE_SSE2

TEST(cmp_zero_128i,regression) {
    union {
        uint8_t n[16];
        __m128i a;
    } uData = {0};
    ASSERT_TRUE(lv::cmp_zero_128i(uData.a));
    uData.n[0] = 1;
    ASSERT_FALSE(lv::cmp_zero_128i(uData.a));
    for(size_t i=1; i<16; ++i) {
        uData.n[i-1] = 0;
        for(size_t j=0; j<8; ++j) {
            uData.n[i] = 1<<j;
            ASSERT_FALSE(lv::cmp_zero_128i(uData.a));
        }
    }
    constexpr __m128i anZeroBuffer1 = {0};
    ASSERT_TRUE(lv::cmp_zero_128i(anZeroBuffer1));
    const __m128i anZeroBuffer2 = _mm_setzero_si128();
    ASSERT_TRUE(lv::cmp_zero_128i(anZeroBuffer2));
}

TEST(cmp_eq_128i,regression) {
    union {
        uint8_t n[16];
        __m128i a;
    } uData = {0};
    constexpr __m128i anZeroBuffer1 = {0};
    ASSERT_TRUE(lv::cmp_eq_128i(uData.a,anZeroBuffer1));
    const __m128i anZeroBuffer2 = _mm_setzero_si128();
    ASSERT_TRUE(lv::cmp_eq_128i(uData.a,anZeroBuffer2));
    std::iota(uData.n,uData.n+16,0);
    ASSERT_FALSE(lv::cmp_eq_128i(uData.a,anZeroBuffer2));
    ASSERT_TRUE(lv::cmp_eq_128i(uData.a,_mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)));
}

namespace {
    template<typename T>
    struct aligned_cast_load_fixture : testing::Test {};
    typedef testing::Types<uint8_t,uint16_t,uint32_t,uint64_t,int8_t,int16_t,int32_t,int64_t,float,double> aligned_cast_load_types;
}
TYPED_TEST_CASE(aligned_cast_load_fixture,aligned_cast_load_types);
TYPED_TEST(aligned_cast_load_fixture,regression) {
    constexpr size_t nVals = sizeof(__m128i)/sizeof(TypeParam);
    std::vector<TypeParam,lv::AlignedMemAllocator<TypeParam,16>> vData(nVals);
    std::iota(vData.begin(),vData.end(),TypeParam(1));
    union {
        TypeParam n[nVals];
        __m128i a;
    } uData = {0};
    std::iota(uData.n,uData.n+nVals,TypeParam(1));
    ASSERT_TRUE(((uintptr_t)vData.data()%16)==0);
    ASSERT_TRUE(lv::cmp_eq_128i(_mm_load_si128((__m128i*)vData.data()),uData.a));
    ASSERT_TRUE(lv::cmp_eq_128i(*(__m128i*)vData.data(),uData.a));
}

TEST(hsum_16ub,regression) {
    
}

#endif //HAVE_SSE2

#if HAVE_SSE4_1


#endif //HAVE_SSE4_1