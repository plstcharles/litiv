
#include "litiv/utils/simd.hpp"
#include "litiv/test.hpp"

#if HAVE_MMX

TEST(hsum_8ui,regression_32bit) {
    std::vector<uint8_t,lv::AlignedMemAllocator<uint8_t,16>> vData(8);
    ASSERT_EQ(((uintptr_t)vData.data()%16),uintptr_t(0));
    ASSERT_EQ(lv::hsum_8ui(*(__m64*)vData.data()),uint32_t(0));
    std::iota(vData.begin(),vData.end(),uint8_t(1));
    ASSERT_EQ(lv::hsum_8ui(*(__m64*)vData.data()),uint32_t((vData.size()*(vData.size()+1))/2));
    std::fill(vData.begin(),vData.end(),uint8_t(255));
    ASSERT_EQ(lv::hsum_8ui(*(__m64*)vData.data()),uint32_t(vData.size()*255));
    for(size_t i=0; i<1000; ++i) {
        for(size_t j=0; j<vData.size(); ++j)
            vData[j] = uint8_t(rand()%256);
        ASSERT_EQ(lv::hsum_8ui(*(__m64*)vData.data()),std::accumulate(vData.begin(),vData.end(),uint32_t(0)));
    }
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
            uData.n[i] = uint8_t(1ULL<<j);
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
    ASSERT_EQ(((uintptr_t)vData.data()%16),uintptr_t(0));
    ASSERT_TRUE(lv::cmp_eq_128i(_mm_load_si128((__m128i*)vData.data()),uData.a));
    ASSERT_TRUE(lv::cmp_eq_128i(*(__m128i*)vData.data(),uData.a));
}

TEST(hsum_8ui,regression) {
    std::vector<uint8_t,lv::AlignedMemAllocator<uint8_t,16>> vData(16);
    ASSERT_EQ(((uintptr_t)vData.data()%16),uintptr_t(0));
    ASSERT_EQ(lv::hsum_8ui(_mm_load_si128((__m128i*)vData.data())),uint32_t(0));
    std::iota(vData.begin(),vData.end(),uint8_t(1));
    ASSERT_EQ(lv::hsum_8ui(_mm_load_si128((__m128i*)vData.data())),uint32_t((vData.size()*(vData.size()+1))/2));
    std::fill(vData.begin(),vData.end(),uint8_t(255));
    ASSERT_EQ(lv::hsum_8ui(_mm_load_si128((__m128i*)vData.data())),uint32_t(vData.size()*255));
    for(size_t i=0; i<1000; ++i) {
        for(size_t j=0; j<vData.size(); ++j)
            vData[j] = uint8_t(rand()%256);
        ASSERT_EQ(lv::hsum_8ui(_mm_load_si128((__m128i*)vData.data())),std::accumulate(vData.begin(),vData.end(),uint32_t(0)));
    }
}

TEST(hsum_32ui,regression) {
    std::vector<uint32_t,lv::AlignedMemAllocator<uint32_t,16>> vData(4);
    ASSERT_EQ(((uintptr_t)vData.data()%16),uintptr_t(0));
    ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),int32_t(0));
    std::iota(vData.begin(),vData.end(),uint32_t(1));
    ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),int32_t((vData.size()*(vData.size()+1))/2));
    std::fill(vData.begin(),vData.end(),uint32_t(255));
    ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),int32_t(vData.size()*255));
    for(size_t i=0; i<1000; ++i) {
        for(size_t j=0; j<vData.size(); ++j)
            vData[j] = uint32_t(rand()%10000);
        ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),int32_t(std::accumulate(vData.begin(),vData.end(),uint32_t(0))));
    }
}

TEST(hsum_32si,regression) {
    std::vector<int32_t,lv::AlignedMemAllocator<int32_t,16>> vData(4);
    ASSERT_EQ(((uintptr_t)vData.data()%16),uintptr_t(0));
    ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),int32_t(0));
    std::iota(vData.begin(),vData.end(),int32_t(1));
    ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),int32_t((vData.size()*(vData.size()+1))/2));
    std::fill(vData.begin(),vData.end(),int32_t(255));
    ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),int32_t(vData.size()*255));
    for(size_t i=0; i<1000; ++i) {
        for(size_t j=0; j<vData.size(); ++j)
            vData[j] = rand()%10000 - 5000;
        ASSERT_EQ(lv::hsum_32i(_mm_load_si128((__m128i*)vData.data())),std::accumulate(vData.begin(),vData.end(),int32_t(0)));
    }
}

TEST(store1_8ui,regression) {
    std::vector<uint8_t,lv::AlignedMemAllocator<uint8_t,16>> vData(16);
    ASSERT_EQ(((uintptr_t)vData.data()%16),uintptr_t(0));
    for(uint8_t v=0; v<255; ++v) {
        lv::store1_8ui((__m128i*)vData.data(),uint8_t(v));
        ASSERT_TRUE(vData[0]==v && std::equal(vData.begin()+1,vData.end(),vData.begin()));
    }
}

TEST(store_8ui,regression) {
    for(size_t i=0; i<10; ++i) {
        std::vector<uint8_t,lv::AlignedMemAllocator<uint8_t,16>> vData((1ULL<<i)*size_t(16));
        ASSERT_EQ(((uintptr_t)vData.data()%16),uintptr_t(0));
        for(uint8_t v=0; v<255; ++v) {
            lv::store_8ui((__m128i*)vData.data(),vData.size()/16,uint8_t(v));
            ASSERT_TRUE(vData[0]==v && std::equal(vData.begin()+1,vData.end(),vData.begin()));
        }
    }
}

TEST(mult_32si,regression) {
    union {
        int32_t n[4];
        __m128i a;
    } a, b, c;
    a.n[0] = 65535; a.n[1] = -512; a.n[2] = 77910; a.n[3] = 0;
    b.n[0] = 2;     b.n[1] = 4431; b.n[2] = -7969; b.n[3] = 240000000;
    c.a = lv::mult_32si(a.a,b.a);
    ASSERT_EQ(c.n[0],131070);
    ASSERT_EQ(c.n[1],-2268672);
    ASSERT_EQ(c.n[2],-620864790);
    ASSERT_EQ(c.n[3],0);
}

TEST(extract_32si,regression) {
    union {
        int32_t n[4];
        __m128i a;
    } uData = {{1,2,3,4}};
    ASSERT_EQ(lv::extract_32si<0>(uData.a),1);
    ASSERT_EQ(lv::extract_32si<1>(uData.a),2);
    ASSERT_EQ(lv::extract_32si<2>(uData.a),3);
    ASSERT_EQ(lv::extract_32si<3>(uData.a),4);
}

#endif //HAVE_SSE2

#if HAVE_SSE4_1

TEST(hmax_8ui,regression) {
    union {
        uint8_t n[16];
        __m128i a;
    } uData = {0};
    ASSERT_EQ(lv::hmax_8ui(uData.a),0);
    for(size_t i=0; i<1000; ++i) {
        uint8_t nMax = (uData.n[0]=(uint8_t)(rand()%256));
        for(size_t j=1; j<16; ++j)
            nMax = std::max(nMax,(uData.n[j]=(uint8_t)(rand()%256)));
        ASSERT_EQ(lv::hmax_8ui(uData.a),nMax);
    }
}

TEST(hmin_8ui,regression) {
    union {
        uint8_t n[16];
        __m128i a;
    } uData = {0};
    ASSERT_EQ(lv::hmin_8ui(uData.a),0);
    for(size_t i=0; i<1000; ++i) {
        uint8_t nMin = (uData.n[0]=(uint8_t)(rand()%256));
        for(size_t j=1; j<16; ++j)
            nMin = std::min(nMin,(uData.n[j]=(uint8_t)(rand()%256)));
        ASSERT_EQ(lv::hmin_8ui(uData.a),nMin);
    }
}

TEST(hmax_32si,regression) {
    union {
        int32_t n[4];
        __m128i a;
    } uData = {0};
    ASSERT_EQ(lv::hmax_32si(uData.a),0);
    for(size_t i=0; i<1000; ++i) {
        int32_t nMax = (uData.n[0]=(int32_t)(rand()%1000-500));
        for(size_t j=1; j<4; ++j)
            nMax = std::max(nMax,(uData.n[j]=(int32_t)(rand()%1000-500)));
        ASSERT_EQ(lv::hmax_32si(uData.a),nMax);
    }
}

#endif //HAVE_SSE4_1