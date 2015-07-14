// note: this is the LBSP 16 bit double-cross indiv RGB/RGBA pattern as used in
// the original article by G.-A. Bilodeau et al.
// 
//  O   O   O          4 ..  3 ..  6
//    O O O           .. 15  8 13 ..
//  O O X O O    =>    0  9  X 11  1
//    O O O           .. 12 10 14 ..
//  O   O   O          7 ..  2 ..  5
//
// must be defined externally:
//      _c              (size_t, number of channels in the image data)
//      _t              (size_t, absolute threshold used for comparisons)
//      _ref            (uchar, 'central' value used for comparisons)
//      _data           (uchar*, image data to be covered by the pattern)
//      _y              (int, pattern rows location in the image data)
//      _x              (int, pattern cols location in the image data)
//      _resc           (size_t, pattern channel location in the image data)
//      _step_row       (size_t, step size between rows, including padding)
//      _res            (ushort, 16 bit result vector)
//       L1dist         (function, returns the absolute difference between two uchars)

#ifdef _val
#error "definitions clash detected"
#else
#define _val(x,y,n) _data[_step_row*(_y+y)+_c*(_x+x)+n]
#endif
#if (!HAVE_SSE4_1 && !HAVE_SSE2)
_res = ((L1dist(_val(-1, 1,_resc),_ref) > _t) << 15) |
       ((L1dist(_val( 1,-1,_resc),_ref) > _t) << 14) |
       ((L1dist(_val( 1, 1,_resc),_ref) > _t) << 13) |
       ((L1dist(_val(-1,-1,_resc),_ref) > _t) << 12) |
       ((L1dist(_val( 1, 0,_resc),_ref) > _t) << 11) |
       ((L1dist(_val( 0,-1,_resc),_ref) > _t) << 10) |
       ((L1dist(_val(-1, 0,_resc),_ref) > _t) << 9)  |
       ((L1dist(_val( 0, 1,_resc),_ref) > _t) << 8)  |
       ((L1dist(_val(-2,-2,_resc),_ref) > _t) << 7)  |
       ((L1dist(_val( 2, 2,_resc),_ref) > _t) << 6)  |
       ((L1dist(_val( 2,-2,_resc),_ref) > _t) << 5)  |
       ((L1dist(_val(-2, 2,_resc),_ref) > _t) << 4)  |
       ((L1dist(_val( 0, 2,_resc),_ref) > _t) << 3)  |
       ((L1dist(_val( 0,-2,_resc),_ref) > _t) << 2)  |
       ((L1dist(_val( 2, 0,_resc),_ref) > _t) << 1)  |
       ((L1dist(_val(-2, 0,_resc),_ref) > _t));
#else //(HAVE_SSE4_1 || HAVE_SSE2)
alignas(16) std::array<uchar,16> _anAlignedVals = {
    _val(-2, 0,_resc),
    _val( 2, 0,_resc),
    _val( 0,-2,_resc),
    _val( 0, 2,_resc),
    _val(-2, 2,_resc),
    _val( 2,-2,_resc),
    _val( 2, 2,_resc),
    _val(-2,-2,_resc),
    _val( 0, 1,_resc),
    _val(-1, 0,_resc),
    _val( 0,-1,_resc),
    _val( 1, 0,_resc),
    _val(-1,-1,_resc),
    _val( 1, 1,_resc),
    _val( 1,-1,_resc),
    _val(-1, 1,_resc),
};
__m128i _anVals = _mm_load_si128((__m128i*)&_anAlignedVals[0]);
__m128i _anRefVals = _mm_set1_epi8(_ref);
#if HAVE_SSE4_1
__m128i _anDistVals = _mm_sub_epi8(_mm_max_epu8(_anVals,_anRefVals),_mm_min_epu8(_anVals,_anRefVals));
__m128i _abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_mm_set1_epi8(0x80)),_mm_set1_epi8(_t^0x80));
#else //HAVE_SSE2
__m128i _anBitFlipper = _mm_set1_epi8(0x80);
__m128i _anDistVals = _mm_xor_si128(_anVals,_anBitFlipper);
_anRefVals = _mm_xor_si128(_anRefVals,_anBitFlipper);
__m128i _abCmpRes = _mm_cmpgt_epi8(_anDistVals,_anRefVals);
__m128i _anDistVals1 = _mm_sub_epi8(_anDistVals,_anRefVals);
__m128i _anDistVals2 = _mm_sub_epi8(_anRefVals,_anDistVals);
_anDistVals = _mm_or_si128(_mm_and_si128(_abCmpRes,_anDistVals1),_mm_andnot_si128(_abCmpRes,_anDistVals2));
_abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_anBitFlipper),_mm_set1_epi8(_t^0x80));
#endif //HAVE_SSE2
_res = _mm_movemask_epi8(_abCmpRes);
#endif //(HAVE_SSE4_1 || HAVE_SSE2)
#undef _val
