// note: this file is used to threshold an LBSP pattern based on a predefined
// lookup array (see LBSP_16bits_dbcross_lookup for more information).
//
// must be defined externally:
//      _t              (size_t, absolute threshold used for comparisons)
//      _ref            (uchar, reference value used for comparisons)
//      _res            (integer type, ouput descriptor bit vector)
//      _anVals         (std::array<uchar,N>, input pattern lookup array)
//       L1dist         (function, returns the absolute difference between two uchars)

CV_DbgAssert(_anVals.size()==2); // @@@ todo: use array template to unroll loops & allow any descriptor size here
#if (!HAVE_SSE4_1 && !HAVE_SSE2)
//_res = 0;
//for(size_t b=0; b<_anVals.size()*8; ++b)
//    _res |= (L1dist(_anVals[b],_ref)>_t)<<b;
_res = ((L1dist(_anVals[15],_ref) > _t) << 15) |
       ((L1dist(_anVals[14],_ref) > _t) << 14) |
       ((L1dist(_anVals[13],_ref) > _t) << 13) |
       ((L1dist(_anVals[12],_ref) > _t) << 12) |
       ((L1dist(_anVals[11],_ref) > _t) << 11) |
       ((L1dist(_anVals[10],_ref) > _t) << 10) |
       ((L1dist(_anVals[9],_ref) > _t) << 9)   |
       ((L1dist(_anVals[8],_ref) > _t) << 8)   |
       ((L1dist(_anVals[7],_ref) > _t) << 7)   |
       ((L1dist(_anVals[6],_ref) > _t) << 6)   |
       ((L1dist(_anVals[5],_ref) > _t) << 5)   |
       ((L1dist(_anVals[4],_ref) > _t) << 4)   |
       ((L1dist(_anVals[3],_ref) > _t) << 3)   |
       ((L1dist(_anVals[2],_ref) > _t) << 2)   |
       ((L1dist(_anVals[1],_ref) > _t) << 1)   |
       ((L1dist(_anVals[0],_ref) > _t));
#else //(HAVE_SSE4_1 || HAVE_SSE2)
CV_DbgAssert(((uintptr_t)(&_anVals[0])&15)==0);
__m128i _anMMXVals = _mm_load_si128((__m128i*)&_anVals[0]);
__m128i _anRefVals = _mm_set1_epi8(char(_ref));
#if HAVE_SSE4_1
__m128i _anDistVals = _mm_sub_epi8(_mm_max_epu8(_anMMXVals,_anRefVals),_mm_min_epu8(_anMMXVals,_anRefVals));
__m128i _abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_mm_set1_epi8(char(0x80))),_mm_set1_epi8(char(_t^0x80)));
#else //HAVE_SSE2
__m128i _anBitFlipper = _mm_set1_epi8(char(0x80));
__m128i _anDistVals = _mm_xor_si128(_anMMXVals,_anBitFlipper);
_anRefVals = _mm_xor_si128(_anRefVals,_anBitFlipper);
__m128i _abCmpRes = _mm_cmpgt_epi8(_anDistVals,_anRefVals);
__m128i _anDistVals1 = _mm_sub_epi8(_anDistVals,_anRefVals);
__m128i _anDistVals2 = _mm_sub_epi8(_anRefVals,_anDistVals);
_anDistVals = _mm_or_si128(_mm_and_si128(_abCmpRes,_anDistVals1),_mm_andnot_si128(_abCmpRes,_anDistVals2));
_abCmpRes = _mm_cmpgt_epi8(_mm_xor_si128(_anDistVals,_anBitFlipper),_mm_set1_epi8(char(_t^0x80)));
#endif //HAVE_SSE20x80u
_res = _mm_movemask_epi8(_abCmpRes);
#endif //(HAVE_SSE4_1 || HAVE_SSE2)
