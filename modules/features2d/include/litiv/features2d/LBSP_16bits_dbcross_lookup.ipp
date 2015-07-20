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
//      _data           (uchar*, image data to be covered by the pattern)
//      _y              (int, pattern rows location in the image data)
//      _x              (int, pattern cols location in the image data)
//      _resc           (size_t, pattern channel location in the image data)
//      _step_row       (size_t, step size between rows, including padding)
//      _anVals         (std::array<uchar,16>, output pattern lookup array)

#ifdef _val
#error "definitions clash detected"
#else
#define _val(x,y,n) _data[_step_row*(_y+y)+_c*(_x+x)+n]
#endif
CV_DbgAssert(_anVals.size()==2);
_anVals = {
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
#undef _val
