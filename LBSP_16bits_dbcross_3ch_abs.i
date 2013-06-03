// note: this is the LBSP 16 bit double-cross indiv RGB pattern as used in
// the original article by G.-A. Bilodeau et al.
// 
//  O   O   O         15 .. 14 .. 13
//    O	O O           .. 12 11 10 ..
//  O O X O O    =>    9  8 ..  7  6
//    O O O           ..  5  4  3 .. 
//  O   O   O          2 ..  1 ..  0
//           3x                     3x
//
// must be defined externally:
//		_t			(uchar, absolute threshold used for comparisons)
//		_data		(uchar*, triple-channel data to be covered by the pattern)
//		_refdata	(uchar*, triple-channel data to be used for comparisons)
//		_y			(int, pattern rows location in the image data)
//		_x			(int, pattern cols location in the image data)
//		_step_row	(int, step size between rows, including padding)
//		_res		(ushort[3], 16 bit result vectors vector)

#if defined(_val) || defined(_absdiff)
#error "definitions clash detected"
#endif
#define _val(x,y,n) _data[_step_row*(_y+y)+3*(_x+x)+n]
#define _absdiff(a,b) (a<b?b-a:a-b)

for(int n=0; n<3; ++n) {
	const uchar _ref = _refdata[_step_row*(_y)+3*(_x)+n];
	_res[n] = ((_absdiff(_val(-2, 2, n),_ref) > _t) << 15)
			+ ((_absdiff(_val( 0, 2, n),_ref) > _t) << 14)
			+ ((_absdiff(_val( 2, 2, n),_ref) > _t) << 13)
			+ ((_absdiff(_val(-1, 1, n),_ref) > _t) << 12)
			+ ((_absdiff(_val( 0, 1, n),_ref) > _t) << 11)
			+ ((_absdiff(_val( 1, 1, n),_ref) > _t) << 10)
			+ ((_absdiff(_val(-2, 0, n),_ref) > _t) << 9)
			+ ((_absdiff(_val(-1, 0, n),_ref) > _t) << 8)
			+ ((_absdiff(_val( 1, 0, n),_ref) > _t) << 7)
			+ ((_absdiff(_val( 2, 0, n),_ref) > _t) << 6)
			+ ((_absdiff(_val(-1,-1, n),_ref) > _t) << 5)
			+ ((_absdiff(_val( 0,-1, n),_ref) > _t) << 4)
			+ ((_absdiff(_val( 1,-1, n),_ref) > _t) << 3)
			+ ((_absdiff(_val(-2,-2, n),_ref) > _t) << 2)
			+ ((_absdiff(_val( 0,-2, n),_ref) > _t) << 1)
			+ ((_absdiff(_val( 2,-2, n),_ref) > _t));
}

#undef _val
#undef _absdiff
		