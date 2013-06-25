// note: this is the LBSP 16 bit double-cross indiv RGB pattern as used in
// the original article by G.-A. Bilodeau et al.
// 
//  O   O   O          4 ..  3 ..  6
//    O O O           .. 15  8 13 ..
//  O O X O O    =>    0  9  X 11  1
//    O O O           .. 12 10 14 ..
//  O   O   O          7 ..  2 ..  5
//           3x                     3x
//
// must be defined externally:
//		_t				(float, relative threshold used for comparisons)
//		_data			(uchar*, triple-channel data to be covered by the pattern)
//		_refdata		(uchar*, triple-channel data to be used for comparisons)
//		_y				(int, pattern rows location in the image data)
//		_x				(int, pattern cols location in the image data)
//		_step_row		(int, step size between rows, including padding)
//		_res			(ushort[3], 16 bit result vectors vector)
//		absdiff_uchar	(function, returns the absolute difference between two uchars)

#ifdef _val
#error "definitions clash detected"
#else
#define _val(x,y,n) _data[_step_row*(_y+y)+3*(_x+x)+n]
#endif

for(int n=0; n<3; ++n) {
	const uchar _ref = _refdata[_step_row*(_y)+3*(_x)+n];
	const uchar _t_curr = (uchar)(_ref*_t);
	_res[n] = ((absdiff_uchar(_val(-1, 1, n),_ref) > _t_curr) << 15)
			+ ((absdiff_uchar(_val( 1,-1, n),_ref) > _t_curr) << 14)
			+ ((absdiff_uchar(_val( 1, 1, n),_ref) > _t_curr) << 13)
			+ ((absdiff_uchar(_val(-1,-1, n),_ref) > _t_curr) << 12)
			+ ((absdiff_uchar(_val( 1, 0, n),_ref) > _t_curr) << 11)
			+ ((absdiff_uchar(_val( 0,-1, n),_ref) > _t_curr) << 10)
			+ ((absdiff_uchar(_val(-1, 0, n),_ref) > _t_curr) << 9)
			+ ((absdiff_uchar(_val( 0, 1, n),_ref) > _t_curr) << 8)
			+ ((absdiff_uchar(_val(-2,-2, n),_ref) > _t_curr) << 7)
			+ ((absdiff_uchar(_val( 2, 2, n),_ref) > _t_curr) << 6)
			+ ((absdiff_uchar(_val( 2,-2, n),_ref) > _t_curr) << 5)
			+ ((absdiff_uchar(_val(-2, 2, n),_ref) > _t_curr) << 4)
			+ ((absdiff_uchar(_val( 0, 2, n),_ref) > _t_curr) << 3)
			+ ((absdiff_uchar(_val( 0,-2, n),_ref) > _t_curr) << 2)
			+ ((absdiff_uchar(_val( 2, 0, n),_ref) > _t_curr) << 1)
			+ ((absdiff_uchar(_val(-2, 0, n),_ref) > _t_curr));
}

#undef _val
		
