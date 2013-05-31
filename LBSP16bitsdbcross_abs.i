// note: this is the LBSP 16 bit double-cross pattern as used in the original article by G.-A. Bilodeau et al
// 
//  O   O   O         15 .. 14 .. 13
//    O	O O           .. 12 11 10 ..
//  O O X O O    =>    9  8 ..  7  6
//    O O O           ..  5  4  3 .. 
//  O   O   O          2 ..  1 ..  0
//
// must be defined externally:
//		_t			(int, absolute threshold used for comparisons)
//		_ref		(uchar, reference value used for comparisons)
//		_data		(uchar*, image data to be covered by the pattern)
//		_y			(int, pattern rows location in the image data)
//		_x			(int, pattern cols location in the image data)
//		_step_row	(int, step size between rows, including padding)
//		_step_col	(int, step size between cols, including padding)
//		_res		(uint16, 16 bit result vector)

#if defined(_val) || defined(_absdiff)
#error "definitions clash detected"
#endif
#define _val(a,b) _data[_step_row*(_y+b)+_step_col*(_x+a)]
#define _absdiff(a,b) (int)(absdiff(a,b))
	    
_res= ((_absdiff(_val(-2, 2),_ref) < _t) << 15)
	+ ((_absdiff(_val( 0, 2),_ref) < _t) << 14)
	+ ((_absdiff(_val( 2, 2),_ref) < _t) << 13)
	+ ((_absdiff(_val(-1, 1),_ref) < _t) << 12)
	+ ((_absdiff(_val( 0, 1),_ref) < _t) << 11)
	+ ((_absdiff(_val( 1, 1),_ref) < _t) << 10)
	+ ((_absdiff(_val(-2, 0),_ref) < _t) << 9)
	+ ((_absdiff(_val(-1, 0),_ref) < _t) << 8)
	+ ((_absdiff(_val( 1, 0),_ref) < _t) << 7)
	+ ((_absdiff(_val( 2, 0),_ref) < _t) << 6)
	+ ((_absdiff(_val(-1,-1),_ref) < _t) << 5)
	+ ((_absdiff(_val( 0,-1),_ref) < _t) << 4)
	+ ((_absdiff(_val( 1,-1),_ref) < _t) << 3)
	+ ((_absdiff(_val(-2,-2),_ref) < _t) << 2)
	+ ((_absdiff(_val( 0,-2),_ref) < _t) << 1)
	+ ((_absdiff(_val( 2,-2),_ref) < _t));

#undef _val
#undef _absdiff
		