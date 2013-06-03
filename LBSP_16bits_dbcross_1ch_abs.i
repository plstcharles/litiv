// note: this is the LBSP 16 bit double-cross single channel pattern as used in
// the original article by G.-A. Bilodeau et al.
// 
//  O   O   O         15 ..  8 .. 13
//    O O O           ..  4  3  6 ..
//  O O X O O    =>    9  0  X  1 11
//    O O O           ..  7  2  5 ..
//  O   O   O         12 .. 10 .. 14
//
// must be defined externally:
//		_t			(uchar, absolute threshold used for comparisons)
//		_data		(uchar*, single-channel data to be covered by the pattern)
//		_refdata	(uchar*, single-channel data to be used for comparisons)
//		_y			(int, pattern rows location in the image data)
//		_x			(int, pattern cols location in the image data)
//		_step_row	(int, step size between rows, including padding)
//		_res		(ushort, 16 bit result vector)

#if defined(_val) || defined(_absdiff)
#error "definitions clash detected"
#endif
#define _val(x,y) _data[_step_row*(_y+y)+_x+x]
#define _absdiff(a,b) (a<b?b-a:a-b)

const uchar _ref = _refdata[_step_row*(_y)+_x];
_res= ((_absdiff(_val(-2, 2),_ref) > _t) << 15)
	+ ((_absdiff(_val( 2,-2),_ref) > _t) << 14)
	+ ((_absdiff(_val( 2, 2),_ref) > _t) << 13)
	+ ((_absdiff(_val(-2,-2),_ref) > _t) << 12)
	+ ((_absdiff(_val( 2, 0),_ref) > _t) << 11)
	+ ((_absdiff(_val( 0,-2),_ref) > _t) << 10)
	+ ((_absdiff(_val(-2, 0),_ref) > _t) << 9)
	+ ((_absdiff(_val( 0, 2),_ref) > _t) << 8)
	+ ((_absdiff(_val(-1,-1),_ref) > _t) << 7)
	+ ((_absdiff(_val( 1, 1),_ref) > _t) << 6)
	+ ((_absdiff(_val( 1,-1),_ref) > _t) << 5)
	+ ((_absdiff(_val(-1, 1),_ref) > _t) << 4)
	+ ((_absdiff(_val( 0, 1),_ref) > _t) << 3)
	+ ((_absdiff(_val( 0,-1),_ref) > _t) << 2)
	+ ((_absdiff(_val( 1, 0),_ref) > _t) << 1)
	+ ((_absdiff(_val(-1, 0),_ref) > _t));

#undef _val
#undef _absdiff
		