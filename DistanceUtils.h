#pragma once

#include <opencv2/core/types_c.h>

//! computes the absolute difference between two unsigned char values
static inline int absdiff_uchar(uchar a, uchar b) {
	return abs(a-b); // should return the same as (a<b?b-a:a-b), but faster when properly optimized
}

//! computes the L1 distance between two 3-ch unsigned char vectors
static inline int L1dist_uchar(uchar* a, uchar* b) {
	return absdiff_uchar(a[0],b[0])+absdiff_uchar(a[1],b[1])+absdiff_uchar(a[2],b[2]);
}

//! computes the squared L2 distance between two 3-ch unsigned char vectors
static inline int L2sqrdist_uchar(uchar* a, uchar* b) {
	return (absdiff_uchar(a[0],b[0])^2)+(absdiff_uchar(a[1],b[1])^2)+(absdiff_uchar(a[2],b[2])^2);
}

//! popcount LUT for 8bit vectors
static const uchar popcount_LUT8[256] = {
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

//! computes the population count of a 16 bits vector using an 8 bits popcount LUT
static inline uchar popcount_ushort_8bitsLUT(ushort x) {
	return popcount_LUT8[(uchar)x] + popcount_LUT8[(uchar)(x>>8)];
}

//! computes the hamming distance between two 16 bits vectors (min=0, max=16)
static inline uchar hdist_ushort_8bitLUT(ushort a, ushort b) {
	return popcount_ushort_8bitsLUT(a^b);
}


