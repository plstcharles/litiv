#pragma once

#include <opencv2/core/types_c.h>

//! computes the absolute difference between two unsigned char values
static inline int absdiff(uchar a, uchar b) {
	return (a<b?b-a:a-b);
}

static inline int L2dist_sqr(uchar* a, uchar* b) {
	const int d0 = absdiff(a[0],b[0]);
	const int d1 = absdiff(a[1],b[1]);
	const int d2 = absdiff(a[2],b[2]);
	return d0*d0 + d1*d1 + d2*d2;
}

static inline int L1dist(uchar* a, uchar* b) {
	return absdiff(a[0],b[0])+absdiff(a[1],b[1])+absdiff(a[2],b[2]);
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
static inline uchar popcount_8bitsLUT(ushort x) {
	return popcount_LUT8[(uchar)x] + popcount_LUT8[(uchar)(x>>8)];
}

//! computes the hamming distance between two 16 bits vectors (min=0, max=16)
static inline uchar hdist_ushort_8bitLUT(ushort a, ushort b) {
	return popcount_8bitsLUT(a^b);
}


