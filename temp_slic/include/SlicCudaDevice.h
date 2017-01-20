/*
This file gathers all the device kernel functions

author : Derue François-Xavier
francois.xavier.derue<at>gmail.com
*/


#pragma once

__global__ void kRgb2CIELab(const cudaTextureObject_t inputImg,
							cudaSurfaceObject_t outputImg,
							int width,
							int height);

__global__ void kInitClusters(const cudaSurfaceObject_t frameLab, 
								float* clusters, 
								int width, 
								int height, 
								int nSpxPerRow, 
								int nSpxPerCol);


__global__ void kAssignment(const cudaSurfaceObject_t frameLab,
							const float* clusters,
							const int width,
							const int height, 
							const int wSpx, 
							const int hSpx, 
							const float wc2,
							cudaSurfaceObject_t labels,
							float* accAtt_g);

__global__ void kUpdate(int nbSpx, float* clusters, float* accAtt_g);


__device__ inline float2 operator-(const float2 & a, const float2 & b) { return make_float2(a.x - b.x, a.y - b.y); }
__device__ inline float3 operator-(const float3 & a, const float3 & b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline int2 operator+(const int2 & a, const int2 & b) { return make_int2(a.x + b.x, a.y + b.y); }

__device__ inline float computeDistance(float2 c_p_xy, float3 c_p_Lab, float areaSpx, float wc2){
	float ds2 = pow(c_p_xy.x, 2) + pow(c_p_xy.y, 2);
	float dc2 = pow(c_p_Lab.x, 2) + pow(c_p_Lab.y, 2) + pow(c_p_Lab.z, 2);
	float dist = sqrt(dc2 + ds2 / areaSpx*wc2);

	return dist;
}

__device__ inline int convertIdx(int2 wg, int lc_idx, int nBloc_per_row){
	int2 relPos2D = make_int2(lc_idx % 5 - 2, lc_idx / 5 - 2);
	int2 glPos2D = wg + relPos2D;

	return glPos2D.y*nBloc_per_row + glPos2D.x;
}