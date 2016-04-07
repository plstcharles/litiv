//
// Created by derue on 15/12/15.
//
#pragma once
#ifndef SLIC_CUDA_SLIC_CUDA_H
#define SLIC_CUDA_SLIC_CUDA_H
#endif //SLIC_CUDA_SLIC_CUDA_H

/*
* Written by Derue Francois-Xavier
* francois-xavier.derue@polymtl.ca
*
* Superpixel oversegmentation
* GPU implementation of the algorithm SLIC of
* Achanta et al. [PAMI 2012, vol. 34, num. 11, pp. 2274-2282]
*
* Library required :
* CUDA
*/

#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "funUtilsSC.h"



#define NMAX_THREAD 256 // depend on gpu

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


class SLIC_cuda {
public:
	static enum InitType{
		SLIC_SIZE,
		SLIC_NSPX
	};
private:
	int m_nPx;
	int m_nSpx;
	int m_diamSpx;
	int m_wSpx, m_hSpx, m_areaSpx;
	int m_width, m_height;
	float m_wc;
	int m_nIteration;
	InitType m_initType;

	//cpu buffer
	float *m_clusters;
	float *m_labels;

	// gpu variable
	// uchar4* frameBGRA_g;
	//float4* frameLab_g;
	float* labels_g;
	float* clusters_g;
	float* accAtt_g;

	//cudaArray
	cudaArray* frameBGRA_array;
	cudaArray* frameLab_array;
	cudaArray* labels_array;

	//texture object
#if __CUDA_ARCH__>=300
	cudaTextureObject_t frameBGRA_tex;
	cudaSurfaceObject_t frameLab_surf;
	cudaSurfaceObject_t labels_surf;
#endif

	//========= methods ===========
	// init centroids uniformly on a grid spaced by diamSpx
	void InitClusters();
	//=subroutine =
	void InitBuffers(); // allocate buffers on gpu
	void SendFrame(cv::Mat& frameLab); //transfer frame to gpu buffer
	void getLabelsFromGpu();
#if __CUDA_ARCH__>=300
	void Rgb2CIELab(cudaTextureObject_t inputImg, cudaSurfaceObject_t outputImg, int width, int height);
#else
	void Rgb2CIELab(int width, int height);
#endif
	//===== Kernel Invocation ======
	void Assignement(); //Assignment
	void Update(); // Update

public:
	SLIC_cuda(){}
	~SLIC_cuda();

	void Initialize(cv::Mat& frame0, int diamSpx_or_Nspx, float wc, int nIteration = 5, InitType initType = SLIC_SIZE);
	void Segment(cv::Mat& frame); // gpu superpixel segmentation
	int getNspx(){ return m_nSpx; }
	cv::Mat getLabels(){ return cv::Mat(m_height, m_width, CV_32F, m_labels); }

	// enforce connectivity between superpixel, discard orphan (optional)
	// implementation from Pascal Mettes : https://github.com/PSMM/SLIC-Superpixels
	void enforceConnectivity();
	//===== Display function =====
	void displayBound(cv::Mat& image, cv::Scalar colour); // cpu draw


};
#if __CUDA_ARCH__>=300
__global__ void k_initClusters(cudaSurfaceObject_t frameLab, float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol);
__global__ void k_assignement(int width, int height, int wSpx, int hSpx, cudaSurfaceObject_t frameLab, cudaSurfaceObject_t labels, float* clusters, float* accAtt, float wc2);
#else
__global__ void k_initClusters(float* clusters, int width, int height, int nSpxPerRow, int nSpxPerCol);
__global__ void k_assignement(int width, int height, int wSpx, int hSpx, float* clusters, float* accAtt, float wc2);
#endif
__global__ void k_update(int nSpx, float* clusters, float* accAtt);
