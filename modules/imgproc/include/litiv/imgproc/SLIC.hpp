/*
Superpixel oversegmentation
GPU implementation of the algorithm SLIC of
Achanta et al. [PAMI 2012, vol. 34, num. 11, pp. 2274-2282]

Library required :
Opencv 3.0 min
CUDA arch>=3.0

Author : Derue Fran?ois-Xavier
francois.xavier.derue@gmail.com


*/

#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

class SlicCuda {
public:
	enum InitType {
		SLIC_SIZE, // initialize with a size of spx
		SLIC_NSPX // initialize with a number of spx
	};
private:
	const int m_deviceId = 0;

	cudaDeviceProp m_deviceProp;

	int m_nbPx;
	int m_nbSpx;
	int m_SpxDiam;
	int m_SpxWidth, m_SpxHeight, m_SpxArea;
	int m_FrameWidth, m_FrameHeight;
	float m_wc;
	int m_nbIteration;
	InitType m_InitType;

	//cpu buffer
	float* h_fClusters;
	float* h_fLabels;

	// gpu variable
	float* d_fClusters;
	float* d_fLabels;
	float* d_fAccAtt;

	//cudaArray
	cudaArray* cuArrayFrameBGRA;
	cudaArray* cuArrayFrameLab;
	cudaArray* cuArrayLabels;

	// Texture and surface Object
	cudaTextureObject_t oTexFrameBGRA;
	cudaSurfaceObject_t oSurfFrameLab;
	cudaSurfaceObject_t oSurfLabels;

	//========= methods ===========

	void initGpuBuffers();
	void uploadFrame(const cv::Mat& frameBGR);
	void gpuRGBA2Lab();

	/*
	Initialize centroids uniformly on a grid with a step of diamSpx
	*/
	void gpuInitClusters();
	void downloadLabels();

	/*
	Assign the closest centroid to each pixel
	*/
	void assignment();

	/*
	Update the clusters' centroids with the belonging pixels
	*/
	void update();

public:
	SlicCuda();
	SlicCuda(const cv::Mat& frame0, const int diamSpxOrNbSpx = 15, const InitType initType = SLIC_SIZE, const float wc = 35, const int nbIteration = 5);
	~SlicCuda();

	/*
	Set up the parameters and initalize all gpu buffer for faster video segmentation.
	*/
	void initialize(const cv::Mat& frame0, const int diamSpxOrNbSpx = 15, const InitType initType = SLIC_SIZE, const float wc = 35, const int nbIteration = 5);

	/*
	Segment a frame in superpixel
	*/
	void segment(const cv::Mat& frame);
	cv::Mat getLabels(){ return cv::Mat(m_FrameHeight, m_FrameWidth, CV_32F, h_fLabels); }

	/*
	Discard orphan clusters (optional)
	*/
	int enforceConnectivity();

	static void displayBound(cv::Mat& image, const float* labels, const cv::Scalar colour); // cpu draw
};

static inline int iDivUp(int a, int b){ return (a%b == 0) ? a / b : a / b + 1; }

/*
Find best width and height from a given diameter to best fit the image size given by imWidth and imHeigh
*/
static void getSpxSizeFromDiam(const int imWidth, const int imHeight, const int diamSpx, int* spxWidth, int* spxHeight);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
