#include "litiv/imgproc/SLIC.hpp"
#include "SLIC_device.hpp"

using namespace std;
using namespace cv;


SlicCuda::SlicCuda(){
	int nbGpu = 0;
	gpuErrchk(cudaGetDeviceCount(&nbGpu));
	cout << "Detected " << nbGpu << " cuda capable gpu" << endl;
	gpuErrchk(cudaSetDevice(m_deviceId));
	gpuErrchk(cudaGetDeviceProperties(&m_deviceProp, m_deviceId));
	if (m_deviceProp.major < 3){
		cerr << "compute capability found = " << m_deviceProp.major << ", compute capability >= 3 required !" << endl;
		exit(EXIT_FAILURE);
	}
}

SlicCuda::~SlicCuda(){
	delete[] h_fClusters;
	delete[] h_fLabels;
	gpuErrchk(cudaFree(d_fClusters));
	gpuErrchk(cudaFree(d_fAccAtt));
	gpuErrchk(cudaFreeArray(cuArrayFrameBGRA));
	gpuErrchk(cudaFreeArray(cuArrayFrameLab));
	gpuErrchk(cudaFreeArray(cuArrayLabels));
}

void SlicCuda::initialize(const cv::Mat& frame0, const int diamSpxOrNbSpx , const InitType initType, const float wc , const int nbIteration ) {
	m_nbIteration = nbIteration;
	m_FrameWidth = frame0.cols;
	m_FrameHeight = frame0.rows;
	m_nbPx = m_FrameWidth*m_FrameHeight;
	m_InitType = initType;
	m_wc = wc;
	if (m_InitType == SLIC_NSPX){
		m_SpxDiam = diamSpxOrNbSpx;
		m_SpxDiam = (int)sqrt(m_nbPx / (float)diamSpxOrNbSpx);
	}
	else m_SpxDiam = diamSpxOrNbSpx;

	getSpxSizeFromDiam(m_FrameWidth, m_FrameHeight, m_SpxDiam, &m_SpxWidth, &m_SpxHeight); // determine w and h of Spx based on diamSpx
	m_SpxArea = m_SpxWidth*m_SpxHeight;
	CV_Assert(m_nbPx%m_SpxArea == 0);
	m_nbSpx = m_nbPx / m_SpxArea;

	h_fClusters = new float[m_nbSpx * 5]; // m_nbSpx * [L,a,b,x,y]
	h_fLabels = new float[m_nbPx];

	initGpuBuffers();
}

void SlicCuda::segment(const Mat& frameBGR) {
	uploadFrame(frameBGR);
	gpuRGBA2Lab();
	gpuInitClusters();

	for (int i = 0; i<m_nbIteration; i++) {
		assignment();
		cudaDeviceSynchronize();
		update();
		cudaDeviceSynchronize();
	}
	downloadLabels();
}

void SlicCuda::initGpuBuffers() {
	//allocate buffers on gpu

	cudaChannelFormatDesc channelDescrBGRA = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	gpuErrchk(cudaMallocArray(&cuArrayFrameBGRA, &channelDescrBGRA, m_FrameWidth, m_FrameHeight));

	cudaChannelFormatDesc channelDescrLab = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	gpuErrchk(cudaMallocArray(&cuArrayFrameLab, &channelDescrLab, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));

	cudaChannelFormatDesc channelDescrLabels = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	gpuErrchk(cudaMallocArray(&cuArrayLabels, &channelDescrLabels, m_FrameWidth, m_FrameHeight, cudaArraySurfaceLoadStore));

	// Specify texture frameBGRA object parameters
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArrayFrameBGRA;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;
	gpuErrchk(cudaCreateTextureObject(&oTexFrameBGRA, &resDesc, &texDesc, NULL));

	// surface frameLab
	cudaResourceDesc rescDescFrameLab;
	memset(&rescDescFrameLab, 0, sizeof(rescDescFrameLab));
	rescDescFrameLab.resType = cudaResourceTypeArray;

	rescDescFrameLab.res.array.array = cuArrayFrameLab;
	gpuErrchk(cudaCreateSurfaceObject(&oSurfFrameLab, &rescDescFrameLab));

	// surface labels
	cudaResourceDesc resDescLabels;
	memset(&resDescLabels, 0, sizeof(resDescLabels));
	resDescLabels.resType = cudaResourceTypeArray;

	resDescLabels.res.array.array = cuArrayLabels;
	gpuErrchk(cudaCreateSurfaceObject(&oSurfLabels, &resDescLabels));

	// buffers clusters , accAtt
	gpuErrchk(cudaMalloc((void**)&d_fClusters, m_nbSpx*sizeof(float) * 5)); // 5-D centroid
	gpuErrchk(cudaMalloc((void**)&d_fAccAtt, m_nbSpx*sizeof(float) * 6)); // 5-D centroid acc + 1 counter
	cudaMemset(d_fAccAtt, 0, m_nbSpx*sizeof(float) * 6);//initialize accAtt to 0

}


void SlicCuda::uploadFrame(const Mat& frameBGR) {
	cv::Mat frameBGRA;
	cv::cvtColor(frameBGR, frameBGRA, CV_BGR2BGRA);
	CV_Assert(frameBGRA.type() == CV_8UC4);
	CV_Assert(frameBGRA.isContinuous());
	gpuErrchk(cudaMemcpyToArray(cuArrayFrameBGRA, 0, 0, (uchar*)frameBGRA.data, m_nbPx* sizeof(uchar4), cudaMemcpyHostToDevice));


	/*uchar* dst = new uchar[4 * m_nbPx];
	cudaMemcpyFromArray(dst, cuArrayFrameBGRA, 0, 0, m_nbPx*sizeof(uchar4), cudaMemcpyDeviceToHost);
	Mat matDst(m_FrameHeight, m_FrameWidth, CV_8UC4, dst);
	cout << matDst << endl;*/
}

void SlicCuda::gpuRGBA2Lab() {
	const int blockW = 16;
	const int blockH = blockW;
	CV_Assert(blockW*blockH <= m_deviceProp.maxThreadsPerBlock);
	dim3 threadsPerBlock(blockW, blockH);
	dim3 numBlocks(iDivUp(m_FrameWidth, blockW), iDivUp(m_FrameHeight, blockH));

	kRgb2CIELab << <numBlocks, threadsPerBlock >> >(oTexFrameBGRA, oSurfFrameLab, m_FrameWidth, m_FrameHeight);

	/*float* dst = new float[4 * m_nbPx];
	cudaMemcpyFromArray(dst, cuArrayFrameLab, 0, 0, m_nbPx*sizeof(float4), cudaMemcpyDeviceToHost);
	Mat matDst(m_FrameHeight, m_FrameWidth, CV_32FC4, dst);
	cout << matDst << endl;*/
}



void SlicCuda::gpuInitClusters() {
	int blockW = 16;
	dim3 threadsPerBlock(blockW);
	dim3 numBlocks(iDivUp(m_nbSpx, blockW));

	kInitClusters << <numBlocks, threadsPerBlock >> >(oSurfFrameLab,
		d_fClusters,
		m_FrameWidth,
		m_FrameHeight,
		m_FrameWidth / m_SpxWidth,
		m_FrameHeight / m_SpxHeight);

	/*float* fTmp = new float[m_nbSpx * 5];
	cudaMemcpy(fTmp, d_fClusters, m_nbSpx * 5 * sizeof(float), cudaMemcpyDeviceToHost);
	Mat matTmp(1, m_nbSpx*5, CV_32F, fTmp);
	cout << matTmp << endl;*/
}

void SlicCuda::assignment(){
	int hMax = m_deviceProp.maxThreadsPerBlock / m_SpxHeight;
	int nBlockPerClust = iDivUp(m_SpxHeight, hMax);

	dim3 blockPerGrid(m_nbSpx, nBlockPerClust);
	dim3 threadPerBlock(m_SpxWidth, std::min(m_SpxHeight, hMax));

	CV_Assert(threadPerBlock.x >= 3 && threadPerBlock.y >= 3);

	float wc2 = m_wc * m_wc;

	kAssignment << < blockPerGrid, threadPerBlock >> >(oSurfFrameLab,
		d_fClusters,
		m_FrameWidth,
		m_FrameHeight,
		m_SpxWidth,
		m_SpxHeight,
		wc2,
		oSurfLabels,
		d_fAccAtt);
}

void SlicCuda::update(){
	dim3 threadsPerBlock(m_deviceProp.maxThreadsPerBlock);
	dim3 numBlocks(iDivUp(m_nbSpx, m_deviceProp.maxThreadsPerBlock));
	kUpdate << <numBlocks, threadsPerBlock >> >(m_nbSpx, d_fClusters, d_fAccAtt);
}

void SlicCuda::downloadLabels(){
	cudaMemcpyFromArray(h_fLabels, cuArrayLabels, 0, 0, m_nbPx* sizeof(float), cudaMemcpyDeviceToHost);
}

int SlicCuda::enforceConnectivity() {
	int label = 0, adjlabel = 0;
	int lims = (m_FrameWidth * m_FrameHeight) / (m_nbSpx);
	lims = lims >> 2;

	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	vector<vector<int> >newLabels;
	for (int i = 0; i < m_FrameHeight; i++) {
		vector<int> nv(m_FrameWidth, -1);
		newLabels.push_back(nv);
	}

	for (int i = 0; i < m_FrameHeight; i++) {
		for (int j = 0; j < m_FrameWidth; j++){
			if (newLabels[i][j] == -1){
				vector<cv::Point> elements;
				elements.push_back(cv::Point(j, i));
				for (int k = 0; k < 4; k++){
					int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
					if (x >= 0 && x < m_FrameWidth && y >= 0 && y < m_FrameHeight){
						if (newLabels[y][x] >= 0){
							adjlabel = newLabels[y][x];
						}
					}
				}
				int count = 1;
				for (int c = 0; c < count; c++){
					for (int k = 0; k < 4; k++){
						int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
						if (x >= 0 && x < m_FrameWidth && y >= 0 && y < m_FrameHeight){
							if (newLabels[y][x] == -1 && h_fLabels[i*m_FrameWidth + j] == h_fLabels[y*m_FrameWidth + x]){
								elements.push_back(cv::Point(x, y));
								newLabels[y][x] = label;//m_labels[i][j];
								count += 1;
							}
						}
					}
				}
				if (count <= lims) {
					for (int c = 0; c < count; c++) {
						newLabels[elements[c].y][elements[c].x] = adjlabel;
					}
					label -= 1;
				}
				label += 1;
			}
		}
	}
	int nbSpxNoOrphan = label; // new number of spx
	for (int i = 0; i < newLabels.size(); i++)
		for (int j = 0; j < newLabels[i].size(); j++)
			h_fLabels[i*m_FrameWidth + j] = (float)newLabels[i][j];

	return nbSpxNoOrphan;
}


void SlicCuda::displayBound(cv::Mat& image, const float* labels, const cv::Scalar colour){
	const int dx8[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	/* Initialize the contour vector and the matrix detailing whether a pixel
	* is already taken to be a contour. */
	vector<cv::Point> contours;
	vector<vector<bool> > istaken;
	for (int i = 0; i < image.rows; i++) {
		vector<bool> nb;
		for (int j = 0; j < image.cols; j++) {
			nb.push_back(false);
		}
		istaken.push_back(nb);
	}
	// Go through all the pixels.
	for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			int nr_p = 0;
			// Compare the pixel to its 8 neighbours.
			for (int k = 0; k < 8; k++) {
				int x = j + dx8[k], y = i + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[y][x] == false && labels[i*image.cols + j] != labels[y*image.cols + x]) {
						nr_p += 1;
					}
				}
			}
			/* Add the pixel to the contour list if desired. */
			if (nr_p >= 2) {
				contours.push_back(cv::Point(j, i));
				istaken[i][j] = true;
			}

		}
	}
	// Draw the contour pixels.
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<cv::Vec3b>(contours[i].y, contours[i].x) = cv::Vec3b((uchar)colour[0], (uchar)colour[1], (uchar)colour[2]);
	}
}
static void getSpxSizeFromDiam(const int imWidth, const int imHeight, const int diamSpx, int* spxWidth, int* spxHeight){
	int wl1, wl2;
	int hl1, hl2;
	wl1 = wl2 = diamSpx;
	hl1 = hl2 = diamSpx;

	while ((imWidth%wl1) != 0) {
		wl1++;
	}
	while ((imWidth%wl2) != 0) {
		wl2--;
	}
	while ((imHeight%hl1) != 0) {
		hl1++;
	}

	while ((imHeight%hl2) != 0) {
		hl2--;
	}
	*spxWidth = ((diamSpx - wl2) < (wl1 - diamSpx)) ? wl2 : wl1;
	*spxHeight = ((diamSpx - hl2) < (hl1 - diamSpx)) ? hl2 : hl1;
}