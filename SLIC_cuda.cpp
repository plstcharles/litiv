//
// Created by derue on 15/12/15.
//

#include <driver_types.h>
#include "SLIC_cuda.h"

using namespace std;

SLIC_cuda::~SLIC_cuda(){
	delete[] m_clusters;
	delete[] m_labels;
	gpuErrchk(cudaFree(clusters_g));
	gpuErrchk(cudaFree(accAtt_g));
	gpuErrchk(cudaFreeArray(frameBGRA_array));
	gpuErrchk(cudaFreeArray(frameLab_array));
	gpuErrchk(cudaFreeArray(labels_array));
}
void SLIC_cuda::Initialize(cv::Mat &frame0, int diamSpx_or_Nspx, float wc, int nIteration, SLIC_cuda::InitType initType) {
	m_nIteration = nIteration;
	m_width = frame0.cols;
	m_height = frame0.rows;
	m_nPx = m_width*m_height;
	m_initType = initType;
	m_wc = wc;
	if (m_initType == SLIC_NSPX){
		m_diamSpx = diamSpx_or_Nspx; m_diamSpx = sqrt(m_nPx / (float)diamSpx_or_Nspx);
	}
	else m_diamSpx = diamSpx_or_Nspx;
	getWlHl(m_width, m_height, m_diamSpx, m_wSpx, m_hSpx); // determine w and h of Spx based on diamSpx
	m_areaSpx = m_wSpx*m_hSpx;
	CV_Assert(m_nPx%m_areaSpx == 0);
	m_nSpx = m_nPx / m_areaSpx; // should be an integer!!

	m_clusters = new float[m_nSpx * 5];
	m_labels = new float[m_nPx];

	InitBuffers();
}
void SLIC_cuda::Segment(cv::Mat &frame) {
	m_nSpx = m_nPx / m_areaSpx;//reinit m_nSpx because of enforceConnectivity
	SendFrame(frame); //ok
	InitClusters();//ok

	for (int i = 0; i<m_nIteration; i++) {
		Assignement();
		cudaDeviceSynchronize();
		Update();
		cudaDeviceSynchronize();
	}
	getLabelsFromGpu();
	enforceConnectivity();
}

void SLIC_cuda::SendFrame(cv::Mat& frameBGR){

	cv::Mat frameBGRA;
	cv::cvtColor(frameBGR, frameBGRA, CV_BGR2BGRA);
	CV_Assert(frameBGRA.type() == CV_8UC4);
	CV_Assert(frameBGRA.isContinuous());

	cudaMemcpyToArray(frameBGRA_array, 0, 0, (uchar*)frameBGRA.data, m_nPx* sizeof(uchar4), cudaMemcpyHostToDevice); //ok
#if __CUDA_ARCH__>=300
	Rgb2CIELab(frameBGRA_tex, frameLab_surf, m_width, m_height); //BGR->Lab gpu
#else
	Rgb2CIELab(m_width, m_height); //BGR->Lab gpu
#endif
}

void SLIC_cuda::getLabelsFromGpu()
{
	cudaMemcpyFromArray(m_labels, labels_array, 0, 0, m_nPx* sizeof(float), cudaMemcpyDeviceToHost);
}

void SLIC_cuda::enforceConnectivity()
{
	int label = 0, adjlabel = 0;
	int lims = (m_width * m_height) / (m_nSpx);
	lims = lims >> 2;


	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };

	vector<vector<int> >newLabels;
	for (int i = 0; i < m_height; i++)
	{
		vector<int> nv(m_width, -1);
		newLabels.push_back(nv);
	}

	for (int i = 0; i < m_height; i++)
	{
		for (int j = 0; j < m_width; j++)
		{
			if (newLabels[i][j] == -1)
			{
				vector<cv::Point> elements;
				elements.push_back(cv::Point(j, i));
				for (int k = 0; k < 4; k++)
				{
					int x = elements[0].x + dx4[k], y = elements[0].y + dy4[k];
					if (x >= 0 && x < m_width && y >= 0 && y < m_height)
					{
						if (newLabels[y][x] >= 0)
						{
							adjlabel = newLabels[y][x];
						}
					}
				}
				int count = 1;
				for (int c = 0; c < count; c++)
				{
					for (int k = 0; k < 4; k++)
					{
						int x = elements[c].x + dx4[k], y = elements[c].y + dy4[k];
						if (x >= 0 && x < m_width && y >= 0 && y < m_height)
						{
							if (newLabels[y][x] == -1 && m_labels[i*m_width + j] == m_labels[y*m_width + x])
							{
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
	m_nSpx = label;
	for (int i = 0; i < newLabels.size(); i++)
	for (int j = 0; j < newLabels[i].size(); j++)
		m_labels[i*m_width + j] = newLabels[i][j];
}

void SLIC_cuda::displayBound(cv::Mat &image, cv::Scalar colour)
{
	//load label from gpu

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

	/* Go through all the pixels. */

	for (int i = 0; i<image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			int nr_p = 0;

			/* Compare the pixel to its 8 neighbours. */
			for (int k = 0; k < 8; k++) {
				int x = j + dx8[k], y = i + dy8[k];

				if (x >= 0 && x < image.cols && y >= 0 && y < image.rows) {
					if (istaken[y][x] == false && m_labels[i*m_width + j] != m_labels[y*m_width + x]) {
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

	/* Draw the contour pixels. */
	for (int i = 0; i < (int)contours.size(); i++) {
		image.at<cv::Vec3b>(contours[i].y, contours[i].x) = cv::Vec3b(colour[0], colour[1], colour[2]);
	}
}


