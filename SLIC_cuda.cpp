//
// Created by derue on 15/12/15.
//

#include "SLIC_cuda.h"

using namespace std;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


SLIC_cuda::SLIC_cuda(int diamSpx, float wc){
    m_diamSpx = diamSpx;
    m_wc = wc;
}
SLIC_cuda::~SLIC_cuda(){
    delete[] m_clusters;
    delete[] m_labels;
}
void SLIC_cuda::Initialize(cv::Mat &frame0) {
    m_width = frame0.cols;
    m_height = frame0.rows;
    m_nPx = m_width*m_height;
    getWlHl(m_width, m_height, m_diamSpx, m_wSpx, m_hSpx); // determine w and h of Spx based on diamSpx
    m_areaSpx = m_wSpx*m_hSpx;
    CV_Assert(m_nPx%m_areaSpx==0);
    m_nSpx = m_nPx/m_areaSpx; // should be an integer!!

    m_clusters = new float[m_nSpx * 5];
    m_labels = new float[m_nPx];

    InitBuffers();


}
void SLIC_cuda::Segment(cv::Mat &frame) {

    SendFrame(frame); //ok
    cudaDeviceSynchronize();
    InitClusters();//ok
    cudaDeviceSynchronize();

    int nIt = 1;
    for(int i=0; i<nIt; i++) {
        //auto start = cv::getTickCount();

        Assignement();
        cudaDeviceSynchronize();
        //auto end = cv::getTickCount();
        //cout<<"runtime gpu "<<(end-start)/cv::getTickFrequency()<<endl;
        Update();
        cudaDeviceSynchronize();
    }

    float* test = new float[m_nPx];
    cv::Mat checkFrame(frame.size(),CV_32F,test);
    gpuErrchk(cudaMemcpy((float*)checkFrame.data,labels_g,m_nPx* sizeof(float),cudaMemcpyDeviceToHost));

    //cout<<checkFrame<<endl;



}

void SLIC_cuda::InitBuffers() {

    //allocate buffers on gpu
    gpuErrchk(cudaMalloc((void**)&frameBGRA_g, m_nPx*sizeof(uchar4))); //4 channels for padding
    gpuErrchk(cudaMalloc((void**)&frameLab_g, m_nPx*sizeof(float4))); //4 channels for padding
    gpuErrchk(cudaMalloc((void**)&labels_g, m_nPx*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&clusters_g, m_nSpx*sizeof(float)*5)); // 5-D centroid
    gpuErrchk(cudaMalloc((void**)&accAtt_g, m_nSpx*sizeof(float)*6)); // 5-D centroid acc + 1 counter
    cudaMemset(accAtt_g, 0, m_nSpx*sizeof(float)*6);//initialize accAtt to 0

}



void SLIC_cuda::SendFrame(cv::Mat& frameBGR){

    cv::Mat frameBGRA;
    cv::cvtColor(frameBGR,frameBGRA,CV_BGR2BGRA);
    CV_Assert(frameBGRA.type()==CV_8UC4);
    CV_Assert(frameBGRA.isContinuous());
    gpuErrchk(cudaMemcpy(frameBGRA_g, (float*)frameBGRA.data, frameBGRA.rows*frameBGRA.cols*frameBGRA.channels()*sizeof(uchar), cudaMemcpyHostToDevice));
    Rgb2CIELab(frameBGRA_g,frameLab_g,m_width,m_height);


    //cv::Mat checkFrame(frameBGR.size(),CV_32FC4,cv::Scalar(0,0,0,0));
    //gpuErrchk(cudaMemcpy((float*)checkFrame.data,frameLab_g,m_nPx* sizeof(float4),cudaMemcpyDeviceToHost));

    //cout<<checkFrame<<endl;
}


void SLIC_cuda::displayBound(cv::Mat &image, cv::Scalar colour)
{
    //load label from gpu
    gpuErrchk(cudaMemcpy(m_labels,labels_g,m_nPx*sizeof(float),cudaMemcpyDeviceToHost));

    const int dx8[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1 };

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
                    if (istaken[y][x] == false && m_labels[i*m_width+j] != m_labels[y*m_width+x]) {
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


