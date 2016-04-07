#include <iostream>
#include <opencv2\opencv.hpp>
#include "SLIC_cuda.h"
using namespace std;
using namespace cv;

int main() {
 
	VideoCapture cap("C:/Recherche_FX/video/TB_CVPR2013/tiger2/img/%04d.jpg"); //change with your videopath
	Mat frame;
	cap >> frame;

	//parameters
	int diamSpx = 8;
	int wc = 35;
	int nIteration = 5;
	SLIC_cuda::InitType initType = SLIC_cuda::SLIC_SIZE;

	//start segmentation
	SLIC_cuda slic_cuda;
	slic_cuda.Initialize(frame,diamSpx,wc,nIteration,initType);
    for(int i = 0; i<10; i++){
        auto start = cv::getTickCount();
		slic_cuda.Segment(frame);
        auto end = cv::getTickCount();
		cout << "runtime segmentation gpu " << (end - start) / cv::getTickFrequency() <<" s"<<endl;
		cv::Mat out = frame.clone();
		slic_cuda.displayBound(out, cv::Scalar(0, 0, 255));
		cv::imshow("out", out);
		waitKey(30);
		cap >> frame;
    }
	waitKey();
    return 0;
}