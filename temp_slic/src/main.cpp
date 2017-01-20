#include <iostream>
#include <opencv2\opencv.hpp>
#include "SlicCudaHost.h"


using namespace std;
using namespace cv;

int main() {
 
	VideoCapture cap("E:/Videos/CVPR_benchmark/ironman/img/%04d.jpg"); //change with your videopath

	// Parameters
	int diamSpx = 15;
	int wc = 35;
	int nIteration = 5;
	SlicCuda::InitType initType = SlicCuda::SLIC_SIZE;

	//start segmentation
	Mat frame;
	cap >> frame;
	SlicCuda oSlicCuda;
	oSlicCuda.initialize(frame, diamSpx, initType, wc, nIteration);
	
	Mat labels;
	int endFrame = cap.get(CAP_PROP_FRAME_COUNT);
    for(int i = 0; i<endFrame; i++){
        auto start = cv::getTickCount();
		oSlicCuda.segment(frame);
		oSlicCuda.enforceConnectivity();
        auto end = cv::getTickCount();
		cout << "runtime segmentation gpu " << (end - start) / cv::getTickFrequency() <<" s"<<endl;
		labels = oSlicCuda.getLabels();
		SlicCuda::displayBound(frame, (float*)labels.data, Scalar(255, 0, 0));
		imshow("segmentation", frame);
		waitKey(1);
		cap >> frame;
    }
    return 0;
}