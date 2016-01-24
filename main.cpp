#include <iostream>
#include <opencv2\opencv.hpp>
#include "SLIC_cuda.h"
using namespace std;
using namespace cv;

int main() {
 
    cv::Mat im = cv::imread("D:/Pictures/test_pic/lena.jpg");

    SLIC_cuda slic(16,35);
    slic.Initialize(im);

    for(int i = 0; i<5; i++){
        auto start = cv::getTickCount();
        slic.Segment(im);
        auto end = cv::getTickCount();
        cout<<"runtime gpu "<<(end-start)/cv::getTickFrequency()<<" for "<<N_ITER<<" iteration"<<endl;
    }


    cv::Mat out = im.clone();
    slic.displayBound(out,cv::Scalar(0,0,255));

    cv::imshow("out",out);


    cv::waitKey();



    return 0;
}