#include <iostream>
#include <opencv2/opencv.hpp>
#include "SLIC_cuda.h"
using namespace std;

int main() {
    cout << "Hello, World!" << endl;


    cv::Mat im = cv::imread("/media/derue/4A30A96F30A962A5/Videos/Tiger1/img/0001.jpg");
    //cv::Mat im = cv::imread("/media/derue/4A30A96F30A962A5/Pictures/test_pic/land1.jpg");

    SLIC_cuda slic(16,35);
    slic.Initialize(im);

    for(int i = 0; i<10; i++){
        auto start = cv::getTickCount();
        slic.Segment(im);
        auto end = cv::getTickCount();
        cout<<"runtime gpu "<<(end-start)/cv::getTickFrequency()<<endl;
    }


    cv::Mat out = im.clone();
    slic.displayBound(out,cv::Scalar(0,0,255));

    cv::imshow("out",out);




    cv::waitKey();

    return 0;
}