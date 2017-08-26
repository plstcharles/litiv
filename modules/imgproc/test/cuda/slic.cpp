
#include "litiv/imgproc/SLIC.hpp"
#include "litiv/test.hpp"

TEST(slic,regression_compute) {
    const cv::Size oImgSize(481,321);
    const cv::Mat oInput = cv::imread(SAMPLES_DATA_ROOT "/108073.jpg");
    ASSERT_TRUE(!oInput.empty() && oInput.type()==CV_8UC3 && oInput.size()==oImgSize);
    SLIC oAlgo;
    oAlgo.initialize(oImgSize,15,SLIC::SLIC_SIZE,35,2);
    oAlgo.segment(oInput);
    oAlgo.enforceConnectivity();
    const cv::Mat& oSPXMask = oAlgo.getLabels();
    //lv::doNotOptimize(oSPXMask); // for some reason, unless we pass the algo output to another lib call, kernels don't execute on MSVC2015 in release...
    ASSERT_TRUE(oSPXMask.size()==oImgSize && oSPXMask.type()==CV_32FC1 && oSPXMask.isContinuous());
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_slic.bin")) {
        //cv::imshow("Segmentation output (NEW)",SLIC::displayBound(oInput,oSPXMask,cv::Scalar(255,0,0)));
        //cv::imshow("Superpixel RGB Mean (NEW)",SLIC::displayMean(oInput,oSPXMask));
        //cv::imshow("Segmentation output (OLD)",SLIC::displayBound(oInput,oSPXMask_ref,cv::Scalar(255,0,0)));
        //cv::imshow("Superpixel RGB Mean (OLD)",SLIC::displayMean(oInput,oSPXMask_ref));
        //cv::imshow("diff",((oSPXMask-oSPXMask_ref)>0.005f)|((oSPXMask-oSPXMask_ref)<-0.005f));
        //cv::waitKey(0);
        if(cv::cuda::DeviceInfo().majorVersion()==5 && cv::cuda::DeviceInfo().minorVersion()==2) {
            // test disabled in 2017/08 due to differing results on various architecture...
            const cv::Mat oSPXMask_ref = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_slic.bin");
            ASSERT_TRUE(oSPXMask_ref.size()==oImgSize && oSPXMask_ref.type()==CV_32FC1 && oSPXMask_ref.isContinuous());
            for(size_t n=0; n<oSPXMask_ref.total(); ++n)
                ASSERT_FLOAT_EQ(((float*)oSPXMask.data)[n],((float*)oSPXMask_ref.data)[n]) << "n=" << n;
        }
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_slic.bin",oSPXMask);
}