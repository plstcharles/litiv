
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
    ASSERT_TRUE(oSPXMask.size()==oImgSize && oSPXMask.type()==CV_32FC1);
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_slic.bin")) {
        const cv::Mat oSPXMask_ref = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_slic.bin");
        ASSERT_TRUE(lv::isEqual<float>(oSPXMask,oSPXMask_ref));
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_slic.bin",oSPXMask);
}