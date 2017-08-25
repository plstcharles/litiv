
#include "litiv/features2d/SC.hpp"
#include "litiv/test.hpp"

TEST(sc,regression_default_params) {
    std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(size_t(2),size_t(5));
    EXPECT_GE(pShapeContext->windowSize().width,0);
    EXPECT_GE(pShapeContext->windowSize().height,0);
    EXPECT_GE(pShapeContext->borderSize(0),0);
    EXPECT_GE(pShapeContext->borderSize(1),0);
    ASSERT_THROW_LV_QUIET(pShapeContext->borderSize(2));
    EXPECT_GT(pShapeContext->descriptorSize(),0);
    EXPECT_EQ(size_t(pShapeContext->descriptorSize())%sizeof(float),size_t(0));
    EXPECT_EQ(pShapeContext->descriptorType(),CV_32F);
    EXPECT_EQ(pShapeContext->descriptorType(),CV_32FC1);
    pShapeContext = std::make_unique<ShapeContext>(0.1,1.0);
    EXPECT_GE(pShapeContext->windowSize().width,0);
    EXPECT_GE(pShapeContext->windowSize().height,0);
    EXPECT_GE(pShapeContext->borderSize(0),0);
    EXPECT_GE(pShapeContext->borderSize(1),0);
    ASSERT_THROW_LV_QUIET(pShapeContext->borderSize(2));
    EXPECT_GT(pShapeContext->descriptorSize(),0);
    EXPECT_EQ(size_t(pShapeContext->descriptorSize())%sizeof(float),size_t(0));
    EXPECT_EQ(pShapeContext->descriptorType(),CV_32F);
    EXPECT_EQ(pShapeContext->descriptorType(),CV_32FC1);
}

TEST(sc,regression_full_compute_abs_nogpu) {
    std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(size_t(2),size_t(40),12,5);
#if HAVE_CUDA
    pShapeContext->enableCUDA(false);
#endif //HAVE_CUDA
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs;
    pShapeContext->compute(oInput,vTargetPts,oOutputDescs);
    ASSERT_EQ(vTargetPts.size(),vTargetPtsOrig.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i)
        ASSERT_EQ(vTargetPts[i].pt,vTargetPtsOrig[i].pt);
    ASSERT_EQ(oOutputDescs.rows,(int)vTargetPts.size());
    cv::Mat_<float> oOutputDescMap;
    pShapeContext->compute2(oInput,vTargetPts,oOutputDescMap);
    ASSERT_EQ(vTargetPts.size(),vTargetPtsOrig.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i)
        ASSERT_EQ(vTargetPts[i].pt,vTargetPtsOrig[i].pt);
    ASSERT_EQ(oOutputDescs.rows,(int)vTargetPts.size());
    ASSERT_EQ(oInput.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(oOutputDescs.cols,oOutputDescMap.size[2]);
    const int nDescSize = oOutputDescMap.size[2];
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs.ptr<float>(i);
        float* aDesc2 = oOutputDescMap.ptr<float>(int(vTargetPts[i].pt.y),int(vTargetPts[i].pt.x));
        ASSERT_TRUE(std::equal(aDesc1,aDesc1+nDescSize,aDesc2,aDesc2+nDescSize));
    }
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_sc_abs.bin")) {
        cv::Mat_<float> oRefDescs = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_sc_abs.bin");
        ASSERT_EQ(oOutputDescs.total(),oRefDescs.total());
        ASSERT_EQ(oOutputDescs.size,oRefDescs.size);
        for(int i=0; i<(int)vTargetPts.size(); ++i) {
            //lvPrint(vTargetPts[i].pt);
            float* aDesc1 = oOutputDescs.ptr<float>(i);
            //lvPrint(cv::Mat_<float>(5,12,aDesc1));
            float* aDesc2 = oRefDescs.ptr<float>(i);
            //lvPrint(cv::Mat_<float>(5,12,aDesc2));
            for(int j=0; j<nDescSize; ++j)
                ASSERT_FLOAT_EQ(aDesc1[j],aDesc2[j]) << "i=" << i << ", j=" << j;
        }
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_sc_abs.bin",oOutputDescs);
    pShapeContext->compute2(oInput,oOutputDescMap);
    ASSERT_EQ(oInput.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(oOutputDescs.cols,oOutputDescMap.size[2]);
}

TEST(sc,regression_full_compute_abs) {
    std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(size_t(2),size_t(40),12,5);
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs;
    pShapeContext->compute(oInput,vTargetPts,oOutputDescs);
    ASSERT_EQ(vTargetPts.size(),vTargetPtsOrig.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i)
        ASSERT_EQ(vTargetPts[i].pt,vTargetPtsOrig[i].pt);
    ASSERT_EQ(oOutputDescs.rows,(int)vTargetPts.size());
    cv::Mat_<float> oOutputDescMap;
    pShapeContext->compute2(oInput,vTargetPts,oOutputDescMap);
    ASSERT_EQ(vTargetPts.size(),vTargetPtsOrig.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i)
        ASSERT_EQ(vTargetPts[i].pt,vTargetPtsOrig[i].pt);
    ASSERT_EQ(oOutputDescs.rows,(int)vTargetPts.size());
    ASSERT_EQ(oInput.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(oOutputDescs.cols,oOutputDescMap.size[2]);
    const int nDescSize = oOutputDescMap.size[2];
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs.ptr<float>(i);
        float* aDesc2 = oOutputDescMap.ptr<float>(int(vTargetPts[i].pt.y),int(vTargetPts[i].pt.x));
        for(int n=0; n<nDescSize; ++n)
            ASSERT_FLOAT_EQ(aDesc1[n],aDesc2[n]) << "for n = " << n;
    }
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_sc_abs.bin")) {
        cv::Mat_<float> oRefDescs = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_sc_abs.bin");
        ASSERT_EQ(oOutputDescs.total(),oRefDescs.total());
        ASSERT_EQ(oOutputDescs.size,oRefDescs.size);
        for(int i=0; i<(int)vTargetPts.size(); ++i) {
            //lvPrint(vTargetPts[i].pt);
            float* aDesc1 = oOutputDescs.ptr<float>(i);
            //lvPrint(cv::Mat_<float>(5,12,aDesc1));
            float* aDesc2 = oRefDescs.ptr<float>(i);
            //lvPrint(cv::Mat_<float>(5,12,aDesc2));
            for(int j=0; j<nDescSize; ++j)
                ASSERT_FLOAT_EQ(aDesc1[j],aDesc2[j]) << "i=" << i << ", j=" << j;
        }
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_sc_abs.bin",oOutputDescs);
    pShapeContext->compute2(oInput,oOutputDescMap);
    ASSERT_EQ(oInput.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(oOutputDescs.cols,oOutputDescMap.size[2]);
}

#if HAVE_CUDA

TEST(sc,regression_compute_abs_gpu_config_6_4) {
    std::unique_ptr<ShapeContext> pShapeContext_gpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),6,4);
    std::unique_ptr<ShapeContext> pShapeContext_nogpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),6,4);
    pShapeContext_nogpu->enableCUDA(false);
    const int nDescSize = 6*4;
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs_gpu,oOutputDescs_nogpu;
    pShapeContext_gpu->compute(oInput,vTargetPts,oOutputDescs_gpu);
    ASSERT_EQ(oOutputDescs_gpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_gpu.cols,nDescSize);
    pShapeContext_nogpu->compute(oInput,vTargetPts,oOutputDescs_nogpu);
    ASSERT_EQ(oOutputDescs_nogpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_nogpu.cols,nDescSize);
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs_gpu.ptr<float>(i);
        float* aDesc2 = oOutputDescs_nogpu.ptr<float>(i);
        for(int n=0; n<nDescSize; ++n)
            ASSERT_FLOAT_EQ(aDesc1[n],aDesc2[n]) << "for i = " << i << ", n = " << n;
    }
}

TEST(sc,regression_compute_abs_gpu_config_8_8) {
    std::unique_ptr<ShapeContext> pShapeContext_gpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),8,8);
    std::unique_ptr<ShapeContext> pShapeContext_nogpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),8,8);
    pShapeContext_nogpu->enableCUDA(false);
    const int nDescSize = 8*8;
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs_gpu,oOutputDescs_nogpu;
    pShapeContext_gpu->compute(oInput,vTargetPts,oOutputDescs_gpu);
    ASSERT_EQ(oOutputDescs_gpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_gpu.cols,nDescSize);
    pShapeContext_nogpu->compute(oInput,vTargetPts,oOutputDescs_nogpu);
    ASSERT_EQ(oOutputDescs_nogpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_nogpu.cols,nDescSize);
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs_gpu.ptr<float>(i);
        float* aDesc2 = oOutputDescs_nogpu.ptr<float>(i);
        for(int n=0; n<nDescSize; ++n)
            ASSERT_FLOAT_EQ(aDesc1[n],aDesc2[n]) << "for i = " << i << ", n = " << n;
    }
}

TEST(sc,regression_compute_abs_gpu_config_8_8_64threads) {
    std::unique_ptr<ShapeContext> pShapeContext_gpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),8,8);
    std::unique_ptr<ShapeContext> pShapeContext_nogpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),8,8);
    pShapeContext_nogpu->enableCUDA(false);
    pShapeContext_gpu->setBlockSize(size_t(64));
    const int nDescSize = 8*8;
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs_gpu,oOutputDescs_nogpu;
    pShapeContext_gpu->compute(oInput,vTargetPts,oOutputDescs_gpu);
    ASSERT_EQ(oOutputDescs_gpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_gpu.cols,nDescSize);
    pShapeContext_nogpu->compute(oInput,vTargetPts,oOutputDescs_nogpu);
    ASSERT_EQ(oOutputDescs_nogpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_nogpu.cols,nDescSize);
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs_gpu.ptr<float>(i);
        float* aDesc2 = oOutputDescs_nogpu.ptr<float>(i);
        for(int n=0; n<nDescSize; ++n)
            ASSERT_FLOAT_EQ(aDesc1[n],aDesc2[n]) << "for i = " << i << ", n = " << n;
    }
}

TEST(sc,regression_compute_abs_gpu_config_12_8) {
    std::unique_ptr<ShapeContext> pShapeContext_gpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),12,8);
    std::unique_ptr<ShapeContext> pShapeContext_nogpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),12,8);
    pShapeContext_nogpu->enableCUDA(false);
    const int nDescSize = 12*8;
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs_gpu,oOutputDescs_nogpu;
    pShapeContext_gpu->compute(oInput,vTargetPts,oOutputDescs_gpu);
    ASSERT_EQ(oOutputDescs_gpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_gpu.cols,nDescSize);
    pShapeContext_nogpu->compute(oInput,vTargetPts,oOutputDescs_nogpu);
    ASSERT_EQ(oOutputDescs_nogpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_nogpu.cols,nDescSize);
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs_gpu.ptr<float>(i);
        float* aDesc2 = oOutputDescs_nogpu.ptr<float>(i);
        for(int n=0; n<nDescSize; ++n)
            ASSERT_FLOAT_EQ(aDesc1[n],aDesc2[n]) << "for i = " << i << ", n = " << n;
    }
}

TEST(sc,regression_compute_abs_gpu_config_16_8) {
    std::unique_ptr<ShapeContext> pShapeContext_gpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),16,8);
    std::unique_ptr<ShapeContext> pShapeContext_nogpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),16,8);
    pShapeContext_nogpu->enableCUDA(false);
    const int nDescSize = 16*8;
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs_gpu,oOutputDescs_nogpu;
    pShapeContext_gpu->compute(oInput,vTargetPts,oOutputDescs_gpu);
    ASSERT_EQ(oOutputDescs_gpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_gpu.cols,nDescSize);
    pShapeContext_nogpu->compute(oInput,vTargetPts,oOutputDescs_nogpu);
    ASSERT_EQ(oOutputDescs_nogpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_nogpu.cols,nDescSize);
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs_gpu.ptr<float>(i);
        float* aDesc2 = oOutputDescs_nogpu.ptr<float>(i);
        for(int n=0; n<nDescSize; ++n)
            ASSERT_FLOAT_EQ(aDesc1[n],aDesc2[n]) << "for i = " << i << ", n = " << n;
    }
}

TEST(sc,regression_compute_abs_gpu_config_16_8_64threads) {
    std::unique_ptr<ShapeContext> pShapeContext_gpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),16,8);
    std::unique_ptr<ShapeContext> pShapeContext_nogpu = std::make_unique<ShapeContext>(size_t(2),size_t(40),16,8);
    pShapeContext_nogpu->enableCUDA(false);
    pShapeContext_gpu->setBlockSize(size_t(64));
    const int nDescSize = 16*8;
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs_gpu,oOutputDescs_nogpu;
    pShapeContext_gpu->compute(oInput,vTargetPts,oOutputDescs_gpu);
    ASSERT_EQ(oOutputDescs_gpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_gpu.cols,nDescSize);
    pShapeContext_nogpu->compute(oInput,vTargetPts,oOutputDescs_nogpu);
    ASSERT_EQ(oOutputDescs_nogpu.rows,(int)vTargetPts.size());
    ASSERT_EQ(oOutputDescs_nogpu.cols,nDescSize);
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs_gpu.ptr<float>(i);
        float* aDesc2 = oOutputDescs_nogpu.ptr<float>(i);
        for(int n=0; n<nDescSize; ++n)
            ASSERT_FLOAT_EQ(aDesc1[n],aDesc2[n]) << "for i = " << i << ", n = " << n;
    }
}

#endif //HAVE_CUDA

TEST(sc,regression_full_compute_rel) {
    std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(0.1,1.0,12,5);
    cv::Mat oInput(257,257,CV_8UC1);
    oInput = 0;
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs;
    pShapeContext->compute(oInput,vTargetPts,oOutputDescs);
    ASSERT_EQ(vTargetPts.size(),vTargetPtsOrig.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i)
        ASSERT_EQ(vTargetPts[i].pt,vTargetPtsOrig[i].pt);
    ASSERT_EQ(oOutputDescs.rows,(int)vTargetPts.size());
    cv::Mat_<float> oOutputDescMap;
    pShapeContext->compute2(oInput,vTargetPts,oOutputDescMap);
    ASSERT_EQ(vTargetPts.size(),vTargetPtsOrig.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i)
        ASSERT_EQ(vTargetPts[i].pt,vTargetPtsOrig[i].pt);
    ASSERT_EQ(oOutputDescs.rows,(int)vTargetPts.size());
    ASSERT_EQ(oInput.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(oOutputDescs.cols,oOutputDescMap.size[2]);
    const int nDescSize = oOutputDescMap.size[2];
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        float* aDesc1 = oOutputDescs.ptr<float>(i);
        float* aDesc2 = oOutputDescMap.ptr<float>(int(vTargetPts[i].pt.y),int(vTargetPts[i].pt.x));
        ASSERT_TRUE(std::equal(aDesc1,aDesc1+nDescSize,aDesc2,aDesc2+nDescSize));
    }
    if(lv::checkIfExists(TEST_CURR_INPUT_DATA_ROOT "/test_sc_rel.bin")) {
        cv::Mat_<float> oRefDescs = lv::read(TEST_CURR_INPUT_DATA_ROOT "/test_sc_rel.bin");
        ASSERT_EQ(oOutputDescs.total(),oRefDescs.total());
        ASSERT_EQ(oOutputDescs.size,oRefDescs.size);
        for(int i=0; i<(int)vTargetPts.size(); ++i) {
            float* aDesc1 = oOutputDescs.ptr<float>(i);
            float* aDesc2 = oRefDescs.ptr<float>(i);
            for(int j=0; j<nDescSize; ++j)
                ASSERT_FLOAT_EQ(aDesc1[j],aDesc2[j]) << "i=" << i << ", j=" << j;
        }
    }
    else
        lv::write(TEST_CURR_INPUT_DATA_ROOT "/test_sc_rel.bin",oOutputDescs);
    pShapeContext->compute2(oInput,oOutputDescMap);
    ASSERT_EQ(oInput.size[0],oOutputDescMap.size[0]);
    ASSERT_EQ(oInput.size[1],oOutputDescMap.size[1]);
    ASSERT_EQ(oOutputDescs.cols,oOutputDescMap.size[2]);
}

TEST(sc,regression_emd_check) {
    std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(0.1,1.0);
    cv::Mat oInput(257,257,CV_8UC1);
    cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
    cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
    oInput = oInput>0;
    std::vector<cv::KeyPoint> vTargetPts = {
        cv::KeyPoint(cv::Point2f(100,100),1.0f),
        cv::KeyPoint(cv::Point2f(110,110),1.0f),
        cv::KeyPoint(cv::Point2f(128,128),1.0f),
        cv::KeyPoint(cv::Point2f(150,150),1.0f),
        cv::KeyPoint(cv::Point2f(185,185),1.0f),
        cv::KeyPoint(cv::Point2f(188,188),1.0f),
    };
    const auto vTargetPtsOrig = vTargetPts;
    cv::Mat_<float> oOutputDescs;
    pShapeContext->compute(oInput,vTargetPts,oOutputDescs);
    ASSERT_EQ(vTargetPts.size(),vTargetPtsOrig.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i)
        ASSERT_EQ(vTargetPts[i].pt,vTargetPtsOrig[i].pt);
    ASSERT_EQ(oOutputDescs.rows,(int)vTargetPts.size());
    for(int i=0; i<(int)vTargetPts.size(); ++i) {
        const float* aDesc1 = oOutputDescs.ptr<float>(0);
        const float* aDesc2 = oOutputDescs.ptr<float>(i);
        if(i==0)
            ASSERT_EQ(pShapeContext->calcDistance_EMD(aDesc1,aDesc2),0.0);
        else
            ASSERT_GT(pShapeContext->calcDistance_EMD(aDesc1,aDesc2),0.0);
    }
}

namespace {

    void sc_abs_perftest(benchmark::State& state) {
        std::unique_ptr<ShapeContext> pShapeContext = std::make_unique<ShapeContext>(size_t(state.range(0)),size_t(state.range(1)));
        cv::Mat oInput(257,257,CV_8UC1);
        oInput = 0;
        cv::circle(oInput,cv::Point(128,128),7,cv::Scalar_<uchar>(255),-1);
        cv::rectangle(oInput,cv::Point(180,180),cv::Point(190,190),cv::Scalar_<uchar>(255),-1);
        oInput = oInput>0;
        cv::Mat_<float> oOutputDescMap;
        while (state.KeepRunning()) {
            pShapeContext->compute2(oInput,oOutputDescMap);
            lvDbgAssert(oInput.size[0]==oOutputDescMap.size[0]);
            lvDbgAssert(oInput.size[1]==oOutputDescMap.size[1]);
            benchmark::DoNotOptimize(oOutputDescMap);
        }
    }
}

BENCHMARK(sc_abs_perftest)->Args({2,20})->Args({2,30})->Args({2,40})->Args({5,40})->Args({5,80})->Repetitions(15)->ReportAggregatesOnly(true);