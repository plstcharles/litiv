
#include "../../../samples/datasets/src/middlebury2005.hpp" // cheat, to avoid copy & reuse in exported headers
#include "litiv/test.hpp"

TEST(datasets_array,regression_custom) {
    // ... @@@@ TODO
}

TEST(datasets_array,regression_specialization) {
    lv::datasets::setParserVerbosity(0);
    using DatasetType = lv::Dataset_<lv::DatasetTask_Cosegm,lv::Dataset_Middlebury2005_demo,lv::NonParallel>;
    const std::string sOutputRootPath = TEST_OUTPUT_DATA_ROOT "/middlebury_test/";
    DatasetType::Ptr pDataset = DatasetType::create(sOutputRootPath,true);
    ASSERT_TRUE(lv::checkIfExists(sOutputRootPath));
    lv::IDataHandlerPtrArray vpBatches = pDataset->getBatches(false);
    ASSERT_EQ(vpBatches.size(),size_t(2));
    ASSERT_EQ(pDataset->getInputCount(),size_t(2));
    ASSERT_EQ(pDataset->getGTCount(),size_t(2));
    for(auto& pBatch : vpBatches) {
        DatasetType::WorkBatch& oBatch = dynamic_cast<DatasetType::WorkBatch&>(*pBatch);
        ASSERT_TRUE(lv::checkIfExists(sOutputRootPath+oBatch.getName()));
        ASSERT_EQ(oBatch.getInputCount(),size_t(1));
        ASSERT_EQ(oBatch.getGTCount(),size_t(1));
        ASSERT_EQ(oBatch.getInputStreamCount(),size_t(2));
        ASSERT_EQ(oBatch.getGTStreamCount(),size_t(2));
        const std::vector<cv::Mat>& vImages = oBatch.getInputArray(0);
        const std::vector<cv::Mat>& vGTMaps = oBatch.getGTArray(0);
        ASSERT_EQ(vImages.size(),size_t(2));
        ASSERT_EQ(vGTMaps.size(),size_t(2));
        ASSERT_TRUE(!vImages[0].empty() && !vImages[1].empty());
        ASSERT_TRUE(vImages[0].size()==vImages[1].size());
        ASSERT_TRUE(!vGTMaps[0].empty() && !vGTMaps[1].empty());
        ASSERT_TRUE(vGTMaps[0].size()==vGTMaps[1].size());
        ASSERT_TRUE(vImages[0].size()==vGTMaps[0].size());
        const cv::Mat oInput0 = cv::imread(std::string(SAMPLES_DATA_ROOT "/middlebury2005_dataset_ex/")+pBatch->getName()+"/view1.png");
        const cv::Mat oInput1 = cv::imread(std::string(SAMPLES_DATA_ROOT "/middlebury2005_dataset_ex/")+pBatch->getName()+"/view5.png");
        ASSERT_TRUE(!oInput0.empty() && !oInput1.empty());
        ASSERT_TRUE(lv::isEqual<uchar>(oInput0,vImages[0]));
        ASSERT_TRUE(lv::isEqual<uchar>(oInput1,vImages[1]));
        const cv::Mat oGT0 = cv::imread(std::string(SAMPLES_DATA_ROOT "/middlebury2005_dataset_ex/")+pBatch->getName()+"/disp1.png",cv::IMREAD_GRAYSCALE);
        const cv::Mat oGT1 = cv::imread(std::string(SAMPLES_DATA_ROOT "/middlebury2005_dataset_ex/")+pBatch->getName()+"/disp5.png",cv::IMREAD_GRAYSCALE);
        ASSERT_TRUE(!oGT0.empty() && !oGT1.empty());
        ASSERT_TRUE(lv::isEqual<uchar>(oGT0,vGTMaps[0]));
        ASSERT_TRUE(lv::isEqual<uchar>(oGT1,vGTMaps[1]));
        const std::vector<cv::Mat> vFeaturesTest = {oInput0,oInput1,oGT0,oGT1};
        std::vector<lv::MatInfo> vPackInfo;
        oBatch.saveFeaturesArray(0,vFeaturesTest,&vPackInfo);
        const std::vector<cv::Mat>& vFeaturesTestOut = oBatch.loadFeaturesArray(0,vPackInfo);
        ASSERT_TRUE(lv::isEqual<uchar>(vFeaturesTestOut[0],oInput0));
        ASSERT_TRUE(lv::isEqual<uchar>(vFeaturesTestOut[1],oInput1));
        ASSERT_TRUE(lv::isEqual<uchar>(vFeaturesTestOut[2],oGT0));
        ASSERT_TRUE(lv::isEqual<uchar>(vFeaturesTestOut[3],oGT1));
    }
}