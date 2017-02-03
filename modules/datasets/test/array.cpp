
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
    }
}