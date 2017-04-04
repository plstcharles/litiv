
#include "litiv/utils/platform.hpp"
#include "litiv/test.hpp"

TEST(filesystem_ops,regression) {
    EXPECT_EQ(lv::addDirSlashIfMissing(""),std::string());
    EXPECT_EQ(lv::addDirSlashIfMissing("c:/"),std::string("c:/"));
    EXPECT_EQ(lv::addDirSlashIfMissing("/"),std::string("/"));
    EXPECT_EQ(lv::addDirSlashIfMissing("."),std::string("./"));
    EXPECT_EQ(lv::addDirSlashIfMissing(".."),std::string("../"));
    EXPECT_EQ(lv::addDirSlashIfMissing("/test/path"),std::string("/test/path/"));
    EXPECT_EQ(lv::addDirSlashIfMissing("/test/path/.."),std::string("/test/path/../"));
    const std::string sDirPath = TEST_OUTPUT_DATA_ROOT "/platformtest/";
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath));
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath+"subdir1"));
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath+"subdir2/"));
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath+"subdir2/subdir3"));
    ASSERT_TRUE(lv::checkIfExists(sDirPath+"subdir2/subdir3"));
    ASSERT_FALSE(lv::checkIfExists(sDirPath+"subdir2/subdir4"));
    std::fstream oTestFile(sDirPath+"test1.txt",std::ios::out);
    ASSERT_TRUE(oTestFile.is_open());
    ASSERT_TRUE(lv::checkIfExists(sDirPath+"test1.txt"));
    ASSERT_FALSE(lv::checkIfExists(sDirPath+"test2.txt"));
    oTestFile << "test" << std::endl;
    oTestFile.close();
    oTestFile = lv::createBinFileWithPrealloc(sDirPath+"test2.bin",1024);
    ASSERT_TRUE(oTestFile.is_open());
    oTestFile << "test" << std::endl;
    oTestFile.close();
    EXPECT_EQ(lv::getFilesFromDir(sDirPath),(std::vector<std::string>{sDirPath+"test1.txt",sDirPath+"test2.bin"}));
    std::vector<std::string> vsFiles1 = lv::getFilesFromDir(sDirPath);
    lv::filterFilePaths(vsFiles1,(std::vector<std::string>{".bin"}),(std::vector<std::string>{}));
    EXPECT_EQ(vsFiles1,(std::vector<std::string>{sDirPath+"test1.txt"}));
    std::vector<std::string> vsFiles2 = lv::getFilesFromDir(sDirPath);
    lv::filterFilePaths(vsFiles2,(std::vector<std::string>{}),(std::vector<std::string>{".txt"}));
    EXPECT_EQ(vsFiles2,(std::vector<std::string>{sDirPath+"test1.txt"}));
    EXPECT_EQ(lv::getSubDirsFromDir(sDirPath),(std::vector<std::string>{sDirPath+"subdir1",sDirPath+"subdir2"}));
    EXPECT_GT(lv::getCurrentPhysMemBytesUsed(),size_t(0));
}