
#include "litiv/utils/platform.hpp"
#include "common.hpp"

TEST(filesystem_ops,regression) {
    EXPECT_EQ(lv::addDirSlashIfMissing(""),std::string());
    EXPECT_EQ(lv::addDirSlashIfMissing("c:/"),std::string("c:/"));
    EXPECT_EQ(lv::addDirSlashIfMissing("/"),std::string("/"));
    EXPECT_EQ(lv::addDirSlashIfMissing("."),std::string("./"));
    EXPECT_EQ(lv::addDirSlashIfMissing(".."),std::string("../"));
    EXPECT_EQ(lv::addDirSlashIfMissing("/test/path"),std::string("/test/path/"));
    EXPECT_EQ(lv::addDirSlashIfMissing("/test/path/.."),std::string("/test/path/../"));
    const std::string sDirPath = TEST_DATA_ROOT "/platformtest/";
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath));
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath+"subdir1"));
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath+"subdir2/"));
    ASSERT_TRUE(lv::createDirIfNotExist(sDirPath+"subdir2/subdir3"));
    std::fstream oTestFile(sDirPath+"test1.txt",std::ios::out);
    ASSERT_TRUE(oTestFile.is_open());
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
    EXPECT_GT(lv::getCurrentPhysMemBytesUsed(),0);
}

namespace {
    template<typename T>
    struct AlignedMemAllocator_fixture : testing::Test {};
    typedef testing::Types<char, short, int, uint8_t, uint16_t, float, double> AlignedMemAllocator_types;
}
TYPED_TEST_CASE(AlignedMemAllocator_fixture,AlignedMemAllocator_types);
TYPED_TEST(AlignedMemAllocator_fixture,regression) {
    std::vector<TypeParam,lv::AlignedMemAllocator<TypeParam,16>> vVec16a(100);
    EXPECT_EQ(((uintptr_t)vVec16a.data()%16),0);
    ASSERT_EQ(vVec16a[0],TypeParam(0));
    ASSERT_TRUE(std::equal(vVec16a.begin()+1,vVec16a.end(),vVec16a.begin()));
    std::vector<TypeParam,lv::AlignedMemAllocator<TypeParam,32>> vVec32a(100);
    EXPECT_EQ(((uintptr_t)vVec32a.data()%32),0);
    ASSERT_EQ(vVec32a[0],TypeParam(0));
    ASSERT_TRUE(std::equal(vVec32a.begin()+1,vVec32a.end(),vVec32a.begin()));
}