#pragma once

#define DATASETUTILS_HARDCODE_FRAME_INDEX      0
#define DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL  uchar(85)

#include "litiv/utils/ParallelUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

namespace DatasetUtils {

    void WriteOnImage(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom=false);
    cv::Mat GetDisplayImage(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oSegmMask, size_t nIdx, cv::Point oDbgPt=cv::Point(-1,-1), cv::Size oRefSize=cv::Size(-1,-1));
    void ValidateKeyPoints(const cv::Mat& oROI, std::vector<cv::KeyPoint>& voKPs);

    class ImagePrecacher {
    public:
        ImagePrecacher(std::function<const cv::Mat&(size_t)> pCallback);
        virtual ~ImagePrecacher();
        const cv::Mat& GetImageFromIndex(size_t nIdx);
        bool StartPrecaching(size_t nTotImageCount, size_t nSuggestedBufferSize);
        void StopPrecaching();
    private:
        void Precache();
        const cv::Mat& GetImageFromIndex_internal(size_t nIdx);
        std::function<const cv::Mat&(size_t)> m_pCallback;
        std::thread m_hPrecacher;
        std::mutex m_oSyncMutex;
        std::condition_variable m_oReqCondVar;
        std::condition_variable m_oSyncCondVar;
        bool m_bIsPrecaching;
        size_t m_nBufferSize;
        size_t m_nTotImageCount;
        std::deque<cv::Mat> m_qoCache;
        std::vector<uchar> m_vcBuffer;
        size_t m_nFirstBufferIdx;
        size_t m_nNextBufferIdx;
        size_t m_nNextExpectedReqIdx;
        size_t m_nNextPrecacheIdx;
        size_t m_nReqIdx,m_nLastReqIdx;
        cv::Mat m_oReqImage,m_oLastReqImage;
    };

    enum eDatasetTypeList {
        eDatasetType_Segm_Video,
        eDatasetType_Segm_Image,
        // ...
    };

    struct EvaluatorBase;
    struct WorkBatch;
    struct WorkGroup;

    struct DatasetInfoBase {
        virtual std::vector<std::shared_ptr<WorkGroup>> ParseDataset();
        virtual void WriteEvalResults(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) const = 0;
        virtual eDatasetTypeList GetType() const = 0;
        std::string m_sDatasetName;
        std::string m_sDatasetRootPath;
        std::string m_sResultsRootPath;
        std::string m_sResultNamePrefix;
        std::string m_sResultNameSuffix;
        std::vector<std::string> m_vsWorkBatchPaths;
        std::vector<std::string> m_vsSkippedNameTokens;
        std::vector<std::string> m_vsGrayscaleNameTokens;
        bool m_bForce4ByteDataAlign;
    };

    class WorkBatch {
    public:
        WorkBatch(const std::string& sBatchName, const DatasetInfoBase& oDatasetInfo, const std::string& sRelativePath=std::string("./"));
        template<typename Tp> static typename std::enable_if<std::is_base_of<WorkBatch,Tp>::value,bool>::type compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {return PlatformUtils::compare_lowercase(i->m_sName,j->m_sName);}
        static bool compare(const WorkBatch* i, const WorkBatch* j) {return PlatformUtils::compare_lowercase(i->m_sName,j->m_sName);}
        static bool compare(const WorkBatch& i, const WorkBatch& j) {return PlatformUtils::compare_lowercase(i.m_sName,j.m_sName);}
        virtual size_t GetTotalImageCount() const = 0;
        virtual double GetExpectedLoad() const = 0;
        virtual cv::Mat ReadResult(size_t nIdx);
        virtual void WriteResult(size_t nIdx, const cv::Mat& oResult);
        virtual bool StartPrecaching(bool bUsingGT, size_t nSuggestedBufferSize=SIZE_MAX);
        void StopPrecaching();
        const std::string m_sName;
        const std::string m_sRelativePath;
        const std::string m_sDatasetPath;
        const std::string m_sResultsPath;
        const std::string m_sResultNamePrefix;
        const std::string m_sResultNameSuffix;
        const bool m_bForcingGrayscale;
        const bool m_bForcing4ByteDataAlign;
        const cv::Mat& GetInputFromIndex(size_t nIdx) {return m_oInputPrecacher.GetImageFromIndex(nIdx);}
        const cv::Mat& GetGTFromIndex(size_t nIdx) {return m_oGTPrecacher.GetImageFromIndex(nIdx);}
        std::shared_ptr<EvaluatorBase> m_pEvaluator;
        std::promise<size_t> m_nImagesProcessed;
    protected:
        ImagePrecacher m_oInputPrecacher;
        ImagePrecacher m_oGTPrecacher;
        virtual cv::Mat GetInputFromIndex_external(size_t nIdx) = 0;
        virtual cv::Mat GetGTFromIndex_external(size_t nIdx) = 0;
    private:
        const cv::Mat& GetInputFromIndex_internal(size_t nIdx);
        const cv::Mat& GetGTFromIndex_internal(size_t nIdx);
        cv::Mat m_oLatestInputImage;
        cv::Mat m_oLatestGTMask;
        WorkBatch& operator=(const WorkBatch&) = delete;
        WorkBatch(const WorkBatch&) = delete;
        friend class WorkGroup;
    };

    class WorkGroup : public WorkBatch {
    public:
        WorkGroup(const std::string& sGroupName, const DatasetInfoBase& oDatasetInfo, const std::string& sRelativePath=std::string("./"));
        virtual size_t GetTotalImageCount() const {return m_nTotImageCount;}
        virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
        virtual bool IsBare() const {return m_bIsBare;}
        std::vector<std::shared_ptr<WorkBatch>> m_vpBatches;
    protected:
        virtual cv::Mat GetInputFromIndex_external(size_t nIdx);
        virtual cv::Mat GetGTFromIndex_external(size_t nIdx);
    private:
        bool m_bIsBare;
        double m_dExpectedLoad;
        size_t m_nTotImageCount;
        WorkGroup& operator=(const WorkGroup&) = delete;
        WorkGroup(const WorkGroup&) = delete;
    };

    namespace Segm {

        namespace Video {

            enum eDatasetList {
                eDataset_CDnet2012,
                eDataset_CDnet2014,
                eDataset_Wallflower,
                eDataset_PETS2001_D3TC1,
                // ...
                eDataset_Custom
            };

            struct DatasetInfo : public DatasetInfoBase {
                virtual void WriteEvalResults(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) const;
                virtual eDatasetTypeList GetType() const {return eDatasetType_Segm_Video;}
                eDatasetList m_eDatasetID;
                size_t m_nResultIdxOffset;
            };

            std::shared_ptr<DatasetInfo> GetDatasetInfo(eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign);

            class Sequence : public WorkBatch {
            public:
                Sequence(const std::string& sSeqName, const DatasetInfo& oDatasetInfo, const std::string& sRelativePath=std::string("./"));
                virtual size_t GetTotalImageCount() const {return m_nTotFrameCount;}
                virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
                virtual void WriteResult(size_t nIdx, const cv::Mat& oResult);
                virtual bool StartPrecaching(bool bUsingGT, size_t nUnused=0);
                cv::Size GetImageSize() const {return m_oSize;}
                const cv::Mat& GetROI() const {return m_oROI;}
                const eDatasetList m_eDatasetID;
                const size_t m_nResultIdxOffset;
            protected:
                virtual cv::Mat GetInputFromIndex_external(size_t nFrameIdx);
                virtual cv::Mat GetGTFromIndex_external(size_t nFrameIdx);
            private:
                double m_dExpectedLoad;
                size_t m_nTotFrameCount;
                std::vector<std::string> m_vsInputFramePaths;
                std::vector<std::string> m_vsGTFramePaths;
                cv::VideoCapture m_voVideoReader;
                size_t m_nNextExpectedVideoReaderFrameIdx;
                cv::Mat m_oROI;
                cv::Size m_oSize;
                std::unordered_map<size_t,size_t> m_mTestGTIndexes;
                Sequence& operator=(const Sequence&) = delete;
                Sequence(const Sequence&) = delete;
            };

        } //namespace Video

        namespace Image {

            enum eDatasetList {
                eDataset_BSDS500_segm_train,
                eDataset_BSDS500_segm_train_valid,
                eDataset_BSDS500_segm_train_valid_test,
                eDataset_BSDS500_edge_train,
                eDataset_BSDS500_edge_train_valid,
                eDataset_BSDS500_edge_train_valid_test,
                // ...
                eDataset_Custom
            };

            struct DatasetInfo : public DatasetInfoBase {
                virtual void WriteEvalResults(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) const;
                virtual eDatasetTypeList GetType() const {return eDatasetType_Segm_Image;}
                eDatasetList m_eDatasetID;
            };

            std::shared_ptr<DatasetInfo> GetDatasetInfo(eDatasetList eDatasetID, const std::string& sDatasetRootPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign);

            class Set : public WorkBatch {
            public:
                Set(const std::string& sSetName, const DatasetInfo& oDatasetInfo, const std::string& sRelativePath=std::string("./"));
                virtual size_t GetTotalImageCount() const {return m_nTotImageCount;}
                virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
                virtual cv::Mat ReadResult(size_t nIdx);
                virtual void WriteResult(size_t nIdx, const cv::Mat& oResult);
                virtual bool StartPrecaching(bool bUsingGT, size_t nUnused=0);
                bool IsConstantImageSize() const {return m_bIsConstantSize;}
                cv::Size GetMaxImageSize() const {return m_oMaxSize;}
                const eDatasetList m_eDatasetID;
            protected:
                virtual cv::Mat GetInputFromIndex_external(size_t nImageIdx);
                virtual cv::Mat GetGTFromIndex_external(size_t nImageIdx);
            private:
                double m_dExpectedLoad;
                size_t m_nTotImageCount;
                std::vector<std::string> m_vsInputImagePaths;
                std::vector<std::string> m_vsGTImagePaths;
                std::vector<std::string> m_vsOrigImageNames;
                std::vector<cv::Size> m_voOrigImageSizes;
                cv::Size m_oMaxSize;
                bool m_bIsConstantSize;
                Set& operator=(const Set&) = delete;
                Set(const Set&) = delete;
            };

        } //namespace Image

    } //namespace Segm

} //namespace DatasetUtils
