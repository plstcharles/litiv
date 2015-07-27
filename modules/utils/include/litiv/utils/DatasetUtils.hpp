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
        static std::vector<std::shared_ptr<WorkGroup>> ParseDataset(const DatasetInfoBase& oInfo);
        virtual eDatasetTypeList GetType() const = 0;
        std::string m_sDatasetName;
        std::string m_sDatasetRootPath;
        std::string m_sResultsRootPath;
        std::shared_ptr<EvaluatorBase> m_pEvaluator;
        std::vector<std::string> m_vsWorkBatchPaths;
        std::vector<std::string> m_vsSkippedNameTokens;
        std::vector<std::string> m_vsGrayscaleNameTokens;
        bool m_bForce4ByteDataAlign;
    };

    class WorkBatch {
    public:
        WorkBatch(const std::string& sBatchName, const std::string& sBatchPath, const DatasetInfoBase& oDatasetInfo, const std::string& sGroupName=std::string());
        template<typename Tp> static typename std::enable_if<std::is_base_of<WorkBatch,Tp>::value,bool>::type compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {return PlatformUtils::compare_lowercase(i->m_sName,j->m_sName);}
        static bool compare(const WorkBatch* i, const WorkBatch* j) {return PlatformUtils::compare_lowercase(i->m_sName,j->m_sName);}
        static bool compare(const WorkBatch& i, const WorkBatch& j) {return PlatformUtils::compare_lowercase(i.m_sName,j.m_sName);}
        virtual size_t GetTotalImageCount() const = 0;
        virtual double GetExpectedLoad() const = 0;
        virtual bool StartPrecaching(size_t nSuggestedBufferSize=SIZE_MAX);
        void StopPrecaching();
        const std::string m_sName;
        const std::string m_sGroupName;
        const std::string m_sPath;
        const bool m_bHasGroundTruth;
        const bool m_bForcingGrayscale;
        const bool m_bForcing4ByteDataAlign;
        const cv::Mat& GetInputFromIndex(size_t nIdx) {return m_oInputPrecacher.GetImageFromIndex(nIdx);}
        const cv::Mat& GetGTFromIndex(size_t nIdx) {return m_oGTPrecacher.GetImageFromIndex(nIdx);}
    protected:
        ImagePrecacher m_oInputPrecacher;
        ImagePrecacher m_oGTPrecacher;
        friend class WorkGroup;
        virtual cv::Mat GetInputFromIndex_external(size_t nIdx) = 0;
        virtual cv::Mat GetGTFromIndex_external(size_t nIdx) = 0;
    private:
        const cv::Mat& GetInputFromIndex_internal(size_t nIdx);
        const cv::Mat& GetGTFromIndex_internal(size_t nIdx);
        cv::Mat m_oLatestInputImage;
        cv::Mat m_oLatestGTMask;
        WorkBatch& operator=(const WorkBatch&) = delete;
        WorkBatch(const WorkBatch&) = delete;
    };

    class WorkGroup : public WorkBatch {
    public:
        WorkGroup(const std::string& sGroupName, const std::string& sGroupPath, const DatasetInfoBase& oDatasetInfo, const std::string& sSuperGroupName=std::string());
        virtual size_t GetTotalImageCount() const {return m_nTotImageCount;}
        virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
        std::vector<std::shared_ptr<WorkBatch>> m_vpBatches;
    protected:
        virtual cv::Mat GetInputFromIndex_external(size_t nFrameIdx);
        virtual cv::Mat GetGTFromIndex_external(size_t nFrameIdx);
    private:
        double m_dExpectedLoad;
        size_t m_nTotImageCount;
        WorkGroup& operator=(const WorkGroup&) = delete;
        WorkGroup(const WorkGroup&) = delete;
    };



    namespace Segm {

        struct BasicMetrics {
            BasicMetrics();
            BasicMetrics operator+(const BasicMetrics& m) const;
            BasicMetrics& operator+=(const BasicMetrics& m);
            uint64_t total() const {return nTP+nTN+nFP+nFN;}
            uint64_t nTP;
            uint64_t nTN;
            uint64_t nFP;
            uint64_t nFN;
            uint64_t nSE; // 'shadow error', not always used/required for eval
            double dTimeElapsed_sec;
        };

        struct Metrics {
            Metrics(const BasicMetrics& m);
            Metrics operator+(const BasicMetrics& m) const;
            Metrics& operator+=(const BasicMetrics& m);
            Metrics operator+(const Metrics& m) const;
            Metrics& operator+=(const Metrics& m);
            double dRecall;
            double dSpecificity;
            double dFPR;
            double dFNR;
            double dPBC;
            double dPrecision;
            double dFMeasure;
            double dMCC;
            double dTimeElapsed_sec; // never averaged, always accumulated
            size_t nWeight; // used to compute averages in overloads only
            static double CalcFMeasure(const BasicMetrics& m);
            static double CalcRecall(const BasicMetrics& m);
            static double CalcPrecision(const BasicMetrics& m);
            static double CalcSpecificity(const BasicMetrics& m);
            static double CalcFalsePositiveRate(const BasicMetrics& m);
            static double CalcFalseNegativeRate(const BasicMetrics& m);
            static double CalcPercentBadClassifs(const BasicMetrics& m);
            static double CalcMatthewsCorrCoeff(const BasicMetrics& m);
        };

        class SegmWorkBatch : public WorkBatch {
        public:
            SegmWorkBatch(const std::string& sBatchName, const std::string& sBatchPath, const DatasetInfoBase& oDatasetInfo, const std::string& sGroupName=std::string());
            BasicMetrics m_oMetrics;
        };

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
                virtual eDatasetTypeList GetType() const {return eDatasetType_Segm_Video;}
                eDatasetList m_eDatasetID;
                std::string m_sResultFrameNamePrefix;
                std::string m_sResultFrameNameSuffix;
                size_t m_nResultIdxOffset;
            };

            std::shared_ptr<DatasetInfo> GetDatasetInfo(eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign);

            class Sequence : public SegmWorkBatch {
            public:
                Sequence(const std::string& sSeqName, const std::string& sSeqPath, const DatasetInfo& oDatasetInfo, const std::string& sSeqGroupName=std::string());
                virtual size_t GetTotalImageCount() const {return m_nTotFrameCount;}
                virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
                cv::Size GetImageSize() const {return m_oSize;}
                const cv::Mat& GetROI() const {return m_oROI;}
                const eDatasetList m_eDatasetID;
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

        }; //namespace Video

        namespace Image {

            class SetInfo;

            enum eDatasetList {
                eDataset_BSDS500_train,
                eDataset_BSDS500_train_valid,
                // ...
                eDataset_Custom
            };

            struct DatasetInfo : public DatasetInfoBase {
                virtual eDatasetTypeList GetType() const {return eDatasetType_Segm_Image;}
                eDatasetList m_eDatasetID;
                std::string m_sResultImageNameExtension;
            };

            std::shared_ptr<DatasetInfo> GetDatasetInfo(eDatasetList eDatasetID, const std::string& sDatasetRootPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign);

        }; //namespace Image

    }; //namespace Segm

}; //namespace DatasetUtils
