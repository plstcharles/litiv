#pragma once

#define DATASETUTILS_USE_AVERAGE_EVAL_METRICS  1
#define DATASETUTILS_HARDCODE_FRAME_INDEX      0

#include "litiv/utils/ParallelUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#if HAVE_GLSL
#include "litiv/utils/GLImageProcUtils.hpp"
#endif //HAVE_GLSL

namespace DatasetUtils {

    void WriteOnImage(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom=false);
    void ValidateKeyPoints(const cv::Mat& oROI, std::vector<cv::KeyPoint>& voKPs);
    typedef std::function<const cv::Mat&(size_t)> ImageQueryByIndexFunc;

    namespace SegmEval {

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

        struct SegmMetrics {
            SegmMetrics(const BasicMetrics& m);
            SegmMetrics operator+(const BasicMetrics& m) const;
            SegmMetrics& operator+=(const BasicMetrics& m);
            SegmMetrics operator+(const SegmMetrics& m) const;
            SegmMetrics& operator+=(const SegmMetrics& m);
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
        };

        double CalcFMeasure(const BasicMetrics& m);
        double CalcRecall(const BasicMetrics& m);
        double CalcPrecision(const BasicMetrics& m);
        double CalcSpecificity(const BasicMetrics& m);
        double CalcFalsePositiveRate(const BasicMetrics& m);
        double CalcFalseNegativeRate(const BasicMetrics& m);
        double CalcPercentBadClassifs(const BasicMetrics& m);
        double CalcMatthewsCorrCoeff(const BasicMetrics& m);

    }; //namespace SegmEval

    class ImagePrecacher {
    public:
        ImagePrecacher(ImageQueryByIndexFunc pCallback);
        virtual ~ImagePrecacher();
        const cv::Mat& GetImageFromIndex(size_t nIdx);
        bool StartPrecaching(size_t nTotImageCount, size_t nSuggestedBufferSize);
        void StopPrecaching();
    private:
        void Precache();
        const cv::Mat& GetImageFromIndex_internal(size_t nIdx);
        ImageQueryByIndexFunc m_pCallback;
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

    class WorkBatch {
    public:
        WorkBatch(const std::string& sName, const std::string& sPath, bool bForceGrayscale=false, bool bUse4chAlign=false);
        static bool compare(const WorkBatch& i, const WorkBatch& j) {return PlatformUtils::compare_lowercase(i.m_sName,j.m_sName);}
        static bool compare(const WorkBatch* i, const WorkBatch* j) {return PlatformUtils::compare_lowercase(i->m_sName,j->m_sName);}
        template<typename Tp> static typename std::enable_if<std::is_base_of<WorkBatch,Tp>::value,bool>::type compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {return PlatformUtils::compare_lowercase(i->m_sName,j->m_sName);}
        virtual size_t GetTotalImageCount() const = 0;
        virtual double GetExpectedLoad() const = 0;
        virtual bool StartPrecaching(size_t nSuggestedBufferSize=SIZE_MAX);
        void StopPrecaching();
        const std::string m_sName;
        const std::string m_sPath;
        const int m_nIMReadInputFlags;
        const bool m_bForcingGrayscale;
        const bool m_bUsing4chAlignment;
        const cv::Mat& GetInputFromIndex(size_t nIdx) {return m_oInputPrecacher.GetImageFromIndex(nIdx);}
        const cv::Mat& GetGTFromIndex(size_t nIdx) {return m_oGTPrecacher.GetImageFromIndex(nIdx);}
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
    };

    namespace VideoSegm {

        class SequenceInfo;
        class CategoryInfo;

        enum eDatasetList {
            eDataset_CDnet2012,
            eDataset_CDnet2014,
            eDataset_Wallflower,
            eDataset_PETS2001_D3TC1,
            eDataset_LITIV2012,
            eDataset_GenericTest,
            // ...
            eDatasetCount
        };

        struct DatasetInfo {
            const eDatasetList eID;
            const std::string sDatasetPath;
            const std::string sResultsPath;
            const std::string sResultPrefix;
            const std::string sResultSuffix;
            const std::vector<std::string> vsFolderPaths;
            const std::vector<std::string> vsGrayscaleNameTokens;
            const std::vector<std::string> vsSkippedNameTokens;
            const size_t nResultIdxOffset;
        };

        DatasetInfo GetDatasetInfo(const eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirPath);
        cv::Mat ReadResult( const std::string& sResultsPath, const std::string& sCatName, const std::string& sSeqName,
                            const std::string& sResultPrefix, size_t nFrameIdx, const std::string& sResultSuffix, int nFlags=cv::IMREAD_GRAYSCALE);
        void WriteResult( const std::string& sResultsPath, const std::string& sCatName, const std::string& sSeqName,
                          const std::string& sResultPrefix, size_t nFrameIdx, const std::string& sResultSuffix,
                          const cv::Mat& oResult, const std::vector<int>& vnComprParams);

        class CategoryInfo : public WorkBatch {
        public:
            CategoryInfo(const std::string& sName, const std::string& sDirectoryPath, eDatasetList eDatasetID,
                         const std::vector<std::string>& vsGrayscaleCategoryNameTokens=std::vector<std::string>(),
                         const std::vector<std::string>& vsSkippedDirectoryNameTokens=std::vector<std::string>(),
                         bool bUse4chAlign=false);
            virtual size_t GetTotalImageCount() const {return m_nTotFrameCount;}
            virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
            static SegmEval::SegmMetrics CalcMetricsFromCategory(const CategoryInfo& oCat, bool bAverage);
            static SegmEval::SegmMetrics CalcMetricsFromCategories(const std::vector<std::shared_ptr<CategoryInfo>>& vpCat, bool bAverage);
            static void WriteMetrics(const std::string& sResultsFilePath, const CategoryInfo& oCat);
            static void WriteMetrics(const std::string& sResultsFilePath, const std::vector<std::shared_ptr<CategoryInfo>>& vpCat);
            const eDatasetList m_eDatasetID;
            std::vector<std::shared_ptr<SequenceInfo>> m_vpSequences;
        protected:
            virtual cv::Mat GetInputFromIndex_external(size_t nFrameIdx);
            virtual cv::Mat GetGTFromIndex_external(size_t nFrameIdx);
        private:
            double m_dExpectedLoad;
            size_t m_nTotFrameCount;
            CategoryInfo& operator=(const CategoryInfo&)=delete;
            CategoryInfo(const CategoryInfo&)=delete;
        };

        class SequenceInfo : public WorkBatch {
        public:
            SequenceInfo(const std::string& sName, const std::string& sParentName, const std::string& sPath,
                         eDatasetList eDatasetID, bool bForceGrayscale=false, bool bUse4chAlign=false);
            virtual size_t GetTotalImageCount() const {return m_nTotFrameCount;}
            virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
            const std::string& GetParentName() const {return m_sParentName;}
            cv::Size GetImageSize() const {return m_oSize;}
            const cv::Mat& GetROI() const {return m_oROI;}
            static void WriteMetrics(const std::string& sResultsFilePath, const SequenceInfo& oSeq);
            const eDatasetList m_eDatasetID;
            const std::string m_sParentName;
            SegmEval::BasicMetrics m_oMetrics;
        protected:
            friend class CategoryInfo;
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
            SequenceInfo& operator=(const SequenceInfo&)=delete;
            SequenceInfo(const CategoryInfo&)=delete;
        };

        namespace CDnet {
            // as defined in the 2012 CDNet scripts/dataset
            const uchar g_nSegmPositive = 255;
            const uchar g_nSegmNegative = 0;
            const uchar g_nSegmOutOfScope = 85;
            const uchar g_nSegmUnknown = 170;
            const uchar g_nSegmShadow = 50;

            void AccumulateMetricsFromResult(const cv::Mat& oSegm, const cv::Mat& oGTFrame, const cv::Mat& oROI, SegmEval::BasicMetrics& m);
            cv::Mat GetDebugDisplayFrame( const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oSegmMask,
                                          const cv::Mat& oGTSegmMask, const cv::Mat& oROI, size_t nFrame, cv::Point oDbgPt=cv::Point(-1,-1));
            cv::Mat GetColoredSegmFrameFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI);
#if HAVE_GLSL
            class Evaluator : public GLImageProcEvaluatorAlgo {
            public:
                Evaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount);
                virtual std::string getComputeShaderSource(size_t nStage) const;
                void getCumulativeCounts(SegmEval::BasicMetrics& m);
                enum eCDNetEvalCountersList {
                    eCDNetEvalCounter_TP,
                    eCDNetEvalCounter_TN,
                    eCDNetEvalCounter_FP,
                    eCDNetEvalCounter_FN,
                    eCDNetEvalCounter_SE,
                    eCDNetEvalCountersCount,
                };
            };
#endif //HAVE_GLSL
        } //namespace CDnet

    } //namespace VideoSegm

    namespace ImageSegm {

        class BatchInfo;
        enum eDatasetList {
            eDataset_BSDS500,
            // ...
            eDatasetCount
        };

        class DatasetInfo {
        public:

        };

        DatasetInfo GetDatasetInfo(const eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirPath);

    } //namespace ImageSegm

}; //namespace DatasetUtils
