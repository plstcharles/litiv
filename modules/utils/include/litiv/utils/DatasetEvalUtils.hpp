#pragma once

#define DATASETUTILS_USE_AVERAGE_EVAL_METRICS  1

#include "litiv/utils/DatasetUtils.hpp"

namespace DatasetUtils {

    namespace Segm {

        struct SegmEvaluator {
#if HAVE_GLSL
        protected:
            struct GLSegmEvaluator : public GLImageProcEvaluatorAlgo {
                GLSegmEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount);
                BasicMetrics getCumulativeMetrics();
                enum eSegmEvalCountersList {
                    eSegmEvalCounter_TP,
                    eSegmEvalCounter_TN,
                    eSegmEvalCounter_FP,
                    eSegmEvalCounter_FN,
                    eSegmEvalCounter_SE,
                    eSegmEvalCountersCount,
                };
            };
        public:
            virtual std::shared_ptr<GLSegmEvaluator> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const = 0;
            virtual BasicMetrics FetchCumulativeMetrics(const std::shared_ptr<GLSegmEvaluator>& pEvaluator) const;
#endif //HAVE_GLSL
            virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, BasicMetrics& m) const = 0;
            virtual cv::Mat GetDebugDisplayImage(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, size_t nIdx, cv::Point oDbgPt=cv::Point(-1,-1)) const;
            virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const = 0;
        };

        struct BinarySegmEvaluator : public SegmEvaluator {
#if HAVE_GLSL
        protected:
            struct GLBinarySegmEvaluator : public SegmEvaluator::GLSegmEvaluator {
                GLBinarySegmEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount);
                virtual std::string getComputeShaderSource(size_t nStage) const;
            };
        public:
            virtual std::shared_ptr<GLSegmEvaluator> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const;
#endif //HAVE_GLSL
            virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, BasicMetrics& m) const;
            virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const;
            static const uchar g_nSegmPositive;
            static const uchar g_nSegmOutOfScope;
            static const uchar g_nSegmNegative;
        };

        namespace Video {

            Metrics CalcMetricsFromCategory(const CategoryInfo& oCat, bool bAverage);
            Metrics CalcMetricsFromCategories(const std::vector<std::shared_ptr<CategoryInfo>>& vpCat, bool bAverage);
            void WriteMetrics(const std::string& sResultsFilePath, const SequenceInfo& oSeq);
            void WriteMetrics(const std::string& sResultsFilePath, const CategoryInfo& oCat);
            void WriteMetrics(const std::string& sResultsFilePath, const std::vector<std::shared_ptr<CategoryInfo>>& vpCat);
            cv::Mat ReadResult( const std::string& sResultsPath, const std::string& sCatName, const std::string& sSeqName,
                                const std::string& sResultPrefix, size_t nFrameIdx, const std::string& sResultSuffix, int nFlags=cv::IMREAD_GRAYSCALE);
            void WriteResult( const std::string& sResultsPath, const std::string& sCatName, const std::string& sSeqName,
                              const std::string& sResultPrefix, size_t nFrameIdx, const std::string& sResultSuffix,
                              const cv::Mat& oResult, const std::vector<int>& vnComprParams);

            class CDnetEvaluator : public SegmEvaluator {
#if HAVE_GLSL
                struct GLCDnetEvaluator : public SegmEvaluator::GLSegmEvaluator {
                    GLCDnetEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount);
                    virtual std::string getComputeShaderSource(size_t nStage) const;
                };
            public:
                virtual std::shared_ptr<SegmEvaluator::GLSegmEvaluator> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const;
#endif //HAVE_GLSL
                virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, BasicMetrics& m) const;
                virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const;
                static const uchar g_nSegmPositive;
                static const uchar g_nSegmOutOfScope;
                static const uchar g_nSegmNegative;
                static const uchar g_nSegmUnknown;
                static const uchar g_nSegmShadow;
            };

        }; //namespace Video

        namespace Image {

        }; //namespace Image

    }; //namespace Segm

}; //namespace DatasetUtils
