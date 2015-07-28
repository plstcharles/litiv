#pragma once

#define DATASETUTILS_USE_AVERAGE_EVAL_METRICS  1

#include "litiv/utils/DatasetUtils.hpp"

namespace DatasetUtils {

    struct EvaluatorBase {
#if HAVE_GLSL
        struct GLEvaluatorBase : public GLImageProcEvaluatorAlgo {
            GLEvaluatorBase(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount, size_t nCountersPerImage);
        };
        virtual std::shared_ptr<GLEvaluatorBase> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const = 0;
#endif //HAVE_GLSL
        virtual void test() {}; // @@@@@@@@@ get rid of
    };

    namespace Segm {

        Metrics CalcMetricsFromWorkGroup(const WorkGroup& oGroup, bool bAverage);
        Metrics CalcMetricsFromWorkGroups(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups, bool bAverage);
        void WriteMetrics(const std::string& sResultsFilePath, const SegmWorkBatch& oBatch);
        void WriteMetrics(const std::string& sResultsFilePath, const WorkGroup& oGroup);
        void WriteMetrics(const std::string& sResultsFilePath, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups);
        cv::Mat GetDisplayImage(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oSegmMask, size_t nIdx, cv::Point oDbgPt=cv::Point(-1,-1));

        struct SegmEvaluator : public EvaluatorBase {
#if HAVE_GLSL
            struct GLSegmEvaluator : public GLEvaluatorBase {
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
#endif //HAVE_GLSL
            virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, BasicMetrics& m) const = 0;
            virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const = 0;
        };

        namespace Video {

            struct BinarySegmEvaluator : public SegmEvaluator {
#if HAVE_GLSL
                struct GLBinarySegmEvaluator : public GLSegmEvaluator {
                    GLBinarySegmEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount);
                    virtual std::string getComputeShaderSource(size_t nStage) const;
                };
                virtual std::shared_ptr<GLEvaluatorBase> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const;
#endif //HAVE_GLSL
                virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, BasicMetrics& m) const;
                virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const;
                static const uchar g_nSegmPositive;
                static const uchar g_nSegmOutOfScope;
                static const uchar g_nSegmNegative;
            };

            struct CDnetEvaluator : public SegmEvaluator {
#if HAVE_GLSL
                struct GLCDnetEvaluator : public GLSegmEvaluator {
                    GLCDnetEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount);
                    virtual std::string getComputeShaderSource(size_t nStage) const;
                };
                virtual std::shared_ptr<GLEvaluatorBase> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const;
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

            // ...

        }; //namespace Image

    }; //namespace Segm

}; //namespace DatasetUtils
