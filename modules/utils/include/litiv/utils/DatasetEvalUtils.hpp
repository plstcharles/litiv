#pragma once

#include "litiv/utils/DatasetUtils.hpp"

namespace DatasetUtils {

    struct BasicMetrics {
        BasicMetrics(std::string sID=std::string());
        BasicMetrics operator+(const BasicMetrics& m) const;
        BasicMetrics& operator+=(const BasicMetrics& m);
        uint64_t total() const {return nTP+nTN+nFP+nFN;}
        uint64_t nTP;
        uint64_t nTN;
        uint64_t nFP;
        uint64_t nFN;
        uint64_t nSE; // 'shadow error', not always used/required for eval
        double dTimeElapsed_sec;
        std::string sInternalID;
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
        std::string sInternalID;
        size_t nWeight; // used to compute averages in overloads only
        static double CalcFMeasure(double dRecall, double dPrecision);
        static double CalcFMeasure(const BasicMetrics& m);
        static double CalcRecall(uint64_t nTP, uint64_t nTPFN);
        static double CalcRecall(const BasicMetrics& m);
        static double CalcPrecision(uint64_t nTP, uint64_t nTPFP);
        static double CalcPrecision(const BasicMetrics& m);
        static double CalcSpecificity(const BasicMetrics& m);
        static double CalcFalsePositiveRate(const BasicMetrics& m);
        static double CalcFalseNegativeRate(const BasicMetrics& m);
        static double CalcPercentBadClassifs(const BasicMetrics& m);
        static double CalcMatthewsCorrCoeff(const BasicMetrics& m);
    };

    struct EvaluatorBase {
#if HAVE_GLSL
        enum eBasicEvalCountersList {
            eBasicEvalCounter_TP,
            eBasicEvalCounter_TN,
            eBasicEvalCounter_FP,
            eBasicEvalCounter_FN,
            eBasicEvalCounter_SE, // 'shadow error', not always used/required for eval
            eBasicEvalCountersCount,
        };
        struct GLEvaluatorBase : public GLImageProcEvaluatorAlgo {
            GLEvaluatorBase(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount, size_t nCountersPerImage=eBasicEvalCountersCount);
        };
        virtual std::shared_ptr<GLEvaluatorBase> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& /*pParent*/, size_t /*nTotImageCount*/) const {return nullptr;}
        virtual void FetchGLEvaluationResults(std::shared_ptr<GLEvaluatorBase> /*pGLEvaluator*/) {}
#endif //HAVE_GLSL
        virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const = 0;
        virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) = 0;
        double dTimeElapsed_sec;
    };

    namespace Segm {

        namespace Video {

            struct BinarySegmEvaluator : public EvaluatorBase {
                BinarySegmEvaluator(std::string sEvalID);
#if HAVE_GLSL
                struct GLBinarySegmEvaluator : public GLEvaluatorBase {
                    GLBinarySegmEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount);
                    virtual std::string getComputeShaderSource(size_t nStage) const;
                    virtual BasicMetrics getCumulativeMetrics();
                };
                virtual std::shared_ptr<GLEvaluatorBase> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const;
                virtual void FetchGLEvaluationResults(std::shared_ptr<GLEvaluatorBase> pGLEvaluator);
#endif //HAVE_GLSL
                virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const;
                virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI);
                static const uchar s_nSegmPositive;
                static const uchar s_nSegmOutOfScope;
                static const uchar s_nSegmNegative;
                BasicMetrics m_oBasicMetrics;
            protected:
                static void WriteEvalResults(const DatasetInfoBase& oInfo, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups, bool bAverageMetrics);
                static Metrics WriteEvalResults(const WorkGroup& oGroup, bool bAverage);
                static Metrics WriteEvalResults(const WorkBatch& oBatch);
                static Metrics CalcMetrics(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups, bool bAverage);
                static Metrics CalcMetrics(const WorkGroup& oGroup, bool bAverage);
                friend class DatasetInfo;
            };

            struct CDnetEvaluator : public BinarySegmEvaluator {
                CDnetEvaluator();
#if HAVE_GLSL
                struct GLCDnetEvaluator : public GLBinarySegmEvaluator {
                    GLCDnetEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount);
                    virtual std::string getComputeShaderSource(size_t nStage) const;
                };
                virtual std::shared_ptr<GLEvaluatorBase> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const;
#endif //HAVE_GLSL
                virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const;
                virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI);
                static const uchar s_nSegmUnknown;
                static const uchar s_nSegmShadow;
            };

        }; //namespace Video

        namespace Image {

            struct BSDS500BoundaryEvaluator : public EvaluatorBase {
                BSDS500BoundaryEvaluator(size_t nThresholdBins=UCHAR_MAX+1);
                virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/) const;
                virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/);
                static const double s_dMaxImageDiagRatioDist;
                struct BSDS500BasicMetrics { // basic eval metrics for a single image
                    BSDS500BasicMetrics(size_t nThresholdsBins);
                    std::vector<uint64_t> vnIndivTP; // one count per threshold
                    std::vector<uint64_t> vnIndivTPFN; // one count per threshold
                    std::vector<uint64_t> vnTotalTP; // one count per threshold
                    std::vector<uint64_t> vnTotalTPFP; // one count per threshold
                    const std::vector<uchar> vnThresholds; // list of thresholds
                };
                std::vector<BSDS500BasicMetrics> m_voBasicMetrics;
                const size_t m_nThresholdBins;
            protected:
                struct BSDS500Score { // edge detection score for a single threshold
                    double dThreshold;
                    double dRecall;
                    double dPrecision;
                    double dFMeasure;
                };
                struct BSDS500Metrics { // high-level metrics for an entire image set
                    std::vector<BSDS500Score> voBestImageScores; // one score per image (best threshold)
                    std::vector<BSDS500Score> voThresholdScores; // one score per threshold (cumul images)
                    BSDS500Score oBestScore; // best score for all thresholds
                    double dMaxRecall;
                    double dMaxPrecision;
                    double dMaxFMeasure;
                    double dAreaPR;
                    double dTimeElapsed_sec;
                };
                static void CalcMetrics(const WorkBatch& oBatch, BSDS500Metrics& oRes);
                static void WriteEvalResults(const DatasetInfoBase& oInfo, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups);
                static void WriteEvalResults(const WorkBatch& oBatch, BSDS500Metrics& oRes);
                static BSDS500Score FindMaxFMeasure(const std::vector<uchar>& vnThresholds, const std::vector<double>& vdRecall, const std::vector<double>& vdPrecision);
                static BSDS500Score FindMaxFMeasure(const std::vector<BSDS500Score>& voScores);
                friend class DatasetInfo;
            };

        }; //namespace Image

    }; //namespace Segm

}; //namespace DatasetUtils
