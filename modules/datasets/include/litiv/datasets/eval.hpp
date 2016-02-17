
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#define DATASETUTILS_VALIDATE_ASYNC_EVALUATORS 1

#include "litiv/datasets/metrics.hpp"

namespace litiv {

    template<eDatasetEvalList eDatasetEval>
    struct IDatasetEvaluator_;

    template<>
    struct IDatasetEvaluator_<eDatasetEval_None> : public IDataset {
        //! writes an overall evaluation report listing packet counts, seconds elapsed and algo speed
        virtual void writeEvalReport() const override;
    };

    template<>
    struct IDatasetEvaluator_<eDatasetEval_BinaryClassifier> : public IDataset {
        //! writes an overall evaluation report listing high-level binary classification metrics
        virtual void writeEvalReport() const override;
        //! accumulates overall metrics from all batch(es)
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const;
        //! calculates overall metrics from all batch(es)
        virtual IMetricsCalculatorPtr getMetrics(bool bAverage) const;
    };

    template<eDatasetEvalList eDatasetEval, eDatasetList eDataset>
    struct DatasetEvaluator_ : public IDatasetEvaluator_<eDatasetEval> {}; // no evaluation specialization by default

    template<eDatasetEvalList eDatasetEval>
    struct IDataReporter_;

    template<>
    struct IDataReporter_<eDatasetEval_None> : public virtual IDataHandler {
        //! writes an evaluation report listing packet counts, seconds elapsed and algo speed for current batch(es)
        virtual void writeEvalReport() const override ;
    protected:
        //! returns a one-line string listing packet counts, seconds elapsed and algo speed for current batch(es)
        std::string writeInlineEvalReport(size_t nIndentSize) const;
        friend struct IDatasetEvaluator_<eDatasetEval_None>;
    };

    template<>
    struct IDataReporter_<eDatasetEval_BinaryClassifier> : IDataReporter_<eDatasetEval_None> {
        //! accumulates basic metrics from current batch(es) --- provides group-impl only
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const;
        //! accumulates high-level metrics from current batch(es)
        virtual IMetricsCalculatorPtr getMetrics(bool bAverage) const;
        //! writes an evaluation report listing high-level metrics for current batch(es)
        virtual void writeEvalReport() const override;
    protected:
        //! returns a one-line string listing high-level metrics for current batch(es)
        std::string writeInlineEvalReport(size_t nIndentSize) const;
        friend struct IDatasetEvaluator_<eDatasetEval_BinaryClassifier>;
    };

    template<eDatasetEvalList eDatasetEval>
    struct IDataEvaluator_ : // no evaluation specialization by default
            public IDataReporter_<eDatasetEval>,
            public IDataConsumer_<eDatasetEval> {};

    template<>
    struct IDataEvaluator_<eDatasetEval_BinaryClassifier> :
            public IDataReporter_<eDatasetEval_BinaryClassifier>,
            public IDataConsumer_<eDatasetEval_BinaryClassifier> {
        //! overrides 'getMetricsBase' from IDataReporter_ for non-group-impl (as always required)
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const override;
        //! overrides 'push' from IDataConsumer_ to simultaneously evaluate the pushed results
        virtual void push(const cv::Mat& oClassif, size_t nIdx) override;
        //! provides a visual feedback on result quality based on evaluation guidelines
        virtual cv::Mat getColoredMask(const cv::Mat& oClassif, size_t nIdx);
        //! resets internal metrics counters to zero
        virtual void resetMetrics();
    protected:
        BinClassifMetricsAccumulatorPtr m_pMetricsBase;
    };

    template<eDatasetEvalList eDatasetEval, ParallelUtils::eParallelAlgoType eImpl>
    struct IAsyncDataEvaluator_ : // no evaluation specialization by default
            public IDataReporter_<eDatasetEval>,
            public IAsyncDataConsumer_<eDatasetEval,eImpl> {
        static_assert(eImpl!=ParallelUtils::eNonParallel,"Cannot use Async eval interface with non-parallel impl");
    };

#if HAVE_GLSL
    template<>
    struct IAsyncDataEvaluator_<eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL> :
            public IDataReporter_<eDatasetEval_BinaryClassifier>,
            public IAsyncDataConsumer_<eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL> {
        //! overrides 'getMetricsBase' from IDataReporter_ for non-group-impl (as always required)
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const override;
        //! returns the ideal size for the GL context window to use for debug display purposes (queries the algo based on dataset specs, if available)
        virtual cv::Size getIdealGLWindowSize() const override;
    protected:
        virtual void _stopProcessing() override;
        virtual void pre_initialize_gl() override;
        virtual void post_initialize_gl() override;
        virtual void pre_apply_gl(size_t nNextIdx, bool bRebindAll) override;
        virtual void post_apply_gl(size_t nNextIdx, bool bRebindAll) override;
        struct GLVideoSegmDataEvaluator : public GLImageProcEvaluatorAlgo {
            GLVideoSegmDataEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount);
            virtual std::string getComputeShaderSource(size_t nStage) const override;
            BinClassifMetricsAccumulatorPtr getMetricsBase();
        };
        std::unique_ptr<GLVideoSegmDataEvaluator> m_pEvalAlgo;
        cv::Mat m_oLastGT,m_oCurrGT,m_oNextGT;
        BinClassifMetricsAccumulatorPtr m_pMetricsBase;
    };

#endif //HAVE_GLSL

    template<eDatasetEvalList eDatasetEval, eDatasetList eDataset>
    struct DataEvaluator_ : public IDataEvaluator_<eDatasetEval> {};

    template<eDatasetEvalList eDatasetEval, eDatasetList eDataset, ParallelUtils::eParallelAlgoType eImpl>
    struct AsyncDataEvaluator_ : public IAsyncDataEvaluator_<eDatasetEval,eImpl> {};

#if 0

    namespace Image {

        namespace Segm {

            struct BSDS500BoundaryEvaluator : public IEvaluator {
                BSDS500BoundaryEvaluator(size_t nThresholdBins=DEFAULT_BSDS500_EDGE_EVAL_THRESHOLD_BINS);
                virtual cv::Mat getColoredSegmMask(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/) const;
                virtual void accumulateMetrics(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/);
                static const double s_dMaxImageDiagRatioDist;
                struct BSDS500ClassifMetricsBase { // basic eval metrics for a single image
                    BSDS500ClassifMetricsBase(size_t nThresholdsBins); // always skips zero threshold
                    std::vector<uint64_t> vnIndivTP; // one count per threshold
                    std::vector<uint64_t> vnIndivTPFN; // one count per threshold
                    std::vector<uint64_t> vnTotalTP; // one count per threshold
                    std::vector<uint64_t> vnTotalTPFP; // one count per threshold
                    const std::vector<uchar> vnThresholds; // list of thresholds
                };
                void setThresholdBins(size_t nThresholdBins);
                size_t getThresholdBins() const;
            protected:
                size_t m_nThresholdBins;
                std::vector<BSDS500ClassifMetricsBase> m_voClassifMetricsBase;
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
                };
                static void CalcMetrics(const WorkBatch& oBatch, BSDS500Metrics& oRes);
                static void writeEvalReport(const DatasetInfoBase& oInfo, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups);
                static void writeEvalReport(const WorkBatch& oBatch, BSDS500Metrics& oRes);
                static BSDS500Score FindMaxFMeasure(const std::vector<uchar>& vnThresholds, const std::vector<double>& vdRecall, const std::vector<double>& vdPrecision);
                static BSDS500Score FindMaxFMeasure(const std::vector<BSDS500Score>& voScores);
                friend struct DatasetInfo;
            };

        } //namespace Segm

    } //namespace Image

#endif

} //namespace litiv
