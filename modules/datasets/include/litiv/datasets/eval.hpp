
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

#define DATASETUTILS_VALIDATE_ASYNC_EVALUATORS 0

#include "litiv/datasets/metrics.hpp"

namespace lv {

    /// dataset evaluator interface for top-level metrics computation & report writing
    template<DatasetEvalList eDatasetEval>
    struct IDatasetEvaluator_;

    template<>
    struct IDatasetEvaluator_<DatasetEval_None> : public IDataset {
        /// writes an overall evaluation report listing packet counts, seconds elapsed and algo speed (default eval)
        virtual void writeEvalReport() const override;
    };

    template<>
    struct IDatasetEvaluator_<DatasetEval_BinaryClassifier> : IDatasetEvaluator_<DatasetEval_None> {
        /// writes an overall evaluation report listing high-level binary classification metrics
        virtual void writeEvalReport() const override;
        /// accumulates overall metrics from all batch(es)
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const;
        /// calculates overall metrics from all batch(es)
        virtual IMetricsCalculatorPtr getMetrics(bool bAverage) const;
    };

    /// default dataset evaluator interface specialization (will use 'non eval' report, by default)
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DatasetEvaluator_ : public IDatasetEvaluator_<eDatasetEval> {};

    /// data reporter interface for work batch that must be specialized based on eval type
    template<DatasetEvalList eDatasetEval>
    struct IDataReporter_;

    template<>
    struct IDataReporter_<DatasetEval_None> : public virtual IDataHandler {
        /// writes an evaluation report listing packet counts, seconds elapsed and algo speed for current batch(es)
        virtual void writeEvalReport() const override;
    protected:
        /// returns a one-line string listing packet counts, seconds elapsed and algo speed for current batch(es)
        virtual std::string writeInlineEvalReport(size_t nIndentSize) const;
        friend struct IDatasetEvaluator_<DatasetEval_None>;
    };

    template<>
    struct IDataReporter_<DatasetEval_BinaryClassifier> : IDataReporter_<DatasetEval_None> {
        /// accumulates basic metrics from current batch(es) --- provides group-impl only
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const;
        /// accumulates high-level metrics from current batch(es)
        virtual IMetricsCalculatorPtr getMetrics(bool bAverage) const;
        /// writes an evaluation report listing high-level metrics for current batch(es)
        virtual void writeEvalReport() const override;
    protected:
        /// returns a one-line string listing high-level metrics for current batch(es)
        virtual std::string writeInlineEvalReport(size_t nIndentSize) const override;
        friend struct IDatasetEvaluator_<DatasetEval_BinaryClassifier>;
    };

    /// default data reporter interface specialization
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DataReporter_ : public IDataReporter_<eDatasetEval> {};

    /// default data evaluator interface specialization (will also determine which consumer interf to use based on eval impl)
    template<DatasetEvalList eDatasetEval, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl>
    struct DataEvaluator_ : // no evaluation specialization by default
            public std::conditional<(eEvalImpl==lv::NonParallel),IDataConsumer_<eDatasetEval>,IAsyncDataConsumer_<eDatasetEval,eEvalImpl>>::type,
            public DataReporter_<eDatasetEval,eDataset> {};

    template<DatasetList eDataset>
    struct DataEvaluator_<DatasetEval_BinaryClassifier,eDataset,lv::NonParallel> :
            public IDataConsumer_<DatasetEval_BinaryClassifier>,
            public DataReporter_<DatasetEval_BinaryClassifier,eDataset> {
        /// overrides 'getMetricsBase' from IDataReporter_ for non-group-impl (as always required)
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const override {
            if(!m_pMetricsBase)
                return BinClassifMetricsAccumulator::create();
            return m_pMetricsBase;
        }
        /// overrides 'push' from IDataConsumer_ to simultaneously evaluate the pushed results
        virtual void push(const cv::Mat& oClassif, size_t nIdx) override {
            IDataConsumer_<DatasetEval_BinaryClassifier>::push(oClassif,nIdx);
            if(getDatasetInfo()->isUsingEvaluator()) {
                auto pLoader = shared_from_this_cast<IDataLoader>(true);
                if(!m_pMetricsBase)
                    m_pMetricsBase = BinClassifMetricsAccumulator::create();
                m_pMetricsBase->accumulate(oClassif,pLoader->getGT(nIdx),pLoader->getInputROI(nIdx));
            }
        }
        /// provides a visual feedback on result quality based on evaluation guidelines
        virtual cv::Mat getColoredMask(const cv::Mat& oClassif, size_t nIdx) {
            auto pLoader = shared_from_this_cast<IDataLoader>(true);
            return BinClassifMetricsAccumulator::getColoredMask(oClassif,pLoader->getGT(nIdx),pLoader->getInputROI(nIdx));
        }
        /// resets internal metrics counters to zero
        virtual void resetMetrics() {
            m_pMetricsBase = BinClassifMetricsAccumulator::create();
        }
    protected:
        BinClassifMetricsAccumulatorPtr m_pMetricsBase;
    };

#if HAVE_GLSL

    /// basic 2D binary classifier evaluator algo interface
    struct GLBinaryClassifierEvaluator : public GLImageProcEvaluatorAlgo {
        GLBinaryClassifierEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount);
        virtual std::string getComputeShaderSource(size_t nStage) const override;
        BinClassifMetricsAccumulatorPtr getMetricsBase();
    };

    template<DatasetList eDataset>
    struct DataEvaluator_<DatasetEval_BinaryClassifier,eDataset,lv::GLSL> :
            public IAsyncDataConsumer_<DatasetEval_BinaryClassifier,lv::GLSL>,
            public DataReporter_<DatasetEval_BinaryClassifier,eDataset> {
        /// overrides 'getMetricsBase' from IDataReporter_ for non-group-impl (as always required)
        virtual IMetricsAccumulatorConstPtr getMetricsBase() const override {
            if(isProcessing())
                lvError("Must stop processing batch before querying metrics under async data evaluator interface");
            else if(!m_pMetricsBase)
                return BinClassifMetricsAccumulator::create();
            return m_pMetricsBase;
        }
    protected:
        /// overrides '_stopProcessing' from IDataHandler to make sure accumulated metrics are fetched from gpu once processing is done
        virtual void _stopProcessing() override {
            if(m_pEvalAlgo && m_pEvalAlgo->getIsGLInitialized()) {
                auto pEvalAlgo = std::dynamic_pointer_cast<GLBinaryClassifierEvaluator>(m_pEvalAlgo);
                lvAssert(pEvalAlgo);
                BinClassifMetricsAccumulatorPtr pMetricsBase = pEvalAlgo->getMetricsBase();
                lvAssert(!DATASETUTILS_VALIDATE_ASYNC_EVALUATORS || !m_pMetricsBase || m_pMetricsBase->isEqual(pMetricsBase));
                m_pMetricsBase = pMetricsBase;
            }
        }
        /// overrides 'post_initialize_gl' from IAsyncDataConsumer_ to initialize an evaluation algo interface
        virtual void post_initialize_gl() override {
            IAsyncDataConsumer_<DatasetEval_BinaryClassifier,lv::GLSL>::post_initialize_gl();
            if(getDatasetInfo()->isUsingEvaluator()) {
                m_pEvalAlgo = std::make_shared<GLBinaryClassifierEvaluator>(m_pAlgo,getTotPackets());
                m_pEvalAlgo->initialize_gl(m_oCurrGT,m_pLoader->getInputROI(m_nCurrIdx));
                m_pMetricsBase = BinClassifMetricsAccumulator::create();
                if(DATASETUTILS_VALIDATE_ASYNC_EVALUATORS) {
                    using namespace std::placeholders;
                    m_lDataCallback = std::bind(&DataEvaluator_<DatasetEval_BinaryClassifier,eDataset,lv::GLSL>::validationCallback,this,_1,_2,_3,_4,_5,_6);
                    m_pAlgo->setOutputFetching(true);
                }
                if(m_pAlgo->m_pDisplayHelper)
                    m_pEvalAlgo->setOutputFetching(true);
                if(m_pAlgo->m_pDisplayHelper && m_pEvalAlgo->m_bUsingDebug)
                    m_pEvalAlgo->setDebugFetching(true);
            }
        }
        /// callback entrypoint for gpu-cpu evaluation validation
        void validationCallback(const cv::Mat& /*oInput*/, const cv::Mat& /*oDebug*/, const cv::Mat& oOutput, const cv::Mat& oGT, const cv::Mat& oROI, size_t /*nIdx*/) {
            lvAssert(m_pMetricsBase && !oOutput.empty() && !oGT.empty());
            m_pMetricsBase->accumulate(oOutput,oGT,oROI);
        }
        BinClassifMetricsAccumulatorPtr m_pMetricsBase;
    };

#endif //HAVE_GLSL

} // namespace lv
