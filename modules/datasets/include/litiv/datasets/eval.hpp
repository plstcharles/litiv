
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

    /// dataset reporter interface forward declaration
    template<DatasetEvalList eDatasetEval>
    struct IDatasetReporter_;

    /// dataset reporter specialization forward declaration
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DatasetReporter_;

    /// metric retriever interface; exposes utility functions to recursively parse metrics through all work batches
    template<typename IDataInterface>
    struct IMetricRetriever_ : public virtual IDataInterface {
        /// returns whether this data handler defines some evaluation procedure or not (always true for this interface)
        virtual bool isEvaluable() const override final {return true;}
        /// accumulates and returns a sum of all base evaluation metrics for the dataset (e.g. sums all classification counters)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const = 0;
        /// accumulates and returns high-level evaluation metrics for the dataset (e.g. computes F-Measure from classification counters)
        virtual IIMetricsCalculatorPtr getMetrics(bool bAverage) const = 0;
    };

    /// data reporter interface forward declaration
    template<DatasetEvalList eDatasetEval>
    struct IDataReporter_;

    /// data reporter specialization for no-eval work batch report writing
    template<>
    struct IDataReporter_<DatasetEval_None> : public virtual IDataHandler {
        /// writes an evaluation report listing packet counts, seconds elapsed and algo speed for current batch(es)
        virtual void writeEvalReport() const override;
    protected:
        /// returns a one-line string listing packet counts, seconds elapsed and algo speed for current batch(es)
        virtual std::string writeInlineEvalReport(size_t nIndentSize) const;
        friend struct IDatasetReporter_<DatasetEval_None>;
    };

    /// data reporter specialization for binary classification work batch report writing
    template<>
    struct IDataReporter_<DatasetEval_BinaryClassifier> :
            public IDataReporter_<DatasetEval_None>,
            protected IMetricRetriever_<IDataHandler> {
        /// writes an evaluation report listing high-level metrics for current batch(es)
        virtual void writeEvalReport() const override;
    protected:
        /// returns a one-line string listing high-level metrics for current batch(es)
        virtual std::string writeInlineEvalReport(size_t nIndentSize) const override;
        friend struct IDatasetReporter_<DatasetEval_BinaryClassifier>;
    };

    /// data reporter specialization for binary classification array work batch report writing
    template<>
    struct IDataReporter_<DatasetEval_BinaryClassifierArray> :
            public IDataReporter_<DatasetEval_None>,
            protected IMetricRetriever_<IDataHandler> {
        /// writes an evaluation report listing high-level metrics for current batch(es)
        virtual void writeEvalReport() const override;
    protected:
        /// returns a one-line string listing high-level metrics for current batch(es)
        virtual std::string writeInlineEvalReport(size_t nIndentSize) const override;
        friend struct IDatasetReporter_<DatasetEval_BinaryClassifierArray>;
    };

    /// data reporter wrapper forward declaration
    template<DatasetEvalList eDatasetEval, DatasetList eDataset, typename ENABLE=void>
    struct DataReporterWrapper_;

    /// data reporter wrapper specialization, pass-through in the case of non-eval
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DataReporterWrapper_<eDatasetEval,eDataset,std::enable_if_t<eDatasetEval==DatasetEval_None>> :
            public IDataReporter_<eDatasetEval> {};

    /// data reporter wrapper specialization, used to simplify getMetricsBase() and getMetrics() recursive calls for specialized templated types
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DataReporterWrapper_<eDatasetEval,eDataset,std::enable_if_t<eDatasetEval!=DatasetEval_None>> :
            public IDataReporter_<eDatasetEval> {
    protected:
        /// accumulates and returns a sum of all base evaluation metrics for children batches (provides group-impl only)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const override {
            lvAssert_(this->isGroup(),"non-group data reporter specialization attempt to call non-overridden method");
            IIMetricsAccumulatorPtr pMetricsBase = IIMetricsAccumulator::create<MetricsAccumulator_<eDatasetEval,eDataset>>();
            for(const auto& pBatch : this->getBatches(true))
                pMetricsBase->accumulate(dynamic_cast<const IMetricRetriever_<IDataHandler>&>(*pBatch).getMetricsBase());
            return pMetricsBase;
        }
        /// accumulates and returns high-level evaluation metrics
        virtual IIMetricsCalculatorPtr getMetrics(bool bAverage) const override {
            if(bAverage && this->isGroup() && !this->isBare()) {
                IDataHandlerPtrArray vpBatches = this->getBatches(true);
                auto ppBatchIter = vpBatches.begin();
                for(; ppBatchIter!=vpBatches.end() && (*ppBatchIter)->getCurrentOutputCount()==0; ++ppBatchIter);
                lvAssert_(ppBatchIter!=vpBatches.end(),"found no processed output packets");
                IIMetricsCalculatorPtr pMetrics = dynamic_cast<const IMetricRetriever_<IDataHandler>&>(**ppBatchIter).getMetrics(bAverage);
                for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
                    if((*ppBatchIter)->getCurrentOutputCount()>0)
                        pMetrics->accumulate(dynamic_cast<const IMetricRetriever_<IDataHandler>&>(**ppBatchIter).getMetrics(bAverage));
                return pMetrics;
            }
            return IIMetricsCalculator::create<MetricsCalculator_<eDatasetEval,eDataset>>(getMetricsBase());
        }
        friend struct DatasetReporter_<eDatasetEval,eDataset>;
    };

    /// data reporter full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DataReporter_ : public DataReporterWrapper_<eDatasetEval,eDataset> {};

    /// data evaluator interface; expose function to reset internal metrics (by default, only packet counts)


    /// data evaluator full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetEvalList eDatasetEval, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl>
    struct DataEvaluator_ : // no evaluation specialization by default
            public std::conditional<(eEvalImpl==lv::NonParallel),IDataConsumer_<eDatasetEval>,IAsyncDataConsumer_<eDatasetEval,eEvalImpl>>::type,
            public DataReporter_<eDatasetEval,eDataset> {};

    /// data evaluator specialization for binary classification work batch performance evaluation
    template<DatasetList eDataset>
    struct DataEvaluator_<DatasetEval_BinaryClassifier,eDataset,lv::NonParallel> :
            public IDataConsumer_<DatasetEval_BinaryClassifier>,
            public DataReporter_<DatasetEval_BinaryClassifier,eDataset> {
        /// provides a visual feedback on result quality based on evaluation guidelines
        virtual cv::Mat getColoredMask(const cv::Mat& oClassif, size_t nIdx) {
            lvAssert_(!oClassif.empty(),"output must be non-empty for display");
            auto pLoader = shared_from_this_cast<IIDataLoader>(true);
            lvAssert_(pLoader->getOutputPacketType()==ImagePacket && pLoader->getGTPacketType()==ImagePacket && pLoader->getGTMappingType()==PixelMapping,"default impl cannot display mask without 1:1 image pixel mapping");
            return BinClassif::getColoredMask(oClassif,pLoader->getGT(nIdx),pLoader->getGTROI(nIdx));
        }
        /// resets internal packet count + classification metrics
        virtual void resetMetrics() override {
            IDataConsumer_<DatasetEval_BinaryClassifier>::resetMetrics();
            m_pMetricsBase = IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifier,eDataset>>();
        }
    protected:
        /// overrides 'getMetricsBase' from IMetricRetriever_ for non-group-impl (as always required)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const override {
            return m_pMetricsBase;
        }
        /// overrides 'processOutput' from IDataConsumer_ to evaluate the provided output packet
        virtual void processOutput(const cv::Mat& oClassif, size_t nIdx) override {
            if(getDatasetInfo()->isUsingEvaluator()) {
                lvAssert_(!oClassif.empty(),"output must be non-empty for evaluation");
                auto pLoader = shared_from_this_cast<IIDataLoader>(true);
                lvAssert_(pLoader->getOutputPacketType()==ImagePacket && pLoader->getGTPacketType()==ImagePacket && pLoader->getGTMappingType()==PixelMapping,"default impl cannot evaluate without 1:1 image pixel mapping");
                m_pMetricsBase->m_oCounters.accumulate(oClassif,pLoader->getGT(nIdx),pLoader->getGTROI(nIdx));
            }
        }
        /// default constructor; automatically creates an instance of the base metrics accumulator object
        inline DataEvaluator_() : m_pMetricsBase(IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifier,eDataset>>()) {}
        /// contains low-level metric accumulation logic
        BinClassifMetricsAccumulatorPtr m_pMetricsBase;
    };

    /// data evaluator specialization for binary classification array work batch performance evaluation
    template<DatasetList eDataset>
    struct DataEvaluator_<DatasetEval_BinaryClassifierArray,eDataset,lv::NonParallel> :
            public IDataConsumer_<DatasetEval_BinaryClassifierArray>,
            public DataReporter_<DatasetEval_BinaryClassifierArray,eDataset> {
        /// provides a visual feedback on result quality based on evaluation guidelines
        virtual std::vector<cv::Mat> getColoredMaskArray(const std::vector<cv::Mat>& vClassif, size_t nIdx) {
            lvAssert_(!vClassif.empty(),"output array must be non-empty for display");
            auto pLoader = shared_from_this_cast<IDataLoader_<Array>>(true);
            lvAssert_(pLoader->getOutputPacketType()==ImageArrayPacket && pLoader->getGTPacketType()==ImageArrayPacket && pLoader->getGTMappingType()==PixelMapping,"default impl cannot display mask without 1:1 image pixel mapping");
            std::vector<cv::Mat> vMasks;
            const std::vector<cv::Mat>& vGTArray = pLoader->getGTArray(nIdx);
            const std::vector<cv::Mat>& vGTROIArray = pLoader->getGTROIArray(nIdx);
            lvAssert_(vClassif.size()==vGTArray.size() && (vGTROIArray.empty() || vClassif.size()==vGTROIArray.size()),"array size mistmatch");
            for(size_t s=0; s<vClassif.size(); ++s)
                vMasks.push_back(BinClassif::getColoredMask(vClassif[s],vGTArray[s],vGTROIArray.empty()?cv::Mat():vGTROIArray[s]));
            return vMasks;
        }
        /// resets internal packet count + classification metrics
        virtual void resetMetrics() override {
            IDataConsumer_<DatasetEval_BinaryClassifierArray>::resetMetrics();
            m_pMetricsBase = IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifierArray,eDataset>>();
        }
    protected:
        /// overrides 'getMetricsBase' from IMetricRetriever_ for non-group-impl (as always required)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const override {
            return m_pMetricsBase;
        }
        /// overrides 'processOutput' from IDataConsumer_ to evaluate the provided output packet
        virtual void processOutput(const std::vector<cv::Mat>& vClassif, size_t nIdx) override {
            if(getDatasetInfo()->isUsingEvaluator()) {
                lvAssert_(!vClassif.empty(),"output array must be non-empty for evaluation");
                auto pLoader = shared_from_this_cast<IDataLoader_<Array>>(true);
                lvAssert_(pLoader->getOutputPacketType()==ImageArrayPacket && pLoader->getGTPacketType()==ImageArrayPacket && pLoader->getGTMappingType()==PixelMapping,"default impl cannot evaluate without 1:1 image pixel mapping");
                lvAssert_(pLoader->getGTStreamCount()==getOutputStreamCount() && vClassif.size()==getOutputStreamCount(),"gt/output array size mismatch");
                const std::vector<cv::Mat>& vGTArray = pLoader->getGTArray(nIdx);
                const std::vector<cv::Mat>& vGTROIArray = pLoader->getGTROIArray(nIdx);
                lvAssert_(vClassif.size()==vGTArray.size() && (vGTROIArray.empty() || vClassif.size()==vGTROIArray.size()),"gt/output array size mistmatch");
                for(size_t s=0; s<vClassif.size(); ++s)
                    m_pMetricsBase->m_vCounters[s].accumulate(vClassif[s],vGTArray[s],vGTROIArray.empty()?cv::Mat():vGTROIArray[s]);
            }
        }
        /// default constructor; automatically creates an instance of the base metrics accumulator object
        inline DataEvaluator_() : m_pMetricsBase(IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifierArray,eDataset>>()) {}
        /// contains low-level metric accumulation logic
        BinClassifMetricsArrayAccumulatorPtr m_pMetricsBase;
    };

#if HAVE_GLSL

    /// basic 2D binary classifier evaluator algo interface
    struct GLBinaryClassifierEvaluator : public GLImageProcEvaluatorAlgo {
        GLBinaryClassifierEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount);
        virtual std::string getComputeShaderSource(size_t nStage) const override;
        BinClassifMetricsAccumulatorPtr getMetricsBase();
    };

    /// data evaluator specialization for binary classification work batch performance evaluation through async (GLSL) algo interface
    template<DatasetList eDataset>
    struct DataEvaluator_<DatasetEval_BinaryClassifier,eDataset,lv::GLSL> :
            public IAsyncDataConsumer_<DatasetEval_BinaryClassifier,lv::GLSL>,
            public DataReporter_<DatasetEval_BinaryClassifier,eDataset> {
        /// resets internal packet count + classification metrics
        virtual void resetMetrics() override {
            IAsyncDataConsumer_<DatasetEval_BinaryClassifier,lv::GLSL>::resetMetrics();
            // ... @@@@ reset glsl eval? need a 'setEvaluationAtomicCounterBuffer' function
            m_pMetricsBase = IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifier,eDataset>>();
        }
    protected:
        /// overrides 'getMetricsBase' from IMetricRetriever_ for non-group-impl (as always required)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const override {
            if(isProcessing())
                lvError("Must stop processing batch before querying metrics under async data evaluator interface");
            return m_pMetricsBase;
        }
        /// overrides '_stopProcessing' from IDataHandler to make sure accumulated metrics are fetched from gpu once processing is done
        virtual void stopProcessing_impl() override {
            if(m_pEvalAlgo && m_pEvalAlgo->getIsGLInitialized()) {
                auto pEvalAlgo = std::dynamic_pointer_cast<GLBinaryClassifierEvaluator>(m_pEvalAlgo);
                lvAssert_(pEvalAlgo,"evaluation algo did not have a GLBinaryClassifierEvaluator interface");
                BinClassifMetricsAccumulatorPtr pMetricsBase = pEvalAlgo->getMetricsBase();
                lvAssert_(!DATASETUTILS_VALIDATE_ASYNC_EVALUATORS || !m_pMetricsBase || m_pMetricsBase->isEqual(pMetricsBase),"gpu evaluation algo did not return same results as cpu evaluation");
                m_pMetricsBase = pMetricsBase;
            }
        }
        /// overrides 'post_initialize_gl' from IAsyncDataConsumer_ to initialize an evaluation algo interface
        virtual void post_initialize_gl() override {
            IAsyncDataConsumer_<DatasetEval_BinaryClassifier,lv::GLSL>::post_initialize_gl();
            if(getDatasetInfo()->isUsingEvaluator()) {
                lvAssert_(m_pLoader->getExpectedOutputCount()>0,"need predetermined limit on eval count");
                m_pEvalAlgo = std::make_shared<GLBinaryClassifierEvaluator>(m_pAlgo,m_pLoader->getExpectedOutputCount());
                m_pEvalAlgo->initialize_gl(m_oCurrGT,m_pLoader->getGTROI(m_nCurrIdx));
                m_pMetricsBase = IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifier,eDataset>>();
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
        void validationCallback(const cv::Mat& /*oInput*/, const cv::Mat& /*oDebug*/, const cv::Mat& oOutput, const cv::Mat& oGT, const cv::Mat& oGTROI, size_t /*nIdx*/) {
            lvAssert_(m_pMetricsBase,"algo needs to be initialized first")
            lvAssert_(!oOutput.empty() && !oGT.empty(),"provided output and gt mats need to be non-empty");
            m_pMetricsBase->m_oCounters.accumulate(oOutput,oGT,oGTROI);
        }
        /// default constructor; automatically creates an instance of the base metrics accumulator object
        inline DataEvaluator_() : m_pMetricsBase(IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifier,eDataset>>()) {}
        BinClassifMetricsAccumulatorPtr m_pMetricsBase;
    };

#endif //HAVE_GLSL

    /// dataset reporter specialization for no-eval top-level report writing
    template<>
    struct IDatasetReporter_<DatasetEval_None> : public virtual IDataset {
        /// writes an overall evaluation report listing packet counts, seconds elapsed and algo speed (default eval)
        virtual void writeEvalReport() const override;
    };

    /// dataset reporter specialization for binary classification top-level report writing
    template<>
    struct IDatasetReporter_<DatasetEval_BinaryClassifier> :
            public IDatasetReporter_<DatasetEval_None>,
            protected IMetricRetriever_<IDataset> {
        /// writes an overall evaluation report listing high-level binary classification metrics
        virtual void writeEvalReport() const override;
    };

    /// dataset reporter specialization for binary classification array top-level report writing
    template<>
    struct IDatasetReporter_<DatasetEval_BinaryClassifierArray> :
            public IDatasetReporter_<DatasetEval_None>,
            protected IMetricRetriever_<IDataset> {
        /// writes an overall evaluation report listing high-level binary classification metrics
        virtual void writeEvalReport() const override;
    };

    /// dataset reporter wrapper forward declaration
    template<DatasetEvalList eDatasetEval, DatasetList eDataset, typename ENABLE=void>
    struct DatasetReporterWrapper_;

    /// dataset reporter wrapper specialization, pass-through in the case of non-eval
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DatasetReporterWrapper_<eDatasetEval,eDataset,std::enable_if_t<eDatasetEval==DatasetEval_None>> :
            public IDatasetReporter_<eDatasetEval> {};

    /// dataset reporter wrapper specialization, used to simplify getMetricsBase() and getMetrics() recursive calls for specialized templated types
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DatasetReporterWrapper_<eDatasetEval,eDataset,std::enable_if_t<eDatasetEval!=DatasetEval_None>> :
            public IDatasetReporter_<eDatasetEval> {
    protected:
        /// accumulates and returns a sum of all base evaluation metrics for the dataset (e.g. sums all classification counters)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const override {
            IIMetricsAccumulatorPtr pMetricsBase = IIMetricsAccumulator::create<MetricsAccumulator_<eDatasetEval,eDataset>>();
            for(const auto& pBatch : this->getBatches(true)) {
                auto pRet = dynamic_cast<IMetricRetriever_<IDataHandler>*>(pBatch.get());
                lvAssert(pRet);
                pMetricsBase->accumulate(pRet->getMetricsBase());
            }
            return pMetricsBase;
        }
        /// accumulates and returns high-level evaluation metrics for the dataset (e.g. computes F-Measure from classification counters)
        virtual IIMetricsCalculatorPtr getMetrics(bool bAverage) const override {
            if(bAverage) {
                IDataHandlerPtrArray vpBatches = this->getBatches(true);
                auto ppBatchIter = vpBatches.begin();
                for(; ppBatchIter!=vpBatches.end() && (*ppBatchIter)->getCurrentOutputCount()==0; ++ppBatchIter);
                lvAssert_(ppBatchIter!=vpBatches.end(),"found no processed output packets");
                IIMetricsCalculatorPtr pMetrics = dynamic_cast<const IMetricRetriever_<IDataHandler>&>(**ppBatchIter).getMetrics(bAverage);
                for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
                    if((*ppBatchIter)->getCurrentOutputCount()>0)
                        pMetrics->accumulate(dynamic_cast<const IMetricRetriever_<IDataHandler>&>(**ppBatchIter).getMetrics(bAverage));
                return pMetrics;
            }
            return IIMetricsCalculator::create<MetricsCalculator_<eDatasetEval,eDataset>>(getMetricsBase());
        }
    };

    /// dataset reporter full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct DatasetReporter_ :
            public DatasetReporterWrapper_<eDatasetEval,eDataset> {};

} // namespace lv
