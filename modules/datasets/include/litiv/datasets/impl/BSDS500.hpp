
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

// note: we should already be in the litiv namespace
#ifndef __LITIV_DATASETS_IMPL_H
#error "This file should never be included directly; use litiv/datasets.hpp instead"
#endif //__LITIV_DATASETS_IMPL_H

// as defined in the BSDS500 scripts/dataset
#define DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS   99
#define DATASETS_BSDS500_EVAL_IMAGE_DIAG_RATIO_DIST 0.0075

struct BSDS500MetricsAccumulator;

enum eBSDS500DatasetGroup {
    eBSDS500Dataset_Training,
    eBSDS500Dataset_Training_Validation,
    eBSDS500Dataset_Training_Validation_Test,
};

template<>
struct DatasetEvaluator_<eDatasetEval_BinaryClassifier,eDataset_BSDS500> :
        public IDatasetEvaluator_<eDatasetEval_None> {
    //! writes an overall evaluation report listing high-level binary classification metrics
    virtual void writeEvalReport() const override;
    //! accumulates overall metrics from all batch(es)
    virtual IMetricsAccumulatorConstPtr getMetricsBase() const;
    //! calculates overall metrics from all batch(es)
    virtual IMetricsCalculatorPtr getMetrics() const;
};

template<eDatasetTaskList eDatasetTask, ParallelUtils::eParallelAlgoType eEvalImpl>
struct Dataset_<eDatasetTask,eDataset_BSDS500,eEvalImpl> :
        public IDataset_<eDatasetTask,eDatasetSource_Image,eDataset_BSDS500,getDatasetEval<eDatasetTask,eDataset_BSDS500>(),eEvalImpl> {
    static_assert(eDatasetTask!=eDatasetTask_Registr,"BSDS500 dataset does not support image registration (no image arrays)");
    static_assert(eDatasetTask!=eDatasetTask_ChgDet,"BSDS500 dataset does not support change detection (no data streaming)");
protected: // should still be protected, as creation should always be done via datasets::create
    Dataset_(
            const std::string& sOutputDirName, // output directory (full) path for debug logs, evaluation reports and results archiving (will be created in BSR dataset folder)
            bool bSaveOutput=false, // defines whether results should be archived or not
            bool bUseEvaluator=true, // defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, // defines whether data packets should be 4-byte aligned (useful for GPU upload)
            double dScaleFactor=1.0, // defines the scale factor to use to resize/rescale read packets
            eBSDS500DatasetGroup eType=eBSDS500Dataset_Training // defines which dataset groups to use
    ) :
            IDataset_<eDatasetTask,eDatasetSource_Image,eDataset_BSDS500,getDatasetEval<eDatasetTask,eDataset_BSDS500>(),eEvalImpl>(
                    "BSDS500",
                    "BSDS500/data/images",
                    std::string(DATASET_ROOT)+"/BSDS500/BSR/"+sOutputDirName+"/",
                    "",
                    ".png",
                    (eType==eBSDS500Dataset_Training)?std::vector<std::string>{"train"}:((eType==eBSDS500Dataset_Training_Validation)?std::vector<std::string>{"train","val"}:std::vector<std::string>{"train","val","test"}),
                    std::vector<std::string>{},
                    std::vector<std::string>{},
                    0,
                    bSaveOutput,
                    bUseEvaluator,
                    bForce4ByteDataAlign,
                    dScaleFactor
            ) {}
};

template<>
struct DataProducer_<eDatasetSource_Image,eDataset_BSDS500> :
        public IDataProducer_<eDatasetSource_Image> {
protected:
    //! data parsing function, dataset-specific (default parser is not satisfactory)
    virtual void parseData() override final;
    //! gt packet load function, dataset-specific (default gt loader is not satisfactory)
    virtual cv::Mat _getGTPacket_impl(size_t nIdx) override final;
};

template<>
struct DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500> :
        public IDataReporter_<eDatasetEval_None> {
    //! accumulates basic metrics from current batch(es) --- provides group-impl only
    virtual IMetricsAccumulatorConstPtr getMetricsBase() const;
    //! accumulates high-level metrics from current batch(es)
    virtual IMetricsCalculatorPtr getMetrics() const;
    //! writes an evaluation report listing high-level metrics for current batch(es)
    virtual void writeEvalReport() const override;
protected:
    //! returns a one-line string listing high-level metrics for current batch(es)
    std::string writeInlineEvalReport(size_t nIndentSize) const;
    //! required so that dataset-level evaluation report can write dataset-specific reports
    friend struct DatasetEvaluator_<eDatasetEval_BinaryClassifier,eDataset_BSDS500>;
};

template<>
struct DataEvaluator_<eDatasetEval_BinaryClassifier,eDataset_BSDS500,ParallelUtils::eNonParallel> :
        public IDataConsumer_<eDatasetEval_BinaryClassifier>,
        public DataReporter_<eDatasetEval_BinaryClassifier,eDataset_BSDS500> {
public:
    //! overrides 'getMetricsBase' from IDataReporter_ for non-group-impl (as always required)
    virtual IMetricsAccumulatorConstPtr getMetricsBase() const;
    //! overrides 'push' from IDataConsumer_ to simultaneously evaluate the pushed results
    virtual void push(const cv::Mat& oClassif, size_t nIdx) override;
    //! provides a visual feedback on result quality based on evaluation guidelines
    virtual cv::Mat getColoredMask(const cv::Mat& oClassif, size_t nIdx);
    //! resets internal metrics counters to zero
    virtual void resetMetrics();
protected:
    std::shared_ptr<BSDS500MetricsAccumulator> m_pMetricsBase;
};
