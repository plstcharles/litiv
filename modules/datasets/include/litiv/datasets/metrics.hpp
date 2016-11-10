
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

#include "litiv/datasets/utils.hpp"

// as defined in the 2012/2014 CDNet evaluation scripts
#define DATASETUTILS_POSITIVE_VAL    uchar(255)
#define DATASETUTILS_NEGATIVE_VAL    uchar(0)
#define DATASETUTILS_OUTOFSCOPE_VAL  uchar(85)
#define DATASETUTILS_UNKNOWN_VAL     uchar(170)
#define DATASETUTILS_SHADOW_VAL      uchar(50)

namespace lv {

    // classfication counters list for binary classifiers (not all counters have to be used)
    struct BinClassif {
        uint64_t nTP; ///< 'true positive' count
        uint64_t nTN; ///< 'true negative' count
        uint64_t nFP; ///< 'false positive' count
        uint64_t nFN; ///< 'false negative' count
        uint64_t nSE; ///< 'special error' count (usage is optional -- useful for e.g. shadow error)
        uint64_t nDC; ///< 'dont care' count (usage is optional -- useful for e.g. unknowns or small ROIs)
        /// counter list used for indexing packed array (useful for aligned gpu upload/download)
        enum CountersList {
            Counter_TP,
            Counter_TN,
            Counter_FP,
            Counter_FN,
            Counter_SE,
            Counter_DC,
            nCountersCount,
        };
        /// returns the total classification count (without 'dont cares', by default)
        inline uint64_t total(bool bWithDontCare=false) const {
            return nTP+nTN+nFP+nFN+(bWithDontCare?nDC:uint64_t(0));
        }
        /// returns whether all internal classifcation counts are equal to those of 'c'
        inline bool isEqual(const BinClassif& c) const {
            return (this->nTP==c.nTP)&&(this->nTN==c.nTN)&&(this->nFP==c.nFP)&&(this->nFN==c.nFN)&&(this->nSE==c.nSE)&&(this->nDC==c.nDC);
        }
        /// accumulates the classification counts of 'c' into the internal counts
        inline void accumulate(const BinClassif& c) {
            nTP += c.nTP; nTN += c.nTN; nFN += c.nFN; nFP += c.nFP; nSE += c.nSE; nDC += c.nDC;
        }
        /// accumulates the pixel-level classification counts of 'oClassif' vs 'oGT' into the internal counts
        void accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI=cv::Mat());
        /// returns a colored classification mask for visualization based on good/bad classifcations of 'oClassif' vs 'oGT'
        static cv::Mat getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI=cv::Mat());
        /// default constructor; sets all counters to zero
        inline BinClassif() : nTP(0),nTN(0),nFP(0),nFN(0),nSE(0),nDC(0) {}
    };

    /// basic metrics accumulator super-interface
    struct IIMetricsAccumulator : lv::enable_shared_from_this<IIMetricsAccumulator> {
        /// virtual destructor for adequate cleanup from IIMetricsAccumulator pointers
        virtual ~IIMetricsAccumulator() = default;
        /// returns the boolean comparison of internal metrics, redirects to overridden function 'isEqual'
        inline bool operator!=(const IIMetricsAccumulator& m) const {
            return !isEqual(m.shared_from_this());
        }
        /// returns the boolean comparison of internal metrics, redirects to overridden function 'isEqual'
        inline bool operator==(const IIMetricsAccumulator& m) const {
            return isEqual(m.shared_from_this());
        }
        /// accumulates internal metrics, redirects to overridden function 'accumulate'
        inline IIMetricsAccumulator& operator+=(const IIMetricsAccumulator& m) {
            return *accumulate(m.shared_from_this());
        }
        /// returns whether the metrics/counters of 'm' are equal to those of this object
        virtual bool isEqual(const std::shared_ptr<const IIMetricsAccumulator>& m) const = 0;
        /// accumulates the metrics/counters of 'm' into those of this object
        virtual std::shared_ptr<IIMetricsAccumulator> accumulate(const std::shared_ptr<const IIMetricsAccumulator>& m) = 0;
        /// returns a new instance of type 'MetricsAccumulator', passing 'args' to its constructor
        template<typename MetricsAccumulator, typename... Targs>
        static std::enable_if_t<std::is_base_of<IIMetricsAccumulator,MetricsAccumulator>::value,std::shared_ptr<MetricsAccumulator>> create(Targs&&... args) {
            struct MetricsAccumulatorWrapper : public MetricsAccumulator {
                MetricsAccumulatorWrapper(Targs&&... _args) : MetricsAccumulator(std::forward<Targs>(_args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            return std::make_shared<MetricsAccumulatorWrapper>(std::forward<Targs>(args)...);
        }
    protected:
        /// derived metrics accumulators should be created via the static 'create' function
        IIMetricsAccumulator() = default;
    };
    using IIMetricsAccumulatorPtr = std::shared_ptr<IIMetricsAccumulator>;
    using IIMetricsAccumulatorConstPtr = std::shared_ptr<const IIMetricsAccumulator>;

    /// default (specializable) forward declaration of the metrics accumulator interface
    template<DatasetEvalList eDatasetEval>
    struct IMetricsAccumulator_;

    /// basic metrics accumulator specialized to evaluate 2d binary classifiers
    template<>
    struct IMetricsAccumulator_<DatasetEval_BinaryClassifier> :
            public IIMetricsAccumulator {
        /// returns whether the binary classification counters of 'm' are equal to those of this object
        virtual bool isEqual(const IIMetricsAccumulatorConstPtr& m) const override;
        /// accumulates the binary classification counters of 'm' into those of this object
        virtual IIMetricsAccumulatorPtr accumulate(const IIMetricsAccumulatorConstPtr& m) override;
        /// contains the actual counters used for binary classification evaluation
        BinClassif m_oCounters;
    protected:
        /// default constructor (must stay protected)
        IMetricsAccumulator_() = default;
    };
    using BinClassifMetricsAccumulator = IMetricsAccumulator_<DatasetEval_BinaryClassifier>;
    using BinClassifMetricsAccumulatorPtr = std::shared_ptr<BinClassifMetricsAccumulator>;
    using BinClassifMetricsAccumulatorConstPtr = std::shared_ptr<const BinClassifMetricsAccumulator>;

    /// basic metrics accumulator specialized to evaluate an array of 2d binary classifiers
    template<>
    struct IMetricsAccumulator_<DatasetEval_BinaryClassifierArray> :
            public IIMetricsAccumulator {
        /// returns whether the binary classification counters of 'm' are equal to those of this object (ignores stream names)
        virtual bool isEqual(const IIMetricsAccumulatorConstPtr& m) const override;
        /// accumulates the binary classification counters of 'm' into those of this object
        virtual IIMetricsAccumulatorPtr accumulate(const IIMetricsAccumulatorConstPtr& m) override;
        /// sum-reduces the binary classification counters of this object into those of a 'BinClassifMetricsAccumulator' object
        virtual BinClassifMetricsAccumulatorPtr reduce() const;
        /// contains the actual counters used for binary classification evaluation
        std::vector<BinClassif> m_vCounters;
        /// contains the stream names used for printing eval reports (should be the same size as m_vCounters)
        std::vector<std::string> m_vsStreamNames;
    protected:
        /// default constructor; resizes the counters array based on array size
        IMetricsAccumulator_(size_t nArraySize=0);
    };
    using BinClassifMetricsArrayAccumulator = IMetricsAccumulator_<DatasetEval_BinaryClassifierArray>;
    using BinClassifMetricsArrayAccumulatorPtr = std::shared_ptr<BinClassifMetricsArrayAccumulator>;
    using BinClassifMetricsArrayAccumulatorConstPtr = std::shared_ptr<const BinClassifMetricsArrayAccumulator>;

    /// basic metrics accumulator full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct MetricsAccumulator_ : public IMetricsAccumulator_<eDatasetEval> {
    protected:
        /// fowards the constructors of the underlying interface (useful if they have mandatory parameters)
        using IMetricsAccumulator_<eDatasetEval>::IMetricsAccumulator_;
    };

    // classfication metrics list for binary classifiers (not all metrics have to be used)
    struct BinClassifMetrics {
        double dRecall; ///< 'recall'/'sensitivity' value [0,1], where 1 is best
        double dSpecificity; ///< 'specificity' (1-FPR) value [0,1], where 1 is best
        double dFPR; ///< 'false positive rate' value [0,1], where 0 is best
        double dFNR; ///< 'false negative rate' value [0,1], where 0 is best
        double dPBC; ///< 'percentage of bad classifications' value [0,100], where 0 is best
        double dPrecision; ///< 'precision' value [0,1], where 1 is best
        double dFMeasure; ///< 'F1 score'/'F-Measure' value [0,1], where 1 is best
        double dMCC; ///< 'Matthew's Correlation Coefficient' value [-1,1], where 1 is best
        /// computes the F-measure metric directly
        static inline double CalcFMeasure(double dRecall, double dPrecision) {
            return (dRecall+dPrecision)>0?(2.0*(dRecall*dPrecision)/(dRecall+dPrecision)):0;
        }
        /// computes the F-measure metric based on 'BinClassif' counters
        static inline double CalcFMeasure(const BinClassif& m) {
            return CalcFMeasure(CalcRecall(m),CalcPrecision(m));
        }
        /// computes the Recall metric directly
        static inline double CalcRecall(uint64_t nTP, uint64_t nTPFN) {
            return nTPFN>0?((double)nTP/nTPFN):0;
        }
        /// computes the Recall metric based on 'BinClassif' counters
        static inline double CalcRecall(const BinClassif& m) {
            return (m.nTP+m.nFN)>0?((double)m.nTP/(m.nTP+m.nFN)):0;
        }
        /// computes the Precision metric directly
        static inline double CalcPrecision(uint64_t nTP, uint64_t nTPFP) {
            return nTPFP>0?((double)nTP/nTPFP):0;
        }
        /// computes the Precision metric based on 'BinClassif' counters
        static inline double CalcPrecision(const BinClassif& m) {
            return (m.nTP+m.nFP)>0?((double)m.nTP/(m.nTP+m.nFP)):0;
        }
        /// computes the Specificity metric based on 'BinClassif' counters
        static inline double CalcSpecificity(const BinClassif& m) {
            return (m.nTN+m.nFP)>0?((double)m.nTN/(m.nTN+m.nFP)):0;
        }
        /// computes the False Positive Rate metric based on 'BinClassif' counters
        static inline double CalcFalsePositiveRate(const BinClassif& m) {
            return (m.nFP+m.nTN)>0?((double)m.nFP/(m.nFP+m.nTN)):0;
        }
        /// computes the False Negative Rate metric based on 'BinClassif' counters
        static inline double CalcFalseNegativeRate(const BinClassif& m) {
            return (m.nTP+m.nFN)>0?((double)m.nFN/(m.nTP+m.nFN)):0;
        }
        /// computes the Percentage of Bad Classifications metric based on 'BinClassif' counters
        static inline double CalcPercentBadClassifs(const BinClassif& m) {
            return m.total()>0?(100.0*(m.nFN+m.nFP)/m.total()):0;
        }
        /// computes Matthew's Correlation Coefficient based on 'BinClassif' counters
        static inline double CalcMatthewsCorrCoeff(const BinClassif& m) {
            return ((m.nTP+m.nFP)>0)&&((m.nTP+m.nFN)>0)&&((m.nTN+m.nFP)>0)&&((m.nTN+m.nFN)>0)?((((double)m.nTP*m.nTN)-(m.nFP*m.nFN))/sqrt(((double)m.nTP+m.nFP)*(m.nTP+m.nFN)*(m.nTN+m.nFP)*(m.nTN+m.nFN))):0;
        }
        /// default contructor requires a base metrics counters, as otherwise, we would obtain NaN's
        inline BinClassifMetrics(const BinClassif& m) : dRecall(CalcRecall(m)),dSpecificity(CalcSpecificity(m)),dFPR(CalcFalsePositiveRate(m)),dFNR(CalcFalseNegativeRate(m)),dPBC(CalcPercentBadClassifs(m)),dPrecision(CalcPrecision(m)),dFMeasure(CalcFMeasure(m)),dMCC(CalcMatthewsCorrCoeff(m)) {}
    };

    /// high-level metrics calculator super-interface (relies on IIMetricsAccumulator internally)
    struct IIMetricsCalculator : lv::enable_shared_from_this<IIMetricsCalculator> {
        /// virtual destructor for adequate cleanup from IIMetricsCalculator pointers
        virtual ~IIMetricsCalculator() = default;
        /// accumulates internal metrics, redirects to overridden function 'accumulate'
        inline IIMetricsCalculator& operator+=(const IIMetricsCalculator& m) {return *accumulate(m.shared_from_this());}
        /// accumulates internal metrics of 'm' into those of this object
        virtual std::shared_ptr<IIMetricsCalculator> accumulate(const std::shared_ptr<const IIMetricsCalculator>& m) = 0;
        /// returns a new instance of type 'MetricsCalculator', passing 'args' to its constructor
        template<typename MetricsCalculator, typename... Targs>
        static std::enable_if_t<std::is_base_of<IIMetricsCalculator,MetricsCalculator>::value,std::shared_ptr<MetricsCalculator>> create(Targs&&... args) {
            struct MetricsCalculatorWrapper : public MetricsCalculator {
                MetricsCalculatorWrapper(Targs&&... _args) : MetricsCalculator(std::forward<Targs>(_args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            return std::make_shared<MetricsCalculatorWrapper>(std::forward<Targs>(args)...);
        }
    protected:
        /// default constructor; assigns the relative weight of this calculator to one
        inline IIMetricsCalculator() : nWeight(1) {}
        /// used to track relative weight of metrics while computing iterative averages in overloads
        size_t nWeight;
    };
    using IIMetricsCalculatorPtr = std::shared_ptr<IIMetricsCalculator>;
    using IIMetricsCalculatorConstPtr = std::shared_ptr<const IIMetricsCalculator>;

    /// default (specializable) forward declaration of the metrics calculator interface
    template<DatasetEvalList eDatasetEval>
    struct IMetricsCalculator_;

    /// metrics calculator specialized to evaluate 2d binary classifiers
    template<>
    struct IMetricsCalculator_<DatasetEval_BinaryClassifier> :
            public IIMetricsCalculator {
        /// accumulates the binary classification metrics of 'm' into those of this object (considers relative weights of metrics)
        virtual IIMetricsCalculatorPtr accumulate(const IIMetricsCalculatorConstPtr& m) override;
        /// contains the actual metrics used for binary classification evaluation
        BinClassifMetrics m_oMetrics;
    protected:
        /// default contructor; requires a base metrics counters, as otherwise, we would obtain NaN's
        IMetricsCalculator_(const IIMetricsAccumulatorConstPtr& m);
        /// default contructor; requires a pre-filled BinClassifMetrics object
        IMetricsCalculator_(const BinClassifMetrics& m);
        /// default contructor; requires a pre-filled BinClassif object
        IMetricsCalculator_(const BinClassif& m);
    };
    using BinClassifMetricsCalculator = IMetricsCalculator_<DatasetEval_BinaryClassifier>;
    using BinClassifMetricsCalculatorPtr = std::shared_ptr<BinClassifMetricsCalculator>;
    using BinClassifMetricsCalculatorConstPtr = std::shared_ptr<const BinClassifMetricsCalculator>;

    /// metrics calculator specialized to evaluate an array of 2d binary classifiers
    template<>
    struct IMetricsCalculator_<DatasetEval_BinaryClassifierArray> :
            public IIMetricsCalculator {
        /// accumulates the binary classification metrics of 'm' into those of this object (considers relative weights of metrics)
        virtual IIMetricsCalculatorPtr accumulate(const IIMetricsCalculatorConstPtr& m) override;
        /// sum-reduces the binary classification metrics of this object into those of a 'BinClassifMetricsCalculator' object
        virtual BinClassifMetricsCalculatorPtr reduce() const;
        /// contains the actual metrics used for binary classification evaluation
        std::vector<BinClassifMetrics> m_vMetrics;
        /// contains the stream names used for printing eval reports (should be the same size as m_vMetrics)
        std::vector<std::string> m_vsStreamNames;
    protected:
        /// default contructor; requires a base metrics counters, as otherwise, we would obtain NaN's
        IMetricsCalculator_(const IIMetricsAccumulatorConstPtr& m);
        /// default contructor; requires pre-filled BinClassifMetrics array object + stream names array
        IMetricsCalculator_(const std::vector<BinClassifMetrics>& vm, const std::vector<std::string>& vs);
    };
    using BinClassifMetricsArrayCalculator = IMetricsCalculator_<DatasetEval_BinaryClassifierArray>;
    using BinClassifMetricsArrayCalculatorPtr = std::shared_ptr<BinClassifMetricsArrayCalculator>;
    using BinClassifMetricsArrayCalculatorConstPtr = std::shared_ptr<const BinClassifMetricsArrayCalculator>;

    /// basic metrics calculator full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct MetricsCalculator_ : public IMetricsCalculator_<eDatasetEval> {
    protected:
        /// fowards the constructors of the underlying interface (useful if they have mandatory parameters)
        using IMetricsCalculator_<eDatasetEval>::IMetricsCalculator_;
    };

    /// metric retriever super-interface; exposes utility functions to recursively parse metrics through all work batches
    struct IIMetricRetriever : public virtual IDataHandler {
        /// virtual destructor for adequate cleanup from IIMetricRetriever pointers
        virtual ~IIMetricRetriever() = default;
        /// returns whether this data handler defines some evaluation procedure or not (always true for this interface)
        virtual bool isEvaluable() const override final {return true;}
        /// accumulates and returns a sum of all base evaluation metrics for children batches, e.g. sums all classification counters (provides group-impl only)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const = 0;
        /// accumulates and returns high-level evaluation metrics, e.g. computes F-Measure from classification counters
        virtual IIMetricsCalculatorPtr getMetrics(bool bAverage) const = 0;
    };

    /// metric retriever interface specialization; exposes utility functions to recursively parse metrics through all work batches
    template<DatasetEvalList eDatasetEval, DatasetList eDataset>
    struct MetricRetriever_ : protected virtual IIMetricRetriever {
        /// accumulates and returns a sum of all base evaluation metrics for children batches, e.g. sums all classification counters (provides group-impl only)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const override {
            lvAssert_(this->isGroup(),"non-group data reporter specialization attempt to call non-overridden method");
            IIMetricsAccumulatorPtr pMetricsBase = IIMetricsAccumulator::create<MetricsAccumulator_<eDatasetEval,eDataset>>();
            for(const auto& pBatch : this->getBatches(true))
                pMetricsBase->accumulate(dynamic_cast<const IIMetricRetriever&>(*pBatch).getMetricsBase());
            return pMetricsBase;
        }
        /// accumulates and returns high-level evaluation metrics, e.g. computes F-Measure from classification counters
        virtual IIMetricsCalculatorPtr getMetrics(bool bAverage) const override final {
            if(bAverage && this->isGroup() && !this->isBare()) {
                IDataHandlerPtrArray vpBatches = this->getBatches(true);
                auto ppBatchIter = vpBatches.begin();
                for(; ppBatchIter!=vpBatches.end() && (*ppBatchIter)->getCurrentOutputCount()==0; ++ppBatchIter) {}
                lvAssert_(ppBatchIter!=vpBatches.end(),"found no processed output packets");
                IIMetricsCalculatorPtr pMetrics = dynamic_cast<const IIMetricRetriever&>(**ppBatchIter).getMetrics(bAverage);
                for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
                    if((*ppBatchIter)->getCurrentOutputCount()>0)
                        pMetrics->accumulate(dynamic_cast<const IIMetricRetriever&>(**ppBatchIter).getMetrics(bAverage));
                return pMetrics;
            }
            return IIMetricsCalculator::create<MetricsCalculator_<eDatasetEval,eDataset>>(getMetricsBase());
        }
    };

    /// metric retriever interface specialization; exposes no functions when eval type is 'none'
    template<DatasetList eDataset>
    struct MetricRetriever_<DatasetEval_None,eDataset> {};

} // namespace lv
