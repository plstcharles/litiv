
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

namespace litiv {

    //! basic metrics accumulator interface
    struct IMetricsAccumulator : CxxUtils::enable_shared_from_this<IMetricsAccumulator> {
        virtual ~IMetricsAccumulator() = default;
        bool operator!=(const IMetricsAccumulator& m) const;
        bool operator==(const IMetricsAccumulator& m) const;
        virtual bool isEqual(const std::shared_ptr<const IMetricsAccumulator>& m) const = 0;
        IMetricsAccumulator& operator+=(const IMetricsAccumulator& m);
        virtual std::shared_ptr<IMetricsAccumulator> accumulate(const std::shared_ptr<const IMetricsAccumulator>& m) = 0;
    protected:
        IMetricsAccumulator() = default;
        IMetricsAccumulator& operator=(const IMetricsAccumulator&) = delete;
        IMetricsAccumulator(const IMetricsAccumulator&) = delete;
    };
    using IMetricsAccumulatorPtr = std::shared_ptr<IMetricsAccumulator>;
    using IMetricsAccumulatorConstPtr = std::shared_ptr<const IMetricsAccumulator>;

    //! high-level metrics calculator interface (relies on IMetricsAccumulator internally)
    struct IMetricsCalculator : CxxUtils::enable_shared_from_this<IMetricsCalculator> {
        virtual ~IMetricsCalculator() = default;
        IMetricsCalculator& operator+=(const IMetricsCalculator& m);
        virtual std::shared_ptr<IMetricsCalculator> accumulate(const std::shared_ptr<const IMetricsCalculator>& m) = 0;
    protected:
        IMetricsCalculator() : nWeight(1) {}
        IMetricsCalculator& operator=(const IMetricsCalculator&) = delete;
        IMetricsCalculator(const IMetricsCalculator&) = delete;
        size_t nWeight; // used to compute iterative averages in overloads
    };
    using IMetricsCalculatorPtr = std::shared_ptr<IMetricsCalculator>;
    using IMetricsCalculatorConstPtr = std::shared_ptr<const IMetricsCalculator>;

    template<eDatasetEvalList eDatasetEval>
    struct MetricsAccumulator_;

    template<> //! basic metrics counter used to evaluate 2d binary classifiers
    struct MetricsAccumulator_<eDatasetEval_BinaryClassifier> :
            public IMetricsAccumulator {
        virtual bool isEqual(const IMetricsAccumulatorConstPtr& m) const override;
        virtual IMetricsAccumulatorPtr accumulate(const IMetricsAccumulatorConstPtr& m) override;
        virtual void accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI=cv::Mat());
        static cv::Mat getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI=cv::Mat());
        inline uint64_t total(bool bWithDontCare=false) const {return nTP+nTN+nFP+nFN+(bWithDontCare?nDC:uint64_t(0));}
        static std::shared_ptr<MetricsAccumulator_<eDatasetEval_BinaryClassifier>> create();
        uint64_t nTP;
        uint64_t nTN;
        uint64_t nFP;
        uint64_t nFN;
        uint64_t nSE; // 'special error' counter (usage is optional -- useful for e.g. shadow error)
        uint64_t nDC; // 'dont care' counter (usage is optional -- useful with unknowns or small ROIs)
        enum eCountersList { // used for packed array indexing
            eCounter_TP,
            eCounter_TN,
            eCounter_FP,
            eCounter_FN,
            eCounter_SE,
            eCounter_DC,
            eCountersCount,
        };
    protected:
        //! default constructor sets all counters to zero
        MetricsAccumulator_();
    };
    using BinClassifMetricsAccumulator = MetricsAccumulator_<eDatasetEval_BinaryClassifier>;
    using BinClassifMetricsAccumulatorPtr = std::shared_ptr<BinClassifMetricsAccumulator>;
    using BinClassifMetricsAccumulatorConstPtr = std::shared_ptr<const BinClassifMetricsAccumulator>;

    template<eDatasetEvalList eDatasetEval>
    struct MetricsCalculator_;

    template<> //! high-level metrics used to evaluate 2d binary classifiers
    struct MetricsCalculator_<eDatasetEval_BinaryClassifier> :
            public IMetricsCalculator {
        virtual IMetricsCalculatorPtr accumulate(const IMetricsCalculatorConstPtr& m) override;
        static std::shared_ptr<MetricsCalculator_<eDatasetEval_BinaryClassifier>> create(const IMetricsAccumulatorConstPtr& m);
        static double CalcFMeasure(double dRecall, double dPrecision);
        static double CalcFMeasure(const BinClassifMetricsAccumulator& m);
        static double CalcRecall(uint64_t nTP, uint64_t nTPFN);
        static double CalcRecall(const BinClassifMetricsAccumulator& m);
        static double CalcPrecision(uint64_t nTP, uint64_t nTPFP);
        static double CalcPrecision(const BinClassifMetricsAccumulator& m);
        static double CalcSpecificity(const BinClassifMetricsAccumulator& m);
        static double CalcFalsePositiveRate(const BinClassifMetricsAccumulator& m);
        static double CalcFalseNegativeRate(const BinClassifMetricsAccumulator& m);
        static double CalcPercentBadClassifs(const BinClassifMetricsAccumulator& m);
        static double CalcMatthewsCorrCoeff(const BinClassifMetricsAccumulator& m);
        double dRecall;
        double dSpecificity;
        double dFPR;
        double dFNR;
        double dPBC;
        double dPrecision;
        double dFMeasure;
        double dMCC;
    protected:
        //! default contructor requires a base metrics counters, as otherwise, we may obtain NaN's
        MetricsCalculator_(const IMetricsAccumulatorConstPtr& m);
    };
    using BinClassifMetricsCalculator = MetricsCalculator_<eDatasetEval_BinaryClassifier>;
    using BinClassifMetricsCalculatorPtr = std::shared_ptr<BinClassifMetricsCalculator>;
    using BinClassifMetricsCalculatorConstPtr = std::shared_ptr<const BinClassifMetricsCalculator>;

} //namespace litiv
