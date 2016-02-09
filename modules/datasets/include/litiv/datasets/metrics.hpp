
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
    struct IMetricsAccumulator {
        virtual ~IMetricsAccumulator() = default;
    };

    //! high-level metrics calculator interface (relies on IMetricsAccumulator internally)
    struct IMetricsCalculator {
        IMetricsCalculator() : nWeight(1) {}
        virtual ~IMetricsCalculator() = default;
    protected:
        size_t nWeight; // used to compute iterative averages in overloads
    };

    template<eDatasetEvalList eDatasetEval>
    struct MetricsAccumulator_;

    template<> //! basic metrics counter used to evaluate 2d binary classifiers
    struct MetricsAccumulator_<eDatasetEval_BinaryClassifier> :
            public IMetricsAccumulator {
        using ThisType = MetricsAccumulator_<eDatasetEval_BinaryClassifier>;
        //! default constructor sets all counters to zero
        MetricsAccumulator_();
        ThisType operator+(const ThisType& m) const;
        ThisType& operator+=(const ThisType& m);
        bool operator==(const ThisType& m) const;
        bool operator!=(const ThisType& m) const;
        virtual void accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI=cv::Mat());
        static cv::Mat getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI=cv::Mat());
        inline uint64_t total(bool bWithDontCare=false) const {return nTP+nTN+nFP+nFN+(bWithDontCare?nDC:uint64_t(0));}
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
    };
    using BinClassifMetricsAccumulator = MetricsAccumulator_<eDatasetEval_BinaryClassifier>;

    template<eDatasetEvalList eDatasetEval>
    struct MetricsCalculator_;

    template<> //! high-level metrics used to evaluate 2d binary classifiers
    struct MetricsCalculator_<eDatasetEval_BinaryClassifier> :
            public IMetricsCalculator {
        using ThisType = MetricsCalculator_<eDatasetEval_BinaryClassifier>;
        using ThisBaseType = MetricsAccumulator_<eDatasetEval_BinaryClassifier>;
        //! default contructor requires a base metrics counters, as otherwise, we may obtain NaN's
        MetricsCalculator_(const ThisBaseType& m);
        ThisType operator+(const ThisType& m) const;
        ThisType& operator+=(const ThisType& m);
        double dRecall;
        double dSpecificity;
        double dFPR;
        double dFNR;
        double dPBC;
        double dPrecision;
        double dFMeasure;
        double dMCC;
        static double CalcFMeasure(double dRecall, double dPrecision);
        static double CalcFMeasure(const ThisBaseType& m);
        static double CalcRecall(uint64_t nTP, uint64_t nTPFN);
        static double CalcRecall(const ThisBaseType& m);
        static double CalcPrecision(uint64_t nTP, uint64_t nTPFP);
        static double CalcPrecision(const ThisBaseType& m);
        static double CalcSpecificity(const ThisBaseType& m);
        static double CalcFalsePositiveRate(const ThisBaseType& m);
        static double CalcFalseNegativeRate(const ThisBaseType& m);
        static double CalcPercentBadClassifs(const ThisBaseType& m);
        static double CalcMatthewsCorrCoeff(const ThisBaseType& m);
    };
    using BinClassifMetricsCalculator = MetricsCalculator_<eDatasetEval_BinaryClassifier>;

} //namespace litiv
