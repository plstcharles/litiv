
// This file is part of the lv framework; visit the original repository at
// https://github.com/plstcharles/lv for more information.
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

#include "litiv/datasets/metrics.hpp"

bool lv::IMetricsAccumulator::operator!=(const IMetricsAccumulator& m) const {
    return !isEqual(m.shared_from_this());
}

bool lv::IMetricsAccumulator::operator==(const IMetricsAccumulator& m) const {
    return isEqual(m.shared_from_this());
}

lv::IMetricsAccumulator& lv::IMetricsAccumulator::operator+=(const IMetricsAccumulator& m) {
    return *accumulate(m.shared_from_this());
}

lv::IMetricsCalculator& lv::IMetricsCalculator::operator+=(const IMetricsCalculator& m) {
    return *accumulate(m.shared_from_this());
}

bool lv::MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::isEqual(const IMetricsAccumulatorConstPtr& m) const {
    const auto& m2 = dynamic_cast<const MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>&>(*m.get());
    return
        (this->nTP==m2.nTP) &&
        (this->nTN==m2.nTN) &&
        (this->nFP==m2.nFP) &&
        (this->nFN==m2.nFN) &&
        (this->nSE==m2.nSE);
}

lv::IMetricsAccumulatorPtr lv::MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::accumulate(const IMetricsAccumulatorConstPtr& m) {
    const auto& m2 = dynamic_cast<const MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>&>(*m.get());
    this->nTP += m2.nTP;
    this->nTN += m2.nTN;
    this->nFP += m2.nFP;
    this->nFN += m2.nFN;
    this->nSE += m2.nSE;
    return shared_from_this();
}

void lv::MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI) {
    lvAssert_(!oClassif.empty() && oClassif.type()==CV_8UC1,"binary classifier results must be non-empty and of type 8UC1");
    lvAssert_(oGT.empty() || oGT.type()==CV_8UC1,"gt mat must be empty, or of type 8UC1")
    lvAssert_(oROI.empty() || oROI.type()==CV_8UC1,"ROI mat must be empty, or of type 8UC1");
    lvAssert_((oGT.empty() || oClassif.size()==oGT.size()) && (oROI.empty() || oClassif.size()==oROI.size()),"all input mat sizes must match");
    if(oGT.empty()) {
        nDC += oClassif.size().area();
        return;
    }
    const size_t step_row = oClassif.step.p[0];
    for(size_t i = 0; i<(size_t)oClassif.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oClassif.data+idx_nstep;
        const uchar* gt_step_ptr = oGT.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j = 0; j<oClassif.cols; ++j) {
            if(gt_step_ptr[j]!=DATASETUTILS_OUTOFSCOPE_VAL &&
               gt_step_ptr[j]!=DATASETUTILS_UNKNOWN_VAL &&
               (oROI.empty() || roi_step_ptr[j]!=dATASETUTILS_NEGATIVE_VAL)) {
                if(input_step_ptr[j]==DATASETUTILS_POSITIVE_VAL) {
                    if(gt_step_ptr[j]==DATASETUTILS_POSITIVE_VAL)
                        ++nTP;
                    else // gt_step_ptr[j]==s_nSegmNegative
                        ++nFP;
                }
                else { // input_step_ptr[j]==s_nSegmNegative
                    if(gt_step_ptr[j]==DATASETUTILS_POSITIVE_VAL)
                        ++nFN;
                    else // gt_step_ptr[j]==s_nSegmNegative
                        ++nTN;
                }
                if(gt_step_ptr[j]==DATASETUTILS_SHADOW_VAL) {
                    if(input_step_ptr[j]==DATASETUTILS_POSITIVE_VAL)
                        ++nSE;
                }
            }
            else
                ++nDC;
        }
    }
}

cv::Mat lv::MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI) {
    lvAssert_(!oClassif.empty() && oClassif.type()==CV_8UC1,"binary classifier results must be non-empty and of type 8UC1");
    lvAssert_(oGT.empty() || oGT.type()==CV_8UC1,"gt mat must be empty, or of type 8UC1")
    lvAssert_(oROI.empty() || oROI.type()==CV_8UC1,"ROI mat must be empty, or of type 8UC1");
    lvAssert_((oGT.empty() || oClassif.size()==oGT.size()) && (oROI.empty() || oClassif.size()==oROI.size()),"all input mat sizes must match");
    if(oGT.empty()) {
        cv::Mat oResult;
        cv::cvtColor(oClassif,oResult,cv::COLOR_GRAY2BGR);
        return oResult;
    }
    cv::Mat oResult(oClassif.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oClassif.step.p[0];
    for(size_t i=0; i<(size_t)oClassif.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oClassif.data+idx_nstep;
        const uchar* gt_step_ptr = oGT.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oClassif.cols; ++j) {
            if(gt_step_ptr[j]!=DATASETUTILS_OUTOFSCOPE_VAL &&
               gt_step_ptr[j]!=DATASETUTILS_UNKNOWN_VAL &&
               (oROI.empty() || roi_step_ptr[j]!=dATASETUTILS_NEGATIVE_VAL)) {
                if(input_step_ptr[j]==DATASETUTILS_POSITIVE_VAL) {
                    if(gt_step_ptr[j]==DATASETUTILS_POSITIVE_VAL)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==dATASETUTILS_NEGATIVE_VAL)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==DATASETUTILS_SHADOW_VAL) {
                        res_step_ptr[j*3+1] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                    else {
                        for(size_t c=0; c<3; ++c)
                            res_step_ptr[j*3+c] = UCHAR_MAX/3;
                    }
                }
                else { // input_step_ptr[j]==s_nSegmNegative
                    if(gt_step_ptr[j]==DATASETUTILS_POSITIVE_VAL) {
                        res_step_ptr[j*3] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                }
            }
            else if(!oROI.empty() && roi_step_ptr[j]==dATASETUTILS_NEGATIVE_VAL) {
                for(size_t c=0; c<3; ++c)
                    res_step_ptr[j*3+c] = UCHAR_MAX/2;
            }
            else {
                for(size_t c=0; c<3; ++c)
                    res_step_ptr[j*3+c] = input_step_ptr[j];
            }
        }
    }
    return oResult;
}

std::shared_ptr<lv::MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>> lv::MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::create() {
    struct MetricsAccumulatorWrapper : public MetricsAccumulator_<DatasetEval_BinaryClassifier> {
        MetricsAccumulatorWrapper() : MetricsAccumulator_<DatasetEval_BinaryClassifier>() {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
    };
    return std::make_shared<MetricsAccumulatorWrapper>();
}

lv::MetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::MetricsAccumulator_() : nTP(0),nTN(0),nFP(0),nFN(0),nSE(0),nDC(0) {}

lv::IMetricsCalculatorPtr lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::accumulate(const IMetricsCalculatorConstPtr& m) {
    const auto& m2 = dynamic_cast<const MetricsCalculator_<lv::DatasetEval_BinaryClassifier>&>(*m.get());
    const size_t nTotWeight = this->nWeight+m2.nWeight;
    this->dRecall = (m2.dRecall*m2.nWeight + this->dRecall*this->nWeight)/nTotWeight;
    this->dSpecificity = (m2.dSpecificity*m2.nWeight + this->dSpecificity*this->nWeight)/nTotWeight;
    this->dFPR = (m2.dFPR*m2.nWeight + this->dFPR*this->nWeight)/nTotWeight;
    this->dFNR = (m2.dFNR*m2.nWeight + this->dFNR*this->nWeight)/nTotWeight;
    this->dPBC = (m2.dPBC*m2.nWeight + this->dPBC*this->nWeight)/nTotWeight;
    this->dPrecision = (m2.dPrecision*m2.nWeight + this->dPrecision*this->nWeight)/nTotWeight;
    this->dFMeasure = (m2.dFMeasure*m2.nWeight + this->dFMeasure*this->nWeight)/nTotWeight;
    this->dMCC = (m2.dMCC*m2.nWeight + this->dMCC*this->nWeight)/nTotWeight;
    this->nWeight = nTotWeight;
    return shared_from_this();
}

lv::BinClassifMetricsCalculatorPtr lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::create(const IMetricsAccumulatorConstPtr& m) {
    struct MetricsCalculatorWrapper : public MetricsCalculator_<DatasetEval_BinaryClassifier> {
        MetricsCalculatorWrapper(const IMetricsAccumulatorConstPtr& m2) : MetricsCalculator_<DatasetEval_BinaryClassifier>(m2) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
    };
    return std::make_shared<MetricsCalculatorWrapper>(m);
}

double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcFMeasure(double dRecall, double dPrecision) {return (dRecall+dPrecision)>0?(2.0*(dRecall*dPrecision)/(dRecall+dPrecision)):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcFMeasure(const BinClassifMetricsAccumulator& m) {return CalcFMeasure(CalcRecall(m),CalcPrecision(m));}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcRecall(uint64_t nTP, uint64_t nTPFN) {return nTPFN>0?((double)nTP/nTPFN):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcRecall(const BinClassifMetricsAccumulator& m) {return (m.nTP+m.nFN)>0?((double)m.nTP/(m.nTP+m.nFN)):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcPrecision(uint64_t nTP, uint64_t nTPFP) {return nTPFP>0?((double)nTP/nTPFP):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcPrecision(const BinClassifMetricsAccumulator& m) {return (m.nTP+m.nFP)>0?((double)m.nTP/(m.nTP+m.nFP)):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcSpecificity(const BinClassifMetricsAccumulator& m) {return (m.nTN+m.nFP)>0?((double)m.nTN/(m.nTN+m.nFP)):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcFalsePositiveRate(const BinClassifMetricsAccumulator& m) {return (m.nFP+m.nTN)>0?((double)m.nFP/(m.nFP+m.nTN)):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcFalseNegativeRate(const BinClassifMetricsAccumulator& m) {return (m.nTP+m.nFN)>0?((double)m.nFN/(m.nTP+m.nFN)):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcPercentBadClassifs(const BinClassifMetricsAccumulator& m) {return m.total()>0?(100.0*(m.nFN+m.nFP)/m.total()):0;}
double lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::CalcMatthewsCorrCoeff(const BinClassifMetricsAccumulator& m) {return ((m.nTP+m.nFP)>0)&&((m.nTP+m.nFN)>0)&&((m.nTN+m.nFP)>0)&&((m.nTN+m.nFN)>0)?((((double)m.nTP*m.nTN)-(m.nFP*m.nFN))/sqrt(((double)m.nTP+m.nFP)*(m.nTP+m.nFN)*(m.nTN+m.nFP)*(m.nTN+m.nFN))):0;}

lv::MetricsCalculator_<lv::DatasetEval_BinaryClassifier>::MetricsCalculator_(const IMetricsAccumulatorConstPtr& m) {
    lvAssert_(m.get(),"bad input pointer");
    const auto& m2 = std::dynamic_pointer_cast<const BinClassifMetricsAccumulator>(m);
    lvAssert_(m2.get(),"input metrics accumulator did not possess a BinClassifMetricsAccumulator interface");
    const BinClassifMetricsAccumulator& m3 = *m2.get();
    dRecall = CalcRecall(m3);
    dSpecificity = CalcSpecificity(m3);
    dFPR = CalcFalsePositiveRate(m3);
    dFNR = CalcFalseNegativeRate(m3);
    dPBC = CalcPercentBadClassifs(m3);
    dPrecision = CalcPrecision(m3);
    dFMeasure = CalcFMeasure(m3);
    dMCC = CalcMatthewsCorrCoeff(m3);
}
