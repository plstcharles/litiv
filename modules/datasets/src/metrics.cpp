
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

#include "litiv/datasets/metrics.hpp"

litiv::MetricsAccumulator_<litiv::eDatasetEval_BinaryClassifier>::MetricsAccumulator_() : nTP(0),nTN(0),nFP(0),nFN(0),nSE(0),nDC(0) {}

litiv::BinClassifMetricsAccumulator litiv::MetricsAccumulator_<litiv::eDatasetEval_BinaryClassifier>::operator+(const BinClassifMetricsAccumulator& m) const {
    BinClassifMetricsAccumulator res(m);
    res.nTP += this->nTP;
    res.nTN += this->nTN;
    res.nFP += this->nFP;
    res.nFN += this->nFN;
    res.nSE += this->nSE;
    return res;
}

litiv::BinClassifMetricsAccumulator& litiv::MetricsAccumulator_<litiv::eDatasetEval_BinaryClassifier>::operator+=(const BinClassifMetricsAccumulator& m) {
    this->nTP += m.nTP;
    this->nTN += m.nTN;
    this->nFP += m.nFP;
    this->nFN += m.nFN;
    this->nSE += m.nSE;
    return *this;
}

bool litiv::MetricsAccumulator_<litiv::eDatasetEval_BinaryClassifier>::operator==(const BinClassifMetricsAccumulator& m) const {
    return
        (this->nTP==m.nTP) &&
        (this->nTN==m.nTN) &&
        (this->nFP==m.nFP) &&
        (this->nFN==m.nFN) &&
        (this->nSE==m.nSE);
}

bool litiv::MetricsAccumulator_<litiv::eDatasetEval_BinaryClassifier>::operator!=(const BinClassifMetricsAccumulator& m) const {
    return !((*this)==m);
}

void litiv::MetricsAccumulator_<litiv::eDatasetEval_BinaryClassifier>::accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI) {
    CV_Assert(oClassif.type()==CV_8UC1 && oGT.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
    CV_Assert(oClassif.size()==oGT.size() && (oROI.empty() || oClassif.size()==oROI.size()));
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

cv::Mat litiv::MetricsAccumulator_<litiv::eDatasetEval_BinaryClassifier>::getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI) {
    CV_Assert(oClassif.type()==CV_8UC1 && oGT.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
    CV_Assert(oClassif.size()==oGT.size() && (oROI.empty() || oClassif.size()==oROI.size()));
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

litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::MetricsCalculator_(const BinClassifMetricsAccumulator& m) :
        dRecall(CalcRecall(m)),
        dSpecificity(CalcSpecificity(m)),
        dFPR(CalcFalsePositiveRate(m)),
        dFNR(CalcFalseNegativeRate(m)),
        dPBC(CalcPercentBadClassifs(m)),
        dPrecision(CalcPrecision(m)),
        dFMeasure(CalcFMeasure(m)),
        dMCC(CalcMatthewsCorrCoeff(m)) {}

litiv::BinClassifMetricsCalculator litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::operator+(const BinClassifMetricsCalculator& m) const {
    BinClassifMetricsCalculator res(m);
    const size_t nTotWeight = this->nWeight+res.nWeight;
    res.dRecall = (res.dRecall*res.nWeight + this->dRecall*this->nWeight)/nTotWeight;
    res.dSpecificity = (res.dSpecificity*res.nWeight + this->dSpecificity*this->nWeight)/nTotWeight;
    res.dFPR = (res.dFPR*res.nWeight + this->dFPR*this->nWeight)/nTotWeight;
    res.dFNR = (res.dFNR*res.nWeight + this->dFNR*this->nWeight)/nTotWeight;
    res.dPBC = (res.dPBC*res.nWeight + this->dPBC*this->nWeight)/nTotWeight;
    res.dPrecision = (res.dPrecision*res.nWeight + this->dPrecision*this->nWeight)/nTotWeight;
    res.dFMeasure = (res.dFMeasure*res.nWeight + this->dFMeasure*this->nWeight)/nTotWeight;
    res.dMCC = (res.dMCC*res.nWeight + this->dMCC*this->nWeight)/nTotWeight;
    res.nWeight = nTotWeight;
    return res;
}

litiv::BinClassifMetricsCalculator& litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::operator+=(const BinClassifMetricsCalculator& m) {
    const size_t nTotWeight = this->nWeight+m.nWeight;
    this->dRecall = (m.dRecall*m.nWeight + this->dRecall*this->nWeight)/nTotWeight;
    this->dSpecificity = (m.dSpecificity*m.nWeight + this->dSpecificity*this->nWeight)/nTotWeight;
    this->dFPR = (m.dFPR*m.nWeight + this->dFPR*this->nWeight)/nTotWeight;
    this->dFNR = (m.dFNR*m.nWeight + this->dFNR*this->nWeight)/nTotWeight;
    this->dPBC = (m.dPBC*m.nWeight + this->dPBC*this->nWeight)/nTotWeight;
    this->dPrecision = (m.dPrecision*m.nWeight + this->dPrecision*this->nWeight)/nTotWeight;
    this->dFMeasure = (m.dFMeasure*m.nWeight + this->dFMeasure*this->nWeight)/nTotWeight;
    this->dMCC = (m.dMCC*m.nWeight + this->dMCC*this->nWeight)/nTotWeight;
    this->nWeight = nTotWeight;
    return *this;
}

double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcFMeasure(double dRecall, double dPrecision) {return (dRecall+dPrecision)>0?(2.0*(dRecall*dPrecision)/(dRecall+dPrecision)):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcFMeasure(const BinClassifMetricsAccumulator& m) {return CalcFMeasure(CalcRecall(m),CalcPrecision(m));}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcRecall(uint64_t nTP, uint64_t nTPFN) {return nTPFN>0?((double)nTP/nTPFN):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcRecall(const BinClassifMetricsAccumulator& m) {return (m.nTP+m.nFN)>0?((double)m.nTP/(m.nTP+m.nFN)):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcPrecision(uint64_t nTP, uint64_t nTPFP) {return nTPFP>0?((double)nTP/nTPFP):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcPrecision(const BinClassifMetricsAccumulator& m) {return (m.nTP+m.nFP)>0?((double)m.nTP/(m.nTP+m.nFP)):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcSpecificity(const BinClassifMetricsAccumulator& m) {return (m.nTN+m.nFP)>0?((double)m.nTN/(m.nTN+m.nFP)):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcFalsePositiveRate(const BinClassifMetricsAccumulator& m) {return (m.nFP+m.nTN)>0?((double)m.nFP/(m.nFP+m.nTN)):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcFalseNegativeRate(const BinClassifMetricsAccumulator& m) {return (m.nTP+m.nFN)>0?((double)m.nFN/(m.nTP+m.nFN)):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcPercentBadClassifs(const BinClassifMetricsAccumulator& m) {return m.total()>0?(100.0*(m.nFN+m.nFP)/m.total()):0;}
double litiv::MetricsCalculator_<litiv::eDatasetEval_BinaryClassifier>::CalcMatthewsCorrCoeff(const BinClassifMetricsAccumulator& m) {return ((m.nTP+m.nFP)>0)&&((m.nTP+m.nFN)>0)&&((m.nTN+m.nFP)>0)&&((m.nTN+m.nFN)>0)?((((double)m.nTP*m.nTN)-(m.nFP*m.nFN))/sqrt(((double)m.nTP+m.nFP)*(m.nTP+m.nFN)*(m.nTN+m.nFP)*(m.nTN+m.nFN))):0;}
