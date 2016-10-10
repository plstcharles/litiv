
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

#include <litiv/datasets/metrics.hpp>
#include "litiv/datasets/metrics.hpp"

void lv::BinClassif::accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI) {
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
               (oROI.empty() || roi_step_ptr[j]!=DATASETUTILS_NEGATIVE_VAL)) {
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

cv::Mat lv::BinClassif::getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& oROI) {
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
               (oROI.empty() || roi_step_ptr[j]!=DATASETUTILS_NEGATIVE_VAL)) {
                if(input_step_ptr[j]==DATASETUTILS_POSITIVE_VAL) {
                    if(gt_step_ptr[j]==DATASETUTILS_POSITIVE_VAL)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==DATASETUTILS_NEGATIVE_VAL)
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
            else if(!oROI.empty() && roi_step_ptr[j]==DATASETUTILS_NEGATIVE_VAL) {
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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

bool lv::IMetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::isEqual(const IIMetricsAccumulatorConstPtr& m) const {
    const auto& m2 = dynamic_cast<const IMetricsAccumulator_<lv::DatasetEval_BinaryClassifier>&>(*m.get());
    return this->m_oCounters.isEqual(m2.m_oCounters);
}

lv::IIMetricsAccumulatorPtr lv::IMetricsAccumulator_<lv::DatasetEval_BinaryClassifier>::accumulate(const IIMetricsAccumulatorConstPtr& m) {
    const auto& m2 = dynamic_cast<const IMetricsAccumulator_<lv::DatasetEval_BinaryClassifier>&>(*m.get());
    this->m_oCounters.accumulate(m2.m_oCounters);
    return shared_from_this();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool lv::IMetricsAccumulator_<lv::DatasetEval_BinaryClassifierArray>::isEqual(const IIMetricsAccumulatorConstPtr& m) const {
    const auto& m2 = dynamic_cast<const IMetricsAccumulator_<lv::DatasetEval_BinaryClassifierArray>&>(*m.get());
    if(this->m_vCounters.size()!=m2.m_vCounters.size())
        return false;
    for(size_t s=0; s<this->m_vCounters.size(); ++s)
        if(!this->m_vCounters[s].isEqual(m2.m_vCounters[s]))
            return false;
    return true;
}

lv::IIMetricsAccumulatorPtr lv::IMetricsAccumulator_<lv::DatasetEval_BinaryClassifierArray>::accumulate(const IIMetricsAccumulatorConstPtr& m) {
    const auto& m2 = dynamic_cast<const IMetricsAccumulator_<lv::DatasetEval_BinaryClassifierArray>&>(*m.get());
    if(m_vCounters.empty())
        m_vCounters.resize(m2.m_vCounters.size());
    else
        lvAssert_(this->m_vCounters.size()==m2.m_vCounters.size(),"array size mismatch");
    if(m_vsStreamNames.empty())
        m_vsStreamNames = m2.m_vsStreamNames;
    else
        lvAssert_(this->m_vsStreamNames.size()==m2.m_vsStreamNames.size(),"array size mismatch");
    lvAssert_(this->m_vCounters.size()==this->m_vsStreamNames.size(),"array size mismatch");
    for(size_t s=0; s<this->m_vCounters.size(); ++s) {
        this->m_vCounters[s].accumulate(m2.m_vCounters[s]);
        if(this->m_vsStreamNames[s].empty())
            this->m_vsStreamNames[s] = m2.m_vsStreamNames[s];
    }
    return shared_from_this();
}

lv::BinClassifMetricsAccumulatorPtr lv::IMetricsAccumulator_<lv::DatasetEval_BinaryClassifierArray>::reduce() const {
    BinClassifMetricsAccumulatorPtr m = IIMetricsAccumulator::create<BinClassifMetricsAccumulator>();
    for(size_t s=0; s<this->m_vCounters.size(); ++s)
        m->m_oCounters.accumulate(this->m_vCounters[s]);
    return m;
}

lv::IMetricsAccumulator_<lv::DatasetEval_BinaryClassifierArray>::IMetricsAccumulator_(size_t nArraySize) :
        m_vCounters(nArraySize),m_vsStreamNames(nArraySize) {}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

lv::IIMetricsCalculatorPtr lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifier>::accumulate(const IIMetricsCalculatorConstPtr& m) {
    const auto& m2 = dynamic_cast<const IMetricsCalculator_<lv::DatasetEval_BinaryClassifier>&>(*m.get());
    const size_t nTotWeight = this->nWeight+m2.nWeight;
    this->m_oMetrics.dRecall = (m2.m_oMetrics.dRecall*m2.nWeight + this->m_oMetrics.dRecall*this->nWeight)/nTotWeight;
    this->m_oMetrics.dSpecificity = (m2.m_oMetrics.dSpecificity*m2.nWeight + this->m_oMetrics.dSpecificity*this->nWeight)/nTotWeight;
    this->m_oMetrics.dFPR = (m2.m_oMetrics.dFPR*m2.nWeight + this->m_oMetrics.dFPR*this->nWeight)/nTotWeight;
    this->m_oMetrics.dFNR = (m2.m_oMetrics.dFNR*m2.nWeight + this->m_oMetrics.dFNR*this->nWeight)/nTotWeight;
    this->m_oMetrics.dPBC = (m2.m_oMetrics.dPBC*m2.nWeight + this->m_oMetrics.dPBC*this->nWeight)/nTotWeight;
    this->m_oMetrics.dPrecision = (m2.m_oMetrics.dPrecision*m2.nWeight + this->m_oMetrics.dPrecision*this->nWeight)/nTotWeight;
    this->m_oMetrics.dFMeasure = (m2.m_oMetrics.dFMeasure*m2.nWeight + this->m_oMetrics.dFMeasure*this->nWeight)/nTotWeight;
    this->m_oMetrics.dMCC = (m2.m_oMetrics.dMCC*m2.nWeight + this->m_oMetrics.dMCC*this->nWeight)/nTotWeight;
    this->nWeight = nTotWeight;
    return shared_from_this();
}

lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifier>::IMetricsCalculator_(const IIMetricsAccumulatorConstPtr& m) :
        m_oMetrics(dynamic_cast<const BinClassifMetricsAccumulator&>(*m.get()).m_oCounters) {}

lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifier>::IMetricsCalculator_(const BinClassifMetrics& m) :
        m_oMetrics(m) {}

lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifier>::IMetricsCalculator_(const BinClassif& m) :
        m_oMetrics(m) {}

////////////////////////////////////////////////////////////////////////////////////////////////////

lv::IIMetricsCalculatorPtr lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifierArray>::accumulate(const IIMetricsCalculatorConstPtr& m) {
    const auto& m2 = dynamic_cast<const IMetricsCalculator_<lv::DatasetEval_BinaryClassifierArray>&>(*m.get());
    lvAssert_(this->m_vMetrics.size()==m2.m_vMetrics.size(),"array size mismatch");
    const size_t nTotWeight = this->nWeight+m2.nWeight;
    for(size_t s=0; s<this->m_vMetrics.size(); ++s) {
        this->m_vMetrics[s].dRecall = (m2.m_vMetrics[s].dRecall*m2.nWeight + this->m_vMetrics[s].dRecall*this->nWeight)/nTotWeight;
        this->m_vMetrics[s].dSpecificity = (m2.m_vMetrics[s].dSpecificity*m2.nWeight + this->m_vMetrics[s].dSpecificity*this->nWeight)/nTotWeight;
        this->m_vMetrics[s].dFPR = (m2.m_vMetrics[s].dFPR*m2.nWeight + this->m_vMetrics[s].dFPR*this->nWeight)/nTotWeight;
        this->m_vMetrics[s].dFNR = (m2.m_vMetrics[s].dFNR*m2.nWeight + this->m_vMetrics[s].dFNR*this->nWeight)/nTotWeight;
        this->m_vMetrics[s].dPBC = (m2.m_vMetrics[s].dPBC*m2.nWeight + this->m_vMetrics[s].dPBC*this->nWeight)/nTotWeight;
        this->m_vMetrics[s].dPrecision = (m2.m_vMetrics[s].dPrecision*m2.nWeight + this->m_vMetrics[s].dPrecision*this->nWeight)/nTotWeight;
        this->m_vMetrics[s].dFMeasure = (m2.m_vMetrics[s].dFMeasure*m2.nWeight + this->m_vMetrics[s].dFMeasure*this->nWeight)/nTotWeight;
        this->m_vMetrics[s].dMCC = (m2.m_vMetrics[s].dMCC*m2.nWeight + this->m_vMetrics[s].dMCC*this->nWeight)/nTotWeight;
    }
    this->nWeight = nTotWeight;
    return shared_from_this();
}

lv::BinClassifMetricsCalculatorPtr lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifierArray>::reduce() const {
    lvAssert_(this->m_vMetrics.size()>0,"need at least array size one");
    BinClassifMetrics m(this->m_vMetrics[0]);
    for(size_t s=1; s<this->m_vMetrics.size(); ++s) {
        m.dRecall += this->m_vMetrics[s].dRecall;
        m.dSpecificity += this->m_vMetrics[s].dSpecificity;
        m.dFPR += this->m_vMetrics[s].dFPR;
        m.dFNR += this->m_vMetrics[s].dFNR;
        m.dPBC += this->m_vMetrics[s].dPBC;
        m.dPrecision += this->m_vMetrics[s].dPrecision;
        m.dFMeasure += this->m_vMetrics[s].dFMeasure;
        m.dMCC += this->m_vMetrics[s].dMCC;
    }
    m.dRecall /= this->m_vMetrics.size();
    m.dSpecificity /= this->m_vMetrics.size();
    m.dFPR /= this->m_vMetrics.size();
    m.dFNR /= this->m_vMetrics.size();
    m.dPBC /= this->m_vMetrics.size();
    m.dPrecision /= this->m_vMetrics.size();
    m.dFMeasure /= this->m_vMetrics.size();
    m.dMCC /= this->m_vMetrics.size();
    return IIMetricsCalculator::create<BinClassifMetricsCalculator>(m);
}

inline std::vector<lv::BinClassifMetrics> initMetricsArray(const lv::BinClassifMetricsArrayAccumulator& m) {
    std::vector<lv::BinClassifMetrics> vMetrics;
    for(const lv::BinClassif& m2 : m.m_vCounters)
        vMetrics.push_back(lv::BinClassifMetrics(m2));
    return vMetrics;
}

lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifierArray>::IMetricsCalculator_(const IIMetricsAccumulatorConstPtr& m) :
        m_vMetrics(initMetricsArray(dynamic_cast<const BinClassifMetricsArrayAccumulator&>(*m.get()))),
        m_vsStreamNames(dynamic_cast<const BinClassifMetricsArrayAccumulator&>(*m.get()).m_vsStreamNames) {
    lvAssert(m_vMetrics.size()==m_vsStreamNames.size());
}

lv::IMetricsCalculator_<lv::DatasetEval_BinaryClassifierArray>::IMetricsCalculator_(const std::vector<BinClassifMetrics>& vm, const std::vector<std::string>& vs) :
        m_vMetrics(vm),m_vsStreamNames(vs) {
    lvAssert(m_vMetrics.size()==m_vsStreamNames.size());
}
