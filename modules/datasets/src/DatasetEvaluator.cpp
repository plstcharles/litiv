
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

#include "litiv/datasets/DatasetEvaluator.hpp"
#include "litiv/imgproc.hpp"
//#include "litiv/utils/ConsoleUtils.hpp" @@@@@ reuse later?

litiv::ClassifMetricsBase::ClassifMetricsBase() : nTP(0),nTN(0),nFP(0),nFN(0),nSE(0) {}

litiv::ClassifMetricsBase litiv::ClassifMetricsBase::operator+(const ClassifMetricsBase& m) const {
    ClassifMetricsBase res(m);
    res.nTP += this->nTP;
    res.nTN += this->nTN;
    res.nFP += this->nFP;
    res.nFN += this->nFN;
    res.nSE += this->nSE;
    return res;
}

litiv::ClassifMetricsBase& litiv::ClassifMetricsBase::operator+=(const ClassifMetricsBase& m) {
    this->nTP += m.nTP;
    this->nTN += m.nTN;
    this->nFP += m.nFP;
    this->nFN += m.nFN;
    this->nSE += m.nSE;
    return *this;
}

litiv::ClassifMetrics::ClassifMetrics(const ClassifMetricsBase& m) :
        dRecall(CalcRecall(m)),
        dSpecificity(CalcSpecificity(m)),
        dFPR(CalcFalsePositiveRate(m)),
        dFNR(CalcFalseNegativeRate(m)),
        dPBC(CalcPercentBadClassifs(m)),
        dPrecision(CalcPrecision(m)),
        dFMeasure(CalcFMeasure(m)),
        dMCC(CalcMatthewsCorrCoeff(m)),
        nWeight(1) {}

litiv::ClassifMetrics litiv::ClassifMetrics::operator+(const ClassifMetrics& m) const {
    ClassifMetrics res(m);
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

litiv::ClassifMetrics& litiv::ClassifMetrics::operator+=(const ClassifMetrics& m) {
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

double litiv::ClassifMetrics::CalcFMeasure(double dRecall, double dPrecision) {return (dRecall+dPrecision)>0?(2.0*(dRecall*dPrecision)/(dRecall+dPrecision)):0;}
double litiv::ClassifMetrics::CalcFMeasure(const ClassifMetricsBase& m) {return CalcFMeasure(CalcRecall(m),CalcPrecision(m));}
double litiv::ClassifMetrics::CalcRecall(uint64_t nTP, uint64_t nTPFN) {return nTPFN>0?((double)nTP/nTPFN):0;}
double litiv::ClassifMetrics::CalcRecall(const ClassifMetricsBase& m) {return (m.nTP+m.nFN)>0?((double)m.nTP/(m.nTP+m.nFN)):0;}
double litiv::ClassifMetrics::CalcPrecision(uint64_t nTP, uint64_t nTPFP) {return nTPFP>0?((double)nTP/nTPFP):0;}
double litiv::ClassifMetrics::CalcPrecision(const ClassifMetricsBase& m) {return (m.nTP+m.nFP)>0?((double)m.nTP/(m.nTP+m.nFP)):0;}
double litiv::ClassifMetrics::CalcSpecificity(const ClassifMetricsBase& m) {return (m.nTN+m.nFP)>0?((double)m.nTN/(m.nTN+m.nFP)):0;}
double litiv::ClassifMetrics::CalcFalsePositiveRate(const ClassifMetricsBase& m) {return (m.nFP+m.nTN)>0?((double)m.nFP/(m.nFP+m.nTN)):0;}
double litiv::ClassifMetrics::CalcFalseNegativeRate(const ClassifMetricsBase& m) {return (m.nTP+m.nFN)>0?((double)m.nFN/(m.nTP+m.nFN)):0;}
double litiv::ClassifMetrics::CalcPercentBadClassifs(const ClassifMetricsBase& m) {return m.total()>0?(100.0*(m.nFN+m.nFP)/m.total()):0;}
double litiv::ClassifMetrics::CalcMatthewsCorrCoeff(const ClassifMetricsBase& m) {return ((m.nTP+m.nFP)>0)&&((m.nTP+m.nFN)>0)&&((m.nTN+m.nFP)>0)&&((m.nTN+m.nFN)>0)?((((double)m.nTP*m.nTN)-(m.nFP*m.nFN))/sqrt(((double)m.nTP+m.nFP)*(m.nTP+m.nFN)*(m.nTN+m.nFP)*(m.nTN+m.nFN))):0;}

litiv::ClassifMetricsBase litiv::IDatasetEvaluator_<litiv::eDatasetType_VideoSegm>::getMetricsBase() const {
    ClassifMetricsBase oMetricsBase;
    for(const auto& pBatch : getBatches())
        oMetricsBase += dynamic_cast<const IMetricsCalculator_<eDatasetType_VideoSegm>&>(*pBatch).getMetricsBase();
    return oMetricsBase;
}

litiv::ClassifMetrics litiv::IDatasetEvaluator_<litiv::eDatasetType_VideoSegm>::getMetrics(bool bAverage) const {
    if(bAverage) {
        IDataHandlerPtrArray vpBatches = getBatches();
        auto ppBatchIter = vpBatches.begin();
        for(; ppBatchIter!=vpBatches.end() && !(*ppBatchIter)->getTotPackets(); ++ppBatchIter);
        CV_Assert(ppBatchIter!=vpBatches.end());
        ClassifMetrics oMetrics = dynamic_cast<const IMetricsCalculator_<eDatasetType_VideoSegm>&>(**ppBatchIter).getMetrics(bAverage);
        for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
            if((*ppBatchIter)->getTotPackets())
                oMetrics += dynamic_cast<const IMetricsCalculator_<eDatasetType_VideoSegm>&>(**ppBatchIter).getMetrics(bAverage);
        return oMetrics;
    }
    return ClassifMetrics(getMetricsBase());
}

void litiv::IDatasetEvaluator_<litiv::eDatasetType_VideoSegm>::writeEvalReport() const {
    if(getBatches().empty() || !isUsingEvaluator()) {
        std::cout << "No report to write for dataset '" << getName() << "', skipping." << std::endl;
        return;
    }
    for(const auto& pGroupIter : getBatches())
        pGroupIter->writeEvalReport();
    const ClassifMetrics& oMetrics = getMetrics(true);
    std::cout << CxxUtils::clampString(getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+"/overall.txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Video segmentation evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        size_t nOverallPacketCount = 0;
        double dOverallTimeElapsed = 0.0;
        for(const auto& pGroupIter : getBatches()) {
            oMetricsOutput << pGroupIter->writeInlineEvalReport(0);
            nOverallPacketCount += pGroupIter->getTotPackets();
            dOverallTimeElapsed += pGroupIter->getProcessTime();
        }
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << "     overall|" <<
                          std::setw(12) << oMetrics.dRecall << "|" <<
                          std::setw(12) << oMetrics.dSpecificity << "|" <<
                          std::setw(12) << oMetrics.dFPR << "|" <<
                          std::setw(12) << oMetrics.dFNR << "|" <<
                          std::setw(12) << oMetrics.dPBC << "|" <<
                          std::setw(12) << oMetrics.dPrecision << "|" <<
                          std::setw(12) << oMetrics.dFMeasure << "|" <<
                          std::setw(12) << oMetrics.dMCC << "\n";
        oMetricsOutput << "\nHz: " << nOverallPacketCount/dOverallTimeElapsed << "\n";
        oMetricsOutput << "\nSHA1:" << LITIV_VERSION_SHA1 << "\n[" << CxxUtils::getTimeStamp() << "]" << std::endl;
    }
}

#if 0//HAVE_GLSL


//template<>
ClassifMetricsBase litiv::IGLEvaluator_<eDatasetType_VideoSegm>::getCumulativeMetrics() {

}

//template<>
void litiv::IEvaluator_<eDatasetType_VideoSegm>::FetchGLEvaluation(std::shared_ptr<IGLEvaluator_<eDatasetType_VideoSegm>> pGLEvaluator) {
    m_oMetricsBase = pGLEvaluator->getCumulativeMetrics();
}

#endif //HAVE_GLSL

// as defined in the 2012 CDNet scripts/dataset
const uchar litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::s_nSegmPositive = 255;
const uchar litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::s_nSegmNegative = 0;
const uchar litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::s_nSegmOutOfScope = DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL;
const uchar litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::s_nSegmUnknown = DATASETUTILS_VIDEOSEGM_UNKNOWN_VAL;
const uchar litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::s_nSegmShadow = DATASETUTILS_VIDEOSEGM_SHADOW_VAL;

litiv::ClassifMetricsBase litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::getMetricsBase() const {
    // @@@@ fetch gl eval output here, if needed
    return m_oMetricsBase;
}

litiv::ClassifMetrics litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::getMetrics(bool) const {
    return ClassifMetrics(getMetricsBase());
}

std::string litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::writeInlineEvalReport(size_t nIndentSize, size_t nCellSize) const {
    if(!getTotPackets())
        return std::string();
    std::stringstream ssStr;
    ssStr << std::fixed;
    const ClassifMetrics& oMetrics = getMetrics(true);
    ssStr << CxxUtils::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
             std::setw(nCellSize) << oMetrics.dRecall << "|" <<
             std::setw(nCellSize) << oMetrics.dSpecificity << "|" <<
             std::setw(nCellSize) << oMetrics.dFPR << "|" <<
             std::setw(nCellSize) << oMetrics.dFNR << "|" <<
             std::setw(nCellSize) << oMetrics.dPBC << "|" <<
             std::setw(nCellSize) << oMetrics.dPrecision << "|" <<
             std::setw(nCellSize) << oMetrics.dFMeasure << "|" <<
             std::setw(nCellSize) << oMetrics.dMCC << "\n";
    return ssStr.str();
}

void litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::writeEvalReport() const {
    if(!getTotPackets()) {
        std::cout << "No report to write for '" << getName() << "', skipping..." << std::endl;
        return;
    }
    const ClassifMetrics& oMetrics = getMetrics(true);
    std::cout << "\t\t" << CxxUtils::clampString(getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+"/../"+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Video segmentation evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << writeInlineEvalReport(0);
        oMetricsOutput << "\nHz: " << getTotPackets()/getProcessTime() << "\n";
        oMetricsOutput << "\nSHA1:" << LITIV_VERSION_SHA1 << "\n[" << CxxUtils::getTimeStamp() << "]" << std::endl;
    }
}

void litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::pushSegmMask(const cv::Mat& oSegm,size_t nIdx) {
    IDataConsumer_<eDatasetType_VideoSegm,eNotGroup>::pushSegmMask(oSegm,nIdx);
    if(getDatasetInfo()->isUsingEvaluator()) {
        auto pProducer = std::dynamic_pointer_cast<IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(shared_from_this());
        CV_Assert(pProducer);
        const cv::Mat& oGTSegm = pProducer->getGTFrame(nIdx);
        const cv::Mat& oROI = pProducer->getROI();
        CV_Assert(oSegm.type()==CV_8UC1 && oGTSegm.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
        CV_Assert(oSegm.size()==oGTSegm.size() && (oROI.empty() || oSegm.size()==oROI.size()));
        const size_t step_row = oSegm.step.p[0];
        for(size_t i = 0; i<(size_t)oSegm.rows; ++i) {
            const size_t idx_nstep = step_row*i;
            const uchar* input_step_ptr = oSegm.data+idx_nstep;
            const uchar* gt_step_ptr = oGTSegm.data+idx_nstep;
            const uchar* roi_step_ptr = oROI.data+idx_nstep;
            for(int j = 0; j<oSegm.cols; ++j) {
                if(gt_step_ptr[j]!=s_nSegmOutOfScope && gt_step_ptr[j]!=s_nSegmUnknown && (oROI.empty() || roi_step_ptr[j]!=s_nSegmNegative)) {
                    if(input_step_ptr[j]==s_nSegmPositive) {
                        if(gt_step_ptr[j]==s_nSegmPositive)
                            ++m_oMetricsBase.nTP;
                        else // gt_step_ptr[j]==s_nSegmNegative
                            ++m_oMetricsBase.nFP;
                    }
                    else { // input_step_ptr[j]==s_nSegmNegative
                        if(gt_step_ptr[j]==s_nSegmPositive)
                            ++m_oMetricsBase.nFN;
                        else // gt_step_ptr[j]==s_nSegmNegative
                            ++m_oMetricsBase.nTN;
                    }
                    if(gt_step_ptr[j]==s_nSegmShadow) {
                        if(input_step_ptr[j]==s_nSegmPositive)
                            ++m_oMetricsBase.nSE;
                    }
                }
            }
        }
    }
}

cv::Mat litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,litiv::eNotGroup>::getColoredSegmMask(const cv::Mat& oSegm, size_t nIdx) {
    auto pProducer = std::dynamic_pointer_cast<IDataProducer_<eDatasetType_VideoSegm,eNotGroup>>(shared_from_this());
    CV_Assert(pProducer);
    const cv::Mat& oGTSegm = pProducer->getGTFrame(nIdx);
    const cv::Mat& oROI = pProducer->getROI();
    CV_Assert(oSegm.type()==CV_8UC1 && oGTSegm.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
    CV_Assert(oSegm.size()==oGTSegm.size() && (oROI.empty() || oSegm.size()==oROI.size()));
    cv::Mat oResult(oSegm.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oSegm.step.p[0];
    for(size_t i=0; i<(size_t)oSegm.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegm.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegm.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oSegm.cols; ++j) {
            if(gt_step_ptr[j]!=s_nSegmOutOfScope && gt_step_ptr[j]!=s_nSegmUnknown && (oROI.empty() || roi_step_ptr[j]!=s_nSegmNegative) ) {
                if(input_step_ptr[j]==s_nSegmPositive) {
                    if(gt_step_ptr[j]==s_nSegmPositive)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==s_nSegmNegative)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==s_nSegmShadow) {
                        res_step_ptr[j*3+1] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                    else {
                        for(size_t c=0; c<3; ++c)
                            res_step_ptr[j*3+c] = UCHAR_MAX/3;
                    }
                }
                else { // input_step_ptr[j]==s_nSegmNegative
                    if(gt_step_ptr[j]==s_nSegmPositive) {
                        res_step_ptr[j*3] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                }
            }
            else if(!oROI.empty() && roi_step_ptr[j]==s_nSegmNegative) {
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

#if 0//HAVE_GLSL

litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,eNotGroup>::GLVideoSegmDataEvaluator::GLVideoSegmDataEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount) :
        GLImageProcEvaluatorAlgo(pParent,nTotFrameCount,(size_t)ClassifMetricsBase::eCount,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

std::string litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,eNotGroup>::GLVideoSegmDataEvaluator::getComputeShaderSource(size_t nStage) const {
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"#version 430\n"
            "#define VAL_POSITIVE     " << (uint)s_nSegmPositive << "\n"
            "#define VAL_NEGATIVE     " << (uint)s_nSegmNegative << "\n"
            "#define VAL_OUTOFSCOPE   " << (uint)s_nSegmOutOfScope << "\n"
            "#define VAL_UNKNOWN      " << (uint)s_nSegmUnknown << "\n"
            "#define VAL_SHADOW       " << (uint)s_nSegmShadow << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_ROIBinding << ", r8ui) readonly uniform uimage2D imgROI;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_OutputBinding << ", r8ui) readonly uniform uimage2D imgInput;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_GTBinding << ", r8ui) readonly uniform uimage2D imgGT;\n";
    if(m_bUsingDebug) ssSrc <<
            "layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ") writeonly uniform uimage2D imgDebug;\n";
    ssSrc <<"layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << ClassifMetricsBase::eCounter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << ClassifMetricsBase::eCounter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << ClassifMetricsBase::eCounter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << ClassifMetricsBase::eCounter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << ClassifMetricsBase::eCounter_SE*4 << ") uniform atomic_uint nSE;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"void main() {\n"
            "    ivec2 imgCoord = ivec2(gl_GlobalInvocationID.xy);\n"
            "    uint nInputSegmVal = imageLoad(imgInput,imgCoord).r;\n"
            "    uint nGTSegmVal = imageLoad(imgGT,imgCoord).r;\n"
            "    uint nROIVal = imageLoad(imgROI,imgCoord).r;\n"
            "    if(nROIVal!=VAL_NEGATIVE) {\n"
            "        if(nGTSegmVal!=VAL_OUTOFSCOPE && nGTSegmVal!=VAL_UNKNOWN) {\n"
            "            if(nInputSegmVal==VAL_POSITIVE) {\n"
            "                if(nGTSegmVal==VAL_POSITIVE) {\n"
            "                    atomicCounterIncrement(nTP);\n"
            "                }\n"
            "                else { // nGTSegmVal==VAL_NEGATIVE\n"
            "                    atomicCounterIncrement(nFP);\n"
            "                }\n"
            "            }\n"
            "            else { // nInputSegmVal==VAL_NEGATIVE\n"
            "                if(nGTSegmVal==VAL_POSITIVE) {\n"
            "                    atomicCounterIncrement(nFN);\n"
            "                }\n"
            "                else { // nGTSegmVal==VAL_NEGATIVE\n"
            "                    atomicCounterIncrement(nTN);\n"
            "                }\n"
            "            }\n"
            "            if(nGTSegmVal==VAL_SHADOW) {\n"
            "                if(nInputSegmVal==VAL_POSITIVE) {\n"
            "                   atomicCounterIncrement(nSE);\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n";
    if(m_bUsingDebug) { ssSrc <<
            "    uvec4 out_color = uvec4(0,0,0,255);\n"
            "    if(nGTSegmVal!=VAL_OUTOFSCOPE && nGTSegmVal!=VAL_UNKNOWN && nROIVal!=VAL_NEGATIVE) {\n"
            "        if(nInputSegmVal==VAL_POSITIVE) {\n"
            "            if(nGTSegmVal==VAL_POSITIVE) {\n"
            "                out_color.g = uint(255);\n"
            "            }\n"
            "            else if(nGTSegmVal==VAL_NEGATIVE) {\n"
            "                out_color.r = uint(255);\n"
            "            }\n"
            "            else if(nGTSegmVal==VAL_SHADOW) {\n"
            "                out_color.rg = uvec2(255,128);\n"
            "            }\n"
            "            else {\n"
            "                out_color.rgb = uvec3(85);\n"
            "            }\n"
            "        }\n"
            "        else { // nInputSegmVal==VAL_NEGATIVE\n"
            "            if(nGTSegmVal==VAL_POSITIVE) {\n"
            "                out_color.rb = uvec2(255,128);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    else if(nROIVal==VAL_NEGATIVE) {\n"
            "        out_color.rgb = uvec3(128);\n"
            "    }\n"
            "    else if(nInputSegmVal==VAL_POSITIVE) {\n"
            "        out_color.rgb = uvec3(255);\n"
            "    }\n"
            "    else if(nInputSegmVal==VAL_NEGATIVE) {\n"
            "        out_color.rgb = uvec3(0);\n"
            "    }\n"
            "    imageStore(imgDebug,imgCoord,out_color);\n";
    }
    ssSrc << "}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

litiv::ClassifMetricsBase litiv::IDataEvaluator_<litiv::eDatasetType_VideoSegm,eNotGroup>::GLVideoSegmDataEvaluator::getCumulativeMetrics() {
    const cv::Mat& oAtomicCountersQueryBuffer = this->getEvaluationAtomicCounterBuffer();
    ClassifMetricsBase oMetricsBase;
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        oMetricsBase.nTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,ClassifMetricsBase::eCounter_TP);
        oMetricsBase.nTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,ClassifMetricsBase::eCounter_TN);
        oMetricsBase.nFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,ClassifMetricsBase::eCounter_FP);
        oMetricsBase.nFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,ClassifMetricsBase::eCounter_FN);
        oMetricsBase.nSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,ClassifMetricsBase::eCounter_SE);
    }
    return oMetricsBase;
}

#endif //HAVE_GLSL

#if 0

#if USE_BSDS500_BENCHMARK
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wwrite-strings"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wformat="
#pragma clang diagnostic ignored "-Wparentheses"
#pragma clang diagnostic ignored "-Wformat-security"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-pedantic-errors"
#endif //__clang__
#if (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wwrite-strings"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wformat="
#pragma GCC diagnostic ignored "-Wparentheses"
#pragma GCC diagnostic ignored "-Wformat-security"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-pedantic-errors"
#endif //(defined(__GNUC__) || defined(__GNUG__))
#ifdef _MSC_VER
#pragma warning(push,0)
#endif //defined(_MSC_VER)
#include "3rdparty/BSDS500/csa.hpp"
#include "3rdparty/BSDS500/kofn.hpp"
#include "3rdparty/BSDS500/match.hpp"
#ifdef _MSC_VER
#pragma warning(pop)
#endif //defined(_MSC_VER)
#if (defined(__GNUC__) || defined(__GNUG__))
#pragma GCC diagnostic pop
#endif //(defined(__GNUC__) || defined(__GNUG__))
#ifdef __clang__
#pragma clang diagnostic pop
#endif //__clang__
#endif //USE_BSDS500_BENCHMARK

// as defined in the BSDS500 scripts/dataset
const double litiv::Image::Segm::BSDS500BoundaryEvaluator::s_dMaxImageDiagRatioDist = 0.0075;

litiv::Image::Segm::BSDS500BoundaryEvaluator::BSDS500BoundaryEvaluator(size_t nThresholdBins) : m_nThresholdBins(nThresholdBins) {CV_Assert(m_nThresholdBins>0 && m_nThresholdBins<=UCHAR_MAX);}

cv::Mat litiv::Image::Segm::BSDS500BoundaryEvaluator::getColoredSegmMask(const cv::Mat& oSegm, const cv::Mat& oGTSegm, const cv::Mat& /*oUnused*/) const {
    CV_Assert(oSegm.type()==CV_8UC1 && oGTSegm.type()==CV_8UC1);
    CV_Assert(oSegm.cols==oGTSegm.cols && (oGTSegm.rows%oSegm.rows)==0 && (oGTSegm.rows/oSegm.rows)>=1);
    CV_Assert(oSegm.step.p[0]==oGTSegm.step.p[0]);
    const double dMaxDist = s_dMaxImageDiagRatioDist*sqrt(double(oSegm.cols*oSegm.cols+oSegm.rows*oSegm.rows));
    const int nMaxDist = (int)ceil(dMaxDist);
    CV_Assert(dMaxDist>0 && nMaxDist>0);
    cv::Mat oSegm_TP(oSegm.size(),CV_16UC1,cv::Scalar_<ushort>(0));
    cv::Mat oSegm_FN(oSegm.size(),CV_16UC1,cv::Scalar_<ushort>(0));
    cv::Mat oSegm_FP(oSegm.size(),CV_16UC1,cv::Scalar_<ushort>(0));
    const size_t nGTMaskCount = size_t(oGTSegm.rows/oSegm.rows);
    for(size_t nGTMaskIdx=0; nGTMaskIdx<nGTMaskCount; ++nGTMaskIdx) {
        cv::Mat oCurrGTSegmMask = oGTSegm(cv::Rect(0,int(oSegm.rows*nGTMaskIdx),oSegm.cols,oSegm.rows));
        cv::Mat oCurrGTSegmMask_dilated,oSegm_dilated;
        cv::Mat oDilateKernel(2*nMaxDist+1,2*nMaxDist+1,CV_8UC1,cv::Scalar_<uchar>(255));
        cv::dilate(oCurrGTSegmMask,oCurrGTSegmMask_dilated,oDilateKernel);
        cv::dilate(oSegm,oSegm_dilated,oDilateKernel);
        cv::add((oSegm&oCurrGTSegmMask_dilated),oSegm_TP,oSegm_TP,cv::noArray(),CV_16U);
        cv::add((oSegm&(oCurrGTSegmMask_dilated==0)),oSegm_FP,oSegm_FP,cv::noArray(),CV_16U);
        cv::add(((oSegm_dilated==0)&oCurrGTSegmMask),oSegm_FN,oSegm_FN,cv::noArray(),CV_16U);
    }
    cv::Mat oSegm_TP_byte, oSegm_FN_byte, oSegm_FP_byte;
    oSegm_TP.convertTo(oSegm_TP_byte,CV_8U,1.0/nGTMaskCount);
    oSegm_FN.convertTo(oSegm_FN_byte,CV_8U,1.0/nGTMaskCount);
    oSegm_FP.convertTo(oSegm_FP_byte,CV_8U,1.0/nGTMaskCount);
    cv::Mat oResult(oSegm.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const std::vector<int> vnMixPairs = {0,2, 1,0, 2,1};
    cv::mixChannels(std::vector<cv::Mat>{oSegm_FN_byte|oSegm_FP_byte,oSegm_FN_byte,oSegm_TP_byte},std::vector<cv::Mat>{oResult},vnMixPairs.data(),vnMixPairs.size()/2);
    return oResult;
}

void litiv::Image::Segm::BSDS500BoundaryEvaluator::accumulateMetrics(const cv::Mat& oSegm, const cv::Mat& oGTSegm, const cv::Mat& /*oUnused*/) {
    CV_Assert(oSegm.type()==CV_8UC1 && oGTSegm.type()==CV_8UC1);
    CV_Assert(oSegm.isContinuous() && oGTSegm.isContinuous());
    CV_Assert(oSegm.cols==oGTSegm.cols && (oGTSegm.rows%oSegm.rows)==0 && (oGTSegm.rows/oSegm.rows)>=1);
    CV_Assert(oSegm.step.p[0]==oGTSegm.step.p[0]);

    const double dMaxDist = s_dMaxImageDiagRatioDist*sqrt(double(oSegm.cols*oSegm.cols+oSegm.rows*oSegm.rows));
    const double dMaxDistSqr = dMaxDist*dMaxDist;
    const int nMaxDist = (int)ceil(dMaxDist);
    CV_Assert(dMaxDist>0 && nMaxDist>0);

    const std::vector<uchar> vuEvalUniqueVals = PlatformUtils::unique_8uc1_values(oSegm);
    BSDS500ClassifMetricsBase oMetricsBase(m_nThresholdBins);
    CV_DbgAssert(m_voMetricsBase.empty() || m_voMetricsBase[0].vnThresholds==oMetricsBase.vnThresholds);
    cv::Mat oCurrSegmMask(oSegm.size(),CV_8UC1), oTmpSegmMask(oSegm.size(),CV_8UC1);
    cv::Mat oSegmTPAccumulator(oSegm.size(),CV_8UC1);
    size_t nNextEvalUniqueValIdx = 0;
    size_t nThresholdBinIdx = 0;
    while(nThresholdBinIdx<oMetricsBase.vnThresholds.size()) {
        cv::compare(oSegm,oMetricsBase.vnThresholds[nThresholdBinIdx],oTmpSegmMask,cv::CMP_GE);
        litiv::thinning(oTmpSegmMask,oCurrSegmMask);

#if USE_BSDS500_BENCHMARK

        ///////////////////////////////////////////////////////
        // code below is adapted from match.cc::matchEdgeMaps()
        ///////////////////////////////////////////////////////

        const double dOutlierCost = 100*dMaxDist;
        CV_Assert(dOutlierCost>1);
        oSegmTPAccumulator = cv::Scalar_<uchar>(0);
        cv::Mat oGTAccumulator(oCurrSegmMask.size(),CV_8UC1,cv::Scalar_<uchar>(0));
        uint64_t nIndivTP = 0;
        uint64_t nGTPosCount = 0;

        static constexpr int multiplier = 100;
        static constexpr int degree = 6;
        static_assert(degree>0,"csa config bad; degree of outlier connections should be > 0");
        static_assert(multiplier>0,"csa config bad; floating-point weights to integers should be > 0");

        for(size_t nGTMaskIdx=0; nGTMaskIdx<size_t(oGTSegm.rows/oCurrSegmMask.rows); ++nGTMaskIdx) {
            cv::Mat oCurrGTSegmMask = oGTSegm(cv::Rect(0,int(oCurrSegmMask.rows*nGTMaskIdx),oCurrSegmMask.cols,oCurrSegmMask.rows));
            cv::Mat oMatchable_SEGM(oCurrSegmMask.size(),CV_8UC1,cv::Scalar_<uchar>(0));
            cv::Mat oMatchable_GT(oCurrSegmMask.size(),CV_8UC1,cv::Scalar_<uchar>(0));
            // Figure out which nodes are matchable, i.e. within maxDist
            // of another node.
            for(int i=0; i<oCurrSegmMask.rows; ++i) {
                for(int j=0; j<oCurrSegmMask.cols; ++j) {
                    if(!oCurrGTSegmMask.at<uchar>(i,j)) continue;
                    for(int u=-nMaxDist; u<=nMaxDist; ++u) {
                        if(i+u<0) continue;
                        if(i+u>=oCurrSegmMask.rows) continue;
                        if(double(u)>dMaxDist) continue;
                        for(int v=-nMaxDist; v<=nMaxDist; ++v) {
                            if(j+v<0) continue;
                            if(j+v>=oCurrSegmMask.cols) continue;
                            if(double(v)>dMaxDist) continue;
                            const double dCurrDistSqr = u*u+v*v;
                            if(dCurrDistSqr>dMaxDistSqr) continue;
                            if(oCurrSegmMask.at<uchar>(i+u,j+v)) {
                                oMatchable_SEGM.at<uchar>(i+u,j+v) = UCHAR_MAX;
                                oMatchable_GT.at<uchar>(i,j) = UCHAR_MAX;
                            }
                        }
                    }
                }
            }

            int nNodeCount_SEGM=0, nNodeCount_GT=0;
            std::vector<cv::Point2i> voNodeToPxLUT_SEGM,voNodeToPxLUT_GT;
            cv::Mat oPxToNodeLUT_SEGM(oCurrSegmMask.size(),CV_32SC1,cv::Scalar_<int>(-1));
            cv::Mat oPxToNodeLUT_GT(oCurrSegmMask.size(),CV_32SC1,cv::Scalar_<int>(-1));
            // Count the number of nodes on each side of the match.
            // Construct nodeID->pixel and pixel->nodeID maps.
            // Node IDs range from [0,nNodeCount_SEGM) and [0,nNodeCount_GT).
            for(int i=0; i<oCurrSegmMask.rows; ++i) {
                for(int j=0; j<oCurrSegmMask.cols; ++j) {
                    cv::Point2i px(j,i);
                    if(oMatchable_SEGM.at<uchar>(px)) {
                        oPxToNodeLUT_SEGM.at<int>(px) = nNodeCount_SEGM;
                        voNodeToPxLUT_SEGM.push_back(px);
                        ++nNodeCount_SEGM;
                    }
                    if(oMatchable_GT.at<uchar>(px)) {
                        oPxToNodeLUT_GT.at<int>(px) = nNodeCount_GT;
                        voNodeToPxLUT_GT.push_back(px);
                        ++nNodeCount_GT;
                    }
                }
            }

            struct Edge {
                int nNodeIdx_SEGM;
                int nNodeIdx_GT;
                double dEdgeDist;
            };
            std::vector<Edge> voEdges;
            // Construct the list of edges between pixels within maxDist.
            for(int i=0; i<oCurrSegmMask.rows; ++i) {
                for(int j=0; j<oCurrSegmMask.cols; ++j) {
                    if(!oMatchable_GT.at<uchar>(i,j)) continue;
                    for(int u=-nMaxDist; u<=nMaxDist; ++u) {
                        if(i+u<0) continue;
                        if(i+u>=oCurrSegmMask.rows) continue;
                        if(double(u)>dMaxDist) continue;
                        for(int v=-nMaxDist; v<=nMaxDist; ++v) {
                            if(j+v<0) continue;
                            if(j+v>=oCurrSegmMask.cols) continue;
                            if(double(v)>dMaxDist) continue;
                            if(!oMatchable_SEGM.at<uchar>(i+u,j+v)) continue;
                            const double dCurrDistSqr = u*u+v*v;
                            if(dCurrDistSqr>dMaxDistSqr) continue;
                            Edge e;
                            e.nNodeIdx_SEGM = oPxToNodeLUT_SEGM.at<int>(i+u,j+v);
                            e.nNodeIdx_GT = oPxToNodeLUT_GT.at<int>(i,j);
                            e.dEdgeDist = sqrt(dCurrDistSqr);
                            CV_DbgAssert(e.nNodeIdx_SEGM>=0 && e.nNodeIdx_SEGM<nNodeCount_SEGM);
                            CV_DbgAssert(e.nNodeIdx_GT>=0 && e.nNodeIdx_GT<nNodeCount_GT);
                            voEdges.push_back(e);
                        }
                    }
                }
            }

            // The cardinality of the match is n.
            const int n = nNodeCount_SEGM+nNodeCount_GT;
            const int nmin = std::min(nNodeCount_SEGM,nNodeCount_GT);
            const int nmax = std::max(nNodeCount_SEGM,nNodeCount_GT);

            // Compute the degree of various outlier connections.
            const int degree_SEGM = std::max(0,std::min(degree,nNodeCount_SEGM-1)); // from map1
            const int degree_GT = std::max(0,std::min(degree,nNodeCount_GT-1)); // from map2
            const int degree_mix = std::min(degree,std::min(nNodeCount_SEGM,nNodeCount_GT)); // between outliers
            const int dmax = std::max(degree_SEGM,std::max(degree_GT,degree_mix));

            CV_DbgAssert(nNodeCount_SEGM==0 || (degree_SEGM>=0 && degree_SEGM<nNodeCount_SEGM));
            CV_DbgAssert(nNodeCount_GT==0 || (degree_GT>=0 && degree_GT<nNodeCount_GT));
            CV_DbgAssert(degree_mix>=0 && degree_mix<=nmin);

            // Count the number of edges.
            int m = 0;
            m += (int)voEdges.size();              // real connections
            m += degree_SEGM*nNodeCount_SEGM; // outlier connections
            m += degree_GT*nNodeCount_GT;     // outlier connections
            m += degree_mix*nmax;             // outlier-outlier connections
            m += n;                           // high-cost perfect match overlay

            // If the graph is empty, then there's nothing to do.
            if(m>0) {
                // Weight of outlier connections.
                const int nOutlierWeight = (int)ceil(dOutlierCost*multiplier);
                // Scratch array for outlier edges.
                std::vector<int> vnOutliers(dmax);
                // Construct the input graph for the assignment problem.
                cv::Mat oGraph(m,3,CV_32SC1);
                int nGraphIdx = 0;
                // real edges
                for(int a=0; a<(int)voEdges.size(); ++a) {
                    int nNodeIdx_SEGM = voEdges[a].nNodeIdx_SEGM;
                    int nNodeIdx_GT = voEdges[a].nNodeIdx_GT;
                    CV_DbgAssert(nNodeIdx_SEGM>=0 && nNodeIdx_SEGM<nNodeCount_SEGM);
                    CV_DbgAssert(nNodeIdx_GT>=0 && nNodeIdx_GT<nNodeCount_GT);
                    oGraph.at<int>(nGraphIdx,0) = nNodeIdx_SEGM;
                    oGraph.at<int>(nGraphIdx,1) = nNodeIdx_GT;
                    oGraph.at<int>(nGraphIdx,2) = (int)rint(voEdges[a].dEdgeDist*multiplier);
                    nGraphIdx++;
                }
                // outliers edges for map1, exclude diagonal
                for(int nNodeIdx_SEGM=0; nNodeIdx_SEGM<nNodeCount_SEGM; ++nNodeIdx_SEGM) {
                    BSDS500::kOfN(degree_SEGM,nNodeCount_SEGM-1,vnOutliers.data());
                    for(int a=0; a<degree_SEGM; a++) {
                        int j = vnOutliers[a];
                        if(j>=nNodeIdx_SEGM) {j++;}
                        CV_DbgAssert(nNodeIdx_SEGM!=j);
                        CV_DbgAssert(j>=0 && j<nNodeCount_SEGM);
                        oGraph.at<int>(nGraphIdx,0) = nNodeIdx_SEGM;
                        oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+j;
                        oGraph.at<int>(nGraphIdx,2) = nOutlierWeight;
                        nGraphIdx++;
                    }
                }
                // outliers edges for map2, exclude diagonal
                for(int nNodeIdx_GT = 0; nNodeIdx_GT<nNodeCount_GT; nNodeIdx_GT++) {
                    BSDS500::kOfN(degree_GT,nNodeCount_GT-1,vnOutliers.data());
                    for(int a = 0; a<degree_GT; a++) {
                        int i = vnOutliers[a];
                        if(i>=nNodeIdx_GT) {i++;}
                        CV_DbgAssert(i!=nNodeIdx_GT);
                        CV_DbgAssert(i>=0 && i<nNodeCount_GT);
                        oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+i;
                        oGraph.at<int>(nGraphIdx,1) = nNodeIdx_GT;
                        oGraph.at<int>(nGraphIdx,2) = nOutlierWeight;
                        nGraphIdx++;
                    }
                }
                // outlier-to-outlier edges
                for(int i = 0; i<nmax; i++) {
                    BSDS500::kOfN(degree_mix,nmin,vnOutliers.data());
                    for(int a = 0; a<degree_mix; a++) {
                        const int j = vnOutliers[a];
                        CV_DbgAssert(j>=0 && j<nmin);
                        if(nNodeCount_SEGM<nNodeCount_GT) {
                            CV_DbgAssert(i>=0 && i<nNodeCount_GT);
                            CV_DbgAssert(j>=0 && j<nNodeCount_SEGM);
                            oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+i;
                            oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+j;
                        }
                        else {
                            CV_DbgAssert(i>=0 && i<nNodeCount_SEGM);
                            CV_DbgAssert(j>=0 && j<nNodeCount_GT);
                            oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+j;
                            oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+i;
                        }
                        oGraph.at<int>(nGraphIdx,2) = nOutlierWeight;
                        nGraphIdx++;
                    }
                }
                // perfect match overlay (diagonal)
                for(int i = 0; i<nNodeCount_SEGM; i++) {
                    oGraph.at<int>(nGraphIdx,0) = i;
                    oGraph.at<int>(nGraphIdx,1) = nNodeCount_GT+i;
                    oGraph.at<int>(nGraphIdx,2) = nOutlierWeight*multiplier;
                    nGraphIdx++;
                }
                for(int i = 0; i<nNodeCount_GT; i++) {
                    oGraph.at<int>(nGraphIdx,0) = nNodeCount_SEGM+i;
                    oGraph.at<int>(nGraphIdx,1) = i;
                    oGraph.at<int>(nGraphIdx,2) = nOutlierWeight*multiplier;
                    nGraphIdx++;
                }
                CV_DbgAssert(nGraphIdx==m);

                // Check all the edges, and set the values up for CSA.
                for(int i = 0; i<m; i++) {
                    CV_DbgAssert(oGraph.at<int>(i,0)>=0 && oGraph.at<int>(i,0)<n);
                    CV_DbgAssert(oGraph.at<int>(i,1)>=0 && oGraph.at<int>(i,1)<n);
                    oGraph.at<int>(i,0) += 1;
                    oGraph.at<int>(i,1) += 1+n;
                }

                // Solve the assignment problem.
                BSDS500::CSA oCSASolver(2*n,m,(int*)oGraph.data);
                CV_Assert(oCSASolver.edges()==n);

                cv::Mat oOutGraph(n,3,CV_32SC1);
                for(int i = 0; i<n; i++) {
                    int a,b,c;
                    oCSASolver.edge(i,a,b,c);
                    oOutGraph.at<int>(i,0) = a-1;
                    oOutGraph.at<int>(i,1) = b-1-n;
                    oOutGraph.at<int>(i,2) = c;
                }

                // Check the solution.
                // Count the number of high-cost edges from the perfect match
                // overlay that were used in the match.
                int nOverlayCount = 0;
                for(int a = 0; a<n; a++) {
                    const int i = oOutGraph.at<int>(a,0);
                    const int j = oOutGraph.at<int>(a,1);
                    const int c = oOutGraph.at<int>(a,2);
                    CV_DbgAssert(i>=0 && i<n);
                    CV_DbgAssert(j>=0 && j<n);
                    CV_DbgAssert(c>=0);
                    // edge from high-cost perfect match overlay
                    if(c==nOutlierWeight*multiplier) {nOverlayCount++;}
                    // skip outlier edges
                    if(i>=nNodeCount_SEGM) {continue;}
                    if(j>=nNodeCount_GT) {continue;}
                    // for edges between real nodes, check the edge weight
                    CV_DbgAssert((int)rint(sqrt((voNodeToPxLUT_SEGM[i].x-voNodeToPxLUT_GT[j].x)*(voNodeToPxLUT_SEGM[i].x-voNodeToPxLUT_GT[j].x)+(voNodeToPxLUT_SEGM[i].y-voNodeToPxLUT_GT[j].y)*(voNodeToPxLUT_SEGM[i].y-voNodeToPxLUT_GT[j].y))*multiplier)==c);
                }

                // Print a warning if any of the edges from the perfect match overlay
                // were used.  This should happen rarely.  If it happens frequently,
                // then the outlier connectivity should be increased.
                if(nOverlayCount>5) {
                    fprintf(stderr,"%s:%d: WARNING: The match includes %d outlier(s) from the perfect match overlay.\n",__FILE__,__LINE__,nOverlayCount);
                }

                // Compute match arrays.
                for(int a = 0; a<n; a++) {
                    // node ids
                    const int i = oOutGraph.at<int>(a,0);
                    const int j = oOutGraph.at<int>(a,1);
                    // skip outlier edges
                    if(i>=nNodeCount_SEGM) {continue;}
                    if(j>=nNodeCount_GT) {continue;}
                    // for edges between real nodes, check the edge weight
                    const cv::Point2i oPx_SEGM = voNodeToPxLUT_SEGM[i];
                    const cv::Point2i oPx_GT = voNodeToPxLUT_GT[j];
                    // record edges
                    CV_Assert(oCurrSegmMask.at<uchar>(oPx_SEGM) && oCurrGTSegmMask.at<uchar>(oPx_GT));
                    oSegmTPAccumulator.at<uchar>(oPx_SEGM) = UCHAR_MAX;
                    ++nIndivTP;
                }
            }
            nGTPosCount += cv::countNonZero(oCurrGTSegmMask);
            oGTAccumulator |= oCurrGTSegmMask;
        }

#else //!USE_BSDS500_BENCHMARK

        oSegmTPAccumulator = cv::Scalar_<uchar>(0); // accP |= ...
        uint64_t nIndivTP = 0; // cntR += ...
        uint64_t nGTPosCount = 0; // sumR += ...
        for(size_t nGTMaskIdx = 0; nGTMaskIdx<size_t(oGTSegm.rows/oCurrSegmMask.rows); ++nGTMaskIdx) {
            cv::Mat oCurrGTSegmMask = oGTSegm(cv::Rect(0,int(oCurrSegmMask.rows*nGTMaskIdx),oCurrSegmMask.cols,oCurrSegmMask.rows));
            for(int i = 0; i<oCurrSegmMask.rows; ++i) {
                for(int j = 0; j<oCurrSegmMask.cols; ++j) {
                    if(!oCurrGTSegmMask.at<uchar>(i,j)) continue;
                    ++nGTPosCount;
                    bool bFoundMatch = false;
                    for(int u = -nMaxDist; u<=nMaxDist && !bFoundMatch; ++u) {
                        if(i+u<0) continue;
                        if(i+u>=oCurrSegmMask.rows) continue;
                        if(double(u)>dMaxDist) continue;
                        for(int v = -nMaxDist; v<=nMaxDist && !bFoundMatch; ++v) {
                            if(j+v<0) continue;
                            if(j+v>=oCurrSegmMask.cols) continue;
                            if(double(v)>dMaxDist) continue;
                            const double dCurrDistSqr = u*u+v*v;
                            if(dCurrDistSqr>dMaxDistSqr) continue;
                            if(oCurrSegmMask.at<uchar>(i+u,j+v)) {
                                ++nIndivTP;
                                oSegmTPAccumulator.at<uchar>(i+u,j+v) = UCHAR_MAX;
                                bFoundMatch = true;
                            }
                        }
                    }
                }
            }
        }

#endif //!USE_BSDS500_BENCHMARK

        //re = TP / (TP + FN)
        CV_Assert(nGTPosCount>=nIndivTP);
        oMetricsBase.vnIndivTP[nThresholdBinIdx] = nIndivTP;
        oMetricsBase.vnIndivTPFN[nThresholdBinIdx] = nGTPosCount;

        //pr = TP / (TP + FP)
        uint64_t nSegmTPAccCount = uint64_t(cv::countNonZero(oSegmTPAccumulator));
        uint64_t nSegmPosCount = uint64_t(cv::countNonZero(oCurrSegmMask));
        CV_Assert(nSegmPosCount>=nSegmTPAccCount);
        oMetricsBase.vnTotalTP[nThresholdBinIdx] = nSegmTPAccCount;
        oMetricsBase.vnTotalTPFP[nThresholdBinIdx] = nSegmPosCount;
        while(nNextEvalUniqueValIdx+1<vuEvalUniqueVals.size() && vuEvalUniqueVals[nNextEvalUniqueValIdx]<=oMetricsBase.vnThresholds[nThresholdBinIdx])
            ++nNextEvalUniqueValIdx;
        while(++nThresholdBinIdx<oMetricsBase.vnThresholds.size() && oMetricsBase.vnThresholds[nThresholdBinIdx]<=vuEvalUniqueVals[nNextEvalUniqueValIdx]) {
            oMetricsBase.vnIndivTP[nThresholdBinIdx] = oMetricsBase.vnIndivTP[nThresholdBinIdx-1];
            oMetricsBase.vnIndivTPFN[nThresholdBinIdx] = oMetricsBase.vnIndivTPFN[nThresholdBinIdx-1];
            oMetricsBase.vnTotalTP[nThresholdBinIdx] = oMetricsBase.vnTotalTP[nThresholdBinIdx-1];
            oMetricsBase.vnTotalTPFP[nThresholdBinIdx] = oMetricsBase.vnTotalTPFP[nThresholdBinIdx-1];
        }

        const float fCompltRatio = float(nThresholdBinIdx)/oMetricsBase.vnThresholds.size();
        litiv::updateConsoleProgressBar("BSDS500 eval:",fCompltRatio);
    }
    litiv::cleanConsoleRow();
    m_voMetricsBase.push_back(oMetricsBase);
}

litiv::Image::Segm::BSDS500BoundaryEvaluator::BSDS500ClassifMetricsBase::BSDS500ClassifMetricsBase(size_t nThresholdsBins) : vnIndivTP(nThresholdsBins,0), vnIndivTPFN(nThresholdsBins,0), vnTotalTP(nThresholdsBins,0), vnTotalTPFP(nThresholdsBins,0), vnThresholds(PlatformUtils::linspace<uchar>(0,UCHAR_MAX,nThresholdsBins,false)) {}

void litiv::Image::Segm::BSDS500BoundaryEvaluator::setThresholdBins(size_t nThresholdBins) {
    CV_Assert(m_nThresholdBins>0 && m_nThresholdBins<=UCHAR_MAX);
    CV_Assert(m_voMetricsBase.empty() || m_voMetricsBase[0].vnThresholds.size()==nThresholdBins); // can't change once we started the eval
    m_nThresholdBins = nThresholdBins;
}

size_t litiv::Image::Segm::BSDS500BoundaryEvaluator::getThresholdBins() const {
    return m_nThresholdBins;
}

void litiv::Image::Segm::BSDS500BoundaryEvaluator::CalcMetrics(const WorkBatch& oBatch, BSDS500Metrics& oRes) {
    auto pEval = std::dynamic_pointer_cast<BSDS500BoundaryEvaluator>(oBatch.m_pEvaluator);
    CV_Assert(pEval!=nullptr && !pEval->m_voMetricsBase.empty());
    oRes.dTimeElapsed_sec = pEval->dTimeElapsed_sec;
    BSDS500ClassifMetricsBase oCumulMetricsBase(pEval->m_nThresholdBins);
    BSDS500ClassifMetricsBase oMaxClassifMetricsBase(1);
    const size_t nImageCount = pEval->m_voMetricsBase.size();
    oRes.voBestImageScores.resize(nImageCount);
    for(size_t nImageIdx = 0; nImageIdx<nImageCount; ++nImageIdx) {
        CV_DbgAssert(!pEval->m_voMetricsBase[nImageIdx].vnIndivTP.empty() && !pEval->m_voMetricsBase[nImageIdx].vnIndivTPFN.empty());
        CV_DbgAssert(!pEval->m_voMetricsBase[nImageIdx].vnTotalTP.empty() && !pEval->m_voMetricsBase[nImageIdx].vnTotalTPFP.empty());
        CV_DbgAssert(pEval->m_voMetricsBase[nImageIdx].vnIndivTP.size()==pEval->m_voMetricsBase[nImageIdx].vnIndivTPFN.size());
        CV_DbgAssert(pEval->m_voMetricsBase[nImageIdx].vnTotalTP.size()==pEval->m_voMetricsBase[nImageIdx].vnTotalTPFP.size());
        CV_DbgAssert(pEval->m_voMetricsBase[nImageIdx].vnIndivTP.size()==pEval->m_voMetricsBase[nImageIdx].vnTotalTP.size());
        CV_DbgAssert(pEval->m_voMetricsBase[nImageIdx].vnThresholds.size()==pEval->m_voMetricsBase[nImageIdx].vnTotalTP.size());
        CV_DbgAssert(nImageIdx==0 || pEval->m_voMetricsBase[nImageIdx].vnIndivTP.size()==pEval->m_voMetricsBase[nImageIdx-1].vnIndivTP.size());
        CV_DbgAssert(nImageIdx==0 || pEval->m_voMetricsBase[nImageIdx].vnThresholds==pEval->m_voMetricsBase[nImageIdx-1].vnThresholds);
        std::vector<BSDS500Score> voImageScore_PerThreshold(pEval->m_nThresholdBins);
        for(size_t nThresholdIdx = 0; nThresholdIdx<pEval->m_nThresholdBins; ++nThresholdIdx) {
            voImageScore_PerThreshold[nThresholdIdx].dRecall = ClassifMetrics::CalcRecall(pEval->m_voMetricsBase[nImageIdx].vnIndivTP[nThresholdIdx],pEval->m_voMetricsBase[nImageIdx].vnIndivTPFN[nThresholdIdx]);
            voImageScore_PerThreshold[nThresholdIdx].dPrecision = ClassifMetrics::CalcPrecision(pEval->m_voMetricsBase[nImageIdx].vnTotalTP[nThresholdIdx],pEval->m_voMetricsBase[nImageIdx].vnTotalTPFP[nThresholdIdx]);
            voImageScore_PerThreshold[nThresholdIdx].dFMeasure = ClassifMetrics::CalcFMeasure(voImageScore_PerThreshold[nThresholdIdx].dRecall,voImageScore_PerThreshold[nThresholdIdx].dPrecision);
            voImageScore_PerThreshold[nThresholdIdx].dThreshold = double(pEval->m_voMetricsBase[nImageIdx].vnThresholds[nThresholdIdx])/UCHAR_MAX;
            oCumulMetricsBase.vnIndivTP[nThresholdIdx] += pEval->m_voMetricsBase[nImageIdx].vnIndivTP[nThresholdIdx];
            oCumulMetricsBase.vnIndivTPFN[nThresholdIdx] += pEval->m_voMetricsBase[nImageIdx].vnIndivTPFN[nThresholdIdx];
            oCumulMetricsBase.vnTotalTP[nThresholdIdx] += pEval->m_voMetricsBase[nImageIdx].vnTotalTP[nThresholdIdx];
            oCumulMetricsBase.vnTotalTPFP[nThresholdIdx] += pEval->m_voMetricsBase[nImageIdx].vnTotalTPFP[nThresholdIdx];
        }
        oRes.voBestImageScores[nImageIdx] = FindMaxFMeasure(voImageScore_PerThreshold);
        size_t nMaxFMeasureIdx = (size_t)std::distance(voImageScore_PerThreshold.begin(),std::max_element(voImageScore_PerThreshold.begin(),voImageScore_PerThreshold.end(),[](const BSDS500Score& n1, const BSDS500Score& n2){
            return n1.dFMeasure<n2.dFMeasure;
        }));
        oMaxClassifMetricsBase.vnIndivTP[0] += pEval->m_voMetricsBase[nImageIdx].vnIndivTP[nMaxFMeasureIdx];
        oMaxClassifMetricsBase.vnIndivTPFN[0] += pEval->m_voMetricsBase[nImageIdx].vnIndivTPFN[nMaxFMeasureIdx];
        oMaxClassifMetricsBase.vnTotalTP[0] += pEval->m_voMetricsBase[nImageIdx].vnTotalTP[nMaxFMeasureIdx];
        oMaxClassifMetricsBase.vnTotalTPFP[0] += pEval->m_voMetricsBase[nImageIdx].vnTotalTPFP[nMaxFMeasureIdx];
    }
    // ^^^ voBestImageScores => eval_bdry_img.txt
    oRes.voThresholdScores.resize(pEval->m_nThresholdBins);
    for(size_t nThresholdIdx = 0; nThresholdIdx<oCumulMetricsBase.vnThresholds.size(); ++nThresholdIdx) {
        oRes.voThresholdScores[nThresholdIdx].dRecall = ClassifMetrics::CalcRecall(oCumulMetricsBase.vnIndivTP[nThresholdIdx],oCumulMetricsBase.vnIndivTPFN[nThresholdIdx]);
        oRes.voThresholdScores[nThresholdIdx].dPrecision = ClassifMetrics::CalcPrecision(oCumulMetricsBase.vnTotalTP[nThresholdIdx],oCumulMetricsBase.vnTotalTPFP[nThresholdIdx]);
        oRes.voThresholdScores[nThresholdIdx].dFMeasure = ClassifMetrics::CalcFMeasure(oRes.voThresholdScores[nThresholdIdx].dRecall,oRes.voThresholdScores[nThresholdIdx].dPrecision);
        oRes.voThresholdScores[nThresholdIdx].dThreshold = double(oCumulMetricsBase.vnThresholds[nThresholdIdx])/UCHAR_MAX;
    }
    // ^^^ voThresholdScores => eval_bdry_thr.txt
    oRes.oBestScore = FindMaxFMeasure(oRes.voThresholdScores);
    oRes.dMaxRecall = ClassifMetrics::CalcRecall(oMaxClassifMetricsBase.vnIndivTP[0],oMaxClassifMetricsBase.vnIndivTPFN[0]);
    oRes.dMaxPrecision = ClassifMetrics::CalcPrecision(oMaxClassifMetricsBase.vnTotalTP[0],oMaxClassifMetricsBase.vnTotalTPFP[0]);
    oRes.dMaxFMeasure = ClassifMetrics::CalcFMeasure(oRes.dMaxRecall,oRes.dMaxPrecision);
    oRes.dAreaPR = 0;
    std::vector<size_t> vnCumulRecallIdx_uniques = PlatformUtils::unique_indexes(oRes.voThresholdScores,
        [&oRes](size_t n1, size_t n2) {
            return oRes.voThresholdScores[n1].dRecall<oRes.voThresholdScores[n2].dRecall;
        },
        [&oRes](size_t n1, size_t n2) {
            return oRes.voThresholdScores[n1].dRecall==oRes.voThresholdScores[n2].dRecall;
        }
    );
    if(vnCumulRecallIdx_uniques.size()>1) {
        std::vector<double> vdCumulRecall_uniques(vnCumulRecallIdx_uniques.size());
        std::vector<double> vdCumulPrecision_uniques(vnCumulRecallIdx_uniques.size());
        for(size_t n = 0; n<vnCumulRecallIdx_uniques.size(); ++n) {
            vdCumulRecall_uniques[n] = oRes.voThresholdScores[vnCumulRecallIdx_uniques[n]].dRecall;
            vdCumulPrecision_uniques[n] = oRes.voThresholdScores[vnCumulRecallIdx_uniques[n]].dPrecision;
        }
        const size_t nInterpReqIdxCount = 100;
        std::vector<double> vdInterpReqIdx(nInterpReqIdxCount+1);
        for(size_t n = 0; n<=nInterpReqIdxCount; ++n)
            vdInterpReqIdx[n] = double(n)/nInterpReqIdxCount;
        std::vector<double> vdInterpVals = PlatformUtils::interp1(vdCumulRecall_uniques,vdCumulPrecision_uniques,vdInterpReqIdx);
        if(!vdInterpVals.empty())
            for(size_t n = 0; n<=vdInterpVals.size(); ++n)
                oRes.dAreaPR += vdInterpVals[n]*0.01;
    }
    // ^^^ oCumulScore,dMaxRecall,dMaxPrecision,dMaxFMeasure,dAreaPR => eval_bdry.txt
}

void litiv::Image::Segm::BSDS500BoundaryEvaluator::writeEvalReport(const std::string& sRootPath, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) {
    if(!vpGroups.empty()) {
        size_t nOverallImageCount = 0;
        std::vector<BSDS500Metrics> voBatchMetrics;
        std::vector<std::string> vsBatchNames;
        for(auto ppGroupIter = vpGroups.begin(); ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
            for(auto ppBatchIter = (*ppGroupIter)->m_vpBatches.begin(); ppBatchIter!=(*ppGroupIter)->m_vpBatches.end(); ++ppBatchIter) {
                voBatchMetrics.push_back(BSDS500Metrics());
                writeEvalReport(**ppBatchIter,voBatchMetrics.back());
                vsBatchNames.push_back(CxxUtils::clampString((*ppBatchIter)->m_sName,10));
                nOverallImageCount += (*ppBatchIter)->GetTotalImageCount();
            }
        }
        if(!voBatchMetrics.empty()) {
            double dBestRecall=0,dBestPrecision=0,dBestFMeasure=0;
            double dMaxRecall=0,dMaxPrecision=0,dMaxFMeasure=0;
            double dTimeElapsed_sec = 0;
            const size_t nBatchCount = voBatchMetrics.size();
            for(size_t n=0; n<nBatchCount; ++n) {
                dBestRecall += voBatchMetrics[n].oBestScore.dRecall/nBatchCount;
                dBestPrecision += voBatchMetrics[n].oBestScore.dPrecision/nBatchCount;
                dBestFMeasure += voBatchMetrics[n].oBestScore.dFMeasure/nBatchCount;
                dMaxRecall += voBatchMetrics[n].dMaxRecall/nBatchCount;
                dMaxPrecision += voBatchMetrics[n].dMaxPrecision/nBatchCount;
                dMaxFMeasure += voBatchMetrics[n].dMaxFMeasure/nBatchCount;
                dTimeElapsed_sec += voBatchMetrics[n].dTimeElapsed_sec;
            }
            std::cout << "\t" << std::setfill(' ') << @@@@std::setw(12) << "ALL-AVG" << " : MaxRcl=" << std::fixed << std::setprecision(4) << dMaxRecall << " MaxPrc=" << dMaxPrecision << " MaxFM=" << dMaxFMeasure << std::endl;
            std::cout << "\t" << std::setfill(' ') << std::setw(12) << " " <<       " : BestRcl=" << std::fixed << std::setprecision(4) << dBestRecall << " BestPrc=" << dBestPrecision << " BestFM=" << dBestFMeasure << std::endl;
#if USE_BSDS500_BENCHMARK
            std::ofstream oMetricsInfoOutput(sRootPath+"/reimpl_eval.txt");
#else //!USE_BSDS500_BENCHMARK
            std::ofstream oMetricsInfoOutput(sRootPath+"/homemade_eval.txt");
#endif //!USE_BSDS500_BENCHMARK
            if(oMetricsInfoOutput.is_open()) {
                oMetricsInfoOutput << "BSDS500 edge detection evaluation for " << voBatchMetrics.size() << " image set(s):" << std::endl;
                oMetricsInfoOutput << std::endl;
                oMetricsInfoOutput << std::fixed << std::setprecision(8);
                oMetricsInfoOutput << "Work batch results :" << std::endl;
                oMetricsInfoOutput << "           MaxRcl     MaxPrc     MaxFM                 BestRcl    BestPrc    BestFM     @Threshold " << std::endl;
                for(size_t g = 0; g<vsBatchNames.size(); ++g)
                    oMetricsInfoOutput << vsBatchNames[g] << " " << voBatchMetrics[g].dMaxRecall << " " << voBatchMetrics[g].dMaxPrecision << " " << voBatchMetrics[g].dMaxFMeasure << "            " << voBatchMetrics[g].oBestScore.dRecall << " " << voBatchMetrics[g].oBestScore.dPrecision << " " << voBatchMetrics[g].oBestScore.dFMeasure << " " << voBatchMetrics[g].oBestScore.dThreshold << std::endl;
                oMetricsInfoOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
                oMetricsInfoOutput << "averaged   " << dMaxRecall << " " << dMaxPrecision << " " << dMaxFMeasure << "            " << dBestRecall << " " << dBestPrecision << " " << dBestFMeasure << std::endl;
                oMetricsInfoOutput << std::endl;
                oMetricsInfoOutput << "Overall FPS: " << nOverallImageCount/dTimeElapsed_sec << std::endl;
                oMetricsOutput << "\nSHA1:" << LITIV_VERSION_SHA1 << "\n[" << CxxUtils::getTimeStamp() << "]" << std::endl;
            }
        }
    }
}

void litiv::Image::Segm::BSDS500BoundaryEvaluator::writeEvalReport(const WorkBatch& oBatch, BSDS500Metrics& oRes) {
    CalcMetrics(oBatch,oRes);
#if USE_BSDS500_BENCHMARK
    const std::string sOutputPath = oBatch.m_sOutputPath+"/../"+oBatch.m_sName+"_reimpl_eval/";
#else //!USE_BSDS500_BENCHMARK
    const std::string sOutputPath = oBatch.m_sOutputPath+"/../"+oBatch.m_sName+"_homemade_eval/";
#endif //!USE_BSDS500_BENCHMARK
    PlatformUtils::CreateDirIfNotExist(sOutputPath);
    std::ofstream oImageScoresOutput(sOutputPath+"/eval_bdry_img.txt");
    if(oImageScoresOutput.is_open())
        for(size_t n=0; n<oRes.voBestImageScores.size(); ++n)
            oImageScoresOutput << cv::format("%10d %10g %10g %10g %10g\n",n+1,oRes.voBestImageScores[n].dThreshold,oRes.voBestImageScores[n].dRecall,oRes.voBestImageScores[n].dPrecision,oRes.voBestImageScores[n].dFMeasure);
    std::ofstream oThresholdMetricsOutput(sOutputPath+"/eval_bdry_thr.txt");
    if(oThresholdMetricsOutput.is_open())
        for(size_t n=0; n<oRes.voThresholdScores.size(); ++n)
            oThresholdMetricsOutput << cv::format("%10g %10g %10g %10g\n",oRes.voThresholdScores[n].dThreshold,oRes.voThresholdScores[n].dRecall,oRes.voThresholdScores[n].dPrecision,oRes.voThresholdScores[n].dFMeasure);
    std::ofstream oOverallMetricsOutput(sOutputPath+"/eval_bdry.txt");
    if(oOverallMetricsOutput.is_open())
        oOverallMetricsOutput << cv::format("%10g %10g %10g %10g %10g %10g %10g %10g\n",oRes.oBestScore.dThreshold,oRes.oBestScore.dRecall,oRes.oBestScore.dPrecision,oRes.oBestScore.dFMeasure,oRes.dMaxRecall,oRes.dMaxPrecision,oRes.dMaxFMeasure,oRes.dAreaPR);
    std::cout << "\t\t" << CxxUtils::clampString(oBatch.m_sName,12) << " : MaxRcl=" << std::fixed << std::setprecision(4) << oRes.dMaxRecall << " MaxPrc=" << oRes.dMaxPrecision << " MaxFM=" << oRes.dMaxFMeasure << std::endl;
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << @@@@" " <<          " : BestRcl=" << std::fixed << std::setprecision(4) << oRes.oBestScore.dRecall << " BestPrc=" << oRes.oBestScore.dPrecision << " BestFM=" << oRes.oBestScore.dFMeasure << "  (@ T=" << std::fixed << std::setprecision(4) << oRes.oBestScore.dThreshold << ")" << std::endl;
}

litiv::Image::Segm::BSDS500BoundaryEvaluator::BSDS500Score litiv::Image::Segm::BSDS500BoundaryEvaluator::FindMaxFMeasure(const std::vector<uchar>& vnThresholds, const std::vector<double>& vdRecall, const std::vector<double>& vdPrecision) {
    CV_Assert(!vnThresholds.empty() && !vdRecall.empty() && !vdPrecision.empty());
    CV_Assert(vnThresholds.size()==vdRecall.size() && vdRecall.size()==vdPrecision.size());
    BSDS500Score oRes;
    oRes.dFMeasure = ClassifMetrics::CalcFMeasure(vdRecall[0],vdPrecision[0]);
    oRes.dPrecision = vdPrecision[0];
    oRes.dRecall = vdRecall[0];
    oRes.dThreshold = double(vnThresholds[0])/UCHAR_MAX;
    for(size_t nThresholdIdx=1; nThresholdIdx<vnThresholds.size(); ++nThresholdIdx) {
        const size_t nInterpCount = 100;
        for(size_t nInterpIdx=0; nInterpIdx<=nInterpCount; ++nInterpIdx) {
            const double dLastInterp = double(nInterpCount-nInterpIdx)/nInterpCount;
            const double dCurrInterp = double(nInterpIdx)/nInterpCount;
            const double dInterpRecall = dLastInterp*vdRecall[nThresholdIdx-1] + dCurrInterp*vdRecall[nThresholdIdx];
            const double dInterpPrecision = dLastInterp*vdPrecision[nThresholdIdx-1] + dCurrInterp*vdPrecision[nThresholdIdx];
            const double dInterpFMeasure = ClassifMetrics::CalcFMeasure(dInterpRecall,dInterpPrecision);
            if(dInterpFMeasure>oRes.dFMeasure) {
                oRes.dThreshold = (dLastInterp*vnThresholds[nThresholdIdx-1] + dCurrInterp*vnThresholds[nThresholdIdx])/UCHAR_MAX;
                oRes.dFMeasure = dInterpFMeasure;
                oRes.dPrecision = dInterpPrecision;
                oRes.dRecall = dInterpRecall;
            }
        }
    }
    return oRes;
}

litiv::Image::Segm::BSDS500BoundaryEvaluator::BSDS500Score litiv::Image::Segm::BSDS500BoundaryEvaluator::FindMaxFMeasure(const std::vector<BSDS500Score>& voScores) {
    CV_Assert(!voScores.empty());
    BSDS500Score oRes = voScores[0];
    for(size_t nScoreIdx=1; nScoreIdx<voScores.size(); ++nScoreIdx) {
        const size_t nInterpCount = 100;
        for(size_t nInterpIdx=0; nInterpIdx<=nInterpCount; ++nInterpIdx) {
            const double dLastInterp = double(nInterpCount-nInterpIdx)/nInterpCount;
            const double dCurrInterp = double(nInterpIdx)/nInterpCount;
            const double dInterpRecall = dLastInterp*voScores[nScoreIdx-1].dRecall + dCurrInterp*voScores[nScoreIdx].dRecall;
            const double dInterpPrecision = dLastInterp*voScores[nScoreIdx-1].dPrecision + dCurrInterp*voScores[nScoreIdx].dPrecision;
            const double dInterpFMeasure = ClassifMetrics::CalcFMeasure(dInterpRecall,dInterpPrecision);
            if(dInterpFMeasure>oRes.dFMeasure) {
                oRes.dThreshold = dLastInterp*voScores[nScoreIdx-1].dThreshold + dCurrInterp*voScores[nScoreIdx].dThreshold;
                oRes.dFMeasure = dInterpFMeasure;
                oRes.dPrecision = dInterpPrecision;
                oRes.dRecall = dInterpRecall;
            }
        }
    }
    return oRes;
}

#endif
