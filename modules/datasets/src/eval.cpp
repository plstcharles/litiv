
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

#include "litiv/datasets/eval.hpp"
//#include "litiv/utils/console.hpp" @@@@@ reuse later?

void lv::IDataReporter_<lv::DatasetEval_None>::writeEvalReport() const {
    if(getCurrentOutputCount()==0) {
        std::cout << "No report to write for '" << getName() << "', skipping..." << std::endl;
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_None>>(true)->IDataReporter_<DatasetEval_None>::writeEvalReport();
    if(isBare())
        return;
    std::ofstream oMetricsOutput(lv::AddDirSlashIfMissing(getOutputPath())+"../"+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Default evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Packets  |   Seconds  |     Hz     \n";
        oMetricsOutput << "------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_None>::writeInlineEvalReport(0);
        oMetricsOutput << lv::getLogStamp();
    }
}

std::string lv::IDataReporter_<lv::DatasetEval_None>::writeInlineEvalReport(size_t nIndentSize) const {
    if(getCurrentOutputCount()==0)
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_None>>(true)->IDataReporter_<DatasetEval_None>::writeInlineEvalReport(nIndentSize+1);
    ssStr << lv::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
             std::setw(nCellSize) << getCurrentOutputCount() << "|" <<
             std::setw(nCellSize) << getCurrentProcessTime() << "|" <<
             std::setw(nCellSize) << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
    return ssStr.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::writeEvalReport() const {
    if(getCurrentOutputCount()==0 || !getDatasetInfo()->isUsingEvaluator()) {
        IDataReporter_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifier>>(true)->IDataReporter_<DatasetEval_BinaryClassifier>::writeEvalReport();
    if(isBare())
        return;
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetrics& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get()).m_oMetrics;
    std::cout << "\t" << lv::clampString(std::string(size_t(!isGroup()),'>')+getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(lv::AddDirSlashIfMissing(getOutputPath())+"../"+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Binary classification evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineEvalReport(0);
        oMetricsOutput << "\nHz: " << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

std::string lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::writeInlineEvalReport(size_t nIndentSize) const {
    if(getCurrentOutputCount()==0)
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifier>>(true)->IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineEvalReport(nIndentSize+1);
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetrics& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get()).m_oMetrics;
    ssStr << lv::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
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

////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IDataReporter_<lv::DatasetEval_BinaryClassifierArray>::writeEvalReport() const {
    if(getCurrentOutputCount()==0 || !getDatasetInfo()->isUsingEvaluator()) {
        IDataReporter_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifierArray>>(true)->IDataReporter_<DatasetEval_BinaryClassifierArray>::writeEvalReport();
    if(isBare())
        return;
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetricsCalculatorPtr pOverallMetrics = dynamic_cast<const BinClassifMetricsArrayCalculator&>(*pMetrics.get()).reduce();
    lvAssert(pOverallMetrics);
    const BinClassifMetrics& oOverallMetrics = pOverallMetrics->m_oMetrics;
    std::cout << "\t" << lv::clampString(std::string(size_t(!isGroup()),'>')+getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oOverallMetrics.dRecall << " Prc=" << oOverallMetrics.dPrecision << " FM=" << oOverallMetrics.dFMeasure << " MCC=" << oOverallMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(lv::AddDirSlashIfMissing(getOutputPath())+"../"+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Binary classification evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Stream   ||     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------||------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_BinaryClassifierArray>::writeInlineEvalReport(0);
        oMetricsOutput << "\nHz: " << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

std::string lv::IDataReporter_<lv::DatasetEval_BinaryClassifierArray>::writeInlineEvalReport(size_t nIndentSize) const {
    if(getCurrentOutputCount()==0)
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifierArray>>(true)->IDataReporter_<DatasetEval_BinaryClassifierArray>::writeInlineEvalReport(nIndentSize+1);
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetricsArrayCalculator& oMetrics = dynamic_cast<const BinClassifMetricsArrayCalculator&>(*pMetrics.get());
    const auto pConsumer = shared_from_this_cast<const IDataConsumer_<DatasetEval_BinaryClassifierArray>>(true);
    lvAssert_(oMetrics.m_vMetrics.size()==pConsumer->getOutputStreamCount(),"output array size mismatch");
    for(size_t s=0; s<oMetrics.m_vMetrics.size(); ++s)
        ssStr << lv::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
                 lv::clampString(pConsumer->getOutputStreamName(s),nCellSize) << "||" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dRecall << "|" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dSpecificity << "|" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dFPR << "|" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dFNR << "|" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dPBC << "|" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dPrecision << "|" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dFMeasure << "|" <<
                 std::setw(nCellSize) << oMetrics.m_vMetrics[s].dMCC << "\n";
    return ssStr.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#if HAVE_GLSL

lv::GLBinaryClassifierEvaluator::GLBinaryClassifierEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent,size_t nTotFrameCount) :
        GLImageProcEvaluatorAlgo(pParent,nTotFrameCount,(size_t)BinClassif::nCountersCount,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

std::string lv::GLBinaryClassifierEvaluator::getComputeShaderSource(size_t nStage) const {
    lvAssert_(nStage<m_nComputeStages,"requested compute shader stage does not exist");
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"#version 430\n"
            "#define VAL_POSITIVE     " << (uint)DATASETUTILS_POSITIVE_VAL << "\n"
            "#define VAL_NEGATIVE     " << (uint)DATASETUTILS_NEGATIVE_VAL << "\n"
            "#define VAL_OUTOFSCOPE   " << (uint)DATASETUTILS_OUTOFSCOPE_VAL << "\n"
            "#define VAL_UNKNOWN      " << (uint)DATASETUTILS_UNKNOWN_VAL << "\n"
            "#define VAL_SHADOW       " << (uint)DATASETUTILS_SHADOW_VAL << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
            "layout(binding=" << GLImageProcAlgo::Image_ROIBinding << ", r8ui) readonly uniform uimage2D imgROI;\n"
            "layout(binding=" << GLImageProcAlgo::Image_OutputBinding << ", r8ui) readonly uniform uimage2D imgInput;\n"
            "layout(binding=" << GLImageProcAlgo::Image_GTBinding << ", r8ui) readonly uniform uimage2D imgGT;\n";
    if(m_bUsingDebug) ssSrc <<
            "layout(binding=" << GLImageProcAlgo::Image_DebugBinding << ") writeonly uniform uimage2D imgDebug;\n";
    ssSrc <<"layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassif::Counter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassif::Counter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassif::Counter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassif::Counter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassif::Counter_SE*4 << ") uniform atomic_uint nSE;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassif::Counter_DC*4 << ") uniform atomic_uint nDC;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"void main() {\n"
            "    ivec2 imgCoord = ivec2(gl_GlobalInvocationID.xy);\n"
            "    uint nInputSegmVal = imageLoad(imgInput,imgCoord).r;\n"
            "    uint nGTSegmVal = imageLoad(imgGT,imgCoord).r;\n"
            "    uint nROIVal = imageLoad(imgROI,imgCoord).r;\n"
            "    if(nROIVal!=VAL_NEGATIVE && nGTSegmVal!=VAL_OUTOFSCOPE && nGTSegmVal!=VAL_UNKNOWN) {\n"
            "        if(nInputSegmVal==VAL_POSITIVE) {\n"
            "            if(nGTSegmVal==VAL_POSITIVE) {\n"
            "                atomicCounterIncrement(nTP);\n"
            "            }\n"
            "            else { // nGTSegmVal==VAL_NEGATIVE\n"
            "                atomicCounterIncrement(nFP);\n"
            "            }\n"
            "        }\n"
            "        else { // nInputSegmVal==VAL_NEGATIVE\n"
            "            if(nGTSegmVal==VAL_POSITIVE) {\n"
            "                atomicCounterIncrement(nFN);\n"
            "            }\n"
            "            else { // nGTSegmVal==VAL_NEGATIVE\n"
            "                atomicCounterIncrement(nTN);\n"
            "            }\n"
            "        }\n"
            "        if(nGTSegmVal==VAL_SHADOW) {\n"
            "            if(nInputSegmVal==VAL_POSITIVE) {\n"
            "               atomicCounterIncrement(nSE);\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    else {\n"
            "        atomicCounterIncrement(nDC);\n"
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

lv::BinClassifMetricsAccumulatorPtr lv::GLBinaryClassifierEvaluator::getMetricsBase() {
    const cv::Mat& oAtomicCountersQueryBuffer = this->getEvaluationAtomicCounterBuffer();
    BinClassifMetricsAccumulatorPtr pMetricsBase = IIMetricsAccumulator::create<IMetricsAccumulator_<DatasetEval_BinaryClassifier>>();
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        pMetricsBase->m_oCounters.nTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassif::Counter_TP);
        pMetricsBase->m_oCounters.nTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassif::Counter_TN);
        pMetricsBase->m_oCounters.nFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassif::Counter_FP);
        pMetricsBase->m_oCounters.nFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassif::Counter_FN);
        pMetricsBase->m_oCounters.nSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassif::Counter_SE);
        pMetricsBase->m_oCounters.nDC += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassif::Counter_DC);
    }
    return pMetricsBase;
}

#endif //HAVE_GLSL

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IDatasetReporter_<lv::DatasetEval_None>::writeEvalReport() const {
    if(getCurrentOutputCount()==0 || getBatches(false).empty()) {
        std::cout << "No report to write for dataset '" << getName() << "', skipping." << std::endl;
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->writeEvalReport();
    std::ofstream oMetricsOutput(getOutputPath()+"/overall.txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Default evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Packets  |   Seconds  |     Hz     \n";
        oMetricsOutput << "------------|------------|------------|------------\n";
        for(const auto& pGroupIter : getBatches(true))
            oMetricsOutput << pGroupIter->shared_from_this_cast<const IDataReporter_<DatasetEval_None>>(true)->IDataReporter_<DatasetEval_None>::writeInlineEvalReport(0);
        oMetricsOutput << "------------|------------|------------|------------\n";
        oMetricsOutput << "     overall|" <<
                       std::setw(12) << getCurrentOutputCount() << "|" <<
                       std::setw(12) << getCurrentProcessTime() << "|" <<
                       std::setw(12) << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IDatasetReporter_<lv::DatasetEval_BinaryClassifier>::writeEvalReport() const {
    if(getCurrentOutputCount()==0 || getBatches(false).empty() || !isUsingEvaluator()) {
        IDatasetReporter_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->writeEvalReport();
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetrics& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get()).m_oMetrics;
    std::cout << lv::clampString(getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+"/overall.txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Binary classification evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        for(const auto& pGroupIter : getBatches(true))
            oMetricsOutput << pGroupIter->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifier>>(true)->IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineEvalReport(0);
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
        oMetricsOutput << "\nHz: " << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IDatasetReporter_<lv::DatasetEval_BinaryClassifierArray>::writeEvalReport() const {
    if(getCurrentOutputCount()==0 || getBatches(false).empty() || !isUsingEvaluator()) {
        IDatasetReporter_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->writeEvalReport();
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetricsArrayCalculator& oMetrics = dynamic_cast<const BinClassifMetricsArrayCalculator&>(*pMetrics.get());
    const BinClassifMetricsCalculatorPtr pOverallMetrics = oMetrics.reduce();
    lvAssert(pOverallMetrics);
    const BinClassifMetrics& oOverallMetrics = pOverallMetrics->m_oMetrics;
    std::cout << lv::clampString(getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oOverallMetrics.dRecall << " Prc=" << oOverallMetrics.dPrecision << " FM=" << oOverallMetrics.dFMeasure << " MCC=" << oOverallMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+"/overall.txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Binary classification evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Stream   ||     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------||------------|------------|------------|------------|------------|------------|------------|------------\n";
        std::vector<std::string> vsStreamNames;
        for(const auto& pGroupIter : getBatches(true)) {
            const auto pReporter = pGroupIter->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifierArray>>(true);
            oMetricsOutput << pReporter->IDataReporter_<DatasetEval_BinaryClassifierArray>::writeInlineEvalReport(0);
            const auto pConsumer = pGroupIter->shared_from_this_cast<const IDataConsumer_<DatasetEval_BinaryClassifierArray>>(true);
            if(vsStreamNames.empty()) {
                vsStreamNames.resize(pConsumer->getOutputStreamCount());
                for(size_t s=0; s<vsStreamNames.size(); ++s)
                    vsStreamNames[s] = pConsumer->getOutputStreamName(s);
            }
            else {
                lvAssert_(vsStreamNames.size()==pConsumer->getOutputStreamCount(),"output stream count mismatch");
                for(size_t s=0; s<vsStreamNames.size(); ++s)
                lvAssert_(vsStreamNames[s]==pConsumer->getOutputStreamName(s),"output stream names mismatch");
            }
        }
        oMetricsOutput << "------------|------------||------------|------------|------------|------------|------------|------------|------------|------------\n";
        lvAssert_(oMetrics.m_vMetrics.size()==vsStreamNames.size(),"output stream count mismatch");
        for(size_t s=0; s<oMetrics.m_vMetrics.size(); ++s)
            oMetricsOutput << "     overall|" << lv::clampString(vsStreamNames[s],12) << "||" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dRecall << "|" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dSpecificity << "|" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dFPR << "|" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dFNR << "|" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dPBC << "|" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dPrecision << "|" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dFMeasure << "|" <<
                           std::setw(12) << oMetrics.m_vMetrics[s].dMCC << "\n";

        oMetricsOutput << "------------|------------||------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << "         overall         ||" <<
                       std::setw(12) << oOverallMetrics.dRecall << "|" <<
                       std::setw(12) << oOverallMetrics.dSpecificity << "|" <<
                       std::setw(12) << oOverallMetrics.dFPR << "|" <<
                       std::setw(12) << oOverallMetrics.dFNR << "|" <<
                       std::setw(12) << oOverallMetrics.dPBC << "|" <<
                       std::setw(12) << oOverallMetrics.dPrecision << "|" <<
                       std::setw(12) << oOverallMetrics.dFMeasure << "|" <<
                       std::setw(12) << oOverallMetrics.dMCC << "\n";
        oMetricsOutput << "\nHz: " << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}
