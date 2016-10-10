
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
        pBatch->writeEvalReport();
    if(isBare())
        return;
    std::ofstream oMetricsOutput(getOutputPath()+(isRoot()?"":"../")+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Default evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Packets  |   Seconds  |     Hz     \n";
        oMetricsOutput << "------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_None>::writeInlineBasicReport(0);
        oMetricsOutput << lv::getLogStamp();
    }
}

std::string lv::IDataReporter_<lv::DatasetEval_None>::writeInlineBasicReport(size_t nIndentSize) const {
    if(getCurrentOutputCount()==0)
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_None>>(true)->IDataReporter_<DatasetEval_None>::writeInlineBasicReport(nIndentSize+1);
    ssStr << lv::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
             std::setw(nCellSize) << getCurrentOutputCount() << "|" <<
             std::setw(nCellSize) << getCurrentProcessTime() << "|" <<
             std::setw(nCellSize) << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
    return ssStr.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::writeEvalReport() const {
    if(getCurrentOutputCount()==0 || !isEvaluating()) {
        IDataReporter_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->writeEvalReport();
    if(isBare())
        return;
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetrics& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get()).m_oMetrics;
    std::cout << "\t" << lv::clampString(std::string(size_t(!isGroup()),'>')+getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+(isRoot()?"":"../")+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Binary classification evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineBinClassifEvalReport(0);
        oMetricsOutput << "\nHz: " << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

std::string lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::writeInlineBinClassifEvalReport(size_t nIndentSize) const {
    if(getCurrentOutputCount()==0)
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifier>>(true)->IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineBinClassifEvalReport(nIndentSize+1);
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
    if(getCurrentOutputCount()==0 || !isEvaluating()) {
        IDataReporter_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pBatch : getBatches(true))
        pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifierArray>>(true)->IDataReporter_<DatasetEval_BinaryClassifierArray>::writeEvalReport();
    if(isBare())
        return;
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetricsArrayCalculator& oMetrics = dynamic_cast<const BinClassifMetricsArrayCalculator&>(*pMetrics.get());
    const BinClassifMetricsCalculatorPtr pReducedMetrics = oMetrics.reduce();
    lvDbgAssert(pReducedMetrics);
    const BinClassifMetrics& oReducedMetrics = pReducedMetrics->m_oMetrics;
    std::cout << "\t" << lv::clampString(std::string(size_t(!isGroup()),'>')+getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oReducedMetrics.dRecall << " Prc=" << oReducedMetrics.dPrecision << " FM=" << oReducedMetrics.dFMeasure << " MCC=" << oReducedMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+(isRoot()?"":"../")+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Binary classification evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Stream   ||     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------||------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_BinaryClassifierArray>::writeInlineBinClassifArrayEvalReport(0);
        oMetricsOutput << "------------|------------||------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_BinaryClassifierArray>::writeInlineBinClassifArrayReducedEvalReport(0);
        oMetricsOutput << "\nHz: " << getCurrentOutputCount()/getCurrentProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

std::string lv::IDataReporter_<lv::DatasetEval_BinaryClassifierArray>::writeInlineBinClassifArrayEvalReport(size_t nIndentSize) const {
    if(getCurrentOutputCount()==0)
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifierArray>>(true)->IDataReporter_<DatasetEval_BinaryClassifierArray>::writeInlineBinClassifArrayEvalReport(nIndentSize+1);
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetricsArrayCalculator& oMetrics = dynamic_cast<const BinClassifMetricsArrayCalculator&>(*pMetrics.get());
    lvDbgAssert(oMetrics.m_vsStreamNames.size()==oMetrics.m_vMetrics.size());
    for(size_t s=0; s<oMetrics.m_vMetrics.size(); ++s)
        ssStr << lv::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
                 lv::clampString((oMetrics.m_vsStreamNames[s].empty())?std::string("s")+std::to_string(s):(oMetrics.m_vsStreamNames[s]),nCellSize) << "||" <<
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

std::string lv::IDataReporter_<lv::DatasetEval_BinaryClassifierArray>::writeInlineBinClassifArrayReducedEvalReport(size_t nIndentSize) const {
    if(getCurrentOutputCount()==0)
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifierArray>>(true)->IDataReporter_<DatasetEval_BinaryClassifierArray>::writeInlineBinClassifArrayReducedEvalReport(nIndentSize+1);
    IIMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvDbgAssert(pMetrics.get());
    const BinClassifMetricsArrayCalculator& oMetrics = dynamic_cast<const BinClassifMetricsArrayCalculator&>(*pMetrics.get());
    const BinClassifMetricsCalculatorPtr pReducedMetrics = oMetrics.reduce();
    lvDbgAssert(pReducedMetrics);
    const BinClassifMetrics& oReducedMetrics = pReducedMetrics->m_oMetrics;
    ssStr << lv::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
             lv::clampString("all/reduced",nCellSize) << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dRecall << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dSpecificity << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dFPR << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dFNR << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dPBC << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dPrecision << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dFMeasure << "|" <<
             std::setw(nCellSize) << oReducedMetrics.dMCC << "\n";
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
