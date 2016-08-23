
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

void lv::IDatasetEvaluator_<lv::DatasetEval_None>::writeEvalReport() const {
    if(getBatches(false).empty()) {
        std::cout << "No report to write for dataset '" << getName() << "', skipping." << std::endl;
        return;
    }
    for(const auto& pGroupIter : getBatches(true))
        pGroupIter->shared_from_this_cast<const IDataReporter_<DatasetEval_None>>(true)->IDataReporter_<DatasetEval_None>::writeEvalReport();
    std::ofstream oMetricsOutput(getOutputPath()+"/overall.txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Default evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Packets  |   Seconds  |     Hz     \n";
        oMetricsOutput << "------------|------------|------------|------------\n";
        size_t nOverallPacketCount = 0;
        double dOverallTimeElapsed = 0.0;
        for(const auto& pGroupIter : getBatches(true)) {
            oMetricsOutput << pGroupIter->shared_from_this_cast<const IDataReporter_<DatasetEval_None>>(true)->IDataReporter_<DatasetEval_None>::writeInlineEvalReport(0);
            nOverallPacketCount += pGroupIter->getTotPackets();
            dOverallTimeElapsed += pGroupIter->getProcessTime();
        }
        oMetricsOutput << "------------|------------|------------|------------\n";
        oMetricsOutput << "     overall|" <<
                          std::setw(12) << nOverallPacketCount << "|" <<
                          std::setw(12) << dOverallTimeElapsed << "|" <<
                          std::setw(12) << nOverallPacketCount/dOverallTimeElapsed << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

void lv::IDatasetEvaluator_<lv::DatasetEval_BinaryClassifier>::writeEvalReport() const {
    if(getBatches(false).empty() || !isUsingEvaluator()) {
        IDatasetEvaluator_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    for(const auto& pGroupIter : getBatches(true))
        pGroupIter->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifier>>(true)->IDataReporter_<DatasetEval_BinaryClassifier>::writeEvalReport();
    IMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvAssert(pMetrics.get());
    const BinClassifMetricsCalculator& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get());
    std::cout << lv::clampString(getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+"/overall.txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Video segmentation evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        size_t nOverallPacketCount = 0;
        double dOverallTimeElapsed = 0.0;
        for(const auto& pGroupIter : getBatches(true)) {
            oMetricsOutput << pGroupIter->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifier>>(true)->IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineEvalReport(0);
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
        oMetricsOutput << lv::getLogStamp();
    }
}

lv::IMetricsAccumulatorConstPtr lv::IDatasetEvaluator_<lv::DatasetEval_BinaryClassifier>::getMetricsBase() const {
    BinClassifMetricsAccumulatorPtr pMetricsBase = BinClassifMetricsAccumulator::create();
    for(const auto& pBatch : getBatches(true))
        pMetricsBase->accumulate(dynamic_cast<const IDataReporter_<DatasetEval_BinaryClassifier>&>(*pBatch).getMetricsBase());
    return pMetricsBase;
}

lv::IMetricsCalculatorPtr lv::IDatasetEvaluator_<lv::DatasetEval_BinaryClassifier>::getMetrics(bool bAverage) const {
    if(bAverage) {
        IDataHandlerPtrArray vpBatches = getBatches(true);
        auto ppBatchIter = vpBatches.begin();
        for(; ppBatchIter!=vpBatches.end() && !(*ppBatchIter)->getTotPackets(); ++ppBatchIter);
        CV_Assert(ppBatchIter!=vpBatches.end());
        IMetricsCalculatorPtr pMetrics = dynamic_cast<const IDataReporter_<DatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage);
        for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
            if((*ppBatchIter)->getTotPackets())
                pMetrics->accumulate(dynamic_cast<const IDataReporter_<DatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage));
        return pMetrics;
    }
    return BinClassifMetricsCalculator::create(getMetricsBase());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void lv::IDataReporter_<lv::DatasetEval_None>::writeEvalReport() const {
    if(!getTotPackets()) {
        std::cout << "No report to write for '" << getName() << "', skipping..." << std::endl;
        return;
    }
    else if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            pBatch->writeEvalReport();
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
    if(!getTotPackets())
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<lv::DatasetEval_None>>(true)->IDataReporter_<DatasetEval_None>::writeInlineEvalReport(nIndentSize+1);
    ssStr << lv::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
             std::setw(nCellSize) << getTotPackets() << "|" <<
             std::setw(nCellSize) << getProcessTime() << "|" <<
             std::setw(nCellSize) << getTotPackets()/getProcessTime() << "\n";
    return ssStr.str();
}

lv::IMetricsAccumulatorConstPtr lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::getMetricsBase() const {
    lvAssert(isGroup()); // non-group specialization should override this method
    BinClassifMetricsAccumulatorPtr pMetricsBase = BinClassifMetricsAccumulator::create();
    for(const auto& pBatch : getBatches(true))
        pMetricsBase->accumulate(dynamic_cast<const IDataReporter_<DatasetEval_BinaryClassifier>&>(*pBatch).getMetricsBase());
    return pMetricsBase;
}

lv::IMetricsCalculatorPtr lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::getMetrics(bool bAverage) const {
    if(bAverage && isGroup() && !isBare()) {
        const IDataHandlerPtrArray& vpBatches = getBatches(true);
        auto ppBatchIter = vpBatches.begin();
        for(; ppBatchIter!=vpBatches.end() && !(*ppBatchIter)->getTotPackets(); ++ppBatchIter);
        CV_Assert(ppBatchIter!=vpBatches.end());
        IMetricsCalculatorPtr pMetrics = dynamic_cast<const IDataReporter_<DatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage);
        for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
            if((*ppBatchIter)->getTotPackets())
                pMetrics->accumulate(dynamic_cast<const IDataReporter_<DatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage));
        return pMetrics; // @@@ check returning metrics weight?
    }
    return BinClassifMetricsCalculator::create(getMetricsBase());
}

void lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::writeEvalReport() const {
    if(!getTotPackets() || !getDatasetInfo()->isUsingEvaluator()) {
        IDataReporter_<lv::DatasetEval_None>::writeEvalReport();
        return;
    }
    else if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            pBatch->writeEvalReport();
    IMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvAssert(pMetrics.get());
    const BinClassifMetricsCalculator& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get());;
    std::cout << "\t" << lv::clampString(std::string(size_t(!isGroup()),'>')+getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(lv::AddDirSlashIfMissing(getOutputPath())+"../"+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Video segmentation evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineEvalReport(0);
        oMetricsOutput << "\nHz: " << getTotPackets()/getProcessTime() << "\n";
        oMetricsOutput << lv::getLogStamp();
    }
}

std::string lv::IDataReporter_<lv::DatasetEval_BinaryClassifier>::writeInlineEvalReport(size_t nIndentSize) const {
    if(!getTotPackets())
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches(true))
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<DatasetEval_BinaryClassifier>>(true)->IDataReporter_<DatasetEval_BinaryClassifier>::writeInlineEvalReport(nIndentSize+1);
    IMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvAssert(pMetrics.get());
    const BinClassifMetricsCalculator& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get());
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
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

#if HAVE_GLSL

lv::GLBinaryClassifierEvaluator::GLBinaryClassifierEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent,size_t nTotFrameCount) :
        GLImageProcEvaluatorAlgo(pParent,nTotFrameCount,(size_t)BinClassifMetricsAccumulator::nCountersCount,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

std::string lv::GLBinaryClassifierEvaluator::getComputeShaderSource(size_t nStage) const {
    lvAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"#version 430\n"
            "#define VAL_POSITIVE     " << (uint)DATASETUTILS_POSITIVE_VAL << "\n"
            "#define VAL_NEGATIVE     " << (uint)dATASETUTILS_NEGATIVE_VAL << "\n"
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
    ssSrc <<"layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::Counter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::Counter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::Counter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::Counter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::Counter_SE*4 << ") uniform atomic_uint nSE;\n"
            "layout(binding=" << GLImageProcAlgo::AtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::Counter_DC*4 << ") uniform atomic_uint nDC;\n";
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
    BinClassifMetricsAccumulatorPtr pMetricsBase = BinClassifMetricsAccumulator::create();
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        pMetricsBase->nTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::Counter_TP);
        pMetricsBase->nTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::Counter_TN);
        pMetricsBase->nFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::Counter_FP);
        pMetricsBase->nFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::Counter_FN);
        pMetricsBase->nSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::Counter_SE);
        pMetricsBase->nDC += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::Counter_DC);
    }
    return pMetricsBase;
}

#endif //HAVE_GLSL
