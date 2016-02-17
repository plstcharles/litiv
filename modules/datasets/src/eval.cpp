
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
//#include "litiv/utils/ConsoleUtils.hpp" @@@@@ reuse later?

void litiv::IDatasetEvaluator_<litiv::eDatasetEval_None>::writeEvalReport() const {
    std::cout << "Writing evaluation report for dataset '" << getName() << "'..." << std::endl;
    if(getBatches().empty()) {
        std::cout << "No report to write for dataset '" << getName() << "', skipping." << std::endl;
        return;
    }
    for(const auto& pGroupIter : getBatches())
        pGroupIter->writeEvalReport();
    std::ofstream oMetricsOutput(getOutputPath()+"/overall.txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Default evaluation report for dataset '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Packets  |   Seconds  |     Hz     \n";
        oMetricsOutput << "------------|------------|------------|------------\n";
        size_t nOverallPacketCount = 0;
        double dOverallTimeElapsed = 0.0;
        for(const auto& pGroupIter : getBatches()) {
            oMetricsOutput << pGroupIter->shared_from_this_cast<const IDataReporter_<eDatasetEval_None>>(true)->IDataReporter_<eDatasetEval_None>::writeInlineEvalReport(0);
            nOverallPacketCount += pGroupIter->getTotPackets();
            dOverallTimeElapsed += pGroupIter->getProcessTime();
        }
        oMetricsOutput << "------------|------------|------------|------------\n";
        oMetricsOutput << "     overall|" <<
                          std::setw(12) << nOverallPacketCount << "|" <<
                          std::setw(12) << dOverallTimeElapsed << "|" <<
                          std::setw(12) << nOverallPacketCount/dOverallTimeElapsed << "\n";
        oMetricsOutput << CxxUtils::getLogStamp();
    }
}

void litiv::IDatasetEvaluator_<litiv::eDatasetEval_BinaryClassifier>::writeEvalReport() const {
    if(getBatches().empty() || !isUsingEvaluator()) {
        std::cout << "No report to write for dataset '" << getName() << "', skipping." << std::endl;
        return;
    }
    for(const auto& pGroupIter : getBatches())
        pGroupIter->writeEvalReport();
    IMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvAssert(pMetrics.get());
    const BinClassifMetricsCalculator& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get());
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
            oMetricsOutput << pGroupIter->shared_from_this_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>>(true)->IDataReporter_<eDatasetEval_BinaryClassifier>::writeInlineEvalReport(0);
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
        oMetricsOutput << CxxUtils::getLogStamp();
    }
}

litiv::IMetricsAccumulatorConstPtr litiv::IDatasetEvaluator_<litiv::eDatasetEval_BinaryClassifier>::getMetricsBase() const {
    BinClassifMetricsAccumulatorPtr pMetricsBase = BinClassifMetricsAccumulator::create();
    for(const auto& pBatch : getBatches())
        pMetricsBase->accumulate(dynamic_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>&>(*pBatch).getMetricsBase());
    return pMetricsBase;
}

litiv::IMetricsCalculatorPtr litiv::IDatasetEvaluator_<litiv::eDatasetEval_BinaryClassifier>::getMetrics(bool bAverage) const {
    if(bAverage) {
        IDataHandlerPtrArray vpBatches = getBatches();
        auto ppBatchIter = vpBatches.begin();
        for(; ppBatchIter!=vpBatches.end() && !(*ppBatchIter)->getTotPackets(); ++ppBatchIter);
        CV_Assert(ppBatchIter!=vpBatches.end());
        IMetricsCalculatorPtr pMetrics = dynamic_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage);
        for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
            if((*ppBatchIter)->getTotPackets())
                pMetrics->accumulate(dynamic_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage));
        return pMetrics;
    }
    return BinClassifMetricsCalculator::create(getMetricsBase());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

void litiv::IDataReporter_<litiv::eDatasetEval_None>::writeEvalReport() const {
    if(!getTotPackets()) {
        std::cout << "No report to write for '" << getName() << "', skipping..." << std::endl;
        return;
    }
    else if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches())
            pBatch->writeEvalReport();
    std::ofstream oMetricsOutput(getOutputPath()+"/../"+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Default evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |   Packets  |   Seconds  |     Hz     \n";
        oMetricsOutput << "------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<eDatasetEval_None>::writeInlineEvalReport(0);
        oMetricsOutput << CxxUtils::getLogStamp();
    }
}

std::string litiv::IDataReporter_<litiv::eDatasetEval_None>::writeInlineEvalReport(size_t nIndentSize) const {
    if(!getTotPackets())
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches())
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<litiv::eDatasetEval_None>>(true)->IDataReporter_<eDatasetEval_None>::writeInlineEvalReport(nIndentSize+1);
    ssStr << CxxUtils::clampString((std::string(nIndentSize,'>')+' '+getName()),nCellSize) << "|" <<
             std::setw(nCellSize) << getTotPackets() << "|" <<
             std::setw(nCellSize) << getProcessTime() << "|" <<
             std::setw(nCellSize) << getTotPackets()/getProcessTime() << "\n";
    return ssStr.str();
}

litiv::IMetricsAccumulatorConstPtr litiv::IDataReporter_<litiv::eDatasetEval_BinaryClassifier>::getMetricsBase() const {
    lvAssert(isGroup()); // non-group specialization should override this method
    BinClassifMetricsAccumulatorPtr pMetricsBase = BinClassifMetricsAccumulator::create();
    for(const auto& pBatch : getBatches())
        pMetricsBase->accumulate(dynamic_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>&>(*pBatch).getMetricsBase());
    return pMetricsBase;
}

litiv::IMetricsCalculatorPtr litiv::IDataReporter_<litiv::eDatasetEval_BinaryClassifier>::getMetrics(bool bAverage) const {
    if(bAverage && isGroup() && !isBare()) {
        const IDataHandlerPtrArray& vpBatches = getBatches();
        auto ppBatchIter = vpBatches.begin();
        for(; ppBatchIter!=vpBatches.end() && !(*ppBatchIter)->getTotPackets(); ++ppBatchIter);
        CV_Assert(ppBatchIter!=vpBatches.end());
        IMetricsCalculatorPtr pMetrics = dynamic_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage);
        for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
            if((*ppBatchIter)->getTotPackets())
                pMetrics->accumulate(dynamic_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>&>(**ppBatchIter).getMetrics(bAverage));
        return pMetrics; // @@@ check returning metrics weight?
    }
    return BinClassifMetricsCalculator::create(getMetricsBase());
}

void litiv::IDataReporter_<litiv::eDatasetEval_BinaryClassifier>::writeEvalReport() const {
    if(!getTotPackets()) {
        std::cout << "No report to write for '" << getName() << "', skipping..." << std::endl;
        return;
    }
    else if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches())
            pBatch->writeEvalReport();
    IMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvAssert(pMetrics.get());
    const BinClassifMetricsCalculator& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get());;
    std::cout << "\t" << CxxUtils::clampString(std::string(size_t(!isGroup()),'>')+getName(),12) << " => Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
    std::ofstream oMetricsOutput(getOutputPath()+"/../"+getName()+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << std::fixed;
        oMetricsOutput << "Video segmentation evaluation report for '" << getName() << "' :\n\n";
        oMetricsOutput << "            |     Rcl    |     Spc    |     FPR    |     FNR    |     PBC    |     Prc    |     FM     |     MCC    \n";
        oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
        oMetricsOutput << IDataReporter_<eDatasetEval_BinaryClassifier>::writeInlineEvalReport(0);
        oMetricsOutput << "\nHz: " << getTotPackets()/getProcessTime() << "\n";
        oMetricsOutput << CxxUtils::getLogStamp();
    }
}

std::string litiv::IDataReporter_<litiv::eDatasetEval_BinaryClassifier>::writeInlineEvalReport(size_t nIndentSize) const {
    if(!getTotPackets())
        return std::string();
    const size_t nCellSize = 12;
    std::stringstream ssStr;
    ssStr << std::fixed;
    if(isGroup() && !isBare())
        for(const auto& pBatch : getBatches())
            ssStr << pBatch->shared_from_this_cast<const IDataReporter_<eDatasetEval_BinaryClassifier>>(true)->IDataReporter_<eDatasetEval_BinaryClassifier>::writeInlineEvalReport(nIndentSize+1);
    IMetricsCalculatorConstPtr pMetrics = getMetrics(true);
    lvAssert(pMetrics.get());
    const BinClassifMetricsCalculator& oMetrics = dynamic_cast<const BinClassifMetricsCalculator&>(*pMetrics.get());
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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

litiv::IMetricsAccumulatorConstPtr litiv::IDataEvaluator_<litiv::eDatasetEval_BinaryClassifier>::getMetricsBase() const {
    if(!m_pMetricsBase)
        return BinClassifMetricsAccumulator::create();
    return m_pMetricsBase;
}

void litiv::IDataEvaluator_<litiv::eDatasetEval_BinaryClassifier>::push(const cv::Mat& oClassif, size_t nIdx) {
    IDataConsumer_<eDatasetEval_BinaryClassifier>::push(oClassif,nIdx);
    if(getDatasetInfo()->isUsingEvaluator()) {
        auto pLoader = shared_from_this_cast<IDataLoader_<eImagePacket>>(true);
        if(!m_pMetricsBase)
            m_pMetricsBase = BinClassifMetricsAccumulator::create();
        m_pMetricsBase->accumulate(oClassif,pLoader->getGT(nIdx),pLoader->getPacketROI(nIdx));
    }
}

cv::Mat litiv::IDataEvaluator_<litiv::eDatasetEval_BinaryClassifier>::getColoredMask(const cv::Mat& oClassif, size_t nIdx) {
    auto pLoader = shared_from_this_cast<IDataLoader_<eImagePacket>>(true);
    return BinClassifMetricsAccumulator::getColoredMask(oClassif,pLoader->getGT(nIdx),pLoader->getPacketROI(nIdx));
}

void litiv::IDataEvaluator_<litiv::eDatasetEval_BinaryClassifier>::resetMetrics() {
    m_pMetricsBase = BinClassifMetricsAccumulator::create();
}

#if HAVE_GLSL

cv::Size litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::getIdealGLWindowSize() const {
    glAssert(m_pEvalAlgo && m_pEvalAlgo->getIsGLInitialized());
    glAssert(getTotPackets()>1);
    cv::Size oWindowSize = shared_from_this_cast<const IDataLoader_<eImagePacket>>(true)->getPacketMaxSize();
    oWindowSize.width *= int(m_pEvalAlgo->m_nSxSDisplayCount);
    return oWindowSize;
}

litiv::IMetricsAccumulatorConstPtr litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::getMetricsBase() const {
    if(isProcessing())
        lvError("Must stop processing batch before querying metrics under async data evaluator interface");
    else if(!m_pMetricsBase)
        return BinClassifMetricsAccumulator::create();
    return m_pMetricsBase;
}

void litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::_stopProcessing() {
    if(m_pEvalAlgo && m_pEvalAlgo->getIsGLInitialized()) {
        BinClassifMetricsAccumulatorPtr pMetricsBase = m_pEvalAlgo->getMetricsBase();
#if DATASETUTILS_VALIDATE_ASYNC_EVALUATORS
        glAssert(!m_pMetricsBase || m_pMetricsBase->isEqual(pMetricsBase));
#endif //DATASETUTILS_VALIDATE_ASYNC_EVALUATORS
        m_pMetricsBase = pMetricsBase;
    }
}

void litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::pre_initialize_gl() {
    IAsyncDataConsumer_<eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::pre_initialize_gl();
    m_oNextGT = m_pLoader->getGT(m_nNextIdx).clone();
    m_oCurrGT = m_pLoader->getGT(m_nCurrIdx).clone();
    m_oLastGT = m_oCurrGT.clone();
    CV_Assert(!m_oCurrGT.empty());
    CV_Assert(m_oCurrGT.isContinuous());
    glAssert(m_oCurrGT.channels()==1 || m_oCurrGT.channels()==4);
}

void litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::post_initialize_gl() {
    IAsyncDataConsumer_<eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::post_initialize_gl();
    m_pEvalAlgo = std::make_unique<GLVideoSegmDataEvaluator>(m_pAlgo,getTotPackets());
    m_pEvalAlgo->initialize_gl(m_oCurrGT,m_pLoader->getPacketROI(m_nCurrIdx));
    m_pMetricsBase = BinClassifMetricsAccumulator::create();
    if(m_pAlgo->m_pDisplayHelper)
        m_pEvalAlgo->setOutputFetching(true);
    if(m_pAlgo->m_pDisplayHelper && m_pEvalAlgo->m_bUsingDebug)
        m_pEvalAlgo->setDebugFetching(true);
    if(DATASETUTILS_VALIDATE_ASYNC_EVALUATORS)
        m_pAlgo->setOutputFetching(true);
}

void litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::pre_apply_gl(size_t nNextIdx, bool bRebindAll) {
    IAsyncDataConsumer_<eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::pre_apply_gl(nNextIdx,bRebindAll);
    if(nNextIdx!=m_nNextIdx)
        m_oNextGT = m_pLoader->getGT(nNextIdx);
}

void litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::post_apply_gl(size_t nNextIdx, bool bRebindAll) {
    glDbgAssert(m_pLoader);
    glDbgAssert(m_pEvalAlgo);
    glDbgAssert(m_pAlgo);
    m_pEvalAlgo->apply_gl(m_oNextGT,bRebindAll);
    m_nLastIdx = m_nCurrIdx;
    m_nCurrIdx = nNextIdx;
    m_nNextIdx = nNextIdx+1;
    if(m_pAlgo->m_pDisplayHelper) {
        m_oCurrInput.copyTo(m_oLastInput);
        m_oNextInput.copyTo(m_oCurrInput);
        m_oCurrGT.copyTo(m_oLastGT);
        m_oNextGT.copyTo(m_oCurrGT);
    }
    if(m_nNextIdx<getTotPackets()) {
        m_oNextInput = m_pLoader->getInput(m_nNextIdx);
        m_oNextGT = m_pLoader->getGT(m_nNextIdx);
    }
    processPacket();
    if(getDatasetInfo()->isSavingOutput() || m_pAlgo->m_pDisplayHelper || DATASETUTILS_VALIDATE_ASYNC_EVALUATORS) {
        cv::Mat oLastOutput,oLastDebug;
        m_pAlgo->fetchLastOutput(oLastOutput);
        if(m_pAlgo->m_pDisplayHelper && m_pEvalAlgo->m_bUsingDebug)
            m_pEvalAlgo->fetchLastDebug(oLastDebug);
        else
            oLastDebug = oLastOutput;
        if(getDatasetInfo()->isSavingOutput())
            save(oLastOutput,m_nLastIdx);
        if(m_pAlgo->m_pDisplayHelper)
            m_pAlgo->m_pDisplayHelper->display(m_oLastInput,oLastDebug,BinClassifMetricsAccumulator::getColoredMask(oLastOutput,m_oLastGT,m_pLoader->getPacketROI(m_nLastIdx)),m_nLastIdx);
        if(DATASETUTILS_VALIDATE_ASYNC_EVALUATORS)
            m_pMetricsBase->accumulate(oLastOutput,m_pLoader->getGT(m_nLastIdx),m_pLoader->getPacketROI(m_nLastIdx));
    }
}

litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::GLVideoSegmDataEvaluator::GLVideoSegmDataEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent,size_t nTotFrameCount) :
        GLImageProcEvaluatorAlgo(pParent,nTotFrameCount,(size_t)BinClassifMetricsAccumulator::eCountersCount,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

std::string litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::GLVideoSegmDataEvaluator::getComputeShaderSource(size_t nStage) const {
    glAssert(nStage<m_nComputeStages);
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
            "layout(binding=" << GLImageProcAlgo::eImage_ROIBinding << ", r8ui) readonly uniform uimage2D imgROI;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_OutputBinding << ", r8ui) readonly uniform uimage2D imgInput;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_GTBinding << ", r8ui) readonly uniform uimage2D imgGT;\n";
    if(m_bUsingDebug) ssSrc <<
            "layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ") writeonly uniform uimage2D imgDebug;\n";
    ssSrc <<"layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::eCounter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::eCounter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::eCounter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::eCounter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::eCounter_SE*4 << ") uniform atomic_uint nSE;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << BinClassifMetricsAccumulator::eCounter_DC*4 << ") uniform atomic_uint nDC;\n";
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

litiv::BinClassifMetricsAccumulatorPtr litiv::IAsyncDataEvaluator_<litiv::eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL>::GLVideoSegmDataEvaluator::getMetricsBase() {
    const cv::Mat& oAtomicCountersQueryBuffer = this->getEvaluationAtomicCounterBuffer();
    BinClassifMetricsAccumulatorPtr pMetricsBase = BinClassifMetricsAccumulator::create();
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        pMetricsBase->nTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::eCounter_TP);
        pMetricsBase->nTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::eCounter_TN);
        pMetricsBase->nFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::eCounter_FP);
        pMetricsBase->nFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::eCounter_FN);
        pMetricsBase->nSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::eCounter_SE);
        pMetricsBase->nDC += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,BinClassifMetricsAccumulator::eCounter_DC);
    }
    return pMetricsBase;
}

#endif //HAVE_GLSL
