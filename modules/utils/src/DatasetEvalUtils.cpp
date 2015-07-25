#include "litiv/utils/DatasetEvalUtils.hpp"

#if HAVE_GLSL

DatasetUtils::EvaluatorBase::GLEvaluatorBase::GLEvaluatorBase(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount, size_t nCountersPerImage)
    :    GLImageProcEvaluatorAlgo(pParent,nTotImageCount,nCountersPerImage,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

DatasetUtils::Segm::SegmEvaluator::GLSegmEvaluator::GLSegmEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount)
    :    GLEvaluatorBase::GLEvaluatorBase(pParent,nTotFrameCount,eSegmEvalCountersCount) {}

DatasetUtils::Segm::BasicMetrics DatasetUtils::Segm::SegmEvaluator::GLSegmEvaluator::getCumulativeMetrics() {
    const cv::Mat& oAtomicCountersQueryBuffer = this->getEvaluationAtomicCounterBuffer();
    BasicMetrics m;
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        m.nTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eSegmEvalCounter_TP);
        m.nTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eSegmEvalCounter_TN);
        m.nFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eSegmEvalCounter_FP);
        m.nFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eSegmEvalCounter_FN);
        m.nSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eSegmEvalCounter_SE);
    }
    return m;
}

#endif //HAVE_GLSL

DatasetUtils::Segm::Metrics DatasetUtils::Segm::CalcMetricsFromWorkGroup(const DatasetUtils::WorkGroup& oGroup, bool bAverage) {
    if(!bAverage) {
        BasicMetrics oCumulBasicMetrics;
        for(auto ppBatchIter =oGroup.m_vpBatches.begin(); ppBatchIter!=oGroup.m_vpBatches.end(); ++ppBatchIter) {
            auto pSeq = std::dynamic_pointer_cast<SegmWorkBatch>(*ppBatchIter);
            CV_Assert(pSeq!=nullptr);
            oCumulBasicMetrics += pSeq->m_oMetrics;
        }
        return Metrics(oCumulBasicMetrics);
    }
    else {
        CV_Assert(!oGroup.m_vpBatches.empty());
        auto pFirstSeq = std::dynamic_pointer_cast<SegmWorkBatch>(oGroup.m_vpBatches.front());
        CV_Assert(pFirstSeq!=nullptr);
        Metrics tmp(pFirstSeq->m_oMetrics);
        for(auto ppBatchIter =oGroup.m_vpBatches.begin()+1; ppBatchIter!=oGroup.m_vpBatches.end(); ++ppBatchIter) {
            auto pSeq = std::dynamic_pointer_cast<SegmWorkBatch>(*ppBatchIter);
            CV_Assert(pSeq!=nullptr);
            tmp += pSeq->m_oMetrics;
        }
        return tmp;
    }
}

DatasetUtils::Segm::Metrics DatasetUtils::Segm::CalcMetricsFromWorkGroups(const std::vector<std::shared_ptr<DatasetUtils::WorkGroup>>& vpGroups, bool bAverage) {
    if(!bAverage) {
        BasicMetrics oCumulBasicMetrics;
        for(auto ppGroupIter =vpGroups.begin(); ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
            for(auto ppBatchIter =(*ppGroupIter)->m_vpBatches.begin(); ppBatchIter!=(*ppGroupIter)->m_vpBatches.end(); ++ppBatchIter) {
                auto pSeq = std::dynamic_pointer_cast<SegmWorkBatch>(*ppBatchIter);
                CV_Assert(pSeq!=nullptr);
                oCumulBasicMetrics += pSeq->m_oMetrics;
            }
        }
        return Metrics(oCumulBasicMetrics);
    }
    else {
        CV_Assert(!vpGroups.empty() && vpGroups[0]!=nullptr);
        Metrics res = CalcMetricsFromWorkGroup(*vpGroups[0],bAverage);
        res.nWeight = 1;
        for(auto ppGroupIter =vpGroups.begin()+1; ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
            CV_Assert((*ppGroupIter)!=nullptr);
            if(!(*ppGroupIter)->m_vpBatches.empty()) {
                Metrics tmp = CalcMetricsFromWorkGroup(**ppGroupIter,bAverage);
                tmp.nWeight = 1;
                res += tmp;
            }
        }
        return res;
    }
}

void DatasetUtils::Segm::WriteMetrics(const std::string& sResultsFilePath, const DatasetUtils::Segm::SegmWorkBatch& oBatch) {
    std::ofstream oMetricsOutput(sResultsFilePath);
    Metrics tmp(oBatch.m_oMetrics);
    const std::string sCurrSeqName = oBatch.m_sName.size()>12?oBatch.m_sName.substr(0,12):oBatch.m_sName;
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " : Rcl=" << std::fixed << std::setprecision(4) << tmp.dRecall << " Prc=" << tmp.dPrecision << " FM=" << tmp.dFMeasure << " MCC=" << tmp.dMCC << std::endl;
    oMetricsOutput << "Results for segm batch '" << oBatch.m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << "nTP nFP nFN nTN nSE nTot" << std::endl; // order similar to the files saved by the CDNet analysis script
    oMetricsOutput << oBatch.m_oMetrics.nTP << " " << oBatch.m_oMetrics.nFP << " " << oBatch.m_oMetrics.nFN << " " << oBatch.m_oMetrics.nTN << " " << oBatch.m_oMetrics.nSE << " " << oBatch.m_oMetrics.total() << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Cumulative metrics :" << std::endl;
    oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    oMetricsOutput << tmp.dRecall << " " << tmp.dSpecificity << " " << tmp.dFPR << " " << tmp.dFNR << " " << tmp.dPBC << " " << tmp.dPrecision << " " << tmp.dFMeasure << " " << tmp.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Work batch FPS: " << oBatch.m_oMetrics.dTimeElapsed_sec/oBatch.GetTotalImageCount() << std::endl;
    oMetricsOutput.close();
}

void DatasetUtils::Segm::WriteMetrics(const std::string& sResultsFilePath, const DatasetUtils::WorkGroup& oGroup) {
    std::ofstream oMetricsOutput(sResultsFilePath);
    oMetricsOutput << "Results for segm work group '" << oGroup.m_sName << "' :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Batch Metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    for(auto ppBatchIter =oGroup.m_vpBatches.begin(); ppBatchIter!=oGroup.m_vpBatches.end(); ++ppBatchIter) {
        auto pSeq = std::dynamic_pointer_cast<SegmWorkBatch>(*ppBatchIter);
        CV_Assert(pSeq!=nullptr);
        Metrics tmp(pSeq->m_oMetrics);
        std::string sName = (*ppBatchIter)->m_sName;
        if(sName.size()>10)
            sName = sName.substr(0,10);
        else if(sName.size()<10)
            sName += std::string(10-sName.size(),' ');
        oMetricsOutput << sName << " " << tmp.dRecall << " " << tmp.dSpecificity << " " << tmp.dFPR << " " << tmp.dFNR << " " << tmp.dPBC << " " << tmp.dPrecision << " " << tmp.dFMeasure << " " << tmp.dMCC << std::endl;
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    Metrics all(CalcMetricsFromWorkGroup(oGroup,DATASETUTILS_USE_AVERAGE_EVAL_METRICS));
    const std::string sCurrGroupName = oGroup.m_sName.size()>12?oGroup.m_sName.substr(0,12):oGroup.m_sName;
    std::cout << "\t" << std::setfill(' ') << std::setw(12) << sCurrGroupName << " : Rcl=" << std::fixed << std::setprecision(4) << all.dRecall << " Prc=" << all.dPrecision << " FM=" << all.dFMeasure << " MCC=" << all.dMCC << std::endl;
    oMetricsOutput << std::string(DATASETUTILS_USE_AVERAGE_EVAL_METRICS?"averaged   ":"cumulative ") << all.dRecall << " " << all.dSpecificity << " " << all.dFPR << " " << all.dFNR << " " << all.dPBC << " " << all.dPrecision << " " << all.dFMeasure << " " << all.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Work group FPS: " << all.dTimeElapsed_sec/oGroup.GetTotalImageCount() << std::endl;
    oMetricsOutput.close();
}

void DatasetUtils::Segm::WriteMetrics(const std::string& sResultsFilePath, const std::vector<std::shared_ptr<DatasetUtils::WorkGroup>>& vpGroups) {
    std::ofstream oMetricsOutput(sResultsFilePath);
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << "Overall results :" << std::endl;
    oMetricsOutput << std::endl;
    oMetricsOutput << std::fixed << std::setprecision(8);
    oMetricsOutput << std::string(DATASETUTILS_USE_AVERAGE_EVAL_METRICS?"Averaged":"Cumulative") << " work group metrics :" << std::endl;
    oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
    size_t nOverallFrameCount = 0;
    for(auto ppGroupIter =vpGroups.begin(); ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
        CV_Assert((*ppGroupIter)!=nullptr);
        if(!(*ppGroupIter)->m_vpBatches.empty()) {
            Metrics tmp(CalcMetricsFromWorkGroup(**ppGroupIter,DATASETUTILS_USE_AVERAGE_EVAL_METRICS));
            std::string sName = (*ppGroupIter)->m_sName;
            if(sName.size()>10)
                sName = sName.substr(0,10);
            else if(sName.size()<10)
                sName += std::string(10-sName.size(),' ');
            oMetricsOutput << sName << " " << tmp.dRecall << " " << tmp.dSpecificity << " " << tmp.dFPR << " " << tmp.dFNR << " " << tmp.dPBC << " " << tmp.dPrecision << " " << tmp.dFMeasure << " " << tmp.dMCC << std::endl;
            nOverallFrameCount += (*ppGroupIter)->GetTotalImageCount();
        }
    }
    oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
    Metrics all(CalcMetricsFromWorkGroups(vpGroups,DATASETUTILS_USE_AVERAGE_EVAL_METRICS));
    oMetricsOutput << "Overall    " << all.dRecall << " " << all.dSpecificity << " " << all.dFPR << " " << all.dFNR << " " << all.dPBC << " " << all.dPrecision << " " << all.dFMeasure << " " << all.dMCC << std::endl;
    oMetricsOutput << std::endl << std::endl;
    oMetricsOutput << "Overall FPS: " << all.dTimeElapsed_sec/nOverallFrameCount << std::endl;
    oMetricsOutput.close();
}

cv::Mat DatasetUtils::Segm::GetDisplayImage(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oSegmMask, const cv::Mat& oROI, size_t nIdx, cv::Point oDbgPt) {
    cv::Mat oInputImgBYTE3, oDebugImgBYTE3, oSegmMaskBYTE3;
    CV_Assert(!oInputImg.empty() && (oInputImg.type()==CV_8UC1 || oInputImg.type()==CV_8UC3 || oInputImg.type()==CV_8UC4));
    CV_Assert(!oDebugImg.empty() && (oDebugImg.type()==CV_8UC1 || oDebugImg.type()==CV_8UC3 || oDebugImg.type()==CV_8UC4));
    CV_Assert(!oSegmMask.empty() && (oSegmMask.type()==CV_8UC1 || oSegmMask.type()==CV_8UC3 || oSegmMask.type()==CV_8UC4));
    CV_Assert(!oROI.empty() && oROI.type()==CV_8UC1);
    if(oInputImg.channels()==1)
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_GRAY2RGB);
    else if(oInputImg.channels()==4)
        cv::cvtColor(oInputImg,oInputImgBYTE3,cv::COLOR_RGBA2RGB);
    else
        oInputImgBYTE3 = oInputImg;
    if(oDebugImg.channels()==1)
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_GRAY2RGB);
    else if(oDebugImg.channels()==4)
        cv::cvtColor(oDebugImg,oDebugImgBYTE3,cv::COLOR_RGBA2RGB);
    else
        oDebugImgBYTE3 = oDebugImg;
    if(oSegmMask.channels()==1)
        cv::cvtColor(oSegmMask,oSegmMaskBYTE3,cv::COLOR_GRAY2RGB);
    else if(oSegmMask.channels()==4)
        cv::cvtColor(oSegmMask,oDebugImgBYTE3,cv::COLOR_RGBA2RGB);
    else
        oSegmMaskBYTE3 = oSegmMask;
    if(oDbgPt!=cv::Point(-1,-1)) {
        cv::circle(oInputImgBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
        cv::circle(oSegmMaskBYTE3,oDbgPt,5,cv::Scalar(255,255,255));
    }
    cv::Mat displayH;
    cv::resize(oInputImgBYTE3,oInputImgBYTE3,cv::Size(320,240));
    cv::resize(oDebugImgBYTE3,oDebugImgBYTE3,cv::Size(320,240));
    cv::resize(oSegmMaskBYTE3,oSegmMaskBYTE3,cv::Size(320,240));

    std::stringstream sstr;
    sstr << "Input #" << nIdx;
    WriteOnImage(oInputImgBYTE3,sstr.str(),cv::Scalar_<uchar>(0,0,255));
    WriteOnImage(oDebugImgBYTE3,"Debug Image",cv::Scalar_<uchar>(0,0,255));
    WriteOnImage(oSegmMaskBYTE3,"Segm Mask",cv::Scalar_<uchar>(0,0,255));

    cv::hconcat(oInputImgBYTE3,oDebugImgBYTE3,displayH);
    cv::hconcat(displayH,oSegmMaskBYTE3,displayH);
    return displayH;
}

const uchar DatasetUtils::Segm::Video::BinarySegmEvaluator::g_nSegmPositive = 255;
const uchar DatasetUtils::Segm::Video::BinarySegmEvaluator::g_nSegmOutOfScope = DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL;
const uchar DatasetUtils::Segm::Video::BinarySegmEvaluator::g_nSegmNegative = 0;

#if HAVE_GLSL

DatasetUtils::Segm::Video::BinarySegmEvaluator::GLBinarySegmEvaluator::GLBinarySegmEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount)
    :    GLSegmEvaluator(pParent,nTotFrameCount) {}

std::string DatasetUtils::Segm::Video::BinarySegmEvaluator::GLBinarySegmEvaluator::getComputeShaderSource(size_t nStage) const {
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"#version 430\n"
            "#define VAL_POSITIVE     " << (uint)g_nSegmPositive << "\n"
            "#define VAL_NEGATIVE     " << (uint)g_nSegmNegative << "\n"
            "#define VAL_OUTOFSCOPE   " << (uint)g_nSegmOutOfScope << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_ROIBinding << ", r8ui) readonly uniform uimage2D imgROI;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_OutputBinding << ", r8ui) readonly uniform uimage2D imgInput;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_GTBinding << ", r8ui) readonly uniform uimage2D imgGT;\n";
    if(m_bUsingDebug) ssSrc <<
            "layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ") writeonly uniform uimage2D imgDebug;\n";
    ssSrc <<"layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_SE*4 << ") uniform atomic_uint nSE;\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"void main() {\n"
            "    ivec2 imgCoord = ivec2(gl_GlobalInvocationID.xy);\n"
            "    uint nInputSegmVal = imageLoad(imgInput,imgCoord).r;\n"
            "    uint nGTSegmVal = imageLoad(imgGT,imgCoord).r;\n"
            "    uint nROIVal = imageLoad(imgROI,imgCoord).r;\n"
            "    if(nROIVal!=VAL_NEGATIVE) {\n"
            "        if(nGTSegmVal!=VAL_OUTOFSCOPE) {\n"
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
            "        }\n"
            "    }\n";
    if(m_bUsingDebug) { ssSrc <<
            "    uvec4 out_color = uvec4(0,0,0,255);\n"
            "    if(nGTSegmVal!=VAL_OUTOFSCOPE && nROIVal!=VAL_NEGATIVE) {\n"
            "        if(nInputSegmVal==VAL_POSITIVE) {\n"
            "            if(nGTSegmVal==VAL_POSITIVE) {\n"
            "                out_color.g = uint(255);\n"
            "            }\n"
            "            else if(nGTSegmVal==VAL_NEGATIVE) {\n"
            "                out_color.r = uint(255);\n"
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
    ssSrc <<"}\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    return ssSrc.str();
}

std::shared_ptr<DatasetUtils::EvaluatorBase::GLEvaluatorBase> DatasetUtils::Segm::Video::BinarySegmEvaluator::CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const {
    return std::shared_ptr<GLEvaluatorBase>(new GLBinarySegmEvaluator(pParent,nTotImageCount));
}

#endif //HAVE_GLSL

void DatasetUtils::Segm::Video::BinarySegmEvaluator::AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, DatasetUtils::Segm::BasicMetrics& m) const {
    CV_DbgAssert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmMask.size()==oGTSegmMask.size() && oSegmMask.size()==oROI.size());
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if(gt_step_ptr[j]!=g_nSegmOutOfScope && roi_step_ptr[j]!=g_nSegmNegative) {
                if(input_step_ptr[j]==g_nSegmPositive) {
                    if(gt_step_ptr[j]==g_nSegmPositive)
                        ++m.nTP;
                    else // gt_step_ptr[j]==g_nSegmNegative
                        ++m.nFP;
                }
                else { // input_step_ptr[j]==g_nSegmNegative
                    if(gt_step_ptr[j]==g_nSegmPositive)
                        ++m.nFN;
                    else // gt_step_ptr[j]==g_nSegmNegative
                        ++m.nTN;
                }
            }
        }
    }
}

cv::Mat DatasetUtils::Segm::Video::BinarySegmEvaluator::GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const {
    CV_DbgAssert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmMask.size()==oGTSegmMask.size() && oSegmMask.size()==oROI.size());
    cv::Mat oResult(oSegmMask.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if(gt_step_ptr[j]!=g_nSegmOutOfScope && roi_step_ptr[j]!=g_nSegmNegative) {
                if(input_step_ptr[j]==g_nSegmPositive) {
                    if(gt_step_ptr[j]==g_nSegmPositive)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==g_nSegmNegative)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else for(size_t c=0; c<3; ++c)
                        res_step_ptr[j*3+c] = UCHAR_MAX/3;
                }
                else { // input_step_ptr[j]==g_nSegmNegative
                    if(gt_step_ptr[j]==g_nSegmPositive) {
                        res_step_ptr[j*3] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                }
            }
            else if(roi_step_ptr[j]==g_nSegmNegative) {
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

cv::Mat DatasetUtils::Segm::Video::ReadResult( const std::string& sResultsPath, const std::string& sGroupName, const std::string& sSeqName,
                                               const std::string& sResultPrefix, size_t nFrameIdx, const std::string& sResultSuffix, int nFlags) {
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nFrameIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sGroupName << "/" << sSeqName << "/" << sResultPrefix << acBuffer.data() << sResultSuffix;
    return cv::imread(sResultFilePath.str(),nFlags);
}

void DatasetUtils::Segm::Video::WriteResult( const std::string& sResultsPath, const std::string& sGroupName, const std::string& sSeqName, const std::string& sResultPrefix,
                                             size_t nFrameIdx, const std::string& sResultSuffix, const cv::Mat& oResult, const std::vector<int>& vnComprParams) {
    std::array<char,10> acBuffer;
    snprintf(acBuffer.data(),acBuffer.size(),"%06lu",nFrameIdx);
    std::stringstream sResultFilePath;
    sResultFilePath << sResultsPath << sGroupName << "/" << sSeqName << "/" << sResultPrefix << acBuffer.data() << sResultSuffix;
    cv::imwrite(sResultFilePath.str(),oResult,vnComprParams);
}

// as defined in the 2012 CDNet scripts/dataset
const uchar DatasetUtils::Segm::Video::CDnetEvaluator::g_nSegmPositive = 255;
const uchar DatasetUtils::Segm::Video::CDnetEvaluator::g_nSegmOutOfScope = 85;
const uchar DatasetUtils::Segm::Video::CDnetEvaluator::g_nSegmNegative = 0;
const uchar DatasetUtils::Segm::Video::CDnetEvaluator::g_nSegmUnknown = 170;
const uchar DatasetUtils::Segm::Video::CDnetEvaluator::g_nSegmShadow = 50;

#if HAVE_GLSL

DatasetUtils::Segm::Video::CDnetEvaluator::GLCDnetEvaluator::GLCDnetEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount)
    :    GLSegmEvaluator(pParent,nTotFrameCount) {}

std::string DatasetUtils::Segm::Video::CDnetEvaluator::GLCDnetEvaluator::getComputeShaderSource(size_t nStage) const {
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"#version 430\n"
            "#define VAL_POSITIVE     " << (uint)g_nSegmPositive << "\n"
            "#define VAL_NEGATIVE     " << (uint)g_nSegmNegative << "\n"
            "#define VAL_OUTOFSCOPE   " << (uint)g_nSegmOutOfScope << "\n"
            "#define VAL_UNKNOWN      " << (uint)g_nSegmUnknown << "\n"
            "#define VAL_SHADOW       " << (uint)g_nSegmShadow << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_ROIBinding << ", r8ui) readonly uniform uimage2D imgROI;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_OutputBinding << ", r8ui) readonly uniform uimage2D imgInput;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_GTBinding << ", r8ui) readonly uniform uimage2D imgGT;\n";
    if(m_bUsingDebug) ssSrc <<
            "layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ") writeonly uniform uimage2D imgDebug;\n";
    ssSrc <<"layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eSegmEvalCounter_SE*4 << ") uniform atomic_uint nSE;\n";
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

std::shared_ptr<DatasetUtils::EvaluatorBase::GLEvaluatorBase> DatasetUtils::Segm::Video::CDnetEvaluator::CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const {
    return std::shared_ptr<GLEvaluatorBase>(new GLCDnetEvaluator(pParent,nTotImageCount));
}

#endif //HAVE_GLSL

void DatasetUtils::Segm::Video::CDnetEvaluator::AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI, DatasetUtils::Segm::BasicMetrics& m) const {
    CV_DbgAssert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmMask.size()==oGTSegmMask.size() && oSegmMask.size()==oROI.size());
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if( gt_step_ptr[j]!=g_nSegmOutOfScope &&
                gt_step_ptr[j]!=g_nSegmUnknown &&
                roi_step_ptr[j]!=g_nSegmNegative ) {
                if(input_step_ptr[j]==g_nSegmPositive) {
                    if(gt_step_ptr[j]==g_nSegmPositive)
                        ++m.nTP;
                    else // gt_step_ptr[j]==g_nSegmNegative
                        ++m.nFP;
                }
                else { // input_step_ptr[j]==g_nSegmNegative
                    if(gt_step_ptr[j]==g_nSegmPositive)
                        ++m.nFN;
                    else // gt_step_ptr[j]==g_nSegmNegative
                        ++m.nTN;
                }
                if(gt_step_ptr[j]==g_nSegmShadow) {
                    if(input_step_ptr[j]==g_nSegmPositive)
                        ++m.nSE;
                }
            }
        }
    }
}

cv::Mat DatasetUtils::Segm::Video::CDnetEvaluator::GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const {
    CV_DbgAssert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && oROI.type()==CV_8UC1);
    CV_DbgAssert(oSegmMask.size()==oGTSegmMask.size() && oSegmMask.size()==oROI.size());
    cv::Mat oResult(oSegmMask.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if( gt_step_ptr[j]!=g_nSegmOutOfScope &&
                gt_step_ptr[j]!=g_nSegmUnknown &&
                roi_step_ptr[j]!=g_nSegmNegative ) {
                if(input_step_ptr[j]==g_nSegmPositive) {
                    if(gt_step_ptr[j]==g_nSegmPositive)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==g_nSegmNegative)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==g_nSegmShadow) {
                        res_step_ptr[j*3+1] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                    else {
                        for(size_t c=0; c<3; ++c)
                            res_step_ptr[j*3+c] = UCHAR_MAX/3;
                    }
                }
                else { // input_step_ptr[j]==g_nSegmNegative
                    if(gt_step_ptr[j]==g_nSegmPositive) {
                        res_step_ptr[j*3] = UCHAR_MAX/2;
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    }
                }
            }
            else if(roi_step_ptr[j]==g_nSegmNegative) {
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
