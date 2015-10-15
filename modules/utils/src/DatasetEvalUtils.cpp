#include "litiv/utils/DatasetEvalUtils.hpp"
#include "litiv/imgproc.hpp"
#include "litiv/utils/ConsoleUtils.hpp"

DatasetUtils::BasicMetrics::BasicMetrics(std::string sID) :
        nTP(0),nTN(0),nFP(0),nFN(0),nSE(0),dTimeElapsed_sec(0),sInternalID(sID) {}

DatasetUtils::BasicMetrics DatasetUtils::BasicMetrics::operator+(const BasicMetrics& m) const {
    CV_Assert(m.sInternalID==sInternalID);
    BasicMetrics res(m);
    res.nTP += this->nTP;
    res.nTN += this->nTN;
    res.nFP += this->nFP;
    res.nFN += this->nFN;
    res.nSE += this->nSE;
    res.dTimeElapsed_sec += this->dTimeElapsed_sec;
    return res;
}

DatasetUtils::BasicMetrics& DatasetUtils::BasicMetrics::operator+=(const BasicMetrics& m) {
    CV_Assert(m.sInternalID==sInternalID);
    this->nTP += m.nTP;
    this->nTN += m.nTN;
    this->nFP += m.nFP;
    this->nFN += m.nFN;
    this->nSE += m.nSE;
    this->dTimeElapsed_sec += m.dTimeElapsed_sec;
    return *this;
}

DatasetUtils::Metrics::Metrics(const BasicMetrics& m) :
        dRecall(CalcRecall(m)),
        dSpecificity(CalcSpecificity(m)),
        dFPR(CalcFalsePositiveRate(m)),
        dFNR(CalcFalseNegativeRate(m)),
        dPBC(CalcPercentBadClassifs(m)),
        dPrecision(CalcPrecision(m)),
        dFMeasure(CalcFMeasure(m)),
        dMCC(CalcMatthewsCorrCoeff(m)),
        dTimeElapsed_sec(m.dTimeElapsed_sec),
        sInternalID(m.sInternalID),
        nWeight(1) {}

DatasetUtils::Metrics DatasetUtils::Metrics::operator+(const BasicMetrics& m) const {
    CV_Assert(m.sInternalID==sInternalID);
    Metrics tmp(m);
    return (*this)+tmp;
}

DatasetUtils::Metrics& DatasetUtils::Metrics::operator+=(const BasicMetrics& m) {
    CV_Assert(m.sInternalID==sInternalID);
    Metrics tmp(m);
    (*this) += tmp;
    return *this;
}

DatasetUtils::Metrics DatasetUtils::Metrics::operator+(const Metrics& m) const {
    CV_Assert(m.sInternalID==sInternalID);
    Metrics res(m);
    const size_t nTotWeight = this->nWeight+res.nWeight;
    res.dRecall = (res.dRecall*res.nWeight + this->dRecall*this->nWeight)/nTotWeight;
    res.dSpecificity = (res.dSpecificity*res.nWeight + this->dSpecificity*this->nWeight)/nTotWeight;
    res.dFPR = (res.dFPR*res.nWeight + this->dFPR*this->nWeight)/nTotWeight;
    res.dFNR = (res.dFNR*res.nWeight + this->dFNR*this->nWeight)/nTotWeight;
    res.dPBC = (res.dPBC*res.nWeight + this->dPBC*this->nWeight)/nTotWeight;
    res.dPrecision = (res.dPrecision*res.nWeight + this->dPrecision*this->nWeight)/nTotWeight;
    res.dFMeasure = (res.dFMeasure*res.nWeight + this->dFMeasure*this->nWeight)/nTotWeight;
    res.dMCC = (res.dMCC*res.nWeight + this->dMCC*this->nWeight)/nTotWeight;
    res.dTimeElapsed_sec += this->dTimeElapsed_sec;
    res.nWeight = nTotWeight;
    return res;
}

DatasetUtils::Metrics& DatasetUtils::Metrics::operator+=(const Metrics& m) {
    CV_Assert(m.sInternalID==sInternalID);
    const size_t nTotWeight = this->nWeight+m.nWeight;
    this->dRecall = (m.dRecall*m.nWeight + this->dRecall*this->nWeight)/nTotWeight;
    this->dSpecificity = (m.dSpecificity*m.nWeight + this->dSpecificity*this->nWeight)/nTotWeight;
    this->dFPR = (m.dFPR*m.nWeight + this->dFPR*this->nWeight)/nTotWeight;
    this->dFNR = (m.dFNR*m.nWeight + this->dFNR*this->nWeight)/nTotWeight;
    this->dPBC = (m.dPBC*m.nWeight + this->dPBC*this->nWeight)/nTotWeight;
    this->dPrecision = (m.dPrecision*m.nWeight + this->dPrecision*this->nWeight)/nTotWeight;
    this->dFMeasure = (m.dFMeasure*m.nWeight + this->dFMeasure*this->nWeight)/nTotWeight;
    this->dMCC = (m.dMCC*m.nWeight + this->dMCC*this->nWeight)/nTotWeight;
    this->dTimeElapsed_sec += m.dTimeElapsed_sec;
    this->nWeight = nTotWeight;
    return *this;
}

double DatasetUtils::Metrics::CalcFMeasure(double dRecall, double dPrecision) {return (dRecall+dPrecision)>0?(2.0*(dRecall*dPrecision)/(dRecall+dPrecision)):0;}
double DatasetUtils::Metrics::CalcFMeasure(const BasicMetrics& m) {return CalcFMeasure(CalcRecall(m),CalcPrecision(m));}
double DatasetUtils::Metrics::CalcRecall(uint64_t nTP, uint64_t nTPFN) {return nTPFN>0?((double)nTP/nTPFN):0;}
double DatasetUtils::Metrics::CalcRecall(const BasicMetrics& m) {return (m.nTP+m.nFN)>0?((double)m.nTP/(m.nTP+m.nFN)):0;}
double DatasetUtils::Metrics::CalcPrecision(uint64_t nTP, uint64_t nTPFP) {return nTPFP>0?((double)nTP/nTPFP):0;}
double DatasetUtils::Metrics::CalcPrecision(const BasicMetrics& m) {return (m.nTP+m.nFP)>0?((double)m.nTP/(m.nTP+m.nFP)):0;}
double DatasetUtils::Metrics::CalcSpecificity(const BasicMetrics& m) {return (m.nTN+m.nFP)>0?((double)m.nTN/(m.nTN+m.nFP)):0;}
double DatasetUtils::Metrics::CalcFalsePositiveRate(const BasicMetrics& m) {return (m.nFP+m.nTN)>0?((double)m.nFP/(m.nFP+m.nTN)):0;}
double DatasetUtils::Metrics::CalcFalseNegativeRate(const BasicMetrics& m) {return (m.nTP+m.nFN)>0?((double)m.nFN/(m.nTP+m.nFN)):0;}
double DatasetUtils::Metrics::CalcPercentBadClassifs(const BasicMetrics& m) {return m.total()>0?(100.0*(m.nFN+m.nFP)/m.total()):0;}
double DatasetUtils::Metrics::CalcMatthewsCorrCoeff(const BasicMetrics& m) {return ((m.nTP+m.nFP)>0)&&((m.nTP+m.nFN)>0)&&((m.nTN+m.nFP)>0)&&((m.nTN+m.nFN)>0)?((((double)m.nTP*m.nTN)-(m.nFP*m.nFN))/sqrt(((double)m.nTP+m.nFP)*(m.nTP+m.nFN)*(m.nTN+m.nFP)*(m.nTN+m.nFN))):0;}

#if HAVE_GLSL

DatasetUtils::EvaluatorBase::GLEvaluatorBase::GLEvaluatorBase(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount, size_t nCountersPerImage) :
        GLImageProcEvaluatorAlgo(pParent,nTotImageCount,nCountersPerImage,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}

#endif //HAVE_GLSL

// as defined in the 2012 CDNet scripts/dataset
const uchar DatasetUtils::Segm::Video::BinarySegmEvaluator::s_nSegmPositive = 255;
const uchar DatasetUtils::Segm::Video::BinarySegmEvaluator::s_nSegmOutOfScope = DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL;
const uchar DatasetUtils::Segm::Video::BinarySegmEvaluator::s_nSegmNegative = 0;

DatasetUtils::Segm::Video::BinarySegmEvaluator::BinarySegmEvaluator(std::string sEvalID) : m_oBasicMetrics(sEvalID) {}

#if HAVE_GLSL

DatasetUtils::Segm::Video::BinarySegmEvaluator::GLBinarySegmEvaluator::GLBinarySegmEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount) :
        GLEvaluatorBase(pParent,nTotFrameCount) {}

std::string DatasetUtils::Segm::Video::BinarySegmEvaluator::GLBinarySegmEvaluator::getComputeShaderSource(size_t nStage) const {
    glAssert(nStage<m_nComputeStages);
    std::stringstream ssSrc;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"#version 430\n"
            "#define VAL_POSITIVE     " << (uint)s_nSegmPositive << "\n"
            "#define VAL_NEGATIVE     " << (uint)s_nSegmNegative << "\n"
            "#define VAL_OUTOFSCOPE   " << (uint)s_nSegmOutOfScope << "\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ssSrc <<"layout(local_size_x=" << m_vDefaultWorkGroupSize.x << ",local_size_y=" << m_vDefaultWorkGroupSize.y << ") in;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_ROIBinding << ", r8ui) readonly uniform uimage2D imgROI;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_OutputBinding << ", r8ui) readonly uniform uimage2D imgInput;\n"
            "layout(binding=" << GLImageProcAlgo::eImage_GTBinding << ", r8ui) readonly uniform uimage2D imgGT;\n";
    if(m_bUsingDebug) ssSrc <<
            "layout(binding=" << GLImageProcAlgo::eImage_DebugBinding << ") writeonly uniform uimage2D imgDebug;\n";
    ssSrc <<"layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_SE*4 << ") uniform atomic_uint nSE;\n";
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

DatasetUtils::BasicMetrics DatasetUtils::Segm::Video::BinarySegmEvaluator::GLBinarySegmEvaluator::getCumulativeMetrics() {
    const cv::Mat& oAtomicCountersQueryBuffer = this->getEvaluationAtomicCounterBuffer();
    BasicMetrics oBasicMetrics;
    for(int nFrameIter=0; nFrameIter<oAtomicCountersQueryBuffer.rows; ++nFrameIter) {
        oBasicMetrics.nTP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eBasicEvalCounter_TP);
        oBasicMetrics.nTN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eBasicEvalCounter_TN);
        oBasicMetrics.nFP += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eBasicEvalCounter_FP);
        oBasicMetrics.nFN += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eBasicEvalCounter_FN);
        oBasicMetrics.nSE += (uint32_t)oAtomicCountersQueryBuffer.at<int32_t>(nFrameIter,eBasicEvalCounter_SE);
    }
    return oBasicMetrics;
}

std::shared_ptr<DatasetUtils::EvaluatorBase::GLEvaluatorBase> DatasetUtils::Segm::Video::BinarySegmEvaluator::CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotImageCount) const {
    return std::shared_ptr<GLEvaluatorBase>(new GLBinarySegmEvaluator(pParent,nTotImageCount));
}

void DatasetUtils::Segm::Video::BinarySegmEvaluator::FetchGLEvaluationResults(std::shared_ptr<GLEvaluatorBase> pGLEvaluator) {
    auto pEval = std::dynamic_pointer_cast<GLBinarySegmEvaluator>(pGLEvaluator);
    if(pEval) {
        std::string sOldEvalName = m_oBasicMetrics.sInternalID;
        m_oBasicMetrics = pEval->getCumulativeMetrics();
        m_oBasicMetrics.sInternalID = sOldEvalName;
    }
}

#endif //HAVE_GLSL

cv::Mat DatasetUtils::Segm::Video::BinarySegmEvaluator::GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const {
    CV_Assert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
    CV_Assert(oSegmMask.size()==oGTSegmMask.size() && (oROI.empty() || oSegmMask.size()==oROI.size()));
    cv::Mat oResult(oSegmMask.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if(gt_step_ptr[j]!=s_nSegmOutOfScope && (oROI.empty() || roi_step_ptr[j]!=s_nSegmNegative) ) {
                if(input_step_ptr[j]==s_nSegmPositive) {
                    if(gt_step_ptr[j]==s_nSegmPositive)
                        res_step_ptr[j*3+1] = UCHAR_MAX;
                    else if(gt_step_ptr[j]==s_nSegmNegative)
                        res_step_ptr[j*3+2] = UCHAR_MAX;
                    else for(size_t c=0; c<3; ++c)
                        res_step_ptr[j*3+c] = UCHAR_MAX/3;
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

void DatasetUtils::Segm::Video::BinarySegmEvaluator::AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) {
    CV_Assert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
    CV_Assert(oSegmMask.size()==oGTSegmMask.size() && (oROI.empty() || oSegmMask.size()==oROI.size()));
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if(gt_step_ptr[j]!=s_nSegmOutOfScope && (oROI.empty() || roi_step_ptr[j]!=s_nSegmNegative) ) {
                if(input_step_ptr[j]==s_nSegmPositive) {
                    if(gt_step_ptr[j]==s_nSegmPositive)
                        ++m_oBasicMetrics.nTP;
                    else // gt_step_ptr[j]==s_nSegmNegative
                        ++m_oBasicMetrics.nFP;
                }
                else { // input_step_ptr[j]==s_nSegmNegative
                    if(gt_step_ptr[j]==s_nSegmPositive)
                        ++m_oBasicMetrics.nFN;
                    else // gt_step_ptr[j]==s_nSegmNegative
                        ++m_oBasicMetrics.nTN;
                }
            }
        }
    }
}

DatasetUtils::Metrics DatasetUtils::Segm::Video::BinarySegmEvaluator::CalcMetrics(const std::vector<std::shared_ptr<WorkGroup>>& vpGroups, bool bAverage) {
    CV_Assert(!vpGroups.empty());
    if(!bAverage) {
        BasicMetrics oCumulBasicMetrics;
        for(auto ppGroupIter =vpGroups.begin(); ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
            for(auto ppBatchIter =(*ppGroupIter)->m_vpBatches.begin(); ppBatchIter!=(*ppGroupIter)->m_vpBatches.end(); ++ppBatchIter) {
                auto pEval = std::dynamic_pointer_cast<BinarySegmEvaluator>((*ppBatchIter)->m_pEvaluator);
                if(pEval!=nullptr && pEval->m_oBasicMetrics.total()>0) {
                    pEval->m_oBasicMetrics.dTimeElapsed_sec = pEval->dTimeElapsed_sec;
                    oCumulBasicMetrics += pEval->m_oBasicMetrics;
                }
            }
        }
        CV_Assert(oCumulBasicMetrics.total()>0);
        return Metrics(oCumulBasicMetrics);
    }
    else {
        size_t nFirstGroupIdx = 0;
        auto pFirstGroup = vpGroups[nFirstGroupIdx++];
        while(pFirstGroup->m_vpBatches.empty()) {
            CV_Assert(vpGroups.size()>nFirstGroupIdx);
            pFirstGroup = vpGroups[nFirstGroupIdx++];
        }
        Metrics oMetrics = CalcMetrics(*pFirstGroup,bAverage);
        oMetrics.nWeight = 1;
        for(auto ppGroupIter =vpGroups.begin()+nFirstGroupIdx; ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
            if(!(*ppGroupIter)->m_vpBatches.empty()) {
                Metrics oTempMetrics = CalcMetrics(**ppGroupIter,bAverage);
                oTempMetrics.nWeight = 1;
                oMetrics += oTempMetrics;
            }
        }
        return oMetrics;
    }
}

DatasetUtils::Metrics DatasetUtils::Segm::Video::BinarySegmEvaluator::CalcMetrics(const WorkGroup& oGroup, bool bAverage) {
    CV_Assert(!oGroup.m_vpBatches.empty() && !oGroup.IsBare());
    if(!bAverage) {
        BasicMetrics oCumulBasicMetrics;
        for(auto ppBatchIter=oGroup.m_vpBatches.begin(); ppBatchIter!=oGroup.m_vpBatches.end(); ++ppBatchIter) {
            auto pEval = std::dynamic_pointer_cast<BinarySegmEvaluator>((*ppBatchIter)->m_pEvaluator);
            if(pEval!=nullptr && pEval->m_oBasicMetrics.total()>0) {
                pEval->m_oBasicMetrics.dTimeElapsed_sec = pEval->dTimeElapsed_sec;
                oCumulBasicMetrics += pEval->m_oBasicMetrics;
            }
        }
        CV_Assert(oCumulBasicMetrics.total()>0);
        return Metrics(oCumulBasicMetrics);
    }
    else {
        size_t nFirstEvalIdx = 0;
        auto pEval = std::dynamic_pointer_cast<BinarySegmEvaluator>(oGroup.m_vpBatches[nFirstEvalIdx++]->m_pEvaluator);
        while(pEval==nullptr || pEval->m_oBasicMetrics.total()==0) {
            CV_Assert(oGroup.m_vpBatches.size()>nFirstEvalIdx);
            pEval = std::dynamic_pointer_cast<BinarySegmEvaluator>(oGroup.m_vpBatches[nFirstEvalIdx++]->m_pEvaluator);
        }
        pEval->m_oBasicMetrics.dTimeElapsed_sec = pEval->dTimeElapsed_sec;
        Metrics oMetrics(pEval->m_oBasicMetrics);
        for(auto ppBatchIter=oGroup.m_vpBatches.begin()+nFirstEvalIdx; ppBatchIter!=oGroup.m_vpBatches.end(); ++ppBatchIter) {
            pEval = std::dynamic_pointer_cast<BinarySegmEvaluator>((*ppBatchIter)->m_pEvaluator);
            if(pEval!=nullptr && pEval->m_oBasicMetrics.total()>0) {
                pEval->m_oBasicMetrics.dTimeElapsed_sec = pEval->dTimeElapsed_sec;
                oMetrics += pEval->m_oBasicMetrics;
            }
        }
        return oMetrics;
    }
}

void DatasetUtils::Segm::Video::BinarySegmEvaluator::WriteEvalResults(const DatasetInfoBase& oInfo, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups, bool bAverageMetrics) {
    if(!vpGroups.empty()) {
        size_t nOverallFrameCount = 0;
        std::vector<Metrics> voGroupMetrics;
        std::vector<std::string> voGroupNames;
        for(auto ppGroupIter = vpGroups.begin(); ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
            if(!(*ppGroupIter)->m_vpBatches.empty() && !(*ppGroupIter)->IsBare()) {
                voGroupMetrics.push_back(WriteEvalResults(**ppGroupIter,bAverageMetrics));
                std::string sGroupName = (*ppGroupIter)->m_sName;
                if(sGroupName.size()>10)
                    sGroupName = sGroupName.substr(0,10);
                else if(sGroupName.size()<10)
                    sGroupName += std::string(10-sGroupName.size(),' ');
                voGroupNames.push_back(sGroupName);
                nOverallFrameCount += (*ppGroupIter)->GetTotalImageCount();
            }
        }
        if(!voGroupMetrics.empty()) {
            const std::string sEvalName = voGroupMetrics[0].sInternalID.empty()?std::string("<default>"):voGroupMetrics[0].sInternalID;
            Metrics oOverallMetrics(CalcMetrics(vpGroups,bAverageMetrics));
            std::cout << "Overall : Rcl=" << std::fixed << std::setprecision(4) << oOverallMetrics.dRecall << " Prc=" << oOverallMetrics.dPrecision << " FM=" << oOverallMetrics.dFMeasure << " MCC=" << oOverallMetrics.dMCC << "      [eval=" << sEvalName << "]" << std::endl;
            std::ofstream oMetricsOutput(oInfo.m_sResultsRootPath+"/overall.txt");
            if(oMetricsOutput.is_open()) {
                oMetricsOutput << "Evaluation set '" << sEvalName << "' for " << voGroupMetrics.size() << " segm work groups:" << std::endl;
                oMetricsOutput << std::endl;
                oMetricsOutput << std::fixed << std::setprecision(8);
                oMetricsOutput << std::string(bAverageMetrics?"Averaged":"Cumulative") << " work group results :" << std::endl;
                oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
                for(size_t g = 0; g<voGroupNames.size(); ++g)
                    oMetricsOutput << voGroupNames[g] << " " << voGroupMetrics[g].dRecall << " " << voGroupMetrics[g].dSpecificity << " " << voGroupMetrics[g].dFPR << " " << voGroupMetrics[g].dFNR << " " << voGroupMetrics[g].dPBC << " " << voGroupMetrics[g].dPrecision << " " << voGroupMetrics[g].dFMeasure << " " << voGroupMetrics[g].dMCC << std::endl;
                oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
                oMetricsOutput << std::string(bAverageMetrics?"averaged   ":"cumulative ") << oOverallMetrics.dRecall << " " << oOverallMetrics.dSpecificity << " " << oOverallMetrics.dFPR << " " << oOverallMetrics.dFNR << " " << oOverallMetrics.dPBC << " " << oOverallMetrics.dPrecision << " " << oOverallMetrics.dFMeasure << " " << oOverallMetrics.dMCC << std::endl;
                oMetricsOutput << std::endl;
                oMetricsOutput << "Overall FPS: " << nOverallFrameCount/oOverallMetrics.dTimeElapsed_sec << std::endl;
                oMetricsOutput << std::endl << std::endl << LITIV_FRAMEWORK_VERSION_SHA1 << std::endl;
            }
        }
    }
}

DatasetUtils::Metrics DatasetUtils::Segm::Video::BinarySegmEvaluator::WriteEvalResults(const WorkGroup& oGroup, bool bAverage) {
    CV_Assert(!oGroup.m_vpBatches.empty() && !oGroup.IsBare());
    std::vector<std::string> voBatchNames;
    std::vector<Metrics> voMetrics;
    for(auto ppBatchIter = oGroup.m_vpBatches.begin(); ppBatchIter!=oGroup.m_vpBatches.end(); ++ppBatchIter) {
        auto pEval = std::dynamic_pointer_cast<BinarySegmEvaluator>((*ppBatchIter)->m_pEvaluator);
        if(pEval!=nullptr && pEval->m_oBasicMetrics.total()>0) {
            size_t nBatchIdx = (size_t)std::distance(oGroup.m_vpBatches.begin(),ppBatchIter);
            CV_Assert(nBatchIdx==0 || voMetrics[nBatchIdx-1].sInternalID==pEval->m_oBasicMetrics.sInternalID);
            voMetrics.push_back(WriteEvalResults(**ppBatchIter));
            std::string sBatchName = (*ppBatchIter)->m_sName;
            if(sBatchName.size()>10)
                sBatchName = sBatchName.substr(0,10);
            else if(sBatchName.size()<10)
                sBatchName += std::string(10-sBatchName.size(),' ');
            voBatchNames.push_back(sBatchName);
        }
    }
    CV_Assert(!voMetrics.empty());
    const std::string sCurrGroupName = oGroup.m_sName.size()>12?oGroup.m_sName.substr(0,12):oGroup.m_sName;
    const std::string sEvalName = voMetrics[0].sInternalID.empty()?std::string("<default>"):voMetrics[0].sInternalID;
    Metrics oGroupMetrics(CalcMetrics(oGroup,bAverage));
    std::cout << "\t" << std::setfill(' ') << std::setw(12) << sCurrGroupName << " : Rcl=" << std::fixed << std::setprecision(4) << oGroupMetrics.dRecall << " Prc=" << oGroupMetrics.dPrecision << " FM=" << oGroupMetrics.dFMeasure << " MCC=" << oGroupMetrics.dMCC << "      [eval=" << sEvalName << "]" << std::endl;
    std::ofstream oMetricsOutput(oGroup.m_sResultsPath+"/../"+oGroup.m_sName+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << "Evaluation set '" << sEvalName << "' for segm work group '" << oGroup.m_sName << "' :" << std::endl;
        oMetricsOutput << std::endl;
        oMetricsOutput << std::fixed << std::setprecision(8);
        oMetricsOutput << "Batch results :" << std::endl;
        oMetricsOutput << "           Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
        for(size_t b = 0; b<voMetrics.size(); ++b)
            oMetricsOutput << voBatchNames[b] << " " << voMetrics[b].dRecall << " " << voMetrics[b].dSpecificity << " " << voMetrics[b].dFPR << " " << voMetrics[b].dFNR << " " << voMetrics[b].dPBC << " " << voMetrics[b].dPrecision << " " << voMetrics[b].dFMeasure << " " << voMetrics[b].dMCC << std::endl;
        oMetricsOutput << "--------------------------------------------------------------------------------------------------" << std::endl;
        oMetricsOutput << std::string(bAverage?"averaged   ":"cumulative ") << oGroupMetrics.dRecall << " " << oGroupMetrics.dSpecificity << " " << oGroupMetrics.dFPR << " " << oGroupMetrics.dFNR << " " << oGroupMetrics.dPBC << " " << oGroupMetrics.dPrecision << " " << oGroupMetrics.dFMeasure << " " << oGroupMetrics.dMCC << std::endl;
        oMetricsOutput << std::endl;
        oMetricsOutput << "Work group FPS: " << oGroup.GetTotalImageCount()/oGroupMetrics.dTimeElapsed_sec << std::endl;
        oMetricsOutput << std::endl << std::endl << LITIV_FRAMEWORK_VERSION_SHA1 << std::endl;
    }
    return oGroupMetrics;
}

DatasetUtils::Metrics DatasetUtils::Segm::Video::BinarySegmEvaluator::WriteEvalResults(const WorkBatch& oBatch) {
    auto pEval = std::dynamic_pointer_cast<BinarySegmEvaluator>(oBatch.m_pEvaluator);
    CV_Assert(pEval!=nullptr && pEval->m_oBasicMetrics.total()>0);
    pEval->m_oBasicMetrics.dTimeElapsed_sec = pEval->dTimeElapsed_sec;
    const BasicMetrics& oCurrBasicMetrics = pEval->m_oBasicMetrics;
    const std::string sCurrSeqName = oBatch.m_sName.size()>12?oBatch.m_sName.substr(0,12):oBatch.m_sName;
    const std::string sEvalName = oCurrBasicMetrics.sInternalID.empty()?std::string("<default>"):oCurrBasicMetrics.sInternalID;
    Metrics oBatchMetrics(oCurrBasicMetrics);
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " : Rcl=" << std::fixed << std::setprecision(4) << oBatchMetrics.dRecall << " Prc=" << oBatchMetrics.dPrecision << " FM=" << oBatchMetrics.dFMeasure << " MCC=" << oBatchMetrics.dMCC << "      [eval=" << sEvalName << "]" << std::endl;
    std::ofstream oMetricsOutput(oBatch.m_sResultsPath+"/../"+oBatch.m_sName+".txt");
    if(oMetricsOutput.is_open()) {
        oMetricsOutput << "Evaluation set '" << sEvalName << "' for segm batch '" << oBatch.m_sName << "' :" << std::endl;
        oMetricsOutput << std::endl;
        oMetricsOutput << "nTP nFP nFN nTN nSE nTot" << std::endl; // order similar to the files saved by the CDNet analysis script
        oMetricsOutput << oCurrBasicMetrics.nTP << " " << oCurrBasicMetrics.nFP << " " << oCurrBasicMetrics.nFN << " " << oCurrBasicMetrics.nTN << " " << oCurrBasicMetrics.nSE << " " << oCurrBasicMetrics.total() << std::endl;
        oMetricsOutput << std::endl;
        oMetricsOutput << std::fixed << std::setprecision(8);
        oMetricsOutput << "Cumulative metrics :" << std::endl;
        oMetricsOutput << "Rcl        Spc        FPR        FNR        PBC        Prc        FM         MCC        " << std::endl;
        oMetricsOutput << oBatchMetrics.dRecall << " " << oBatchMetrics.dSpecificity << " " << oBatchMetrics.dFPR << " " << oBatchMetrics.dFNR << " " << oBatchMetrics.dPBC << " " << oBatchMetrics.dPrecision << " " << oBatchMetrics.dFMeasure << " " << oBatchMetrics.dMCC << std::endl;
        oMetricsOutput << std::endl;
        oMetricsOutput << "Work batch FPS: " << oBatch.GetTotalImageCount()/oCurrBasicMetrics.dTimeElapsed_sec << std::endl;
        oMetricsOutput << std::endl << std::endl << LITIV_FRAMEWORK_VERSION_SHA1 << std::endl;
    }
    return oBatchMetrics;
}

// as defined in the 2012 CDNet scripts/dataset
const uchar DatasetUtils::Segm::Video::CDnetEvaluator::s_nSegmUnknown = 170;
const uchar DatasetUtils::Segm::Video::CDnetEvaluator::s_nSegmShadow = 50;

DatasetUtils::Segm::Video::CDnetEvaluator::CDnetEvaluator() : BinarySegmEvaluator("CDNET_EVAL") {}

#if HAVE_GLSL

DatasetUtils::Segm::Video::CDnetEvaluator::GLCDnetEvaluator::GLCDnetEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount) :
        GLBinarySegmEvaluator(pParent,nTotFrameCount) {}

std::string DatasetUtils::Segm::Video::CDnetEvaluator::GLCDnetEvaluator::getComputeShaderSource(size_t nStage) const {
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
    ssSrc <<"layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_TP*4 << ") uniform atomic_uint nTP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_TN*4 << ") uniform atomic_uint nTN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_FP*4 << ") uniform atomic_uint nFP;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_FN*4 << ") uniform atomic_uint nFN;\n"
            "layout(binding=" << GLImageProcAlgo::eAtomicCounterBuffer_EvalBinding << ", offset=" << eBasicEvalCounter_SE*4 << ") uniform atomic_uint nSE;\n";
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

cv::Mat DatasetUtils::Segm::Video::CDnetEvaluator::GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) const {
    CV_Assert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
    CV_Assert(oSegmMask.size()==oGTSegmMask.size() && (oROI.empty() || oSegmMask.size()==oROI.size()));
    cv::Mat oResult(oSegmMask.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        uchar* res_step_ptr = oResult.data+idx_nstep*3;
        for(int j=0; j<oSegmMask.cols; ++j) {
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

void DatasetUtils::Segm::Video::CDnetEvaluator::AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& oROI) {
    CV_Assert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1 && (oROI.empty() || oROI.type()==CV_8UC1));
    CV_Assert(oSegmMask.size()==oGTSegmMask.size() && (oROI.empty() || oSegmMask.size()==oROI.size()));
    const size_t step_row = oSegmMask.step.p[0];
    for(size_t i=0; i<(size_t)oSegmMask.rows; ++i) {
        const size_t idx_nstep = step_row*i;
        const uchar* input_step_ptr = oSegmMask.data+idx_nstep;
        const uchar* gt_step_ptr = oGTSegmMask.data+idx_nstep;
        const uchar* roi_step_ptr = oROI.data+idx_nstep;
        for(int j=0; j<oSegmMask.cols; ++j) {
            if(gt_step_ptr[j]!=s_nSegmOutOfScope && gt_step_ptr[j]!=s_nSegmUnknown && (oROI.empty() || roi_step_ptr[j]!=s_nSegmNegative) ) {
                if(input_step_ptr[j]==s_nSegmPositive) {
                    if(gt_step_ptr[j]==s_nSegmPositive)
                        ++m_oBasicMetrics.nTP;
                    else // gt_step_ptr[j]==s_nSegmNegative
                        ++m_oBasicMetrics.nFP;
                }
                else { // input_step_ptr[j]==s_nSegmNegative
                    if(gt_step_ptr[j]==s_nSegmPositive)
                        ++m_oBasicMetrics.nFN;
                    else // gt_step_ptr[j]==s_nSegmNegative
                        ++m_oBasicMetrics.nTN;
                }
                if(gt_step_ptr[j]==s_nSegmShadow) {
                    if(input_step_ptr[j]==s_nSegmPositive)
                        ++m_oBasicMetrics.nSE;
                }
            }
        }
    }
}

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
#include "3rdparty/BSDS500/csa.hh"
#include "3rdparty/BSDS500/kofn.hh"
#include "3rdparty/BSDS500/match.hh"
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
const double DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::s_dMaxImageDiagRatioDist = 0.0075;

DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::BSDS500BoundaryEvaluator(size_t nThresholdBins) : m_nThresholdBins(nThresholdBins) {CV_Assert(m_nThresholdBins>0 && m_nThresholdBins<=UCHAR_MAX);}

cv::Mat DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/) const {
    CV_Assert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1);
    CV_Assert(oSegmMask.cols==oGTSegmMask.cols && (oGTSegmMask.rows%oSegmMask.rows)==0 && (oGTSegmMask.rows/oSegmMask.rows)>=1);
    CV_Assert(oSegmMask.step.p[0]==oGTSegmMask.step.p[0]);
    const double dMaxDist = s_dMaxImageDiagRatioDist*sqrt(double(oSegmMask.cols*oSegmMask.cols+oSegmMask.rows*oSegmMask.rows));
    const int nMaxDist = (int)ceil(dMaxDist);
    CV_Assert(dMaxDist>0 && nMaxDist>0);
    cv::Mat oSegmMask_TP(oSegmMask.size(),CV_16UC1,cv::Scalar_<ushort>(0));
    cv::Mat oSegmMask_FN(oSegmMask.size(),CV_16UC1,cv::Scalar_<ushort>(0));
    cv::Mat oSegmMask_FP(oSegmMask.size(),CV_16UC1,cv::Scalar_<ushort>(0));
    const size_t nGTMaskCount = size_t(oGTSegmMask.rows/oSegmMask.rows);
    for(size_t nGTMaskIdx=0; nGTMaskIdx<nGTMaskCount; ++nGTMaskIdx) {
        cv::Mat oCurrGTSegmMask = oGTSegmMask(cv::Rect(0,int(oSegmMask.rows*nGTMaskIdx),oSegmMask.cols,oSegmMask.rows));
        cv::Mat oCurrGTSegmMask_dilated,oSegmMask_dilated;
        cv::Mat oDilateKernel(2*nMaxDist+1,2*nMaxDist+1,CV_8UC1,cv::Scalar_<uchar>(255));
        cv::dilate(oCurrGTSegmMask,oCurrGTSegmMask_dilated,oDilateKernel);
        cv::dilate(oSegmMask,oSegmMask_dilated,oDilateKernel);
        cv::add((oSegmMask&oCurrGTSegmMask_dilated),oSegmMask_TP,oSegmMask_TP,cv::noArray(),CV_16U);
        cv::add((oSegmMask&(oCurrGTSegmMask_dilated==0)),oSegmMask_FP,oSegmMask_FP,cv::noArray(),CV_16U);
        cv::add(((oSegmMask_dilated==0)&oCurrGTSegmMask),oSegmMask_FN,oSegmMask_FN,cv::noArray(),CV_16U);
    }
    cv::Mat oSegmMask_TP_byte, oSegmMask_FN_byte, oSegmMask_FP_byte;
    oSegmMask_TP.convertTo(oSegmMask_TP_byte,CV_8U,1.0/nGTMaskCount);
    oSegmMask_FN.convertTo(oSegmMask_FN_byte,CV_8U,1.0/nGTMaskCount);
    oSegmMask_FP.convertTo(oSegmMask_FP_byte,CV_8U,1.0/nGTMaskCount);
    cv::Mat oResult(oSegmMask.size(),CV_8UC3,cv::Scalar_<uchar>(0));
    const std::vector<int> vnMixPairs = {0,2, 1,0, 2,1};
    cv::mixChannels(std::vector<cv::Mat>{oSegmMask_FN_byte|oSegmMask_FP_byte,oSegmMask_FN_byte,oSegmMask_TP_byte},std::vector<cv::Mat>{oResult},vnMixPairs.data(),vnMixPairs.size()/2);
    return oResult;
}

void DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/) {
    CV_Assert(oSegmMask.type()==CV_8UC1 && oGTSegmMask.type()==CV_8UC1);
    CV_Assert(oSegmMask.isContinuous() && oGTSegmMask.isContinuous());
    CV_Assert(oSegmMask.cols==oGTSegmMask.cols && (oGTSegmMask.rows%oSegmMask.rows)==0 && (oGTSegmMask.rows/oSegmMask.rows)>=1);
    CV_Assert(oSegmMask.step.p[0]==oGTSegmMask.step.p[0]);

    const double dMaxDist = s_dMaxImageDiagRatioDist*sqrt(double(oSegmMask.cols*oSegmMask.cols+oSegmMask.rows*oSegmMask.rows));
    const double dMaxDistSqr = dMaxDist*dMaxDist;
    const int nMaxDist = (int)ceil(dMaxDist);
    CV_Assert(dMaxDist>0 && nMaxDist>0);

    const std::vector<uchar> vuEvalUniqueVals = PlatformUtils::unique_8uc1_values(oSegmMask);
    BSDS500BasicMetrics oBasicMetrics(m_nThresholdBins);
    CV_DbgAssert(m_voBasicMetrics.empty() || m_voBasicMetrics[0].vnThresholds==oBasicMetrics.vnThresholds);
    cv::Mat oCurrSegmMask(oSegmMask.size(),CV_8UC1), oTmpSegmMask(oSegmMask.size(),CV_8UC1);
    cv::Mat oSegmTPAccumulator(oSegmMask.size(),CV_8UC1);
    size_t nNextEvalUniqueValIdx = 0;
    size_t nThresholdBinIdx = 0;
    while(nThresholdBinIdx<oBasicMetrics.vnThresholds.size()) {
        cv::compare(oSegmMask,oBasicMetrics.vnThresholds[nThresholdBinIdx],oTmpSegmMask,cv::CMP_GE);
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

        // CSA code needs integer weights.  Use this multiplier to convert
        // floating-point weights to integers.
        static const int multiplier = 100;
        // The degree of outlier connections.
        static const int degree = 6;
        CV_Assert(degree > 0);
        CV_Assert(multiplier > 0);


        for(size_t nGTMaskIdx=0; nGTMaskIdx<size_t(oGTSegmMask.rows/oCurrSegmMask.rows); ++nGTMaskIdx) {
            cv::Mat oCurrGTSegmMask = oGTSegmMask(cv::Rect(0,int(oCurrSegmMask.rows*nGTMaskIdx),oCurrSegmMask.cols,oCurrSegmMask.rows));
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
            m += voEdges.size();              // real connections
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
        for(size_t nGTMaskIdx = 0; nGTMaskIdx<size_t(oGTSegmMask.rows/oCurrSegmMask.rows); ++nGTMaskIdx) {
            cv::Mat oCurrGTSegmMask = oGTSegmMask(cv::Rect(0,int(oCurrSegmMask.rows*nGTMaskIdx),oCurrSegmMask.cols,oCurrSegmMask.rows));
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
        oBasicMetrics.vnIndivTP[nThresholdBinIdx] = nIndivTP;
        oBasicMetrics.vnIndivTPFN[nThresholdBinIdx] = nGTPosCount;

        //pr = TP / (TP + FP)
        uint64_t nSegmTPAccCount = uint64_t(cv::countNonZero(oSegmTPAccumulator));
        uint64_t nSegmPosCount = uint64_t(cv::countNonZero(oCurrSegmMask));
        CV_Assert(nSegmPosCount>=nSegmTPAccCount);
        oBasicMetrics.vnTotalTP[nThresholdBinIdx] = nSegmTPAccCount;
        oBasicMetrics.vnTotalTPFP[nThresholdBinIdx] = nSegmPosCount;
        while(nNextEvalUniqueValIdx+1<vuEvalUniqueVals.size() && vuEvalUniqueVals[nNextEvalUniqueValIdx]<=oBasicMetrics.vnThresholds[nThresholdBinIdx])
            ++nNextEvalUniqueValIdx;
        while(++nThresholdBinIdx<oBasicMetrics.vnThresholds.size() && oBasicMetrics.vnThresholds[nThresholdBinIdx]<=vuEvalUniqueVals[nNextEvalUniqueValIdx]) {
            oBasicMetrics.vnIndivTP[nThresholdBinIdx] = oBasicMetrics.vnIndivTP[nThresholdBinIdx-1];
            oBasicMetrics.vnIndivTPFN[nThresholdBinIdx] = oBasicMetrics.vnIndivTPFN[nThresholdBinIdx-1];
            oBasicMetrics.vnTotalTP[nThresholdBinIdx] = oBasicMetrics.vnTotalTP[nThresholdBinIdx-1];
            oBasicMetrics.vnTotalTPFP[nThresholdBinIdx] = oBasicMetrics.vnTotalTPFP[nThresholdBinIdx-1];
        }

        const float fCompltRatio = float(nThresholdBinIdx)/oBasicMetrics.vnThresholds.size();
        litiv::updateConsoleProgressBar("BSDS500 eval:",fCompltRatio);
    }
    litiv::cleanConsoleRow();
    m_voBasicMetrics.push_back(oBasicMetrics);
}

DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::BSDS500BasicMetrics::BSDS500BasicMetrics(size_t nThresholdsBins) : vnIndivTP(nThresholdsBins,0), vnIndivTPFN(nThresholdsBins,0), vnTotalTP(nThresholdsBins,0), vnTotalTPFP(nThresholdsBins,0), vnThresholds(PlatformUtils::linspace<uchar>(0,UCHAR_MAX,nThresholdsBins,false)) {}

void DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::setThresholdBins(size_t nThresholdBins) {
    CV_Assert(m_nThresholdBins>0 && m_nThresholdBins<=UCHAR_MAX);
    CV_Assert(m_voBasicMetrics.empty() || m_voBasicMetrics[0].vnThresholds.size()==nThresholdBins); // can't change once we started the eval
    m_nThresholdBins = nThresholdBins;
}

size_t DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::getThresholdBins() const {
    return m_nThresholdBins;
}

void DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::CalcMetrics(const WorkBatch& oBatch, BSDS500Metrics& oRes) {
    auto pEval = std::dynamic_pointer_cast<BSDS500BoundaryEvaluator>(oBatch.m_pEvaluator);
    CV_Assert(pEval!=nullptr && !pEval->m_voBasicMetrics.empty());
    oRes.dTimeElapsed_sec = pEval->dTimeElapsed_sec;
    BSDS500BasicMetrics oCumulBasicMetrics(pEval->m_nThresholdBins);
    BSDS500BasicMetrics oMaxBasicMetrics(1);
    const size_t nImageCount = pEval->m_voBasicMetrics.size();
    oRes.voBestImageScores.resize(nImageCount);
    for(size_t nImageIdx = 0; nImageIdx<nImageCount; ++nImageIdx) {
        CV_DbgAssert(!pEval->m_voBasicMetrics[nImageIdx].vnIndivTP.empty() && !pEval->m_voBasicMetrics[nImageIdx].vnIndivTPFN.empty());
        CV_DbgAssert(!pEval->m_voBasicMetrics[nImageIdx].vnTotalTP.empty() && !pEval->m_voBasicMetrics[nImageIdx].vnTotalTPFP.empty());
        CV_DbgAssert(pEval->m_voBasicMetrics[nImageIdx].vnIndivTP.size()==pEval->m_voBasicMetrics[nImageIdx].vnIndivTPFN.size());
        CV_DbgAssert(pEval->m_voBasicMetrics[nImageIdx].vnTotalTP.size()==pEval->m_voBasicMetrics[nImageIdx].vnTotalTPFP.size());
        CV_DbgAssert(pEval->m_voBasicMetrics[nImageIdx].vnIndivTP.size()==pEval->m_voBasicMetrics[nImageIdx].vnTotalTP.size());
        CV_DbgAssert(pEval->m_voBasicMetrics[nImageIdx].vnThresholds.size()==pEval->m_voBasicMetrics[nImageIdx].vnTotalTP.size());
        CV_DbgAssert(nImageIdx==0 || pEval->m_voBasicMetrics[nImageIdx].vnIndivTP.size()==pEval->m_voBasicMetrics[nImageIdx-1].vnIndivTP.size());
        CV_DbgAssert(nImageIdx==0 || pEval->m_voBasicMetrics[nImageIdx].vnThresholds==pEval->m_voBasicMetrics[nImageIdx-1].vnThresholds);
        std::vector<BSDS500Score> voImageScore_PerThreshold(pEval->m_nThresholdBins);
        for(size_t nThresholdIdx = 0; nThresholdIdx<pEval->m_nThresholdBins; ++nThresholdIdx) {
            voImageScore_PerThreshold[nThresholdIdx].dRecall = Metrics::CalcRecall(pEval->m_voBasicMetrics[nImageIdx].vnIndivTP[nThresholdIdx],pEval->m_voBasicMetrics[nImageIdx].vnIndivTPFN[nThresholdIdx]);
            voImageScore_PerThreshold[nThresholdIdx].dPrecision = Metrics::CalcPrecision(pEval->m_voBasicMetrics[nImageIdx].vnTotalTP[nThresholdIdx],pEval->m_voBasicMetrics[nImageIdx].vnTotalTPFP[nThresholdIdx]);
            voImageScore_PerThreshold[nThresholdIdx].dFMeasure = Metrics::CalcFMeasure(voImageScore_PerThreshold[nThresholdIdx].dRecall,voImageScore_PerThreshold[nThresholdIdx].dPrecision);
            voImageScore_PerThreshold[nThresholdIdx].dThreshold = double(pEval->m_voBasicMetrics[nImageIdx].vnThresholds[nThresholdIdx])/UCHAR_MAX;
            oCumulBasicMetrics.vnIndivTP[nThresholdIdx] += pEval->m_voBasicMetrics[nImageIdx].vnIndivTP[nThresholdIdx];
            oCumulBasicMetrics.vnIndivTPFN[nThresholdIdx] += pEval->m_voBasicMetrics[nImageIdx].vnIndivTPFN[nThresholdIdx];
            oCumulBasicMetrics.vnTotalTP[nThresholdIdx] += pEval->m_voBasicMetrics[nImageIdx].vnTotalTP[nThresholdIdx];
            oCumulBasicMetrics.vnTotalTPFP[nThresholdIdx] += pEval->m_voBasicMetrics[nImageIdx].vnTotalTPFP[nThresholdIdx];
        }
        oRes.voBestImageScores[nImageIdx] = FindMaxFMeasure(voImageScore_PerThreshold);
        size_t nMaxFMeasureIdx = (size_t)std::distance(voImageScore_PerThreshold.begin(),std::max_element(voImageScore_PerThreshold.begin(),voImageScore_PerThreshold.end(),[](const BSDS500Score& n1, const BSDS500Score& n2){
            return n1.dFMeasure<n2.dFMeasure;
        }));
        oMaxBasicMetrics.vnIndivTP[0] += pEval->m_voBasicMetrics[nImageIdx].vnIndivTP[nMaxFMeasureIdx];
        oMaxBasicMetrics.vnIndivTPFN[0] += pEval->m_voBasicMetrics[nImageIdx].vnIndivTPFN[nMaxFMeasureIdx];
        oMaxBasicMetrics.vnTotalTP[0] += pEval->m_voBasicMetrics[nImageIdx].vnTotalTP[nMaxFMeasureIdx];
        oMaxBasicMetrics.vnTotalTPFP[0] += pEval->m_voBasicMetrics[nImageIdx].vnTotalTPFP[nMaxFMeasureIdx];
    }
    // ^^^ voBestImageScores => eval_bdry_img.txt
    oRes.voThresholdScores.resize(pEval->m_nThresholdBins);
    for(size_t nThresholdIdx = 0; nThresholdIdx<oCumulBasicMetrics.vnThresholds.size(); ++nThresholdIdx) {
        oRes.voThresholdScores[nThresholdIdx].dRecall = Metrics::CalcRecall(oCumulBasicMetrics.vnIndivTP[nThresholdIdx],oCumulBasicMetrics.vnIndivTPFN[nThresholdIdx]);
        oRes.voThresholdScores[nThresholdIdx].dPrecision = Metrics::CalcPrecision(oCumulBasicMetrics.vnTotalTP[nThresholdIdx],oCumulBasicMetrics.vnTotalTPFP[nThresholdIdx]);
        oRes.voThresholdScores[nThresholdIdx].dFMeasure = Metrics::CalcFMeasure(oRes.voThresholdScores[nThresholdIdx].dRecall,oRes.voThresholdScores[nThresholdIdx].dPrecision);
        oRes.voThresholdScores[nThresholdIdx].dThreshold = double(oCumulBasicMetrics.vnThresholds[nThresholdIdx])/UCHAR_MAX;
    }
    // ^^^ voThresholdScores => eval_bdry_thr.txt
    oRes.oBestScore = FindMaxFMeasure(oRes.voThresholdScores);
    oRes.dMaxRecall = Metrics::CalcRecall(oMaxBasicMetrics.vnIndivTP[0],oMaxBasicMetrics.vnIndivTPFN[0]);
    oRes.dMaxPrecision = Metrics::CalcPrecision(oMaxBasicMetrics.vnTotalTP[0],oMaxBasicMetrics.vnTotalTPFP[0]);
    oRes.dMaxFMeasure = Metrics::CalcFMeasure(oRes.dMaxRecall,oRes.dMaxPrecision);
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

void DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::WriteEvalResults(const DatasetInfoBase& oInfo, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups) {
    if(!vpGroups.empty()) {
        size_t nOverallImageCount = 0;
        std::vector<BSDS500Metrics> voBatchMetrics;
        std::vector<std::string> vsBatchNames;
        for(auto ppGroupIter = vpGroups.begin(); ppGroupIter!=vpGroups.end(); ++ppGroupIter) {
            for(auto ppBatchIter = (*ppGroupIter)->m_vpBatches.begin(); ppBatchIter!=(*ppGroupIter)->m_vpBatches.end(); ++ppBatchIter) {
                voBatchMetrics.push_back(BSDS500Metrics());
                WriteEvalResults(**ppBatchIter,voBatchMetrics.back());
                std::string sBatchName = (*ppBatchIter)->m_sName;
                if(sBatchName.size()>10)
                    sBatchName = sBatchName.substr(0,10);
                else if(sBatchName.size()<10)
                    sBatchName += std::string(10-sBatchName.size(),' ');
                vsBatchNames.push_back(sBatchName);
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
            std::cout << "\t" << std::setfill(' ') << std::setw(12) << "ALL-AVG" << " : MaxRcl=" << std::fixed << std::setprecision(4) << dMaxRecall << " MaxPrc=" << dMaxPrecision << " MaxFM=" << dMaxFMeasure << std::endl;
            std::cout << "\t" << std::setfill(' ') << std::setw(12) << " " <<       " : BestRcl=" << std::fixed << std::setprecision(4) << dBestRecall << " BestPrc=" << dBestPrecision << " BestFM=" << dBestFMeasure << std::endl;
#if USE_BSDS500_BENCHMARK
            std::ofstream oMetricsInfoOutput(oInfo.m_sResultsRootPath+"/reimpl_eval.txt");
#else //!USE_BSDS500_BENCHMARK
            std::ofstream oMetricsInfoOutput(oInfo.m_sResultsRootPath+"/homemade_eval.txt");
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
                oMetricsInfoOutput << std::endl << std::endl << LITIV_FRAMEWORK_VERSION_SHA1 << std::endl;
            }
        }
    }
}

void DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::WriteEvalResults(const WorkBatch& oBatch, BSDS500Metrics& oRes) {
    CalcMetrics(oBatch,oRes);
#if USE_BSDS500_BENCHMARK
    const std::string sResultPath = oBatch.m_sResultsPath+"/../"+oBatch.m_sName+"_reimpl_eval/";
#else //!USE_BSDS500_BENCHMARK
    const std::string sResultPath = oBatch.m_sResultsPath+"/../"+oBatch.m_sName+"_homemade_eval/";
#endif //!USE_BSDS500_BENCHMARK
    PlatformUtils::CreateDirIfNotExist(sResultPath);
    std::ofstream oImageScoresOutput(sResultPath+"/eval_bdry_img.txt");
    if(oImageScoresOutput.is_open())
        for(size_t n=0; n<oRes.voBestImageScores.size(); ++n)
            oImageScoresOutput << cv::format("%10d %10g %10g %10g %10g\n",n+1,oRes.voBestImageScores[n].dThreshold,oRes.voBestImageScores[n].dRecall,oRes.voBestImageScores[n].dPrecision,oRes.voBestImageScores[n].dFMeasure);
    std::ofstream oThresholdMetricsOutput(sResultPath+"/eval_bdry_thr.txt");
    if(oThresholdMetricsOutput.is_open())
        for(size_t n=0; n<oRes.voThresholdScores.size(); ++n)
            oThresholdMetricsOutput << cv::format("%10g %10g %10g %10g\n",oRes.voThresholdScores[n].dThreshold,oRes.voThresholdScores[n].dRecall,oRes.voThresholdScores[n].dPrecision,oRes.voThresholdScores[n].dFMeasure);
    std::ofstream oOverallMetricsOutput(sResultPath+"/eval_bdry.txt");
    if(oOverallMetricsOutput.is_open())
        oOverallMetricsOutput << cv::format("%10g %10g %10g %10g %10g %10g %10g %10g\n",oRes.oBestScore.dThreshold,oRes.oBestScore.dRecall,oRes.oBestScore.dPrecision,oRes.oBestScore.dFMeasure,oRes.dMaxRecall,oRes.dMaxPrecision,oRes.dMaxFMeasure,oRes.dAreaPR);
    const std::string sCurrSeqName = oBatch.m_sName.size()>12?oBatch.m_sName.substr(0,12):oBatch.m_sName;
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << sCurrSeqName << " : MaxRcl=" << std::fixed << std::setprecision(4) << oRes.dMaxRecall << " MaxPrc=" << oRes.dMaxPrecision << " MaxFM=" << oRes.dMaxFMeasure << std::endl;
    std::cout << "\t\t" << std::setfill(' ') << std::setw(12) << " " <<          " : BestRcl=" << std::fixed << std::setprecision(4) << oRes.oBestScore.dRecall << " BestPrc=" << oRes.oBestScore.dPrecision << " BestFM=" << oRes.oBestScore.dFMeasure << "  (@ T=" << std::fixed << std::setprecision(4) << oRes.oBestScore.dThreshold << ")" << std::endl;
}

DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::BSDS500Score DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::FindMaxFMeasure(const std::vector<uchar>& vnThresholds, const std::vector<double>& vdRecall, const std::vector<double>& vdPrecision) {
    CV_Assert(!vnThresholds.empty() && !vdRecall.empty() && !vdPrecision.empty());
    CV_Assert(vnThresholds.size()==vdRecall.size() && vdRecall.size()==vdPrecision.size());
    BSDS500Score oRes;
    oRes.dFMeasure = Metrics::CalcFMeasure(vdRecall[0],vdPrecision[0]);
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
            const double dInterpFMeasure = Metrics::CalcFMeasure(dInterpRecall,dInterpPrecision);
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

DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::BSDS500Score DatasetUtils::Segm::Image::BSDS500BoundaryEvaluator::FindMaxFMeasure(const std::vector<BSDS500Score>& voScores) {
    CV_Assert(!voScores.empty());
    BSDS500Score oRes = voScores[0];
    for(size_t nScoreIdx=1; nScoreIdx<voScores.size(); ++nScoreIdx) {
        const size_t nInterpCount = 100;
        for(size_t nInterpIdx=0; nInterpIdx<=nInterpCount; ++nInterpIdx) {
            const double dLastInterp = double(nInterpCount-nInterpIdx)/nInterpCount;
            const double dCurrInterp = double(nInterpIdx)/nInterpCount;
            const double dInterpRecall = dLastInterp*voScores[nScoreIdx-1].dRecall + dCurrInterp*voScores[nScoreIdx].dRecall;
            const double dInterpPrecision = dLastInterp*voScores[nScoreIdx-1].dPrecision + dCurrInterp*voScores[nScoreIdx].dPrecision;
            const double dInterpFMeasure = Metrics::CalcFMeasure(dInterpRecall,dInterpPrecision);
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
