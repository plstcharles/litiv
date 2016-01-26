
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

#pragma once

#define DEFAULT_BSDS500_EDGE_EVAL_THRESHOLD_BINS 99

#include "litiv/datasets/DatasetUtils.hpp"

namespace litiv {

    struct ClassifMetricsBase {
        ClassifMetricsBase();
        ClassifMetricsBase operator+(const ClassifMetricsBase& m) const;
        ClassifMetricsBase& operator+=(const ClassifMetricsBase& m);
        uint64_t total() const {return nTP+nTN+nFP+nFN;}
        uint64_t nTP;
        uint64_t nTN;
        uint64_t nFP;
        uint64_t nFN;
        uint64_t nSE; // 'shadow error', not always used/required for eval
    };

    struct ClassifMetrics {
        ClassifMetrics(const ClassifMetricsBase& m);
        ClassifMetrics operator+(const ClassifMetrics& m) const;
        ClassifMetrics& operator+=(const ClassifMetrics& m);
        double dRecall;
        double dSpecificity;
        double dFPR;
        double dFNR;
        double dPBC;
        double dPrecision;
        double dFMeasure;
        double dMCC;
        size_t nWeight; // used to compute iterative averages in overloads only
        static double CalcFMeasure(double dRecall, double dPrecision);
        static double CalcFMeasure(const ClassifMetricsBase& m);
        static double CalcRecall(uint64_t nTP, uint64_t nTPFN);
        static double CalcRecall(const ClassifMetricsBase& m);
        static double CalcPrecision(uint64_t nTP, uint64_t nTPFP);
        static double CalcPrecision(const ClassifMetricsBase& m);
        static double CalcSpecificity(const ClassifMetricsBase& m);
        static double CalcFalsePositiveRate(const ClassifMetricsBase& m);
        static double CalcFalseNegativeRate(const ClassifMetricsBase& m);
        static double CalcPercentBadClassifs(const ClassifMetricsBase& m);
        static double CalcMatthewsCorrCoeff(const ClassifMetricsBase& m);
    };

    template<eDatasetTypeList eDatasetType>
    struct IDatasetEvaluator_ : public IDataset {
        virtual void writeEvalReport() const override {
            std::cout << "Writing evaluation report for dataset '" << getDatasetName() << "'..." << std::endl;
            if(getBatches().empty()) {
                std::cout << "\tNo report to write for dataset '" << getDatasetName() << "', skipping." << std::endl;
                return;
            }
            for(const auto& pGroupIter : getBatches())
                pGroupIter->writeEvalReport();
            std::ofstream oMetricsOutput(getResultsRootPath()+"/overall.txt");
            if(oMetricsOutput.is_open()) {
                oMetricsOutput << std::fixed;
                oMetricsOutput << "Default evaluation report for dataset '" << getDatasetName() << "' :\n\n";
                oMetricsOutput << "            |Packets     |Seconds     |Hz          \n";
                oMetricsOutput << "------------|------------|------------|------------\n";
                size_t nOverallPacketCount = 0;
                double dOverallTimeElapsed = 0.0;
                for(const auto& pGroupIter : getBatches()) {
                    oMetricsOutput << pGroupIter->writeInlineEvalReport(0);
                    nOverallPacketCount += pGroupIter->getTotPackets();
                    dOverallTimeElapsed += pGroupIter->getProcessTime();
                }
                oMetricsOutput << "------------|------------|------------|------------\n";
                oMetricsOutput << "overall     |" <<
                std::setw(12) << nOverallPacketCount << "|" <<
                std::setw(12) << dOverallTimeElapsed << "|" <<
                std::setw(12) << nOverallPacketCount/dOverallTimeElapsed << "\n";
                oMetricsOutput << "\n" << LITIV_VERSION_SHA1 << "\n" << CxxUtils::getTimeStamp() << std::endl;
            }
        }
    };

    template<>
    struct IDatasetEvaluator_<eDatasetType_VideoSegm> : public IDataset {
        virtual ClassifMetricsBase getMetricsBase() const;
        virtual ClassifMetrics getMetrics(bool bAverage) const;
        virtual void writeEvalReport() const override;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset>
    struct DatasetEvaluator_ : public IDatasetEvaluator_<eDatasetType> {}; // can be specialized for custom operations

#if 0//HAVE_GLSL // put in IDataEvaluator_<eDatasetType_VideoSegm>? @@@@@

    enum eClassifMetricsBaseCountersList {
        eClassifMetricsBaseCounter_TP,
        eClassifMetricsBaseCounter_TN,
        eClassifMetricsBaseCounter_FP,
        eClassifMetricsBaseCounter_FN,
        eClassifMetricsBaseCounter_SE, // 'shadow error', not always used/required for eval
        eClassifMetricsBaseCountersCount,
    };

    struct IGLEvaluator {
        virtual std::string getComputeShaderSource(size_t nStage) const = 0;
        virtual ~IGLEvaluator() = default;
    };

    template<eDatasetTypeList eDatasetType>
    struct IGLEvaluator_;

    template<>
    struct IGLEvaluator_<eDatasetType_VideoSegm> :
            public GLImageProcEvaluatorAlgo {
        IGLEvaluator_(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount) :
                GLImageProcEvaluatorAlgo(pParent,nTotFrameCount,(size_t)eClassifMetricsBaseCountersCount,pParent->getIsUsingDisplay()?CV_8UC4:-1,CV_8UC1,true) {}
        virtual std::string getComputeShaderSource(size_t nStage) const;
        virtual ClassifMetricsBase getCumulativeMetrics();
    };

    template<eDatasetList eDataset>
    struct GLEvaluator_;

    template<>
    struct GLEvaluator_<eDataset_VideoSegm_CDnet> :
            public IGLEvaluator_<eDatasetType_VideoSegm> {
        GLEvaluator_(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount) :
                IGLEvaluator_(pParent,nTotFrameCount) {}
        virtual std::string getComputeShaderSource(size_t nStage) const;
    };

#endif //HAVE_GLSL

    template<eDatasetTypeList eDatasetType>
    struct IMetricsCalculator_ : public virtual IDataHandler {
        virtual std::string writeInlineEvalReport(size_t nIndentSize, size_t nCellSize=12) const override {
            if(!this->getTotPackets())
                return std::string();
            std::stringstream ssStr;
            ssStr << std::fixed;
            if(this->isGroup() && !this->isBare())
                for(const auto& pBatch : this->getBatches())
                    ssStr << pBatch->writeInlineEvalReport(nIndentSize+1);
            ssStr << std::setw(nCellSize) << (std::string('>',nIndentSize)+this->getName()) << "|" <<
            std::setw(nCellSize) << this->getTotPackets() << "|" <<
            std::setw(nCellSize) << this->getProcessTime() << "|" <<
            std::setw(nCellSize) << this->getTotPackets()/this->getProcessTime() << "\n";
            return ssStr.str();
        }

        virtual void writeEvalReport() const override {
            if(!this->getTotPackets()) {
                std::cout << "No report to write for '" << this->getName() << "', skipping..." << std::endl;
                return;
            }
            if(this->isGroup() && !this->isBare()) {
                for(const auto& pBatch : this->getBatches())
                    pBatch->writeEvalReport();
            }
            std::ofstream oMetricsOutput(this->getResultsPath()+"/../"+this->getName()+".txt");
            if(oMetricsOutput.is_open()) {
                oMetricsOutput << std::fixed;
                oMetricsOutput << "Default evaluation report for '" << this->getName() << "' :\n\n";
                oMetricsOutput << "            |Packets     |Seconds     |Hz          \n";
                oMetricsOutput << "------------|------------|------------|------------\n";
                oMetricsOutput << this->writeInlineEvalReport(0);
                oMetricsOutput << "\n" << LITIV_VERSION_SHA1 << "\n" << CxxUtils::getTimeStamp() << std::endl;
            }
        }
    };

    template<>
    struct IMetricsCalculator_<eDatasetType_VideoSegm> : public virtual IDataHandler { // contains group-impl only
        virtual ClassifMetricsBase getMetricsBase() const {
            ClassifMetricsBase oMetricsBase;
            for(const auto& pBatch : getBatches())
                oMetricsBase += dynamic_cast<const IMetricsCalculator_<eDatasetType_VideoSegm>&>(*pBatch).getMetricsBase();
            return oMetricsBase;
        }
        virtual ClassifMetrics getMetrics(bool bAverage) const {
            if(bAverage && isGroup() && !isBare()) {
                const IDataHandlerPtrArray& vpBatches = getBatches();
                auto ppBatchIter = vpBatches.begin();
                for(; ppBatchIter!=vpBatches.end() && !(*ppBatchIter)->getTotPackets(); ++ppBatchIter);
                CV_Assert(ppBatchIter!=vpBatches.end());
                ClassifMetrics oMetrics(dynamic_cast<const IMetricsCalculator_<eDatasetType_VideoSegm>&>(**ppBatchIter).getMetrics(bAverage));
                for(; ppBatchIter!=vpBatches.end(); ++ppBatchIter)
                    if((*ppBatchIter)->getTotPackets())
                        oMetrics += dynamic_cast<const IMetricsCalculator_<eDatasetType_VideoSegm>&>(**ppBatchIter).getMetrics(bAverage);
                // @@@ check returning metrics weight?
                return oMetrics;
            }
            return ClassifMetrics(getMetricsBase());
        }

        virtual std::string writeInlineEvalReport(size_t nIndentSize, size_t nCellSize=12) const override {
            if(!getTotPackets())
                return std::string();
            std::stringstream ssStr;
            ssStr << std::fixed;
            if(isGroup() && !isBare())
                for(const auto& pBatch : getBatches())
                    ssStr << pBatch->writeInlineEvalReport(nIndentSize+1);
            const ClassifMetrics& oMetrics = getMetrics(true);
            ssStr << std::setw(nCellSize) << (std::string('>',nIndentSize)+getName()) << "|" <<
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

        virtual void writeEvalReport() const override {
            if(!getTotPackets()) {
                std::cout << "No report to write for '" << getName() << "', skipping..." << std::endl;
                return;
            }
            if(isGroup() && !isBare()) {
                for(const auto& pBatch : getBatches())
                    pBatch->writeEvalReport();
            }
            const ClassifMetrics& oMetrics = getMetrics(true);
            std::cout << "\t" << std::setw(12) << (std::string('>',size_t(!isGroup()))+getName()) << " : Rcl=" << std::fixed << std::setprecision(4) << oMetrics.dRecall << " Prc=" << oMetrics.dPrecision << " FM=" << oMetrics.dFMeasure << " MCC=" << oMetrics.dMCC << std::endl;
            std::ofstream oMetricsOutput(getResultsPath()+"/../"+getName()+".txt");
            if(oMetricsOutput.is_open()) {
                oMetricsOutput << std::fixed;
                oMetricsOutput << "Video segmentation evaluation report for '" << getName() << "' :\n\n";
                oMetricsOutput << "            |Rcl         |Spc         |FPR         |FNR         |PBC         |Prc         |FM          |MCC         \n";
                oMetricsOutput << "------------|------------|------------|------------|------------|------------|------------|------------|------------\n";
                oMetricsOutput << writeInlineEvalReport(0);
                oMetricsOutput << "\nHz: " << getTotPackets()/getProcessTime() << "\n";
                oMetricsOutput << "\n" << LITIV_VERSION_SHA1 << "\n" << CxxUtils::getTimeStamp() << std::endl;
            }
        }

    };

    template<eDatasetTypeList eDatasetType, bool bGroup>
    struct IDataEvaluator_ :
            public IMetricsCalculator_<eDatasetType>,
            public IDataConsumer_<eDatasetType,bGroup> {};

    template<>
    struct IDataEvaluator_<eDatasetType_VideoSegm,TNoGroup> : public IDataConsumer_<eDatasetType_VideoSegm,TNoGroup> {
        virtual ClassifMetricsBase getMetricsBase() const;
        virtual ClassifMetrics getMetrics(bool bAverage) const;
        virtual std::string writeInlineEvalReport(size_t nIndentSize, size_t nCellSize=12) const override;
        virtual void writeEvalReport() const override;
        virtual cv::Mat getColoredSegmMaskFromResult(const cv::Mat& oSegm, size_t nIdx);

#if HAVE_GLSL
        //virtual std::shared_ptr<GLEvaluator> CreateGLEvaluator(const std::shared_ptr<GLImageProcAlgo>& pParent, size_t nTotFrameCount) const {
        //    return std::shared_ptr<GLEvaluator>(new GLEvaluator(pParent,nTotFrameCount));
        //}
        //virtual void FetchGLEvaluationResults(std::shared_ptr<IGLEvaluator_<eDatasetType_VideoSegm>> pGLEvaluator);
#endif //HAVE_GLSL

        static const uchar s_nSegmPositive;
        static const uchar s_nSegmNegative;
        static const uchar s_nSegmOutOfScope;
        static const uchar s_nSegmUnknown;
        static const uchar s_nSegmShadow;

    protected:
        virtual void _pushResult(const cv::Mat& oSegm, size_t nIdx) override;
        ClassifMetricsBase m_oMetricsBase;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, bool bGroup>
    struct DataEvaluator_ : public IDataEvaluator_<eDatasetType,bGroup> {};

#if 0

    namespace Image {

        namespace Segm {

            struct BSDS500BoundaryEvaluator : public IEvaluator {
                BSDS500BoundaryEvaluator(size_t nThresholdBins=DEFAULT_BSDS500_EDGE_EVAL_THRESHOLD_BINS);
                virtual cv::Mat GetColoredSegmMaskFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/) const;
                virtual void AccumulateMetricsFromResult(const cv::Mat& oSegmMask, const cv::Mat& oGTSegmMask, const cv::Mat& /*oUnused*/);
                static const double s_dMaxImageDiagRatioDist;
                struct BSDS500ClassifMetricsBase { // basic eval metrics for a single image
                    BSDS500ClassifMetricsBase(size_t nThresholdsBins); // always skips zero threshold
                    std::vector<uint64_t> vnIndivTP; // one count per threshold
                    std::vector<uint64_t> vnIndivTPFN; // one count per threshold
                    std::vector<uint64_t> vnTotalTP; // one count per threshold
                    std::vector<uint64_t> vnTotalTPFP; // one count per threshold
                    const std::vector<uchar> vnThresholds; // list of thresholds
                };
                void setThresholdBins(size_t nThresholdBins);
                size_t getThresholdBins() const;
            protected:
                size_t m_nThresholdBins;
                std::vector<BSDS500ClassifMetricsBase> m_voClassifMetricsBase;
                struct BSDS500Score { // edge detection score for a single threshold
                    double dThreshold;
                    double dRecall;
                    double dPrecision;
                    double dFMeasure;
                };
                struct BSDS500Metrics { // high-level metrics for an entire image set
                    std::vector<BSDS500Score> voBestImageScores; // one score per image (best threshold)
                    std::vector<BSDS500Score> voThresholdScores; // one score per threshold (cumul images)
                    BSDS500Score oBestScore; // best score for all thresholds
                    double dMaxRecall;
                    double dMaxPrecision;
                    double dMaxFMeasure;
                    double dAreaPR;
                };
                static void CalcMetrics(const WorkBatch& oBatch, BSDS500Metrics& oRes);
                static void WriteEvalResults(const DatasetInfoBase& oInfo, const std::vector<std::shared_ptr<WorkGroup>>& vpGroups);
                static void WriteEvalResults(const WorkBatch& oBatch, BSDS500Metrics& oRes);
                static BSDS500Score FindMaxFMeasure(const std::vector<uchar>& vnThresholds, const std::vector<double>& vdRecall, const std::vector<double>& vdPrecision);
                static BSDS500Score FindMaxFMeasure(const std::vector<BSDS500Score>& voScores);
                friend struct DatasetInfo;
            };

        } //namespace Segm

    } //namespace Image

#endif

} //namespace litiv
