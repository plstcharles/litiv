
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

#ifndef _LITIV_DATASETS_IMPL_H_
#error "This file should never be included directly; use litiv/datasets.hpp instead"
#endif //_LITIV_DATASETS_IMPL_H_

// as defined in the BSDS500 scripts/dataset
#define DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS   99
#define DATASETS_BSDS500_EVAL_IMAGE_DIAG_RATIO_DIST 0.0075

#include "litiv/datasets.hpp" // for parsers only, not truly required here

namespace lv {

    enum BSDS500DatasetGroup {
        BSDS500Dataset_Training,
        BSDS500Dataset_Training_Validation,
        BSDS500Dataset_Training_Validation_Test,
    };

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_BSDS500,eEvalImpl> :
            public IDataset_<eDatasetTask,DatasetSource_Image,Dataset_BSDS500,lv::getDatasetEval<eDatasetTask,Dataset_BSDS500>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via datasets::create
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in BSR dataset folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bForce4ByteDataAlign=false, ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
                double dScaleFactor=1.0, ///< defines the scale factor to use to resize/rescale read packets
                BSDS500DatasetGroup eType=BSDS500Dataset_Training ///< defines which dataset groups to use
        ) :
                IDataset_<eDatasetTask,DatasetSource_Image,Dataset_BSDS500,lv::getDatasetEval<eDatasetTask,Dataset_BSDS500>(),eEvalImpl>(
                        "BSDS500",
                        lv::datasets::getDatasetsRootPath()+"BSDS500/data/images/",
                        lv::datasets::getDatasetsRootPath()+"BSDS500/BSR/"+lv::AddDirSlashIfMissing(sOutputDirName),
                        "",
                        ".png",
                        getWorkBatchDirNames(eType),
                        std::vector<std::string>(),
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        bForce4ByteDataAlign,
                        dScaleFactor
                ) {}
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames(BSDS500DatasetGroup eType=BSDS500Dataset_Training) {
            static const std::vector<std::string> s_vsWorkBatchDirs_train = {"train"};
            static const std::vector<std::string> s_vsWorkBatchDirs_trainval = {"train","val"};
            static const std::vector<std::string> s_vsWorkBatchDirs_trainvaltest = {"train","val","test"};
            if(eType==BSDS500Dataset_Training)
                return s_vsWorkBatchDirs_train;
            else if(eType==BSDS500Dataset_Training_Validation)
                return s_vsWorkBatchDirs_trainval;
            else
                return s_vsWorkBatchDirs_trainvaltest;
        }
    };

    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_Image,Dataset_BSDS500> :
            public IDataProducer_<DatasetSource_Image> {
    protected:
        /// default constructor; initializes base class with gt packet type as non-image (needs unpacking), and gt mapping as index only
        DataProducer_() :
                IDataProducer_<DatasetSource_Image>(NotImagePacket,lv::getOutputPacketType<eDatasetTask,Dataset_BSDS500>(),IndexMapping,lv::getIOMappingType<eDatasetTask,Dataset_BSDS500>()) {}
        /// data parsing function, dataset-specific (default parser is not satisfactory)
        virtual void parseData() override final {
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            lv::GetFilesFromDir(this->getDataPath(),this->m_vsInputPaths);
            lv::FilterFilePaths(this->m_vsInputPaths,{},{".jpg",".png",".bmp"});
            if(this->m_vsInputPaths.empty())
                lvError_("BSDS500 set '%s' did not possess any jpg/png/bmp image file",this->getName().c_str());
            lv::GetSubDirsFromDir(this->getRoot()->getDataPath()+"../groundTruth_bdry_images/"+this->getRelativePath(),this->m_vsGTPaths);
            if(this->m_vsGTPaths.empty())
                lvError_("BSDS500 set '%s' did not possess any groundtruth image folders",this->getName().c_str());
            else if(this->m_vsGTPaths.size()!=this->m_vsInputPaths.size())
                lvError_("BSDS500 set '%s' input/groundtruth count mismatch",this->getName().c_str());
            this->m_mGTIndexLUT.clear();
            for(size_t n=0; n<this->m_vsInputPaths.size(); ++n)
                this->m_mGTIndexLUT[n] = n;
            // make sure folders are non-empty, and folders & images are similarliy ordered
            std::vector<std::string> vsTempPaths;
            for(size_t nImageIdx=0; nImageIdx<this->m_vsGTPaths.size(); ++nImageIdx) {
                lv::GetFilesFromDir(this->m_vsGTPaths[nImageIdx],vsTempPaths);
                lvAssert(!vsTempPaths.empty());
                const size_t nLastInputSlashPos = this->m_vsInputPaths[nImageIdx].find_last_of("/\\");
                const std::string sInputFullName = nLastInputSlashPos==std::string::npos?this->m_vsInputPaths[nImageIdx]:this->m_vsInputPaths[nImageIdx].substr(nLastInputSlashPos+1);
                const size_t nLastGTSlashPos = this->m_vsGTPaths[nImageIdx].find_last_of("/\\");
                lvAssert(sInputFullName.find(nLastGTSlashPos==std::string::npos?this->m_vsGTPaths[nImageIdx]:this->m_vsGTPaths[nImageIdx].substr(nLastGTSlashPos+1))!=std::string::npos);
            }
            this->m_bIsInputConstantSize = this->m_bIsGTConstantSize = true;
            this->m_oInputMaxSize = this->m_oGTMaxSize = cv::Size(0,0);
            this->m_vInputSizes.clear();
            this->m_vInputSizes.reserve(this->m_vsInputPaths.size());
            this->m_vGTSizes.clear();
            this->m_vGTSizes.reserve(this->m_vsGTPaths.size());
            const double dScale = this->getScaleFactor();
            for(size_t nImageIdx=0; nImageIdx<this->m_vsInputPaths.size(); ++nImageIdx) {
                const cv::Mat oCurrInput = cv::imread(this->m_vsInputPaths[nImageIdx],this->isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
                lvAssert(!oCurrInput.empty() && (oCurrInput.size()==cv::Size(321,481) || oCurrInput.size()==cv::Size(481,321)));
                lv::GetFilesFromDir(this->m_vsGTPaths[nImageIdx],vsTempPaths);
                lvAssert(!vsTempPaths.empty());
                this->m_vInputSizes.push_back(cv::Size(int(oCurrInput.cols*dScale),int(oCurrInput.rows*dScale)));
                this->m_vGTSizes.push_back(cv::Size(int(oCurrInput.cols*dScale),int(oCurrInput.rows*vsTempPaths.size()*dScale)));
                this->m_oInputMaxSize.width = std::max(this->m_oInputMaxSize.width,this->m_vInputSizes[nImageIdx].width);
                this->m_oInputMaxSize.height = std::max(this->m_oInputMaxSize.height,this->m_vInputSizes[nImageIdx].height);
                this->m_oGTMaxSize.width = std::max(this->m_oGTMaxSize.width,this->m_vGTSizes[nImageIdx].width);
                this->m_oGTMaxSize.height = std::max(this->m_oGTMaxSize.height,this->m_vGTSizes[nImageIdx].height);
            }
            lvAssert(this->m_vInputSizes.size()>0);
        }
        /// gt packet load function, dataset-specific (default gt loader is not satisfactory)
        virtual cv::Mat getRawGT(size_t nIdx) override final {
            lvDbgExceptionWatch;
            // 'this' is always required here since function name lookup is done during instantiation because of not-fully-specialized class template
            if(this->m_mGTIndexLUT.count(nIdx)) {
                const size_t nGTIdx = this->m_mGTIndexLUT[nIdx];
                if(nGTIdx<this->m_vsGTPaths.size()) {
                    std::vector<std::string> vsTempPaths;
                    lv::GetFilesFromDir(this->m_vsGTPaths[nIdx],vsTempPaths);
                    lvAssert(!vsTempPaths.empty());
                    cv::Mat oTempRefGTImage = cv::imread(vsTempPaths[0],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oTempRefGTImage.empty() && (oTempRefGTImage.size()==cv::Size(481,321) || oTempRefGTImage.size()==cv::Size(321,481)));
                    if(oTempRefGTImage.size()!=this->m_vInputSizes[nGTIdx])
                        cv::resize(oTempRefGTImage,oTempRefGTImage,this->m_vInputSizes[nGTIdx],0,0,cv::INTER_NEAREST);
                    cv::Mat oGTMask(oTempRefGTImage.rows*int(vsTempPaths.size()),oTempRefGTImage.cols,CV_8UC1);
                    for(size_t nGTImageIdx=0; nGTImageIdx<vsTempPaths.size(); ++nGTImageIdx) {
                        cv::Mat oTempGTImage = cv::imread(vsTempPaths[nGTImageIdx],cv::IMREAD_GRAYSCALE);
                        lvAssert(!oTempGTImage.empty() && (oTempGTImage.size()==cv::Size(481,321) || oTempGTImage.size()==cv::Size(321,481)));
                        if(oTempGTImage.size()!=this->m_vInputSizes[nGTIdx])
                            cv::resize(oTempGTImage,oTempGTImage,this->m_vInputSizes[nGTIdx],0,0,cv::INTER_NEAREST);
                        lvAssert(oTempGTImage.size()==oTempRefGTImage.size());
                        oTempGTImage.copyTo(oGTMask(cv::Rect(0,oTempRefGTImage.rows*int(nGTImageIdx),oTempRefGTImage.cols,oTempRefGTImage.rows)));
                    }
                    lvAssert(oGTMask.size()==this->m_vGTSizes[nGTIdx]);
                    return oGTMask;
                }
            }
            return cv::Mat();
        }
    };

    struct BSDS500Counters { // edge detection counters for a single image
        BSDS500Counters(size_t nThresholdsBins) : // always skips zero threshold
                vnIndivTP(nThresholdsBins,0),
                vnIndivTPFN(nThresholdsBins,0),
                vnTotalTP(nThresholdsBins,0),
                vnTotalTPFP(nThresholdsBins,0),
                vnThresholds(lv::linspace<uchar>(0,UCHAR_MAX,nThresholdsBins,false)) {
            lvAssert(nThresholdsBins>0 && nThresholdsBins<=UCHAR_MAX);
        }
        std::vector<uint64_t> vnIndivTP; // one count per threshold
        std::vector<uint64_t> vnIndivTPFN; // one count per threshold
        std::vector<uint64_t> vnTotalTP; // one count per threshold
        std::vector<uint64_t> vnTotalTPFP; // one count per threshold
        std::vector<uchar> vnThresholds; // list of thresholds
        static bool isEqual(const BSDS500Counters& a, const BSDS500Counters& b) {
            return
                    (a.vnThresholds==b.vnThresholds) &&
                    (a.vnIndivTP==b.vnIndivTP) &&
                    (a.vnIndivTPFN==b.vnIndivTPFN) &&
                    (a.vnTotalTP==b.vnTotalTP) &&
                    (a.vnTotalTPFP==b.vnTotalTPFP);
        }
    };

    template<>
    struct MetricsAccumulator_<DatasetEval_BinaryClassifier,Dataset_BSDS500> :
            public IIMetricsAccumulator {
        virtual bool isEqual(const IIMetricsAccumulatorConstPtr& m) const override;
        virtual std::shared_ptr<IIMetricsAccumulator> accumulate(const IIMetricsAccumulatorConstPtr& m) override;
        void accumulate(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& /*oROI*/);
        static cv::Mat getColoredMask(const cv::Mat& oClassif, const cv::Mat& oGT, const cv::Mat& /*oROI*/);
        std::vector<BSDS500Counters> m_voMetricsBase; // one counter block per image
        const size_t m_nThresholdBins;
    protected:
        MetricsAccumulator_(size_t nThresholdBins=DATASETS_BSDS500_EVAL_DEFAULT_THRESH_BINS) :
                m_nThresholdBins(nThresholdBins) {lvAssert(m_nThresholdBins>0 && m_nThresholdBins<=UCHAR_MAX);}
    };

    struct BSDS500Score { // edge detection score for a single threshold
        double dThreshold;
        double dRecall;
        double dPrecision;
        double dFMeasure;
    };

    template<>
    struct MetricsCalculator_<DatasetEval_BinaryClassifier,Dataset_BSDS500> :
            public IIMetricsCalculator {
        virtual IIMetricsCalculatorPtr accumulate(const IIMetricsCalculatorConstPtr& m) override;
        // high-level metrics for an entire image set
        std::vector<BSDS500Score> voBestImageScores; // one score per image (best threshold)
        std::vector<BSDS500Score> voThresholdScores; // one score per threshold (cumul images)
        BSDS500Score oBestScore; // best score for all thresholds
        double dMaxRecall;
        double dMaxPrecision;
        double dMaxFMeasure;
        double dAreaPR;
    protected:
        std::vector<BSDS500Counters> m_voMetricsBase; // one counter block per image (used for image set accumulation only)
        const size_t m_nThresholdBins;
        void updateScores();
        /// default contructor requires a base metrics counters, as otherwise, we may obtain NaN's
        MetricsCalculator_(const IIMetricsAccumulatorConstPtr& m) :
                m_voMetricsBase(dynamic_cast<const MetricsAccumulator_<DatasetEval_BinaryClassifier,Dataset_BSDS500>&>(*m.get()).m_voMetricsBase),
                m_nThresholdBins(dynamic_cast<const MetricsAccumulator_<DatasetEval_BinaryClassifier,Dataset_BSDS500>&>(*m.get()).m_nThresholdBins) {
            updateScores();
        }
    };

    template<>
    struct DataReporter_<DatasetEval_BinaryClassifier,Dataset_BSDS500> :
            public DataReporterWrapper_<DatasetEval_BinaryClassifier,Dataset_BSDS500> {
        /// writes an evaluation report listing custom high-level metrics for current batch(es)
        virtual void writeEvalReport() const override;
    protected:
        /// returns a one-line string listing custom high-level metrics for current batch(es)
        std::string writeInlineEvalReport(size_t nIndentSize) const;
        friend struct DatasetReporter_<DatasetEval_BinaryClassifier,Dataset_BSDS500>;
    };

    template<>
    struct DataEvaluator_<DatasetEval_BinaryClassifier,Dataset_BSDS500,lv::NonParallel> :
            public IDataConsumer_<DatasetEval_BinaryClassifier>,
            public DataReporter_<DatasetEval_BinaryClassifier,Dataset_BSDS500> {
    public:
        /// provides a visual feedback on result quality based on evaluation guidelines
        virtual cv::Mat getColoredMask(const cv::Mat& oClassif, size_t nIdx);
        /// resets internal metrics counters to zero
        virtual void resetMetrics() override;
    protected:
        /// overrides 'getMetricsBase' from IIMetricRetriever for non-group-impl (as always required)
        virtual IIMetricsAccumulatorConstPtr getMetricsBase() const override final;
        /// overrides 'processOutput' from IDataConsumer_ to evaluate the provided output packet
        virtual void processOutput(const cv::Mat& oClassif, size_t nIdx) override;
        /// default constructor; automatically creates an instance of the base metrics accumulator object
        inline DataEvaluator_() : m_pMetricsBase(IIMetricsAccumulator::create<MetricsAccumulator_<DatasetEval_BinaryClassifier,Dataset_BSDS500>>()) {}
        /// contains low-level metric accumulation logic
        std::shared_ptr<MetricsAccumulator_<DatasetEval_BinaryClassifier,Dataset_BSDS500>> m_pMetricsBase;
    };

    template<>
    struct DatasetReporter_<DatasetEval_BinaryClassifier,Dataset_BSDS500> :
            public DatasetReporterWrapper_<DatasetEval_BinaryClassifier,Dataset_BSDS500> {
        /// writes an overall evaluation report listing high-level binary classification metrics
        virtual void writeEvalReport() const override;
    };

} // namespace lv
