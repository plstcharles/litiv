
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

// note: we should already be in the litiv namespace
#ifndef __LITIV_DATASETS_IMPL_H
#error "This file should never be included directly; use litiv/datasets.hpp instead"
#endif //__LITIV_DATASETS_IMPL_H

enum eBSDS500DatasetGroup {
    eBSDS500Dataset_Training,
    eBSDS500Dataset_Training_Validation,
    eBSDS500Dataset_Training_Validation_Test,
};

template<ParallelUtils::eParallelAlgoType eEvalImpl>
struct Dataset_<eDatasetType_ImageEdgDet,eDataset_ImageEdgDet_BSDS500,eEvalImpl> :
        public IDataset_<eDatasetType_ImageEdgDet,eDataset_ImageEdgDet_BSDS500,eEvalImpl> {
protected: // should still be protected, as creation should always be done via datasets::create
    Dataset_(
            const std::string& sOutputDirName, // output directory (full) path for debug logs, evaluation reports and results archiving (will be created in BSR dataset folder)
            bool bSaveOutput=false, // defines whether results should be archived or not
            bool bUseEvaluator=true, // defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, // defines whether data packets should be 4-byte aligned (useful for GPU upload)
            double dScaleFactor=1.0, // defines the scale factor to use to resize/rescale read packets
            eBSDS500DatasetGroup eType=eBSDS500Dataset_Training // defines which dataset groups to use
    ) :
            IDataset_<eDatasetType_ImageEdgDet,eDataset_ImageEdgDet_BSDS500,eEvalImpl>(
                    "BSDS500",
                    "BSDS500/data/images",
                    std::string(DATASET_ROOT)+"/BSDS500/BSR/"+sOutputDirName+"/",
                    "",
                    ".png",
                    (eType==eBSDS500Dataset_Training)?std::vector<std::string>{"train"}:((eType==eBSDS500Dataset_Training_Validation)?std::vector<std::string>{"train","val"}:std::vector<std::string>{"train","val","test"}),
                    std::vector<std::string>{},
                    std::vector<std::string>{},
                    0,
                    bSaveOutput,
                    bUseEvaluator,
                    bForce4ByteDataAlign,
                    dScaleFactor
            ) {}
};

template<>
struct DataProducer_<eDatasetType_ImageEdgDet,eDataset_ImageEdgDet_BSDS500,eNotGroup> :
        public IDataProducer_<eDatasetType_ImageEdgDet,eNotGroup> {
protected:
    /*
    class Set : public WorkBatch {
    public:
        Set(const std::string& sSetName, const DatasetInfo& oDataset, const std::string& sRelativePath=std::string("./"));
        virtual size_t GetTotalImageCount() const {return m_nTotImageCount;}
        virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
        virtual cv::Mat ReadResult(size_t nIdx);
        virtual void WriteResult(size_t nIdx, const cv::Mat& oResult);
        virtual bool StartPrecaching(bool bUsingGT, size_t nUnused=0);
        bool IsConstantImageSize() const {return m_bIsConstantSize;}
        cv::Size GetMaxImageSize() const {return m_oMaxSize;}
        const eDatasetList m_eDatasetID;
    protected:
        virtual cv::Mat GetInputFromIndex_external(size_t nImageIdx);
        virtual cv::Mat GetGTFromIndex_external(size_t nImageIdx);
    private:
        double m_dExpectedLoad;
        size_t m_nTotImageCount;
        std::vector<std::string> m_vsInputImagePaths;
        std::vector<std::string> m_vsGTImagePaths;
        std::vector<std::string> m_vsOrigImageNames;
        std::vector<cv::Size> m_voOrigImageSizes;
        cv::Size m_oMaxSize;
        bool m_bIsConstantSize;
        Set& operator=(const Set&) = delete;
        Set(const Set&) = delete;
    };
    */
    virtual void parseData() override final {
        /*
        PlatformUtils::GetFilesFromDir(m_sDatasetPath,m_vsInputImagePaths);
        PlatformUtils::FilterFilePaths(m_vsInputImagePaths,{},{".jpg"});
        if(m_vsInputImagePaths.empty())
            throw std::runtime_error(cv::format("Image set '%s' did not possess any image file",sSetName.c_str()));
        m_oMaxSize = cv::Size(481,321);
        m_nTotImageCount = m_vsInputImagePaths.size();
        m_dExpectedLoad = (double)m_oMaxSize.area()*m_nTotImageCount*(int(!m_bForcingGrayscale)+1);
        if(m_eDatasetID==eDataset_BSDS500_edge_train || m_eDatasetID==eDataset_BSDS500_edge_train_valid || m_eDatasetID==eDataset_BSDS500_edge_train_valid_test) {
            PlatformUtils::GetSubDirsFromDir(oDatasetInfo.m_sDatasetRootPath+"/../groundTruth_bdry_images/"+sRelativePath,m_vsGTImagePaths);
            if(m_vsGTImagePaths.empty())
                throw std::runtime_error(cv::format("Image set '%s' did not possess any groundtruth image folders",sSetName.c_str()));
            else if(m_vsGTImagePaths.size()!=m_vsInputImagePaths.size())
                throw std::runtime_error(cv::format("Image set '%s' input/groundtruth count mismatch",sSetName.c_str()));
            // make sure folders are non-empty, and folders & images are similarliy ordered
            std::vector<std::string> vsTempPaths;
            for(size_t nImageIdx=0; nImageIdx<m_vsGTImagePaths.size(); ++nImageIdx) {
                PlatformUtils::GetFilesFromDir(m_vsGTImagePaths[nImageIdx],vsTempPaths);
                CV_Assert(!vsTempPaths.empty());
                const size_t nLastInputSlashPos = m_vsInputImagePaths[nImageIdx].find_last_of("/\\");
                const std::string sInputImageFullName = nLastInputSlashPos==std::string::npos?m_vsInputImagePaths[nImageIdx]:m_vsInputImagePaths[nImageIdx].substr(nLastInputSlashPos+1);
                const size_t nLastGTSlashPos = m_vsGTImagePaths[nImageIdx].find_last_of("/\\");
                CV_Assert(sInputImageFullName.find(nLastGTSlashPos==std::string::npos?m_vsGTImagePaths[nImageIdx]:m_vsGTImagePaths[nImageIdx].substr(nLastGTSlashPos+1))!=std::string::npos);
            }
            m_pEvaluator = std::shared_ptr<EvaluatorBase>(new BSDS500BoundaryEvaluator());
        }
        else { //m_eDatasetID==eDataset_BSDS500_segm_train || m_eDatasetID==eDataset_BSDS500_segm_train_valid || m_eDatasetID==eDataset_BSDS500_segm_train_valid_test
            // current impl cannot parse GT/evaluate (matlab files only)
            CV_Error(0,"missing impl");
        }
        */
        lvError("Missing impl");
    }
    virtual cv::Mat _getGTPacket_impl(size_t nIdx) override final {
        cv::Mat oFrame;
        auto res = m_mTestGTIndexes.find(nFrameIdx);
        if(res!=m_mTestGTIndexes.end()) {
            oFrame = cv::imread(m_vsGTFramePaths[res->second],cv::IMREAD_GRAYSCALE);
            if(oFrame.size()!=m_oSize)
                cv::resize(oFrame,oFrame,m_oSize,0,0,cv::INTER_NEAREST);
        }
        else
            oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL));
        return oFrame;
    }
};
