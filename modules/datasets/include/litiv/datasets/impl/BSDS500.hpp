
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

#define DEFAULT_BSDS500_EDGE_EVAL_THRESHOLD_BINS 99

enum eBSDS500DatasetGroup {
    eBSDS500Dataset_Training,
    eBSDS500Dataset_Training_Validation,
    eBSDS500Dataset_Training_Validation_Test,
};

template<eDatasetTaskList eDatasetTask, ParallelUtils::eParallelAlgoType eEvalImpl>
struct Dataset_<eDatasetTask,eDataset_BSDS500,eEvalImpl> :
        public IDataset_<eDatasetTask,eDatasetSource_Image,eDataset_BSDS500,getDatasetEval<eDatasetTask,eDataset_BSDS500>(),eEvalImpl> {
    static_assert(eDatasetTask!=eDatasetTask_Registr,"BSDS500 dataset does not support image registration (no image arrays)");
    static_assert(eDatasetTask!=eDatasetTask_ChgDet,"BSDS500 dataset does not support change detection (no data streaming)");
protected: // should still be protected, as creation should always be done via datasets::create
    Dataset_(
            const std::string& sOutputDirName, // output directory (full) path for debug logs, evaluation reports and results archiving (will be created in BSR dataset folder)
            bool bSaveOutput=false, // defines whether results should be archived or not
            bool bUseEvaluator=true, // defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, // defines whether data packets should be 4-byte aligned (useful for GPU upload)
            double dScaleFactor=1.0, // defines the scale factor to use to resize/rescale read packets
            eBSDS500DatasetGroup eType=eBSDS500Dataset_Training // defines which dataset groups to use
    ) :
            IDataset_<eDatasetTask,eDatasetSource_Image,eDataset_BSDS500,getDatasetEval<eDatasetTask,eDataset_BSDS500>(),eEvalImpl>(
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
struct DataProducer_<eDatasetSource_Image,eDataset_BSDS500> :
        public IDataProducer_<eDatasetSource_Image> {
protected:
    virtual void parseData() override final {
        PlatformUtils::GetFilesFromDir(getDataPath(),m_vsInputImagePaths);
        PlatformUtils::FilterFilePaths(m_vsInputImagePaths,{},{".jpg",".png",".bmp"});
        if(m_vsInputImagePaths.empty())
            lvErrorExt("BSDS500 set '%s' did not possess any jpg/png/bmp image file",getName().c_str());
        PlatformUtils::GetSubDirsFromDir(getDatasetInfo()->getDatasetPath()+"/../groundTruth_bdry_images/"+getRelativePath(),m_vsGTImagePaths);
        if(m_vsGTImagePaths.empty())
            lvErrorExt("BSDS500 set '%s' did not possess any groundtruth image folders",getName().c_str());
        else if(m_vsGTImagePaths.size()!=m_vsInputImagePaths.size())
            lvErrorExt("BSDS500 set '%s' input/groundtruth count mismatch",getName().c_str());
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
        m_bIsConstantSize = true;
        m_oMaxSize = cv::Size(481,321);
        m_voImageOrigSizes.clear();
        m_vbImageTransposed.clear();
        m_voImageOrigSizes.reserve(m_vsInputImagePaths.size());
        m_vbImageTransposed.reserve(m_vsInputImagePaths.size());
        for(size_t n=0; n<m_vsInputImagePaths.size(); ++n) {
            cv::Mat oCurrInput = cv::imread(m_vsInputImagePaths[n],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
            lvAssert(!oCurrInput.empty());
            m_voImageOrigSizes.push_back(oCurrInput.size());
            m_vbImageTransposed.push_back(oCurrInput.size()==cv::Size(321,481));
        }
        m_nImageCount = m_vsInputImagePaths.size();
        const double dScale = getDatasetInfo()->getScaleFactor();
        if(dScale!=1.0)
            m_oMaxSize = cv::Size(int(m_oMaxSize.width*dScale),int(m_oMaxSize.height*dScale));
        m_voImageSizes = std::vector<cv::Size>(m_nImageCount,m_oMaxSize);
        CV_Assert(m_nImageCount>0);
    }
    virtual cv::Mat _getGTPacket_impl(size_t nIdx) override final {
        if(m_vsGTImagePaths.size()>nIdx) {
            std::vector<std::string> vsTempPaths;
            PlatformUtils::GetFilesFromDir(m_vsGTImagePaths[nIdx],vsTempPaths);
            CV_Assert(!vsTempPaths.empty());
            cv::Mat oTempRefGTImage = cv::imread(vsTempPaths[0],cv::IMREAD_GRAYSCALE);
            CV_Assert(!oTempRefGTImage.empty());
            CV_Assert(m_voImageOrigSizes[nIdx]==cv::Size() || m_voImageOrigSizes[nIdx]==oTempRefGTImage.size());
            CV_Assert(oTempRefGTImage.size()==cv::Size(481,321) || oTempRefGTImage.size()==cv::Size(321,481));
            m_voImageOrigSizes[nIdx] = oTempRefGTImage.size();
            if(oTempRefGTImage.size()==cv::Size(321,481))
                cv::transpose(oTempRefGTImage,oTempRefGTImage);
            cv::Mat oGTMask(int(oTempRefGTImage.rows*vsTempPaths.size()),oTempRefGTImage.cols,CV_8UC1);
            for(size_t nGTImageIdx=0; nGTImageIdx<vsTempPaths.size(); ++nGTImageIdx) {
                cv::Mat oTempGTImage = cv::imread(vsTempPaths[nGTImageIdx],cv::IMREAD_GRAYSCALE);
                CV_Assert(!oTempGTImage.empty() && (oTempGTImage.size()==cv::Size(481,321) || oTempGTImage.size()==cv::Size(321,481)));
                if(oTempGTImage.size()==cv::Size(321,481))
                    cv::transpose(oTempGTImage,oTempGTImage);
                oTempGTImage.copyTo(cv::Mat(oGTMask,cv::Rect(0,int(oTempGTImage.rows*nGTImageIdx),oTempGTImage.cols,oTempGTImage.rows)));
            }
            if(getPacketSize(nIdx).area()>0 && oGTMask.size()!=getPacketSize(nIdx))
                cv::resize(oGTMask,oGTMask,getPacketSize(nIdx),0,0,cv::INTER_NEAREST);
            return oGTMask;
        }
        return cv::Mat(getPacketSize(nIdx),CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUTOFSCOPE_VAL));
    }
};

template<>
struct DataEvaluator_<eDatasetEval_BinaryClassifier,eDataset_BSDS500> :
        public IDataEvaluator_<eDatasetEval_BinaryClassifier> {
protected:
/*    virtual void writeEdgeMask(const cv::Mat& oEdges, size_t nIdx) const override final {
        CV_Assert(!oEdges.empty());
        cv::Mat oEdgesOutput = oEdges;
        auto pProducer = shared_from_this_cast<const IDataProducer_<eDatasetType_ImageEdgDet,eNotGroup>>(true);
        if(pProducer->getInputImageSize(nIdx)==cv::Size(321,481))
            cv::transpose(oEdgesOutput,oEdgesOutput);
        IDataConsumer_<eDatasetType_ImageEdgDet>::writeEdgeMask(oEdgesOutput,nIdx);
    }
    virtual cv::Mat readEdgeMask(size_t nIdx) const override final {
        cv::Mat oEdgesOutput = IDataConsumer_<eDatasetType_ImageEdgDet>::readEdgeMask(nIdx);
        CV_Assert(!oEdgesOutput.empty());
        auto pProducer = shared_from_this_cast<const IDataProducer_<eDatasetType_ImageEdgDet,eNotGroup>>(true);
        if(pProducer->getInputImageSize(nIdx)==cv::Size(321,481))
            cv::transpose(oEdgesOutput,oEdgesOutput);
        return oEdgesOutput;
    }*/
};