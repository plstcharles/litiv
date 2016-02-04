
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

template<>
struct Dataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_PETS2001D3TC1> : public IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_PETS2001D3TC1> {
protected: // should still be protected, as creation should always be done via datasets::create
    Dataset_(
            const std::string& sOutputDirName, // output directory (full) path for debug logs, evaluation reports and results archiving (will be created in PETS dataset folder)
            bool bSaveOutput=false, // defines whether results should be archived or not
            bool bUseEvaluator=true, // defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, // defines whether data packets should be 4-byte aligned (useful for GPU upload)
            double dScaleFactor=1.0 // defines the scale factor to use to resize/rescale read packets
    ) :
            IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_PETS2001D3TC1>(
                    "PETS2001 Dataset#3",
                    "PETS2001/DATASET3",
                    std::string(DATASET_ROOT)+"/PETS2001/DATASET3/"+sOutputDirName+"/",
                    "bin",
                    ".png",
                    std::vector<std::string>{"TESTING"},
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
struct DataProducer_<eDatasetType_VideoSegm, eDataset_VideoSegm_PETS2001D3TC1, eNotGroup> :
        public IDataProducer_<eDatasetType_VideoSegm,eNotGroup> {
        
    virtual void parseData() override final {
        /*
        std::vector<std::string> vsVideoSeqPaths;
        PlatformUtils::GetFilesFromDir(m_sDatasetPath,vsVideoSeqPaths);
        if(vsVideoSeqPaths.size()!=1)
            throw std::runtime_error(cv::format("Sequence '%s': bad subdirectory for PETS2001 parsing (should contain only one video sequence file)",sSeqName.c_str()));
        std::vector<std::string> vsGTSubdirPaths;
        PlatformUtils::GetSubDirsFromDir(m_sDatasetPath,vsGTSubdirPaths);
        if(vsGTSubdirPaths.size()!=1)
            throw std::runtime_error(cv::format("Sequence '%s': bad subdirectory for PETS2001 parsing (should contain only one GT subdir)",sSeqName.c_str()));
        m_voVideoReader.open(vsVideoSeqPaths[0]);
        if(!m_voVideoReader.isOpened())
            throw std::runtime_error(cv::format("Sequence '%s': video file could not be opened",sSeqName.c_str()));
        PlatformUtils::GetFilesFromDir(vsGTSubdirPaths[0],m_vsGTFramePaths);
        if(m_vsGTFramePaths.empty())
            throw std::runtime_error(cv::format("Sequence '%s': did not possess any valid GT frames",m_sDatasetPath.c_str()));
        const std::string sGTFilePrefix("image_");
        const size_t nInputFileNbDecimals = 4;
        for(auto iter=m_vsGTFramePaths.begin(); iter!=m_vsGTFramePaths.end(); ++iter)
            m_mTestGTIndexes.insert(std::pair<size_t,size_t>((size_t)atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),iter-m_vsGTFramePaths.begin()));
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            throw std::runtime_error(cv::format("Sequence '%s': did not possess valid GT file(s)",sSeqName.c_str()));
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
        m_oSize = oTempImg.size();
        m_nNextExpectedVideoReaderFrameIdx = 0;
        m_nTotFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
        CV_Assert(m_nTotFrameCount>0);
        m_dExpectedLoad = (double)m_oSize.height*m_oSize.width*m_nTotFrameCount*(int(!m_bForcingGrayscale)+1);
        m_pEvaluator = std::shared_ptr<EvaluatorBase>(new BinarySegmEvaluator("PETS2001_EVAL"));
        */
        lvError("Missing impl");
    }
    
    virtual cv::Mat _getGTPacket_impl(size_t nIdx) override final {
        cv::Mat oFrame;
        auto res = m_mTestGTIndexes.find(nIdx);
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
