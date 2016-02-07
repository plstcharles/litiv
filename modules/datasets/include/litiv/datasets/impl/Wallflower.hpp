
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

template<ParallelUtils::eParallelAlgoType eEvalImpl>
struct Dataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_Wallflower,eEvalImpl> :
        public IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_Wallflower,eEvalImpl> {
protected: // should still be protected, as creation should always be done via datasets::create
    Dataset_(
            const std::string& sOutputDirName, // output directory (full) path for debug logs, evaluation reports and results archiving (will be created in Wallflower dataset folder)
            bool bSaveOutput=false, // defines whether results should be archived or not
            bool bUseEvaluator=true, // defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, // defines whether data packets should be 4-byte aligned (useful for GPU upload)
            double dScaleFactor=1.0 // defines the scale factor to use to resize/rescale read packets
    ) :
            IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_Wallflower,eEvalImpl>(
                    "Wallflower",
                    "Wallflower/dataset",
                    std::string(DATASET_ROOT)+"/Wallflower/"+sOutputDirName+"/",
                    "bin",
                    ".png",
                    std::vector<std::string>{""},
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
struct DataProducer_<eDatasetType_VideoSegm,eDataset_VideoSegm_Wallflower,eNotGroup> :
        public IDataProducer_<eDatasetType_VideoSegm,eNotGroup> {
protected:
    virtual void parseData() override final {
        // @@@@ untested since 2016/01 refactoring
        std::vector<std::string> vsImgPaths;
        PlatformUtils::GetFilesFromDir(getDataPath(),vsImgPaths);
        bool bFoundScript=false, bFoundGTFile=false;
        const std::string sGTFilePrefix("hand_segmented_");
        const size_t nInputFileNbDecimals = 5;
        const std::string sInputFileSuffix(".bmp");
        for(auto iter=vsImgPaths.begin(); iter!=vsImgPaths.end(); ++iter) {
            if(*iter==getDataPath()+"/script.txt")
                bFoundScript = true;
            else if(iter->find(sGTFilePrefix)!=std::string::npos) {
                m_mTestGTIndexes.insert(std::pair<size_t,size_t>(atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str()),m_vsGTFramePaths.size()));
                m_vsGTFramePaths.push_back(*iter);
                bFoundGTFile = true;
            }
            else {
                if(iter->find(sInputFileSuffix)!=iter->size()-sInputFileSuffix.size())
                    lvErrorExt("Wallflower sequence '%s' contained an unknown file ('%s')",getName().c_str(),iter->c_str());
                m_vsInputFramePaths.push_back(*iter);
            }
        }
        if(!bFoundGTFile || !bFoundScript || m_vsInputFramePaths.empty() || m_vsGTFramePaths.size()!=1)
            lvErrorExt("Wallflower sequence '%s' did not possess the required groundtruth and input files",getName().c_str());
        cv::Mat oTempImg = cv::imread(m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            lvErrorExt("Wallflower sequence '%s' did not possess a valid GT file",getName().c_str());
        m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
        m_oSize = oTempImg.size();
        m_nFrameCount = m_vsInputFramePaths.size();
        CV_Assert(m_nFrameCount>0);
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
