
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

template<eDatasetTaskList eDatasetTask, ParallelUtils::eParallelAlgoType eEvalImpl>
struct Dataset_<eDatasetTask,eDataset_Wallflower,eEvalImpl> :
        public IDataset_<eDatasetTask,eDatasetSource_Video,eDataset_Wallflower,getDatasetEval<eDatasetTask,eDataset_Wallflower>(),eEvalImpl> {
    static_assert(eDatasetTask!=eDatasetTask_Registr,"Wallflower dataset does not support image registration (no image arrays)");
protected: // should still be protected, as creation should always be done via datasets::create
    Dataset_(
            const std::string& sOutputDirName, //!< output directory (full) path for debug logs, evaluation reports and results archiving (will be created in Wallflower dataset folder)
            bool bSaveOutput=false, //!< defines whether results should be archived or not
            bool bUseEvaluator=true, //!< defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, //!< defines whether data packets should be 4-byte aligned (useful for GPU upload)
            double dScaleFactor=1.0 //!< defines the scale factor to use to resize/rescale read packets
    ) :
            IDataset_<eDatasetTask,eDatasetSource_Video,eDataset_Wallflower,getDatasetEval<eDatasetTask,eDataset_Wallflower>(),eEvalImpl>(
                    "Wallflower",
                    PlatformUtils::AddDirSlashIfMissing(DATASET_ROOT)+"Wallflower/dataset/",
                    PlatformUtils::AddDirSlashIfMissing(DATASET_ROOT)+"Wallflower/"+PlatformUtils::AddDirSlashIfMissing(sOutputDirName),
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

template<eDatasetTaskList eDatasetTask>
struct DataProducer_<eDatasetTask,eDatasetSource_Video,eDataset_Wallflower> :
        public DataProducer_c<eDatasetTask,eDatasetSource_Video> {
protected:
    virtual void parseData() override final {
        // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
        // @@@@ untested since 2016/01 refactoring
        std::vector<std::string> vsImgPaths;
        PlatformUtils::GetFilesFromDir(this->getDataPath(),vsImgPaths);
        bool bFoundScript=false, bFoundGTFile=false;
        const std::string sGTFilePrefix("hand_segmented_");
        const size_t nInputFileNbDecimals = 5;
        const std::string sInputFileSuffix(".bmp");
        this->m_mGTIndexLUT.clear();
        for(auto iter=vsImgPaths.begin(); iter!=vsImgPaths.end(); ++iter) {
            if(*iter==this->getDataPath()+"/script.txt")
                bFoundScript = true;
            else if(iter->find(sGTFilePrefix)!=std::string::npos) {
                this->m_mGTIndexLUT[(size_t)atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str())] = this->m_vsGTFramePaths.size();
                this->m_vsGTFramePaths.push_back(*iter);
                bFoundGTFile = true;
            }
            else {
                if(iter->find(sInputFileSuffix)!=iter->size()-sInputFileSuffix.size())
                    lvErrorExt("Wallflower sequence '%s' contained an unknown file ('%s')",this->getName().c_str(),iter->c_str());
                this->m_vsInputPaths.push_back(*iter);
            }
        }
        if(!bFoundGTFile || !bFoundScript || this->m_vsInputPaths.empty() || this->m_vsGTFramePaths.size()!=1)
            lvErrorExt("Wallflower sequence '%s' did not possess the required groundtruth and input files",this->getName().c_str());
        cv::Mat oTempImg = cv::imread(this->m_vsGTFramePaths[0]);
        if(oTempImg.empty())
            lvErrorExt("Wallflower sequence '%s' did not possess a valid GT file",this->getName().c_str());
        this->m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
        this->m_oOrigSize = this->m_oROI.size();
        const double dScale = this->getDatasetInfo()->getScaleFactor();
        if(dScale!=1.0)
            cv::resize(this->m_oROI,this->m_oROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
        this->m_oSize = this->m_oROI.size();
        this->m_nFrameCount = this->m_vsInputPaths.size();
        CV_Assert(this->m_nFrameCount>0);
    }
};
