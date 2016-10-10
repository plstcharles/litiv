
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

#include "litiv/datasets.hpp" // for parsers only, not truly required here

namespace lv {

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_Wallflower,eEvalImpl> :
            public IDataset_<eDatasetTask,DatasetSource_Video,Dataset_Wallflower,lv::getDatasetEval<eDatasetTask,Dataset_Wallflower>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via datasets::create
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in Wallflower dataset folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bForce4ByteDataAlign=false, ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
                double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        ) :
                IDataset_<eDatasetTask,DatasetSource_Video,Dataset_Wallflower,lv::getDatasetEval<eDatasetTask,Dataset_Wallflower>(),eEvalImpl>(
                        "Wallflower",
                        lv::datasets::getDatasetsRootPath()+"Wallflower/dataset/",
                        lv::datasets::getDatasetsRootPath()+"Wallflower/"+lv::AddDirSlashIfMissing(sOutputDirName),
                        "bin",
                        ".png",
                        getWorkBatchDirNames(),
                        std::vector<std::string>(),
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        bForce4ByteDataAlign,
                        dScaleFactor
                ) {}
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames() {
            static const std::vector<std::string> s_vsWorkBatchDirs = {""}; // 'current' directory should contain the videos
            return s_vsWorkBatchDirs;
        }
    };

    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_Video,Dataset_Wallflower> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_Video,Dataset_Wallflower> {
    protected:
        virtual void parseData() override final {
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            // @@@@ untested since 2016/01 refactoring
            std::vector<std::string> vsImgPaths;
            lv::GetFilesFromDir(this->getDataPath(),vsImgPaths);
            bool bFoundScript=false, bFoundGTFile=false;
            const std::string sGTFilePrefix("hand_segmented_");
            const size_t nInputFileNbDecimals = 5;
            const std::string sInputFileSuffix(".bmp");
            this->m_mGTIndexLUT.clear();
            for(auto iter=vsImgPaths.begin(); iter!=vsImgPaths.end(); ++iter) {
                if(*iter==this->getDataPath()+"script.txt")
                    bFoundScript = true;
                else if(iter->find(sGTFilePrefix)!=std::string::npos) {
                    this->m_mGTIndexLUT[(size_t)atoi(iter->substr(iter->find(sGTFilePrefix)+sGTFilePrefix.size(),nInputFileNbDecimals).c_str())] = this->m_vsGTPaths.size();
                    this->m_vsGTPaths.push_back(*iter);
                    bFoundGTFile = true;
                }
                else {
                    if(iter->find(sInputFileSuffix)!=iter->size()-sInputFileSuffix.size())
                        lvError_("Wallflower sequence '%s' contained an unknown file ('%s')",this->getName().c_str(),iter->c_str());
                    this->m_vsInputPaths.push_back(*iter);
                }
            }
            if(!bFoundGTFile || !bFoundScript || this->m_vsInputPaths.empty() || this->m_vsGTPaths.size()!=1)
                lvError_("Wallflower sequence '%s' did not possess the required groundtruth and input files",this->getName().c_str());
            cv::Mat oTempImg = cv::imread(this->m_vsGTPaths[0]);
            if(oTempImg.empty())
                lvError_("Wallflower sequence '%s' did not possess a valid GT file",this->getName().c_str());
            this->m_oInputROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
            const double dScale = this->getScaleFactor();
            if(dScale!=1.0)
                cv::resize(this->m_oInputROI,this->m_oInputROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
            this->m_oGTROI = this->m_oInputROI;
            this->m_oInputSize = this->m_oGTSize = this->m_oInputROI.size();
            this->m_nFrameCount = this->m_vsInputPaths.size();
            lvAssert_(this->m_nFrameCount>0,"could not find any input frames");
        }
    };

} // namespace lv
