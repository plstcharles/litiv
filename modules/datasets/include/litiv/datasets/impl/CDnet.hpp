
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
    struct Dataset_<eDatasetTask,Dataset_CDnet,eEvalImpl> :
            public IDataset_<eDatasetTask,DatasetSource_Video,Dataset_CDnet,lv::getDatasetEval<eDatasetTask,Dataset_CDnet>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via datasets::create
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in CDnet dataset folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bForce4ByteDataAlign=false, ///< defines whether data packets should be 4-byte aligned
                double dScaleFactor=1.0, ///< defines the scale factor to use to resize/rescale read packets
                bool b2014=true ///< defines whether to use the 2012 or 2014 version of the dataset (each should have its own folder in dataset root)
        ) :
                IDataset_<eDatasetTask,DatasetSource_Video,Dataset_CDnet,lv::getDatasetEval<eDatasetTask,Dataset_CDnet>(),eEvalImpl>(
                        b2014?"CDnet 2014":"CDnet 2012",
                        lv::datasets::getRootPath()+std::string(b2014?"CDNet2014/dataset/":"CDNet/dataset/"),
                        lv::datasets::getRootPath()+std::string(b2014?"CDNet2014/":"CDNet/")+lv::addDirSlashIfMissing(sOutputDirName),
                        getWorkBatchDirNames(b2014),
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        bForce4ByteDataAlign,
                        dScaleFactor
                ) {}
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames(bool b2014=true) {
            static const std::vector<std::string> s_vsWorkBatchDirs_2014 = {"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"};
            static const std::vector<std::string> s_vsWorkBatchDirs_2012 = {"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"};
            //static const std::vector<std::string> tmp = std::vector<std::string>{"baseline_highway"};
            //return tmp;
            if(b2014)
                return s_vsWorkBatchDirs_2014;
            else
                return s_vsWorkBatchDirs_2012;
        }
    };

    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_Video,Dataset_CDnet> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_Video,Dataset_CDnet> {
    protected:
        virtual void parseData() override final {
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            const bool bIsGrayscale = this->getName().find("thermal")!=std::string::npos || this->getName().find("turbulence")!=std::string::npos;
            const std::vector<std::string> vsSubDirs = lv::getSubDirsFromDir(this->getDataPath());
            auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"groundtruth");
            auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"input");
            if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
                lvError_("CDnet sequence '%s' did not possess the required groundtruth and input directories",this->getName().c_str());
            this->m_vsInputPaths = lv::getFilesFromDir(*inputDir);
            this->m_vsGTPaths = lv::getFilesFromDir(*gtDir);
            this->m_nFrameCount = this->m_vsInputPaths.size();
            lvAssert_(this->m_nFrameCount>0,"could not find any input frames");
            if(this->m_vsGTPaths.size()!=this->m_vsInputPaths.size())
                lvError_("CDnet sequence '%s' did not possess same amount of GT & input frames",this->getName().c_str());
            cv::Mat oROI = cv::imread(this->getDataPath()+"ROI.bmp",cv::IMREAD_GRAYSCALE);
            cv::Mat oTempROI = cv::imread(this->getDataPath()+"ROI.jpg");
            if(oROI.empty() || oTempROI.empty())
                lvError_("CDnet sequence '%s' did not possess ROI.bmp/ROI.jpg files",this->getName().c_str());
            if(oROI.size()!=oTempROI.size()) {
                std::cerr << "CDnet sequence '" << this->getName().c_str() << "' ROI images size mismatch; will keep smallest overlap." << std::endl;
                oROI = oROI(cv::Rect(0,0,std::min(oROI.cols,oTempROI.cols),std::min(oROI.rows,oTempROI.rows))).clone();
            }
            this->m_oInputROI = oROI>0;
            const double dScale = this->getScaleFactor();
            if(dScale!=1.0)
                cv::resize(this->m_oInputROI,this->m_oInputROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
            this->m_oGTROI = this->m_oInputROI;
            this->m_oInputInfo = lv::MatInfo{this->m_oInputROI.size(),(bIsGrayscale?CV_8UC1:this->is4ByteAligned()?CV_8UC4:CV_8UC3)};
            this->m_oGTInfo = lv::MatInfo{this->m_oInputROI.size(),CV_8UC1};
            this->m_mGTIndexLUT.clear();
            for(size_t i=0; i<this->m_nFrameCount; ++i)
                this->m_mGTIndexLUT[i] = i; // direct gt path index to frame index mapping
        }
    };

    template<DatasetEvalList eDatasetEval, lv::ParallelAlgoType eEvalImpl>
    struct DataEvaluator_<eDatasetEval,Dataset_CDnet,eEvalImpl> :
            public DataEvaluatorWrapper_<eDatasetEval,Dataset_CDnet,eEvalImpl> {
        // standardized names needed here since we might evaluate externally using matlab/python scripts
        virtual std::string getOutputName(size_t nPacketIdx) const override final {
            std::array<char,32> acBuffer;
            snprintf(acBuffer.data(),acBuffer.size(),"bin%06zu",nPacketIdx+1);
            return std::string(acBuffer.data());
        }
    };

} // namespace lv
