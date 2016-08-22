
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

template<eDatasetTaskList eDatasetTask, lv::eParallelAlgoType eEvalImpl>
struct Dataset_<eDatasetTask,eDataset_CDnet,eEvalImpl> :
        public IDataset_<eDatasetTask,eDatasetSource_Video,eDataset_CDnet,getDatasetEval<eDatasetTask,eDataset_CDnet>(),eEvalImpl> {
    static_assert(eDatasetTask!=eDatasetTask_Registr,"CDnet dataset does not support image registration (no image arrays)");
protected: // should still be protected, as creation should always be done via datasets::create
    Dataset_(
            const std::string& sOutputDirName, ///< output directory (full) path for debug logs, evaluation reports and results archiving (will be created in CDnet dataset folder)
            bool bSaveOutput=false, ///< defines whether results should be archived or not
            bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
            double dScaleFactor=1.0, ///< defines the scale factor to use to resize/rescale read packets
            bool b2014=true ///< defines whether to use the 2012 or 2014 version of the dataset (each should have its own folder in dataset root)
    ) :
            IDataset_<eDatasetTask,eDatasetSource_Video,eDataset_CDnet,getDatasetEval<eDatasetTask,eDataset_CDnet>(),eEvalImpl>(
                    b2014?"CDnet 2014":"CDnet 2012",
                    lv::AddDirSlashIfMissing(EXTERNAL_DATA_ROOT)+std::string(b2014?"CDNet2014/dataset/":"CDNet/dataset/"),
                    lv::AddDirSlashIfMissing(EXTERNAL_DATA_ROOT)+std::string(b2014?"CDNet2014/":"CDNet/")+lv::AddDirSlashIfMissing(sOutputDirName),
                    "bin",
                    ".png",
                    /*std::vector<std::string>{"baseline_highway_cut2"},*/b2014?std::vector<std::string>{"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"}:std::vector<std::string>{"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"},
                    std::vector<std::string>{},
                    b2014?std::vector<std::string>{"thermal","turbulence"}:std::vector<std::string>{"thermal"},
                    1,
                    bSaveOutput,
                    bUseEvaluator,
                    bForce4ByteDataAlign,
                    dScaleFactor
            ) {}
};

template<eDatasetTaskList eDatasetTask>
struct DataProducer_<eDatasetTask,eDatasetSource_Video,eDataset_CDnet> :
        public DataProducer_c<eDatasetTask,eDatasetSource_Video> {
protected:
    virtual void parseData() override final {
        // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
        std::vector<std::string> vsSubDirs;
        lv::GetSubDirsFromDir(this->getDataPath(),vsSubDirs);
        auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"groundtruth");
        auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"input");
        if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
            lvErrorExt("CDnet sequence '%s' did not possess the required groundtruth and input directories",this->getName().c_str());
        lv::GetFilesFromDir(*inputDir,this->m_vsInputPaths);
        lv::GetFilesFromDir(*gtDir,this->m_vsGTPaths);
        if(this->m_vsGTPaths.size()!=this->m_vsInputPaths.size())
            lvErrorExt("CDnet sequence '%s' did not possess same amount of GT & input frames",this->getName().c_str());
        this->m_oROI = cv::imread(lv::AddDirSlashIfMissing(this->getDataPath())+"ROI.bmp",cv::IMREAD_GRAYSCALE);
        cv::Mat oTempROI = cv::imread(lv::AddDirSlashIfMissing(this->getDataPath())+"ROI.jpg");
        if(this->m_oROI.empty() || oTempROI.empty())
            lvErrorExt("CDnet sequence '%s' did not possess ROI.bmp/ROI.jpg files",this->getName().c_str());
        if(this->m_oROI.size()!=oTempROI.size()) {
            std::cerr << "CDnet sequence '" << this->getName().c_str() << "' ROI images size mismatch; will keep smallest overlap." << std::endl;
            this->m_oROI = this->m_oROI(cv::Rect(0,0,std::min(this->m_oROI.cols,oTempROI.cols),std::min(this->m_oROI.rows,oTempROI.rows))).clone();
        }
        this->m_oROI = this->m_oROI>0;
        this->m_oOrigSize = this->m_oROI.size();
        const double dScale = this->getDatasetInfo()->getScaleFactor();
        if(dScale!=1.0)
            cv::resize(this->m_oROI,this->m_oROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
        this->m_oSize = this->m_oROI.size();
        this->m_nFrameCount = this->m_vsInputPaths.size();
        CV_Assert(this->m_nFrameCount>0);
        this->m_mGTIndexLUT.clear();
        for(size_t i=0; i<this->m_nFrameCount; ++i)
            this->m_mGTIndexLUT[i] = i; // direct gt path index to frame index mapping
    }
};
