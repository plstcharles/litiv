
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/datasets.hpp"

namespace lv {

    static const DatasetList Dataset_Middlebury2005_demo = DatasetList(Dataset_Custom+1); // cheat; might cause problems if exposed in multiple external/custom specializations

    // override the dataset evaluation type utility getter function to always set the eval type for this dataset as 'none'
    template<>
    constexpr DatasetEvalList getDatasetEval<DatasetTask_Cosegm,Dataset_Middlebury2005_demo>() {
        // @@@ update sample to use default multiclassif evaluator once implemented in the module?
        return DatasetEval_None;
    }

    // override the dataset evaluation type utility getter function to always set the eval type for this dataset as 'none'
    template<>
    constexpr DatasetEvalList getDatasetEval<DatasetTask_Registr,Dataset_Middlebury2005_demo>() {
        return DatasetEval_None;
    }

    // top-level dataset interface specialization; this allows us to simplify the default constructor
    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_Middlebury2005_demo,eEvalImpl> :
            public IDataset_<eDatasetTask,DatasetSource_ImageArray,Dataset_Middlebury2005_demo,DatasetEval_None,eEvalImpl> {
        static_assert((eDatasetTask==DatasetTask_Cosegm || eDatasetTask==DatasetTask_Registr),"bad task chosen for middlebury stereo dataset");
    protected: // should still be protected, as creation should always be done via datasets::create
        Dataset_(
            const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving
            bool bSaveOutput=false ///< defines whether results should be archived or not
        ) :
                IDataset_<eDatasetTask,DatasetSource_ImageArray,Dataset_Middlebury2005_demo,DatasetEval_None,eEvalImpl>(
                        "middlebury2005",
                        lv::AddDirSlashIfMissing(SAMPLES_DATA_ROOT)+"middlebury2005_dataset_ex/",
                        sOutputDirName,
                        "",
                        "",
                        std::vector<std::string>{"art","dolls"},
                        std::vector<std::string>(),
                        std::vector<std::string>(),
                        bSaveOutput,
                        false,
                        false,
                        1.0
                ) {}
    };

    // dataset group handler specialization; this allows us to completely bypass the notion of 'groups', as there are no categories in middlebury2005
    template<DatasetTaskList eDatasetTask>
    struct DataGroupHandler_<eDatasetTask,DatasetSource_ImageArray,Dataset_Middlebury2005_demo> :
            public DataGroupHandler {
    protected:
        // override the 'parseData()' member from DataGroupHandler to bypass dataset categories (there are none)
        virtual void parseData() override {
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            this->m_vpBatches.clear(); // clear all children if re-parsing
            this->m_bIsBare = true; // in this dataset, work batch groups are always bare (i.e. there are no data 'categories')
            this->m_vpBatches.push_back(this->createWorkBatch(this->getName(),this->getRelativePath())); // push directly with current-level params
        }
    };

    // work batch data producer specialization; this allows us to define what data will be parsed, and how it should be exposed to the end user
    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_ImageArray,Dataset_Middlebury2005_demo> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_ImageArray,Dataset_Middlebury2005_demo> {
        // return the array size of each input packet
        virtual size_t getInputStreamCount() const override final {
            return 2; // this dataset always exposes two input images simultaneously (i.e. one per stereo head)
        }
        // return the array size of each gt packet
        virtual size_t getGTStreamCount() const override final {
            return 2; // this dataset always exposes two gt disparity maps simultaneously (i.e. one per stereo head)
        }
        // OPTIONAL: provide a user-friendly name for input streams (in this case, it specifies the left/right stereo head)
        virtual std::string getInputStreamName(size_t nStreamIdx) const override final {
            return ((nStreamIdx==0)?"Left input":(nStreamIdx==1)?"Right input":"UNKNOWN");
        }
        // OPTIONAL: provide a user-friendly name for gt streams (in this case, it specifies the left/right stereo head)
        virtual std::string getGTStreamName(size_t nStreamIdx) const override final {
            return ((nStreamIdx==0)?"Left gt":(nStreamIdx==1)?"Right gt":"UNKNOWN");
        }
    protected:
        // implement the 'parseData()' abstract member from IDataHandler with our own parsing routine
        virtual void parseData() override final {
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            // we have to fill the members declared in 'IDataProducer_<DatasetSource_ImageArray>', which is the base data producer for this dataset
            this->m_vvsInputPaths.push_back(std::vector<std::string>{this->getDataPath()+"view1.png",this->getDataPath()+"view5.png"}); // batch contains only one 'packet', i.e. one stereo image array containing two images
            const cv::Mat oInput_L = cv::imread(this->m_vvsInputPaths[0][0]); // load 'left' image from the stereo pair
            const cv::Mat oInput_R = cv::imread(this->m_vvsInputPaths[0][1]); // load 'right' image from the stereo pair
            lvAssert(!oInput_L.empty() && !oInput_R.empty() && oInput_L.size()==oInput_R.size()); // make sure the data fits expectations
            this->m_vvsGTPaths.push_back(std::vector<std::string>{this->getDataPath()+"disp1.png",this->getDataPath()+"disp5.png"}); // do the same for GT data (one packet = two disparity maps, one for each stereo head)
            const cv::Mat oGT_L = cv::imread(this->m_vvsGTPaths[0][0],cv::IMREAD_GRAYSCALE); // load 'left' disparity map
            const cv::Mat oGT_R = cv::imread(this->m_vvsGTPaths[0][1],cv::IMREAD_GRAYSCALE); // load 'right' disparity map
            lvAssert(!oGT_L.empty() && !oGT_R.empty() && oGT_L.size()==oGT_R.size() && oGT_L.size()==oInput_L.size());
            this->m_bIsInputConstantSize = this->m_bIsGTConstantSize = true; // all packets have the same size (there is only one packet anyway)
            this->m_vvInputSizes = this->m_vvGTSizes = std::vector<std::vector<cv::Size>>{{oInput_L.size(),oInput_R.size()}}; // all packets have the same size
            this->m_oInputMaxSize = this->m_oGTMaxSize = oInput_L.size(); // max input frame size is already known (there is only one packet size anyway)
            this->m_mGTIndexLUT[0] = 0; // gt packet with index #0 is associated with input packet with index #0
        }
    };

} // namespace lv
