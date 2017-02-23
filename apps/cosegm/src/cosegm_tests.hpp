
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

#include <opencv2/imgcodecs.hpp>
#include "litiv/utils.hpp"
#include "litiv/datasets.hpp"

namespace lv {

    static const DatasetList Dataset_CosegmTests = DatasetList(Dataset_Custom+1); // cheat; might cause problems if exposed in multiple external/custom specializations

    struct ICosegmTestDataset {
        virtual int isUsingHalfGT() const = 0;
        virtual std::string getFeaturesDirName() const = 0;
    };

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_CosegmTests,eEvalImpl> :
            public ICosegmTestDataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl> {
    protected:
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in VAP trimodal dataset results folder)
                const std::string& sFeaturesDirName, ///< defines the name of the directory where precomputed features may be found (in the sequence directory)
                int nUseHalfGT=1, ///< defines whether the input masks given should be half gt or not (useful for training) --- 1=thermalgt, -1=rgbgt, 0=nogt
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl>(
                        "cosegm_tests",
                        lv::datasets::getRootPath()+"litiv/cosegm_tests/",
                        lv::datasets::getRootPath()+"litiv/cosegm_tests/results/"+lv::addDirSlashIfMissing(sOutputDirName),
                        std::vector<std::string>{"test01"},
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        false,
                        dScaleFactor
                ),m_nUsingHalfGT(nUseHalfGT),m_sFeaturesDirName(sFeaturesDirName) {}
        virtual int isUsingHalfGT() const override final {return m_nUsingHalfGT;}
        virtual std::string getFeaturesDirName() const override final {return m_sFeaturesDirName;}
    protected:
        const int m_nUsingHalfGT;
        const std::string m_sFeaturesDirName;
    };

    template<DatasetTaskList eDatasetTask>
    struct DataGroupHandler_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests> :
            public DataGroupHandler {
    protected:
        virtual void parseData() override {
            this->m_vpBatches.clear();
            this->m_bIsBare = true;
            this->m_vpBatches.push_back(this->createWorkBatch(this->getName(),this->getRelativePath()));
        }
    };

    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests> {

        virtual size_t getInputStreamCount() const override final {
            return 4; // 2x input images, 2x approx fg masks
        }

        virtual size_t getGTStreamCount() const override final {
            return 2; // 2x fg masks
        }

        virtual std::string getFeaturesName(size_t nPacketIdx) const override final {
            std::array<char,32> acBuffer;
            snprintf(acBuffer.data(),acBuffer.size(),nPacketIdx<size_t(1e7)?"%06zu":"%09zu",nPacketIdx);
            const ICosegmTestDataset& oDataset = dynamic_cast<const ICosegmTestDataset&>(*this->getRoot());
            return lv::addDirSlashIfMissing(oDataset.getFeaturesDirName())+std::string(acBuffer.data());
        }

    protected:

        virtual void parseData() override final {
            const ICosegmTestDataset& oDataset = dynamic_cast<const ICosegmTestDataset&>(*this->getRoot());
            this->m_vvsInputPaths.clear();
            this->m_vvsGTPaths.clear();
            this->m_vInputInfos.resize(getInputStreamCount());
            this->m_vGTInfos.resize(getGTStreamCount());
            int nCurrIdx = 0;
            while(true) {
                const std::vector<std::string> vCurrInputPaths = {
                    cv::format("%simg%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format((oDataset.isUsingHalfGT()==-1?"%sgt%05da.png":"%smask%05da.png"),this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%simg%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format((oDataset.isUsingHalfGT()==1?"%sgt%05db.png":"%smask%05db.png"),this->getDataPath().c_str(),nCurrIdx),
                };
                const cv::Mat oInput0 = cv::imread(vCurrInputPaths[0],cv::IMREAD_COLOR);
                const cv::Mat oMask0 = cv::imread(vCurrInputPaths[1],cv::IMREAD_GRAYSCALE);
                const cv::Mat oInput1 = cv::imread(vCurrInputPaths[2],cv::IMREAD_GRAYSCALE);
                const cv::Mat oMask1 = cv::imread(vCurrInputPaths[3],cv::IMREAD_GRAYSCALE);
                if(oInput0.empty() || oMask0.empty() || oInput1.empty() || oMask1.empty())
                    break;
                lvAssert(oInput0.size()==oInput1.size() && oMask0.size()==oMask1.size() && oInput0.size()==oMask0.size());
                this->m_vvsInputPaths.push_back(vCurrInputPaths);
                if(this->m_vvsInputPaths.size()==size_t(1)) {
                    this->m_vInputInfos[0] = lv::MatInfo(oInput0);
                    this->m_vInputInfos[1] = lv::MatInfo(oMask0);
                    this->m_vInputInfos[2] = lv::MatInfo(oInput1);
                    this->m_vInputInfos[3] = lv::MatInfo(oMask1);
                }
                else {
                    lvAssert(lv::MatInfo(oInput0)==this->m_vInputInfos[0]);
                    lvAssert(lv::MatInfo(oMask0)==this->m_vInputInfos[1]);
                    lvAssert(lv::MatInfo(oInput1)==this->m_vInputInfos[2]);
                    lvAssert(lv::MatInfo(oMask1)==this->m_vInputInfos[3]);
                }
                const std::vector<std::string> vCurrGTPaths = {
                    cv::format("%sgt%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%sgt%05db.png",this->getDataPath().c_str(),nCurrIdx),
                };
                const cv::Mat oGT0 = cv::imread(vCurrGTPaths[0],cv::IMREAD_GRAYSCALE);
                const cv::Mat oGT1 = cv::imread(vCurrGTPaths[1],cv::IMREAD_GRAYSCALE);
                lvAssert(!oGT0.empty() && !oGT1.empty() && oGT0.size()==oGT1.size() && oInput0.size()==oGT0.size());
                this->m_vvsGTPaths.push_back(vCurrGTPaths);
                if(this->m_vvsGTPaths.size()==size_t(1)) {
                    this->m_vGTInfos[0] = lv::MatInfo(oGT0);
                    this->m_vGTInfos[1] = lv::MatInfo(oGT1);
                }
                else {
                    lvAssert(lv::MatInfo(oGT0)==this->m_vGTInfos[0]);
                    lvAssert(lv::MatInfo(oGT1)==this->m_vGTInfos[1]);
                }
                this->m_mGTIndexLUT[nCurrIdx] = nCurrIdx;
                ++nCurrIdx;
            }
            lvAssert(this->m_vvsInputPaths.size()>0 && this->m_vvsInputPaths[0].size()>0);
            const cv::Size oOrigSize = this->m_vInputInfos[0].size();
            lvAssert(oOrigSize.area()>0);
            const double dScale = this->getScaleFactor();
            std::array<cv::Mat,2> aROIs = {
                cv::imread(this->getDataPath()+"roi0.png",cv::IMREAD_GRAYSCALE),
                cv::imread(this->getDataPath()+"roi1.png",cv::IMREAD_GRAYSCALE)
            };
            this->m_vInputROIs.resize(4);
            this->m_vGTROIs.resize(2);
            for(size_t a=0; a<2; ++a) {
                if(aROIs[a].empty())
                    aROIs[a] = cv::Mat(oOrigSize,CV_8UC1,cv::Scalar_<uchar>(255));
                else
                    lvAssert(aROIs[a].size()==oOrigSize);
                if(dScale!=1.0)
                    cv::resize(aROIs[a],aROIs[a],cv::Size(),dScale,dScale,cv::INTER_NEAREST);
                this->m_vInputROIs[2*a] = aROIs[a].clone();
                this->m_vInputROIs[2*a+1] = aROIs[a].clone();
                this->m_vGTROIs[a] = aROIs[a].clone();
            }
            this->m_vInputInfos[0] = lv::MatInfo(aROIs[0].size(),CV_8UC3);
            this->m_vInputInfos[1] = lv::MatInfo(aROIs[0].size(),CV_8UC1);
            this->m_vInputInfos[2] = lv::MatInfo(aROIs[1].size(),CV_8UC1);
            this->m_vInputInfos[3] = lv::MatInfo(aROIs[1].size(),CV_8UC1);
            this->m_vGTInfos[0] = lv::MatInfo(aROIs[0].size(),CV_8UC1);
            this->m_vGTInfos[1] = lv::MatInfo(aROIs[1].size(),CV_8UC1);
        }

    };

} // namespace lv
