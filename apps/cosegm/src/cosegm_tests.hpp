
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
        virtual bool isUsingHalfGT() const = 0;
    };

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_CosegmTests,eEvalImpl> :
            public ICosegmTestDataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl> {
    protected:
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in VAP trimodal dataset results folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                double dScaleFactor=1.0, ///< defines the scale factor to use to resize/rescale read packets
                bool bUseHalfGT=1.0 ///< defines whether the input masks given should be half gt or not (useful for training)
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl>(
                        "cosegm_tests",
                        lv::datasets::getRootPath()+"litiv/cosegm_tests/",
                        lv::datasets::getRootPath()+"litiv/cosegm_tests/results/"+lv::addDirSlashIfMissing(sOutputDirName),
                        "bin",
                        ".png",
                        std::vector<std::string>{"test01"},
                        std::vector<std::string>(),
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        false,
                        dScaleFactor
                ),m_bUsingHalfGT(bUseHalfGT) {}
        virtual bool isUsingHalfGT() const override final {return m_bUsingHalfGT;}
    protected:
        const bool m_bUsingHalfGT;
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
            return 2;
        }
        virtual bool isStreamGrayscale(size_t nStreamIdx) const override final {
            return nStreamIdx!=0;
        }
    protected:
        virtual void parseData() override final {
            const ICosegmTestDataset& oDataset = dynamic_cast<const ICosegmTestDataset&>(*this->getRoot());
            this->m_bUsingHalfGT = oDataset.isUsingHalfGT();
            this->m_vvsInputPaths.clear();
            this->m_vvsGTPaths.clear();
            int nCurrIdx = 0;
            while(true) {
                std::vector<std::string> vCurrPaths = {
                    cv::format("%simg%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%simg%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smask%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format((this->m_bUsingHalfGT?"%sgt%05db.png":"%smask%05db.png"),this->getDataPath().c_str(),nCurrIdx),
                };
                const cv::Mat oInput0 = cv::imread(vCurrPaths[0]);
                const cv::Mat oInput1 = cv::imread(vCurrPaths[1]);
                const cv::Mat oMask0 = cv::imread(vCurrPaths[2],cv::IMREAD_GRAYSCALE);
                const cv::Mat oMask1 = cv::imread(vCurrPaths[3],cv::IMREAD_GRAYSCALE);
                if(!oInput0.empty() && !oInput1.empty() && !oMask0.empty() && !oMask1.empty()) {
                    lvAssert(oInput0.size()==oInput1.size() && oMask0.size()==oMask1.size() && oInput0.size()==oMask0.size());
                    this->m_vvsInputPaths.push_back(vCurrPaths);
                    this->m_vvsGTPaths.push_back(
                        std::vector<std::string>{
                            cv::format("%sgt%05da.png",this->getDataPath().c_str(),nCurrIdx),
                            cv::format("%sgt%05db.png",this->getDataPath().c_str(),nCurrIdx),
                        }
                    );
                    const cv::Mat oGT0 = cv::imread(this->m_vvsGTPaths.back()[0],cv::IMREAD_GRAYSCALE);
                    const cv::Mat oGT1 = cv::imread(this->m_vvsGTPaths.back()[1],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oGT0.empty() && !oGT1.empty() && oGT0.size()==oGT1.size() && oInput0.size()==oGT0.size());
                    this->m_mGTIndexLUT[nCurrIdx] = nCurrIdx;
                    ++nCurrIdx;
                }
                else
                    break;
            }
            lvAssert(this->m_vvsInputPaths.size()>0 && this->m_vvsInputPaths[0].size()>0);
            const cv::Size oOrigSize = cv::imread(this->m_vvsInputPaths[0][0]).size();
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
                this->m_vInputROIs[a] = aROIs[a].clone();
                this->m_vInputROIs[a+2] = aROIs[a].clone();
                this->m_vGTROIs[a] = aROIs[a].clone();
            }
            this->m_vInputSizes = std::vector<cv::Size>{aROIs[0].size(),aROIs[1].size(),aROIs[0].size(),aROIs[1].size()};
            this->m_vGTSizes = std::vector<cv::Size>{aROIs[0].size(),aROIs[1].size()};
            this->m_oMaxInputSize = this->m_oMaxGTSize = aROIs[0].size();
        }

        virtual cv::Mat getRawInput(size_t nPacketIdx) override final {
            if(nPacketIdx>=this->m_vvsInputPaths.size())
                return cv::Mat();
            const std::vector<std::string>& vsInputPaths = this->m_vvsInputPaths[nPacketIdx];
            const std::vector<cv::Size>& vsInputSizes = this->getInputSizeArray(nPacketIdx);
            cv::Mat oPacket;
            for(size_t nStreamIdx=0; nStreamIdx<vsInputPaths.size(); ++nStreamIdx) {
                cv::Mat oCurrImg = cv::imread(vsInputPaths[nStreamIdx],this->isStreamGrayscale(nStreamIdx)?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
                if(oCurrImg.empty()) // if a single image is missing/cannot load, we skip the entire packet
                    return cv::Mat();
                const cv::Size& oPacketSize = vsInputSizes[nStreamIdx];
                if(oCurrImg.size()!=oPacketSize)
                    cv::resize(oCurrImg,oCurrImg,oPacketSize,0,0,cv::INTER_NEAREST);
                if(oPacket.empty())
                    oPacket = oCurrImg;
                else {
                    lvDbgAssert(oPacket.isContinuous() && oCurrImg.isContinuous());
                    lvDbgAssert(size_t(oPacket.dataend-oPacket.datastart)==oPacket.total()*oPacket.elemSize());
                    cv::Mat oNewPacket(int(oPacket.total()*oPacket.elemSize()+oCurrImg.total()*oCurrImg.elemSize()),1,CV_8UC1);
                    std::copy(oPacket.datastart,oPacket.dataend,oNewPacket.data);
                    std::copy(oCurrImg.datastart,oCurrImg.dataend,oNewPacket.data+uintptr_t(oPacket.dataend-oPacket.datastart));
                    oPacket = oNewPacket;
                }
            }
            return oPacket;
        }

        virtual void unpackInput(size_t nPacketIdx, std::vector<cv::Mat>& vUnpackedInput) override final {
            const cv::Mat& oInput = this->getInput(nPacketIdx)/*.clone()*/;
            const std::vector<cv::Size>& vSizes = this->getInputSizeArray(nPacketIdx);
            if(oInput.empty()) {
                for(size_t s=0; s<vSizes.size(); ++s)
                    vUnpackedInput[s] = cv::Mat();
                return;
            }
            lvAssert(oInput.isContinuous());
            size_t nCurrPacketIdxOffset = 0;
            const size_t nTotPacketSize = oInput.elemSize()*oInput.total();
            for(size_t s=0; s<vSizes.size(); ++s) {
                const size_t nCurrPacketSize = (s==0?3:1)*vSizes[s].area();
                lvAssert_(nCurrPacketIdxOffset+nCurrPacketSize<=nTotPacketSize,"unpack out-of-bounds");
                vUnpackedInput[s] = (nCurrPacketSize>0)?cv::Mat(vSizes[s],(s==0?CV_8UC3:CV_8UC1),(void*)(oInput.data+nCurrPacketIdxOffset)):cv::Mat();
                nCurrPacketIdxOffset += nCurrPacketSize;
            }
            lvAssert_(nCurrPacketIdxOffset==nTotPacketSize,"unpack has leftover data");
        }

        bool m_bUsingHalfGT;
    };

} // namespace lv
