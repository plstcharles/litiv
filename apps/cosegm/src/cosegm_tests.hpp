
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
#include <fstream>

#define BORDER_EXPAND_TYPE cv::BORDER_REPLICATE

namespace lv {

    static const DatasetList Dataset_CosegmTests = DatasetList(Dataset_Custom+1); // cheat; might cause problems if exposed in multiple external/custom specializations

    struct ICosegmTestDataset {
        virtual int getExtraPixelBorderSize() const = 0;
        virtual int isUsingGTMaskAsInput() const = 0;
        virtual bool isEvaluatingStereoDisp() const = 0;
    };

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_CosegmTests,eEvalImpl> :
            public ICosegmTestDataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl> {
    protected:
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in VAP trimodal dataset results folder)
                int nExtraPixelBorderSize, ///< defines the extra border size to add to all input images to avoid oob-errors when extracting features
                int nUseGTMaskAsInput=1, ///< defines whether the input masks given should be half gt or not (useful for training) --- 1 = use thermal gt mask, 2 = use rgb gt mask, 0 = use no gt mask
                bool bEvalStereoDisp=false, ///< defines whether we should evaluate fg/bg segmentation or stereo disparity
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl>(
                        "cosegm_tests",
                        lv::datasets::getRootPath()+"litiv/cosegm_tests/",
                        lv::datasets::getRootPath()+"litiv/cosegm_tests/results/"+lv::addDirSlashIfMissing(sOutputDirName),
                        //std::vector<std::string>{"test01"},
                        std::vector<std::string>{"art"},
                        //std::vector<std::string>{"art_mini"},
                        //std::vector<std::string>{"noiseless"},
                        //std::vector<std::string>{"noiseless_mini"},
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        false,
                        dScaleFactor
                ),m_nExtraPixelBorderSize(nExtraPixelBorderSize),m_nUseGTMaskAsInput(nUseGTMaskAsInput),m_bEvalStereoDisp(bEvalStereoDisp) {}
        virtual int getExtraPixelBorderSize() const override final {return m_nExtraPixelBorderSize;}
        virtual int isUsingGTMaskAsInput() const override final {return m_nUseGTMaskAsInput;}
        virtual bool isEvaluatingStereoDisp() const override final {return m_bEvalStereoDisp;}
    protected:
        const int m_nExtraPixelBorderSize;
        const int m_nUseGTMaskAsInput;
        const bool m_bEvalStereoDisp;
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

        int isUsingGTMaskAsInput() const {
            return dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).isUsingGTMaskAsInput();
        }

        bool isEvaluatingStereoDisp() const {
            return dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).isEvaluatingStereoDisp();
        }

        std::string getFeaturesDirName() const {
            return this->m_sFeaturesDirName;
        }

        void setFeaturesDirName(const std::string& sDirName) {
            this->m_sFeaturesDirName = sDirName;
            lv::createDirIfNotExist(this->getFeaturesPath()+sDirName);
        }

        size_t getMinDisparity() const {
            return this->m_nMinDisp;
        }

        size_t getMaxDisparity() const {
            return this->m_nMaxDisp;
        }

        size_t getDisparityStep() const {
            return this->m_nDispStep;
        }

        virtual size_t getInputStreamCount() const override final {
            return 4; // 2x input images, 2x approx fg masks
        }

        virtual size_t getGTStreamCount() const override final {
            return 2; // 2x fg masks or disp maps
        }

        virtual std::string getFeaturesName(size_t nPacketIdx) const override final {
            std::array<char,32> acBuffer;
            snprintf(acBuffer.data(),acBuffer.size(),nPacketIdx<size_t(1e7)?"%06zu":"%09zu",nPacketIdx);
            return lv::addDirSlashIfMissing(this->m_sFeaturesDirName)+std::string(acBuffer.data());
        }

    protected:

        virtual void parseData() override final {
            this->m_vvsInputPaths.clear();
            this->m_vvsGTPaths.clear();
            this->m_vInputInfos.resize(getInputStreamCount());
            this->m_vGTInfos.resize(getGTStreamCount());
            const double dScale = this->getScaleFactor();
            const int nExtraPxBorderSize = dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).getExtraPixelBorderSize();
            int nCurrIdx = 0;
            while(true) {
                const std::vector<std::string> vCurrInputPaths = {
                    cv::format("%simg%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format(((this->isUsingGTMaskAsInput()&2)?"%sgtmask%05da.png":"%smask%05da.png"),this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%simg%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format(((this->isUsingGTMaskAsInput()&1)?"%sgtmask%05db.png":"%smask%05db.png"),this->getDataPath().c_str(),nCurrIdx),
                };
                cv::Mat oInput0 = cv::imread(vCurrInputPaths[0],cv::IMREAD_UNCHANGED);
                cv::Mat oMask0 = cv::imread(vCurrInputPaths[1],cv::IMREAD_GRAYSCALE);
                cv::Mat oInput1 = cv::imread(vCurrInputPaths[2],cv::IMREAD_UNCHANGED);
                cv::Mat oMask1 = cv::imread(vCurrInputPaths[3],cv::IMREAD_GRAYSCALE);
                if(oInput0.empty() || oMask0.empty() || oInput1.empty() || oMask1.empty())
                    break;
                /*const std::vector<std::string> in = {
                    cv::format("%simg%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%simg%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smask%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smask%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%sgtmask%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%sgtmask%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%sgtdisp%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%sgtdisp%05db.png",this->getDataPath().c_str(),nCurrIdx),
                };
                const std::vector<std::string> out = {
                    cv::format("%smini/img%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smini/img%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smini/mask%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smini/mask%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smini/gtmask%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smini/gtmask%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smini/gtdisp%05da.png",this->getDataPath().c_str(),nCurrIdx),
                    cv::format("%smini/gtdisp%05db.png",this->getDataPath().c_str(),nCurrIdx),
                };
                cv::Rect croproi(13,132,340,200);
                for(size_t i=0; i<in.size(); ++i) {
                    cv::Mat img = cv::imread(in[i],cv::IMREAD_UNCHANGED);
                    if(!img.empty())
                        cv::imwrite(out[i],img(croproi));
                }*/
                lvAssert(oInput0.size()==oInput1.size() && oMask0.size()==oMask1.size() && oInput0.size()==oMask0.size());
                //cv::imwrite(cv::format("%stmp/img%05da.png",this->getDataPath().c_str(),nCurrIdx),oInput0);
                //cv::imwrite(cv::format(((this->isUsingGTMaskAsInput()&2)?"%stmp/gtmask%05da.png":"%stmp/mask%05da.png"),this->getDataPath().c_str(),nCurrIdx),oMask0);
                //cv::imwrite(cv::format("%stmp/img%05db.png",this->getDataPath().c_str(),nCurrIdx),oInput1);
                //cv::imwrite(cv::format(((this->isUsingGTMaskAsInput()&1)?"%stmp/gtmask%05db.png":"%stmp/mask%05db.png"),this->getDataPath().c_str(),nCurrIdx),oMask1);
                if(nExtraPxBorderSize>0) {
                    cv::copyMakeBorder(oInput0,oInput0,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                    cv::copyMakeBorder(oMask0,oMask0,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                    cv::copyMakeBorder(oInput1,oInput1,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                    cv::copyMakeBorder(oMask1,oMask1,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                    /*cv::Mat test0,test1;
                    cv::resize(oInput0,test0,cv::Size(),20,20,cv::INTER_NEAREST);
                    cv::imshow("test0",test0);
                    lvPrint(oInput0);
                    cv::resize(oInput1,test1,cv::Size(),20,20,cv::INTER_NEAREST);
                    cv::imshow("test1",test1);
                    lvPrint(oInput1);
                    cv::waitKey(0);*/
                }
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
                if(this->isEvaluating()) {
                    const std::vector<std::string> vCurrGTPaths = {
                        cv::format(this->isEvaluatingStereoDisp()?"%sgtdisp%05da.png":"%sgtmask%05da.png",this->getDataPath().c_str(),nCurrIdx),
                        cv::format(this->isEvaluatingStereoDisp()?"%sgtdisp%05db.png":"%sgtmask%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    };
                    cv::Mat oGT0 = cv::imread(vCurrGTPaths[0],cv::IMREAD_GRAYSCALE);
                    cv::Mat oGT1 = cv::imread(vCurrGTPaths[1],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oGT0.empty() && !oGT1.empty() && oGT0.size()==oGT1.size() && oInput0.size()==oGT0.size());
                    if(nExtraPxBorderSize>0) {
                        cv::copyMakeBorder(oGT0,oGT0,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                        cv::copyMakeBorder(oGT1,oGT1,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                    }
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
                }
                ++nCurrIdx;
            }
            lvAssert(this->m_vvsInputPaths.size()>0 && this->m_vvsInputPaths[0].size()>0);
            const cv::Size oOrigSize = this->m_vInputInfos[0].size();
            lvAssert(oOrigSize.area()>0);
            std::array<cv::Mat,2> aROIs = {
                cv::imread(this->getDataPath()+"roi0.png",cv::IMREAD_GRAYSCALE),
                cv::imread(this->getDataPath()+"roi1.png",cv::IMREAD_GRAYSCALE)
            };
            this->m_vInputROIs.resize(4);
            if(this->isEvaluating())
                this->m_vGTROIs.resize(2);
            for(size_t a=0; a<2; ++a) {
                if(aROIs[a].empty())
                    aROIs[a] = cv::Mat(oOrigSize,CV_8UC1,cv::Scalar_<uchar>(255));
                else {
                    if(nExtraPxBorderSize>0)
                        cv::copyMakeBorder(aROIs[a],aROIs[a],nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                    lvAssert(aROIs[a].size()==oOrigSize);
                }
                if(dScale!=1.0)
                    cv::resize(aROIs[a],aROIs[a],cv::Size(),dScale,dScale,cv::INTER_NEAREST);
                this->m_vInputROIs[2*a] = aROIs[a].clone();
                this->m_vInputROIs[2*a+1] = aROIs[a].clone();
                if(this->isEvaluating())
                    this->m_vGTROIs[a] = aROIs[a].clone();
            }
            this->m_vInputInfos[0] = lv::MatInfo(aROIs[0].size(),this->m_vInputInfos[0].type);
            this->m_vInputInfos[1] = lv::MatInfo(aROIs[0].size(),this->m_vInputInfos[1].type);
            this->m_vInputInfos[2] = lv::MatInfo(aROIs[1].size(),this->m_vInputInfos[2].type);
            this->m_vInputInfos[3] = lv::MatInfo(aROIs[1].size(),this->m_vInputInfos[3].type);
            if(this->isEvaluating()) {
                this->m_vGTInfos[0] = lv::MatInfo(aROIs[0].size(),this->m_vGTInfos[0].type);
                this->m_vGTInfos[1] = lv::MatInfo(aROIs[1].size(),this->m_vGTInfos[1].type);
            }
            this->m_nMinDisp = size_t(0);
            this->m_nMinDisp = size_t(100);
            this->m_nDispStep = size_t(1);
            std::ifstream oDispRangeFile(this->getDataPath()+"drange.txt");
            if(oDispRangeFile.is_open() && !oDispRangeFile.eof()) {
                oDispRangeFile >> this->m_nMinDisp;
                if(!oDispRangeFile.eof()) {
                    oDispRangeFile >> this->m_nMaxDisp;
                    if(!oDispRangeFile.eof())
                        oDispRangeFile >> this->m_nDispStep;
                }
            }
            this->m_nMinDisp *= dScale;
            this->m_nMaxDisp *= dScale;
            this->m_nDispStep = std::max((size_t)std::round(this->m_nDispStep*dScale),size_t(1));
            lvAssert(this->m_nMaxDisp>this->m_nMinDisp);
            this->m_nMaxDisp -= (this->m_nMaxDisp-this->m_nMinDisp)%this->m_nDispStep;
            lvAssert(((this->m_nMaxDisp-this->m_nMinDisp)%this->m_nDispStep)==0);
            lvAssert(this->m_nMaxDisp>this->m_nMinDisp);
        }

        virtual std::vector<cv::Mat> getRawInputArray(size_t nPacketIdx) override final {
            const int nExtraPxBorderSize = dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).getExtraPixelBorderSize();
            std::vector<cv::Mat> vInputs = IDataProducer_<DatasetSource_VideoArray>::getRawInputArray(nPacketIdx);
            for(size_t nStreamIdx=0; nStreamIdx<vInputs.size(); ++nStreamIdx) {
                if(!vInputs[nStreamIdx].empty() && nExtraPxBorderSize>0)
                    cv::copyMakeBorder(vInputs[nStreamIdx],vInputs[nStreamIdx],nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
            }
            return vInputs;
        }

        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) override final {
            lvAssert(this->isEvaluating());
            const int nExtraPxBorderSize = dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).getExtraPixelBorderSize();
            std::vector<cv::Mat> vGTs = IDataProducer_<DatasetSource_VideoArray>::getRawGTArray(nPacketIdx);
            for(size_t nStreamIdx=0; nStreamIdx<vGTs.size(); ++nStreamIdx) {
                if(!vGTs[nStreamIdx].empty()) {
                    if(this->isEvaluatingStereoDisp() && this->getScaleFactor()!=0)
                        vGTs[nStreamIdx] *= this->getScaleFactor();
                    if(nExtraPxBorderSize>0)
                        cv::copyMakeBorder(vGTs[nStreamIdx],vGTs[nStreamIdx],nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,nExtraPxBorderSize,BORDER_EXPAND_TYPE);
                }
            }
            return vGTs;
        }

        size_t m_nMinDisp,m_nMaxDisp,m_nDispStep;
        std::string m_sFeaturesDirName;

    };

} // namespace lv
