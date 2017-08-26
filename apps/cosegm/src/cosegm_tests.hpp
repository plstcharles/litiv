
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

#define BORDER_EXPAND_TYPE cv::BORDER_REPLICATE

namespace lv {

    static const DatasetList Dataset_CosegmTests = DatasetList(Dataset_Custom+1337); // cheat; might cause problems if exposed in multiple external/custom specializations

    struct ICosegmTestDataset {
        virtual bool isEvaluatingDisparities() const = 0;
        virtual bool isLoadingFrameSubset() const = 0;
        virtual int isLoadingInputMasks() const = 0;
    };

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_CosegmTests,eEvalImpl> :
            public ICosegmTestDataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl> {
    protected:
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bEvalDisparities=false, ///< defines whether we should evaluate fg/bg segmentation or stereo disparities
                bool bLoadFrameSubset=false, ///< defines whether only a subset of the dataset's frames will be loaded or not
                int nLoadInputMasks=0, ///< defines whether the input stream should be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
                double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests,DatasetEval_BinaryClassifierArray,eEvalImpl>(
                        "cosegm_tests",
                        lv::datasets::getRootPath()+"litiv/cosegm_tests/",
                        DataHandler::createOutputDir(lv::datasets::getRootPath()+"litiv/cosegm_tests/results/",sOutputDirName),
                        std::vector<std::string>{"test01"},
                        //std::vector<std::string>{"art"},
                        //std::vector<std::string>{"art_mini"},
                        //std::vector<std::string>{"noiseless"},
                        //std::vector<std::string>{"noiseless_mini"},
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        false,
                        dScaleFactor
                ),
                m_bEvalDisparities(bEvalDisparities),
                m_bLoadFrameSubset(bLoadFrameSubset),
                m_nLoadInputMasks(nLoadInputMasks) {}
        virtual bool isEvaluatingDisparities() const override final {return m_bEvalDisparities;}
        virtual bool isLoadingFrameSubset() const override {return m_bLoadFrameSubset;}
        virtual int isLoadingInputMasks() const override final {return m_nLoadInputMasks;}
    protected:
        const bool m_bEvalDisparities;
        const bool m_bLoadFrameSubset;
        const int m_nLoadInputMasks;
    };

    template<DatasetTaskList eDatasetTask>
    struct DataGroupHandler_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests> :
            public DataGroupHandler {
    protected:
        virtual void parseData() override {
            this->m_vpBatches.clear();
            this->m_bIsBare = true;
            if(!lv::string_contains_token(this->getName(),this->getSkipTokens()))
                this->m_vpBatches.push_back(this->createWorkBatch(this->getName(),this->getRelativePath()));
        }
    };

    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_VideoArray,Dataset_CosegmTests> {

        virtual size_t getInputStreamCount() const override final {
            return size_t(2*(this->isLoadingInputMasks()?2:1)); // 2x input images (+ 2x approx fg masks)
        }

        virtual size_t getGTStreamCount() const override final {
            return 2; // 2x fg masks or disp maps
        }

        bool isEvaluatingDisparities() const {
            return dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).isEvaluatingDisparities();
        }

        bool isLoadingFrameSubset() const {
            return dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).isLoadingFrameSubset();
        }

        int isLoadingInputMasks() const {
            return dynamic_cast<const ICosegmTestDataset&>(*this->getRoot()).isLoadingInputMasks();
        }

        std::string getFeaturesDirName() const {
            return this->m_sFeaturesDirName;
        }

        void setFeaturesDirName(const std::string& sDirName) {
            this->m_sFeaturesDirName = sDirName;
            lv::createDirIfNotExist(this->getFeaturesPath()+sDirName);
        }

        virtual std::string getFeaturesName(size_t nPacketIdx) const override final {
            std::array<char,32> acBuffer;
            snprintf(acBuffer.data(),acBuffer.size(),nPacketIdx<size_t(1e7)?"%06zu":"%09zu",nPacketIdx);
            return lv::addDirSlashIfMissing(this->m_sFeaturesDirName)+std::string(acBuffer.data());
        }

        size_t getMinDisparity() const {
            return this->m_nMinDisp;
        }

        size_t getMaxDisparity() const {
            return this->m_nMaxDisp;
        }

    protected:

        virtual void parseData() override final {
            lvDbgExceptionWatch;
            const int nLoadInputMasks = this->isLoadingInputMasks();
            const bool bLoadFrameSubset = this->isLoadingFrameSubset();
            const bool bUseInterlacedMasks = nLoadInputMasks!=0;
            const bool bUseApproxMask0 = (nLoadInputMasks&2)==0;
            const bool bUseApproxMask1 = (nLoadInputMasks&4)==0;
            const size_t nInputStreamCount = this->getInputStreamCount();
            const size_t nGTStreamCount = this->getGTStreamCount();
            constexpr size_t nInputStreamIdx0 = 0;
            const size_t nInputStreamIdx1 = bUseInterlacedMasks?2:1;
            constexpr size_t nInputMaskStreamIdx0 = 1;
            constexpr size_t nInputMaskStreamIdx1 = 3;
            this->m_vvsInputPaths.clear();
            this->m_vvsGTPaths.clear();
            this->m_vInputInfos.resize(getInputStreamCount());
            this->m_vGTInfos.resize(getGTStreamCount());
            const double dScale = this->getScaleFactor();
            int nCurrIdx = 0;
            while(true) {
                std::vector<std::string> vCurrInputPaths(nInputStreamCount);
                vCurrInputPaths[nInputStreamIdx0] = cv::format("%simg%05da.png",this->getDataPath().c_str(),nCurrIdx);
                vCurrInputPaths[nInputStreamIdx1] = cv::format("%simg%05db.png",this->getDataPath().c_str(),nCurrIdx);
                cv::Mat oInput0 = cv::imread(vCurrInputPaths[nInputStreamIdx0],cv::IMREAD_UNCHANGED);
                cv::Mat oInput1 = cv::imread(vCurrInputPaths[nInputStreamIdx1],cv::IMREAD_UNCHANGED);
                if(oInput0.empty() || oInput1.empty())
                    break;
                lvAssert(oInput0.size()==oInput1.size());
                cv::Mat oMask0,oMask1;
                if(bUseInterlacedMasks) {
                    vCurrInputPaths[nInputMaskStreamIdx0] = cv::format(((!bUseApproxMask0)?"%sgtmask%05da.png":"%smask%05da.png"),this->getDataPath().c_str(),nCurrIdx);
                    vCurrInputPaths[nInputMaskStreamIdx1] = cv::format(((!bUseApproxMask1)?"%sgtmask%05db.png":"%smask%05db.png"),this->getDataPath().c_str(),nCurrIdx);
                    oMask0 = cv::imread(vCurrInputPaths[nInputMaskStreamIdx0],cv::IMREAD_GRAYSCALE);
                    oMask1 = cv::imread(vCurrInputPaths[nInputMaskStreamIdx1],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oMask0.empty() && !oMask1.empty());
                    lvAssert(oMask0.size()==oMask1.size());
                    lvAssert(oInput0.size()==oMask0.size());
                }
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
                //cv::imwrite(cv::format("%stmp/img%05da.png",this->getDataPath().c_str(),nCurrIdx),oInput0);
                //cv::imwrite(cv::format(((this->isUsingGTMaskAsInput()&2)?"%stmp/gtmask%05da.png":"%stmp/mask%05da.png"),this->getDataPath().c_str(),nCurrIdx),oMask0);
                //cv::imwrite(cv::format("%stmp/img%05db.png",this->getDataPath().c_str(),nCurrIdx),oInput1);
                //cv::imwrite(cv::format(((this->isUsingGTMaskAsInput()&1)?"%stmp/gtmask%05db.png":"%stmp/mask%05db.png"),this->getDataPath().c_str(),nCurrIdx),oMask1);
                this->m_vvsInputPaths.push_back(vCurrInputPaths);
                if(this->m_vvsInputPaths.size()==size_t(1)) {
                    this->m_vInputInfos[nInputStreamIdx0] = lv::MatInfo(oInput0);
                    this->m_vInputInfos[nInputStreamIdx1] = lv::MatInfo(oInput1);
                    if(bUseInterlacedMasks) {
                        this->m_vInputInfos[nInputMaskStreamIdx0] = lv::MatInfo(oMask0);
                        this->m_vInputInfos[nInputMaskStreamIdx1] = lv::MatInfo(oMask1);
                    }
                }
                else {
                    lvAssert(lv::MatInfo(oInput0)==this->m_vInputInfos[nInputStreamIdx0]);
                    lvAssert(lv::MatInfo(oInput1)==this->m_vInputInfos[nInputStreamIdx1]);
                    if(bUseInterlacedMasks) {
                        lvAssert(lv::MatInfo(oMask0)==this->m_vInputInfos[nInputMaskStreamIdx0]);
                        lvAssert(lv::MatInfo(oMask1)==this->m_vInputInfos[nInputMaskStreamIdx1]);
                    }
                }
                if(this->isEvaluating()) {
                    const std::vector<std::string> vCurrGTPaths = {
                        cv::format(this->isEvaluatingDisparities()?"%sgtdisp%05da.png":"%sgtmask%05da.png",this->getDataPath().c_str(),nCurrIdx),
                        cv::format(this->isEvaluatingDisparities()?"%sgtdisp%05db.png":"%sgtmask%05db.png",this->getDataPath().c_str(),nCurrIdx),
                    };
                    cv::Mat oGT0 = cv::imread(vCurrGTPaths[0],cv::IMREAD_GRAYSCALE);
                    cv::Mat oGT1 = cv::imread(vCurrGTPaths[1],cv::IMREAD_GRAYSCALE);
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
                }
                ++nCurrIdx;
            }
            lvAssert(this->m_vvsInputPaths.size()>0 && this->m_vvsInputPaths[0].size()>0);
            const cv::Size oOrigSize = this->m_vInputInfos[0].size();
            lvAssert(oOrigSize.area()>0);
            this->m_vInputROIs.resize(nInputStreamCount);
            if(this->isEvaluating())
                this->m_vGTROIs.resize(nGTStreamCount);
            std::array<cv::Mat,2> aROIs = {
                cv::imread(this->getDataPath()+"roi0.png",cv::IMREAD_GRAYSCALE),
                cv::imread(this->getDataPath()+"roi1.png",cv::IMREAD_GRAYSCALE)
            };
            for(size_t a=0; a<2; ++a) {
                if(aROIs[a].empty())
                    aROIs[a] = cv::Mat(oOrigSize,CV_8UC1,cv::Scalar_<uchar>(255));
                else
                    lvAssert(aROIs[a].size()==oOrigSize);
                if(dScale!=1.0)
                    cv::resize(aROIs[a],aROIs[a],cv::Size(),dScale,dScale,cv::INTER_NEAREST);
                if(bUseInterlacedMasks) {
                    this->m_vInputROIs[2*a] = aROIs[a].clone();
                    this->m_vInputROIs[2*a+1] = aROIs[a].clone();
                }
                else
                    this->m_vInputROIs[a] = aROIs[a].clone();
                if(this->isEvaluating())
                    this->m_vGTROIs[a] = aROIs[a].clone();
            }
            this->m_vInputInfos[nInputStreamIdx0] = lv::MatInfo(aROIs[0].size(),this->m_vInputInfos[nInputStreamIdx0].type);
            this->m_vInputInfos[nInputStreamIdx1] = lv::MatInfo(aROIs[1].size(),this->m_vInputInfos[nInputStreamIdx1].type);
            if(bUseInterlacedMasks) {
                this->m_vInputInfos[nInputMaskStreamIdx0] = lv::MatInfo(aROIs[0].size(),this->m_vInputInfos[nInputMaskStreamIdx0].type);
                this->m_vInputInfos[nInputMaskStreamIdx1] = lv::MatInfo(aROIs[1].size(),this->m_vInputInfos[nInputMaskStreamIdx1].type);
            }
            if(this->isEvaluating()) {
                this->m_vGTInfos[0] = lv::MatInfo(aROIs[0].size(),this->m_vGTInfos[0].type);
                this->m_vGTInfos[1] = lv::MatInfo(aROIs[1].size(),this->m_vGTInfos[1].type);
            }
            this->m_nMinDisp = size_t(0);
            this->m_nMaxDisp = size_t(100);
            std::ifstream oDispRangeFile(this->getDataPath()+"drange.txt");
            if(oDispRangeFile.is_open() && !oDispRangeFile.eof()) {
                oDispRangeFile >> this->m_nMinDisp;
                if(!oDispRangeFile.eof())
                    oDispRangeFile >> this->m_nMaxDisp;
            }
            this->m_nMinDisp *= dScale;
            this->m_nMaxDisp *= dScale;
            lvAssert(this->m_nMaxDisp>this->m_nMinDisp);
        }

        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) override final {
            lvAssert(this->isEvaluating());
            std::vector<cv::Mat> vGTs = IDataProducer_<DatasetSource_VideoArray>::getRawGTArray(nPacketIdx);
            for(size_t nStreamIdx=0; nStreamIdx<vGTs.size(); ++nStreamIdx) {
                if(!vGTs[nStreamIdx].empty()) {
                    if(this->isEvaluatingDisparities() && this->getScaleFactor()!=0)
                        vGTs[nStreamIdx] *= this->getScaleFactor();
                }
            }
            return vGTs;
        }

        size_t m_nMinDisp,m_nMaxDisp;
        std::string m_sFeaturesDirName;

    };

} // namespace lv
