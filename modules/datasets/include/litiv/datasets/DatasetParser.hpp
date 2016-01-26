
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

#include "litiv/datasets/DatasetUtils.hpp"
#include "litiv/datasets/DatasetEvaluator.hpp"

namespace litiv {

    struct DataHandler : public virtual IDataHandler {
        virtual const std::string& getName() const override final {return m_sBatchName;}
        virtual const std::string& getPath() const override final {return m_sDatasetPath;}
        virtual const std::string& getResultsPath() const override final {return m_sResultsPath;}
        virtual const std::string& getRelativePath() const override final {return m_sRelativePath;}
        virtual bool isGrayscale() const override final {return m_bForcingGrayscale;}
        virtual IDatasetPtr getDatasetInfo() const override final {return m_pDataset;}
    protected:
        DataHandler(const std::string& sBatchName, std::shared_ptr<IDataset> pDataset, const std::string& sRelativePath);
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const override final {
            if(isGroup()) {
                size_t nCurrPacketCount = 0;
                auto vpBatches = getBatches();
                auto ppBatchIter = vpBatches.begin();
                while(ppBatchIter!=vpBatches.end()) {
                    const size_t nNextPacketIncr = (*ppBatchIter)->getTotPackets();
                    if(nPacketIdx<nCurrPacketCount+nNextPacketIncr)
                        break;
                    nCurrPacketCount += nNextPacketIncr;
                    ++ppBatchIter;
                }
                CV_Assert(ppBatchIter!=vpBatches.end());
                nPacketIdx -= nCurrPacketCount;
                return *ppBatchIter;
            }
            else {
                CV_Assert(nPacketIdx<getTotPackets());
                return shared_from_this();
            }
        }
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) override final {
            if(isGroup()) {
                size_t nCurrPacketCount = 0;
                auto vpBatches = getBatches();
                auto ppBatchIter = vpBatches.begin();
                while(ppBatchIter!=vpBatches.end()) {
                    const size_t nNextPacketIncr = (*ppBatchIter)->getTotPackets();
                    if(nPacketIdx<nCurrPacketCount+nNextPacketIncr)
                        break;
                    nCurrPacketCount += nNextPacketIncr;
                    ++ppBatchIter;
                }
                CV_Assert(ppBatchIter!=vpBatches.end());
                nPacketIdx -= nCurrPacketCount;
                return *ppBatchIter;
            }
            else {
                CV_Assert(nPacketIdx<getTotPackets());
                return shared_from_this();
            }
        }
    private:
        const std::string m_sBatchName;
        const std::string m_sRelativePath;
        const std::string m_sDatasetPath;
        const std::string m_sResultsPath;
        const bool m_bForcingGrayscale;
        const IDatasetPtr m_pDataset;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, bool bGroup>
    struct DataProducer_ : public IDataProducer_<eDatasetType,bGroup> {};

    template<> // fully specialized dataset producer type for default CDnet (2012+2014) handling
    struct DataProducer_<eDatasetType_VideoSegm, eDataset_VideoSegm_CDnet, TNoGroup> :
            public IDataProducer_<eDatasetType_VideoSegm,TNoGroup> {

        virtual void parseDataset() override final {
            std::vector<std::string> vsSubDirs;
            PlatformUtils::GetSubDirsFromDir(getPath(),vsSubDirs);
            auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getPath()+"/groundtruth");
            auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getPath()+"/input");
            if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
                lvErrorExt("CDnet Sequence '%s' did not possess the required groundtruth and input directories",getName().c_str());
            PlatformUtils::GetFilesFromDir(*inputDir,m_vsInputFramePaths);
            PlatformUtils::GetFilesFromDir(*gtDir,m_vsGTFramePaths);
            if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
                lvErrorExt("CDnet Sequence '%s' did not possess same amount of GT & input frames",getName().c_str());
            m_oROI = cv::imread(getPath()+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
            if(m_oROI.empty())
                lvErrorExt("CDnet Sequence '%s' did not possess a ROI.bmp file",getName().c_str());
            m_oROI = m_oROI>0; // @@@@@ check throw here???
            m_oSize = m_oROI.size();
            m_nFrameCount = m_vsInputFramePaths.size();
            CV_Assert(m_nFrameCount>0);
            // note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
        }

        virtual cv::Mat _getGTPacket_impl(size_t nIdx) override final {
            cv::Mat oFrame = cv::imread(m_vsGTFramePaths[nIdx],cv::IMREAD_GRAYSCALE);
            if(oFrame.empty())
                oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL));
            else if(oFrame.size()!=m_oSize)
                cv::resize(oFrame,oFrame,m_oSize,0,0,cv::INTER_NEAREST);
            return oFrame;
        }
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset>
    struct IDataset_ : public DatasetEvaluator_<eDatasetType,eDataset> { // dataset interface specialization for smaller impl sizes

        struct WorkBatch final :
                public DataHandler,
                public DataProducer_<eDatasetType,eDataset,TNoGroup>,
                public DataEvaluator_<eDatasetType,eDataset,TNoGroup> {
            virtual ~WorkBatch() = default;
            virtual eDatasetTypeList getDatasetType() const override final {return eDatasetType;}
            virtual eDatasetList getDataset() const override final {return eDataset;}
            virtual bool isBare() const override final {return false;}
            virtual bool isGroup() const override final {return false;}
            virtual IDataHandlerPtrArray getBatches() const override final {return IDataHandlerPtrArray();}
            virtual double getProcessTime() const override final {return m_dElapsedTime_sec;}
            bool isProcessing() {return m_bIsProcessing;}
            void startProcessing() { // used to start batch timer & init other time-critical components via _startProcessing
                if(!m_bIsProcessing) {
                    _startProcessing();
                    m_oStopWatch.tick();
                    m_bIsProcessing = true;
                }
            }
            void stopProcessing() { // used to stop batch timer & release other time-critical components via _stopProcessing (also implies stopPrecaching)
                if(m_bIsProcessing) {
                    m_dElapsedTime_sec = m_oStopWatch.tock();
                    m_bIsProcessing = false;
                    _stopProcessing();
                    stopPrecaching();
                }
            }
            template<typename ...Targs>
            static std::shared_ptr<WorkBatch> create(const Targs& ... args) {
                return std::shared_ptr<WorkBatch>(new WorkBatch(args...));
            }
        private:
            WorkBatch(const std::string& sBatchName, IDatasetPtr pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sBatchName,pDataset,sRelativePath),m_dElapsedTime_sec(0),m_bIsProcessing(false) {parseDataset();}
            WorkBatch& operator=(const WorkBatch&) = delete;
            WorkBatch(const WorkBatch&) = delete;
            friend class WorkBatchGroup;
            CxxUtils::StopWatch m_oStopWatch;
            double m_dElapsedTime_sec;
            bool m_bIsProcessing;
        };

        struct WorkBatchGroup final :
                public DataHandler,
                public DataProducer_<eDatasetType,eDataset,TGroup>,
                public DataEvaluator_<eDatasetType,eDataset,TGroup> {
            virtual ~WorkBatchGroup() = default;
            virtual eDatasetTypeList getDatasetType() const override final {return eDatasetType;}
            virtual eDatasetList getDataset() const override final {return eDataset;}
            virtual bool isBare() const override final {return m_bIsBare;}
            virtual bool isGroup() const override final {return true;}
            virtual IDataHandlerPtrArray getBatches() const override final {return m_vpBatches;}
            virtual void startPrecaching(bool bPrecacheGT, size_t nSuggestedBufferSize=SIZE_MAX) override final {for(const auto& pBatch : getBatches()) pBatch->startPrecaching(bPrecacheGT,nSuggestedBufferSize);}
            virtual void stopPrecaching() override final {for(const auto& pBatch : getBatches()) pBatch->stopPrecaching();}
            virtual void parseDataset() override final {for(const auto& pBatch : getBatches()) pBatch->parseDataset();}
            virtual double getProcessTime() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessTime();});}
            virtual double getExpectedLoad() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getExpectedLoad();});}
            virtual size_t getTotPackets() const override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getTotPackets();});}
            template<typename ...Targs>
            static std::shared_ptr<WorkBatchGroup> create(const Targs& ... args) {
                return std::shared_ptr<WorkBatchGroup>(new WorkBatchGroup(args...));
            }
        private:
            WorkBatchGroup(const std::string& sGroupName, std::shared_ptr<IDataset> pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sGroupName,pDataset,sRelativePath+"/"+sGroupName+"/"),m_bIsBare(false) {
                PlatformUtils::CreateDirIfNotExist(getResultsPath());
                if(!PlatformUtils::string_contains_token(getName(),pDataset->getSkippedNameTokens())) {
                    std::cout << "[" << pDataset->getDatasetName() << "] -- Parsing directory '" << pDataset->getDatasetRootPath()+sRelativePath << "' for work group '" << getName() << "'..." << std::endl;
                    std::vector<std::string> vsWorkBatchPaths;
                    // all subdirs are considered work batch directories (if none, the category directory itself is a batch)
                    PlatformUtils::GetSubDirsFromDir(getPath(),vsWorkBatchPaths);
                    if(vsWorkBatchPaths.empty()) {
                        m_vpBatches.push_back(WorkBatch::create(getName(),pDataset,getRelativePath()));
                        m_bIsBare = true;
                    }
                    else {
                        for(const auto& sPathIter : vsWorkBatchPaths) {
                            const size_t nLastSlashPos = sPathIter.find_last_of("/\\");
                            const std::string sNewBatchName = nLastSlashPos==std::string::npos?sPathIter:sPathIter.substr(nLastSlashPos+1);
                            if(!PlatformUtils::string_contains_token(sNewBatchName,pDataset->getSkippedNameTokens()))
                                m_vpBatches.push_back(WorkBatch::create(sNewBatchName,pDataset,getRelativePath()+"/"+sNewBatchName+"/"));
                        }
                    }
                }
            }
            WorkBatchGroup& operator=(const WorkBatchGroup&) = delete;
            WorkBatchGroup(const WorkBatchGroup&) = delete;
            IDataHandlerPtrArray m_vpBatches;
            bool m_bIsBare;
        };

        virtual const std::string& getDatasetName() const override final {return m_sDatasetName;}
        virtual const std::string& getDatasetRootPath() const override final {return m_sDatasetRootPath;}
        virtual const std::string& getResultsRootPath() const override final {return m_sResultsRootPath;}
        virtual const std::string& getResultsNamePrefix() const override final {return m_sResultNamePrefix;}
        virtual const std::string& getResultsNameSuffix() const override final {return m_sResultNameSuffix;}
        virtual const std::vector<std::string>& getWorkBatchPaths() const override final {return m_vsWorkBatchPaths;}
        virtual const std::vector<std::string>& getSkippedNameTokens() const override final {return m_vsSkippedNameTokens;}
        virtual const std::vector<std::string>& getGrayscaleNameTokens() const override final {return m_vsGrayscaleNameTokens;}
        virtual size_t getOutputIdxOffset() const override final {return m_nOutputIdxOffset;}
        virtual bool isSavingResults() const override final {return m_bSavingResults;}
        virtual bool is4ByteAligned() const override final {return m_bForce4ByteDataAlign;}
        virtual double getScaleFactor() const override final {return m_dScaleFactor;}

        virtual double getExpectedLoad() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getExpectedLoad();});}
        virtual size_t getTotPackets() const override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getTotPackets();});}
        virtual double getProcessTime() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessTime();});}
        virtual size_t getProcessedPacketsCountPromise() override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCountPromise();});}
        virtual size_t getProcessedPacketsCount() const override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCount();});}

        virtual void parseDataset() override final {
            std::cout << "Parsing dataset '" << getDatasetName() << "'..." << std::endl;
            m_vpBatches.clear();
            if(!getResultsRootPath().empty())
                PlatformUtils::CreateDirIfNotExist(getResultsRootPath());
            for(const auto& sPathIter : getWorkBatchPaths())
                m_vpBatches.push_back(WorkBatchGroup::create(sPathIter,this->shared_from_this()));
        }

        virtual IDataHandlerPtrArray getBatches() const override final {
            return m_vpBatches;
        }

        virtual IDataHandlerPtrQueue getSortedBatches() const override final {
            IDataHandlerPtrQueue vpBatches(&IDataHandler::compare_load<IDataHandler>);
            std::function<void(const IDataHandlerPtr&)> lPushBatches = [&](const IDataHandlerPtr& pBatch) {
                if(pBatch->isGroup())
                    for(const auto& pSubBatch : pBatch->getBatches())
                        lPushBatches(pSubBatch);
                else
                    vpBatches.push(pBatch);
            };
            for(const auto& pBatch : getBatches())
                lPushBatches(pBatch);
            return vpBatches;
        }

    protected:
        IDataset_(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                  const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                  const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                  size_t nOutputIdxOffset, bool bSaveResults, bool bForce4ByteDataAlign, double dScaleFactor) :
                m_sDatasetName(sDatasetName),m_sDatasetRootPath(sDatasetRootPath),m_sResultsRootPath(sResultsRootPath),
                m_sResultNamePrefix(sResultNamePrefix),m_sResultNameSuffix(sResultNameSuffix),m_vsWorkBatchPaths(vsWorkBatchPaths),
                m_vsSkippedNameTokens(vsSkippedNameTokens),m_vsGrayscaleNameTokens(vsGrayscaleNameTokens),m_nOutputIdxOffset(nOutputIdxOffset),
                m_bSavingResults(bSaveResults),m_bForce4ByteDataAlign(bForce4ByteDataAlign),m_dScaleFactor(dScaleFactor) {}
        const std::string m_sDatasetName;
        const std::string m_sDatasetRootPath;
        const std::string m_sResultsRootPath;
        const std::string m_sResultNamePrefix;
        const std::string m_sResultNameSuffix;
        const std::vector<std::string> m_vsWorkBatchPaths;
        const std::vector<std::string> m_vsSkippedNameTokens;
        const std::vector<std::string> m_vsGrayscaleNameTokens;
        const size_t m_nOutputIdxOffset;
        const bool m_bSavingResults;
        const bool m_bForce4ByteDataAlign;
        const double m_dScaleFactor;
        IDataHandlerPtrArray m_vpBatches;
    private:
        IDataset_& operator=(const IDataset_&) = delete;
        IDataset_(const IDataset_&) = delete;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset>
    struct Dataset_ : public IDataset_<eDatasetType,eDataset> {
        // if the dataset type/id is not specialized, redirect to default IDataset_ constructor
        private:
        friend struct datasets;
        using IDataset_<eDatasetType,eDataset>::IDataset_;
    };

    template<>
    struct Dataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet> final :
            public IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet> {
        Dataset_(const std::string& sDatasetRootPath, const std::string& sResultsDirName, bool bSaveResults=true, bool bForce4ByteDataAlign=false, double dScaleFactor=1.0, bool b2014=true) :
                IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet>(
                    b2014?"CDnet 2014":"CDnet 2012",
                    b2014?sDatasetRootPath+"/CDNet2014/dataset/":sDatasetRootPath+"/CDNet/dataset/",
                    b2014?sDatasetRootPath+"/CDNet2014/"+sResultsDirName+"/":sDatasetRootPath+"/CDNet/"+sResultsDirName+"/",
                    "bin",
                    ".png",
                    b2014?std::vector<std::string>{"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"}:std::vector<std::string>{"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"},
                    std::vector<std::string>{},
                    b2014?std::vector<std::string>{"thermal","turbulence"}:std::vector<std::string>{"thermal"},
                    1,
                    bSaveResults,
                    bForce4ByteDataAlign,
                    dScaleFactor) {}
    };

    namespace datasets {
        using CustomVideoSegm = Dataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_Custom>;
        using CDnet = Dataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet>;
    } //namespace datasets

} //namespace litiv
