
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

#define LITIV_DATASET_IMPL_BEGIN(eDatasetType,eDataset) \
    template<> \
    struct Dataset_<eDatasetType,eDataset> : public IDataset_<eDatasetType,eDataset> { \
    private: \
        template<eDatasetTypeList eDatasetType, eDatasetList eDataset, typename ...Targs> \
        friend IDatasetPtr datasets::create(const Targs& ... args)
#define LITIV_DATASET_IMPL_END() }

namespace litiv {

    namespace datasets {

        template<eDatasetTypeList eDatasetType, eDatasetList eDataset, typename ...Targs>
        IDatasetPtr create(const Targs& ... args);

        template<eDatasetTypeList eDatasetType, typename ...Targs>
        IDatasetPtr create(const Targs& ... args);

    } //namespace datasets

    struct DataHandler : public virtual IDataHandler {
        virtual const std::string& getName() const override final {return m_sBatchName;}
        virtual const std::string& getDataPath() const override final {return m_sDataPath;}
        virtual const std::string& getOutputPath() const override final {return m_sOutputPath;}
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
        const std::string m_sDataPath;
        const std::string m_sOutputPath;
        const bool m_bForcingGrayscale;
        const IDatasetPtr m_pDataset;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, bool bGroup>
    struct DataProducer_ : public IDataProducer_<eDatasetType,bGroup> {};

    template<> // fully specialized dataset producer type for default CDnet (2012+2014) handling
    struct DataProducer_<eDatasetType_VideoSegm, eDataset_VideoSegm_CDnet, TNoGroup> :
            public IDataProducer_<eDatasetType_VideoSegm,TNoGroup> {

        virtual void parseData() override final {
            std::vector<std::string> vsSubDirs;
            std::cout << getName() << "->getDataPath() = " << getDataPath() << std::endl;
            PlatformUtils::GetSubDirsFromDir(getDataPath(),vsSubDirs);
            auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getDataPath()+"/groundtruth");
            auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getDataPath()+"/input");
            if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
                lvErrorExt("CDnet Sequence '%s' did not possess the required groundtruth and input directories",getName().c_str());
            PlatformUtils::GetFilesFromDir(*inputDir,m_vsInputFramePaths);
            PlatformUtils::GetFilesFromDir(*gtDir,m_vsGTFramePaths);
            if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
                lvErrorExt("CDnet Sequence '%s' did not possess same amount of GT & input frames",getName().c_str());
            m_oROI = cv::imread(getDataPath()+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
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
                    this->setProcessedPacketsPromise();
                }
            }
            template<typename ...Targs>
            static std::shared_ptr<WorkBatch> create(const Targs& ... args) {
                return std::shared_ptr<WorkBatch>(new WorkBatch(args...));
            }
        private:
            WorkBatch(const std::string& sBatchName, IDatasetPtr pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sBatchName,pDataset,sRelativePath),m_dElapsedTime_sec(0),m_bIsProcessing(false) {parseData();}
            WorkBatch& operator=(const WorkBatch&) = delete;
            WorkBatch(const WorkBatch&) = delete;
            friend struct WorkBatchGroup;
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
            virtual void parseData() override final {for(const auto& pBatch : getBatches()) pBatch->parseData();}
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
                if(!PlatformUtils::string_contains_token(getName(),pDataset->getSkippedDirTokens())) {
                    std::cout << "[" << pDataset->getName() << "] -- Parsing directory '" << pDataset->getDatasetPath()+sRelativePath << "' for work group '" << getName() << "'..." << std::endl;
                    std::vector<std::string> vsWorkBatchPaths;
                    // all subdirs are considered work batch directories (if none, the category directory itself is a batch)
                    PlatformUtils::GetSubDirsFromDir(getDataPath(),vsWorkBatchPaths);
                    if(vsWorkBatchPaths.empty()) {
                        m_vpBatches.push_back(WorkBatch::create(getName(),pDataset,getRelativePath()));
                        m_bIsBare = true;
                    }
                    else {
                        for(const auto& sPathIter : vsWorkBatchPaths) {
                            const size_t nLastSlashPos = sPathIter.find_last_of("/\\");
                            const std::string sNewBatchName = nLastSlashPos==std::string::npos?sPathIter:sPathIter.substr(nLastSlashPos+1);
                            if(!PlatformUtils::string_contains_token(sNewBatchName,pDataset->getSkippedDirTokens()))
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

        virtual const std::string& getName() const override final {return m_sDatasetName;}
        virtual const std::string& getDatasetPath() const override final {return m_sDatasetPath;}
        virtual const std::string& getOutputPath() const override final {return m_sOutputPath;}
        virtual const std::string& getOutputNamePrefix() const override final {return m_sOutputNamePrefix;}
        virtual const std::string& getOutputNameSuffix() const override final {return m_sOutputNameSuffix;}
        virtual const std::vector<std::string>& getWorkBatchDirs() const override final {return m_vsWorkBatchDirs;}
        virtual const std::vector<std::string>& getSkippedDirTokens() const override final {return m_vsSkippedDirTokens;}
        virtual const std::vector<std::string>& getGrayscaleDirTokens() const override final {return m_vsGrayscaleDirTokens;}
        virtual size_t getOutputIdxOffset() const override final {return m_nOutputIdxOffset;}
        virtual bool isSavingOutput() const override final {return m_bSavingOutput;}
        virtual bool isUsingEvaluator() const override final {return m_bUsingEvaluator;}
        virtual bool is4ByteAligned() const override final {return m_bForce4ByteDataAlign;}
        virtual double getScaleFactor() const override final {return m_dScaleFactor;}

        virtual double getExpectedLoad() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getExpectedLoad();});}
        virtual size_t getTotPackets() const override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getTotPackets();});}
        virtual double getProcessTime() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessTime();});}
        virtual size_t getProcessedPacketsCountPromise() override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCountPromise();});}
        virtual size_t getProcessedPacketsCount() const override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCount();});}

        virtual void parseDataset() override final {
            std::cout << "Parsing dataset '" << getName() << "'..." << std::endl;
            m_vpBatches.clear();
            if(!getOutputPath().empty())
                PlatformUtils::CreateDirIfNotExist(getOutputPath());
            for(const auto& sPathIter : getWorkBatchDirs())
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
        IDataset_(
                const std::string& sDatasetName,
                const std::string& sDatasetDirName,
                const std::string& sOutputDirName,
                const std::string& sOutputNamePrefix,
                const std::string& sOutputNameSuffix,
                const std::vector<std::string>& vsWorkBatchDirs,
                const std::vector<std::string>& vsSkippedDirTokens,
                const std::vector<std::string>& vsGrayscaleDirTokens,
                size_t nOutputIdxOffset,
                bool bSaveOutput,
                bool bUseEvaluator,
                bool bForce4ByteDataAlign,
                double dScaleFactor
        ) :
                m_sDatasetName(sDatasetName),
                m_sDatasetPath(std::string(DATASET_ROOT)+"/"+sDatasetDirName+"/"),
                m_sOutputPath(std::string(DATASET_ROOT)+"/"+sDatasetDirName+"/"+sOutputDirName+"/"),
                m_sOutputNamePrefix(sOutputNamePrefix),
                m_sOutputNameSuffix(sOutputNameSuffix),
                m_vsWorkBatchDirs(vsWorkBatchDirs),
                m_vsSkippedDirTokens(vsSkippedDirTokens),
                m_vsGrayscaleDirTokens(vsGrayscaleDirTokens),
                m_nOutputIdxOffset(nOutputIdxOffset),
                m_bSavingOutput(bSaveOutput),
                m_bUsingEvaluator(bUseEvaluator),
                m_bForce4ByteDataAlign(bForce4ByteDataAlign),
                m_dScaleFactor(dScaleFactor) {}
        const std::string m_sDatasetName;
        const std::string m_sDatasetPath;
        const std::string m_sOutputPath;
        const std::string m_sOutputNamePrefix;
        const std::string m_sOutputNameSuffix;
        const std::vector<std::string> m_vsWorkBatchDirs;
        const std::vector<std::string> m_vsSkippedDirTokens;
        const std::vector<std::string> m_vsGrayscaleDirTokens;
        const size_t m_nOutputIdxOffset;
        const bool m_bSavingOutput;
        const bool m_bUsingEvaluator;
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
        using IDataset_<eDatasetType,eDataset>::IDataset_;
        template<eDatasetTypeList eDatasetType2, eDatasetList eDataset2, typename ...Targs>
        friend IDatasetPtr datasets::create(const Targs& ... args);
    };

    LITIV_DATASET_IMPL_BEGIN(eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet);
        Dataset_(const std::string& sOutputDirName, bool bSaveOutput=false, bool bUseEvaluator=true, bool bForce4ByteDataAlign=false, double dScaleFactor=1.0, bool b2014=true) :
                IDataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet>(
                        b2014?"CDnet 2014":"CDnet 2012",
                        b2014?"CDNet2014/dataset":"CDNet/dataset",
                        sOutputDirName,
                        "bin",
                        ".png",
                        std::vector<std::string>{"baseline_highway_cut2"},//b2014?std::vector<std::string>{"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"}:std::vector<std::string>{"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"},
                        std::vector<std::string>{},
                        b2014?std::vector<std::string>{"thermal","turbulence"}:std::vector<std::string>{"thermal"},
                        1,
                        bSaveOutput,
                        bUseEvaluator,
                        bForce4ByteDataAlign,
                        dScaleFactor
                ) {}
    LITIV_DATASET_IMPL_END();

    namespace datasets {

        template<eDatasetTypeList eDatasetType, eDatasetList eDataset, typename ...Targs>
        IDatasetPtr create(const Targs& ... args) {
            auto pDataset = IDatasetPtr(new Dataset_<eDatasetType,eDataset>(args...));
            pDataset->parseDataset();
            return pDataset;
        }

        template<eDatasetTypeList eDatasetType, typename ...Targs>
        IDatasetPtr create(const Targs& ... args) {
            return create<eDatasetType,getCustomDatasetEnum(eDatasetType)>(args...);
        }

    } //namespace datasets

} //namespace litiv
