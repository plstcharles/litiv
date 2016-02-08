
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

#include "litiv/datasets/utils.hpp"
#include "litiv/datasets/eval.hpp"

namespace litiv {

    namespace datasets {

        template<eDatasetTypeList eDatasetType, eDatasetList eDataset, ParallelUtils::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args);

        template<eDatasetTypeList eDatasetType, ParallelUtils::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args);

    } //namespace datasets

    struct DataHandler : public virtual IDataHandler {
        virtual const std::string& getName() const override final;
        virtual const std::string& getDataPath() const override final;
        virtual const std::string& getOutputPath() const override final;
        virtual const std::string& getRelativePath() const override final;
        virtual bool isGrayscale() const override final;
        virtual IDatasetPtr getDatasetInfo() const override final;
    protected:
        DataHandler(const std::string& sBatchName, std::shared_ptr<IDataset> pDataset, const std::string& sRelativePath);
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const override final;
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) override final;
    private:
        const std::string m_sBatchName;
        const std::string m_sRelativePath;
        const std::string m_sDataPath;
        const std::string m_sOutputPath;
        const bool m_bForcingGrayscale;
        const IDatasetPtr m_pDataset;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, ParallelUtils::eParallelAlgoType eEvalImpl>
    struct IDataset_ : public DatasetEvaluator_<eDatasetType,eDataset> {
        struct WorkBatch :
                public DataHandler,
                public DataProducer_<eDatasetType,eDataset,eNotGroup>,
                public std::conditional<(eEvalImpl==ParallelUtils::eNonParallel),DataEvaluator_<eDatasetType,eDataset>,AsyncDataEvaluator_<eDatasetType,eDataset,eEvalImpl>>::type {
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
            template<typename... Targs>
            static std::shared_ptr<WorkBatch> create(Targs&&... args) {
                struct WorkBatchWrapper : public WorkBatch {
                    WorkBatchWrapper(Targs&&... args) : WorkBatch(std::forward<Targs>(args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
                };
                return std::make_shared<WorkBatchWrapper>(std::forward<Targs>(args)...);
            }
        protected:
            WorkBatch(const std::string& sBatchName, IDatasetPtr pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sBatchName,pDataset,sRelativePath),m_dElapsedTime_sec(0),m_bIsProcessing(false) {parseData();}
            WorkBatch& operator=(const WorkBatch&) = delete;
            WorkBatch(const WorkBatch&) = delete;
            CxxUtils::StopWatch m_oStopWatch;
            double m_dElapsedTime_sec;
            bool m_bIsProcessing;
        };

        struct WorkBatchGroup :
                public DataHandler,
                public DataProducer_<eDatasetType,eDataset,eGroup>,
                public IMetricsCalculator_<eDatasetType>,
                public IDataCounter_<eGroup> {
            virtual ~WorkBatchGroup() = default;
            virtual eDatasetTypeList getDatasetType() const override final {return eDatasetType;}
            virtual eDatasetList getDataset() const override final {return eDataset;}
            virtual bool isBare() const override final {return m_bIsBare;}
            virtual bool isGroup() const override final {return true;}
            virtual IDataHandlerPtrArray getBatches() const override final {return m_vpBatches;}
            virtual void startPrecaching(bool bPrecacheGT, size_t nSuggestedBufferSize=SIZE_MAX) override final {for(const auto& pBatch : getBatches()) pBatch->startPrecaching(bPrecacheGT,nSuggestedBufferSize);}
            virtual void stopPrecaching() override final {for(const auto& pBatch : getBatches()) pBatch->stopPrecaching();}
            virtual double getProcessTime() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getProcessTime();});}
            virtual double getExpectedLoad() const override final {return CxxUtils::accumulateMembers<double,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getExpectedLoad();});}
            virtual size_t getTotPackets() const override final {return CxxUtils::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(),[](const IDataHandlerPtr& p){return p->getTotPackets();});}
            template<typename... Targs>
            static std::shared_ptr<WorkBatchGroup> create(Targs&&... args) {
                struct WorkBatchGroupWrapper : public WorkBatchGroup {
                    WorkBatchGroupWrapper(Targs&&... args) : WorkBatchGroup(std::forward<Targs>(args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
                };
                return std::make_shared<WorkBatchGroupWrapper>(std::forward<Targs>(args)...);
            }
        protected:
            virtual void parseData() override final { for(const auto& pBatch : getBatches()) pBatch->parseData(); }
            WorkBatchGroup(const std::string& sGroupName, std::shared_ptr<IDataset> pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sGroupName,pDataset,sRelativePath+"/"+sGroupName+"/"),m_bIsBare(false) {
                if(!PlatformUtils::string_contains_token(getName(),pDataset->getSkippedDirTokens())) {
                    std::cout << "\tParsing directory '" << pDataset->getDatasetPath()+sRelativePath << "' for work group '" << getName() << "'..." << std::endl;
                    std::vector<std::string> vsWorkBatchPaths;
                    // all subdirs are considered work batch directories (if none, the category directory itself is a batch, and 'bare')
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
                const std::string& sDatasetName, // user-friendly dataset name (used for identification only)
                const std::string& sDatasetDirName, // dataset directory name in the dataset root path (the latter is set in CMake)
                const std::string& sOutputDirPath, // output directory (full) path for debug logs, evaluation reports and results archiving
                const std::string& sOutputNamePrefix, // output name prefix for results archiving (if null, only packet idx will be used as file name)
                const std::string& sOutputNameSuffix, // output name suffix for results archiving (if null, no file extension will be used)
                const std::vector<std::string>& vsWorkBatchDirs, // array of directory names for top-level work batch groups
                const std::vector<std::string>& vsSkippedDirTokens, // array of tokens which allow directories to be skipped if one is found in their name
                const std::vector<std::string>& vsGrayscaleDirTokens, // array of tokens which allow directories to be treated as grayscale input only if one is found in their name
                size_t nOutputIdxOffset, // output packet idx offset value used when archiving results
                bool bSaveOutput, // defines whether results should be archived or not
                bool bUseEvaluator, // defines whether results should be fully evaluated, or simply acknowledged
                bool bForce4ByteDataAlign, // defines whether data packets should be 4-byte aligned (useful for GPU upload)
                double dScaleFactor // defines the scale factor to use to resize/rescale read packets
        ) :
                m_sDatasetName(sDatasetName),
                m_sDatasetPath(std::string(DATASET_ROOT)+"/"+sDatasetDirName+"/"),
                m_sOutputPath(sOutputDirPath),
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

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, ParallelUtils::eParallelAlgoType eEvalImpl>
    struct Dataset_ : public IDataset_<eDatasetType,eDataset,eEvalImpl> {
        // if the dataset type/id is not specialized, this redirects creation to default IDataset_ constructor
        using IDataset_<eDatasetType,eDataset,eEvalImpl>::IDataset_;
    };

#define __LITIV_DATASETS_IMPL_H
#include "litiv/datasets/impl/BSDS500.hpp"
#include "litiv/datasets/impl/CDnet.hpp"
//#include "litiv/datasets/impl/LITIV2012b.hpp"  @@@@ still need to work on interfaces for eDatasetType_VideoRegistr
#include "litiv/datasets/impl/PETS2001.hpp"
#include "litiv/datasets/impl/Wallflower.hpp"
#undef __LITIV_DATASETS_IMPL_H

    namespace datasets {

        template<eDatasetTypeList eDatasetType, eDatasetList eDataset, ParallelUtils::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args) {
            struct DatasetWrapper : public Dataset_<eDatasetType,eDataset,eEvalImpl> {
                DatasetWrapper(Targs&&... args) : Dataset_<eDatasetType,eDataset,eEvalImpl>(std::forward<Targs>(args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            IDatasetPtr pDataset = std::make_shared<DatasetWrapper>(std::forward<Targs>(args)...);
            pDataset->parseDataset();
            return pDataset;
        }

        template<eDatasetTypeList eDatasetType, ParallelUtils::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args) {
            return create<eDatasetType,getCustomDatasetEnum(eDatasetType),eEvalImpl>(std::forward<Targs>(args)...);
        }

    } //namespace datasets

} //namespace litiv

