
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
#include "litiv/datasets/metrics.hpp"
#include "litiv/datasets/eval.hpp"

namespace lv {

    namespace datasets {

        /// global dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<eDatasetTaskList eDatasetTask, eDatasetList eDataset, lv::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args);
        /// global dataset object creation method (uses 'custom' dataset interface, forwards extra args to dataset constructor)
        template<eDatasetTaskList eDatasetTask, lv::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args);

    } // namespace datasets

    /// full implementation of basic data handler interface functions (used in work batch & group impl)
    struct DataHandler : public virtual IDataHandler {
        /// returns the work batch/group name
        virtual const std::string& getName() const override final;
        /// returns the work batch/group data path
        virtual const std::string& getDataPath() const override final;
        /// returns the work batch/group output path
        virtual const std::string& getOutputPath() const override final;
        /// returns the work batch/group relative path offset w.r.t. dataset root
        virtual const std::string& getRelativePath() const override final;
        /// returns whether the work batch/group data will be treated as grayscale
        virtual bool isGrayscale() const override final;
        /// returns a pointer to this work batch/group's parent dataset interface
        virtual IDatasetPtr getDatasetInfo() const override final;
    protected:
        /// fills internal impl parameters based on batch name, dataset parameters & current relative data path
        DataHandler(const std::string& sBatchName, std::shared_ptr<IDataset> pDataset, const std::string& sRelativePath);
        /// returns the children batch associated with the given packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const override final;
        /// returns the children batch associated with the given packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) override final;
    private:
        const std::string m_sBatchName;
        const std::string m_sRelativePath;
        const std::string m_sDataPath;
        const std::string m_sOutputPath;
        const bool m_bForcingGrayscale;
        const IDatasetPtr m_pDataset;
    };

    /// top-level dataset interface where work batches & groups are implemented based on template policies --- all internal methods can be overriden via dataset impl headers
    template<eDatasetTaskList eDatasetTask, eDatasetSourceList eDatasetSource, eDatasetList eDataset, eDatasetEvalList eDatasetEval, lv::eParallelAlgoType eEvalImpl>
    struct IDataset_ : public DatasetEvaluator_<eDatasetEval,eDataset> {
        /// fully implemented work batch interface with template specializations
        struct WorkBatch :
                public DataHandler,
                public DataProducer_<eDatasetTask,eDatasetSource,eDataset>,
                public DataEvaluator_<eDatasetEval,eDataset,eEvalImpl> {
            /// default destructor, should stay public so smart pointers can access it
            virtual ~WorkBatch() = default;
            /// returns the currently implemented task type for this work batch
            virtual eDatasetTaskList getDatasetTask() const override final {return eDatasetTask;}
            /// returns the currently implemented source type for this work batch
            virtual eDatasetSourceList getDatasetSource() const override final {return eDatasetSource;}
            /// returns the currently implemented dataset type for this work batch
            virtual eDatasetList getDataset() const override final {return eDataset;}
            /// returns the currently implemented evaluation type for this work batch
            virtual eDatasetEvalList getDatasetEval() const override final {return eDatasetEval;}
            /// always returns false for non-group work batches
            virtual bool isBare() const override final {return false;}
            /// always returns false for non-group work batches
            virtual bool isGroup() const override final {return false;}
            /// always returns an empty data handler array for non-group work batches
            virtual IDataHandlerPtrArray getBatches(bool /*bWithHierarchy*/) const override final {return IDataHandlerPtrArray();}
            /// returns the current (or final) duration elapsed between start/stopProcessing calls
            virtual double getProcessTime() const override final {return m_dElapsedTime_sec;}
            /// returns whether the work batch is still being processed or not (i.e. between start/stopProcessing calls)
            virtual bool isProcessing() const override final {return m_bIsProcessing;}
            /// sets the work batch in 'processing' mode, initializing timers, packet counters and other time-critical evaluation components (if any)
            void startProcessing() {
                if(!m_bIsProcessing) {
                    _startProcessing();
                    m_oStopWatch.tick();
                    m_bIsProcessing = true;
                }
            }
            /// exits 'processing' mode, releasing time-critical evaluation components (if any) and setting the processed packets promise
            void stopProcessing() {
                if(m_bIsProcessing) {
                    m_dElapsedTime_sec = m_oStopWatch.tock();
                    m_bIsProcessing = false;
                    _stopProcessing();
                    this->stopAsyncPrecaching();
                    this->setProcessedPacketsPromise();
                }
            }
            /// work batch object creation method with dataset impl specialization (forwards extra args to work batch constructor)
            template<typename... Targs>
            static std::shared_ptr<WorkBatch> create(Targs&&... args) {
                struct WorkBatchWrapper : public WorkBatch {
                    WorkBatchWrapper(Targs&&... args) : WorkBatch(std::forward<Targs>(args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
                };
                return std::make_shared<WorkBatchWrapper>(std::forward<Targs>(args)...);
            }
        protected:
            /// work batch default constructor (protected, objects should always be instantiated via 'create' member function)
            WorkBatch(const std::string& sBatchName, IDatasetPtr pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sBatchName,pDataset,sRelativePath),m_dElapsedTime_sec(0),m_bIsProcessing(false) {parseData();}
            WorkBatch& operator=(const WorkBatch&) = delete;
            WorkBatch(const WorkBatch&) = delete;
            lv::StopWatch m_oStopWatch;
            double m_dElapsedTime_sec;
            bool m_bIsProcessing;
        };
        /// fully implemented work group interface with template specializations
        struct WorkBatchGroup :
                public DataHandler,
                public DataCounter_<eGroup>,
                public DataReporter_<eDatasetEval,eDataset> {
            /// default destructor, should stay public so smart pointers can access it
            virtual ~WorkBatchGroup() = default;
            /// returns which processing task this work batch/group was built for
            virtual eDatasetTaskList getDatasetTask() const override final {return eDatasetTask;}
            /// returns which data source this work batch/group was built for
            virtual eDatasetSourceList getDatasetSource() const override final {return eDatasetSource;}
            /// returns which dataset this work batch/group was built for
            virtual eDatasetList getDataset() const override final {return eDataset;}
            /// returns which evaluation method this work batch/group was built for
            virtual eDatasetEvalList getDatasetEval() const override final {return eDatasetEval;}
            /// returns whether the work group is a pass-through container
            virtual bool isBare() const override final {return m_bIsBare;}
            /// always returns true for work groups
            virtual bool isGroup() const override final {return true;}
            /// returns this work group's children (work batch array)
            virtual IDataHandlerPtrArray getBatches(bool bWithHierarchy) const override final {
                if(bWithHierarchy)
                    return m_vpBatches;
                IDataHandlerPtrArray vpBatches;
                std::function<void(const IDataHandlerPtr&)> lPushBatches = [&](const IDataHandlerPtr& pBatch) {
                    if(pBatch->isGroup())
                        for(const auto& pSubBatch : pBatch->getBatches(true))
                            lPushBatches(pSubBatch);
                    else
                        vpBatches.push_back(pBatch);
                };
                for(const auto& pBatch : getBatches(true))
                    lPushBatches(pBatch);
                return vpBatches;
            }
            /// returns whether any of this work group's children batches is currently processing data
            virtual bool isProcessing() const override final {for(const auto& pBatch : getBatches(true)) if(pBatch->isProcessing()) return true; return false;}
            /// returns the current (or final) duration elapsed between start/stopProcessing calls, recursively queried for all children work batches
            virtual double getProcessTime() const override final {return lv::accumulateMembers<double,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessTime();});}
            /// accumulates the expected CPU load for this data batch based on all children work batches load
            virtual double getExpectedLoad() const override final {return lv::accumulateMembers<double,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getExpectedLoad();});}
            /// accumulate total packet count from all children work batches
            virtual size_t getTotPackets() const override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getTotPackets();});}
            /// work group object creation method with dataset impl specialization (forwards extra args to work group constructor)
            template<typename... Targs>
            static std::shared_ptr<WorkBatchGroup> create(Targs&&... args) {
                struct WorkBatchGroupWrapper : public WorkBatchGroup {
                    WorkBatchGroupWrapper(Targs&&... args) : WorkBatchGroup(std::forward<Targs>(args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
                };
                return std::make_shared<WorkBatchGroupWrapper>(std::forward<Targs>(args)...);
            }
        protected:
            /// recursively calls parse data on all childrens
            virtual void parseData() override final { for(const auto& pBatch : getBatches(true)) pBatch->parseData(); }
            /// work group default constructor (protected, objects should always be instantiated via 'create' member function)
            WorkBatchGroup(const std::string& sGroupName, std::shared_ptr<IDataset> pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sGroupName,pDataset,lv::AddDirSlashIfMissing(sRelativePath)+sGroupName+"/"),m_bIsBare(false) {
                if(!lv::string_contains_token(getName(),pDataset->getSkippedDirTokens())) {
                    std::cout << "\tParsing directory '" << pDataset->getDatasetPath()+sRelativePath << "' for work group '" << getName() << "'..." << std::endl;
                    std::vector<std::string> vsWorkBatchPaths;
                    // all subdirs are considered work batch directories (if none, the category directory itself is a batch, and 'bare')
                    lv::GetSubDirsFromDir(getDataPath(),vsWorkBatchPaths);
                    if(vsWorkBatchPaths.empty()) {
                        m_vpBatches.push_back(WorkBatch::create(getName(),pDataset,getRelativePath()));
                        m_bIsBare = true;
                    }
                    else {
                        for(const auto& sPathIter : vsWorkBatchPaths) {
                            const size_t nLastSlashPos = sPathIter.find_last_of("/\\");
                            const std::string sNewBatchName = nLastSlashPos==std::string::npos?sPathIter:sPathIter.substr(nLastSlashPos+1);
                            if(!lv::string_contains_token(sNewBatchName,pDataset->getSkippedDirTokens()))
                                m_vpBatches.push_back(WorkBatch::create(sNewBatchName,pDataset,lv::AddDirSlashIfMissing(getRelativePath())+sNewBatchName+"/"));
                        }
                    }
                }
            }
            WorkBatchGroup& operator=(const WorkBatchGroup&) = delete;
            WorkBatchGroup(const WorkBatchGroup&) = delete;
            IDataHandlerPtrArray m_vpBatches;
            bool m_bIsBare;
        };
        /// returns the dataset name
        virtual const std::string& getName() const override final {return m_sDatasetName;}
        /// returns the root data path
        virtual const std::string& getDatasetPath() const override final {return m_sDatasetPath;}
        /// returns the root output path
        virtual const std::string& getOutputPath() const override final {return m_sOutputPath;}
        /// returns the output file name prefix for results archiving
        virtual const std::string& getOutputNamePrefix() const override final {return m_sOutputNamePrefix;}
        /// returns the output file name suffix for results archiving
        virtual const std::string& getOutputNameSuffix() const override final {return m_sOutputNameSuffix;}
        /// returns the directory names of top-level work batches
        virtual const std::vector<std::string>& getWorkBatchDirs() const override final {return m_vsWorkBatchDirs;}
        /// returns the directory name tokens which, if found, should be skipped
        virtual const std::vector<std::string>& getSkippedDirTokens() const override final {return m_vsSkippedDirTokens;}
        /// returns the directory name tokens which, if found, should be treated as grayscale
        virtual const std::vector<std::string>& getGrayscaleDirTokens() const override final {return m_vsGrayscaleDirTokens;}
        /// returns the output file/packet index offset for results archiving
        virtual size_t getOutputIdxOffset() const override final {return m_nOutputIdxOffset;}
        /// returns the input data scaling scaling factor
        virtual double getScaleFactor() const override final {return m_dScaleFactor;}
        /// returns whether we should save the results through DataConsumers or not
        virtual bool isSavingOutput() const override final {return m_bSavingOutput;}
        /// returns whether we should evaluate the results through DataConsumers or not
        virtual bool isUsingEvaluator() const override final {return m_bUsingEvaluator;}
        /// returns whether loaded data should be 4-byte aligned or not (4-byte alignment is ideal for GPU upload)
        virtual bool is4ByteAligned() const override final {return m_bForce4ByteDataAlign;}
        /// returns the total number of packets in the dataset (recursively queried from work batches)
        virtual size_t getTotPackets() const override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getTotPackets();});}
        /// returns the total time it took to process the dataset (recursively queried from work batches)
        virtual double getProcessTime() const override final {return lv::accumulateMembers<double,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessTime();});}
        /// returns the total processed packet count, blocking if processing is not finished yet (recursively queried from work batches)
        virtual size_t getProcessedPacketsCountPromise() override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCountPromise();});}
        /// returns the total processed packet count (recursively queried from work batches)
        virtual size_t getProcessedPacketsCount() const override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessedPacketsCount();});}
        /// clears all batches and reparses them from the dataset metadata
        virtual void parseDataset() override final {
            std::cout << "Parsing dataset '" << getName() << "'..." << std::endl;
            m_vpBatches.clear();
            if(!getOutputPath().empty())
                lv::CreateDirIfNotExist(getOutputPath());
            for(const auto& sPathIter : getWorkBatchDirs())
                m_vpBatches.push_back(WorkBatchGroup::create(sPathIter,this->shared_from_this()));
        }
        /// returns the array of work batches (or groups) contained in this dataset
        virtual IDataHandlerPtrArray getBatches(bool bWithHierarchy) const override final {
            if(bWithHierarchy)
                return m_vpBatches;
            IDataHandlerPtrArray vpBatches;
            for(const auto& pBatch : getBatches(true))
                if(pBatch->isGroup())
                    for(const auto& pSubBatch : pBatch->getBatches(false))
                        vpBatches.push_back(pSubBatch);
                else
                    vpBatches.push_back(pBatch);
            return vpBatches;
        }
        /// returns the array of work batches (or groups) contained in this dataset, sorted by expected CPU load
        virtual IDataHandlerPtrQueue getSortedBatches(bool bWithHierarchy) const override final {
            IDataHandlerPtrQueue vpBatches(&IDataHandler::compare_load<IDataHandler>);
            for(const auto& pBatch : getBatches(bWithHierarchy))
                vpBatches.push(pBatch);
            return vpBatches;
        }
    protected:
        /// full dataset constructor; see individual parameter comments for descriptions
        IDataset_(
                const std::string& sDatasetName, ///< user-friendly dataset name (used for identification only)
                const std::string& sDatasetDirPath, ///< dataset directory (full) path where work batches can be found
                const std::string& sOutputDirPath, ///< output directory (full) path for debug logs, evaluation reports and results archiving
                const std::string& sOutputNamePrefix, ///< output name prefix for results archiving (if null, only packet idx will be used as file name)
                const std::string& sOutputNameSuffix, ///< output name suffix for results archiving (if null, no file extension will be used)
                const std::vector<std::string>& vsWorkBatchDirs, ///< array of directory names for top-level work batch groups (one group typically contains multiple work batches)
                const std::vector<std::string>& vsSkippedDirTokens, ///< array of tokens which allow directories to be skipped if one is found in their name
                const std::vector<std::string>& vsGrayscaleDirTokens, ///< array of tokens which allow directories to be treated as grayscale input only if one is found in their name
                size_t nOutputIdxOffset, ///< output packet idx offset value used when archiving results
                bool bSaveOutput, ///< defines whether results should be archived or not
                bool bUseEvaluator, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bForce4ByteDataAlign, ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
                double dScaleFactor ///< defines the scale factor to use to resize/rescale read packets
        ) :
                m_sDatasetName(sDatasetName),
                m_sDatasetPath(lv::AddDirSlashIfMissing(sDatasetDirPath)),
                m_sOutputPath(lv::AddDirSlashIfMissing(sOutputDirPath)),
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

    /// returns the eval type policy to use based on the dataset task type (can also be overriden by dataset type)
    template<eDatasetTaskList eDatasetTask, eDatasetList eDataset>
    constexpr eDatasetEvalList getDatasetEval() {
        // note: these are only defaults, they can be overriden via full specialization in their impl header
        return (eDatasetTask==eDatasetTask_ChgDet)?eDatasetEval_BinaryClassifier:
               (eDatasetTask==eDatasetTask_Segm)?eDatasetEval_Segm:
               (eDatasetTask==eDatasetTask_Registr)?eDatasetEval_Registr:
               (eDatasetTask==eDatasetTask_EdgDet)?eDatasetEval_BinaryClassifier:
               // ...
               throw -1; // undefined behavior
    }

    /// returns the source type policy to use based on the dataset task type (can also be overriden by dataset type)
    template<eDatasetTaskList eDatasetTask, eDatasetList eDataset>
    constexpr eDatasetSourceList getDatasetSource() {
        // note: these are only defaults, they can be overriden via full specialization in their impl header
        return (eDatasetTask==eDatasetTask_ChgDet)?eDatasetSource_Video:
               (eDatasetTask==eDatasetTask_Segm)?eDatasetSource_Video:
               (eDatasetTask==eDatasetTask_Registr)?eDatasetSource_VideoArray:
               (eDatasetTask==eDatasetTask_EdgDet)?eDatasetSource_Image:
               // ...
               throw -1; // undefined behavior
    }

    /// dataset interface that must be specialized based on task & eval types, and dataset (in impl headers, if required)
    template<eDatasetTaskList eDatasetTask, eDatasetList eDataset, lv::eParallelAlgoType eEvalImpl>
    struct Dataset_;

    #define __LITIV_DATASETS_IMPL_H
    #include "litiv/datasets/impl/BSDS500.hpp"
    #include "litiv/datasets/impl/CDnet.hpp"
    //#include "litiv/datasets/impl/LITIV2012b.hpp"  @@@@ still need to work on interfaces for eDatasetType_VideoRegistr
    #include "litiv/datasets/impl/PETS2001.hpp"
    #include "litiv/datasets/impl/Wallflower.hpp"
    #undef __LITIV_DATASETS_IMPL_H

    /// default dataset interface implementation w/ default specialization & constructor pass-through
    template<eDatasetTaskList eDatasetTask, eDatasetList eDataset, lv::eParallelAlgoType eEvalImpl>
    struct Dataset_ : public IDataset_<eDatasetTask,getDatasetSource<eDatasetTask,eDataset>(),eDataset,getDatasetEval<eDatasetTask,eDataset>(),eEvalImpl> {
        // if the task/dataset is not specialized, this redirects creation to the default IDataset_ constructor
        using IDataset_<eDatasetTask,getDatasetSource<eDatasetTask,eDataset>(),eDataset,getDatasetEval<eDatasetTask,eDataset>(),eEvalImpl>::IDataset_;
    };

    namespace datasets {

        /// global dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<eDatasetTaskList eDatasetTask, eDatasetList eDataset, lv::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args) {
            struct DatasetWrapper : public Dataset_<eDatasetTask,eDataset,eEvalImpl> {
                DatasetWrapper(Targs&&... args) : Dataset_<eDatasetTask,eDataset,eEvalImpl>(std::forward<Targs>(args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            IDatasetPtr pDataset = std::make_shared<DatasetWrapper>(std::forward<Targs>(args)...);
            pDataset->parseDataset();
            return pDataset;
        }
        /// global dataset object creation method (uses 'custom' dataset interface, forwards extra args to dataset constructor)
        template<eDatasetTaskList eDatasetTask, lv::eParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args) {
            return create<eDatasetTask,eDataset_Custom,eEvalImpl>(std::forward<Targs>(args)...);
        }

    } // namespace datasets

} // namespace lv
