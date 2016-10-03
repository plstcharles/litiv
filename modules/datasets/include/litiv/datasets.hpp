
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

        /// returns the path where datasets should be found on the system (the default is given by the EXTERNAL_DATA_ROOT cmake variable)
        const std::string& getDatasetsRootPath();
        /// sets the path where datasets should be found on the system (will be kept using a global variable)
        void setDatasetsRootPath(const std::string& sNewPath);

        /// global dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args);
        /// global dataset object creation method (uses 'custom' dataset interface, forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl, typename... Targs>
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
        /// returns the children batch associated with the given (input) packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const override final;
        /// returns the children batch associated with the given (input) packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) override final;
    private:
        const std::string m_sBatchName;
        const std::string m_sRelativePath;
        const std::string m_sDataPath;
        const std::string m_sOutputPath;
        const bool m_bForcingGrayscale;
        const IDatasetPtr m_pDataset;
    };

    /// top-level dataset interface where work batches & groups are implemented based on template policies --- all internal methods can be overridden via dataset impl headers
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset, DatasetEvalList eDatasetEval, lv::ParallelAlgoType eEvalImpl>
    struct IDataset_ : public DatasetReporter_<eDatasetEval,eDataset> {
        static_assert(lv::isDatasetSpecValid<eDatasetTask,eDatasetSource,eDataset,eDatasetEval>(),"dataset does not support the required task/source/eval combo");

        /// static dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<typename... Targs>
        static inline IDatasetPtr create(Targs&&... args) {
            return lv::datasets::create<eDatasetTask,eDataset,eEvalImpl>(std::forward<Targs>(args)...);
        }

        /// work batch group implementation forward declaration (required before friending, as it hides some templates from top class)
        struct WorkBatchGroup;

        /// fully implemented work batch interface with template specializations
        struct WorkBatch :
                public DataHandler,
                public DataProducer_<eDatasetTask,eDatasetSource,eDataset>,
                public DataEvaluator_<eDatasetEval,eDataset,eEvalImpl> {
            /// default destructor, should stay public so smart pointers can access it
            virtual ~WorkBatch() = default;
            /// returns the currently implemented task type for this work batch
            virtual DatasetTaskList getDatasetTask() const override final {return eDatasetTask;}
            /// returns the currently implemented source type for this work batch
            virtual DatasetSourceList getDatasetSource() const override final {return eDatasetSource;}
            /// returns the currently implemented evaluation type for this work batch
            virtual DatasetEvalList getDatasetEval() const override final {return eDatasetEval;}
            /// returns the currently implemented dataset type for this work batch
            virtual DatasetList getDataset() const override final {return eDataset;}
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
            inline void startProcessing() {
                if(!this->m_bIsProcessing) {
                    this->startProcessing_impl();
                    this->m_bIsProcessing = true;
                    this->m_oStopWatch.tick();
                }
            }
            /// exits 'processing' mode, releasing time-critical evaluation components (if any) and setting the processed packets promise
            inline void stopProcessing() {
                if(this->m_bIsProcessing) {
                    this->m_dElapsedTime_sec = this->m_oStopWatch.tock();
                    this->stopProcessing_impl();
                    this->m_bIsProcessing = false;
                    this->stopAsyncPrecaching();
                    this->setProcessedOutputCountPromise();
                }
            }
        protected:
            /// work batch default constructor (protected, objects should always be instantiated via 'create' member function)
            WorkBatch(const std::string& sBatchName, IDatasetPtr pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sBatchName,pDataset,sRelativePath),m_dElapsedTime_sec(0),m_bIsProcessing(false) {parseData();}
            WorkBatch& operator=(const WorkBatch&) = delete;
            WorkBatch(const WorkBatch&) = delete;
            friend struct WorkBatchGroup;
            lv::StopWatch m_oStopWatch;
            double m_dElapsedTime_sec;
            bool m_bIsProcessing;
        };

        /// fully implemented work group interface with template specializations
        struct WorkBatchGroup :
                public DataHandler,
                public GroupDataParser_<eDatasetTask,eDatasetSource,eDataset>,
                public IDataCounter_<Group>,
                public DataReporter_<eDatasetEval,eDataset> {
            /// default destructor, should stay public so smart pointers can access it
            virtual ~WorkBatchGroup() = default;
            /// returns which processing task this work batch/group was built for
            virtual DatasetTaskList getDatasetTask() const override final {return eDatasetTask;}
            /// returns which data source this work batch/group was built for
            virtual DatasetSourceList getDatasetSource() const override final {return eDatasetSource;}
            /// returns which evaluation method this work batch/group was built for
            virtual DatasetEvalList getDatasetEval() const override final {return eDatasetEval;}
            /// returns which dataset this work batch/group was built for
            virtual DatasetList getDataset() const override final {return eDataset;}
        protected:
            /// creates and returns a work batch for a given relative dataset path
            virtual IDataHandlerPtr createWorkBatch(const std::string& sBatchName, const std::string& sRelativePath=std::string("./")) const override {
                return std::shared_ptr<WorkBatch>(new WorkBatch(sBatchName,getDatasetInfo(),sRelativePath));
            }
            /// work group default constructor (protected, objects should always be instantiated via 'create' member function)
            WorkBatchGroup(const std::string& sGroupName, std::shared_ptr<IDataset> pDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sGroupName,pDataset,lv::AddDirSlashIfMissing(sRelativePath)+sGroupName+"/") {parseData();}
            WorkBatchGroup& operator=(const WorkBatchGroup&) = delete;
            WorkBatchGroup(const WorkBatchGroup&) = delete;
            friend struct IDataset_<eDatasetTask,eDatasetSource,eDataset,eDatasetEval,eEvalImpl>;
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
        /// returns the total input packet count in the dataset (recursively queried from work batches)
        virtual size_t getInputCount() const override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getInputCount();});}
        /// returns the total gt packet count in the dataset (recursively queried from work batches)
        virtual size_t getGTCount() const override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getGTCount();});}
        /// returns the total output packet count expected to be processed by the dataset evaluator (recursively queried from work batches)
        virtual size_t getExpectedOutputCount() const override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getExpectedOutputCount();});}
        /// returns the total output packet count so far processed by the dataset evaluator (recursively queried from work batches)
        virtual size_t getProcessedOutputCount() const override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessedOutputCount();});}
        /// returns the total output packet count so far processed by the dataset evaluator, blocking if processing is not finished yet (recursively queried from work batches)
        virtual size_t getProcessedOutputCountPromise() override final {return lv::accumulateMembers<size_t,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessedOutputCountPromise();});}
        /// returns the total time it took to process the dataset (recursively queried from work batches)
        virtual double getProcessTime() const override final {return lv::accumulateMembers<double,IDataHandlerPtr>(getBatches(true),[](const IDataHandlerPtr& p){return p->getProcessTime();});}
        /// clears all batches and reparses them from the dataset metadata
        virtual void parseDataset() override final {
            std::cout << "Parsing dataset '" << getName() << "'..." << std::endl;
            m_vpBatches.clear();
            if(!getOutputPath().empty())
                lv::CreateDirIfNotExist(getOutputPath());
            for(const auto& sPathIter : getWorkBatchDirs())
                m_vpBatches.push_back(std::shared_ptr<WorkBatchGroup>(new WorkBatchGroup(sPathIter,this->shared_from_this())));
        }
        /// returns the array of work batches (or groups) contained in this dataset
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
        /// returns the array of work batches (or groups) contained in this dataset, sorted by expected CPU load
        virtual IDataHandlerPtrQueue getSortedBatches(bool bWithHierarchy) const override final {
            IDataHandlerPtrQueue vpBatches(&IDataHandler::compare_load<IDataHandler>);
            for(const auto& pBatch : getBatches(bWithHierarchy))
                vpBatches.push(pBatch);
            return vpBatches;
        }
    protected:
        /// full dataset constructor; parameters are passed through lv::datasets::create<...>(...), and may be caught/simplified by a specialization
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

    /// dataset interface that must be specialized based on task & eval types, and dataset (in impl headers, if required)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_;

} // namespace lv

#define _LITIV_DATASETS_IMPL_H_
// will include all specializations of 'Dataset_<...>' and its interfaces
#include "litiv/datasets/impl/all.hpp"
#undef _LITIV_DATASETS_IMPL_H_

namespace lv {

    /// dataset full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_ : public IDataset_<eDatasetTask,getDatasetSource<eDatasetTask,eDataset>(),eDataset,getDatasetEval<eDatasetTask,eDataset>(),eEvalImpl> {
        // if the task/dataset is not specialized, this redirects creation to the default IDataset_ constructor
        using IDataset_<eDatasetTask,getDatasetSource<eDatasetTask,eDataset>(),eDataset,getDatasetEval<eDatasetTask,eDataset>(),eEvalImpl>::IDataset_;
    };

    namespace datasets {

        /// global dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args) {
            struct DatasetWrapper : public Dataset_<eDatasetTask,eDataset,eEvalImpl> {
                DatasetWrapper(Targs&&... args) : Dataset_<eDatasetTask,eDataset,eEvalImpl>(std::forward<Targs>(args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            IDatasetPtr pDataset = std::make_shared<DatasetWrapper>(std::forward<Targs>(args)...);
            pDataset->parseDataset();
            return pDataset;
        }
        /// global dataset object creation method (uses 'custom' dataset interface, forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl, typename... Targs>
        IDatasetPtr create(Targs&&... args) {
            return create<eDatasetTask,Dataset_Custom,eEvalImpl>(std::forward<Targs>(args)...);
        }

    } // namespace datasets

} // namespace lv
