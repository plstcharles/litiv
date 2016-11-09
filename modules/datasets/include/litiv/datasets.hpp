
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

    /// dataset interface specialization forward declaration
    template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_;

    /// pointer type for highest-level dataset specialization implementation
    template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl>
    using DatasetPtr_ = std::shared_ptr<Dataset_<eDatasetTask,eDataset,eEvalImpl>>;

    namespace datasets {

        /// returns the path where datasets should be found on the system (the default is given by the EXTERNAL_DATA_ROOT cmake variable)
        const std::string& getDatasetsRootPath();
        /// sets the path where datasets should be found on the system (will be kept using a global variable)
        void setDatasetsRootPath(const std::string& sNewPath);

        /// global dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl, typename... Targs>
        DatasetPtr_<eDatasetTask,eDataset,eEvalImpl> create(Targs&&... args);
        /// global dataset object creation method (uses 'custom' dataset interface, forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl, typename... Targs>
        DatasetPtr_<eDatasetTask,Dataset_Custom,eEvalImpl> create(Targs&&... args);

    } // namespace datasets

    /// top-level dataset interface where work batches & groups are implemented based on template policies --- all internal methods can be overridden via dataset impl headers
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset, DatasetEvalList eDatasetEval, lv::ParallelAlgoType eEvalImpl>
    struct IDataset_ :
            public DatasetHandler_<eDatasetTask,eDatasetSource,eDataset>,
            public DataGroupHandler_<eDatasetTask,eDatasetSource,eDataset>,
            public DataTemplSpec_<eDatasetTask,eDatasetSource,eDataset,eDatasetEval>,
            public DatasetReporter_<eDatasetEval,eDataset> {
        static_assert(lv::isDatasetSpecValid<eDatasetTask,eDatasetSource,eDataset,eDatasetEval>(),"dataset does not support the required task/source/eval combo");

        /// static dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<typename... Targs>
        static inline DatasetPtr_<eDatasetTask,eDataset,eEvalImpl> create(Targs&&... args) {
            return lv::datasets::create<eDatasetTask,eDataset,eEvalImpl>(std::forward<Targs>(args)...);
        }

        /// internal specialization of dataset interface pointer
        using Ptr = DatasetPtr_<eDatasetTask,eDataset,eEvalImpl>;

        /// work batch group implementation forward declaration (required before friending, as it hides some templates from top class)
        struct WorkBatchGroup;

        /// fully implemented+specialized work batch for the current dataset specialization
        struct WorkBatch :
                public DataHandler_<eDatasetTask,eDatasetSource,eDataset>,
                public DataProducer_<eDatasetTask,eDatasetSource,eDataset>,
                public DataTemplSpec_<eDatasetTask,eDatasetSource,eDataset,eDatasetEval>,
                public DataEvaluator_<eDatasetEval,eDataset,eEvalImpl> {
            /// default destructor, should stay public so smart pointers can access it
            virtual ~WorkBatch() = default;
            /// returns the time taken so far to process the work batch data (i.e. between start/stopProcessing calls)
            virtual double getCurrentProcessTime() const override final {return this->m_bIsProcessing?this->m_oStopWatch.elapsed():this->m_dFinalElapsedTime;}
            /// returns the final time taken to process the work batch data (i.e. between start/stopProcessing calls)
            virtual double getFinalProcessTime() override final {return this->m_dElapsedTimeFuture.valid()?(this->m_dFinalElapsedTime=this->m_dElapsedTimeFuture.get()):this->m_dFinalElapsedTime;}
            /// returns whether the work batch is still being processed or not (i.e. between start/stopProcessing calls)
            virtual bool isProcessing() const override final {return m_bIsProcessing;}
            /// always returns false for non-group work batches
            virtual bool isBare() const override final {return false;}
            /// always returns false for non-group work batches
            virtual bool isGroup() const override final {return false;}
            /// always returns an empty data handler array for non-group work batches
            virtual IDataHandlerPtrArray getBatches(bool /*bWithHierarchy*/) const override final {return IDataHandlerPtrArray();}
            /// sets the work batch in 'processing' mode, initializing timers, packet counters and other time-critical evaluation components (if any)
            inline void startProcessing() {
                lvDbgExceptionWatch;
                if(!this->m_bIsProcessing) {
                    m_dElapsedTimePromise = std::promise<double>();
                    m_dElapsedTimeFuture = m_dElapsedTimePromise.get_future();
                    m_dFinalElapsedTime = 0.0;
                    this->resetOutputCount();
                    this->startProcessing_impl();
                    this->m_bIsProcessing = true;
                    this->m_oStopWatch.tick();
                }
            }
            /// exits 'processing' mode, releasing time-critical evaluation components (if any) and setting the processed packets promise
            inline void stopProcessing() {
                lvDbgExceptionWatch;
                if(this->m_bIsProcessing) {
                    this->m_dElapsedTimePromise.set_value(this->m_oStopWatch.tock());
                    this->getFinalProcessTime();
                    this->setOutputCountPromise();
                    this->stopProcessing_impl();
                    this->m_bIsProcessing = false;
                    this->stopPrecaching();
                }
            }
        protected:
            /// work batch instances can only be created by work groups via their protected 'createWorkBatch' function
            WorkBatch(const std::string& sBatchName, const std::string& sRelativePath, const IDataHandler& oParent) :
                    DataHandler_<eDatasetTask,eDatasetSource,eDataset>(sBatchName,sRelativePath,oParent),
                    m_dFinalElapsedTime(0),m_bIsProcessing(false) {}
            WorkBatch& operator=(const WorkBatch&) = delete;
            WorkBatch(const WorkBatch&) = delete;
            friend struct WorkBatchGroup;
            lv::StopWatch m_oStopWatch;
            std::promise<double> m_dElapsedTimePromise;
            std::future<double> m_dElapsedTimeFuture;
            double m_dFinalElapsedTime; ///< returns final time elapsed between start/stop processing calls (always in seconds)
            bool m_bIsProcessing; ///< returns whether in between start/stop processing calls
        };

        /// fully implemented+specialized work batch group for the current dataset specialization
        struct WorkBatchGroup :
                public DataHandler_<eDatasetTask,eDatasetSource,eDataset>,
                public DataGroupHandler_<eDatasetTask,eDatasetSource,eDataset>,
                public DataTemplSpec_<eDatasetTask,eDatasetSource,eDataset,eDatasetEval>,
                public DataReporter_<eDatasetEval,eDataset> {
            /// default destructor, should stay public so smart pointers can access it
            virtual ~WorkBatchGroup() = default;

        protected:
            /// creates and returns a work batch for a given relative dataset path
            virtual IDataHandlerPtr createWorkBatch(const std::string& sBatchName, const std::string& sRelativePath) const override {
                lvDbgExceptionWatch;
                static_assert((!std::is_abstract<WorkBatch>::value),"Work batch class must be non-abstract (check for missing virtual pure impls in interface specializations)");
                auto p = std::shared_ptr<WorkBatch>(new WorkBatch(sBatchName,sRelativePath,*this));
                p->parseData();
                return p;
            }
            /// work group instances can only be created by dataset handlers
            WorkBatchGroup(const std::string& sGroupName, const std::string& sRelativePath, const IDataHandler& oParent) :
                    DataHandler_<eDatasetTask,eDatasetSource,eDataset>(sGroupName,sRelativePath,oParent) {}
            WorkBatchGroup& operator=(const WorkBatchGroup&) = delete;
            WorkBatchGroup(const WorkBatchGroup&) = delete;
            friend struct IDataset_<eDatasetTask,eDatasetSource,eDataset,eDatasetEval,eEvalImpl>;
            bool m_bIsBare;
        };

        /// creates and returns a work batch group for a given relative dataset path
        virtual IDataHandlerPtr createWorkBatch(const std::string& sBatchName, const std::string& sRelativePath) const override {
            lvDbgExceptionWatch;
            static_assert((!std::is_abstract<WorkBatchGroup>::value),"Work batch group class must be non-abstract (check for missing virtual pure impls in interface specializations)");
            auto p = std::shared_ptr<WorkBatchGroup>(new WorkBatchGroup(sBatchName,sRelativePath,*this));
            p->parseData();
            return p;
        }
        /// clears all batches and reparses them from the dataset metadata
        virtual void parseData() override final {
            lvDbgExceptionWatch;
            std::cout << "Parsing directory '" << this->getDataPath() << "' for dataset '" << this->getName() << "'..." << std::endl;
            this->m_vpBatches.clear();
            this->m_bIsBare = false; // always false by default for top level
            if(!this->getOutputPath().empty())
                lv::CreateDirIfNotExist(this->getOutputPath());
            for(const auto& sPathIter : this->getWorkBatchDirs())
                this->m_vpBatches.push_back(createWorkBatch(sPathIter,lv::AddDirSlashIfMissing(sPathIter)));
        }
    protected:
        /// full dataset constructor (copied from DatasetHandler to avoid msvc2015 bug); parameters are passed through lv::datasets::create<...>(...), and may be caught/simplified by a specialization
		IDataset_(
			const std::string& sDatasetName, ///< user-friendly dataset name (used for identification only)
			const std::string& sDatasetDirPath, ///< dataset directory (full) path where work batches can be found
			const std::string& sOutputDirPath, ///< output directory (full) path for debug logs, evaluation reports and results archiving
			const std::string& sOutputNamePrefix, ///< output name prefix for results archiving (if null, only packet idx will be used as file name)
			const std::string& sOutputNameSuffix, ///< output name suffix for results archiving (if null, no file extension will be used)
			const std::vector<std::string>& vsWorkBatchDirs, ///< array of directory names for top-level work batch groups (one group typically contains multiple work batches)
			const std::vector<std::string>& vsSkippedDirTokens, ///< array of tokens which allow directories to be skipped if one is found in their name
			const std::vector<std::string>& vsGrayscaleDirTokens, ///< array of tokens which allow directories to be treated as grayscale input only if one is found in their name
			bool bSaveOutput, ///< defines whether results should be archived or not
			bool bUseEvaluator, ///< defines whether results should be fully evaluated, or simply acknowledged
			bool bForce4ByteDataAlign, ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
			double dScaleFactor ///< defines the scale factor to use to resize/rescale read packets
		) : DatasetHandler_<eDatasetTask,eDatasetSource,eDataset>(sDatasetName,sDatasetDirPath,sOutputDirPath,sOutputNamePrefix,sOutputNameSuffix,vsWorkBatchDirs,vsSkippedDirTokens,vsGrayscaleDirTokens,bSaveOutput,bUseEvaluator,bForce4ByteDataAlign,dScaleFactor) {}
    };

} // namespace lv

#define _LITIV_DATASETS_IMPL_H_
// will include all specializations of 'Dataset_<...>' and its interfaces
#include "litiv/datasets/impl/all.hpp"
#undef _LITIV_DATASETS_IMPL_H_

namespace lv {

    /// dataset full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_ : public IDataset_<eDatasetTask,lv::getDatasetSource<eDatasetTask,eDataset>(),eDataset,lv::getDatasetEval<eDatasetTask,eDataset>(),eEvalImpl> {
        // if the task/dataset is not specialized, this redirects creation to the default DatasetHandler_ constructor
		using IDataset_<eDatasetTask,lv::getDatasetSource<eDatasetTask,eDataset>(),eDataset,lv::getDatasetEval<eDatasetTask,eDataset>(),eEvalImpl>::IDataset_;
    };

    namespace datasets {

        /// global dataset object creation method with dataset impl specialization (forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, DatasetList eDataset, lv::ParallelAlgoType eEvalImpl, typename... Targs>
        DatasetPtr_<eDatasetTask,eDataset,eEvalImpl> create(Targs&&... args) {
            lvDbgExceptionWatch;
            static_assert((!std::is_abstract<Dataset_<eDatasetTask,eDataset,eEvalImpl>>::value),"Requested dataset class must be non-abstract (check for missing virtual pure impls in interface specializations)");
            struct DatasetWrapper : public Dataset_<eDatasetTask,eDataset,eEvalImpl> {
                DatasetWrapper(Targs&&... _args) : Dataset_<eDatasetTask,eDataset,eEvalImpl>(std::forward<Targs>(_args)...) {} // cant do 'using BaseCstr::BaseCstr;' since it keeps the access level
            };
            auto p = std::make_shared<DatasetWrapper>(std::forward<Targs>(args)...);
            p->parseData();
            return p;
        }
        /// global dataset object creation method (uses 'custom' dataset interface, forwards extra args to dataset constructor)
        template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl, typename... Targs>
        DatasetPtr_<eDatasetTask,Dataset_Custom,eEvalImpl> create(Targs&&... args) {
            return create<eDatasetTask,Dataset_Custom,eEvalImpl>(std::forward<Targs>(args)...);
        }

    } // namespace datasets

} // namespace lv
