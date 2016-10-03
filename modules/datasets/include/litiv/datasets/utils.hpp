
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

#include "litiv/utils/parallel.hpp"
#include "litiv/utils/opencv.hpp"
#include "litiv/utils/platform.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace lv {

    enum DatasetTaskList { // from the task type, we can derive the source and eval types
        DatasetTask_Segm,
        DatasetTask_Cosegm,
        DatasetTask_Registr, // @@@ specialization todo
        DatasetTask_EdgDet,
        // ...
    };

    enum DatasetSourceList { // from the source type, we can derive the input packet policy
        DatasetSource_Video,
        DatasetSource_VideoArray,
        DatasetSource_Image,
        DatasetSource_ImageArray,
        // ...
    };

    enum DatasetEvalList { // from the eval type, we can derive the gt packet mapping policy
        DatasetEval_BinaryClassifier,
        DatasetEval_BinaryClassifierArray,
        DatasetEval_MultiClassifier, // @@@ specialization todo
        DatasetEval_MultiClassifierArray, // @@@ specialization todo
        DatasetEval_Registr, // @@@ specialization todo
        DatasetEval_BoundingBox, // @@@ specialization todo
        // ...
        DatasetEval_None // will only count packets & monitor processing time
    };

    enum DatasetList { // dataset types are used for impl specializations only
        Dataset_CDnet,
        Dataset_Wallflower,
        Dataset_PETS2001D3TC1,
        Dataset_LITIV2012b,
        Dataset_BSDS500,
        Dataset_VAPtrimod2016,
        // ...
        Dataset_Custom // 'datasets::create' will forward all parameters from Dataset constr
    };

    enum GroupPolicy { // used to toggle group policy functions in data handler interfaces
        Group,
        NotGroup,
    };

    enum ArrayPolicy { // used to toggle data array policy functions in data handler interfaces
        Array,
        NotArray,
    };

    enum PacketPolicy { // used to toggle packet policy functions in data handler interfaces
        ImagePacket,
        ImageArrayPacket,
        NotImagePacket
    };

    enum MappingPolicy { // used to determine how data packets (input/output, or gt/output) can be mapped
        PixelMapping,
        IndexMapping,
        BatchMapping,
        NoMapping
    };

    /// returns the gt packet type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr PacketPolicy getGTPacketType() {
        return
                (eDatasetTask==DatasetTask_Segm)?ImagePacket:
                (eDatasetTask==DatasetTask_Cosegm)?ImageArrayPacket:
                (eDatasetTask==DatasetTask_Registr)?NotImagePacket:
                (eDatasetTask==DatasetTask_EdgDet)?ImagePacket:
                // ...
                throw -1; // undefined behavior
    }

    /// returns the output packet type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr PacketPolicy getOutputPacketType() {
        return
                (eDatasetTask==DatasetTask_Segm)?ImagePacket:
                (eDatasetTask==DatasetTask_Cosegm)?ImageArrayPacket:
                (eDatasetTask==DatasetTask_Registr)?NotImagePacket:
                (eDatasetTask==DatasetTask_EdgDet)?ImagePacket:
                // ...
                throw -1; // undefined behavior
    }

    /// returns the GT packet mapping style policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr MappingPolicy getGTMappingType() {
        return
                (eDatasetTask==DatasetTask_Segm)?PixelMapping:
                (eDatasetTask==DatasetTask_Cosegm)?IndexMapping:
                (eDatasetTask==DatasetTask_Registr)?BatchMapping:
                (eDatasetTask==DatasetTask_EdgDet)?PixelMapping:
                // ...
                throw -1; // undefined behavior
    }

    /// returns the I/O packet mapping style policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr MappingPolicy getIOMappingType() {
        return
                (eDatasetTask==DatasetTask_Segm)?PixelMapping:
                (eDatasetTask==DatasetTask_Cosegm)?IndexMapping:
                (eDatasetTask==DatasetTask_Registr)?BatchMapping:
                (eDatasetTask==DatasetTask_EdgDet)?PixelMapping:
                // ...
                throw -1; // undefined behavior
    }

    /// returns the eval type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr DatasetEvalList getDatasetEval() {
        // note: these are only defaults, they can be overridden via full specialization in their impl header
        return
                (eDatasetTask==DatasetTask_Segm)?DatasetEval_BinaryClassifier:
                (eDatasetTask==DatasetTask_Cosegm)?DatasetEval_BinaryClassifierArray:
                (eDatasetTask==DatasetTask_Registr)?DatasetEval_Registr:
                (eDatasetTask==DatasetTask_EdgDet)?DatasetEval_BinaryClassifier:
                // ...
                DatasetEval_None; // undefined behavior
    }

    /// returns the array type policy to use based on the dataset eval type
    template<DatasetEvalList eDatasetEval>
    constexpr ArrayPolicy getArrayPolicy() {
        return ((eDatasetEval==DatasetEval_BinaryClassifierArray)||(eDatasetEval==DatasetEval_MultiClassifierArray))?Array:NotArray;
    }

    /// returns the source type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr DatasetSourceList getDatasetSource() {
        // note: these are only defaults, they can be overridden via full specialization in their impl header
        return
                (eDatasetTask==DatasetTask_Segm)?DatasetSource_Video:
                (eDatasetTask==DatasetTask_Cosegm)?DatasetSource_VideoArray:
                (eDatasetTask==DatasetTask_Registr)?DatasetSource_VideoArray:
                (eDatasetTask==DatasetTask_EdgDet)?DatasetSource_Image:
                // ...
                DatasetSource_Video; // undefined behavior
    }

    /// returns whether task, source, and eval types are all compatible (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset, DatasetEvalList eDatasetEval>
    constexpr bool isDatasetSpecValid() {
        return
                (eDatasetTask==DatasetTask_Segm)?(((eDatasetSource==DatasetSource_Video)||(eDatasetSource==DatasetSource_Image))&&((eDatasetEval==DatasetEval_BinaryClassifier)||(eDatasetEval==DatasetEval_MultiClassifier)||(eDatasetEval==DatasetEval_None))):
                (eDatasetTask==DatasetTask_Cosegm)?(((eDatasetSource==DatasetSource_VideoArray)||(eDatasetSource==DatasetSource_ImageArray))&&((eDatasetEval==DatasetEval_BinaryClassifierArray)||(eDatasetEval==DatasetEval_MultiClassifierArray)||(eDatasetEval==DatasetEval_None))):
                (eDatasetTask==DatasetTask_Registr)?(((eDatasetSource==DatasetSource_VideoArray)||(eDatasetSource==DatasetSource_ImageArray))&&((eDatasetEval==DatasetEval_Registr)||(eDatasetEval==DatasetEval_None))):
                (eDatasetTask==DatasetTask_EdgDet)?(((eDatasetSource==DatasetSource_Video)||(eDatasetSource==DatasetSource_Image))&&((eDatasetEval==DatasetEval_BinaryClassifier)||(eDatasetEval==DatasetEval_None))):
                // ...
                false; // undefined behavior
    }

    struct IDataset;
    struct IDataHandler;
    using IDatasetPtr = std::shared_ptr<IDataset>;
    using IDataHandlerPtr = std::shared_ptr<IDataHandler>;
    using IDataHandlerPtrArray = std::vector<IDataHandlerPtr>;
    using IDataHandlerConstPtr = std::shared_ptr<const IDataHandler>;
    using IDataHandlerConstPtrArray = std::vector<IDataHandlerConstPtr>;
    using IDataHandlerPtrQueue = std::priority_queue<IDataHandlerPtr,IDataHandlerPtrArray,std::function<bool(const IDataHandlerPtr&,const IDataHandlerPtr&)>>;
    using AsyncDataCallbackFunc = std::function<void(const cv::Mat& /*oInput*/,const cv::Mat& /*oDebug*/,const cv::Mat& /*oOutput*/,const cv::Mat& /*oGT*/,const cv::Mat& /*oGTROI*/,size_t /*nIdx*/)>;

    /// fully abstract dataset interface (dataset parser & evaluator implementations will derive from this)
    struct IDataset : lv::enable_shared_from_this<IDataset> {
        /// virtual destructor for adequate cleanup from IDataset pointers
        virtual ~IDataset() = default;
        /// returns the dataset name
        virtual const std::string& getName() const = 0;
        /// returns the root data path
        virtual const std::string& getDatasetPath() const = 0;
        /// returns the root output path
        virtual const std::string& getOutputPath() const = 0;
        /// returns the output file name prefix for results archiving
        virtual const std::string& getOutputNamePrefix() const = 0;
        /// returns the output file name suffix for results archiving
        virtual const std::string& getOutputNameSuffix() const = 0;
        /// returns the directory names of top-level work batches
        virtual const std::vector<std::string>& getWorkBatchDirs() const = 0;
        /// returns the directory name tokens which, if found, should be skipped
        virtual const std::vector<std::string>& getSkippedDirTokens() const = 0;
        /// returns the directory name tokens which, if found, should be treated as grayscale
        virtual const std::vector<std::string>& getGrayscaleDirTokens() const = 0;
        /// returns the output file/packet index offset for results archiving
        virtual size_t getOutputIdxOffset() const = 0;
        /// returns the input data scaling scaling factor
        virtual double getScaleFactor() const = 0;
        /// returns whether we should save the results through DataConsumers or not
        virtual bool isSavingOutput() const = 0;
        /// returns whether we should evaluate the results through DataConsumers or not
        virtual bool isUsingEvaluator() const = 0;
        /// returns whether loaded data should be 4-byte aligned or not (4-byte alignment is ideal for GPU upload)
        virtual bool is4ByteAligned() const = 0;
        /// returns whether this dataset defines some evaluation procedure or not
        virtual bool isEvaluable() const {return false;}
        /// returns the total input packet count in the dataset (recursively queried from work batches)
        virtual size_t getInputCount() const = 0;
        /// returns the total gt packet count in the dataset (recursively queried from work batches)
        virtual size_t getGTCount() const = 0;
        /// returns the total output packet count expected to be processed by the dataset evaluator (recursively queried from work batches)
        virtual size_t getExpectedOutputCount() const = 0;
        /// returns the total output packet count so far processed by the dataset evaluator (recursively queried from work batches)
        virtual size_t getProcessedOutputCount() const = 0;
        /// returns the total output packet count so far processed by the dataset evaluator, blocking if processing is not finished yet (recursively queried from work batches)
        virtual size_t getProcessedOutputCountPromise() = 0;
        /// returns the total time it took to process the dataset (recursively queried from work batches)
        virtual double getProcessTime() const = 0;
        /// clears all batches and reparses them from the dataset metadata
        virtual void parseDataset() = 0;
        /// writes the dataset-level evaluation report
        virtual void writeEvalReport() const = 0;
        /// returns the array of work batches (or groups, if requested with hierarchy) contained in this dataset
        virtual IDataHandlerPtrArray getBatches(bool bWithHierarchy) const = 0;
        /// returns the array of work batches (or groups, if requested with hierarchy) contained in this dataset, sorted by expected CPU load
        virtual IDataHandlerPtrQueue getSortedBatches(bool bWithHierarchy) const = 0;
    };

    /// fully abstract data handler interface (work batch and work group implementations will derive from this)
    struct IDataHandler : lv::enable_shared_from_this<IDataHandler> {
        /// returns the work batch/group name
        virtual const std::string& getName() const = 0;
        /// returns the work batch/group data path
        virtual const std::string& getDataPath() const = 0;
        /// returns the work batch/group output path
        virtual const std::string& getOutputPath() const = 0;
        /// returns the work batch/group relative path offset w.r.t. dataset root
        virtual const std::string& getRelativePath() const = 0;
        /// returns the expected CPU load of the work batch/group (only relevant for intra-dataset load comparisons)
        virtual double getExpectedLoad() const = 0;
        /// returns the total input packet count for this work batch/group
        virtual size_t getInputCount() const = 0;
        /// returns the total gt packet count for this work batch/group
        virtual size_t getGTCount() const = 0;
        /// returns the total output packet count expected to be processed by the work batch/group evaluator
        virtual size_t getExpectedOutputCount() const = 0;
        /// returns the total output packet count so far processed by the work batch/group evaluator
        virtual size_t getProcessedOutputCount() const = 0;
        /// returns the total output packet count so far processed by the work batch/group evaluator, blocking if processing is not finished yet
        virtual size_t getProcessedOutputCountPromise() = 0;
        /// returns a name (not necessarily used for parsing) associated with an input data packet index (useful for data archiving)
        virtual std::string getInputName(size_t nPacketIdx) const;
        /// returns a name that should be given to an output data packet based on its index (useful for data archiving)
        virtual std::string getOutputName(size_t nPacketIdx) const;
        /// returns whether the work batch/group data will be treated as grayscale
        virtual bool isGrayscale() const = 0;
        /// returns whether the work group is a pass-through container (always false for work batches)
        virtual bool isBare() const = 0;
        /// returns whether this data handler interface points to a work batch or a work group
        virtual bool isGroup() const = 0;
        /// returns whether this dataset defines some evaluation procedure or not
        virtual bool isEvaluable() const {return false;}
        /// returns this work group's children (work batch array)
        virtual IDataHandlerPtrArray getBatches(bool bWithHierarchy) const = 0;
        /// returns a pointer to this work batch/group's parent dataset interface
        virtual IDatasetPtr getDatasetInfo() const = 0;
        /// returns which processing task this work batch/group was built for
        virtual DatasetTaskList getDatasetTask() const = 0;
        /// returns which data source this work batch/group was built for
        virtual DatasetSourceList getDatasetSource() const = 0;
        /// returns which evaluation method this work batch/group was built for
        virtual DatasetEvalList getDatasetEval() const = 0;
        /// returns which dataset this work batch/group was built for
        virtual DatasetList getDataset() const = 0;
        /// writes the batch-level evaluation report
        virtual void writeEvalReport() const = 0;
        /// virtual destructor for adequate cleanup from IDataHandler pointers
        virtual ~IDataHandler() = default;
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT=false, size_t nSuggestedBufferSize=SIZE_MAX) = 0;
        /// kills the asynchronyzed precacher, and clears internal buffers
        virtual void stopPrecaching() = 0;
        /// returns whether this work batch (or any of this work group's children batches) is currently processing data
        virtual bool isProcessing() const = 0;
        /// returns the current (or final) duration elapsed between start/stopProcessing calls (recursively queried for work groups)
        virtual double getProcessTime() const = 0;
    protected:
        /// work batch/group comparison function based on names
        template<typename Tp>
        static std::enable_if_t<std::is_base_of<IDataHandler,Tp>::value,bool> compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return lv::compare_lowercase(i->getName(),j->getName());
        }
        /// work batch/group comparison function based on expected CPU load
        template<typename Tp>
        static std::enable_if_t<std::is_base_of<IDataHandler,Tp>::value,bool> compare_load(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return i->getExpectedLoad()<j->getExpectedLoad();
        }
        /// work batch/group comparison function based on names
        static bool compare(const IDataHandler* i, const IDataHandler* j);
        /// work batch/group comparison function based on expected CPU load
        static bool compare_load(const IDataHandler* i, const IDataHandler* j);
        /// work batch/group comparison function based on names
        static bool compare(const IDataHandler& i, const IDataHandler& j);
        /// work batch/group comparison function based on expected CPU load
        static bool compare_load(const IDataHandler& i, const IDataHandler& j);
        /// returns the children batch associated with the given packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const = 0;
        /// returns the children batch associated with the given packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) = 0;
        /// overrideable method to be called when the user 'starts processing' the data batch (by default, always inits counters after calling this)
        virtual void startProcessing_impl() {}
        /// overrideable method to be called when the user 'stops processing' the data batch (by default, always stops counters before calling this)
        virtual void stopProcessing_impl() {}
        /// local folder data parsing function, dataset-specific
        virtual void parseData() = 0;
        template<DatasetEvalList eDatasetEval>
        friend struct IDatasetReporter_; // required for external interface to write eval reports
        template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset, DatasetEvalList eDatasetEval, lv::ParallelAlgoType eEvalImpl>
        friend struct IDataset_; // required for data handler sorting and other top-level dataset utility functions
        friend struct IGroupDataParser; // required for recursive parsing of work batch data through groups
    };

    /// group data parser interface for work batch groups @@@@ replace by IDataParser + templ spec? like counter (mostly final overrides)
    struct IGroupDataParser : public virtual IDataHandler {
        /// returns whether the work group is a pass-through container
        virtual bool isBare() const override final;
        /// always returns true for work groups
        virtual bool isGroup() const override final;
        /// returns this work group's children (work batch array)
        virtual IDataHandlerPtrArray getBatches(bool bWithHierarchy) const override final;
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT=false, size_t nSuggestedBufferSize=SIZE_MAX) override final;
        /// kills the asynchronyzed precacher, and clears internal buffers
        virtual void stopPrecaching() override final;
        /// returns whether any of this work group's children batches is currently processing data
        virtual bool isProcessing() const override final;
        /// returns the current (or final) duration elapsed between start/stopProcessing calls, recursively queried for all children work batches
        virtual double getProcessTime() const override final;
        /// accumulates the expected CPU load for this data batch based on all children work batches load
        virtual double getExpectedLoad() const override final;
        /// accumulate total input packet count from all children work batches
        virtual size_t getInputCount() const override final;
        /// accumulate total gt packet count from all children work batches
        virtual size_t getGTCount() const override final;
    protected:
        /// creates and returns a work batch for a given relative dataset path
        virtual IDataHandlerPtr createWorkBatch(const std::string& sBatchName, const std::string& sRelativePath=std::string("./")) const = 0;
        /// creates group/nongroup workbatches based on internal datset info and current relative path, and recursively calls parse data on all childrens
        virtual void parseData() override;
        /// default constructor; automatically sets 'isBare' to true
        inline IGroupDataParser() : m_bIsBare(true) {}
        /// contains children work batches (which may also be groups containing children)
        IDataHandlerPtrArray m_vpBatches;
        /// defines whether the group is pass-through (with zero or one non-group child) or not
        bool m_bIsBare;
    };

    /// group data parser full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset>
    struct GroupDataParser_ : public IGroupDataParser {};

    /// general-purpose data packet precacher, fully implemented (i.e. can be used stand-alone)
    struct DataPrecacher {
        /// attaches to data loader (will halt auto-precaching if an empty packet is fetched)
        DataPrecacher(std::function<const cv::Mat&(size_t)> lDataLoaderCallback);
        /// default destructor (joins the precaching thread, if still running)
        ~DataPrecacher();
        /// fetches a packet, with or without precaching enabled (should never be called concurrently, returned packets should never be altered directly, and a single packet loaded twice is assumed identical)
        const cv::Mat& getPacket(size_t nIdx);
        /// initializes precaching with a given buffer size (starts up thread)
        bool startAsyncPrecaching(size_t nSuggestedBufferSize);
        /// joins precaching thread and clears all internal buffers
        void stopAsyncPrecaching();
        /// returns whether the precaching thread has already been started or not
        inline bool isActive() const {return m_bIsActive;}
    private:
        void entry(const size_t nBufferSize);
        const std::function<const cv::Mat&(size_t)> m_lCallback;
        std::thread m_hWorker;
        std::mutex m_oSyncMutex;
        std::condition_variable m_oReqCondVar;
        std::condition_variable m_oSyncCondVar;
        std::atomic_bool m_bIsActive;
        size_t m_nReqIdx,m_nLastReqIdx;
        std::atomic_size_t m_nAnswIdx;
        cv::Mat m_oReqPacket,m_oLastReqPacket;
        DataPrecacher& operator=(const DataPrecacher&) = delete;
        DataPrecacher(const DataPrecacher&) = delete;
    };

    /// data loader super-interface for work batch, exposes basic packet get functions and internal precacher wiring
    struct IIDataLoader : public virtual IDataHandler {
        /// returns the input data packet type policy (used for internal packet auto-transformations)
        inline PacketPolicy getInputPacketType() const {return m_eInputType;}
        /// returns the gt data packet type policy (used for internal packet auto-transformations)
        inline PacketPolicy getGTPacketType() const {return m_eGTType;}
        /// returns the output data packet type policy (used for internal packet auto-transformations)
        inline PacketPolicy getOutputPacketType() const {return m_eOutputType;}
        /// returns the gt/output data packet mapping type policy (used for internal packet auto-transformations)
        inline MappingPolicy getGTMappingType() const {return m_eGTMappingType;}
        /// returns the input/output data packet mapping type policy (used for internal packet auto-transformations)
        inline MappingPolicy getIOMappingType() const {return m_eIOMappingType;}
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT=false, size_t nSuggestedBufferSize=SIZE_MAX) override;
        /// kills the asynchronyzed precacher, and clears internal buffers
        virtual void stopPrecaching() override;
        /// returns an input packet by index (works both with and without precaching enabled)
        const cv::Mat& getInput(size_t nPacketIdx);
        /// returns a gt packet by index (works both with and without precaching enabled)
        const cv::Mat& getGT(size_t nPacketIdx);
        /// returns the ROI associated with an input packet by index (returns empty mat by default)
        virtual const cv::Mat& getInputROI(size_t nPacketIdx) const;
        /// returns the ROI associated with a gt packet by index (returns empty mat by default)
        virtual const cv::Mat& getGTROI(size_t nPacketIdx) const;
        /// returns the size associated with an input packet by index (returns empty size by default) @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputSize(size_t nPacketIdx) const;
        /// returns the size associated with a gt packet by index (returns empty size by default) @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTSize(size_t nPacketIdx) const;
        /// returns the maximum size associated with any input packet (returns empty size by default) @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputMaxSize() const;
        /// returns the maximum size associated with any gt packet (returns empty size by default) @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTMaxSize() const;
    protected:
        /// types serve to automatically transform packets & define default implementations
        IIDataLoader(PacketPolicy eInputType, PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        /// input packet load function, pre-transformations (can return empty mats)
        virtual cv::Mat getRawInput(size_t nPacketIdx) = 0;
        /// gt packet load function, pre-transformations (can return empty mats)
        virtual cv::Mat getRawGT(size_t nPacketIdx) = 0;
        /// input packet transformation function (used e.g. for rescaling and color space conversion)
        virtual const cv::Mat& getInput_redirect(size_t nPacketIdx);
        /// gt packet transformation function (used e.g. for rescaling and color space conversion)
        virtual const cv::Mat& getGT_redirect(size_t nPacketIdx);
    private:
        /// holds the loaded copies of the latest input/gt packets queried by the precachers
        cv::Mat m_oLatestInput,m_oLatestGT;
        /// precacher objects which may spin up a thread to pre-fetch data packets
        DataPrecacher m_oInputPrecacher,m_oGTPrecacher;
        /// input/gt/output packet policy types
        const PacketPolicy m_eInputType,m_eGTType,m_eOutputType;
        /// output-gt and input-output mapping policy types
        const MappingPolicy m_eGTMappingType,m_eIOMappingType;
    };

    /// default (specializable) forward declaration of the data loader interface
    template<ArrayPolicy ePolicy>
    struct IDataLoader_;

    /// data loader specialization for non-array processing (exposes more auto-transform getters for packets, and simple ROIs)
    template<>
    struct IDataLoader_<NotArray> : public IIDataLoader {
    protected:
        /// pass-through constructor to super-interface
        using IIDataLoader::IIDataLoader;
    };

    /// data loader specialization for array processing (exposes unpacked array getters)
    template<>
    struct IDataLoader_<Array> : public IIDataLoader {
        /// returns the number of parallel input streams (defaults to 1)
        virtual size_t getInputStreamCount() const;
        /// returns the number of parallel gt streams (defaults to 1)
        virtual size_t getGTStreamCount() const;
        /// returns the (friendly) name of an input stream specified by index
        virtual std::string getInputStreamName(size_t nStreamIdx) const;
        /// returns the (friendly) name of a gt stream specified by index
        virtual std::string getGTStreamName(size_t nStreamIdx) const;
        /// unpacks and returns an input array by packet index, with each stream its own cv::Mat (works both with and without precaching enabled)
        const std::vector<cv::Mat>& getInputArray(size_t nPacketIdx);
        /// unpacks and returns a gt array by packet index, with each stream its own cv::Mat (works both with and without precaching enabled)
        const std::vector<cv::Mat>& getGTArray(size_t nPacketIdx);
        /// unpacks and returns an input ROI array by packet index, with each stream its own cv::Mat
        virtual const std::vector<cv::Mat>& getInputROIArray(size_t nPacketIdx) const;
        /// unpacks and returns a gt ROI array by packet index, with each stream its own cv::Mat
        virtual const std::vector<cv::Mat>& getGTROIArray(size_t nPacketIdx) const;
        /// returns the size array associated with an input packet by index (returns empty vector by default) @@@@@ override later to make size N-Dim?
        virtual const std::vector<cv::Size>& getInputSizeArray(size_t nPacketIdx) const;
        /// returns the size associated with a gt packet by index (returns empty vector by default) @@@@@ override later to make size N-Dim?
        virtual const std::vector<cv::Size>& getGTSizeArray(size_t nPacketIdx) const;
        /// returns whether an input data stream should be force-loaded as grayscale
        virtual bool isStreamGrayscale(size_t nStreamIdx) const;
    protected:
        /// pass-through constructor to super-interface
        using IIDataLoader::IIDataLoader;
        /// hides the 'packed' input accessor from public interface
        using IIDataLoader::getInput;
        /// hides the 'packed' gt accessor from public interface
        using IIDataLoader::getGT;
        /// hides useless non-array-only function from public interface (can be unhidden by derived class)
        using IIDataLoader::getInputROI;
        /// hides useless non-array-only function from public interface (can be unhidden by derived class)
        using IIDataLoader::getGTROI;
        /// hides useless non-array-only function from public interface (can be unhidden by derived class)
        using IIDataLoader::getInputSize;
        /// hides useless non-array-only function from public interface (can be unhidden by derived class)
        using IIDataLoader::getGTSize;
        /// input 'unpacking' function, which essentially unmerges the streams in a packet and assigns them to individual mats in the vector
        virtual void unpackInput(size_t nPacketIdx, std::vector<cv::Mat>& vUnpackedInput);
        /// gt 'unpacking' function, which essentially unmerges the streams in a packet and assigns them to individual mats in the vector
        virtual void unpackGT(size_t nPacketIdx, std::vector<cv::Mat>& vUnpackedGT);
    private:
        std::vector<cv::Mat> m_vLatestUnpackedInput,m_vLatestUnpackedGT;
    };

    /// default (specializable) forward declaration of the data producer interface
    template<DatasetSourceList eDatasetSource>
    struct IDataProducer_;

    /// data producer specialization for video processing
    template<>
    struct IDataProducer_<DatasetSource_Video> :
            public IDataLoader_<NotArray> {
        /// returns the total frame count for this work batch/group (redirects to getInputCount())
        inline size_t getFrameCount() const {return getInputCount();}
        /// returns the ROI associated with all frames
        virtual const cv::Mat& getFrameROI() const;
        /// returns the size associated with all frames @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getFrameSize() const;
        /// returns the total input frame count for this work batch/group
        virtual size_t getInputCount() const override;
        /// returns the total gt frame count for this work batch/group
        virtual size_t getGTCount() const override;
        /// compute the expected CPU load for this data batch based on frame size, frame count, and channel count
        virtual double getExpectedLoad() const override;
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT=false, size_t nSuggestedBufferSize=SIZE_MAX) override;
    protected:
        /// specialized constructor; still need to specify gt type, output type, and mappings
        IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        virtual const cv::Mat& getInputROI(size_t nPacketIdx) const override;
        virtual const cv::Mat& getGTROI(size_t nPacketIdx) const override;
        virtual const cv::Size& getInputSize(size_t nPacketIdx) const override;
        virtual const cv::Size& getGTSize(size_t nPacketIdx) const override;
        virtual const cv::Size& getInputMaxSize() const override;
        virtual const cv::Size& getGTMaxSize() const override;
        virtual cv::Mat getRawInput(size_t nPacketIdx) override;
        virtual cv::Mat getRawGT(size_t nPacketIdx) override;
        virtual void parseData() override;
        size_t m_nFrameCount; ///< needed as a separate variable for VideoCapture+imread support
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::string> m_vsInputPaths,m_vsGTPaths;
        cv::VideoCapture m_voVideoReader;
        size_t m_nNextExpectedVideoReaderFrameIdx;
        cv::Mat m_oInputROI,m_oGTROI;
        cv::Size m_oInputSize,m_oGTSize;
    };

    /// data producer specialization for multi-video processing
    template<>
    struct IDataProducer_<DatasetSource_VideoArray> :
            public IDataLoader_<Array> {
        /// returns the total frame count for this work batch/group (redirects to getInputCount())
        inline size_t getFrameCount() const {return getInputCount();}
        /// returns the ROIs associated with all frames
        virtual const std::vector<cv::Mat>& getFrameROIArray() const;
        /// returns the sizes associated with all frames @@@@@ override later to make size N-Dim?
        virtual const std::vector<cv::Size>& getFrameSizeArray() const;
        /// returns the total input frame count for this work batch/group --- all streams should be sync'd
        virtual size_t getInputCount() const override;
        /// returns the total gt frame count for this work batch/group --- all streams should be sync'd
        virtual size_t getGTCount() const override;
        /// compute the expected CPU load for this data batch based on frame size, frame count, and channel count
        virtual double getExpectedLoad() const override;
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT=false, size_t nSuggestedBufferSize=SIZE_MAX) override;
    protected:
        /// specialized constructor; still need to specify gt type, output type, and mappings
        IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        virtual const std::vector<cv::Mat>& getInputROIArray(size_t nPacketIdx) const override;
        virtual const std::vector<cv::Mat>& getGTROIArray(size_t nPacketIdx) const override;
        virtual const std::vector<cv::Size>& getInputSizeArray(size_t nPacketIdx) const override;
        virtual const std::vector<cv::Size>& getGTSizeArray(size_t nPacketIdx) const override;
        virtual const cv::Size& getInputMaxSize() const override;
        virtual const cv::Size& getGTMaxSize() const override;
        virtual cv::Mat getRawInput(size_t nPacketIdx) override; ///< loads and returns a 'packed' input packet
        virtual cv::Mat getRawGT(size_t nPacketIdx) override; ///< loads and returns a 'packed' gt packet
        //virtual void parseData() override;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::vector<std::string>> m_vvsInputPaths,m_vvsGTPaths; // first dimension is packet index, 2nd is stream index
        std::vector<cv::Mat> m_vInputROIs,m_vGTROIs; // one ROI per stream
        std::vector<cv::Size> m_vInputSizes,m_vGTSizes;
        cv::Size m_oMaxInputSize,m_oMaxGTSize;
    };

    /// data producer specialization for image processing
    template<>
    struct IDataProducer_<DatasetSource_Image> :
            public IDataLoader_<NotArray> {
        /// returns the total image count for this work batch/group (redirects to getInputCount())
        inline size_t getImageCount() const {return getInputCount();}
        /// returns whether all input images in this batch have the same size
        virtual bool isInputConstantSize() const;
        /// returns whether all gt images in this batch have the same size
        virtual bool isGTConstantSize() const;
        /// returns the total input image count for this work batch/group
        virtual size_t getInputCount() const override;
        /// returns the total gt image count for this work batch/group
        virtual size_t getGTCount() const override;
        /// compute the expected CPU load for this data batch based on max image size, image count, and channel count
        virtual double getExpectedLoad() const override;
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT=false, size_t nSuggestedBufferSize=SIZE_MAX) override;
        /// returns the size associated with an input image by index @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputSize(size_t nPacketIdx) const override;
        /// returns the size associated with a gt image by index @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTSize(size_t nPacketIdx) const override;
        /// returns the maximum size associated with an input image @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputMaxSize() const override;
        /// returns the maximum size associated with a gt image @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTMaxSize() const override;
        /// returns the file name associated with an input data packet index (useful for data archiving)
        virtual std::string getInputName(size_t nPacketIdx) const override;
    protected:
        /// specialized constructor; still need to specify gt type, output type, and mappings
        IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        virtual cv::Mat getRawInput(size_t nPacketIdx) override;
        virtual cv::Mat getRawGT(size_t nPacketIdx) override;
        virtual void parseData() override;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::string> m_vsInputPaths,m_vsGTPaths;
        std::vector<cv::Size> m_vInputSizes,m_vGTSizes;
        bool m_bIsInputConstantSize,m_bIsGTConstantSize;
        cv::Size m_oInputMaxSize,m_oGTMaxSize;
    };

    /// data producer specialization for multi-image processing
    template<>
    struct IDataProducer_<DatasetSource_ImageArray> : // @@@ to be tested
            public IDataLoader_<Array> {
        /// returns the total image count for this work batch/group (redirects to getInputCount())
        inline size_t getImageCount() const {return getInputCount();}
        /// returns whether all input images in this batch have the same size
        virtual bool isInputConstantSize() const;
        /// returns whether all gt images in this batch have the same size
        virtual bool isGTConstantSize() const;
        /// returns the total input image count for this work batch/group
        virtual size_t getInputCount() const override;
        /// returns the total gt image count for this work batch/group
        virtual size_t getGTCount() const override;
        /// compute the expected CPU load for this data batch based on max image size, image count, and channel count
        virtual double getExpectedLoad() const override;
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT=false, size_t nSuggestedBufferSize=SIZE_MAX) override;
        /// returns the size associated with an input image by index @@@@@ override later to make size N-Dim?
        virtual const std::vector<cv::Size>& getInputSizeArray(size_t nPacketIdx) const override;
        /// returns the size associated with a gt image by index @@@@@ override later to make size N-Dim?
        virtual const std::vector<cv::Size>& getGTSizeArray(size_t nPacketIdx) const override;
        /// returns the maximum size associated with an input image @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputMaxSize() const override;
        /// returns the maximum size associated with a gt image @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTMaxSize() const override;
    protected:
        /// specialized constructor; still need to specify gt type, output type, and mappings
        IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        virtual cv::Mat getRawInput(size_t nPacketIdx) override; ///< loads and returns a 'packed' input packet
        virtual cv::Mat getRawGT(size_t nPacketIdx) override; ///< loads and returns a 'packed' gt packet
        //virtual void parseData() override;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::vector<std::string>> m_vvsInputPaths,m_vvsGTPaths; ///< one path per packet per stream
        std::vector<std::vector<cv::Size>> m_vvInputSizes,m_vvGTSizes; ///< one size per packet per stream
        bool m_bIsInputConstantSize,m_bIsGTConstantSize; ///< all streams use same logic
        cv::Size m_oInputMaxSize,m_oGTMaxSize;
    };

    /// data producer constructor wrapper for cleaner specializations in 'impl' headers (if needed, custom data producers should inherit this instead of IDataProducer_)
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset>
    struct IDataProducerWrapper_ : public IDataProducer_<eDatasetSource> {
        IDataProducerWrapper_() : IDataProducer_<eDatasetSource>(lv::getGTPacketType<eDatasetTask,eDataset>(),lv::getOutputPacketType<eDatasetTask,eDataset>(),lv::getGTMappingType<eDatasetTask,eDataset>(),lv::getIOMappingType<eDatasetTask,eDataset>()) {}
    };

    /// data producer full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset>
    struct DataProducer_ : public IDataProducerWrapper_<eDatasetTask,eDatasetSource,eDataset> {};

    /// general-purpose, stand-alone data packet writer
    struct DataWriter {
        /// attaches to data archiver (the callback is the actual 'writing' action, with a signature similar to 'queue')
        DataWriter(std::function<size_t(const cv::Mat&,size_t)> lDataArchiverCallback);
        /// default destructor (joins the writing thread, if still running)
        ~DataWriter();
        /// queues a packet, with or without async writing enabled, and returns its position in queue
        size_t queue(const cv::Mat& oPacket, size_t nIdx);
        /// returns the current queue size, in packets
        inline size_t getCurrentQueueCount() const {return m_nQueueCount;}
        /// returns the current queue size, in bytes
        inline size_t getCurrentQueueSize() const {return m_nQueueSize;}
        /// initializes async writing with a given queue size (in bytes) and a number of threads
        bool startAsyncWriting(size_t nSuggestedQueueSize, bool bDropPacketsIfFull=false, size_t nWorkers=1);
        /// joins writing thread and clears all internal buffers
        void stopAsyncWriting();
        /// returns whether the wariting thread has already been started or not
        inline bool isActive() const {return m_bIsActive;}
    private:
        void entry();
        const std::function<size_t(const cv::Mat&,size_t)> m_lCallback;
        std::vector<std::thread> m_vhWorkers;
        std::mutex m_oSyncMutex;
        std::condition_variable m_oQueueCondVar;
        std::condition_variable m_oClearCondVar;
        std::map<size_t,cv::Mat> m_mQueue;
        std::atomic_bool m_bIsActive;
        bool m_bAllowPacketDrop;
        size_t m_nQueueMaxSize;
        std::atomic_size_t m_nQueueSize;
        std::atomic_size_t m_nQueueCount;
        DataWriter& operator=(const DataWriter&) = delete;
        DataWriter(const DataWriter&) = delete;
    };

    /// default (specializable) forward declaration of the data archiver interface (used to save/load outputs)
    template<ArrayPolicy ePolicy>
    struct IDataArchiver_;

    /// data archiver specialization for non-array output processing
    template<>
    struct IDataArchiver_<NotArray> : public virtual IDataHandler {
    protected:
        /// saves a processed data packet locally based on idx and packet name (if available), with optional flags (-1 = internal defaults)
        virtual void save(const cv::Mat& oOutput, size_t nIdx, int nFlags=-1);
        /// loads a processed data packet based on idx and packet name (if available), with optional flags (-1 = internal defaults)
        virtual cv::Mat load(size_t nIdx, int nFlags=-1);
    };

    /// data archiver specialization for array output processing
    template<>
    struct IDataArchiver_<Array> : public virtual IDataHandler {
    protected:
        /// saves a processed data packet array locally based on idx and packet name (if available), with optional flags (-1 = internal defaults)
        virtual void saveArray(const std::vector<cv::Mat>& vOutput, size_t nIdx, int nFlags=-1);
        /// loads a processed data packet array based on idx and packet name (if available), with optional flags (-1 = internal defaults)
        virtual std::vector<cv::Mat> loadArray(size_t nIdx, int nFlags=-1);
    };

    /// default (specializable) forward declaration of the data counter interface (used only for packet counting)
    template<GroupPolicy ePolicy>
    struct IDataCounter_;

    /// data counter specialization for individual work batches (exposes counting logic)
    template<>
    struct IDataCounter_<NotGroup> : public virtual IDataHandler {
    protected:
        /// increments processed packets count
        inline void countOutput(size_t nPacketIdx) {m_mProcessedPackets.insert(nPacketIdx);}
        /// sets the processed packets count promise for async implementations
        inline void setProcessedOutputCountPromise() {m_nProcessedPacketsPromise.set_value(m_mProcessedPackets.size());}
        /// resets the processed packets count
        inline void resetProcessedOutputCount() {m_mProcessedPackets.clear();}
        /// gets current processed packets count
        virtual size_t getProcessedOutputCount() const override final;
        /// gets processed packets count from promise for async implementations (blocks until stopProcessing is called)
        virtual size_t getProcessedOutputCountPromise() override final;
    private:
        std::unordered_set<size_t> m_mProcessedPackets;
        std::promise<size_t> m_nProcessedPacketsPromise;
    };

    /// data counter specialization for work batch groups (exposes recursive counting logic)
    template<>
    struct IDataCounter_<Group> : public virtual IDataHandler {
        /// returns the total output packet count expected to be processed by the data consumer
        virtual size_t getExpectedOutputCount() const override final;
        /// returns the total output packet count so far processed by the data consumer
        virtual size_t getProcessedOutputCount() const override final;
        /// returns the total output packet count so far processed by the data consumer, blocking if processing is not finished yet
        virtual size_t getProcessedOutputCountPromise() override final;
    };

    /// default (specializable) forward declaration of the data consumer interface
    template<DatasetEvalList eDatasetEval, typename ENABLE=void>
    struct IDataConsumer_;

    /// data consumer specialization for receiving processed packets (evaluation entrypoint)
    template<DatasetEvalList eDatasetEval>
    struct IDataConsumer_<eDatasetEval,std::enable_if_t<getArrayPolicy<eDatasetEval>()==NotArray>> :
            public IDataArchiver_<NotArray>,
            public IDataCounter_<NotGroup> {
        /// returns the total output packet count expected to be processed by the data consumer (defaults to GT count)
        virtual size_t getExpectedOutputCount() const override {
            return getGTCount();
        }
        /// pushes an output (processed) data packet array for writing and/or evaluation
        inline void push(const cv::Mat& oOutput, size_t nPacketIdx) {
            lvAssert_(isProcessing(),"data processing must be toggled via 'startProcessing()' before pushing packets");
            countOutput(nPacketIdx);
            processOutput(oOutput,nPacketIdx);
            if(getDatasetInfo()->isSavingOutput() && !oOutput.empty())
                this->save(oOutput,nPacketIdx);
        }
    protected:
        /// processes an output packet (does nothing by default, but may be overridden for evaluation/pipelining)
        virtual void processOutput(const cv::Mat& /*oOutput*/, size_t /*nPacketIdx*/) {}
    };

    /// data consumer specialization for receiving processed packet arrays (evaluation entrypoint)
    template<DatasetEvalList eDatasetEval>
    struct IDataConsumer_<eDatasetEval,std::enable_if_t<getArrayPolicy<eDatasetEval>()==Array>> :
            public IDataArchiver_<Array>,
            public IDataCounter_<NotGroup> {
        /// returns the number of parallel output streams (defaults to GT stream count if loader is array-based, and 1 otherwise)
        virtual size_t getOutputStreamCount() const {
            auto pLoader = shared_from_this_cast<IDataLoader_<Array>>();
            if(pLoader)
                return pLoader->getGTStreamCount();
            return 1;
        }
        /// returns the total output packet count expected to be processed by the data consumer (defaults to GT count)
        virtual size_t getExpectedOutputCount() const override {
            return getGTCount();
        }
        /// returns the (friendly) name of an output stream specified by index
        virtual std::string getOutputStreamName(size_t nStreamIdx) const {
            return cv::format("out[%02d]",(int)nStreamIdx);
        }
        /// pushes an output (processed) data packet array for writing and/or evaluation
        inline void push(const std::vector<cv::Mat>& vOutput, size_t nPacketIdx) {
            lvAssert_(isProcessing(),"data processing must be toggled via 'startProcessing()' before pushing packets");
            lvAssert_(vOutput.empty() || vOutput.size()==getOutputStreamCount(),"bad output array size");
            countOutput(nPacketIdx);
            processOutput(vOutput,nPacketIdx);
            if(getDatasetInfo()->isSavingOutput() && !vOutput.empty())
                this->saveArray(vOutput,nPacketIdx);
        }
    protected:
        /// processes an output array packet (does nothing by default, but may be overridden for evaluation/pipelining)
        virtual void processOutput(const std::vector<cv::Mat>& /*vOutput*/, size_t /*nPacketIdx*/) {}
    };

    /// default (specializable) forward declaration of the async data consumer interface used for receiving processed packets (evaluation entrypoint)
    template<DatasetEvalList eDatasetEval, lv::ParallelAlgoType eImpl>
    struct IAsyncDataConsumer_;

#if HAVE_GLSL

    /// async data consumer specialization for non-array binary classification processing
    template<>
    struct IAsyncDataConsumer_<DatasetEval_BinaryClassifier,lv::GLSL> :
            public IDataArchiver_<NotArray>,
            public IDataCounter_<NotGroup> {
        /// returns the ideal size for the GL context window to use for debug display purposes (queries the algo based on dataset specs, if available)
        virtual cv::Size getIdealGLWindowSize() const;
        /// initializes internal params & calls 'initialize_gl' on algo with expanded args list
        template<typename Talgo, typename... Targs>
        void initialize_gl(const std::shared_ptr<Talgo>& pAlgo, Targs&&... args) {
            m_pAlgo = pAlgo;
            pre_initialize_gl();
            pAlgo->initialize_gl(m_oCurrInput,m_pLoader->getInputROI(m_nCurrIdx),std::forward<Targs>(args)...);
            post_initialize_gl();
        }
        /// calls 'apply_gl' from 'Talgo' interface with expanded args list
        template<typename Talgo, typename... Targs>
        void apply_gl(const std::shared_ptr<Talgo>& pAlgo, size_t nNextIdx, bool bRebindAll, Targs&&... args) {
            m_pAlgo = pAlgo;
            pre_apply_gl(nNextIdx,bRebindAll);
            // @@@@@ allow apply with new roi for each packet? (like init?)
            pAlgo->apply_gl(m_oNextInput,bRebindAll,std::forward<Targs>(args)...);
            post_apply_gl(nNextIdx,bRebindAll);
        }
    protected:
        /// initializes internal async packet fetching indexes
        IAsyncDataConsumer_();
        /// called just before the GL algorithm/evaluator are initialized
        virtual void pre_initialize_gl();
        /// called just after the GL algorithm/evaluator are initialized
        virtual void post_initialize_gl();
        /// called just before the GL algorithm/evaluator process a new packet
        virtual void pre_apply_gl(size_t nNextIdx, bool bRebindAll);
        /// called just after the GL algorithm/evaluator process a new packet
        virtual void post_apply_gl(size_t nNextIdx, bool bRebindAll);
        /// utility function for output/debug mask display (should be overloaded if also evaluating results)
        virtual void getColoredMasks(cv::Mat& oOutput, cv::Mat& oDebug, const cv::Mat& oGT=cv::Mat(), const cv::Mat& oGTROI=cv::Mat());
        std::shared_ptr<lv::IParallelAlgo_GLSL> m_pAlgo;
        std::shared_ptr<GLImageProcEvaluatorAlgo> m_pEvalAlgo;
        std::shared_ptr<IIDataLoader> m_pLoader;
        cv::Mat m_oLastInput,m_oCurrInput,m_oNextInput;
        cv::Mat m_oLastGT,m_oCurrGT,m_oNextGT;
        size_t m_nLastIdx,m_nCurrIdx,m_nNextIdx;
        AsyncDataCallbackFunc m_lDataCallback;
    };

#endif //HAVE_GLSL

} // namespace lv
