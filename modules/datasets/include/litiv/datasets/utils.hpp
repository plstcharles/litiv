
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

#include "litiv/utils/algo.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <unordered_map>
#include <fstream>
#include <stack>

#ifdef _MSC_VER
// disable some very verbose warnings, use #pragma warning(enable:###) to re-enable
#pragma warning(disable:4250) // disables C4250, "'class1' : inherits 'class2::member' via dominance"
#endif //_MSC_VER

namespace lv {

    // forward declarations (real declarations further down)
    struct IDataHandler;
    using IDataHandlerPtr = std::shared_ptr<IDataHandler>;
    using IDataHandlerPtrArray = std::vector<IDataHandlerPtr>;
    using IDataHandlerConstPtr = std::shared_ptr<const IDataHandler>;
    using IDataHandlerConstPtrArray = std::vector<IDataHandlerConstPtr>;
    using AsyncDataCallbackFunc = std::function<void(const cv::Mat& /*oInput*/,const cv::Mat& /*oDebug*/,const cv::Mat& /*oOutput*/,const cv::Mat& /*oGT*/,const cv::Mat& /*oGTROI*/,size_t /*nIdx*/)>;

    /// list of computer vision tasks that can be studied using a dataset
    enum DatasetTaskList { // note: from the task type, we can derive the source and eval types
        DatasetTask_Segm, ///< image/video segmentation task id
        DatasetTask_Cosegm, ///< image/video cosegmentation task id (always array-based)
        DatasetTask_StereoReg, ///< image/video registration task id (always array-based)
        DatasetTask_EdgDet, ///< image edge detection task id
        // ...
        DatasetTask_Unspecified ///< unspecified task id; requires full specialization of dataset interfaces
    };

    /// list of data source types that can be offered by datasets
    enum DatasetSourceList { // note: from the source type, we can derive the input packet policy
        DatasetSource_Video, ///< video source id (all packets have same size)
        DatasetSource_VideoArray, ///< synchonized videos source id (array size assumed constant)
        DatasetSource_Image, ///< image source id (packets can have different sizes)
        DatasetSource_ImageArray, ///< image arrays source id (size can vary for each packet)
        // ...
        DatasetSource_Unspecified ///< unspecified source id; requires full specialization of dataset interfaces
    };

    /// list of evaluation approaches that can be used for a task
    enum DatasetEvalList { // note: from the eval type, we can derive the gt packet mapping policy
        DatasetEval_BinaryClassifier, ///< binary classification evaluation id
        DatasetEval_BinaryClassifierArray, ///< binary classification (multi-array) evaluation id
        DatasetEval_MultiClassifier, ///< multilabel classification evaluation id @@@ wip/todo
        DatasetEval_MultiClassifierArray, ///< multilabel classification (multi-array) evaluation id @@@ wip/todo
        DatasetEval_StereoDisparityEstim, ///< multilabel classification specialized for stereo disparity evaluation id
        DatasetEval_BoundingBox, ///< bounding box (for detection/tracking) evaluation id @@@ wip/todo
        // ...
        DatasetEval_None ///< no evaluation id; data consumer will only count packets & monitor processing time
    };

    /// list of datasets with built-in parsing/evaluation support
    enum DatasetList { // note: these types are used for impl specializations only
        Dataset_CDnet, ///< ChangeDetection.net (2012/2014) dataset id
        Dataset_Wallflower, ///< Wallflower dataset id
        Dataset_PETS2001D3TC1, ///< PETS2001 Dataset 3 Track 1 dataset id
        Dataset_BSDS500, ///< Berkeley Segmentation Dataset (BSDS) dataset id
        Dataset_LITIV_stcharles2015, ///< LITIV planar registration (rev2) dataset id
        Dataset_LITIV_stcharles2018, ///< LITIV cosegm/registration (rev1) dataset id
        Dataset_LITIV_bilodeau2014, ///< LITIV stereo registration (rev3) dataset id
        Dataset_VAP_trimod2016, ///< VAP Trimodal people Segmentation dataset id
        // ...
        Dataset_Custom ///< custom dataset id; 'datasets::create(...)' will forward parameters to custom constructor
    };

    /// array policy list; used to toggle data array policy functions in data handler interfaces
    enum ArrayPolicy {
        Array,
        NotArray,
    };

    /// packet policy list; used to toggle packet policy functions in data handler interfaces
    enum PacketPolicy {
        ImagePacket, ///< image packet type id; allows packets to be saved/loaded via image container formats automatically
        ImageArrayPacket, ///< image array packet type id; allows packets to be saved/loaded via image container formats automatically
        // ...
        UnspecifiedPacket ///< unspecified packet type id; forces packets to be saved/loaded via binary archive only
    };

    /// mapping policy list; used to detail the link between input/output and gt/output streams (in decreasing order of strictness)
    enum MappingPolicy {
        ElemMapping=0, ///< element-based mapping id; it means packets, matrices, and elements (or pixels) are mapped between streams
        ArrayMapping, ///< array-based mapping id; it means all packets and matrices are mapped between stream (but not their elements)
        IndexMapping, ///< index-based mapping id; it means packets are mapped between streams (but not their content)
        NoMapping ///< no mapping id; it means there is no logical link between the packets of different streams
    };

    /// returns the gt packet type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr PacketPolicy getGTPacketType() {
        // note: these are only defaults, they can be overridden via full method specialization w/ task type + dataset id
        return
            (eDatasetTask==DatasetTask_Segm)?ImagePacket:
            (eDatasetTask==DatasetTask_Cosegm)?ImageArrayPacket:
            (eDatasetTask==DatasetTask_StereoReg)?ImageArrayPacket:
            (eDatasetTask==DatasetTask_EdgDet)?ImagePacket:
            // ...
            lvStdError_(domain_error,"unknown input task id");
    }

    /// returns the output packet type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr PacketPolicy getOutputPacketType() {
        // note: these are only defaults, they can be overridden via full method specialization w/ task type + dataset id
        return
            (eDatasetTask==DatasetTask_Segm)?ImagePacket:
            (eDatasetTask==DatasetTask_Cosegm)?ImageArrayPacket:
            (eDatasetTask==DatasetTask_StereoReg)?ImageArrayPacket:
            (eDatasetTask==DatasetTask_EdgDet)?ImagePacket:
            // ...
            lvStdError_(domain_error,"unknown input task id");
    }

    /// returns the GT packet mapping style policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr MappingPolicy getGTMappingType() {
        // note: these are only defaults, they can be overridden via full method specialization w/ task type + dataset id
        return
            (eDatasetTask==DatasetTask_Segm)?ElemMapping:
            (eDatasetTask==DatasetTask_Cosegm)?ElemMapping:
            (eDatasetTask==DatasetTask_StereoReg)?ElemMapping:
            (eDatasetTask==DatasetTask_EdgDet)?ElemMapping:
            // ...
            lvStdError_(domain_error,"unknown input task id");
    }

    /// returns the I/O packet mapping style policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr MappingPolicy getIOMappingType() {
        // note: these are only defaults, they can be overridden via full method specialization w/ task type + dataset id
        return
            (eDatasetTask==DatasetTask_Segm)?ElemMapping:
            (eDatasetTask==DatasetTask_Cosegm)?IndexMapping: // may use interlaced input streams for same segmentation output
            (eDatasetTask==DatasetTask_StereoReg)?IndexMapping:
            (eDatasetTask==DatasetTask_EdgDet)?ElemMapping:
            // ...
            lvStdError_(domain_error,"unknown input task id");
    }

    /// returns the eval type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr DatasetEvalList getDatasetEval() {
        // note: these are only defaults, they can be overridden via full method specialization w/ task type + dataset id
        return
            (eDatasetTask==DatasetTask_Segm)?DatasetEval_BinaryClassifier:
            (eDatasetTask==DatasetTask_Cosegm)?DatasetEval_BinaryClassifierArray:
            (eDatasetTask==DatasetTask_StereoReg)?DatasetEval_StereoDisparityEstim:
            (eDatasetTask==DatasetTask_EdgDet)?DatasetEval_BinaryClassifier:
            // ...
            DatasetEval_None;
    }

    /// returns the source type policy to use based on the dataset task type (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetList eDataset>
    constexpr DatasetSourceList getDatasetSource() {
        // note: these are only defaults, they can be overridden via full method specialization w/ task type + dataset id
        return
            (eDatasetTask==DatasetTask_Segm)?DatasetSource_Video:
            (eDatasetTask==DatasetTask_Cosegm)?DatasetSource_VideoArray:
            (eDatasetTask==DatasetTask_StereoReg)?DatasetSource_ImageArray:
            (eDatasetTask==DatasetTask_EdgDet)?DatasetSource_Image:
            // ...
            lvStdError_(domain_error,"unknown input task id");
    }

    /// returns whether task, source, and eval types are all compatible (can also be overridden by dataset type)
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset, DatasetEvalList eDatasetEval>
    constexpr bool isDatasetSpecValid() {
        return
            (eDatasetTask==DatasetTask_Segm)?(((eDatasetSource==DatasetSource_Video)||(eDatasetSource==DatasetSource_Image))&&((eDatasetEval==DatasetEval_BinaryClassifier)||(eDatasetEval==DatasetEval_MultiClassifier)||(eDatasetEval==DatasetEval_None))):
            (eDatasetTask==DatasetTask_Cosegm)?(((eDatasetSource==DatasetSource_VideoArray)||(eDatasetSource==DatasetSource_ImageArray))&&((eDatasetEval==DatasetEval_BinaryClassifierArray)||(eDatasetEval==DatasetEval_MultiClassifierArray)||(eDatasetEval==DatasetEval_None))):
            (eDatasetTask==DatasetTask_StereoReg)?(((eDatasetSource==DatasetSource_VideoArray)||(eDatasetSource==DatasetSource_ImageArray))&&((eDatasetEval==DatasetEval_StereoDisparityEstim)||(eDatasetEval==DatasetEval_None))):
            (eDatasetTask==DatasetTask_EdgDet)?(((eDatasetSource==DatasetSource_Video)||(eDatasetSource==DatasetSource_Image))&&((eDatasetEval==DatasetEval_BinaryClassifier)||(eDatasetEval==DatasetEval_None))):
            // ...
            false;
    }

    /// returns the array type policy to use based on the dataset eval type
    template<DatasetEvalList eDatasetEval>
    constexpr ArrayPolicy getOutputArrayPolicy() { // helper func for data consumers
        return
            (eDatasetEval==DatasetEval_BinaryClassifier)?NotArray:
            (eDatasetEval==DatasetEval_BinaryClassifierArray)?Array:
            (eDatasetEval==DatasetEval_MultiClassifier)?NotArray:
            (eDatasetEval==DatasetEval_MultiClassifierArray)?Array:
            (eDatasetEval==DatasetEval_StereoDisparityEstim)?Array:
            (eDatasetEval==DatasetEval_BoundingBox)?NotArray:
            // ...
            lvStdError_(domain_error,"unknown input eval id");
    }

    /// default (specializable) forward declaration of output array policy helper, to toggle with NoEval
    template<DatasetEvalList eDatasetEval, typename ENABLE=void>
    struct OutputArrayPolicyHelper;

    /// required due to MSVC2015 failure to use constexpr functions in SFINAE expressions
    template<DatasetEvalList eDatasetEval>
    struct OutputArrayPolicyHelper<eDatasetEval,std::enable_if_t<(eDatasetEval==DatasetEval_None)>> {};

    /// required due to MSVC2015 failure to use constexpr functions in SFINAE expressions
    template<DatasetEvalList eDatasetEval>
    struct OutputArrayPolicyHelper<eDatasetEval,std::enable_if_t<(eDatasetEval!=DatasetEval_None)>> {
        static constexpr ArrayPolicy value = getOutputArrayPolicy<eDatasetEval>();
    };

    /// fully abstract data handler interface (work batch and work group implementations will derive from this)
    struct IDataHandler : lv::enable_shared_from_this<IDataHandler> {
        /// virtual destructor for adequate cleanup from IDataHandler pointers
        virtual ~IDataHandler() = default;
        /// returns the work batch/group name
        virtual const std::string& getName() const = 0;
        /// returns the work batch/group data path (always slash-terminated)
        virtual const std::string& getDataPath() const = 0;
        /// returns the work batch/group output path (always slash-terminated)
        virtual const std::string& getOutputPath() const = 0;
        /// returns the work batch/group features path (for save/load ops; always slash-terminated)
        virtual const std::string& getFeaturesPath() const = 0;
        /// returns the work batch/group relative path offset w.r.t. parent (always slash-terminated)
        virtual const std::string& getRelativePath() const = 0;
        /// returns a name (not necessarily used for parsing) associated with an input data packet index (useful for data archiving)
        virtual std::string getInputName(size_t nPacketIdx) const;
        /// returns a name that should be given to an output data packet based on its index (useful for data archiving)
        virtual std::string getOutputName(size_t nPacketIdx) const;
        /// returns a name that should be given to features data packet based on its index (useful for data archiving)
        virtual std::string getFeaturesName(size_t nPacketIdx) const;
        /// returns the tokens which, if found in a batch/group name or directory at runtime, should force it to be skipped
        virtual const std::vector<std::string>& getSkipTokens() const = 0;
        /// returns a string containing a tree representation of the batch/group data, with the given prefix
        virtual std::string printDataStructure(const std::string& sPrefix) const = 0;
        /// returns the data scaling factor to apply when loading packets
        virtual double getScaleFactor() const = 0;
        /// returns the total size (in bytes) of input data that may be loaded by the work batch/group
        virtual size_t getExpectedLoadSize() const = 0;
        /// returns the total input packet count for this work batch/group
        virtual size_t getInputCount() const = 0;
        /// returns the total gt packet count for this work batch/group
        virtual size_t getGTCount() const = 0;
        /// returns the output packet count expected to be processed by the work batch/group evaluator
        virtual size_t getExpectedOutputCount() const = 0;
        /// returns the output packet count so far processed by the work batch/group evaluator
        virtual size_t getCurrentOutputCount() const = 0;
        /// returns the final output packet count processed by the work batch/group evaluator, blocking if processing is not finished yet
        virtual size_t getFinalOutputCount() = 0;
        /// returns the time taken so far to process the work batch/group data
        virtual double getCurrentProcessTime() const = 0;
        /// returns the final time taken to process the work batch/group data, blocking if processing is not finished yet
        virtual double getFinalProcessTime() = 0;
        /// returns the top-level data handler (typically a work batch group) for this dataset
        virtual IDataHandlerConstPtr getRoot() const = 0;
        /// returns the current data handler's parent (will be null if already top level)
        virtual IDataHandlerConstPtr getParent() const = 0;
        /// resets internal work batch/group evaluation and packet count metrics
        virtual void resetMetrics() = 0;
        /// returns whether this data handler interface points to the dataset's top level (root) interface or not
        virtual bool isRoot() const = 0;
        /// returns whether loaded data should be 4-byte aligned or not
        virtual bool is4ByteAligned() const = 0;
        /// returns whether the pushed results will be saved in the output directory or not
        virtual bool isSavingOutput() const = 0;
        /// returns whether the pushed results will be evaluated or not
        virtual bool isEvaluating() const = 0;
        /// returns whether this dataset defines an evaluation procedure or not
        virtual bool isEvaluable() const {return false;}
        /// returns whether this work batch/group is currently processing data
        virtual bool isProcessing() const = 0;
        /// returns whether the work group is a pass-through container (always false for work batches)
        virtual bool isBare() const = 0;
        /// returns whether this data handler interface points to a work batch or a work group
        virtual bool isGroup() const = 0;
        /// returns the children of this work batch/group (if any) as a work batch array
        virtual IDataHandlerPtrArray getBatches(bool bWithHierarchy) const = 0;
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
        /// initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheInputOnly=true, size_t nSuggestedBufferSize=SIZE_MAX) = 0;
        /// kills the asynchronyzed precacher, and clears internal buffers
        virtual void stopPrecaching() = 0;
    protected:
        /// work batch/group comparison function based on names
        template<typename Tp>
        static std::enable_if_t<std::is_base_of<IDataHandler,Tp>::value,bool> compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return lv::compare_lowercase(i->getName(),j->getName());
        }
        /// work batch/group comparison function based on expected input data load size
        template<typename Tp>
        static std::enable_if_t<std::is_base_of<IDataHandler,Tp>::value,bool> compare_load(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return i->getExpectedLoadSize()<j->getExpectedLoadSize();
        }
        /// work batch/group comparison function based on names
        static bool compare(const IDataHandler* i, const IDataHandler* j);
        /// work batch/group comparison function based on expected CPU load
        static bool compare_load(const IDataHandler* i, const IDataHandler* j);
        /// work batch/group comparison function based on names
        static bool compare(const IDataHandler& i, const IDataHandler& j);
        /// work batch/group comparison function based on expected CPU load
        static bool compare_load(const IDataHandler& i, const IDataHandler& j);
        /// overrideable method to be called when the user 'starts processing' the data batch (by default, always inits counters after calling this)
        virtual void startProcessing_impl() {}
        /// overrideable method to be called when the user 'stops processing' the data batch (by default, always stops counters before calling this)
        virtual void stopProcessing_impl() {}
        /// local folder data parsing function, dataset-specific
        virtual void parseData() = 0;
    private:
        friend struct IDataGroupHandler; // required for recursive parsing of work batch data through groups
    };

    /// full implementation of basic data handler interface functions (used in work batch & group impl)
    struct DataHandler : public virtual IDataHandler {
        /// returns the work batch/group name
        virtual const std::string& getName() const override final;
        /// returns the work batch/group data path (always slash-terminated)
        virtual const std::string& getDataPath() const override final;
        /// returns the work batch/group output path (always slash-terminated)
        virtual const std::string& getOutputPath() const override final;
        /// returns the work batch/group features path (for save/load ops; always slash-terminated)
        virtual const std::string& getFeaturesPath() const override final;
        /// returns the work batch/group relative path offset w.r.t. parent (always slash-terminated)
        virtual const std::string& getRelativePath() const override final;
        /// returns the tokens which, if found in a batch name or directory at runtime, should force it to be skipped
        virtual const std::vector<std::string>& getSkipTokens() const override;
        /// returns a string containing a tree representation of the batch/group data, with the given prefix
        virtual std::string printDataStructure(const std::string& sPrefix) const override;
        /// returns the data scaling factor to apply when loading packets
        virtual double getScaleFactor() const override;
        /// returns the top-level data handler (typically a work batch group) for this dataset
        virtual IDataHandlerConstPtr getRoot() const override;
        /// returns the current data handler's parent (will be null if already top level)
        virtual IDataHandlerConstPtr getParent() const override;
        /// returns whether this data handler interface points to the dataset's top level (root) interface or not (always false here)
        virtual bool isRoot() const override final;
        /// returns whether loaded data should be 4-byte aligned or not
        virtual bool is4ByteAligned() const override;
        /// returns whether the pushed results will be saved in the output directory or not
        virtual bool isSavingOutput() const override;
        /// returns whether the pushed results will be evaluated or not
        virtual bool isEvaluating() const override;
        /// assembles & creates an output directory path string from a global part (which is created if needed) and a local suffix
        static std::string createOutputDir(const std::string& sGlobalDir, const std::string& sLocalDirSuffix);
    protected:
        /// fills internal impl parameters based on batch name, dataset parameters & current relative data path
        DataHandler(const std::string& sBatchName, const std::string& sRelativePath, const IDataHandler& oParent);
        const std::string m_sBatchName; ///< name of the work batch (typically taken from the batch's data directory)
        const std::string m_sRelativePath; ///< relative path from the dataset root to this work batch's root
        const std::string m_sDataPath; ///< path where the input for this work batch can be found
        const std::string m_sOutputPath; ///< path where the data generated by this work batch will be saved
        const std::string m_sFeaturesPath; ///< path for saving/loading precomputed features (useful when extraction is too slow)
        const IDataHandler& m_oParent; ///< holds a ref to this work batch's parent (should stay valid for the batch's lifespan)
        const IDataHandler& m_oRoot; ///< holds a ref to the dataset root (should stay valid for the batch's lifespan)
    };

    /// data handler full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset>
    struct DataHandler_ : public DataHandler {
    protected:
        using DataHandler::DataHandler;
    };

    /// group data parser interface for work batch groups
    struct DataGroupHandler : public virtual IDataHandler {
        /// accumulates and returns the expected load size from all children work batch data loads
        virtual size_t getExpectedLoadSize() const override final;
        /// accumulates and returns the total input packet count from all children work batch counts
        virtual size_t getInputCount() const override final;
        /// accumulate and returns the total gt packet count from all children work batch counts
        virtual size_t getGTCount() const override final;
        /// accumulates and returns the output packet count expected to be processed from all children work batch counts
        virtual size_t getExpectedOutputCount() const override final;
        /// accumulates and returns the output packet count so far processed from all children work batch counts
        virtual size_t getCurrentOutputCount() const override final;
        /// accumulates and returns the final output packet count processed from all children work batch counts, blocking if processing is not finished yet
        virtual size_t getFinalOutputCount() override final;
        /// accumulates and returns the time taken so far to process all children work batches
        virtual double getCurrentProcessTime() const override final;
        /// accumulates and returns the final time taken to process all children work batches, blocking if processing is not finished yet
        virtual double getFinalProcessTime() override final;
        /// resets all internal children work batch evaluation and packet count metrics
        virtual void resetMetrics() override final;
        /// returns whether *any* children work batch is currently processing data
        virtual bool isProcessing() const override final;
        /// returns whether the work group is a pass-through container
        virtual bool isBare() const override final;
        /// returns whether this data handler interface points to a work group (always true here)
        virtual bool isGroup() const override final;
        /// returns this work group's children batches
        virtual IDataHandlerPtrArray getBatches(bool bWithHierarchy) const override final;
        /// initializes precaching in all children work batches
        virtual void startPrecaching(bool bPrecacheInputOnly=true, size_t nSuggestedBufferSize=SIZE_MAX) override final;
        /// stops precaching in all children work batches
        virtual void stopPrecaching() override final;
    protected:
        /// creates and returns a work batch for a given relative dataset path
        virtual IDataHandlerPtr createWorkBatch(const std::string& sBatchName, const std::string& sRelativePath) const = 0;
        /// creates group/nongroup workbatches based on internal dataset info and current relative path, and recursively calls parse data on all childrens
        virtual void parseData() override;
        /// protected default constructor; automatically sets 'isBare' to true
        inline DataGroupHandler() : m_bIsBare(true) {}
        /// contains the group's children work batches (which may also be groups, themselves containing children)
        IDataHandlerPtrArray m_vpBatches;
        /// defines whether the group is pass-through (i.e. contains zero or one non-group child) or not
        bool m_bIsBare;
    };

    /// group data parser full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset>
    struct DataGroupHandler_ : public DataGroupHandler {};

    /// data handler specialized templace getters interface (shared by work batches, groups, and dataset interfaces)
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset, DatasetEvalList eDatasetEval>
    struct DataTemplSpec_ : public virtual IDataHandler {
        /// returns which processing task this work batch/group was built for
        virtual DatasetTaskList getDatasetTask() const override final {return eDatasetTask;}
        /// returns which data source this work batch/group was built for
        virtual DatasetSourceList getDatasetSource() const override final {return eDatasetSource;}
        /// returns which evaluation method this work batch/group was built for
        virtual DatasetEvalList getDatasetEval() const override final {return eDatasetEval;}
        /// returns which dataset this work batch/group was built for
        virtual DatasetList getDataset() const override final {return eDataset;}
    };

    /// general-purpose data packet precacher, fully implemented (i.e. can be used stand-alone)
    struct DataPrecacher {
        /// attaches to data loader (will halt auto-precaching if an empty packet is fetched)
        DataPrecacher(std::function<cv::Mat(size_t)> lDataLoaderCallback);
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
        /// returns the last requested packet index (i.e. the index to data still being held)
        inline size_t getLastReqIdx() const {return m_nLastReqIdx;}
    private:
        void entry(const size_t nBufferSize);
        const std::function<cv::Mat(size_t)> m_lCallback;
        std::thread m_hWorker;
        std::exception_ptr m_pWorkerException;
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
        virtual void startPrecaching(bool bPrecacheInputOnly=true, size_t nSuggestedBufferSize=SIZE_MAX) override;
        /// kills the asynchronyzed precacher, and clears internal buffers
        virtual void stopPrecaching() override;
        /// returns an input packet by index (works both with and without precaching enabled)
        const cv::Mat& getInput(size_t nPacketIdx);
        /// returns a gt packet by index (works both with and without precaching enabled)
        const cv::Mat& getGT(size_t nPacketIdx);
        /// loads a user-defined features data packet by index (works both with and without precaching enabled)
        const cv::Mat& loadFeatures(size_t nPacketIdx);
        /// saves a user-defined features data packet by index (useful when extraction is hard/slow)
        void saveFeatures(size_t nPacketIdx, const cv::Mat& oFeatures) const;
        /// returns the ROI associated with an input packet by index (returns empty mat by default)
        virtual const cv::Mat& getInputROI(size_t nPacketIdx) const;
        /// returns the ROI associated with a gt packet by index (returns empty mat by default)
        virtual const cv::Mat& getGTROI(size_t nPacketIdx) const;
        /// returns the size/type associated with an input packet by index
        virtual lv::MatInfo getInputInfo(size_t nPacketIdx) const = 0;
        /// returns the size/type associated with a gt packet by index
        virtual lv::MatInfo getGTInfo(size_t nPacketIdx) const = 0;
        /// returns whether the input packets are constant-sized and constant-typed
        virtual bool isInputInfoConst() const = 0;
        /// returns whether the gt packets are constant-sized and constant-typed
        virtual bool isGTInfoConst() const = 0;
    protected:
        /// types serve to automatically transform packets & define default implementations
        IIDataLoader(PacketPolicy eInputType, PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        /// features packet load function (can return empty mat)
        virtual cv::Mat loadRawFeatures(size_t nPacketIdx);
        /// input packet transformation function (used e.g. for rescaling and color space conversion on images)
        virtual cv::Mat getInput_redirect(size_t nPacketIdx);
        /// gt packet transformation function (used e.g. for rescaling and color space conversion on images)
        virtual cv::Mat getGT_redirect(size_t nPacketIdx);
    private:
        /// required friend for access to precachers
        template<ArrayPolicy ePolicy>
        friend struct IDataLoader_;
        /// precacher objects which may spin up a thread to pre-fetch data packets
        DataPrecacher m_oInputPrecacher,m_oGTPrecacher,m_oFeaturesPrecacher;
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
        /// needed for raw calls in redirection methods
        friend struct IIDataLoader;
        /// pass-through constructor to super-interface
        using IIDataLoader::IIDataLoader;
        /// input packet load function, pre-transformations (can return empty mat)
        virtual cv::Mat getRawInput(size_t nPacketIdx) = 0;
        /// gt packet load function, pre-transformations (can return empty mat)
        virtual cv::Mat getRawGT(size_t nPacketIdx) = 0;
    };

    /// data loader specialization for array processing (exposes unpacked array getters)
    template<>
    struct IDataLoader_<Array> : public IIDataLoader {
        /// returns the number of parallel input streams (must be overloaded)
        virtual size_t getInputStreamCount() const = 0;
        /// returns the number of parallel gt streams (defaults to 0)
        virtual size_t getGTStreamCount() const;
        /// returns the (friendly) name of an input stream specified by index
        virtual std::string getInputStreamName(size_t nStreamIdx) const;
        /// returns the (friendly) name of a gt stream specified by index
        virtual std::string getGTStreamName(size_t nStreamIdx) const;
        /// unpacks and returns an input array by packet index, with each stream its own cv::Mat (works both with and without precaching enabled)
        const std::vector<cv::Mat>& getInputArray(size_t nPacketIdx);
        /// unpacks and returns a gt array by packet index, with each stream its own cv::Mat (works both with and without precaching enabled)
        const std::vector<cv::Mat>& getGTArray(size_t nPacketIdx);
        /// unpacks and returns a user-defined features data packet by index, with provided pack info (works both with and without precaching enabled)
        const std::vector<cv::Mat>& loadFeaturesArray(size_t nPacketIdx, const std::vector<lv::MatInfo>& vPackingInfo);
        /// packs and saves a user-defined features data array packet by index, with each (input) stream its own cv::Mat (useful when necessary features extraction is hard/slow)
        void saveFeaturesArray(size_t nPacketIdx, const std::vector<cv::Mat>& oFeatures, std::vector<lv::MatInfo>* pvOutputPackingInfo=nullptr) const;
        /// unpacks and returns an input ROI array by packet index, with each stream its own cv::Mat (returns vector of empty mats by default)
        virtual const std::vector<cv::Mat>& getInputROIArray(size_t nPacketIdx) const;
        /// unpacks and returns a gt ROI array by packet index, with each stream its own cv::Mat (returns vector of empty mats by default)
        virtual const std::vector<cv::Mat>& getGTROIArray(size_t nPacketIdx) const;
        /// returns the size/type array associated with an input packet by index
        virtual std::vector<lv::MatInfo> getInputInfoArray(size_t nPacketIdx) const = 0;
        /// returns the size/type array associated with a gt packet by index
        virtual std::vector<lv::MatInfo> getGTInfoArray(size_t nPacketIdx) const = 0;
    protected:
        /// needed for raw calls in redirection methods
        friend struct IIDataLoader;
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
        /// hides the 'packed' input array info getter from public interface (throws by default)
        virtual lv::MatInfo getInputInfo(size_t nPacketIdx) const override;
        /// hides the 'packed' gt array info getter from public interface (throws by default)
        virtual lv::MatInfo getGTInfo(size_t nPacketIdx) const override;
        /// input array load function, pre-transformations (can return vector of empty mats)
        virtual std::vector<cv::Mat> getRawInputArray(size_t nPacketIdx) = 0;
        /// gt array load function, pre-transformations (can return vector of empty mats)
        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) = 0;
    private:
        std::vector<cv::Mat> m_vLatestUnpackedInput,m_vLatestUnpackedGT,m_vLatestUnpackedFeatures;
        mutable std::vector<cv::Mat> m_vEmptyInputROIArray,m_vEmptyGTROIArray;
    };

    /// default (specializable) forward declaration of the data producer interface
    /// note: producer must be fully specialized via DataProducer_<...> if using 'unspecified' source packet template type
    template<DatasetSourceList eDatasetSource>
    struct IDataProducer_;

    /// data producer specialization for video processing
    template<>
    struct IDataProducer_<DatasetSource_Video> :
            public IDataLoader_<NotArray> {
        /// returns the ROI associated with all frames
        virtual const cv::Mat& getFrameROI() const;
        /// returns the size info associated with all input frames
        virtual lv::MatSize getFrameSize() const;
        /// returns the size/type associated with all input frames
        virtual lv::MatInfo getInputInfo() const;
        /// returns the size/type associated with all gt frames
        virtual lv::MatInfo getGTInfo() const;
        /// returns the total frame count for this work batch/group (redirects to getInputCount())
        inline size_t getFrameCount() const {return getInputCount();}
        /// returns the total input frame count for this work batch/group
        virtual size_t getInputCount() const override;
        /// returns the total gt frame count for this work batch/group
        virtual size_t getGTCount() const override;
        /// compute the expected data load size for this batch based on frame size, frame count, and channel count
        virtual size_t getExpectedLoadSize() const override;
    protected:
        /// specialized constructor; still need to specify gt type, output type, and mappings
        IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        virtual const cv::Mat& getInputROI(size_t nPacketIdx) const override; // hidden; we assume input roi = frame roi
        virtual const cv::Mat& getGTROI(size_t nPacketIdx) const override; // hidden; we assume gt roi = frame roi
        virtual lv::MatInfo getInputInfo(size_t nPacketIdx) const override; // hidden; we assume all input packets info constant
        virtual lv::MatInfo getGTInfo(size_t nPacketIdx) const override; // hidden; we assume all gtpackets info constant
        virtual bool isInputInfoConst() const override final; // hidden; we assume 'yes' (obvious due to producer spec)
        virtual bool isGTInfoConst() const override final; // hidden; we assume 'yes' (obvious due to producer spec)
        virtual cv::Mat getRawInput(size_t nPacketIdx) override;
        virtual cv::Mat getRawGT(size_t nPacketIdx) override;
        virtual void parseData() override;
        size_t m_nFrameCount; ///< needed as a separate variable for VideoCapture+imread support
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::string> m_vsInputPaths,m_vsGTPaths;
        cv::VideoCapture m_voVideoReader;
        size_t m_nNextExpectedVideoReaderFrameIdx;
        cv::Mat m_oInputROI,m_oGTROI;
        lv::MatInfo m_oInputInfo,m_oGTInfo;
    };

    /// data producer specialization for multi-video processing
    template<>
    struct IDataProducer_<DatasetSource_VideoArray> :
            public IDataLoader_<Array> {
        /// returns the ROIs associated with all frames
        virtual const std::vector<cv::Mat>& getFrameROIArray() const;
        /// returns the sizes associated with all frames
        virtual std::vector<cv::Size> getFrameSizeArray() const;
        /// returns the size/type array associated with all input frames
        virtual std::vector<lv::MatInfo> getInputInfoArray() const;
        /// returns the size/type array associated with all gt frames
        virtual std::vector<lv::MatInfo> getGTInfoArray() const;
        /// returns the total frame count for this work batch/group (redirects to getInputCount())
        inline size_t getFrameCount() const {return getInputCount();}
        /// returns the total input frame count for this work batch/group --- we assume all streams are sync'd
        virtual size_t getInputCount() const override;
        /// returns the total gt frame count for this work batch/group --- we assume all streams are sync'd
        virtual size_t getGTCount() const override;
        /// compute the expected data load size for this batch based on frame size, frame count, channel count, and stream count
        virtual size_t getExpectedLoadSize() const override;
    protected:
        /// specialized constructor; still need to specify gt type, output type, and mappings
        IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        virtual const std::vector<cv::Mat>& getInputROIArray(size_t nPacketIdx) const override; // hidden; we assume input roi = frame roi
        virtual const std::vector<cv::Mat>& getGTROIArray(size_t nPacketIdx) const override; // hidden; we assume gt roi = frame roi
        virtual std::vector<lv::MatInfo> getInputInfoArray(size_t nPacketIdx) const override; // hidden; we assume all input packets info constant
        virtual std::vector<lv::MatInfo> getGTInfoArray(size_t nPacketIdx) const override; // hidden; we assume all gt packets info constant
        virtual bool isInputInfoConst() const override final; // hidden; we assume 'yes' (obvious due to producer spec)
        virtual bool isGTInfoConst() const override final; // hidden; we assume 'yes' (obvious due to producer spec)
        //virtual void parseData() override; // we provide no default impl; you need to override this yourself!
        virtual std::vector<cv::Mat> getRawInputArray(size_t nPacketIdx) override;
        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) override;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::vector<std::string>> m_vvsInputPaths,m_vvsGTPaths; // first dimension is packet index, 2nd is stream index
        std::vector<cv::Mat> m_vInputROIs,m_vGTROIs; // one ROI per stream
        std::vector<lv::MatInfo> m_vInputInfos,m_vGTInfos;
    };

    /// data producer specialization for image processing
    template<>
    struct IDataProducer_<DatasetSource_Image> :
            public IDataLoader_<NotArray> {
        /// returns the total image count for this work batch/group (redirects to getInputCount())
        inline size_t getImageCount() const {return getInputCount();}
        /// returns the total input image count for this work batch/group
        virtual size_t getInputCount() const override;
        /// returns the total gt image count for this work batch/group
        virtual size_t getGTCount() const override;
        /// compute the expected data load size for this batch based on image sizes, image count, and channel count
        virtual size_t getExpectedLoadSize() const override;
        /// returns the size/type associated with an input image by index
        virtual lv::MatInfo getInputInfo(size_t nPacketIdx) const override;
        /// returns the size/type associated with a gt image by index
        virtual lv::MatInfo getGTInfo(size_t nPacketIdx) const override;
        /// returns whether the input packets are constant-sized and constant-typed
        virtual bool isInputInfoConst() const override;
        /// returns whether the gt packets are constant-sized and constant-typed
        virtual bool isGTInfoConst() const override;
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
        std::vector<lv::MatInfo> m_vInputInfos,m_vGTInfos;
        bool m_bIsInputInfoConst,m_bIsGTInfoConst;
    };

    /// data producer specialization for multi-image processing
    template<>
    struct IDataProducer_<DatasetSource_ImageArray> :
            public IDataLoader_<Array> {
        /// returns the total image count for this work batch/group (redirects to getInputCount())
        inline size_t getImageCount() const {return getInputCount();}
        /// returns the total input image count for this work batch/group
        virtual size_t getInputCount() const override;
        /// returns the total gt image count for this work batch/group
        virtual size_t getGTCount() const override;
        /// compute the expected data load size for this batch based on image sizes, image count, channel count, and stream count
        virtual size_t getExpectedLoadSize() const override;
        /// returns the size associated with an input image by index
        virtual std::vector<lv::MatInfo> getInputInfoArray(size_t nPacketIdx) const override;
        /// returns the size associated with a gt image by index
        virtual std::vector<lv::MatInfo> getGTInfoArray(size_t nPacketIdx) const override;
        /// returns whether the input packets are constant-sized and constant-typed
        virtual bool isInputInfoConst() const override;
        /// returns whether the gt packets are constant-sized and constant-typed
        virtual bool isGTInfoConst() const override;
    protected:
        /// specialized constructor; still need to specify gt type, output type, and mappings
        IDataProducer_(PacketPolicy eGTType, PacketPolicy eOutputType, MappingPolicy eGTMappingType, MappingPolicy eIOMappingType);
        //virtual void parseData() override; // we provide no default impl; you need to override this yourself!
        virtual std::vector<cv::Mat> getRawInputArray(size_t nPacketIdx) override;
        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) override;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::vector<std::string>> m_vvsInputPaths,m_vvsGTPaths; // one path per packet per stream
        std::vector<std::vector<lv::MatInfo>> m_vvInputInfos,m_vvGTInfos; // one size/type per packet per stream
        bool m_bIsInputInfoConst,m_bIsGTInfoConst;
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
        /// returns whether the given packet could be added to the queue (true), or it would be dropped (false)
        bool queue_check(const cv::Mat& oPacket, size_t nIdx);
        /// queues a packet, with or without async writing enabled, and returns its position in queue
        size_t queue(const cv::Mat& oPacket, size_t nIdx);
        /// returns the current queue size, in packets
        inline size_t getCurrentQueueCount() const {return m_nQueueCount;}
        /// returns the current queue size, in bytes
        inline size_t getCurrentQueueSize() const {return m_nQueueSize;}
        /// returns the maximum queue size, in bytes
        inline size_t getMaxQueueSize() const {return m_nQueueMaxSize;}
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
        std::stack<std::pair<std::exception_ptr,size_t>> m_vWorkerExceptions;
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
        /// loads an output data packet based on idx, with optional flags (-1 = internal defaults)
        virtual cv::Mat loadOutput(size_t nIdx, int nFlags=-1);
    protected:
        /// saves an output data packet based on idx, with optional flags (-1 = internal defaults)
        virtual void saveOutput(const cv::Mat& oOutput, size_t nIdx, int nFlags=-1);
    };

    /// data archiver specialization for array output processing
    template<>
    struct IDataArchiver_<Array> : public virtual IDataHandler {
        /// loads an output data packet array based on idx, with optional flags (-1 = internal defaults)
        virtual std::vector<cv::Mat> loadOutputArray(size_t nIdx, int nFlags=-1);
        /// returns the number of parallel output streams (defaults to input or GT stream count if loader is array-based & one mapping allows it)
        virtual size_t getOutputStreamCount() const;
    protected:
        /// saves an output data packet array based on idx, with optional flags (-1 = internal defaults)
        virtual void saveOutputArray(const std::vector<cv::Mat>& vOutput, size_t nIdx, int nFlags=-1);
    };

    /// data counter interface for non-group work batches (exposes output packet counting logic)
    struct IDataCounter : public virtual IDataHandler {
        /// returns the output packet count so far processed by the work batch evaluator
        virtual size_t getCurrentOutputCount() const override final;
        /// returns the final output packet count processed by the work batch evaluator, blocking if processing is not finished yet
        virtual size_t getFinalOutputCount() override final;
    protected:
        /// checks output with index 'nPacketIdx' as processed
        void countOutput(size_t nPacketIdx);
        /// sets the processed packets count promise for async count fetching
        void setOutputCountPromise();
        /// resets the processed packets count (and reinitializes promise)
        void resetOutputCount();
        /// default constructor (calls resetOutputCount to initialize all members)
        inline IDataCounter() {resetOutputCount();}
    private:
        std::unordered_set<size_t> m_mProcessedPackets;
        std::promise<size_t> m_nPacketCountPromise;
        std::future<size_t> m_nPacketCountFuture;
        size_t m_nFinalPacketCount;
    };

    /// default (specializable) forward declaration of the data consumer interface
    template<DatasetEvalList eDatasetEval, typename ENABLE=void>
    struct IDataConsumer_;

    /// data consumer specialization for process monitoring only (no-eval entrypoint)
    template<DatasetEvalList eDatasetEval>
    struct IDataConsumer_<eDatasetEval,std::enable_if_t<eDatasetEval==DatasetEval_None>> :
            public IDataCounter {
        /// returns the total output packet count expected to be processed by the data consumer (defaults to 0)
        virtual size_t getExpectedOutputCount() const override {
            return 0;
        }
        /// resets internal packet count metrics (no evaluation metrics for this interface)
        virtual void resetMetrics() override {
            resetOutputCount();
        }
        /// pushes an output packet index for counting/speed analysis
        inline void push(size_t nPacketIdx) {
            lvAssert_(isProcessing(),"data processing must be toggled via 'startProcessing()' before pushing indices");
            countOutput(nPacketIdx);
        }
    };

    /// data consumer specialization for receiving processed packets (evaluation entrypoint)
    template<DatasetEvalList eDatasetEval>
    struct IDataConsumer_<eDatasetEval,std::enable_if_t<OutputArrayPolicyHelper<eDatasetEval>::value==NotArray>> :
            public IDataArchiver_<NotArray>,
            public IDataCounter {
        /// returns the total output packet count expected to be processed by the data consumer (defaults to GT count)
        virtual size_t getExpectedOutputCount() const override {
            return getGTCount();
        }
        /// resets internal packet count metrics (no evaluation metrics for this interface)
        virtual void resetMetrics() override {
            resetOutputCount();
        }
        /// pushes an output (processed) data packet for writing and/or evaluation
        inline void push(const cv::Mat& oOutput, size_t nPacketIdx) {
            lvAssert_(isProcessing(),"data processing must be toggled via 'startProcessing()' before pushing packets");
            countOutput(nPacketIdx);
            processOutput(oOutput,nPacketIdx);
            if(isSavingOutput() && !oOutput.empty())
                this->saveOutput(oOutput,nPacketIdx);
        }
    protected:
        /// processes an output packet (does nothing by default, but may be overridden for evaluation/pipelining)
        virtual void processOutput(const cv::Mat& /*oOutput*/, size_t /*nPacketIdx*/) {}
    };

    /// data consumer specialization for receiving processed packet arrays (evaluation entrypoint)
    template<DatasetEvalList eDatasetEval>
    struct IDataConsumer_<eDatasetEval,std::enable_if_t<OutputArrayPolicyHelper<eDatasetEval>::value==Array>> :
            public IDataArchiver_<Array>,
            public IDataCounter {
        /// returns the total output packet count expected to be processed by the data consumer (defaults to GT count)
        virtual size_t getExpectedOutputCount() const override {
            return getGTCount();
        }
        /// resets internal packet count metrics (no evaluation metrics for this interface)
        virtual void resetMetrics() override {
            resetOutputCount();
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
            if(isSavingOutput() && !vOutput.empty())
                this->saveOutputArray(vOutput,nPacketIdx);
        }
        /// pushes an output (processed) data packet array for writing and/or evaluation
        template<size_t nArraySize>
        inline void push(const std::array<cv::Mat,nArraySize>& aOutput, size_t nPacketIdx) {
            return push(std::vector<cv::Mat>(aOutput.begin(),aOutput.end()),nPacketIdx);
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
            public IDataCounter {
        /// returns the ideal size for the GL context window to use for debug display purposes (queries the algo based on dataset specs, if available)
        virtual cv::Size getIdealGLWindowSize() const;
        /// returns the total output packet count expected to be processed by the data consumer (defaults to GT count)
        virtual size_t getExpectedOutputCount() const override {
            return getGTCount();
        }
        /// resets internal packet count metrics (no evaluation metrics for this interface)
        virtual void resetMetrics() override {
            resetOutputCount();
        }
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

    /// full implementation of basic dataset handler interface functions
    struct DatasetHandler : public virtual IDataHandler {
        /// returns the dataset name
        virtual const std::string& getName() const override final;
        /// returns the dataset data path (always slash-terminated)
        virtual const std::string& getDataPath() const override final;
        /// returns the dataset output path (always slash-terminated)
        virtual const std::string& getOutputPath() const override final;
        /// returns the dataset features path (for save/load ops; always slash-terminated)
        virtual const std::string& getFeaturesPath() const override final;
        /// returns the dataset relative path offset (always empty string, since this is root interface)
        virtual const std::string& getRelativePath() const override final;
        /// returns the directory names of potential top-level work batches/groups
        virtual const std::vector<std::string>& getWorkBatchDirs() const;
        /// returns the tokens which, if found in a batch/group name or directory at runtime, should force it to be skipped
        virtual const std::vector<std::string>& getSkipTokens() const override final;
        /// returns a string containing a tree representation of the dataset, with the given prefix
        virtual std::string printDataStructure(const std::string& sPrefix) const override;
        /// returns the data scaling factor to apply when loading packets
        virtual double getScaleFactor() const override final;
        /// returns the top-level data handler for this dataset (i.e. this instance)
        virtual IDataHandlerConstPtr getRoot() const override;
        /// returns the current data handler's parent (always null here, since we are the root)
        virtual IDataHandlerConstPtr getParent() const override;
        /// returns whether this data handler interface points to the dataset's top level (root) interface or not (always true here)
        virtual bool isRoot() const override final;
        /// returns whether loaded data should be 4-byte aligned or not
        virtual bool is4ByteAligned() const override final;
        /// returns whether the pushed results will be saved in the output directory or not
        virtual bool isSavingOutput() const override final;
        /// returns whether the pushed results will be evaluated or not
        virtual bool isEvaluating() const override final;
    protected:
        /// full dataset handler constructor; parameters are passed through lv::datasets::create<...>(...), and may be caught/simplified by a specialization
        DatasetHandler(
            const std::string& sDatasetName, ///< user-friendly dataset name (used for identification only)
            const std::string& sDatasetDirPath, ///< root path from which work batches can be parsed
            const std::string& sOutputDirPath, ///< root path for work batch output (debug logs, evaluation reports, and generated results)
            const std::vector<std::string>& vsWorkBatchDirs, ///< array of directory names for top-level work batch groups (one group typically contains multiple work batches)
            const std::vector<std::string>& vsSkippedDirTokens, ///< array of tokens which allow directories to be skipped if one is found in their name
            bool bSaveOutput, ///< defines whether results should be archived or not
            bool bUseEvaluator, ///< defines whether results should be fully evaluated, or simply acknowledged
            bool bForce4ByteDataAlign=false, ///< defines whether data packets should be 4-byte aligned
            double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        );
    private:
        const std::string m_sDatasetName; ///< user-friendly dataset name (used for identification only)
        const std::string m_sDatasetPath; ///< root path from which work batches can be parsed
        const std::string m_sRelativePath; ///< relative path from which work batches can be parsed (always empty by default)
        const std::string m_sOutputPath; ///< root path for work batch output (debug logs, evaluation reports, and generated results)
        const std::string m_sFeaturesPath; ///< path for saving/loading precomputed dataset-wide features (useful when extraction is too slow)
        const std::vector<std::string> m_vsWorkBatchDirs; ///< array of directory names for top-level work batch groups (one group typically contains multiple work batches)
        const std::vector<std::string> m_vsSkippedDirTokens; ///< array of tokens which allow directories to be skipped if one is found in their name
        const bool m_bSavingOutput; ///< defines whether results should be archived or not
        const bool m_bUsingEvaluator; ///< defines whether results should be fully evaluated, or simply acknowledged
        const bool m_bForce4ByteDataAlign; ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
        const double m_dScaleFactor; ///< defines the scale factor to use to resize/rescale read packets
    };

    /// dataset handler full (default) specialization --- can be overridden by dataset type in 'impl' headers
    template<DatasetTaskList eDatasetTask, DatasetSourceList eDatasetSource, DatasetList eDataset>
    struct DatasetHandler_ : public DatasetHandler {
    protected:
        using DatasetHandler::DatasetHandler;
    };

} // namespace lv
