
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

#include "litiv/utils/ParallelUtils.hpp"
#include "litiv/utils/OpenCVUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// as defined in the 2012/2014 CDNet evaluation scripts
#define DATASETUTILS_POSITIVE_VAL    uchar(255)
#define dATASETUTILS_NEGATIVE_VAL    uchar(0)
#define DATASETUTILS_OUTOFSCOPE_VAL  uchar(85)
#define DATASETUTILS_UNKNOWN_VAL     uchar(170)
#define DATASETUTILS_SHADOW_VAL      uchar(50)

// as defined in the bsds500 evaluation script
#define DATASETUTILS_IMAGEEDGDET_EVAL_THRESHOLD_BINS 99

namespace litiv {

    enum eDatasetTaskList { // from the task type, we can derive the source and eval types
        eDatasetTask_ChgDet,
        eDatasetTask_Segm,
        eDatasetTask_Registr,
        eDatasetTask_EdgDet,
        // ...
    };

    enum eDatasetSourceList { // from the source type, we can derive the input packet policy
        eDatasetSource_Video,
        eDatasetSource_VideoArray,
        eDatasetSource_Image,
        eDatasetSource_ImageArray,
        // ...
    };

    enum eDatasetEvalList { // from the eval type, we can derive the gt packet mapping policy
        eDatasetEval_BinaryClassifier,
        eDatasetEval_Registr,
        eDatasetEval_Segm,
        eDatasetEval_BoundingBox,
        // ...
        eDatasetEval_None // will only count packets & monitor processing time
    };

    enum eDatasetList { // dataset types are used for impl specializations only
        eDataset_CDnet,
        eDataset_Wallflower,
        eDataset_PETS2001D3TC1,
        eDataset_LITIV2012b,
        eDataset_BSDS500,
        // ...
        eDataset_Custom // 'datasets::create' will forward all parameters from Dataset constr
    };

    enum eGroupPolicy { // used to toggle group policy functions in data handler interfaces
        eGroup,
        eNotGroup,
    };

    enum ePacketPolicy { // used to toggle packet policy functions in data handler interfaces
        eImagePacket,
        eNotImagePacket
    };

    enum eGTMappingPolicy { // used to determine if input packet transformations should also be applied to GT (e.g. scaling)
        eDirectPixelMapping,
        ePacketIdxMapping,
        eBatchMapping,
        eNoMapping
    };

    //! returns the GT packet mapping style policy to use based on the dataset task type
    template<eDatasetTaskList eDatasetTask>
    constexpr eGTMappingPolicy getGTMappingType() {
        return (eDatasetTask==eDatasetTask_ChgDet)?eDirectPixelMapping:
               (eDatasetTask==eDatasetTask_Segm)?eDirectPixelMapping:
               (eDatasetTask==eDatasetTask_Registr)?eBatchMapping:
               (eDatasetTask==eDatasetTask_EdgDet)?ePacketIdxMapping:
               // ...
               throw -1; // undefined behavior
    }

    struct IDataset;
    struct IDataHandler;
    using IDatasetPtr = std::shared_ptr<IDataset>;
    using IDataHandlerPtr = std::shared_ptr<IDataHandler>;
    using IDataHandlerPtrArray = std::vector<IDataHandlerPtr>;
    using IDataHandlerConstPtr = std::shared_ptr<const IDataHandler>;
    using IDataHandlerConstPtrArray = std::vector<IDataHandlerConstPtr>;
    using IDataHandlerPtrQueue = std::priority_queue<IDataHandlerPtr,IDataHandlerPtrArray,std::function<bool(const IDataHandlerPtr&,const IDataHandlerPtr&)>>;
    using AsyncDataCallbackFunc = std::function<void(const cv::Mat& /*oInput*/,const cv::Mat& /*oDebug*/,const cv::Mat& /*oOutput*/,const cv::Mat& /*oGT*/,const cv::Mat& /*oROI*/,size_t /*nIdx*/)>;

    //! fully abstract dataset interface (dataset parser & evaluator implementations will derive from this)
    struct IDataset : CxxUtils::enable_shared_from_this<IDataset> {
        //! returns the dataset name
        virtual const std::string& getName() const = 0;
        //! returns the root data path
        virtual const std::string& getDatasetPath() const = 0;
        //! returns the root output path
        virtual const std::string& getOutputPath() const = 0;
        //! returns the output file name prefix for results archiving
        virtual const std::string& getOutputNamePrefix() const = 0;
        //! returns the output file name suffix for results archiving
        virtual const std::string& getOutputNameSuffix() const = 0;
        //! returns the directory names of top-level work batches
        virtual const std::vector<std::string>& getWorkBatchDirs() const = 0;
        //! returns the directory name tokens which, if found, should be skipped
        virtual const std::vector<std::string>& getSkippedDirTokens() const = 0;
        //! returns the directory name tokens which, if found, should be treated as grayscale
        virtual const std::vector<std::string>& getGrayscaleDirTokens() const = 0;
        //! returns the output file/packet index offset for results archiving
        virtual size_t getOutputIdxOffset() const = 0;
        //! returns the input data scaling scaling factor
        virtual double getScaleFactor() const = 0;
        //! returns whether we should save the results through DataConsumers or not
        virtual bool isSavingOutput() const = 0;
        //! returns whether we should evaluate the results through DataConsumers or not
        virtual bool isUsingEvaluator() const = 0;
        //! returns whether loaded data should be 4-byte aligned or not (4-byte alignment is ideal for GPU upload)
        virtual bool is4ByteAligned() const = 0;
        //! virtual destructor for adequate cleanup from IDataset pointers
        virtual ~IDataset() = default;
        //! returns the total number of packets in the dataset (recursively queried from work batches)
        virtual size_t getTotPackets() const = 0;
        //! returns the total time it took to process the dataset (recursively queried from work batches)
        virtual double getProcessTime() const = 0;
        //! returns the total processed packet count, blocking if processing is not finished yet (recursively queried from work batches)
        virtual size_t getProcessedPacketsCountPromise() = 0;
        //! returns the total processed packet count (recursively queried from work batches)
        virtual size_t getProcessedPacketsCount() const = 0;
        //! clears all batches and reparses them from the dataset metadata
        virtual void parseDataset() = 0;
        //! writes the dataset-level evaluation report
        virtual void writeEvalReport() const = 0;
        //! returns the array of work batches (or groups) contained in this dataset
        virtual IDataHandlerPtrArray getBatches() const = 0;
        //! returns the array of work batches (or groups) contained in this dataset, sorted by expected CPU load
        virtual IDataHandlerPtrQueue getSortedBatches() const = 0;
    };

    //! fully abstract data handler interface (work batch and work group implementations will derive from this)
    struct IDataHandler : CxxUtils::enable_shared_from_this<IDataHandler> {
        //! returns the work batch/group name
        virtual const std::string& getName() const = 0;
        //! returns the work batch/group data path
        virtual const std::string& getDataPath() const = 0;
        //! returns the work batch/group output path
        virtual const std::string& getOutputPath() const = 0;
        //! returns the work batch/group relative path offset w.r.t. dataset root
        virtual const std::string& getRelativePath() const = 0;
        //! returns the expected CPU load of the work batch/group (only relevant for intra-dataset load comparisons)
        virtual double getExpectedLoad() const = 0;
        //! returns the total packet count for this work batch/group
        virtual size_t getTotPackets() const = 0;
        //! returns whether the work batch/group data will be treated as grayscale
        virtual bool isGrayscale() const = 0;
        //! returns whether the work group is a pass-through container (always false for work batches)
        virtual bool isBare() const = 0;
        //! returns whether this data handler interface points to a work batch or a work group
        virtual bool isGroup() const = 0;
        //! returns this work group's children (work batch array)
        virtual IDataHandlerPtrArray getBatches() const = 0; // @@@@@ rename batches at dataset level to something else?
        //! returns a pointer to this work batch/group's parent dataset interface
        virtual IDatasetPtr getDatasetInfo() const = 0;
        //! returns which processing task this work batch/group was built for
        virtual eDatasetTaskList getDatasetTask() const = 0;
        //! returns which data source this work batch/group was built for
        virtual eDatasetSourceList getDatasetSource() const = 0;
        //! returns which dataset this work batch/group was built for
        virtual eDatasetList getDataset() const = 0;
        //! returns which evaluation method this work batch/group was built for
        virtual eDatasetEvalList getDatasetEval() const = 0;
        //! writes the batch-level evaluation report
        virtual void writeEvalReport() const = 0;
        //! virtual destructor for adequate cleanup from IDataHandler pointers
        virtual ~IDataHandler() = default;
        //! returns whether this work batch (or any of this work group's children batches) is currently processing data
        virtual bool isProcessing() const = 0;
        //! returns the current (or final) duration elapsed between start/stopProcessing calls (recursively queried for work groups)
        virtual double getProcessTime() const = 0;
        //! returns the total processed packet count, blocking if processing is not finished yet (recursively queried for work groups)
        virtual size_t getProcessedPacketsCountPromise() = 0;
        //! returns the total processed packet count (recursively queried from work batches)
        virtual size_t getProcessedPacketsCount() const = 0;
    protected:
        //! work batch/group comparison function based on names
        template<typename Tp>
        static typename std::enable_if<std::is_base_of<IDataHandler,Tp>::value,bool>::type compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return PlatformUtils::compare_lowercase(i->getName(),j->getName());
        }
        //! work batch/group comparison function based on expected CPU load
        template<typename Tp>
        static typename std::enable_if<std::is_base_of<IDataHandler,Tp>::value,bool>::type compare_load(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return i->getExpectedLoad()<j->getExpectedLoad();
        }
        //! work batch/group comparison function based on names
        static bool compare(const IDataHandler* i, const IDataHandler* j);
        //! work batch/group comparison function based on expected CPU load
        static bool compare_load(const IDataHandler* i, const IDataHandler* j);
        //! work batch/group comparison function based on names
        static bool compare(const IDataHandler& i, const IDataHandler& j);
        //! work batch/group comparison function based on expected CPU load
        static bool compare_load(const IDataHandler& i, const IDataHandler& j);
        //! returns the internal name of a given data packet (useful for data archiving)
        virtual std::string getPacketName(size_t nPacketIdx) const;
        //! returns the children batch associated with the given packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const = 0;
        //! returns the children batch associated with the given packet index; will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) = 0;
        //! overrideable method to be called when the user 'starts processing' the data batch (by default, always inits counters after calling this)
        virtual void _startProcessing() {}
        //! overrideable method to be called when the user 'stops processing' the data batch (by default, always stops counters before calling this)
        virtual void _stopProcessing() {}
        //! local folder data parsing function, dataset-specific
        virtual void parseData() = 0;
        template<eDatasetEvalList eDatasetEval>
        friend struct IDatasetEvaluator_; // required for dataset evaluator interface to write eval reports
        template<eDatasetTaskList eDatasetTask, eDatasetSourceList eDatasetSource, eDatasetList eDataset, eDatasetEvalList eDatasetEval, ParallelUtils::eParallelAlgoType eEvalImpl>
        friend struct IDataset_; // required for data handler sorting and other top-level dataset utility functions
    };

    //! general-purpose data packet precacher, fully implemented (i.e. can be used stand-alone)
    struct DataPrecacher {
        //! attaches to data loader (will halt auto-precaching if an empty packet is fetched)
        DataPrecacher(std::function<const cv::Mat&(size_t)> lDataLoaderCallback);
        //! default destructor (joins the precaching thread, if still running)
        ~DataPrecacher();
        //! fetches a packet, with or without precaching enabled -- should never be called concurrently, and packets should never be altered directly
        const cv::Mat& getPacket(size_t nIdx);
        //! initializes precaching with a given buffer size (starts up thread)
        bool startPrecaching(size_t nSuggestedBufferSize);
        //! joins precaching thread and clears all internal buffers
        void stopPrecaching();
    private:
        //! precacher entry point
        void precache(size_t nBufferSize);
        const std::function<const cv::Mat&(size_t)> m_lCallback;
        std::thread m_hPrecacher;
        std::mutex m_oSyncMutex;
        std::condition_variable m_oReqCondVar;
        std::condition_variable m_oSyncCondVar;
        bool m_bIsPrecaching;
        size_t m_nReqIdx,m_nLastReqIdx;
        cv::Mat m_oReqPacket,m_oLastReqPacket;
        DataPrecacher& operator=(const DataPrecacher&) = delete;
        DataPrecacher(const DataPrecacher&) = delete;
    };

    //! data loader interface for work batch, applies basic packet transforms where needed and relies on precacher
    struct IDataLoader : public virtual IDataHandler {
        //! returns the input data packet type policy (used for internal packet auto-transformations)
        inline ePacketPolicy getInputPacketType() const {return m_eInputType;}
        //! returns the gt data packet mapping type policy (used for internal packet auto-transformations)
        inline eGTMappingPolicy getGTMappingType() const {return m_eGTMappingType;}
        //! initializes data spooling by starting an asynchronyzed precacher to pre-fetch data packets based on queried ids
        virtual void startPrecaching(bool bPrecacheGT, size_t nSuggestedBufferSize=SIZE_MAX);
        //! kills the asynchronyzed precacher, and clears internal buffers
        virtual void stopPrecaching();
        //! returns an input packet by index (with both with and without precaching enabled)
        const cv::Mat& getInput(size_t nPacketIdx) {return m_oInputPrecacher.getPacket(nPacketIdx);}
        //! returns a gt packet by index (with both with and without precaching enabled)
        const cv::Mat& getGT(size_t nPacketIdx) {return m_oGTPrecacher.getPacket(nPacketIdx);}
        //! returns whether an input packet should be transposed or not (only applicable to image packets)
        virtual bool isInputTransposed(size_t /*nPacketIdx*/) const {return false;}
        //! returns whether a gt packet should be transposed or not (only applicable to image packets)
        virtual bool isGTTransposed(size_t /*nPacketIdx*/) const {return false;}
        //! returns the roi associated with an input packet (only applicable to image packets, or dataset-specific)
        virtual const cv::Mat& getInputROI(size_t /*nPacketIdx*/) const {return cv::emptyMat();}
        //! returns the roi associated with an input packet (only applicable to image packets, or dataset-specific)
        virtual const cv::Mat& getGTROI(size_t /*nPacketIdx*/) const {return cv::emptyMat();}
        //! returns the size of a pre-transformed input packet @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputSize(size_t nPacketIdx) const = 0;
        //! returns the size of a pre-transformed gt packet @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTSize(size_t nPacketIdx) const = 0;
        //! returns the original size of an input packet @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputOrigSize(size_t nPacketIdx) const = 0;
        //! returns the original size of a GT packet @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTOrigSize(size_t nPacketIdx) const = 0;
        //! returns the maximum size of all input packets for this data batch @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getInputMaxSize() const = 0;
        //! returns the maximum size of all gt packets for this data batch @@@@@ override later to make size N-Dim?
        virtual const cv::Size& getGTMaxSize() const = 0;
    protected:
        //! will automatically apply byte-alignment/scale in packet redirection if using image packets
        IDataLoader(ePacketPolicy eInputType, eGTMappingPolicy eGTMappingType);
        //! input packet load function, dataset-specific (can return empty mats)
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) = 0;
        //! gt packet load function, dataset-specific (can return empty mats)
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) = 0;
    private:
        cv::Mat m_oLatestInputPacket, m_oLatestGTPacket;
        DataPrecacher m_oInputPrecacher,m_oGTPrecacher;
        const cv::Mat& _getInputPacket_redirect(size_t nIdx);
        const cv::Mat& _getGTPacket_redirect(size_t nIdx);
        const ePacketPolicy m_eInputType;
        const eGTMappingPolicy m_eGTMappingType;
    };

    //! data producer interface for work batch that must be specialized based on source type (wraps data loader interface)
    template<eDatasetSourceList eDatasetSource>
    struct IDataProducer_;

    template<>
    struct IDataProducer_<eDatasetSource_Video> :
            public IDataLoader {
        //! redirects to getTotPackets()
        inline size_t getFrameCount() const {return getTotPackets();}
        //! compute the expected CPU load for this data batch based on frame size, frame count, and channel count
        virtual double getExpectedLoad() const override;
        //! initializes frame precaching for this work batch (will try to allocate enough memory for the entire sequence)
        virtual void startPrecaching(bool bUsingGT, size_t /*nUnused*/=0) override;
        //! returns the ROI associated with the video sequence (if any)
        virtual const cv::Mat& getROI() const {return m_oROI;}
        //! return the (constant) frame size used in this video sequence, post-transformations
        virtual const cv::Size& getFrameSize() const {return m_oSize;}
        //! return the original (constant) frame size used in this video sequence
        virtual const cv::Size& getFrameOrigSize() const {return m_oOrigSize;}

    protected:
        IDataProducer_(eGTMappingPolicy eGTMappingType);
        virtual size_t getTotPackets() const override;
        virtual bool isInputTransposed(size_t nPacketIdx) const override final;
        virtual bool isGTTransposed(size_t nPacketIdx) const override;
        virtual const cv::Mat& getInputROI(size_t nPacketIdx) const override final;
        virtual const cv::Mat& getGTROI(size_t nPacketIdx) const override;
        virtual const cv::Size& getInputSize(size_t nPacketIdx) const override final;
        virtual const cv::Size& getGTSize(size_t nPacketIdx) const override;
        virtual const cv::Size& getInputOrigSize(size_t nPacketIdx) const override final;
        virtual const cv::Size& getGTOrigSize(size_t nPacketIdx) const override;
        virtual const cv::Size& getInputMaxSize() const override final;
        virtual const cv::Size& getGTMaxSize() const override;
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) override;
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) override;
        virtual void parseData() override;
        size_t m_nFrameCount;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::string> m_vsInputPaths,m_vsGTPaths;
        cv::VideoCapture m_voVideoReader;
        size_t m_nNextExpectedVideoReaderFrameIdx;
        bool m_bTransposeFrames;
        cv::Mat m_oROI;
        cv::Size m_oOrigSize,m_oSize;
    };

    template<>
    struct IDataProducer_<eDatasetSource_Image> :
            public IDataLoader {
        //! redirects to getTotPackets()
        inline size_t getImageCount() const {return getTotPackets();}
        //! compute the expected CPU load for this data batch based on max image size, image count, and channel count
        virtual double getExpectedLoad() const override;
        //! initializes image precaching for this work batch (will try to allocate enough memory for the entire set)
        virtual void startPrecaching(bool bUsingGT, size_t /*nUnused*/=0) override;
        //! returns whether all input images in this batch have the same size
        virtual bool isInputConstantSize() const;
        //! returns whether all input images in this batch have the same size
        virtual bool isGTConstantSize() const;
        //! returns whether an input packet should be transposed or not (only applicable to image packets)
        virtual bool isInputTransposed(size_t nPacketIdx) const override;
        //! returns whether a gt packet should be transposed or not (only applicable to image packets)
        virtual bool isGTTransposed(size_t nPacketIdx) const override;
        //! returns the roi associated with an input image packet
        virtual const cv::Mat& getInputROI(size_t nPacketIdx) const override;
        //! returns the roi associated with a gt image packet
        virtual const cv::Mat& getGTROI(size_t nPacketIdx) const override;
        //! returns the size of a pre-transformed input image packet
        virtual const cv::Size& getInputSize(size_t nPacketIdx) const override;
        //! returns the size of a pre-transformed gt image packet
        virtual const cv::Size& getGTSize(size_t nPacketIdx) const override;
        //! returns the original size of an input image packet
        virtual const cv::Size& getInputOrigSize(size_t nPacketIdx) const override;
        //! returns the original size of a gt image packet
        virtual const cv::Size& getGTOrigSize(size_t nPacketIdx) const override;
        //! returns the maximum size of all input image packets for this data batch
        virtual const cv::Size& getInputMaxSize() const override;
        //! returns the maximum size of all image packets for this data batch
        virtual const cv::Size& getGTMaxSize() const override;
        //! returns the (file) name associated with an image packet (useful for archiving/evaluation)
        virtual std::string getPacketName(size_t nPacketIdx) const override;

    protected:
        IDataProducer_(eGTMappingPolicy eGTMappingType);
        virtual size_t getTotPackets() const override;
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) override;
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) override;
        virtual void parseData() override;
        size_t m_nImageCount;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::string> m_vsInputPaths,m_vsGTPaths;
        std::vector<cv::Size> m_voInputSizes,m_voGTSizes;
        std::vector<cv::Size> m_voInputOrigSizes,m_voGTOrigSizes;
        std::vector<bool> m_vbInputTransposed,m_vbGTTransposed;
        bool m_bIsInputConstantSize,m_bIsGTConstantSize;
        cv::Size m_oInputMaxSize,m_oGTMaxSize;
    };

    //! data producer interface specialization default constructor override for cleaner implementations
    template<eDatasetTaskList eDatasetTask, eDatasetSourceList eDatasetSource>
    struct DataProducer_c : public IDataProducer_<eDatasetSource> {
        DataProducer_c() : IDataProducer_<eDatasetSource>(getGTMappingType<eDatasetTask>()) {}
    };

    //! default data producer interface specialization (will attempt to load data using predefined functions)
    template<eDatasetTaskList eDatasetTask, eDatasetSourceList eDatasetSource, eDatasetList eDataset>
    struct DataProducer_ : public DataProducer_c<eDatasetTask,eDatasetSource> {};

    //! data counter interface for work batch/group (toggled via policy) for processed packet counting
    template<eGroupPolicy ePolicy>
    struct DataCounter_;

    template<>
    struct DataCounter_<eNotGroup> : public virtual IDataHandler {
    protected:
        //! default constructor
        DataCounter_() : m_nProcessedPackets(0) {}
        //! increments processed packets count
        inline void processPacket() {++m_nProcessedPackets;}
        //! sets processed packets count promise for async implementations
        inline void setProcessedPacketsPromise() {m_nProcessedPacketsPromise.set_value(m_nProcessedPackets);}
        //! gets processed packets count from promise for async implementations (blocks until stopProcessing is called)
        virtual size_t getProcessedPacketsCountPromise() override final;
        //! gets current processed packets count
        virtual size_t getProcessedPacketsCount() const override final;
    private:
        size_t m_nProcessedPackets;
        std::promise<size_t> m_nProcessedPacketsPromise;
    };

    template<>
    struct DataCounter_<eGroup> : public virtual IDataHandler {
        //! gets processed packets count from children batch promises for async implementations
        virtual size_t getProcessedPacketsCountPromise() override final;
        //! gets current processed packets count from children batches
        virtual size_t getProcessedPacketsCount() const override final;
    };

    //! data archiver interface for work batches for processed packet saving/loading from disk
    struct IDataArchiver : public virtual IDataHandler {
    protected:
        //! saves a processed data packet locally based on idx and packet name (if available)
        virtual void save(const cv::Mat& oOutput, size_t nIdx) const;
        //! loads a processed data packet based on idx and packet name (if available)
        virtual cv::Mat load(size_t nIdx) const;
    };

    //! data consumer interface for work batches for receiving processed packets
    template<eDatasetEvalList eDatasetEval>
    struct IDataConsumer_ :
            public IDataArchiver,
            public DataCounter_<eNotGroup> {
        //! push a processed data packet for writing and/or evaluation (also registers it as 'done' for internal purposes)
        virtual void push(const cv::Mat& oOutput, size_t nIdx) {
            lvDbgAssert(isProcessing());
            processPacket();
            if(getDatasetInfo()->isSavingOutput())
                save(oOutput,nIdx);
        }
    };

    //! async data consumer interface for work batches for receiving processed packets & async context setup/init
    template<eDatasetEvalList eDatasetEval, ParallelUtils::eParallelAlgoType eImpl>
    struct IAsyncDataConsumer_;

#if HAVE_GLSL

    template<>
    struct IAsyncDataConsumer_<eDatasetEval_BinaryClassifier,ParallelUtils::eGLSL> :
            public IDataArchiver,
            public DataCounter_<eNotGroup> {
        //! returns the ideal size for the GL context window to use for debug display purposes (queries the algo based on dataset specs, if available)
        virtual cv::Size getIdealGLWindowSize() const;
        //! initializes internal params & calls 'initialize_gl' on algo with expanded args list
        template<typename Talgo, typename... Targs>
        void initialize_gl(const std::shared_ptr<Talgo>& pAlgo, Targs&&... args) {
            m_pAlgo = pAlgo;
            pre_initialize_gl();
            pAlgo->initialize_gl(m_oCurrInput,m_pLoader->getInputROI(m_nCurrIdx),std::forward<Targs>(args)...);
            post_initialize_gl();
        }
        //! calls 'apply_gl' from 'Talgo' interface with expanded args list
        template<typename Talgo, typename... Targs>
        void apply_gl(const std::shared_ptr<Talgo>& pAlgo, size_t nNextIdx, bool bRebindAll, Targs&&... args) {
            m_pAlgo = pAlgo;
            pre_apply_gl(nNextIdx,bRebindAll);
            // @@@@@ allow apply with new roi for each packet? (like init?)
            pAlgo->apply_gl(m_oNextInput,bRebindAll,std::forward<Targs>(args)...);
            post_apply_gl(nNextIdx,bRebindAll);
        }
    protected:
        //! initializes internal async packet fetching indexes
        IAsyncDataConsumer_();
        //! called just before the GL algorithm/evaluator are initialized
        virtual void pre_initialize_gl();
        //! called just after the GL algorithm/evaluator are initialized
        virtual void post_initialize_gl();
        //! called just before the GL algorithm/evaluator process a new packet
        virtual void pre_apply_gl(size_t nNextIdx, bool bRebindAll);
        //! called just after the GL algorithm/evaluator process a new packet
        virtual void post_apply_gl(size_t nNextIdx, bool bRebindAll);
        //! utility function for output/debug mask display (should be overloaded if also evaluating results)
        virtual void getColoredMasks(cv::Mat& oOutput, cv::Mat& oDebug, const cv::Mat& oGT=cv::Mat(), const cv::Mat& oROI=cv::Mat());
        std::shared_ptr<ParallelUtils::IParallelAlgo_GLSL> m_pAlgo;
        std::shared_ptr<GLImageProcEvaluatorAlgo> m_pEvalAlgo;
        std::shared_ptr<IDataLoader> m_pLoader;
        cv::Mat m_oLastInput,m_oCurrInput,m_oNextInput;
        cv::Mat m_oLastGT,m_oCurrGT,m_oNextGT;
        size_t m_nLastIdx,m_nCurrIdx,m_nNextIdx;
        AsyncDataCallbackFunc m_lDataCallback;
    };

#endif //HAVE_GLSL

} //namespace litiv
