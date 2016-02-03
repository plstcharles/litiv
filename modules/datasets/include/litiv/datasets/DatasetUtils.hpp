
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

#define DATASETUTILS_VIDEOSEGM_OUTOFSCOPE_VAL  uchar(85)
#define DATASETUTILS_VIDEOSEGM_UNKNOWN_VAL     uchar(170)
#define DATASETUTILS_VIDEOSEGM_SHADOW_VAL      uchar(50)

#include "litiv/utils/ParallelUtils.hpp"
#include "litiv/utils/OpenCVUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace litiv {

    enum eDatasetTypeList {
        eDatasetType_VideoSegm,
        eDatasetType_VideoRegistr,
        eDatasetType_ImageSegm,
        eDatasetType_ImageEdgDet,
        // ...
    };

    enum eDatasetList {

        //// VIDEO SEGMENTATION
        eDataset_VideoSegm_CDnet,
        eDataset_VideoSegm_Wallflower,
        eDataset_VideoSegm_PETS2001D3TC1,
        //eDataset_VideoSegm_...
        eDataset_VideoSegm_Custom,

        //// VIDEO REGISTRATION
        eDataset_VideoRegistr_LITIV2012b,
        //eDataset_VideoRegistr_...
        eDataset_VideoRegistr_Custom,

        //// IMAGE SEGMENTATION
        eDataset_ImageSegm_BSDS500,
        //eDataset_ImageSegm_...
        eDataset_ImageSegm_Custom,

        //// IMAGE EDGE DETECTION
        eDataset_ImageEdgDet_BSDS500,
        //eDataset_ImageEdgDet_...
        eDataset_ImageEdgDet_Custom

    };

    enum eGroupPolicy {
        eGroup,
        eNotGroup,
    };

    constexpr eDatasetList getCustomDatasetEnum(eDatasetTypeList eDatasetType) {
        return (eDatasetType==eDatasetType_VideoSegm)?eDataset_VideoSegm_Custom:
               (eDatasetType==eDatasetType_VideoRegistr)?eDataset_VideoRegistr_Custom:
               (eDatasetType==eDatasetType_ImageSegm)?eDataset_ImageSegm_Custom:
               (eDatasetType==eDatasetType_ImageEdgDet)?eDataset_ImageEdgDet_Custom:
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

    template<eDatasetTypeList eDatasetType>
    struct IDataReader_;
    template<eDatasetTypeList eDatasetType, eGroupPolicy ePolicy>
    struct IDataProducer_;
    template<eGroupPolicy ePolicy>
    struct IDataCounter_;
    template<eDatasetTypeList eDatasetType>
    struct IDataRecorder_;
    template<eDatasetTypeList eDatasetType, eGroupPolicy ePolicy>
    struct IDataConsumer_;

    template<eDatasetTypeList eDatasetType, ParallelUtils::eParallelAlgoType eImpl>
    struct AsyncEvaluationWrapper_; // standalone algo evaluation wrapper

    struct IDataset : std::enable_shared_from_this<IDataset> {
        virtual const std::string& getName() const = 0;
        virtual const std::string& getDatasetPath() const = 0;
        virtual const std::string& getOutputPath() const = 0;
        virtual const std::string& getOutputNamePrefix() const = 0;
        virtual const std::string& getOutputNameSuffix() const = 0;
        virtual const std::vector<std::string>& getWorkBatchDirs() const = 0;
        virtual const std::vector<std::string>& getSkippedDirTokens() const = 0;
        virtual const std::vector<std::string>& getGrayscaleDirTokens() const = 0;
        virtual size_t getOutputIdxOffset() const = 0;
        virtual double getScaleFactor() const = 0;
        virtual bool isSavingOutput() const = 0;
        virtual bool isUsingEvaluator() const = 0;
        virtual bool is4ByteAligned() const = 0;
        virtual ~IDataset() = default;

        virtual size_t getTotPackets() const = 0;
        virtual double getProcessTime() const = 0;
        virtual double getExpectedLoad() const = 0;
        virtual size_t getProcessedPacketsCountPromise() = 0;
        virtual size_t getProcessedPacketsCount() const = 0;

        virtual void parseDataset() = 0;
        virtual void writeEvalReport() const = 0;
        virtual IDataHandlerPtrArray getBatches() const = 0;
        virtual IDataHandlerPtrQueue getSortedBatches() const = 0;
    };

    struct IDataHandler : std::enable_shared_from_this<IDataHandler> {
        virtual const std::string& getName() const = 0;
        virtual const std::string& getDataPath() const = 0;
        virtual const std::string& getOutputPath() const = 0;
        virtual const std::string& getRelativePath() const = 0;
        virtual double getExpectedLoad() const = 0;
        virtual size_t getTotPackets() const = 0;
        virtual bool isGrayscale() const = 0;
        virtual bool isBare() const = 0;
        virtual bool isGroup() const = 0;
        virtual IDataHandlerPtrArray getBatches() const = 0; // @@@@@ rename batches at dataset level to something else?
        virtual IDatasetPtr getDatasetInfo() const = 0;
        virtual eDatasetTypeList getDatasetType() const = 0;
        virtual eDatasetList getDataset() const = 0;
        virtual std::string writeInlineEvalReport(size_t nIndentSize, size_t nCellSize=12) const = 0;
        virtual void writeEvalReport() const = 0;
        virtual void parseData() = 0;
        virtual ~IDataHandler() = default;

        virtual void startPrecaching(bool bPrecacheGT, size_t nSuggestedBufferSize=SIZE_MAX) = 0; // starts prefetching data packets
        virtual void stopPrecaching() = 0; // stops prefetching data packets (for work batches, is also called in stopProcessing)
        virtual double getProcessTime() const = 0; // returns the current (or final) duration elapsed between start/stopProcessing calls
        virtual size_t getProcessedPacketsCountPromise() = 0;
        virtual size_t getProcessedPacketsCount() const = 0;

        template<typename Tp>
        static typename std::enable_if<std::is_base_of<IDataHandler,Tp>::value,bool>::type compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return PlatformUtils::compare_lowercase(i->getName(),j->getName());
        }
        template<typename Tp>
        static typename std::enable_if<std::is_base_of<IDataHandler,Tp>::value,bool>::type compare_load(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {
            return i->getExpectedLoad()<j->getExpectedLoad();
        }
        static bool compare(const IDataHandler* i, const IDataHandler* j);
        static bool compare_load(const IDataHandler* i, const IDataHandler* j);
        static bool compare(const IDataHandler& i, const IDataHandler& j);
        static bool compare_load(const IDataHandler& i, const IDataHandler& j);
    protected:
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const = 0; // will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) = 0; // will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual void _startProcessing() {}
        virtual void _stopProcessing() {}
    };

    struct DataPrecacher {
        // @@@@ rewrite to allow streaming with no limit? (might just need to modify init and set tot=inf)
        // @@@@ current impl expects all packets to be the same size
        DataPrecacher(std::function<const cv::Mat&(size_t)> lDataLoaderCallback);
        ~DataPrecacher();
        const cv::Mat& getPacket(size_t nIdx);
        bool startPrecaching(size_t nTotPacketCount, size_t nSuggestedBufferSize);
        void stopPrecaching();
    protected:
        void precache();
        const cv::Mat& getPacket_internal(size_t nIdx);
        const std::function<const cv::Mat&(size_t)> m_lCallback;
        std::thread m_hPrecacher;
        std::mutex m_oSyncMutex;
        std::condition_variable m_oReqCondVar;
        std::condition_variable m_oSyncCondVar;
        bool m_bIsPrecaching;
        size_t m_nBufferSize;
        size_t m_nPacketCount;
        std::deque<cv::Mat> m_qoCache;
        std::vector<uchar> m_vcBuffer;
        size_t m_nFirstBufferIdx;
        size_t m_nNextBufferIdx;
        size_t m_nNextExpectedReqIdx;
        size_t m_nNextPrecacheIdx;
        size_t m_nReqIdx,m_nLastReqIdx;
        cv::Mat m_oReqPacket,m_oLastReqPacket;
    private:
        DataPrecacher& operator=(const DataPrecacher&) = delete;
        DataPrecacher(const DataPrecacher&) = delete;
    };

    template<eGroupPolicy ePolicy>
    struct IDataLoader_ : public virtual IDataHandler {};

    template<>
    struct IDataLoader_<eNotGroup> : public virtual IDataHandler {
        virtual void startPrecaching(bool bPrecacheGT, size_t nSuggestedBufferSize=SIZE_MAX) override;
        virtual void stopPrecaching() override;
    protected:
        IDataLoader_();
        DataPrecacher m_oInputPrecacher,m_oGTPrecacher;
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) = 0;
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) = 0;
    private:
        cv::Mat m_oLatestInputPacket, m_oLatestGTPacket;
        const cv::Mat& _getInputPacket_redirect(size_t nIdx);
        const cv::Mat& _getGTPacket_redirect(size_t nIdx);
    };

    template<>
    struct IDataReader_<eDatasetType_VideoSegm> : public virtual IDataHandler {
        inline size_t getFrameCount() const {return getTotPackets();}
        virtual const cv::Mat& getInputFrame(size_t nFrameIdx) = 0;
        virtual const cv::Mat& getGTFrame(size_t nFrameIdx) = 0;
    };

    template<>
    struct IDataProducer_<eDatasetType_VideoSegm,eGroup> :
            public IDataLoader_<eGroup>,
            public IDataReader_<eDatasetType_VideoSegm> {
        virtual const cv::Mat& getInputFrame(size_t nFrameIdx) override final;
        virtual const cv::Mat& getGTFrame(size_t nFrameIdx) override final;
    };

    template<>
    struct IDataProducer_<eDatasetType_VideoSegm,eNotGroup> :
            public IDataLoader_<eNotGroup>,
            public IDataReader_<eDatasetType_VideoSegm> {
        virtual double getExpectedLoad() const override;
        virtual size_t getTotPackets() const override;
        virtual void startPrecaching(bool bUsingGT, size_t /*nUnused*/=0) override;
        inline const cv::Size& getFrameSize() const {return m_oSize;}
        inline const cv::Mat& getROI() const {return m_oROI;}
        virtual const cv::Mat& getInputFrame(size_t nFrameIdx) override final;
        virtual const cv::Mat& getGTFrame(size_t nFrameIdx) override final;
        virtual void parseData() override;
    protected:
        IDataProducer_() : m_nFrameCount(0),m_nNextExpectedVideoReaderFrameIdx(size_t(-1)) {}
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) override;
        virtual cv::Mat _getGTPacket_impl(size_t) override;
        size_t m_nFrameCount;
        std::vector<std::string> m_vsInputFramePaths;
        std::vector<std::string> m_vsGTFramePaths;
        cv::VideoCapture m_voVideoReader;
        size_t m_nNextExpectedVideoReaderFrameIdx;
        cv::Mat m_oROI;
        cv::Size m_oOrigSize,m_oSize;
        std::unordered_map<size_t,size_t> m_mTestGTIndexes;
    };

    template<>
    struct IDataCounter_<eNotGroup> : public virtual IDataHandler {
    protected:
        IDataCounter_() : m_nProcessedPackets(0) {}
        inline void processPacket() {++m_nProcessedPackets;}
        inline void setProcessedPacketsPromise() {m_nProcessedPacketsPromise.set_value(m_nProcessedPackets);}
        virtual size_t getProcessedPacketsCountPromise() override;
        virtual size_t getProcessedPacketsCount() const override;
    private:
        size_t m_nProcessedPackets;
        std::promise<size_t> m_nProcessedPacketsPromise;
    };

    template<>
    struct IDataCounter_<eGroup> : public virtual IDataHandler {
        virtual size_t getProcessedPacketsCountPromise() override final;
        virtual size_t getProcessedPacketsCount() const override final;
    };

    template<>
    struct IDataRecorder_<eDatasetType_VideoSegm> : public virtual IDataHandler {
        virtual void pushSegmMask(const cv::Mat& oSegm, size_t nIdx) = 0;
    protected:
        virtual cv::Mat readSegmMask(size_t nIdx) const = 0; // used for output reevaluation only
        virtual void writeSegmMask(const cv::Mat& oSegm, size_t nIdx) const = 0;
        friend struct IDataConsumer_<eDatasetType_VideoSegm,eGroup>;
    };

    template<>
    struct IDataConsumer_<eDatasetType_VideoSegm,eGroup> :
            public IDataCounter_<eGroup>,
            public IDataRecorder_<eDatasetType_VideoSegm> {
        virtual void pushSegmMask(const cv::Mat& oSegm, size_t nIdx) override final;
    protected:
        virtual cv::Mat readSegmMask(size_t nIdx) const override final;
        virtual void writeSegmMask(const cv::Mat& oSegm, size_t nIdx) const override final;
    };

    template<>
    struct IDataConsumer_<eDatasetType_VideoSegm,eNotGroup> :
            public IDataCounter_<eNotGroup>,
            public IDataRecorder_<eDatasetType_VideoSegm> {
        virtual void pushSegmMask(const cv::Mat& oSegm, size_t nIdx) override;
    protected:
        virtual cv::Mat readSegmMask(size_t nIdx) const override;
        virtual void writeSegmMask(const cv::Mat& oSegm, size_t nIdx) const override;
        // cannot use partial template specialization here... for some reason.
        friend struct AsyncEvaluationWrapper_<eDatasetType_VideoSegm,ParallelUtils::eGLSL>;
        //friend struct AsyncEvaluationWrapper_<eDatasetType_VideoSegm,ParallelUtils::eCUDA>;
        //friend struct AsyncEvaluationWrapper_<eDatasetType_VideoSegm,ParallelUtils::eOpenCL>;
    };

#if HAVE_GLSL

    template<>
    struct AsyncEvaluationWrapper_<eDatasetType_VideoSegm,ParallelUtils::eGLSL> {
        //! default constructor (grabs a copy of pAlgo and readies internal fetching; you still need to call 'initialize_gl' however)
        AsyncEvaluationWrapper_(const std::shared_ptr<ParallelUtils::IParallelAlgo_GLSL>& pAlgo, const IDataHandlerPtr& pSequence);
        //! returns the ideal size for the GL context window to use for debug display purposes (queries the algo based on dataset specs)
        virtual cv::Size getIdealGLWindowSize();
        //! always called by 'initialize_gl' before initializing algo; override this if implementing async evaluator
        virtual void pre_initialize_gl();
        //! casts the algo to 'Talgo' type, and calls 'initialize_gl' with expanded args list
        template<typename Talgo, typename... Targs>
        void initialize_gl(Targs&&... args) {
            Talgo& oAlgo = dynamic_cast<Talgo&>(*m_pAlgo);
            pre_initialize_gl();
            oAlgo.initialize_gl(m_oCurrInput,m_pProducer->getROI(),std::forward<Targs>(args)...);
        }
        //! always called by 'apply_gl' after processing via algo; override this if implementing async evaluator
        virtual void post_apply_gl();
        //! casts the algo to 'Talgo' type, and calls 'apply_gl' with expanded args list
        template<typename Talgo, typename... Targs>
        void apply_gl(size_t nNextIdx, Targs&&... args) {
            Talgo& oAlgo = dynamic_cast<Talgo&>(*m_pAlgo);
            if(nNextIdx!=m_nNextIdx)
                m_oNextInput = m_pProducer->getInputFrame(nNextIdx);
            oAlgo.apply_gl(m_oNextInput,std::forward<Targs>(args)...);
            m_nLastIdx = m_nCurrIdx;
            m_nCurrIdx = nNextIdx;
            m_nNextIdx = nNextIdx+1;
            post_apply_gl();
        }
    protected:
        // add toggle for double-eval in here somewhere, or in actual evaluator interface?
        std::shared_ptr<ParallelUtils::IParallelAlgo_GLSL> m_pAlgo;
        std::shared_ptr<cv::DisplayHelper> m_pDisplayHelper;
        std::shared_ptr<IDataProducer_<eDatasetType_VideoSegm,eNotGroup>> m_pProducer;
        std::shared_ptr<IDataConsumer_<eDatasetType_VideoSegm,eNotGroup>> m_pConsumer;
        const bool m_bPreserveInputs;
        cv::Mat m_oLastInput,m_oCurrInput,m_oNextInput;
        size_t m_nLastIdx,m_nCurrIdx,m_nNextIdx;
        size_t m_nFrameCount;
    };

    using AsyncEvaluationWrapper_VideoSegm_GLSL = AsyncEvaluationWrapper_<eDatasetType_VideoSegm,ParallelUtils::eGLSL>;
#endif //HAVE_GLSL

#if 0
    namespace Video {

        namespace Segm {

            enum eDatasetList {
                eDataset_CDnet2012,
                eDataset_CDnet2014,
                eDataset_Wallflower,
                eDataset_PETS2001_D3TC1,
                // ...
                eDataset_Custom
            };

            struct DatasetInfo : public DatasetInfoBase {
                DatasetInfo();
                DatasetInfo(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                            const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                            const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                            bool bForce4ByteDataAlign, double dScaleFactor, eDatasetList eDatasetID, size_t nResultIdxOffset);
                virtual void WriteEvalResults(const std::vector<std::shared_ptr<WorkBatchGroup>>& vpGroups) const;
                virtual eDatasetTypeList GetType() const {return eDatasetType_Video_Segm;}
                eDatasetList m_eDatasetID;
                size_t m_nResultIdxOffset;
            };

            std::shared_ptr<DatasetInfo> GetDatasetInfo(eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign);

            class Sequence : public WorkBatch {
            public:
                Sequence(const std::string& sSeqName, const DatasetInfo& oDataset, const std::string& sRelativePath=std::string("./"));
                virtual size_t GetTotalImageCount() const {return m_nTotFrameCount;}
                virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
                virtual void WriteResult(size_t nIdx, const cv::Mat& oResult);
                virtual bool StartPrecaching(bool bUsingGT, size_t nUnused=0);
                cv::Size GetImageSize() const {return m_oSize;}
                const cv::Mat& GetROI() const {return m_oROI;}
                const eDatasetList m_eDatasetID;
                const size_t m_nResultIdxOffset;
            protected:
                virtual cv::Mat GetInputFromIndex_external(size_t nFrameIdx);
                virtual cv::Mat GetGTFromIndex_external(size_t nFrameIdx);
            private:
                double m_dExpectedLoad;
                size_t m_nTotFrameCount;
                std::vector<std::string> m_vsInputFramePaths;
                std::vector<std::string> m_vsGTFramePaths;
                cv::VideoCapture m_voVideoReader;
                size_t m_nNextExpectedVideoReaderFrameIdx;
                cv::Mat m_oROI;
                cv::Size m_oOrigSize,m_oSize;
                double m_dScaleFactor;
                std::unordered_map<size_t,size_t> m_mTestGTIndexes;
                Sequence& operator=(const Sequence&) = delete;
                Sequence(const Sequence&) = delete;
            };

        } //namespace Segm

        namespace Registr {

            enum eDatasetList {
                eDataset_LITIV2012b,
                // ...
                eDataset_Custom
            };

            struct DatasetInfo : public DatasetInfoBase {
                DatasetInfo();
                DatasetInfo(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                            const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                            const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                            bool bForce4ByteDataAlign, double dScaleFactor, eDatasetList eDatasetID, size_t nResultIdxOffset);
                virtual void WriteEvalResults(const std::vector<std::shared_ptr<WorkBatchGroup>>& vpGroups) const;
                virtual eDatasetTypeList GetType() const {return eDatasetType_Video_Registr;}
                eDatasetList m_eDatasetID;
                size_t m_nResultIdxOffset;
            };

            std::shared_ptr<DatasetInfo> GetDatasetInfo(eDatasetList eDatasetID, const std::string& sDatasetRootDirPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign);

            class Sequence : public WorkBatch {
            public:
                Sequence(const std::string& sSeqName, const DatasetInfo& oDataset, const std::string& sRelativePath=std::string("./"));
                virtual size_t GetTotalImageCount() const {return m_nTotFrameCount;}
                virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
                virtual void WriteResult(size_t nIdx, const cv::Mat& oResult);
                virtual bool StartPrecaching(bool bUsingGT, size_t nUnused=0);
                cv::Size GetImageSize() const {return m_oSize;}
                const cv::Mat& GetROI() const {return m_oROI;}
                const eDatasetList m_eDatasetID;
                const size_t m_nResultIdxOffset;
            protected:
                virtual cv::Mat GetInputFromIndex_external(size_t nFrameIdx);
                virtual cv::Mat GetGTFromIndex_external(size_t nFrameIdx);
            private:
                double m_dExpectedLoad;
                size_t m_nTotFrameCount;
                std::vector<std::string> m_vsInputFramePaths;
                std::vector<std::string> m_vsGTFramePaths;
                cv::VideoCapture m_voVideoReader;
                size_t m_nNextExpectedVideoReaderFrameIdx;
                cv::Mat m_oROI;
                cv::Size m_oOrigSize,m_oSize;
                double m_dScaleFactor;
                std::unordered_map<size_t,size_t> m_mTestGTIndexes;
                Sequence& operator=(const Sequence&) = delete;
                Sequence(const Sequence&) = delete;
            };
        } //namespace Registr

    } //namespace Video

    namespace Image {

        namespace Segm {

            enum eDatasetList {
                eDataset_BSDS500_segm_train,
                eDataset_BSDS500_segm_train_valid,
                eDataset_BSDS500_segm_train_valid_test,
                eDataset_BSDS500_edge_train,
                eDataset_BSDS500_edge_train_valid,
                eDataset_BSDS500_edge_train_valid_test,
                // ...
                eDataset_Custom
            };

            struct DatasetInfo : public DatasetInfoBase {
                DatasetInfo();
                DatasetInfo(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                            const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                            const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                            bool bForce4ByteDataAlign, double dScaleFactor, eDatasetList eDatasetID);
                virtual void WriteEvalResults(const std::vector<std::shared_ptr<WorkBatchGroup>>& vpGroups) const;
                virtual eDatasetTypeList GetType() const {return eDatasetType_Image_Segm;}
                eDatasetList m_eDatasetID;
            };

            std::shared_ptr<DatasetInfo> GetDatasetInfo(eDatasetList eDatasetID, const std::string& sDatasetRootPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign);

            class Set : public WorkBatch {
            public:
                Set(const std::string& sSetName, const DatasetInfo& oDataset, const std::string& sRelativePath=std::string("./"));
                virtual size_t GetTotalImageCount() const {return m_nTotImageCount;}
                virtual double GetExpectedLoad() const {return m_dExpectedLoad;}
                virtual cv::Mat ReadResult(size_t nIdx);
                virtual void WriteResult(size_t nIdx, const cv::Mat& oResult);
                virtual bool StartPrecaching(bool bUsingGT, size_t nUnused=0);
                bool IsConstantImageSize() const {return m_bIsConstantSize;}
                cv::Size GetMaxImageSize() const {return m_oMaxSize;}
                const eDatasetList m_eDatasetID;
            protected:
                virtual cv::Mat GetInputFromIndex_external(size_t nImageIdx);
                virtual cv::Mat GetGTFromIndex_external(size_t nImageIdx);
            private:
                double m_dExpectedLoad;
                size_t m_nTotImageCount;
                std::vector<std::string> m_vsInputImagePaths;
                std::vector<std::string> m_vsGTImagePaths;
                std::vector<std::string> m_vsOrigImageNames;
                std::vector<cv::Size> m_voOrigImageSizes;
                cv::Size m_oMaxSize;
                bool m_bIsConstantSize;
                Set& operator=(const Set&) = delete;
                Set(const Set&) = delete;
            };

        } //namespace Segm

    } //namespace Image

#endif

} //namespace litiv
