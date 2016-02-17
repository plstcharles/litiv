
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

    enum eDatasetTaskList {
        eDatasetTask_ChgDet,
        eDatasetTask_Segm,
        eDatasetTask_Registr,
        eDatasetTask_EdgDet,
        // ...
    };

    enum eDatasetSourceList {
        eDatasetSource_Video,
        eDatasetSource_VideoArray,
        eDatasetSource_Image,
        eDatasetSource_ImageArray,
        // ...
    };

    enum eDatasetList {
        eDataset_CDnet,
        eDataset_Wallflower,
        eDataset_PETS2001D3TC1,
        eDataset_LITIV2012b,
        eDataset_BSDS500,
        // ...
        eDataset_Custom // 'datasets::create' will forward all parameters from Dataset constr
    };

    enum eDatasetEvalList {
        eDatasetEval_BinaryClassifier,
        eDatasetEval_Registr,
        eDatasetEval_Segm,
        eDatasetEval_BoundingBox,
        // ...
        eDatasetEval_None // will only count packets & monitor processing time
    };

    enum eGroupPolicy { // used to toggle group policy template in data handler interfaces
        eGroup,
        eNotGroup,
    };

    enum ePacketPolicy { // used to toggle packet policy template in data handler interfaces
        eImagePacket,
        eNotImagePacket
    };

    struct IDataset;
    struct IDataHandler;
    using IDatasetPtr = std::shared_ptr<IDataset>;
    using IDataHandlerPtr = std::shared_ptr<IDataHandler>;
    using IDataHandlerPtrArray = std::vector<IDataHandlerPtr>;
    using IDataHandlerConstPtr = std::shared_ptr<const IDataHandler>;
    using IDataHandlerConstPtrArray = std::vector<IDataHandlerConstPtr>;
    using IDataHandlerPtrQueue = std::priority_queue<IDataHandlerPtr,IDataHandlerPtrArray,std::function<bool(const IDataHandlerPtr&,const IDataHandlerPtr&)>>;

    struct IDataset : CxxUtils::enable_shared_from_this<IDataset> {
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

    struct IDataHandler : CxxUtils::enable_shared_from_this<IDataHandler> {
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
        virtual eDatasetTaskList getDatasetTask() const = 0;
        virtual eDatasetSourceList getDatasetSource() const = 0;
        virtual eDatasetList getDataset() const = 0;
        virtual eDatasetEvalList getDatasetEval() const = 0;
        virtual void writeEvalReport() const = 0;
        virtual ~IDataHandler() = default;
        virtual void startPrecaching(bool bPrecacheGT, size_t nSuggestedBufferSize=SIZE_MAX) = 0; // starts prefetching data packets
        virtual void stopPrecaching() = 0; // stops prefetching data packets (for work batches, is also called in stopProcessing)
        virtual bool isProcessing() const = 0;
        virtual double getProcessTime() const = 0; // returns the current (or final) duration elapsed between start/stopProcessing calls
        virtual size_t getProcessedPacketsCountPromise() = 0;
        virtual size_t getProcessedPacketsCount() const = 0;

    protected:
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
        virtual std::string getPacketName(size_t nPacketIdx) const;
        virtual IDataHandlerConstPtr getBatch(size_t& nPacketIdx) const = 0; // will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        virtual IDataHandlerPtr getBatch(size_t& nPacketIdx) = 0; // will throw if out of range, and readjust nPacketIdx for returned batch range otherwise
        template<eDatasetEvalList eDatasetEval>
        friend struct IDatasetEvaluator_;
        template<eDatasetTaskList eDatasetTask, eDatasetSourceList eDatasetSource, eDatasetList eDataset, eDatasetEvalList eDatasetEval, ParallelUtils::eParallelAlgoType eEvalImpl>
        friend struct IDataset_;
        virtual void _startProcessing() {}
        virtual void _stopProcessing() {}
        //! data parsing function, dataset-specific
        virtual void parseData() = 0;
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

    struct IIDataLoader : public virtual IDataHandler {
        inline ePacketPolicy getPacketType() const {return m_ePacketType;}
        virtual void startPrecaching(bool bPrecacheGT, size_t nSuggestedBufferSize=SIZE_MAX) override;
        virtual void stopPrecaching() override;
    protected:
        IIDataLoader(ePacketPolicy ePacket); // will automatically apply byte-alignment/scale in redirect if using image packets
        DataPrecacher m_oInputPrecacher,m_oGTPrecacher;
        //! input packet load function, dataset-specific
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) = 0;
        //! gt packet load function, dataset-specific
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) = 0;
    private:
        cv::Mat m_oLatestInputPacket, m_oLatestGTPacket;
        const cv::Mat& _getInputPacket_redirect(size_t nIdx);
        const cv::Mat& _getGTPacket_redirect(size_t nIdx);
        const ePacketPolicy m_ePacketType;
    };

    template<ePacketPolicy ePacket>
    struct IDataLoader_ : public IIDataLoader {
    protected:
        IDataLoader_() : IIDataLoader(ePacket) {}
    };

    template<>
    struct IDataLoader_<eImagePacket> : public IIDataLoader {
        virtual bool isPacketTransposed(size_t nPacketIdx) const = 0;
        virtual const cv::Mat& getPacketROI(size_t nPacketIdx) const = 0;
        virtual const cv::Size& getPacketSize(size_t nPacketIdx) const = 0;
        virtual const cv::Size& getPacketOrigSize(size_t nPacketIdx) const = 0;
        virtual const cv::Size& getPacketMaxSize() const = 0;
        const cv::Mat& getInput(size_t nPacketIdx) {return m_oInputPrecacher.getPacket(nPacketIdx);}
        const cv::Mat& getGT(size_t nPacketIdx) {return m_oGTPrecacher.getPacket(nPacketIdx);}
    protected:
        IDataLoader_() : IIDataLoader(eImagePacket) {}
    };

    template<eDatasetSourceList eDatasetSource>
    struct IDataProducer_;

    template<>
    struct IDataProducer_<eDatasetSource_Video> :
            public IDataLoader_<eImagePacket> {
        //! redirects to getTotPackets()
        inline size_t getFrameCount() const {return getTotPackets();}
        virtual double getExpectedLoad() const override;
        virtual void startPrecaching(bool bUsingGT, size_t /*nUnused*/=0) override;
        virtual const cv::Mat& getROI() const {return m_oROI;}
        virtual const cv::Size& getFrameSize() const {return m_oSize;}
        virtual const cv::Size& getFrameOrigSize() const {return m_oOrigSize;}

    protected:
        IDataProducer_();
        virtual size_t getTotPackets() const override;
        virtual bool isPacketTransposed(size_t /*nPacketIdx*/) const override final {return m_bTransposeFrames;}
        virtual const cv::Mat& getPacketROI(size_t /*nPacketIdx*/) const override final {return getROI();}
        virtual const cv::Size& getPacketSize(size_t /*nPacketIdx*/) const override final {return getFrameSize();}
        virtual const cv::Size& getPacketOrigSize(size_t /*nPacketIdx*/) const override final {return getFrameOrigSize();}
        virtual const cv::Size& getPacketMaxSize() const override final {return getFrameSize();}
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) override;
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) override;
        virtual void parseData() override;
        size_t m_nFrameCount;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::string> m_vsInputFramePaths;
        std::vector<std::string> m_vsGTFramePaths;
        cv::VideoCapture m_voVideoReader;
        size_t m_nNextExpectedVideoReaderFrameIdx;
        bool m_bTransposeFrames;
        cv::Mat m_oROI;
        cv::Size m_oOrigSize,m_oSize;
    };

    template<>
    struct IDataProducer_<eDatasetSource_Image> :
            public IDataLoader_<eImagePacket> {
        //! redirects to getTotPackets()
        inline size_t getImageCount() const {return getTotPackets();}
        virtual double getExpectedLoad() const override;
        virtual void startPrecaching(bool bUsingGT, size_t /*nUnused*/=0) override;
        virtual bool isConstantSize() const {return m_bIsConstantSize;}
        virtual bool isPacketTransposed(size_t nPacketIdx) const override;
        virtual const cv::Mat& getPacketROI(size_t nPacketIdx) const override;
        virtual const cv::Size& getPacketSize(size_t nPacketIdx) const override;
        virtual const cv::Size& getPacketOrigSize(size_t nPacketIdx) const override;
        virtual const cv::Size& getPacketMaxSize() const override;
        virtual std::string getPacketName(size_t nPacketIdx) const override;

    protected:
        IDataProducer_();
        virtual size_t getTotPackets() const override;
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) override;
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) override;
        virtual void parseData() override;
        size_t m_nImageCount;
        std::unordered_map<size_t,size_t> m_mGTIndexLUT;
        std::vector<std::string> m_vsInputImagePaths;
        std::vector<std::string> m_vsGTImagePaths;
        std::vector<cv::Size> m_voImageSizes;
        std::vector<cv::Size> m_voImageOrigSizes;
        std::vector<bool> m_vbImageTransposed;
        bool m_bIsConstantSize;
        cv::Size m_oMaxSize;
        const cv::Mat m_oDefaultEmptyROI;
    };

    template<eDatasetSourceList eDatasetSource, eDatasetList eDataset>
    struct DataProducer_ : public IDataProducer_<eDatasetSource> {};

    template<eGroupPolicy ePolicy>
    struct DataCounter_;

    template<>
    struct DataCounter_<eNotGroup> : public virtual IDataHandler {
    protected:
        DataCounter_() : m_nProcessedPackets(0) {}
        inline void processPacket() {++m_nProcessedPackets;}
        inline void setProcessedPacketsPromise() {m_nProcessedPacketsPromise.set_value(m_nProcessedPackets);}
        virtual size_t getProcessedPacketsCountPromise() override final;
        virtual size_t getProcessedPacketsCount() const override final;
    private:
        size_t m_nProcessedPackets;
        std::promise<size_t> m_nProcessedPacketsPromise;
    };

    template<>
    struct DataCounter_<eGroup> : public virtual IDataHandler {
        virtual size_t getProcessedPacketsCountPromise() override final;
        virtual size_t getProcessedPacketsCount() const override final;
    };

    struct IDataArchiver : public virtual IDataHandler {
    protected:
        virtual void save(const cv::Mat& oClassif, size_t nIdx) const;
        virtual cv::Mat load(size_t nIdx) const;
    };

    template<eDatasetEvalList eDatasetEval>
    struct IDataConsumer_ :
            public IDataArchiver,
            public DataCounter_<eNotGroup> {
        virtual void push(const cv::Mat& oClassif, size_t nIdx) {
            processPacket();
            if(getDatasetInfo()->isSavingOutput())
                save(oClassif,nIdx);
        }
    };

    using DataCallbackFunc = std::function<void(const cv::Mat& /*oInput*/,const cv::Mat& /*oDebug*/,const cv::Mat& /*oOutput*/,const cv::Mat& /*oGT*/,const cv::Mat& /*oROI*/,size_t /*nIdx*/)>;
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
            pAlgo->initialize_gl(m_oCurrInput,m_pLoader->getPacketROI(m_nCurrIdx),std::forward<Targs>(args)...);
            post_initialize_gl();
        }
        //! casts the algo to 'Talgo' type, and calls 'apply_gl' with expanded args list
        template<typename Talgo, typename... Targs>
        void apply_gl(const std::shared_ptr<Talgo>& pAlgo, size_t nNextIdx, bool bRebindAll, Targs&&... args) {
            m_pAlgo = pAlgo;
            pre_apply_gl(nNextIdx,bRebindAll);
            // @@@@@ allow apply with new roi for each packet? (like init?)
            pAlgo->apply_gl(m_oNextInput,bRebindAll,std::forward<Targs>(args)...);
            post_apply_gl(nNextIdx,bRebindAll);
        }
    protected:
        IAsyncDataConsumer_();
        virtual void pre_initialize_gl();
        virtual void post_initialize_gl();
        virtual void pre_apply_gl(size_t nNextIdx, bool bRebindAll);
        virtual void post_apply_gl(size_t nNextIdx, bool bRebindAll);
        virtual void getColoredMasks(cv::Mat& oOutput, cv::Mat& oDebug, const cv::Mat& oGT=cv::Mat(), const cv::Mat& oROI=cv::Mat());
        std::shared_ptr<ParallelUtils::IParallelAlgo_GLSL> m_pAlgo;
        std::shared_ptr<GLImageProcEvaluatorAlgo> m_pEvalAlgo;
        std::shared_ptr<IDataLoader_<eImagePacket>> m_pLoader;
        cv::Mat m_oLastInput,m_oCurrInput,m_oNextInput;
        cv::Mat m_oLastGT,m_oCurrGT,m_oNextGT;
        size_t m_nLastIdx,m_nCurrIdx,m_nNextIdx;
        DataCallbackFunc m_lDataCallback;
    };

#endif //HAVE_GLSL

} //namespace litiv
