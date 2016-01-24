
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

#define DATASETUTILS_HARDCODE_FRAME_INDEX      0
#define DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL  uchar(85)

#include "litiv/utils/ParallelUtils.hpp"
#include "litiv/utils/PlatformUtils.hpp"
#include "litiv/utils/DatasetEvalUtils.hpp"

namespace DatasetUtils {

    enum eDatasetTypeList {
        eDatasetType_VideoSegm,
        eDatasetType_VideoRegistr,
        eDatasetType_ImageSegm,
        eDatasetType_ImageEdgDet,
        // ...
    };

    enum eDatasetList {

        //// VIDEO SEGMENTATION
        eDataset_VideoSegm_CDnet2012,
        eDataset_VideoSegm_CDnet2014,
        eDataset_VideoSegm_Wallflower,
        eDataset_VideoSegm_PETS2001D3TC1,
        //eDataset_VideoSegm_...
        eDataset_VideoSegm_Custom,

        //// VIDEO REGISTRATION
        eDataset_VideoReg_LITIV2012b,
        //eDataset_VideoReg_...
        eDataset_VideoReg_Custom,

        //// IMAGE SEGMENTATION
        eDataset_ImageSegm_BSDS500,
        //eDataset_ImageSegm_...
        eDataset_ImageSegm_Custom,

        //// IMAGE EDGE DETECTION
        eDataset_ImageEdgDet_BSDS500,
        //eDataset_ImageEdgDet_...
        eDataset_ImageEdgDet_Custom

    };

    struct IDataHandler;
    struct IDataEvaluator;
    template<eDatasetTypeList eDatasetType>
    struct IDataEvaluator_;

    using IDataHandlerPtr = std::shared_ptr<IDataHandler>;
    using IDataHandlerPtrArray = std::vector<IDataHandlerPtr>;
    using IDataHandlerPtrQueue = std::priority_queue<IDataHandlerPtr,IDataHandlerPtrArray,std::function<bool(const IDataHandlerPtr&,const IDataHandlerPtr&)>>;

    struct IDataset {
        virtual const std::string& getDatasetName() const = 0;
        virtual const std::string& getDatasetRootPath() const = 0;
        virtual const std::string& getResultsRootPath() const = 0;
        virtual const std::string& getResultsNamePrefix() const = 0;
        virtual const std::string& getResultsNameSuffix() const = 0;
        virtual const std::vector<std::string>& getWorkBatchPaths() const = 0;
        virtual const std::vector<std::string>& getSkippedNameTokens() const = 0;
        virtual const std::vector<std::string>& getGrayscaleNameTokens() const = 0;
        virtual size_t getOutputIdxOffset() const = 0;
        virtual double getScaleFactor() const = 0;
        virtual bool is4ByteAligned() const = 0;
        virtual bool hasEvaluator() const {return false;}
        virtual ~IDataset() = default;

        virtual void parseDataset() = 0;
        virtual void writeEvalReport() const = 0;
        virtual IDataHandlerPtrArray getBatches() const = 0;
        virtual IDataHandlerPtrQueue getSortedBatches() const = 0;
    };

    struct IDataHandler {
        virtual const std::string& getName() const = 0;
        virtual const std::string& getPath() const = 0;
        virtual const std::string& getResultsPath() const = 0;
        virtual const std::string& getRelativePath() const = 0;
        virtual double getExpectedLoad() const = 0;
        virtual size_t getTotPackets() const = 0;
        virtual bool hasEvaluator() const {return false;}
        virtual bool isGrayscale() const = 0;
        virtual bool isBare() const = 0;
        virtual bool isGroup() const = 0;
        virtual IDataHandlerPtrArray getBatches() const = 0;
        virtual const IDataset& getDatasetInfo() const = 0;
        virtual eDatasetTypeList getDatasetType() const = 0;
        virtual eDatasetList getDataset() const = 0;
        virtual std::string writeInlineEvalReport(size_t nIndentSize, size_t nCellSize=12) const = 0;
        virtual void writeEvalReport() const = 0;
        virtual ~IDataHandler() = default;

        virtual void startProcessing() = 0; // used to start batch timer & init other time-critical components via _startProcessing
        virtual void stopProcessing() = 0; // used to stop batch timer & release other time-critical components via _stopProcessing
        virtual double getProcessTime() const = 0; // returns the current (or final) duration elapsed between start/stopProcessing calls

        template<typename Tp> static typename std::enable_if<std::is_base_of<IDataHandler,Tp>::value,bool>::type compare(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {return PlatformUtils::compare_lowercase(i->getName(),j->getName());}
        template<typename Tp> static typename std::enable_if<std::is_base_of<IDataHandler,Tp>::value,bool>::type compare_load(const std::shared_ptr<Tp>& i, const std::shared_ptr<Tp>& j) {return i->getExpectedLoad()<j->getExpectedLoad();}
        static bool compare(const IDataHandler* i, const IDataHandler* j) {return PlatformUtils::compare_lowercase(i->getName(),j->getName());}
        static bool compare_load(const IDataHandler* i, const IDataHandler* j) {return i->getExpectedLoad()<j->getExpectedLoad();}
        static bool compare(const IDataHandler& i, const IDataHandler& j) {return PlatformUtils::compare_lowercase(i.getName(),j.getName());}
        static bool compare_load(const IDataHandler& i, const IDataHandler& j) {return i.getExpectedLoad()<j.getExpectedLoad();}
    protected:
        virtual void _startProcessing() {}
        virtual void _stopProcessing() {}
    };

    void writeOnImage(cv::Mat& oImg, const std::string& sText, const cv::Scalar& vColor, bool bBottom=false);
    cv::Mat getDisplayImage(const cv::Mat& oInputImg, const cv::Mat& oDebugImg, const cv::Mat& oSegmMask, size_t nIdx, cv::Point oDbgPt=cv::Point(-1,-1), cv::Size oRefSize=cv::Size(-1,-1));
    void validateKeyPoints(const cv::Mat& oROI, std::vector<cv::KeyPoint>& voKPs);

    struct DataPrecacher {
        // @@@@ rewrite to allow streaming with no limit? (might just need to modify init and set tot=inf)
        // @@@@ current impl expects all packets to be the same size
        DataPrecacher(const std::function<const cv::Mat&(size_t)>& lCallback);
        virtual ~DataPrecacher();
        const cv::Mat& getPacket(size_t nIdx);
        bool startPrecaching(size_t nTotPacketCount, size_t nSuggestedBufferSize);
        void stopPrecaching();
    private:
        void precache();
        const cv::Mat& getPacket_internal(size_t nIdx);
        const std::function<const cv::Mat&(size_t)>& m_lCallback;
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

    struct DataHandler : public IDataHandler {
        DataHandler(const std::string& sBatchName, const IDataset& oDataset, const std::string& sRelativePath);
    protected:
        const std::string m_sBatchName;
        const std::string m_sRelativePath;
        const std::string m_sDatasetPath;
        const std::string m_sResultsPath;
        const bool m_bForcingGrayscale;
        const IDataset& m_oDataset;
        virtual const std::string& getName() const override {return m_sBatchName;}
        virtual const std::string& getPath() const override {return m_sDatasetPath;}
        virtual const std::string& getResultsPath() const override {return m_sResultsPath;}
        virtual const std::string& getRelativePath() const override {return m_sRelativePath;}
        virtual bool isGrayscale() const override {return m_bForcingGrayscale;}
        virtual const IDataset& getDatasetInfo() const override {return m_oDataset;}
    };

    struct IDataProducer : public IDataHandler { // generalized producer (exposes common interface for all dataset types)
        IDataProducer();
        virtual bool startPrecaching(bool bUsingGT, size_t nTotPacketCount, size_t nSuggestedBufferSize=SIZE_MAX);
        virtual void stopPrecaching();
    protected:
        DataPrecacher m_oInputPrecacher;
        DataPrecacher m_oGTPrecacher;
        virtual cv::Mat _getInputPacket_impl(size_t nIdx) = 0;
        virtual cv::Mat _getGTPacket_impl(size_t nIdx) = 0;
    private:
        cv::Mat m_oLatestInputPacket, m_oLatestGTPacket;
        const cv::Mat& _getInputPacket_redirect(size_t nIdx);
        const cv::Mat& _getGTPacket_redirect(size_t nIdx);
    };

    template<eDatasetTypeList eDatasetType>
    struct IDataProducer_; // specialized producer (exposes interface for single dataset types)

    template<> // all method impl can go in CPP as template possibilities are tightly defined
    struct IDataProducer_<eDatasetType_VideoSegm> : public IDataProducer {
        IDataProducer_() : m_nFrameCount(0),m_nNextExpectedVideoReaderFrameIdx(size_t(-1)) {}

        virtual double getExpectedLoad() const override {return m_oROI.empty()?0.0:(double)cv::countNonZero(m_oROI)*m_nFrameCount*(int(!isGrayscale())+1);}
        virtual size_t getTotPackets() const override {return m_nFrameCount;}
        virtual bool startPrecaching(bool bUsingGT, size_t /*nUnused=0*/, size_t /*nUnused=0*/) override {
            return IDataProducer::startPrecaching(bUsingGT,m_nFrameCount,m_oSize.area()*(m_nFrameCount+1)*(isGrayscale()?1:getDatasetInfo().is4ByteAligned()?4:3));
        }

        virtual cv::Size getFrameSize() const {return m_oSize;}
        virtual const cv::Mat& getROI() const {return m_oROI;}

        virtual const cv::Mat& getInputFrame(size_t nFrameIdx) {return m_oInputPrecacher.getPacket(nFrameIdx);}
        virtual const cv::Mat& getGTFrame(size_t nFrameIdx) {return m_oGTPrecacher.getPacket(nFrameIdx);}

    protected:

        virtual cv::Mat _getInputPacket_impl(size_t nIdx) override {
            cv::Mat oFrame;
            if(!m_voVideoReader.isOpened())
                oFrame = cv::imread(m_vsInputFramePaths[nIdx],isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
            else {
                if(m_nNextExpectedVideoReaderFrameIdx!=nIdx) {
                    m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,(double)nIdx);
                    m_nNextExpectedVideoReaderFrameIdx = nIdx+1;
                }
                else
                    ++m_nNextExpectedVideoReaderFrameIdx;
                m_voVideoReader >> oFrame;
                if(isGrayscale() && oFrame.channels()>1)
                    cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2GRAY);
            }
            if(getDatasetInfo().is4ByteAligned() && oFrame.channels()==3)
                cv::cvtColor(oFrame,oFrame,cv::COLOR_BGR2BGRA);
            if(oFrame.size()!=m_oSize)
                cv::resize(oFrame,oFrame,m_oSize,0,0,cv::INTER_NEAREST);
            return oFrame;
        }

        virtual cv::Mat _getGTPacket_impl(size_t) override {
            return cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL));
        }

        size_t m_nFrameCount;
        std::vector<std::string> m_vsInputFramePaths;
        std::vector<std::string> m_vsGTFramePaths;
        cv::VideoCapture m_voVideoReader;
        size_t m_nNextExpectedVideoReaderFrameIdx;
        cv::Mat m_oROI;
        cv::Size m_oOrigSize,m_oSize;
        std::unordered_map<size_t,size_t> m_mTestGTIndexes;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, typename enable=void>
    struct DataProducer_; // no impl and no IDataProducer_ interface prevents compilation when not specialized

    template<eDatasetList eDataset> // partially specialized dataset producer type for default VideoSegm handling
    struct DataProducer_<eDatasetType_VideoSegm, eDataset> :
            public IDataProducer_<eDatasetType_VideoSegm> {
        static_assert(eDataset>=eDataset_VideoSegm_CDnet2012 && eDataset<=eDataset_VideoSegm_Custom,"dataset id is invalid for dataset type");
        DataProducer_()  {
            cv::Mat oTempImg;
            m_voVideoReader.open(getPath());
            if(!m_voVideoReader.isOpened()) {
                PlatformUtils::GetFilesFromDir(getPath(),m_vsInputFramePaths);
                if(!m_vsInputFramePaths.empty()) {
                    oTempImg = cv::imread(m_vsInputFramePaths[0]);
                    m_nFrameCount = m_vsInputFramePaths.size();
                }
            }
            else {
                m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
                m_voVideoReader >> oTempImg;
                m_voVideoReader.set(cv::CAP_PROP_POS_FRAMES,0);
                m_nFrameCount = (size_t)m_voVideoReader.get(cv::CAP_PROP_FRAME_COUNT);
            }
            if(oTempImg.empty())
                throw std::runtime_error(cv::format("Sequence '%s': video could not be opened via VideoReader or imread (you might need to implement your own DataProducer_ interface)",getName().c_str()));
            m_oOrigSize = oTempImg.size();
            if(getDatasetInfo().m_dScaleFactor!=1.0)
                cv::resize(oTempImg,oTempImg,cv::Size(),getDatasetInfo().m_dScaleFactor,getDatasetInfo().m_dScaleFactor,cv::INTER_NEAREST);
            m_oROI = cv::Mat(oTempImg.size(),CV_8UC1,cv::Scalar_<uchar>(255));
            m_oSize = oTempImg.size();
            m_nNextExpectedVideoReaderFrameIdx = 0;
            CV_Assert(m_nFrameCount>0);
        }
    };

    template<eDatasetList eDataset> // partially specialized dataset producer type for default CDnet (2012+2014) handling
    struct DataProducer_<eDatasetType_VideoSegm, eDataset, typename std::enable_if<((eDataset==eDataset_VideoSegm_CDnet2012)||(eDataset==eDataset_VideoSegm_CDnet2014))>::type> final :
            public IDataProducer_<eDatasetType_VideoSegm> {
        DataProducer_()  {
            std::vector<std::string> vsSubDirs;
            PlatformUtils::GetSubDirsFromDir(getPath(),vsSubDirs);
            auto gtDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getPath()+"/groundtruth");
            auto inputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),getPath()+"/input");
            if(gtDir==vsSubDirs.end() || inputDir==vsSubDirs.end())
                throw std::runtime_error(cv::format("CDnet Sequence '%s' did not possess the required groundtruth and input directories",getName().c_str()));
            PlatformUtils::GetFilesFromDir(*inputDir,m_vsInputFramePaths);
            PlatformUtils::GetFilesFromDir(*gtDir,m_vsGTFramePaths);
            if(m_vsGTFramePaths.size()!=m_vsInputFramePaths.size())
                throw std::runtime_error(cv::format("CDnet Sequence '%s' did not possess same amount of GT & input frames",getName().c_str()));
            m_oROI = cv::imread(getPath()+"/ROI.bmp",cv::IMREAD_GRAYSCALE);
            if(m_oROI.empty())
                throw std::runtime_error(cv::format("CDnet Sequence '%s' did not possess a ROI.bmp file",getName().c_str()));
            m_oROI = m_oROI>0;
            m_oSize = m_oROI.size();
            m_nFrameCount = m_vsInputFramePaths.size();
            CV_Assert(m_nFrameCount>0);
            // note: in this case, no need to use m_vnTestGTIndexes since all # of gt frames == # of test frames (but we assume the frames returned by 'GetFilesFromDir' are ordered correctly...)
            //m_pEvaluator = std::shared_ptr<EvaluatorBase>(new CDnetEvaluator()); @@@@
        }

        virtual cv::Mat _getGTPacket_impl(size_t nIdx) override final {
            cv::Mat oFrame = cv::imread(m_vsGTFramePaths[nIdx],cv::IMREAD_GRAYSCALE);
            if(oFrame.empty())
                oFrame = cv::Mat(m_oSize,CV_8UC1,cv::Scalar_<uchar>(DATASETUTILS_OUT_OF_SCOPE_DEFAULT_VAL));
            else if(oFrame.size()!=m_oSize)
                cv::resize(oFrame,oFrame,m_oSize,0,0,cv::INTER_NEAREST);
            return oFrame;
        }
    };

    template<eDatasetTypeList eDatasetType>
    struct IDataConsumer_; // specialized consumer (exposes interface for single dataset types -- no need for generalized interface)

    template<>
    struct IDataConsumer_<eDatasetType_VideoSegm> : public IDataEvaluator_<eDatasetType_VideoSegm> {
        IDataConsumer_() : m_nProcessedFrames(0) {}

        virtual void pushSegmResult(const cv::Mat& /*oUnused*/, size_t /*nIdx*/) {++m_nProcessedFrames;}
        virtual void _stopProcessing() override {m_nPacketsProcessed.set_value(m_nProcessedFrames);}
        virtual size_t getProcessedFramesCountPromise() {return m_nPacketsProcessed.get_future().get();}
        virtual size_t getProcessedFramesCount() {return m_nProcessedFrames;}

        virtual cv::Mat readSegmResult(size_t nIdx) const {
            CV_Assert(!getDatasetInfo().m_sResultNameSuffix.empty());
            std::array<char,10> acBuffer;
            snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
            std::stringstream sResultFilePath;
            sResultFilePath << getResultsPath() << getDatasetInfo().m_sResultNamePrefix << acBuffer.data() << getDatasetInfo().m_sResultNameSuffix;
            return cv::imread(sResultFilePath.str(),isGrayscale()?cv::IMREAD_GRAYSCALE:cv::IMREAD_COLOR);
        }

        virtual void writeSegmResult(const cv::Mat& oSegm, size_t nIdx) const {
            CV_Assert(!getDatasetInfo().m_sResultNameSuffix.empty());
            std::array<char,10> acBuffer;
            snprintf(acBuffer.data(),acBuffer.size(),"%06zu",nIdx);
            std::stringstream sResultFilePath;
            sResultFilePath << getResultsPath() << getDatasetInfo().m_sResultNamePrefix << acBuffer.data() << getDatasetInfo().m_sResultNameSuffix;
            const std::vector<int> vnComprParams = {cv::IMWRITE_PNG_COMPRESSION,9};
            cv::imwrite(sResultFilePath.str(),oSegm,vnComprParams);
        }

    protected:
        size_t m_nProcessedFrames;
        std::promise<size_t> m_nPacketsProcessed;
    };


    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, typename enable=void>
    struct DataConsumer_; // no impl and no IDataConsumer_ interface prevents compilation when not specialized

    template<eDatasetList eDataset> // partially specialized dataset consumer type for default VideoSegm handling
    struct DataConsumer_<eDatasetType_VideoSegm, eDataset> : public IDataConsumer_<eDatasetType_VideoSegm,Evaluator_<eDatasetType_VideoSegm,eDataset>> {};

    template<eDatasetList eDataset> // partially specialized dataset producer type for default CDnet (2012+2014) handling
    struct DataConsumer_<eDatasetType_VideoSegm, eDataset, typename std::enable_if<((eDataset==eDataset_VideoSegm_CDnet2012)||(eDataset==eDataset_VideoSegm_CDnet2014))>::type> final :
            public IDataConsumer_<eDatasetType_VideoSegm,Evaluator_<eDatasetType_VideoSegm,eDataset>> {

        DataConsumer_() {
            //m_pEvaluator = std::make_shared(); @@@@
        }

        virtual void pushSegmResult(const cv::Mat& oSegm, size_t nIdx) override {
            //m_pEvaluator-> @@@@
            IDataConsumer_<eDatasetType_VideoSegm>::pushSegmResult(oSegm,nIdx);
        }
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset>
    struct IDataset_ : public IDatasetEvaluator_<eDatasetType,eDataset> { // dataset interface specialization for smaller impl sizes
        IDataset_(const std::string& sDatasetName, const std::string& sDatasetRootPath, const std::string& sResultsRootPath,
                  const std::string& sResultNamePrefix, const std::string& sResultNameSuffix, const std::vector<std::string>& vsWorkBatchPaths,
                  const std::vector<std::string>& vsSkippedNameTokens, const std::vector<std::string>& vsGrayscaleNameTokens,
                  size_t nOutputIdxOffset, double dScaleFactor, bool bForce4ByteDataAlign) :
                m_sDatasetName(sDatasetName),m_sDatasetRootPath(sDatasetRootPath),m_sResultsRootPath(sResultsRootPath),
                m_sResultNamePrefix(sResultNamePrefix),m_sResultNameSuffix(sResultNameSuffix),m_vsWorkBatchPaths(vsWorkBatchPaths),
                m_vsSkippedNameTokens(vsSkippedNameTokens),m_vsGrayscaleNameTokens(vsGrayscaleNameTokens),
                m_nOutputIdxOffset(nOutputIdxOffset),m_dScaleFactor(dScaleFactor),m_bForce4ByteDataAlign(bForce4ByteDataAlign) {}
        struct WorkBatch :
                public DataHandler,
                public DataProducer_<eDatasetType,eDataset>,
                public DataConsumer_<eDatasetType,eDataset> {
            WorkBatch(const std::string& sBatchName, const IDataset& oDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sBatchName,oDataset,sRelativePath) {}
            virtual ~WorkBatch() = default;
            virtual eDatasetTypeList getDatasetType() const override final {return eDatasetType;}
            virtual eDatasetList getDataset() const override final {return eDataset;}
            virtual bool isBare() const override final {return false;}
            virtual bool isGroup() const override final {return false;}
            virtual IDataHandlerPtrArray getBatches() const override final {return IDataHandlerPtrArray();}
            virtual void startProcessing() override final {_startProcessing(); m_oStopWatch.tick();}
            virtual void stopProcessing() override final {_stopProcessing(); m_dElapsedTime_sec = m_oStopWatch.tock();}
            virtual double getProcessTime() const override final {return m_dElapsedTime_sec;}
        private:
            WorkBatch& operator=(const WorkBatch&) = delete;
            WorkBatch(const WorkBatch&) = delete;
            friend class WorkBatchGroup;
            CxxUtils::StopWatch m_oStopWatch;
            double m_dElapsedTime_sec;
        };
        struct WorkBatchGroup :
                public DataHandler { // @@@@@@@ inherit DataEvaluator_<1,2>? or interface?
            WorkBatchGroup(const std::string& sGroupName, const IDataset& oDataset, const std::string& sRelativePath=std::string("./")) :
                    DataHandler(sGroupName,oDataset,sRelativePath+"/"+sGroupName+"/"), m_bIsBare(false) {
                PlatformUtils::CreateDirIfNotExist(m_sResultsPath);
                if(!PlatformUtils::string_contains_token(getName(),m_oDataset.m_vsSkippedNameTokens)) {
                    std::cout << "[" << m_oDataset.m_sDatasetName << "] -- Parsing directory '" << m_oDataset.m_sDatasetRootPath+sRelativePath << "' for work group '" << getName() << "'..." << std::endl;
                    std::vector<std::string> vsWorkBatchPaths;
                    // all subdirs are considered work batch directories (if none, the category directory itself is a batch)
                    PlatformUtils::GetSubDirsFromDir(m_sDatasetPath,vsWorkBatchPaths);
                    if(vsWorkBatchPaths.empty()) {
                        m_vpBatches.push_back(std::make_shared<WorkBatch>(getName(),m_oDataset,m_sRelativePath));
                        m_bIsBare = true;
                    }
                    else {
                        for(auto&& sPathIter : vsWorkBatchPaths) {
                            const size_t nLastSlashPos = sPathIter.find_last_of("/\\");
                            const std::string sNewBatchName = nLastSlashPos==std::string::npos?sPathIter:sPathIter.substr(nLastSlashPos+1);
                            if(!PlatformUtils::string_contains_token(sNewBatchName,m_oDataset.m_vsSkippedNameTokens))
                                m_vpBatches.push_back(std::make_shared<WorkBatch>(sNewBatchName,m_oDataset,m_sRelativePath+"/"+sNewBatchName+"/"));
                        }
                    }
                }
            }
            virtual ~WorkBatchGroup() = default;
            virtual double getExpectedLoad() const override {
                return std::accumulate(m_vpBatches.begin(),m_vpBatches.end(),0.0,[&](double dSum, const IDataHandlerPtr& p) {
                    return dSum + p->getExpectedLoad();
                });
            }
            virtual size_t getTotPackets() const override {
                return std::accumulate(m_vpBatches.begin(),m_vpBatches.end(),0.0,[&](size_t nSum, const IDataHandlerPtr& p) {
                    return nSum + p->getTotPackets();
                });
            }
            virtual bool isBare() const override final {return m_bIsBare;}
            virtual bool isGroup() const override final {return true;}
            virtual IDataHandlerPtrArray getBatches() const override final {return m_vpBatches;}
            virtual void startProcessing() override final {_startProcessing();}
            virtual void stopProcessing() override final {_stopProcessing();}
            virtual double getProcessTime() const override final {
                return std::accumulate(m_vpBatches.begin(),m_vpBatches.end(),0.0,[&](double dSum, const IDataHandlerPtr& p) {
                    return dSum + p->getProcessTime();
                });
            }
        private:
            bool m_bIsBare;
            IDataHandlerPtrArray m_vpBatches;
            WorkBatchGroup& operator=(const WorkBatchGroup&) = delete;
            WorkBatchGroup(const WorkBatchGroup&) = delete;
        };

        virtual const std::string& getDatasetName() const override final {return m_sDatasetName;}
        virtual const std::string& getDatasetRootPath() const override final {return m_sDatasetRootPath;}
        virtual const std::string& getResultsRootPath() const override final {return m_sResultsRootPath;}
        virtual const std::string& getResultsNamePrefix() const override final {return m_sResultNamePrefix;}
        virtual const std::string& getResultsNameSuffix() const override final {return m_sResultNameSuffix;}
        virtual const std::vector<std::string>& getWorkBatchPaths() const override final {return m_vsWorkBatchPaths;}
        virtual const std::vector<std::string>& getSkippedNameTokens() const override final {return m_vsSkippedNameTokens;}
        virtual const std::vector<std::string>& getGrayscaleNameTokens() const override final {return m_vsGrayscaleNameTokens;}
        virtual size_t getOutputIdxOffset() const override final {return m_nOutputIdxOffset;}
        virtual double getScaleFactor() const override final {return m_dScaleFactor;}
        virtual bool is4ByteAligned() const override final {return m_bForce4ByteDataAlign;}

        virtual void parseDataset() override final {
            m_vpBatches.clear();
            if(!m_sResultsRootPath.empty())
                PlatformUtils::CreateDirIfNotExist(m_sResultsRootPath);
            for(auto&& sPathIter : m_vsWorkBatchPaths)
                m_vpBatches.push_back(std::make_shared<WorkBatchGroup>(sPathIter,*this));
        }

        virtual IDataHandlerPtrArray getBatches() const override final {
            return m_vpBatches;
        }

        virtual IDataHandlerPtrQueue getSortedBatches() const override final {
            IDataHandlerPtrQueue vpBatches(&IDataHandler::compare_load);
            std::function<void(const IDataHandlerPtr&)> lPushBatches = [&](const IDataHandlerPtr& pBatch) {
                if(pBatch->isGroup())
                    for(auto&& pSubBatch : pBatch->getBatches())
                        lPushBatches(pSubBatch);
                else
                    vpBatches.push(pBatch);
            };
            for(auto&& pBatch : getBatches())
                lPushBatches(pBatch);
            return vpBatches;
        }

    protected:
        const std::string m_sDatasetName;
        const std::string m_sDatasetRootPath;
        const std::string m_sResultsRootPath;
        const std::string m_sResultNamePrefix;
        const std::string m_sResultNameSuffix;
        const std::vector<std::string> m_vsWorkBatchPaths;
        const std::vector<std::string> m_vsSkippedNameTokens;
        const std::vector<std::string> m_vsGrayscaleNameTokens;
        const size_t m_nOutputIdxOffset;
        const double m_dScaleFactor;
        const bool m_bForce4ByteDataAlign;

        IDataHandlerPtrArray m_vpBatches;
    };

    template<eDatasetTypeList eDatasetType, eDatasetList eDataset, typename enable=void>
    struct Dataset_; // no impl and no IDataset interface prevents compilation when not specialized

    template<> // fully specialized dataset for default VideoSegm dataset handling
    struct Dataset_<eDatasetType_VideoSegm, eDataset_VideoSegm_Custom> :
            public IDataset_<eDatasetType_VideoSegm, eDataset_VideoSegm_Custom> {
        using IDataset_<eDatasetType_VideoSegm, eDataset_VideoSegm_Custom>::IDataset_<eDatasetType_VideoSegm, eDataset_VideoSegm_Custom>;
    };

    template<eDatasetList eDataset> // partially specialized dataset for default CDnet (2012+2014) handling
    struct Dataset_<eDatasetType_VideoSegm, eDataset, typename std::enable_if<((eDataset==eDataset_VideoSegm_CDnet2012)||(eDataset==eDataset_VideoSegm_CDnet2014))>::type> final :
            public IDataset_<eDatasetType_VideoSegm, eDataset_VideoSegm_Custom> {
        Dataset_(const std::string& sDatasetRootPath, const std::string& sResultsDirName, bool bForce4ByteDataAlign, double dScaleFactor) :
                IDataset_( eDataset==eDataset_VideoSegm_CDnet2012?"CDnet 2012":"CDnet 2014",
                           eDataset==eDataset_VideoSegm_CDnet2012?sDatasetRootPath+"/CDNet/dataset/":sDatasetRootPath+"/CDNet2014/dataset/",
                           eDataset==eDataset_VideoSegm_CDnet2012?sDatasetRootPath+"/CDNet/"+sResultsDirName+"/":sDatasetRootPath+"/CDNet2014/"+sResultsDirName+"/",
                           "bin",
                           ".png",
                           eDataset==eDataset_VideoSegm_CDnet2012?std::vector<std::string>{"baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","shadow","thermal"}:std::vector<std::string>{"badWeather","baseline","cameraJitter","dynamicBackground","intermittentObjectMotion","lowFramerate","nightVideos","PTZ","shadow","thermal","turbulence"},
                           std::vector<std::string>{},
                           eDataset==eDataset_VideoSegm_CDnet2012?std::vector<std::string>{"thermal"}:std::vector<std::string>{"thermal","turbulence"},
                           1,
                           dScaleFactor,
                           bForce4ByteDataAlign) {}
    };
    typedef Dataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet2012> CDnet2012;
    typedef Dataset_<eDatasetType_VideoSegm,eDataset_VideoSegm_CDnet2014> CDnet2014;

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

} //namespace DatasetUtils
