
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2017 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#ifndef _LITIV_DATASETS_IMPL_H_
#error "This file should never be included directly; use litiv/datasets.hpp instead"
#endif //_LITIV_DATASETS_IMPL_H_

#include "litiv/datasets.hpp" // for parsers only, not truly required here
#include <opencv2/calib3d.hpp>

#ifndef DATASETS_LITIV2018_LOAD_CALIB_DATA
#define DATASETS_LITIV2018_LOAD_CALIB_DATA 0
#endif //ndef(DATASETS_LITIV2018_LOAD_CALIB_DATA)
#define DATASETS_LITIV2018_RECTIFIED_SIZE cv::Size(640,480)
#ifndef DATASETS_LITIV2018_VERSION
#define DATASETS_LITIV2018_VERSION "-v1"
#endif //ndef(DATASETS_LITIV2018_VERSION)
#ifndef DATASETS_LITIV2018_FLIP_RGB
#define DATASETS_LITIV2018_FLIP_RGB 1
#endif //ndef(DATASETS_LITIV2018_FLIP_RGB)

namespace lv {

    /// parameter interface for LITIV cosegm/registration dataset loader impl
    struct ILITIVStCharles2018Dataset {
        /// returns whether the depth stream should be loaded & provided as input or not (if not, the dataset is used as a bimodal one)
        virtual bool isLoadingDepth() const = 0;
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        virtual bool isUndistorting() const = 0;
        /// returns whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
        virtual bool isHorizRectifying() const = 0;
        /// returns whether we should evaluate fg/bg segmentation or stereo disparities
        virtual bool isEvaluatingDisparities() const = 0;
        /// returns whether frames should be flipped to use inverted disparities (from rgb to thermal) or not
        virtual bool isFlippingDisparities() const = 0;
        /// returns whether only a subset of the dataset's frames will be loaded or not
        virtual bool isLoadingFrameSubset() const = 0;
        /// returns whether only a subset of the dataset's frames will be evaluated or not
        virtual bool isEvaluatingOnlyFrameSubset() const = 0;
        /// returns the evaluation window temporal size used around each frame in subset mode
        virtual int getEvalTemporalWindowSize() const = 0;
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        virtual int isLoadingInputMasks() const = 0;
        /// list of stream indices used in the array-based implementation below
        enum StreamIdxList {
            LITIV2018_RGB=0,
            LITIV2018_LWIR=1,
            LITIV2018_Depth=2,
            // @@@@ should add streams for body joints & nir images
            //LITIV2018_NIR,
            //LITIV2018_Joints,
        };
    };

    /// dataset loader impl specialization for LITIV cosegm/registration dataset -- instantiated via lv::datasets::create(...)
    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_LITIV_stcharles2018,eEvalImpl> :
            public ILITIVStCharles2018Dataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_stcharles2018,lv::getDatasetEval<eDatasetTask,Dataset_LITIV_stcharles2018>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via lv::datasets::create
        /// specialization constructor, with all required extra parameters; these will be forwarded by lv::datasets::create(...)
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in dataset results folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bLoadDepth=true, ///< defines whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
                bool bUndistort=true, ///< defines whether images should be undistorted when loaded or not, using the calib files provided with the dataset
                bool bHorizRectify=false, ///< defines whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
                bool bEvalDisparities=false, ///< defines whether we should evaluate fg/bg segmentation or stereo disparities
                bool bFlipDisparities=false, ///< defines whether frames should be flipped to use inverted disparities (from rgb to thermal) or not
                bool bLoadFrameSubset=false, ///< defines whether only a subset of the dataset's frames will be loaded or not
                bool bEvalOnlyFrameSubset=false, ///< defines whether only a subset of the dataset's frames will be evaluated or not (if bLoadFrameSubset==true, has no effect)
                int nEvalTemporalWindowSize=0, ///< defines the evaluation window temporal size around each frame to eval in subset mode (if bLoadFrameSubset==false and bEvalOnlyFrameSubset==false, has no effect)
                int nLoadInputMasks=0, ///< defines whether the input stream should be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
                double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_stcharles2018,lv::getDatasetEval<eDatasetTask,Dataset_LITIV_stcharles2018>(),eEvalImpl>(
                        "LITIV-stcharles2018" DATASETS_LITIV2018_VERSION,
                        lv::datasets::getRootPath()+"litiv/stcharles2018" DATASETS_LITIV2018_VERSION "/",
                        DataHandler::createOutputDir(lv::datasets::getRootPath()+"litiv/stcharles2018" DATASETS_LITIV2018_VERSION "/results/",sOutputDirName),
                        getWorkBatchDirNames(),
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        false,
                        dScaleFactor
                ),
                m_bLoadDepth(bLoadDepth),
                m_bUndistort(bUndistort||bHorizRectify),
                m_bHorizRectify(bHorizRectify),
                m_bEvalDisparities(bEvalDisparities),
                m_bFlipDisparities(bFlipDisparities),
                m_bLoadFrameSubset(bLoadFrameSubset),
                m_bEvalOnlyFrameSubset(bEvalOnlyFrameSubset),
                m_nEvalTemporalWindowSize(nEvalTemporalWindowSize),
                m_nLoadInputMasks(nLoadInputMasks) {}
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames() {
        #if DATASETS_LITIV2018_LOAD_CALIB_DATA
            static const std::vector<std::string> s_vsWorkBatchDirs = {"calib02"};
        #else //!DATASETS_LITIV2018_LOAD_CALIB_DATA
            static const std::vector<std::string> s_vsWorkBatchDirs = {"vid01","vid02","vid03"};
        #endif //!DATASETS_LITIV2018_LOAD_CALIB_DATA
            return s_vsWorkBatchDirs;
        }
        /// returns whether the depth stream should be loaded & provided as input or not (if not, the dataset is used as a bimodal one)
        virtual bool isLoadingDepth() const override {return m_bLoadDepth;}
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        virtual bool isUndistorting() const override {return m_bUndistort;}
        /// returns whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
        virtual bool isHorizRectifying() const override {return m_bHorizRectify;}
        /// returns whether we should evaluate fg/bg segmentation or stereo disparities
        virtual bool isEvaluatingDisparities() const override {return m_bEvalDisparities;}
        /// returns whether frames should be flipped to use inverted disparities (from rgb to thermal) or not
        virtual bool isFlippingDisparities() const override {return m_bFlipDisparities;}
        /// returns whether only a subset of the dataset's frames will be loaded or not
        virtual bool isLoadingFrameSubset() const override {return m_bLoadFrameSubset;}
        /// returns whether only a subset of the dataset's frames will be evaluated or not
        virtual bool isEvaluatingOnlyFrameSubset() const override {return m_bEvalOnlyFrameSubset;}
        /// returns the evaluation window temporal size used around each frame in subset mode
        virtual int getEvalTemporalWindowSize() const override {return m_nEvalTemporalWindowSize;}
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        virtual int isLoadingInputMasks() const override {return m_nLoadInputMasks;}
    protected:
        const bool m_bLoadDepth;
        const bool m_bUndistort;
        const bool m_bHorizRectify;
        const bool m_bEvalDisparities;
        const bool m_bFlipDisparities;
        const bool m_bLoadFrameSubset;
        const bool m_bEvalOnlyFrameSubset;
        const int m_nEvalTemporalWindowSize;
        const int m_nLoadInputMasks;
    };

    /// data grouper handler impl specialization for LITIV cosegm/registration dataset; will skip groups entirely & forward data to batches
    template<DatasetTaskList eDatasetTask>
    struct DataGroupHandler_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_stcharles2018> :
            public DataGroupHandler {
    protected:
        virtual void parseData() override {
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            this->m_vpBatches.clear();
            this->m_bIsBare = true; // in this dataset, work batch groups are always bare
            if(!lv::string_contains_token(this->getName(),this->getSkipTokens()))
                this->m_vpBatches.push_back(this->createWorkBatch(this->getName(),this->getRelativePath()));
        }
    };

    /// data producer impl specialization for LITIV cosegm/registration dataset; provides required data i/o and undistort support
    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_stcharles2018> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_stcharles2018> {
        /// returns the number of parallel input streams (depends on whether loading depth or not)
        virtual size_t getInputStreamCount() const override final {
            // @@@@ should add streams for body joints & nir images
            return size_t((m_bLoadDepth?3:2)*(m_nLoadInputMasks?2:1));
        }
        /// returns the number of parallel gt streams (depends on whether loading depth or not)
        virtual size_t getGTStreamCount() const override final {
            return size_t(m_bLoadDepth?3:2);
        }
        /// returns the (friendly) name of an input stream specified by index
        virtual std::string getInputStreamName(size_t nStreamIdx) const override final {
            lvAssert(nStreamIdx<getInputStreamCount());
            if(m_nLoadInputMasks) {
                const std::string sBaseName = (((nStreamIdx/2)==ILITIVStCharles2018Dataset::LITIV2018_RGB)?"RGB":((nStreamIdx/2)==ILITIVStCharles2018Dataset::LITIV2018_LWIR)?"LWIR":((nStreamIdx/2)==ILITIVStCharles2018Dataset::LITIV2018_Depth)?"DEPTH":"UNKNOWN");
                return sBaseName+std::string((nStreamIdx%2)?"_MASK":"");
            }
            else
                return ((nStreamIdx==ILITIVStCharles2018Dataset::LITIV2018_RGB)?"RGB":(nStreamIdx==ILITIVStCharles2018Dataset::LITIV2018_LWIR)?"LWIR":(nStreamIdx==ILITIVStCharles2018Dataset::LITIV2018_Depth)?"DEPTH":"UNKNOWN");
        }
        /// returns the (friendly) name of a gt stream specified by index
        virtual std::string getGTStreamName(size_t nStreamIdx) const override final {
            lvAssert(nStreamIdx<getGTStreamCount());
            const std::string sBaseName = ((nStreamIdx==ILITIVStCharles2018Dataset::LITIV2018_RGB)?"RGB":(nStreamIdx==ILITIVStCharles2018Dataset::LITIV2018_LWIR)?"LWIR":(nStreamIdx==ILITIVStCharles2018Dataset::LITIV2018_Depth)?"DEPTH":"UNKNOWN");
            return sBaseName+"_GT";
        }
        /// returns whether the depth stream should be loaded & provided as input or not (if not, the dataset is used as a bimodal one)
        bool isLoadingDepth() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isLoadingDepth();
        }
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        bool isUndistorting() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isUndistorting();
        }
        /// returns whether images should be horizontally rectified when loaded or not, using the calib files provided with the dataset
        bool isHorizRectifying() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isHorizRectifying();
        }
        /// returns whether we should evaluate fg/bg segmentation or stereo disparities
        bool isEvaluatingDisparities() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isEvaluatingDisparities();
        }
        /// returns whether frames should be flipped to use inverted disparities (from rgb to thermal) or not
        bool isFlippingDisparities() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isFlippingDisparities();
        }
        /// returns whether only a subset of the dataset's frames will be loaded or not
        bool isLoadingFrameSubset() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isLoadingFrameSubset();
        }
        /// returns whether only a subset of the dataset's frames will be evaluated or not
        bool isEvaluatingOnlyFrameSubset() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isEvaluatingOnlyFrameSubset();
        }
        /// returns the evaluation window temporal size used around each frame in subset mode
        int getEvalTemporalWindowSize() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).getEvalTemporalWindowSize();
        }
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        int isLoadingInputMasks() const {
            return dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot()).isLoadingInputMasks();
        }
        /// returns the name of the directory where feature packets should be saved
        std::string getFeaturesDirName() const {
            return this->m_sFeaturesDirName;
        }
        /// sets the name of the directory where feature packets should be saved
        void setFeaturesDirName(const std::string& sDirName) {
            this->m_sFeaturesDirName = sDirName;
            lv::createDirIfNotExist(this->getFeaturesPath()+sDirName);
        }
        /// returns the name of a feature packet (with its directory name as prefix)
        virtual std::string getFeaturesName(size_t nPacketIdx) const override final {
            std::array<char,32> acBuffer;
            snprintf(acBuffer.data(),acBuffer.size(),nPacketIdx<size_t(1e7)?"%06zu":"%09zu",nPacketIdx);
            return lv::addDirSlashIfMissing(this->m_sFeaturesDirName)+std::string(acBuffer.data());
        }
        /// returns the minimum scene disparity offset
        size_t getMinDisparity() const {
            return this->m_nMinDisp;
        }
        /// returns the maximum scene disparity offset
        size_t getMaxDisparity() const {
            return this->m_nMaxDisp;
        }
        /// returns whether the given packet index is at a break between two temporal windows (only useful when subset-processing, always false otherwise)
        bool isTemporalWindowBreak(size_t nPacketIdx) const {
            if(nPacketIdx==0u)
                return true;
            const bool bLoadFrameSubset = this->isLoadingFrameSubset();
            const bool bEvalOnlyFrameSubset = this->isEvaluatingOnlyFrameSubset();
            if(!bLoadFrameSubset && !bEvalOnlyFrameSubset)
                return false;
            lvAssert_(!m_mSubset.empty(),"must parse data first");
            const int nEvalTemporalWindowSize = this->getEvalTemporalWindowSize();
            lvAssert_(nEvalTemporalWindowSize>=0,"bad temporal window size");
            if(nEvalTemporalWindowSize==0 || nPacketIdx>=m_mSubset.size()) // treat oob lookups as temporal breaks
                return true;
            auto pPacketIter = m_mSubset.begin();
            const size_t nPrevPacketIdx = nPacketIdx-1u;
            std::advance(pPacketIter,nPrevPacketIdx);
            const size_t nRealPrevPacketIdx = *pPacketIter;
            std::advance(pPacketIter,1u);
            const size_t nRealCurrPacketIdx = *pPacketIter;
            return ((nRealPrevPacketIdx+1u)!=nRealCurrPacketIdx);
        }

    protected:
        virtual void parseData() override final {
            // note: this function is called right after the constructor, so initialize everything for other calls here
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            const ILITIVStCharles2018Dataset& oDataset = dynamic_cast<const ILITIVStCharles2018Dataset&>(*this->getRoot());
            const bool bLoadFrameSubset = this->isLoadingFrameSubset();
            const bool bEvalOnlyFrameSubset = this->isEvaluatingOnlyFrameSubset();
            const bool bIsLoadingCalibData = bool(DATASETS_LITIV2018_LOAD_CALIB_DATA);
            const int nEvalTemporalWindowSize = this->getEvalTemporalWindowSize();
            lvAssert_(nEvalTemporalWindowSize>=0,"bad temporal window size");
            this->m_bLoadDepth = oDataset.isLoadingDepth();
            this->m_nLoadInputMasks = oDataset.isLoadingInputMasks();
            this->m_bUndistort = oDataset.isUndistorting();
            this->m_bHorizRectify = oDataset.isHorizRectifying();
            lvAssert_(!this->m_bHorizRectify || this->m_bUndistort,"if rectification is needed, must enable undistortion too");
            const bool bUseInterlacedMasks = this->m_nLoadInputMasks!=0;
            const bool bEvalDisparityMaps = this->isEvaluatingDisparities();
            const bool bFlipDisparities = this->isFlippingDisparities();
            lvAssert_(!bEvalDisparityMaps || this->m_bHorizRectify,"if evaluating disparity maps, must enable rectification");
            const size_t nInputStreamCount = this->getInputStreamCount();
            const size_t nGTStreamCount = this->getGTStreamCount();
            const size_t nInputRGBStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_RGB*(bUseInterlacedMasks?2:1));
            const size_t nInputLWIRStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_LWIR*(bUseInterlacedMasks?2:1));
            const size_t nInputDepthStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_Depth*(bUseInterlacedMasks?2:1));
            const size_t nInputRGBMaskStreamIdx = nInputRGBStreamIdx+1;
            const size_t nInputLWIRMaskStreamIdx = nInputLWIRStreamIdx+1;
            const size_t nInputDepthMaskStreamIdx = nInputDepthStreamIdx+1;
            const size_t nGTRGBStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_RGB);
            const size_t nGTLWIRStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_LWIR);
            const size_t nGTDepthStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_Depth);
            const double dScale = this->getScaleFactor();
            ////////////////////////////////
            if(bIsLoadingCalibData) {
                lvAssert_(!bLoadFrameSubset && !bEvalOnlyFrameSubset,"calib data must not be loaded with subset flag on (it is already sampled)");
                lvAssert_(!this->m_bLoadDepth,"calib data cannot be loaded with depth frames");
                lvAssert_(!this->m_nLoadInputMasks,"calib data cannot be loaded with input masks");
            }
            const std::string sDirNameSuffix = bIsLoadingCalibData?"_subset":"";
            const std::vector<std::string> vsSubDirs = lv::getSubDirsFromDir(this->getDataPath());
            auto psRGBGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+(bEvalDisparityMaps?"rgb_gt_disp":"rgb_gt_masks"));
            auto psRGBMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"rgb_masks"+sDirNameSuffix);
            auto psRGBFramesDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"rgb"+sDirNameSuffix);
            if(psRGBFramesDir==vsSubDirs.end())
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess the required RGB frame subdirectory in folder '%s'",this->getName().c_str(),this->getDataPath().c_str());
            std::vector<std::string> vsRGBFramePaths = lv::getFilesFromDir(*psRGBFramesDir);
            std::vector<std::string> vsRGBMaskPaths = (psRGBMasksDir==vsSubDirs.end())?std::vector<std::string>{}:lv::getFilesFromDir(*psRGBMasksDir);
            std::vector<std::string> vsRGBGTPaths = (psRGBGTDir==vsSubDirs.end())?std::vector<std::string>{}:lv::getFilesFromDir(*psRGBGTDir);
            lv::filterFilePaths(vsRGBFramePaths,{},{".jpg"});
            lv::filterFilePaths(vsRGBMaskPaths,{},{".png"});
            lv::filterFilePaths(vsRGBGTPaths,{},{".png"});
            lvAssert__(vsRGBFramePaths.size()>0u && vsRGBMaskPaths.size()<=vsRGBFramePaths.size() && vsRGBGTPaths.size()<=vsRGBFramePaths.size(),"bad input rgb packet count for sequence '%s'",this->getName().c_str());
            auto psLWIRGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+(bEvalDisparityMaps?"lwir_gt_disp":"lwir_gt_masks"));
            auto psLWIRMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"lwir_masks"+sDirNameSuffix);
            auto psLWIRFramesDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"lwir"+sDirNameSuffix);
            if(psLWIRFramesDir==vsSubDirs.end())
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess the required LWIR frame subdirectory",this->getName().c_str());
            std::vector<std::string> vsLWIRFramePaths = lv::getFilesFromDir(*psLWIRFramesDir);
            std::vector<std::string> vsLWIRMaskPaths = (psLWIRMasksDir==vsSubDirs.end())?std::vector<std::string>{}:lv::getFilesFromDir(*psLWIRMasksDir);
            std::vector<std::string> vsLWIRGTPaths = (psLWIRGTDir==vsSubDirs.end())?std::vector<std::string>{}:lv::getFilesFromDir(*psLWIRGTDir);
            lv::filterFilePaths(vsLWIRFramePaths,{},{".jpg"});
            lv::filterFilePaths(vsLWIRMaskPaths,{},{".png"});
            lv::filterFilePaths(vsLWIRGTPaths,{},{".png"});
            lvAssert__(vsLWIRFramePaths.size()>0u && vsLWIRMaskPaths.size()<=vsLWIRFramePaths.size() && vsLWIRGTPaths.size()<=vsLWIRFramePaths.size(),"bad input lwir packet count for sequence '%s'",this->getName().c_str());
            const size_t nTotInputPackets = bIsLoadingCalibData?std::max(vsRGBFramePaths.size(),vsLWIRFramePaths.size()):vsRGBFramePaths.size();
            auto psDepthGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+(bEvalDisparityMaps?"depth_gt_disp":"depth_gt_masks"));
            auto psDepthMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"depth_masks"+sDirNameSuffix);
            auto psDepthFramesDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"depth"+sDirNameSuffix);
            std::vector<std::string> vsDepthFramePaths,vsDepthMaskPaths,vsDepthGTPaths;
            if(this->m_bLoadDepth) {
                if(psDepthFramesDir==vsSubDirs.end())
                    lvError_("LITIV-stcharles2018 sequence '%s' did not possess the required depth frame subdirectory",this->getName().c_str());
                vsDepthFramePaths = lv::getFilesFromDir(*psDepthFramesDir);
                vsDepthMaskPaths = (psDepthMasksDir==vsSubDirs.end())?std::vector<std::string>{}:lv::getFilesFromDir(*psDepthMasksDir);
                vsDepthGTPaths = (psDepthGTDir==vsSubDirs.end())?std::vector<std::string>{}:lv::getFilesFromDir(*psDepthGTDir);
                lv::filterFilePaths(vsDepthFramePaths,{},{".png"});
                lv::filterFilePaths(vsDepthMaskPaths,{},{".png"});
                lv::filterFilePaths(vsDepthGTPaths,{},{".png"});
                lvAssert__(vsDepthFramePaths.size()>0u && vsDepthMaskPaths.size()<=vsDepthFramePaths.size() && vsDepthGTPaths.size()<=vsDepthFramePaths.size(),"bad input depth packet count for sequence '%s'",this->getName().c_str());
            }
            ////////////////////////////////
            const auto lIndexExtractor = [&](const std::string& sFilePath) {
                const size_t nLastInputSlashPos = sFilePath.find_last_of("/\\");
                const std::string sInputFileNameExt = nLastInputSlashPos==std::string::npos?sFilePath:sFilePath.substr(nLastInputSlashPos+1);
                const size_t nLastInputDotPos = sInputFileNameExt.find_last_of('.');
                const std::string sInputFileName = (nLastInputDotPos==std::string::npos)?sInputFileNameExt:sInputFileNameExt.substr(0,nLastInputDotPos);
                return (size_t)std::stoi(sInputFileName);
            };
            this->m_mRealSubset.clear();
            this->m_mSubset.clear();
            if(!bIsLoadingCalibData) {
                lvAssert_(vsLWIRFramePaths.size()==nTotInputPackets && (!this->m_bLoadDepth || vsDepthFramePaths.size()==nTotInputPackets),"input packet array sizes mismatch");
                for(size_t nCurrIdx=0u; nCurrIdx<nTotInputPackets; ++nCurrIdx) {
                    lvAssert(lIndexExtractor(vsRGBFramePaths[nCurrIdx])==nCurrIdx);
                    lvAssert(lIndexExtractor(vsLWIRFramePaths[nCurrIdx])==nCurrIdx);
                    lvAssert(!this->m_bLoadDepth || lIndexExtractor(vsDepthFramePaths[nCurrIdx])==nCurrIdx);
                }
                lvAssert(vsRGBMaskPaths.empty() || lIndexExtractor(vsRGBMaskPaths.back())<nTotInputPackets);
                lvAssert(vsRGBGTPaths.empty() || lIndexExtractor(vsRGBGTPaths.back())<nTotInputPackets);
                lvAssert(vsLWIRMaskPaths.empty() || lIndexExtractor(vsLWIRMaskPaths.back())<nTotInputPackets);
                lvAssert(vsLWIRGTPaths.empty() || lIndexExtractor(vsLWIRGTPaths.back())<nTotInputPackets);
                lvAssert(!this->m_bLoadDepth || vsDepthMaskPaths.empty() || lIndexExtractor(vsDepthMaskPaths.back())<nTotInputPackets);
                lvAssert(!this->m_bLoadDepth || vsDepthGTPaths.empty() || lIndexExtractor(vsDepthGTPaths.back())<nTotInputPackets);
                if(bLoadFrameSubset || bEvalOnlyFrameSubset) {
                    const std::string sSubsetFilePath = this->getDataPath()+"subset.txt";
                    std::ifstream oSubsetFile(sSubsetFilePath);
                    lvAssert__(oSubsetFile.is_open(),"could not open frame subset file at '%s'",sSubsetFilePath.c_str());
                    std::string sLineBuffer;
                    while(oSubsetFile) {
                        lvDbgExceptionWatch;
                        if(!std::getline(oSubsetFile,sLineBuffer))
                            break;
                        if(sLineBuffer[0]=='#')
                            continue;
                        const int nCurrIdx = std::stoi(sLineBuffer);
                        lvAssert(nCurrIdx>=0 && nCurrIdx<(int)nTotInputPackets);
                        this->m_mRealSubset.insert(size_t(nCurrIdx));
                        for(int nOffsetIdx=nCurrIdx-nEvalTemporalWindowSize; nOffsetIdx<=nCurrIdx+nEvalTemporalWindowSize; ++nOffsetIdx)
                            if(nOffsetIdx>=0 && nOffsetIdx<(int)nTotInputPackets)
                                this->m_mSubset.insert(size_t(nOffsetIdx));
                    }
                    lvAssert(this->m_mSubset.size()>0);
                }
                else {
                    for(size_t nCurrIdx=0u; nCurrIdx<nTotInputPackets; ++nCurrIdx) {
                        this->m_mRealSubset.insert(nCurrIdx);
                        this->m_mSubset.insert(nCurrIdx);
                    }
                }
            }
            else {
                for(size_t nCurrIdx=0u; nCurrIdx<nTotInputPackets; ++nCurrIdx) {
                    const size_t nRGBIdx = (nCurrIdx<vsRGBFramePaths.size())?lIndexExtractor(vsRGBFramePaths[nCurrIdx]):SIZE_MAX;
                    if(nRGBIdx!=SIZE_MAX) {
                        this->m_mRealSubset.insert(nRGBIdx);
                        this->m_mSubset.insert(nRGBIdx);
                    }
                    const size_t nLWIRIdx = (nCurrIdx<vsLWIRFramePaths.size())?lIndexExtractor(vsLWIRFramePaths[nCurrIdx]):SIZE_MAX;
                    if(nLWIRIdx!=SIZE_MAX) {
                        this->m_mRealSubset.insert(nLWIRIdx);
                        this->m_mSubset.insert(nLWIRIdx);
                    }
                }
            }
            const size_t nInputPackets = this->m_mSubset.size();
            const auto lSubsetCleaner = [&](std::vector<std::string>& vsPaths, const std::set<size_t>& mSubset, bool bMustIncludeAll=true) {
                lvAssert(!mSubset.empty());
                std::vector<std::string> vsKeptPaths;
                if(bMustIncludeAll) {
                    lvAssert(vsPaths.size()>*mSubset.rbegin());
                    for(size_t nIdx : mSubset) {
                        lvAssert(vsPaths[nIdx].find(lv::putf("%05d",(int)nIdx))!=std::string::npos);
                        lvAssert(lIndexExtractor(vsPaths[nIdx])==nIdx);
                        vsKeptPaths.push_back(vsPaths[nIdx]);
                    }
                }
                else {
                    for(size_t nIdx : mSubset) {
                        for(const auto& sPath : vsPaths) {
                            if(sPath.find(lv::putf("%05d",(int)nIdx))!=std::string::npos) {
                                lvAssert(lIndexExtractor(vsPaths[nIdx])==nIdx);
                                vsKeptPaths.push_back(vsPaths[nIdx]);
                            }
                        }
                    }
                }
                vsPaths = vsKeptPaths;
            };
            ////////////////////////////////
            const cv::Size oRGBSize(1920,1080),oLWIRSize(320,240),oDepthSize(512,424),oRectifSize(DATASETS_LITIV2018_RECTIFIED_SIZE);
            cv::Mat oRGBROI = cv::imread(this->getDataPath()+"rgb_roi.png",cv::IMREAD_GRAYSCALE);
            cv::Mat oLWIRROI = cv::imread(this->getDataPath()+"lwir_roi.png",cv::IMREAD_GRAYSCALE);
            cv::Mat oDepthROI = cv::imread(this->getDataPath()+"depth_roi.png",cv::IMREAD_GRAYSCALE);
            if(!oRGBROI.empty()) {
                lvAssert(oRGBROI.type()==CV_8UC1 && oRGBROI.size()==oRGBSize);
                oRGBROI = oRGBROI>128;
            }
            else
                oRGBROI = cv::Mat(oRGBSize,CV_8UC1,cv::Scalar_<uchar>(255));
            if(!oLWIRROI.empty()) {
                lvAssert(oLWIRROI.type()==CV_8UC1 && oLWIRROI.size()==oLWIRSize);
                oLWIRROI = oLWIRROI>128;
            }
            else
                oLWIRROI = cv::Mat(oLWIRSize,CV_8UC1,cv::Scalar_<uchar>(255));
            if(!oDepthROI.empty()) {
                lvAssert(oDepthROI.type()==CV_8UC1 && oDepthROI.size()==oDepthSize);
                oDepthROI = oDepthROI>128;
            }
            else
                oDepthROI = cv::Mat(oDepthSize,CV_8UC1,cv::Scalar_<uchar>(255));
            ////////////////////////////////
            this->m_nMinDisp = size_t(0);
            this->m_nMaxDisp = size_t(100);
            this->m_nLWIRDispOffset = 0;
            std::ifstream oDispRangeFile(this->getDataPath()+"drange.txt");
            if(oDispRangeFile.is_open() && !oDispRangeFile.eof()) {
                oDispRangeFile >> this->m_nMinDisp;
                if(!oDispRangeFile.eof())
                    oDispRangeFile >> this->m_nMaxDisp;
            }
            if(this->m_nMinDisp>size_t(0)) {
                this->m_nLWIRDispOffset = int(this->m_nMinDisp);
                this->m_nMaxDisp -= this->m_nMinDisp;
                this->m_nMinDisp = size_t(0);
            }
            this->m_nLWIRDispOffset = (int)std::round(dScale*this->m_nLWIRDispOffset);
            this->m_nMinDisp = (size_t)std::round(dScale*this->m_nMinDisp);
            this->m_nMaxDisp = (size_t)std::round(dScale*this->m_nMaxDisp);
            lvAssert(this->m_nMaxDisp>this->m_nMinDisp);
            ////////////////////////////////
            if(this->m_bUndistort) {
                const std::string sCalibDataFilePath = this->getDataPath()+"calib/calibdata.yml";
                cv::FileStorage oParamsFS(sCalibDataFilePath,cv::FileStorage::READ);
                lvAssert__(oParamsFS.isOpened(),"could not open calibration yml file at '%s'",sCalibDataFilePath.c_str());
                oParamsFS["aCamMats0"] >> this->m_oRGBCameraParams;
                oParamsFS["aDistCoeffs0"] >> this->m_oRGBDistortParams;
                lvAssert_(!this->m_oRGBCameraParams.empty() && !this->m_oRGBDistortParams.empty(),"failed to load RGB camera calibration parameters");
                oParamsFS["aCamMats1"] >> this->m_oLWIRCameraParams;
                oParamsFS["aDistCoeffs1"] >> this->m_oLWIRDistortParams;
                lvAssert_(!this->m_oLWIRCameraParams.empty() && !this->m_oLWIRDistortParams.empty(),"failed to load LWIR camera calibration parameters");
                if(this->m_bHorizRectify) {
                    lvAssert_(!this->m_bLoadDepth,"missing depth image rectification impl"); // @@@
                    std::array<cv::Mat,2> aRectifRotMats,aRectifProjMats;
                    cv::Mat oDispToDepthMap,oRotMat,oTranslMat;
                    oParamsFS["oRotMat"] >> oRotMat;
                    oParamsFS["oTranslMat"] >> oTranslMat;
                    lvAssert_(!this->m_oLWIRCameraParams.empty() && !this->m_oLWIRDistortParams.empty(),"failed to load LWIR camera calibration parameters");
                    // @@@@ calib individual heads using cv::calibrateCamera, and recalc cam mats using 'getOptimalNewCameraMatrix' for common size (imageSize param)
                    cv::stereoRectify(this->m_oRGBCameraParams,this->m_oRGBDistortParams,
                                      this->m_oLWIRCameraParams,this->m_oLWIRDistortParams,
                                      oRectifSize,oRotMat,oTranslMat,
                                      aRectifRotMats[0],aRectifRotMats[1],
                                      aRectifProjMats[0],aRectifProjMats[1],
                                      oDispToDepthMap,
                                      0,//cv::CALIB_ZERO_DISPARITY,
                                      -1,oRectifSize);
                    cv::initUndistortRectifyMap(this->m_oRGBCameraParams,this->m_oRGBDistortParams,
                                                aRectifRotMats[0],aRectifProjMats[0],oRectifSize,
                                                CV_16SC2,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2);
                    cv::initUndistortRectifyMap(this->m_oLWIRCameraParams,this->m_oLWIRDistortParams,
                                                aRectifRotMats[1],aRectifProjMats[1],oRectifSize,
                                                CV_16SC2,this->m_oLWIRCalibMap1,this->m_oLWIRCalibMap2);
                }
                else {
                    const double dUndistortMapCameraMatrixAlpha = -1.0;
                    cv::initUndistortRectifyMap(this->m_oRGBCameraParams,this->m_oRGBDistortParams,cv::Mat(),
                                                (dUndistortMapCameraMatrixAlpha<0)?this->m_oRGBCameraParams:cv::getOptimalNewCameraMatrix(this->m_oRGBCameraParams,this->m_oRGBDistortParams,oRGBSize,dUndistortMapCameraMatrixAlpha,oRGBSize,0),
                                                oRGBSize,CV_16SC2,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2);
                    cv::initUndistortRectifyMap(this->m_oLWIRCameraParams,this->m_oLWIRDistortParams,cv::Mat(),
                                                (dUndistortMapCameraMatrixAlpha<0)?this->m_oLWIRCameraParams:cv::getOptimalNewCameraMatrix(this->m_oLWIRCameraParams,this->m_oLWIRDistortParams,oLWIRSize,dUndistortMapCameraMatrixAlpha,oLWIRSize,0),
                                                oLWIRSize,CV_16SC2,this->m_oLWIRCalibMap1,this->m_oLWIRCalibMap2);
                }
                //////////////
                cv::remap(oRGBROI.clone(),oRGBROI,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_LINEAR);
                cv::remap(oLWIRROI.clone(),oLWIRROI,this->m_oLWIRCalibMap1,this->m_oLWIRCalibMap2,cv::INTER_LINEAR);
                oRGBROI = oRGBROI>128;
                oLWIRROI = oLWIRROI>128;
                if(this->m_nLWIRDispOffset!=0)
                    lv::shift(oLWIRROI.clone(),oLWIRROI,cv::Point2f(0.0f,-float(this->m_nLWIRDispOffset)));
                cv::erode(oRGBROI,oRGBROI,cv::Mat(),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar_<uchar>(0));
                cv::erode(oLWIRROI,oLWIRROI,cv::Mat(),cv::Point(-1,-1),1,cv::BORDER_CONSTANT,cv::Scalar_<uchar>(0));
                cv::Mat oRGBROI_undist = cv::imread(this->getDataPath()+"rgb_undist_roi.png",cv::IMREAD_GRAYSCALE);
                cv::Mat oLWIRROI_undist = cv::imread(this->getDataPath()+"LWIR_undist_roi.png",cv::IMREAD_GRAYSCALE);
                if(!oRGBROI_undist.empty()) {
                    lvAssert(oRGBROI_undist.type()==CV_8UC1 && oRGBROI_undist.size()==oRGBSize);
                    oRGBROI_undist = oRGBROI_undist>128;
                }
                else
                    oRGBROI_undist = cv::Mat(oRGBSize,CV_8UC1,cv::Scalar_<uchar>(255));
                if(!oLWIRROI_undist.empty()) {
                    lvAssert(oLWIRROI_undist.type()==CV_8UC1 && oLWIRROI_undist.size()==oLWIRSize);
                    oLWIRROI_undist = oLWIRROI_undist>128;
                }
                else
                    oLWIRROI_undist = cv::Mat(oLWIRSize,CV_8UC1,cv::Scalar_<uchar>(255));
                oRGBROI &= oRGBROI_undist;
                oLWIRROI &= oLWIRROI_undist;
            }
            ////////////////////////////////
            this->m_vOrigGTInfos.resize(nGTStreamCount);
            this->m_vOrigInputInfos.resize(nInputStreamCount);
            this->m_vOrigGTInfos[nGTRGBStreamIdx] = lv::MatInfo{oRGBROI};
            this->m_vOrigGTInfos[nGTLWIRStreamIdx] = lv::MatInfo{oLWIRROI};
            this->m_vOrigInputInfos[nInputRGBStreamIdx] = lv::MatInfo{oRGBROI.size(),CV_8UC3};
            this->m_vOrigInputInfos[nInputLWIRStreamIdx] = lv::MatInfo{oLWIRROI.size(),CV_8UC1};
            if(this->m_bLoadDepth) {
                this->m_vOrigGTInfos[nGTDepthStreamIdx] = lv::MatInfo{oDepthROI};
                this->m_vOrigInputInfos[nInputDepthStreamIdx] = lv::MatInfo{oDepthROI.size(),CV_16UC1};
            }
            if(bUseInterlacedMasks) {
                this->m_vOrigInputInfos[nInputRGBMaskStreamIdx] = lv::MatInfo{oRGBROI};
                this->m_vOrigInputInfos[nInputLWIRMaskStreamIdx] = lv::MatInfo{oLWIRROI};
                if(this->m_bLoadDepth)
                    this->m_vOrigInputInfos[nInputDepthMaskStreamIdx] = lv::MatInfo{oDepthROI};
            }
            if(dScale!=1.0) {
                cv::resize(oRGBROI,oRGBROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
                cv::resize(oLWIRROI,oLWIRROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
                cv::resize(oDepthROI,oDepthROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
            }
            this->m_vGTROIs.resize(nGTStreamCount);
            this->m_vGTInfos.resize(nGTStreamCount);
            this->m_vInputROIs.resize(nInputStreamCount);
            this->m_vInputInfos.resize(nInputStreamCount);
            this->m_vGTROIs[nGTRGBStreamIdx] = oRGBROI.clone();
            this->m_vGTROIs[nGTLWIRStreamIdx] = oLWIRROI.clone();
            this->m_vGTInfos[nGTRGBStreamIdx] = lv::MatInfo{oRGBROI};
            this->m_vGTInfos[nGTLWIRStreamIdx] = lv::MatInfo{oLWIRROI};
            this->m_vInputROIs[nInputRGBStreamIdx] = oRGBROI.clone();
            this->m_vInputROIs[nInputLWIRStreamIdx] = oLWIRROI.clone();
            this->m_vInputInfos[nInputRGBStreamIdx] = lv::MatInfo{oRGBROI.size(),CV_8UC3};
            this->m_vInputInfos[nInputLWIRStreamIdx] = lv::MatInfo{oLWIRROI.size(),CV_8UC1};
            if(this->m_bLoadDepth) {
                this->m_vGTROIs[nGTDepthStreamIdx] = oDepthROI.clone();
                this->m_vGTInfos[nGTDepthStreamIdx] = lv::MatInfo{oDepthROI};
                this->m_vInputROIs[nInputDepthStreamIdx] = oDepthROI.clone();
                this->m_vInputInfos[nInputDepthStreamIdx] = lv::MatInfo{oDepthROI.size(),CV_16UC1};
            }
            if(bUseInterlacedMasks) {
                this->m_vInputROIs[nInputRGBMaskStreamIdx] = oRGBROI.clone();
                this->m_vInputROIs[nInputLWIRMaskStreamIdx] = oLWIRROI.clone();
                this->m_vInputInfos[nInputRGBMaskStreamIdx] = lv::MatInfo{oRGBROI};
                this->m_vInputInfos[nInputLWIRMaskStreamIdx] = lv::MatInfo{oLWIRROI};
                if(this->m_bLoadDepth) {
                    this->m_vInputROIs[nInputDepthMaskStreamIdx] = oLWIRROI.clone();
                    this->m_vInputInfos[nInputDepthMaskStreamIdx] = lv::MatInfo{oDepthROI};
                }
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////
            if(vsRGBFramePaths.empty() || lv::MatInfo(cv::imread(vsRGBFramePaths[0],cv::IMREAD_COLOR))!=this->m_vOrigInputInfos[nInputRGBStreamIdx])
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected RGB frame packet size/type",this->getName().c_str());
            if(bUseInterlacedMasks && (vsRGBMaskPaths.empty() || lv::MatInfo(cv::imread(vsRGBMaskPaths[0],cv::IMREAD_GRAYSCALE))!=this->m_vOrigInputInfos[nInputRGBMaskStreamIdx]))
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected RGB mask packet size",this->getName().c_str());
            if(vsLWIRFramePaths.empty() || lv::MatInfo(cv::imread(vsLWIRFramePaths[0],cv::IMREAD_GRAYSCALE))!=this->m_vOrigInputInfos[nInputLWIRStreamIdx])
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected LWIR frame packet size/type",this->getName().c_str());
            if(bUseInterlacedMasks && (vsLWIRMaskPaths.empty() || lv::MatInfo(cv::imread(vsLWIRMaskPaths[0],cv::IMREAD_GRAYSCALE))!=this->m_vOrigInputInfos[nInputLWIRMaskStreamIdx]))
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected LWIR mask packet size",this->getName().c_str());
            if(this->m_bLoadDepth) {
                if(vsDepthFramePaths.empty() || lv::MatInfo(cv::imread(vsDepthFramePaths[0],cv::IMREAD_ANYDEPTH))!=this->m_vOrigInputInfos[nInputDepthStreamIdx])
                    lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected depth frame packet size/type",this->getName().c_str());
                if(bUseInterlacedMasks && (vsDepthMaskPaths.empty() || lv::MatInfo(cv::imread(vsDepthMaskPaths[0],cv::IMREAD_GRAYSCALE))!=this->m_vOrigInputInfos[nInputDepthMaskStreamIdx]))
                    lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected depth mask packet size",this->getName().c_str());
            }
            if(bLoadFrameSubset) {
                lSubsetCleaner(vsRGBFramePaths,this->m_mSubset);
                lSubsetCleaner(vsLWIRFramePaths,this->m_mSubset);
                if(this->m_bLoadDepth)
                    lSubsetCleaner(vsDepthFramePaths,this->m_mSubset);
                if(bUseInterlacedMasks) {
                    lSubsetCleaner(vsRGBMaskPaths,this->m_mSubset);
                    lSubsetCleaner(vsLWIRMaskPaths,this->m_mSubset);
                    if(this->m_bLoadDepth)
                        lSubsetCleaner(vsDepthMaskPaths,this->m_mSubset);
                }
            }
            if(!bIsLoadingCalibData) {
                lvAssert_(vsRGBFramePaths.size()==nInputPackets,"missing RGB frame indices");
                lvAssert_(vsLWIRFramePaths.size()==nInputPackets,"missing LWIR frame indices");
                if(this->m_bLoadDepth)
                    lvAssert_(vsDepthFramePaths.size()==nInputPackets,"missing depth frame indices");
                if(bUseInterlacedMasks) {
                    lvAssert_(vsRGBFramePaths.size()==vsRGBMaskPaths.size(),"bad RGB frame/mask index overlap");
                    lvAssert_(vsLWIRFramePaths.size()==vsLWIRMaskPaths.size(),"bad LWIR frame/mask index overlap");
                    if(this->m_bLoadDepth)
                        lvAssert_(vsDepthFramePaths.size()==vsDepthMaskPaths.size(),"bad depth frame/mask index overlap");
                }
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////
            if(!vsRGBGTPaths.empty() && lv::MatInfo(cv::imread(vsRGBGTPaths[0],cv::IMREAD_GRAYSCALE))!=this->m_vOrigGTInfos[nGTRGBStreamIdx])
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected RGB GT packet size/type",this->getName().c_str());
            if(!vsLWIRGTPaths.empty() && lv::MatInfo(cv::imread(vsLWIRGTPaths[0],cv::IMREAD_GRAYSCALE))!=this->m_vOrigGTInfos[nGTLWIRStreamIdx])
                lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected LWIR GT packet size/type",this->getName().c_str());
            if(this->m_bLoadDepth) {
                if(!vsDepthGTPaths.empty() && lv::MatInfo(cv::imread(vsDepthGTPaths[0],cv::IMREAD_GRAYSCALE))!=this->m_vOrigGTInfos[nGTDepthStreamIdx])
                    lvError_("LITIV-stcharles2018 sequence '%s' did not possess expected depth GT packet size/type",this->getName().c_str());
            }
            if(bLoadFrameSubset || bEvalOnlyFrameSubset) {
                lSubsetCleaner(vsRGBGTPaths,this->m_mRealSubset,false);
                lSubsetCleaner(vsLWIRGTPaths,this->m_mRealSubset,false);
                if(this->m_bLoadDepth)
                    lSubsetCleaner(vsDepthGTPaths,this->m_mRealSubset,false);
            }
            lvAssert(vsRGBGTPaths.size()<=nInputPackets);
            lvAssert(vsLWIRGTPaths.size()<=nInputPackets);
            if(this->m_bLoadDepth)
                lvAssert(vsDepthGTPaths.size()<=nInputPackets);
            //////////////////////////////////////////////////////////////////////////////////////////////////
            this->m_vvsInputPaths.resize(nInputPackets);
            for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx) {
                const size_t nOrigPacketIdx = *std::next(this->m_mSubset.begin(),nInputPacketIdx);
                this->m_vvsInputPaths[nInputPacketIdx].resize(nInputStreamCount);
                if(bIsLoadingCalibData) {
                    const auto lIndexMatcher = [&](const std::string& sFilePath) {
                        return sFilePath.find(lv::putf("%05d",(int)nOrigPacketIdx))!=std::string::npos;
                    };
                    auto pRGBFramePath = std::find_if(vsRGBFramePaths.begin(),vsRGBFramePaths.end(),lIndexMatcher);
                    if(pRGBFramePath!=vsRGBFramePaths.end())
                        this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx] = *pRGBFramePath;
                    else
                        this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx] = std::string();
                    auto pLWIRFramePath = std::find_if(vsLWIRFramePaths.begin(),vsLWIRFramePaths.end(),lIndexMatcher);
                    if(pLWIRFramePath!=vsLWIRFramePaths.end())
                        this->m_vvsInputPaths[nInputPacketIdx][nInputLWIRStreamIdx] = *pLWIRFramePath;
                    else
                        this->m_vvsInputPaths[nInputPacketIdx][nInputLWIRStreamIdx] = std::string();
                }
                else {
                    lvAssert(lIndexExtractor(vsRGBFramePaths[nInputPacketIdx])==nOrigPacketIdx);
                    this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx] = vsRGBFramePaths[nInputPacketIdx];
                    //const size_t nLastInputSlashPos = this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx].find_last_of("/\\");
                    //const std::string sInputFileNameExt = nLastInputSlashPos==std::string::npos?this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx]:this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx].substr(nLastInputSlashPos+1);
                    //cv::imwrite(this->getDataPath()+"rgb_subset/"+sInputFileNameExt,cv::imread(this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx]));
                    lvAssert(lIndexExtractor(vsLWIRFramePaths[nInputPacketIdx])==nOrigPacketIdx);
                    this->m_vvsInputPaths[nInputPacketIdx][nInputLWIRStreamIdx] = vsLWIRFramePaths[nInputPacketIdx];
                    if(this->m_bLoadDepth) {
                        lvAssert(lIndexExtractor(vsDepthFramePaths[nInputPacketIdx])==nOrigPacketIdx);
                        this->m_vvsInputPaths[nInputPacketIdx][nInputDepthStreamIdx] = vsDepthFramePaths[nInputPacketIdx];
                    }
                    if(bUseInterlacedMasks) {
                        lvAssert(lIndexExtractor(vsRGBMaskPaths[nInputPacketIdx])==nOrigPacketIdx);
                        this->m_vvsInputPaths[nInputPacketIdx][nInputRGBMaskStreamIdx] = vsRGBMaskPaths[nInputPacketIdx];
                        lvAssert(lIndexExtractor(vsLWIRMaskPaths[nInputPacketIdx])==nOrigPacketIdx);
                        this->m_vvsInputPaths[nInputPacketIdx][nInputLWIRMaskStreamIdx] = vsLWIRMaskPaths[nInputPacketIdx];
                        if(this->m_bLoadDepth) {
                            lvAssert(lIndexExtractor(vsDepthMaskPaths[nInputPacketIdx])==nOrigPacketIdx);
                            this->m_vvsInputPaths[nInputPacketIdx][nInputDepthMaskStreamIdx] = vsDepthMaskPaths[nInputPacketIdx];
                        }
                    }
                }
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////
            this->m_vvsGTPaths = std::vector<std::vector<std::string>>(nInputPackets);
            this->m_mGTIndexLUT.clear();
            for(size_t nGTPacketIdx=0; nGTPacketIdx<nInputPackets; ++nGTPacketIdx) {
                this->m_vvsGTPaths[nGTPacketIdx].resize(nGTStreamCount);
                const auto lIndexMatcher = [&](const std::string& sFilePath) {
                    return sFilePath.find(lv::putf("%05d",(int)nGTPacketIdx))!=std::string::npos;
                };
                auto pRGBGTPath = std::find_if(vsRGBGTPaths.begin(),vsRGBGTPaths.end(),lIndexMatcher);
                if(pRGBGTPath!=vsRGBGTPaths.end())
                    this->m_vvsGTPaths[nGTPacketIdx][nGTRGBStreamIdx] = *pRGBGTPath;
                auto pLWIRGTPath = std::find_if(vsLWIRGTPaths.begin(),vsLWIRGTPaths.end(),lIndexMatcher);
                if(pLWIRGTPath!=vsLWIRGTPaths.end())
                    this->m_vvsGTPaths[nGTPacketIdx][nGTLWIRStreamIdx] = *pLWIRGTPath;
                if(this->m_bLoadDepth) {
                    auto pDepthGTPath = std::find_if(vsDepthGTPaths.begin(),vsDepthGTPaths.end(),lIndexMatcher);
                    if(pDepthGTPath!=vsDepthGTPaths.end())
                        this->m_vvsGTPaths[nGTPacketIdx][nGTDepthStreamIdx] = *pDepthGTPath;
                }
                this->m_mGTIndexLUT[nGTPacketIdx] = nGTPacketIdx; // direct gt path index to frame index mapping
            }
            //////////////////////////////////////////////////////////////////////////////////////////////////
            if(bFlipDisparities) {
                for(size_t nStreamIdx=0; nStreamIdx<this->getInputStreamCount(); ++nStreamIdx)
                    cv::flip(this->m_vInputROIs[nStreamIdx],this->m_vInputROIs[nStreamIdx],1);
                for(size_t nStreamIdx=0; nStreamIdx<this->getGTStreamCount(); ++nStreamIdx)
                    cv::flip(this->m_vGTROIs[nStreamIdx],this->m_vGTROIs[nStreamIdx],1);
            }
        }
        virtual std::vector<cv::Mat> getRawInputArray(size_t nPacketIdx) override final {
            lvDbgExceptionWatch;
            if(nPacketIdx>=this->m_vvsInputPaths.size())
                return std::vector<cv::Mat>(this->getInputStreamCount());
            const cv::Size oRGBSize(1920,1080),oLWIRSize(320,240),oDepthSize(512,424);
            const bool bIsLoadingCalibData = bool(DATASETS_LITIV2018_LOAD_CALIB_DATA);
            const bool bUseInterlacedMasks = this->m_nLoadInputMasks!=0;
            const bool bFlipDisparities = this->isFlippingDisparities();
            const size_t nInputRGBStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_RGB*(bUseInterlacedMasks?2:1));
            const size_t nInputLWIRStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_LWIR*(bUseInterlacedMasks?2:1));
            const size_t nInputDepthStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_Depth*(bUseInterlacedMasks?2:1));
            const size_t nInputRGBMaskStreamIdx = nInputRGBStreamIdx+1;
            const size_t nInputLWIRMaskStreamIdx = nInputLWIRStreamIdx+1;
            const size_t nInputDepthMaskStreamIdx = nInputDepthStreamIdx+1;
            const std::vector<std::string>& vsInputPaths = this->m_vvsInputPaths[nPacketIdx];
            lvDbgAssert(!vsInputPaths.empty() && vsInputPaths.size()==this->getInputStreamCount());
            const std::vector<lv::MatInfo>& vInputInfos = this->m_vInputInfos;
            lvDbgAssert(!vInputInfos.empty() && vInputInfos.size()==getInputStreamCount());
            std::vector<cv::Mat> vInputs(getInputStreamCount());
            ///////////////////////////////////////////////////////////////////////////////////
            lvAssert__(bIsLoadingCalibData || !vsInputPaths[nInputRGBStreamIdx].empty(),"could not open RGB input frame #%d (empty path)",(int)nPacketIdx);
            cv::Mat oRGBPacket = vsInputPaths[nInputRGBStreamIdx].empty()?cv::Mat(oRGBSize,CV_8UC3,cv::Scalar::all(0)):cv::imread(vsInputPaths[nInputRGBStreamIdx],cv::IMREAD_COLOR);
            lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC3 && oRGBPacket.size()==oRGBSize);
        #if DATASETS_LITIV2018_FLIP_RGB
            cv::flip(oRGBPacket,oRGBPacket,1); // must pre-flip rgb frames due to original camera flip
        #endif //DATASETS_LITIV2018_FLIP_RGB
            if(oRGBPacket.size()!=vInputInfos[nInputRGBStreamIdx].size())
                cv::resize(oRGBPacket,oRGBPacket,vInputInfos[nInputRGBStreamIdx].size(),0,0,cv::INTER_CUBIC);
            if(this->m_bUndistort || this->m_bHorizRectify)
                cv::remap(oRGBPacket.clone(),oRGBPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_CUBIC);
            vInputs[nInputRGBStreamIdx] = oRGBPacket;
            if(bUseInterlacedMasks) {
                cv::Mat oRGBMaskPacket = cv::imread(vsInputPaths[nInputRGBMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oRGBMaskPacket.empty() && oRGBMaskPacket.type()==CV_8UC1 && oRGBMaskPacket.size()==oRGBSize);
                if(oRGBMaskPacket.size()!=vInputInfos[nInputRGBStreamIdx].size())
                    cv::resize(oRGBMaskPacket,oRGBMaskPacket,vInputInfos[nInputRGBStreamIdx].size(),cv::INTER_LINEAR);
                if(this->m_bUndistort || this->m_bHorizRectify)
                    cv::remap(oRGBMaskPacket.clone(),oRGBMaskPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_LINEAR);
                oRGBMaskPacket = oRGBMaskPacket>128;
                vInputs[nInputRGBMaskStreamIdx] = oRGBMaskPacket;
            }
            ///////////////////////////////////////////////////////////////////////////////////
            lvAssert__(bIsLoadingCalibData || !vsInputPaths[nInputLWIRStreamIdx].empty(),"could not open LWIR input frame #%d (empty path)",(int)nPacketIdx);
            cv::Mat oLWIRPacket = vsInputPaths[nInputLWIRStreamIdx].empty()?cv::Mat(oLWIRSize,CV_8UC1,cv::Scalar::all(0)):cv::imread(vsInputPaths[nInputLWIRStreamIdx],cv::IMREAD_GRAYSCALE);
            lvAssert(!oLWIRPacket.empty() && oLWIRPacket.type()==CV_8UC1 && oLWIRPacket.size()==oLWIRSize);
            if(oLWIRPacket.size()!=vInputInfos[nInputLWIRStreamIdx].size())
                cv::resize(oLWIRPacket,oLWIRPacket,vInputInfos[nInputLWIRStreamIdx].size(),0,0,cv::INTER_CUBIC);
            if(this->m_bUndistort || this->m_bHorizRectify) {
                cv::remap(oLWIRPacket.clone(),oLWIRPacket,this->m_oLWIRCalibMap1,this->m_oLWIRCalibMap2,cv::INTER_CUBIC);
                if(this->m_nLWIRDispOffset!=0)
                    lv::shift(oLWIRPacket.clone(),oLWIRPacket,cv::Point2f(0.0f,-float(this->m_nLWIRDispOffset)));
            }
            vInputs[nInputLWIRStreamIdx] = oLWIRPacket;
            if(bUseInterlacedMasks) {
                cv::Mat oLWIRMaskPacket = cv::imread(vsInputPaths[nInputLWIRMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oLWIRMaskPacket.empty() && oLWIRMaskPacket.type()==CV_8UC1 && oLWIRMaskPacket.size()==oLWIRSize);
                if(oLWIRMaskPacket.size()!=vInputInfos[nInputLWIRStreamIdx].size())
                    cv::resize(oLWIRMaskPacket,oLWIRMaskPacket,vInputInfos[nInputLWIRStreamIdx].size(),cv::INTER_LINEAR);
                if(this->m_bUndistort || this->m_bHorizRectify) {
                    cv::remap(oLWIRMaskPacket.clone(),oLWIRMaskPacket,this->m_oLWIRCalibMap1,this->m_oLWIRCalibMap2,cv::INTER_LINEAR);
                    if(this->m_nLWIRDispOffset!=0)
                        lv::shift(oLWIRMaskPacket.clone(),oLWIRMaskPacket,cv::Point2f(0.0f,-float(this->m_nLWIRDispOffset)));
                }
                oLWIRMaskPacket = oLWIRMaskPacket>128;
                vInputs[nInputLWIRMaskStreamIdx] = oLWIRMaskPacket;
            }
            ///////////////////////////////////////////////////////////////////////////////////
            if(this->m_bLoadDepth) {
                cv::Mat oDepthPacket = cv::imread(vsInputPaths[nInputDepthStreamIdx],cv::IMREAD_ANYDEPTH);
                lvAssert(!oDepthPacket.empty() && oDepthPacket.type()==CV_16UC1 && oDepthPacket.size()==oDepthSize);
                if(oDepthPacket.size()!=vInputInfos[nInputDepthStreamIdx].size())
                    cv::resize(oDepthPacket,oDepthPacket,vInputInfos[nInputDepthStreamIdx].size(),0,0,cv::INTER_CUBIC);
                // depth should be already undistorted
                lvAssert_(!this->m_bHorizRectify,"missing depth image rectification impl");
                vInputs[nInputDepthStreamIdx] = oDepthPacket;
                if(bUseInterlacedMasks) {
                    cv::Mat oDepthMaskPacket = cv::imread(vsInputPaths[nInputDepthMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oDepthMaskPacket.empty() && oDepthMaskPacket.type()==CV_8UC1 && oDepthMaskPacket.size()==oDepthSize);
                    if(oDepthMaskPacket.size()!=vInputInfos[nInputDepthStreamIdx].size())
                        cv::resize(oDepthMaskPacket,oDepthMaskPacket,vInputInfos[nInputDepthStreamIdx].size(),cv::INTER_LINEAR);
                    lvAssert_(!this->m_bHorizRectify,"missing depth image rectification impl");
                    oDepthMaskPacket = oDepthMaskPacket>128;
                    vInputs[nInputDepthMaskStreamIdx] = oDepthMaskPacket;
                }
            }
            if(bFlipDisparities)
                for(size_t nInputStreamIdx=0; nInputStreamIdx<this->getInputStreamCount(); ++nInputStreamIdx)
                    if(!vInputs[nInputStreamIdx].empty())
                        cv::flip(vInputs[nInputStreamIdx],vInputs[nInputStreamIdx],1);
            return vInputs;
        }
        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) override final {
            lvDbgExceptionWatch;
            const cv::Size oRGBSize(1920,1080),oLWIRSize(320,240),oDepthSize(512,424);
            const bool bFlipDisparities = this->isFlippingDisparities();
            const bool bEvalDisparityMaps = this->isEvaluatingDisparities();
            const size_t nGTRGBStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_RGB);
            const size_t nGTLWIRStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_LWIR);
            const size_t nGTDepthStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_Depth);
            std::vector<cv::Mat> vGTs(getGTStreamCount());
            if(this->m_mGTIndexLUT.count(nPacketIdx)) {
                const size_t nGTIdx = this->m_mGTIndexLUT[nPacketIdx];
                lvDbgAssert(nGTIdx<this->m_vvsGTPaths.size());
                const std::vector<std::string>& vsGTMasksPaths = this->m_vvsGTPaths[nGTIdx];
                lvDbgAssert(!vsGTMasksPaths.empty() && vsGTMasksPaths.size()==getGTStreamCount());
                const std::vector<lv::MatInfo>& vGTInfos = this->m_vGTInfos;
                lvDbgAssert(!vGTInfos.empty() && vGTInfos.size()==getGTStreamCount());
                cv::Mat oRGBPacket = cv::imread(vsGTMasksPaths[nGTRGBStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC1 && oRGBPacket.size()==oRGBSize);
                cv::Mat oLWIRPacket = cv::imread(vsGTMasksPaths[nGTLWIRStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oLWIRPacket.empty() && oLWIRPacket.type()==CV_8UC1 && oLWIRPacket.size()==oLWIRSize);
                cv::Mat oDepthPacket;
                if(this->m_bLoadDepth) {
                    oDepthPacket = cv::imread(vsGTMasksPaths[nGTDepthStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oDepthPacket.empty() && oDepthPacket.type()==CV_8UC1 && oDepthPacket.size()==oDepthSize);
                }
                if(bEvalDisparityMaps) {
                    // assume loaded gt packets are already properly rectified
                    if(oRGBPacket.size()!=vGTInfos[nGTLWIRStreamIdx].size()) {
                        cv::resize(oRGBPacket,oRGBPacket,vGTInfos[nGTLWIRStreamIdx].size(),0,0,cv::INTER_NEAREST);
                        cv::Mat oOldDontCareMask = (oRGBPacket==255);
                        oRGBPacket *= this->getScaleFactor();
                        cv::Mat_<uchar>(oRGBPacket.size(),uchar(255)).copyTo(oRGBPacket,oOldDontCareMask);
                    }
                    if(oLWIRPacket.size()!=vGTInfos[nGTLWIRStreamIdx].size()) {
                        cv::resize(oLWIRPacket,oLWIRPacket,vGTInfos[nGTLWIRStreamIdx].size(),0,0,cv::INTER_NEAREST);
                        cv::Mat oOldDontCareMask = (oLWIRPacket==255);
                        oLWIRPacket *= this->getScaleFactor();
                        cv::Mat_<uchar>(oLWIRPacket.size(),uchar(255)).copyTo(oLWIRPacket,oOldDontCareMask);
                    }
                    if(this->m_bLoadDepth && oDepthPacket.size()!=vGTInfos[nGTLWIRStreamIdx].size()) {
                        cv::resize(oDepthPacket,oDepthPacket,vGTInfos[nGTLWIRStreamIdx].size(),0,0,cv::INTER_NEAREST);
                        cv::Mat oOldDontCareMask = (oDepthPacket==255);
                        oDepthPacket *= this->getScaleFactor();
                        cv::Mat_<uchar>(oDepthPacket.size(),uchar(255)).copyTo(oDepthPacket,oOldDontCareMask);
                    }
                }
                else {
                #if DATASETS_LITIV2018_FLIP_RGB
                    cv::flip(oRGBPacket,oRGBPacket,1); // must pre-flip rgb frames due to original camera flip
                #endif //DATASETS_LITIV2018_FLIP_RGB
                    if(oRGBPacket.size()!=vGTInfos[nGTRGBStreamIdx].size())
                        cv::resize(oRGBPacket,oRGBPacket,vGTInfos[nGTRGBStreamIdx].size(),0,0,cv::INTER_LINEAR);
                    if(this->m_bUndistort || this->m_bHorizRectify)
                        cv::remap(oRGBPacket.clone(),oRGBPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_LINEAR);
                    oRGBPacket = oRGBPacket>128;
                    if(oLWIRPacket.size()!=vGTInfos[nGTLWIRStreamIdx].size())
                        cv::resize(oLWIRPacket,oLWIRPacket,vGTInfos[nGTLWIRStreamIdx].size(),0,0,cv::INTER_LINEAR);
                    if(this->m_bUndistort || this->m_bHorizRectify) {
                        cv::remap(oLWIRPacket.clone(),oLWIRPacket,this->m_oLWIRCalibMap1,this->m_oLWIRCalibMap2,cv::INTER_LINEAR);
                        if(this->m_nLWIRDispOffset!=0)
                            lv::shift(oLWIRPacket.clone(),oLWIRPacket,cv::Point2f(0.0f,-float(this->m_nLWIRDispOffset)));
                    }
                    oLWIRPacket = oLWIRPacket>128;
                    if(this->m_bLoadDepth) {
                        if(oDepthPacket.size()!=vGTInfos[nGTDepthStreamIdx].size())
                            cv::resize(oDepthPacket,oDepthPacket,vGTInfos[nGTDepthStreamIdx].size(),0,0,cv::INTER_LINEAR);
                        // depth should be already undistorted
                        lvAssert_(!this->m_bHorizRectify,"missing depth image rectification impl");
                        oDepthPacket = oDepthPacket>128;
                    }
                }
                vGTs[nGTRGBStreamIdx] = oRGBPacket;
                vGTs[nGTLWIRStreamIdx] = oLWIRPacket;
                if(this->m_bLoadDepth)
                    vGTs[nGTDepthStreamIdx] = oDepthPacket;
                if(bFlipDisparities)
                    for(size_t nGTStreamIdx=0; nGTStreamIdx<this->getGTStreamCount(); ++nGTStreamIdx)
                        if(!vGTs[nGTStreamIdx].empty())
                            cv::flip(vGTs[nGTStreamIdx],vGTs[nGTStreamIdx],1);
            }
            return vGTs;
        }
        std::set<size_t> m_mRealSubset,m_mSubset;
        bool m_bLoadDepth,m_bUndistort,m_bHorizRectify;
        int m_nLoadInputMasks;
        int m_nLWIRDispOffset;
        size_t m_nMinDisp,m_nMaxDisp;
        std::string m_sFeaturesDirName;
        cv::Mat m_oRGBCameraParams;
        cv::Mat m_oLWIRCameraParams;
        cv::Mat m_oRGBDistortParams;
        cv::Mat m_oLWIRDistortParams;
        cv::Mat m_oRGBCalibMap1,m_oRGBCalibMap2;
        cv::Mat m_oLWIRCalibMap1,m_oLWIRCalibMap2;
        std::vector<lv::MatInfo> m_vOrigInputInfos,m_vOrigGTInfos;
    public:
    #if DATASETS_LITIV2018_LOAD_CALIB_DATA
        std::vector<bool> isCalibInputValid(size_t nPacketIdx) {
            lvDbgExceptionWatch;
            if(nPacketIdx>=this->m_vvsInputPaths.size())
                return std::vector<bool>(this->getInputStreamCount(),false);
            const bool bUseInterlacedMasks = this->m_nLoadInputMasks!=0;
            const size_t nInputRGBStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_RGB*(bUseInterlacedMasks?2:1));
            const size_t nInputLWIRStreamIdx = size_t(ILITIVStCharles2018Dataset::LITIV2018_LWIR*(bUseInterlacedMasks?2:1));
            const std::vector<std::string>& vsInputPaths = this->m_vvsInputPaths[nPacketIdx];
            lvDbgAssert(!vsInputPaths.empty() && vsInputPaths.size()==this->getInputStreamCount());
            const std::vector<lv::MatInfo>& vInputInfos = this->m_vInputInfos;
            lvDbgAssert(!vInputInfos.empty() && vInputInfos.size()==getInputStreamCount());
            std::vector<bool> vValid(getInputStreamCount(),false);
            vValid[nInputRGBStreamIdx] = !(vsInputPaths[nInputRGBStreamIdx].empty()?cv::Mat():cv::imread(vsInputPaths[nInputRGBStreamIdx])).empty();
            vValid[nInputLWIRStreamIdx] = !(vsInputPaths[nInputLWIRStreamIdx].empty()?cv::Mat():cv::imread(vsInputPaths[nInputLWIRStreamIdx])).empty();
            return vValid;
        }
    #endif //DATASETS_LITIV2018_LOAD_CALIB_DATA
    };

} // namespace lv
