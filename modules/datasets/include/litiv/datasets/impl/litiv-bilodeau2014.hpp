
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

namespace lv {

    /// parameter interface for LITIV stereo registration dataset loader impl
    struct ILITIVBilodeau2014Dataset {
        /// returns whether the implementation is loading all video frames via avi's instead of specific stills via jpg's
        virtual bool isLoadingFullVideos() const = 0;
        /// returns whether we should evaluate fg/bg segmentation or stereo disparity
        virtual bool isEvaluatingStereoDisp() const = 0;
        /// returns which 'person' sets will be loaded as work batches (-1=all sets, (1<<(X-1))=load set 'XPerson')
        virtual int isLoadingPersonSets() const = 0;
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        virtual int isLoadingInputMasks() const = 0;
        /// returns whether a specific person variable (1/2/3/4/5) is set in a given flag
        static bool checkPersonSetInFlag(int nPersonFlag, int nPersonId) {
            lvAssert_(nPersonId>=1 && nPersonId<=5,"person set id only goes from 1 to 5");
            return bool(nPersonFlag&(1<<(nPersonId-1)));
        }
    };

    /// dataset loader impl specialization for LITIV stereo registration dataset -- instantiated via lv::datasets::create(...)
    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_LITIV_bilodeau2014,eEvalImpl> :
            public ILITIVBilodeau2014Dataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_bilodeau2014,lv::getDatasetEval<eDatasetTask,Dataset_LITIV_bilodeau2014>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via lv::datasets::create
        /// specialization constructor, with all required extra parameters; these will be forwarded by lv::datasets::create(...)
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in root directory's results folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=false, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bLoadFullVideos=false, ///< defines whether we should load full videos via avi's instead of specific stills via jpg's
                bool bEvalStereoDisp=true, ///< defines whether we should evaluate fg/bg segmentation or stereo disparity
                int nLoadPersonSets=-1, ///< defines which 'person' sets will be loaded as work batches (-1=all sets, (1<<(X))=load set 'XPerson')
                int nLoadInputMasks=0, ///< defines whether the input stream should be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
                double dScaleFactor=1.0 ///< defines the scale factor to use to resize/rescale read packets
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_bilodeau2014,lv::getDatasetEval<eDatasetTask,Dataset_LITIV_bilodeau2014>(),eEvalImpl>(
                        "LITIV-bilodeau2014",
                        lv::datasets::getRootPath()+"litiv/bilodeau2014/",
                        lv::datasets::getRootPath()+"litiv/bilodeau2014/results/"+lv::addDirSlashIfMissing(sOutputDirName),
                        getWorkBatchDirNames(nLoadPersonSets),
                        std::vector<std::string>(),
                        bSaveOutput,
                        bUseEvaluator,
                        false,
                        dScaleFactor
                ),
                m_bLoadFullVideos(bLoadFullVideos),
                m_bEvalStereoDisp(bEvalStereoDisp),
                m_nLoadPersonSets(nLoadPersonSets),
                m_nLoadInputMasks(nLoadInputMasks) {
            lvAssert_(nLoadPersonSets!=0,"must load at least one person set");
        }
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames(int nLoadPersonSets) {
            // dataset contains 3 videos with GT, cut into different sequences
            std::vector<std::string> vsWorkBatchDirs;
            if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,1) ||
               ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,2) ||
               ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,3) ||
               ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,4))
                vsWorkBatchDirs.push_back("vid1");
            if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,1) ||
               ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,2))
                vsWorkBatchDirs.push_back("vid2/cut1");
            if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,2) ||
               ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,3) ||
               ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nLoadPersonSets,4))
                vsWorkBatchDirs.push_back("vid2/cut2");
            vsWorkBatchDirs.push_back("vid3");
            return vsWorkBatchDirs;
        }
        /// returns whether the implementation is loading all video frames via avi's instead of specific stills via jpg's
        virtual bool isLoadingFullVideos() const override {return m_bLoadFullVideos;}
        /// returns whether we should evaluate fg/bg segmentation or stereo disparity
        virtual bool isEvaluatingStereoDisp() const override {return m_bEvalStereoDisp;}
        /// returns which 'person' sets will be loaded as work batches (-1=all sets, (1<<(X))=load set 'XPerson')
        virtual int isLoadingPersonSets() const override {return m_nLoadPersonSets;}
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        virtual int isLoadingInputMasks() const override {return m_nLoadInputMasks;}
    protected:
        const bool m_bLoadFullVideos;
        const bool m_bEvalStereoDisp;
        const int m_nLoadPersonSets;
        const int m_nLoadInputMasks;
    };

    /// data grouper handler impl specialization for litiv stereo registration dataset; will skip groups entirely & forward data to batches
    template<DatasetTaskList eDatasetTask>
    struct DataGroupHandler_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_bilodeau2014> :
            public DataGroupHandler {
    protected:
        virtual void parseData() override {
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            this->m_vpBatches.clear();
            this->m_bIsBare = true;
            if(!lv::string_contains_token(this->getName(),this->getSkipTokens())) {
                const ILITIVBilodeau2014Dataset& oDataset = dynamic_cast<const ILITIVBilodeau2014Dataset&>(*this->getRoot());
                const bool bLoadingFullVideos = oDataset.isLoadingFullVideos();
                if(bLoadingFullVideos)
                    m_vpBatches.push_back(createWorkBatch(getName(),getRelativePath()));
                else {
                    const int nPersonSetsFlag = oDataset.isLoadingPersonSets();
                    std::vector<std::string> vsWorkBatchPaths;
                    if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nPersonSetsFlag,1))
                        vsWorkBatchPaths.push_back("1Person");
                    if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nPersonSetsFlag,2))
                        vsWorkBatchPaths.push_back("2Person");
                    if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nPersonSetsFlag,3))
                        vsWorkBatchPaths.push_back("3Person");
                    if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nPersonSetsFlag,4))
                        vsWorkBatchPaths.push_back("4Person");
                    if(ILITIVBilodeau2014Dataset::checkPersonSetInFlag(nPersonSetsFlag,5))
                        vsWorkBatchPaths.push_back("5Person");
                    for(const auto& sPathIter : vsWorkBatchPaths) {
                        const std::string sNewBatchName = this->getName()+"/"sPathIter;
                        if(lv::checkIfExists(this->getDataPath()+sNewBatchName))
                            m_vpBatches.push_back(createWorkBatch(sNewBatchName,getRelativePath()+lv::addDirSlashIfMissing(sNewBatchName)));
                    }
                    this->m_bIsBare = m_vpBatches.empty();
                }
            }
        }
    };

    /// data producer impl specialization for litiv stereo registration dataset; provides required data i/o implementations
    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_bilodeau2014> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_VideoArray,Dataset_LITIV_bilodeau2014> {
        /// returns the number of parallel input streams (depends on whether loading masks or not)
        virtual size_t getInputStreamCount() const override final {
            return size_t(2*(m_nLoadInputMasks?2:1));
        }
        /// returns the number of parallel gt streams (always 2, disparity/segm masks)
        virtual size_t getGTStreamCount() const override final {
            return size_t(2);
        }
        /// returns the (friendly) name of an input stream specified by index
        virtual std::string getInputStreamName(size_t nStreamIdx) const override final {
            lvAssert(nStreamIdx<getInputStreamCount());
            if(m_nLoadInputMasks) {
                const std::string sBaseName = (((nStreamIdx/2)==0)?"RGB":((nStreamIdx/2)==1)?"THERMAL":"UNKNOWN");
                return sBaseName+std::string((nStreamIdx%2)?"_MASK":"");
            }
            else
                return ((nStreamIdx==0)?"RGB":(nStreamIdx==1)?"THERMAL":"UNKNOWN");
        }
        /// returns the (friendly) name of a gt stream specified by index
        virtual std::string getGTStreamName(size_t nStreamIdx) const override final {
            lvAssert(nStreamIdx<getGTStreamCount());
            const std::string sBaseName = ((nStreamIdx==0)?"RGB":(nStreamIdx==1)?"THERMAL":"UNKNOWN");
            return sBaseName+"_GT";
        }
        /// returns whether the implementation is loading all video frames via avi's instead of specific stills via jpg's
        bool isLoadingFullVideos() const {
            return dynamic_cast<const ILITIVBilodeau2014Dataset&>(*this->getRoot()).isLoadingFullVideos();
        }
        /// returns whether we should evaluate fg/bg segmentation or stereo disparity
        bool isEvaluatingStereoDisp() const {
            return dynamic_cast<const ILITIVBilodeau2014Dataset&>(*this->getRoot()).isEvaluatingStereoDisp();
        }
        /// returns which 'person' sets will be loaded as work batches (-1=all sets, (1<<(X-1))=load set 'XPerson')
        int isLoadingPersonSets() const {
            return dynamic_cast<const ILITIVBilodeau2014Dataset&>(*this->getRoot()).isLoadingPersonSets();
        }
        /// returns whether the input stream will be interlaced with fg/bg masks (0=no interlacing masks, -1=all gt masks, 1=all approx masks, (1<<(X+1))=gt mask for stream 'X')
        bool isLoadingInputMasks() const {
            return dynamic_cast<const ILITIVBilodeau2014Dataset&>(*this->getRoot()).isLoadingInputMasks();
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

    protected:
        virtual void parseData() override final {
            // note: this function is called right after the constructor, so initialize everything for other calls here
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            const ILITIVBilodeau2014Dataset& oDataset = dynamic_cast<const ILITIVBilodeau2014Dataset&>(*this->getRoot());
            this->m_bLoadFullVideos = oDataset.isLoadingFullVideos();
            this->m_nLoadInputMasks = oDataset.isLoadingInputMasks();
            this->m_bEvalStereoDisp = oDataset.isEvaluatingStereoDisp();
            const bool bUseInterlacedMasks = this->m_nLoadInputMasks!=0;
            const bool bUseApproxRGBMask = (this->m_nLoadInputMasks&2)==0;
            const bool bUseApproxThermalMask = (this->m_nLoadInputMasks&4)==0;
            const size_t nInputStreamCount = this->getInputStreamCount();
            const size_t nGTStreamCount = this->getGTStreamCount();
            constexpr size_t nInputRGBStreamIdx = 0;
            const size_t nInputThermalStreamIdx = bUseInterlacedMasks?2:1;
            constexpr size_t nInputRGBMaskStreamIdx = 1;
            constexpr size_t nInputThermalMaskStreamIdx = 3;
            constexpr size_t nGTRGBMaskStreamIdx = 0;
            constexpr size_t nGTThermalMaskStreamIdx = 1;
            const std::vector<std::string> vsSubDirs = lv::getSubDirsFromDir(this->getDataPath());
            auto psDisparityMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"IRDisparitymap");
            auto psGTMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"IRForegroundmap");
            auto psApproxMasksDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"Foreground");
            auto psInputDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"videoFrames");
            if((psDisparityMasksDir==vsSubDirs.end()) || (psApproxMasksDir==vsSubDirs.end()) || (psInputDir==vsSubDirs.end()))
                lvError_("LITIV-bilodeau2014 sequence '%s' did not possess the required input directories",this->getName().c_str());
            this->m_vInputROIs.resize(nInputStreamCount);
            this->m_vGTROIs.resize(nGTStreamCount);
            this->m_vInputInfos.resize(nInputStreamCount);
            this->m_vGTInfos.resize(nGTStreamCount);
            const cv::Size oImageSize(480,360);
            cv::Mat oGlobalROI(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
            const double dScale = this->getScaleFactor();
            if(dScale!=1.0)
                cv::resize(oGlobalROI,oGlobalROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
            for(size_t nStreamIdx=0; nStreamIdx<nInputStreamCount; ++nStreamIdx)
                this->m_vInputInfos[nStreamIdx] = lv::MatInfo{oGlobalROI.size(),(nStreamIdx==nInputRGBStreamIdx?CV_8UC3:CV_8UC1)};
            for(size_t nStreamIdx=0; nStreamIdx<nGTStreamCount; ++nStreamIdx)
                this->m_vGTInfos[nStreamIdx] = lv::MatInfo{oGlobalROI.size(),CV_8UC1};
            if(this->m_bLoadFullVideos) {
                lvAssert(false); // missing impl @@@@
                // need to add video reader, override getInputCount, ...
            }
            else {
                const std::vector<std::string> vsInputPaths = lv::getFilesFromDir(*psInputDir);
                const std::vector<std::string> vsApproxMasksPaths = lv::getFilesFromDir(*psApproxMasksDir);
                const std::vector<std::string> vsDisparityMasksPaths = lv::getFilesFromDir(*psApproxMasksDir);
                const std::vector<std::string> vsGTMasksPaths = lv::getFilesFromDir(*psApproxMasksDir);
                //////////////////////////////////////////////////////////////////////////////////////////
                std::vector<std::string> vsThermalInputPaths = vsInputPaths;
                lv::filterFilePaths(vsThermalInputPaths,{},{"IR"});
                if(vsThermalInputPaths.empty() || cv::imread(vsThermalInputPaths[0]).size()!=oImageSize)
                    lvError_("LITIV-bilodeau2014 sequence '%s' did not possess expected thermal input data",this->getName().c_str());
                for(size_t nInputPacketIdx=0; nInputPacketIdx<vsThermalInputPaths.size(); ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][nInputThermalStreamIdx] = vsThermalInputPaths[nInputPacketIdx];
                const size_t nInputPackets = vsThermalInputPaths.size();
                this->m_vvsInputPaths.resize(nInputPackets);
                std::vector<std::string> vsTempInputFileNames(nInputPackets);
                for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx) {
                    this->m_vvsInputPaths[nInputPacketIdx].resize(nInputStreamCount);
                    this->m_vvsInputPaths[nInputPacketIdx][nInputRGBStreamIdx] = vsThermalInputPaths[nInputPacketIdx];
                    const size_t nLastInputSlashPos = vsThermalInputPaths[nInputPacketIdx].find_last_of("/\\");
                    const std::string sInputFileNameExt = nLastInputSlashPos==std::string::npos?vsThermalInputPaths[nInputPacketIdx]:vsThermalInputPaths[nInputPacketIdx].substr(nLastInputSlashPos+1);
                    const size_t nLastInputDotPos = sInputFileNameExt.find_last_of('.');
                    vsTempInputFileNames[nInputPacketIdx] = nLastInputDotPos==std::string::npos?sInputFileNameExt:sInputFileNameExt.substr(0,nLastInputDotPos);
                }
                if(bUseInterlacedMasks && bUseApproxThermalMask) {
                    std::vector<std::string> vsThermalApproxMasksPaths = vsApproxMasksPaths;
                    lv::filterFilePaths(vsThermalApproxMasksPaths,{},{"VisForeground"});
                    if(vsThermalApproxMasksPaths.empty() || cv::imread(vsThermalApproxMasksPaths[0]).size()!=oImageSize)
                        lvError_("LITIV-bilodeau2014 sequence '%s' did not possess expected thermal approx mask data",this->getName().c_str());
                    lvAssert_(vsThermalApproxMasksPaths.size()==nInputPackets,"thermal approx mask count did not match input image count");
                    for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                        this->m_vvsInputPaths[nInputPacketIdx][nInputThermalMaskStreamIdx] = vsThermalApproxMasksPaths[nInputPacketIdx];
                }
                cv::Mat oThermalROI = cv::imread(this->getDataPath()+"IRROI.png",cv::IMREAD_GRAYSCALE);
                if(!oThermalROI.empty()) {
                    lvAssert(oThermalROI.type()==CV_8UC1 && oThermalROI.size()==oImageSize);
                    oThermalROI = oThermalROI>0;
                }
                else {
                    const cv::Mat oInitThermalInput = cv::imread(vsThermalInputPaths[0]);
                    lvAssert(!oInitThermalInput.empty() && oInitThermalInput.size()==oImageSize && oInitThermalInput.type()==CV_8UC3);
                    oThermalROI = (cv::imread(vsThermalInputPaths[0])!=cv::Scalar_<uchar>(255,255,255));
                    cv::erode(oThermalROI,oThermalROI,cv::Mat());
                    lvAssert(oThermalROI.type()==CV_8UC1 && cv::countNonZero(oThermalROI)>0);
                }
                if(oThermalROI.size()!=this->m_vInputInfos[nInputThermalStreamIdx].size())
                    cv::resize(oThermalROI,oThermalROI,this->m_vInputInfos[nInputThermalStreamIdx].size(),0,0,cv::INTER_NEAREST);
                this->m_vInputROIs[nInputThermalStreamIdx] = oThermalROI.clone();
                if(bUseInterlacedMasks)
                    this->m_vInputROIs[nInputThermalMaskStreamIdx] = oThermalROI.clone();
                this->m_vGTROIs[nGTThermalMaskStreamIdx] = oThermalROI.clone();
                std::vector<std::string> vsThermalGTMasksPaths;
                if(this->m_bEvalStereoDisp) {
                    vsThermalGTMasksPaths = vsDisparityMasksPaths;
                    lv::filterFilePaths(vsThermalGTMasksPaths,{},{"DisparityIR"});
                }
                else {
                    lvIgnore(psGTMasksDir);
                    lvAssert(false); // @@@@ missing impl; dataset does not contain gt segm masks
                }
                if(vsThermalGTMasksPaths.empty() || cv::imread(vsThermalGTMasksPaths[0]).size()!=oImageSize)
                    lvError_("LITIV-bilodeau2014 sequence '%s' did not possess expected thermal gt data",this->getName().c_str());
                const size_t nGTPackets = vsThermalGTMasksPaths.size();
                this->m_vvsGTPaths.resize(nGTPackets);
                this->m_mGTIndexLUT.clear();
                for(size_t nGTPacketIdx=0; nGTPacketIdx<nGTPackets; ++nGTPacketIdx) {
                    // input and gt mask names might not match @@@@@@@@@@@@@@@@@@@@
                    this->m_vvsGTPaths[nGTPacketIdx].resize(nGTStreamCount);
                    this->m_vvsGTPaths[nGTPacketIdx][nGTThermalMaskStreamIdx] = vsThermalGTMasksPaths[nGTPacketIdx];
                    const size_t nLastGTSlashPos = vsThermalGTMasksPaths[nGTPacketIdx].find_last_of("/\\");
                    const std::string sGTFileNameExt = nLastGTSlashPos==std::string::npos?vsThermalGTMasksPaths[nGTPacketIdx]:vsThermalGTMasksPaths[nGTPacketIdx].substr(nLastGTSlashPos+1);
                    const size_t nLastGTDotPos = sGTFileNameExt.find_last_of('.');
                    const std::string sGTFileName = nLastGTDotPos==std::string::npos?sGTFileNameExt:sGTFileNameExt.substr(0,nLastGTDotPos);
                    lvAssert(!sGTFileName.empty());
                    size_t nInputPacketIdx = 0;
                    for(; nInputPacketIdx<vsTempInputFileNames.size(); ++nInputPacketIdx)
                        if(sGTFileName==vsTempInputFileNames[nInputPacketIdx])
                            break;
                    lvAssert(nInputPacketIdx<vsTempInputFileNames.size());
                    this->m_mGTIndexLUT[nInputPacketIdx] = nGTPacketIdx; // direct gt path index to frame index mapping
                }
                if(bUseInterlacedMasks && !bUseApproxThermalMask) {
                    lvAssert(false); // @@@@ missing impl; dataset does not contain gt segm masks
                }
                //////////////////////////////////////////////////////////////////////
                std::vector<std::string> vsRGBInputPaths = vsInputPaths;
                lv::filterFilePaths(vsRGBInputPaths,{},{"Vis"});
                if(vsRGBInputPaths.empty() || cv::imread(vsRGBInputPaths[0]).size()!=oImageSize)
                    lvError_("LITIV-bilodeau2014 sequence '%s' did not possess expected RGB input data",this->getName().c_str());
                if(vsRGBInputPaths.size()!=nInputPackets)
                    lvError_("LITIV-bilodeau2014 sequence '%s' did not possess same amount of RGB/thermal frames",this->getName().c_str());
                if(bUseInterlacedMasks && bUseApproxRGBMask) {
                    std::vector<std::string> vsRGBApproxMasksPaths = vsApproxMasksPaths;
                    lv::filterFilePaths(vsRGBApproxMasksPaths,{},{"VisForeground"});
                    if(vsRGBApproxMasksPaths.empty() || cv::imread(vsRGBApproxMasksPaths[0]).size()!=oImageSize)
                        lvError_("LITIV-bilodeau2014 sequence '%s' did not possess expected RGB approx mask data",this->getName().c_str());
                    lvAssert_(vsRGBApproxMasksPaths.size()==nInputPackets,"rgb approx mask count did not match input image count");
                    for(size_t nInputPacketIdx=0; nInputPacketIdx<nInputPackets; ++nInputPacketIdx)
                        this->m_vvsInputPaths[nInputPacketIdx][nInputRGBMaskStreamIdx] = vsRGBApproxMasksPaths[nInputPacketIdx];
                }
                cv::Mat oRGBROI = cv::imread(this->getDataPath()+"VisROI.png",cv::IMREAD_GRAYSCALE);
                if(!oRGBROI.empty()) {
                    lvAssert(oRGBROI.type()==CV_8UC1 && oRGBROI.size()==oImageSize);
                    oRGBROI = oRGBROI>0;
                }
                else {
                    const cv::Mat oInitRGBInput = cv::imread(vsRGBInputPaths[0]);
                    lvAssert(!oInitRGBInput.empty() && oInitRGBInput.size()==oImageSize && oInitRGBInput.type()==CV_8UC3);
                    oRGBROI = (cv::imread(vsRGBInputPaths[0])!=cv::Scalar_<uchar>(255,255,255));
                    cv::erode(oRGBROI,oRGBROI,cv::Mat());
                    lvAssert(oRGBROI.type()==CV_8UC1 && cv::countNonZero(oRGBROI)>0);
                }
                if(oRGBROI.size()!=this->m_vInputInfos[nInputRGBStreamIdx].size())
                    cv::resize(oRGBROI,oRGBROI,this->m_vInputInfos[nInputRGBStreamIdx].size(),0,0,cv::INTER_NEAREST);
                this->m_vInputROIs[nInputRGBStreamIdx] = oRGBROI.clone();
                if(bUseInterlacedMasks)
                    this->m_vInputROIs[nInputRGBMaskStreamIdx] = oRGBROI.clone();
                this->m_vGTROIs[nGTRGBMaskStreamIdx] = oRGBROI.clone();
                if(bUseInterlacedMasks && !bUseApproxRGBMask) {
                    lvAssert(false); // @@@@ missing impl; dataset does not contain gt segm masks
                }
            }
            this->m_nMinDisp = size_t(0);
            this->m_nMaxDisp = size_t(50);
            std::ifstream oDispRangeFile(this->getDataPath()+"drange.txt");
            if(oDispRangeFile.is_open() && !oDispRangeFile.eof()) {
                oDispRangeFile >> this->m_nMinDisp;
                if(!oDispRangeFile.eof())
                    oDispRangeFile >> this->m_nMaxDisp;
            }
            this->m_nMinDisp *= dScale;
            this->m_nMaxDisp *= dScale;
            lvAssert(this->m_nMaxDisp>this->m_nMinDisp);
        }
        virtual std::vector<cv::Mat> getRawInputArray(size_t nPacketIdx) override final {
            lvDbgExceptionWatch;
            if(nPacketIdx>=this->m_vvsInputPaths.size())
                return std::vector<cv::Mat>(this->getInputStreamCount());
            const cv::Size oImageSize(480,360);
            const bool bUseInterlacedMasks = this->m_nLoadInputMasks!=0;
            constexpr size_t nInputRGBStreamIdx = 0;
            const size_t nInputThermalStreamIdx = bUseInterlacedMasks?2:1;
            constexpr size_t nInputRGBMaskStreamIdx = 1;
            constexpr size_t nInputThermalMaskStreamIdx = 3;
            std::vector<cv::Mat> vInputs(getInputStreamCount());
            const std::vector<lv::MatInfo>& vInputInfos = this->m_vInputInfos;
            lvDbgAssert(!vInputInfos.empty() && vInputInfos.size()==getInputStreamCount());
            if(this->m_bLoadFullVideos) {
                lvAssert(false); // missing impl @@@@
            }
            else {
                const std::vector<std::string>& vsInputPaths = this->m_vvsInputPaths[nPacketIdx];
                lvDbgAssert(!vsInputPaths.empty() && vsInputPaths.size()==this->getInputStreamCount());
                ///////////////////////////////////////////////////////////////////////////////////
                cv::Mat oRGBPacket = cv::imread(vsInputPaths[nInputRGBStreamIdx],cv::IMREAD_COLOR);
                lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC3 && oRGBPacket.size()==oImageSize);
                if(oRGBPacket.size()!=vInputInfos[nInputRGBStreamIdx].size())
                    cv::resize(oRGBPacket,oRGBPacket,vInputInfos[nInputRGBStreamIdx].size(),0,0,cv::INTER_CUBIC);
                vInputs[nInputRGBStreamIdx] = oRGBPacket;
                if(bUseInterlacedMasks) {
                    cv::Mat oRGBMaskPacket = cv::imread(vsInputPaths[nInputRGBMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oRGBMaskPacket.empty() && oRGBMaskPacket.type()==CV_8UC1 && oRGBMaskPacket.size()==oImageSize);
                    if(oRGBMaskPacket.size()!=vInputInfos[nInputRGBStreamIdx].size())
                        cv::resize(oRGBMaskPacket,oRGBMaskPacket,vInputInfos[nInputRGBStreamIdx].size(),cv::INTER_NEAREST);
                    vInputs[nInputRGBMaskStreamIdx] = oRGBMaskPacket;
                }
                ///////////////////////////////////////////////////////////////////////////////////
                cv::Mat oThermalPacket = cv::imread(vsInputPaths[nInputThermalStreamIdx],cv::IMREAD_GRAYSCALE);
                lvAssert(!oThermalPacket.empty() && oThermalPacket.type()==CV_8UC1 && oThermalPacket.size()==oImageSize);
                if(oThermalPacket.size()!=vInputInfos[nInputThermalStreamIdx].size())
                    cv::resize(oThermalPacket,oThermalPacket,vInputInfos[nInputThermalStreamIdx].size());
                vInputs[nInputThermalStreamIdx] = oThermalPacket;
                if(bUseInterlacedMasks) {
                    cv::Mat oThermalMaskPacket = cv::imread(vsInputPaths[nInputThermalMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oThermalMaskPacket.empty() && oThermalMaskPacket.type()==CV_8UC1 && oThermalMaskPacket.size()==oImageSize);
                    if(oThermalMaskPacket.size()!=vInputInfos[nInputThermalStreamIdx].size())
                        cv::resize(oThermalMaskPacket,oThermalMaskPacket,vInputInfos[nInputThermalStreamIdx].size(),cv::INTER_NEAREST);
                    vInputs[nInputThermalMaskStreamIdx] = oThermalMaskPacket;
                }
            }
            return vInputs;
        }
        virtual std::vector<cv::Mat> getRawGTArray(size_t nPacketIdx) override final {
            lvDbgExceptionWatch;
            const cv::Size oImageSize(480,360);
            constexpr size_t nGTRGBMaskStreamIdx = 0;
            constexpr size_t nGTThermalMaskStreamIdx = 1;
            if(this->m_mGTIndexLUT.count(nPacketIdx)) {
                const size_t nGTIdx = this->m_mGTIndexLUT[nPacketIdx];
                lvDbgAssert(nGTIdx<this->m_vvsGTPaths.size());
                const std::vector<lv::MatInfo>& vGTInfos = this->m_vGTInfos;
                lvDbgAssert(!vGTInfos.empty() && vGTInfos.size()==getGTStreamCount());
                std::vector<cv::Mat> vGTs(getGTStreamCount());
                if(this->m_bLoadFullVideos) {
                    lvAssert(false); // missing impl @@@@
                }
                else {
                    const std::vector<std::string>& vsGTMasksPaths = this->m_vvsGTPaths[nGTIdx];
                    lvDbgAssert(!vsGTMasksPaths.empty() && vsGTMasksPaths.size()==getGTStreamCount());
                    /*cv::Mat oRGBPacket = cv::imread(vsGTMasksPaths[nGTRGBMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC1 && oRGBPacket.size()==oImageSize);*/
                    // @@@@ no RGB packet in current dataset (gt is only for thermal)
                    lvAssert(this->m_bEvalStereoDisp); // missing impl for 'dont care' rgb packet if eval foreground masks
                    cv::Mat oRGBPacket = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255)); // init with oob 'dont care' value for disparity
                    if(oRGBPacket.size()!=vGTInfos[nGTRGBMaskStreamIdx].size())
                        cv::resize(oRGBPacket,oRGBPacket,vGTInfos[nGTRGBMaskStreamIdx].size(),0,0,cv::INTER_NEAREST);
                    vGTs[nGTRGBMaskStreamIdx] = oRGBPacket;
                    cv::Mat oThermalPacket = cv::imread(vsGTMasksPaths[nGTThermalMaskStreamIdx],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oThermalPacket.empty() && oThermalPacket.type()==CV_8UC1 && oThermalPacket.size()==oImageSize);
                    if(oThermalPacket.size()!=vGTInfos[nGTThermalMaskStreamIdx].size())
                        cv::resize(oThermalPacket,oThermalPacket,vGTInfos[nGTThermalMaskStreamIdx].size(),0,0,cv::INTER_NEAREST);
                    vGTs[nGTThermalMaskStreamIdx] = oThermalPacket;
                }
                return vGTs;
            }
            return std::vector<cv::Mat>();
        }
        bool m_bLoadFullVideos;
        bool m_bEvalStereoDisp;
        int m_nLoadInputMasks;
        size_t m_nMinDisp,m_nMaxDisp;
        std::string m_sFeaturesDirName;
    };

} // namespace lv
