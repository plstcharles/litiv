
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

namespace lv {

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_VAPtrimod2016,eEvalImpl> :
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016,getDatasetEval<eDatasetTask,Dataset_VAPtrimod2016>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via datasets::create
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in VAP trimodal dataset results folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bForce4ByteDataAlign=false, ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
                double dScaleFactor=1.0, ///< defines the scale factor to use to resize/rescale read packets
                bool bLoadDepth=true ///< defines whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016,getDatasetEval<eDatasetTask,Dataset_VAPtrimod2016>(),eEvalImpl>(
                        "VAP-trimodal2016",
                        lv::datasets::getDatasetsRootPath()+"vap/rgbdt-stereo/",
                        lv::datasets::getDatasetsRootPath()+"vap/rgbdt-stereo/results/"+lv::AddDirSlashIfMissing(sOutputDirName),
                        "bin",
                        ".png",
                        getWorkBatchDirNames(),
                        getSkippedWorkBatchDirNames(bLoadDepth),
                        getGrayscaleWorkBatchDirNames(),
                        0,
                        bSaveOutput,
                        bUseEvaluator,
                        bForce4ByteDataAlign,
                        dScaleFactor
                ) {}
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames() {
            static const std::vector<std::string> s_vsWorkBatchDirs = {"Scene 1","Scene 2","Scene 3"};// @@@@ groups = these, batches...? will not auto-dig deeper?
            return s_vsWorkBatchDirs;
        }
        /// returns the names of all work batch directories which should be skipped for this dataset speialization
        static const std::vector<std::string>& getSkippedWorkBatchDirNames(bool bLoadDepth=true) {
            static const std::vector<std::string> s_vsSkippedWorkBatchDirs_nodepth = {"SyncD"};
            static const std::vector<std::string> s_vsSkippedWorkBatchDirs = {};
            return bLoadDepth?s_vsSkippedWorkBatchDirs:s_vsSkippedWorkBatchDirs_nodepth;
        }
        /// returns the names of all work batch directories which should be treated as grayscale for this dataset speialization
        static const std::vector<std::string>& getGrayscaleWorkBatchDirNames() {
            static const std::vector<std::string> s_vsGrayscaleWorkBatchDirs = {"SyncD","SyncT"};
            return s_vsGrayscaleWorkBatchDirs;
        }
    };

    template<DatasetTaskList eDatasetTask>
    struct GroupDataParser_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016> :
            public IGroupDataParser {
    protected:
        virtual void parseData() override {
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            this->m_vpBatches.clear();
            this->m_bIsBare = true;
            // in this dataset, work batch groups are always bare
            if(!lv::string_contains_token(this->getName(),this->getDatasetInfo()->getSkippedDirTokens()))
                this->m_vpBatches.push_back(this->createWorkBatch(this->getName(),this->getRelativePath()));
        }
    };

    template<DatasetTaskList eDatasetTask>
    struct DataProducer_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016> :
            public IDataProducerWrapper_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016> {
        virtual size_t getInputStreamCount() const override final {
            return m_bLoadDepth?3:2;
        }
        virtual size_t getGTStreamCount() const override final {
            return m_bLoadDepth?3:2;
        }
        virtual std::string getInputStreamName(size_t nStreamIdx) const override final {
            return ((nStreamIdx==0)?"RGB":(nStreamIdx==1)?"THERMAL":(nStreamIdx==2)?"DEPTH":"UNKNOWN");
        }
        virtual std::string getGTStreamName(size_t nStreamIdx) const override final {
            return getInputStreamName(nStreamIdx)+"_GT";
        }
        virtual bool isStreamGrayscale(size_t nStreamIdx) const override final {
            return nStreamIdx!=0;
        }
    protected:
        virtual void parseData() override final {
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            std::vector<std::string> vsSubDirs;
            lv::GetSubDirsFromDir(this->getDataPath(),vsSubDirs);
            auto psDepthGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"depthMasks");
            auto psDepthDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"SyncD");
            auto psRGBGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"rgbMasks");
            auto psRGBDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"SyncRGB");
            auto psThermalGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"thermalMasks");
            auto psThermalDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),lv::AddDirSlashIfMissing(this->getDataPath())+"SyncT");
            if((psDepthDir==vsSubDirs.end() || psDepthGTDir==vsSubDirs.end()) || (psRGBDir==vsSubDirs.end() || psRGBGTDir==vsSubDirs.end()) || (psThermalDir==vsSubDirs.end() || psThermalGTDir==vsSubDirs.end()))
                lvError_("VAPtrimod2016 sequence '%s' did not possess the required groundtruth and input directories",this->getName().c_str());
            this->m_bLoadDepth = !lv::string_contains_token("SyncD",this->getDatasetInfo()->getSkippedDirTokens());
            const size_t nStreamCount = this->m_bLoadDepth?3:2;
            this->m_vInputROIs.resize(nStreamCount);
            this->m_vGTROIs.resize(nStreamCount);
            this->m_vInputSizes.resize(nStreamCount);
            this->m_vGTSizes.resize(nStreamCount);
            cv::Mat oGlobalROI(480,640,CV_8UC1,cv::Scalar_<uchar>(255));
            const double dScale = this->getDatasetInfo()->getScaleFactor();
            if(dScale!=1.0)
                cv::resize(oGlobalROI,oGlobalROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
            for(size_t s=0; s<nStreamCount; ++s) {
                this->m_vInputROIs[s] = oGlobalROI.clone();
                this->m_vGTROIs[s] = oGlobalROI.clone();
                this->m_vInputSizes[s] = this->m_vGTSizes[s] = oGlobalROI.size();
            }
            this->m_oMaxInputSize = this->m_oMaxGTSize = oGlobalROI.size();
            //
            // NOTE: internal stream indexing
            //    stream[0] = RGB     (default:CV_8UC3)
            //    stream[1] = thermal (default:?)
            //    stream[2] = depth   (default:?) --- if enabled only
            //
            std::vector<std::string> vsRGBPaths;
            lv::GetFilesFromDir(*psRGBDir,vsRGBPaths);
            if(vsRGBPaths.empty() || cv::imread(vsRGBPaths[0]).size()!=cv::Size(640,480))
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected RGB data",this->getName().c_str());
            this->m_vvsInputPaths.resize(vsRGBPaths.size());
            std::vector<std::string> vsTempInputFileNames(vsRGBPaths.size());
            for(size_t nInputPacketIdx=0; nInputPacketIdx<vsRGBPaths.size(); ++nInputPacketIdx) {
                this->m_vvsInputPaths[nInputPacketIdx].resize(nStreamCount);
                this->m_vvsInputPaths[nInputPacketIdx][0] = vsRGBPaths[nInputPacketIdx];
                const size_t nLastInputSlashPos = vsRGBPaths[nInputPacketIdx].find_last_of("/\\");
                const std::string sInputFileNameExt = nLastInputSlashPos==std::string::npos?vsRGBPaths[nInputPacketIdx]:vsRGBPaths[nInputPacketIdx].substr(nLastInputSlashPos+1);
                const size_t nLastInputDotPos = sInputFileNameExt.find_last_of('.');
                vsTempInputFileNames[nInputPacketIdx] = nLastInputDotPos==std::string::npos?sInputFileNameExt:sInputFileNameExt.substr(0,nLastInputDotPos);
            }
            std::vector<std::string> vsRGBGTPaths;
            lv::GetFilesFromDir(*psRGBGTDir,vsRGBGTPaths);
            if(vsRGBGTPaths.empty() || cv::imread(vsRGBGTPaths[0]).size()!=cv::Size(640,480))
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected RGB gt data",this->getName().c_str());
            this->m_vvsGTPaths.resize(vsRGBGTPaths.size());
            this->m_mGTIndexLUT.clear();
            for(size_t nGTPacketIdx=0; nGTPacketIdx<vsRGBGTPaths.size(); ++nGTPacketIdx) {
                this->m_vvsGTPaths[nGTPacketIdx].resize(nStreamCount);
                this->m_vvsGTPaths[nGTPacketIdx][0] = vsRGBGTPaths[nGTPacketIdx];
                const size_t nLastGTSlashPos = vsRGBGTPaths[nGTPacketIdx].find_last_of("/\\");
                const std::string sGTFileNameExt = nLastGTSlashPos==std::string::npos?vsRGBGTPaths[nGTPacketIdx]:vsRGBGTPaths[nGTPacketIdx].substr(nLastGTSlashPos+1);
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
            std::vector<std::string> vsThermalPaths;
            lv::GetFilesFromDir(*psThermalDir,vsThermalPaths);
            if(vsThermalPaths.empty() || cv::imread(vsThermalPaths[0]).size()!=cv::Size(640,480))
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected thermal data",this->getName().c_str());
            if(vsThermalPaths.size()!=vsRGBPaths.size())
                lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/thermal frames",this->getName().c_str());
            for(size_t nInputPacketIdx=0; nInputPacketIdx<vsThermalPaths.size(); ++nInputPacketIdx)
                this->m_vvsInputPaths[nInputPacketIdx][1] = vsThermalPaths[nInputPacketIdx];
            std::vector<std::string> vsThermalGTPaths;
            lv::GetFilesFromDir(*psThermalGTDir,vsThermalGTPaths);
            if(vsThermalGTPaths.empty() || cv::imread(vsThermalGTPaths[0]).size()!=cv::Size(640,480))
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected thermal gt data",this->getName().c_str());
            if(vsThermalGTPaths.size()!=vsRGBGTPaths.size())
                lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/thermal gt frames",this->getName().c_str());
            for(size_t nGTPacketIdx=0; nGTPacketIdx<vsThermalGTPaths.size(); ++nGTPacketIdx)
                this->m_vvsGTPaths[nGTPacketIdx][1] = vsThermalPaths[nGTPacketIdx];
            if(this->m_bLoadDepth) {
                std::vector<std::string> vsDepthPaths;
                lv::GetFilesFromDir(*psDepthDir,vsDepthPaths);
                if(vsDepthPaths.empty() || cv::imread(vsDepthPaths[0]).size()!=cv::Size(640,480))
                    lvError_("VAPtrimod2016 sequence '%s' did not possess expected depth data",this->getName().c_str());
                if(vsDepthPaths.size()!=vsRGBPaths.size())
                    lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/depth frames",this->getName().c_str());
                for(size_t nInputPacketIdx=0; nInputPacketIdx<vsDepthPaths.size(); ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][2] = vsDepthPaths[nInputPacketIdx];
                std::vector<std::string> vsDepthGTPaths;
                lv::GetFilesFromDir(*psDepthGTDir,vsDepthGTPaths);
                if(vsDepthGTPaths.empty() || cv::imread(vsDepthGTPaths[0]).size()!=cv::Size(640,480))
                    lvError_("VAPtrimod2016 sequence '%s' did not possess expected depth gt data",this->getName().c_str());
                if(vsDepthGTPaths.size()!=vsRGBGTPaths.size())
                    lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/depth gt frames",this->getName().c_str());
                for(size_t nGTPacketIdx=0; nGTPacketIdx<vsDepthGTPaths.size(); ++nGTPacketIdx)
                    this->m_vvsGTPaths[nGTPacketIdx][2] = vsDepthGTPaths[nGTPacketIdx];
            }
        }
        virtual void unpackInput(size_t nPacketIdx, std::vector<cv::Mat>& vUnpackedInput) override final {
            // no need to clone if getInput does not allow reentrancy --- output mats in the vector will stay valid for as long as oInput is valid (typically until next getInput call)
            const cv::Mat& oInput = this->getInput(nPacketIdx)/*.clone()*/;
            if(oInput.empty()) {
                for(size_t s=0; s<vUnpackedInput.size(); ++s)
                    vUnpackedInput[s] = cv::Mat();
            }
            else {
                lvDbgAssert(vUnpackedInput.size()==this->m_bLoadDepth?3:2);
                lvDbgAssert(this->getInputPacketType()==ImageArrayPacket);
                const std::vector<cv::Size>& vSizes = this->m_vInputSizes;
                lvDbgAssert(vSizes.size()==vUnpackedInput.size() && vSizes[0].area()>0);
                lvDbgAssert(oInput.size().area()*(int)oInput.elemSize()==vSizes[0].area()*(this->m_bLoadDepth?6:4));
                lvDbgAssert(oInput.isContinuous());
                const size_t nArea = (size_t)vSizes[0].area();
                vUnpackedInput[0] = cv::Mat(vSizes[0],CV_8UC3,oInput.data);
                vUnpackedInput[1] = cv::Mat(vSizes[0],CV_8UC1,oInput.data+nArea*vUnpackedInput[0].elemSize());
                if(this->m_bLoadDepth)
                    vUnpackedInput[2] = cv::Mat(vSizes[0],CV_16UC1,oInput.data+nArea*vUnpackedInput[0].elemSize()+nArea*vUnpackedInput[1].elemSize());
            }
        }
        virtual cv::Mat getRawInput(size_t nPacketIdx) override final {
            if(nPacketIdx>=this->m_vvsInputPaths.size())
                return cv::Mat();
            const std::vector<std::string>& vsInputPaths = this->m_vvsInputPaths[nPacketIdx];
            lvDbgAssert(!vsInputPaths.empty() && vsInputPaths.size()==getInputStreamCount() && vsInputPaths.size()==(this->m_bLoadDepth?3:2));
            const std::vector<cv::Size>& vInputSizes = this->m_vInputSizes;
            lvDbgAssert(vInputSizes.size()==vsInputPaths.size() && vInputSizes[0].area()>0);
            cv::Mat oFullPacket(1,vInputSizes[0].area()*(this->m_bLoadDepth?6:4),CV_8UC1);
            lvAssert_(!this->getDatasetInfo()->is4ByteAligned(),"missing conversion/alignment impl");
            ptrdiff_t nPacketOffset = 0;
            const auto lAppendPacket = [&](const cv::Mat& oNewPacket) {
                lvDbgAssert(oNewPacket.size()==vInputSizes[0]);
                lvDbgAssert(oFullPacket.isContinuous() && oNewPacket.isContinuous());
                std::copy(oNewPacket.datastart,oNewPacket.dataend,oFullPacket.data+nPacketOffset);
                nPacketOffset += ptrdiff_t(oNewPacket.dataend-oNewPacket.datastart);
            };
            cv::Mat oRGBPacket = cv::imread(vsInputPaths[0]);
            lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC3 && oRGBPacket.size()==cv::Size(640,480));
            if(oRGBPacket.size()!=vInputSizes[0])
                cv::resize(oRGBPacket,oRGBPacket,vInputSizes[0]);
            lAppendPacket(oRGBPacket);
            cv::Mat oThermalPacket = cv::imread(vsInputPaths[1],cv::IMREAD_GRAYSCALE);
            lvAssert(!oThermalPacket.empty() && oThermalPacket.type()==CV_8UC1 && oThermalPacket.size()==cv::Size(640,480));
            lvDbgAssert(vInputSizes[1]==vInputSizes[0]);
            if(oThermalPacket.size()!=vInputSizes[0])
                cv::resize(oThermalPacket,oThermalPacket,vInputSizes[0]);
            lAppendPacket(oThermalPacket);
            if(this->m_bLoadDepth) {
                cv::Mat oDepthPacket = cv::imread(vsInputPaths[2],cv::IMREAD_ANYDEPTH);
                lvAssert(!oDepthPacket.empty() && oDepthPacket.type()==CV_16UC1 && oDepthPacket.size()==cv::Size(640,480));
                lvDbgAssert(vInputSizes[2]==vInputSizes[0]);
                if(oDepthPacket.size()!=vInputSizes[0])
                    cv::resize(oDepthPacket,oDepthPacket,vInputSizes[0]);
                lAppendPacket(oDepthPacket);
            }
            lvDbgAssert(nPacketOffset==ptrdiff_t(oFullPacket.dataend-oFullPacket.datastart));
            return oFullPacket;
        }
        bool m_bLoadDepth; ///< defines whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
    };

} // namespace lv
