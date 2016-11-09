
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

#define DATASETS_VAP_FIX_SCENE2_DISTORT 1
#define DATASETS_VAP_FIX_SCENE3_OFFSET  1

namespace lv {

    struct IVAPtrimod2016Dataset {
        /// returns whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
        virtual bool isLoadingDepth() const = 0;
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        virtual bool isUndistorting() const = 0;
    };

    template<DatasetTaskList eDatasetTask, lv::ParallelAlgoType eEvalImpl>
    struct Dataset_<eDatasetTask,Dataset_VAPtrimod2016,eEvalImpl> :
            public IVAPtrimod2016Dataset,
            public IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016,lv::getDatasetEval<eDatasetTask,Dataset_VAPtrimod2016>(),eEvalImpl> {
    protected: // should still be protected, as creation should always be done via datasets::create
        Dataset_(
                const std::string& sOutputDirName, ///< output directory name for debug logs, evaluation reports and results archiving (will be created in VAP trimodal dataset results folder)
                bool bSaveOutput=false, ///< defines whether results should be archived or not
                bool bUseEvaluator=true, ///< defines whether results should be fully evaluated, or simply acknowledged
                bool bForce4ByteDataAlign=false, ///< defines whether data packets should be 4-byte aligned (useful for GPU upload)
                double dScaleFactor=1.0, ///< defines the scale factor to use to resize/rescale read packets
                bool bLoadDepth=true, ///< defines whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
                bool bUndistort=true ///< defines whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        ) :
                IDataset_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016,lv::getDatasetEval<eDatasetTask,Dataset_VAPtrimod2016>(),eEvalImpl>(
                        "VAP-trimodal2016",
                        lv::datasets::getDatasetsRootPath()+"vap/rgbdt-stereo/",
                        lv::datasets::getDatasetsRootPath()+"vap/rgbdt-stereo/results/"+lv::AddDirSlashIfMissing(sOutputDirName),
                        "bin",
                        ".png",
                        getWorkBatchDirNames(),
                        std::vector<std::string>(),
                        getGrayscaleWorkBatchDirNames(),
                        bSaveOutput,
                        bUseEvaluator,
                        bForce4ByteDataAlign,
                        dScaleFactor
                ), m_bLoadDepth(bLoadDepth),m_bUndistort(bUndistort) {}
        /// returns the names of all work batch directories available for this dataset specialization
        static const std::vector<std::string>& getWorkBatchDirNames() {
            static const std::vector<std::string> s_vsWorkBatchDirs = {"Scene 1","Scene 2","Scene 3"};
            return s_vsWorkBatchDirs;
        }
        /// returns the names of all work batch directories which should be treated as grayscale for this dataset speialization
        static const std::vector<std::string>& getGrayscaleWorkBatchDirNames() {
            static const std::vector<std::string> s_vsGrayscaleWorkBatchDirs = {"SyncD","SyncT"};
            return s_vsGrayscaleWorkBatchDirs;
        }
        /// returns whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
        virtual bool isLoadingDepth() const override {return m_bLoadDepth;}
        /// returns whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        virtual bool isUndistorting() const override {return m_bUndistort;}
    protected:
        const bool m_bLoadDepth;
        const bool m_bUndistort;
    };

    template<DatasetTaskList eDatasetTask>
    struct DataGroupHandler_<eDatasetTask,DatasetSource_VideoArray,Dataset_VAPtrimod2016> :
            public DataGroupHandler {
    protected:
        virtual void parseData() override {
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            this->m_vpBatches.clear();
            this->m_bIsBare = true; // in this dataset, work batch groups are always bare
            if(!lv::string_contains_token(this->getName(),this->getSkippedDirTokens()))
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
            lvDbgExceptionWatch;
            // 'this' is required below since name lookup is done during instantiation because of not-fully-specialized class template
            std::vector<std::string> vsSubDirs;
            lv::GetSubDirsFromDir(this->getDataPath(),vsSubDirs);
            auto psDepthGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"depthMasks");
            auto psDepthDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"SyncD");
            auto psRGBGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"rgbMasks");
            auto psRGBDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"SyncRGB");
            auto psThermalGTDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"thermalMasks");
            auto psThermalDir = std::find(vsSubDirs.begin(),vsSubDirs.end(),this->getDataPath()+"SyncT");
            if((psDepthDir==vsSubDirs.end() || psDepthGTDir==vsSubDirs.end()) || (psRGBDir==vsSubDirs.end() || psRGBGTDir==vsSubDirs.end()) || (psThermalDir==vsSubDirs.end() || psThermalGTDir==vsSubDirs.end()))
                lvError_("VAPtrimod2016 sequence '%s' did not possess the required groundtruth and input directories",this->getName().c_str());
            const IVAPtrimod2016Dataset& oDataset = dynamic_cast<const IVAPtrimod2016Dataset&>(*this->getRoot());
            this->m_bLoadDepth = oDataset.isLoadingDepth();
            this->m_bUndistort = oDataset.isUndistorting();
            const size_t nStreamCount = this->m_bLoadDepth?3:2;
            this->m_vInputROIs.resize(nStreamCount);
            this->m_vGTROIs.resize(nStreamCount);
            this->m_vInputSizes.resize(nStreamCount);
            this->m_vGTSizes.resize(nStreamCount);
            cv::Mat oGlobalROI(480,640,CV_8UC1,cv::Scalar_<uchar>(255));
            const double dScale = this->getScaleFactor();
            if(dScale!=1.0)
                cv::resize(oGlobalROI,oGlobalROI,cv::Size(),dScale,dScale,cv::INTER_NEAREST);
            for(size_t s=0; s<nStreamCount; ++s)
                this->m_vInputSizes[s] = this->m_vGTSizes[s] = oGlobalROI.size();
            this->m_oMaxInputSize = this->m_oMaxGTSize = oGlobalROI.size();
            const cv::Size oImageSize(640,480);
            const double dUndistortMapCameraMatrixAlpha = -1.0;
            if(this->m_bUndistort) {
                cv::FileStorage oParamsFS(this->getDataPath()+"calibVars.yml",cv::FileStorage::READ);
                lvAssert_(oParamsFS.isOpened(),"could not open calibration yml file");
                oParamsFS["rgbCamMat"] >> this->m_oRGBCameraParams;
                oParamsFS["rgbDistCoeff"] >> this->m_oRGBDistortParams;
                lvAssert_(!this->m_oRGBCameraParams.empty() && !this->m_oRGBDistortParams.empty(),"failed to load RGB camera calibration parameters");
                cv::initUndistortRectifyMap(this->m_oRGBCameraParams,this->m_oRGBDistortParams,cv::Mat(),
                                            (dUndistortMapCameraMatrixAlpha<0)?this->m_oRGBCameraParams:cv::getOptimalNewCameraMatrix(this->m_oRGBCameraParams,this->m_oRGBDistortParams,oImageSize,dUndistortMapCameraMatrixAlpha,oImageSize,0),
                                            oImageSize,CV_16SC2,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2);
                oParamsFS["tCamMat"] >> this->m_oThermalCameraParams;
                oParamsFS["tDistCoeff"] >> this->m_oThermalDistortParams;
                lvAssert_(!this->m_oThermalCameraParams.empty() && !this->m_oThermalDistortParams.empty(),"failed to load thermal camera calibration parameters");
                cv::initUndistortRectifyMap(this->m_oThermalCameraParams,this->m_oThermalDistortParams,cv::Mat(),
                                            (dUndistortMapCameraMatrixAlpha<0)?this->m_oThermalCameraParams:cv::getOptimalNewCameraMatrix(this->m_oThermalCameraParams,this->m_oThermalDistortParams,oImageSize,dUndistortMapCameraMatrixAlpha,oImageSize,0),
                                            oImageSize,CV_16SC2,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2);
            }
            //
            // NOTE: internal stream indexing
            //    stream[0] = RGB     (default:CV_8UC3)
            //    stream[1] = thermal (default:?)
            //    stream[2] = depth   (default:?) --- if enabled only
            //
            std::vector<std::string> vsRGBPaths;
            lv::GetFilesFromDir(*psRGBDir,vsRGBPaths);
            if(vsRGBPaths.empty() || cv::imread(vsRGBPaths[0]).size()!=oImageSize)
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
            cv::Mat oRGBROI = cv::imread(this->getDataPath()+"rgb_roi.png",cv::IMREAD_GRAYSCALE);
            if(!oRGBROI.empty()) {
                lvAssert(oRGBROI.type()==CV_8UC1 && oRGBROI.size()==oImageSize);
                oRGBROI = oRGBROI>0;
            }
            else
                oRGBROI = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
            if(oRGBROI.size()!=this->m_vInputSizes[0])
                cv::resize(oRGBROI,oRGBROI,this->m_vInputSizes[0],0,0,cv::INTER_NEAREST);
            if(this->m_bUndistort)
                cv::remap(oRGBROI.clone(),oRGBROI,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_NEAREST);
            this->m_vInputROIs[0] = oRGBROI.clone();
            this->m_vGTROIs[0] = oRGBROI.clone();
            std::vector<std::string> vsRGBGTPaths;
            lv::GetFilesFromDir(*psRGBGTDir,vsRGBGTPaths);
            if(vsRGBGTPaths.empty() || cv::imread(vsRGBGTPaths[0]).size()!=oImageSize)
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
            if(vsThermalPaths.empty() || cv::imread(vsThermalPaths[0]).size()!=oImageSize)
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected thermal data",this->getName().c_str());
            if(vsThermalPaths.size()!=vsRGBPaths.size())
                lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/thermal frames",this->getName().c_str());
            for(size_t nInputPacketIdx=0; nInputPacketIdx<vsThermalPaths.size(); ++nInputPacketIdx)
                this->m_vvsInputPaths[nInputPacketIdx][1] = vsThermalPaths[nInputPacketIdx];
            cv::Mat oThermalROI = cv::imread(this->getDataPath()+"thermal_roi.png",cv::IMREAD_GRAYSCALE);
            if(!oThermalROI.empty()) {
                lvAssert(oThermalROI.type()==CV_8UC1 && oThermalROI.size()==oImageSize);
                oThermalROI = oThermalROI>0;
            }
            else
                oThermalROI = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
            if(oThermalROI.size()!=this->m_vInputSizes[1])
                cv::resize(oThermalROI,oThermalROI,this->m_vInputSizes[1],0,0,cv::INTER_NEAREST);
            if(this->m_bUndistort)
                cv::remap(oThermalROI.clone(),oThermalROI,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2,cv::INTER_NEAREST);
            this->m_vInputROIs[1] = oThermalROI.clone();
            this->m_vGTROIs[1] = oThermalROI.clone();
            std::vector<std::string> vsThermalGTPaths;
            lv::GetFilesFromDir(*psThermalGTDir,vsThermalGTPaths);
            if(vsThermalGTPaths.empty() || cv::imread(vsThermalGTPaths[0]).size()!=oImageSize)
                lvError_("VAPtrimod2016 sequence '%s' did not possess expected thermal gt data",this->getName().c_str());
            if(vsThermalGTPaths.size()!=vsRGBGTPaths.size())
                lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/thermal gt frames",this->getName().c_str());
            for(size_t nGTPacketIdx=0; nGTPacketIdx<vsThermalGTPaths.size(); ++nGTPacketIdx)
                this->m_vvsGTPaths[nGTPacketIdx][1] = vsThermalGTPaths[nGTPacketIdx];
            if(this->m_bLoadDepth) {
                std::vector<std::string> vsDepthPaths;
                lv::GetFilesFromDir(*psDepthDir,vsDepthPaths);
                if(vsDepthPaths.empty() || cv::imread(vsDepthPaths[0]).size()!=oImageSize)
                    lvError_("VAPtrimod2016 sequence '%s' did not possess expected depth data",this->getName().c_str());
                if(vsDepthPaths.size()!=vsRGBPaths.size())
                    lvError_("VAPtrimod2016 sequence '%s' did not possess same amount of RGB/depth frames",this->getName().c_str());
                for(size_t nInputPacketIdx=0; nInputPacketIdx<vsDepthPaths.size(); ++nInputPacketIdx)
                    this->m_vvsInputPaths[nInputPacketIdx][2] = vsDepthPaths[nInputPacketIdx];
                cv::Mat oDepthROI = cv::imread(this->getDataPath()+"depth_roi.png",cv::IMREAD_GRAYSCALE);
                if(!oDepthROI.empty()) {
                    lvAssert(oDepthROI.type()==CV_8UC1 && oDepthROI.size()==oImageSize);
                    oDepthROI = oDepthROI>0;
                }
                else
                    oDepthROI = cv::Mat(oImageSize,CV_8UC1,cv::Scalar_<uchar>(255));
                if(oDepthROI.size()!=this->m_vInputSizes[2])
                    cv::resize(oDepthROI,oDepthROI,this->m_vInputSizes[2],0,0,cv::INTER_NEAREST);
                this->m_vInputROIs[2] = oDepthROI.clone();
                this->m_vGTROIs[2] = oDepthROI.clone();
                std::vector<std::string> vsDepthGTPaths;
                lv::GetFilesFromDir(*psDepthGTDir,vsDepthGTPaths);
                if(vsDepthGTPaths.empty() || cv::imread(vsDepthGTPaths[0]).size()!=oImageSize)
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
                lvDbgAssert(vUnpackedInput.size()==(this->m_bLoadDepth?3:2));
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
            lvDbgAssert(vInputSizes.size()==vsInputPaths.size() && (lv::accumulateMembers<int,cv::Size>(vInputSizes,[](const cv::Size& s){return s.area();}))>0);
            cv::Mat oFullPacket(1,vInputSizes[0].area()*3+vInputSizes[1].area()+(this->m_bLoadDepth?vInputSizes[2].area()*2:0),CV_8UC1);
            lvAssert_(!this->is4ByteAligned(),"missing conversion/alignment impl");
            ptrdiff_t nPacketOffset = 0;
            const auto lAppendPacket = [&](const cv::Mat& oNewPacket) {
                lvDbgAssert(oFullPacket.isContinuous() && oNewPacket.isContinuous());
                std::copy(oNewPacket.datastart,oNewPacket.dataend,oFullPacket.data+nPacketOffset);
                nPacketOffset += ptrdiff_t(oNewPacket.dataend-oNewPacket.datastart);
            };
            cv::Mat oRGBPacket = cv::imread(vsInputPaths[0]);
            lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC3 && oRGBPacket.size()==cv::Size(640,480));
            if(oRGBPacket.size()!=vInputSizes[0])
                cv::resize(oRGBPacket,oRGBPacket,vInputSizes[0]);
            if(this->m_bUndistort)
                cv::remap(oRGBPacket.clone(),oRGBPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_LINEAR);
            lAppendPacket(oRGBPacket);
            lvDbgAssert(nPacketOffset<ptrdiff_t(oFullPacket.dataend-oFullPacket.datastart));
            cv::Mat oThermalPacket = cv::imread(vsInputPaths[1],cv::IMREAD_GRAYSCALE);
            lvAssert(!oThermalPacket.empty() && oThermalPacket.type()==CV_8UC1 && oThermalPacket.size()==cv::Size(640,480));
            if(oThermalPacket.size()!=vInputSizes[1])
                cv::resize(oThermalPacket,oThermalPacket,vInputSizes[1]);
            if(this->m_bUndistort)
                cv::remap(oThermalPacket.clone(),oThermalPacket,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2,cv::INTER_LINEAR);
            lAppendPacket(oThermalPacket);
            lvDbgAssert(nPacketOffset<=ptrdiff_t(oFullPacket.dataend-oFullPacket.datastart));
            if(this->m_bLoadDepth) {
                cv::Mat oDepthPacket = cv::imread(vsInputPaths[2],cv::IMREAD_ANYDEPTH);
                lvAssert(!oDepthPacket.empty() && oDepthPacket.type()==CV_16UC1 && oDepthPacket.size()==cv::Size(640,480));
                if(oDepthPacket.size()!=vInputSizes[2])
                    cv::resize(oDepthPacket,oDepthPacket,vInputSizes[2]);
                // depth should be already undistorted
                lAppendPacket(oDepthPacket);
            }
            lvDbgAssert(nPacketOffset==ptrdiff_t(oFullPacket.dataend-oFullPacket.datastart));
            return oFullPacket;
        }
        virtual cv::Mat getRawGT(size_t nPacketIdx) override final {
            if(this->m_mGTIndexLUT.count(nPacketIdx)) {
                const size_t nGTIdx = this->m_mGTIndexLUT[nPacketIdx];
                lvDbgAssert(nGTIdx<this->m_vvsGTPaths.size());
                const std::vector<std::string>& vsGTPaths = this->m_vvsGTPaths[nGTIdx];
                lvDbgAssert(!vsGTPaths.empty() && vsGTPaths.size()==getGTStreamCount() && vsGTPaths.size()==(this->m_bLoadDepth?3:2));
                const std::vector<cv::Size>& vGTSizes = this->m_vGTSizes;
                const int nTotPacketSize = lv::accumulateMembers<int,cv::Size>(vGTSizes,[](const cv::Size& s){return s.area();});
                lvDbgAssert(vGTSizes.size()==vsGTPaths.size() && nTotPacketSize>0);
                cv::Mat oFullPacket(1,nTotPacketSize,CV_8UC1);
                lvAssert_(!this->is4ByteAligned(),"missing conversion/alignment impl");
                ptrdiff_t nPacketOffset = 0;
                const auto lAppendPacket = [&](const cv::Mat& oNewPacket) {
                    lvDbgAssert(oFullPacket.isContinuous() && oNewPacket.isContinuous());
                    std::copy(oNewPacket.datastart,oNewPacket.dataend,oFullPacket.data+nPacketOffset);
                    nPacketOffset += ptrdiff_t(oNewPacket.dataend-oNewPacket.datastart);
                };
                cv::Mat oRGBPacket = cv::imread(vsGTPaths[0],cv::IMREAD_GRAYSCALE);
                lvAssert(!oRGBPacket.empty() && oRGBPacket.type()==CV_8UC1 && oRGBPacket.size()==cv::Size(640,480));
                if(oRGBPacket.size()!=vGTSizes[0])
                    cv::resize(oRGBPacket,oRGBPacket,vGTSizes[0],0,0,cv::INTER_NEAREST);
                if(this->m_bUndistort)
                    cv::remap(oRGBPacket.clone(),oRGBPacket,this->m_oRGBCalibMap1,this->m_oRGBCalibMap2,cv::INTER_NEAREST);
                lAppendPacket(oRGBPacket);
                lvDbgAssert(nPacketOffset<nTotPacketSize);
                cv::Mat oThermalPacket = cv::imread(vsGTPaths[1],cv::IMREAD_GRAYSCALE);
                lvAssert(!oThermalPacket.empty() && oThermalPacket.type()==CV_8UC1 && oThermalPacket.size()==cv::Size(640,480));
#if DATASETS_VAP_FIX_SCENE3_OFFSET
                // fail: calibration really breaks up for scene 3 (need to translate [x,y]=[13,4], and it's still not great)
                if(this->getName()=="Scene 3") {
                    static const cv::Mat_<double> oAffineTransf = (cv::Mat_<double>(2,3) << 1.0,0.0,13.0,0.0,1.0,4.0);
                    cv::warpAffine(oThermalPacket.clone(),oThermalPacket,oAffineTransf,oThermalPacket.size());
                }
#endif //DATASETS_VAP_FIX_SCENE3_OFFSET
                if(oThermalPacket.size()!=vGTSizes[1])
                    cv::resize(oThermalPacket,oThermalPacket,vGTSizes[1],0,0,cv::INTER_NEAREST);
#if DATASETS_VAP_FIX_SCENE2_DISTORT
                // fail: input 'distorted' thermal gt images are actually already undistorted in 'Scene 2' (so no need to remap again)
                if(this->getName()!="Scene 2")
#endif //DATASETS_VAP_FIX_SCENE2_DISTORT
                {
                    if(this->m_bUndistort)
                        cv::remap(oThermalPacket.clone(),oThermalPacket,this->m_oThermalCalibMap1,this->m_oThermalCalibMap2,cv::INTER_NEAREST);
                }
                lAppendPacket(oThermalPacket);
                lvDbgAssert(nPacketOffset<=nTotPacketSize);
                if(this->m_bLoadDepth) {
                    cv::Mat oDepthPacket = cv::imread(vsGTPaths[2],cv::IMREAD_GRAYSCALE);
                    lvAssert(!oDepthPacket.empty() && oDepthPacket.type()==CV_8UC1 && oDepthPacket.size()==cv::Size(640,480));
                    if(oDepthPacket.size()!=vGTSizes[2])
                        cv::resize(oDepthPacket,oDepthPacket,vGTSizes[2]);
                    // depth should be already undistorted
                    lAppendPacket(oDepthPacket);
                }
                lvDbgAssert(nPacketOffset==nTotPacketSize);
                return oFullPacket;
            }
            return cv::Mat();
        }
        bool m_bLoadDepth; ///< defines whether the depth stream should be loaded or not (if not, the dataset is used as a bimodal one)
        bool m_bUndistort; ///< defines whether images should be undistorted when loaded or not, using the calib files provided with the dataset
        cv::Mat m_oRGBCameraParams;
        cv::Mat m_oThermalCameraParams;
        cv::Mat m_oRGBDistortParams;
        cv::Mat m_oThermalDistortParams;
        cv::Mat m_oRGBCalibMap1,m_oRGBCalibMap2;
        cv::Mat m_oThermalCalibMap1,m_oThermalCalibMap2;
    };

} // namespace lv
